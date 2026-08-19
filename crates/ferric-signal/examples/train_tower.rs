//! Train the actual transformer autoencoder through the FSQ bottleneck, on synthetic physics.
//!
//!   cargo run -p ferric-signal --example train_tower --release
//!
//! The MLP run in `train_tokenizer` proved the bottleneck and the gradient path. This runs the real
//! towers — the same code the `Tensor` forward uses, held to it numerically by
//! `the_var_tower_matches_the_tensor_tower` — so the model that trains is the model that runs.
//!
//! Scope, stated up front: a small configuration on synthetic signals for a short run. This is a
//! demonstration that the pipeline learns end to end, not a sensor foundation model.

use ferric_core::Context;
use ferric_signal::{
    decoder_forward_var, forward_var, mse, straight_through, DecoderWeights, EncoderConfig,
    EncoderWeights, Fsq, Patcher, RevIn,
};
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::HashSet;
use std::sync::Arc;

const STEPS: usize = 400;

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (((z >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
    }).collect()
}

fn synthetic(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        let t = i as f32 * 0.002;
        match (i / 256) % 5 {
            0 => (-1.2 * t).exp() * (2.0 * std::f32::consts::PI * 11.0 * t).sin() * 3.0,
            1 => 20.0 + 45.0 * (-0.8 * t).exp(),
            2 => if (t * 60.0).fract() < 0.35 { 5.0 } else { 0.0 },
            3 => (2.0 * std::f32::consts::PI * (4.0 + 30.0 * t) * t).sin() * 1.5,
            _ => 0.5 * (0.7 * t).sin() + fill(i as u64, 1, 0.6)[0],
        }
    }).collect()
}

fn main() {
    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example measures nothing without one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);

    let cfg = EncoderConfig { patch_len: 16, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 128, latent_dim: 5 };
    let q = Fsq::signal_15bit();

    let raw = synthetic(cfg.patch_len * 256);
    let rev = RevIn::fit(&raw, 1).unwrap();
    let patcher = Patcher::contiguous(cfg.patch_len).unwrap();
    let patches = patcher.patchify(&rev.apply(&raw).unwrap()).unwrap();
    let t = patches.len() / cfg.patch_len;
    let x = Tensor::from_vec(&ctx, &patches, &[t, cfg.patch_len]);

    let mut ep = EncoderWeights::deterministic(&ctx, cfg, 1).unwrap().params_flat();
    let mut dp = DecoderWeights::deterministic(&ctx, cfg, 2).unwrap().params_flat();
    let n_enc = ep.len();
    let mut all: Vec<Tensor> = ep.iter().chain(dp.iter()).cloned().collect();
    let mut opt = Adam::new(&all, 2e-3);

    println!("\nTRANSFORMER AUTOENCODER  d_model {} layers {} heads {}  latent {} -> {} codes",
             cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.latent_dim, q.codebook_size());
    println!("  {} encoder + {} decoder parameters, {t} patches\n",
             cfg.params().total(), cfg.decoder_params());
    println!("  {:>6}  {:>12}  {:>14}", "step", "recon MSE", "codes used");
    println!("  {:->6}  {:->12}  {:->14}", "", "", "");

    let mut first = 0.0f32;
    for step in 0..=STEPS {
        let vars: Vec<Var> = all.iter().cloned().map(Var::leaf).collect();
        let xv = Var::leaf(x.clone());
        let z = forward_var(&ctx, cfg, &vars[..n_enc], &xv).unwrap();
        let zq = straight_through(&ctx, &z, &q);
        let recon = decoder_forward_var(&ctx, cfg, &vars[n_enc..], &zq).unwrap();
        let loss = mse(&recon, &xv);
        loss.backward();

        let l = pollster::block_on(loss.value().to_vec())[0];
        if step == 0 { first = l; }
        if step % 100 == 0 || step == STEPS {
            let zv = pollster::block_on(z.value().to_vec());
            let used: HashSet<u32> = zv.chunks(cfg.latent_dim)
                .filter_map(|r| q.quantize(r).ok().map(|c| q.to_index(&c).unwrap()))
                .collect();
            println!("  {step:>6}  {l:>12.6}  {:>8} / {}", used.len(), q.codebook_size());
        }
        let grads: Vec<Tensor> = vars.iter().enumerate()
            .map(|(i, v)| v.grad().unwrap_or_else(|| panic!("parameter {i} got no gradient")))
            .collect();
        opt.step(&mut all, &grads);
    }
    ep = all[..n_enc].to_vec();
    dp = all[n_enc..].to_vec();
    let _ = (&ep, &dp);

    let vars: Vec<Var> = all.iter().cloned().map(Var::leaf).collect();
    let xv = Var::leaf(x.clone());
    let z = forward_var(&ctx, cfg, &vars[..n_enc], &xv).unwrap();
    let zq = straight_through(&ctx, &z, &q);
    let recon = decoder_forward_var(&ctx, cfg, &vars[n_enc..], &zq).unwrap();
    let last = pollster::block_on(mse(&recon, &xv).value().to_vec())[0];

    let pv = pollster::block_on(x.to_vec());
    let m = pv.iter().sum::<f32>() / pv.len() as f32;
    let var = pv.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / pv.len() as f32;

    println!("\nRESULT");
    println!("  reconstruction MSE     {last:.6}   (from {first:.6}, {:.0}x lower)", first / last.max(1e-12));
    println!("  share of variance      {:.2}%   ({:.1} dB SNR)", last / var * 100.0,
             10.0 * (var / last.max(1e-12)).log10());
    println!("\n  Scope: {STEPS} steps, small config, synthetic signals. This shows the real towers");
    println!("  learn end to end through the discrete bottleneck. It is not a foundation model.\n");
}
