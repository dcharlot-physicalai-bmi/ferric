//! Train the discrete bottleneck on synthetic physics, and measure what it actually did.
//!
//!   cargo run -p ferric-signal --example train_tokenizer --release
//!
//! ## Why synthetic, and why this is the honest first number
//!
//! Training on synthetic physical processes needs no data licence and, more usefully, the ground
//! truth is known: a damped oscillator, a thermal decay and a PWM square wave are exactly the kinds
//! of behaviour a sensor tokenizer has to represent, and if the bottleneck cannot carry them it
//! will not carry a bearing trace either.
//!
//! ## What is measured, and the one that matters
//!
//! Reconstruction error says whether the codes carry the signal. **Codebook utilisation tests FSQ's
//! central claim**: that codebook collapse — the failure that makes VQ-VAE training fragile, where
//! most of the vocabulary goes unused and the model quietly operates on a handful of codes — cannot
//! happen when there is no codebook to collapse. That claim is worth measuring rather than
//! repeating, and this prints the number.
//!
//! The encoder here is an MLP, not the transformer tower. The tower's forward pass is written
//! against `Tensor`; training it needs the same pass rebuilt on `Var`, which is mechanical and not
//! yet done. So this measures **the bottleneck and the training path**, and says so.

use ferric_core::Context;
use ferric_signal::{mse, straight_through, Fsq, Patcher, RevIn};
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::HashSet;
use std::sync::Arc;

const PATCH: usize = 16;
const HIDDEN: usize = 64;
const LATENT: usize = 5;
const STEPS: usize = 1500;

/// Deterministic pseudo-random, so a reported number can be reproduced exactly.
fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (((z >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Five physical processes, each with its own parameters, all deterministic.
fn synthetic(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.002;
        let which = (i / 512) % 5;
        let v = match which {
            // Damped spring-mass: the textbook case the phenomenological work uses.
            0 => (-1.2 * t).exp() * (2.0 * std::f32::consts::PI * 11.0 * t).sin() * 3.0,
            // Thermal decay toward ambient.
            1 => 20.0 + 45.0 * (-0.8 * t).exp(),
            // PWM at a fixed duty: hard edges, which a smooth basis finds hard.
            2 => if (t * 60.0).fract() < 0.35 { 5.0 } else { 0.0 },
            // Chirp: frequency the model has not seen at that amplitude.
            3 => (2.0 * std::f32::consts::PI * (4.0 + 30.0 * t) * t).sin() * 1.5,
            // Broadband noise on a slow drift.
            _ => 0.5 * (0.7 * t).sin() + fill(i as u64, 1, 0.6)[0],
        };
        out.push(v);
    }
    out
}

struct Mlp {
    w: Vec<Tensor>,
}

impl Mlp {
    fn new(ctx: &Arc<Context>) -> Self {
        let s = |fan: usize| 1.0 / (fan as f32).sqrt();
        Self {
            w: vec![
                Tensor::from_vec(ctx, &fill(1, HIDDEN * PATCH, s(PATCH)), &[HIDDEN, PATCH]),
                Tensor::from_vec(ctx, &fill(2, LATENT * HIDDEN, s(HIDDEN)), &[LATENT, HIDDEN]),
                Tensor::from_vec(ctx, &fill(3, HIDDEN * LATENT, s(LATENT)), &[HIDDEN, LATENT]),
                Tensor::from_vec(ctx, &fill(4, PATCH * HIDDEN, s(HIDDEN)), &[PATCH, HIDDEN]),
            ],
        }
    }
    fn vars(&self) -> Vec<Var> {
        self.w.iter().cloned().map(Var::leaf).collect()
    }
}

/// `x @ wᵀ` for an HF-layout weight `[out, in]`, differentiably.
fn linear(x: &Var, w: &Var) -> Var {
    x.matmul(&w.transpose(1, 0))
}

fn main() {
    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example measures nothing without one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);

    let raw = synthetic(PATCH * 640);
    let rev = RevIn::fit(&raw, 1).unwrap();
    let patches = Patcher::contiguous(PATCH).unwrap().patchify(&rev.apply(&raw).unwrap()).unwrap();
    let t = patches.len() / PATCH;
    let x = Tensor::from_vec(&ctx, &patches, &[t, PATCH]);

    let q = Fsq::signal_15bit();
    let mut net = Mlp::new(&ctx);
    let mut opt = Adam::new(&net.w, 3e-3);

    println!("\nTRAINING  {t} patches of {PATCH} samples, 5 synthetic physical processes");
    println!("  bottleneck {LATENT} dims x 8 levels = {} codes\n", q.codebook_size());
    println!("  {:>6}  {:>12}  {:>14}", "step", "recon MSE", "codes used");
    println!("  {:->6}  {:->12}  {:->14}", "", "", "");

    let forward = |vars: &[Var], xv: &Var| -> (Var, Var) {
        let h = linear(xv, &vars[0]).silu();
        let z = linear(&h, &vars[1]);
        let zq = straight_through(&ctx, &z, &q);
        let g = linear(&zq, &vars[2]).silu();
        (linear(&g, &vars[3]), zq)
    };

    let mut first = 0.0f32;
    for step in 0..=STEPS {
        let vars = net.vars();
        let xv = Var::leaf(x.clone());
        let (recon, _) = forward(&vars, &xv);
        let loss = mse(&recon, &xv);
        loss.backward();

        let l = pollster::block_on(loss.value().to_vec())[0];
        if step == 0 {
            first = l;
        }
        if step % 250 == 0 || step == STEPS {
            let used = codes_used(&ctx, &vars, &x, &q);
            println!("  {step:>6}  {l:>12.6}  {used:>8} / {}", q.codebook_size());
        }
        let grads: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
        opt.step(&mut net.w, &grads);
    }

    // ---- what actually happened ----
    let vars = net.vars();
    let xv = Var::leaf(x.clone());
    let (recon, _) = forward(&vars, &xv);
    let last = pollster::block_on(mse(&recon, &xv).value().to_vec())[0];
    let used = codes_used(&ctx, &vars, &x, &q);

    // Variance of the normalized input: MSE relative to it is the fraction of signal NOT captured.
    let pv = pollster::block_on(x.to_vec());
    let mean = pv.iter().sum::<f32>() / pv.len() as f32;
    let var = pv.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / pv.len() as f32;

    println!("\nRESULT");
    println!("  reconstruction MSE      {last:.6}   (from {first:.6}, {:.0}x lower)", first / last.max(1e-12));
    println!("  as a share of variance  {:.2}%   ({:.1} dB SNR)",
             last / var * 100.0, 10.0 * (var / last.max(1e-12)).log10());
    println!("  codes used              {used} of {} ({:.1}%)",
             q.codebook_size(), used as f64 / q.codebook_size() as f64 * 100.0);
    println!("  distinct patches        {t}");
    // Diagnose the utilisation rather than reporting it flat: how many LEVELS does each latent
    // dimension actually reach, and how wide is the latent before bounding?
    let h = linear(&Var::leaf(x.clone()), &vars[0]).silu();
    let zraw = pollster::block_on(linear(&h, &vars[1]).value().to_vec());
    println!("\n  PER-DIMENSION DIAGNOSIS");
    let mut product = 1usize;
    for d in 0..LATENT {
        let col: Vec<f32> = zraw.iter().skip(d).step_by(LATENT).copied().collect();
        let m = col.iter().sum::<f32>() / col.len() as f32;
        let sd = (col.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / col.len() as f32).sqrt();
        let levels: std::collections::HashSet<u32> =
            col.iter().map(|&v| q.quantize(&[v; 1][..].iter().cycle().take(LATENT).copied().collect::<Vec<_>>()).unwrap()[d]).collect();
        product *= levels.len().max(1);
        println!("    dim {d}: latent sd {sd:>7.3}  reaches {} of 8 levels", levels.len());
    }
    println!("    product of reached levels = {product}, against {used} codes observed");

    println!("\n  Codebook utilisation is the number that tests FSQ's central claim. There is no");
    println!("  codebook to collapse, so every code stays reachable by construction; a VQ-VAE at");
    println!("  this scale is where collapse would show up as a handful of codes carrying");
    println!("  everything. Used codes cannot exceed distinct patches, which is the real ceiling");
    println!("  here at {t}.");
    println!("\n  Scope: an MLP bottleneck, not the transformer tower, and synthetic signals.");
    println!("  This measures the BOTTLENECK and the TRAINING PATH, not a sensor foundation model.\n");
}

fn codes_used(ctx: &Arc<Context>, vars: &[Var], x: &Tensor, q: &Fsq) -> usize {
    let h = linear(&Var::leaf(x.clone()), &vars[0]).silu();
    let z = pollster::block_on(linear(&h, &vars[1]).value().to_vec());
    let _ = ctx;
    let mut seen = HashSet::new();
    for row in z.chunks(LATENT) {
        if let Ok(c) = q.quantize(row) {
            seen.insert(q.to_index(&c).unwrap());
        }
    }
    seen.len()
}
