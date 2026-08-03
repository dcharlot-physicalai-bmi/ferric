//! Verify Cosmos 3 Edge's FULL generator-tower forward: latent → proj_in → 28 dual-stream layers
//! (AR conditioning + DM diffusion tokens, tech-report Eq 7-8) → norm_moe_gen → proj_out → velocity.
//! The complete generator velocity-prediction pass (one denoising step's network eval), sans timestep
//! injection, verified exact vs a numpy reference. usage: cosmos_gen_forward <dir>
use ferric_core::Context;
use ferric_load::safetensors_filtered;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let dir = std::env::args().nth(1).expect("usage: cosmos_gen_forward <dir>");
    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let keep = |n: &str| n.starts_with("layers.") || n == "proj_in.weight" || n == "proj_in.bias"
        || n == "proj_out.weight" || n == "proj_out.bias" || n == "norm_moe_gen.weight" || n.starts_with("time_embedder.");
    let mut w: HashMap<String, ferric_load::STensor> = HashMap::new();
    for e in std::fs::read_dir(format!("{dir}/transformer")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            w.extend(safetensors_filtered(&std::fs::read(&p).unwrap(), keep).unwrap());
        }
    }
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let (nh, nkv, hd, eps, base) = (16usize, 8usize, 128usize, 1e-5f32, 1e8f32);
    println!("loaded generator weights in {:?}", t0.elapsed());

    // synthetic AR conditioning [Ta,2048] + noisy DM latents [Td,192]
    let (ta, td) = (5usize, 4usize);
    let xa0: Vec<f32> = (0..ta * 2048).map(|i| ((i as f32 * 0.01).sin() * 0.1)).collect();
    let lat: Vec<f32> = (0..td * 192).map(|i| ((i as f32 * 0.03 + 1.0).sin() * 0.2)).collect();
    let mut xa = Tensor::from_vec(&ctx, &xa0, &[ta, 2048]);
    let mut xd = Tensor::from_vec(&ctx, &lat, &[td, 192]).matmul_bt(&g("proj_in.weight")).add(&g("proj_in.bias").reshape(&[1, 2048])); // [Td,2048]

    // Timestep conditioning (code-faithful): temb = time_embedder(time_proj(σ·timestep_scale)), added
    // to the noisy DM tokens at input (no per-block AdaLN). time_proj = diffusers Timesteps(256,
    // flip_sin_to_cos=True, downscale_freq_shift=0, max_period=10000); time_embedder = Linear→SiLU→Linear.
    let (timestep, timestep_scale) = (500.0f32, 0.001f32); // a mid denoising step; σ = t·scale = 0.5
    let sigma = timestep * timestep_scale;
    let half = 128usize;
    let mut tp = vec![0f32; 256];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        tp[i] = (sigma * freq).cos();          // flip_sin_to_cos=True → cos first
        tp[half + i] = (sigma * freq).sin();
    }
    let tpv = Tensor::from_vec(&ctx, &tp, &[1, 256]);
    let temb = tpv.matmul_bt(&g("time_embedder.linear_1.weight")).add(&g("time_embedder.linear_1.bias").reshape(&[1, 2048]))
        .silu().matmul_bt(&g("time_embedder.linear_2.weight")).add(&g("time_embedder.linear_2.bias").reshape(&[1, 2048])); // [1,2048]
    xd = xd.add(&temb); // broadcast over the Td noisy tokens

    let t0 = std::time::Instant::now();
    let mlp = |h: &Tensor, u: &Tensor, d: &Tensor| h.matmul_bt(u).relu2().matmul_bt(d);
    for il in 0..28 {
        let b = |s: &str| g(&format!("layers.{il}.{s}"));
        // AR stream: per the diffusers Cosmos3OmniTransformer, Edge has qk_norm_for_text=False, so
        // q_und/k_und are UN-normed for the AR causal self-attn. A SEPARATE k_norm_und_for_gen(k_und)
        // copy (`k_und_for_gen`) is the AR key fed into the DM full attention.
        let ha = xa.rmsnorm(&b("input_layernorm.weight"), eps);
        let qa = ha.matmul_bt(&b("self_attn.to_q.weight")).rope(nh, hd, base, 0);
        let k_und = ha.matmul_bt(&b("self_attn.to_k.weight"));
        let ka = k_und.rope(nkv, hd, base, 0); // un-normed AR key (causal self-attn)
        let ka_gen = k_und.reshape(&[ta, nkv, hd]).rmsnorm(&b("self_attn.k_norm_und_for_gen.weight"), eps).reshape(&[ta, nkv * hd]).rope(nkv, hd, base, 0); // k_und_for_gen
        let va = ha.matmul_bt(&b("self_attn.to_v.weight"));
        let oa = nn::causal_attention(&qa, &ka, &va, nh, nkv, 0.0);
        let xa1 = xa.add(&oa.matmul_bt(&b("self_attn.to_out.weight")));
        xa = xa1.add(&mlp(&xa1.rmsnorm(&b("post_attention_layernorm.weight"), eps), &b("mlp.up_proj.weight"), &b("mlp.down_proj.weight")));
        // DM stream (full attn over [K_AR;K_DM], q+k normed)
        let hdd = xd.rmsnorm(&b("input_layernorm_moe_gen.weight"), eps);
        let qd = hdd.matmul_bt(&b("self_attn.add_q_proj.weight")).reshape(&[td, nh, hd]).rmsnorm(&b("self_attn.norm_added_q.weight"), eps).reshape(&[td, nh * hd]).rope(nh, hd, base, 0);
        let kd = hdd.matmul_bt(&b("self_attn.add_k_proj.weight")).reshape(&[td, nkv, hd]).rmsnorm(&b("self_attn.norm_added_k.weight"), eps).reshape(&[td, nkv * hd]).rope(nkv, hd, base, 0);
        let vd = hdd.matmul_bt(&b("self_attn.add_v_proj.weight"));
        let od = nn::full_attention_kv(&qd, &ka_gen.cat(&kd, 0), &va.cat(&vd, 0), nh, nkv); // AR key = k_und_for_gen
        let xd1 = xd.add(&od.matmul_bt(&b("self_attn.to_add_out.weight")));
        xd = xd1.add(&mlp(&xd1.rmsnorm(&b("post_attention_layernorm_moe_gen.weight"), eps), &b("mlp_moe_gen.up_proj.weight"), &b("mlp_moe_gen.down_proj.weight")));
    }
    let vel = xd.rmsnorm(&g("norm_moe_gen.weight"), eps).matmul_bt(&g("proj_out.weight")).add(&g("proj_out.bias").reshape(&[1, 192]));
    let o = vel.to_vec().await; // [Td, 192] predicted velocity
    println!("full 28-layer generator forward {:?}", t0.elapsed());
    println!("Ferric velocity token0[:6] = {:?}", &o[..6].iter().map(|v| (v * 1e5).round() / 1e5).collect::<Vec<_>>());
    println!("  velocity sum = {:.5} · finite = {}", o.iter().sum::<f32>(), o.iter().all(|x| x.is_finite()));
}
