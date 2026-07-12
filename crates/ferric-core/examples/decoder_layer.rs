//! A complete modern (Llama / SmolVLA-style) transformer DECODER LAYER, composed entirely from
//! Ferric on-GPU kernels and validated end-to-end against a plain-Rust CPU reference:
//!
//!   x → RMSNorm → QKV → RoPE(Q,K) → causal multi-head attention → out-proj → +residual
//!     → RMSNorm → SwiGLU FFN (SiLU-gated) → +residual → y
//!
//! Everything stays on the device between ops (one readback at the end); the same source runs on
//! native Metal/Vulkan/DX12 and browser WebGPU.
use ferric_core::{cpu, matmul_cpu, max_abs_diff, Context};

fn fill(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.2 - 0.1).collect()
}
fn emul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x * y).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let (t, h, dh, hff, base, eps) = (8usize, 4usize, 16usize, 128usize, 10000.0f32, 1e-5f32);
    let d = h * dh;

    let xd = fill(t * d, 1.0);
    let wn1: Vec<f32> = fill(d, 2.0).iter().map(|v| v + 1.0).collect();
    let wn2: Vec<f32> = fill(d, 3.0).iter().map(|v| v + 1.0).collect();
    let (wq, wk, wv, wo) = (fill(d * d, 4.0), fill(d * d, 5.0), fill(d * d, 6.0), fill(d * d, 7.0));
    let (wg, wu, wd) = (fill(d * hff, 8.0), fill(d * hff, 9.0), fill(hff * d, 10.0));

    // ---- GPU: on-device, ops chain buffer→buffer ----
    let (u, um, us) = (|v: &Vec<f32>, s: &[usize]| ctx.tensor(v, s), t as u32, d as u32);
    let x = u(&xd, &[t, d]);
    let (twn1, twn2) = (u(&wn1, &[d]), u(&wn2, &[d]));
    let (tq, tk, tv, to) = (u(&wq, &[d, d]), u(&wk, &[d, d]), u(&wv, &[d, d]), u(&wo, &[d, d]));
    let (tg, tu2, td) = (u(&wg, &[d, hff]), u(&wu, &[d, hff]), u(&wd, &[hff, d]));

    let rms1 = ctx.rmsnorm_t(&x, &twn1, um, us, eps);
    let q = ctx.rope_t(&ctx.mm(&rms1, &tq, um, us, us), um, h as u32, dh as u32, base);
    let k = ctx.rope_t(&ctx.mm(&rms1, &tk, um, us, us), um, h as u32, dh as u32, base);
    let v = ctx.mm(&rms1, &tv, um, us, us);
    let attn = ctx.mha_causal_t(&q, &k, &v, um, h as u32, h as u32, dh as u32);
    let x2 = ctx.add_t(&x, &ctx.mm(&attn, &to, um, us, us));
    let rms2 = ctx.rmsnorm_t(&x2, &twn2, um, us, eps);
    let g = ctx.mm(&rms2, &tg, um, us, hff as u32);
    let up = ctx.mm(&rms2, &tu2, um, us, hff as u32);
    let silu = ctx.mul_t(&g, &ctx.sigmoid_t(&g));
    let down = ctx.mm(&ctx.mul_t(&silu, &up), &td, um, hff as u32, us);
    let y = ctx.add_t(&x2, &down);
    let y_gpu = ctx.to_vec(&y).await.unwrap();

    // ---- CPU reference: the same math in plain Rust ----
    let rms1 = cpu::rmsnorm(&xd, &wn1, t, d, eps);
    let q = cpu::rope(&matmul_cpu(&rms1, &wq, t, d, d), t, h, dh, base);
    let k = cpu::rope(&matmul_cpu(&rms1, &wk, t, d, d), t, h, dh, base);
    let v = matmul_cpu(&rms1, &wv, t, d, d);
    let attn = cpu::mha_causal(&q, &k, &v, t, h, h, dh);
    let x2 = cpu::add(&xd, &matmul_cpu(&attn, &wo, t, d, d));
    let rms2 = cpu::rmsnorm(&x2, &wn2, t, d, eps);
    let g = matmul_cpu(&rms2, &wg, t, d, hff);
    let up = matmul_cpu(&rms2, &wu, t, d, hff);
    let silu = emul(&g, &cpu::sigmoid(&g));
    let down = matmul_cpu(&emul(&silu, &up), &wd, t, hff, d);
    let y_cpu = cpu::add(&x2, &down);

    let diff = max_abs_diff(&y_gpu, &y_cpu);
    println!("Ferric decoder layer · {:?} · T={t} H={h} dh={dh} D={d}", ctx.backend);
    println!("  RMSNorm→RoPE→causal-MHA→proj→res→RMSNorm→SwiGLU→res");
    println!("  max|gpu - cpu| = {diff:.3e}  (f32 reduction-order drift over ~8 chained matmuls)");
    assert!(diff < 1e-3, "decoder layer mismatch {diff}");
    println!("✅ A full modern decoder LAYER runs on-GPU in Ferric — matches the CPU reference");
}
