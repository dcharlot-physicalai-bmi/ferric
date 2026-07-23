//! det_perf — what does determinism cost at real model shapes?
//!
//! Times the deterministic kernel set at transformer-scale shapes (not the
//! probe's 12×64 toys): warmup then N timed iterations, wall-clock per op.
//! The storage-chain kernels (rmsnorm/layernorm/softmax) pay real memory
//! traffic for their pins; the matmul MAC pays XOR barriers. This measures
//! the price honestly so the fast-path/det-path decision is data, not vibes.
//!
//! Timing method: N dispatches are queued back-to-back and ONE readback
//! syncs at the end — per-op cost excludes the 8 MB readback that otherwise
//! dominates (the earlier per-op-readback numbers were transfer-bound).
//!
//!   cargo run --release -p ferric-core --example det_perf

use ferric_core::Context;
use std::time::Instant;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f32::from_bits(0x3F80_0000 | (s >> 41) as u32) - 1.5
        })
        .collect()
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = Context::new().await.expect("gpu context");
    println!("fabric: {:?} ({})", ctx.backend, ctx.adapter_name);

    // ── matmul 512×512×512 (0.27 GFLOP/op) ──
    let n = 512usize;
    let a = det(n * n, 1);
    let b = det(n * n, 2);
    let at = ctx.tensor(&a, &[n, n]);
    let bt = ctx.tensor(&b, &[n, n]);
    for _ in 0..3 {
        let c = ctx.mm(&at, &bt, n as u32, n as u32, n as u32);
        ctx.to_vec(&c).await.unwrap();
    }
    let iters = 20;
    let t0 = Instant::now();
    let mut last = ctx.mm(&at, &bt, n as u32, n as u32, n as u32);
    for _ in 1..iters {
        last = ctx.mm(&at, &bt, n as u32, n as u32, n as u32);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gflops = 2.0 * (n as f64).powi(3) / (ms / 1000.0) / 1e9;
    println!("matmul  512³          {ms:8.2} ms/op  {gflops:7.1} GFLOP/s");

    // ── rmsnorm 512 rows × 4096 ──
    let (rows, d) = (512usize, 4096usize);
    let x = det(rows * d, 3);
    let w: Vec<f32> = det(d, 4).iter().map(|v| v + 1.0).collect();
    let xt = ctx.tensor(&x, &[rows, d]);
    let wt = ctx.tensor(&w, &[d]);
    for _ in 0..3 {
        let y = ctx.rmsnorm_t(&xt, &wt, rows as u32, d as u32, 1e-5);
        ctx.to_vec(&y).await.unwrap();
    }
    let t0 = Instant::now();
    let mut last = ctx.rmsnorm_t(&xt, &wt, rows as u32, d as u32, 1e-5);
    for _ in 1..iters {
        last = ctx.rmsnorm_t(&xt, &wt, rows as u32, d as u32, 1e-5);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gbs = (rows * d * 4 * 2) as f64 / (ms / 1000.0) / 1e9;
    println!("rmsnorm 512×4096      {ms:8.2} ms/op  {gbs:7.1} GB/s eff");

    // ── rmsnorm-TREE 512 × 4096 (the roadmap kernel) ──
    for _ in 0..3 {
        let y = ctx.rmsnorm_tree_t(&xt, &wt, rows as u32, d as u32, 1e-5);
        ctx.to_vec(&y).await.unwrap();
    }
    let t0 = Instant::now();
    let mut last = ctx.rmsnorm_tree_t(&xt, &wt, rows as u32, d as u32, 1e-5);
    for _ in 1..iters {
        last = ctx.rmsnorm_tree_t(&xt, &wt, rows as u32, d as u32, 1e-5);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gbs = (rows * d * 4 * 2) as f64 / (ms / 1000.0) / 1e9;
    println!("rms-TREE 512×4096     {ms:8.2} ms/op  {gbs:7.1} GB/s eff");

    // ── layernorm 512 × 4096 ──
    let bsv = det(d, 5);
    let bt2 = ctx.tensor(&bsv, &[d]);
    for _ in 0..3 {
        let y = ctx.layernorm_t(&xt, &wt, &bt2, rows as u32, d as u32, 1e-5);
        ctx.to_vec(&y).await.unwrap();
    }
    let t0 = Instant::now();
    let mut last = ctx.layernorm_t(&xt, &wt, &bt2, rows as u32, d as u32, 1e-5);
    for _ in 1..iters {
        last = ctx.layernorm_t(&xt, &wt, &bt2, rows as u32, d as u32, 1e-5);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("layernorm 512×4096    {ms:8.2} ms/op");

    // ── layernorm-TREE 512 × 4096 ──
    for _ in 0..3 {
        let y = ctx.layernorm_tree_t(&xt, &wt, &bt2, rows as u32, d as u32, 1e-5);
        ctx.to_vec(&y).await.unwrap();
    }
    let t0 = Instant::now();
    let mut last = ctx.layernorm_tree_t(&xt, &wt, &bt2, rows as u32, d as u32, 1e-5);
    for _ in 1..iters {
        last = ctx.layernorm_tree_t(&xt, &wt, &bt2, rows as u32, d as u32, 1e-5);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("ln-TREE 512×4096      {ms:8.2} ms/op");

    // ── softmax 512 × 4096 ──
    for _ in 0..3 {
        let y = ctx.softmax_t(&xt, rows as u32, d as u32);
        ctx.to_vec(&y).await.unwrap();
    }
    let t0 = Instant::now();
    let mut last = ctx.softmax_t(&xt, rows as u32, d as u32);
    for _ in 1..iters {
        last = ctx.softmax_t(&xt, rows as u32, d as u32);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("softmax 512×4096      {ms:8.2} ms/op");

    // ── causal MHA T=256, H=8, dh=64 ──
    let (tt, h, dh) = (256usize, 8usize, 64usize);
    let q = det(tt * h * dh, 6);
    let k = det(tt * h * dh, 7);
    let v = det(tt * h * dh, 8);
    let qt = ctx.tensor(&q, &[tt, h * dh]);
    let kt = ctx.tensor(&k, &[tt, h * dh]);
    let vt = ctx.tensor(&v, &[tt, h * dh]);
    for _ in 0..3 {
        let y = ctx.mha_causal_t(&qt, &kt, &vt, tt as u32, h as u32, h as u32, dh as u32);
        ctx.to_vec(&y).await.unwrap();
    }
    let t0 = Instant::now();
    let mut last = ctx.mha_causal_t(&qt, &kt, &vt, tt as u32, h as u32, h as u32, dh as u32);
    for _ in 1..iters {
        last = ctx.mha_causal_t(&qt, &kt, &vt, tt as u32, h as u32, h as u32, dh as u32);
    }
    ctx.to_vec(&last).await.unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("mha T=256 H=8 dh=64   {ms:8.2} ms/op");
}
