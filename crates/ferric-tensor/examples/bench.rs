//! Matmul throughput: the tiled (workgroup-shared-memory) fast-path vs the naive one-thread-per-
//! output kernel. Reports GFLOPS for both and verifies they agree. Fast GEMM is the foundation every
//! SOTA runtime stands on.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.001 + s).sin()) * 0.1).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Ferric matmul benchmark · {:?}", ctx.adapter_name);
    for &d in &[256usize, 512, 1024] {
        let a = Tensor::from_vec(&ctx, &seq(d * d, 1.0), &[d, d]);
        let b = Tensor::from_vec(&ctx, &seq(d * d, 2.0), &[d, d]);
        let flop = 2.0 * (d as f64).powi(3);
        let iters = 20;

        // warm up + validate equality
        let t = a.matmul_tiled(&b).to_vec().await;
        let nv = a.matmul_naive(&b).to_vec().await;
        let diff = t.iter().zip(&nv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);

        let t0 = Instant::now();
        for _ in 0..iters { let _ = a.matmul_tiled(&b).to_vec().await; }
        let tiled_s = t0.elapsed().as_secs_f64() / iters as f64;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = a.matmul_naive(&b).to_vec().await; }
        let naive_s = t0.elapsed().as_secs_f64() / iters as f64;

        println!("  {d}×{d}×{d}:  tiled {:>7.1} GFLOP/s   naive {:>7.1} GFLOP/s   speedup {:.2}×   max|Δ|={:.1e}",
            flop / tiled_s / 1e9, flop / naive_s / 1e9, naive_s / tiled_s, diff);
        assert!(diff < 1e-2, "tiled != naive");
    }
    println!("✅ Tiled matmul validated + benchmarked (shared-memory GEMM fast-path)");
}
