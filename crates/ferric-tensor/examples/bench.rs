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
    for &d in &[512usize, 1024, 2048, 4096] {
        let a = Tensor::from_vec(&ctx, &seq(d * d, 1.0), &[d, d]);
        let b = Tensor::from_vec(&ctx, &seq(d * d, 2.0), &[d, d]);
        let flop = 2.0 * (d as f64).powi(3);
        let iters = 30;

        // warm up + validate equality
        let t = a.matmul_tiled(&b).to_vec().await;
        let nv = a.matmul_naive(&b).to_vec().await;
        let diff = t.iter().zip(&nv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);

        // Queue all iters, sync ONCE — awaiting each iter would time submit+fence (~1ms), not the
        // kernel, and at these sizes that overhead is a large fraction of the measurement.
        let bench = |f: &dyn Fn() -> Tensor| {
            let mut last = None;
            let t0 = Instant::now();
            for _ in 0..iters { last = Some(f()); }
            let _ = pollster::block_on(last.unwrap().to_vec());
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let rt = a.matmul_rt(&b).to_vec().await;
        let rtdiff = rt.iter().zip(&nv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let tiled_s = bench(&|| a.matmul_tiled(&b));
        let naive_s = bench(&|| a.matmul_naive(&b));
        let rt_s = bench(&|| a.matmul_rt(&b));

        println!("  {d}⁴:  reg-tiled {:>7.1}   naive {:>7.1}   tiled {:>7.1} GFLOP/s   rt/naive {:.2}×   max|Δ|={:.1e}",
            flop / rt_s / 1e9, flop / naive_s / 1e9, flop / tiled_s / 1e9, naive_s / rt_s, diff.max(rtdiff));
        assert!(diff < 1e-2 && rtdiff < 1e-2, "kernels disagree");
    }
    println!("✅ Tiled matmul validated + benchmarked (shared-memory GEMM fast-path)");
}
