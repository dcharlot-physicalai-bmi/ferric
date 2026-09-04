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

        // warm up + validate equality
        let warm = Instant::now();
        let t = a.matmul_tiled(&b).to_vec().await;
        let one = warm.elapsed().as_secs_f64();

        // ⛔ `iters` WAS A CONSTANT 30, AND ON A SOFTWARE RASTERIZER THAT COST 113 MINUTES OF CI.
        // llvmpipe sustains ~2 GFLOP/s, so one 4096³ matmul is ~69 s; 30 iterations across three
        // kernels was 96% of the entire Linux validation step — to report throughput figures that
        // mean nothing on a CPU. The count is now DERIVED from a measured single iteration against a
        // wall-clock budget, so fast fabrics still get 30 samples and slow ones get as few as one.
        // The equality check above is unaffected and still runs at every size: correctness is the
        // part worth paying for here, the GFLOP/s number is not.
        const BUDGET_S: f64 = 2.0;   // per kernel, per size
        let iters = ((BUDGET_S / one.max(1e-6)).floor() as usize).clamp(1, 30);
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

        // n= is not decoration: a throughput number from ONE sample and from thirty must not print
        // identically, or a reader cannot tell a measurement from a single timing.
        println!("  {d}⁴:  reg-tiled {:>7.1}   naive {:>7.1}   tiled {:>7.1} GFLOP/s   rt/naive {:.2}×   max|Δ|={:.1e}  (n={iters})",
            flop / rt_s / 1e9, flop / naive_s / 1e9, flop / tiled_s / 1e9, naive_s / rt_s, diff.max(rtdiff));
        assert!(diff < 1e-2 && rtdiff < 1e-2, "kernels disagree");
    }
    println!("✅ Tiled matmul validated + benchmarked (shared-memory GEMM fast-path)");
}
