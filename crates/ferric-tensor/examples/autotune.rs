//! Per-device GEMM autotuning: for each shape, measure the naive vs register-blocked-tiled kernel
//! and cache the winner; `matmul` then dispatches the measured-fastest kernel automatically. No
//! single kernel wins on every GPU — this is how GEMM stays fast *portably* across the fabric.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.001 + s).sin()) * 0.1).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Ferric GEMM autotuner · {:?}", ctx.adapter_name);
    for &d in &[256usize, 512, 1024] {
        let a = Tensor::from_vec(&ctx, &seq(d * d, 1.0), &[d, d]);
        let b = Tensor::from_vec(&ctx, &seq(d * d, 2.0), &[d, d]);
        let choice = a.autotune_matmul(&b).await;
        // matmul now auto-dispatches the winner for this shape bucket; verify correctness
        let got = a.matmul(&b).to_vec().await;
        let refv = a.matmul_naive(&b).to_vec().await;
        let diff = got.iter().zip(&refv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("  {d}×{d}×{d}: autotuner selected {:<5}  · matmul dispatches it · max|Δ|={:.1e}", choice, diff);
        assert!(diff < 1e-2);
    }
    println!("✅ GEMM autotuning: matmul selects the measured-fastest kernel per device+shape");
}
