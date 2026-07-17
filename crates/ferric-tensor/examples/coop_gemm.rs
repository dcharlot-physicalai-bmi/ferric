//! Full cooperative-matrix GEMM (C=A·B on the hardware matrix unit) — validate vs naive + benchmark.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.001 + s).sin()) * 0.1).collect() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} · {} · coop_matrix={}", ctx.backend, ctx.adapter_name, ctx.coop_matrix);
    if !ctx.coop_gemm_ok() { println!("⏭  coop GEMM not usable here (Vulkan SPIR-V backend bug)"); return; }
    for &d in &[512usize, 1024, 2048] {
        let a = Tensor::from_vec(&ctx, &seq(d * d, 1.0), &[d, d]);
        let b = Tensor::from_vec(&ctx, &seq(d * d, 2.0), &[d, d]);
        let coop = a.matmul_coop(&b).to_vec().await;
        let nv = a.matmul_naive(&b).to_vec().await;
        let e = coop.iter().zip(&nv).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let flop = 2.0 * (d as f64).powi(3);
        let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t = Instant::now();
            for _ in 0..20 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64() / 20.0 };
        let _ = a.matmul_coop(&b).to_vec().await; // warm
        let ct = bench(&|| a.matmul_coop(&b));
        let nt = bench(&|| a.matmul_naive(&b));
        let prec = if e < 1e-4 { "exact f32" } else { "TF32" };
        println!("  {d}³: coop {:>7.1} GFLOP/s   naive {:>7.1}   coop/naive {:.2}×   max|Δ|={:.1e} ({prec})",
            flop / ct / 1e9, flop / nt / 1e9, nt / ct, e);
        assert!(e < 6e-2, "coop diverged beyond TF32 tolerance");
    }
    println!("✅ cooperative-matrix GEMM on the hardware matrix unit — Metal exact f32, NVIDIA TF32");
}
