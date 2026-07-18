//! Two-pass quant prefill: dequant Q2_0 → f32[K,N], then the fast f32 coop GEMM (9.4 TFLOP/s),
//! vs the scalar dequant-in-kernel Q2_0 matmul. The scalar path is dequant-bound (~1.2 TFLOP/s);
//! two-pass pays the dequant once and lets the tensor cores run at full speed. Measures whether
//! that amortizes across realistic prefill M on NVIDIA.
use ferric_core::Context;
use ferric_gguf::quant_q2_0;
use ferric_tensor::{Q2_0Weights, Tensor};
use std::sync::Arc; use std::time::Instant;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} · {} · coop={}", ctx.backend, ctx.adapter_name, ctx.coop_gemm_ok());
    if !ctx.coop_gemm_ok() { println!("⏭  no coop GEMM here"); return; }
    let (k, n) = (5120usize, 5120usize);
    let wf: Vec<f32> = (0..n * k).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
    let mut packed = Vec::new();
    for r in 0..n { packed.extend(quant_q2_0(&wf[r * k..(r + 1) * k])); }
    let qw = Q2_0Weights::from_bytes(&ctx, &packed, n, k);
    println!("  weight {k}×{n}  ·  two-pass = dequant→f32[K,N] + coop GEMM\n");
    println!("  {:>5} {:>12} {:>12} {:>10} {:>9}", "M", "scalar GF/s", "2pass GF/s", "speedup", "rel|Δ|");
    for m in [64usize, 128, 256, 512, 1024, 2048] {
        let x = Tensor::from_vec(&ctx, &(0..m * k).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[m, k]);
        let two = x.matmul_q2_0_coop2pass(&qw).to_vec().await;
        let scalar = x.matmul_q2_0(&qw).to_vec().await;
        let e = two.iter().zip(&scalar).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let sc = scalar.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t = Instant::now();
            for _ in 0..30 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64() / 30.0 };
        let _ = x.matmul_q2_0_coop2pass(&qw).to_vec().await;
        let tt = bench(&|| x.matmul_q2_0_coop2pass(&qw));
        let st = bench(&|| x.matmul_q2_0(&qw));
        let flop = 2.0 * (m as f64) * (k as f64) * (n as f64);
        println!("  {m:>5} {:>12.0} {:>12.0} {:>9.1}× {:>9.1e}", flop / st / 1e9, flop / tt / 1e9, st / tt, e / sc);
    }
    println!("\n  (two-pass includes the per-call dequant; a cached-f16 weight would drop that)");
}
