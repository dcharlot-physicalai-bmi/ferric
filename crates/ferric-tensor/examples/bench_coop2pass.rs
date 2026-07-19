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
    println!("{:?} · {} · coop_gemm={} coop16={}", ctx.backend, ctx.adapter_name, ctx.coop_gemm_ok(), ctx.coop16_ok());
    if !ctx.coop_gemm_ok() && !ctx.coop16_ok() { println!("⏭  no coop GEMM here"); return; }
    let (k, n) = (5120usize, 5120usize);
    let wf: Vec<f32> = (0..n * k).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
    let mut packed = Vec::new();
    for r in 0..n { packed.extend(quant_q2_0(&wf[r * k..(r + 1) * k])); }
    let qw = Q2_0Weights::from_bytes(&ctx, &packed, n, k);
    println!("  weight {k}×{n}  ·  two-pass = dequant→f32[K,N] + coop GEMM\n");
    // On Vulkan the 8×8-f32 coop path is zeros; use the f16 tensor-core path (matmul_q2_0_coop16).
    let use16 = ctx.coop16_ok();
    let path = |x: &Tensor| if use16 { x.matmul_q2_0_coop16(&qw) } else { x.matmul_q2_0_coop2pass(&qw) };
    println!("  coop path: {}\n", if use16 { "coop16 (f16 tensor cores, Vulkan)" } else { "coop2pass (8×8 f32, Metal)" });
    println!("  {:>5} {:>12} {:>12} {:>10} {:>9}", "M", "scalar GF/s", "coop GF/s", "speedup", "rel|Δ|");
    // Stop at 1024: the timing loop allocates a fresh dequant buffer per rep, and 30×(105MB f32 +
    // f16 copies) at M=2048 OOMs a 6GB card — a bench artifact, not a limit of a single call.
    for m in [64usize, 128, 256, 512, 1024] {
        let x = Tensor::from_vec(&ctx, &(0..m * k).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[m, k]);
        let two = path(&x).to_vec().await;
        let scalar = x.matmul_q2_0(&qw).to_vec().await;
        let e = two.iter().zip(&scalar).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let sc = scalar.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t = Instant::now();
            for _ in 0..30 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64() / 30.0 };
        let _ = path(&x).to_vec().await;
        let tt = bench(&|| path(&x));
        let st = bench(&|| x.matmul_q2_0(&qw));
        let flop = 2.0 * (m as f64) * (k as f64) * (n as f64);
        println!("  {m:>5} {:>12.0} {:>12.0} {:>9.1}× {:>9.1e}", flop / st / 1e9, flop / tt / 1e9, st / tt, e / sc);
    }
    println!("\n  (coop includes the per-call dequant; rel|Δ| vs scalar is f16-input precision on Vulkan)");
}
