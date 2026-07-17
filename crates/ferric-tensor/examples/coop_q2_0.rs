//! Quant-coop prefill: Q2_0 matmul on the tensor-core matrix unit (dequant tile → coop) vs the
//! scalar Q2_0 kernel. This is where the 6-32× meets a real ternary model's prefill.
use ferric_core::Context;
use ferric_gguf::quant_q2_0;
use ferric_tensor::{Q2_0Weights, Tensor};
use std::sync::Arc; use std::time::Instant;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} · {} · coop={}", ctx.backend, ctx.adapter_name, ctx.coop_gemm_ok());
    if !ctx.coop_shared_ok() { println!("⏭  coop-from-shared not usable here (Metal-only; NVIDIA SPIR-V bug)"); return; }
    // prefill shape: M tokens × K in → N out (Bonsai-ish layer)
    for (m, k, n) in [(64usize, 2048usize, 2048usize), (256, 5120, 5120)] {
        let wf: Vec<f32> = (0..n * k).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
        let mut packed = Vec::new();
        for r in 0..n { packed.extend(quant_q2_0(&wf[r * k..(r + 1) * k])); }
        let qw = Q2_0Weights::from_bytes(&ctx, &packed, n, k);
        let x = Tensor::from_vec(&ctx, &(0..m * k).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[m, k]);
        let coop = x.matmul_q2_0_coop(&qw).to_vec().await;
        let scalar = x.matmul_q2_0(&qw).to_vec().await;
        let e = coop.iter().zip(&scalar).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let sc = scalar.iter().map(|v|v.abs()).fold(1e-3,f32::max);
        let bench = |f:&dyn Fn()->Tensor| { let mut l=None; let t=Instant::now();
            for _ in 0..30 { l=Some(f()); } let _=pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64()/30.0 };
        let _ = x.matmul_q2_0_coop(&qw).to_vec().await;
        let ct = bench(&|| x.matmul_q2_0_coop(&qw));
        let st = bench(&|| x.matmul_q2_0(&qw));
        let flop = 2.0*(m as f64)*(k as f64)*(n as f64);
        println!("  [{m}×{k}]·[{k}×{n}]: coop {:.0} GFLOP/s  scalar {:.0}  {:.1}×  rel|Δ|={:.1e}",
            flop/ct/1e9, flop/st/1e9, st/ct, e/sc);
        assert!(e/sc < 6e-2, "coop q2_0 diverged");
    }
    println!("✅ Q2_0 tensor-core prefill matmul validated + benchmarked");
}
