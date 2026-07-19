//! Isolate which dimension breaks the RB coop GEMM. a=0.01, b=0.02 → every c[i] = K·2e-4.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} coop={}", ctx.backend, ctx.coop_gemm_ok());
    // M multiple of 16 → RB kernel; M=8 (not 16-aligned) → the plain COOP_GEMM_WGSL 8×8 kernel.
    for (m, k, n) in [(512usize, 512usize, 512usize), (16, 2048, 2048), (2048, 5120, 5120),
                      (8, 2048, 2048), (8, 5120, 5120), (8, 512, 512), (24, 512, 512)] {
        let a = Tensor::from_vec(&ctx, &vec![0.01f32; m * k], &[m, k]);
        let b = Tensor::from_vec(&ctx, &vec![0.02f32; k * n], &[k, n]);
        let c = a.matmul_coop(&b).to_vec().await;
        let expect = k as f32 * 0.01 * 0.02;
        let e = c.iter().map(|v| (v - expect).abs()).fold(0f32, f32::max);
        let ok = e / expect < 0.05;
        println!("  [{m:>4}×{k:>4}]·[{k}×{n}]  c[0]={:.4} expect {:.4}  relΔ={:.2e}  {}",
            c[0], expect, e / expect, if ok { "OK" } else { "❌ WRONG" });
    }
}
