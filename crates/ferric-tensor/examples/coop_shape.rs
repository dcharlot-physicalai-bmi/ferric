//! Isolate the naga panic: does matmul_coop compile at the two-pass shape (M=64, non-square)?
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} coop={}", ctx.backend, ctx.coop_gemm_ok());
    for (m, k, n) in [(64usize, 5120usize, 5120usize), (512, 5120, 5120)] {
        let a = Tensor::from_vec(&ctx, &vec![0.01f32; m * k], &[m, k]);
        let b = Tensor::from_vec(&ctx, &vec![0.02f32; k * n], &[k, n]);
        let c = a.matmul_coop(&b).to_vec().await;
        println!("  matmul_coop [{m}×{k}]·[{k}×{n}] OK, c[0]={:.4}", c[0]);
    }
    println!("✅ matmul_coop compiles at these shapes");
}
