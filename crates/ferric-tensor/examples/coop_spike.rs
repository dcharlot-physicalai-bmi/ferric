//! Prove the cooperative-matrix (tensor-core) path: 8×8 matmul on the hardware matrix unit vs CPU.
use ferric_core::Context;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("backend {:?} · {} · coop_matrix={} subgroups={}", ctx.backend, ctx.adapter_name, ctx.coop_matrix, ctx.subgroups);
    if !ctx.coop_gemm_ok() { println!("⏭  coop matrix not usable here"); return; }
    let a: Vec<f32> = (0..64).map(|i| ((i as f32 * 0.1).sin())).collect();
    let b: Vec<f32> = (0..64).map(|i| ((i as f32 * 0.07).cos())).collect();
    let got = ferric_tensor::run_coop_matmul_test(&ctx, &a, &b).await;
    // CPU reference C = A·B, row-major 8×8
    let mut exp = vec![0f32; 64];
    for i in 0..8 { for j in 0..8 { let mut s = 0f32; for k in 0..8 { s += a[i*8+k]*b[k*8+j]; } exp[i*8+j] = s; } }
    let e = got.iter().zip(&exp).map(|(x,y)|(x-y).abs()).fold(0f32, f32::max);
    println!("{} coop_mat 8×8 matmul vs CPU · max|Δ| = {e:.2e}", if e < 1e-4 { "✅ tensor-core path WORKS" } else { "❌" });
    println!("  got[0..4]={:?}", &got[0..4]);
    assert!(e < 1e-4);
}
