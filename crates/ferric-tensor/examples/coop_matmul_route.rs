//! Verify coop routes through the general matmul() under FERRIC_COOP, exact vs naive on Metal.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc; use std::time::Instant;
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.001 + s).sin()) * 0.1).collect() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let d = 2048;
    let a = Tensor::from_vec(&ctx, &seq(d*d, 1.0), &[d, d]);
    let b = Tensor::from_vec(&ctx, &seq(d*d, 2.0), &[d, d]);
    let routed = a.matmul(&b).to_vec().await;         // FERRIC_COOP decides
    let nv = a.matmul_naive(&b).to_vec().await;
    let e = routed.iter().zip(&nv).map(|(x,y)|(x-y).abs()).fold(0f32,f32::max);
    let bench = |f: &dyn Fn()->Tensor| { let mut l=None; let t=Instant::now();
        for _ in 0..20 { l=Some(f()); } let _=pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64()/20.0 };
    let mt = bench(&|| a.matmul(&b));
    let flop = 2.0*(d as f64).powi(3);
    let coop = std::env::var("FERRIC_COOP").is_ok();
    println!("matmul() {d}³: {:.0} GFLOP/s  (FERRIC_COOP={}) max|Δ| vs naive = {:.1e}", flop/mt/1e9, coop, e);
}
