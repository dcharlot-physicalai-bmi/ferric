//! Sanity check for the new sin/cos Tensor + autograd ops: values and first-order gradients vs analytic.
use ferric_tensor::{Tensor, Var};
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let xs = [0.3f32, 1.1, -0.7, 2.5];
    let x = Var::leaf(Tensor::from_vec(&ctx, &xs, &[4]));
    // value check
    let s = x.sin(); let c = x.cos();
    let sv = s.value().to_vec().await; let cv = c.value().to_vec().await;
    for i in 0..4 { assert!((sv[i] - xs[i].sin()).abs() < 1e-5 && (cv[i] - xs[i].cos()).abs() < 1e-5, "value mismatch at {i}"); }
    println!("  sin/cos VALUES ok: sin={:?}", sv.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());
    // grad of sum(sin) wrt x = cos(x)
    let loss = x.sin().sum(&[0]); loss.backward();
    let g = x.grad().unwrap().to_vec().await;
    for i in 0..4 { assert!((g[i] - xs[i].cos()).abs() < 1e-4, "d sin grad mismatch at {i}: {} vs {}", g[i], xs[i].cos()); }
    println!("  d/dx sum(sin) = cos(x) ok: {:?}", g.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());
    // grad of sum(cos) wrt x = -sin(x)
    let x2 = Var::leaf(Tensor::from_vec(&ctx, &xs, &[4]));
    let loss2 = x2.cos().sum(&[0]); loss2.backward();
    let g2 = x2.grad().unwrap().to_vec().await;
    for i in 0..4 { assert!((g2[i] + xs[i].sin()).abs() < 1e-4, "d cos grad mismatch at {i}"); }
    println!("  d/dx sum(cos) = -sin(x) ok: {:?}", g2.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());
    println!("\n  ✓ sin/cos ops correct (value + gradient) — angle composition is now in-graph.");
}
