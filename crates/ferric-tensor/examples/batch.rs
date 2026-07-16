//! Command batching: many dispatches, one submit — same result. Confirms Context::batch is correct
//! (identical output to the unbatched path) and reports the submit reduction.
use ferric_core::Context;
use ferric_tensor::{op_counters, reset_op_counters, Tensor};
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let a = Tensor::from_vec(&ctx, &(0..256).map(|i| i as f32 * 0.01).collect::<Vec<_>>(), &[16, 16]);

    // unbatched: a chain of 50 ops
    let chain = |a: &Tensor| { let mut x = a.clone(); for _ in 0..50 { x = x.mul(a).add(a).relu(); } x };
    reset_op_counters();
    let plain = chain(&a).to_vec().await;
    let (d0, s0) = op_counters();

    reset_op_counters();
    let batched = ferric_tensor::batch(&ctx, || chain(&a)).to_vec().await;
    let (d1, s1) = op_counters();

    let e = plain.iter().zip(&batched).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let ok = e == 0.0 && d0 == d1;
    println!("  unbatched: {d0} dispatches → {s0} submits");
    println!("  batched:   {d1} dispatches → {s1} submits");
    println!("  {} identical result (max|Δ| = {e:.1e}), {}× fewer submits",
        if ok { "✅" } else { "❌" }, if s1 > 0 { s0 / s1 } else { s0 });
    assert!(ok && s1 < s0);
    println!("✅ command batching: N dispatches, 1 submit, same numbers");
}
