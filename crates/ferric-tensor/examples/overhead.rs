//! How much does a dispatch cost before any real work happens? Bonsai runs ~600+ ops per token, so
//! per-op overhead is multiplied by a large constant.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let a = Tensor::from_vec(&ctx, &vec![1.0f32; 256], &[16, 16]);
    let _ = a.mul(&a).to_vec().await; // warm
    for n in [100usize, 600] {
        let t0 = std::time::Instant::now();
        let mut last = None;
        for _ in 0..n { last = Some(a.mul(&a)); }
        let _ = last.unwrap().to_vec().await;
        let dt = t0.elapsed().as_secs_f64();
        println!("  {n:>4} tiny ops (queued, 1 sync): {:>7.2} ms total → {:>6.3} ms/op", dt * 1e3, dt * 1e3 / n as f64);
    }
    // how many ops does one Bonsai decode step issue? ~600-700; multiply through
    println!("\n  Bonsai issues ~640 ops/token → per-op overhead is paid 640x per token");
}
