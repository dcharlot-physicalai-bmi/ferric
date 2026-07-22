//! Chain/thrash probe: the qkv pattern (one x through THREE different-shaped linears, repeated)
//! vs the same-shape pattern. Today's single-slot ResidentCache rebuilds on every shape change.
use ferric_core::Context;
use ferric_tensor::{device_sync, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| 0.01 * (((i + salt) % 13) as f32 - 6.0)).collect()
}

fn main() {
    let ctx = Arc::new(pollster::block_on(Context::new()).unwrap());
    std::env::set_var("FERRIC_METAL4", "1");
    let (rows, d) = (256usize, 2048usize);
    let x = Tensor::from_vec(&ctx, &gen(rows * d, 1), &[rows, d]);
    let wq = Tensor::from_vec(&ctx, &gen(2048 * d, 2), &[2048, d]);
    let wk = Tensor::from_vec(&ctx, &gen(512 * d, 3), &[512, d]);
    let wv = Tensor::from_vec(&ctx, &gen(512 * d, 4), &[512, d]);

    // same-shape: 3x q-proj (cache hits after first)
    let _ = pollster::block_on(x.matmul_bt(&wq).to_vec());
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = x.matmul_bt(&wq);
        let _ = x.matmul_bt(&wq);
        let _ = x.matmul_bt(&wq);
    }
    device_sync(&ctx);
    println!("same-shape 3-chain: {:.3} ms/iter", t0.elapsed().as_secs_f64() / 10.0 * 1e3);

    // qkv: 3 different shapes per iter (thrash if single-slot)
    let _ = pollster::block_on(x.matmul_bt(&wv).to_vec());
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = x.matmul_bt(&wq);
        let _ = x.matmul_bt(&wk);
        let _ = x.matmul_bt(&wv);
    }
    device_sync(&ctx);
    println!("qkv 3-chain:        {:.3} ms/iter", t0.elapsed().as_secs_f64() / 10.0 * 1e3);
}
