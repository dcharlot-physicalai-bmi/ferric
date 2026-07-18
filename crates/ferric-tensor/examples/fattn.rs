//! Fused single-query attention must equal the composed reshape/permute/matmul/softmax/matmul path.
use ferric_core::Context;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.13 + s).sin()) * 0.3).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    // S=2049/4096/5000 cross the 2048-key chunk boundary — validate the online-softmax combination.
    for (nh, nkv, dh, s) in [(16usize, 8usize, 128usize, 1usize), (16, 8, 128, 7), (16, 8, 128, 64), (16, 16, 64, 300), (8, 2, 96, 129), (16, 8, 128, 2049), (16, 16, 64, 4096), (8, 8, 64, 5000)] {
        let q = Tensor::from_vec(&ctx, &seq(nh * dh, 1.0), &[1, nh * dh]);
        let k = Tensor::from_vec(&ctx, &seq(s * nkv * dh, 2.0), &[s, nkv * dh]);
        let v = Tensor::from_vec(&ctx, &seq(s * nkv * dh, 3.0), &[s, nkv * dh]);
        let fused = q.fused_decode_attention(&k, &v, nh, nkv, dh).to_vec().await;
        // composed reference (bypass the fused fast-path by calling the pieces directly is hard; instead
        // temporarily compare against the math via the OLD composed path — replicate it here):
        let reference = nn::decode_attention_composed(&q, &k, &v, nh, nkv).to_vec().await;
        let e = maxdiff(&fused, &reference);
        let p = e < 2e-4; ok &= p;
        println!("{} nh={nh} nkv={nkv} dh={dh} S={s:<4} max|Δ|={e:.1e}", if p { "✅" } else { "❌" });
    }
    println!("{}", if ok { "✅ fused decode attention == composed path" } else { "❌ diverged" });
    assert!(ok);
}
