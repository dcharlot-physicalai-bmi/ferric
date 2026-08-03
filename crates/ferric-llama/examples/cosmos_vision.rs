//! Run Cosmos 3 Edge's SigLIP vision encoder + projector on Ferric. Feeds deterministic synthetic
//! patches (reproducible in a numpy reference) and prints the [64, 2048] vision tokens for comparison.
//! usage: cosmos_vision <cosmos3-edge-dir>
use ferric_core::Context;
use ferric_llama::cosmos::CosmosVision;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let dir = std::env::args().nth(1).expect("usage: cosmos_vision <dir>");
    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let m = CosmosVision::load(&ctx, &dir).unwrap();
    println!("loaded Cosmos vision encoder in {:?} (27 SigLIP layers + projector)", t0.elapsed());

    // Deterministic synthetic patches [256, 768] — the numpy reference regenerates these identically.
    let patches: Vec<f32> = (0..256 * 768).map(|i| ((i as f32 * 0.017).sin() * 0.5)).collect();
    let t0 = std::time::Instant::now();
    let out = m.forward(&patches).to_vec().await; // [64, 2048]
    println!("forward {:?} · out shape [64, 2048]", t0.elapsed());
    println!("  token0[:6] = {:?}", &out[..6].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("  token63[:6] = {:?}", &out[63 * 2048..63 * 2048 + 6].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    let sum: f64 = out.iter().map(|&x| x as f64).sum();
    println!("  sum = {sum:.4} · finite = {}", out.iter().all(|x| x.is_finite()));
}
