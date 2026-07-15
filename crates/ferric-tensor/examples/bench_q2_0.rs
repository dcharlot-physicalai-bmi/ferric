//! Benchmark `matmul_q2_0` at Bonsai-27B's real shapes, and report achieved weight-read bandwidth.
//! Decode of a ternary LLM is memory-bound: every token must read the whole weight, so
//! bytes/second against the roofline is the number that matters, not FLOPs.
use ferric_core::Context;
use ferric_gguf::quant_q2_0;
use ferric_tensor::{Q2_0Weights, Tensor};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Ferric · matmul_q2_0 at Bonsai-27B shapes");
    println!("  kernel: {}\n", std::env::var("FERRIC_Q2_0_KERNEL").unwrap_or_else(|_| "auto (per-shape)".into()));
    println!("  {:<22} {:>5} {:>10} {:>11} {:>12}", "shape (in→out)", "toks", "packed", "time", "GB/s");

    // (in, out) for the projections that dominate a Bonsai layer, plus the LM head — at 338 MB the
    // head is far too big to sit in cache, so it's the one shape here that measures true DRAM
    // streaming. The others (~24 MB) fit in the SLC and are re-read across reps, which flatters
    // them: the real model touches each weight once per token, always cold.
    let shapes = [(5120usize, 17408usize, "ffn_gate/up"), (17408, 5120, "ffn_down"), (5120, 10240, "gdn qkv"), (5120, 12288, "attn q"), (5120, 248320, "lm_head (cold)")];
    for (inn, out, name) in shapes {
        // build packed weights once
        let w: Vec<f32> = (0..out * inn).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
        let mut packed = Vec::with_capacity(out * (inn / 128) * 34);
        for r in 0..out { packed.extend(quant_q2_0(&w[r * inn..(r + 1) * inn])); }
        let qw = Q2_0Weights::from_bytes(&ctx, &packed, out, inn);
        let bytes = qw.nbytes();

        for toks in [1usize, 5] {
            let x = Tensor::from_vec(&ctx, &(0..toks * inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[toks, inn]);
            // warm up (shader compile + first dispatch)
            let _ = x.matmul_q2_0(&qw).to_vec().await;
            // Queue the reps and sync ONCE at the end. Awaiting each rep would measure
            // submit+fence+readback latency (~1 ms) rather than the kernel — dispatches are async.
            let reps = if bytes > 100_000_000 { 4 } else { 20 };
            let t0 = std::time::Instant::now();
            let mut last = None;
            for _ in 0..reps { last = Some(x.matmul_q2_0(&qw)); }
            let _ = last.unwrap().to_vec().await;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            // each token must read the full weight once
            let gbs = (bytes * toks) as f64 / dt / 1e9;
            println!("  {:<22} {:>5} {:>9.1}M {:>9.2}ms {:>11.1}", format!("{name} {inn}→{out}"), toks, bytes as f64 / 1e6, dt * 1e3, gbs);
        }
    }
    println!("\n  (M5 Max unified memory is ~400-546 GB/s — that ceiling is what decode is chasing)");
}
