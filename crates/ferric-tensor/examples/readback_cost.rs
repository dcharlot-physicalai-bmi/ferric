//! **What one GPU→CPU round trip costs, because expert streaming needs one per layer per token.**
//!
//! `docs/expert-streaming-decision.md` makes this measurement a gate. Streaming routed experts
//! requires knowing which experts a layer chose *on the CPU*, so it can fetch the misses — and that
//! is a readback. `Tensor::moe_topk` exists specifically to delete that readback
//! (`dtype.rs:1710`, "kills the per-layer CPU readback sync") and `qwen35.rs` records the result as
//! "Zero syncs". So streaming re-introduces, once per (token, MoE layer), exactly the stall the FFN
//! fast path was written to remove. This prices it.
//!
//! ## What is measured, and why it is a LOWER BOUND
//!
//! A round trip here is `submit → map → await → read`, on a tensor the size a router selection
//! actually is (`[T, 2k]` — 16 floats for one token at top-8, i.e. 64 bytes). At that size the cost
//! is pure latency; bytes are irrelevant.
//!
//! ⚠ **The real cost is higher than this number.** In a forward, the sync lands mid-pipeline: it
//! drains work already enqueued and stalls the layers behind it, and `ferric_tensor::batch` defers
//! submission for a whole layer at a time (`lib.rs:1802`), so a mid-layer readback forces an early
//! flush and forfeits the batching that removed ~38 ms/token. This example deliberately measures the
//! round trip in isolation instead of trying to model that, because a clean lower bound is more
//! useful than an unvalidated estimate — and if the lower bound already dominates, the decision is
//! made without needing the exact figure.
//!
//! ⚠ It also prints a WARM and a COLD figure. The first readback after other GPU work pays for
//! draining that work; a readback in a tight loop does not. Streaming's syncs are all of the first
//! kind, so the cold column is the relevant one and reporting only the warm one would flatter it.
//!
//!   cargo run -p ferric-tensor --example readback_cost --release
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn spread(v: &[f64]) -> f64 {
    let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    if lo <= 0.0 { f64::INFINITY } else { hi / lo }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("GPU round-trip cost — the sync expert streaming would pay per (token, layer)\n");

    // A router selection row: [T, 2k] for T=1, k=8. 64 bytes.
    let sel = Tensor::from_vec(&ctx, &vec![1.0f32; 16], &[1, 16]);
    // Something to make the queue non-empty, so the COLD case is a real drain rather than a no-op.
    let busy = Tensor::from_vec(&ctx, &vec![0.5f32; 1 << 20], &[1024, 1024]);

    let mut warm = Vec::new();
    for _ in 0..200 {
        let t = Instant::now();
        let v = sel.to_vec().await;
        warm.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(v[0]);
    }
    let (warm_spread, warm_ms) = (spread(&warm), median(&mut warm));

    let mut cold = Vec::new();
    for _ in 0..60 {
        // Enqueue real work first, then read back — which is what a mid-forward sync does.
        let w = busy.mul(&busy).rmsnorm_weightless(1e-5);
        let t = Instant::now();
        let v = sel.to_vec().await;
        cold.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((v[0], w.shape[0]));
    }
    let (cold_spread, cold_ms) = (spread(&cold), median(&mut cold));

    println!("  {:<34} {:>10}  {:>9}", "round trip", "ms", "spread");
    println!("  {:-<56}", "");
    println!("  {:<34} {:>10.4}  {:>8.2}x", "warm (tight loop, idle queue)", warm_ms, warm_spread);
    println!("  {:<34} {:>10.4}  {:>8.2}x", "cold (queue has pending work)", cold_ms, cold_spread);

    // Qwen3.6-35B-A3B: every block carries a routed FFN.
    for (name, layers) in [("Qwen3.6-35B-A3B (measured)", 40usize), ("a 60-layer MoE", 60)] {
        println!("\n  {name}: {layers} MoE layers");
        for (label, ms) in [("warm", warm_ms), ("cold", cold_ms)] {
            let per_tok = ms * layers as f64;
            println!("    {label:<5} {:>8.1} ms/token of pure sync  →  ceiling {:>7.1} tok/s \
                      even if fetch and compute were FREE", per_tok, 1000.0 / per_tok);
        }
    }

    println!("\n  ⚠ LOWER BOUND. A real streaming sync also drains the enqueued layer and forfeits");
    println!("     the per-layer batching that removed ~38 ms/token (qwen35.rs). The true cost is");
    println!("     above the cold row, never below it.");
    println!("\n  The gate in docs/expert-streaming-decision.md: if this dominates, per-expert");
    println!("  residency does not pay for its syncs and Ferric should stream whole LAYERS instead —");
    println!("  PrefetchCache already does that, and needs NO readback because layer order is known.");
}
