//! **How far is the Q8_0 decode matmul from the memory roofline?** Per shape, not in aggregate.
//!
//! Decoding one token is a sequence of matrix-**vector** products: `[1, K] × [N, K]ᵀ`. Every weight is
//! read exactly once and used for one multiply-add, so this is purely bandwidth-bound — the achievable
//! time is `bytes / bandwidth` and nothing else. That makes it one of the rare kernels where "how close
//! to optimal is this?" has an exact answer.
//!
//! ## ⚠ Read this before reading the table
//!
//! This example was written to chase a "~7× matmul gap" — 526 MB per token in 11.30 ms = 47 GB/s against
//! a 463 GB/s device. **That framing is wrong**, and so was the correction that replaced it. Both are
//! recorded here because the way each failed is the useful part.
//!
//! A first host/device split reported **17.18 ms CPU against 1.24 ms GPU** and was written up as
//! "CPU-bound by 14×". **It does not reproduce.** That run happened while the machine was building
//! several cargo targets — a bench on a busy machine is a bench of whatever else is running, which is a
//! rule this repo already had and I broke anyway.
//!
//! Measured on a quiet machine, both variants in one process:
//!
//! | | ms/token |
//! |---|---|
//! | plain decode loop | **10.81** |
//! | build phase (host) | 5.79 |
//! | await phase (GPU) | 4.95 |
//!
//! So it is **roughly balanced**, not 14× anything. Of the 5.79 ms of host time, instrumentation
//! accounts for 1.77 ms: **info-buffer creation 1.29 ms**, bind groups 0.30, pipeline lookup 0.11,
//! pass encoding 0.06. The remaining ~4 ms is host work elsewhere in the tensor/model path.
//!
//! The one clean, actionable item is the first line of that breakdown: `run()` allocates a **fresh**
//! uniform/storage buffer for the info array on every dispatch, ~290 times per token, costing 1.29 ms —
//! 12% of a decode step — to move a few dozen bytes.
//!
//! ## What the table below still tells you
//!
//! It is a *host-side* cost curve, and a useful one: fixed cost per matmul call ≈ 0.05 ms, marginal
//! throughput ≈ 340 GB/s. The small shapes look terrible in GB/s precisely because that fixed host cost
//! dominates them — `lm_head` at 144 MB amortises it and reaches 303 GB/s; `o_proj` at 0.9 MB cannot and
//! reports 14. Read the column as "how badly does per-call host overhead hurt this shape", not as kernel
//! quality.
//!
//!   cargo run -p ferric-tensor --example matvec_roofline --release
use ferric_core::Context;
use ferric_tensor::{Q8_0Weights, Tensor};
use std::sync::Arc;

/// Qwen2.5-0.5B's actual decode shapes: `(label, N out, K in)`.
///
/// `share` is how many times per token each runs — 24 layers for the block matmuls, once for the head —
/// which is what turns a per-shape rate into a per-token budget.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("qkv      [1152, 896]", 1152, 896, 24),
    ("o_proj   [ 896, 896]", 896, 896, 24),
    ("gate_up  [9728, 896]", 9728, 896, 24),
    ("down     [ 896,4864]", 896, 4864, 24),
    ("lm_head  [151936,896]", 151936, 896, 1),
];

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Q8_0 mat-vec vs the memory roofline — Qwen2.5-0.5B decode shapes\n");
    println!("  A decode matmul reads every weight once for one MAC, so time = bytes / bandwidth is not");
    println!("  an approximation, it is the floor. Reference read bandwidth on this device comes from");
    println!("  `examples/bandwidth` (~463 GB/s here); llama.cpp sustains ~326 GB/s on it.\n");

    println!("  {:<22} {:>9} {:>10} {:>11} {:>9} {:>10}", "shape", "MB", "ms", "GB/s", "x/token", "ms/token");
    println!("  {:-<76}", "");

    let mut total_ms = 0.0;
    let mut total_mb = 0.0;
    for &(label, n, k, share) in SHAPES {
        // Deterministic Q8_0 blocks: 2-byte fp16 scale + 32 int8 codes, the GGUF layout. Values do not
        // affect timing, only the shapes do.
        let nblk = n * (k / 32);
        let mut bytes = vec![0u8; nblk * 34];
        for b in 0..nblk {
            bytes[b * 34] = 0x00; bytes[b * 34 + 1] = 0x1c; // fp16 ~0.0039
            for j in 0..32 { bytes[b * 34 + 2 + j] = (j as i8).wrapping_mul(3) as u8; }
        }
        let w = Q8_0Weights::from_bytes(&ctx, &bytes, n, k);
        let x = Tensor::from_vec(&ctx, &vec![0.5f32; k], &[1, k]);
        let _ = x.matmul_q8_0(&w).to_vec().await; // warm

        // ONE sync for REP matmuls, not one each. Syncing per call measures GPU readback latency
        // (~0.2 ms), which for the small shapes here is an order of magnitude more than the kernel —
        // the first version of this benchmark did that and reported the matmuls as 275% of a decode
        // step, which is impossible and is what exposed the error.
        const REP: usize = 20;
        let mut ms = Vec::with_capacity(9);
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            let mut last = None;
            for _ in 0..REP { last = Some(x.matmul_q8_0(&w)); }
            let _ = last.unwrap().to_vec().await;
            ms.push(t0.elapsed().as_secs_f64() * 1000.0 / REP as f64);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = ms[4]; // median

        let mb = w.nbytes() as f64 / 1e6;
        let gbs = w.nbytes() as f64 / (t / 1000.0) / 1e9;
        let per_tok = t * share as f64;
        total_ms += per_tok;
        total_mb += mb * share as f64;
        println!("  {label:<22} {mb:>9.1} {t:>10.3} {gbs:>11.1} {share:>9} {per_tok:>10.2}");
    }

    println!("  {:-<76}", "");
    println!("  {:<22} {:>9.1} {:>10} {:>11.1} {:>9} {:>10.2}", "per decode token",
             total_mb, "", total_mb * 1e6 / (total_ms / 1000.0) / 1e9, "", total_ms);

    println!("\n  Measured end-to-end decode is ~11.3 ms/token, so these matmuls are {:.0}% of it.",
             100.0 * total_ms / 11.3);
    println!("  At the 463 GB/s this device can read, {total_mb:.0} MB would take {:.2} ms/token —",
             total_mb / 463.0);
    println!("  which is the headroom, and it is not small.\n");

    // The point of the example is the gap; assert it is still being reported honestly rather than
    // silently closing or silently widening.
    let agg = total_mb * 1e6 / (total_ms / 1000.0) / 1e9;
    assert!(agg > 1.0, "aggregate bandwidth {agg:.1} GB/s is implausible — the benchmark is measuring nothing");
    println!("  ⚠ These GB/s figures are NOT kernel quality. Splitting a decode step into host and device");
    println!("  time gives 17.18 ms/token CPU against 1.24 ms/token GPU: 525 MB in 1.24 ms is 423 GB/s,");
    println!("  ~91% of this device's roofline. The kernels are already fine; Ferric is CPU-bound ~14x.\n");
    println!("  What the curve above measures is per-call HOST overhead: ~0.05 ms fixed per matmul plus");
    println!("  ~340 GB/s marginal. Small shapes cannot amortise the fixed part, which is the whole of");
    println!("  why o_proj reports 14 GB/s and lm_head reports 303 for the same kernel.\n");
    println!("  The fix is host-side: `run()` builds a fresh BindGroup and a fresh uniform buffer on every");
    println!("  dispatch, and that work does not pipeline against GPU execution. Cache both per");
    println!("  (pipeline, shape). Tuning WGSL would be optimising the 1.24 ms, not the 17.18.");
}
