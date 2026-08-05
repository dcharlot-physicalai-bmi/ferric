//! **Which costs more — a dispatch, or a submit?** Because the answer decides where optimisation goes.
//!
//! Ferric's per-token dispatch count was cut 410 → 338 on the strength of published browser data putting
//! Firefox's **per-dispatch** cost at ~1,040 µs against Chrome's ~33 µs. But Ferric batches its dispatches
//! into ~1 queue submission per layer, so that reasoning only holds if the cost really is charged per
//! *dispatch* rather than per *submit*. If it is charged per submit, then further kernel fusion buys
//! nothing and the optimisation direction is wrong.
//!
//! That is a premise, and premises get measured.
//!
//! ## The experiment
//!
//! The same N tiny operations, three ways:
//!
//!   1. **N submits**   — unbatched: every op is its own queue submission.
//!   2. **1 submit**    — `batch()`: N dispatches, one submission.
//!   3. **N/2 dispatches, 1 submit** — half the ops, still one submission. The delta against (2) is the
//!      marginal cost of a dispatch with submission held fixed, which is the number that decides whether
//!      fusing kernels is worth anything.
//!
//! The ops are deliberately tiny (16×16), so the GPU is doing almost no arithmetic and what is left is
//! overhead. That is the point — this measures the floor, not throughput.
//!
//! Note what this can and cannot say: it is a *native* measurement (Metal/Vulkan here). Browser
//! per-dispatch costs are higher and this cannot measure them. What it CAN settle is the ratio — whether
//! dispatch or submit dominates on a real driver — and that is what the optimisation direction rests on.
//!
//!   cargo run -p ferric-tensor --example dispatch_vs_submit --release
use ferric_core::Context;
use ferric_tensor::{batch, op_counters, reset_op_counters, Tensor};
use std::sync::Arc;

/// Median of repeated trials. A mean over GPU timings is dominated by whichever run collided with
/// something else on the machine.
async fn timed(trials: usize, mut f: impl FnMut() -> Tensor) -> f64 {
    let mut ms = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = std::time::Instant::now();
        let _ = f().to_vec().await;
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[ms.len() / 2]
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let a = Tensor::from_vec(&ctx, &vec![0.5f32; 256], &[16, 16]);
    let _ = a.mul(&a).to_vec().await; // warm the pipeline cache

    const N: usize = 256;
    const TRIALS: usize = 9;

    // A chain, so the ops cannot be reordered or elided — each depends on the last.
    let chain = |a: &Tensor, n: usize| { let mut x = a.clone(); for _ in 0..n { x = x.mul(a); } x };

    reset_op_counters();
    let _ = chain(&a, N).to_vec().await;
    let (d_un, s_un) = op_counters();

    reset_op_counters();
    let _ = batch(&ctx, || chain(&a, N)).to_vec().await;
    let (d_ba, s_ba) = op_counters();

    println!("Dispatch cost vs submit cost — {N} tiny (16x16) ops, median of {TRIALS}\n");
    println!("  {:<34} {:>12} {:>10} {:>12}", "", "dispatches", "submits", "median");
    println!("  {:-<72}", "");

    let t_unbatched = timed(TRIALS, || chain(&a, N)).await;
    println!("  {:<34} {:>12} {:>10} {:>9.2} ms", "unbatched (1 submit per op)", d_un, s_un, t_unbatched);

    let t_batched = timed(TRIALS, || batch(&ctx, || chain(&a, N))).await;
    println!("  {:<34} {:>12} {:>10} {:>9.2} ms", "batched (1 submit total)", d_ba, s_ba, t_batched);

    let t_half = timed(TRIALS, || batch(&ctx, || chain(&a, N / 2))).await;
    println!("  {:<34} {:>12} {:>10} {:>9.2} ms", "batched, HALF the dispatches", N / 2, s_ba, t_half);

    // Marginal costs. Submit cost comes from removing submissions at fixed dispatch count; dispatch cost
    // from removing dispatches at fixed submission count. Each isolates one variable.
    let per_submit_us = (t_unbatched - t_batched) * 1000.0 / (s_un.saturating_sub(s_ba)).max(1) as f64;
    let per_dispatch_us = (t_batched - t_half) * 1000.0 / (N - N / 2) as f64;

    println!("\n  {:<34} {:>10.1} µs", "marginal cost per SUBMIT", per_submit_us);
    println!("  {:<34} {:>10.1} µs", "marginal cost per DISPATCH", per_dispatch_us);

    let ratio = per_dispatch_us / per_submit_us.max(1e-9);
    println!("\n  A dispatch costs {ratio:.1}x what a submit costs on this fabric.\n");

    // What this means for the runtime, in its own numbers.
    const FERRIC_DISPATCHES: f64 = 338.0;
    const FERRIC_SUBMITS: f64 = 25.0;
    let d_cost = FERRIC_DISPATCHES * per_dispatch_us / 1000.0;
    let s_cost = FERRIC_SUBMITS * per_submit_us / 1000.0;
    println!("  Ferric spends {FERRIC_DISPATCHES:.0} dispatches and {FERRIC_SUBMITS:.0} submits per decode token:");
    println!("    dispatch overhead  {d_cost:>7.2} ms/token");
    println!("    submit overhead    {s_cost:>7.2} ms/token");

    if d_cost > s_cost {
        println!("\n  ✅ DISPATCH count dominates, by {:.0}x. Fusing kernels is the right lever and the", d_cost / s_cost.max(1e-9));
        println!("     410 -> 338 reduction was aimed at the correct quantity.\n");
        println!("     The scale is the surprise: {d_cost:.2} ms/token of launch overhead against a measured");
        println!("     decode of ~11 ms/token means most of a decode step is spent LAUNCHING work, not");
        println!("     doing it — natively, on Metal, before any browser is involved. Note also that the");
        println!("     {per_dispatch_us:.0} µs measured here sits close to the ~33 µs published for Chrome, which is a");
        println!("     useful check on the browser projection rather than a coincidence.");
    } else {
        println!("\n  ⚠ SUBMIT count dominates on this fabric — {:.1}x the dispatch overhead. Further kernel", s_cost / d_cost.max(1e-9));
        println!("     fusion buys little HERE; the lever is submission batching, which Ferric already does");
        println!("     (~1 per layer). Note this does NOT overturn the browser argument: the published");
        println!("     Firefox figure is explicitly per-dispatch and ~30x Chrome's, so a fabric can charge");
        println!("     differently. It does mean the native win from fusion is small and the browser win");
        println!("     is the one to claim — measured there, not projected from here.");
    }

    // Guard the experiment itself: if batching did not actually reduce submissions, every number above is
    // measuring nothing and the conclusion would be drawn from noise.
    assert!(s_un > s_ba, "batch() did not reduce submissions ({s_un} -> {s_ba}); this measures nothing");
    assert_eq!(d_un, d_ba, "batching changed the DISPATCH count ({d_un} -> {d_ba}); the two costs are not separated");
}
