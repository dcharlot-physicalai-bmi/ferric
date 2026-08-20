//! **Does carrying the gated-delta-net state ISOLATE a snapshot of it?**
//!
//! `gdn_state.rs` proves the recurrence is correct when the state is carried forward. This proves the
//! adjacent property that speculative decoding actually depends on: that *keeping a copy* of the state
//! and then stepping the recurrence leaves the copy alone.
//!
//! It is not the same question, and the answer used to be no. `gated_delta_rule_stateful` writes the
//! state buffer **in place** — its own comment says "hand it a buffer it owns either way" — but it
//! obtained that buffer with `state.contiguous()`, and `Tensor::contiguous()` returns `self.clone()`
//! when the input is already contiguous, which a carried state always is. `Tensor::buf` is an
//! `Arc<wgpu::Buffer>`, so that clone is the SAME GPU allocation. Every handle to the state therefore
//! observed the step, including handles taken specifically to undo it.
//!
//! That made `qwen35::Cache::snapshot()` a lie for exactly one field. Its doc says "tensors are
//! immutable Arc-shared buffers, so cloning the cache clones handles, never GPU data" — true of the KV
//! entries, whose `cat` allocates fresh, and false of the GDN state, whose kernel mutates. The live
//! consumer is `ferric-serve`'s speculative-decode rollback: `let snap = cache.snapshot(); … cache =
//! snap;` restored a snapshot the draft steps had already advanced, so a rejected draft left the
//! recurrent state in the future while `pos` went back. Nothing errors; the model just attends to a
//! state that never corresponded to any accepted prefix.
//!
//! The failure is invisible to `gdn_state.rs` because that example never keeps a second handle: it
//! reassigns `st = s` every step, so aliasing and copying are indistinguishable there.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (i as f32 * 0.37 + s).sin()).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (t, h, dk, dv) = (8usize, 4usize, 16usize, 16usize);
    let mk = |o: f32, d: usize| Tensor::from_vec(&ctx, &seq(t * h * d, o), &[t, h, d]);
    let (q, k, v) = (mk(1.0, dk), mk(2.0, dk), mk(3.0, dv));
    let gbv: Vec<f32> = (0..t * h)
        .flat_map(|i| [-0.1 - 0.05 * ((i % 3) as f32), 0.3 + 0.1 * ((i % 4) as f32)])
        .collect();
    let gb = Tensor::from_vec(&ctx, &gbv, &[t, h, 2]);
    let nar = |x: &Tensor, a: usize, b: usize| x.narrow(0, a, b - a).contiguous();

    // Prefill, then take a snapshot exactly the way Cache::snapshot does: a handle clone.
    let (_, st) = nar(&q, 0, 4).gated_delta_rule_stateful(
        &nar(&k, 0, 4), &nar(&v, 0, 4), &nar(&gb, 0, 4), h, dk, dv, None);
    let snapshot = st.clone();
    let before = snapshot.to_vec().await;

    // Step the recurrence forward from `st`, as a draft would.
    let (_, advanced) = nar(&q, 4, 5).gated_delta_rule_stateful(
        &nar(&k, 4, 5), &nar(&v, 4, 5), &nar(&gb, 4, 5), h, dk, dv, Some(&st));
    let after = snapshot.to_vec().await;

    let drift = maxdiff(&before, &after);
    let moved = maxdiff(&before, &advanced.to_vec().await);
    println!("gated-delta-net state isolation");
    println!("  the step itself moved the state by max|Δ| = {moved:.4} (it must move, or this proves nothing)");
    println!("  the SNAPSHOT drifted by            max|Δ| = {drift:.4} (it must not move at all)");

    assert!(moved > 1e-3,
            "the recurrence did not advance ({moved:.2e}); this example cannot detect aliasing when \
             the step is a no-op, so fix the inputs before trusting the result below");
    assert!(drift == 0.0,
            "SNAPSHOT ALIASED THE LIVE STATE: a handle taken before the step observed it anyway \
             (max|Δ| = {drift:.4}). gated_delta_rule_stateful writes the state in place, so it must be \
             handed a buffer no one else holds — Tensor::contiguous() returns self.clone() for an \
             already-contiguous input and does not provide that. Speculative-decode rollback in \
             ferric-serve restores such a snapshot, so this corrupts the recurrent state on any \
             rejected draft, silently and fluently.");
    println!("  ✅ the snapshot is isolated: stepping the live state left it bit-identical");
}
