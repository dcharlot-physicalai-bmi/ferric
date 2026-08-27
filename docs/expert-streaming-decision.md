# Expert streaming on the qwen35 runtime: what it costs, before anyone builds it

**Status: decision record, nothing implemented.** Written 2026-08-26 after three measurements
converged on the same constraint. The point is to establish the price before paying it.

## The gap

Colibri and FreeToken both run frontier MoE models on hardware that cannot hold them, by streaming
routed experts from NVMe. Ferric cannot. `Qwen35::load` is all-or-nothing — it packs every expert
into GPU slabs (`qwen35.rs`, `MoeExperts::Slab`) — and `ferric_tier::ExpertCache`, which exists
precisely to hold a resident subset, **has no runtime consumer anywhere in the tree**.

Everything built this session sits above that gap: the fabric split decides where a missing expert
executes, the weighted mirror decides which device serves it, the cache-policy sweep tunes what to
retain. All of it presumes a runtime that can miss. None can.

## The mechanism is feasible — the kernel already supports it

`matmul_q4_k_swiglu_id(w, selw, k, eff)` (`dtype.rs:1755`) reads an expert id out of `selw` and
indexes `w`. Nothing requires `w` to hold *all* experts. A resident subset plus an id→slot remap of
`selw` runs on the existing kernel unchanged, so the slab becomes a cache rather than a table. That
part is a small change.

## ⛔ The price: streaming re-introduces exactly the sync the fast path was built to delete

To fetch a missing expert you must first know which experts were chosen. Knowing that on the CPU is a
readback. And `moe_topk` exists *for the express purpose of removing that readback* —
`dtype.rs:1710` says so in its first line: "MoE router top-k, entirely on the GPU — **kills the
per-layer CPU readback sync**." `qwen35.rs:894` records the result: "**Zero syncs**, and the dispatch
count is independent of T."

So expert streaming and the zero-sync FFN path are **mutually exclusive by construction**. Streaming
costs one GPU→CPU round trip per (token, MoE layer) — on this checkpoint, ~40 per token.

## ⛔ And speculation cannot bridge it — measured, not assumed

The obvious escape is to prefetch: predict the next layer's experts while the current one computes,
so the fetch overlaps the sync. `examples/route_predictability.rs` measured whether that is possible
on this checkpoint:

```text
  identity  (L's set -> L+1)               2.7%    <- at the 3.1% random floor
  per-layer popularity [held-out]         10.0%
  previous token, same layer              41.0%
```

**Cross-layer routing prediction is at chance.** The only predictor available while layer L runs is
layer L's own set, and it carries no information about L+1. Colibri's published 71.6% does not
transfer here. So the sync cannot be hidden behind a speculative fetch, and the 41% temporal signal
does not help either — it predicts what the cache *already holds*, not what to fetch next.

## What that means for the decision

Expert streaming on this runtime is **a deliberate trade of throughput for capacity**, not a free
capability. It buys "runs at all" on a machine that cannot hold the model, and it pays ~40 syncs per
token plus the fetch itself. That is consistent with what the field reports: Colibri measures
0.05–6.8 tok/s and FreeToken 14.9–83 tok/s depending on residency, against the tens-to-hundreds a
resident model achieves.

**The trade is worth making** — a model you cannot run has a throughput of zero — but it should be
implemented as an explicitly budgeted mode that a caller opts into, never as a default, and the sync
cost should be measured on the first working version rather than assumed to be small.

## Order of work, if it is taken up

1. **Measure the sync first.** Time a forward with a per-(token, layer) `to_vec()` on `selw` against
   the current zero-sync path. That number decides whether the rest is worth writing, and it needs no
   new code beyond a timing loop around the existing `route_trace` hook.
2. `MoeExperts::Streamed { cache: ExpertCache, slab: … }` plus an id→slot remap kernel.
3. `Qwen35::load` gains a byte budget; below it, MoE layers take the streamed variant.
4. Placement-invariance test: the same logits at every budget. `ferric-tier` already enforces this
   shape for layers (`tests/placement_invariance.rs`); experts need the same guarantee.

⚠ Step 1 is a gate, not a formality. If the sync dominates, the answer may be that Ferric should
stream at a coarser granularity — whole layers, which `PrefetchCache` already does and which needs no
readback because layer order is known — and accept that per-expert residency is not worth its syncs
on this architecture.
