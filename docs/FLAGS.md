# Ferric runtime flags

26 `FERRIC_*` environment variables exist, and to a user they currently look alike — a 13× performance
win, a measured regression, and a switch that silently changes numerics are all "just an env var."
This file is the taxonomy. **The category matters more than the flag.**

## Why anything is opt-in at all

Ferric's differentiating guarantee is **bit-reproducible output across fabrics** (CPU / Metal / Vulkan /
WebGPU). Any flag that changes floating-point accumulation order breaks that guarantee, however small the
delta. So a fast path is *not* promoted to default merely because it is faster and "close enough."

Measured on Metal, `matmul() 2048³`:

| path | throughput | max&#124;Δ&#124; vs naive |
|---|---|---|
| naive (default) | 643 GFLOP/s | — |
| `FERRIC_COOP=1` | **8383 GFLOP/s (13×)** | **5.0e-8** |

5.0e-8 is sub-epsilon for f32 (~1.2e-7) — it is reassociation, not a bug. It is also **not zero**, which
is why coop stays opt-in on the inference path. A 13× speedup does not buy the right to silently change
someone's logits.

> ⚠️ A prior audit recorded this path as "bit-identical (relΔ 0.0e0)" and recommended making it the
> default. Direct measurement contradicts that. Verify precision claims before promoting a flag.

---

## A — Performance wins (opt-in *because* they change numerics)

Enable deliberately for training and prefill, where bit-exactness was never promised. Do **not** enable
for a run whose output must match another fabric bit-for-bit.

| flag | effect |
|---|---|
| `FERRIC_COOP` | Cooperative-matrix (tensor-core) f32 GEMM. **13× on Metal**, Δ 5.0e-8. Metal-only via `coop_gemm_ok()`. |
| `FERRIC_COOP16` | 16×16 f16-input coop path (NVIDIA tensor cores / Intel XMX). Vulkan-only via `coop16_ok()`. |
| `FERRIC_COOP_2PASS` | Two-pass coop variant. |
| `FERRIC_METAL4` | Metal-4 tensor-unit resident GEMM. fp16 inputs by contract ⇒ precision-changing. |
| `FERRIC_SUBGROUP` / `FERRIC_SGGEMV` | Subgroup-accelerated decode GEMV. |
| `FERRIC_Q4K_TRANS` | Transposed Q4_K path. |
| `FERRIC_Q2_0_KERNEL`, `FERRIC_Q2_0_SPLITK_MAX` | Q2_0 kernel selection / split-K bound. |

## B — Measured negatives (documented dead ends — do not enable)

Kept in-tree so the dead end is not re-discovered. Both are **correct**, just slower.

| flag | why it stays off |
|---|---|
| `FERRIC_MEGA` | Whole-FFN megakernel. Token-for-token identical but **~2× slower at decode** (occupancy-bound). `dtype.rs:954`, `qwen3.rs:327`. |
| `FERRIC_Q4K_TRANS_M` | Measured to **hurt**. |

Also recorded as a dead end in code, though not a flag: ternary coop **K=32** blocking (+7.4% slower —
more shared memory cut occupancy; the win came from N-tiling instead, which raises arithmetic intensity).

## C — Escape hatches (disable a default fast path)

Useful for A/B-ing a suspected fast-path bug against the reference path.

`FERRIC_NOFLASH` · `FERRIC_NOFUSE` · `FERRIC_NOSPEC` · `FERRIC_NOWINDOW` · `FERRIC_NO_SUBGROUP`

## D — Debug / diagnostics (never in production)

`FERRIC_PROFILE` · `FERRIC_DUMP_IDS` · `FERRIC_EMB_DEBUG` · `FERRIC_LAYER_SUMS` · `FERRIC_SPEC_DBG` ·
`FERRIC_COOP_SHARED_FORCE` (overrides the coop capability gate for debugging)

## E — Configuration (not performance switches)

`FERRIC_MAX_BINDING` · `FERRIC_MAX_LAYERS` · `FERRIC_MLMODELC_DIR` · `FERRIC_SPEC_DRAFT`

---

## Rule for adding a flag

1. State the **category** (A–E) in the doc comment at the gate site.
2. If it changes numerics, say so and give the measured **max|Δ|** — not "negligible".
3. If it was measured and lost, keep it with the number and the reason, like `FERRIC_MEGA`. A recorded
   negative is worth more than a deleted branch: it stops the next person re-running the experiment.
