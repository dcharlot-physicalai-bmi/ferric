# Ferric vs the field, measured — September 2026

Every number in the first three sections was measured **on this machine, this week**, not quoted from
a README. Where a figure is external it is cited. `docs/SOTA.md` is the standing scorecard; this is a
dated snapshot with a ranked plan, written because "best cross-fabric runtime, latest models, best
performance" is three different scoreboards and Ferric is in a very different place on each.

## 1. Correctness — Ferric leads, and it is now measured rather than asserted

The comparison that matters is **same fabric**. Cross-fabric comparisons are dominated by the fabric.

| comparison | relative Δ (`sum_abs`, 151936 logits) | argmax |
|---|---|---|
| Ferric-Metal vs **llama.cpp-Metal** | **5.35e-05** | both 12095 |
| llama.cpp-Metal vs **llama.cpp-CPU** | **1.79e-02** | both 12095 |

⭐ **The fabric term is 335x the implementation term.** One model (`qwen3-0.6b-q5km`), one prompt,
identical token ids. This retires a question that stood open for a long stretch: the ~1e-3 residual
between Ferric and llama.cpp on hyv4 is **6.25x smaller than llama.cpp's own CPU-vs-Metal
disagreement**. Ferric is closer to llama.cpp-CPU than llama.cpp-Metal is.

Alongside that: Metal and browser WebGPU are **bit-identical**, NVIDIA Vulkan agrees to ~1e-5 with
identical argmax (`docs/SOTA.md`), and greedy decode is token-for-token identical to llama.cpp on
Qwen3-0.6B-Q4_K_M / Q5_K_M and Qwen2.5-0.5B-Q6_K.

**No other project ships proven same-code cross-vendor parity.** This is the moat. It is also the
thing least visible on a benchmark chart.

## 2. Performance — Ferric is ~10x behind, and the reason is NOT what the docs assumed

Same model, same machine, both on Metal:

| | tok/s | ms/token |
|---|---|---|
| llama.cpp (`llama-bench` tg24, 3 reps) | **369.25 ± 7.44** | 2.7 |
| Ferric (warm, 24 tokens) | **38.5** | 26 |

**9.6x.** Four explanations are ruled out by measurement, not argument:

| hypothesis | verdict | evidence |
|---|---|---|
| host-side command preparation | **false** | 1.1–2.5% of wall (MNN reports 49.5–75.2% on GPU; not us) |
| memory-bandwidth-bound | **false** | 424 MiB and 974 MiB models both decode at **26 ms/token** — 2.3x the bytes, same time |
| dispatch-count-bound | **false** | 478 vs 178 dispatches/token — 2.7x, same time |
| submit-count-bound | **false** | 57 vs 33 submits/token — 1.7x, same time |

⭐ **What the profiler says instead: attention is 52% of a decode step.**

    attn     252.3 ms   52.0%
    ffn      175.0 ms   36.1%
    lm_head   50.3 ms   10.4%
    embed      7.4 ms    1.5%

At a ~17-token context the attention FLOPs are negligible, so 52% is **per-op overhead in the
attention path**, not arithmetic. That also explains the model-size independence above: a fixed
per-layer dispatch pattern over a short sequence costs the same whatever the weights weigh. Ferric
issues **478 dispatches/token across 28 layers — 17 per layer**, where a fused implementation needs
a handful.

⛔ **This retires "we are bandwidth-bound", which this repo's docs said for months.** The 47 GB/s
figure was bytes ÷ wall clock, and `docs/RUNTIME-PARITY-2026.md` already flagged it as false; the
two-model experiment above settles it.

## 3. Model coverage — the honest count

`crates/ferric-llama/src/arch.rs` is a registry with a status per architecture, and the taxonomy is
the point: `Verified` means diffed against a reference implementation **on real weights**.

| status | count | meaning |
|---|---|---|
| **Verified** | **15** | compared against a reference on real weights and matched |
| Loads | 15 | coherent output, never diffed — a wrong RoPE looks exactly like this |
| Parts | 5 | components exist, no loader wires them |
| Untried | 4 | synthetic checkpoint only |

Against llama.cpp's **50+ text architectures** and ~45k GGUF checkpoints on Hugging Face. So on raw
breadth Ferric is behind roughly 2:1 — but on *evidence per architecture* nothing else publishes a
status taxonomy at all, and 15 reference-verified is a real number.

⭐ **And Ferric is ahead where it counts most**: `hyv4` (Tencent Hy4, 770B/49B) is **supported by no
upstream runtime** — llama.cpp needs two out-of-tree patches. As of today Ferric runs all 78 blocks
of the real 213.66 GiB checkpoint, streamed, and emits the same greedy token as Tencent's own
implementation. That row moved `Untried → Verified` this week.

## 4. The field, September 2026

**Server tier** (not Ferric's game, but sets the performance narrative): SGLang ~16,200 tok/s vs
vLLM ~12,500 on prefix-heavy Llama-3.1-8B (H100), TensorRT-LLM highest absolute with a compile step.

**Local tier** (Ferric's actual competitor): llama.cpp. v0.4.0 shipped **2026-09-04** with
Qwen3.8-Flash-Next (`qwen4exp`), Nemotron-3-Puzzle-75B-A9B, and **lazy tensor reading — weights
pulled from disk on demand**, which is the same idea as Ferric's `load_streaming`. ggml 0.23.0 added
sparse flash attention for DeepSeek-V4, GLM and qwen4exp.

**Browser tier** (Ferric's differentiator): WebLLM runs Llama-3.1-8B-4bit at ~41 tok/s on an M3 Max
(~80% of native MLC); transformers.js v4 was rewritten in C++ with a WebGPU backend for 3–10x over
v3. Browser is generally 5–10x slower than native. Ferric's measured browser figures (LFM2.5-1.2B:
6.8 tok/s cold, ~20 warm, 37.6 with q4_0 KV) are **behind WebLLM on a model 7x smaller** — the
cross-fabric *correctness* lead is not yet a cross-fabric *performance* lead.

**Rust tier**: candle (inference, cuBLAS-direct), mistral.rs (quantized, OpenAI-compatible server),
burn+CubeCL (the only true same-code cross-platform peer, no bit-exact parity guarantee), ratchet
(WebGPU, inference-only). None combines native + browser + training + verified parity.

**Research worth ingesting**: dispatch overhead is characterised across 4 vendors / 3 backends / 3
browsers (arXiv 2604.02344) — **wgpu-native on Metal is the worst measured at 71.1 µs/dispatch vs
Safari's 31.7**, and kernel fusion cutting 312 dispatches bought 53% end-to-end. ⚠ That paper's
batch=1 conclusion does **not** transfer here: Ferric measured a 29% dispatch cut at 0.00 ms. Read it
for the fusion result, not the diagnosis. Also live: speculative decoding × sparse attention (NSA,
EAGLE-series), NPU table-lookup low-bit inference (T-MAN), CPU-GPU cooperative execution with AMX.

## 5. Ranked plan — what actually closes the gap

Ordered by measured impact per unit of work, not by novelty.

1. **Fuse the decode attention path.** It is 52% of a decode step at a context where its FLOPs round
   to zero. This is the single biggest lever in the runtime and it is roadmap item #3, still
   unchecked. Target: 17 dispatches/layer → single digits.
2. **A real flash-attention/GQA WGSL kernel** (online softmax, tiled), which is the same work item
   and also unlocks long context, where the current path will degrade fastest.
3. **Per-shape/per-device kernel autotuning** (roadmap #6). Measured elsewhere at +41%; matters more
   for Ferric than anyone because WebGPU spans the widest hardware range.
4. **Close the browser performance gap** to WebLLM. The correctness story is already better; the
   throughput story is 7x worse per parameter, and browser-first is the stated thesis.
5. **Latest-model cadence.** `qwen4exp` (Qwen3.8-Flash-Next) and Nemotron-3-Puzzle-75B-A9B landed
   upstream on 2026-09-04. The standing rule is support within 30 days.
6. **Promote `Loads` → `Verified`.** Fifteen architectures generate coherent text and have never been
   diffed against a reference. `scripts/hyv4_vs_reference.sh` is now a reusable shape: same file,
   both implementations, gate `sum_abs` **and** the greedy pick.

⛔ **What NOT to do**: chase server-tier throughput. vLLM/SGLang/TensorRT-LLM own batched serving on
NVIDIA and Ferric will not take that, nor should it. The defensible position is *one codebase, every
fabric, verified identical, latest models, browser included* — and of those five, only performance
is currently red.
