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

⚠ **A fifth hypothesis was raised and then RETRACTED — by the instrument that raised it.** The
category profiler reported `attn` at 52% (qwen3) and 62% (llama3.2) of a decode step, which looked
like the answer and was written into the first version of this document. It is not sound. `prof()`
calls `device_sync()` at **every category boundary** — two per layer — and the profiled run costs
+25% (qwen3) and **+52%** (llama3.2) over the unprofiled one. The model with *more* sync overhead is
exactly the one reporting the *higher* attention share:

| | profiled | unprofiled | syncs/token | added ms per sync |
|---|---|---|---|---|
| qwen3-0.6b | 32.3 ms | 25.9 ms | 56 | 0.114 |
| llama3.2-1b | 39.8 ms | 26.2 ms | 32 | 0.425 |

Measured directly instead, without a readback in the timed loop
(`crates/ferric-tensor/examples/fattn_bench.rs`), the fused decode-attention kernel costs:

| S (cache length) | ms/call | MFLOP | GFLOP/s |
|---|---|---|---|
| 17 | 0.051 | 0.14 | 2.7 |
| 128 | 0.042 | 1.05 | 24.7 |
| 512 | 0.057 | 4.19 | 73.9 |
| 2048 | 0.090 | 16.78 | 186.0 |

⭐ **Flat to S=128 — a ~0.04 ms launch floor**, then it scales. At 28 layers that is **1.43 ms/token,
5.5% of the step** (llama3.2: 0.64 ms, 2.4%). The category profile said 52% and 62%. **The attention
core is not the bottleneck, and the ranked plan below was rewritten because of it.**

### What the gap actually decomposes into

The clean experiment is a **controlled pair**: two checkpoints of the *same architecture* — identical
graph, identical 478 dispatches/token — differing only in quantisation, so the only variable is bytes.

| | weight bytes | ms/token (warm) |
|---|---|---|
| qwen3-0.6b-**q5km** | 424 MB | 25.70 |
| qwen3-0.6b-**q4km** | 378 MB | 25.05 |

Δ46 MB → Δ0.65 ms, so the **marginal cost of weight bytes is 71 GB/s effective**. Against the
measured WGSL read ceiling (385–475 GB/s) that is 5.4–6.7x, and against llama.cpp's ~326 GB/s
streaming rate it is **4.6x**. Splitting each model's step by that rate:

| model | total | bytes term | remainder | remainder ÷ dispatches |
|---|---|---|---|---|
| qwen3-0.6b-q5km | 25.70 ms | 5.99 (23%) | 19.71 | 41.2 µs |
| qwen3-0.6b-q4km | 25.05 ms | 5.34 (21%) | 19.71 | 41.2 µs |
| llama3.2-1b-q6k | 26.10 ms | 13.76 (53%) | 12.34 | 69.3 µs |

⭐ **Two terms, and neither alone closes the gap.** The bytes term *by itself* — 6.0 ms — is already
**2.2x llama.cpp's entire 2.7 ms step**. The remainder is 12–20 ms/token on top of that.

⭐ **The remainder's per-dispatch figure is corroborated three independent ways**: derived here by
subtraction (41.2 and 69.3 µs), measured as the launch floor of the fused attention kernel
(~40 µs, `fattn_bench` above), and published for wgpu-native on Metal (**71.1 µs**, arXiv 2604.02344,
the worst of four backends measured — Safari's Metal is 31.7 µs and Dawn's Vulkan 23.8 µs).

⚠ **But it does NOT scale linearly with dispatch count**, and that matters: 2.7x the dispatches
(478 vs 178) buys only 1.6x the remainder. So dispatch count is part of it and not all of it — which
is exactly consistent with this repo's earlier finding that cutting 29% of dispatches moved wall time
by 0.00 ms. Both observations are true; the remainder has a per-dispatch component and a per-token
component, and this experiment cannot separate them.

⛔ **Still not established**: what the per-token component *is*. Five candidate mechanisms are dead
(host-side preparation, total bytes, dispatch count alone, submit count, the attention kernel), and
the two live terms are now measured rather than guessed — but the remainder's floor needs per-op GPU
timing that does not sync to attribute.

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

1. **The quantised matmul kernels — 71 GB/s against a 385–475 GB/s ceiling.** This is now measured
   with a controlled pair, not inferred: it is a 4.6x gap against llama.cpp's streaming rate, and the
   bytes term alone already exceeds llama.cpp's whole step by 2.2x. `matmul_q4_k_splitk` /
   `matmul_q5_k_splitk` / `matmul_q6_k_splitk` are where the work is.
2. **Cut the per-token remainder** (12–20 ms). Its per-dispatch figure (41–69 µs) matches published
   wgpu-native-on-Metal cost, which is the **worst of four backends measured** — Safari's Metal is
   2.2x cheaper and Dawn's Vulkan 3x. Some of this is wgpu's, not Ferric's, and that is worth knowing
   before optimising around it.
3. **Build a non-syncing profiler** (timestamp queries) to separate the remainder's per-dispatch and
   per-token halves. Every attribution attempt here either synced per boundary — distorting the split
   badly enough to produce a retracted 52% claim — or aggregated to a whole step.
4. **A real flash-attention/GQA WGSL kernel** (online softmax, tiled) for PREFILL and long context,
   where the measured curve above does start to scale. Not for short-context decode: the fused
   kernel already costs 5.5% there and fusing it further cannot buy back 9.6x.
5. **Per-shape/per-device kernel autotuning** (roadmap #6). Measured elsewhere at +41%; matters more
   for Ferric than anyone because WebGPU spans the widest hardware range.
6. **Close the browser performance gap** to WebLLM. The correctness story is already better; the
   throughput story is 7x worse per parameter, and browser-first is the stated thesis.
7. **Latest-model cadence.** `qwen4exp` (Qwen3.8-Flash-Next) and Nemotron-3-Puzzle-75B-A9B landed
   upstream on 2026-09-04. The standing rule is support within 30 days.
8. **Promote `Loads` → `Verified`.** Fifteen architectures generate coherent text and have never been
   diffed against a reference. `scripts/hyv4_vs_reference.sh` is now a reusable shape: same file,
   both implementations, gate `sum_abs` **and** the greedy pick.

⛔ **What NOT to do**: chase server-tier throughput. vLLM/SGLang/TensorRT-LLM own batched serving on
NVIDIA and Ferric will not take that, nor should it. The defensible position is *one codebase, every
fabric, verified identical, latest models, browser included* — and of those five, only performance
is currently red.
