# SOTA feature matrix — what Ferric needs to be the go-to runtime on all hardware

**Focus.** Feature completeness across every fabric, not throughput on any one machine. A runtime is
adopted for what it *can do*; speed on a given laptop is a tuning question and is tracked separately
in `composable-engine-survey.md`.

**Provenance.** ✅ verified present in the tree · ❌ verified absent · ⚠ partial, with the limit named
· ❓ not verified this pass. Absences are code-checked, not assumed.

---

## A. Serving & throughput features

| feature | state | note |
|---|---|---|
| Structured / constrained output | ✅ | JSON-schema-constrained sampling in-runtime, deterministic across fabrics. **Ahead of most.** |
| Tool / function calling | ✅ | plus an MCP module in `ferric-serve` |
| Embeddings + pooling | ✅ | LAST-pooling, matches reference conventions |
| Chunked prefill | ✅ | verified bit-identical to whole-history, incl. windowed models |
| Prefix cache | ⚠ | **one slot**. SGLang's RadixAttention generalises to a radix tree over arbitrary branching — the shape agent loops actually produce |
| Speculative decoding | ⚠ | a draft/verify path exists; **not verified EAGLE-class**. EAGLE 3.1 (vLLM) and EAGLE3 dynamic-tree kernels (TensorRT-LLM) both shipped in the last 45 days |
| **Continuous batching** | ❌ | `ferric-serve`: *"One request at a time…"* — but ⚠ **the serialization claim does not survive a first measurement**, see §I. Still absent as a feature; the size of the win is now unclear. |
| Paged / block KV cache | ❓ | contiguous `KvBuf` today; paging is what makes many-session serving memory-viable |
| Multi-model in one process | ❌ | SGLang ships it; vLLM does not |
| Disaggregated prefill/decode | ❌ | standard in large deployments |
| **KV cache quantization** | ❌ | no implementation. Directly sets max context per GB — matters most on the small devices Ferric targets |

## B. Hardware coverage — the "all hardware" axis

| fabric | state | note |
|---|---|---|
| Metal / Vulkan / DX12 / GL | ✅ | via `wgpu`, one codebase |
| **Browser WebGPU** | ✅ | same source to `wasm32`. Genuinely rare |
| CPU fallback | ❓ | wgpu GL/software path only; no hand-vectorised CPU kernels (AVX-512 / NEON) verified |
| **CUDA native** | ❌ | NVIDIA reached through Vulkan, not cuBLAS/CUTLASS/cuDNN. Leaves tensor-core throughput on the table on the most common accelerator |
| **ROCm native** | ❌ | AMD via Vulkan; AITER/CK unused |
| **NPUs** | ❌ | no Hexagon (MNN shipped it Jul 2026), no Ascend/CANN, no CoreML/ANE, no OpenVINO |
| Multi-GPU: tensor / pipeline / expert parallel | ⚠ | `ferric-dist` does **pipeline** (layer sharding) only; no tensor or expert parallelism |
| RDMA / high-speed interconnect | ❌ | Velox has UCX; NCCL/NVLink equivalents absent |

**The honest read.** `wgpu` buys breadth cheaply and is why "runs everywhere" is defensible at all.
It also caps the ceiling on every vendor's flagship path: no tensor cores via CUDA, no matrix cores
via ROCm, no NPU at all. **Breadth is real; depth on any single vendor is not.**

## C. Model capability

| feature | state | note |
|---|---|---|
| Dense GQA transformers | ✅ | qwen2/3, llama, phi3, gemma/2/3 |
| MoE (incl. shared experts) | ✅ | qwen35moe, deepseek2, indexed expert kernels |
| MLA (latent attention) | ✅ | deepseek2, legacy `attn_kv_b` path |
| SSM / Mamba-2 hybrids | ✅ | nemotron_h, lfm2 short-conv |
| Per-layer embeddings, shared KV | ✅ | gemma4 |
| RoPE: NEOX/NORM, YaRN, Llama-3 scaling, partial, 2-D | ✅ | all reference-verified as of 2026-08-15 |
| Vision / multimodal | ⚠ | Muse Glimmer ViT + mmproj works; gemma4 mmproj **unwired**; no audio/video |
| **Multi-token prediction (MTP)** | ❌ | skipped on nemotron; gemma4 ships an `mtp-` file unused. Free speedup left on the floor |
| **LoRA / adapter serving** | ❌ | only DeepSeek's `q_lora_rank`/`kv_lora_rank` MLA internals exist — that is not adapter serving |
| Absorbed-MLA path | ❌ | refused at load; it is the variant that makes the KV cache small |
| Q-LoRA-factored attention | ❌ | refused at load; needed for full-size DeepSeek |

## D. Quantization

| format | state |
|---|---|
| Q4_K, Q5_0, Q6_K, Q8_0, IQ4_XS, IQ4_NL, Q2_0 ternary | ✅ packed kernels |
| **MXFP4** | ❌ — **GPT-OSS ships in it**; without this those weights cannot be run as released |
| FP8 (E4M3/E5M2) | ❓ |
| AWQ / GPTQ | ⚠ a ternary-GPTQ example exists; general support unverified |
| KV cache quant | ❌ (see §A) |

## E. Where Ferric is genuinely ahead

Not throughput. These are real and uncontested in the surveys:

1. **Joules as an objective function.** `ferric-joule` measures, and refuses to report unmeasured
   figures. Zero mentions of joule/watt/kWh across NVIDIA Switchyard *and* DeepSeek Harness.
2. **Sound verification.** `ferric-certify`; interval certificates; determinism across fabrics.
3. **One codebase, native and browser.** Ratchet and Burn overlap the fabric, not the whole span.
4. **Architecture registry that refuses unknowns** rather than loading them down a near-miss path.
5. **Constrained decoding in-runtime**, deterministic across fabrics.

## F. Ranked build order

Ranked by *adoption blocked per unit of work*, which is not the same as difficulty:

1. **Continuous batching** — turns a demo server into a usable one. Nothing else in §A matters while
   the server is single-request.
2. **MXFP4** — a format gate: GPT-OSS cannot run as released without it.
3. **KV cache quantization** — largest context-per-GB lever, and it compounds on exactly the small
   devices Ferric claims.
4. **RadixAttention-style prefix tree** — a data structure, not a kernel; aimed at agent loops, which
   is what the new open-weight frontier is being used for.
5. **LoRA adapter serving** — the standard way people ship fine-tunes.
6. **MTP** — the weights are already on disk, unused.
7. **CUDA/ROCm native backends** — the largest ceiling raise, and the largest amount of work; it is a
   second backend tree, not a tuning pass.
8. **NPU backends** (CoreML/ANE, Hexagon, Ascend) — where on-device inference is going.

## G. ⭐ RLM makes continuous batching load-bearing, not optional

Verified 2026-08-16. Recursive Language Models (`alexzhang13/rlm`, MIT, 5.5k★) reach local models
through their OpenAI client — the README says *"For local models, we recommend using vLLM (which
interfaces with the OpenAI client)"*. `ferric-serve` is OpenAI-compatible, and it works **today, with
no new code**:

```
/v1/models           → {"object":"list","data":[{"id":"ferric",…}]}
/v1/chat/completions → {"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}],
                        "usage":{"prompt_tokens":15,"completion_tokens":1,"total_tokens":16}}
```

**And that is exactly where the single-request server stops being a nice-to-have gap.** RLM's whole
shape is fan-out: `rlm_query_batched` spawns parallel sub-calls bounded by `max_concurrent_subcalls`,
one per slice. Point that at Ferric and every parallel sub-call **serialises**, so the paradigm's
central move degrades to a sequential loop — while `ferric_joule::recursion` shows the energy saving
survives (parallelism buys latency, not joules) the *latency* case for recursion collapses entirely.

So the scheduler in `ferric_llama::sched` is not a generic serving improvement. **It is the specific
thing that makes Ferric usable as an RLM backend**, which is the fastest-moving harness paradigm in
the field and the one whose reference implementation (`PrimeIntellect-ai/prime-agent`, 16.4k★) is
being pushed daily. Wiring the scheduler into `ferric-serve` is now the highest-value remaining item
in §F, ahead of MXFP4.

## I. ✅ Serialization CONFIRMED — and the first measurement was wrong

Measured properly on 2026-08-16: 64 max_tokens so decode dominates fixed overhead, per-request
timings rather than a wall-clock total, and `completion_tokens`/`finish_reason` reported so an early
stop cannot masquerade as concurrency.

```
SEQUENTIAL baseline   2.05s  2.06s  2.28s          (all out=64, finish=stop)
4 CONCURRENT          2.67s  4.71s  7.06s  8.87s   (all out=64, finish=stop)
wall total 9.12s
```

**The staircase is the evidence.** Each request completes ~2.1 s after the previous — 2.67 → 4.71
(+2.04) → 7.06 (+2.35) → 8.87 (+1.81) — which is queueing, not overlap. Wall total 9.12 s against a
2.06 s single request is **4.4x for 4x the work**. The crate docs are right: the server serializes.

**My earlier measurement said 1.75x and was wrong.** It used 24 max_tokens, where fixed overhead is a
large share of each request and the signal is swamped. The error was in the *experiment design*, not
the machine — the same class as timing inside a `cargo run` loop, and the fix was the same: make the
thing being measured dominate what surrounds it.

Worth noting which way each error pointed. The bad short-generation run made the server look *better*
than it is and would have talked me out of building continuous batching; the bad `cargo run` loop made
the machine look *worse* than it is. **Neither direction is safe, and the convenient one is the more
dangerous** — I only re-ran this because the result flattered a conclusion I had already started
building on.

So §F item 1 stands, and the win is now **measured rather than reasoned**. Same five requests, same
tokens, only `max_batch` changed (`examples/continuous_batching`, Qwen2.5-0.5B):

| max_batch | batched steps | total |
|---|---|---|
| 1 | 15 | 970.21 ms |
| 2 | 9 | 914.16 ms |
| 3 | 7 | 672.84 ms |
| **5** | **5** | **315.65 ms** |

**3.07x** — real, and **short of the ~4-5x** I claimed one commit earlier from "one weight read serves
the batch". Two reasons, both structural rather than tuning:

1. **Prefill is still serial.** Each admitted request prefills alone before joining the batch, so a
   full slate of 5 pays 5 sequential prefills up front. Chunked prefill interleaved into the decode
   batch is the standard fix, and Ferric already has chunked prefill verified bit-identical — it is
   the *scheduling* of it that is missing.
2. **The batch drains.** Budgets are uneven by design in that example, so the batch is only full at
   the start; average occupancy over the run is well below `max_batch`. `Scheduler::occupancy()`
   exists to make exactly this visible.

Both point at the same next lever, and neither is a reason to doubt the feature: 3.07x on five short
sequences is a large win for a data-structure change with no kernel work.

## K. ⚠ Batched decode exists on ONE runtime of five

Checked before starting the transport work rather than during it. `forward_batch` is implemented on:

| runtime | batched decode | covers |
|---|---|---|
| `qwen3` (Dense) | ✅ | qwen2, qwen3, llama, phi3, gemma, gemma2, gemma3 |
| `qwen35` (Hybrid) | ❌ | qwen35, qwen35moe, laguna |
| `lfm2` | ❌ | lfm2 |
| `gemma4` | ❌ | gemma4 |
| `deepseek2` | ❌ | deepseek2 |

So continuous batching in `ferric-serve` is **Dense-only** as things stand. That covers most
architectures by count, and none of the four newest ones — including DeepSeek and Gemma 4, the two
added this week, and the hybrid path that carries Ferric's ternary work.

**This is not a small gap to close.** Batching requires advancing N independent sequences in one
forward, and each of these runtimes carries a *different* per-sequence state:

- `qwen35` — gated-delta-net recurrent state per layer, plus KV on the attention layers
- `lfm2` — short-conv state (`l_cache-1` timesteps of the PRE-conv signal) plus KV
- `gemma4` — KV, but with **shared** caches where blocks ≥15 read block 13/14
- `deepseek2` — MLA latent KV with asymmetric head widths

Each needs its own batched forward, and each is a place where a batched path that crossed sequences
would still emit fluent text — the failure mode `examples/continuous_batching` checks for by
re-running every sequence solo.

**Consequence for the build order.** §F item 1 splits in two: wiring the transport (Dense-only, ships
a real win for the majority path) and extending batched decode to the other four runtimes (four
separate pieces of model work, each needing its own solo-equivalence proof). The first does not block
on the second, and the server must fall back to serial rather than silently mis-batching a runtime
that cannot.

## J. Not assessed

CPU SIMD kernels; FP8; paged KV internals; guided-decoding coverage vs xgrammar/outlines; audio and
video multimodal; scheduler fairness and preemption; observability surface.
