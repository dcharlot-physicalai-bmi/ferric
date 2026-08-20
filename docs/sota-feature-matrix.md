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
| **Continuous batching** | ✅ | Shipped and wired: `ferric-serve/src/batch.rs` runs the accept/dispatch loop (`serve_loop` at lib.rs:785), `Model::supports_batching()` gates it per runtime (lib.rs:421), and `--max-batch N` / `--no-batch` control it. Guided-decoding and tool-calling requests keep the untouched serial path. Four tests, each mutation-proven: batched responses identical to serial, concurrent requests interleaving rather than queueing, a late request joining a batch in flight, and the runtime gate actually being consulted. Both cross-sequence-leak mutations (every row reading seq 0; off-by-one row mapping) fail it. Serialization measured at 4.4x, payoff at 3.07x — not the ~5x a naive weight-read argument suggests, because prefill is still serial and the batch drains |
| Paged / block KV cache | ❓ | contiguous `KvBuf` today; paging is what makes many-session serving memory-viable |
| Multi-model in one process | ❌ | SGLang ships it; vLLM does not |
| Disaggregated prefill/decode | ❌ | standard in large deployments |
| **KV cache quantization** | ⚠ **q8_0 usable, 4-bit is not** | Wired into the **dense** runtime behind `FERRIC_KVQ`, default OFF and bit-identical to the pre-wiring build (FNV over raw logit bits vs a separate worktree at `b94825e`, 12/12 steps). **q8_0**: 3.76x context per GB (24576 -> 6528 B/token), ppl 31.21 -> 31.76 (+1.8%), mean KL 1.5e-2, top-1 91.9%, generation matched f32 for all 48 tokens. **q4_0/q4_1**: kernels bit-exact vs a CPU reference (0/99360 and 0/110400 packed words differ) but the format is too lossy here — rel-rmse ~0.10, 1−cos ~5e-3, both diverge at generated token 1. Their ppl ordering (106.6 vs 196.7) is NOT evidence about the formats: once both are off the rails those are two chaotic trajectories, not a comparison. Measured on qwen2.5-0.5b-q8_0. Wired on **ALL FIVE runtimes** (`qwen3`, `gemma4`, `lfm2`, `deepseek2`, `qwen35`), each verified default-off unchanged against a real checkpoint and batched-decode solo-equivalent under `FERRIC_KVQ=q8_0`. ⚠ **The qwen35 wiring shipped broken once and the equivalence example caught it.** That runtime has SEPARATE batched attention functions from its solo ones; the first pass updated only the solo sites, so a quantized cache fell through to the `_ =>` arm and every token attended to ITSELF ALONE, as though the sequence had no history — finite logits, fluent text, no error. Any edit to one of that file's four cache blocks must be made to all four. ⚠ **deepseek2 gains MORE from this than it looks, not less.** MLA's small cache is the ABSORBED path, which `DeepSeek2::load` refuses; the legacy `attn_kv_b` path decompresses to `n_head*qk_dim` keys and `n_head*v_head` values and caches those, so the cache is far larger than the latent. K and V therefore have DIFFERENT widths (3072 vs 2048 on Coder-V2-Lite) — the only wired runtime where they do. `qwen35` is left because its `LayerCache` is `Attn { k: Tensor, v: Tensor }` built with `cat` rather than `KvBuf`, and `attn`/`lag_attn` take `&mut Option<LayerCache>` without knowing the format: wiring it means threading `fmt` through 8 call sites or pre-initialising the variant from the layer schedule. The `Clone`/`snapshot()` blocker is resolved; what remains is plumbing. ⚠ **Two ratios, and the smaller one is routinely misread.** `kv_bytes()` counts ALLOCATED capacity while `kv_f32_bytes()` counts LIVE rows, and both stores grow by doubling, so their quotient understates the format by up to 2x. Measured on LFM2.5-1.2B at 262 rows: allocated-vs-live **1.93x** where the format ratio is **3.76x** — the entire gap is one doubling. `kv_live_bytes()` exists to report the second. q4_0 the same: 3.64x allocated, **7.11x** live. On gemma4, q8_0 is token-identical to f32 over 16 generated tokens; on lfm2, token-identical over 12 while q4_0 diverges at token 3 but stays coherent — notably better than on qwen2.5-0.5b, where q4_0 diverged at token 1 and collapsed, consistent with only 6 of lfm2's 16 blocks carrying KV at all. The conv state is never quantized: it is bounded by `l_cache`, not by context. The shared-KV indirection is preserved (a shared block dequantizes `q[kv_src(il)]`, not `q[il]` — the latter hands blocks >= kv_from_start an EMPTY cache, whose attention output is finite and fluent). ✅ **KV quant and prefix caching now COMPOSE.** `QKvCache` supported exactly one access shape — sequential append, then full dequantize — so every feature wanting a different one had to refuse. `deep_clone`, a correct `Clone` (via a lazily-learned `Context`, since `Clone::clone` takes no arguments) and `clone_prefix` shipped 2026-08-16, and `PrefixCache` now stores either representation. A cross-representation seed is treated as a MISS, not a panic: a process can change `FERRIC_KVQ` while an in-memory cache outlives the change, and refusing to reuse costs a prefill where reusing the wrong layout costs correctness. ✅ **Batched decode composes too** (gemma4 and lfm2, verified solo-equivalent under `FERRIC_KVQ=q8_0` at n=2/3/4 and n=2/3/4/8 respectively). That refusal turned out to be conservative rather than structural: both stores index by SEQUENCE then layer, so batching needs no fork at all — row `i` appends to cache `i` exactly as the solo path does. What replaced the refusal is a check that all caches in one batch agree on representation. |

## B. Hardware coverage — the "all hardware" axis

| fabric | state | note |
|---|---|---|
| Metal / Vulkan / DX12 / GL | ✅ | via `wgpu`, one codebase |
| **Browser WebGPU** | ✅ | same source to `wasm32`. Genuinely rare |
| CPU fallback | ❓ | wgpu GL/software path only; no hand-vectorised CPU kernels (AVX-512 / NEON) verified |
| **CUDA native** | ❌ | NVIDIA reached through Vulkan, not cuBLAS/CUTLASS/cuDNN. Leaves tensor-core throughput on the table on the most common accelerator |
| **ROCm native** | ❌ | AMD via Vulkan; AITER/CK unused |
| **NPUs** | ⚠ | ⚠ **this row was wrong.** CoreML/ANE **exists** (`ferric-tensor/src/npu_coreml.rs`, receipt-gated on `MLComputePlan`) and WebNN **exists** (`ferric-web/examples/npu.rs`). Both serve one op (`bmm`) into a scheduler **no LLM runtime calls**, so no token of real inference moves through either. Still absent: Hexagon/QNN (MNN shipped it Jul 2026), Ascend/CANN, OpenVINO, DirectML. See `docs/backend-expansion.md` |
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
| **Multi-token prediction (MTP)** | ⚠ | ⚠ **this row said ❌ and that was wrong.** `qwen35` has a working draft block — `Qwen35::mtp_forward` / `mtp_forward_h` with its own `MtpCache` — and `ferric-serve` drives speculative rollback against it. Its cache stays f32 under `FERRIC_KVQ` on purpose: one layer, history discarded every verify round. Still absent: nemotron skips MTP at load, and gemma4 ships an `mtp-` file nothing reads |
| **LoRA / adapter serving** | ❌ | only DeepSeek's `q_lora_rank`/`kv_lora_rank` MLA internals exist — that is not adapter serving |
| Absorbed-MLA path | ❌ | refused at load; it is the variant that makes the KV cache small |
| Q-LoRA-factored attention | ❌ | refused at load; needed for full-size DeepSeek |

## D. Quantization

| format | state |
|---|---|
| Q4_K, Q5_0, Q6_K, Q8_0, IQ4_XS, IQ4_NL, Q2_0 ternary | ✅ packed kernels |
| **MXFP4** | ✅ **read path + packed kernel** | ggml type **39**, 32-value / 17-byte block. Dequant bit-identical to ggml over 11,124,608 real elements and the full 4096-pair (E8M0 scale x E2M1 code) grid. The packed GPU kernel is verified end to end: on a 72-tensor MXFP4 checkpoint it is **token-identical to the f32 dense fallback** over 16 steps (logit FNV differs by ~1 ulp, which is the accumulation order, not the values). Resident at the format's own 0.53125 B/elem rather than f32's 4 — asserted through `QMatrix::from_bytes` AND through `block_bytes(39)`, the predicate every loader consults; both defect vectors are mutation-checked. ⚠ The E2M1 conversion holds a factor of two back (doubled table x 2^(e-128)) because 2^128 is not representable in f32 — a from-the-spec `2^(e-127)` sends every element of an e=255 block to inf where ggml returns finite. Caught by the exhaustive grid, not by reading |
| FP8 (E4M3/E5M2) | ❓ |
| AWQ / GPTQ | ⚠ a ternary-GPTQ example exists; general support unverified |
| KV cache quant | ⚠ q8_0/q4_0/q4_1 primitives in `ferric-tensor`, unwired (see §A, §L) |

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
2. ~~**MXFP4**~~ — **DONE** 2026-08-16: read path bit-exact vs ggml, and the packed kernel shipped, so
   an MXFP4 weight is resident at 0.53125 B/elem instead of being expanded ~7.5x to f32 on load.
   Verified token-identical to the dense fallback end to end on a real 72-tensor checkpoint.
3. ~~**KV cache quantization**~~ — q8_0 **done** 2026-08-16 for the dense runtime (3.76x context/GB
   at +1.8% ppl). Two things remain, and the first is the bigger win:
   a. ⭐ **The block axis is wrong for K.** The shipped layout quantizes 32 consecutive channels
      along a row — exactly where K's outlier channels live. Measured on captured K/V:
      per-block(32)-along-row gives K asym rel-rmse **0.09591** at 5.00 bits/value, while
      per-channel x 32 tokens gives **0.03495** at 5.11 bits/value: near 3x better error for
      the same bit budget. 4-bit KV is unusable today because of the AXIS, not the bit width,
      so this is the change that would make it usable.
   b. The other four runtimes: only `qwen3` is wired.
4. ~~**RadixAttention-style prefix tree**~~ — ⚠ **this item was already done when it was written.**
   `ferric-llama/src/prefix.rs` shipped in `f88b4d8` ("5.17x on an agent workload, 84% of prefill
   skipped") on top of `ferric_kv::RadixIndex`, and `ferric-kv` also carries `PagedKv`, `BlockPool`
   and refcounted block tables. What was genuinely missing was narrower: `RadixIndex` is a *trie*
   with no node splitting, no reference counting and no eviction, which is what the 2026-08-16 work
   added. Ranking a shipped feature as unstarted is the same staleness as the NPU row in §B.
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

## K. ✅ Batched decode on ALL FIVE runtimes

This section read "exists on ONE runtime of five" until 2026-08-16. `forward_batch` is now implemented
and **adversarially verified token-identical to solo decode** on every runtime, f32 and KV-quantized:

| runtime | batched decode | covers | verified at |
|---|---|---|---|
| `qwen3` (Dense) | ✅ | qwen2, qwen3, llama, phi3, gemma, gemma2, gemma3 | n=2..5 |
| `qwen35` (Hybrid) | ✅ | qwen35, qwen35moe, laguna | n=2, incl. `FERRIC_KVQ=q8_0` |
| `lfm2` | ✅ | lfm2 | n=2/3/4/8, incl. q8_0 |
| `gemma4` | ✅ | gemma4 | n=2/3/4, past the 512 window, incl. q8_0 |
| `deepseek2` | ✅ | deepseek2 | n=2/4, MLA + MoE + YaRN, incl. q8_0 |

Continuous batching in `ferric-serve` is therefore no longer Dense-only.

⚠ **Every one of these was proven by re-running each sequence SOLO and comparing token ids** — not by
inspection, and not by "the output looks right". Two of the five had a real defect that only that
comparison caught: `gemma4`'s first equivalence example missed a dropped rope scaling (token ids
identical, logits off by 8.2e-1, printed but not asserted), and `qwen35`'s KV-quant wiring left the
BATCHED cache arms unwritten, so every token attended to itself alone — finite, fluent, silent.

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

## L. ⭐ KV-cache quantization — the append constraint picks the scheme, and it is measured

`ferric_tensor::kvquant` (2026-08-16). `QKvCache` is `KvBuf`'s quantized twin: preallocate, append
rows, double when full — except a row is stored as llama.cpp blocks. **8.5 bits/value (q8_0), 5.0
(q4_1), 4.5 (q4_0)** against f32's 32, so **3.76x / 6.40x / 7.11x more context per GB.** On
Llama-3.2-1B that is 536.9 MB → 142.6 MB of KV at 8k context.

**The scheme is not a free choice — the granularity has to survive one append per token.** Stated as
code in `append_cost`, and measured on **captured** K/V from a real prefill (`examples/kv_capture.rs`
→ `examples/kv_quant_error.rs`), not on gaussians. Relative RMSE, mean over all layers, Llama-3.2-1B
(Qwen2.5-0.5B agrees to the third digit):

| granularity | 8-bit K | 8-bit V | 4-bit K | 4-bit V | append |
|---|---|---|---|---|---|
| per-tensor | 0.0157 | 0.0189 | 0.282 | 0.335 | FULL REQUANT |
| per-token | 0.0127 | 0.0116 | 0.229 | 0.209 | in place |
| **per-block(32) along the row** | **0.0072** | **0.0062** | **0.131** | **0.113** | **in place** ← shipped |
| per-channel × 32 tokens | 0.0038 | 0.0054 | 0.069 | 0.097 | group-flush |
| per-channel | 0.0045 | 0.0067 | 0.080 | 0.121 | FULL REQUANT |

**Three results worth keeping.**

1. **"K needs per-channel" is half right, and the half that is wrong is the half people quote.**
   Static per-channel does beat per-block(32) on K (0.0045 vs 0.0072, ~1.6x) — and is *worse than
   per-block(32) on V* (0.0067 vs 0.0062). The K/V asymmetry is real and reproduces on both models.
2. **The per-channel variant that wins is the one grouped over a token window**, which also happens
   to be the only per-channel variant that can be appended to at all. Plain per-channel shares one
   scale with tokens that have not been generated yet, so a new token that moves a channel's max
   invalidates every code already stored for it — O(len·width) per token, the exact quadratic
   `KvBuf` exists to avoid.
3. **f16 scales cost nothing measurable.** Per-block(32) 8-bit is 0.00721 with f32 scales and 0.00721
   with the f16 scales q8_0 actually stores.

**Verification.** GPU kernels are diffed against a CPU reference (`kvquant::reference`) that mirrors
them line for line; on the captured tensors **0 of 504,832 packed words differ** on Llama-3.2-1B and
0 of 187,680 on Qwen2.5-0.5B, all three formats, and every dequantized value is bit-identical.
Row-at-a-time appends are required to equal a one-shot quantize word for word, on a width whose
per-row block count is *odd*, so consecutive rows share a scale word and the boundary case runs on
every append.

**What remains.** Wiring. Each runtime's `Cache` holds `(KvBuf, KvBuf)` per layer; the change is to
hold `(QKvCache, QKvCache)` and dequantize into the attention read. That is a small, obvious edit per
runtime — and it is *not* free of a decision: dequantizing the whole cache per step trades the memory
win for bandwidth, so the version worth having reads the packed blocks inside the attention kernel.
The layout was chosen for exactly that: `QKvCache`'s codes/scales split is word-for-word the layout
`Q8_0Weights`/`Q4_0Weights`/`Q4_1Weights` use, so a quantized K cache already *is* a packed weight
matrix in the shape `matmul_q*` reads. **Verified as a layout, not as a call**: those types only
construct from GGUF bytes today, so handing them a live cache needs a from-buffers constructor first.

## M. The K-axis change, scoped precisely (2026-08-17)

§A item 3a says 4-bit KV is unusable because of the *axis*, not the bit width. Measured on captured
K/V, mean over 24 layers, relative RMSE:

| granularity | bits/value | K asym | V asym | append cost |
|---|---|---|---|---|
| per-block(32) along the row — **what ships** | 5.00 | 0.09591 | 0.08660 | in-place |
| per-channel x 32 tokens | 5.11 | **0.03495** | 0.07010 | group-flush |

Near 3x better error on K for the same bit budget. The win is K-specific (V barely moves), which is
what outlier channels predict: the shipped layout puts 32 consecutive CHANNELS in one block, so one
outlier crushes the other 31.

### It is an INDEX CHANGE, not new arithmetic

Checked against the kernels rather than assumed, because "new kernels" was the reason this kept being
deferred. Both directions are the same 32-value block with the same scale math; only the addressing
differs:

- **quantize** reads `src[base + j]` for `j in 0..32`. Grouped wants `src[base + j*width]` —
  a source STRIDE, one term.
- **dequantize** is one invocation per output element and derives `(block, lane)` from `(r, c)`:
  shipped `b = (row0+r)*nblk + (c>>5)`, `j = c & 31`; grouped `b = (r/32)*width + c`, `j = r & 31`.

The pack/unpack, the E8M0/f16 scales, the two-blocks-share-a-word rounding: all unchanged.

### What actually has to be built

1. **Index mappings** (~20 lines) — plus widening the kernels' `array<vec4<u32>, 2>` info buffer,
   which is full: `info[0] = (n, grid_w, blk_start, blk_end)`, `info[1] = (base, row_stride, nblk,
   row0)`. A third `vec4` touches all six kernels.
2. **Staging** (~60 lines) — a group is not quantizable until 32 tokens exist, so the partial group
   stays f32 in the cache. Bounded at `31 x width x 4` per cache: ~63 KB per gemma4 layer, ~1.9 MB
   over 15 KV-owning layers x2. Re-quantizing the partial group on every append instead would
   quantize already-quantized values, compounding loss.
3. **Plumbing** (~40 lines) — K and V are separate `QKvCache`s already, so the asymmetric scheme the
   measurement recommends (K grouped, V per-block) needs only a per-side format, not a new type.
4. **Tests** — the CPU oracle already exists: `group_of` / `n_groups` handle `PerChannelGroup(g)`.
   Diff the GPU path against it, and mutation-test the stride (a `step` of 1 where `width` is meant
   silently reproduces today's layout and passes any test that does not compare against the oracle).

### Built 2026-08-17: `GroupedKvCache`

The estimate above was still too big. It needed **no new kernel and no info-buffer change**: a group is
`[GROUP, width]`, and transposed it is `[width, GROUP]` — which the existing block quantizer reads as
`width` rows of exactly one 32-value block each, i.e. one block per channel spanning GROUP tokens. So
`GroupedKvCache` is a permute plus `QKvCache`, inheriting the pack/unpack, the scale formats and the
shared-word rounding rather than reimplementing them. Block `g*width + c` means a flush appends
`width` blocks at the end and never rewrites an earlier one.

Measured, 4-bit, on synthesised outlier-channel data (4 of 64 channels ~30x the rest):
**per-block(32) along row 0.13261 → per-channel x 32 tokens 0.06540**, a 2.0x improvement, consistent
with the 2.7x measured on real captured K.

Verified by three arrival patterns that must agree — one shot, one row per step, and a ragged schedule
straddling the group boundary — on input deliberately non-smooth in BOTH axes, so a crossed permute
cannot look plausible. Three mutations fail it: identity permute, dropped staged tail, dropped
transpose (which silently reproduces today's layout).

**Costs, stated:** the tail below `GROUP` stays f32, bounded at `(GROUP-1) * width * 4` independent of
context; and `dequantize` pays one permute over the cache on top of the dequantize it already pays. A
fused transposing dequantize kernel would remove the second; it is not written.

### `KvStore`: K and V as different KINDS, not different parameters

The pairing the data asks for is asymmetric, so a single shared "format" field could not express it.
`KvStore::{Block, Grouped}` lets each side be configured independently; they were already separate
objects, this makes them separately *typed*.

⚠ `clone_prefix` is available on `Block` and **refused** (`None`) on `Grouped`. A prefix ending
mid-group would have to split a quantized block, and rounding the length down to a group boundary
would return FEWER tokens than asked for while reporting success — a prefix cache that quietly caches
less than it claims. `None` makes the caller treat it as a miss: one prefill, no correctness risk.
Same judgment as the cross-representation seed in §A.

### Wired into the dense runtime, and the axis beats the bit width

`qwen3` now holds `(KvStore, KvStore)` and configures the two sides separately.
`FERRIC_KVQ_K_AXIS=grouped` opts K into the token-grouped layout; unset keeps block/block, which is
byte-for-byte today's behaviour (verified: the f32 decode fingerprint is still `35968411ed3b5190`,
the value recorded against a separate-worktree baseline earlier).

Teacher-forced perplexity, one token per step (the append path), same 160-token passage for every
row, qwen2.5-0.5b-q8_0. **f32 = 31.2113.**

| fmt | bits/val | ppl, block K | ppl, **grouped** K | top-1 block | top-1 **grouped** |
|---|---|---|---|---|---|
| q8_0 | 8.5 | 31.7614 | **31.1885** | 91.88% | **98.12%** |
| q4_0 | 4.5 | **106.5980** | **32.2271** | 53.75% | **92.50%** |
| q4_1 | 5.0 | **196.6680** | **31.8704** | 47.50% | **96.25%** |

4-bit KV goes from **catastrophic to +3.3% perplexity** purely by changing which axis shares a scale.
Mean KL on q4_0 drops 1.116 -> 2.749e-2, a factor of 40.

**It also resolves an anomaly this document recorded earlier.** §A noted that q4_1 (5.0 bits) scored
WORSE than q4_0 (4.5 bits), which is backwards for an affine codebook, and concluded the ordering was
meaningless because both had left the rails. With the right axis the ordering is correct — q4_1
**31.8704** beats q4_0 **32.2271** — confirming that reading: the inversion was an artifact of two
broken configurations, not a property of the formats.

⚠ q8_0-grouped's ppl sits a hair BELOW f32 (31.1885 vs 31.2113, mean KL 9.9e-5). That is noise on one
passage, not quantization improving a model, and should not be quoted as a win.

⚠ **One model, one passage.** The effect is far too large to be an artifact, but a sweep across
checkpoints and context lengths has not been run.

`prefix.rs` refuses to cache a grouped side rather than round a prefix down to a group boundary, so
prefix caching and grouped K do not yet compose. The other four runtimes still hold `QKvCache` pairs.

## N. DeepSeek V4 Flash — a different architecture, and Ferric already holds half the format gate

`deepseek2` is the GGUF **architecture string** for DeepSeek V2/V3, not a generation label. V4 Flash is
not a variant of it: llama.cpp registers a separate `LLM_ARCH_DEEPSEEK4`, and **stock upstream cannot
load V4 at all** — the working implementations are community forks.

Verified 2026-08-17 from the port write-up and the GGUF cards, not from prior belief.

### What V4 Flash actually is

| piece | detail |
|---|---|
| attention | per-layer compressors `attn_compressor_{ape,kv,gate,norm}` producing a **latent K cache**; compression ratios alternate `0`/`4`/`128` across 43 layers |
| indexer | a SEPARATE pass — `indexer_compressor_*` plus `indexer_proj` — driving sparse attention at `top_k=512` (the "lightning indexer") |
| residuals | **replaced by hyper-connections**: count=4, Sinkhorn iterations=20, eps=1e-6, per-layer `hc_attn_*`/`hc_ffn_*` plus an `output_hc_*` triple |
| MoE | 256 experts x 6 active + 1 shared (from V3.2), with a NEW `sqrtsoftplus` gate |
| KV | **three K caches per layer** — standard SWA, compressed-attention, and indexer K where `compress_ratios[il] == 4` |
| weights | FP8 **e4m3** attention weights with **e8m0 per-block scales**; **FP4** routed experts. There is NO fp16/bf16 distribution — a converter must handle these natively |

### Why Ferric is closer than the version number suggests

The weight formats are half done as of today. **MXFP4 shipped this session** — E2M1 elements under a
shared **E8M0** block scale, bit-identical to ggml over 11,124,608 real elements and the full 4096-pair
(scale, code) grid, including the factor-of-two ggml holds back because `2^(e-127)` at `e=255` is not
representable in f32. V4's routed experts are FP4, and its attention scales are the same E8M0 encoding
Ferric now decodes exactly.

The **FP8 e4m3 element decode now exists too** (`E4M3_TO_F32_BITS`, exhaustive against
`torch.float8_e4m3fn`), and the F8_E4M3_B128 container it sits in has been reconciled against real
files and the fork's source — see the verdicts below.

The larger work is the forward graph — compressors, the indexer pass, hyper-connections with Sinkhorn
normalisation, and `sqrtsoftplus` gating — none of which resembles `deepseek2`'s MLA.

### F8_E4M3_B128 format verdicts — three sources reconciled, 2026-08-19

Ferric shipped the format on two labelled assumptions plus a type-id guess. Three independent lanes —
(A) byte-level measurement of the only located native V4 GGUF
(`nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF` → `DeepSeek-V4-Flash-FP4-FP8-native.gguf`, 156,148,189,760
bytes, x-repo-commit `0b34e0b6`), (B) an independent re-measurement of the same file plus the
`teamblobfish/DeepSeek-V4-Flash-GGUF` requants, and (C) the converter/dequant source
(nisparks/llama.cpp `wip/deepseek-v4-support` @ `9d36408`, cross-checked against ggml-org master @
`b062ba7`) — were reconciled against the code. A header measurement outranks source reading outranks
a model card.

| claim | shipped as | verdict | evidence |
|---|---|---|---|
| 129 bytes per 128-element block | assumed | **CONFIRMED** | all 365 type-42 tensors in the native file measure exactly `n/128 × 129` by offset arithmetic (two independent parsers, zero deviants); fork struct `static_assert sizeof == 129` |
| payload first, scale byte LAST | assumed | **REFUTED — corrected** | byte 0 of every block is the scale (2 distinct values, entropy ≈ 0 at phase 0 of 129; phases 1–128 full FP8 spread, sign bits ≈ 50%) in two tensors × two independent probe methods, calibrated on the file's known scale-first MXFP4; fork struct `{ uint8_t e; uint8_t qs[128]; }`; converter writes scale to byte 0 (`convert_hf_to_gguf.py:9429-9432`). `deq_f8_e4m3_b128` now reads **scale first** |
| E8M0 scale bias `2^(e-127)` (`e8m0_bias127`), not ggml's halved `2^(e-128)` | assumed | **CONFIRMED (source level only)** | fork dequant uses `GGML_E8M0_TO_FP32` = `e << 23` = bias-127 (`ggml-quants.c:649`, `ggml-impl.h:439-473`); upstream converter reads the same checkpoint bytes as `torch.exp2(bits − 127.0)`. File bytes CANNOT discriminate: observed scale bytes 115/116 fit both conventions one octave apart. The halved convention remains MXFP4-only. Code unchanged |
| type id 42 for F8, colliding with a Q2_0 | assumed | **CONFIRMED — and the collision is worse** | 42 confirmed in the native file (365 tensors) and in fork source (`gguf-py constants.py:4115`). But ggml-org **master** also assigns 42 — to its own `Q2_0` (`block_q2_0`: f16 + 16 code bytes = **64 values / 18 bytes**, ≠ PrismML's 128/34). `resolve_type_42` now knows all three strides and names mainline Q2_0 in its refusal (still undecoded here). Never key on `general.file_type` either: nisparks ftype 41 = `MOSTLY_F8_E4M3_MXFP4`, upstream ftype 41 = `MOSTLY_Q2_0`, and the native file says 41 |
| element format = `torch.float8_e4m3fn` | verified vs torch | **corroborated** | fork's `ggml_f8_e4m3fn_to_fp32` implements the same fn-variant semantics (no infs, `0x7F`/`0xFF` NaN, max 448); payload sign-bit statistics consistent. NOT yet diffed against reference dequantized values from a real file |

One deliberate divergence, kept: ggml's E8M0 macro ships its NaN branch commented out, so a `0xFF`
scale decodes to **+Inf** there; Ferric keeps OCP's **NaN** so a poisoned scale is caught rather than
multiplied through. Unreachable from any sane quantizer (it would mean a block scale of 2^128).

Corroborating context from the same sweep: `general.architecture = deepseek4` confirmed in real files
and all four fork lineages; MXFP4 (type 39) carries the routed experts at exactly ggml's 17 B / 32
values; upstream mainline has **no F8 type at all** (it dequantizes FP8 at convert time), so stock
llama.cpp builds reject any type-42 file with "invalid ggml type 42" whichever meaning was intended.
Caution for the eventual loader: the nisparks lineage and upstream diverge on tensor names
(`attn_compress_kv` in real circulating files vs `attn_compressor_kv` upstream) and on KV keys —
parse on the KV-key set actually present, not on an assumed lineage.

**Still unverified, and stated as such:** (1) the E8M0 bias verdict rests on the fork's source, not
on file bytes — a dequantized-reference diff would close it; (2) per-element payload decode against
reference values from a real file (order + fn semantics presumed from source + statistics); (3)
whether OTHER V4 GGUF quantizers reuse id 42 or another id — one native file exists and its hosting
repo could not be positively named (CAS `_id` matched no probed public repo); (4) PrismML Q2_0's own
34-byte geometry was not re-verified by any of these sources; (5) the MXFP4 scale convention inside
V4 files (stride matches ggml exactly; semantics untestable from sizes); (6) nothing here validates
the forward graph — compressors, indexer, hyper-connections, `sqrtsoftplus` remain unimplemented and
unmeasured.

### And the KV work compounds here

V4 keeps **three** K caches per layer. KV memory is therefore more binding on this architecture than on
any runtime Ferric currently ships, which makes §M's grouped-K result (4-bit from perplexity 106.6 to
32.2, at 7.11x the context per GB) worth more here than where it was measured.

### Addendum 2026-08-20: the element decode is now verified on REAL V4 weights

`blk.0.attn_kv_latent.weight` (2,097,152 elements, 16,384 blocks) fetched by HTTP range from the live
156 GB file and decoded two ways: Ferric's shipped `deq_f8_e4m3_b128`, and independently
`torch.float8_e4m3fn` × `exp2(e−127)` (the arithmetic upstream's own checkpoint converter uses).

**Bit-differing elements: 0 of 2,097,152.** Stats are real-weight-shaped (amax 0.15625, rms 0.026438,
zero NaN/Inf), which a wrong layout or wrong bias could not produce — the refuted payload-first order
decodes the same bytes into NaN-bearing garbage, and the wrong bias shifts every value one octave.

Still open on the format: nothing — layout, size, bias and element decode are all verified against the
file or bit-exact against a reference on the file's own bytes. Open overall: the forward graph, and
whether OTHER quantizers' V4 files match this one (only a single native F8 file exists publicly).

## O. Browser runtimes: demonstrated in a REAL Chrome tab (2026-08-20)

`web/runtimes.html` + a puppeteer driver run the shipped wasm bundle (0.95 MB — the entire runtime)
in headless Chrome with WebGPU, Dawn/Tint compiling the WGSL rather than our patched naga. This rung
exists because compiling to wasm32 proves nothing about running: `std::time::Instant` LINKS on wasm
and only panics at runtime, so this session's per-dispatch profiling crashed every tab while every
`cargo check --target wasm32` reported 0 errors in good faith. Fixed with a one-place `profclock`
shim (native Instant, wasm no-op; decode fingerprint bit-identical).

| tab run | verdict |
|---|---|
| LFM2.5-1.2B Q4_K_M (700 MB), f32 KV | ✅ " Paris. The city is known…" — word-identical to native, 6.8 tok/s cold / ~20 warm |
| same, **q8_0 KV** | ✅ identical output, receipt `kv q8_0 K:block` asserted IN FORCE on the page |
| gemma4 E2B Q4_K_M (3.27 GB) | ❌ `TypeError: Failed to fetch` — Chrome refuses the whole-file response into an ArrayBuffer. NOT the server (curl delivers all 3,427,880,384 bytes in 2.4 s), NOT wasm, NOT WebGPU: the load path's shape |

Two harness lessons, both now permanent in the driver: Chrome's persistent profile serves CACHED
pages for the same URL (`setCacheEnabled(false)` — never test a page you may not be running), and
the page renders its KV line AFTER applying the config so it is a receipt, not a default.

**The gemma4 blocker is the whole-file load path, precisely located.** `FerricModel::load(Vec<u8>)`
requires fetch → ArrayBuffer → copy, three simultaneous multi-GB tenants. The fix that changes the
ceiling is a STREAMING load — read the response body in chunks into wasm, upload weights as they
arrive, never hold the file twice — and `ferric-gguf`'s `Backing`/`GgufBacked`/`header_probe` were
built for exactly that access pattern. Until then the honest browser matrix is: dense family + LFM2
proven in-tab; gemma4 wired and native-proven but tab-blocked on load shape, not on capability.

### §O addendum: the small-model thesis, receipted in one tab (2026-08-20)

The strategic claim under test: parameters do not need to store the world when the runtime lives
inside it — a tab has native fetch into all data, so weights supply COMPETENCE and retrieval supplies
KNOWLEDGE, which is why ternary-compressed small models are sufficient rather than a compromise.

Every row is a real headless-Chrome run with the KV receipt asserted in force on the page:

| stack (PrismML ternary bonsai-1.7B, 2.125 bpw, ~450 MB) | verdict | tok/s |
|---|---|---|
| f32 KV | ✅ correct | 10.7 |
| q8_0 grouped-K KV | ✅ token-identical | 14.6 |
| **q4_0 grouped-K KV** | ✅ token-identical | 15.1 |
| q4_0 grouped-K + **retrieval-grounded QA** (`?ctx=`) | ✅ answered "0.95" | 7.8 |

The last row is the decisive one: the question asks for a fact that POSTDATES any possible training
data (this month's wasm bundle size), so no parameters anywhere could contain it — the answer came
from the document the tab fetched at inference time. Ternary weights + 4.5-bit KV + a 0.95 MB runtime
is ~12x smaller end-to-end than the f16/f32 default, output unchanged, and the quantized rows are
FASTER because decode at this size is bandwidth-bound. What this deprioritises, deliberately: the V4
forward graph (a 156 GB MoE is the premise this bets against) and the multi-GB streaming tab loader.

**LFM2 grouped-K (2026-08-20):** the K axis is now shared across runtimes via `KvStore` and one env
predicate. Measured on LFM2.5-1.2B, 24 greedy tokens vs f32: block-K q4_0 diverges at token **2**,
grouped-K q4_0 at token **12**, grouped-K q8_0 **never** — the dense ordering, reproduced on the
edge-native architecture. Batched decode composed untouched (solo-equivalent n=2/3/4/8 under grouped
q4_0), because the axis lives in the TYPE rather than in per-runtime flags. In-tab receipt:
`16 layers · lfm2 · BrowserWebGpu · kv q4_0 K:grouped`, identical output, **37.6 tok/s** — the
fastest LFM2 tab run recorded, quantized KV being less memory traffic on a bandwidth-bound decode.
