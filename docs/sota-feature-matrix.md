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
| **heterogeneous fabric split** | ⭐ **`ferric_tier::fabric`** — when `m` experts are missing from device memory, some cross the bus and the rest compute IN PLACE on the host, concurrently. Latency-optimal is FreeToken's `q* = m·BP/BH`, derived from the residual host bandwidth `BR = BH − BP`. ⭐ **Energy gives a different answer BY KIND**: `E(q) = q·S·e_dev + (m−q)·S·e_host` is LINEAR in q, so its optimum is always a CORNER (all-bus or all-host) while the latency optimum is interior — they cannot coincide unless the paths cost the same per byte. The useful point is neither: minimise joules subject to a latency budget, verified against brute force at 120 budgets × 3 energy models. ⛔ `split_for_latency` shipped ROUNDING TO NEAREST and was 5% slower than its own claimed minimum — the latency V is asymmetric (falls at S/BR, rises at S/BP), so when BR < BP the optimum is the CEILING, not the nearest. Caught by the profiler printing a faster plan two lines under the one labelled "min latency". ⚠ `FabricProfile` has NO `Default`: measured or nothing |
| **HF checkpoints, no conversion** | ⭐ **`ferric_load::hf::HfCheckpoint` implements `GgufSource`** over `config.json` + safetensors, so EVERY runtime accepts a published checkpoint unchanged. This was the last and largest llama.cpp dependency: not a line of code and not a crate in the tree, but its Python converter as a mandatory step in front of every model — invisible in `Cargo.lock` and total in practice. Verified on LFM2-350M against `ref_logits.bin` from **HuggingFace transformers**, the implementation the weights shipped with: argmax 1020 = 1020, max |Δ| 0.0001, **correlation 1.000000**, all 148 tensors mapped. ⚠ Geometry is REVERSE-THE-SHAPE-KEEP-THE-BYTES — GGUF reports `ne[]` fastest-varying first, so `[out, in]` in PyTorch is `[in, out]` in GGUF over identical row-major data; transposing to 'fix' the mismatch yields a model that loads, runs and is wrong. ⚠ Per-`model_type` maps; `lfm2` only so far, and each new one should arrive with something that checks the weights land where the runtime thinks they do |
| **Q2_K, Q3_K** | ⚠ **read correctly, NO packed kernel** — the small-quant tier every large-model repo publishes was unreadable until 2026-08-26; now dequantized (84 B and 110 B per 256) with the layout traps that make them silent: Q2_K's `qs` byte index selects the ELEMENT while the shift selects the SUB-BLOCK, and Q3_K's high-bit plane is INVERTED (set = add nothing, clear = subtract 4) with its bit selector running 1..128 across BOTH halves, so a per-half index overwrites the other half's bits. ⚠ `QMatrix::block_bytes` does NOT list 10 or 11, so a Q2_K model loads through the **f32 dense fallback**: correct output, ~16x the resident bytes, and no error anywhere — the same trap `iq4_real_weights.rs` documents for IQ4_XS. ✅ **PACKED KERNELS SHIPPED 2026-08-26** — `Q2_KWeights`/`Q3_KWeights` + WGSL, wired into `block_bytes` and `QShard`, so Q2_K/Q3_K weights are now resident at 84 B and 110 B per 256 instead of 4 B per weight (**10.6x** on Llama-3.2-1B's four largest eligible tensors: 237 MB vs 2504 MB). ⭐ Verified by a **one-hot probe** — with `x = e_k` the matmul reduces to column `k` of `W`, every other term a hard zero, so the output IS the kernel's dequantization and accumulation order cannot contribute: **worst Δ 0.00e0**, bit-exact against `deq_raw` across 8 tensor/format pairs × 11 boundary-straddling positions. Random-vector Δ 4.49e-7 is pure f32 reordering. ⚠ An assertion that passes at exactly zero is the one most worth doubting, so it was mutation-tested: inverting Q3_K's high-bit sense gives **1.688e11** relative and reading Q2_K's `qs` sequentially gives **7.919e10** — a wrong block layout does not drift a few percent, it reconstructs a different number entirely, which is why nothing downstream catches it. Q3_K's sixteen 6-bit scales are unshuffled at LOAD, not per matmul |
| **writing quants** | ✅ **Ferric's own quantizer**, `ferric-gguf::quantize` — Q2_K/Q3_K f32→blocks in pure Rust, so producing a test file no longer means shelling out to another project's binary. Least-squares refinement against the codes actually emitted beats plain min/max on 8/8 cases (Q2_K 0.1774→0.1540, Q3_K 0.1388→0.1303 NRMSE). ⭐ **`dmin` is a SIGNED f16 and a negative one puts a sub-block's floor ABOVE zero**, which no conventional quantizer emits and every conforming reader already decodes — DC-offset cost x3.50 → x1.00. ⛔ And **worth 0.00% on real weights**: 0 of 1,484,800 super-blocks on Llama-3.2-1B chose it, because the polarity is per-SUPER-block and one negative weight among 256 settles it |
| FP8 (E4M3/E5M2) | ✅ **read path** in `ferric-load` — E4M3 verified against the format definition and E5M2 against `half::f16` (E5M2 IS f16's top byte: same exponent width, same bias), both **exhaustive over all 256 bytes**. ⚠ An FP8 tensor is NOT the weight — the value is that byte times a scale in a sibling `_scale_inv`/`_scale` tensor, so `get()` REFUSES and `get_scaled()` applies it; handing back the bare bytes gives right shapes and wrong magnitudes with nothing to catch it. No packed FP8 kernel yet |
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

**Fleet-uniform + fused (2026-08-20):** gemma4 and deepseek2 joined `KvStore`, so ONE type, one env
var and one constructor shape carry the K axis across all five runtimes. gemma4's measured spread
matches the pattern (q4_0 block diverges at token **1**, grouped at **7**, q8_0-grouped **never** —
third architecture, same ordering). And the grouped layout's last overhead is gone: a **fused
transposing dequantize** derives each output element's source block in index math, replacing the
per-read permute+contiguous over the whole history. Bit-identical to the two-pass oracle on all three
formats (zero differing bits; both index-math mutations caught), and batched decode re-verified
solo-equivalent through the fused path on lfm2 (n=2..8), gemma4 (n=2..4, past the sliding window) and
deepseek2 (n=2/4). Grouped-K now costs a bounded f32 staging tail and nothing per read.

**In-tab corpus retrieval (2026-08-20): one ternary model is BOTH retriever and generator.**
`?corpus=<url>` fetches a document set; the page embeds every chunk AND the question locally via
`embed()`, ranks by cosine, grounds generation in the winners, and renders the chosen chunk index as
a receipt the driver asserts on. On an 8-chunk corpus of distinct technical facts, bonsai-1.7B
(ternary, ~450 MB, NOT embedding-trained) ranked the correct chunk FIRST (cosine 0.819 vs 0.772
runner-up) and answered "0.95" — a fact postdating all training data. Retrieval and generation both
local, only the corpus remote, with an auditable record of which knowledge grounded the answer. The
two assertions are separate BY DESIGN: a wrong ranking is caught even when generation gets lucky,
and a generation failure is isolated from retrieval — the subject-vs-claim discipline applied to the
demo itself.

**At scale, with precomputed vectors (2026-08-20):** corpus embeddings are a function of (corpus,
model), so `embed_corpus` computes them ONCE natively and ships them as a ~1 MB `.fvec` beside the
corpus; the tab pays ONE question-embed plus dot products per query, independent of corpus size. On a
120-chunk corpus the correct chunk still ranked FIRST at the same cosine and margin as the 8-chunk
run (19:0.819 vs 0.772) and the answer held.

The file's model-identity check REFUTED its own first design and produced a finding: a bit-hash of a
probe embedding MISMATCHES across fabrics for the SAME model, because kernel selection is
capability-dependent (subgroup reduce on native Metal, plain on Chrome/Dawn) — the same dispatch takes
a different reduction order by construction, and 28 layers accumulate it to ulp order. v2 ships the
probe VECTOR and checks cosine > 0.999; the measured cross-fabric agreement renders on every run
(probe-cos 1.000000 — real at the bit level, invisible at six decimals). Rule: across fabrics, model
identity is a TOLERANCE check, never a bit-hash.

## P. The measurement instrument was broken where it mattered most (2026-08-21)

`ferric-joule` gates every energy claim this project makes, so its own soundness is upstream of all
of them. Preparing to *use* it for the first time surfaced four defects, three of which silently
neutralised the crate's central argument.

| # | defect | why it mattered |
|---|---|---|
| 1 | `Saving.successes` was ONE `u64` shared by both arms | `per_success()` divided both arms by the same number, so `fraction()`, `percent()` and the ranking came out bit-identical to `per_attempt()`. **A field that cannot change any comparison cannot correct one** — joules-per-completed-task was decorative |
| 2 | `Routed::against` charged both arms the ladder's own success count | the worst possible site: routing buys its saving by resolving work on weaker rungs, so the arm that saves the energy is exactly the arm whose success count differs. Savings read as free |
| 3 | the reversal test could not fail | it compared two *different* `Saving`s and reduced to `1180/164 < 1180/55` — the monotonicity of division, true for every input |
| 4 | "a `Saving` cannot be constructed by hand" was prose | every field was `pub`. Now `#[non_exhaustive]` + a `compile_fail` doc-test, the only place this is checkable, since doc-tests compile as an external crate |

**And the crate's own citation was overstated.** It read "44.6 kJ against 21.5 kJ, which *reverses*
the ranking energy-per-query gives you". The arithmetic says otherwise: 6.19x → 2.08x is a
**narrowing**, same winner. A real reversal needs a wider success gap, and now shows up as the two
savings **disagreeing in sign**, which `Display` warns about explicitly.

**Why it survived 82 tests:** `Saving` had *no consumer anywhere outside its own tests*. Nothing
exercised the type, so nothing could expose it. The fix therefore includes the paths a real consumer
needs — `compare_tasks()`, which grades the closures and **tallies** successes per arm so a caller
cannot report a rate no arm demonstrated, and `grade_tasks()` for machines with no readable sensor.

All four fixes are mutation-proven: reinstating each defect turns the suite red.

## Q. "Weights on the outside", graded — and what the control arm did to it (2026-08-21)

`crates/ferric-web/examples/lookup_vs_weights.rs`. The thesis under test: *a model small enough to
ship to a tab, with the corpus outside its weights, answers more questions correctly than a model
several times larger answering from memory.* 22 real, checkable facts; 66 corpus chunks with **every
answer flanked by two topically adjacent distractors** (Chernobyl beside Three Mile Island and
Fukushima; gold beside silver and platinum); both arms graded by the same word-boundary matcher.

**The prediction was registered in the source before the first run, and it was refuted.**

| arm | bytes | score |
|---|---|---|
| weights INSIDE — qwen1.5b, closed book | 1117 MB | **20/22 (91%)** |
| CONTROL — qwen3-0.6b, closed book | 397 MB | 14/22 (64%) |
| weights OUTSIDE — qwen3-0.6b + 66 chunks, generator as its own retriever | 397 MB | 14/22 (64%) |

**The control is the whole bench.** Retrieval's net contribution was **+0**: two questions answered
only *with* the passage, two *lost* by having it. The 64% was entirely what the small model already
knew, and on two questions an irrelevant passage displaced an answer the model would have given
unaided. Without the control this reads as "small + retrieval scores 64%" and credits the corpus with
work the corpus did not do.

### The cause was a retriever the code had already warned about

`FerricModel::embed` pools the last hidden state, and its own doc says doing that on a checkpoint not
trained for embedding "hands back plausible cosine scores that mean nothing". **Its guard checks the
runtime kind (`Dense`), not whether the weights were trained for the task**, so a generative model
passes it silently. Swapping in a checkpoint actually trained to embed:

| | generator as retriever | trained retriever |
|---|---|---|
| retrieval@1 | 8/22 | **19/22** |
| mean top1–top2 margin | 0.0108 | **0.1188** (11x) |
| distinct passages for 22 questions | 14 | **22** (no collapse) |
| lookup arm score | 14/22 | **18/22** |
| retrieval's net contribution | **+0** (+2 / −2) | **+4** (+4 / −0) |

**But a working retriever is not free.** `qwen3-embed-0.6b` is 639 MB, so the candidate ships 1036 MB
against the baseline's 1117 MB — the size ratio the thesis rests on falls from **2.8x to 1.08x**.
Pricing only the generator would be the same class of error as a baseline measured at 3.4% MFU.

**Standing result on memorised world facts: at equal bytes, memorising beat looking up, 20 to 18.**
That is the case most favourable to weights-inside, since every fact is in the baseline's training
data by construction.

### The other half: facts no model can have memorised

The bench above is the case *most favourable to weights-inside*, because every fact in it is in the
baseline's training data by construction. The regime the thesis is actually about is knowledge a model
cannot have — post-cutoff, private, local, specific. So the same harness runs against 22 invented
instruments with invented parameters (`web/qa_novel.tsv`, `web/qa_novel_corpus.txt`), each answer
flanked by two siblings carrying **different** values, every number in the corpus globally unique so
no answer can be matched by a word-boundary hit on another instrument's figure, and generated from a
fixed seed so it reproduces from the file alone.

**No positional shortcut can pass it.** Every passage carries three numbers (value, revision,
catalogue) and only half lead with the answer, so `copy-the-first-number` scores 11/22,
`copy-the-last` 11/22, `copy-the-middle` 0/22. Anything above 11 required reading the question. The
first version of this corpus put the answer first every time, where a pure copying heuristic would
have scored 22/22 — indistinguishable from the result the best arm actually got.

## R. The full matrix (2026-08-21)

| system | total bytes | memorised world facts | invented facts (66) | invented facts (1000) |
|---|---|---|---|---|
| **weights INSIDE** — qwen1.5b, closed book | 1117 MB | **20/22 (91%)** | **0/22** | **0/22** |
| **weights OUTSIDE** — Bonsai-1.7B **ternary** + **Q4_K_M** retriever | **860 MB** | **20/22 (91%)** | **22/22** | **22/22** |
| weights OUTSIDE — Bonsai-1.7B ternary + Q8_0 retriever | 1102 MB | 20/22 (91%) | 22/22 | 22/22 |
| weights OUTSIDE — qwen3-0.6b + Q8_0 retriever | 1036 MB | 18/22 (82%) | 19/22 | — |
| *control* — Bonsai-1.7B ternary, closed book | 463 MB | 17/22 | 0/22 | 0/22 |
| *control* — qwen3-0.6b, closed book | 397 MB | 14/22 | 0/22 | — |

**At 77% of the memoriser's bytes, the ternary + corpus system ties it on the facts it memorised and
beats it 22–0 on the facts it could not — and holds that across a 45x range of corpus sizes.**

On the invented corpus the ternary arm made zero errors at either corpus size: retrieval@1 22/22,
all 22 passages distinct, net contribution +22. On world facts it retrieves 20/22 with the 4-bit
retriever against 19/22 with the 8-bit one, so quantisation costs nothing on the *harder* retrieval
problem either — paraphrased questions against topically adjacent distractors, where the margin sits
at 0.1207 rather than the invented corpus's 0.16–0.21.

### What the three rows say that one row cannot

1. **Memorisation has a hard zero.** Outside its training data the 1117 MB model scores 0/22, and no
   parameter count moves that. This is the whole asymmetry: one architecture degrades, the other
   stops.
2. **Ternary buys the tie.** The 0.6B dense generator loses the world-facts arm 18–20; the 1.7B
   ternary generator draws it 20–20 in **1.2x the bytes of the 0.6B**. Ternary compression is what
   makes "small enough to ship" and "good enough to tie" the same model.
3. **The retriever is the new bottleneck, and it is the wrong size.** `qwen3-embed-0.6b` is 639 MB
   against the ternary generator's 463 MB — in a weights-outside system the *retriever* is the larger
   half. It is also the half the field already ships at 20–100 MB (MiniLM, bge-small, gte-small), so
   the concrete engineering path is a small trained retriever beside a ternary generator, and the
   1102 MB measured here is an upper bound, not the target.
4. **Retrieval quality is a capability, not a given.** Pooling a generator's hidden state retrieves
   8/22; a checkpoint trained to embed retrieves 22/22. The failure mode of the first is not an
   error — it is ordinary-looking cosine scores that rank near-arbitrarily, which a single-question
   demo cannot detect. §Q is the record of this project shipping exactly that demo.

This is the measurable form of the browser-native thesis: a tab has the network, so the corpus is free
to be current, private, and arbitrarily large, and the only thing that must fit in memory is the part
that reads it.

## S. The retrieval decay curve, and what a small retriever costs (2026-08-21)

§R asserted that a browser tab's corpus "is free to be current, private, and arbitrarily large". That
sentence rode on a single 66-chunk measurement. `crates/ferric-web/examples/retrieval_scale.rs`
measures it: one embedding pass over 1000 chunks, then arithmetic over nested subsets that all share
the same 22 answer-bearing passages, so the only variable is how many distractors the right answer
must outrank.

| chunks | retrieval@1 | margin (top1−top2) | top1 | second place | distinct passages |
|---|---|---|---|---|---|
| 22 | **22/22** | 0.4016 | 0.8299 | 0.4283 | 22 |
| 50 | **22/22** | 0.2809 | 0.8299 | 0.5490 | 22 |
| 100 | **22/22** | 0.2324 | 0.8299 | 0.5975 | 22 |
| 200 | **22/22** | 0.2121 | 0.8299 | 0.6178 | 22 |
| 400 | **22/22** | 0.1947 | 0.8299 | 0.6352 | 22 |
| 700 | **22/22** | 0.1760 | 0.8299 | 0.6539 | 22 |
| 1000 | **22/22** | 0.1610 | 0.8299 | 0.6689 | 22 |

**retrieval@1 is perfect across a 45x range, and no question ever collapses onto another's passage.**

**`top1` never moves.** It cannot — the question and its passage are unchanged at every size. The
entire effect is the *runner-up* climbing, 0.4283 → 0.6689, as more distractors get more chances to
look relevant. Stating the result as "retrieval degrades with corpus size" would be wrong in a way
this column makes obvious: nothing about the right answer degrades, the field behind it crowds.

**Decay is steep then flat.** The first few distractors cost the most (−0.1023 per doubling from 22
to 50); by 200 chunks it settles to **−0.0225 per doubling** averaged over the last four steps.
Extrapolating that rate, the margin would reach zero near **140,000 chunks** — an *extrapolation*
from seven points that assumes the settled rate holds, not a measurement, and the practical ceiling
sits well below it since ranking becomes a coin flip as the margin approaches zero. What is measured
is that 1000 chunks is nowhere near the limit.

### The retriever quantizes for free

| retriever | size | retrieval@1 | margin @66 | end-to-end |
|---|---|---|---|---|
| Q8_0 | 639 MB | 22/22 | 0.1826 | 22/22 |
| **Q4_K_M** | **396 MB** | 22/22 | 0.2132 | **22/22** |

The margin difference runs in the candidate's favour, which on 22 questions is sampling noise: the
claim is **no degradation**, never improvement. What matters is the size. With a 4-bit retriever the
weights-outside system ships **860 MB against the memoriser's 1117 MB** — so the headline moves from
"ties at equal bytes" to **wins at 1.3x smaller**, still 22/22 against 0/22 on facts nothing could
have memorised.

## T. Quantisation moves an embedding ~50x more than the fabric does (2026-08-21)

The tab's model-identity gate accepts `probe cosine > 0.999`, a threshold set in §O to tolerate
cross-fabric kernel selection. That raises a question the artifacts can now answer: **is a
requantised model the same model by that test?** It decides whether one `.fvec` can serve every
client regardless of which quantisation they downloaded.

Same 120-chunk corpus, same checkpoint, embedded twice — `qwen3-embed-0.6b` at Q8_0 and at Q4_K_M:

| comparison | probe cosine | the gate |
|---|---|---|
| same weights, **different fabric** (native Metal vs Chrome/Dawn) | **1.000000** | passes |
| same checkpoint, **different quantisation** (Q8_0 vs Q4_K_M) | **0.986131** | **fails** |

All 120 corpus vectors move together: min 0.9637, median 0.9777, max 0.9832 — **120/120 below the
gate**. So the threshold is well calibrated: it admits the noise it was written for and rejects a
change two orders of magnitude larger.

**But "different model" is not the same as "unusable vectors".** Ranking each chunk against the
*other* file's vectors, **120/120 still rank themselves first**. The geometry moved uniformly enough
that retrieval would have worked. The gate refuses a mix that would in fact have functioned — and
that is the right default, because it cannot see ranking, only geometry, and a genuinely different
model can also preserve some rankings by luck.

The deployment consequence is concrete: **corpus vectors are per-quantisation, not per-model.** One
`.fvec` cannot serve clients that chose different quantisations; each needs its own, which is one
`embed_corpus` run.

The page now splits the diagnostic by magnitude rather than reporting one message. Above 0.95 it says
the file was built with a different *quantisation* and names the fix; below, it says different model.
A near-miss almost always means the former, and the old wording sent the reader hunting the latter.

## U. Re-measured after the tokenizer fix — the result stands (2026-08-21)

Everything in §Q–§T was measured before `04151cf`, which corrected a real defect in the tokenizer:
`tokenizer.ggml.pre` was read nowhere, so the Qwen family got GPT-2's pre-tokenizer and mid-word
punctuation split differently from the reference. The invented-facts corpus is built from
**hyphenated proper nouns**, and those names *are* the retrieval signal — every distractor is a
structurally identical sentence differing only in the entity. So every retrieval figure had been
computed through a component since proven wrong on precisely the text carrying the signal.

Re-run on the 1000-chunk corpus with the identical retriever and corpus:

| chunks | margin before | margin after | Δ |
|---|---|---|---|
| 22 | 0.4016 | 0.4042 | +0.0026 |
| 100 | 0.2324 | 0.2326 | +0.0002 |
| 400 | 0.1947 | 0.1960 | +0.0013 |
| 1000 | 0.1610 | 0.1619 | +0.0009 |

**retrieval@1 is 22/22 at every size in BOTH runs, and 22 distinct passages in both.** `top1` moves
0.8299 → 0.8305. Mean margin change **+0.0011, and all seven sizes move the same direction** — a
small systematic improvement, consistent with vectors that now match the reference checkpoint.

**Nothing in §Q–§T changes.** The reason the fix barely moved the numbers is the reason the bench
could not have caught the bug: corpus and query pass through the same tokenizer, so ranking is
invariant to a consistent change in it. That is a property of the measurement, not a defence of it —
the vectors *were* wrong against the reference, and only a comparison with an independent
implementation could show it.

**Provenance:** figures in §Q–§T were produced before `04151cf` and re-confirmed after it. §S's
extrapolated 140,000-chunk ceiling uses the settled decay rate, which moves from −0.0225 to −0.0223
per doubling — inside the noise of a seven-point fit either way, so the figure is unchanged and
remains an extrapolation rather than a measurement.

## V. Closing the open-weight gaps (2026-08-25)

Prompted by Stripe's $7–8B acquisition of OpenRouter — model routing priced as infrastructure — a
sweep of Ferric against the 2026 open-weight tooling landscape. Ferric was stronger than expected on
the standards (an **MCP client** over stdio and Streamable-HTTP already, plus guided decoding, tool
calling, speculative decode, LoRA, imatrix). The real holes were in loading and retrieval.

| gap | status | evidence |
|---|---|---|
| **Sharded GGUF** (`model-00001-of-0000N`) | ✅ closed | 310 tensors byte-identical to the merged original, opened from part 1, 2 AND 3 |
| Streaming vs sharded input | ✅ refuses | positional readers cannot address multi-file checkpoints |
| **WordPiece** | ✅ closed | 3/3 vs `llama-tokenize` |
| **BERT encoders** | ✅ closed | 0.999999 (F16), 0.999996 (Q4_K_M) vs `llama-embedding` |
| **XLM-RoBERTa** | ✅ closed | 0.999995–1.000000 |
| **Cross-encoder rerankers** | ✅ closed | 6.585 vs 6.570; −8.366 vs −8.361 |
| **`/v1/rerank`** | ✅ closed | live HTTP, Cohere response shape, `top_n` + `return_documents` |
| Q4_K matmul fidelity | ✅ verified | 0.999996 — retires a risk under every quantised result here |

**Sharded GGUF was not a missing feature — it was "cannot open the file."** Every large checkpoint on
HuggingFace ships split; llama.cpp, vLLM and Ollama all follow the parts. Ferric loaded part 1 alone,
saw 128 of 310 tensors, and failed with `no tensor 'blk.11.attn_v.weight'` — a message naming a
tensor rather than the 182 in files it never opened. Two checks now guard it, because the first alone
was insufficient: the tensor count against `split.tensors.count`, **and** each shard's declared data
span against its actual length (truncating a part to 200 KB leaves the header intact, so the count
check passed and 310 tensors loaded).

### ⛔ The most expensive lesson: verification does not transfer between files

A "0.9615 XLM-R encoder divergence" was recorded here as an open bug and chased through nine
hypotheses — quantisation, the pooler, GELU, token types, the LayerNorm epsilon, the position offset.
Every elimination was correct. None was the cause. `bert_reference` hardcoded a **WordPiece**
tokenizer, right for `tokenizer.ggml.model == "bert"` and wrong for XLM-R's `"t5"`, and fed the
encoder four tokens for "Paris" where llama.cpp uses three.

Three things worth carrying forward:

1. **The tokenizer family is declared, not implied.** `bge-small` and `bge-reranker-v2-m3` are both
   `general.architecture = bert`; one is WordPiece, the other SentencePiece. `bert::Reranker` now owns
   its tokenizer so no caller re-derives this.
2. **A better number can mean a wronger parameter.** A position offset of 2 scored cosine 0.9726
   against the correct 0's 0.9615 — with another defect present, tuning toward the final scalar moved
   *away* from the reference. `llama-eval-callback` settled it: the reference's position-embedding sum
   for three tokens is −37.445366 and rows 0..2 sum to −37.445358.
3. **Per-op tracing is bounded; end-to-end comparison is not.** `FERRIC_BERT_TRACE=1` prints a sum per
   checkpoint tensor against the reference's 512. The first line disagreed on token *count* — 4 vs 3 —
   which no parameter sweep could have reached.

## W. The first non-transformer: Nemotron-H, verified (2026-08-25)

`general.architecture = nemotron_h` — NVIDIA's Mamba-2 / attention / MLP hybrid, current (official
NVIDIA GGUF, llama.cpp support March 2026). Of 42 blocks in Nemotron-3-Nano-4B, **21 are state-space
mixers, 17 are ReLU² MLPs, and four are attention**. Sequence mixing is a recurrence, not a quadratic
attention matrix, which is the point of the family.

**Verified at the distribution.** On the same 8-token prompt the top-10 match the reference token for
token *in the same order*, logprobs agreeing to ~0.01:

| rank | llama.cpp | Ferric |
|---|---|---|
| 1 | `<\|im_end\|>` −0.7757 | `<\|im_end\|>` −0.7843 |
| 2 | " The" −2.5101 | " The" −2.4986 |
| 3 | " This" −3.1766 | " This" −3.1770 |
| 4 | " It" −3.3102 | " It" −3.3060 |

### What the trace bought, measured

The per-op trace was built **before** the first end-to-end comparison — the ordering the BERT port
earned by spending nine hypotheses on a harness bug. It localised the one real defect on its first
run:

```
embd        -0.302475  vs  -0.302475   exact
blk0 (SSM)  -2.730604  vs  -175.85     WRONG
```

Embedding right, first SSM block wrong, so the search was one block wide. **The grouped norm is two
steps, not one**: the reference normalises each 960-wide group with *no weight*, then multiplies by
the full `{960, 8}` = 7680 tensor whose values differ per group. Passing that 7680-wide weight to
`rmsnorm` over 960-wide rows reads the wrong slice for seven groups in eight. After the fix, 0.10%.

### ⛔ The metadata lies about RoPE

The file declares `rope.dimension_count = 78`. The reference graph contains **zero** ROPE ops —
attention here is position-free because the SSM layers carry position. Trusting the key would have
rotated 78 of 128 head dims and produced fluent, wrong text with every shape assertion passing.

Also read off the graph rather than inferred: SILU covers **all** of xBC so the D-skip uses the silu'd
x; `dt` bypasses the conv while `x` does not; SWIGLU takes `z` **first**; and Ferric's `ssm_scan` adds
`D·x` internally where ggml emits it as a separate node, so doing both would double the skip term.

### Two metrics that do NOT grade fidelity

**Per-block sums.** Their error tracks the cancellation ratio `|sum|/max|v|` — 5.18 → 0.10%, 0.14 →
13.85%. A sum that is a near-cancellation of large opposing values turns a tiny per-element difference
into a huge relative error *in the sum* while the tensor is fine. Sums localise a gross defect; they
cannot measure agreement.

**Greedy token paths.** Both runtimes emit " Paris." then split. The leader at that position holds
under half the mass, so which of several near-ties wins says nothing. The distribution comparison is
the test; the token path is not.

⚠ **No incremental state yet.** Every decode step re-runs the whole prefix — O(T²), which is precisely
what a state-space model exists to avoid. Carrying conv and scan state is a separate correctness
problem, and conflating it with mixer correctness is how a state bug gets misattributed.
