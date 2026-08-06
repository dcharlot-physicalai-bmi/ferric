# Runtime parity survey — August 2026

Where Ferric stands against the AI runtimes across Rust, C++, Python and Zig, and what was done about it.

## The landscape

| runtime | language | what it has that matters |
|---|---|---|
| **llama.cpp / ggml** | C++ | **Official WebGPU backend** ("LlamaWeb", [arXiv 2605.20706](https://arxiv.org/abs/2605.20706), MSR): 29–33% lower peak memory, 45–69% higher throughput than prior browser paths, within 5–50% of *native* decode, across 16 devices / 8 vendors. Vulkan across NVIDIA/AMD/Intel/Qualcomm. Speculative decoding, server with batching. |
| **vLLM** | Python/C++ | PagedAttention, continuous batching, chunked prefill, prefix caching, guided + speculative decoding, **disaggregated prefill/decode** (prefill is compute-bound, decode memory-bound, so they go on different GPUs). |
| **SGLang** | Python | **RadixAttention** — KV caches in a radix tree so prefixes are reused *across concurrent requests*. |
| **ZML** | Zig | One codebase → NVIDIA / AMD / TPU / Trainium / Intel / Metal via MLIR + OpenXLA + PJRT. Tensor-parallel on all of them. Trace → lower → compile → execute, JAX-style. |
| **Candle / mistral.rs / Burn** | Rust | Candle runs transformers in-browser via wasm; mistral.rs adds quantized inference and samplers on top; Burn covers training. |

## Where Ferric already stands

Cross-fabric bit-reproducibility (CPU/Metal/Vulkan/WebGPU), browser inference on WebGPU, GGUF-native quantized kernels, streaming weights with placement-invariance, sound certificates, and a training path. **ZML is the closest peer on breadth-of-fabric; nobody else pairs it with reproducibility.**

## Gaps found, and what shipped

Surveyed `crates/*/src/`: **paged KV, prefix caching, continuous batching, chunked prefill and block tables were all entirely absent.** Ferric's cache was one contiguous buffer per session — it cannot share anything and reallocates as it grows. That is the single largest parity gap against vLLM/SGLang, and it is the foundation the others build on.

### `ferric-kv` — paged KV + radix prefix sharing

Both ideas as **pure integer bookkeeping**: block ids, refcounts, a prefix tree. The storage they describe belongs to the caller — a GPU buffer natively, a JS array in a browser, a `Vec` in a test — which is what lets the dangerous cases be microsecond unit tests on any target, including wasm.

- **`PagedKv`** — block tables, refcounted sharing, `fork` for prefix reuse. Measured: 11 sequences behind a 512-token prompt cost barely more than one.
- **`RadixIndex`** — longest-prefix lookup over token sequences. On a 20-turn agent workload behind one system prompt: **>90% of prefill tokens reused**.

Two design points worth stating because they are where this goes silently wrong:

**Matches are whole blocks only.** A prefix agreeing on 100 tokens with a 16-token block yields 6 blocks (96 tokens), not 100. The seventh block holds KV computed from *different* preceding tokens; handing it over looks valid and makes the model quietly wrong.

**There is no copy-on-write, deliberately.** vLLM needs it because it shares partially-filled blocks. `fork` here shares **whole blocks only**, and that one rule makes writing into a shared block unreachable — a sharing sequence is always block-aligned, so its next append starts a fresh block. Cost: at most `block_tokens − 1` tokens of extra prefill per fork, 15 out of thousands. Benefit: an entire class of silent cross-sequence corruption cannot occur.

COW *was* written first. It came out when its test turned out to assert nothing and the path proved unreachable — dead safety machinery is worse than none, because it implies a hazard is handled.

## A cross-fabric finding worth acting on

**Firefox's WebGPU per-dispatch cost is ~1,040 µs against Chrome's 33 µs — 30× worse** ([browser benchmark data](https://deciphertech.io/blogs/how-browser-llm-inference-became-production-ready-in-2026-what-the-benchmark-data-reveals/)). Any design that issues many small dispatches per token is not portable; it is Chrome-only with a Firefox-shaped cliff.

This section previously said Ferric was already "the right shape" here because it batches into regions per layer via `ferric_tensor::batch`. **Measuring it overturned that** (`examples/dispatch_budget.rs`, exact host-side counts, no timing involved):

`batch` collapses *queue submissions* — ~1 per layer, genuinely good. But the Firefox penalty is per **dispatch**, and dispatches were **410 per token, 17.1 per layer**. Submission batching does not protect against this cliff at all. At Chrome's *own* 33 µs that is 13.5 ms/token of pure launch overhead — so in a browser Ferric is dispatch-bound, not compute-bound.

Tracing which kernels those were corrected the obvious next guess too. QKV, flash-attention and add+rmsnorm are **already fused**. The fat was `gather` — pure data movement doing no math — running 3× per layer to materialise the q/k/v windows of the fused QKV output.

All three are now gone, and two further fusions followed (q|k RoPE into one dispatch; the K and V cache appends into one), each verified byte-identical before being wired in: **410 → 290 dispatches/token, a 29% cut**, with `gather` at literally zero per token.

### ⚠ And it made no difference. The correction matters more than the optimisation.

This section previously claimed the browser went "495 → 271 ms/token, 1.83×". That was a **single-sample comparison of a fetch-bound metric** — at a 64 MB budget the demo re-fetches ~3.5 GB of layer weights per run, so its ms/token measures range-request traffic, not compute. Repeated, it spreads 491–1622 ms/token. The claim was noise.

Measured properly — all layers pinned so nothing is fetched during generation, median of repeated runs, baseline built from `f88b4d8` in a separate worktree:

| | dispatches/token | native decode | browser decode |
|---|---|---|---|
| baseline (`f88b4d8`) | 410 | **11.30 ms/token** | 25–29 ms/token |
| after all fusions | 290 (−29%) | **11.30 ms/token** | 25–27 ms/token |

**Identical.** The microbenchmark predicted ~3.4 ms/token of savings and delivered zero.

The reasoning error is worth stating plainly, because it is a general one: `dispatch_vs_submit` measures per-dispatch cost using **empty** kernels, which is launch latency against an idle GPU. Multiplying that by a model's dispatch count assumes launch cost is *additive* with compute. It is not — a GPU pipelines, so a dispatch carrying real work has its setup overlapped with the previous dispatch's execution, and the cost hides behind compute. "85% of a decode step is launch overhead" was false.

### What Ferric is actually bound by

Measured the same day: Ferric moves ~526 MB of q8_0 weights per decode token in 11.30 ms — **47 GB/s**. On the same machine `examples/bandwidth` reads at **463 GB/s**, and llama.cpp sustains **~326 GB/s** decoding a 27B model. So Ferric runs at roughly **10% of roofline and ~7× off llama.cpp**, and the gap is in matmul kernel efficiency — weight streaming, tiling, dequant — which no amount of dispatch fusion touches. That is the real target.

The fusions were kept: they are correctness-preserving, verified byte-identical, and a simpler program. And a fabric charging ~1,040 µs per dispatch (the published Firefox figure) may serialise where this one pipelines — but that is an **unverified hypothesis**, since no Firefox measurement has been taken.

### What was deliberately *not* done, and why it matters more

The tempting fix was to relax the `offset == 0` guard in `is_contiguous()` so every view could skip packing. A 15-agent audit of all 57 kernels — each SAFE verdict then handed to an adversarial skeptic — killed that idea:

- **Only 3 kernels honour a tensor's buffer offset** (`cat`, `gather`, `binary`). 52 index from element 0. An earlier comment in this repo claimed "9", counted by grepping `offset as u32`; that grep double-counts call sites and, worse, catches `rope`/`rope_scaled`, whose `offset` parameter is the **sequence position**. The error ran in the dangerous direction — it made the guard look optional.
- **Two hazards are not kernels at all.** `reshape` kept the buffer but hardcoded `offset: 0`; `reduce` discarded the tensor wrapper entirely. Both were correct only because `contiguous()` always materialises. Relax the guard and every `reshape` silently aliases the head of the parent buffer — same shapes, same numel, no assert, fluent wrong text.
- **It would have introduced a platform split.** `metal4_linear` honours offsets while its WGSL fallbacks `matmul_bt`/`matmul_bt_act` do not, so correctness would have become macOS-and-`FERRIC_METAL4`-dependent, silently.

So the fix was two kernels, not 52: `kv_write` and `rope` now read a **row-major strided view in place** (`src_off` and `src_row_stride` as their own info slots, never reusing the rope-position slot). One subtlety was load-bearing — `out` is allocated at exactly `numel`, so folding the source offset into rope's single shared base would have pushed every write past the end, and **WGSL silently discards out-of-bounds stores**, so rope would have returned zeros with no crash. Input and output bases are kept separate.

`reshape` and `reduce` were fixed anyway (a no-op today, a landmine otherwise), and `dispatch_budget` asserts the per-layer count so a future fan-out fails there instead of quietly becoming a Firefox cliff.

## Closed since: chunked prefill, and a ceiling that was not where anyone would look

**Chunked prefill** shipped (`nn::chunked_attention`, `examples/chunked_prefill.rs`): identical predictions
at every chunk size, peak score memory cut ~16× at 4096 tokens.

Chasing its first failure found something worth more than the feature. A one-pass prefill was **rejected
outright above ~862 tokens** — and the cause was not the quadratic `[T,T]` attention everyone assumes. It
was **`swiglu`**, an *elementwise* kernel dispatching `t · n_ff / 64` workgroups, which crosses WebGPU's
65,535-per-dimension limit **linearly in prompt length**, long before the quadratic term matters. Same
latent cliff in `gather_rows` (the embedding path of every model, `idx.len() · d`).

Both now fold into a 2-D grid — the convention several kernels already used — so single-pass prefill runs
to 4096 tokens and beyond, verified against the CPU reference. `run()` also asserts the per-dimension
limit *with the kernel name*, so the next one is a diagnosis rather than an opaque driver rejection.

The general lesson: on a per-element kernel the *linear* dimensions hit the dispatch wall first, and a
hard driver rejection reads nothing like the memory-pressure slowdown you go in expecting.

**Continuous batching** shipped (`ferric_kv::Scheduler`, `examples/continuous_batching.rs`). Measured on
Qwen2.5-0.5B against a static baseline over the same 6 uneven requests:

| | wall | total wait (steps) | 3-token requests' wait |
|---|---|---|---|
| static batching | 1334 ms | 408 | 272 |
| continuous batching | 1243 ms | 106 | **16** |

Generated tokens are identical under both. Note the wall clock is *nearly the same* — that is the honest
shape of this win and the reason the metric is per-request latency rather than tokens/sec: throughput can
look unchanged while every individual request is slow, which is precisely the failure being fixed.

The scheduler is bookkeeping only, so its real hazards are microsecond unit tests on any target including
wasm: a request lost under memory pressure, an overdrawn pool, a preempted sequence coming back short (it
keeps its tokens and recomputes — truncating its KV instead would leave it attending to a prefix of its
own history, fluent and wrong).

## Still open

- **Disaggregated prefill/decode** — matters at multi-GPU scale, which is not Ferric's near-term shape.
- **Wiring `ferric-kv` into `qwen3::Cache`** — the crate is complete and tested; the model still uses the contiguous cache.


## Where decode time actually goes (profiled, 2026-08-05)

Four bottleneck hypotheses were proposed and falsified this session. Rather than propose a fifth, the
decode loop was profiled with `sample` and the candidates measured directly. What is now **established**:

| claim | verdict | evidence |
|---|---|---|
| dispatch-launch-bound | **false** | 410 → 290 dispatches/token (−29%) changed decode by 0.00 ms |
| memory-bandwidth-bound | **false** | "47 GB/s" was bytes ÷ wall clock, and wall clock is ~half host time |
| CPU-bound by 14× | **false** | measured on a busy machine; does not reproduce |
| logits readback (608 KB/token) | **false** | 0.45 ms/token, 4% of a step |
| info-buffer allocation | **real but unfixable this way** | 1.29 ms/token; pooling made it *worse* (1.47–1.78) |

The profile: **62.9% of main-thread time is `__psynch_cvwait`**, plus 6.3% `mach_msg2_trap` — about 69%
**blocked**, not computing. The main thread is not CPU-saturated. Everything else is scattered below 1%
(wgpu resource-tracker drop glue, SipHash, Metal's range allocator, `memset`) with no hot function.

So the shape is: **10.81 ms/token = 5.79 ms host + 4.95 ms device, fully serialised.** Single-sequence
greedy decode has a hard dependency — token *N+1* needs token *N*'s output — so the host cannot run ahead
and the device idles while commands are built, and vice versa. There is no hot spot to fix because the
cost is the *round trip*, ~290 dispatches deep, once per token.

That reframes the remedies, and notably the most promising one is already built:

- **Batch sequences — and this does NOT currently work.** It was tempting to assume
  `ferric_kv::Scheduler` amortises one host build across many sequences. It cannot:
  `Qwen3::forward_cached(tokens, cache)` takes exactly **one** sequence's cache, so a step with N live
  sequences issues N separate forward passes — N host builds, N GPU round trips. Measured
  (`examples/batch_throughput.rs`, settled machine, median of 5):

  | live sequences | ms/step | tokens/sec | vs 1 seq |
  |---|---|---|---|
  | 1 | 11.15 | 89.7 | 1.00× |
  | 2 | 22.47 | 89.0 | 0.99× |
  | 4 | 44.52 | 89.8 | 1.00× |
  | 8 | 89.54 | 89.3 | 1.00× |

  Exactly flat. Cost is perfectly linear in the sequence count. The scheduler removes head-of-line
  blocking — a real and measured win on per-request *latency* — but cannot amortise work the execution
  layer has no way to share.

  The fix is a **batched decode**: stack N sequences' tokens into `[N, d]` and run one forward, which
  needs each sequence's KV attended separately inside one kernel. That is exactly what paged attention
  provides, and why `ferric-kv`'s `PagedKv` is complete, tested, and unwired. Unlike the four falsified
  hypotheses above, this rests on a structural fact about the code rather than on an interpretation of
  a timing: one forward reads the 525 MB of weights **once**, N forwards read it N times.

  **How much is actually available — spiked, not assumed.** Comparing one N-token forward against N
  single-token forwards isolates exactly that amortisation, using existing code:

  | N | N × 1-token | 1 × N-token | speedup |
  |---|---|---|---|
  | 2 | 21.7 ms | 18.8 ms | 1.16× |
  | 4 | 42.6 ms | 19.4 ms | 2.20× |
  | 8 | 87.1 ms | 24.9 ms | **3.50×** |

  So roughly **3.5× at 8 sequences, not the naive 8×** — attention work grows with N, and the host build
  does not vanish. Still the largest available win by a wide margin, and now with a realistic target
  rather than a ceiling.

  ⚠ Caveat on those numbers: taken at load average 12.4, which is higher than ideal. The trend is
  monotonic and the N=1 row agrees with independent measurements, so the shape is trustworthy; the exact
  multipliers should be re-taken on a quiet machine before anyone quotes them. A later attempt at load
  40 produced nonsense (N=1 at 0.58×) and was discarded — `batch_throughput.rs` now refuses to run above
  load 8 for exactly this reason.

  ### ✅ Built — `Qwen3::forward_batch`

  All three pieces landed: `Tensor::rope_at` (per-row absolute positions, since each sequence sits
  wherever its own history reached), a per-sequence attention loop, and per-sequence KV append.

  | seqs | N separate | 1 batched | speedup |
  |---|---|---|---|
  | 1 | 10.79 ms | 10.78 ms | 1.00× |
  | 2 | 21.73 ms | 13.28 ms | 1.64× |
  | 4 | 43.10 ms | 18.14 ms | 2.38× |
  | 8 | 86.48 ms | 29.61 ms | **2.92×** |

  Against the 3.5× spike, which had no per-sequence attention loop — that loop is the difference, and it
  is exactly what paged attention would fold away. **This is the first change this session that moved a
  wall-clock number**, after four falsified hypotheses that did not.

  Correctness is asserted before any timing: every sequence's tokens must be **identical** to decoding it
  alone, checked over 24 steps at 2/4/8 sequences with deliberately *unequal* prompt lengths so a shared
  RoPE position or a crossed cache cannot cancel out. A batched path that leaked between sequences would
  still emit fluent text; there is no crash to catch it.

  Still open: fold the per-sequence attention loop into one paged-attention kernel, which is the
  remaining gap between 2.92× and linear.
- **Reduce host build time.** ~4 ms of the 5.79 remains unattributed by `host_ns()`; the profile shows it
  is diffuse rather than concentrated, so this is death-by-a-thousand-cuts, not one fix.
- **Speculative decoding** would break the serial dependency — the only remedy that attacks the structure
  rather than its symptoms.

The lesson for anyone continuing: this file has recorded four confident wrong answers. Do not add a
fifth without a host/device split *and* an A/B against a baseline built from a `git worktree`.
