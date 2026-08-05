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

All three are now gone: **410 → 338 dispatches/token, a 17.6% cut**, with every exactness check unchanged and `gather` at literally zero per token. In the browser, where the analysis said Ferric is dispatch-bound rather than compute-bound, that shows up directly: the headed-Chrome demo went **495 → 271 ms/token, 1.83×**, on the same model and prompt.

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
