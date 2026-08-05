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

**Firefox's WebGPU per-dispatch cost is ~1,040 µs against Chrome's 33 µs — 30× worse** ([browser benchmark data](https://deciphertech.io/blogs/how-browser-llm-inference-became-production-ready-in-2026-what-the-benchmark-data-reveals/)). Any design that issues many small dispatches per token is not portable; it is Chrome-only with a Firefox-shaped cliff. Ferric batches into 10 regions per layer via `ferric_tensor::batch`, which is the right shape — but this is a portability budget worth measuring against, not assuming.

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
