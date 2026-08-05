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

## Still open

- **Chunked prefill** — bounds memory for long prompts; kimi-k3 lists it as its own highest-value gap too.
- **Continuous batching** — now unblocked by paged KV, but needs a scheduler.
- **Disaggregated prefill/decode** — matters at multi-GPU scale, which is not Ferric's near-term shape.
- **Wiring `ferric-kv` into `qwen3::Cache`** — the crate is complete and tested; the model still uses the contiguous cache.
