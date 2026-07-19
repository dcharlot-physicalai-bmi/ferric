# Ferric — Path to Dominance

Goal: **the dominant Rust framework for all of AI.** This document is the strategic
counterpart to `SOTA.md` (the technical scorecard). SOTA.md says *where we stand on
kernels*; this says *which battles to fight*. Grounded in three 2026 competitive surveys
(the Rust field, the cross-language incumbents, and the "all of AI" adoption map).

## The field, and the bar each incumbent sets

**In Rust (direct competitors):**
- **Burn / CubeCL** — the primary adversary. ~140 TFLOP/s portable tensor-core matmul from
  one source (CUDA/ROCm/Metal/Vulkan/WebGPU), trains, funded team (Tracel AI). ~2 years
  ahead of us on kernels. **But: no bit-exact cross-vendor parity, no LLM runtime, browser
  is a reduced path.**
- **Candle** (HuggingFace) — owns *adoption*: 20.7k stars, the model zoo everyone forks, HF
  gravity. Inference-only, kernels competent-not-SOTA.
- **mistral.rs** — owns *serving* in Rust: PagedAttention, continuous batching, FP8 KV-cache,
  OpenAI+Anthropic APIs, 45+ architectures, tensor-parallel, PyO3 bindings. Solo-maintained.
- **luminal** — funded ($5.3M seed, YC) search-based compiler; ~80% H100 peak; inference-only.

**Cross-language (the bar for "all of AI"):**
- **llama.cpp** — local inference king. ~45k GGUF checkpoints on HF; ~290 tok/s decode top
  NVIDIA, ~66–94 tok/s Apple; **~60% memory-bandwidth utilization on Metal**. k-quant/i-quant/
  imatrix quality below 3 bits. The bar for *local*.
- **vLLM** — serving king. PagedAttention cuts KV waste 60–80% → **<4%**; continuous batching;
  OpenAI-compatible API; up to 24× naive HF throughput. CUDA-locked.
- **TensorRT-LLM** — NVIDIA peak. FP4 on Blackwell, 10k–60k tok/s/GPU. Unreachable without a
  4-bit hardware path; CUDA-locked.
- **PyTorch 2 / JAX-XLA** — the training fortress. 54–65% MFU across thousands of accelerators.

## Our honest gaps (unsentimental)

| Gap | Where we are | The bar |
|---|---|---|
| **Adoption surface** | no Python bindings, no HF Hub loader, no OpenAI API | the actual adoption vector — `tokenizers`/`safetensors`/`outlines-core` all won as invisible-Rust-under-Python |
| **Decode roofline** | ~20–57% of roofline, ~8× off llama.cpp | ~60% MBU / ~290 tok/s to be credible for local |
| **Serving** | none | paged KV + continuous batching + OpenAI API (mistral.rs already ships this in Rust) |
| **Kernels** | naive default; coop opt-in at 3.9 TFLOP/s | CubeCL 140 TFLOP/s; autotuning + fusion (the "free 40%") |
| **Model coverage** | ~3 families | Candle's zoo |
| **Training at scale** | single-device | 54% MFU to 2,048 GPUs — **a fortress; a trap for us** |

## The open territory nobody holds — our moat

All three surveys converged on the same unoccupied ground, which is exactly our proven
capability:

1. **Cross-vendor determinism is now a named, hot problem.** Thinking Machines' 2025 paper
   *Defeating Nondeterminism in LLM Inference* showed Qwen3-235B at temp-0 produced **80
   unique completions in 1,000 runs**; their fix costs performance and is single-vendor. We
   already ship bit-identical Metal↔WebGPU and fp32-close NVIDIA. **Nobody else claims
   cross-vendor parity.** The field treats determinism as a cost; we make it a product.
2. **Train + infer + native + browser in one artifact.** The set `{trains} ∩ {same-code
   native+browser} ∩ {heterogeneous scheduler}` is occupied by **no one**.
3. **The browser has no CUDA moat.** vLLM/TRT/MLX/Mojo/ZML have *no browser at all*;
   WebLLM/transformers.js have no native-tensor-core or training path.

## Where dominance is winnable vs where it's a trap

| Winnable (concentrate) | Trap (avoid as a primary bet) |
|---|---|
| **Interop / format gravity** — read GGUF/safetensors/ONNX + HF Hub + PyO3 → inherit the corpus *and* users | **Training at scale** — fortress; our edges are worthless to that buyer |
| **Edge / browser** — no CUDA moat, incumbents thin, our bit-identity is differentiated | **Fine-tuning** — PyTorch+HF owned (TRL/Unsloth/Axolotl) |
| **Structured generation / agent runtime** — Rust-shaped hot loop, un-consolidated | **Hardware-portability-as-a-product** — crowded (Modular/tinygrad/IREE/XLA) |
| **Determinism as a feature** — a named problem we already solve | **Chasing vLLM tokens/sec on NVIDIA** — treadmill on a metric where our edges don't count |

## The single most underappreciated lever: "invisible Rust"

`tokenizers`, `safetensors`, `outlines-core` all won by being the fast correct thing *under*
Python — nobody notices they're Rust. Even vLLM/SGLang are C++/CUDA cores behind Python
control planes. **The market buys drop-in speed and reach, not a language.** Ship `pip
install ferric` + HF Hub loader + OpenAI-compatible endpoint so people adopt us without
writing a line of Rust — then let determinism/browser/portability be why they stay.

## The one-line positioning the whole field leaves open

> The fastest **portable** AI runtime that runs the same kernel source from native tensor
> cores to the browser tab, is **bit-consistent across GPU vendors**, trains as well as
> infers, and runs the standard GGUF ecosystem — pure Rust, adoptable from Python.

## Prioritized path (build this, not that)

- **P0 — Adoption ticket ("invisible Rust").** OpenAI-compatible server (drop-in for the
  entire ecosystem) → HF Hub GGUF loader → PyO3 `pip install ferric`. Highest leverage in the
  whole analysis; makes us adoptable without anyone writing Rust.
- **P0 — Weaponize determinism.** Turn our cross-vendor bit-parity into a headline benchmark +
  the lead pitch (regulated / eval / finance care deeply). The Thinking Machines paper named
  the problem for us.
- **P1 — Close the decode roofline gap** to ~60% MBU. The *credibility* gap for local inference.
- **P1 — Portable serving essentials** (paged KV + continuous batching), framed as "the only
  *portable* serving engine," not a vLLM race.
- **P2 — Structured / guided decoding + agent-runtime layer** (see the agentic-ecosystem ingest
  list) — a clean, un-consolidated, Rust-shaped wedge that makes the core useful for real apps.

**Reading of "done":** P0 → adoptable & differentiated; P1 → credible for local + serving;
P2 → the runtime under the agentic ecosystem. At that point Ferric is the only stack that is
SOTA-portable *and* browser-competitive *and* trainable *and* deterministic across vendors —
the intersection no incumbent holds.
