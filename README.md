# Ferric

[![CI](https://github.com/dcharlot-physicalai-bmi/ferric/actions/workflows/ci.yml/badge.svg)](https://github.com/dcharlot-physicalai-bmi/ferric/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)

**One pure-Rust AI compute ecosystem across every fabric — cloud, local, browser.** Maximally
optimized, heterogeneous, no Python/C++ in the hot path. A model defined once runs the *same code* on
a datacenter GPU, a laptop, an edge robot, and a browser tab.

> **Status: runs a current-generation flagship, end to end.** The cross-fabric thesis is proven — one
> Rust kernel runs *bit-identical* on a native GPU and in the browser — and Ferric assembles that into a
> full language-model runtime: embedding → RoPE grouped-query attention → RMSNorm → SwiGLU, a KV cache
> and autoregressive generation, the standard quantized-GGUF ecosystem, and the **Qwen3.5/3.6 GDN-hybrid**
> (gated delta net + periodic full attention). The OpenAI-compatible server runs **Qwen3.6-27B** (a 2026
> flagship) with **in-runtime constrained decoding** (schema-conformant JSON), plus a pure-Rust ONNX
> importer and safetensors loader. Every step is validated against a reference. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why
Browser AI today is JavaScript / C++-WASM (ONNX Runtime Web, WebLLM, Transformers.js). Rust's AI story
is fragmented and weak in the browser, and the one Rust WebGPU ONNX runtime is stalled. **No one has a
single coherent Rust ecosystem that is genuinely the same across cloud + local + browser.** Ferric is
that: build on the SOTA pure-Rust compute layer (`wgpu`), own the kernels, model runtime, weight
loading, and browser packaging — and, the eventual differentiator, a heterogeneous scheduler that
partitions a model across whatever compute is present.

## What's proven
Everything below is validated against an independent reference (CPU, `onnxruntime`, or `numpy`) and
builds fully offline from vendored source.

| Milestone | Result |
|---|---|
| One matmul kernel, native **Metal** vs browser **WebGPU** | bit-identical, `max‖gpu−cpu‖ = 7.15e-7` |
| Transformer kernels (matmul, flash attention, RoPE, RMSNorm, LayerNorm, SwiGLU, softmax) | validated vs CPU |
| Pure-Rust **ONNX** importer (~22 ops) — MLP, self-attention, real **SmolVLA** components | matches `onnxruntime` |
| Full modern **decoder layer** (RoPE GQA attention + SwiGLU) on-GPU | matches CPU |
| Multi-layer **LM forward pass** (embed → N layers → head → logits) | matches CPU `1.9e-6` |
| **KV-cache** autoregressive generation | *exact* vs full recompute |
| Pure-Rust **safetensors** loader (F32/F16/BF16) | exact round-trip |
| **Llama/SmolLM** checkpoint bridge (HF layout, GQA, tied embeddings) | matches `numpy` `1.2e-6` |
| **General tensor runtime** — arbitrary rank, strided views, broadcasting, any-axis reductions, batched matmul | matches CPU ref |
| **Autograd** — reverse-mode; an MLP *and* a transformer trained on the GPU | grad-check `1.5e-5`, 100% acc |
| **Runs a REAL model** — SmolLM2-135M (30-layer Llama, GQA, tied, bf16) end-to-end | logits match numpy `1.4e-6` |
| **Generates REAL text** — HF tokenizer + greedy decode | *"The capital of France is the capital of the country."* |
| **Model families** — BitNet/PrismML (ternary), Liquid LFM2 (conv1d), EBM (Langevin), JEPA/V-JEPA2 | all validated |
| **Runs a CURRENT flagship** — Qwen3.6-27B (2026 GDN-hybrid: 48 gated-delta-net + 16 full-attn, Q4_K_M) on the OpenAI server | coherent (*"…Tokyo … Senso-ji Temple"*) |
| **Structured output** — in-runtime constrained JSON (schema: required/optional, int `min`/`max`, `maxLength`, typed/bounded arrays, enum), native + browser/WebGPU | schema-conformant on Qwen3.6-27B |

## Crates
- [`ferric-core`](crates/ferric-core) — L0/L1: the `wgpu` `Context` + the cross-fabric kernel set + CPU references.
- [`ferric-tensor`](crates/ferric-tensor) — L2: the general N-D tensor runtime (strided views, broadcasting, reductions, batched matmul) + reverse-mode autograd for training.
- [`ferric-onnx`](crates/ferric-onnx) — L3: a pure-Rust ONNX importer that runs graphs on Ferric tensor ops.
- [`ferric-load`](crates/ferric-load) — safetensors reader (fp16/bf16 dequant) — how real checkpoints enter.
- [`ferric-llama`](crates/ferric-llama) — maps a Llama/SmolLM safetensors checkpoint onto the kernels and runs it.
- [`ferric-web`](crates/ferric-web) — the same core compiled to WASM, running on browser WebGPU.

## Quickstart
```bash
# a full multi-layer transformer LM forward pass, on your GPU, vs a CPU reference
cargo run --release -p ferric-core --example tiny_llm

# KV-cache autoregressive generation
cargo run --release -p ferric-core --example generate

# load a Llama/SmolLM-layout safetensors checkpoint and run it
cargo run --release -p ferric-llama --example run_llama

# real ONNX (incl. SmolVLA components) matching onnxruntime
cargo run --release -p ferric-onnx --example run_projectors

# the SAME kernel in the browser (WASM + WebGPU)
cd crates/ferric-web && wasm-pack build --release --target web --out-dir pkg
#   then serve the crate dir and open index.html
```

## Self-reliance
Ferric depends on no external project at runtime: strategic crates (`wgpu`, `naga`) are **forked**
in-tree under [`forks/`](forks) and wired via `[patch.crates-io]`, and the full dependency tree can be
vendored for hermetic, offline builds.

**Offline builds:** run `cargo vendor` once to populate `vendor/` and print the `[source]` stanza for
`.cargo/config.toml`; thereafter `cargo build --offline` uses only in-repo source. (`vendor/` is not
committed — it's reconstructible.)

## License
Apache-2.0. From the [Institute for Physical AI](https://physicalai-bmi.org).
