# Ferric vs. the world — SOTA scorecard & roadmap

A grounded competitive analysis (2026 landscape) of Ferric against the leading AI runtimes across
every ecosystem, and the concrete list of what remains to be unambiguously state-of-the-art. This is
how we know when we're done: each gap below is a checkbox.

## The field we're measured against
- **C++:** llama.cpp/ggml, ONNX Runtime, TVM/MLC-LLM, ExecuTorch, vLLM, TensorRT-LLM.
- **Python:** tinygrad, PyTorch 2 (torch.compile / Inductor / Triton), JAX/XLA.
- **Swift/Zig/Mojo:** MLX + mlx-swift, ZML (Zig), Mojo/MAX.
- **Rust:** candle, burn + CubeCL, luminal, ratchet, wonnx, tract.
- **Browser/WebGPU:** WebLLM/MLC-web, transformers.js, ONNX Runtime Web, tinygrad-WEBGPU, and the
  research SOTA reference *LlamaWeb* ("Llamas on the Web", arXiv 2605.20706).

## Where Ferric is already ahead of everyone
These are the axes no single competitor holds together — Ferric's moat is **capability surface**, not
FLOPS:

1. **One codebase, bit-identical native ↔ browser (validated).** No mainstream project ships *proven
   bit-identical* native-vs-WebGPU numerics as a first-class property. Burn is the only true same-code
   cross-platform peer and doesn't guarantee bit-exact parity; ratchet/wonnx are cross-platform but
   inference-only; WebLLM/transformers.js/tinygrad are browser-only with no native same-code path.
2. **Training that actually trains transformers, *inside* the cross-fabric runtime.** The set
   {trains transformers} ∩ {same code native+browser} ∩ {heterogeneous scheduler} is currently
   occupied by **no one**. Every browser engine (WebLLM, transformers.js, ratchet, wonnx) is
   inference-only.
3. **A heterogeneous scheduler spanning CPU + GPU + cloud(TCP) + browser(WebGPU) in one graph.**
   Burn has `burn-remote`; nobody places work across local CPU/GPU, a TCP cloud worker, *and* a
   browser tab behind one `Device` abstraction.
4. **Portable ingest (safetensors + ONNX) + a real Llama forward + int8/int4 weight-only quant matmul
   + eager autograd**, all pure-Rust and self-reliant (vendored + forked deps, offline builds).

## Where Ferric is behind (be honest)
**Raw kernel quality.** burn/CubeCL and LlamaWeb are ahead on GEMM and attention. Our matmul is a
well-threaded but *naive* one-thread-per-output kernel (~425 GFLOP/s at 1024³ on an M5 Max — the
"multi-thread WebGPU GEMM" tier), whereas register+workgroup tiling reaches **>1 TFLOP/s** and
subgroup-matrix (tensor cores) more. We have no general flash-attention path, no GGUF/k-quants, no
tokenizer, no kernel fusion, no autotuning.

## The 2026 WebGPU platform reality (shapes what's even possible in-browser)
- **Subgroups + `shader-f16`: STABLE** in Chrome 134+ (2.3–2.9× on matrix-vector shaders). Usable
  in-browser today — a real, unclaimed perf lever.
- **Subgroup-matrix / cooperative-matrix (the WGSL path to tensor cores): NOT stable in browsers.**
  Native-only via Dawn/wgpu. So the best *browser* GEMM = register/workgroup tiling + f16 + subgroup
  reductions; tensor cores are a native-only accelerant to feature-gate. Conveniently, our in-browser
  competitiveness does **not** depend on it (nobody else has it in-browser either).

## Roadmap — the checklist to unambiguous SOTA
Priority order (both the C++ and Rust/browser surveys converge on this):

- [x] **Pipeline caching** — compile each WGSL kernel once, not every dispatch. (~4× on small ops; the
      single biggest per-op overhead, and table stakes every runtime has.) **Shipped.**
- [ ] **1. Tiled + 8×8 register-blocked GEMM + vec4 loads.** The naive→>1 TFLOP lever. A 4×4 tiled
      kernel exists (`matmul_tiled`) as the foundation but doesn't yet beat naive; needs 8×8 + vec4 +
      unroll. Pure core WGSL, no extensions, identical native/browser.
- [ ] **2. Subgroup-accelerated GEMV (decode) + `shader-f16`.** Memory-bound decode is where LLMs live;
      subgroups are browser-stable now. Feature-detect + fall back.
- [ ] **3. General flash-attention WGSL** (online-softmax, tiled, GQA) + paged / quantized KV cache.
      Proven viable on WebGPU (ORT FA2, LlamaWeb FlashDecoding).
- [ ] **4. GGUF loader + k-quant / i-quant dequant kernels** (Q4_K_M, Q6_K, Q8_0…). Unlocks the entire
      llama.cpp/HF quantized-model corpus. candle already reads these; we can't yet.
- [ ] **5. Tokenizer** (byte-level BPE / SentencePiece). Table stakes for prompt→text; pure Rust, wasm-clean.
- [ ] **6. Per-shape / per-device kernel autotuning** + a persisted cache. LlamaWeb got +41% average;
      matters *more* for us because WebGPU spans wildly heterogeneous GPUs.
- [ ] **7. Kernel fusion** (matmul epilogue + RMSNorm/RoPE/SwiGLU; attention fusion) via a pass over the
      tensor graph — the memory-bandwidth win LlamaWeb blames its weak prefill on.
- [ ] **8. Cooperative-matrix (tensor-core) tile-matmul — native-gated.** The last mile to cuBLAS/Metal
      parity on native; feature-detected, browser falls back to #1/#2.

**Reading of "done":** land 1–3 → kernel-credible; 4–5 → ergonomically complete end-to-end; 6–7 →
capture the "free" 40%+ everyone else banks; 8 → native parity. At that point Ferric is the only stack
that is simultaneously SOTA-Rust *and* browser-competitive *and* trainable *and* deterministic across
cloud+local+browser — a combination no single competitor holds.
