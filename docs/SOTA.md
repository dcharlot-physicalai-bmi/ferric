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

## Model-family coverage (Liquid AI, PrismML, BitNet, EBM, JEPA)
Researched the exact ops each family needs (July 2026). Most reduce to primitives Ferric already has.

| Family | What it is | New primitives needed | Status |
|---|---|---|---|
| **BitNet / ternary** (`microsoft/bitnet-b1.58-2B-4T`) | Transformer w/ ternary `{−1,0,+1}` BitLinear + int8 activations + ReLU² FFN | ternary matmul, ReLU², GGUF ternary blocks | ternary matmul ✅ (1.9e-6, 1/16 mem) · ReLU² ✅ · GGUF Q2_0/TQ2_0 ⬜ |
| **PrismML** (Caltech spinout: Bonsai / Ternary Bonsai 1.7/4/8B) | **Architecturally identical to BitNet** — standard transformer, every linear ternary 1.58-bit (group-128, fp16 scale), ships GGUF `Q2_0` | *same as BitNet* — ternary matmul + GGUF `Q2_0` | ternary matmul ✅ · GGUF Q2_0 ⬜ |
| **Liquid AI LFM2** (`LiquidAI/LFM2-1.2B`) | 16 blocks = 10 gated short-conv + 6 GQA; SwiGLU MLP; RMSNorm; RoPE | **causal depthwise conv1d (L=3)** + gating (⊙) | conv1d ✅ (1.5e-7) · gated block ✅ · GQA/RoPE/RMSNorm/SwiGLU ✅ — **LFM2 block fully covered** |
| **EBM / JEM** | scalar energy `E(x)` + Langevin sampling `x -= ε∇ₓE + √ε·𝒩` | grad-w.r.t-input (✅), logsumexp, host loop | **✅ RUNS** — `examples/ebm.rs` Langevin-descends the energy (−0.12→−1.46) via autograd-∇ₓE; logsumexp composed from primitives |
| **JEPA** (I-JEPA, V-JEPA 2) | ViT encoder + predictor, latent-space prediction | patch embed (unfold+matmul), **non-causal attention**, GELU (✅), 3D RoPE (V-JEPA2) | `bidirectional_attention` ✅ · GELU ✅ · patch-embed = reshape+matmul ✅ · **encoder composable** · 3D-RoPE / mask-token ⬜ (V-JEPA2 predictor only) |

**Key insight:** PrismML ≡ BitNet (both ternary transformers), so ternary matmul + GGUF-ternary covers
*two* families. LFM2 needed only conv1d (done). EBM needs almost nothing new (we already do
grad-w.r.t-input). JEPA is a standard ViT + a few small primitives.

**Remaining model-family gaps (all small/tractable):** GGUF ternary blocks (`Q2_0`/`TQ2_0`), on-device
RNG + `logsumexp` (EBM sampling), non-causal attention + additive mask + patch-embed + 3D-RoPE (JEPA).

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
- [ ] **1. Tiled register-blocked GEMM + vec4 loads + autotuning.** The naive→>1 TFLOP lever.
      **Measured finding (M5 Max, wgpu→Metal):** the naive one-thread-per-output kernel is a *strong*
      baseline here — Metal auto-vectorizes/caches it to **~587 GFLOP/s at 1024³** (very different from
      the research's "naive = 1.6 GFLOP/s" on other GPUs). Both a 4×4 and an 8×8 register-blocked
      shared-memory kernel (`matmul_tiled`) were implemented and validated bit-exact but **do not beat
      naive on this hardware** (~0.4× at 1024³) — a straightforward tiled kernel isn't enough; the
      published >1 TFLOP results require **vec4 loads + a bounds-check-free interior fast-path + loop
      unrolling + per-device autotuning**, and the win is GPU-specific. **This is precisely why #6
      (autotuning) is not optional:** there is no single kernel that wins on every GPU, so the correct
      SOTA move is to keep both kernels and *select per device+shape by measurement* (on M5 Max that
      selects naive; on GPUs where tiled wins, it selects tiled). Do #6 to make GEMM portably fast.
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
