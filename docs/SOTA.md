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
| **BitNet / ternary** (`microsoft/bitnet-b1.58-2B-4T`) | Transformer w/ ternary `{−1,0,+1}` BitLinear + int8 activations + ReLU² FFN | ternary matmul, ReLU², GGUF ternary blocks | ✅ ternary matmul (1.9e-6, 1/16 mem) · ReLU² · **GGUF TQ2_0 loads** |
| **PrismML** (Caltech spinout: Ternary Bonsai **27B**, + 1.7/4/8B) | **NOT a plain ternary transformer** — a Qwen3.5 hybrid: 64 blocks = 48 **gated delta net** linear-attention + 16 full **gated** GQA (partial RoPE, QK-norm), every projection ternary in their own `Q2_0` (2.125 bpw) | gated delta rule, l2norm, softplus, `cat`/`narrow`, Q2_0-native matmul, partial RoPE | ✅ **RUNS — 27B validated vs their llama.cpp fork, max\|Δ\| = 7.8e-4** (`examples/run_bonsai.rs`) |
| **Liquid AI LFM2** (`LiquidAI/LFM2-1.2B`) | 16 blocks = 10 gated short-conv + 6 GQA; SwiGLU MLP; RMSNorm; RoPE | **causal depthwise conv1d (L=3)** + gating (⊙) | conv1d ✅ (1.5e-7) · gated block ✅ · GQA/RoPE/RMSNorm/SwiGLU ✅ — **LFM2 block fully covered** |
| **EBM / JEM** | scalar energy `E(x)` + Langevin sampling `x -= ε∇ₓE + √ε·𝒩` | grad-w.r.t-input (✅), logsumexp, host loop | **✅ RUNS** — `examples/ebm.rs` Langevin-descends the energy (−0.12→−1.46) via autograd-∇ₓE; logsumexp composed from primitives |
| **JEPA** (I-JEPA, V-JEPA 2) | ViT encoder + predictor, latent-space prediction | patch embed (unfold+matmul), non-causal attention, GELU, 3D RoPE, mask-token | ✅ **FULLY RUNS** — `examples/jepa.rs`: patch-embed→bidirectional encoder w/ **3D RoPE** (5.96e-8)→GELU MLP→**mask-token blend**→predictor, end to end |

**Correction (2026-07):** the earlier note here claimed *PrismML ≡ BitNet*. Running the real
Bonsai-27B disproved that — it is a **Qwen3.5 hybrid**, three-quarters linear attention, and needs a
gated-delta-rule recurrence that a plain ternary transformer never exercises. Assuming the two were
the same would have covered one family, not two.

**All 5 families run, and three are now validated against a real downloaded checkpoint:**

| Model | Params | Reference | Agreement |
|---|---|---|---|
| SmolLM2-135M | 135M | numpy | 1.4e-6 |
| Liquid LFM2-350M | 350M | transformers | 4.7e-6 |
| **PrismML Ternary Bonsai-27B** | **27B** | **their llama.cpp fork** | **7.8e-4** |

Remaining: GGUF `TQ1_0`/`I2_S` ternary variants; true multi-chain SGLD via on-device RNG; and
Bonsai's 4-bit HQQ **vision tower** (the text path is done).

**Speed, honestly.** Correctness landed before speed. On the same M5 Max, Bonsai-27B:

| | Ferric (first run) | Ferric (now) | llama.cpp (Metal) |
|---|---|---|---|
| load | 10 s | 6 s | 0.25 s (mmap) |
| prefill (5 tok) | 2.5 s | **1.35 s** | 0.08 s |
| decode | re-prefilled (17.3 s / 12 tok) | **cached — 3.8 s / 12 tok** | 0.26 s / 12 tok |

Decode is ~290 ms/token against llama.cpp's 22 ms — still ~13× off (was ~32×). The remaining causes
are known: every op is its own dispatch (the fusion compiler isn't applied to this path yet), the KV
cache re-concatenates rather than writing into a reserved buffer, and load re-uploads instead of
mmap-ing. The honest claim stays **"Ferric runs Bonsai-27B correctly,"** not that it matches
llama.cpp's speed.

### What the perf work actually taught (measured, not assumed)
1. **The 34-byte Q2_0 block is hostile to a GPU.** `f16 d` + 32 code bytes isn't a multiple of 4, so
   a shader can't address codes as `u32` and falls back to a byte-extract that re-reads the same word
   once per weight. Repacking on upload into an aligned codes array + separate scales array (same
   bytes, our choice of layout) lets the inner loop read 8 words per block instead of 128.
2. **Benchmark methodology dominated the first conclusions.** Awaiting each rep measured
   submit+fence+readback (~1 ms), not the kernel — it made every shape look like a flat ~2.3 ms and
   hid all real differences. Dispatches are async; queue the reps and sync once.
3. **The kernel selector is output count, not K depth.** Flat (one thread per output) starves at
   decode-sized output counts; split-K (workgroup per output, 64× the threads) wins there by up to
   2.6×, and *loses* by 1.6× at prefill where flat already saturates. Crossover ≈16K outputs.
   Selecting per shape beats either kernel fixed: **1.35 s vs 2.21 s (flat) / 2.01 s (split-K)**.
4. **A 1D dispatch grid caps at 65535 workgroups = 4.19M threads** — which a real LM head passes at
   17 tokens (17 × 248320 = 4.22M). This was latent in `groups(n)` from the start and only surfaced
   once generation ran long enough; big-vocab models hit it early.

### The cache: state carry is the whole game
Both halves of the hybrid resume. Attention keeps K/V; the gated delta net carries its recurrent
state **and** the short conv's receptive field — a lone token can't be convolved, so the carried tail
is prepended (which doubles as the causal zero-padding a fresh sequence needs). The correctness bar
is equivalence, and it's asserted two ways: `gdn_state` shows the recurrence carries **bit-exactly**
(0.0e0) across every split, and `run_bonsai --verify-cache` requires cached decode to emit token ids
identical to re-prefilling — currently 12/12 identical at 4.5× the speed.


## Device coverage — CPU / GPU / NPU (honest)
`sched::detect_devices()` + `examples/devices.rs` enumerate and use everything present:
- **GPU:** every wgpu adapter across all backends (Metal/Vulkan/DX12/GL) — `Context::enumerate()` /
  `for_adapter(i)`, so a multi-GPU box gets one device per GPU. Verified on M5 Max.
- **CPU:** always a device, now **multi-threaded across all cores** (`cpu_bmm` splits the batch over
  `available_parallelism()` — 18 cores on M5 Max; the measured split hands the CPU real work).
- **NPU — actually runs.** WebGPU/wgpu cannot dispatch to an NPU, so the reachable path is a real
  execution-provider: **WebNN** (`navigator.ml`), and on macOS Chrome's WebNN is backed by **CoreML →
  the Apple Neural Engine**. `examples/npu.rs` (in `ferric-web`) launches Chrome with WebNN enabled,
  gets a `deviceType:'npu'` context, and **dispatches a matmul that executes on the ANE** through the
  browser-worker bridge — result matches CPU to **fp16 precision** (the ANE is an fp16 engine; 7.6e-2).
  Ferric also ships `probe_npu()` + the `NpuBackend` trait + `Device::Npu` slot; it never fakes NPU
  work on the GPU. (Chrome-flag gotcha: **two `--enable-features` flags — only the last wins**; merge
  `WebMachineLearningNeuralNetwork` with the WebGPU features into one flag.) Native CoreML/DirectML EPs
  are the follow-up; WebNN is the portable one and it works today.

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
