# How people beat/match CUDA portably (2026) — and what Ferric should do

Synthesis of four research sweeps (Mojo/MAX · portable kernel compilers · browser/edge WebGPU ·
vLLM/SGLang/DeepSeek). Written for Ferric: pure-Rust wgpu/WGSL, targeting Metal + Vulkan + DX12 +
browser-WebGPU, cross-fabric bit-consistent, runs the standard quantized GGUF ecosystem.

## The one framing

The order-of-magnitude wins are **algorithmic and memory-layout**, not instruction-level. Vendor
kernels (TMA, WGMMA, FP8 tensor cores, hand asm) buy the *last ~1.5–3×* on one specific chip. The
scheduling/memory layer — paging, batching, prefix reuse, KV compression, speculative decode, whole-
graph fusion — is where serving throughput actually comes from, and it is **almost entirely hardware-
agnostic and WGSL-implementable.** A portable runtime captures the majority of the practical win
because most of it lives *above* the kernel.

Honest ceiling on the kernel itself: portable stacks reach **~60–100% of the vendor library on plain
GEMM** and can **beat vendor asm on fused kernels**. Modular's own Blackwell matmul tops out at ~85%
of cuBLAS; an independent HPC paper puts Mojo compute-bound kernels at 38–82% of CUDA/HIP. So: budget
an ~80–85% GEMM ceiling, and **win on fusion, layout ergonomics, and browser/edge reach** instead.

## The three references to study (in order)

1. **LlamaWeb / `ggml-webgpu`** (arXiv 2605.20706; now in llama.cpp, PR #17031). The single best
   blueprint — a WebGPU LLM runtime that is *the same shape as Ferric*. It beats WebLLM +54% decode /
   transformers.js +69%, 29–33% lower memory. Its playbook: static arena allocation; single command-
   buffer batching; rotating pre-allocated buffer slots (emulating missing push-constants); pipeline
   cache keyed on specialization; **templated dequant-in-kernel** (one GEMM/GEMV body, dequant routine
   injected per format — 23 formats incl `q1_0` 1-bit Bonsai); three kernel families **`reg_tile`
   (portable fallback), `sg_mat` (subgroup-matrix/tensor-core), FlashDecoding**; and
   **performance-portable tile sizes swept over thousands of configs on 4 GPUs / 4 vendors** (+41%
   avg kernel speedup vs Apple-only tuning). Weights stream through only four 1 MB buffers.

2. **CubeCL (Burn)** — our exact pure-Rust precedent. One kernel codebase, matches/beats cuBLAS on an
   RTX 4080 (CUDA *and* Vulkan), same source on Metal + WebGPU. Design: a four-level
   Tile→Stage→Global→Batch hierarchy, **the Tile level a trait** that binds to tensor cores /
   simdgroup_matrix / software fallback; double-buffering; register-pinned accumulators; vectorized
   shared-mem writes that self-distribute across banks. Their honest weak spot: variant selection is a
   heuristic, not per-shape autotuned.

3. **antirez/ds4** ("DeepSeek 4 Flash") — one codebase across Metal/CUDA/ROCm; **unify the graph + KV
   memory format, specialize only the per-hardware kernels** (sessions port across backends, survive
   disk checkpointing). Plus: selective/asymmetric quant (only routed MoE experts 2-bit), and KV cache
   as a "first-class disk citizen" (stream compressed KV from SSD → memory becomes a speed spectrum,
   not a cliff).

## The convergent recipe for "portable AND fast"

Every system that pulls it off uses some subset of:

1. **Layout as a first-class compile-time object** (CUTLASS CuTe, Mojo LayoutTensor, TK/HipKittens
   tiles). A `Layout = (shape, stride)` with a composition algebra (compose/tile/distribute/vectorize);
   generate index math from it instead of hand-writing offsets. **Highest-leverage idea, fully
   portable — pure index math.** Swizzle modeled separately as `base ∘ Swizzle<B,M,S>` (XOR bit-
   permute) to kill shared-memory bank conflicts, keyed to tile shape.
2. **Tile abstraction with interface/implementation split** — architect kernels as tiles; bind the
   *inner* tile op to a matrix-unit primitive when present, else a subgroup/vectorized fallback. The
   HipKittens thesis: tile DSLs that separate interface from implementation are the portable path to
   peak; compilers (Triton/Mojo) sacrifice some performance (Triton ~62–101% of cuBLAS; Mojo ~85%).
3. **Cooperative-matrix as the portable tensor-core spelling** — one `mma` that lowers to NVIDIA
   tensor cores, AMD MFMA, Apple simdgroup_matrix, Intel XMX. The matrix carries an A/B/Accumulator
   role. `VK_NV_cooperative_matrix2` shows the ideal shape: workgroup-scope matrices (impl auto-stages
   through shared mem), tensor addressing with boundary clamp/pad, accumulator↔A/B conversions for
   fusion, row/col reductions on the accumulator (softmax without a shared-mem round trip), and
   **in-load decode callbacks to dequantize int4/sub-byte weights** — exactly our packed-quant need.
4. **Measured autotuning** (tinygrad BEAM search; LlamaWeb config sweep) — enumerate {tile sizes,
   workgroup size, subgroup use, buffering depth, vector width}, *run each on device*, cache the winner
   per (device, shape-class). Because Ferric spans Apple-ALU vs NVIDIA-tensor-cores vs mobile, an
   analytic cost model will not port — measured search is mandatory.
5. **Fuse aggressively** — the consistent 2025–2026 lesson: portable kernels *lose* to vendor GEMM on
   plain matmul but *win* on fused ops (attention, MoE dispatch, dequant-matmul). MAX's real vLLM win
   was whole-graph fusion (RoPE+attention+proj), not kernels beating cuBLAS.
6. **Async staging / double buffering** — no TMA in WGSL, but the pattern (prefetch tile N+1 to shared
   mem while computing N) is portable and is where most software-tier speed comes from.

## WGSL/wgpu surface — what's actually available (the gating facts)

- **Subgroups: SHIPPED** natively (wgpu `SUBGROUP`/`SUBGROUP_BARRIER`) AND in-browser (Chrome 134,
  Feb 2025). `subgroupAdd/Ballot/Broadcast/Shuffle`. Measured 2.3–2.9× on matrix-vector shaders.
  **Usable today. Biggest safe win** — subgroup reductions replace our 6-barrier split-K trees.
- **shader-f16: SHIPPED** (native + Chrome 120). **DP4a** (`dot4I8Packed`) shipped Chrome 123 — INT8
  dot products for integer-quant GEMM.
- **Cooperative matrix (`coop_mat`): EXPERIMENTAL, merged** — naga/wgpu **PR #8251 (2025-12-22)**,
  `coop_mat<elem,dims,role>` + `coopLoad`/`coopStore`/`coopMatMulAdd`, **Vulkan + Metal only**, scope =
  Vulkan∩Metal intersection, **MulAdd-only**, no DX12, WIP. Our fork already carries
  `EXPERIMENTAL_COOPERATIVE_MATRIX`. **Native tensor-core GEMM is reachable today**; browser waits on
  Dawn shipping subgroup-matrix (standardizing now, gpuweb WGSL WG Jan/Mar 2026).
- **M5 (this dev machine) is the first Apple GPU with real in-GPU matrix hardware** — our "naive beats
  tiled" result was true on ALU-only silicon; `coop_mat`→simdgroup_matrix could genuinely change it
  here. Worth re-measuring on M5.

## Where Ferric already is (independently built = validated)

Per-layer command batching ✓ · pipeline cache keyed on content hash ✓ · dequant-in-kernel per format
(Q2_0/Q4_0/Q4_K/Q5_K/Q6_K/Q8_0) ✓ · fused single-query attention = FlashDecoding ✓ · fused SwiGLU ✓ ·
KV cache ✓ · vec4 activation loads ✓ · rows-aware kernel selection ✓ · cross-fabric bit-consistency ✓
(the thing NO competitor has) · runs `q1_0`/Q2_0 ternary Bonsai ✓ · validated token-for-token vs
llama.cpp ✓.

**Gaps vs the SOTA blueprints:** `sg_mat`/`coop_mat` tensor-core path (worth ~+88% prefill) · **one**
templated GEMM body instead of six near-duplicate kernels · static arena + weight streaming (we
re-upload; blocks >buffer-limit models, esp. Safari) · measured autotuner over tile configs · a real
GEMM (we use naive one-thread-per-output — fine at decode, weak at prefill) · the portable *algorithmic*
wins below.

## The prioritized plan for Ferric

**Tier 1 — adopt now, pure WGSL/host, no new hardware, all portable (biggest value/effort):**
1. **Subgroup reductions** in the split-K quant matmuls + FlashDecoding — replaces barrier trees;
   ~2–3× on memory-bound decode GEMV. Feature-detect `SUBGROUP`; keep the barrier path as fallback.
2. **Templated dequant-in-kernel**: collapse the six per-format matmuls into one GEMM/GEMV body with
   the dequant routine injected (LlamaWeb model). Less code, one place to optimize, trivial to add
   formats. Builds directly on the `QMatrix` enum.
3. **Static arena + buffer pooling + weight streaming** (LlamaWeb): stop re-uploading; stream weights
   through a few fixed buffers; shard any tensor above `maxStorageBufferBindingSize` (Safari 256 MB–1 GB
   is the real gate). Unlocks >buffer-limit models in-browser and cuts the ~1.8 s load.
4. **Whole-graph / epilogue fusion** at the Ferric IR level (already have `fuse.rs`/`Lazy`): fuse
   RMSNorm→matmul, residual adds, RoPE into neighbors. MAX's actual serving win.
5. **Quantized KV cache (INT8 → INT4)**: dequant-in-shader; decode is bandwidth-bound so it's a direct
   speedup and unlocks long context on memory-limited devices.
6. **Speculative decoding (self-speculative / quantized draft, EAGLE-3-style)**: biggest *single-stream*
   latency win (up to ~2.5–4×), and batch=1 is exactly the edge/browser regime. Pure control flow.

**Tier 2 — prototype behind a capability flag (native win today, browser later):**
7. **`coop_mat` GEMM path** wired to naga PR #8251 (Vulkan + Metal), with the `reg_tile` register-blocked
   WGSL kernel as the universal fallback, selected by capability detection from ONE source. This is the
   prefill moat (~+88%). **Re-measure the tiled-vs-naive question on the M5's new matrix hardware first.**
8. **Layout algebra + swizzle as a Rust host type** that generates WGSL index math + bank-conflict-free
   shared-mem layouts. The structural investment that makes 7 (and a real tiled GEMM) tractable.
9. **Measured autotuner**: sweep {tile, workgroup, subgroup, buffering, vector width} on device, cache
   per (device, shape-class). Mandatory given how differently Apple/NVIDIA/mobile behave.

**Tier 3 — algorithmic serving layer (throughput, mostly host-side, all portable):**
10. Paged KV cache + block table · continuous batching (mix prefill/decode) · chunked prefill ·
    RadixAttention-style cross-request prefix reuse. These are the vLLM/SGLang order-of-magnitude
    *serving* wins and need no new kernels.

**Do NOT chase:** FA3 async pipelines, TMA/WGMMA, FP8/FP4 tensor-core peak, DualPipe/DeepEP multi-GPU
comms — last-1.5–3× on one chip, no WebGPU equivalent.

## The honest positioning this research confirms

Ferric will not beat cuBLAS/CUDA on raw NVIDIA GEMM throughput — nobody portable does (best is ~85%).
The defensible, and genuinely unoccupied, claim: **the fastest *portable* AI runtime that shares one
kernel source across native tensor cores (naga `coop_mat`, today) and the browser, is bit-consistent
across fabrics, and runs the standard quantized ecosystem end-to-end — as a pure-Rust library.**
LlamaWeb proved the browser numbers but is a C++ fork of llama.cpp with no native-tensor-core or
cross-fabric-determinism story; Mojo/MAX has the kernels but *no browser at all*; CubeCL has the
portable Rust kernels but not the LLM runtime. Ferric is the intersection.
