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

1. **One codebase across three fabrics and two GPU vendors — proven on a full 1.7B LLM.** The same
   pure-Rust `Qwen3` forward runs PrismML Ternary Bonsai-1.7B and produces the same last-position
   logits on **Apple Metal** (`run_qwen3 --dump`), **browser WebGPU** (`bonsai_logits`), and **NVIDIA
   Vulkan** (`run_qwen3 --dump` on an RTX 4050):

   | | Apple Metal | browser WebGPU | NVIDIA Vulkan |
   |---|---|---|---|
   | argmax | 12095 " Paris" | 12095 | 12095 |
   | top logit | +17.94083 | +17.94083 | +17.94083 |
   | Σ all 151,669 logits | −395780.136 | −395780.136 | −395780.119 |

   Metal and WebGPU (both on Apple) are **bit-identical**; NVIDIA/Vulkan agrees to **fp32 rounding**
   (~1e-5 per logit — the sum differs by 0.017 across 151,669 terms, i.e. floating-point accumulation
   order on a different vendor's shader compiler, not a bug), with identical argmax and top-8 ranking.
   No mainstream project ships *proven* same-code parity across vendors like this. Burn is the only
   true same-code cross-platform peer and doesn't guarantee bit-exact parity; ratchet/wonnx are
   cross-platform but inference-only; WebLLM/transformers.js/tinygrad are browser-only with no native
   same-code path.
2. **Training that actually trains transformers, *inside* the cross-fabric runtime.** The set
   {trains transformers} ∩ {same code native+browser} ∩ {heterogeneous scheduler} is currently
   occupied by **no one**. Every browser engine (WebLLM, transformers.js, ratchet, wonnx) is
   inference-only.
3. **A heterogeneous scheduler spanning CPU + GPU + cloud(TCP) + browser(WebGPU) in one graph.**
   Burn has `burn-remote`; nobody places work across local CPU/GPU, a TCP cloud worker, *and* a
   browser tab behind one `Device` abstraction.
4. **Portable ingest (safetensors + ONNX) + a real Llama forward + eager autograd**, all pure-Rust
   and self-reliant (vendored + forked deps, offline builds).
5. **Runs standard quantized GGUFs end-to-end, weights never expanded to f32.** The full common
   spectrum dequants *inside* the matmul kernel — PrismML **Q2_0** ternary, llama.cpp **Q4_0**,
   **Q4_K**, **Q5_K**, **Q6_K**, **Q8_0** (2-bit through 8-bit) — each validated exact vs a
   dequantize-then-f32 reference and, at a 4096 GEMV, **1.8–1.9× faster + ~7× lighter** than that
   path. A `QMatrix` unifies them behind one `matmul_q`,
   and the Qwen3 loader is format-agnostic, so **a genuine `Q4_K_M` model off Hugging Face runs on
   Ferric** — `unsloth/Qwen3-0.6B-Q4_K_M` (which mixes Q4_K and Q6_K, even within one qkv) generates
   coherent text: *"The capital of France is Paris."* And it runs **in a browser tab** too: the same
   wasm/WebGPU path produces logits matching native Metal (argmax 12095, Σ −379147.98 both), so an
   arbitrary standard quantized model is cross-fabric-consistent from one codebase. One loader serves
   the **whole dense Qwen family** — `qwen3` (QK-norm) *and* `qwen2` (QKV bias, no QK-norm) — by
   feature-detecting from tensor presence; `Qwen2.5-0.5B-Instruct-Q6_K` off HF generates *"…Paris. It
   is the largest city in Europe…"* **And it's correct against the reference, not just coherent:**
   Ferric's greedy decode is **token-for-token identical to llama.cpp** on `Qwen3-0.6B-Q4_K_M`,
   `Qwen3-0.6B-Q5_K_M`, and `Qwen2.5-0.5B-Q6_K` (`scripts/validate_vs_llamacpp.sh`) — matching argmax
   at every step means the logits are right across all six quant kernels and both architectures.
   Nothing else pure-Rust runs the standard quantized ecosystem *and*
   the browser from one source. **Llama runs too**: `Llama-3.2-1B-Instruct-Q6_K` (arch `llama`)
   generates *"The capital of France is Paris, the city…"* once two Llama-isms are handled — the
   `rope_freqs` per-frequency RoPE scaling and, the load-bearing one, prepending **BOS** (Llama
   collapses without it; Qwen sets `add_bos_token=false` so nothing changes there). (The older
   non-K formats Q5_0/Q4_1 and the tiktoken pre-tokenizer edge cases are the next reach.)

## Where Ferric is behind (be honest)
**Raw GEMM throughput for prefill/training.** The general f32 matmul is a one-thread-per-output
kernel at ~550–710 GFLOP/s at 1024³ on an M5 Max (thermal-dependent). Getting past that is where
burn/CubeCL and LlamaWeb lead — but **not**, as this doc once claimed, via workgroup tiling. Measured
across 512³–4096³, three ways:

| kernel | 1024³ | 4096³ | vs naive |
|---|---|---|---|
| naive (1 thread/output, coalesced) | ~710 | ~450 | 1.0× |
| register-tiled (1×8 per thread) | ~280 | ~280 | 0.6× |
| shared-memory tiled (64×64, 4×4 micro) | ~215 | ~229 | 0.3× |

Among *hand-written WGSL* kernels the naive one **wins** (cross-vendor: same on the NVIDIA RTX 4050),
because the hardware caches already serve the reuse tiling exists to capture. **But that was never the
ceiling — the matrix hardware was**, and it *is* reachable from portable WGSL. `matmul_coop` uses WGSL
cooperative-matrix (`coop_mat8x8`, `coopMultiplyAdd`) lowering to the GPU's matrix unit — one kernel,
both vendors:

| | naive | WGSL tiled | **coop_mat** |
|---|---|---|---|
| Apple M5 Max, Metal (exact f32) | ~700 GFLOP/s | ~215 | **~4300 (6×), bit-identical (0.0e0)** |
| NVIDIA RTX 4050, Vulkan (TF32) | ~293 GFLOP/s | — | **9400 @2048³ (32×), Δ~6e-3** |

The M5 is the first Apple GPU with real in-GPU matrix hardware; the 4050 exposes NVIDIA tensor cores
via `KHR_cooperative_matrix`. This **corrects the earlier "portable WGSL can't reach tensor cores"
claim** — it reaches simdgroup_matrix AND NVIDIA tensor cores today, through our naga fork
(`EXPERIMENTAL_COOPERATIVE_MATRIX`, `unsafe ExperimentalFeatures::enabled()` token). A naga SPIR-V
codegen bug (coop pointer index not cached) is worked around by let-binding the indices. Precision
differs by vendor — Metal computes exact f32, NVIDIA computes f32-coop as **TF32** (~1e-2, like
PyTorch's default) — which, along with fp-order, is why coop stays a fast-path, not the bit-identical
cross-fabric default. Remaining: wire coop_mat into the matmul selector, 16×16 + larger register
tiles, and the quant-matmul prefill path on NVIDIA (Metal already ships; blocked on a diagnosed
wgpu coop-load barrier gap).

**Unbounded exact attention (both paths default).** Both hot-path attention kernels stream their
keys in 2048-key chunks with an **online softmax** (running max/sum + a per-thread output accumulator),
so neither ever materializes an intermediate scores tensor and neither has a context-length cap:
- **Prefill** — `flash_attention_prefill`, one workgroup per (query, head), O(head_dim) state instead
  of the composed path's [nh,T,T]. Measured 2.0× at T=1024 up to **12.4× at T=5000** (saves 400 MB of
  scores); the win grows with T because the composed path is O(T²) memory traffic.
- **Decode** — `fused_decode_attention`, one workgroup per head, replaces the ~12-dispatch composed
  single-query path at any cache length.
Both are **exact** (same math as the composed reference), so they are the **default**, not opt-in:
validated across the 2048-chunk boundary (prefill T=3000/5000, decode S=2049/4096/5000 all max|Δ|~2e-7),
all three reference models stay token-for-token identical to llama.cpp, and the 1.7B cross-fabric
fingerprint is bit-identical. Verified end to end on a real model: a 3640-token prompt prefills
(2 chunks) and a 2200-token generation crosses the boundary at steady throughput, neither degrading
to the composed fallback.

**Decode speed vs llama.cpp.** Bonsai-27B decodes at ~171 ms/token against llama.cpp's 22 ms (~8×).
The gap is now dominated by two things: llama.cpp reaches ~90% of the 325 GB/s memory roofline on its
Q2_0 GEMV where Ferric's reaches ~57%, and llama.cpp's **zero-copy mmap load** (0.25 s) exploits
Apple unified memory to skip the copy+repack that costs Ferric ~1.8 s warm. Closing either needs
work Ferric hasn't done: subgroup-level GEMV reductions, and a GPU-consumable on-disk layout so
weights can be mapped rather than repacked.

**Browser ceiling.** The moat is one codebase native↔browser, and it is now demonstrated on a *real*
model: **PrismML Ternary Bonsai 1.7B (Qwen3 dense, ternary Q2_0) generates in a browser tab on
WebGPU** — the same `Qwen3` Rust code as native, fed a fetched `Vec<u8>` via `GgufSource`. Verified in
headless Chrome: coherent output, **19.7 ms/token (~50 tok/s)**, load 279 ms. No mainstream project
runs a ternary hybrid-family LLM in a tab. The remaining ceiling is size: WebGPU per-buffer/total
memory limits cap what fits, so the 7 GB 27B needs weight streaming/sharding Ferric doesn't have yet
(1.7B/4B are fine).

*Closed since this section was first written:* GGUF + k-quants (Q4_K/Q6_K) and PrismML ternary
(Q1_0/Q2_0/TQ2_0); an exact GPT-2/BPE tokenizer; the elementwise+matmul-epilogue **fusion compiler**
(`fuse.rs`, whole-graph `Lazy`); kernel **autotuning** (naive-vs-tiled GEMM, per-shape Q2_0 kernel
choice); and projection fusion in the model itself.

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
| load | 10 s | 2.5 s | 0.25 s (mmap) |
| prompt (5 tok) | 2.5 s | **0.38 s** | 0.08 s |
| decode | re-prefilled (288 ms/tok) | **cached+batched+fused — 171 ms/tok** | 22 ms/tok |
| Q2_0 matmul (cold) | ~70 GB/s | **~101–186 GB/s** | — (ceiling: 325 GB/s) |

Decode is ~171 ms/token against llama.cpp's 22 ms — ~8× off (was ~32×). The honest claim stays
**"Ferric runs Bonsai-27B correctly,"** not that it matches llama.cpp's speed.

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
5. **Micro-benchmark gains did not transfer, because the benchmark was cache-resident.** Re-reading
   one 24 MB weight 20× leaves it in the SLC; the real model touches each weight once per token,
   always cold from DRAM. Coalescing the split-K loads (stride by *word*, not by block) won 30–40% on
   the bench — `gdn qkv` @1 token 0.34 → 0.24 ms — and **zero** end-to-end (295 → 301 ms/token, noise).
   The only honest shape here is the 338 MB LM head, far too big to cache: cold, both kernels land at
   ~66–70 GB/s. *(This item originally concluded "so we're DRAM-bound" — **wrong**, and disproved by
   items 7–8 below: we were latency-bound the whole time, and the same cold LM head now runs at 101
   GB/s. Coalescing genuinely didn't matter; the reason wasn't the one stated.)*
   A `vec4<u32>` variant on the *codes* was tried and is *worse*: it cuts work units 4×, wrecking
   load balance, and Apple already coalesces consecutive `u32` loads. (vec4 on the *activations* is a
   different matter entirely — see item 8.)
6. **Per-dispatch overhead is real but not the story:** ~0.06 ms for a trivial op × ~640 ops/token
   ≈ 38 ms, about 13% of decode.

7. **Find the wall before climbing it.** A shader that only reads (512 MB buffer, far bigger than
   any cache) hits **239 GB/s scalar / 325 GB/s vec4** — llama.cpp streams Bonsai at ~326 GB/s, so
   WGSL reaches the same roofline and the memory path was never our wall. Our matmul sat at 22% of
   bandwidth *and* 22% of ALU: at 22% of both, it was **latency-bound**, and the fix had to be in the
   inner loop.
8. **It was the activation loads, not the weights.** The scalar form issues 16 `x` loads per code
   word — 5120 per output against 320 code loads — so `x` dominated the instruction stream and
   starved the kernel of issue slots. Reading `x` as `vec4<f32>` and reducing each 4 weights with
   `dot()` took the cold LM head from **70.5 → 101 GB/s** (1 token) and **80.9 → 181 GB/s**
   (5 tokens); `ffn_gate` @5 tokens 77.5 → **186.5 GB/s**. From 22% to ~57% of roofline.
9. **The textbook fix lost.** Output-major (transposed) weights make adjacent threads read adjacent
   words — and measured *worse* (cold LM head 70.5 → 49.1 GB/s). Row-major already works because each
   thread streams 1280 contiguous bytes and consumes whole cache lines on its own; coalescing across
   threads buys nothing, while output-major scatters each thread's own stream ~1 MB per step.

10. **Occupancy-starved GEMVs want fewer, wider matmuls.** At one token every projection is a
    memory-bound GEMV that can't fill the machine, so projections sharing an input were merged into a
    single wider matmul: FFN gate+up (64 layers), attention q+k+v (16), GDN qkv+z+α+β (48). Q2_0 is
    row-major, so stacking outputs is just concatenating raw bytes at load — no repack, no new kernel;
    the forward splits the result with zero-copy narrows. Gate+up alone measured **1.79×** as one
    [5120→34816] vs two [5120→17408]; end to end decode went **288 → 179 ms/token**, logits unchanged.

**Where decode time goes now** (~180 ms/token, measured with the built-in `FERRIC_PROFILE` timer —
one sync'd submit per category, so it attributes GPU work not op count):

| category | share | what it is |
|---|---|---|
| FFN (64 layers) | 44% | gate+up matmul, down matmul, SwiGLU |
| GDN mixer (48 layers) | 41% | in_proj + out matmuls, conv, gates, the recurrence |
| attention (16 layers) | 14% | q/k/v + out matmuls, decode-attention |
| lm_head + embed | ~1.5% | negligible — the 338 MB head is one matmul over one token |

This **corrects an earlier guess in this doc** ("gated-delta-net ~14 ms, matmuls dominate ~110 ms").
The recurrence kernel is minor; FFN and the GDN mixer cost about the same, and both are dominated by
their Q2_0 matmuls, which at these decode shapes run ~57 GB/s effective. Measured-not-assumed is the
rule: several confident conclusions in this section died on contact with a correct measurement.

**A second machine changed the picture — and the tooling.** An NVIDIA RTX 4050 (Vulkan) is far less
thermally noisy than the M5 Max (±1 ms vs ±30 ms run-to-run), so it's now the machine decode work is
*measured* on. Bonsai-**1.7B** (Qwen3 dense) there:

| | before | after fused attention | note |
|---|---|---|---|
| decode | 35 ms/tok | **32 ms/tok** | fused single-query attention, ~9% |

Profiling the 1.7B decode on the 4050 (attention in *every* layer, unlike the 27B hybrid) showed
**attention at 54%** — because `decode_attention` composed ~12 dispatches (reshape/permute/contiguous/
broadcast/matmul/softmax/matmul) with intermediate tensors. `fused_decode_attention` collapses that to
**one workgroup-per-head kernel** (scores → softmax → weighted-V in shared memory, validated == the
composed path to ~1e-8); attention dropped to 49% and decode to 32 ms. SwiGLU was also fused
(silu·mul → one kernel) — correct, but a single-dispatch saving is below the ~1 ms resolution.

The honest read: at batch-1 decode every op is a tiny GEMV that underfills the GPU, so decode is
**dispatch/latency-bound on a long dependent chain** (~15 kernels/layer × 28–64 layers), not
bandwidth-bound. The Q2_0 GEMV itself is already well-tuned (word-strided coalesced loads, vec4
activations, full workgroup occupancy — an "idle-threads" worry was checked and disproved) and sits
at ~32% of the memory roofline because on-the-fly ternary unpacking is compute-bound, which is
inherent to the format. So the remaining levers are structural: fuse whole layers to shorten the
dependent chain, a preallocated write-in-place KV cache (vs today's re-concatenate) for long context,
and eventually multi-token decode — not shaving individual kernels.

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
