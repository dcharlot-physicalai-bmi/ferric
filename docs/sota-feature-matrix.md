# SOTA feature matrix — what Ferric needs to be the go-to runtime on all hardware

**Focus.** Feature completeness across every fabric, not throughput on any one machine. A runtime is
adopted for what it *can do*; speed on a given laptop is a tuning question and is tracked separately
in `composable-engine-survey.md`.

**Provenance.** ✅ verified present in the tree · ❌ verified absent · ⚠ partial, with the limit named
· ❓ not verified this pass. Absences are code-checked, not assumed.

---

## A. Serving & throughput features

| feature | state | note |
|---|---|---|
| Structured / constrained output | ✅ | JSON-schema-constrained sampling in-runtime, deterministic across fabrics. **Ahead of most.** |
| Tool / function calling | ✅ | plus an MCP module in `ferric-serve` |
| Embeddings + pooling | ✅ | LAST-pooling, matches reference conventions |
| Chunked prefill | ✅ | verified bit-identical to whole-history, incl. windowed models |
| Prefix cache | ⚠ | **one slot**. SGLang's RadixAttention generalises to a radix tree over arbitrary branching — the shape agent loops actually produce |
| Speculative decoding | ⚠ | a draft/verify path exists; **not verified EAGLE-class**. EAGLE 3.1 (vLLM) and EAGLE3 dynamic-tree kernels (TensorRT-LLM) both shipped in the last 45 days |
| **Continuous batching** | ❌ | `ferric-serve`: *"One request at a time (the GPU serializes anyway); continuous batching is the P1 follow-up."* **This is the single biggest serving gap.** |
| Paged / block KV cache | ❓ | contiguous `KvBuf` today; paging is what makes many-session serving memory-viable |
| Multi-model in one process | ❌ | SGLang ships it; vLLM does not |
| Disaggregated prefill/decode | ❌ | standard in large deployments |
| **KV cache quantization** | ❌ | no implementation. Directly sets max context per GB — matters most on the small devices Ferric targets |

## B. Hardware coverage — the "all hardware" axis

| fabric | state | note |
|---|---|---|
| Metal / Vulkan / DX12 / GL | ✅ | via `wgpu`, one codebase |
| **Browser WebGPU** | ✅ | same source to `wasm32`. Genuinely rare |
| CPU fallback | ❓ | wgpu GL/software path only; no hand-vectorised CPU kernels (AVX-512 / NEON) verified |
| **CUDA native** | ❌ | NVIDIA reached through Vulkan, not cuBLAS/CUTLASS/cuDNN. Leaves tensor-core throughput on the table on the most common accelerator |
| **ROCm native** | ❌ | AMD via Vulkan; AITER/CK unused |
| **NPUs** | ❌ | no Hexagon (MNN shipped it Jul 2026), no Ascend/CANN, no CoreML/ANE, no OpenVINO |
| Multi-GPU: tensor / pipeline / expert parallel | ⚠ | `ferric-dist` does **pipeline** (layer sharding) only; no tensor or expert parallelism |
| RDMA / high-speed interconnect | ❌ | Velox has UCX; NCCL/NVLink equivalents absent |

**The honest read.** `wgpu` buys breadth cheaply and is why "runs everywhere" is defensible at all.
It also caps the ceiling on every vendor's flagship path: no tensor cores via CUDA, no matrix cores
via ROCm, no NPU at all. **Breadth is real; depth on any single vendor is not.**

## C. Model capability

| feature | state | note |
|---|---|---|
| Dense GQA transformers | ✅ | qwen2/3, llama, phi3, gemma/2/3 |
| MoE (incl. shared experts) | ✅ | qwen35moe, deepseek2, indexed expert kernels |
| MLA (latent attention) | ✅ | deepseek2, legacy `attn_kv_b` path |
| SSM / Mamba-2 hybrids | ✅ | nemotron_h, lfm2 short-conv |
| Per-layer embeddings, shared KV | ✅ | gemma4 |
| RoPE: NEOX/NORM, YaRN, Llama-3 scaling, partial, 2-D | ✅ | all reference-verified as of 2026-08-15 |
| Vision / multimodal | ⚠ | Muse Glimmer ViT + mmproj works; gemma4 mmproj **unwired**; no audio/video |
| **Multi-token prediction (MTP)** | ❌ | skipped on nemotron; gemma4 ships an `mtp-` file unused. Free speedup left on the floor |
| **LoRA / adapter serving** | ❌ | only DeepSeek's `q_lora_rank`/`kv_lora_rank` MLA internals exist — that is not adapter serving |
| Absorbed-MLA path | ❌ | refused at load; it is the variant that makes the KV cache small |
| Q-LoRA-factored attention | ❌ | refused at load; needed for full-size DeepSeek |

## D. Quantization

| format | state |
|---|---|
| Q4_K, Q5_0, Q6_K, Q8_0, IQ4_XS, IQ4_NL, Q2_0 ternary | ✅ packed kernels |
| **MXFP4** | ❌ — **GPT-OSS ships in it**; without this those weights cannot be run as released |
| FP8 (E4M3/E5M2) | ❓ |
| AWQ / GPTQ | ⚠ a ternary-GPTQ example exists; general support unverified |
| KV cache quant | ❌ (see §A) |

## E. Where Ferric is genuinely ahead

Not throughput. These are real and uncontested in the surveys:

1. **Joules as an objective function.** `ferric-joule` measures, and refuses to report unmeasured
   figures. Zero mentions of joule/watt/kWh across NVIDIA Switchyard *and* DeepSeek Harness.
2. **Sound verification.** `ferric-certify`; interval certificates; determinism across fabrics.
3. **One codebase, native and browser.** Ratchet and Burn overlap the fabric, not the whole span.
4. **Architecture registry that refuses unknowns** rather than loading them down a near-miss path.
5. **Constrained decoding in-runtime**, deterministic across fabrics.

## F. Ranked build order

Ranked by *adoption blocked per unit of work*, which is not the same as difficulty:

1. **Continuous batching** — turns a demo server into a usable one. Nothing else in §A matters while
   the server is single-request.
2. **MXFP4** — a format gate: GPT-OSS cannot run as released without it.
3. **KV cache quantization** — largest context-per-GB lever, and it compounds on exactly the small
   devices Ferric claims.
4. **RadixAttention-style prefix tree** — a data structure, not a kernel; aimed at agent loops, which
   is what the new open-weight frontier is being used for.
5. **LoRA adapter serving** — the standard way people ship fine-tunes.
6. **MTP** — the weights are already on disk, unused.
7. **CUDA/ROCm native backends** — the largest ceiling raise, and the largest amount of work; it is a
   second backend tree, not a tuning pass.
8. **NPU backends** (CoreML/ANE, Hexagon, Ascend) — where on-device inference is going.

## G. Not assessed

CPU SIMD kernels; FP8; paged KV internals; guided-decoding coverage vs xgrammar/outlines; audio and
video multimodal; scheduler fairness and preemption; observability surface.
