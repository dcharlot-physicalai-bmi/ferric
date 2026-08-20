# Backend expansion: NPU and vendor-native accelerators

Feasibility study for `sota-feature-matrix.md` §F items 7 and 8 (CUDA/ROCm native, NPU backends).
Read-only review. No Rust was written for this document.

**Method.** Every claim about the tree is code-checked and the check is shown. Every claim about an
external crate is taken from the crates.io or GitHub API on 2026-08-16, not from prior belief.
Statements are labelled **verified** (I ran it or read it), **read** (documented by the vendor or
maintainer, not executed here), or **projected** (my estimate, not measured).

**No performance numbers appear in this document.** Machine load average was 18 to 21 throughout
(three other agents building), so any throughput figure measured now would be noise. Where a
performance effect is argued it is argued from structure and labelled as unmeasured.

---

## 0. Corrections to `sota-feature-matrix.md` §B

The task briefed me from §B. Two of its claims do not survive contact with the tree, and one is
overstated. This matters because both errors point the same way: they understate what Ferric already
has, and §F item 8 is sized as if from zero.

### 0.1 "no CoreML/ANE" is false. The ANE execution provider exists and runs.

`crates/ferric-tensor/src/npu_coreml.rs` (312 lines) is a working CoreML execution provider that
dispatches on the Apple Neural Engine. I ran its test on this machine (M5 Max, macOS 26.5.2):

```
$ cargo test -p ferric-tensor --lib npu_coreml -- --nocapture
plan: [ios16.cast→ANE ios16.cast→ANE ios16.matmul→ANE ios16.cast→ANE]
ANE confirmed: true
  bmm 512x512x512: err 6.529e-5
  bmm 100x300x200: err 4.150e-5
  bmm 64x1024x64: err 4.446e-5
test npu_coreml::tests::coreml_npu_loads_and_bmm_matches_the_fp16_oracle ... ok
test result: ok. 2 passed; 0 failed
```

**Verified, with the strength of the evidence named.** The matmul is scheduled to the Neural Engine
by Apple's own `MLComputePlan`, and the result matches an fp16 oracle. `docs/NPU.md` already
documents this as done since 2026-07. §B was not updated.

⚠ **What the green suite does and does not prove (correction, 2026-08-16).** The transcript above is
composite rather than a verbatim capture, and the distinction matters. `ane_confirmed` is **printed,
never asserted** — it appears at `npu_coreml.rs:40` (field), `:72` (assignment), `:264` (`eprintln!`)
and `sched.rs:166` (read), and in no assertion anywhere in the tree. The only guarded claim in that
test is the numeric one, `err < 2e-2`, which CoreML satisfies **on CPU** just as well as on the ANE.
The test also early-returns silently on any machine where the CoreML EP fails to load. So:

- the ANE dispatch is evidenced by an **observational receipt** (the printed compute plan), not by a
  test that can fail on it;
- a green suite is therefore *not* evidence the ANE executed — only the printed plan is;
- the numeric tolerance is loose in absolute terms: `2e-2` against a peak output of 0.0628 and RMS
  0.0262 for the 512-cube case is ~32% of peak and ~76% of RMS. It discriminates in practice (three
  independent fault injections were each caught) but it is not a tight bound.

The §0.1 conclusion stands. It rests on a receipt, and a receipt is weaker than a guard. Turning
`ane_confirmed` into an assertion behind an explicit "ANE required" env gate is the cheap fix.

The honest form of the §B row is not "no CoreML/ANE". It is: *an ANE execution provider exists,
is gated on a compute-plan receipt, and serves exactly one op (`bmm`) into a scheduler that no LLM
runtime calls.* That is a much more useful sentence, and section 1 explains why.

### 0.2 "no tensor cores via CUDA" is true but the surrounding sentence is overstated.

§B's honest-read paragraph says the wgpu choice means "no tensor cores via CUDA, no matrix cores via
ROCm". The first clause is literally true and the implication is not. `crates/ferric-core/src/lib.rs`
requests `wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX` and `SHADER_F16`, and there are **14**
cooperative-matrix kernel entry points:

```
matmul_coop            matmul_coop16          matmul_q2_0_coop       matmul_q2_0_coop16
matmul_q2_0_coop2pass  matmul_q4_k_coop       matmul_q5_k_coop       matmul_q6_k_coop
matmul_q8_0_coop       matmul_ternary_coop    matmul_ternary_coop4   matmul_ternary_coop_n16
matmul_ternary_coop_n32                       matmul_ternary_coop_n64
```

`coop16_ok()` gates on `coop_matrix && shader_f16 && backend == Vulkan`, and the source comment
records that this was developed against an RTX 4050. WGSL `coop_mat` lowers to
`VK_KHR_cooperative_matrix`, which *is* the tensor-core instruction path on NVIDIA, and to
`simdgroup_matrix` on Metal.

So tensor cores are reached. What is absent is the **vendor library** path: cuBLAS, cuBLASLt,
CUTLASS, cuDNN. The gap is library-quality GEMM (autotuned tile schedules, split-K, epilogue
fusion), not hardware access. That is a materially smaller and differently-shaped gap than §B
implies, and it changes the ranking in section 5.

### 0.3 What is accurately absent

**Verified absent** by grep across `crates/`: Hexagon/QNN, Ascend/CANN, OpenVINO, DirectML,
cuBLAS/CUTLASS/cuDNN, and any AMD AITER/CK path. No occurrence of any of these names outside of
comments describing them as future work. §B is correct on all of these.

⚠ **Correction (2026-08-16).** An earlier revision of this section also listed **WebNN** as verified
absent. That was wrong, and wrong in the same way this document criticises §B for being wrong:
`crates/ferric-web/examples/npu.rs` is an executable WebNN path with a real assertion, dispatching
through `Device::BrowserWorker` and reporting which backend WebNN actually bound (npu / gpu / cpu /
none). §1.1 of this very document lists that file among the seam's consumers, so the two sections
contradicted each other. A grep that returns nothing is evidence about the grep, not about the tree.

### 0.4 Not everything is WGSL/wgpu

The task asked whether anything today is not WGSL/wgpu. **Three things are**, and they are the
existing precedents that any new backend should be designed against:

| path | file | mechanism | not wgpu because |
|---|---|---|---|
| Metal 4 tensor units | `ferric-tensor/src/metal4.rs` (2037 lines) | `objc2-metal`, MTL4 command model, MetalPerformancePrimitives `tensor_ops::matmul2d`, precompiled `metal4_gemm.metallib` embedded | wgpu cannot express cooperative tensor ops through the MTL4 path |
| Apple ANE | `ferric-tensor/src/npu_coreml.rs` | `objc2-core-ml`, compiled `.mlmodelc` embedded (4 files, about 16 KB) | WebGPU cannot target an NPU at all |
| CPU vector units | `ferric-tensor/src/cpu_simd.rs` (410 lines) | plain safe Rust shaped for LLVM autovectorisation, `std::thread::scope` | deliberately not intrinsics, to stay portable to wasm |

`metal4.rs` is the most important precedent. It interoperates at the **`wgpu::Buffer` level**, not
the host-buffer level: `metal4::wgpu_buffer_raw()` extracts the underlying `MTLBuffer` from a wgpu
buffer, so `bmm_resident` runs a native Metal kernel directly on wgpu-allocated memory with no
host round trip. It is wired into the real path at `ferric-tensor/src/lib.rs:764` and `:783`
(`matmul_bt` and `matmul_bt_act` try `metal4_linear` before falling back to WGSL).

**This is the shape a native kernel backend should take in Ferric, and it already exists and works.**

---

## 1. The seam, quantified

The task asked: what is the seam a new backend implements, how many ops, is it a trait or an enum?

**The answer is that there are two seams, they are not connected, and the narrow one is the one
that does not matter.**

### 1.1 Seam A: `NpuBackend`, a real trait, 2 methods, reaching nothing

`crates/ferric-tensor/src/sched.rs:45`:

```rust
pub trait NpuBackend: Send + Sync {
    fn name(&self) -> String;
    fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32>;
    fn linear_relu(&self, x: &[f32], rows: usize, w: &[f32], in_: usize, out: usize) -> Vec<f32> {
        self.bmm(x, w, 1, rows, in_, out).iter().map(|v| v.max(0.0)).collect()
    }
}
```

**One required compute method.** `linear_relu` has a default. The contract is host `&[f32]` in,
`Vec<f32>` out, so the backend owns its own upload and readback.

Alongside it is a closed enum, `sched::Device`, with 6 variants (`Gpu`, `Cpu`, `Remote`,
`BrowserWorker`, `Npu`, `Metal4`). Adding a device *kind* means editing that enum plus its three
match arms (`name`, `bmm`, `linear_relu`). Adding a device *instance* of an existing kind means
only implementing the trait. `sched.rs` is not owned by the concurrent workflow, so this is editable.

**The problem.** Grep for every consumer of this seam:

```
$ grep -rn --include='*.rs' -l "NpuBackend|Device::Npu|sched::Fabric|tensor::sched" crates/
crates/ferric-tensor/examples/adaptive.rs
crates/ferric-tensor/examples/devices.rs
crates/ferric-tensor/examples/fabric.rs
crates/ferric-tensor/examples/npu_routing.rs
crates/ferric-tensor/src/npu_coreml.rs
crates/ferric-tensor/src/sched.rs
crates/ferric-web/examples/browser_worker.rs
crates/ferric-web/examples/npu.rs
```

**Verified: two library files and six examples. Zero LLM runtimes.** `ferric-llama` never reaches
`ferric_tensor::sched`, `Fabric`, `Device` or `NpuBackend`. Implementing `NpuBackend` gives you a
device in the heterogeneous-scheduler demo and moves no token of real inference.

(Stated precisely on purpose: `ferric-llama` does have a `sched` module of its own —
`crates/ferric-llama/src/sched.rs`, the continuous-batching scheduler — which is an unrelated
namespace. An earlier revision said "does not reference `sched` anywhere", which is false as written.)

### 1.2 Seam B: the actual inference path, which is not a trait

`ferric-llama`'s seven runtime modules (`qwen3`, `qwen35`, `deepseek2`, `gemma4`, `lfm2`, `cosmos`,
`mla`) call inherent methods on two concrete structs:

- `ferric_core::Context` (a struct holding `wgpu::Device` + `wgpu::Queue` + capability flags)
- `ferric_tensor::Tensor` / `QTensor` / `QMatrix` (an `Arc<wgpu::Buffer>` plus shape and strides)

There is **no trait**. `grep '^\s*pub trait ' ` across `ferric-core/src`, `ferric-tensor/src` and
`ferric-llama/src` returns exactly one hit, and it is `NpuBackend` from seam A. A backend cannot be
"implemented" here in the Rust sense. It can only be added as a fallible fast-path branch inside
each op, which is precisely what `metal4_linear` does.

**Op count.** Intersecting the public API of `ferric-tensor/{lib,dtype,nn}.rs` and
`ferric-core/kernels.rs` (213 public functions) with the method names actually called from
`crates/ferric-llama/src/*.rs` gives **71 distinct ops**:

| category | n | ops |
|---|---|---|
| matmul / linear | 13 | `matmul` `matmul_bt` `matmul_q` `matmul_q4_k_id_wsum` `matmul_q4_k_swiglu_id` `matmul_q5_0_id_wsum` `matmul_q6_k_id_wsum` `matmul_q8_0_id_wsum` `matmul_q8_0_swiglu_id` `mm_bt` `linear` `try_matmul_swiglu` `try_ffn_mega` |
| elementwise / activation | 21 | `add` `add_t` `mul` `mul_t` `silu` `gelu` `gelu_tanh` `relu2` `sigmoid` `sigmoid_t` `sqrt` `exp` `sin` `cos` `abs` `max` `sum` `softplus` `softcap` `swiglu` `scalar` |
| RoPE | 9 | `rope` `rope_at` `rope_at_ex` `rope_interleaved` `rope_partial` `rope_scaled` `rope_scaled_interleaved` `rope_t` `apply_rope_costable` |
| norm | 5 | `rmsnorm` `rmsnorm_t` `rmsnorm_weightless` `layernorm` `add_rmsnorm` |
| SSM / conv | 3 | `gated_delta_rule_stateful` `depthwise_conv1d_causal` `conv2d` |
| attention | 2 | `mha_causal_t` `flash_attention_prefill` |
| MoE routing | 2 | `moe_topk` `moe_topk_ex` |
| gather | 2 | `gather0` `gather_rows` |
| quant / misc | 2 | `dequant` `resize_bilinear` |
| view / host | 12 | `reshape` `narrow` `permute` `view` `contiguous` `len` `clone_prefix` `broadcast_to` `transpose` `cat` `to_vec` `tensor` |

**59 compute ops and 12 view/host ops.** The view/host ops are stride arithmetic and copies; they do
not need a backend kernel. So the honest number for a *complete* backend is **59 compute ops**, and
for a backend that only wants the bandwidth-dominated majority of decode time it is far fewer
(section 5.1).

Underneath, the WGSL surface is larger than the op surface because of quantization: **135 `@compute`
entry points** (108 in `ferric-tensor`, 27 in `ferric-core`) and **39 `pub fn matmul_*` variants** in
`dtype.rs` alone. (Counted 2026-08-16; now 142 = 115 + 27, the extra 7 being
`crates/ferric-tensor/src/kvquant.rs`, added after this count. Recorded so the figure is understood as
a snapshot of a moving tree rather than re-derived badly later.)

### 1.3 The fact that decides everything below: weights are quantized blocks, dequantized inside the kernel

`matmul_q` dispatches on `QShard`, which is a GGUF block format:

```rust
QShard::Q2_0(w) => self.matmul_q2_0(w),   QShard::Q4_0(w) => self.matmul_q4_0(w),
QShard::Q4_1(w) => self.matmul_q4_1(w),   QShard::Q5_0(w) => self.matmul_q5_0(w),
QShard::Q5_1(w) => self.matmul_q5_1(w),   QShard::Q4_K(w) => self.matmul_q4_k(w),
QShard::Q5_K(w) => self.matmul_q5_k(w),   QShard::Q6_K(w) => self.matmul_q6_k(w),
QShard::Q8_0(w) => self.matmul_q8_0(w),   QShard::Iq4Xs(w) => self.matmul_iq4_xs(w),
```

The weight never exists as a dense tensor. Each kernel reads packed codes plus per-block scales and
reconstructs values in registers. That is the whole point (decode is bandwidth-bound, so the weight
is read in its compressed form), and it is the single hardest constraint on every backend below.

A backend that cannot accept "here is a buffer of Q4_K blocks, here is the block layout, produce
`x·Wᵀ`" cannot serve Ferric's hot path without first materialising a dense fp16 copy of every
weight, which discards the reason the format was chosen.

---

## 2. Kernel backend versus graph backend

This is the split the task asked for, and it is the correct organising principle.

**A kernel backend** takes individual ops and buffers. You keep your own graph, your own memory,
your own control flow. Ferric's seam B is a kernel seam, and `metal4.rs` proves a native kernel
backend can slot into it at the `wgpu::Buffer` level.

**A graph backend** takes a whole model, compiles it ahead of time into a vendor binary, and gives
you back one opaque `run(inputs) -> outputs`. You do not choose kernels, you do not see
intermediates, and you cannot interleave your own ops. Anything the compiler does not support
either falls back to CPU silently or fails to convert.

**What a graph backend costs Ferric specifically**, beyond generic awkwardness:

1. **The quantization formats die at the boundary.** Graph compilers accept their own quantization
   schemes (CoreML palettisation, QNN `SFIXED_POINT`, OpenVINO NNCF). GGUF K-quants are not among
   them. You must dequantize to fp16 offline, which multiplies weight footprint by roughly 4x
   against Q4_K and removes the bandwidth advantage that motivates the format. **Projected**, from
   the bits-per-weight ratio, not measured.
2. **The architecture registry loses its meaning.** §E item 4 of the feature matrix says Ferric
   "refuses unknowns rather than loading them down a near-miss path". A graph backend inverts this:
   an unsupported op is silently partitioned to CPU by the vendor runtime, and the model runs, wrong
   or slow, with no refusal.
3. **Determinism across fabrics is lost.** §E item 2 claims determinism and `ferric-certify`
   soundness. A vendor graph compiler chooses its own fusions, layouts and accumulation orders, and
   changes them between SDK versions. Bit-identical cross-fabric output cannot be promised through it.
4. **Sampling, KV cache and continuous batching sit outside the graph.** Ferric's KV cache, prefix
   cache, guided decoding and the scheduler from §F item 1 all operate between ops. A whole-model
   graph gives no seam for them without recompiling per shape and per cache state.

The blunt version: **a graph backend is not a backend for Ferric. It is a second, parallel runtime
that happens to live in the same repository.** It should only be considered where it is the sole way
to reach the silicon at all.

---

## 3. Per-backend assessment

Crate data from the crates.io and GitHub APIs, 2026-08-16.

### 3.1 CoreML / ANE (Apple)

- **Integration surface: GRAPH.** CoreML executes a compiled `.mlmodelc`. This review did not locate
  a runtime graph builder or an op-level entry point in the CoreML API surface, and `docs/NPU.md`
  reached the same conclusion in 2026-07 (*"there is no offline Rust path to author one"*). **Read**,
  not exhaustively proven: the claim is that the documented API is graph-only.
- **Rust crates.** `objc2-core-ml` **0.3.2**, last published 2025-10-04, part of `madsmtm/objc2`,
  pure Rust, already vendored in `vendor/objc2-core-ml` and already used. Alternatives found:
  `coreml` 0.3.5 (`doom-fish/coreml-rs`, 0 GitHub stars, 224 downloads), `coreml-native` 0.2.0,
  `candle-coreml` 0.3.1 (last updated 2025-09-10), `rlx-coreml` 0.2.14 (`MIT-RLX/rlx`, 22 stars,
  554 downloads). None is more capable than what Ferric already uses.
- **The authoring gap, which is the real constraint.** `objc2-core-ml` binds the *inference* API.
  Authoring a `.mlmodelc` requires `coremltools` (Python) and `coremlcompiler` (Xcode). **This
  review did not locate a Rust crate that writes MIL / `.mlpackage` files.** Searches for a Rust
  MIL author returned only inference bindings. Consequently the shipped EP embeds **one** compiled
  shape, `(1,8,512,64)·(1,8,64,512)`, and tiles all other shapes onto it with host-side K-summation.
  That is why it is a `bmm`-only device.
- **Pure Rust?** **Yes, and it already is.** `objc2-core-ml` depends only on `objc2` and
  `objc2-foundation`. No `cc`, no `bindgen`. CoreML.framework is an OS framework already present on
  every Mac, not a vendor blob Ferric ships. The `.mlmodelc` bytes in the repo are a build artifact
  of Apple's compiler, which is a supply-chain fact worth naming but is not a linked C++ library.
- **Cost to go further.** To serve seam B rather than seam A you would need a compiled CoreML
  program per (op, shape) class, authored out-of-band in Python, embedded, and selected at runtime,
  with fp16 dense weights. Given section 1.3, **I do not think this is worth doing**, and section 5
  ranks it accordingly.

### 3.2 Hexagon / QNN (Qualcomm)

- **Integration surface: GRAPH.** QNN (now QAIRT) converts and quantizes a model offline, then runs
  it via `libQnnHtp.so` on the HTP tensor accelerator. **Read.**
- **One genuine nuance.** The Hexagon **HVX vector unit** is separately programmable at kernel level,
  and Qualcomm published XNNPACK-on-HVX in March 2026, plus `qualcomm/hexagon-mlir` (BSD-3-Clause,
  186 stars, created 2025-12-19, last pushed 2026-07-02) which compiles Triton kernels and PyTorch
  models. So an op-level path to HVX exists in principle. **This review did not locate any Rust
  binding to it**; hexagon-mlir's documented entry points are Triton and PyTorch.
- **Rust crates.** Thin. `rlx-qnn` **0.2.14** (`MIT-RLX/rlx`, Apache-2.0, 22 stars, **117 total
  downloads**, updated 2026-08-12) is the only real candidate. Its own documentation is candid that
  validation is on the x86 CPU reference backend: *"Both validate without a Snapdragon device: QNN's
  x86-64 CPU reference backend (`libQnnCpu.so`) runs on a commodity Linux host"*, with *"remaining:
  real HTP silicon soak"*. Also found: `ironaccelerator-qnn` 2.2.0 (82 downloads). At 117 and 82
  downloads these are pre-adoption; **this is an empty lane, and that is the finding.**
- **Pure Rust?** **No.** `libQnnHtp.so` is a proprietary Qualcomm binary distributed under an SDK
  licence requiring acceptance of terms. Ferric would dlopen a closed vendor blob it cannot vendor,
  cannot patch, and very likely cannot redistribute. **This directly contradicts the `.cargo/config.toml`
  policy**, which states: *"Ferric is self-contained: every dependency's source lives in `vendor/`.
  We build only from source we own and can patch."*
- **Hardware access.** No Snapdragon device is present in this environment, so nothing here could be
  validated on silicon even if written.
- **Note on the §B citation.** MNN 3.6.1 shipped a Hexagon backend on 2026-07-22. **Verified** by
  search. §B's parenthetical is accurate.

### 3.3 Ascend / CANN (Huawei)

- **Integration surface: mixed, mostly GRAPH.** ACL/AscendCL offers a runtime API, but the deployed
  path is an offline-converted `.om` model.
- **Rust crates. This is the emptiest lane of the six.** `cann` / `cann-sys` **0.1.1**
  (`cann-rs/cann-rs`): **0 GitHub stars, 72 total downloads**, licensed `MIT OR Apache-2.0`
  (an earlier revision of this document said "no licence declared" and that was false — the
  crates.io API reports `MIT OR Apache-2.0` for the crate record and for 0.1.1), last pushed
  2026-06-24. `yijunyu/tile-rs` (Apache-2.0, 9 stars, created 2026-02-23) compiles Rust kernels to
  Ascend and others via MLIR and is the only op-level Rust lane located, but at 9 stars it is a
  research prototype. ONNX Runtime's CANN execution provider is classified **community-maintained,
  preview**, and its prebuilt binaries are **Python-only**, so the `ort` route requires building
  ONNX Runtime from source.
- **Pure Rust?** **No.** Requires the CANN SDK (`libascendcl`), a proprietary Huawei binary.
- **Hardware access.** No Ascend device present. Unvalidatable here.
- **Assessment.** A 0-star, 72-download binding to a proprietary SDK that cannot be tested here,
  for hardware that is export-restricted in several jurisdictions. Recommend explicitly declining,
  and recording that as a decision rather than an omission. The licence is **not** among the reasons:
  it is a permissive `MIT OR Apache-2.0`, and the decline rests on the proprietary SDK dependency, the
  absence of hardware to validate against, and the maturity signals.

### 3.4 OpenVINO (Intel)

- **Integration surface: GRAPH,** but the friendliest of the graph backends. It compiles an IR or
  ONNX model and targets CPU, iGPU and Intel NPU from one artifact.
- **Rust crates.** `openvino` and `openvino-sys` **0.11.0**, published 2026-05-06,
  **maintained by Intel themselves** (`intel/openvino-rs`, Apache-2.0, 134 stars). This is the only
  lane of the six with a first-party vendor-maintained Rust binding. Roughly 2.5M downloads.
  Build dependencies are `openvino-finder` and `env_logger` only, with **no `cc` and no `bindgen`**;
  it locates and loads the installed OpenVINO C API shared library.
- **Pure Rust?** **No.** The OpenVINO runtime is a large C++ shared library that must be installed
  on the target. The *Rust source* stays pure and needs no C compiler at build time, but the
  deployed artifact depends on a vendor C++ runtime.
- **Strategic note.** Ferric's advantage on Intel silicon is that wgpu already covers the iGPU
  through Vulkan and DX12. OpenVINO's marginal gain is the Intel NPU specifically, and it arrives
  bundled with every cost in section 2.

### 3.5 CUDA native

- **Integration surface: KERNEL. This is the important result of the whole review.** CUDA is not a
  graph backend. You get the driver API for memory and launches, NVRTC to compile kernel source at
  runtime, and cuBLAS/cuBLASLt/cuDNN as optional library calls. This maps onto seam B directly.
- **Rust crate.** `cudarc` **0.19.9**, updated **2026-08-11**, **7,084,146 downloads**,
  `chelsea0x3b/cudarc` (Apache-2.0, 1206 stars, pushed 2026-08-12). Actively maintained. It wraps
  the driver API, NVRTC, cuBLAS, cuBLASLt, cuDNN, cuSPARSE, cuSOLVER, cuFFT, cuRAND and NCCL.
  (The older `cust` from Rust-CUDA is **0.3.2, last published 2022-02-16**: stale, do not use.)
- **Pure Rust? This is better than expected, and the detail matters.** `cudarc`'s only non-optional
  dependency is **`libloading ^0.9.0`**, verified through the crates.io dependency API. No `cc`, no
  `bindgen`, no `cmake`. Its default `dynamic-loading` feature *"will not require any libraries to
  be present at build time"*; it dlopens the driver at runtime.
  **And `libloading` is already vendored in this tree** (`vendor/libloading` exists). So adding
  cudarc adds one vendored pure-Rust crate and zero C toolchain.
  The honest qualifier: at **runtime** on an NVIDIA box you are calling closed NVIDIA libraries. But
  that is already true today, because the Vulkan path calls the same closed driver. **Ferric's
  pure-Rust property, as expressed by its own build policy, survives CUDA-native intact.** That is
  a strategic fact and it is the opposite of what §B implies.
- **NCCL bonus.** `cudarc` exposes NCCL, which is the named gap in §B's RDMA/interconnect row.
- **Hardware access.** No NVIDIA GPU in this environment. Everything here is **read**, not verified
  on silicon. The existing coop16 comments indicate an RTX 4050 has been used previously, so a
  reference machine exists somewhere in the project's fleet; I could not confirm its availability.

### 3.6 ROCm native

- **Integration surface: KERNEL,** structurally identical to CUDA. HIP is a near-copy of the CUDA
  driver API, and `hiprtc` mirrors NVRTC.
- **Rust crates, ranked by what the data supports.**
  - `cubecl-hip-sys` **7.14.6085000**, updated 2026-07-30, **1,063,209 downloads**, from
    `tracel-ai/cubecl` (Apache-2.0, 2319 stars, pushed 2026-08-14). This is Burn's ROCm layer and is
    by a wide margin the healthiest ROCm binding in the Rust ecosystem. Non-optional deps are `libc`
    and `regex`; **no `cc`, no `bindgen`.**
  - `rocm-rs` 0.5.2 (`RustNSparks/rocm-rs`, MIT, 103 stars, 4,778 downloads, pushed 2026-07-03).
    Broader library coverage, far less adoption.
  - `hip-sys` / `hipblas-sys` **0.1.1 / 0.1.0, both last published 2023-07-25**. Unmaintained; do
    not use.
  - `rlx-rocm` 0.2.14 (1,882 downloads), `ironaccelerator-rocm` 2.2.0 (80 downloads). Pre-adoption.
- **A signal worth recording.** ONNX Runtime's ROCm execution provider is now marked
  **deprecated** in the official EP documentation. Whatever AMD's direction is, routing to ROCm
  through ONNX Runtime is a shrinking path.
- **AITER / Composable Kernel.** §B is correct that these are unused. **This review did not locate
  a Rust binding to AITER or to Composable Kernel.** Both are C++ template libraries, and CK in
  particular is header-template-heavy, which is close to a worst case for FFI. Reaching them would
  require a C shim, which *would* introduce `cc` into the build and break the property that section
  3.5 shows CUDA preserves.
- **Pure Rust?** Yes at build time via `cubecl-hip-sys` (dlopen the HIP runtime). No, if AITER/CK
  are wanted.
- **Hardware access.** No AMD GPU present. **Read**, not verified.

---

## 4. The pure-Rust ledger

Ferric's central claim deserves a precise statement rather than a slogan, so I checked it.

**Verified, from `Cargo.lock`:** zero occurrences of `cc`, `bindgen` or `cmake`. The only `-sys`
crates are OS-binding shims (`jni-sys`, `ndk-sys`, `wayland-sys`, `windows-sys`, `js-sys`,
`web-sys`, `linux-raw-sys`, `renderdoc-sys`). `pkg-config` appears, pulled only by `khronos-egl` and
`wayland-sys`, both Linux windowing transitives of wgpu. **Nothing in this workspace compiles C or
C++ today.** All 171 dependencies are vendored (260 MB in `vendor/`), and `.cargo/config.toml`
redirects crates.io to that directory, so any new dependency must be vendored to build at all.

Against that standard:

| backend | C/C++ compiled at build time | vendor blob at runtime | vendorable source | verdict |
|---|---|---|---|---|
| CUDA via `cudarc` | **no** (`libloading` only, already vendored) | NVIDIA driver, already required by the Vulkan path | yes | **preserves the property** |
| ROCm via `cubecl-hip-sys` | **no** (`libc`, `regex`) | HIP runtime | yes | **preserves the property** |
| ROCm via AITER / CK | **yes**, needs a C++ shim | AITER/CK | no | **breaks it** |
| CoreML via `objc2-core-ml` | **no** | OS framework, not shipped | yes, already vendored | **preserves the property** |
| OpenVINO via `openvino-sys` | no | **large Intel C++ runtime, must be installed** | binding yes, runtime no | **weakens it** |
| QNN via `rlx-qnn` | no | **proprietary `libQnnHtp.so`, licence-gated** | **no** | **breaks it** |
| CANN via `cann-sys` | no | **proprietary `libascendcl`** | **no** | **breaks it** |
| any route via `ort` | no | **prebuilt ONNX Runtime C++ binary; and Microsoft ships prebuilts only for CUDA/TensorRT, so every other EP means building ONNX Runtime from source** | no | **breaks it** |

The clean conclusion: **the two backends that raise the ceiling most (CUDA and ROCm native) are also
the two that cost the pure-Rust property nothing.** Every NPU lane except Apple's costs it outright.
This inverts the intuition in §F, where NPUs are listed as the destination and CUDA/ROCm as the
grind.

---

## 5. Ranked recommendation

Ranked by ceiling raised per unit of work, with the pure-Rust cost carried explicitly.

### 5.1 Rank 1: CUDA native via `cudarc`, scoped to the quantized matvec

**Reasoning.** It is a kernel backend, so it fits seam B without architectural change. It costs
nothing in build purity (section 4). It targets the most common accelerator. `metal4.rs` already
demonstrates the exact integration pattern: a fallible native fast-path tried ahead of the WGSL
kernel, operating on the same buffers.

**Scope it narrowly.** Do not port 59 ops. The `ferric-bandwidth-ceiling` finding in this project's
own memory is that the gap against llama.cpp is *entirely* the quantized matmul. So the first
increment is the `matmul_q*` family for the two or three formats that actually ship (Q4_K, Q6_K,
Q8_0), written as CUDA C and compiled at runtime with **NVRTC**, which keeps kernel source in the
repo as text and adds no build step. Everything else stays on WGSL via the existing Vulkan path,
which already works on NVIDIA.

**Critically, cuBLAS is the wrong tool for the first increment.** cuBLAS wants dense fp16. Ferric's
hot op is a quantized matvec that dequantizes in registers. An NVRTC-compiled custom kernel is both
the higher-performance and the more architecture-preserving choice. cuBLASLt becomes interesting
later for batched prefill GEMM, once §F item 1 lands and prefill is actually batched.

**Blocker to name up front: this machine has no NVIDIA GPU.** **Verified** via
`system_profiler SPDisplaysDataType`: one adapter, `Apple M5 Max`, vendor Apple (0x106b), zero
NVIDIA or AMD matches. Per this project's reference-diff rule, this work cannot be validated here.
It should not start until a target machine is confirmed available, because a CUDA kernel that has
never run is worth less than nothing.

**Unmeasured.** I make no claim about the size of the speedup. The argument is structural: a
hand-written quantized matvec reads each weight byte once in its packed form, which is the same
argument that makes the existing WGSL kernel bandwidth-bound.

### 5.2 Rank 2: ROCm native via `cubecl-hip-sys`, as a port of rank 1

**Reasoning.** HIP is near-identical to CUDA, so this is largely a retarget of rank 1's kernels
through `hiprtc` rather than fresh design. Do it second and reuse, do not do it in parallel.

**Explicitly decline AITER and Composable Kernel** for now, and record why: they are C++ template
libraries with no Rust binding located, and reaching them introduces a C++ shim that breaks the
property section 4 establishes. That is a bigger loss than the kernel quality is a gain.

### 5.3 Rank 3: finish what already exists on Apple, rather than starting a new backend

The ANE EP works but serves one op into a seam no runtime calls. Two increments are available and
both are small relative to any item above:

1. **Wire `metal4`'s resident path more widely.** This is native, already integrated at
   `lib.rs:764`, already correct, and reaches Apple's tensor units. Extending it beyond
   `matmul_bt` / `matmul_bt_act` is incremental work on proven code.
2. **Decide the ANE's fate deliberately.** Serving seam B through CoreML needs compiled programs per
   op and shape, authored in Python, with dense fp16 weights. Given section 1.3 that is a poor
   trade. The defensible position is to keep the ANE as a measured, honestly-gated scheduler device
   and say so, rather than leave §B claiming it does not exist.

**Correct §B either way.** It currently understates the tree on two counts (sections 0.1 and 0.2).

### 5.4 Rank 4: OpenVINO, only if an Intel NPU target is named

Best-maintained binding of the NPU group and the only first-party one. Still a graph backend with
every cost in section 2, and wgpu already covers Intel iGPUs. Worth doing only against a concrete
deployment target, not speculatively.

### 5.5 Declined, with reasons recorded

- **Hexagon / QNN.** Graph-only for the HTP; the sole Rust crate has 117 downloads and validates on
  an x86 reference backend rather than silicon; requires a licence-gated proprietary blob that
  cannot be vendored; no Snapdragon hardware here. Revisit if a Rust binding to the **HVX** vector
  unit appears, since that would be a kernel lane rather than a graph lane.
- **Ascend / CANN.** 0-star binding (permissively licensed, `MIT OR Apache-2.0`); ONNX Runtime's EP is community-preview with
  Python-only prebuilts; proprietary SDK; no hardware; export-restricted.
- **`ort` as a universal shortcut.** Superficially attractive because one dependency reaches CoreML,
  QNN, CANN, OpenVINO, DirectML, CUDA and TensorRT. Rejected because it makes Ferric a wrapper
  around a prebuilt ONNX Runtime C++ binary, abandons the GGUF quantization path, abandons
  determinism, and abandons the pure-Rust claim. Microsoft ships prebuilt binaries only for CUDA and
  TensorRT, so every NPU EP would additionally require building ONNX Runtime from source. `ort`
  itself is healthy (2.0.0-rc.13, 15.8M downloads, updated 2026-07-28) and is still the wrong
  dependency for this runtime.

---

## 6. What this review did not establish

1. **No NVIDIA, AMD, Qualcomm, Intel-NPU or Ascend hardware was available.** Every claim in
   sections 3.2 through 3.6 is from documentation and registry metadata. Nothing was executed on
   any of that silicon.
2. **No performance figure of any kind was measured or is claimed**, deliberately, per the load
   constraint. The bandwidth argument in 5.1 is structural and unmeasured.
3. **`cudarc`'s dynamic-loading claim was read, not exercised.** I verified its dependency graph
   through the crates.io API (`libloading` only), which is strong evidence, but I did not vendor it
   or compile anything against it.
4. **Kernel-level HVX reachability from Rust is unresolved.** `qualcomm/hexagon-mlir` exists and is
   BSD-3, but this review did not locate a Rust entry point to it and did not read its source.
5. **The 71-op count is call-site derived**, from intersecting declared public functions with method
   names appearing in `ferric-llama/src`. Ops reached only through re-exports or trait methods, or
   named dynamically, would be missed. It is a good estimate of the seam's width, not a proof.
6. **`ferric-onnx` was not assessed as an integration route.** It exists as a pure-Rust ONNX importer
   with a starter op set (MatMul, Add, Relu) running on Ferric's own kernels. It is an importer, not
   a backend, so it does not bear on this question, but it was not examined in depth.
7. **No claim is made about whether §B's other rows are accurate.** I checked the three rows I was
   briefed on. `sota-feature-matrix.md` was not audited as a whole, and given that two of three rows
   checked were wrong, an audit of the rest is probably warranted.
