# Ferric — architecture

**One pure-Rust AI ecosystem across every compute fabric: cloud, local, browser.**

A model defined once runs — the same code, no Python/C++ in the hot path — on a datacenter GPU, a
laptop, an edge robot, and a browser tab, with a scheduler that partitions work across whatever
heterogeneous compute is present for the fastest path.

## Why this doesn't exist yet
The mature browser-AI stacks (ONNX Runtime Web, WebLLM, Transformers.js, LiteRT.js) are all
JavaScript / C++-WASM. The Rust side is real but fragmented and early: `candle` (great CPU, weak
WebGPU), `burn`+`CubeCL` (strong cross-backend kernels, tiny browser demos), `ratchet` (HF, pre-V1,
LLM/audio only), `wgml`/`inferi` (dimforge, early), and `wonnx` — the one Rust WebGPU **ONNX** runtime
— stalled and op-incomplete. **No one has assembled a single coherent Rust ecosystem that is genuinely
the same across cloud + local + browser, heterogeneous-scheduled.** That is Ferric.

## The layers

| Layer | Responsibility | Strategy |
|---|---|---|
| **L0 Fabric** | one device API over WebGPU / Vulkan / Metal / DX12 / CUDA / ROCm / CPU | **leverage `wgpu`** (portable); reach lower for CUDA/ROCm where it pays |
| **L1 Kernels** | compute written once in Rust, compiled to every backend | **leverage/extend `CubeCL`** — SOTA "one kernel, all fabrics" (autotune + fusion) |
| **L2 Tensor / autodiff / fusion** | ndarray, lazy graph, kernel fusion, autograd | leverage `burn` core where it fits |
| **L3 Model runtime + IR** ⭐ | load ONNX / GGUF / safetensors, execute, quantize (int4/fp16) | **ours** — the ONNX-in-Rust path is the stalled gap |
| **L5 Cloud serving** ⭐ | batching, streaming, multi-GPU inference server | **ours** — the cloud leg |
| **L6 Browser runtime** ⭐ | WASM+WebGPU packaging, weight streaming + IndexedDB cache | **ours** — our proven strength |
| **L7 Heterogeneous scheduler** ⭐⭐ | partition a model across CPU+GPU+NPU / multi-device / multi-node | **ours — the real differentiator; no Rust framework has this** |

**Strategic bet:** don't reinvent L0–L2 — `wgpu` + `CubeCL` + `burn` are SOTA and pure-Rust; rebuilding
them is years and worse. Ferric's novel, unclaimed contribution is **L3 + L5 + L6 + L7 and the
unification** — one API/runtime that makes cloud, local, and browser genuinely the same thing.

## Proof of the core thesis (done)
The scariest unknown — *does one pure-Rust kernel really run, validated identically, on a native GPU
AND in the browser?* — is **retired**. `crates/ferric-core` wraps `wgpu` into a `Context` + a matmul
kernel (WGSL, written once). The identical source was run:

- **Native** (Apple M5 Max, Metal): `max|gpu − cpu| = 7.153e-7`
- **Browser** (`wasm32` + WebGPU, headless Chrome): `max|gpu − cpu| = 7.153e-7` — **bit-identical**

Same Rust, same WGSL, same numbers, on two fabrics. That is the foundation stone.

## Workspace
```
ferric/
  crates/
    ferric-core/   L0/L1 — Context + kernels on wgpu; CPU reference + validation.
                   examples/matmul_native.rs  → runs + validates on the native GPU.
    ferric-web/    L6 seed — the same core in WASM; ferric_matmul_demo() runs on browser WebGPU.
                   index.html + run.mjs        → headless-Chrome validation harness.
  docs/ARCHITECTURE.md
```

## Roadmap
1. ✅ **Cross-fabric core** — matmul validated native Metal + browser WebGPU (bit-identical).
2. **Kernel set toward a transformer** — layernorm, softmax, attention, gelu/silu, elementwise; tiled
   matmul + fusion. Validate each cross-fabric.
3. **L3 model runtime** — a graph IR + ONNX importer (the stalled gap), quantized weights (int4/fp16),
   KV cache. Milestone: run a real transformer / robot policy (the SmolVLA components we already ran in
   JS) natively **and** in-browser, from pure Rust.
4. **L6 browser runtime** — weight streaming + IndexedDB cache + a clean JS/TS surface.
5. **L5 cloud serving** — batched, streaming, multi-GPU inference server on the same runtime.
6. **L7 heterogeneous scheduler** — cost-model-driven partitioning across CPU+GPU+NPU / devices. The
   differentiator.

Everything above L2 is one runtime; the fabric (cloud GPU, laptop, browser) is a deployment target, not
a rewrite.

## Self-reliance & ownership (we depend on nothing external)
Ferric is fully self-contained. Every dependency's source is vendored in-repo (`vendor/`, 132 crates
incl. `wgpu`, `naga`, `ash`/Vulkan, `metal`/`objc2`, `glow`, `d3d12`); the build uses **only** that
source (`.cargo/config.toml` → `replace-with = "vendored-sources"`) and succeeds `--offline`. Proven:
the cross-fabric matmul builds with zero crates.io access and still validates bit-identical (7.153e-7).

**Ownership ladder (per crate):**
1. **Vendored** — source in-repo, patchable in place. (all 132, today)
2. **Promoted fork** — strategic crates copied to `forks/<crate>/` and wired via `[patch.crates-io]`;
   cargo builds our copy (not checksum-locked), so we edit + evolve freely. **`naga` is the first**
   (shader translation → our kernel codegen/fusion). Verified: cargo compiles `naga (forks/naga)`.
3. **Aligned** — refactored into the Ferric idiom (naming, API, our optimizations), upstream-tracked
   but not upstream-dependent.

**Promotion order (strategic first, commodity as-needed):**
`naga` (done) → **`wgpu` / `wgpu-core` / `wgpu-hal` / `wgpu-types` / `wgpu-naga-bridge` (done — the
whole device layer, verified bit-identical native Metal + browser WebGPU from the forks)** → `CubeCL` (L1 kernel language) → `burn` core (L2) →
`candle` (reference ops). Backend bindings (`ash`, `metal`, `glow`, `d3d12`) stay vendored + patchable
— commodity we own but don't rewrite unless it pays.

**Policy:** track upstream for fixes/features, cherry-pick, but never be *blocked* by it. Our structure
(one monorepo, one release cadence, patch-anything) is the agility advantage — we reach SOTA faster and
maintain on our own terms.
