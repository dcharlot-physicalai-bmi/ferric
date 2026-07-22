# Why Ferric's wgpu/naga forks diverge from stock: strict float semantics

*Internal engineering record, 2026-07-22. The reference patches in this
directory document exactly how our forks differ from stock v30 (the current
upstream release, which our forks are based on). This is our determinism
guarantee's foundation — it lives in our forks, maintained by us.*

## The problem we measured

The same WGSL compute shader produces different f32 results on different
backends, even when every operation in it is IEEE-exact:

1. **Metal compiles shaders with fast-math by default.** Stock wgpu-hal
   creates `MTLCompileOptions::new()` and sets `preserveInvariance` but not
   `mathMode`, so Metal's default (permitting reassociation) applies.
   Forensic measurement (bit-exact CPU replica of a barriered Newton
   sequence): NVIDIA/Vulkan deviated on **0/768** inputs; Metal deviated on
   **183/768**, each by 1 ULP.
2. **SPIR-V permits fma contraction unless `NoContraction` is decorated.**
   Stock naga doesn't decorate, and NVIDIA's compiler fuses `a*b + c` freely
   — *differently from Metal* on expressions with more than one product.
3. **MSL contracts fma even under `MTLMathModeSafe`** — fp-contract is a
   separate, default-on pragma, so fixing (1) alone still leaves
   Metal ≠ Vulkan on any mul+add chain.

Source-level workarounds (integer-domain bitcast barriers) are insufficient:
compilers legally fold constant XORs and reassociate through inlined calls.
We measured each of these failure modes directly before changing anything.

## What our forks change

- `forks/wgpu-hal` (metal/device.rs): `setMathMode(MTLMathModeSafe)`, with
  `setFastMathEnabled(false)` as the pre-macOS-15 fallback — `0001`.
- `forks/naga` (spv/block.rs): `NoContraction` decoration on float binary
  arithmetic results (`FMul/FAdd/FSub/FDiv/FRem/VectorTimesScalar`) — `0002`.
- `forks/naga` (msl/writer.rs): `#pragma STDC FP_CONTRACT OFF` in the
  shader preamble — `0003`.

With all three, WGSL evaluates as-written IEEE round-to-nearest on both
backends — the same semantics as our CPU references and wasm.

## Measured result

Seven compute kernels — matmul, RMSNorm, sqrt, RoPE, causal attention,
sigmoid, and a full 3-layer transformer forward — produce **bit-identical
output hashes on Apple M5 Max (Metal) and RTX 4050 (Vulkan)**, with sqrt
verified 0/768 against a plain-IEEE CPU replica on both
(`ferric-core/examples/fabric_probe.rs`). Accuracy vs CPU references:
1e-8-scale deltas; matmul exactly 0.

## Trade-off, owned deliberately

Strict math forgoes fma fusion, trading some throughput for exact
reproducibility. For Ferric that trade IS the product: cross-fabric
bit-identity is what makes verified-behavior deployment (Ferrite),
deterministic replay, and browser↔native parity possible. Performance-
sensitive paths that don't need the guarantee (coop-matrix GEMM, subgroup
kernels) are already feature-gated off the bit-identical default path.

## Maintenance notes

- On each fork rebase to a new upstream release, re-apply these three
  patches (they're small and localized) and re-run `fabric_probe` on both
  test machines before trusting any cross-fabric claim.
- The det-math WGSL transcendentals (`kernels.rs DET_MATH_WGSL`) are the
  other half of the guarantee — vendor `exp/sin/cos/sqrt` differ across
  GPUs regardless of compiler flags.
- Browser fabrics (Dawn/Tint in Chrome) compile WGSL with their own stack;
  parity there must be measured separately, not assumed.
