# Upstream proposal: reproducible float math across wgpu backends

*Prepared 2026-07-22 by the Institute for Physical AI @ BMI (Ferric project).
Reference patches (against wgpu/naga v30) in this directory. NOT yet
submitted — awaiting a decision on venue (issue vs PR) and framing.*

## Problem

The same WGSL compute shader produces different f32 results on different
wgpu backends, even when every operation in it is IEEE-exact:

1. **Metal compiles shaders with fast-math by default.** wgpu-hal creates
   `MTLCompileOptions::new()` and sets `preserveInvariance`, but never
   `mathMode` — and Metal's default permits reassociation. Forensic
   measurement (bit-exact CPU replica of a barriered Newton sequence):
   NVIDIA/Vulkan deviated on **0/768** inputs; Metal deviated on **183/768**,
   each by 1 ULP.
2. **SPIR-V permits fma contraction unless `NoContraction` is decorated.**
   naga never decorates, and NVIDIA's compiler fuses `a*b + c` freely — and
   *differently from Metal* on expressions with more than one product.
3. **MSL contracts fma even under `MTLMathModeSafe`** — fp-contract is a
   separate, default-on pragma (`FP_CONTRACT`), so fixing (1) alone still
   leaves Metal ≠ Vulkan on any mul+add chain.

Net effect: cross-device reproducibility of WGSL compute is impossible today
from the WGSL layer alone. Source-level workarounds (integer-domain
bitcast barriers) fail because compilers legally fold constant XORs and
reassociate through inlined calls.

## Fix (three one-liners, conceptually)

- wgpu-hal/metal: `options.setMathMode(MTLMathModeSafe)` (fallback
  `setFastMathEnabled(false)` pre-macOS-15) — `0001`.
- naga/spv: decorate float binary arithmetic results (`FMul/FAdd/FSub/FDiv/
  FRem/VectorTimesScalar`) with `NoContraction` — `0002`.
- naga/msl: emit `#pragma STDC FP_CONTRACT OFF` in the preamble — `0003`.

With all three, WGSL evaluates as-written IEEE round-to-nearest on both
backends. Measured result: seven compute kernels (matmul, RMSNorm, sqrt,
RoPE, causal attention, sigmoid, and a full 3-layer transformer forward)
produce **bit-identical output hashes on Apple M5 Max (Metal) and RTX 4050
(Vulkan)**, with sqrt verified 0/768 against a plain-IEEE CPU replica on
both. Accuracy vs CPU references: 1e-8-scale deltas; matmul exactly 0.

## Upstream framing (important)

Unconditional strict math trades performance (fma fusion) for
reproducibility, so upstream should likely expose it as an **opt-in**, e.g.:

- a `ShaderModuleDescriptor` / device-level flag
  (`float_semantics: Strict | Fast`), defaulting to today's behavior; or
- a naga writer option mirrored into both backends.

The WebGPU spec conversation is adjacent: browsers (Dawn/Tint) make their
own choices here, and portable-determinism interest exists in the ML-on-
WebGPU community. Worth citing the measurements above; the forensic
methodology (per-kernel FNV of output bits + CPU IEEE replica) is easy for
maintainers to reproduce.

## Status in Ferric

Applied unconditionally in Ferric's maintained forks (`forks/wgpu-hal`,
`forks/naga`) since 2026-07-22 — determinism is the product there. The
Ferrite deploy layer (github.com/dcharlot-physicalai-bmi/ferrite) depends on
it for cross-fabric verified-behavior updates.
