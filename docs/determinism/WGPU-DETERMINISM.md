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

## Browser fabric (Dawn/Tint) — measured 2026-07-22

`ferric-web::ferric_fabric_probe` runs the identical probe in Chrome
(headless, WebGPU via ANGLE-Metal on the M5 Max). Result: **not
bit-identical with the strict-IEEE native path** — Tint compiles WGSL with
fma contraction, and the browser's compiler is not ours to configure.

The diagnostic detail: every browser row reproduces, bit-exactly, a value
from a specific earlier round of the native experiments — mm/rmsnorm/rope/
sigmoid equal the pre-NoContraction native values; sqrt and mha equal the
runtime-zero-barriered rounds under contraction-allowed compilation. Two
implications, both measured rather than assumed:

1. The browser is *deterministic*, with contraction-allowed semantics.
2. The runtime-zero `det_bar` barriers pin op-level behavior even under
   contracting compilers — the barriered kernels landed on identical bits
   across Tint, fast-math Metal, and fusing NVIDIA.

Path to browser↔native bit-identity, if wanted: a "portable-det" kernel
profile with runtime-zero barriers on every float op (including inside
det_exp/det_recip/det_sincos Horner chains) — WGSL-only, so it works on any
conforming WebGPU implementation, at a to-be-measured throughput cost.
Until that ships: browser parity is validated by tolerance (CPU reference
deltas ~1e-7), not by digest. Also note "the browser" is not one fabric:
Dawn picks ANGLE-Metal / Vulkan / D3D per platform, and contraction choices
can differ across them (the mha row already shows two contracted compilers
disagreeing) — cross-browser digests must be measured per platform.

## Portable-det profile — 2026-07-22, second measurement round

All det functions and every kernel accumulation/product chain are now
barriered (runtime-zero `det_bar` at each step; op order preserved exactly).
Native regression: **all 7 probe rows unchanged** on Metal — the barriers are
value no-ops under strict compilation, as designed. `matmul_native` timing on
the 64×48×32 validation case: 5.95 ms before → 4.85 ms after (noise-level;
the case is overhead-dominated — a real perf ceiling needs larger shapes).

Browser (Dawn/Tint, ANGLE-Metal) after portable-det: **mm, rope, and sigmoid
now match the strict-native digests bit-exactly** — barrier-forced plain
sequences hold in a compiler we cannot configure. Still divergent: sqrt and
rmsnorm (both via det_rsqrt) and mha (+ demo-lm, which composes them). The
sqrt forensic in-browser reproduces native fast-math Metal's fingerprint
EXACTLY (183/768 deviations, first at i=0, cpu 3f1db1f4 vs gpu 3f1db1f3, 1
ULP low) — some Metal fast-math transform inside the Newton chain survives
per-step XOR barriers. Open question, scoped for a micro-kernel forensic
session: emit per-iteration intermediates from the GPU and diff against the
CPU replica stage by stage to identify the exact op. Until resolved,
browser↔native digest parity covers the mm/rope/sigmoid kernel class;
rsqrt-dependent kernels remain tolerance-validated in the browser.

Scoreboard: CPU cross-arch ✅ digest · GPU cross-vendor ✅ digest (Vulkan
re-verify of portable-det pending Tailscale re-auth of the test box) ·
browser mm/rope/sigmoid ✅ digest · browser rsqrt-class ⏳ forensic.

## Browser campaign round 2 — the memory-pin discovery (2026-07-22)

Stage-by-stage forensics (Context::det_rsqrt_stage — one dispatch per stage,
so the chain stays register-resident up to the exported value) delivered the
mechanism: **ANGLE/Tint transforms register-only float chains straight
through integer-XOR barriers, but cannot optimize across stores to a private
array at runtime-opaque, mutually-distinct indices.** The identical Newton
chain scored 183/768 (1 ULP low) as registers and 0/768 through the array.

Applied: det_rsqrt rewritten as a z-offset array chain → **browser sqrt now
bit-matches native strict (c7d7cc05…, 0/768 vs plain-IEEE CPU)**. Browser
scoreboard: **mm, sqrt, rope, sigmoid = 4/7 rows digest-identical with
native Metal AND native Vulkan.** Native regression clean throughout; all
validations green.

Two instructive failures, preserved for the follow-up:
- det_bar itself as a store/load at the SAME index expression gets
  store-forwarded by Tint (barrier elided; browser reverted to fast-math).
  Index expressions must differ or chain across slots.
- Array-pinning det_exp/det_recip the same way made rope/sigmoid REGRESS in
  the browser (native unmoved) — ANGLE appears to register-promote small
  dynamic arrays differently per inlining context, re-exposing fast-math.
  Reverted; XOR versions retained (they converge in the probed contexts).

Still open (scoped): rmsnorm + mha (and demo-lm, their composite) in the
browser — the residue lives in the interaction between det functions and
their inlining context under ANGLE, not in the accumulation loops (memory-
pinned accumulators changed nothing). Next tool: per-stage forensic kernels
for the rmsnorm row (recip → rsqrt-arg → inv → output chain) to name the op.
Native (Metal+Vulkan) remains 7/7 digest-identical and is the shipping
guarantee; browser is 4/7 digest + tolerance elsewhere.

## 2026-07-23 — BROWSER PARITY COMPLETE: 7/7 rows, three fabrics, one digest

The rmsnorm forensic escalation resolved the last divergences and completed
the triangle: **all seven probe rows — including the full 3-layer
transformer forward — are bit-identical across Chrome (Dawn/Tint via
ANGLE-Metal), native Apple Metal, and native NVIDIA Vulkan**, with sqrt
verified 0/768 against the plain-IEEE CPU replica on all three.

The address-space hierarchy that emerged (the campaign's core lesson):
1. **Private arrays** (even at runtime-opaque indices) — register-promoted
   in some inlining contexts; unreliable under Tint/ANGLE.
2. **Workgroup memory without barriers** — legally promotable too (no
   cross-thread visibility guarantees); held in dynamic-trip loops in one
   kernel context and fused in another. Context-dependent; unreliable.
3. **Storage (device) memory** — host-visible; NO conforming compiler may
   promote or fuse across it. **The only universally reliable pin.**

Final rmsnorm structure: per-row scalar chain (accumulation, ms·recip,
rsqrt Newton, output products) entirely through a 16-slot-per-row storage
scratch buffer, SSA-style unique slots for straight-line steps, two
loop-carried slots for accumulations. mha and demo-lm converged with NO
further changes — their divergence had been inherited from rmsnorm inputs
end-to-end. Native digests never moved throughout; all validations green;
matmul timing unchanged.

What this buys the ecosystem: a policy/model in the demo-LM kernel class
now has ONE behavioral signature valid in the training browser, on any
Metal device, and on any Vulkan device — Ferrite eval vectors recorded in
any of the three verify bit-exactly on the other two. Browser-trained →
deployed continuity is a checkable digest, not a claim.

Follow-ups (scoped): apply the storage-scratch pattern to LAYERNORM and
SOFTMAX (same class, not in the demo path); perf-profile the storage
traffic at real model shapes and consider a fast-path/det-path kernel pair
if it costs; extend the probe with a per-fabric CI row.

## 2026-07-23 — kernel set complete: 9/9 rows, three fabrics

layernorm and softmax received the storage-chain treatment (same per-row
scalar pattern as rmsnorm; softmax's exp-sum accumulates through a scratch
slot with each exp pinned by its store to out[]). Probe extended to 9 rows
so their parity is measured, not assumed:

    mm · rmsnorm · sqrt · rope · mha · sigmoid · layernorm · softmax ·
    demo-lm (full transformer) — identical digests on Chrome (Dawn/Tint),
    Apple Metal, and NVIDIA Vulkan; sqrt 0/768 vs plain-IEEE CPU on all
    three. All CPU-reference validations green (layernorm 4.8e-7,
    softmax 3.7e-9); KV-cache decode exact.

These kernels are free for anyone to use, study, and build on (MIT OR
Apache-2.0) — the probe, the CPU replicas, and this document are the
reproduction kit. Remaining scoped work: perf profile of the storage
scratch at real model shapes; a per-fabric CI row.

## 2026-07-23 — the measured price of determinism (Metal, M5 Max)

`examples/det_perf.rs` at transformer shapes, A/B against the pre-campaign
kernels (worktree at 102d625, same harness):

| kernel                | fast    | deterministic | cost  |
|-----------------------|---------|---------------|-------|
| matmul 512³           | 2.80 ms · 95.9 GF/s | 3.26 ms · 82.4 GF/s | +16% |
| rmsnorm 512×4096      | 2.22 ms | 6.43 ms       | +190% |
| layernorm 512×4096    | 2.03 ms | 3.69 ms       | +82%  |
| softmax 512×4096      | 2.10 ms | 2.48 ms       | +18%  |
| mha T=256 H=8 dh=64   | 3.25 ms | 4.25 ms       | +31%  |

Reading: XOR-barrier kernels (matmul, softmax, mha) pay 16–31%; the
storage-chain norms pay for per-element device-memory round-trips, rmsnorm
worst at 2.9×. Both variants are the same naive one-thread-per-row designs,
so absolute throughput is modest either way (GEMM tiling remains the known
roadmap item). Conclusion for the roadmap: a deterministic parallel-tree
reduction for the norms (storage-pinned partials, one pin per tree level
instead of per element) should beat BOTH variants while keeping the digest
guarantee — better than maintaining a fast/det kernel pair. Until then:
these numbers are the honest price list, and at education/demo scale
(the probe, the portal benches, Ferrite eval vectors) it is negligible.
