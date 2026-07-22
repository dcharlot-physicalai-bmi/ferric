# Metal 4 tensor-unit GEMM — a native backend for the fabric

**Correction (verified 2026-07, macOS 26.5.2 / Apple M5 Max / Metal 4):** the cooperative matmul
path this codebase uses via wgpu is `simdgroup_matrix` (Metal 3-era). Metal 4 ships
**`MetalPerformancePrimitives`** with native **tensor-unit** ops — `matmul2d` / `convolution2d`
(`mpp::tensor_ops`, fp16→fp32, cooperative across SIMD-groups) — which supersede it and are NOT
reachable through wgpu. So on Apple silicon the fabric currently leaves the tensor units unused.

This is a **GPU** path (the M-series tensor units), not the ANE — see `NPU.md` for the ANE.

## What's verified

- The Metal 4 GEMM kernel **compiles here** (Xcode 26.6 toolchain, `metal -std=metal4.0`):
  ```metal
  #include <metal_stdlib>
  #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
  using namespace metal; using namespace mpp::tensor_ops;
  kernel void matMul(tensor<device half,  dextents<int32_t,2>> A,
                     tensor<device half,  dextents<int32_t,2>> B,
                     tensor<device float, dextents<int32_t,2>> C,
                     uint2 tgid [[threadgroup_position_in_grid]]) {
      constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
      matmul2d<desc, execution_simdgroups<4>> op;
      auto tA = A.slice(0, tgid.y*64);
      auto tB = B.slice(tgid.x*32, 0);
      auto tC = C.slice(tgid.x*32, tgid.y*64);
      op.run(tA, tB, tC);
  }
  ```
  (→ ~5.3 KB AIR). `matmul2d_descriptor(m, n, k=dynamic, transpose_l, transpose_r, relaxed_precision)`.
  Note: tensors are **host-bound `MTLTensor` kernel args** — there is no raw-pointer in-kernel
  constructor; the `tensor_inline` variant exists but the ergonomic path is host tensors.

- **The host API is already vendored**: `objc2-metal 0.3.2` has `MTLTensor`, `MTLTensorDescriptor`,
  `MTLTensorExtents`, and the full **MTL4** command model (`MTL4CommandQueue`, `MTL4CommandBuffer`,
  `MTL4CommandAllocator`, `MTL4ComputeCommandEncoder`, `MTL4ArgumentTable`, `MTL4MachineLearningPipeline`).
  So the backend is buildable in clean Rust — no raw `msg_send`.

## Build plan (a `MetalTensorGemm` backend)

1. Runtime-compile the kernel: `device.newLibraryWithSource(MSL, opts)` with the Metal-4 language
   version, get the `matMul` function, `newComputePipelineState`.
2. Allocate A/B (fp16) and C (fp32) as `MTLBuffer`s; create `MTLTensor`s over them via
   `MTLTensorDescriptor` (dimensions = `MTLTensorExtents`, `dataType`, `usage = .compute`).
3. MTL4 dispatch: `MTL4CommandAllocator` → `MTL4CommandBuffer` → `MTL4ComputeCommandEncoder`;
   bind tensors through an `MTL4ArgumentTable`; `dispatchThreadgroups((M+63)/64, (N+31)/32, 1)` with
   `threadsPerThreadgroup = simdWidth*4`; commit on an `MTL4CommandQueue`; read back C.
4. **Verify-first** (the house rule): assert `max|gpu − cpu_reference|` within fp tolerance before
   trusting it, exactly as the existing WGSL matmul is checked.
5. Wire it into the fabric as a device/kernel option the adaptive `Planner` (see `sched.rs`) routes
   large fp16 matmuls to on Apple; keep the WGSL path as the portable default + oracle.

Deliberate follow-on (not a blind speed-run): the MTL4 command model is new API surface and deserves
careful, verified implementation. The feasibility is proven — kernel compiles, host bindings present.
