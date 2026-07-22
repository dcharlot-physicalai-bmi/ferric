# The real NPU execution-provider

**DONE (2026-07): the fabric dispatches on the real Apple Neural Engine.**
`ferric_tensor::npu_coreml::CoreMlNpu` is a CoreML execution-provider whose ANE use is
**confirmed by `MLComputePlan`** (Apple's own scheduler receipt — the honesty gate below);
`detect_devices()` adds it as `Device::Npu` only when the receipt shows the Neural Engine.

What the receipts taught us (all measured from Rust, see `npu_coreml::plan_experiments`):

| model pattern                                   | preferred device |
|-------------------------------------------------|------------------|
| rank-2 matmul (512×512, dynamic operands)       | CPU              |
| rank-2 linear+relu (fixed weights)              | CPU              |
| 3×3 conv, 64 ch, 56×56                          | CPU              |
| **rank-4 matmul (1,8,512,64)·(1,8,64,512)**     | **ANE** (all ops)|
| **1×1 conv, 512 ch, 32×32**                     | **ANE**          |

The embedded EP uses the rank-4 matmul (≈16 KB `.mlmodelc`, authored with coremltools MIL +
`coremlcompiler`); arbitrary `bmm` shapes tile onto it (512×512 output tiles, K covered 8
lanes × 64 per prediction, lanes host-summed). Measured on M5 Max: ~500 µs dispatch overhead,
~700 GFLOP/s through the host-boundary tiling — real numbers from real silicon, calibrated by
the same `Planner` as every other device. On this box the M5's GPU tensor units still win
every size class; the point is the fabric *knows* that by measurement, not assumption.

---

## The original plan (kept for the record)

The fabric's NPU **routing + dispatch path is done and verified**
(`crates/ferric-tensor/examples/npu_routing.rs`): `Device::Npu(Arc<dyn NpuBackend>)`
dispatches correctly and the adaptive `Planner` routes across CPU + GPU + NPU. A real
execution-provider just implements the same `NpuBackend` trait and is routed identically.
`probe_npu()` stays honest — `dispatchable: false` until a real EP is wired; the ANE is
never faked on the GPU/CPU.

## Apple ANE (via CoreML) — the concrete path

Investigated 2026-07: `objc2-core-ml` (v0.3.2, crates.io) provides the bindings needed and,
crucially, **`MLComputePlan` / `MLComputePlanDeviceUsage` + `MLNeuralEngineComputeDevice`** —
so you can *prove per-operation* that the ANE (not the GPU/CPU) ran. What it does **not**
provide is a runtime graph builder: CoreML runs a **compiled model** (`.mlmodelc`), and there
is no offline Rust path to author one.

To finish, in an environment with coremltools/Xcode:
1. Generate a small matmul / `linear_relu` CoreML model (`coremltools`), compile to `.mlmodelc`,
   embed the bytes (or ship alongside).
2. Vendor `objc2-core-ml` (+ transitive deps) into `vendor/` via `cargo vendor`.
3. Impl `NpuBackend` for a `CoreMlNpu`: load the model with
   `MLModelConfiguration.computeUnits = .all` (ANE-eligible), feed inputs as `MLMultiArray`,
   run, read back.
4. **Honesty gate:** query `MLComputePlan` for the model and report the per-op device usage;
   only add `Device::Npu` in `detect_devices()` when the plan shows ANE dispatch. Never claim
   the ANE unless the compute plan confirms it.

Note: the ANE is tuned for fp16 conv/attention patterns; a plain fp32 matmul may be scheduled
to GPU/CPU by CoreML even with `.all`. Measure (via `MLComputePlan`) before claiming a win.

## Browser (WebNN) — the other path

`navigator.ml` with `deviceType: 'npu'` is the browser route (maps to CoreML/DirectML/etc.).
Absent in current Chrome here (even behind the experimental flags). When available, implement
the `NpuBackend` equivalent against `MLGraphBuilder` in the `ferric-web` layer.
