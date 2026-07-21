# Wiring a real NPU execution-provider

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
