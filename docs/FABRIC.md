# The Compute Fabric

**Thesis: every modern device is a heterogeneous fabric — CPU, GPU, GPU tensor units, NPU — and
smart software knows what it is running on and adapts by measurement, not assumption.** Ferric's L7
scheduler (`ferric_tensor::sched`) is that software: it enumerates every compute unit it can
honestly reach, calibrates a cost model per device on the actual machine, and routes each workload
to the unit the measurements predict fastest.

Everything below is verified in-repo (tests + asserting examples) and was measured on an
M5 Max / macOS 26.5 / Xcode 26.6. Numbers are that box's; the point of the fabric is that it
re-derives them wherever it lands.

## The device roster

| device | reached via | notes |
|---|---|---|
| `Cpu` | native loops | always present; the low-overhead floor (~16 µs dispatch) |
| `Gpu` | wgpu (WebGPU/Metal/Vulkan/DX12) | the portable tier — same kernels native and in-browser |
| `Metal4` | raw MTL4 + MetalPerformancePrimitives | the M-series GPU **tensor units** (`tensor_ops::matmul2d` / `convolution2d`), unreachable through wgpu |
| `Npu` | CoreML execution provider | the **real Apple Neural Engine** — added only with a receipt (below) |

`detect_devices()` builds the roster; `Planner::calibrate` fits `t = overhead + flops/throughput`
per device (2 probe sizes, 2 warm-ups, min-of-5; the rate is clamped — shape-cached devices can
return both probes latency-dominated and degenerate the fit to infinity).

Calibrated on the M5 Max (per-op dispatch; a chained op-DAG flush amortizes the external floor
across its runs — see The op-DAG below):

```
GPU:Apple M5 Max          overhead ~290 µs   throughput ~100 GFLOP/s   (portable WGSL matmul)
CPU                       overhead  ~16 µs   throughput ~3.2 GFLOP/s
Metal4-TensorUnits (fp16) overhead  ~90 µs   throughput  2.1+ TFLOP/s  (host boundary)
NPU:ANE (CoreML, fp16)    overhead ~500 µs   throughput ~700 GFLOP/s   (host-boundary tiling)
```

Routing on this box: tiny → CPU, everything else → tensor units. The ANE never wins a size class
*here* — the M5's GPU tensor units dominate — and the fabric knows that **by measurement**. On
silicon where the NPU is the strong unit, the same Planner routes to it.

## Tensor-unit coverage (the `Metal4` device)

All resident paths run on **wgpu's own MTLDevice** (via `as_hal` interop) with zero host copies:
pad/f16-convert, the tensor op, and an activation-capable un-pad execute as one MTL4 command
buffer reading and writing wgpu buffers directly. Opt-in via `FERRIC_METAL4=1` under the
precision doctrine (fp16-input paths never silently change default results — the `FERRIC_COOP`
precedent), with a ~1e8-flop floor.

| path | measured (M5 Max) |
|---|---|
| `matmul` (NN) resident | 11.9 TFLOP/s @2048³, 13.9× over WGSL |
| `matmul_bt`/`matmul_bt_act` (NT + fused relu/silu/gelu/sigmoid) | 5.0–8.3× on llama-class linears |
| Q2_0 ternary prefill (`matmul_q2_0`, rows ≥ 32: dequant-once → NT GEMM) | 9.3× @512 tokens (13.6 TFLOP/s) |
| Var/Adam training step (all six GEMMs resident) | 3.3–4.3× @batch 1024 × width 2048 |
| `conv2d` (NHWC, runtime-compiled per-config PSOs) | ~2.2× single, ~3–7× batched |
| `Var::conv2d` autograd (dX = strided transposed conv, dW = input×grad correlation, portable WGSL) | FD-verified ~3e-4; conv layers train |

## The op-DAG

`batch()` builds an ordered **segment list** — Wgsl segments (one encoder each) interleaved with
External segments (deferred tensor-unit ops, GEMM and conv alike, as `ExtOp`s owning `Arc`s to
their buffers). At flush, segments execute in order; each External segment splits into runs at
config repeats (same-config ops share scratch and argument tables) and encodes every op of a run
into ONE MTL4 command buffer with inter-op barriers.

Runs **chain GPU-side**: `MTL4CommandQueue::waitForEvent/signalEvent` orders dependent command
buffers on the queue (proven exact by the `floor_probe` copy-chain test), so a whole flush pays
**one** host wait. The floor decomposition that motivated it:

```
empty MTL4 cb, host wait:        ~172 µs   (the platform's commit→completion round trip)
8 cbs, host wait each:          ~1089 µs
8 cbs, queue-chained, 1 wait:    ~279 µs   (~35 µs marginal per chained cb)
```

Measured effect: a same-shape 3-chain (which must split into 3 runs) drops 1.7 → 0.68 ms deferred
(2.4×); the q/k/v pattern runs 1.5–1.7× faster deferred than immediate, bit-identical results.
Beneath it all, the three device caches are bounded MRU lists (16 GEMM / 8 conv / 8 host-boundary
shapes), so alternating shapes — every real model — never rebuild.

Honest boundary: a training step's GEMMs alternate with elementwise WGSL, leaving External
segments mostly single-op — the DAG is roughly neutral there. It pays where resident ops cluster.

## The ANE (the `Npu` device)

CoreML runs compiled graphs, so the EP embeds one (~16 KB `.mlmodelc`) and tiles arbitrary `bmm`
shapes onto it. **The honesty gate:** `detect_devices` adds the device only after `MLComputePlan`
— Apple's own scheduler receipt — confirms the Neural Engine runs the model's matmul. No receipt,
no claim; the ANE is never faked on CPU/GPU.

ANE eligibility is empirical (receipts read from Rust; `npu_coreml::plan_experiments` prints them
for any `.mlmodelc`):

| model pattern | preferred device |
|---|---|
| rank-2 matmul (dynamic operands) | CPU |
| rank-2 linear+relu (fixed weights) | CPU |
| 3×3 conv, 64 ch, 56×56 | CPU |
| **rank-4 matmul (1,8,512,64)·(1,8,64,512), both dynamic** | **ANE — every op** |
| **1×1 conv, 512 ch** | **ANE** |

## Empirical contracts (not in anyone's documentation)

Facts established by experiment in this repo, each pinned by a test:

- **MPP `convolution2d`** computes cross-correlation with the source window implicitly shifted by
  −k/2 per axis (SAME-centering); `set_offsets(k/2 + tile_origin·stride)` recovers corner-anchored
  VALID windows exactly. Tiling = per-tile dest slices + per-tile offsets. Naive descriptor use is
  silently wrong on every output element.
- **Batching a conv's N into the descriptor serializes batches inside each tile** (measured
  slower than WGSL); batch must ride the dispatch grid with per-batch tensor slices.
- **MTLTensor descriptors must match the backing buffer's storage mode** (wgpu storage buffers
  are Private; Metal validates).
- **Tensor *objects* need residency**, not just their backing buffers; `queue.addResidencySet`
  leaks (32-set cap) — attach per command buffer, re-attached after every begin.
- **wgpu staged uploads flush only on `queue.submit`** — `device.poll` alone never runs them.
  The sync contract is submit **then** poll.
- **wgpu's buffer-init tracker lazily zero-fills "uninitialized" buffers on first wgpu use**,
  clobbering external-queue writes. `clear_buffer` in the flush submit both zeroes and marks
  initialized (GPU-side, free); the resident out-pool makes it once-per-buffer.
- **An open `batch()` is invisible to `queue.submit([])`** — any path handing work to an external
  queue must `flush_batch` first or it reads inputs that haven't been computed yet.
- **fp16 gradient underflow is real**: a full-mean loss puts gradients (~1e-7) below fp16's
  normal range and the tensor units crush them — observed as slower convergence, restored by
  per-sample loss scale. This is *why* the fp16 paths are opt-in.
- **The dispatch floor is the platform's**: an EMPTY MTL4 command buffer round-trips in ~172 µs.
  You cannot shrink it; you can only pay it less often — queue-side event chaining orders
  dependent cbs on the GPU (~35 µs marginal each) with one host wait per flush.
- **Same-config ops cannot share a command buffer** — they share scratch and argument tables, and
  argument tables are read at execution, so the last `setAddress` would win for both dispatches.
  Runs split on config repeats; queue chaining makes the split nearly free.
- **In-flight cbs pin their cache entries**: an LRU eviction while a committed command buffer
  still references an entry's scratch frees memory the GPU is reading. Locks are held across a
  chained flush and eviction happens only after its final wait.

## Doctrine

1. **Measure, don't assume.** Devices are calibrated on the machine they run on; optimizations are
   profiled before they are built (the prset cache and TN/NT backward kernels were both killed by
   measurement before they existed).
2. **Never fake a device.** `dispatchable: false` until a real EP is wired; ANE claims require the
   compute-plan receipt.
3. **Precision changes are opt-in.** fp16-input fast paths (`FERRIC_METAL4`, `FERRIC_COOP`) never
   silently alter default results; every resident path is verified against an fp16-input oracle.
4. **Portable floor first.** Every op has a WGSL baseline that runs everywhere (browser included);
   accelerators are routed above it, never instead of it.

## Known limits / open items

- The ~172 µs platform floor is paid once per op outside `batch()`, once per flush inside it;
  fully hiding it would need async tensor semantics (results promised, waited at readback).
- Quantized resident recipes beyond Q2_0 (Q4_K/Q5_K/Q6_K) pending.
- Conv autograd is first-order only (no double-backward); its backward runs on the portable floor.
- An ANE conv1x1 EP variant is receipt-confirmed but awaits a consumer.
- `vendor/` is machine-local (gitignored): rebuilding elsewhere needs `cargo vendor`. The CoreML
  models regenerate from `scripts/npu-models/` (coremltools MIL + `coremlcompiler`); the EP's
  `.mlmodelc` is embedded in-tree either way.
