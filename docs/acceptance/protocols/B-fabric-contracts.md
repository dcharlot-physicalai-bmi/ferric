<!-- ferric-accept-protocol
id: B.fabric-contracts.v1
track: B
-->

# Track B — one behavioral contract per fabric

## Mission

Every target fabric has one behavioral contract: the same kernel, the same dtype, the same shape, the same profile of numeric flags, on that fabric's own driver or browser. The receipt names those fields and satisfies only that cell. Cells the tree does not implement stay required. A software rasterizer does not close a hardware cell or a browser cell.

Tolerance id: `B.fabric-cell.v1`.

The required cells, and no others, are:

| cell_id | Real browser | What a pass is |
|---|---|---|
| `gpu.wgpu.metal` | no | WGSL kernel versus the CPU reference, bitwise-equal to another WGSL fabric when the contract says so |
| `gpu.wgpu.vulkan.hardware` | no | Same, on a hardware Vulkan adapter |
| `gpu.wgpu.dx12` | no | Same, on a DX12 adapter |
| `gpu.wgpu.webgpu.browser` | yes | Same kernel in a real browser's WebGPU device |
| `gpu.wgpu.vulkan.lavapipe` | no | The software stand-in only. `does_not_satisfy` includes `gpu.wgpu.vulkan.hardware` and `gpu.wgpu.webgpu.browser` |
| `cpu.host` | no | `Device::Cpu` |
| `remote.tcp` | no | `Device::Remote` doing the named op, not a local GPU relabeled |
| `browser.worker` | yes | `Device::BrowserWorker` with the tab actually connected |
| `npu.execution-provider` | no | A compute-plan receipt that the NPU ran the op. No plan, no pass |
| `metal4.tensor-units` | no | fp16 oracle, dtype not f32. Does not satisfy `gpu.wgpu.metal` |
| `cuda.native` | no | The CUDA path in `crates/ferric-tensor/src/cuda.rs`, on an NVIDIA device |
| `rocm.native` | no | A ROCm kernel. Absent in the tree until one exists. Vulkan-on-AMD is the hardware Vulkan cell |
| `webnn.browser` | yes | WebNN bound in a real browser. API missing is a withhold |
| `tenstorrent.host` | no | A compute kernel on the device. Host plumbing is not the kernel |

## Independent reference

For a WGSL f32 cell the reference is the CPU implementation of that kernel in `ferric-core` (`crates/ferric-core/examples/kernels_native.rs` lists add, silu, layernorm, softmax, matmul_bt, attention). For the cross-fabric bitwise claim the reference is the other fabric's output on the same inputs, not the CPU. For `metal4.tensor-units` the reference is the fp16 CPU oracle in `crates/ferric-core/examples/metal4_tensor_gemm.rs`. For `npu.execution-provider` the reference is Apple's `MLComputePlan` (or the equivalent plan on another EP) naming the NPU, as `docs/FABRIC.md` already requires.

`Device` variants are declared in `crates/ferric-tensor/src/sched.rs`: `Gpu`, `Cpu`, `Remote`, `BrowserWorker`, `Npu`, `Metal4`.

## Preregistered tolerances

These bars are preregistered with the protocol.

- Cross-fabric WGSL identity, where the README's contract is bit-identity: the compared f32 outputs are bitwise equal. The historical CPU delta 7.15e-7 is a historical author record (`hist.readme.cross-fabric-number`), not a tolerance to widen toward.
- In-tree kernel gate, which does not replace the bitwise contract: `d < 1e-4` in `kernels_native.rs`, and `diff < 1e-4` in `matmul_native.rs`.
- `metal4.tensor-units`: `err < 5e-2` against the fp16 oracle. A receipt with `dtype` `f32` on this cell is rejected by the checker.
- Profile: every category-A flag in `docs/FLAGS.md` that is set (`FERRIC_COOP`, `FERRIC_COOP16`, `FERRIC_METAL4`, and the others listed there) is named in `fabric.profile`. The default profile is the string `default`, meaning none of those flags are set. `docs/FLAGS.md` records that coop is not bit-identical; a coop profile cannot satisfy a bitwise-default cell.
- Shape and dtype are the concrete values dispatched, not a family name.
- `fabric.satisfies` is exactly `[cell_id]`.

## Procedure

One receipt per cell. Repeat the steps with that cell's command.

1. Clean tree. Record `git rev-parse HEAD`.
2. Select the cell. Set `fabric.profile`, `fabric.kernel`, `fabric.dtype`, `fabric.shape`, `fabric.driver.name`.
3. WGSL f32 cells, default profile: `cargo run -p ferric-core --example kernels_native` and `cargo run -p ferric-core --example matmul_native` on that adapter. Record `ctx.adapter_name`. For the bitwise pair, run the same binary inputs on the second fabric and compare f32 bit patterns.
4. Browser cells: build with `./web/build.sh` from this clone (the script's own root, not a path outside it). Serve with `./web/serve.sh`. Drive a real browser. Record `browser.name`, `browser.real` true, and the adapter string the page reports. Reject the run if that string contains lavapipe, llvmpipe, or swiftshader.
5. Lavapipe cell only: the CI-shaped command is a Vulkan run whose adapter contains `lavapipe`. Do not copy its artifact into another cell's receipt.
6. Metal4: `cargo run -p ferric-core --example metal4_tensor_gemm` on macOS. dtype on the receipt is the fp16 contract.
7. NPU: run the plan probe. `fabric.compute_plan_receipt` is true only when the plan names the NPU for the op. Otherwise withhold.
8. `rocm.native`: if no ROCm kernel is in the tree, withhold. Do not mark the cell passed because Vulkan ran on an AMD GPU.

`gpu.wgpu.vulkan.lavapipe` is the software cell. `crates/ferric-web` is the browser build. `docs/FABRIC.md` is the NPU honesty rule.

## Receipt

`cargo test -p ferric-accept` rejects a browser pass whose adapter contains lavapipe, a lavapipe receipt that omits lavapipe, an NPU pass without `compute_plan_receipt`, a metal4 pass with dtype `f32`, and any `satisfies` list other than the one cell.

## Withhold

Withhold when the device is not in the machine, the browser has no WebGPU or WebNN, the NPU plan names the CPU or GPU, or the kernel is not implemented. An absent cell stays `status: required` in `docs/acceptance/corpus/fabric-cells.json`. Absence is not deleted from the matrix.

## Downgrade refusal

The cell list is not shortened because a backend is missing. Lavapipe is not loosened into a stand-in for hardware Vulkan or for `gpu.wgpu.webgpu.browser`. The fp16 metal4 gate is not loosened into a claim on the f32 metal cell. The bitwise cross-fabric bar is not loosened to 1e-4. A profile that changes numerics is not omitted from the receipt.
