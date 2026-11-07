# Fleet commands

The worker runs these. This agent did not run them on a Mac or on the Framework.

```bash
cargo run -p ferric-fleet -- plan
cargo run -p ferric-fleet -- acquire --dir "$FLEET_ARTIFACTS"
cargo run -p ferric-fleet -- preflight --adapter-name "$WGPU_ADAPTER_NAME"
cargo run -p ferric-fleet -- check-bytes "$FLEET_ARTIFACTS/config.json" config.json
```

`acquire` downloads the six pinned files from Hugging Face revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and checks size and sha256. `model.safetensors` is 906712520 bytes. Its pin is the LFS oid. The full file was not read when the pin was written.

`preflight` refuses an adapter name containing `llvmpipe`, `lavapipe`, or `swiftshader`. It does not open a device. The worker passes the name wgpu reports. RADV on the Framework is eligible. llvmpipe on that same machine is not.

Measurement windows stay separate: `load`, `compile`, `warmup`, `steady-infer`, `joint-step`. Around each window the worker drains wgpu the way `Context::flush` does: `queue.submit` of an empty batch, then `device.poll(wait_indefinitely)`. Steady inference and the joint step also read results back after the second drain.

RAPL package and core stay separate. ROCm PPT is not added to the package. macmon is not wall energy. A DRAM field that is all zeros is unavailable, not a measurement of zero. A busy Mac is not an attributable baseline. Five seconds is not a default duration.

`model_receipts` is 0. `joint_step_executed` is false.
