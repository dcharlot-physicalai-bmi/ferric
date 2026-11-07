# Fleet commands

The worker runs these. This agent did not run them on a Mac or on the Framework.

```bash
cargo run -p ferric-fleet -- plan
cargo run -p ferric-fleet -- acquire --dir "$FLEET_ARTIFACTS"
cargo run -p ferric-fleet -- preflight --adapter-name "$WGPU_ADAPTER_NAME"
cargo run -p ferric-fleet -- check-bytes "$FLEET_ARTIFACTS/config.json" config.json
```

`acquire` downloads the six pinned files from Hugging Face revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and checks size and sha256. `model.safetensors` is 906712520 bytes. Its pin is the LFS oid `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`. That hash was checked against the full file on the VM that built `ferric-smolvla`.

The model command is separate from this harness:

```bash
cargo run -p ferric-smolvla --release -- infer \
  --weights "$FLEET_ARTIFACTS/model.safetensors" \
  --tokenizer /path/to/SmolVLM2-tokenizer.json \
  --stats "$FLEET_ARTIFACTS/policy_preprocessor_step_5_normalizer_processor.safetensors" \
  --joints /path/to/joints.json \
  --action-stats-key so100.buffer.action
```

`infer` prints one JSON object. `forward_backend` is `ferric_core::cpu f32`. `wgpu_adapter` is the adapter `Context::new` opened. `wgpu_software_adapter` is true when that name is a software rasterizer. A machine that exposes no adapter records `wgpu_adapter` `none` and `wgpu_adapter_present` false; the action chunk is still the CPU forward. `gpu_energy` is false on every path. `robot_actuated` is false. The joint command is action row 0 after `bind_clock`. `ferric-accept` `authorize_rollout` is still `ProposedEntryInvoked` and this command does not call it.

`preflight` refuses an adapter name containing `llvmpipe`, `lavapipe`, or `swiftshader`. It does not open a device. The worker passes the name wgpu reports. RADV on the Framework is eligible. llvmpipe on that same machine is not.

Measurement windows stay separate: `load`, `compile`, `warmup`, `steady-infer`, `joint-step`. Around each window the worker drains wgpu the way `Context::flush` does: `queue.submit` of an empty batch, then `device.poll(wait_indefinitely)`. Steady inference and the joint step also read results back after the second drain.

RAPL package and core stay separate. ROCm PPT is not added to the package. macmon is not wall energy. A DRAM field that is all zeros is unavailable, not a measurement of zero. A busy Mac is not an attributable baseline. Five seconds is not a default duration.

`model_receipts` is 0. `joint_step_executed` is false.
