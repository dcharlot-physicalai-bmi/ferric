# Fleet commands

The worker runs these. This agent did not run them on a Mac or on the Framework.

```bash
cargo run -p ferric-fleet -- plan
cargo run -p ferric-fleet -- acquire --dir "$FLEET_ARTIFACTS"
cargo run -p ferric-fleet -- preflight --adapter-name "$WGPU_ADAPTER_NAME"
cargo run -p ferric-fleet -- check-bytes "$FLEET_ARTIFACTS/config.json" config.json
```

`acquire` downloads the six pinned files from Hugging Face revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and checks size and sha256. `model.safetensors` is 906712520 bytes. Its pin is the LFS oid `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`. That hash was checked against the full file on the VM that built `ferric-smolvla`.

The model commands are separate from this harness. `infer` is an execution command. It refuses unless `--embodiment-map` names `observation.state` and `action` stats keys that exist in the stats file. The pinned file's prefixes are `so100.buffer.action`, `so100-blue.buffer.action`, and `so100-red.buffer.action`. There is no `observation.state` key. MEAN_STD does not become IDENTITY. `so100.buffer.action` is not assumed.

```bash
cargo run -p ferric-smolvla --release -- infer \
  --weights "$FLEET_ARTIFACTS/model.safetensors" \
  --tokenizer crates/ferric-smolvla/tests/fixtures/tokenizer.json \
  --stats "$FLEET_ARTIFACTS/policy_preprocessor_step_5_normalizer_processor.safetensors" \
  --embodiment-map /path/to/embodiment-map.json

cargo run -p ferric-smolvla --release -- network-forward \
  --weights "$FLEET_ARTIFACTS/model.safetensors" \
  --tokenizer crates/ferric-smolvla/tests/fixtures/tokenizer.json \
  --stats "$FLEET_ARTIFACTS/policy_preprocessor_step_5_normalizer_processor.safetensors" \
  --trace-dir /tmp/smolvla-rust

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo run -p ferric-smolvla --release -- gpu \
  --weights "$FLEET_ARTIFACTS/model.safetensors" \
  --tokenizer crates/ferric-smolvla/tests/fixtures/tokenizer.json \
  --stats "$FLEET_ARTIFACTS/policy_preprocessor_step_5_normalizer_processor.safetensors"

cargo run -p ferric-smolvla --release -- simulate \
  --actions /tmp/smolvla-rust/actions6.f32 \
  --joints crates/ferric-smolvla/tests/fixtures/joints.json
```

`network-forward` is the non-executable f32 tensor path. Its images are `input_class: synthetic_protocol`. `gpu` dispatches that graph through wgpu and reads it back. A lavapipe adapter is software. `gpu_energy` is false. `simulate` is `simulation-validation`. It calls `authorize_rollout`, records `ProposedEntryInvoked`, and does not actuate a robot.

`docs/fleet/smolvla-cpu-infer.json` is an earlier run that unnormalized with the unpinned key `so100.buffer.action` and padded state without mean/std. It is not an execution receipt.

The quiet driver and the f32 bound are `docs/fleet/ENERGY-DRIVE.md` and `docs/fleet/TOLERANCE.md`. The RADV gfx1151 file `docs/fleet/smolvla-wgpu-radv-gfx1151-energy.json` stays `attributable: false`.

`preflight` refuses an adapter name containing `llvmpipe`, `lavapipe`, or `swiftshader`. It does not open a device. The worker passes the name wgpu reports. RADV on the Framework is eligible. llvmpipe on that same machine is not.

Measurement windows stay separate: `load`, `compile`, `warmup`, `steady-infer`, `joint-step`. Around each window the worker drains wgpu the way `Context::flush` does: `queue.submit` of an empty batch, then `device.poll(wait_indefinitely)`. Steady inference and the joint step also read results back after the second drain.

RAPL package and core stay separate. ROCm PPT is not added to the package. macmon is not wall energy. A DRAM field that is all zeros is unavailable, not a measurement of zero. A busy Mac is not an attributable baseline. Five seconds is not a default duration.

`model_receipts` is 0. `joint_step_executed` is false.
