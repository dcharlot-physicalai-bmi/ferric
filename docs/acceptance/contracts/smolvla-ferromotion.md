# SmolVLA contract for Ferromotion (bmi-4)

Status of `ferric.smolvla.predict_action_chunk.v1`: **proposed**. This repository does not run SmolVLA. `predict_action_chunk()` in `crates/ferric-accept/src/smolvla.rs` returns a refusal. The checker tests the pins and the byte layout.

Pin: LeRobot commit `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a` (2026-07-23). Files: `configuration_smolvla.py`, `processor_smolvla.py`, `modeling_smolvla.py`, `processor/normalize_processor.py`, `policies/common/vla_utils.py`. Model id named by the card: `lerobot/smolvla_base`. No weight sha256 is in this tree. Paper: https://arxiv.org/abs/2506.01844 (2 June 2025). Card: https://huggingface.co/lerobot/smolvla_base.

## What Ferric has today

| Surface | Support |
|---|---|
| Full policy, images, language, flow matching, action chunk | Not implemented. Not tested. |
| `crates/ferric-onnx/examples/run_projectors.rs` | An example that reads `testdata/proj`, which is not in the tree. CI records it as manual. It is not this contract. |
| `tiny_llm` / decoder comments that say "SmolVLA-style" | A shape of a language block. Not the VLA. |
| This contract crate | Checks horizon, Aloha-delta refusal, MEAN_STD, state pad, newline, SigLIP range, resize geometry, chunk digest. |

## Call Ferromotion should make later

`predict_action_chunk`, not `select_action`. `select_action` pops one step from a queue and the source asserts that RTC is unsupported there. The chunk call returns the sliced tensor.

## Shape, order, dtype, units, frame, time

- Network noise / pre-slice tensor: `(batch, chunk_size, max_action_dim)` = `(batch, 50, 32)` by default, f32. `sample_actions` allocates that shape. Its docstring says `num_steps`; `num_steps` is the Euler count, default **10**, and is not the horizon.
- After unpad: `(batch, chunk_size, action_dim)` with `action_dim = action_feature.shape[0]`, the dataset width, not 32 unless the dataset is 32.
- Queue for `select_action` keeps `n_action_steps` (default 50). `n_action_steps > chunk_size` is a `ValueError`.
- Canonical bytes for one robot: time-major f32, row `t` then action dimension. Codec version 1. Domain `ferric.smolvla.action-chunk`.
- Unit after unnormalize: dataset-native. Not degrees, not radians. A Hub comment that the range is [-100, 100] is not this formula.
- Frame: embodiment name supplied by the caller (`so100_follower` or otherwise). The model does not convert frames.
- Time: action index `k` is chunk step `k`. `observation_delta_indices` is `[0]` (current frame). Dataset fps is not in `SmolVLAConfig`. The codec tag for time is step, not seconds.

## Preprocess, in code order

Processor `input_steps` (the docstring lists normalize earlier; the list is the pin):

1. Rename observations.
2. Add batch dimension.
3. Task string ends with one newline.
4. Tokenize with `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, padding `longest`, side right, max length 48. Tokenization is not implemented here.
5. Move to device.
6. Normalize. Visual: IDENTITY. State and action: MEAN_STD.

Then inside the model:

7. Image rank 5 uses the last time index. Layout `(batch, channel, height, width)`.
8. `resize_with_pad(img, height=config[1], width=config[0], pad_value=0)`. Default config is `(width, height) = (512, 512)`. Bilinear, `align_corners=False`, pad on the **left and the top**. `prepare_images` says 224; the call does not.
9. `pixel * 2 - 1`. The comment assumes the input was in `[0, 1]`. This contract refuses a pixel outside that interval. The LeRobot call does not.
10. State rank 3 uses the last time index. `pad_vector` to 32 with zeros on the right. A vector already longer than 32 is left unchanged by `pad_vector`. This contract refuses it, because the state projection is `Linear(32, ...)`.

Postprocess: unnormalize, then CPU. Unnormalize is `y * std + mean`. Forward is `(x - mean) / (std + 1e-8)`. The inverse does **not** multiply by `std + eps`.

`use_delta_joint_actions_aloha=True` raises `NotImplementedError` in the config. `adapt_to_pi_aloha` defaults false. Empty image list is a `ValueError`. Missing cameras become zeros inside `prepare_images`; `empty_cameras` defaults to 0 and would add features shaped `(3, 480, 640)`.

## Determinism

If `noise` is omitted, `sample_actions` draws it. That profile is stochastic and cannot be a reference-agreement receipt. A deterministic receipt carries the noise sha256, the weight sha256, and the stats sha256. Flow count stays 10 unless the pinned config on the receipt says otherwise. `load_vlm_weights` defaults false on the dataclass; that is not a statement about `from_pretrained`.

## Digest

Version 1 covers: magic, codec version, LeRobot commit, model id, weight sha256, stats sha256, noise sha256 (empty if stochastic), embodiment name, horizon, action dim, dtype/layout/unit/frame/time tags, and the f32 values. It does not cover wall time, GPU name, or energy.

## Bounds

| Bound | Value | On violation |
|---|---|---|
| `n_action_steps` | ≤ `chunk_size` (50) | refuse |
| state width | ≤ 32 | refuse (stricter than `pad_vector`) |
| action width | 1..=32 | refuse |
| tokens | ≤ 48 | refuse when a length is presented |
| pixels before the SigLIP map | `[0, 1]` | refuse |
| Aloha joint deltas | false | refuse |

No SmolVLA forward was run to produce a fixture. The tests construct chunks and stats in process.
