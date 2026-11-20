# SmolVLA chunk contract v3 for Ferromotion (bmi-4) and Ferromorphic (bmi-2)

Status: **proposed**. This document is the contract text. It does not change the codec. `crates/ferric-accept/src/smolvla.rs` stays on version 2 until this text is reviewed. No golden for version 3 is invented here.

## What stays

Version 2 stays parseable forever. Magic `FERRIC-SMOLVLA-CHUNK` and a NUL, version `u32` = 2, and the rest of the version 2 layout in `docs/acceptance/contracts/smolvla-ferromotion.md` are unchanged. The two protocol fixtures stay byte-identical, and their digests stay:

- `docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json` → `6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86`
- `docs/acceptance/contracts/fixtures/smolvla-empty-chunk-v2.json` → `6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e`

A reader branches on the version integer after the shared magic. Version 2 takes the version 2 layout. Version 3 takes the layout below. Any other version is refused. A version 2 golden is not a version 3 chunk, and a version 3 chunk is not a version 2 golden.

## Version 3 header

Little-endian, same string rule as version 2: a `u32` byte length, then UTF-8, no trailing NUL. The three provenance hashes stay fixed 64-byte lowercase hex.

| Order | Field |
|---|---|
| 21 | magic `FERRIC-SMOLVLA-CHUNK` and a NUL |
| u32 | version `3` |
| string | LeRobot commit |
| string | model id |
| 64 | weight sha256 |
| 64 | stats sha256 |
| string | stats key, one of `so100`, `so100-blue`, `so100-red` |
| 64 | noise sha256 |
| string | embodiment name |
| u64 | `t0_num` |
| u64 | `t0_den` |
| u64 | `period_num` |
| u64 | `period_den` |
| u32 | `n_action_steps` |
| u32 | `action_dim` |
| u32 | joint count, equal to `action_dim` |
| repeat | joint name, unit, frame, then f32 min, f32 max |
| u8 | `1` for f32 |
| u8 | `1` for time-major, joint order |
| f32… | values, row `t` then joint `0..action_dim` |

`t0` and the period are reduced rationals of nanoseconds: numerator `u64`, denominator `u64`. The denominator is greater than 0. `gcd(num, den) = 1`, computed with the Euclidean algorithm on the two `u64`s. A zero denominator, or a fraction that is not already reduced, is refused. The encoder does not reduce on the caller's behalf.

Every product and sum used to place step `k` is checked `u128`. Overflow refuses the chunk. Step `k` (0-based, `k < n_action_steps`) is the rational

```
t(k) = t0_num / t0_den + k * period_num / period_den
     = (t0_num * period_den + k * period_num * t0_den) / (t0_den * period_den)
```

The digest stores the four reduced inputs, not this sum. A caller who asks for an integer nanosecond gets one only when the denominator divides the numerator exactly. Otherwise that conversion is refused. The rational itself is the time.

For 30 Hz the period is `100000000/3` nanoseconds. `10^9 / 30 = 1000000000/30 = 100000000/3`, and `gcd(100000000, 3) = 1`. An integer `step_duration_ns` cannot say this; that is why version 2's two `u64` clocks are not the version 3 clock.

## Joints

`lerobot/smolvla_base` config at Hub revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` has `output_features.action.shape = [6]` and `normalization_mapping` `VISUAL: IDENTITY`, `STATE: MEAN_STD`, `ACTION: MEAN_STD`.

Both dataset cards record `fps` 30:

- `lerobot/svla_so100_pickplace` `meta/info.json`, `robot_type` `so100`
- `lerobot/svla_so101_pickplace` `meta/info.json`, `robot_type` `so100_follower`

LeRobot `src/lerobot/robots/so_follower/so_follower.py` at `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a` builds the bus in this order. `use_degrees` defaults true (`SOFollowerConfig.use_degrees: bool = True`), so the five arm motors are `MotorNormMode.DEGREES`. The gripper is always `MotorNormMode.RANGE_0_100`.

The version 3 unit tokens are `deg` and `percent_range`. `deg` is the contract spelling of `DEGREES`. `percent_range` is the contract spelling of `RANGE_0_100`. Any other unit string is refused. `deg` on the gripper, or `percent_range` on an arm joint, is refused.

The normative SO-100 names are the `svla_so100_pickplace` action names, in that order, which is the same motor order as the bus:

| Index | Name | Motor id | Unit token | MotorNormMode |
|---|---|---|---|---|
| 0 | `main_shoulder_pan` | 1 `shoulder_pan` | `deg` | `DEGREES` |
| 1 | `main_shoulder_lift` | 2 `shoulder_lift` | `deg` | `DEGREES` |
| 2 | `main_elbow_flex` | 3 `elbow_flex` | `deg` | `DEGREES` |
| 3 | `main_wrist_flex` | 4 `wrist_flex` | `deg` | `DEGREES` |
| 4 | `main_wrist_roll` | 5 `wrist_roll` | `deg` | `DEGREES` |
| 5 | `main_gripper` | 6 `gripper` | `percent_range` | `RANGE_0_100` |

`svla_so101_pickplace` records a different action-name spelling of the same six motors: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`. That spelling is not this table. A mixed list, a reordering, or a `.pos` name in an SO-100 chunk is refused. `action_dim` for this embodiment is 6.

`min` and `max` stay caller limits with `min < max`, both finite, as in version 2. This text does not invent a degree interval.

## Stats key

`policy_postprocessor_step_0_unnormalizer_processor.safetensors` at Hub revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` is 640 bytes. Its sha256 is `490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8`. The same bytes are `policy_preprocessor_step_5_normalizer_processor.safetensors` at that revision. The version 3 stats hash is this file. The stats key names which tensors inside it were used.

Each tensor is six little-endian f32s. The sha256 is of those 24 bytes:

| Key | Tensor | sha256 |
|---|---|---|
| `so100` | `so100.buffer.action.mean` | `e22c3af117ece699b816d50f60df8a1e7e4728e1babf5297634b74b8483ad924` |
| `so100` | `so100.buffer.action.std` | `a09262820c5532e2fffb6da68c7ee6d9670d0da7080f70aadd0b34b7af8755b9` |
| `so100-blue` | `so100-blue.buffer.action.mean` | `8f178ffad9bca313223b2ebf56102779a017dbb0409c557259fa8888e02155fb` |
| `so100-blue` | `so100-blue.buffer.action.std` | `a4762d32295adf3bf7d652ea86cb3e7f2cc575b496e7e30a220a572e927a6820` |
| `so100-red` | `so100-red.buffer.action.mean` | `f85d5119f589c058249515c89ed780b4bba4fdaa30b81277bb738e513b7e6cc8` |
| `so100-red` | `so100-red.buffer.action.std` | `4deb2fcd2af0aa4667f6b09748b5e527e543ca36c13e204b75273e7cc62c8bf6` |

A chunk records one of those three keys. A missing key, a different key, or a tensor sha256 that does not match the row is refused.

## Provenance

The weight pin is `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`, the check at `crates/ferric-smolvla/src/bin/ferric-smolvla.rs:85`. Version 2's sentence "No weight sha256 is in this tree" is stale and is removed there.

`crates/ferric-smolvla/src/bin/ferric-smolvla.rs:301-302` still defaults a missing stats hash to `"11" * 32` and a missing noise hash to `"22" * 32`. Version 3 removes those defaults. A missing stats hash or a missing noise hash is refused. It is not filled in. The weight default on line 300 is already the pin above; stats and noise have no such pin and get none here.

## Digest

`digest_sha256` is SHA-256 of the full canonical byte string as emitted, magic and header included. The hash is not appended to those bytes. Ferromorphic chunk binding 1 calls this value `chunk_sha256`.

## Pin label

The LeRobot sources for the version 2 pins are commit `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a` (2026-07-23). GitHub compare puts that commit on the ancestry of tag `v0.6.1` (`7e241bd630a3719a56157a497ce5d08f244784f1`, 112 commits later) and of tag `v0.7.0` (`410c095eacc0011d998a73383eb94abed2715af0`, 364 commits later). The label for the pin is `a0eb860d (v0.6.1 lineage)`. Tag `v0.7.0` remains the name of `410c095e` only.
