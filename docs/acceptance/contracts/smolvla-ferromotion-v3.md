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
| u64 | `t0_num_ns` |
| u64 | `t0_den` |
| u64 | `period_num_ns` |
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
t(k) = t0_num_ns / t0_den + k * period_num_ns / period_den
     = (t0_num_ns * period_den + k * period_num_ns * t0_den) / (t0_den * period_den)
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

Embodiment token: `so100`. That is `robot_type` in the dataset info below, and it is the stats key `so100` (not `so100-blue` or `so100-red`).

Frame token, every joint: `so100_joint`. `meta/info.json` has no frame field. The token names the `so_follower` joint-command space, not a Cartesian frame. Any other frame string on this embodiment is refused.

Limits are a closed interval, `min < max`, value inside, as in version 2. The gripper limit is the `RANGE_0_100` interval, 0 to 100, not the dataset's observed maximum. The five arm limits are the dataset `action` min and max. Those are observed extrema in `meta/stats.json`, not a mechanical stop in `so_follower.py`.

| Index | min | max |
|---|---|---|
| 0 `main_shoulder_pan` | -37.177734375 | 72.7734375 |
| 1 `main_shoulder_lift` | -89.47265625 | 41.1328125 |
| 2 `main_elbow_flex` | -49.04296875 | 74.619140625 |
| 3 `main_wrist_flex` | 9.66796875 | 96.591796875 |
| 4 `main_wrist_roll` | 56.25 | 123.57421875 |
| 5 `main_gripper` | 0 | 100 |

The dataset's observed gripper action max is 34.93077850341797. A command between that and 100 is still inside the mode interval and is accepted. A command outside 0–100 is refused.

## Dataset meta

`lerobot/svla_so100_pickplace` `meta/info.json` is pinned at Hub revision `728583b5eaf9e739a7f119e2def466fa1d552402` (dataset `sha` on 2026-11-27). The file sha256 is `aec8d0c7bd3d2dc693c160bd58f9fb6aabcbb24d4b28e2f09950a3aa6b246eef`. It records `robot_type` `so100`, `fps` 30, and the six action names above. It does not state units.

`meta/stats.json` at the same revision has sha256 `5f53e159022a2ecf14bb01dd0623fcc498bffa719a776f1ef1858c6098169b89`. The arm min and max in the table above are `action.min` and `action.max` from that file. The units in the joint table are derived, not read from the dataset: `src/lerobot/robots/so_follower/so_follower.py` at `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a`, lines 50–59, with `use_degrees: bool = True` at `config_so_follower.py:42`.

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

Embodiment and dataset pick the key:

| Embodiment token | Dataset | Stats key |
|---|---|---|
| `so100` | `lerobot/svla_so100_pickplace` | `so100` |
| `so100-blue` | a blue SO-100 policy that names that prefix | `so100-blue` |
| `so100-red` | a red SO-100 policy that names that prefix | `so100-red` |

`so100-blue` and `so100-red` are keys in the same safetensors file. They are not `svla_so100_pickplace`. A `so100` chunk that cites `so100-blue` is refused.

The header carries each tensor sha256, not only the file sha256. For the action key that is the two rows above. For STATE, the model safetensors has no `observation.state` tensors. STATE mean and std come from `meta/stats.json` `observation.state` at the revision above, cast to little-endian f32 in joint order. Their sha256 values, of those 24 bytes, are:

| Tensor | sha256 |
|---|---|
| `observation.state` mean | `c0087077ccbf5362ee8dab2ec5cae2ee915123b176ade6c0be1fad3feb7a7eca` |
| `observation.state` std | `09d52284cad4193fdaf1fe35c89dab6aa2711514b3043296b7811c13cb8f13ef` |

MEAN_STD is `src/lerobot/processor/normalize_processor.py` at `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a`, lines 349–362, `eps` default `1e-8` (line 97), and the saved preprocessor and postprocessor configs both set `"eps": 1e-08`.

Forward, STATE into the network: `(x - mean) / (std + 1e-8)`.

Inverse, ACTION out of the network: `y * std + mean`. The inverse does not add epsilon. A missing mean or std for a MEAN_STD feature is refused. It is not IDENTITY.

## Noise

`noise_sha256` is SHA-256 of this byte string, and of nothing smaller:

1. UTF-8 `f32`.
2. `u32` little-endian rank `3`.
3. Three `u64` little-endian extents: `1`, `50`, `32`. That is `x_T` shaped `(batch, chunk_size, max_action_dim)`.
4. UTF-8 `le`.
5. The `50 * 32` little-endian f32 payload, time-major then joint.
6. The `u64` little-endian seed.
7. UTF-8 `xorshift64-box-muller-v1`.

The payload is `noise_from_seed` in `crates/ferric-smolvla/src/lib.rs`: state starts at `seed | 1`; each step is `x ^= x << 13; x ^= x >> 7; x ^= x << 17`; the unit is `(x >> 11) as f64 / 2^53`; pairs go through Box–Muller and are cast to f32. A missing noise hash is refused. The placeholders `"11" * 32` and `"22" * 32` are not values of this hash.

## Interop with ferromorphic-chunk-binding 1

The binding's chunk fields, and where they sit in this canonical layout:

| Binding field | Ferric field |
|---|---|
| `chunk_sha256` | `digest_sha256`, SHA-256 of the full canonical bytes, magic and header included |
| `t0_num_ns` | `t0_num_ns` |
| `t0_den` | `t0_den` |
| `period_num_ns` | `period_num_ns` |
| `period_den` | `period_den` |
| `n_action_steps` | `n_action_steps` |
| `envelope_sha256` | not a byte of the chunk. The envelope that cites this chunk carries it. |
| `epoch` | not a byte of the chunk. The envelope carries it. |
| `chunk_index` | not a byte of the chunk. The envelope carries it, and it is strictly increasing inside one episode. |

The chunk period `period_num_ns / period_den` must equal the envelope's `period_num_ns / period_den` exactly, both rationals already reduced. A chunk whose period is a different reduced fraction is refused.

`t0` must equal `origin_ns + k · period` for some integer `k`, with `origin_ns` and `period` taken from the episode below. The sum is the checked `u128` arithmetic above. A `t0` that is not on that grid is refused.

## ferromorphic-episode 2

Fields, in this order:

1. `version`
2. `epoch`
3. `dt_num_ns`
4. `dt_den`
5. `dt_bits`
6. `period_num_ns`
7. `period_den`
8. `origin_ns`
9. `episode_nonce`

For this SO-100 contract the only legal clock is 30 Hz with 20 substeps. `period_num_ns / period_den` is `100000000/3`. `dt_num_ns / dt_den` is `5000000/3`, which is that period divided by 20. Both fractions are already reduced (`gcd` with 3 is 1). Any other period or dt is refused.

## Provenance

The weight pin is `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`, the check at `crates/ferric-smolvla/src/bin/ferric-smolvla.rs:85`. Version 2's sentence "No weight sha256 is in this tree" is stale and is removed there.

`crates/ferric-smolvla/src/bin/ferric-smolvla.rs:301-302` still defaults a missing stats hash to `"11" * 32` and a missing noise hash to `"22" * 32`. Version 3 removes those defaults. A missing stats hash or a missing noise hash is refused. It is not filled in. The weight default on line 300 is already the pin above; stats and noise have no such pin and get none here.

## Digest

`digest_sha256` is SHA-256 of the full canonical byte string as emitted, magic and header included. The hash is not appended to those bytes. Ferromorphic chunk binding 1 calls this value `chunk_sha256`.

## Pin label

The LeRobot sources for the version 2 pins are commit `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a` (2026-07-23). GitHub compare puts that commit on the ancestry of tag `v0.6.1` (`7e241bd630a3719a56157a497ce5d08f244784f1`, 112 commits later) and of tag `v0.7.0` (`410c095eacc0011d998a73383eb94abed2715af0`, 364 commits later). The label for the pin is `a0eb860d (v0.6.1 lineage)`. Tag `v0.7.0` remains the name of `410c095e` only.
