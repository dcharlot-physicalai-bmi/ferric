# SmolVLA reference delta, LeRobot v0.7.0

Primary sources, checked 2026-11-09:

- Release `v0.7.0` is commit `410c095eacc0011d998a73383eb94abed2715af0`, published 2026-11-05T17:13:20Z. The notes include PR #4292.
- PR #4292, merge `c1789b1306ca451cdc61c4576c3eba691bd9b974` (2026-10-21), adds two lines to `prepare_images`: if the image is `uint8`, cast to float32 and divide by 255 before resize. The shared resize helper is unchanged. The same two lines are in the tag.
- arXiv:2609.00363 is "Deterministic LLM Inference Across GPU Kernels". It is not a SmolVLA or LeRobot paper, and it does not describe this preprocessing change.

`lerobot/smolvla_base` Hub HEAD on this date is still `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` (card `lastModified` 2026-10-06). v0.7.0 does not pin a weight revision; `from_pretrained("lerobot/smolvla_base")` follows that Hub HEAD. The local `model.safetensors` measures `906712520` bytes and sha256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`, which matches the expected pin.

The pinned inputs are f32 pixels in `[0, 1]`. On those tensors the uint8 branch is not taken. All three prepared images are bitwise identical between the a0eb860d path and the v0.7.0 path (`max_abs` 0, `rmse` 0, `n` 786432). One remeasured f32 forward matches `smolvla-f32-reference.json` on every stage. Later-stage deltas are 0 because that input is bitwise identical. The graph was not run twice.

A uint8 view of camera 0, which is not the pinned protocol, is a different measured result: `max_abs` 506.5058898925781, `rmse` 287.66815185546875. The old path leaves the 0..255 values in the resize and the following `* 2 - 1` reaches 507.5. The v0.7.0 path stays in `[-1, 0.9941176176071167]`.

Ferric's default `prepare_image` accepts only f32 in `[0, 1]`, resizes, then maps to `[-1, 1]`. That matches both LeRobot revisions for the pinned float inputs. It does not accept uint8, so it matches neither the old uint8 path nor a silent switch onto the new one. `prepare_uint8_v070` is the named v0.7.0 path: divide by 255, then the existing resize. `forward` does not call it.
