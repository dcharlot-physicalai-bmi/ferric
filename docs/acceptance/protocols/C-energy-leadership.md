<!-- ferric-accept-protocol
id: C.energy-leadership.v1
track: C
-->

# Track C — matched-quality energy

## Mission

Leadership is a saving on prefill, on decode, and on training, each at matched quality. Every figure has a gross joule number and an above-idle joule number. The class is measured or derived, in the sense of `ferric_joule::Class`. An estimate is not a claim. A baseline that fails the quality gate is not a baseline. ferrotherm calibrates the window A/B/A before the workload is attributable.

Tolerance id: `C.matched-energy.v1`.

## Independent reference

The quality reference is track A, same git revision, same weight digest. The energy reference is a sensor:

| Source in tree | Class the code assigns | Boundary |
|---|---|---|
| `ferric_joule::Macmon` | measured at the sensor; the server integral is derived | accelerator rails, and the SoC beside them when the CPU rail is not stuck at 0 (`crates/ferric-serve/src/energy.rs`) |
| `ferric_joule::Rapl` | measured | package domain |
| NVML `power.draw`, as named in the `ferric-joule` crate docs | measured | GPU board |
| TDP times a utilisation guess | estimated | not claimable |

ferrotherm is the external metrology instrument. It is not a type in this repository. macmon does not stand in for it on a leadership pass. `scripts/prefill_joules.sh` is the in-tree prefill procedure and it already calls its integral a derived figure.

`scripts/speech_joules_browser.sh` sets `FERRIC=/Users/dcharlot/vibe-coding/ferric`. Do not run that path. Do not read or edit that directory. A browser energy run uses this clone's `web/build.sh` and `web/serve.sh`.

## Preregistered tolerances

These bars are preregistered with the protocol.

- Quality: the track A receipt for the same revision is `outcome: pass` before either arm's joules may enter a comparison. The receipt id is `quality_gate_receipt_id`.
- Gross and above-idle are both finite numbers on a pass. Above-idle below zero fails. The server already returns null when the window is at or under the idle baseline or inside twice the idle spread (`crates/ferric-serve/src/energy.rs`). Null is a withhold.
- Class `estimated` cannot pass. `Class::claimable` is false for it (`crates/ferric-joule/src/lib.rs`, "Estimates may not.").
- A sample above 1000 W voids the window. That ceiling is `CEILING_W` in `energy.rs` and `PREFILL_JOULES_CEILING_W` in `scripts/prefill_joules.sh`. It is not raised to keep a number.
- Prefill attribution follows `scripts/prefill_joules.sh`: each chunk against the idle gaps on either side; gaps that disagree by more than 25% of the work are refused; each chunk's joules per token lies within 2× of the median chunk.
- ferrotherm A/B/A: calibration A1 with no workload, workload B, calibration A2. A1 and A2 must agree within the instrument's own repeatability, recorded on the receipt as `ferrotherm_aba.a1_a2_agree`. Disagreement withholds B. The two A's are not averaged into a pass.
- A saving is two readings. `ferric_joule` has no `Saving::new` for a hand-built delta. Both arms ran, both passed quality, both boundaries match.
- Three workloads, three receipts: prefill, decode, training. One of them does not cover the others.

The field note in `ferric-joule` about a 3.4% model-FLOPs baseline is historical context for why a broken baseline is not a win. This protocol does not adopt 40% as a new numeric gate. The gate is: the baseline passed the same quality predicate.

## Procedure

1. Clean tree, record the revision and the weight-file sha256.
2. Quality: execute track A for that checkpoint. Keep its receipt id.
3. Prefill: `scripts/prefill_joules.sh <model.gguf> <prompt_tokens>`. The script shells out to `macmon` and `cargo build -p ferric-llama --release --example prefill_joules`. Keep the raw sample log as an artifact.
4. Decode: the same A/B/A structure around a decode example that loads and warms outside the window. `scripts/bonsai2_decode_joules.sh` is the in-tree decode-shaped script when that checkpoint is the one under test; otherwise the same idle-gap rules as prefill, named in the command array.
5. Training: `cargo run -p ferric-tensor --release --example train` (or the example whose assert is the quality gate), meter wrapped around the steps, warmup outside the window. The example's existing assert is copied into the quality receipt from the source at this revision. It is not edited in the commit that reports the joules.
6. ferrotherm, when the instrument's revision is pinned: record A1, B, A2 on its clock and on the Ferric meter's clock. If the revision is not pinned, `status` is `withheld`. Do not copy macmon into the ferrotherm fields.
7. Baseline arm: the named competent system, same prompts, same quality predicate, its own meter boundary written next to Ferric's. Different boundaries do not compare (`Saving::claimable` refuses that in `ferric-joule`).

## Receipt

`energy.class`, `energy.gross_joules`, `energy.above_idle_joules`, `energy.boundary_gross`, `energy.boundary_above_idle`, `quality_gate_receipt_id`. A leadership pass also has `energy.ferrotherm_aba.a1_a2_agree` true. The checker rejects estimated, negative above-idle, and a missing quality id.

## Withhold

Withhold when the meter is absent (the crate returns `None`, not zero), when the idle baseline moved, when the CPU rail reads 0 and a SoC number would hide that, when quality failed, when ferrotherm is unpinned, or when only one of prefill, decode, and training was run and the claim says all three.

## Downgrade refusal

Gross is not loosened into a substitute for above-idle. Above-idle is not loosened into a substitute for gross. Derived is not relabeled measured. Estimated is not relabeled derived. A failed quality gate is not dropped to keep a joule ratio. The 1000 W ceiling is not raised. The developer-machine path in `scripts/speech_joules_browser.sh` is not used, and its absence is not filled with a made-up browser joule number.
