<!-- ferric-accept-issue
track: C
title: Matched-quality workload and system energy leadership
-->

# Matched-quality workload and system energy leadership

## Mission

Prefill, decode, and training each report gross joules and above-idle joules at matched quality. The class is measured or derived. Estimated is not a claim. A saving has two arms that both passed the quality gate. ferrotherm A/B/A calibration agrees before the workload is attributable.

## Acceptance

Protocol `docs/acceptance/protocols/C-energy-leadership.md`, tolerance `C.matched-energy.v1`, schema `docs/acceptance/schemas/fabric-energy.schema.json`. Three measured receipts, one per workload, each naming `quality_gate_receipt_id` and one fabric cell. A cell receipt has `closes_track_c` false. Closing the track also has ferrotherm `a1_a2_agree`, all three workloads, and a latency target whose minimum-energy configuration is the reported point. `ferric-accept` rejects an estimated class, a model estimate (including Apple IOReport as Zeus documents it), a negative above-idle figure, a hardware counter labeled as a wall, a copied ML.ENERGY leaderboard joule, and a missing quality id.

The methodology pin is Zeus `zeus-v0.16.0` (2026-07-07) and the ML.ENERGY Benchmark, NeurIPS 2025 Datasets and Benchmarks, with leaderboard v3 (December 2025). Zeus is an external cross-check, not a Ferric dependency. Leaderboard rows from H100 and B200 are their measurements.

## Evidence separation

`crates/ferric-serve/src/energy.rs` labeling the server figure derived, and `ferric-joule` refusing estimates, are historical contracts of the code. They are not a joule measurement of a model. `scripts/speech_joules_browser.sh` points at a developer-machine tree this workstream does not touch. The ML.ENERGY leaderboard and the Hugging Face dataset `ml-energy/benchmark-v3` are external published results. They are not Ferric receipts. Targets: `target.c.energy-leadership`. Measured set: empty. No fabric cell was metered for this change.

## Status on 2026-11-09

Issue number 4 is open with this title. This file is the body for `dcharlot-physicalai-bmi` to apply if that issue is edited or reopened. This workstream does not call the issues API and does not open a pull request.

`docs/acceptance/corpus/measured.json` still has `receipts: []`. No prefill, decode, or training joule was measured. `gpu_energy` is not claimed for the SmolVLA forward on `cursor/fleet-energy-harness-65da`. The fleet harness refusing to add RAPL package and core, or PPT and package, is a code contract, not a watt measurement. Zeus and the ML.ENERGY leaderboard stay external cross-checks.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. On 2026-11-09 that list shows issues 2 through 7 open, one per track. Issue 4 is this file. Do not open a second issue for the same protocol.

## Identity

The actor that prepared this body is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE account is not an acceptable actor. This body stays in the tree for dcharlot-physicalai-bmi to apply. It does not link an issue URL.
