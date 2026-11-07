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

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. No existing energy-leadership issue.

## Identity

GitHub actor available to this workstream is cursor[bot], not dcharlot-physicalai-bmi. david@openie.dev is the OpenIE run owner and is not an acceptable actor for opening this issue. The body was prepared in-tree and the issue was not opened.
