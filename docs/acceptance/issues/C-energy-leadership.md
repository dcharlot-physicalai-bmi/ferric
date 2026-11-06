<!-- ferric-accept-issue
track: C
title: Matched-quality workload and system energy leadership
-->

# Matched-quality workload and system energy leadership

## Mission

Prefill, decode, and training each report gross joules and above-idle joules at matched quality. The class is measured or derived. Estimated is not a claim. A saving has two arms that both passed the quality gate. ferrotherm A/B/A calibration agrees before the workload is attributable.

## Acceptance

Protocol `docs/acceptance/protocols/C-energy-leadership.md`, tolerance `C.matched-energy.v1`. Three measured receipts, one per workload, each naming `quality_gate_receipt_id`. A leadership close also has ferrotherm `a1_a2_agree`. `ferric-accept` rejects an estimated class, a negative above-idle figure, and a missing quality id.

## Evidence separation

`crates/ferric-serve/src/energy.rs` labeling the server figure derived, and `ferric-joule` refusing estimates, are historical contracts of the code. They are not a joule measurement of a model. `scripts/speech_joules_browser.sh` points at a developer-machine tree this workstream does not touch. Targets: `target.c.energy-leadership`. Measured set: empty.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. No existing energy-leadership issue.

## Identity

GitHub actor available to this workstream is cursor[bot], not dcharlot-physicalai-bmi. david@openie.dev is the OpenIE run owner and is not an acceptable actor for opening this issue. The body was prepared in-tree and the issue was not opened.
