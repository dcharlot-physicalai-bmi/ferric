<!-- ferric-accept-issue
track: A
title: Full-model correctness against independent references
-->

# Full-model correctness against independent references

## Mission

Registered architectures agree with an independent reference on tokenization, GGUF import, full logit rows, and generation. Malformed inputs and resource overruns are refused by name. The KV-cache comparison against full recompute stays at max abs 0, which is stricter than the in-tree assert of 2e-3.

## Acceptance

Protocol `docs/acceptance/protocols/A-full-model-correctness.md`, tolerance `A.full-model.v1`.

A closing receipt is `evidence_class: measured`, `tolerance.id` `A.full-model.v1`, and all six stages pass: tokenization, import, logits, generation, malformed, resource_bounds. A quantized file is not the math proof. Greedy identity is not the logit proof. `cargo test -p ferric-accept` accepts that shape and rejects a pass that skips a stage.

## Evidence separation

Targets live in `docs/acceptance/corpus/targets.json` (`target.a.full-model`). Historical author claims, including the 1e-3/1e-4 band and the generate.rs assert, live in `docs/acceptance/corpus/historical.json`. Measured receipts for this track are empty as of 2026-11-06 (`docs/acceptance/corpus/measured.json`). Do not close this issue on a historical sentence.

## Status on 2026-11-09

Issue number 2 is open with this title. This file is the body for `dcharlot-physicalai-bmi` to apply if that issue is edited or reopened. This workstream does not call the issues API and does not open a pull request.

`docs/acceptance/corpus/measured.json` still has `receipts: []`. That is the measured set. A SmolVLA f32 network forward on `cursor/fleet-energy-harness-65da` is not a closing receipt for `A.full-model.v1`. The pinned stats file has no `observation.state` mean or std, so MEAN_STD execution is refused instead of being treated as IDENTITY. Images used on that branch are synthetic protocol inputs. The six closing stages remain tokenization, import, logits, generation, malformed, and resource_bounds. Do not close issue 2 on the fleet forward.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. On 2026-11-09 that list shows issues 2 through 7 open, one per track. Issue 2 is this file. Do not open a second issue for the same protocol.

## Identity

The actor that prepared this body is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE account is not an acceptable actor. This body stays in the tree for dcharlot-physicalai-bmi to apply. It does not link an issue URL.
