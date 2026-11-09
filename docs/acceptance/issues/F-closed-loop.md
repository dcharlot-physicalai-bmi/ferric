<!-- ferric-accept-issue
track: F
title: Closed-loop ferrotherm, ferromorphic, and ferromotion task
-->

# Closed-loop ferrotherm, ferromorphic, and ferromotion task

## Mission

One revision-pinned task, `f.closed-loop.ferromotion-sin3x.v1`, with reference agreement and synchronized gross and above-idle energy. The interfaces are ferrotherm metrology, ferromorphic event-NIR, and ferromotion dynamics control. All three revisions are pinned on a pass. A synthetic ferric-signal series does not count as event-NIR.

## Acceptance

Protocol `docs/acceptance/protocols/F-closed-loop.md`, tolerance `F.closed-loop.v1`. Both arms meet MSE < 5e-3 and max abs versus sin(3x) < 0.15. The probe vectors agree within 1e-3. Energy covers the task window and ferrotherm A/B/A agrees. `ferric-accept` rejects an unpinned revision and `sensor: synthetic`.

## Evidence separation

The MLP assert in `crates/ferric-tensor/examples/ferromotion_mlp.rs` is an in-tree gate and a historical record. It does not include a second implementation or a joule figure. The 1e-3 agreement bar is a preregistered target, not an observed error. ferrotherm and ferromorphic are not crates in this repository. Target: `target.f.closed-loop`. Measured set: empty.

## Status on 2026-11-09

Issue number 7 is open with this title. This file is the body for `dcharlot-physicalai-bmi` to apply if that issue is edited or reopened. This workstream does not call the issues API and does not open a pull request.

`docs/acceptance/corpus/measured.json` still has `receipts: []`. A simulation clock step on `cursor/fleet-energy-harness-65da`, when present, keeps `authorize_rollout` at `ProposedEntryInvoked` and `robot_actuated` false. That gate is not `f.closed-loop.ferromotion-sin3x.v1`. It is not event-NIR, not a ferrotherm window, and not a joule measurement. Synthetic images do not count as a sensor.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. On 2026-11-09 that list shows issues 2 through 7 open, one per track. Issue 7 is this file. Do not open a second issue for the same protocol.

## Identity

The actor that prepared this body is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE account is not an acceptable actor. This body stays in the tree for dcharlot-physicalai-bmi to apply. It does not link an issue URL.
