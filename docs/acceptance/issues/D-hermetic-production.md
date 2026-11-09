<!-- ferric-accept-issue
track: D
title: Hermetic production, offline rebuilds, bounds, and recovery
-->

# Hermetic production, offline rebuilds, bounds, and recovery

## Mission

Two independent machines rebuild offline from one immutable manifest, including a Cargo.lock digest and one rustc commit, and the release artifact digests match. A budget that cannot hold the working set is refused before tokens. Recovery reads the same bytes at a larger budget.

## Acceptance

Protocol `docs/acceptance/protocols/D-hermetic-production.md`, tolerance `D.hermetic.v1`. The measured receipt needs `hermetic.lockfile_sha256`, two machines, `artifact_sha256_agree`, `bounds_refused`, and `recovery_identical`. As of 2026-11-06 the tree has no Cargo.lock and no committed vendor directory, so a pass is blocked until a manifest exists. This issue does not ask this branch to add the lockfile.

## Evidence separation

Forks under `forks/` and the README's vendor instructions are historical description. `ferric-tier` tests are the bounds procedure, not a completed two-machine rebuild. Target: `target.d.hermetic`. Measured set: empty.

## Status on 2026-11-09

Issue number 5 is open with this title. This file is the body for `dcharlot-physicalai-bmi` to apply if that issue is edited or reopened. This workstream does not call the issues API and does not open a pull request.

`docs/acceptance/corpus/measured.json` still has `receipts: []`. `Cargo.lock` is still gitignored. No two-machine offline rebuild was run, and no release artifact digests were compared. This issue still does not ask for a lockfile to be added on this branch.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. On 2026-11-09 that list shows issues 2 through 7 open, one per track. Issue 5 is this file. Do not open a second issue for the same protocol.

## Identity

The actor that prepared this body is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE account is not an acceptable actor. This body stays in the tree for dcharlot-physicalai-bmi to apply. It does not link an issue URL.
