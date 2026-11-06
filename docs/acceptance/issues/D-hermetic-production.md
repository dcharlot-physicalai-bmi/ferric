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

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. No existing hermetic-build issue.

## Identity

GitHub actor available to this workstream is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE address david@openie.dev is not an acceptable actor. This body was prepared in-tree and the issue was not opened.
