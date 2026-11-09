<!-- ferric-accept-issue
track: E
title: Dated owner-attributed prior-art leadership matrix
-->

# Dated owner-attributed prior-art leadership matrix

## Mission

Prove integrated leadership with a matrix that has a date, an owner on the physicalai-bmi attribution, a pinned peer revision per cell, a pinned Ferric revision, and a measured receipt per leadership cell. The goal in `docs/STRATEGY.md` stays the mission. The 27 September 2026 parity audit stays an input.

## Acceptance

Protocol `docs/acceptance/protocols/E-prior-art-matrix.md`, tolerance `E.leadership-matrix.v1`. The minimum candidate set is O01 energy, cross-vendor WGSL identity, S36 batch API, F13 ternary, E27 determinism, and F02 i-quants. `matrix.leadership_cells_open` is 0 only when each of those has a measured receipt. Dropping a candidate is a looser matrix and does not close the issue.

## Evidence separation

`docs/parity/MATRIX.md` is historical_author, owner David Jean Charlot, commit `d36f756`. Sentences in that file are not re-measurements. Target: `target.e.leadership-matrix`. Measured set: empty.

## Status on 2026-11-09

Issue number 6 is open with this title. This file is the body for `dcharlot-physicalai-bmi` to apply if that issue is edited or reopened. This workstream does not call the issues API and does not open a pull request.

`docs/acceptance/corpus/measured.json` still has `receipts: []`. No leadership cell has a new measured receipt. `docs/parity/MATRIX.md` stays `historical_author`. `matrix.leadership_cells_open` is not zero.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. On 2026-11-09 that list shows issues 2 through 7 open, one per track. Issue 6 is this file. Do not open a second issue for the same protocol.

## Identity

The actor that prepared this body is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE account is not an acceptable actor. This body stays in the tree for dcharlot-physicalai-bmi to apply. It does not link an issue URL.
