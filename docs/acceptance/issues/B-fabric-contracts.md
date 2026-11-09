<!-- ferric-accept-issue
track: B
title: One behavioral contract for every target fabric
-->

# One behavioral contract for every target fabric

## Mission

Each required fabric cell has its own receipt: profile, kernel, dtype, shape, driver. Real-browser cells run in a real browser. Lavapipe does not close hardware Vulkan or WebGPU. The NPU cell requires a compute-plan receipt. Metal4 does not close the f32 WGSL metal cell. Cells with no kernel stay required.

## Acceptance

Protocol `docs/acceptance/protocols/B-fabric-contracts.md`, tolerance `B.fabric-cell.v1`. The cell list is `docs/acceptance/corpus/fabric-cells.json` and `REQUIRED_FABRIC_CELLS` in `crates/ferric-accept`. One measured receipt per cell, `fabric.satisfies` equal to that cell only. The issue stays open until every cell has a pass or an explicit withhold that does not delete the cell.

## Evidence separation

The README's 7.15e-7 cross-fabric number, the M5 Max roster, the browser Paris sentence, and the 1e-4 / 5e-2 in-tree asserts are historical author records. They are not new measurements. Targets are `target.b.fabric-cells`. The measured set is empty.

## Status on 2026-11-09

Issue number 3 is open with this title. This file is the body for `dcharlot-physicalai-bmi` to apply if that issue is edited or reopened. This workstream does not call the issues API and does not open a pull request.

`docs/acceptance/corpus/measured.json` still has `receipts: []`. No required fabric cell has a measured pass. A lavapipe or llvmpipe dispatch, including one recorded on `cursor/fleet-energy-harness-65da`, does not close hardware Vulkan, Metal, DX12, or a real-browser WebGPU cell. Cells with no kernel stay required.

## Dedup

On 2026-11-06, `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned []. On 2026-11-09 that list shows issues 2 through 7 open, one per track. Issue 3 is this file. Do not open a second issue for the same protocol.

## Identity

The actor that prepared this body is cursor[bot], not dcharlot-physicalai-bmi. The OpenIE account is not an acceptable actor. This body stays in the tree for dcharlot-physicalai-bmi to apply. It does not link an issue URL.
