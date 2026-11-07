# Ferric acceptance

This directory is the evidence foundation for six workstreams. It is not a scoreboard, and it does not contain a passing result.

Three kinds of statement are kept apart, on purpose:

| Class | What it is | Where it lives |
|---|---|---|
| Target | The mission bar. Not an observation. | `corpus/targets.json`, `corpus/tolerances.json`, `corpus/fabric-cells.json`, the protocols |
| Historical author | A sentence or an assert already in the tree, cited to a path, a commit, and David Jean Charlot's physicalai-bmi attribution | `corpus/historical.json` |
| Measured | A command that was run, on a clean revision, with artifacts | `corpus/measured.json` |

`corpus/measured.json` has an empty `receipts` array. This workstream ran the checker in `crates/ferric-accept` and did not run a model, a browser, a power meter, an offline rebuild, or the closed-loop task. An empty measured set is the accurate state. Filling it with a placeholder pass is what the checker rejects.

## Tracks

| Track | Protocol | Issue body |
|---|---|---|
| A full-model correctness | `protocols/A-full-model-correctness.md` | `issues/A-full-model-correctness.md` |
| B one contract per fabric | `protocols/B-fabric-contracts.md` | `issues/B-fabric-contracts.md` |
| C matched-quality energy | `protocols/C-energy-leadership.md` | `issues/C-energy-leadership.md` |
| D hermetic production | `protocols/D-hermetic-production.md` | `issues/D-hermetic-production.md` |
| E prior-art leadership matrix | `protocols/E-prior-art-matrix.md` | `issues/E-prior-art-matrix.md` |
| F closed loop | `protocols/F-closed-loop.md` | `issues/F-closed-loop.md` |

The issue bodies are ready to open. They were not opened. See `ACTOR.md`.

## Check

```bash
cargo test -p ferric-accept
cargo run -p ferric-accept --example check
```

The crate has no dependency on the tensor stack. `cargo test --workspace` already runs it, because the workspace members include `crates/*`. No CI workflow file is involved.

Schemas for anything that is not Rust: `schemas/receipt.schema.json`, `schemas/fabric-cell.schema.json`, `schemas/tolerance.schema.json`, and `schemas/fabric-energy.schema.json`. The last one is the per-fabric energy methodology (Zeus v0.16.0 and the ML.ENERGY Benchmark). A schema-valid energy object is not a measurement. The Rust checker is normative for the cross-field rules (a target cannot carry joules, a pass must use the mission tolerance id, a hardware counter cannot claim a wall outlet, lavapipe cannot close a browser cell). `corpus/measured.json` stays empty until a run exists.

## What a later pass looks like

A measured receipt is a new JSON object that `check_receipt` accepts, stored with the run's log, not a sentence added to a protocol. Protocols do not contain `outcome: pass`. Mission tolerances are not widened in the same change as a receipt that needs the wider bar.
