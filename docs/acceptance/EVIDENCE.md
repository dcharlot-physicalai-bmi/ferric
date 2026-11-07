# How to read a statement in this repository

A number in Ferric has a class. Mixing the classes is how a target becomes a claimed result, and how an old sentence becomes a new measurement. This foundation keeps them in separate files.

## Measured

A measured receipt was produced by running the protocol's command on a clean git revision. It carries:

- `git.revision` (40 hex characters) and `git.dirty: false`
- the command array
- the machine's OS and adapter
- an independent comparator
- the preregistered tolerance id for that track
- sha256 of the raw log

`outcome: pass` is legal only here, and only with the mission tolerance id (`A.full-model.v1`, `B.fabric-cell.v1`, `C.matched-energy.v1`, `D.hermetic.v1`, `E.leadership-matrix.v1`, `F.closed-loop.v1`). An in-tree assert that is looser (KV cache at 2e-3, matmul at 1e-4, Metal4 at 5e-2) can be cited as history. It cannot be substituted for the mission id.

Energy follows `ferric_joule::Class`. `measured` and `derived` can carry a claim. `estimated` cannot. Gross joules and above-idle joules are different fields. A null from `crates/ferric-serve/src/energy.rs` is unresolved, not zero.

## Target

A target is the bar the project is held to. `not_a_measurement` is true. `outcome` is `not_applicable`. Observation fields (`joules`, `max_abs_error`, `wer`, `tokens_per_second`, `observed`) are rejected on a target object.

The mission is not reduced to whatever the tree already checks. Track A's KV bar stays max abs 0. Track B keeps ROCm and the browser as required cells. Track C keeps prefill, decode, and training. Track F keeps all three interfaces.

The 1e-3 cross-implementation bar on task `f.closed-loop.ferromotion-sin3x.v1` is a target registered in `corpus/tolerances.json`. It is not an error anyone has recorded for that pair of runs.

## Historical author

A historical record quotes a file that is already in the tree. The checker re-reads the file and requires the excerpt to be present. The git commit, date, and author email are the ones `git log` reported for that file on 2026-11-06. The author is David Jean Charlot, attributed as `david@physicalai-bmi.org` or the `dcharlot-physicalai-bmi` noreply address. `not_a_new_measurement` is true.

Kinds:

- `prose` — a sentence in a doc or a comment
- `in_tree_gate` — an assert or a refusal the code already performs
- `audit` — `docs/parity/MATRIX.md`, dated 27 September 2026

Quoting the audit's "verified" label does not make a new measurement. The label stays in the historical file.

## What this workstream measured

The checker tests in `crates/ferric-accept` were run in order to see whether the corpus matches the rules. That is a check of documents. It is not a logit error, a joule figure, a browser transcript, or a rebuild digest. No such numbers were produced here, and none are invented in the corpus. Published ML.ENERGY leaderboard rows are that project's measurements on their hardware. They are not rows in `corpus/measured.json`.
