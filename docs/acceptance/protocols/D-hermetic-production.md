<!-- ferric-accept-protocol
id: D.hermetic-production.v1
track: D
-->

# Track D — hermetic production

## Mission

A production build is rebuildable offline, on a machine that did not author it, from an immutable manifest, and the artifact digest matches. A budget that cannot hold the working set is refused before a token is emitted. After the budget is raised, the bytes match the read taken under the larger budget.

Tolerance id: `D.hermetic.v1`.

## Independent reference

The second machine is the reference. It does not share a target directory, a cargo registry cache, or a vendor directory it created itself. It consumes the manifest the first machine published: source revision, `Cargo.lock` bytes, vendor archive, and `rustc` commit (`rustc -vV`).

In-tree pieces that already exist, and what they do not yet prove:

- `forks/` is patched via `[patch.crates-io]` in the workspace `Cargo.toml`. That is source ownership. It is not an offline rebuild.
- README "Self-reliance" says `cargo vendor` populates `vendor/` and that `vendor/` is not committed. Confirmed on 2026-11-06: `vendor/` is not in the tree.
- Confirmed on 2026-11-06: `Cargo.lock` is not in the tree. A floating resolve is not an immutable manifest. This protocol does not add a lockfile. Adding one would change what `cargo build --workspace` resolves, which is outside this evidence foundation.
- `ferric-tier` refuses a short budget (`BudgetTooSmall`) and tests that bytes do not depend on which tier served them (`bytes_are_identical_whatever_tier_serves_them`, `crates/ferric-tier/tests/placement_invariance.rs`).

## Preregistered tolerances

These bars are preregistered with the protocol.

- `hermetic.offline` is true: the second build is `cargo build --offline --release -p ferric-serve` (and the same for `ferric-cli`) with no network.
- `hermetic.lockfile_sha256` is the sha256 of the `Cargo.lock` bytes both machines used. Without a lockfile the field cannot be filled and the receipt cannot pass.
- `hermetic.rustc_commit` is identical on both machines.
- `hermetic.machines` names two machines. One authoring laptop is not two machines.
- `hermetic.artifact_sha256_agree` is true for the named release binaries.
- `hermetic.bounds_refused` is true: `cargo test -p ferric-tier --lib budget_that_cannot_hold_one_layer_is_refused_up_front` passes, and a streaming read of unstaged bytes is `TierError::NotStaged` rather than zeros (`web/README.md` records that refusal by name).
- `hermetic.recovery_identical` is true: `cargo test -p ferric-tier --test placement_invariance` passes at this revision. The test is the recovery: same logical reads, more than one budget, identical bytes.

A green `cargo test -p ferric-tier` on one machine is the bounds half. It is not the offline-rebuild half.

## Procedure

1. On machine A, from a clean revision, produce the manifest: the git revision, the `Cargo.lock` that was resolved for that revision, the vendor archive from `cargo vendor`, and `rustc -vV`. Hash each. This protocol does not commit that lockfile into this branch.
2. Copy only the manifest and the vendor archive to machine B. Machine B has no cargo git index and no crates.io route. Confirm with a failed fetch or an interface that is down, and record that in the log.
3. On machine B: `cargo build --offline --release -p ferric-serve -p ferric-cli`.
4. sha256 the two `target/release/ferric-serve` binaries and the two `ferric-cli` binaries. Equal digests are the agree bit. A mismatch fails. Do not patch one side.
5. Bounds, either machine, this revision: `cargo test -p ferric-tier --lib budget_that_cannot_hold_one_layer_is_refused_up_front` and `cargo test -p ferric-tier --test placement_invariance`.
6. Recovery is the placement test itself: the small budget and the large budget return the same bytes. Quote the test output in the artifact. Do not summarize it as "ok" without the log.

## Receipt

The checker requires `offline`, a 64-character `lockfile_sha256`, `rustc_commit`, at least two machines, `artifact_sha256_agree`, `bounds_refused`, and `recovery_identical`. A tree that still has no lockfile stays withheld.

## Withhold

Withhold when `Cargo.lock` is absent, when machine B contacted the network, when the compiler commits differ, when only one machine ran, or when the placement test was not executed. Do not describe the forks' presence as a completed hermetic rebuild.

## Downgrade refusal

Offline is not loosened to "the network was available but unused in spirit." A missing lockfile is not loosened to a hash of `Cargo.toml` alone. Two builds on one machine are not loosened into two machines. A budget miss that still returns logits is not a pass. Recovery that changes bytes and calls the change small is a looser bar, and this protocol refuses it.
