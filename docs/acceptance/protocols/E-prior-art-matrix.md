<!-- ferric-accept-protocol
id: E.prior-art-matrix.v1
track: E
-->

# Track E — dated, owner-attributed prior art

## Mission

Integrated leadership is a matrix a reader can date and attribute. Every leadership cell names the peer's revision, Ferric's revision, the owner, and a measured receipt. The mission in `docs/STRATEGY.md` is the dominant Rust framework for all of AI. That sentence stays the mission. The September 2026 parity audit does not discharge it.

Tolerance id: `E.leadership-matrix.v1`.

## Independent reference

Peers are the ones `docs/parity/MATRIX.md` already audited, at the commits that audit pinned, plus any peer added later at a commit recorded in the cell. Ferric's side is a revision of `dcharlot-physicalai-bmi/ferric`. The owner email is the physicalai-bmi attribution used in git (`david@physicalai-bmi.org` or the `dcharlot-physicalai-bmi` noreply). An OpenIE address is not an owner for this matrix.

`docs/parity/MATRIX.md` is dated 27 September 2026 in its heading and last touched in commit `d36f756` by David Jean Charlot `<david@physicalai-bmi.org>`. It is `evidence_class: historical_author` (`hist.parity.energy-peers`, `hist.parity.date`). Cells it marks Ferric-verified where peers are thin — energy per request (O01, 0/14), the batch API, ternary, determinism, i-quants — are candidates for leadership cells. Each candidate still needs a measured receipt before it counts as proved.

## Preregistered tolerances

These bars are preregistered with the protocol.

- `matrix.date` is a calendar date on the receipt, the date the matrix was closed, not the audit's date copied forward.
- `matrix.owner_email` passes `author_is_physicalai` in `ferric-accept`. `openie` in the email fails the receipt.
- `matrix.leadership_cells_open` is 0 on a pass. A cell without a measured receipt id is open.
- `matrix.peer_pins_sha256` is the sha256 of the file that lists each peer repository and commit. The file is an artifact of the receipt.
- A cell whose only citation is `docs/parity/MATRIX.md` stays historical. The checker does not have a path that turns that file into `outcome: pass`.
- Leadership is the integrated claim: the cells that the audit says peers do not hold, demonstrated together on one Ferric revision, not one feature in isolation quoted from 2026. The protocol does not drop a cell from that set to make the remainder look closed.

The candidate set, taken from the "Where Ferric stands that peers do not" table and the determinism paragraph of that matrix, is the minimum leadership set:

- O01 energy per request or per token
- cross-vendor bitwise identity of one WGSL kernel (track B)
- S36 batch API
- F13 ternary / 1-bit
- E27 deterministic / batch-invariant inference
- F02 GGUF i-quants

Removing one of these from the set is a looser matrix. Adding peers is allowed. Reclassifying a candidate as historical without a measured receipt is required, not optional.

## Procedure

1. Copy the candidate list above into the matrix artifact. Date it. Sign it with the owner email.
2. For each candidate, attach the measured receipt id from the track that owns the measurement (C for O01, B for cross-vendor identity, and the existing conformance scripts for S36, F13, E27, F02 once they are run at this revision and written up as receipts). A script that exists is not yet that receipt.
3. For each peer that the 2026 audit said lacks the feature, pin the commit the statement was checked against. If the peer has since shipped it, the cell records that and stops claiming absence. A stale "0/14" is not leadership.
4. `cargo run -p ferric-accept --example check` on the corpus, then a separate measured receipt with `tolerance.id` `E.leadership-matrix.v1` once every candidate is closed.
5. Keep `docs/acceptance/corpus/historical.json` as the place the audit's sentences live. Do not paste those sentences into the measured receipt as if they were re-run.

## Receipt

Pass requires `matrix.date`, a physicalai-bmi `matrix.owner_email`, `matrix.leadership_cells_open` = 0, and `matrix.peer_pins_sha256`. Anything short of that is withheld.

## Withhold

Withhold while any candidate lacks a measured receipt, while a peer pin is a branch name rather than a commit, or while the owner email is not the physicalai-bmi attribution. The historical ledger is the honest state until then.

## Downgrade refusal

The mission sentence is not rewritten to the subset already marked verified in 2026. A candidate is not dropped because its measurement is missing. The audit date is not loosened into the matrix date. Peer absence is not loosened from a pinned commit to a recollection. An OpenIE identity is not an acceptable owner.
