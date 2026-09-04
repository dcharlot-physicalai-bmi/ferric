# What is verified in Ferric's low-bit and attention paths, and what is not

Generated 2026-09-04 from the tree, not from memory. Counts are live:
**12 Kani harnesses**, **248 tests** (51 `ferric-gguf`, 83 `ferric-tensor`, 114 `ferric-llama`),
wasm32 clean on all three.

This document exists because "verified" is not one thing. A bounded model check, an exhaustive GPU
differential, a probabilistic proof over a finite field, a derived rounding bound and a measured
amplification factor make five different claims of five different strengths, and a reader deciding
whether to trust a number needs to know which one they are holding. Every row below says what the
claim is, and the last section says plainly what nothing here covers.

---

## 1. Bounded model checking — Kani 0.67, all inputs in range

Run by `scripts/proofs.sh` (needs `-Z stubbing`; per-crate timeout; the verdict is read off the
`Complete - N … 0 failures` summary line, because piping through `grep` matched that line whether it
said 0 failures or 2). CI job `proofs`, pinned to 0.67.0 — **first ran in CI on 2026-09-04** (run 33906791929), and
the log was read rather than the green tick trusted: 12 `VERIFICATION:- SUCCESSFUL` lines and two
`Complete - N … 0 failures` summaries (7 gguf + 5 llama), matching the local harness count exactly.
A job that skips silently and a job that verifies produce the same coloured tick.

| harness | claim |
|---|---|
| `codebook_is_the_thirty_two_patterns` | the STQ1_0 codebook is exactly the 32 three-of-four ternary patterns, distinct, sign half mirrored |
| `only_three_of_four_groups_are_encodable` | pack↔decode is inverse on every legal group; every illegal one is refused, not silently rounded |
| `decoder_places_every_group_where_the_encoder_says` | for all 64 groups × 32 codes, the real decoder's positions match the real encoder map |
| `vec4_traversal_addresses_every_lane_where_the_decoder_does` | the vec4 shader's addressing, mirrored in Rust, agrees with the decoder for all 4×4×4×4 positions |
| `ksigns_is_an_injective_even_parity_code` | the IQ sign table is a parity code, which is why it is computed rather than carried |
| `iq2_subscale_is_in_the_sign_word` | IQ2_XXS reads its sub-scale from the word it shares with the sign indices, not the index word |
| `iq3_control_words_live_after_the_index_bytes` | IQ3_XXS's two block halves are not interleaved |
| `every_layer_reuses_a_real_preceding_selection` | DSA: causality, and every source is itself a full layer |
| `the_source_is_the_most_recent_full_layer` | nothing full sits between a layer and its source — layer 2 takes layer 1, not layer 0 |
| `sources_are_monotone` | the index cache only ever advances |
| `a_non_full_first_layer_is_refused` | a checkpoint with nothing to reuse is refused at construction |
| `live_cache_count_is_the_number_of_self_sourcing_layers` | allocation equals what is written |

The DSA harnesses hold for **every `is_full` pattern up to 12 layers**, not just the shipped one.

**Stubbed, and therefore excluded:** `rd_f16`. `half` reaches runtime CPU-feature detection whose C
string literals Kani cannot encode. The stub is position-sensitive (1.0 for the bit pattern of 1.0,
0.0 otherwise) so a wrong offset still fails — a constant-returning stub let "scale read from the
front of the block" pass. **f16 arithmetic is not verified; which bytes are read is.**

---

## 2. Exact algebra — GF(2⁶¹−1), Schwartz–Zippel

`ferric-llama/src/exact.rs`. The hyper-connection closed form and both MLA absorption folds are
polynomial identities once gates and attention weights are treated as given coefficients — which
they are; they hold for *any* gate, sigmoid or not. Coefficients are ±1, so an integer identity
holds over ℚ exactly when it holds over GF(p).

One exact agreement at a random point bounds "the identity is false" at **1.3e-18**; eight
independent trials put it below **1e-140**. Exact `==`, no tolerance, and no dependency — Ferric
builds only from `vendor/`, and u64 residues mod a Mersenne prime never overflow a u128 product.

---

## 3. Exhaustive GPU differentials

| test | what it sweeps |
|---|---|
| `every_group_position_reaches_the_kernel` | 64 groups × 32 codes = 2048 rows, one distinct group each, all three traversal forms, on the real GPU |
| `every_codebook_pattern_reaches_the_kernel` | all 32 codebook patterns driven deliberately |
| `packed_matmul_matches_dequant_then_matmul` | packed kernel against a decoder verified on Tencent's published weights |
| `packed_iq_matmuls_match_dequant_then_matmul` | same, IQ2_XXS and IQ3_XXS |
| `stq1_0_interop` (example) | Ferric's decode against the *same weights published at a second quant level* — cos 0.8416, 99.3% sign agreement, with both wrong-layout controls at chance |
| `hyv4_real_moe` (example) | Tencent's **block 1 entire** — real attention *and* real experts — run from **both published builds** and compared: IQ2_XXS/IQ3_XXS against Q4_K/Q6_K, cos **0.96671**, against a measured no-routed-path floor of 0.29180 |
| `hyv4_real_block` (example) | block 0's real attention, hyper-connection and indexer weights, with both wrong-orientation controls |

The cross-quant check is the strongest thing available without a reference implementation, and
its shape is worth stating plainly: the two builds are the *same trained tensors* rounded by two
quantisers into formats decoded by two different code paths, so neither arm can borrow the other's
bug. What it therefore reaches is exactly what the builds do not share — **their formats**. A bug
applied to both arms is invisible to it by construction (expert order reversed in the loader, the
routed scale applied before the renormalisation rather than after); those stay with the synthetic
golden hash.

⚠ **Ordering the arms was not enough, and mutation is what showed it.** Gating on
`real > controls` survives a broken decoder, because a shader bug moves every arm of that build
together: three of four IQ2/IQ3 shader mutations kept the ordering while the real cosine fell to
0.832, 0.341 and 0.686. The gate is now an absolute floor placed between the pristine 0.96671 and
the closest mutation at 0.83199 — from the measured ladder, not chosen to pass.

⚠ **And the first mutation round was aimed at dead code.** Mutating `deq_iq2_xxs`/`deq_iq3_xxs` in
`ferric-gguf` left every cosine unchanged to five decimals: the loader sends expert slabs down the
**packed** path, so what real weights run is the WGSL, not the CPU decoder. A row of identical
verdicts is not a weak test — it is a test pointed at nothing.

`every_group_position` exists because the older test gave every group of a row the *same* slot, so a
kernel writing group g's lanes to group g′'s position still matched. Breaking that symmetry is what
ties the WGSL text to the Kani mirror.

Comparisons are **exact (`==`) where the reference is exactly representable** — integer activations
× {−1,0,+1} weights × d=1.0 keep every partial sum an exact integer below 2²⁴, so GPU and dense are
bit-identical regardless of accumulation order. A tolerance there would do no work while looking
like it did.

---

## 4. Derived rounding bounds — and one that is honestly loose

Not chosen numbers. Count the roundings, bound each against the **operand scale** Σ|terms| — the
quantity forward error is proportional to. Dividing by |result| instead reports tens of ulps on a
correct GPU the moment two terms cancel.

| path | bound | observed / bound |
|---|---|---|
| hc reduce, 6 sublayers | `(m+4)·ε·S_x` | **0.084** |
| hc state, 6 sublayers | `m·ε·S_H` | **0.269** |
| hc single step vs exact f64 oracle | — | **0.827 / 0.653 ulps** |
| DSA score | `ε·[DK·Σ|iw|Σ|qk| + H·Σ|iw·relu|]` | **0.020** |

Each carries a **floor** on observed/bound, so a bound that becomes decorative fails the test. The
old hand-picked `2e-4` in `hc` sat near 6e-4 on that scale — 200× looser than the arithmetic
warranted, and it would fail the floor.

⚠ **The DSA bound is ~190 ulps against an observed 0.69, and that is not slack to remove.** γₙ is a
worst case; on a random-sign dot the operand scale exceeds the result by ~√n while n roundings are
charged, so a rigorous order-independent bound is inherently ~√n·n loose. Tightening it would
require the kernel's accumulation order, which differs across fabrics. The bound's value there is
that it now *scales with the operands*, not that it is tight. It catches a 512-ulp injection and
**does not** catch an 8-ulp one — stated in the test rather than discovered later.

---

## 5. A measurement, not a bound: what the attention does to score error

The MLA paths differ *before* a softmax, whose response to score error is a perturbation bound, not
a rounding count. Composing worst cases through it lands far above observed, so no bound is
asserted. Measured instead:

```
perturbation 1e-4 -> amplification 0.002
perturbation 1e-3 -> amplification 0.001
perturbation 1e-2 -> amplification 0.001
```

**This attention attenuates score error ~500×.** Which corrected the model: the two paths differ in
the key fold (before the softmax, attenuated) *and* the value fold (after it, multiplied by
nothing). The split is **1% scores / 99% values**.

> A port of absorbed MLA should spend its precision on the **value** contraction, not the score one.

Asserted, so the claim fails if it stops being true. The most informative mutation is the
**survivor**, predicted in advance: a 64-ulp key-fold error survives, exactly as a 500× attenuation
requires. A test that cannot catch it is not weak — it is reporting the architecture.

---

## 6. Mutation testing — every claim above

Every proof and every bound was mutation-tested; a check that cannot fail is worth nothing.
Round tallies: **10/10, 6/6, 6/6, 6/6, 4/4 + 6/6 + 4/4, 3/3, 4/4**.

Three rounds of that found the checks themselves were wrong:

- **The first Kani proofs were vacuous — 1 of 4 caught.** `stride16_map_is_a_bijection` proved the
  map injective and in range; a *contiguous* map is also injective and in range. Another re-derived
  its indices inline and never called the code it claimed to check. **A proof that restates the
  formula is a tautology.** The rewrite runs the real decoder against the real encoder map — two
  files, two derivations, neither calling the other.
- **The test generator was one-signed, in seven places.** `(s >> 33) / 2³¹ − 1.0` is uniform in
  **[−1, 0)**, maximum −4.7e-10. Every "random" input in `hc`, `mla`, `dsa`, `quantize`, `dtype` and
  `nn` was negative. Worst case: with every logit negative and sinks at 0.0/+2.25, **the sink always
  won the max** — in the tests for the feature whose point is that it competes in that max. One
  published number moved: the synthetic STQ1_0 least-squares-vs-amax ratio was **8.7×, is 6.7×**.
  Every module now has a `the_fixture_generator_is_two_signed` test, checked to fail on the old one.
- **A whole mutation round landed on code the test never runs** — see §3. Four verdicts identical to
  five decimal places, which reads like robustness and is the opposite.

---

## 7. What none of this covers

- **That Ferric's decode matches Tencent's bytes.** That is empirical, settled by
  `stq1_0_interop.rs` against real published weights — not by any proof here.
- **The WGSL text**, beyond what the 2048-row differential exercises. Kani verifies the Rust mirror
  of the shader's addressing; the text is tied to it only by that differential. naga validates every
  quant shader with no GPU, but **structure only** — never semantics.
- **The indexer score's floating point.** ReLU is not a polynomial, so the GF(p) method does not
  reach it. Runtime tests only.
- **That hyv4's *arithmetic* is Tencent's arithmetic.** No reference implementation builds on this
  machine and the smallest checkpoint is 213.66 GiB against ~47 GB free, so nothing here compares
  this forward against theirs. Both cross-quant arms run the *same* forward code; a
  wrong-but-consistent formula agrees with itself. The formula is covered by the Kani harnesses,
  the GF(p) identities and the synthetic golden hash — not by any real-weights run.
- **Any block but 0 and 1, and any routing wider than 4 experts.** `hyv4_real_moe` slices 4 experts
  of the published 256 and runs top-2. Nothing exercises 256-way routing, an expert past index 3, or
  generation of any kind. The `hyv4` arch row stays `Untried` and `resolve` still refuses the string.
- **The energy figures' sensitivity to input sign.** They were measured on all-negative activations
  (same defect as §6). Both arms always saw identical data, so every ratio is a valid differential —
  but whether the ratios shift on two-signed input is **unmeasured, not unchanged**.
- **~~The CI proofs job.~~** Now run — see §1.
- **~~Linux test execution.~~** Also now run, and it paid for itself immediately.
  `examples/m4prof.rs` imported the macOS-only `metal4` module unguarded, so `cargo test
  --workspace` died at compile and the software-GPU job had **never run a test** — red and unread
  since 2026-07-22, which made the macOS job silently the whole of CI. Unblocked in `f9e3746`:
  **566 s, all tests pass** (Metal's own `Test workspace` takes 898 s, so lavapipe is not the slow
  part). The very next step then found a real portability defect and a real cost defect:
  - `examples/flash.rs` **panicked** — `causal_attention` materialises `[nh,T,T]`, which at
    T=3000/nh=8 is 288,000,000 bytes against lavapipe's `max_storage_buffer_binding_size` of
    134,217,728. `Context::new` requests `adapter.limits()`, so Metal grants far more and that case
    had only ever run because of the hardware under it. **The baseline that cannot allocate is the
    exact condition flash attention exists to remove**, so the fix compares at the largest head
    count that fits (T unchanged, so the 2048-key chunk boundary is still crossed) and then runs
    flash at the full head count where the baseline cannot run at all.
  - `examples/bench.rs` ran **113 minutes** — 96% of the whole step. llvmpipe sustains ~2 GFLOP/s,
    so a fixed 30 iterations of a 137-GFLOP matmul across three kernels is nearly two hours, to
    report throughput that means nothing on a CPU. `iters` is now derived from a measured single
    iteration against a wall-clock budget, and the sample count is **printed** (`n=30` … `n=1`), so
    a one-sample figure cannot be mistaken for a thirty-sample one. The equality check is untouched.
  ⚠ The constrained flash branch would otherwise have shipped never having executed on any machine
  its author could see — the same trap that put the panic there. `FERRIC_MAX_BINDING` forces it on
  any device; run at lavapipe's exact 134217728 it takes the small path and still proves equality.

---

## 8. The gates that refuse rather than report

The energy harness will not produce a number it cannot stand behind. Five independent refusals, each
naming its cause:

| gate | refuses when |
|---|---|
| scanner pre-flight | `syspolicyd`/`XprotectService`/`mds` above 20% — a rebuild's Gatekeeper scan, which the run itself caused |
| idle floor | > 1.0 W on the accelerator rail, taken as the **minimum of six windows** (idle is a floor, not an average) |
| calibration | the dense reference below 130 GB/s — a flat battery throttles hard *and stays perfectly quiet doing it* |
| `Saving::claimable` | an arm under one second |
| marginal-power assertion | work drawing less than idle — a physical impossibility that means a contended baseline |

The floors come from measured bands (dense sits at 143.6–151.2 GB/s across five clean runs), not
from numbers picked to pass. An earlier 100 GB/s floor let a half-recovered machine through at
101.7, and that was a threshold chosen to feel safe rather than derived — the same error the
tolerance work exists to remove.
