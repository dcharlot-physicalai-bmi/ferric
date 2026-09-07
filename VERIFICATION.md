# What is verified in Ferric's low-bit and attention paths, and what is not

Generated 2026-09-05 from the tree, not from memory. Counts are live:
**12 Kani harnesses**, **286 tests** (63 `ferric-gguf`, 83 `ferric-tensor`, 119 `ferric-llama`,
13 `ferric-tokenizer`, 8 `ferric-serve`), wasm32 clean on all four library crates.

This document exists because "verified" is not one thing. A bounded model check, an exhaustive GPU
differential, a probabilistic proof over a finite field, an equivalence chain, a derived rounding
bound and a measured amplification factor make six different claims of six different strengths, and
a reader deciding whether to trust a number needs to know which one they are holding. Every row
below says what the claim is, and the last section says plainly what nothing here covers.

⚠ These counts go stale, and stale counts in a document about verification are the exact failure it
warns about. Two claims in this file were false when the session that wrote them ended — the CI
proofs job "has not yet run", and a registry note asserting no real checkpoint had loaded — and both
were caught by a test, not by re-reading. Regenerate rather than trusting the numbers above.

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

## 3c. Reference comparison — the only hyv4 check that is not a self-comparison

Every other hyv4 check in this repo compares Ferric to Ferric: decode against prefill, batched
against solo, one quantisation against another, a golden hash against its own earlier self. **All of
them are satisfied by a wrong-but-consistent implementation.** This one is not.

`llama.cpp`'s `src/models/hyv4.cpp` is Tencent's own graph, written by other people from the same
spec. `scripts/hyv4_vs_reference.sh` runs it and Ferric **over the same file** — the synthetic
checkpoint Ferric's own GGUF writer emits:

| prompt | reference `sum_abs` | ferric `sum_abs` | relative Δ |
|---|---|---|---|
| `dlh` (3,11,7) | 21.410659 | 21.410664 | 2.3e-07 |
| `a` (0) | 20.593453 | 20.593319 | 6.5e-06 |
| `mzq` (12,25,16) | 18.677007 | 18.675997 | **5.4e-05** |
| `abcde` (0,1,2,3,4) | 19.971650 | 19.971348 | 1.5e-05 |
| `k9x` (10,35,23) | 25.421876 | 25.421875 | 3.9e-08 |
| `zzz` (25,25,25) | 21.914916 | 21.915052 | 6.2e-06 |
| `q7` (16,33) | 22.550225 | 22.549913 | 1.4e-05 |

A NEON CPU and a Metal GPU reducing in different orders. The six individual values `eval-callback`
prints for `dlh` match Ferric's to every digit shown.

⛔ **The gate is on `sum_abs`, not `sum`.** It used to compare the signed sum of the 40 logits, and
that quantity CANCELS: on the real checkpoint (§3d) per-block sums agreed to 8.4e-6 while the values
behind them were ~1e-3 apart per element. The same seven prompts under the old metric read
`4.1e-5 … 9.6e-4` — an order of magnitude *worse*-looking than the truth, and blind to the failure
mode it was supposed to catch. Both sides now emit `sum_abs`, and both are parsed **by field name**;
the old parse read a fixed line offset, which would have silently repointed at a different number
the moment either dump grew a line.

⛔ **"No reference implementation builds on this machine" was repeated for an entire session and was
simply never tested.** The patches apply cleanly at llama.cpp `0cea36222` and build CPU-only in
minutes; the CUDA half of patch 0002 is additive, and STQ1_0 has a CPU path. **Retest a blocker
before quoting it** — the same session also carried "~47 GB free" when the machine had 302 GB.

Two gaps in Ferric's own checkpoint had to close before the file was readable by anyone else, and
both are worth knowing:

- llama.cpp refuses a `gpt2`-model GGUF with **no `tokenizer.ggml.merges`**, before it will build a
  graph. Ferric's loader ignores those keys, so nothing here had ever needed them.
- A vocabulary of `t0`…`t39` **cannot segment any text**. The model then loads perfectly and reports
  *"there are not input tokens to process"*. A model that loads but cannot be given input is not an
  oracle. Single-character tokens fixed it.

⚠ **What this does NOT establish.** It is a *synthetic* checkpoint at d=32 with short prompts. It
does not cover the real weights, 256-way routing, a block past 1, long context, or a `top_k` that
actually selects — the synthetic config admits every position. The gate is `2e-4` **relative on
`sum_abs`** against the observed ladder `3.9e-8 … 5.4e-5` (~4x the worst), which is ~25x tighter than
the metric it replaced. That is one observation set on one adapter, and two Metal adapters are known
to disagree on the golden hash, so widen it only with a new measured ladder and never to make a
failure go away.

⚠ **It also cannot see the clamp defect of §3d.** The synthetic checkpoint's activations never leave
±10, where hyv4's SwiGLU clamp is the identity, so this comparison is bit-identical with that bug
present and with it fixed. A gate whose fixture never crosses a rule's threshold does not test the
rule.

---

## 3d. The real weights — one real defect, and a residual that is not one

§3c compares Ferric to Tencent's implementation on a **synthetic** checkpoint. This section is the
same comparison on the **real** 213.66 GiB `Hy4-preview-STQ1_0.gguf`, five tokens
(`802 8778 299 12749 341` — "The capital of France is", verified identical in both logs, `embd` sum
−7.720856 on both sides).

⚠ **Ferric runs on Metal; the reference runs on CPU.** llama.cpp's Metal backend crashes in
`ggml_metal_op_mul_mat_id` on hyv4's MoE, so CPU is the only reference path that exists. Every
number below is Ferric-on-Metal (M3 Ultra) against llama.cpp-on-NEON, and some part of the residual
is that and not either implementation being wrong.

### The defect: the SwiGLU clamp is a ROUTED-expert rule

hyv4 clamps the SwiGLU logits of its **routed** experts. Ferric applied that clamp to the **shared**
expert and the **dense** FFN as well. llama.cpp's own loader says so where it reads the key:
*"routed-expert SwiGLU logits clamp (shared/dense experts are NOT clamped, so `swiglu_clamp_shexp`
is intentionally left at its 0 default)"*.

| block 29, `ffn` split | Ferric (before) | reference |
|---|---|---|
| routed (8 of 256 experts) | −287.39 | −294.79 |
| shared expert | −337.78 | **−849.87** |

The routed half was already right; the shared half was 2.5x small. After the fix: shexp −857.93,
`ffn_out` −1145.32 against the reference's −1144.68.

⛔ **No test in this repo could have caught it.** The clamp is the IDENTITY while activations stay
inside ±10, and nothing in the synthetic checkpoint ever exceeds it — the synthetic comparison is
bit-identical with the bug present and with it fixed. It first bites at block 29 of the real
weights. A rule that only applies above a threshold needs a fixture that crosses the threshold.

### The retracted explanation

An earlier version of this section attributed the block-29 divergence to the two implementations
**selecting different experts** — the top-8 argsort over 256 near-tied router logits is
discontinuous, so a 0.003 difference flips rank 8. That reasoning was sound and the observation was
real (two of five tokens differed by exactly one expert, at rank 8), but it was **not the cause**.
Forcing the reference's own 385-entry routing table into Ferric via `FERRIC_FORCE_ROUTING` did not
close the gap. Selection divergence was a true fact that explained nothing, and it cost the
investigation a detour. The instrument that settled it — splitting `ffn_out` into routed and shared
— should have come first, because it is the split that separates the two candidate causes.

### The instrument was also wrong: `sum` cancels

Per-block agreement was originally read off the **sum** of each activation, and the sums agreed to
~8.4e-6. They were flattering the comparison: `result_norm`'s last row was −63.405 against −55.541,
a ~1e-3 per-element difference under a sum that matched to five decimals, because the errors
offset. Both sides now report `sum_abs`, which cannot cancel
(`common_debug_print_tensor` in the reference, `dump()` in `hyv4.rs`). **Every number below is
`sum_abs`.** §3c's gate was moved onto it too.

### What agreement actually is, all 78 blocks

With the clamp fixed and routing forced identical, relative `sum_abs` difference per block:

| tensor | mean | block 0 | worst |
|---|---|---|---|
| `attn_norm` (attention input) | **7.08e-05** | 6.7e-07 | 3.5e-04 (b67) |
| `attn_out` | **9.49e-04** | 2.3e-04 | 8.7e-03 (b77) |
| `routed` (8 of 256 experts) | 1.54e-03 | — | 1.3e-02 (b77) |
| `shexp` (shared expert) | 1.15e-03 | — | 1.2e-02 (b77) |
| `ffn_out` | 1.35e-03 | 4.5e-05 | 1.1e-02 (b77) |
| `l_out` (block output) | 8.74e-04 | 1.0e-04 | 2.4e-03 (b77) |

⭐ **The curve is FLAT, not stepped.** `l_out` runs 2.3e-03 at block 1, ~8e-04 through the middle,
2.4e-03 at block 77 — it does not grow with depth and has no step anywhere. A second defect of the
clamp's kind would appear as a step, the way block 29 did. There is no such step.

### Inside attention: no step there either

The input to attention agrees 13x better than its output (7.08e-05 → 9.49e-04), which reads like
localisation. It is not. Dumping the five stages between them — `kv_cmpr`, `q_pe`/`k_pe`,
`attn_kqv` (the attention core's output), `attn_gated`, `attn_out` — shows **the 13x is a
composition, not a jump**:

| stage | mean relative Δ | x previous |
|---|---|---|
| `attn_norm` (the input) | 7.08e-05 | — |
| `kv_cmpr` | 2.19e-04 | **3.1x** |
| `q_pe` / `k_pe` | 4.47e-04 / 4.97e-04 | ~2.1x |
| `attn_kqv` (attention core out) | 8.00e-04 | 1.6x |
| `attn_gated` | 9.43e-04 | 1.2x |
| `attn_out` | 9.49e-04 | 1.006x |

⛔ **"The residual is born in attention" was too strong, and this table is the correction.** The
largest single increment is the **first projection** — one 6144→576 matmul plus an RMS norm, 3.1x —
not the attention core, and not the gate. Growth then *tapers*: the output projection adds 0.6%.
Every stage is a matmul or a normalisation reducing over thousands of terms, and each one grows the
disagreement a little, which is what two different summation orders (NEON vs Metal) do.

⭐ **A monotone taper with no step is the signature of accumulation, not of a defect.** A wrong gate
would spike at `attn_gated`; a wrong RoPE at `q_pe`; a wrong mask or softmax at `attn_kqv`. None of
them does. This does not *prove* the residual is only fabric numerics — see the "not claimed" note
below, which still stands — but it removes every mechanism inside attention that a single bug could
occupy.

⚠ These are different tensors of different shapes, so the table is a sequence of measurements along
the dataflow, **not a strict error-propagation budget**. The ratios say where disagreement grows,
not how much each op contributes in isolation.

### Three explanations for the ~1e-3, all tested, all refuted

1. **The low-bit expert kernels.** Block 0's FFN is entirely Q6_K and agrees ~10x better than every
   later block, whose routed experts are IQ2_XXS (2.06 bpw) / IQ3_XXS (3.06 bpw) / STQ1_0 (1.31
   bpw). Tempting — and wrong. The **shared** expert is Q6_K and sits inside the same MoE blocks:
   it disagrees by 1.15e-03 against the routed experts' 1.54e-03. A 6.5-bit path and a 2-bit path
   fed the same input disagree by the same amount, so the low-bpw kernels are not the source; both
   inherit the error from their common input. (`packed_iq_matmuls_match_dequant_then_matmul` in
   `ferric-tensor` independently pins both kernels against dequantise-then-matmul at 1e-4.)
2. **Dense-vs-MoE.** `leading_dense_block_count` is **1**, so block 0 differs from every other block
   in two ways at once — dense-vs-MoE and Q6_K-vs-low-bpw — and the two are perfectly confounded.
   The routed/shared split above is what breaks the confound; neither survives it.
3. **The reference's KV cache dtype.** llama.cpp defaults `-ctk`/`-ctv` to **f16**, whose relative
   spacing (2⁻¹⁰ ≈ 9.8e-04) sits almost exactly on the measured `attn_out` gap of 9.49e-04 — while
   Ferric's `MlaCache` holds f32. A striking fit, and false: re-running the reference with
   `-ctk f32 -ctv f32` leaves `attn_out` **bit-identical** at every block (verified by hand:
   block 0 `sum_abs` 2950.567254, block 77 442536.124366, both runs). The flag was not inert — the
   stored `cache_k_l0` changed from `(f16)` to `(f32)` and 22,694 log lines differ — so this is a
   refutation, not a vacuous test. ⚠ That run also picked a different `n_ctx` (260608 vs 256), so it
   was not a pure single-variable change; the bit-identical attention holds regardless.

The DSA lightning indexer contributes but does not explain it either: the 21 blocks that run its
top-k selection disagree by 1.32e-03 against the other 57 blocks' 8.13e-04 — a factor of 1.6, on a
baseline that is already there without any sparse selection.

### Ferric's own routing, without the crutch

Every per-block number above is measured with `FERRIC_FORCE_ROUTING` pinning the reference's expert
*ids* (weights and expert arithmetic stay Ferric's). Run without it, on Ferric's own argsort:

| tensor | forced routing | Ferric's own routing |
|---|---|---|
| `attn_norm` | 7.08e-05 | 2.01e-04 |
| `attn_out` | 9.49e-04 | 2.44e-03 |
| `ffn_out` | 1.35e-03 | 3.12e-03 |
| `l_out` | 8.74e-04 | 1.09e-03 |

✅ Agreement degrades by ~2.5x and **stays the same order of magnitude through all 78 blocks**. It
does not blow up, which is the thing worth knowing: Ferric tracks Tencent's implementation on its
own expert selection. Some of that 2.5x is not error at all — a different expert at rank 8 is a
genuinely different function, so the argsort is *supposed* to produce divergence there.

End-to-end, the last row's 120832 logits:

| | `sum` | `sum_abs` | relative Δ on `sum_abs` |
|---|---|---|---|
| reference | −377666.72 | 421398.64 | — |
| Ferric, own routing | −376036.64 | 420188.58 | **2.87e-03** |
| Ferric, forced routing | −385843.10 | 429466.50 | 1.91e-02 |

⭐ **Forced routing agrees better at every block and worse end-to-end — by 6.7x.** Both metrics rank
it the same way, so this is not a cancellation artifact; it is real. Pinning the reference's expert
ids while the combining weights stay Ferric's builds a **hybrid that is neither implementation**,
and being closer at each intermediate step does not make the output closer.

⛔ **So forced routing is a diagnostic instrument, not a fidelity result.** It exists to make
intermediate activations comparable across the argsort discontinuity, which is the only reason the
per-block table above can be read at all. It must not be quoted as "Ferric agrees this well" — the
honest end-to-end number is the unforced one, 2.87e-03.

### What is claimed, and what is not

✅ **Claimed**: on the real checkpoint, Ferric and Tencent's implementation agree to
**~1e-3 relative on activation magnitude at every one of the 78 blocks** with routing forced,
and **2.87e-03 end-to-end on Ferric's own routing** — flat with depth, with no step that would
indicate a second defect. The tokenizer, embedding dequantisation, hyper-connections, MLA
projections, RoPE, the DSA indexer, attention, the gate, the output projection, the 256-expert
MoE over three quantised expert formats, and the shared expert are all inside that.

⛔ **Not claimed**: that the ~1e-3 is *only* CPU-vs-Metal numerics. Everything measured is
consistent with it — bounded, flat with depth, roughly sign-balanced (Ferric larger on 40 of 78
blocks for `attn_out`), and inside a block it accumulates as a smooth taper across five large
reductions with no step at any one of them. Four candidate mechanisms have been tested and refuted
(low-bpw kernels, dense-vs-MoE, the reference's KV dtype, the DSA top-k) and the per-stage bisect
leaves no room inside attention for a single bug. **But "every mechanism I thought to test is
refuted" is not "it is numerics."** No experiment here has measured the fabric contribution
directly — that would need the same Ferric build on a second adapter, which this machine does not
have. It is unexplained, and the honest word for it is unexplained.

⛔ **Not claimed**: anything past five tokens, or any decode step. This is one prefill.

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

## 5b. Equivalence chains — decode against prefill, prefill against a reference

`mla.rs` declined to implement cached decode for a stated reason: *"a decode path would be
unverified code wearing a verified module's name."* That was right for the oracles available then.
What answers it is a **chain**, not a new reference:

```
AMD's real Instella module  ──maxΔ 5.96e-7──▶  MLA prefill  ──exact──▶  MLA cached decode
                             (instella_gmla)                (incremental == full re-run)
```

Every link is checkable here. The same shape then extends to the whole hyv4 graph — hyper-connections,
the DSA indexer and its per-full-layer key cache, clamped-SwiGLU MoE, absolute-position RoPE — where
any split of a sequence into decode blocks must reproduce `forward` over the whole thing.

| check | regime |
|---|---|
| `cached_decode_equals_a_full_re_run` | single-token, with and without an attention sink |
| `block_decode_in_uneven_chunks_equals_the_whole` | 1+2+2, 2+1+2, 3+2, 5 |
| `the_offset_mask_hides_exactly_the_future` | the mask asserted directly, not only through an equality |
| `hyv4_synthetic` (example, **in CI**) | whole graph, 5 splits × `top_k` {64, 2}, on **three adapters** |
| `hyv4_real_moe` (example, local) | whole graph on **Tencent's real weights**, both quantisations, `top_k` {2048, 2} |

⚠ **Two oracle holes were found here, and neither was a shortage of test data.** Both were a
*configuration that could not express the bug*:

- **At `tq == 1` the offset causal mask is a no-op** — row 0 spans `(off+1)..tkv`, empty when
  `off == tkv-1`. Single-token decode therefore cannot detect a wrong offset: every query
  legitimately sees everything. Only blocks with `tq > 1` make intra-block causality observable, and
  the splits are **uneven** because equal chunks make `off` a multiple of the chunk size, which
  several wrong formulas also satisfy.
- **At `top_k = 64` the DSA mask is all zeros**, so its column order is unobservable and an
  index-cache that prepends instead of appending survives. Only `top_k = 2` selects. The real-weights
  example inherited this exact blind spot (published `top_k` is 2048 against 8 positions) until it
  was run at `top_k = 2` as well.

⭐ **Changing the REGIME is what found both.** More tokens would have found neither. The two axes now
covered independently are hardware (three adapters, two backends) and numerics (random vs trained
weights, dense vs sparse selection).

⛔ **A bit-hash is a local regression lock, never a portability claim — and it is per ADAPTER, not
per backend.** The same synthetic checkpoint hashes three different ways:
`Apple M5 Max 0x29142d075f1beadc`, `Apple Paravirtual device 0x13cfd14821cf04d5` (**also Metal**),
`llvmpipe 0xf690016066e7574b`. Kernel selection reads capabilities, and a paravirtualised GPU does
not advertise what an M5 Max does. Every *reported quantity* agrees to four decimals across all
three (0.2993, 0.4989, 0.00e0) — only the hash, which amplifies one bit into a different number,
separates them. The lock is keyed by adapter and prefix-matched; an unrecorded adapter reports
rather than asserts.

⭐ **Order the portable evidence before any absolute lock.** The hash originally sat *ahead* of these
equivalences, so the first CI run on a new fabric died on a fabric-specific value without ever
running the checks that hold everywhere — while the session notes already claimed decode was verified
"on both fabrics". Self-comparisons first.

---

## 6. Mutation testing — every claim above

Every proof and every bound was mutation-tested; a check that cannot fail is worth nothing.
Round tallies: **10/10, 6/6, 6/6, 6/6, 4/4 + 6/6 + 4/4, 3/3, 4/4, 4/4, 4/4, 2/2**.

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
- **~~That hyv4's *arithmetic* is Tencent's arithmetic.~~** Checked on a synthetic checkpoint (§3c)
  **and on the real 213.66 GiB weights for blocks 0–28** (§3d), which does exercise 256-way routing
  and the quantised expert formats. **Blocks 29–77 are untested, not wrong**: expert selection
  diverges there on marginal experts, after which the two implementations compute different
  functions and cannot be compared end to end. Settling them needs routing forced identical.
- **Any block but 0 and 1, and any routing wider than 4 experts.** `hyv4_real_moe` slices 4 experts
  of the published 256 and runs top-2. Nothing exercises 256-way routing or an expert past index 3.
- **~~Batched decode for hyv4.~~** Done — `Hyv4::decode_batch`, verified token-identical to solo
  decode at n = 2/3/4 on sequences of **different lengths**, which is the discriminating case:
  borrowing sequence 0's `n_past` gives every row the right answer when every row is at the same
  position. Three mutations caught (rope from sequence 0, top-k offset from sequence 0, `n_past`
  never advancing). `supports_batching` is now `true`, and the same equality holds on **Tencent's
  real weights** at three different sequence lengths under sparse selection.
- **~~A generation loop.~~** `hyv4_synthetic` now drives greedy decode → argmax → feed-back and
  requires a fresh `forward` over the produced sequence to predict the same token at every position.
  ⚠ Labelled in the source as a COMPOSITION check, largely subsumed by the decode oracle: it drives
  the same machinery and the splits run first, so every library mutation constructible against it is
  caught there. What it independently pins is the loop's **shape** — which row of a multi-token
  decode predicts the next token, and that feeding it back lands at the right position. **Sampling
  beyond greedy is still not covered** here; `ferric-serve` owns temperature and top-p.
- **Embeddings from hyv4.** `forward_hidden` refuses outright: hyv4 exposes no pre-head hidden
  state, and returning logits would make every embedding wrong while looking like a vector.
- **Serving it at all.** `hyv4` is now a first-class `Runtime::Hyv4` wired through `ferric-serve`
  (it previously named `Runtime::DeepSeek2` as a placeholder — one status edit away from loading a
  hyv4 checkpoint *as a DeepSeek2 model*). The row still carries `Status::Untried` and `resolve`
  still refuses the string, so nothing can reach that dispatch.
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
  its author could see — the same trap that put the panic there. **`FERRIC_MAX_BINDING` now lives in
  `ferric-core`**, clamping `max_storage_buffer_binding_size` in `required_limits` so *any* code path
  meets the constrained fabric locally. Clamping that one limit isolates the variable exactly, rather
  than swapping in a downlevel profile that would move unrelated ones.
- **Fixing these one CI round-trip at a time was the wrong loop.** With the hook, one local sweep of
  all **42** example commands the Linux job runs found every remaining failure at once — seven, of
  which only one was the binding limit:
  - `bandwidth` asked for a 512 MiB binding. Its premise is *"buffer >> last-level cache"*, not the
    literal 512 MiB, so it now clamps to what the device will bind **and says so** — a bandwidth
    figure from a smaller buffer is a different measurement and must not print as the same one.
  - `q4_k` `q5_k` `q6_k` `coop_q4_k` `coop_q5_k` `coop_q6_k` failed **identically with and without the
    clamp** — the control mattered — on `attempt to multiply with overflow`: an unwrapped
    `j * 2654435761` inside a seeded hash whose every other operation is `wrapping_*`. Release
    already wrapped, so `wrapping_mul` changes no value; it makes **debug agree with release**, and
    debug is what CI runs. 17 sites across 9 files. These had never run in CI anywhere.
  - `bandwidth` then aborted at **exit 134 after printing correct output**: dropping a `wgpu::Buffer`
    reaches `SnatchLock::read` → `LockTrace::enter`, which calls `LocalKey::take`/`set` on
    wgpu-core's own thread-local — and those **panic once it is destroyed**. Ferric caches
    `wgpu::Buffer` in its own `thread_local!`, destruction order between two crates' TLS is
    unspecified, and a panic in a `Drop` aborts. Latent in *any* program exiting with a cached wgpu
    resource alive, and **debug-only**, so it fires in CI and never in a release build. Fixed in
    `forks/wgpu-core/src/snatch.rs` with `try_with`: a destructor must not panic. Mutation-tested —
    forcing the guarded body to panic aborts a normal run, so the recursion check still executes.
  ⭐ **Confirmed on the real fabric, not inferred from the local sweep** — run 33927133667 on
  `3b049d6` is the first fully green CI run: software GPU ✓, real Metal ✓, proofs ✓. 58 examples
  executed on llvmpipe; `bandwidth` clamped and said so (3.9 GB/s scalar, 5.3 vec4 — against 292 on
  Metal, which is what a CPU rasterizer should look like); `flash` took the constrained path at
  nh=2 and nh=1; `bench` chose n=4/1/1/1; and Q4_K/Q5_K/Q6_K validated exact
  (max|Δ|/scale ≈ 1e-6) **for the first time ever in CI**. The validation step went 7054 s → 968 s
  → 1218 s and now passes; `Test workspace` is 575 s.
  ⚠ The local sweep clamped Metal's limit — it could not cover anything depending on llvmpipe's
  *behaviour* rather than its reported limits. The `LockTrace` abort in particular was only ever
  reproduced on Metal. That gap is what this CI run closed.

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
