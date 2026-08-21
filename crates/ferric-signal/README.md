# ferric-signal

**Open sensor-language tokenization for physical signals.** Pure Rust. Turns a sensor stream into
discrete tokens that live in the same vocabulary as text, so a decoder can read a measurement and a
word through one embedding table — and ships a **determinism receipt** so anyone can check that a
given token stream came from a given signal.

```
ferric-signal tokenize vibration.csv --channels 2 --receipt run-a.kv --save-weights model.fsig
ferric-signal tokenize vibration.csv --channels 2 --weights model.fsig --receipt run-b.kv
ferric-signal verify   run-a.kv run-b.kv
ferric-signal cost     --window 1024
```

```
DIFFERENT SPEC. These runs were not asked the same question, so comparing
their tokens would mean nothing.
  differs: patch_len  16  vs  32
  differs: stride     16  vs  32
```

## Why

Reviewing the field in August 2026: the time-series **forecasting** foundation models are open but do
not speak language (Chronos, TimesFM, Moirai, Toto, Timer, Granite TTM). The models that do speak
language about sensors are either narrow — wearable IMU — or reachable only through an API. This
review located **no joules-per-token figure published for any of them.**

The tokenizer is the piece everything else depends on, and it is also by a wide margin the *small*
part: a discrete bottleneck over five dimensions with eight levels each spans 32,768 codes and needs
**no codebook at all**. So the component that makes the rest possible is the one that costs least to
open.

## What is here, and how far each part is verified

| Module | What it does | How it is checked |
|---|---|---|
| `fsq` | Finite scalar quantization | Bijection and round trip **exhaustive over all 32,768 codes** |
| `patch` | Patching + reversible per-channel normalization | Checked against its own inverse |
| `encoder` | Tower shape, parameter accounting, positions | Sizing checked against the published 9.5M; positions against the closed form |
| `tower` | Encoder and decoder forward passes | Allocated parameters equal the arithmetic; bit-exact across runs |
| `vocab` | Signal codes and text tokens in one id space | Every id in the combined space round trips |
| `cost` | Operations and bytes per token | Exact arithmetic, hand-checked |
| `receipt` | Recomputable claim about a token stream | Field-by-field digest sensitivity |
| `sha256` | Self-contained hash | Published test vectors, including the 1M-character one |
| `store` | Save and load weights | Save/load must yield **identical tokens**; every truncation point refused |
| `train` | Straight-through estimator | Gradient must behave as if rounding were the identity |
| `language` | Mixed text/signal sequences, causal LM | Encode/decode inverses; a position cannot see its future |

**95 tests.** Fourteen are mutation-controlled: each was verified to FAIL when the line it names is
broken, because a test that survives the mutation of the thing it names is decorative. Three real
bugs were found that way and none of them raised an error on their own — a quantizer that silently
halved its resolution, a stuck sensor that normalised to a full-amplitude signal made of rounding
dust, and a bottleneck whose latents grew until the code space collapsed from 32,768 to 27.

## Measured results, and what they are worth

Trained on **synthetic** physical processes — damped oscillation, thermal decay, PWM, chirp, noise on
drift — whose ground truth is known by construction.

| | |
|---|---|
| Transformer autoencoder, reconstruction | **25.5 dB SNR**, 0.28% of variance, 444x loss reduction |
| Signal to text, on trained examples | **5 of 5** processes named from their tokens alone |
| Signal to text, **held out** | **4 of 10 (40%)** against a 20% chance baseline |

**The third row is the one that matters.** Five examples over five classes is memorisable by a lookup
table, so the 5-of-5 proves nothing on its own. The held-out split runs the same processes with
different parameters — a faster decay, a wider duty cycle, a different chirp span — and 40% against
20% chance is the memorisation signature: the architecture learns the mapping, and has nothing to
generalise from. **That gap is what a real corpus would close.**

### Scaling the synthetic corpus: three curves, two of them instructively wrong

Train on N variants per process (bounded parameter families from `synth`), hold out a disjoint set,
and guard the split **where the model looks**: any held-out example whose token sequence also
appears in training is excluded and counted, because per-example normalization cancels each
family's affine parameters and raw-signal distinctness proves nothing downstream of a lossy
front end.

| held-out accuracy at N variants/kind | 1 | 2 | 4 | 8 | 16 | control |
|---|---|---|---|---|---|---|
| untrained tokenizer | 38% | 44% | 50% | 53% | **60%** | 20% |
| tokenizer retrained per size, fixed 400 steps | 47% | 50% | 47% | 38% | 19% | 19% |
| one tokenizer, trained once on the full pool, 1200 steps | 32% | 21% | 12% | 38% | 38% | 19% |

Chance is 20%, every control lands on it, and each control prints its realized label agreement so
the null is readable. Held-out n is 15–19 per point, so a single step is within noise; the shapes
across five sizes are the evidence. The tokenizer never sees a held-out variant in any row — the
split holds across the whole model, not just the LM.

**Row 1, the best curve, is borrowing.** It climbs monotonically and is still climbing at 16
variants, but the untrained tokenizer collapses the entire thermal family to shared token sequences
— every thermal example is excluded as token-identical — and part of its generalization rides on
exactly that blurriness: codes shared between train and held-out act as smoothing.

**Row 2 is a confound inverting a conclusion.** Retraining the tokenizer per corpus size at a fixed
step budget makes tokenizer quality a hidden variable of the x-axis: reconstruction error scales
almost linearly with corpus size (0.003 to 0.020), the thermal family re-collapses, and at 16
variants even train accuracy breaks. A reader shown only this curve would conclude that data hurts.
Any scaling study that retrains a preprocessing component per corpus size at fixed budget carries
this exact risk.

**Row 3 revises this crate's own previous conclusion, and the revision is the finding.** An earlier
commit concluded "the next lever is training the tokenizer, not scaling the LM." Measured, that is
incomplete: one well-trained tokenizer — reconstruction 0.007, the thermal family separated, the
split guard nearly quiet at small sizes — held fixed across the sweep is worse than the untrained
one at every corpus size. The mechanism is the frozen embedding. A sharp tokenizer routes each
signal through near-unique codes; an unseen code is an untrained embedding row carrying nothing;
and the blurry tokenizer's collisions were doing real representation-sharing work that the
frozen-embedding LM depended on. Training the tokenizer alone makes held-out worse.

**A fourth measurement tested row 3's own explanation, and refuted it.** The mechanism above —
sharp codes starve because the frozen embedding cannot learn them — predicts that unfreezing the
table closes the gap. The crate now has a differentiable lookup (`embed_var`: a one-hot matrix
times the table, so matmul's existing backward is exactly the scatter-add a trainable embedding
needs), and the full grid at 16 variants per kind says otherwise:

| held-out at N=16 | frozen embedding | trainable embedding |
|---|---|---|
| untrained tokenizer | **60%** | 53% |
| trained tokenizer | 38% | 31% |

Trainable embeddings lift the sharp tokenizer at small and mid sizes — 12% to 31% at N=4, 38% to
44% at N=8 — and give it back at N=16, where the two embedding conditions sit one example apart,
inside noise at these sample sizes. The blurry tokenizer wins every cell. The control in the
trainable configuration lands at 25% against the 20% chance line, one example above it.

### The direct test: sweeping codebook resolution at fixed corpus

The conclusions above say resolution must be matched to coverage. That is testable in one
parameter, so it was tested: hold the corpus at 16 variants per kind and sweep the quantizer from
2 to 12 levels per latent dimension.

| levels/dim | codes | distinct sequences | held-out | excluded | code overlap |
|---|---|---|---|---|---|
| 2 | 32 | **1 of 80** | undefined | 40 of 40 | 100% |
| 3 | 243 | 62 of 80 | **62%** | 14 | 99% |
| 4 | 1,024 | 60 of 80 | 57% | 12 | 96% |
| 6 | 7,776 | 65 of 80 | 58% | 9 | 91% |
| 8 | 32,768 | 66 of 80 | **62%** | 11 | 76% |
| 12 | 248,832 | 65 of 80 | 48% | 9 | 56% |

**A plateau bounded by two failure modes, not a peak.** At 32 codes the tokenizer emits ONE
sequence for all eighty training examples — damped, thermal, square, chirp and noise become the
same token stream, train accuracy falls to guessing, and every held-out example is excluded as
token-identical. Held-out accuracy is reported as undefined rather than as a number, because with
no distinct sequences there is nothing to score. At 248,832 codes the opposite failure begins:
overlap falls to 56%, held-out signals land on codes the corpus never visited, and accuracy drops
to 48%.

Between those walls, across a **135x span of codebook size**, held-out sits at 57–62% — flat within
noise at these sample sizes (n is 26–31 after exclusions, so one example is 3–4%). 243 codes and
32,768 codes perform the same.

Two things follow, and the second is the more useful:

- **Resolution is the causal variable; training the tokenizer was a proxy for it.** The earlier
  rows are re-read by this sweep: the untrained tokenizer won not because it was untrained but
  because it was COARSE, and coarse in the useful range. Training sharpened it past the plateau.
- **At this corpus scale the codebook can be two orders of magnitude smaller for the same
  accuracy.** The 32,768-code default is inherited from the published description of a model
  trained on 23 billion tokens; at 80 training examples it buys nothing over 243 codes. A codebook
  is sized for a corpus, and a reproduction inherits the number without inheriting the corpus.

What the whole set of measurements supports:

- **Generalization in this regime flows through shared codes, and a tokenizer sharp enough to
  eliminate sharing starves the model regardless of what is trainable downstream.** Codebook
  resolution has to be matched to the coverage the corpus can actually provide — a tokenizer can
  be too good for its data. This is the small-scale form of a familiar production rule: vocabulary
  size is chosen to fit the corpus, and the two are grown together.
- **A tokenizer's training budget must scale with its corpus.** At fixed budget it silently becomes
  the bottleneck, and its degradation masquerades as a data effect with the opposite sign.
- **Unfreezing the embedding is necessary machinery, not a cure.** It converts coverage into
  accuracy faster mid-curve, and it does nothing for codes the corpus never visits.

The first version of this experiment reported a higher, non-monotone curve with no split guard at
all, and it was wrong: held-out examples were token-identical to training examples, and denser
sampling made that more likely — a curve partly manufactured by its own x-axis. The guard, the
control calibration and the per-batch class balance all came out of an adversarial review of the
protocol; every number above postdates it.

## What is NOT here

**There are no published weights for real sensors.** Everything above is synthetic, with labels the
generator wrote. Nothing has been compared against a reference implementation's outputs, because no
reference weights were located.

**The embedding table is frozen** during language training, because the autograd layer has no row
gather. A frozen embedding cannot learn that two signal codes mean similar things, which caps what a
small corpus can teach.

**Energy is not measured unless a meter exists.** `cargo run --example token_cost` reports the exact
operation count always, and prints `NOT MEASURED` when no hardware counter is readable rather than
printing a zero. A zero is indistinguishable from a very efficient run, which is how a great many
published efficiency claims come to be wrong.

## Two findings worth stating

**A per-token cost is not a constant.** Attention is quadratic, so the cost of a token is a property
of the *window* it sat in. The same token costs **5.4x more at 8192 patches than at 16**. Quoting
"X joules per token" without the window is not a tight figure missing context; it is an
underspecified one.

**At short windows this is bandwidth-bound**, at 8.1 FLOP per weight byte against the ~100–300 a
modern part needs to be compute-bound. On a sensor node the bill is reading the weights, not the
arithmetic — so quantize the weights rather than optimising the math.

## The receipt

Two digests. `spec_digest` covers everything that *determines* the tokens — signal, normalization
statistics, patching, encoder shape, weights, quantizer levels, vocabulary layout. `token_digest`
covers what came out. A mismatch is therefore diagnosable rather than merely alarming: same spec and
different tokens means the computation diverged; different spec means you compared two different
things, and that is reported first.

The platform is deliberately **outside** the spec digest. If it were inside, every machine would
digest differently and the comparison that matters — same question, same answer, different hardware
— could never be posed.

Receipts travel as flat key/value pairs, the shape [ferroscope](https://github.com/dcharlot-physicalai-bmi)
carries inside an MCAP recording. Interoperating on the format rather than by linking two build
graphs is deliberate.

## Licence

Apache-2.0. Institute for Physical AI @ Bailey Military Institute.
