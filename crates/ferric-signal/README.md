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

**107 tests**, fourteen of them mutation-controlled: each was verified to fail when the line it
names is broken. Several silent defects were caught that way and are documented at the code that
fixes them.

## Measured results

The first three results below are **synthetic**: five parameterized physical process families from
`synth`, whose labels this crate's own generator writes. The real-sensor section that follows is the
UCI hydraulic corpus, with labels the rig's operators wrote.

**Reconstruction.** The transformer autoencoder trains end to end through the FSQ bottleneck to
25.5 dB SNR (0.28% of variance) on synthetic physics. Single run.

**Signal to text.** A causal decoder over the hybrid vocabulary learns to name the process that
produced a token stream. Held-out variants are new parameter draws from the same families, and any
held-out example whose token sequence also appears in training is excluded from the score and
counted — per-example normalization cancels each family's affine parameters, so distinctness in the
raw signal does not imply distinctness in the tokens.

At 16 training variants per process, five seeds each, chance 20%, n≈26–29 after exclusions:

| codebook | five seeds | mean | sd |
|---|---|---|---|
| 243 codes | 62, 54, 65, 69, 54 | **60.8%** | 6.2 |
| 32,768 codes | 62, 48, 45, 52, 55 | **52.4%** | 5.9 |

**Read that spread before reading anything into a gap.** At this sample size a difference of five
points is noise. The 8-point gap between these two rows is about one pooled standard deviation:
suggestive that the smaller codebook is no worse and possibly better, not a demonstration.

**One structural result is not a matter of sampling.** Below roughly 200 codes the tokenizer stops
discriminating: at 32 codes it emits a single token sequence for all eighty training examples, every
process collapses to the same stream, and held-out accuracy is undefined rather than low. That is a
property of the quantizer's resolution against the data, not a seed effect.

**Real sensor data: signal to text works, and the earlier null here was a protocol defect.**
`examples/hydraulic` ingests the UCI Condition Monitoring of Hydraulic Systems corpus (CC BY 4.0,
not redistributed here) — 2,205 cycles, 17 channels at four sampling rates, five independent label
axes. 400 strided cycles, 300 train / 100 held out, the split fixed by cycle position before
anything is fitted. Three seeds.

| axis | held-out | sd | majority | chance | control |
|---|---|---|---|---|---|
| cooler | **84.7%** | 1.7 | 36.0% | 33% | 36.0% |
| valve | **72.3%** | 4.1 | 54.0% | 25% | 55.0% |
| pump_leak | 57.0% | 0.8 | 55.0% | 33% | 52.7% |
| accumulator | **43.3%** | 3.4 | 33.0% | 25% | 35.7% |
| stable | **85.3%** | 0.5 | 63.0% | 50% | 66.7% |

`control` is the same protocol with the cycle-to-caption assignment permuted — same split, same
class balance, same caption vocabulary, same correlations *within* a caption. It lands on the
majority baseline for every axis, which is what makes the left column mean anything. It matters
most for the four axes scored with the earlier caption words teacher-forced: those five labels are
not independent in this rig's design, so a decoder could in principle answer `stable` from `cooler`
and never read a sensor. Under permutation that shortcut survives and the signal does not, and the
control does not rise. **`pump_leak` is a null** — 57.0% against a 55.0% majority, with the control
at 52.7%.

**What actually caused the earlier null, which was not what this README previously said.** An older
version of this experiment reported every axis at or below majority, sd 0.0 across seeds, and named
the untrained tokenizer as the cause. That diagnosis was wrong. Holding the corpus, the split, the
tokenizer and the number of examples seen fixed, and varying only how training examples are
presented:

| protocol | cooler | valve | pump_leak | accumulator | stable |
|---|---|---|---|---|---|
| batch 1, **corpus order** | 37.0 ±0.0 | 54.0 ±0.0 | 55.0 ±0.0 | 23.3 ±6.9 | 72.0 ±0.0 |
| batch 1, shuffled | 54.3 ±6.1 | 66.7 ±3.8 | 58.3 ±1.7 | 37.3 ±4.6 | 85.0 ±1.6 |
| batch 8, shuffled | **84.7 ±1.7** | **72.3 ±4.1** | 57.0 ±0.8 | **43.3 ±3.4** | **85.3 ±0.5** |

The first row is the published null, reproduced exactly: one word emitted for every held-out cycle
on three axes, sd 0.0. **This corpus is stored in experimental-condition order**, and the module
already knew that — it is why cycles are sampled with a stride instead of a prefix. The same fact
was never applied to the training loop, which walked the sampled cycles in index order and so fed
the decoder long runs of near-identical labels. Shuffling the presentation is most of the recovery;
accumulating gradients over 8 examples per step is another 30 points on `cooler`.

The guard against a silent repeat is now in the output. Every run reports how many **distinct words**
the model actually emitted at each axis position across the held-out set. A model answering the same
word every time scores that word's frequency and reads as a weak learner in an accuracy column;
`1.0 of 3` says outright that it never varied.

**A probe located the null before it was fixed, and predicted where the signal was.** Multinomial
naive Bayes over (channel, code) counts — no language model, no capacity to speak — asks whether the
token stream separates the classes at all:

| axis | majority | probe, untrained tokenizer | probe, trained tokenizer | permutation control |
|---|---|---|---|---|
| cooler | 36.0% | 75.0% | 76.0% | +9.0 pt |
| valve | 54.0% | 59.0% | 54.0% | +1.0 pt |
| pump_leak | 55.0% | 62.0% | 60.0% | +2.0 pt |
| accumulator | 33.0% | 48.0% | 54.0% | +5.0 pt |
| stable | 63.0% | 77.0% | 81.0% | +4.0 pt |

Run while the decoder was still emitting a constant, this said the tokens carried `cooler`, `stable`
and `accumulator` well clear of the control and `pump_leak` barely. The repaired decoder landed in
exactly those places, and `pump_leak` is the axis that stays at majority in every cell. **At 100
held-out examples the control sits at +1 to +9 points; at 10 it reached +40**, which is the whole
reason the probe is reported with one.

RevIn normalises every channel of every cycle to zero mean and unit scale before patching, so
absolute pressure and absolute temperature are gone by construction. Whatever is being read is shape.

**The tokenizer now trains on real sensor data, and it is not the binding constraint.** Encoder,
FSQ bottleneck and decoder train to reconstruct the corpus's own patches, on the training cycles
only — an unsupervised objective is still leakage if it is fitted to held-out signals. 1,200 steps
per patch length, one channel-cycle per step so training and inference see the same receptive field:

| patch | held-out MSE | held-out SNR | codes visited |
|---|---|---|---|
| 10 samples | 0.156 | 8.1 dB | — |
| 100 samples | 0.067 | 11.3 dB | — |
| untrained tokenizer | — | — | 3,222 / 32,768 (9.8%) |
| trained tokenizer | — | — | 12,561 / 32,768 (38.3%) |

Training quadruples code-space use and gives the first real-data reconstruction figures in this
crate. It does **not** clearly buy accuracy: against the untrained tokenizer it moves `accumulator`
+10.3 points and `stable` −8.7, with everything else inside a seed's spread. Reconstruction fidelity
and label separability are different objectives, and optimising the first did not deliver the
second.

**Sizing the vocabulary to the corpus is free.** Restricting the signal vocabulary to the codes the
*training* cycles use — held-out codes outside that set map to one reserved id, and 0.39% of
held-out tokens land there — gives 3,046 embedding rows against 32,788, and the same accuracy:

| vocabulary | rows | cooler | valve | pump_leak | accumulator | stable |
|---|---|---|---|---|---|---|
| full codebook | 32,788 | 80.7 ±8.6 | 68.7 ±4.7 | 58.0 ±1.4 | 42.7 ±2.1 | 86.7 ±1.2 |
| compacted | 3,046 | 84.7 ±1.7 | 72.3 ±4.1 | 57.0 ±0.8 | 43.3 ±3.4 | 85.3 ±0.5 |

**10.8x fewer rows, every axis within a seed's spread.** The output head is what scales with
vocabulary, so this is the traffic argument below, measured on real data rather than derived.

**The energy accounting is arithmetic, not measurement.** A decoder touches its vocabulary twice per
position — one row on the way in, every row at the output head on the way out — so the head is what
grows with codebook size:

| codes | rows | output head | read per token generated |
|---|---|---|---|
| 243 | 252 | 0.06 MB | **63 KiB** |
| 32,768 | 32,777 | 8.39 MB | **8,195 KiB** |

130x the traffic per token. This crate's arithmetic intensity is about 8 FLOP per weight byte
against the 100–300 a modern part needs to be compute-bound, so at the edge bytes moved set the
bill. `vocab_cost` computes it and the accounting is tested.

**What follows for sizing a codebook.** Accuracy differences across this range are below what the
held-out sets here resolve. Traffic differs by two orders of magnitude and is exact. Where one axis
cannot separate the options and the other separates them by 130x, size the codebook on the axis that
can tell them apart — and size it to your own corpus, since 32,768 is inherited from the description
of a model trained on 23 billion tokens. On the hydraulic corpus that is not an argument by
analogy: the compacted vocabulary above is **10.8x smaller at equal accuracy**, measured.

**Method notes.** Every figure is a single configuration unless a seed count is given, and a number
without one should be treated as unresolved at the five-point level. Real-data figures are three
seeds with a majority baseline and a label-permutation control; synthetic figures state their own
seed count. The synthetic protocol's split guard, control calibration and batch class balance came
out of an adversarial review that found the first version of that experiment unsound, and the
real-data protocol's presentation-order defect is documented above with the ladder that isolated it.
`examples/scaling_curve.rs` documents what it checks and what it cannot.

## What is NOT here

**There are no published weights.** The tokenizer trains on the hydraulic corpus inside
`examples/hydraulic` and its digest is printed, but no checkpoint is shipped and the corpus is not
redistributed here. Nothing has been compared against a reference implementation's outputs, because
no reference weights were located.

**One corpus is not a claim about sensors in general.** Everything measured on real data comes from
a single hydraulic test rig, five label axes, one split. Three of the four downloaded corpora are
Matlab v5 `.mat` files this crate cannot yet read, which is the next piece of engineering.

**The embedding table trains through a materialized one-hot**, not a native row gather, because the
autograd layer has none. A one-hot `[t, rows]` times the table is gather in the forward pass and
matmul's existing backward is the scatter-add — correct, but it allocates `t x rows` floats per
step, which is tens of megabytes at these sizes and would want a real gather in a deployment. The
cost of NOT training the table was measured on synthetic pairs at 60% against 38% held-out;
`examples/scaling_curve.rs --train-embeddings` is the switch.

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
