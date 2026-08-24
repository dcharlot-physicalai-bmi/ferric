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

Everything below is **synthetic**: five parameterized physical process families from `synth`, whose
labels this crate's own generator writes. Nothing has been trained on real sensor data.

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

**What follows for sizing a codebook.** Accuracy differences across this range are below what ~30
held-out examples and five seeds resolve. Traffic differs by two orders of magnitude and is exact.
Where one axis cannot separate the options and the other separates them by 130x, size the codebook
on the axis that can tell them apart — and size it to your own corpus, since 32,768 is inherited
from the description of a model trained on 23 billion tokens.

**Method notes.** Everything above is a single configuration unless a seed count is given. Numbers
without a seed count are one run and should be treated as unresolved at the five-point level. The
protocol's split guard, control calibration and batch class balance came out of an adversarial
review that found the first version of the experiment unsound; `examples/scaling_curve.rs` documents
what it checks and what it cannot.

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
