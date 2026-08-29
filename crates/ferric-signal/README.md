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
| `synth` | Parameterized physical processes with known ground truth | The generator behind every synthetic figure here; each family is deterministic and reproducible |
| `bench` | Majority baseline, token probe, permutation control | The control must collapse on features that certainly carry the label, and shrink as n grows |
| `caption` | Signal-to-text training, caption vocabulary, compaction | Shared by both corpus ingests, so the protocol is one implementation |
| `mat` | MATLAB v5 reader, the format sensor corpora ship in | **717 channels across three corpora agree with `scipy.io` exactly**; no truncation yields content |
| `inflate` | DEFLATE/zlib decompression | Streams compressed elsewhere, including the dynamic-Huffman branch; the Adler-32 trailer is checked |

**170 tests**, fourteen of them mutation-controlled: each was verified to fail when the line it
names is broken. Several silent defects were caught that way and are documented at the code that
fixes them.

## How to read a number from this crate

Every held-out figure here is reported with four columns beside it, because each one separates a
failure that accuracy alone merges. They cost almost nothing to compute and each has changed a
verdict in this repository at least once.

| column | what it separates | what it has caught here |
|---|---|---|
| **majority** | "at chance" from "predicting the training prior" | a null published as a result: valve at 50.7% against 25% chance reads as twice chance, and is exactly its majority baseline |
| **control** | a real effect from what permuted labels can produce | a probe effect of +9.5 sitting inside a control of +12.2 — not a result — and the same axis clearing its control at a larger held-out size |
| **said** | "answered one thing every time" from "at chance" | a decoder emitting one word for every held-out example, which scores that word's frequency and reads as a weak learner |
| **off-axis** | "wrong word" from "not a word at all" | 27% of one axis's predictions leaving the caption vocabulary, invisible behind an accuracy of 23% |

Two properties of the control are worth stating outright, because both have bitten here. **It is a
function of the held-out size** — the same probe on the same tokens gave +40 points at 10 held-out
examples and +1 at 450 — so it is computed in the run that produced the figure and never quoted
from another. And **worst-of-N grows with N**, so the number of permutations belongs beside it;
five rounds swung five points between draws on this corpus, and twenty is the default.

A figure without a seed count is a single run. At these sample sizes held-out accuracy carries a
standard deviation of several points, so a difference smaller than that is not a measurement.

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
| cooler | 36.0% | 75.0% | 76.0% | +11.0 pt |
| valve | 54.0% | 59.0% | 54.0% | +2.0 pt |
| pump_leak | 55.0% | 62.0% | 60.0% | +3.0 pt |
| accumulator | 33.0% | 48.0% | 54.0% | +6.0 pt |
| stable | 63.0% | 77.0% | 81.0% | +3.0 pt |

Run while the decoder was still emitting a constant, this said the tokens carried `cooler`, `stable`
and `accumulator` well clear of the control and `pump_leak` barely. The repaired decoder landed in
exactly those places, and `pump_leak` is the axis that stays at majority in every cell. **At 100
held-out examples the control sits at +2 to +11 points; at 10 it reached +40**, which is the whole
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

**10.8x fewer rows, every axis within a seed's spread** — and the same 10.8x off the training
traffic, exactly. A trainable embedding here goes through a materialized one-hot `[t, rows]`,
because the autograd layer has no row gather, so the lookup moves `t x rows` floats where a gather
would move `t x d_model`:

| run | positions | rows | one-hot | a gather | ratio |
|---|---|---|---|---|---|
| hydraulic, full codebook | 594 | 32,788 | **77.90 MB** | 0.15 MB | 512x |
| hydraulic, compacted | 594 | 3,046 | 7.24 MB | 0.15 MB | 48x |
| rotating, compacted | 405 | 7,883 | 12.77 MB | 0.10 MB | 123x |

Per optimizer step, before the backward pass touches the same matrix again. The ratio is
`rows / d_model` exactly, so it grows with the vocabulary and not with the sequence — which is what
makes sizing a codebook to its corpus an energy result and not only a modelling one. `embed_cost`
computes it and `cargo run --example token_cost` prints it.

**A second real corpus: rotating machinery, and a fault type that survives an unseen operating
point.** 45 recordings, four accelerometers at 25.6 kHz, a perfectly balanced 3 torques x 15
conditions. 30 one-second windows per recording spread across the whole of it, tokenized by an
**untrained** encoder, read by the same naive-Bayes probe. 1,350 windows, 450 held out.

| axis | classes | chance | majority | within recording | across recording |
|---|---|---|---|---|---|
| fault type | 5 | 20.0% | 33.3% | **67.8%** | **65.6%** |
| torque | 3 | 33.3% | 33.3% | **51.3%** | not askable |
| severity | 5 | 20.0% | 33.3% | **51.8%** | **52.9%** |
| *permutation control* | | | | −0.7 / +4.9 / +1.8 pt | +2.0 / — / +2.4 pt |

The two columns are two different questions. *Within recording* holds out the last third of every
recording, so train and test windows share a machine, a mounting and a day's noise floor — the
protocol most of this literature uses, and it flatters. *Across recording* holds out every 4 Nm
recording: nothing about a held-out recording was seen, and **training contains no 4 Nm at all**.

**Fault type costs 2.2 points to move between them.** A random projection through a discrete
bottleneck separates five fault types at twice the majority baseline on machines and at an
operating point it was never trained on. The torque axis is reported as *not askable* under that
split rather than scored, because its held-out class is absent from training — scoring it would
produce a number that looks like a failure and is a question that was never posed.

At 180 held-out windows the same run gave fault +24.5 with a control of +0.0, and torque +9.5
against a control of **+7.2** — a positive-looking axis that is not one. At 450 the controls fall
to a few points and torque separates properly at +18. That is the control being worth its cost
twice: once by vetoing, once by clearing.

**Every control here is the worst of twenty permutations, and the round count belongs next to the
figure.** Worst-of-N grows with N, which is the direction to err in. Five rounds on the hydraulic
corpus gave +9.0 points on an axis where a different five gave +2.0 — a control that swings five
points between draws is under-sampled, and under-sampling a control makes results look better than
they are.

**The decoder says it in words, on recordings it never saw.** Training a causal decoder over the
hybrid vocabulary to emit a three-word caption — fault, torque, severity — under the *across
recording* split, 900 train / 450 held out, three seeds:

| axis | mean | sd | majority | chance | distinct words emitted |
|---|---|---|---|---|---|
| fault type | **48.2%** | 5.2 | 33.3% | 20.0% | 4.7 of 5 |
| torque | — | — | — | — | not askable under this split |
| severity | 14.1% | 18.4 | 33.3% | 20.0% | 3.3 of 5 |

Fault clears the majority baseline by 14.9 points while emitting nearly the full vocabulary, so it
is answering rather than guessing a constant. **Severity is a null**: below majority, and a
standard deviation of 18.4 across three seeds that ran 0%, 40% and 2% — which is not a weak
signal, it is no signal with a lot of variance.

**And part of the severity failure is not a wrong answer at all.** A cheaper configuration of the
same experiment — 12 windows per recording instead of 30, the same epoch budget four times faster —
reproduces fault at 41.9% ± 4.7 against the same 33.3% majority, and adds the column that says what
the decoder was doing at each position:

| axis | mean | sd | majority | distinct words | not a word for this axis |
|---|---|---|---|---|---|
| fault type | 41.9% | 4.7 | 33.3% | 4.3 of 5 | **0%** |
| torque | 0.0% | 0.0 | not askable | 1.7 of 3 | **0%** |
| severity | 23.0% | 13.1 | 33.3% | 3.0 of 5 | **27%** |

Torque is the clean control for the column: 0.0% accuracy with 0% off-axis is a decoder answering
with legal torque words and never the right one, which is exactly what an axis whose held-out class
was never trained on should look like. Severity is the other failure — **27% of its predictions are
not a severity word at all** — and the two are indistinguishable in an accuracy column.

**The budget, not the architecture, was what stood between the two.** At 250 optimizer steps —
2.2 passes over the training set, where the hydraulic corpus had 6.7 — every axis collapsed to one
word, standard deviation 0.000, sitting exactly on its majority baseline. That is the same
signature as the presentation-order null above, and it would have read as a modelling result. The
probe had already placed the fault signal in the tokens at 65.6%, so the diagnosis was decoder-side
before anything was retrained, and 800 steps confirmed it.

**The probe is still ahead of the decoder.** Run at matched size — 12 windows per recording, 180
held out, same tokens, same splits, three seeds for the decoder:

| axis | split | probe | probe's control | decoder | majority |
|---|---|---|---|---|---|
| fault | within recording | 57.8% | +1.7 | 45.6% ± 2.3 | 33.3% |
| fault | across recording | 53.9% | +0.6 | 41.9% ± 4.7 | 33.3% |
| torque | within recording | 42.8% | **+12.2** | 34.3% ± 2.8 | 33.3% |
| severity | within recording | 51.1% | +5.6 | 33.7% ± 1.8 | 33.3% |
| severity | across recording | 48.3% | +1.7 | 23.0% ± 13.1 | 33.3% |

Three things fall out of reading it as a whole rather than a row at a time.

**The language half generalizes as well as the representation does.** Crossing from within-recording
to across-recording costs the probe 3.9 points on fault and the decoder 3.7. Whatever the decoder is
failing to do, it is not failing to transfer.

**Severity is a decoder failure, not a token failure — and not a generalization failure either.**
The probe finds 17.8 points of severity signal above majority with a control of +5.6, and the
decoder gets none of it: 33.7% against a 33.3% baseline *within recording*, where train and test
share a machine. The tokens carry it and the decoder cannot read it at all.

**A registered prediction about decoder width, refuted.** The probe pools a document into a count
over 6,534 observed (channel, code) pairs; the decoder pools it into `d_model` numbers. At 64 that
is a narrower summary by two orders of magnitude, and it was the obvious candidate for the axis the
probe reads and the decoder cannot. The prediction — written down and committed before the run —
was that doubling `d_model` would lift severity above its majority baseline. Doubling it made
everything worse:

| decoder | fault | torque | severity | distinct words, fault |
|---|---|---|---|---|
| `d_model` 64 | **45.6% ± 2.3** | 34.3% ± 2.8 | 33.7% ± 1.8 | 4.0 of 5 |
| `d_model` 128 | 24.4% ± 6.3 | 34.3% ± 1.7 | 31.9% ± 1.7 | **1.0 of 5** |

**And the `said` column says which kind of worse.** `1.0 of 5` is one word for every held-out
window — the collapse signature of a model that has not been trained enough, not of one that has
overfitted. An overfitted decoder emits varied words and generalizes badly; this one stopped
answering. So the wider model was **under-trained at a step budget that was adequate for the
narrower one**, and the experiment as designed cannot separate "width does not help" from "width
needs more steps."

That is a flaw in the design, and it is worth naming precisely: **matched epochs is not matched
training adequacy when capacity differs.** Both configurations saw the same 3,200 examples over the
same 360 training examples — 8.9 passes each — and that was sufficient for one and not the other.
Holding the obvious quantity fixed was not holding the relevant one fixed.

**Tripling the budget instead does not move severity either, and it costs fault ten points.** Same
width, same corpus, 1,200 optimizer steps against 400:

| budget | fault | torque | severity | distinct words, fault |
|---|---|---|---|---|
| 400 steps (8.9 passes) | **45.6% ± 2.3** | 34.3% ± 2.8 | 33.7% ± 1.8 | 4.0 of 5 |
| 1,200 steps (26.7 passes) | 35.7% ± 2.3 | 37.8% ± 3.5 | 30.9% ± 2.3 | 4.7 of 5 |

**The two failures are now both on the table, and one column tells them apart at the same low
accuracy.** The wide model scored 24.4% emitting *one* word for every window; the long-trained one
scores 35.7% emitting *4.7 of 5*. Under-training collapses variety; overfitting keeps it and loses
accuracy. Read as accuracy alone, both rows say "worse" and nothing else.

So **severity sits on its majority baseline in five of five seed-runs across two budgets**, while
the probe reads 51.1% from the same tokens with a control of +5.6. Neither of the two obvious knobs
touches it. What remains is inductive bias, and that is a more specific claim than "the decoder is
weaker": a counting model fits 6,534 features from 360 examples because that is what counting
models do, and a half-million-parameter decoder given three supervised words per 410-token document
does not. **The probe is not a weaker model that happens to win — it is the right shape for this
much data**, which is a statement about how much sensor-text pairing a sensor-language model needs
rather than about this architecture.

**A third registered prediction, also refuted — and the three together say something specific.**
The probe is naive Bayes over counts, so the obvious remaining explanation was that the decoder
never sees a pooled summary. `--pool` adds the mean of every signal embedding to each caption
position: the summary a counting model works from, handed over directly. It did not help.

| decoder, within recording | fault | torque | severity |
|---|---|---|---|
| baseline | **45.6% ± 2.3** | 34.3% ± 2.8 | 33.7% ± 1.8 |
| `d_model` 128 | 24.4% ± 6.3 | 34.3% ± 1.7 | 31.9% ± 1.7 |
| 3× the budget | 35.7% ± 2.3 | 37.8% ± 3.5 | 30.9% ± 2.3 |
| mean-pooled | 38.7% ± 5.7 | 37.6% ± 3.9 | 34.1% ± 2.6 |

Severity sits on its 33.3% majority in every row. Width collapses the decoder, budget overfits it,
and pooling costs fault seven points while moving severity 0.4.

**Why pooling failed is the useful part.** A mean over embeddings is not a count over codes. Naive
Bayes carries a 6,534-dimensional histogram — one number per observed (channel, code) pair — and
averaging 64-dimensional vectors destroys exactly the per-code identity it runs on. So the probe's
advantage is a representation two orders of magnitude wider than the decoder's *that requires no
training at all*, because counts are given rather than learned.

That closes the line: **the gap is not reachable by architecture at this data volume.** A learned
model has to acquire its pooling, a 6,534-dimensional pooling cannot be acquired from 360 examples,
and the width that would represent it needs a budget that overfits 360 examples first. The three
refutations are consistent and they point at the corpus rather than the design.

The transferable form: **how much sensor-text pairing a sensor-language model needs is set by the
dimensionality of the evidence, not by the size of the model.** That is a claim about this field
rather than about this crate, and it came out of three predictions that were all wrong.

**Torque at this size is inside its own control** — the probe's +9.5 over majority against a
control of **+12.2**. It is not a result at 180 held-out examples. At 450 the same probe gives 51.3%
with a control of +4.9 and it is one. The same axis, the same tokens, and the answer changes with
the held-out size: which is the whole reason the control is printed beside the figure and never
inferred from a previous run.

**A third corpus, three times the recordings, and a much weaker signal.** The three refutations
above point at labelled *recordings* as the binding constraint — windows from one recording are not
independent of each other, recordings are — so `examples/cwru` ingests the CWRU bearing set, whose
condition labels live in the HTML pages beside the data. **All 161 files resolve to a label with
none left over**, parsed from the pages rather than transcribed, and 138 carry both accelerometers.
Three axes: fault type (six classes, including outer-race defects at three clock positions, which
are separate conditions and not a detail), fault diameter, and motor load — balanced 41/40/40/40,
which makes load the across-operating-point split.

**⛔ Sample rate is confounded with the label in this corpus, and a fixed-sample window turns that
into a leak.** 105 recordings are at 12 kHz and 56 at 48 kHz — and **every healthy recording is at
48 kHz, none at 12**. A window of a fixed number of samples therefore spans 2.1 s or 0.5 s
depending on the file, so a model can separate healthy from faulty by detecting the sampling rate
and never look at a bearing. The other classes split roughly 70/30 across rates, so they leak
partially too.

Controlling it — one rate only — is what the two columns below differ by:

| axis | mixed rates | rate-controlled (12 kHz) |
|---|---|---|
| fault | 30.7% vs 25.7% majority, control +3.1 → 1.6× | 28.8% vs 25.0%, control +2.8 → **1.4×** |
| diameter | 40.7% vs 34.3% majority, control +3.8 → 1.7× | 42.7% vs 41.7%, control +2.8 → **inside its control** |

**The diameter result was substantially the confound.** It survives the mixed-rate run and does not
survive the controlled one. Fault survives both and clears its control by 1.4× — present, and not
by much.

So against the rotating corpus's fault axis at 65.6% against a +2.0 control — **sixteen times its
control** — CWRU separates neither axis convincingly once the rate is held fixed. **Three times the
recordings bought a weaker signal, not a stronger one.** Recordings alone were not the binding
constraint; the corpora differ in how separable their labels are from the shape of a vibration
window, and CWRU asks a harder question with a trap in it.

Rate control costs something and the run prints it: at 12 kHz the healthy class does not exist at
all, so `fault` becomes five kinds of defect with no negative case. `--rate all` restores the
contaminated corpus for anyone who wants to see the difference for themselves.

**A confound I introduced nearly turned that into a false negative.** The first CWRU run used a
4,096-sample window, giving documents of 32 tokens against the rotating run's 400 — twelve times
shorter, because the corpora are at different sample rates and I carried a default across. At that
length fault scored 21.4% *below* its 25.7% majority and the honest report would have been "this
corpus separates nothing". Matching the document length moved it to 30.7%, above majority, and
code-space use from 11.7% to 23.7%. Comparing two corpora means matching what the model sees, not
what the flags say.

**Two traps in this corpus, both silent.** The 2 Nm unbalance recordings are spelled
`Unbalalnce` — left alone that is a sixth fault class containing exactly one torque, so "fault
type" and "torque" become partly the same question. And recordings are not the same length: 60 s
for the bearing faults, 120 s for misalignment and unbalance, 300 s for normal. Taking every
non-overlapping window would hand `Normal` five times the examples of `BPFI` and make the class
balance an artifact of recording length; a fixed number of windows per recording keeps the design
balanced. Both are handled, and the normalisation count is printed.

**Reading the corpora at all: a MATLAB v5 reader, checked against another implementation.** Three
of the four public sensor corpora this crate was pointed at ship as `.mat`, and none could be
opened. `mat` reads all three: both byte orders, both tag forms, numeric and char and struct and
cell including nesting, and zlib-compressed elements through `inflate`.

| corpus | files | series | agreement with `scipy.io` |
|---|---|---|---|
| CWRU bearings | 161 | 518 | **518 / 518** |
| Rotating machinery (vibration) | 1 sampled | 61 | **61 / 61** |
| Wind-turbine drivetrain (compressed) | 1 sampled | 138 | **138 / 138** |

Agreement is checked against a table written by a different implementation — a parser validated
only against its own output agrees with its own bugs. `--check` compares length, first sample, last
sample and the sum of every series, and exits non-zero on any disagreement. One 92 MB compressed
file yields 180.7M samples in 1.8 s.

**Four things the real corpora caught that no fixture would have.**

*Compressed elements are not padded.* Every other element is padded to an eight-byte boundary. A
compressed one is followed immediately by the next. The wind corpus's first element is 1,594 bytes
and the next tag is at 1,730, not 1,736 — pad it and a 92 MB file reads as one variable followed by
a garbage tag.

*`channels()` first required at most one dimension above 1*, which is right for `[N, 1]` and wrong
for every multi-channel recording. On the rotating corpus that returned 57 metadata scalars and
**none of its 6.1M samples** — parsed successfully, and empty. The recording is `[1536000, 4]`, four
accelerometers for sixty seconds. MATLAB is column-major, so each column is already contiguous.

*A variable this reader cannot decode is stepped over and recorded, not fatal.* The wind corpus
stores an `mxOPAQUE` — a MATLAB object — in the middle of its channel list, with twelve channels
including the tachometer after it. Failing the file to avoid a MATLAB object would throw those away;
silently returning fewer variables would be worse. So `MatFile::skipped` says what was stepped over
and why, and a file where *nothing* decoded is an error rather than an empty file.

*MATLAB's unnamed function-workspace element is interpreter state, not a recording.* It arrived as a
1×1152 unnamed channel. It stays visible in `vars` under the conventional name and is kept out of
`channels()`.

And **the reader refuses exactly the 15 CWRU files `scipy` refuses**, all truncated downloads,
naming the byte where each ran short — a half-file tokenizes perfectly well and would otherwise
have been scored as data.

**`inflate` is written out rather than depended on.** This crate carries two path dependencies and
`pollster`, and wrote its own SHA-256 for the same reason. The Adler-32 trailer is checked, which
is what makes a silent partial decode impossible: a sensor channel that is wrong in its second half
still tokenizes.

**One tokenizer across four corpora, at the published size, and
[released](https://github.com/dcharlot-physicalai-bmi/ferric/releases/tag/ferric-signal-tokenizer-v0.1).** `examples/universal` trains a single encoder, FSQ bottleneck and decoder on four
independent sensor corpora at once: **9,576,448 parameters, 0.8% from the published 9.5M**. Four
rigs, four laboratories, sampling rates spanning four orders of magnitude, and one corpus that is
not vibration at all. Held out by *recording*, reported per corpus, trained round-robin so corpus
size does not decide whose gradients win.

| corpus | machine | held-out SNR | strongest 15-bit baseline | margin |
|---|---|---|---|---|
| hydraulic | test rig, pressures and flows | **9.2 dB** | 5.2 dB | **+4.0** |
| wind | turbine nacelle | **7.1 dB** | 5.1 dB | **+2.0** |
| rotating | gearbox rig, 25.6 kHz | 2.1 dB | 1.4 dB | **+0.7** |
| CWRU | bearing stand, 12–48 kHz | 1.6 dB | 1.6 dB | 0.0 |

The baseline spends *exactly* the same bit rate and learns nothing: 15 bits naming which of 128 DCT
coefficients is largest and quantizing it, at whichever of four matched-budget coders is strongest
on that corpus. Model and baseline come from separate runs over the same deterministic corpus load;
the held-out window counts match exactly, which is what makes them comparable, and both are printed.

**The earlier "a trained tokenizer loses to one DCT coefficient" is retired, and the reason is data
volume rather than bit rate.** The first version of this table gave CWRU 1.1 dB against a baseline
that beat it. Feeding the vibration corpora roughly ten times the training windows — 212 → 1,496 for
CWRU and 192 → 1,088 for rotating, same architecture, same rate, same 24,000 steps — moved CWRU to
1.6 dB and rotating to 2.1, converting a loss into a tie and a tie into a win. Code-space use went
from 16.1% to 28.5% over the same change.

That was the experiment named as the one that would separate the two explanations, and it separated
them: the vibration corpora were starved, not rate-limited.

**What survives is smaller and more specific.** High-rate vibration is still close to the limit of
what 15 bits per 128 samples carries — the learned tokenizer exceeds an untrained single coefficient
by 0.7 dB and 0.0 dB there, against 4.0 and 2.0 on the smoother corpora. Four rigs, one set of
weights, and the same bit budget buys very different fidelity depending on the physics. That is the
honest content of "universal": it works everywhere, and what it is worth varies by an order of
magnitude.

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

**The published weights reproduce their own table.** The four-corpus tokenizer above is released as
[`ferric-signal-tokenizer-v0.1`](https://github.com/dcharlot-physicalai-bmi/ferric/releases/tag/ferric-signal-tokenizer-v0.1)
— both towers, 38 MB, digest `2cd72ffc…`. `--load` rebuilds it, re-tokenizes the held-out set and
reconstructs from it, returning every SNR above to the decimal beside the same matched-bit-rate
baseline. Those are figures the artifact produces, not figures the training run remembered.

**No language model ships.** The decoder in that file reconstructs a signal from tokens, which is
what makes the SNR checkable; it does not produce words. And nothing here has been compared against
a reference implementation's outputs, because no reference weights were located.

**One corpus is not a claim about sensors in general.** Everything measured on real data comes from
a single hydraulic test rig, five label axes, one split. The other corpora now open (see below) but
nothing has been trained on them.

**`mat` does not read every class.** `mxSPARSE`, `mxOBJECT`, `mxOPAQUE` and complex arrays are
refused by name. `inflate` decompresses and does not compress.

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

**This is bandwidth-bound at every window, and the weight-only figure says otherwise by 847x.**
The intensity quoted here was 8.1 FLOP per *weight* byte against the ~100–300 a modern part needs
to be compute-bound. That figure omitted the attention score matrix, which `tower::attention_v`
materializes three times per layer at `[n_heads, t, t]` and which is quadratic where the weights are
flat:

| window | FLOP per weight byte | FLOP per byte actually moved | score bytes ÷ weight bytes |
|---|---|---|---|
| 16 | 8.1 | **8.04** | 0.003 |
| 256 | 145.5 | **79.7** | 0.83 |
| 282 | 162.3 | **81.1** | 1.00 |
| 512 | 326.3 | **75.8** | 3.3 |
| 8192 | 22,141.4 | **26.1** | 846 |

At 16 patches the two agree to 1%, which is why the published figure stands where it was quoted. At
8192 the weight-only number says 22,141 FLOP per byte — comfortably compute-bound, and wrong by
**847x**. The honest figure is 26.1, still far below the 100–300 threshold.

**The honest intensity is not monotonic.** It peaks at about 81 FLOP per byte near a window of 282
patches — exactly where score traffic overtakes weight traffic — and falls away in both directions,
never reaching compute-bound anywhere. The weight-only figure rises without limit and would tell you
to use the longest window available. That is the opposite of what the traffic does.

So the original conclusion survives and its scope was wrong: on a sensor node the bill is bytes
moved, at *every* window — but past a few hundred patches those bytes are the score matrix, not the
weights, and quantizing weights stops helping. `TokenCost::flops_per_byte` is the figure to use;
`flops_per_weight_byte` is kept, and now documents that it answers a narrower question.

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
