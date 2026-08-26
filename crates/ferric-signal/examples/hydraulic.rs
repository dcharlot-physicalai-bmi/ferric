//! Ingest a real multi-rate sensor corpus: UCI Condition Monitoring of Hydraulic Systems.
//!
//!   cargo run -p ferric-signal --example hydraulic --release -- --data <dir> [--cycles N]
//!         [--tok-steps N] [--train] [--steps N] [--seeds N]
//!
//! Dataset: Helwig, Pignanelli & Schütze, "Condition monitoring of a complex hydraulic system
//! using multivariate statistics", I2MTC 2015. UCI Machine Learning Repository, CC BY 4.0.
//! Not redistributed here; point `--data` at your own copy.
//!
//! ## Why this corpus, and what it exercises that synthetic data cannot
//!
//! 2,205 cycles of a hydraulic test rig, 17 channels recorded at **four different sampling rates**
//! in the same recording — six pressures and motor power at 100 Hz, two flows at 10 Hz, four
//! temperatures, vibration and three virtual channels at 1 Hz. Every synthetic signal this crate
//! has been measured on was single-rate, so the multi-rate path has never been exercised.
//!
//! **A patch is one second of its own channel.** That keeps a patch physically comparable across
//! rates rather than comparable in samples, so a 100 Hz channel yields 100-sample patches and a
//! 10 Hz channel yields 10-sample patches, and both cover the same second of machine time. The
//! 1 Hz channels get ten-second patches, because a one-sample patch is a linear layer wearing a
//! transformer's clothes.
//!
//! Each channel becomes its own run inside one sequence, separated by the vocabulary's channel
//! marker — which is what that marker was added for, and is here fed real multi-channel data for
//! the first time.
//!
//! ## The split is decided before anything is trained
//!
//! **Cycles are sampled with a stride, never as a prefix.** The corpus is ordered by experimental
//! condition: the first 100 cycles all carry cooler=3, valve=100, leak=0, accumulator=130,
//! stable=1 — one condition, five identical labels, a single-class corpus wearing the shape of a
//! hundred examples. Striding across all 2,205 cycles is what makes the label axes vary.
//!
//! The held-out quarter is fixed by cycle position before the tokenizer sees a sample, and the
//! **tokenizer trains on the training cycles only**. Tokenizer training is unsupervised, which
//! makes it tempting to fit it on everything — but a tokenizer fitted to held-out signals has
//! already encoded them, and the language model downstream inherits that. Transduction is still
//! leakage when the number being reported is held-out accuracy.
//!
//! ## What this example claims
//!
//! With `--tok-steps 0` (the default) the encoder is random, and the structure counts describe
//! what an untrained tokenizer does to real multi-rate data. With `--tok-steps N` the same
//! corpus is tokenized **twice** — once by the random encoder, once by one trained through the
//! discrete bottleneck on the training cycles — and every downstream number is reported for both
//! arms at identical seeds. That A/B is the only thing that can attribute a change to the
//! tokenizer rather than to the run.

use ferric_core::Context;
use ferric_signal::{
    cross_entropy, decoder_forward_var, embed_var, forward_var, lm_forward_var, mse,
    straight_through, DecoderWeights, EncoderConfig, EncoderWeights, Example, Fsq, HybridVocab,
    Patcher, RevIn, SensorLm, Sequencer, Span, Weights,
};
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::sync::Arc;

/// (name, samples per cycle, samples per patch). Patch length is one second of machine time where
/// that is at least eight samples, and ten seconds otherwise.
const CHANNELS: &[(&str, usize, usize)] = &[
    ("PS1", 6000, 100), ("PS2", 6000, 100), ("PS3", 6000, 100),
    ("PS4", 6000, 100), ("PS5", 6000, 100), ("PS6", 6000, 100),
    ("EPS1", 6000, 100),
    ("FS1", 600, 10), ("FS2", 600, 10),
    ("TS1", 60, 10), ("TS2", 60, 10), ("TS3", 60, 10), ("TS4", 60, 10),
    ("VS1", 60, 10), ("CE", 60, 10), ("CP", 60, 10), ("SE", 60, 10),
];

const LABELS: &[&str] = &["cooler", "valve", "pump_leak", "accumulator", "stable"];

/// Tokenizer tower shape. Small on purpose: this is the resolution the corpus is tokenized at in
/// both arms, so it is held fixed and only the WEIGHTS differ between them.
fn tok_cfg(patch: usize) -> EncoderConfig {
    EncoderConfig { patch_len: patch, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Deterministic Fisher-Yates, so "shuffled" is reproducible and a reported number can be rerun.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

/// Read one channel file: one row per cycle, tab-separated samples.
///
/// STREAMED line by line rather than read whole. A 100 Hz channel is 87 MB on disk and about 93 MB
/// as a `String`, and holding several of those alongside their parsed `f32` copies was enough to
/// get this example OOM-killed (exit 137) on a machine that was also holding several GB of freshly
/// downloaded corpus in page cache. `BufReader` keeps one line live at a time.
///
/// A row whose width does not match the documented width is an ERROR, not a truncation: a channel
/// silently short by one sample would shift every patch after it and still tokenize.
fn read_channel(dir: &str, name: &str, want_cols: usize, want: &[usize]) -> Result<Vec<Vec<f32>>, String> {
    let path = format!("{dir}/{name}.txt");
    let f = std::fs::File::open(&path).map_err(|e| format!("{path}: {e}"))?;
    let wanted: HashSet<usize> = want.iter().copied().collect();
    let last = want.iter().copied().max().unwrap_or(0);
    let mut rows = Vec::with_capacity(want.len());
    for (i, line) in BufReader::new(f).lines().enumerate() {
        if i > last {
            break;
        }
        if !wanted.contains(&i) {
            continue;
        }
        let line = line.map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        let vals: Result<Vec<f32>, _> = line.split_whitespace().map(|v| v.parse::<f32>()).collect();
        let vals = vals.map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        if vals.is_empty() {
            continue;
        }
        if vals.len() != want_cols {
            return Err(format!("{path}:{}: {} samples, expected {want_cols}", i + 1, vals.len()));
        }
        rows.push(vals);
    }
    Ok(rows)
}

/// One word per (axis, value) pair, plus a terminator at id 0. A caption is then five words and an
/// end marker, and the five axes stay separable at scoring time.
fn build_words(labels: &[Vec<i32>]) -> (Vec<String>, Vec<Vec<u32>>) {
    let mut words = vec!["<end>".to_string()];
    let mut per_axis: Vec<Vec<i32>> = vec![Vec::new(); LABELS.len()];
    for row in labels {
        for (a, &v) in row.iter().enumerate() {
            if !per_axis[a].contains(&v) {
                per_axis[a].push(v);
            }
        }
    }
    for a in 0..LABELS.len() {
        per_axis[a].sort_unstable();
        for v in &per_axis[a] {
            words.push(format!("{}={}", LABELS[a], v));
        }
    }
    let idx = |a: usize, v: i32| -> u32 {
        let mut base = 1usize;
        for k in 0..a {
            base += per_axis[k].len();
        }
        (base + per_axis[a].iter().position(|&x| x == v).unwrap()) as u32
    };
    let caps: Vec<Vec<u32>> = labels
        .iter()
        .map(|row| {
            let mut c: Vec<u32> = (0..LABELS.len()).map(|a| idx(a, row[a])).collect();
            c.push(0);
            c
        })
        .collect();
    (words, caps)
}

/// Train one tokenizer — encoder, FSQ bottleneck, decoder — to reconstruct its own patches.
///
/// **A training step is one channel-cycle, not a batch of them.** The encoder is bidirectional
/// over whatever sequence it is handed, so pooling cycles into one tensor would let patches from
/// one cycle attend to patches from another; the tokens would then depend on the batch. That is
/// the bug this example already shipped and corrected once, and the fix has to hold on the
/// training path too, or training and inference would see different receptive fields.
///
/// Returns the trained encoder, the decoder's parameters (needed to measure reconstruction, and
/// nothing else — the decoder is a training-time artifact) and the loss at the first and last step.
fn train_tokenizer(
    ctx: &Arc<Context>,
    cfg: EncoderConfig,
    q: &Fsq,
    blocks: &[&Vec<f32>],
    steps: usize,
    seed: u64,
) -> Result<(EncoderWeights, Vec<Tensor>, f32, f32, String), String> {
    let enc = EncoderWeights::deterministic(ctx, cfg, seed).map_err(|e| format!("{e:?}"))?;
    let dec = DecoderWeights::deterministic(ctx, cfg, seed ^ 0x5DEE).map_err(|e| format!("{e:?}"))?;
    let n_enc = enc.params_flat().len();
    let mut params: Vec<Tensor> = enc.params_flat().into_iter().chain(dec.params_flat()).collect();
    let mut opt = Adam::new(&params, 2e-3);

    let order = shuffled(blocks.len(), seed ^ 0xA11CE);
    let (mut first, mut last) = (0.0f32, 0.0f32);
    for step in 0..steps {
        let b = blocks[order[step % order.len()]];
        let t = b.len() / cfg.patch_len;
        let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
        let x = Var::leaf(Tensor::from_vec(ctx, b, &[t, cfg.patch_len]));
        let z = forward_var(ctx, cfg, &vars[..n_enc], &x).map_err(|e| format!("{e:?}"))?;
        let zq = straight_through(ctx, &z, q);
        let recon = decoder_forward_var(ctx, cfg, &vars[n_enc..], &zq).map_err(|e| format!("{e:?}"))?;
        let loss = mse(&recon, &x);
        loss.backward();
        let l = pollster::block_on(loss.value().to_vec())[0];
        if step == 0 {
            first = l;
        }
        // A diverged tokenizer that keeps running produces tokens, and every downstream number
        // would then be reported off a NaN encoder. Stop where it happened.
        if !l.is_finite() {
            return Err(format!("tokenizer loss became {l} at step {step}"));
        }
        last = l;
        let grads: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
        opt.step(&mut params, &grads);
    }

    // Round-trip the trained encoder through the on-disk format rather than keeping the live
    // tensors: it checks every shape by name, and it means the weights that tokenize are weights
    // that could have been loaded from a file.
    let names = EncoderWeights::tensor_names(cfg.n_layers);
    let refs: Vec<(&str, &Tensor)> =
        names.iter().map(|s| s.as_str()).zip(params[..n_enc].iter()).collect();
    let w = Weights::from_tensors(&refs);
    let digest = w.digest();
    let enc = EncoderWeights::from_weights(ctx, cfg, &w).map_err(|e| format!("{e:?}"))?;
    Ok((enc, params[n_enc..].to_vec(), first, last, digest))
}

/// Reconstruction error of a trained tokenizer over a set of blocks, and the variance of those
/// blocks, so the error can be reported as a share of the signal rather than in bare units.
fn recon_error(
    ctx: &Arc<Context>,
    cfg: EncoderConfig,
    q: &Fsq,
    enc: &EncoderWeights,
    dec_params: &[Tensor],
    blocks: &[&Vec<f32>],
) -> (f64, f64) {
    let (mut se, mut n, mut sum, mut sumsq) = (0.0f64, 0usize, 0.0f64, 0.0f64);
    let dvars: Vec<Var> = dec_params.iter().cloned().map(Var::leaf).collect();
    for b in blocks {
        let t = b.len() / cfg.patch_len;
        let lat = pollster::block_on(
            enc.forward(ctx, &Tensor::from_vec(ctx, b, &[t, cfg.patch_len])).unwrap().to_vec(),
        );
        let deq: Vec<f32> = (0..t)
            .flat_map(|i| {
                let c = q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap();
                q.dequantize(&c).unwrap()
            })
            .collect();
        let zq = Var::leaf(Tensor::from_vec(ctx, &deq, &[t, cfg.latent_dim]));
        let r = pollster::block_on(
            decoder_forward_var(ctx, cfg, &dvars, &zq).unwrap().value().to_vec(),
        );
        for (a, e) in r.iter().zip(b.iter()) {
            se += ((a - e) as f64) * ((a - e) as f64);
            sum += *e as f64;
            sumsq += (*e as f64) * (*e as f64);
            n += 1;
        }
    }
    let mean = sum / n as f64;
    (se / n as f64, sumsq / n as f64 - mean * mean)
}

/// Compact the signal vocabulary down to the codes the TRAINING cycles actually use.
///
/// The full codebook is 32,768 rows, and this corpus visits a few thousand of them. Every unvisited
/// row is an untrained embedding AND a class in the output softmax, so the decoder spends most of
/// its capacity and all of its traffic on codes that cannot occur. The README's own conclusion from
/// the synthetic sweep was to size the codebook to the corpus rather than inherit 32,768 from a
/// model trained on 23 billion tokens; this is that, applied to real data.
///
/// **Built from the training cycles only.** A vocabulary fitted to every code the held-out cycles
/// contain has already been told something about them. Held-out codes outside the training set map
/// to one reserved id, and the share of held-out tokens that lands there is returned, because a
/// compaction that silently discards a third of the held-out signal is not a free win.
fn compact(
    per_cycle: &[Vec<Vec<u32>>],
    train_idx: &[usize],
    held_idx: &[usize],
) -> (Vec<Vec<Vec<u32>>>, u32, f64) {
    let mut seen: Vec<u32> = train_idx
        .iter()
        .flat_map(|&i| per_cycle[i].iter().flatten().copied())
        .collect::<HashSet<u32>>()
        .into_iter()
        .collect();
    seen.sort_unstable();
    let map: HashMap<u32, u32> =
        seen.iter().enumerate().map(|(k, &c)| (c, k as u32)).collect();
    let unk = seen.len() as u32;
    let remapped: Vec<Vec<Vec<u32>>> = per_cycle
        .iter()
        .map(|chans| {
            chans
                .iter()
                .map(|run| run.iter().map(|c| map.get(c).copied().unwrap_or(unk)).collect())
                .collect()
        })
        .collect();
    let (mut miss, mut total) = (0usize, 0usize);
    for &i in held_idx {
        for run in &per_cycle[i] {
            for c in run {
                total += 1;
                if !map.contains_key(c) {
                    miss += 1;
                }
            }
        }
    }
    (remapped, unk + 1, miss as f64 / total.max(1) as f64 * 100.0)
}

/// What one seed produced: held-out accuracy per axis, and whether the model said anything.
struct SeedResult {
    acc: Vec<f64>,
    /// Distinct words the model actually emitted at each axis position across the held-out set.
    distinct: Vec<usize>,
}

/// Train signal-to-text for one seed.
#[allow(clippy::too_many_arguments)]
fn train_seed(
    ctx: &Arc<Context>,
    seq: &Sequencer,
    rows_tokens: &[Vec<Vec<u32>>],
    caps: &[Vec<u32>],
    train_idx: &[usize],
    held_idx: &[usize],
    steps: usize,
    batch: usize,
    lm_cfg: EncoderConfig,
    seed: u64,
) -> SeedResult {
    let rows = seq.embedding_rows();
    let lm_cfg = lm_cfg;
    let cfg = lm_cfg;
    let lm = SensorLm::deterministic(ctx, cfg, rows, seed).unwrap();
    // The embedding table trains with the rest: a signal code the corpus never visited is an
    // untrained row, and this corpus visits only a fraction of the code space.
    let mut params: Vec<Tensor> = std::iter::once(lm.embed.clone()).chain(lm.params_flat()).collect();
    let mut opt = Adam::new(&params, 2e-3);

    let build = |i: usize| -> Example {
        let prompt = seq.encode(&[Span::Signal(rows_tokens[i].clone())]).unwrap();
        let target = seq.encode(&[Span::Text(caps[i].clone())]).unwrap();
        let target_from = prompt.len();
        let mut tokens = prompt;
        tokens.extend(target);
        Example { tokens, target_from }
    };

    // GRADIENT ACCUMULATION over `batch` examples per optimizer step.
    //
    // `lm_forward_var` takes `[t, d]`, with no batch axis, and giving it two examples stacked
    // would let one attend to the other — the same coupling the tokenizer path already had to
    // correct. Summing gradients instead leaves every example encoded on its own and only
    // averages what the optimizer sees. At batch 1 the gradient from a single 600-token sequence
    // with a six-word target is noisy enough that the cheapest descent direction is the label
    // marginal, which is exactly the constant-output failure the `distinct` column below counts.
    let mut order = shuffled(train_idx.len(), seed ^ 0x5EED);
    let mut cursor = 0usize;
    for _ in 0..steps {
        let mut acc: Vec<Tensor> = Vec::new();
        for _ in 0..batch.max(1) {
            if cursor >= order.len() {
                order = shuffled(train_idx.len(), seed ^ 0x5EED ^ cursor as u64);
                cursor = 0;
            }
            let i = train_idx[order[cursor]];
            cursor += 1;
            let e = build(i);
            let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
            let emb = embed_var(ctx, &vars[0], &e.tokens).unwrap();
            let logits = lm_forward_var(ctx, cfg, &vars[1..], &emb).unwrap();
            let loss = cross_entropy(ctx, &logits, &e.tokens, e.target_from, rows).unwrap();
            loss.backward();
            let grads: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
            acc = if acc.is_empty() {
                grads
            } else {
                acc.iter()
                    .zip(grads.iter())
                    .map(|(a, g)| Var::leaf(a.clone()).add(&Var::leaf(g.clone())).value().clone())
                    .collect()
            };
        }
        let inv = 1.0 / batch.max(1) as f32;
        let grads: Vec<Tensor> = acc
            .iter()
            .map(|a| {
                let sc = Var::leaf(Tensor::from_vec(ctx, &[inv], &[1])).broadcast_to(&a.shape);
                Var::leaf(a.clone()).mul(&sc).value().clone()
            })
            .collect();
        opt.step(&mut params, &grads);
    }

    // Score each axis at its own caption position: the model sees the signal plus the caption
    // words before this axis, and must produce this axis's word.
    let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
    let mut right = vec![0usize; LABELS.len()];
    // What the model EMITTED, not just whether it was right. A model that answers the same word
    // for every held-out cycle scores the frequency of that word and looks like a weak learner in
    // an accuracy column; counting distinct predictions says outright that it never varied.
    let mut said: Vec<HashSet<usize>> = vec![HashSet::new(); LABELS.len()];
    for &i in held_idx {
        let e = build(i);
        for a in 0..LABELS.len() {
            let upto = e.target_from + a;
            let emb = embed_var(ctx, &vars[0], &e.tokens[..upto]).unwrap();
            let logits =
                pollster::block_on(lm_forward_var(ctx, cfg, &vars[1..], &emb).unwrap().value().to_vec());
            let last = (upto - 1) * rows as usize;
            let row = &logits[last..last + rows as usize];
            let best = row.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
            said[a].insert(best);
            if best == seq.vocab().text(caps[i][a]).unwrap() as usize {
                right[a] += 1;
            }
        }
    }
    SeedResult {
        acc: right.iter().map(|&r| r as f64 / held_idx.len() as f64 * 100.0).collect(),
        distinct: said.iter().map(|s| s.len()).collect(),
    }
}

/// Multinomial naive Bayes over channel-tagged code counts: does the TOKEN STREAM carry the label
/// at all, independent of whether a language model can read it?
///
/// This is the instrument that makes a null attributable. If signal-to-text sits on the majority
/// baseline, there are two very different reasons — the tokens do not encode the fault, or they do
/// and the decoder cannot extract it — and an accuracy column cannot tell them apart. A classifier
/// with no capacity to speak, reading the same tokens, separates them.
///
/// A feature is (channel, code), not code alone: the same code from a pressure channel and from a
/// temperature channel is not the same evidence.
fn nb_probe(
    per_cycle: &[Vec<Vec<u32>>],
    labels: &[Vec<i32>],
    train_idx: &[usize],
    held_idx: &[usize],
    axis: usize,
) -> f64 {
    let feat = |c: usize, code: u32| -> u64 { (c as u64) << 32 | code as u64 };
    let classes: Vec<i32> = {
        let mut v: Vec<i32> = train_idx.iter().map(|&i| labels[i][axis]).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    // Per class: total count of each feature, and the class's total mass.
    let mut counts: Vec<HashMap<u64, f64>> = vec![HashMap::new(); classes.len()];
    let mut totals = vec![0.0f64; classes.len()];
    let mut vocab: HashSet<u64> = HashSet::new();
    for &i in train_idx {
        let k = classes.iter().position(|&v| v == labels[i][axis]).unwrap();
        for (c, run) in per_cycle[i].iter().enumerate() {
            for &code in run {
                *counts[k].entry(feat(c, code)).or_insert(0.0) += 1.0;
                totals[k] += 1.0;
                vocab.insert(feat(c, code));
            }
        }
    }
    let v = vocab.len() as f64;
    let alpha = 1.0f64;
    let mut right = 0usize;
    for &i in held_idx {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for k in 0..classes.len() {
            let denom = totals[k] + alpha * v;
            let mut lp = 0.0f64;
            for (c, run) in per_cycle[i].iter().enumerate() {
                for &code in run {
                    let n = counts[k].get(&feat(c, code)).copied().unwrap_or(0.0);
                    lp += ((n + alpha) / denom).ln();
                }
            }
            if lp > best.0 {
                best = (lp, k);
            }
        }
        if classes[best.1] == labels[i][axis] {
            right += 1;
        }
    }
    right as f64 / held_idx.len() as f64 * 100.0
}

/// Majority-class accuracy: the most frequent TRAIN value, scored on HELD OUT.
fn majority(labels: &[Vec<i32>], train_idx: &[usize], held_idx: &[usize], axis: usize) -> f64 {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &i in train_idx {
        *counts.entry(labels[i][axis]).or_insert(0) += 1;
    }
    let maj = counts.iter().max_by_key(|(_, &c)| c).map(|(&v, _)| v).unwrap_or(0);
    held_idx.iter().filter(|&&i| labels[i][axis] == maj).count() as f64 / held_idx.len() as f64 * 100.0
}

/// Tokenize every cycle of every channel with a given set of encoders.
fn tokenize(
    ctx: &Arc<Context>,
    q: &Fsq,
    encoders: &[(usize, EncoderConfig, EncoderWeights)],
    patches: &[Vec<Vec<f32>>],
    n_cycles: usize,
) -> (Vec<Vec<Vec<u32>>>, HashSet<u32>) {
    let mut per_cycle: Vec<Vec<Vec<u32>>> = vec![Vec::new(); n_cycles];
    let mut all_codes: HashSet<u32> = HashSet::new();
    for (ch, &(_, _, patch)) in CHANNELS.iter().enumerate() {
        let (_, cfg, enc) = encoders.iter().find(|(p, _, _)| *p == patch).unwrap();
        // ONE FORWARD PER CYCLE, and that is not an optimisation to be undone. Batching every
        // cycle of a channel into a single tensor was faster and WRONG: the encoder's attention
        // is over the whole sequence it is given, so a 400-cycle batch let patches from one cycle
        // attend to patches from another. Tokenization then depends on what else happened to be
        // in the batch, which breaks per-cycle determinism and quietly couples held-out cycles to
        // training ones. It surfaced only because 24,000 patches squared exceeded the 4 GB buffer
        // limit; at 100 cycles it would have run and been silently wrong.
        for (ci, block) in patches[ch].iter().enumerate() {
            let t = block.len() / patch;
            let lat = pollster::block_on(
                enc.forward(ctx, &Tensor::from_vec(ctx, block, &[t, patch])).unwrap().to_vec(),
            );
            let codes: Vec<u32> = (0..t)
                .map(|i| {
                    q.to_index(&q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap())
                        .unwrap()
                })
                .collect();
            all_codes.extend(&codes);
            per_cycle[ci].push(codes);
        }
    }
    (per_cycle, all_codes)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = flag(&args, "--data") else {
        eprintln!("usage: --data <dir containing PS1.txt .. profile.txt> [--cycles N] [--tok-steps N] [--train]");
        std::process::exit(2);
    };
    let max_cycles: usize = flag(&args, "--cycles").and_then(|v| v.parse().ok()).unwrap_or(200);
    let tok_steps: usize = flag(&args, "--tok-steps").and_then(|v| v.parse().ok()).unwrap_or(0);

    // Labels first: if the profile does not parse there is no point reading 663 MB of signal.
    let profile = match std::fs::read_to_string(format!("{dir}/profile.txt")) {
        Ok(t) => t,
        Err(e) => { eprintln!("error: {dir}/profile.txt: {e}"); std::process::exit(1); }
    };
    let all_labels: Vec<Vec<i32>> = profile
        .lines()
        .filter_map(|l| {
            let v: Vec<i32> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            (v.len() == LABELS.len()).then_some(v)
        })
        .collect();
    // Stride, not prefix. See the module docs: a prefix is one experimental condition.
    let stride = (all_labels.len() / max_cycles.max(1)).max(1);
    let picked: Vec<usize> = (0..all_labels.len()).step_by(stride).take(max_cycles).collect();
    let labels: Vec<Vec<i32>> = picked.iter().map(|&i| all_labels[i].clone()).collect();
    println!("\nUCI HYDRAULIC  {} of {} cycles, every {stride}th (ordered corpus: a prefix would be one condition)",
             labels.len(), all_labels.len());

    // THE SPLIT IS FIXED HERE, before a tokenizer or a language model sees a sample. Split by
    // stride phase, not contiguously: the corpus is ordered by condition, so a contiguous split
    // would put whole conditions on one side of it.
    let n = labels.len();
    let held_idx: Vec<usize> = (0..n).filter(|i| i % 4 == 3).collect();
    let train_idx: Vec<usize> = (0..n).filter(|i| i % 4 != 3).collect();

    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example needs one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);
    let q = Fsq::new(vec![8u32; 5]).unwrap();

    // ---- read and normalize every channel once ----
    println!("  reading {} channels...", CHANNELS.len());
    let mut patches: Vec<Vec<Vec<f32>>> = Vec::with_capacity(CHANNELS.len());
    for &(name, cols, patch) in CHANNELS {
        let rows = match read_channel(&dir, name, cols, &picked) {
            Ok(r) => r,
            Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
        };
        if rows.len() != n {
            eprintln!("error: {name} gave {} cycles, expected {n}", rows.len());
            std::process::exit(1);
        }
        let patcher = Patcher::contiguous(patch).unwrap();
        // Per-cycle, per-channel normalization: a pressure channel and a temperature channel
        // differ by orders of magnitude in units, and the model should see shape.
        let blocks: Vec<Vec<f32>> = rows
            .iter()
            .map(|raw| {
                let rev = RevIn::fit(raw, 1).unwrap();
                patcher.patchify(&rev.apply(raw).unwrap()).unwrap()
            })
            .collect();
        patches.push(blocks);
    }

    // One encoder per distinct patch length, so every channel at a given resolution shares a
    // tokenizer and the code counts are comparable across channels.
    let lengths: Vec<usize> = {
        let mut v: Vec<usize> = CHANNELS.iter().map(|&(_, _, p)| p).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let untrained: Vec<(usize, EncoderConfig, EncoderWeights)> = lengths
        .iter()
        .map(|&p| (p, tok_cfg(p), EncoderWeights::deterministic(&ctx, tok_cfg(p), 7).unwrap()))
        .collect();
    println!("  {} channels, {} distinct patch lengths, codebook {} codes",
             CHANNELS.len(), lengths.len(), q.codebook_size());
    println!("  split: {} train / {} held out, fixed by cycle position\n", train_idx.len(), held_idx.len());

    // ---- optionally train a tokenizer per patch length, on TRAIN cycles only ----
    let mut trained: Option<Vec<(usize, EncoderConfig, EncoderWeights)>> = None;
    if tok_steps > 0 {
        println!("  TOKENIZER TRAINING  {tok_steps} steps per patch length, train cycles only");
        println!("  {:>6} {:>8} {:>10} {:>12} {:>12} {:>10}  {}",
                 "patch", "blocks", "step 0", "final", "held MSE", "held SNR", "digest");
        println!("  {:->6} {:->8} {:->10} {:->12} {:->12} {:->10}  {:->12}", "", "", "", "", "", "", "");
        let mut out = Vec::new();
        for &p in &lengths {
            let cfg = tok_cfg(p);
            let chans: Vec<usize> =
                (0..CHANNELS.len()).filter(|&c| CHANNELS[c].2 == p).collect();
            let mut tr: Vec<&Vec<f32>> = Vec::new();
            let mut hd: Vec<&Vec<f32>> = Vec::new();
            for &c in &chans {
                for &i in &train_idx {
                    tr.push(&patches[c][i]);
                }
                for &i in &held_idx {
                    hd.push(&patches[c][i]);
                }
            }
            let (enc, dec, first, last, digest) = match train_tokenizer(&ctx, cfg, &q, &tr, tok_steps, 11) {
                Ok(v) => v,
                Err(e) => { eprintln!("error: patch {p}: {e}"); std::process::exit(1); }
            };
            // Held-out reconstruction, through the ACTUAL quantizer rather than the
            // straight-through estimator: at inference there is no gradient to pass, and a figure
            // measured on the estimator's output would be measuring the training path.
            let (mse_h, var_h) = recon_error(&ctx, cfg, &q, &enc, &dec, &hd);
            let snr = 10.0 * (var_h / mse_h.max(1e-12)).log10();
            println!("  {p:>6} {:>8} {first:>10.5} {last:>12.5} {mse_h:>12.5} {snr:>9.1}dB  {}",
                     tr.len(), &digest[..12]);
            out.push((p, cfg, enc));
        }
        println!();
        trained = Some(out);
    }

    // ---- tokenize, one arm per tokenizer ----
    let mut arms: Vec<(&str, Vec<Vec<Vec<u32>>>, HashSet<u32>)> = Vec::new();
    let (pc, codes) = tokenize(&ctx, &q, &untrained, &patches, n);
    arms.push(("untrained", pc, codes));
    if let Some(t) = &trained {
        let (pc, codes) = tokenize(&ctx, &q, t, &patches, n);
        arms.push(("trained", pc, codes));
    }

    let seq = Sequencer::new(HybridVocab::new(64, q.clone()).unwrap());
    println!("  {:<11} {:>12} {:>14} {:>12} {:>12}",
             "tokenizer", "codes used", "of codebook", "distinct seq", "tokens");
    println!("  {:-<11} {:->12} {:->14} {:->12} {:->12}", "", "", "", "", "");
    for (name, pc, codes) in &arms {
        let mut distinct: HashSet<Vec<u32>> = HashSet::new();
        let mut toks = 0usize;
        for chans in pc {
            toks += chans.iter().map(|c| c.len()).sum::<usize>();
            distinct.insert(seq.encode(&[Span::Signal(chans.clone())]).unwrap());
        }
        println!("  {name:<11} {:>12} {:>13.1}% {:>12} {toks:>12}",
                 codes.len(), codes.len() as f64 / q.codebook_size() as f64 * 100.0, distinct.len());
    }

    // Label structure: what a caption over this corpus would have to say.
    println!("\n  labels present in these {n} cycles:");
    for (i, nm) in LABELS.iter().enumerate() {
        let mut vals: Vec<i32> = labels.iter().map(|r| r[i]).collect();
        vals.sort_unstable();
        vals.dedup();
        println!("    {nm:<12} {} distinct: {vals:?}", vals.len());
    }
    let combos: HashSet<Vec<i32>> = labels.iter().cloned().collect();
    println!("    {} distinct label COMBINATIONS — a caption is compositional here, not one-of-N",
             combos.len());

    // ---- the probe: do the tokens carry the label at all? ----
    println!("\n  TOKEN PROBE  naive Bayes over (channel, code) counts. No language model, no");
    println!("  capacity to speak — just whether the token stream separates the classes.\n");
    // THE CONTROL. A probe that beats majority is only evidence if the SAME probe, on the same
    // tokens, fails when the labels are permuted. Without this, a probe that is simply
    // mis-scored — scoring on training rows, leaking an index — reads as a discovery. Five
    // permutations, each scored against its own permuted majority, and the worst case is
    // reported: one lucky permutation is not a control.
    let perms: Vec<Vec<Vec<i32>>> = (0..5)
        .map(|k| {
            let order = shuffled(n, 0xC0FF_EE00 + k);
            order.iter().map(|&i| labels[i].clone()).collect()
        })
        .collect();

    print!("  {:<13} {:>10}", "axis", "majority");
    for (name, _, _) in &arms {
        print!(" {:>12}", *name);
    }
    println!(" {:>12}", "control");
    print!("  {:-<13} {:->10}", "", "");
    for _ in &arms {
        print!(" {:->12}", "");
    }
    println!(" {:->12}", "");
    let mut control_worst = 0.0f64;
    for a in 0..LABELS.len() {
        let maj = majority(&labels, &train_idx, &held_idx, a);
        print!("  {:<13} {maj:>9.1}%", LABELS[a]);
        for (_, pc, _) in &arms {
            let acc = nb_probe(pc, &labels, &train_idx, &held_idx, a);
            let mark = if acc > maj + 1e-9 { "*" } else { " " };
            print!(" {acc:>11.1}%{mark}");
        }
        // The control runs on the LAST arm's tokens — the best tokenizer available, so the
        // control is run where a spurious result would be easiest to get.
        let last_arm = &arms[arms.len() - 1].1;
        let over: Vec<f64> = perms
            .iter()
            .map(|pl| {
                nb_probe(last_arm, pl, &train_idx, &held_idx, a) - majority(pl, &train_idx, &held_idx, a)
            })
            .collect();
        let worst = over.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        control_worst = control_worst.max(worst);
        print!(" {:>+11.1}pt", worst);
        println!();
    }
    println!("\n  * beats the majority baseline. `control` is the same probe on the same tokens with");
    println!("  the CYCLE-TO-LABEL assignment permuted, reported as the best of five permutations");
    println!("  against each permutation's own majority: {control_worst:+.1} points at worst. A control");
    println!("  near zero is what makes a starred column mean something.");
    println!("\n  RevIn normalises every channel of every cycle to zero mean and unit scale before");
    println!("  patching, so absolute pressure and absolute temperature are gone by construction.");
    println!("  Whatever the probe reads is SHAPE.");
    println!("\n  A probe at or below majority means the tokens do not separate that axis, which is a");
    println!("  statement about the TOKENIZER; a probe above majority that the language model");
    println!("  cannot match is a statement about the DECODER.");

    if !args.iter().any(|a| a == "--train") {
        println!("\n  Pass --train to run signal-to-text on these captions.\n");
        return;
    }

    // ---- signal to text on real captions ----
    let (words, caps) = build_words(&labels);
    let vocab_words = words.len() as u32;
    let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(600);
    let seeds: usize = flag(&args, "--seeds").and_then(|v| v.parse().ok()).unwrap_or(3);
    let batch: usize = flag(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(8);
    let lm_dim: usize = flag(&args, "--lm-dim").and_then(|v| v.parse().ok()).unwrap_or(64);
    let lm_layers: usize = flag(&args, "--lm-layers").and_then(|v| v.parse().ok()).unwrap_or(2);
    let lm_cfg = EncoderConfig {
        patch_len: 16, d_model: lm_dim, n_layers: lm_layers, n_heads: 4, d_ff: lm_dim * 2, latent_dim: 5,
    };
    let vocabs: Vec<&str> = match flag(&args, "--vocab").as_deref() {
        Some("full") => vec!["full"],
        Some("compact") => vec!["compact"],
        _ => vec!["full", "compact"],
    };

    println!("\n  SIGNAL -> TEXT  {} train / {} held out, {vocab_words} caption words",
             train_idx.len(), held_idx.len());
    println!("  {steps} optimizer steps x batch {batch} = {} examples seen, {seeds} seeds",
             steps * batch);
    println!("  decoder: d_model {lm_dim}, {lm_layers} layers");
    println!("  captions are compositional: five axes scored separately");

    // Every (tokenizer, vocabulary) cell runs at the SAME seeds, so a difference between cells is
    // a difference between cells and not between draws.
    let mut cells: Vec<(String, Vec<Vec<Vec<u32>>>, Sequencer)> = Vec::new();
    for (name, pc, _) in &arms {
        for v in &vocabs {
            match *v {
                "compact" => {
                    let (rm, size, unk_pct) = compact(pc, &train_idx, &held_idx);
                    println!("    {name}/compact: {size} signal rows (from {}), {unk_pct:.2}% of held-out tokens unseen in training",
                             q.codebook_size());
                    let fq = Fsq::new(vec![size]).unwrap();
                    cells.push((format!("{name}/compact"),
                                rm,
                                Sequencer::new(HybridVocab::new(vocab_words, fq).unwrap())));
                }
                _ => cells.push((format!("{name}/full"),
                                 pc.clone(),
                                 Sequencer::new(HybridVocab::new(vocab_words, q.clone()).unwrap()))),
            }
        }
    }

    for (name, pc, seq2) in &cells {
        println!("\n  cell: {name}  ({} embedding rows)", seq2.embedding_rows());
        let mut runs: Vec<SeedResult> = Vec::new();
        for s in 0..seeds {
            let r = train_seed(&ctx, seq2, pc, &caps, &train_idx, &held_idx, steps, batch, lm_cfg, 3 + s as u64 * 17);
            println!("    seed {}: {}", 3 + s * 17,
                     r.acc.iter().enumerate().map(|(a, v)| format!("{}={v:.0}%", LABELS[a])).collect::<Vec<_>>().join("  "));
            runs.push(r);
        }

        // MAJORITY BASELINE. "At chance" and "predicting the training majority" look identical in
        // an accuracy column and are different failures: the second means the model learned the
        // label prior and ignored the signal. Scoring the majority explicitly separates them, and
        // a model that merely ties it has read nothing.
        println!("\n    {:<13} {:>8} {:>7} {:>10} {:>8} {:>9}",
                 "axis", "mean", "sd", "majority", "chance", "said");
        println!("    {:-<13} {:->8} {:->7} {:->10} {:->8} {:->9}", "", "", "", "", "", "");
        for a in 0..LABELS.len() {
            let v: Vec<f64> = runs.iter().map(|r| r.acc[a]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
            let n_vals = labels.iter().map(|r| r[a]).collect::<HashSet<_>>().len();
            let maj_acc = majority(&labels, &train_idx, &held_idx, a);
            // How many DIFFERENT words the model emitted at this position, averaged over seeds.
            // 1.0 means it answered the same thing for every held-out cycle.
            let said = runs.iter().map(|r| r.distinct[a] as f64).sum::<f64>() / runs.len() as f64;
            let verdict = if said <= 1.0 {
                "  <- one word for every cycle"
            } else if m > maj_acc + 1e-9 {
                ""
            } else {
                "  <- at or below majority"
            };
            println!("    {:<13} {m:>7.1}% {sd:>6.1} {maj_acc:>9.1}% {:>7.0}% {said:>8.1} of {n_vals}{verdict}",
                     LABELS[a], 100.0 / n_vals as f64);
        }
    }

    println!("\n  Chance is one over the number of values that axis takes. {seeds} seeds, n={} held out,",
             held_idx.len());
    println!("  so one example is {:.1} points: read the spread before reading any gap.",
             100.0 / held_idx.len() as f64);
    println!("  Real sensor data, real labels; the corpus is small.\n");
}
