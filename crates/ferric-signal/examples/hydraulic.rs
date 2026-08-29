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
//! ## Presentation order is part of the protocol, not a detail
//!
//! The same fact that forces a strided sample — the corpus is stored in experimental-condition
//! order — also makes walking it in index order the wrong way to train. An earlier version of this
//! example did exactly that, and produced a clean null: one word emitted for every held-out cycle
//! on three of five axes, sd 0.0 across seeds, every axis at its majority baseline. It was reported
//! as evidence about the tokenizer. It was evidence about the loop.
//!
//! Holding corpus, split, tokenizer and examples-seen fixed and varying only presentation
//! (400 cycles, 300 train, 2,000 examples, three seeds, held-out accuracy):
//!
//! ```text
//!   protocol                    cooler     valve  pump_leak  accumulator    stable
//!   majority                      36.0      54.0       55.0         33.0      63.0
//!   batch 1, corpus order    37.0+-0.0 54.0+-0.0  55.0+-0.0    23.3+-6.9 72.0+-0.0
//!   batch 1, shuffled        54.3+-6.1 66.7+-3.8  58.3+-1.7    37.3+-4.6 85.0+-1.6
//!   batch 8, shuffled        84.7+-1.7 72.3+-4.1  57.0+-0.8    43.3+-3.4 85.3+-0.5
//! ```
//!
//! `--sequential` reproduces the first row. The `said` column in every run's output counts the
//! distinct words the decoder actually emitted, so a repeat of this failure announces itself
//! instead of arriving as a plausible-looking accuracy column.
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
    build_words, compact, decoder_forward_var, forward_var, majority, mse, nb_probe,
    permutation_control, shuffled, straight_through, train_captions, DecoderWeights, EncoderConfig,
    EncoderWeights, Fsq, HybridVocab, Patcher, RevIn, Sequencer, Span, Weights,
};
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::HashSet;
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
    let mut labels: Vec<Vec<i32>> = picked.iter().map(|&i| all_labels[i].clone()).collect();
    // THE CONTROL RUN. Permuting which cycle carries which caption destroys the signal-to-label
    // correspondence and leaves everything else — split, class balance, caption vocabulary, the
    // correlations WITHIN a caption — exactly as it was. A cell that still scores above its
    // majority under this is not reading the signal.
    //
    // It matters most for axes 1..4, which are scored with the earlier caption words teacher-
    // forced. Those five axes are not independent in this rig's experimental design, so a decoder
    // could in principle answer `stable` from `cooler` and never look at a sensor. Under
    // permutation that shortcut survives and the signal does not, which is what makes the two
    // separable at all.
    if args.iter().any(|a| a == "--control") {
        let order = shuffled(labels.len(), 0xBADC_0DE);
        labels = order.iter().map(|&i| labels[i].clone()).collect();
        println!("  CONTROL: cycle-to-caption assignment permuted");
    }
    let labels = labels;
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

    // One label vector per axis, which is what the library instruments take: a probe asks about
    // ONE axis at a time, and bundling the five into a row was only ever convenient here.
    let per_axis: Vec<Vec<i32>> =
        (0..LABELS.len()).map(|a| labels.iter().map(|r| r[a]).collect()).collect();

    // ---- the probe: do the tokens carry the label at all? ----
    println!("\n  TOKEN PROBE  naive Bayes over (channel, code) counts. No language model, no");
    println!("  capacity to speak — just whether the token stream separates the classes.\n");
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
        let maj = majority(&per_axis[a], &train_idx, &held_idx);
        print!("  {:<13} {maj:>9.1}%", LABELS[a]);
        for (_, pc, _) in &arms {
            let acc = nb_probe(pc, &per_axis[a], &train_idx, &held_idx);
            let mark = if acc > maj + 1e-9 { "*" } else { " " };
            print!(" {acc:>11.1}%{mark}");
        }
        // The control runs on the LAST arm's tokens — the best tokenizer available, so it is run
        // where a spurious result would be easiest to get.
        let last_arm = &arms[arms.len() - 1].1;
        let worst =
            permutation_control(last_arm, &per_axis[a], &train_idx, &held_idx, 20, 0xC0FF_EE00);
        control_worst = control_worst.max(worst);
        print!(" {:>+11.1}pt", worst);
        println!();
    }
    println!("\n  * beats the majority baseline. `control` is the same probe on the same tokens with");
    println!("  the CYCLE-TO-LABEL assignment permuted, reported as the worst of twenty permutations");
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
    let (words, caps) = build_words(LABELS, &labels);
    let vocab_words = words.len() as u32;
    let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(600);
    let seeds: usize = flag(&args, "--seeds").and_then(|v| v.parse().ok()).unwrap_or(3);
    let batch: usize = flag(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(8);
    let sequential = args.iter().any(|a| a == "--sequential");
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
    println!("  decoder: d_model {lm_dim}, {lm_layers} layers, examples presented {}",
             if sequential { "in corpus order" } else { "shuffled" });
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
        let mut runs: Vec<ferric_signal::SeedResult> = Vec::new();
        for s in 0..seeds {
            let r = train_captions(
                &ctx, seq2, pc, &caps, &train_idx, &held_idx, steps, batch, lm_cfg, sequential,
                false, LABELS.len(), 3 + s as u64 * 17,
            )
            .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
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
            let n_vals = per_axis[a].iter().collect::<HashSet<_>>().len();
            let maj_acc = majority(&per_axis[a], &train_idx, &held_idx);
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
