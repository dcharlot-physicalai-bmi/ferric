//! Ingest a real multi-rate sensor corpus: UCI Condition Monitoring of Hydraulic Systems.
//!
//!   cargo run -p ferric-signal --example hydraulic --release -- --data <dir> [--cycles N]
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
//! ## What this example does and does not claim
//!
//! **Cycles are sampled with a stride, never as a prefix.** The corpus is ordered by experimental
//! condition: the first 100 cycles all carry cooler=3, valve=100, leak=0, accumulator=130,
//! stable=1 — one condition, five identical labels, a single-class corpus wearing the shape of a
//! hundred examples. Striding across all 2,205 cycles is what makes the label axes vary.
//!
//! It ingests, tokenizes and reports structure: how many distinct token sequences the corpus
//! produces, how much of the codebook it visits, and how those vary by channel. **The encoder is
//! untrained**, so these numbers describe what an untrained tokenizer does to real data. They are
//! not accuracy results and there is no model here to be accurate.

use ferric_core::Context;
use ferric_signal::{
    cross_entropy, embed_var, lm_forward_var, EncoderConfig, EncoderWeights, Example, Fsq,
    HybridVocab, Patcher, RevIn, SensorLm, Sequencer, Span,
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

/// Train signal-to-text for one seed. Returns per-axis held-out accuracy.
#[allow(clippy::too_many_arguments)]
fn train_seed(
    ctx: &Arc<Context>,
    seq: &Sequencer,
    rows_tokens: &[Vec<Vec<u32>>],
    caps: &[Vec<u32>],
    train_idx: &[usize],
    held_idx: &[usize],
    steps: usize,
    seed: u64,
) -> Vec<f64> {
    let rows = seq.embedding_rows();
    let cfg = EncoderConfig { patch_len: 16, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 128, latent_dim: 5 };
    let lm = SensorLm::deterministic(ctx, cfg, rows, seed).unwrap();
    // The embedding table trains with the rest: a signal code the corpus never visited is an
    // untrained row, and this corpus visits only about a quarter of the code space.
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

    for step in 0..steps {
        let i = train_idx[step % train_idx.len()];
        let e = build(i);
        let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
        let emb = embed_var(ctx, &vars[0], &e.tokens).unwrap();
        let logits = lm_forward_var(ctx, cfg, &vars[1..], &emb).unwrap();
        let loss = cross_entropy(ctx, &logits, &e.tokens, e.target_from, rows).unwrap();
        loss.backward();
        let grads: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
        opt.step(&mut params, &grads);
    }

    // Score each axis at its own caption position: the model sees the signal plus the caption
    // words before this axis, and must produce this axis's word.
    let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
    let mut right = vec![0usize; LABELS.len()];
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
            if best == seq.vocab().text(caps[i][a]).unwrap() as usize {
                right[a] += 1;
            }
        }
    }
    right.iter().map(|&r| r as f64 / held_idx.len() as f64 * 100.0).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = flag(&args, "--data") else {
        eprintln!("usage: --data <dir containing PS1.txt .. profile.txt> [--cycles N]");
        std::process::exit(2);
    };
    let max_cycles: usize = flag(&args, "--cycles").and_then(|v| v.parse().ok()).unwrap_or(200);

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

    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example needs one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);
    let q = Fsq::new(vec![8u32; 5]).unwrap();
    let seq = Sequencer::new(HybridVocab::new(64, q.clone()).unwrap());

    // One encoder per distinct patch length. Untrained and fixed, so every channel is tokenized by
    // the same weights at the same resolution and the counts below are comparable.
    let mut encoders: Vec<(usize, EncoderConfig, EncoderWeights)> = Vec::new();
    for &(_, _, patch) in CHANNELS {
        if !encoders.iter().any(|(p, _, _)| *p == patch) {
            let cfg = EncoderConfig { patch_len: patch, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 };
            let w = EncoderWeights::deterministic(&ctx, cfg, 7).unwrap();
            encoders.push((patch, cfg, w));
        }
    }
    println!("  {} channels, {} distinct patch lengths, codebook {} codes\n",
             CHANNELS.len(), encoders.len(), q.codebook_size());

    println!("  {:<6} {:>6} {:>8} {:>9} {:>9} {:>10}", "chan", "rate", "patches", "tokens", "distinct", "codes used");
    println!("  {:-<6} {:->6} {:->8} {:->9} {:->9} {:->10}", "", "", "", "", "", "");

    let mut all_codes: HashSet<u32> = HashSet::new();
    let mut per_cycle: Vec<Vec<Vec<u32>>> = vec![Vec::new(); labels.len()];
    let mut total_tokens = 0usize;

    for &(name, cols, patch) in CHANNELS {
        let rows = match read_channel(&dir, name, cols, &picked) {
            Ok(r) => r,
            Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
        };
        let (_, cfg, enc) = encoders.iter().find(|(p, _, _)| *p == patch).unwrap();
        let patcher = Patcher::contiguous(patch).unwrap();
        let mut chan_codes: HashSet<u32> = HashSet::new();
        let mut seqs: HashSet<Vec<u32>> = HashSet::new();
        let mut toks = 0usize;

        // ONE FORWARD PER CYCLE, and that is not an optimisation to be undone. Batching every
        // cycle of a channel into a single tensor was faster and WRONG: the encoder's attention is
        // over the whole sequence it is given, so a 400-cycle batch let patches from one cycle
        // attend to patches from another. Tokenization then depends on what else happened to be in
        // the batch, which breaks per-cycle determinism and quietly couples held-out cycles to
        // training ones. It surfaced only because 24,000 patches squared exceeded the 4 GB buffer
        // limit; at 100 cycles it would have run and been silently wrong.
        //
        // The memory problem that motivated batching is fixed where it belonged, in the streamed
        // reader above.
        for (ci, raw) in rows.iter().enumerate() {
            // Per-cycle, per-channel normalization: a pressure channel and a temperature channel
            // differ by orders of magnitude in units, and the model should see shape.
            let rev = RevIn::fit(raw, 1).unwrap();
            let patches = patcher.patchify(&rev.apply(raw).unwrap()).unwrap();
            let t = patches.len() / patch;
            let lat = pollster::block_on(
                enc.forward(&ctx, &Tensor::from_vec(&ctx, &patches, &[t, patch])).unwrap().to_vec(),
            );
            let codes: Vec<u32> = (0..t)
                .map(|i| {
                    q.to_index(&q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap())
                        .unwrap()
                })
                .collect();
            chan_codes.extend(&codes);
            all_codes.extend(&codes);
            toks += codes.len();
            seqs.insert(codes.clone());
            per_cycle[ci].push(codes);
        }
        total_tokens += toks;
        let rate = cols / 60;
        println!("  {name:<6} {rate:>5}H {:>8} {toks:>9} {:>9} {:>10}",
                 rows.first().map(|r| r.len() / patch).unwrap_or(0), seqs.len(), chan_codes.len());
    }

    // One sequence per cycle: every channel as its own run, separated by the channel marker.
    let mut seq_lens = Vec::new();
    let mut distinct_cycles: HashSet<Vec<u32>> = HashSet::new();
    for chans in &per_cycle {
        if chans.len() != CHANNELS.len() { continue; }
        let ids = seq.encode(&[Span::Signal(chans.clone())]).unwrap();
        seq_lens.push(ids.len());
        distinct_cycles.insert(ids);
    }

    println!("\n  corpus: {} cycles, {total_tokens} signal tokens, {} of {} codes visited ({:.1}%)",
             per_cycle.len(), all_codes.len(), q.codebook_size(),
             all_codes.len() as f64 / q.codebook_size() as f64 * 100.0);
    if let (Some(&lo), Some(&hi)) = (seq_lens.iter().min(), seq_lens.iter().max()) {
        println!("  one cycle = {lo}..{hi} tokens across {} channel runs", CHANNELS.len());
    }
    println!("  distinct cycle sequences: {} of {}", distinct_cycles.len(), seq_lens.len());

    // Label structure: what a caption over this corpus would have to say.
    println!("\n  labels present in these {} cycles:", labels.len());
    for (i, nm) in LABELS.iter().enumerate() {
        let mut vals: Vec<i32> = labels.iter().map(|r| r[i]).collect();
        vals.sort_unstable();
        vals.dedup();
        println!("    {nm:<12} {} distinct: {vals:?}", vals.len());
    }
    let combos: HashSet<Vec<i32>> = labels.iter().cloned().collect();
    println!("    {} distinct label COMBINATIONS — a caption is compositional here, not one-of-N",
             combos.len());
    if !args.iter().any(|a| a == "--train") {
        println!("\n  The encoder is untrained: these describe what an untrained tokenizer does to real");
        println!("  multi-rate data. They are structure, not accuracy.");
        println!("  Pass --train to run signal-to-text on these captions.\n");
        return;
    }

    // ---- signal to text on real captions ----
    let complete: Vec<usize> = (0..per_cycle.len()).filter(|&i| per_cycle[i].len() == CHANNELS.len()).collect();
    let (words, caps) = build_words(&labels);
    // Split by stride phase, not contiguously: the corpus is ordered by condition, so a contiguous
    // split would put whole conditions on one side of it.
    let held_idx: Vec<usize> = complete.iter().copied().filter(|i| i % 4 == 3).collect();
    let train_idx: Vec<usize> = complete.iter().copied().filter(|i| i % 4 != 3).collect();

    let vocab_words = words.len() as u32;
    let seq2 = Sequencer::new(HybridVocab::new(vocab_words, q.clone()).unwrap());
    let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(600);
    let seeds: usize = flag(&args, "--seeds").and_then(|v| v.parse().ok()).unwrap_or(3);

    println!("\n  SIGNAL -> TEXT  {} train / {} held out, {vocab_words} caption words, {steps} steps, {seeds} seeds",
             train_idx.len(), held_idx.len());
    println!("  captions are compositional: five axes scored separately\n");

    let mut runs: Vec<Vec<f64>> = Vec::new();
    for s in 0..seeds {
        let acc = train_seed(&ctx, &seq2, &per_cycle, &caps, &train_idx, &held_idx, steps, 3 + s as u64 * 17);
        println!("    seed {}: {}", 3 + s * 17,
                 acc.iter().enumerate().map(|(a, v)| format!("{}={v:.0}%", LABELS[a])).collect::<Vec<_>>().join("  "));
        runs.push(acc);
    }

    // MAJORITY BASELINE. "At chance" and "predicting the training majority" look identical in an
    // accuracy column and are different failures: the second means the model learned the label
    // prior and ignored the signal. Scoring the majority explicitly separates them, and a model
    // that merely ties it has read nothing.
    println!("\n  {:<13} {:>8} {:>7} {:>10} {:>10}", "axis", "mean", "sd", "majority", "chance");
    println!("  {:-<13} {:->8} {:->7} {:->10} {:->10}", "", "", "", "", "");
    for a in 0..LABELS.len() {
        let v: Vec<f64> = runs.iter().map(|r| r[a]).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
        let n_vals = labels.iter().map(|r| r[a]).collect::<HashSet<_>>().len();
        // Most frequent value in TRAIN, scored on HELD OUT.
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for &i in &train_idx {
            *counts.entry(labels[i][a]).or_insert(0) += 1;
        }
        let maj = counts.iter().max_by_key(|(_, &c)| c).map(|(&v, _)| v).unwrap_or(0);
        let maj_acc = held_idx.iter().filter(|&&i| labels[i][a] == maj).count() as f64
            / held_idx.len() as f64 * 100.0;
        let verdict = if m > maj_acc + 1e-9 { "" } else { "  <- at or below majority" };
        println!("  {:<13} {m:>7.1}% {sd:>6.1} {maj_acc:>9.1}% {:>9.0}%{verdict}",
                 LABELS[a], 100.0 / n_vals as f64);
    }
    println!("\n  Chance is one over the number of values that axis takes. {seeds} seeds, n={} held out,",
             held_idx.len());
    println!("  so one example is {:.1} points: read the spread before reading any gap.",
             100.0 / held_idx.len() as f64);
    println!("  Real sensor data, real labels; the tokenizer is untrained and the corpus is small.\n");
}
