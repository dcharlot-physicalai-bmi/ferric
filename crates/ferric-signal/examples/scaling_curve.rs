//! Held-out accuracy against synthetic corpus size: the curve that prices the corpus decision.
//!
//!   cargo run -p ferric-signal --example scaling_curve --release -- --variants 4
//!   cargo run -p ferric-signal --example scaling_curve --release -- --variants 8 --control
//!
//! ## The question this answers
//!
//! The first signal-to-text run trained on ONE example per process and scored 40% on held-out
//! variants against a 20% chance baseline — the memorisation signature. Before anyone sources or
//! generates a corpus, the cheap measurement is the slope: train on N variants per process, hold
//! out a fixed disjoint set, and watch what N buys.
//!
//! ## Protocol, stated so it can be attacked — and what the first version got wrong
//!
//! An adversarial review of the first version of this file found the split unsound: per-example
//! normalization cancels each family's affine parameters, and two "different" variants can land
//! close enough in normalized space to tokenize IDENTICALLY, at which point a held-out example is
//! a training example in the only representation the model sees — and denser sampling makes that
//! MORE likely, manufacturing a climbing curve. The protocol now guards the split directly:
//!
//! - **Split soundness is checked where the model looks.** Every held-out example whose code
//!   SEQUENCE also appears in the training set is excluded from the score, and the exclusion
//!   count is printed. Raw-signal distinctness proves nothing here and is not claimed.
//! - Train variants are `0..N` per kind; held-out variants are `100..104` per kind. `N` is
//!   required to stay at or below 100 so the index ranges cannot overlap.
//! - The tokenizer (encoder + FSQ) is UNTRAINED and FIXED across every run, seed 7. The language
//!   half is what trains; the embedding table is frozen throughout.
//! - Compute per step is constant: every step accumulates gradients over a 5-example batch. The
//!   training set is laid out variant-major, so every batch of 5 contains one example of EACH
//!   kind at every corpus size — class composition does not vary with N. What a bigger corpus
//!   changes is visits per example: at N=1 each example is seen ~600 times, at N=16 ~37 times.
//! - **What `--control` does and does not test.** It replaces each training label with a
//!   deterministic random one. That tests memorisation capacity and EVAL-SIDE contamination (a
//!   path from the true label into evaluation). It does NOT test the train/held-out split: a
//!   leaked example enters training with a random label and still scores at chance. The split is
//!   guarded by the code-sequence exclusion above, not by the control. The realized agreement
//!   between random and true labels is printed, because with 5 kinds it varies by draw and the
//!   null is only readable next to it.
//! - "Chance" is 1-in-5: the score demands the argmax over all 32,777 rows land on the one true
//!   word among 5, so 20% is what guessing uniformly among the five words would earn, and the
//!   model can do worse by preferring a signal token.

use ferric_core::Context;
use ferric_signal::{
    cross_entropy, lm_forward_var, synth, EncoderConfig, EncoderWeights, Fsq, HybridVocab, Patcher,
    RevIn, SensorLm, Sequencer, Span, Task,
};
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::HashSet;
use std::sync::Arc;

const WORDS: [&str; 6] = ["<end>", "damped", "thermal", "square", "chirp", "noise"];
const PATCH: usize = 16;
const PATCHES: usize = 6;
const BATCH: usize = 5;
const HELD_BASE: usize = 100;
const HELD_PER_KIND: usize = 4;

fn num(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let variants = num(&args, "--variants", 4);
    let steps = num(&args, "--steps", 600);
    let control = args.iter().any(|a| a == "--control");

    if variants < 1 || variants > HELD_BASE {
        eprintln!("--variants must be in 1..={HELD_BASE}: train variants are 0..N and held-out \
                   variants are {HELD_BASE}..{}, and the ranges must not overlap",
                  HELD_BASE + HELD_PER_KIND);
        std::process::exit(2);
    }

    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example measures nothing without one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);

    let q = Fsq::signal_15bit();
    let seq = Sequencer::new(HybridVocab::new(WORDS.len() as u32, q.clone()).unwrap());
    let rows = seq.embedding_rows();

    // The tokenizer: untrained, FIXED across every run in the sweep.
    let enc_cfg = EncoderConfig { patch_len: PATCH, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 };
    let enc = EncoderWeights::deterministic(&ctx, enc_cfg, 7).unwrap();
    let patcher = Patcher::contiguous(PATCH).unwrap();
    let ld = enc_cfg.latent_dim;

    let codes_for = |kind: usize, variant: usize| -> Vec<u32> {
        let raw = synth::signal(kind, variant, PATCH * PATCHES);
        let rev = RevIn::fit(&raw, 1).unwrap();
        let patches = patcher.patchify(&rev.apply(&raw).unwrap()).unwrap();
        let t = patches.len() / PATCH;
        let lat = pollster::block_on(
            enc.forward(&ctx, &Tensor::from_vec(&ctx, &patches, &[t, PATCH])).unwrap().to_vec(),
        );
        (0..t)
            .map(|i| q.to_index(&q.quantize(&lat[i * ld..(i + 1) * ld]).unwrap()).unwrap())
            .collect()
    };

    // VARIANT-MAJOR layout: every consecutive block of 5 holds one example of each kind, so batch
    // class-composition is identical at every corpus size and cannot confound the curve.
    let mut train = Vec::new();
    let mut train_codes: HashSet<u32> = HashSet::new();
    let mut train_seqs: HashSet<Vec<u32>> = HashSet::new();
    for v in 0..variants {
        for kind in 0..synth::KINDS {
            let codes = codes_for(kind, v);
            train_codes.extend(&codes);
            train_seqs.insert(codes.clone());
            let label = if control {
                1 + (mix((kind as u64) * 31 + (v as u64) * 7 + 9) % synth::KINDS as u64) as u32
            } else {
                (kind + 1) as u32
            };
            let e = seq.example(Task::SignalToText, &[label, 0], &[codes], &[]).unwrap();
            train.push((kind, v, label, e));
        }
    }
    let agree = train.iter().filter(|(k, _, l, _)| *l == (*k + 1) as u32).count();

    let cfg = EncoderConfig { patch_len: PATCH, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 128, latent_dim: 5 };
    let lm = SensorLm::deterministic(&ctx, cfg, rows, 3).unwrap();
    let mut params = lm.params_flat();
    let mut opt = Adam::new(&params, 3e-3);

    println!("\nSCALING  {} variants/kind = {} examples, {} steps, batch {} (one of each kind per batch){}",
             variants, train.len(), steps, BATCH,
             if control { "  [CONTROL: random labels]" } else { "" });
    println!("  tokenizer fixed and untrained; embedding frozen; LM trains");
    println!("  distinct signal codes in the training set: {}", train_codes.len());
    if control {
        println!("  realized agreement of random labels with truth: {agree} of {} ({:.0}%)",
                 train.len(), agree as f64 / train.len() as f64 * 100.0);
    }
    println!();

    // Argmax over the whole vocabulary at the last prompt position. `total_cmp` so a NaN logit
    // orders deterministically instead of panicking inside an unwrap with no context.
    let eval_one = |params: &[Tensor], codes: Vec<u32>, want_word: u32| -> bool {
        let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
        let ids = seq.encode(&[Span::Signal(vec![codes])]).unwrap();
        let emb = Var::leaf(lm.embed_tokens(&ids));
        let logits = pollster::block_on(lm_forward_var(&ctx, cfg, &vars, &emb).unwrap().value().to_vec());
        let last = (ids.len() - 1) * rows as usize;
        let row = &logits[last..last + rows as usize];
        let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        best == seq.vocab().text(want_word).unwrap() as usize
    };

    // Score each training example against the label it was TRAINED with — under --control that is
    // the random one, so this measures memorisation capacity rather than truth.
    let train_acc = |params: &[Tensor]| -> (usize, usize) {
        let mut right = 0;
        for (kind, v, label, _) in &train {
            if eval_one(params, codes_for(*kind, *v), *label) {
                right += 1;
            }
        }
        (right, train.len())
    };

    // Held-out scoring with THE SPLIT GUARD: an example whose code sequence appears verbatim in
    // the training set is not held out in any sense the model can perceive, so it is excluded and
    // counted. This is the check the first version lacked, and the reason it lacked honest numbers.
    struct Held {
        right: usize,
        scored: usize,
        collided: usize,
        overlap_pct: f64,
        per_kind: Vec<(usize, usize)>,
    }
    let held_acc = |params: &[Tensor]| -> Held {
        let mut h = Held { right: 0, scored: 0, collided: 0, overlap_pct: 0.0, per_kind: vec![(0, 0); synth::KINDS] };
        let mut overlap = 0.0f64;
        let mut n = 0usize;
        for kind in 0..synth::KINDS {
            for v in HELD_BASE..HELD_BASE + HELD_PER_KIND {
                let codes = codes_for(kind, v);
                let seen = codes.iter().filter(|c| train_codes.contains(c)).count();
                overlap += seen as f64 / codes.len() as f64;
                n += 1;
                if train_seqs.contains(&codes) {
                    h.collided += 1;
                    continue;
                }
                let ok = eval_one(params, codes, (kind + 1) as u32);
                h.right += ok as usize;
                h.scored += 1;
                h.per_kind[kind].0 += ok as usize;
                h.per_kind[kind].1 += 1;
            }
        }
        h.overlap_pct = overlap / n as f64 * 100.0;
        h
    };

    println!("  {:>6}  {:>12}  {:>11}  {:>14}", "step", "loss", "train acc", "held-out");
    println!("  {:->6}  {:->12}  {:->11}  {:->14}", "", "", "", "");

    for step in 0..=steps {
        let mut total = 0.0f32;
        let mut grads_acc: Option<Vec<Tensor>> = None;
        for j in 0..BATCH {
            let (_, _, _, e) = &train[(step * BATCH + j) % train.len()];
            let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
            let emb = Var::leaf(lm.embed_tokens(&e.tokens));
            let logits = lm_forward_var(&ctx, cfg, &vars, &emb).unwrap();
            let loss = cross_entropy(&ctx, &logits, &e.tokens, e.target_from, rows).unwrap();
            loss.backward();
            total += pollster::block_on(loss.value().to_vec())[0];
            let g: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
            grads_acc = Some(match grads_acc {
                None => g,
                Some(acc) => acc.iter().zip(&g).map(|(a, b)| a.add(b)).collect(),
            });
        }
        let grads: Vec<Tensor> = grads_acc.unwrap().iter().map(|g| g.mul(&g.scalar(1.0 / BATCH as f32))).collect();
        if step % (steps / 4).max(1) == 0 || step == steps {
            let (tr, tn) = train_acc(&params);
            let h = held_acc(&params);
            println!("  {step:>6}  {:>12.6}  {:>7} / {tn:<3}  {:>7} / {:<3}", total / BATCH as f32, tr, h.right, h.scored);
        }
        opt.step(&mut params, &grads);
    }

    let (tr, tn) = train_acc(&params);
    let h = held_acc(&params);
    let pct = if h.scored > 0 { h.right as f64 / h.scored as f64 * 100.0 } else { f64::NAN };
    println!("\nRESULT variants={variants} control={control} train={tr}/{tn} held={}/{} ({pct:.0}%) collided_excluded={} code_overlap={:.0}% label_agreement={agree}/{} chance=20%",
             h.right, h.scored, h.collided, h.overlap_pct, train.len());
    let by_kind: Vec<String> = h.per_kind.iter().enumerate()
        .map(|(k, (r, n))| format!("{}={r}/{n}", synth::name(k)))
        .collect();
    println!("  per kind: {}", by_kind.join("  "));
    println!("  Split guard: held-out examples token-identical to a training example are EXCLUDED");
    println!("  and counted above; the control tests eval-side contamination and memorisation");
    println!("  capacity, not the split. Held-out n={} gives ~{:.0}% granularity per example.",
             h.scored, if h.scored > 0 { 100.0 / h.scored as f64 } else { f64::NAN });
    println!("  The tokenizer is untrained and the embedding frozen: this measures the LM half only.\n");
}
