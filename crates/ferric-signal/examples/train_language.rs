//! Signal in, words out: train the language half on synthetic sensor-text pairs.
//!
//!   cargo run -p ferric-signal --example train_language --release
//!
//! ## Where the pairs come from, and what that does and does not show
//!
//! Sensor-language training needs signals with descriptions attached. The published model this
//! crate reproduces used 23B such tokens; the academic work used narrow wearable corpora. Neither
//! is available here, so the pairs are **generated**: five physical processes whose ground truth is
//! known by construction, each captioned programmatically.
//!
//! That makes the labels correct by construction, which is the point — and it also bounds the
//! claim. Success here shows the architecture learns a signal→text mapping end to end. It does not
//! show anything about real sensors, and it cannot: the model is being asked to recover a label the
//! generator wrote.
//!
//! The embedding table is FROZEN (`Var` has no row gather), so training moves the blocks and the
//! head. A frozen embedding cannot learn that two signal codes mean similar things, which caps what
//! a small corpus can teach.

use ferric_core::Context;
use ferric_signal::{
    cross_entropy, lm_forward_var, EncoderConfig, EncoderWeights, Fsq, HybridVocab, Patcher, RevIn,
    SensorLm, Sequencer, Span, Task,
};
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::sync::Arc;

/// A tiny caption vocabulary. Word 0 is reserved as a terminator.
const WORDS: [&str; 6] = ["<end>", "damped", "thermal", "square", "chirp", "noise"];
const PATCH: usize = 16;
const PATCHES: usize = 6;
const STEPS: usize = 600;

/// `v` selects a VARIANT of the same process: a different damping rate, duty cycle, chirp span.
/// Variant 0 is what the model trains on; the others are held out.
fn process_v(kind: usize, i: usize, v: usize) -> f32 {
    let t = i as f32 * 0.002;
    let k = 1.0 + v as f32 * 0.6;
    match kind {
        0 => (-1.2 * k * t).exp() * (2.0 * std::f32::consts::PI * 11.0 * k * t).sin() * 3.0,
        1 => 20.0 + 45.0 * (-0.8 * k * t).exp(),
        2 => if (t * 60.0 * k).fract() < 0.35 + 0.15 * v as f32 { 5.0 } else { 0.0 },
        3 => (2.0 * std::f32::consts::PI * (4.0 * k + 30.0 * k * t) * t).sin() * 1.5,
        _ => {
            let mut s = (i as u64 + v as u64 * 7919).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            s ^= s >> 31;
            0.5 * (0.7 * k * t).sin() + ((s >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 1.2
        }
    }
}

fn process(kind: usize, i: usize) -> f32 { process_v(kind, i, 0) }

fn main() {
    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example measures nothing without one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);

    let q = Fsq::signal_15bit();
    let vocab = HybridVocab::new(WORDS.len() as u32, q.clone()).unwrap();
    let seq = Sequencer::new(vocab);
    let rows = seq.embedding_rows();

    // The tokenizer: untrained but FIXED, so the same process always yields the same codes. The
    // language half is what is being trained here.
    let enc_cfg = EncoderConfig { patch_len: PATCH, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 };
    let enc = EncoderWeights::deterministic(&ctx, enc_cfg, 7).unwrap();
    let patcher = Patcher::contiguous(PATCH).unwrap();

    // Build one example per process: its signal codes, captioned with its own word.
    let mut examples = Vec::new();
    for kind in 0..WORDS.len() - 1 {
        let raw: Vec<f32> = (0..PATCH * PATCHES).map(|i| process(kind, i)).collect();
        let rev = RevIn::fit(&raw, 1).unwrap();
        let patches = patcher.patchify(&rev.apply(&raw).unwrap()).unwrap();
        let t = patches.len() / PATCH;
        let lat = pollster::block_on(
            enc.forward(&ctx, &Tensor::from_vec(&ctx, &patches, &[t, PATCH])).unwrap().to_vec(),
        );
        let codes: Vec<u32> = (0..t)
            .map(|i| q.to_index(&q.quantize(&lat[i * 5..(i + 1) * 5]).unwrap()).unwrap())
            .collect();
        // Caption: the process's own word, then the terminator.
        let e = seq.example(Task::SignalToText, &[(kind + 1) as u32, 0], &[codes], &[]).unwrap();
        examples.push((kind, e));
    }

    let cfg = EncoderConfig { patch_len: PATCH, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 128, latent_dim: 5 };
    let lm = SensorLm::deterministic(&ctx, cfg, rows, 3).unwrap();
    let mut params = lm.params_flat();
    let mut opt = Adam::new(&params, 3e-3);

    println!("\nSIGNAL -> TEXT  {} processes, {} embedding rows ({} words + {} codes + 3 markers)",
             examples.len(), rows, WORDS.len(), q.codebook_size());
    println!("  the tokenizer is fixed; the language half is what trains\n");
    println!("  {:>6}  {:>12}  {:>10}", "step", "loss", "correct");
    println!("  {:->6}  {:->12}  {:->10}", "", "", "");

    let predict = |params: &[Tensor]| -> usize {
        let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
        let mut right = 0;
        for (kind, e) in &examples {
            // Feed the prompt plus the first target position, and read the argmax at the last step.
            let upto = e.target_from;
            let ids = &e.tokens[..upto];
            let emb = Var::leaf(lm.embed_tokens(ids));
            let logits = pollster::block_on(
                lm_forward_var(&ctx, cfg, &vars, &emb).unwrap().value().to_vec(),
            );
            let last = (upto - 1) * rows as usize;
            let row = &logits[last..last + rows as usize];
            let best = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if best == seq.vocab().text((kind + 1) as u32).unwrap() as usize {
                right += 1;
            }
        }
        right
    };

    let mut first = 0.0f32;
    for step in 0..=STEPS {
        let mut total = 0.0f32;
        let mut grads_acc: Option<Vec<Tensor>> = None;
        for (_, e) in &examples {
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
        let n = examples.len() as f32;
        let grads: Vec<Tensor> = grads_acc.unwrap().iter().map(|g| g.mul(&g.scalar(1.0 / n))).collect();
        let l = total / n;
        if step == 0 { first = l; }
        if step % 150 == 0 || step == STEPS {
            println!("  {step:>6}  {l:>12.6}  {:>7} / {}", predict(&params), examples.len());
        }
        opt.step(&mut params, &grads);
    }

    println!("\nRESULT");
    let right = predict(&params);
    println!("  loss {first:.4} -> final; {right} of {} processes named correctly from their tokens alone",
             examples.len());
    for (kind, e) in &examples {
        let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
        let ids = &e.tokens[..e.target_from];
        let emb = Var::leaf(lm.embed_tokens(ids));
        let logits = pollster::block_on(lm_forward_var(&ctx, cfg, &vars, &emb).unwrap().value().to_vec());
        let last = (e.target_from - 1) * rows as usize;
        let row = &logits[last..last + rows as usize];
        let best = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let said = seq.vocab().kind(best as u32).ok().and_then(|k| match k {
            ferric_signal::TokenKind::Text(w) => WORDS.get(w as usize).copied(),
            _ => Some("<a signal code>"),
        }).unwrap_or("?");
        println!("    {:>8} -> said {:?}", WORDS[kind + 1], said);
    }
    // ---- THE CHECK THAT SEPARATES LEARNING FROM MEMORISATION ----
    //
    // Five examples and five classes means a lookup table scores 5 of 5. So: same processes,
    // DIFFERENT parameters — a faster decay, a wider duty cycle, a different chirp span. The model
    // has never seen these token sequences.
    println!("\nHELD OUT  (same processes, different parameters — never seen)");
    let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
    let mut held_right = 0;
    let mut held_total = 0;
    for v in 1..=2usize {
        for kind in 0..WORDS.len() - 1 {
            let raw: Vec<f32> = (0..PATCH * PATCHES).map(|i| process_v(kind, i, v)).collect();
            let rev = RevIn::fit(&raw, 1).unwrap();
            let patches = patcher.patchify(&rev.apply(&raw).unwrap()).unwrap();
            let t = patches.len() / PATCH;
            let lat = pollster::block_on(
                enc.forward(&ctx, &Tensor::from_vec(&ctx, &patches, &[t, PATCH])).unwrap().to_vec(),
            );
            let codes: Vec<u32> = (0..t)
                .map(|i| q.to_index(&q.quantize(&lat[i * 5..(i + 1) * 5]).unwrap()).unwrap())
                .collect();
            let ids = seq.encode(&[Span::Signal(vec![codes])]).unwrap();
            let emb = Var::leaf(lm.embed_tokens(&ids));
            let logits = pollster::block_on(
                lm_forward_var(&ctx, cfg, &vars, &emb).unwrap().value().to_vec(),
            );
            let last = (ids.len() - 1) * rows as usize;
            let row = &logits[last..last + rows as usize];
            let best = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let ok = best == seq.vocab().text((kind + 1) as u32).unwrap() as usize;
            held_right += ok as usize;
            held_total += 1;
            let said = seq.vocab().kind(best as u32).ok().and_then(|k| match k {
                ferric_signal::TokenKind::Text(w) => WORDS.get(w as usize).copied(),
                _ => Some("<a signal code>"),
            }).unwrap_or("?");
            println!("    variant {v}  {:>8} -> said {:<16} {}", WORDS[kind + 1], format!("{said:?}"),
                     if ok { "ok" } else { "MISS" });
        }
    }
    println!("  held-out: {held_right} of {held_total} correct ({:.0}%)",
             held_right as f64 / held_total as f64 * 100.0);
    println!("  Chance is 1 in {} words. {} of {} on TRAINED examples proves nothing on its own:",
             WORDS.len() - 1, right, examples.len());
    println!("  five examples over five classes is memorisable by a lookup table.");

    println!("\n  Scope: SYNTHETIC pairs with programmatic captions, a frozen embedding table, and");
    println!("  five processes. This shows the architecture learns a signal->text mapping. It shows");
    println!("  nothing about real sensors: the label was written by the generator.\n");
}
