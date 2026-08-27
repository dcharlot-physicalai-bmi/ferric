//! Train ONE tokenizer across four independent sensor corpora, and publish it.
//!
//!   cargo run -p ferric-signal --example universal --release -- \
//!       --hydraulic <dir> --cwru <dir> --rotating <dir> --wind <dir> \
//!       [--windows N] [--steps N] [--out model.fsig]
//!
//! ## Why this exists
//!
//! This crate's own "what is NOT here" has said, since it was written, that there are no published
//! weights. In the landscape it reviewed, that is not a gap peculiar to this crate: the sensor
//! tokenizer everything else is built on is described in a paper and reachable only behind an API.
//! A tokenizer is also the *cheap* part — a discrete bottleneck over five dimensions with eight
//! levels each spans 32,768 codes and needs no codebook — so it is the piece that costs least to
//! open and unlocks the most.
//!
//! The claim a "universal" sensor tokenizer makes is that one set of weights turns any physical
//! signal into tokens. That claim is only testable against signals that do not resemble each
//! other, so this trains on four corpora at once and reports reconstruction on each SEPARATELY.
//! A single pooled number would let one easy corpus carry three hard ones.
//!
//! | corpus | machine | channels | rate |
//! |---|---|---|---|
//! | UCI hydraulic | hydraulic test rig | pressures, flows, temperatures, power | 1–100 Hz |
//! | CWRU | bearing test stand | drive-end and fan-end accelerometers | 12/48 kHz |
//! | Rotating machinery | gearbox and shaft rig | four accelerometers | 25.6 kHz |
//! | Wind drivetrain | wind-turbine nacelle | bearing and tower accelerometers | 74 kHz |
//!
//! Four rigs, four laboratories, four sampling regimes, and — for the hydraulic corpus —
//! quantities that are not vibration at all.
//!
//! ## What makes a patch comparable across all of that
//!
//! Nothing about absolute time or amplitude. Rates here span four orders of magnitude and the
//! quantities are not commensurable — a pressure in bar and an acceleration in g have no exchange
//! rate. What IS comparable is SHAPE: RevIn normalises every window of every channel to zero mean
//! and unit scale before patching, so the tokenizer sees a dimensionless waveform and a patch is a
//! fixed number of samples rather than a fixed duration.
//!
//! That is a modelling decision with a cost, and it is stated rather than hidden: a tokenizer built
//! this way cannot represent absolute level, so a channel that has drifted and one that has not are
//! the same to it unless the drift shows up inside the window.
//!
//! ## Held out by RECORDING, not by window
//!
//! Windows from one recording are not independent of each other. The held-out set here is built
//! from files and cycles the training set never touched, per corpus, so the reconstruction figures
//! are about unseen machines and not about unseen seconds of a seen machine.

use ferric_signal::{
    best_dct_baseline, decoder_forward_var, forward_var, mse, shuffled, straight_through, DecoderWeights,
    EncoderConfig, EncoderWeights, Fsq, MatFile, Patcher, RevIn, Weights,
};
use ferric_core::Context;
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

/// Samples per patch. A fixed sample count, not a fixed duration — see the module docs.
const PATCH: usize = 128;
/// Samples per window, so a window is 16 patches.
const WINDOW: usize = PATCH * 16;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// One corpus's contribution: normalized, patched windows, split by recording.
struct Corpus {
    name: &'static str,
    train: Vec<Vec<f32>>,
    held: Vec<Vec<f32>>,
    /// Recordings actually read, and how many were passed over to bound the I/O.
    files: usize,
    skipped: usize,
}

/// Cut `count` windows spread across a channel, normalize each, and patch it.
///
/// Spread rather than taken from the start: a recording's first second is not representative of it,
/// and every corpus here is ordered by something.
fn windows_of(chan: &[f32], count: usize, patcher: &Patcher, out: &mut Vec<Vec<f32>>) {
    if chan.len() < WINDOW || count == 0 {
        return;
    }
    let stride = if count > 1 { (chan.len() - WINDOW) / (count - 1) } else { 0 };
    for w in 0..count {
        let start = w * stride;
        let raw = &chan[start..start + WINDOW];
        let Ok(rev) = RevIn::fit(raw, 1) else { continue };
        let Ok(norm) = rev.apply(raw) else { continue };
        let Ok(px) = patcher.patchify(&norm) else { continue };
        out.push(px);
    }
}

/// Every numeric series in a `.mat` file long enough to window, longest first.
fn mat_channels(path: &std::path::Path, want: usize) -> Vec<Vec<f32>> {
    let Ok(bytes) = std::fs::read(path) else { return Vec::new() };
    let Ok(m) = MatFile::parse(&bytes) else { return Vec::new() };
    let mut v: Vec<Vec<f32>> = m
        .channels()
        .into_iter()
        .filter(|(_, s)| s.len() >= WINDOW)
        .map(|(_, s)| s.iter().map(|&x| x as f32).collect())
        .collect();
    v.sort_by_key(|c| std::cmp::Reverse(c.len()));
    v.truncate(want);
    v
}

fn mat_files(dir: &str) -> Vec<std::path::PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut v: Vec<_> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "mat"))
        .collect();
    v.sort();
    v
}

/// A `.mat` corpus: files are the unit of the split, so no recording appears on both sides.
///
/// `max_files` bounds the I/O rather than the science. Reading all 45 rotating recordings means
/// parsing 3.4 GB to keep a few hundred windows; a strided subset spans the same conditions at a
/// fraction of the cost. The count actually used is printed, so a reader can see what was skipped.
fn load_mat(
    name: &'static str,
    dir: &str,
    per_file: usize,
    chans: usize,
    patcher: &Patcher,
    max_files: usize,
) -> Corpus {
    let all = mat_files(dir);
    let stride = (all.len() / max_files.max(1)).max(1);
    let files: Vec<_> = all.iter().step_by(stride).take(max_files).cloned().collect();
    let mut train = Vec::new();
    let mut held = Vec::new();
    // Every fourth file is held out; with fewer than four the LAST file is, so a small corpus
    // still has a held-out set instead of silently reporting none.
    let held_every = if files.len() >= 4 { 4 } else { files.len().max(1) };
    for (i, p) in files.iter().enumerate() {
        let is_held = if files.len() >= 4 { i % held_every == 3 } else { i + 1 == files.len() };
        let into = if is_held { &mut held } else { &mut train };
        for c in mat_channels(p, chans) {
            windows_of(&c, per_file, patcher, into);
        }
    }
    Corpus { name, train, held, files: files.len(), skipped: all.len() - files.len() }
}

/// The hydraulic corpus ships as text, one row per cycle. Cycles are the unit of the split.
fn load_hydraulic(dir: &str, cycles: usize, patcher: &Patcher) -> Corpus {
    // Only the channels long enough to hold a window at this patch length; the 1 Hz channels are
    // 60 samples per cycle and cannot.
    const CHANS: &[(&str, usize)] = &[
        ("PS1", 6000), ("PS2", 6000), ("PS3", 6000), ("PS4", 6000), ("PS5", 6000), ("PS6", 6000),
        ("EPS1", 6000),
    ];
    let mut train = Vec::new();
    let mut held = Vec::new();
    for &(name, cols) in CHANS {
        let path = format!("{dir}/{name}.txt");
        let Ok(f) = std::fs::File::open(&path) else { continue };
        for (i, line) in BufReader::new(f).lines().enumerate() {
            if i >= cycles {
                break;
            }
            let Ok(line) = line else { continue };
            let vals: Vec<f32> = line.split_whitespace().filter_map(|v| v.parse().ok()).collect();
            if vals.len() != cols {
                continue;
            }
            let into = if i % 4 == 3 { &mut held } else { &mut train };
            windows_of(&vals, 2, patcher, into);
        }
    }
    Corpus { name: "hydraulic", train, held, files: cycles, skipped: 0 }
}

/// Reconstruction of a set of windows through the ACTUAL quantizer, not the straight-through
/// estimator: at inference there is no gradient to pass, and a figure measured on the estimator's
/// output would be measuring the training path.
fn recon(
    ctx: &Arc<Context>,
    cfg: EncoderConfig,
    q: &Fsq,
    enc: &EncoderWeights,
    dec: &[Var],
    blocks: &[Vec<f32>],
    codes: &mut HashSet<u32>,
) -> (f64, f64) {
    let (mut se, mut n, mut sum, mut sumsq) = (0.0f64, 0usize, 0.0f64, 0.0f64);
    for b in blocks {
        let t = b.len() / cfg.patch_len;
        let Ok(lat_t) = enc.forward(ctx, &Tensor::from_vec(ctx, b, &[t, cfg.patch_len])) else {
            continue;
        };
        let lat = pollster::block_on(lat_t.to_vec());
        let mut deq = Vec::with_capacity(t * cfg.latent_dim);
        for i in 0..t {
            let c = q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap();
            codes.insert(q.to_index(&c).unwrap());
            deq.extend(q.dequantize(&c).unwrap());
        }
        let zq = Var::leaf(Tensor::from_vec(ctx, &deq, &[t, cfg.latent_dim]));
        let Ok(rv) = decoder_forward_var(ctx, cfg, dec, &zq) else { continue };
        let r = pollster::block_on(rv.value().to_vec());
        for (a, e) in r.iter().zip(b.iter()) {
            se += ((a - e) as f64) * ((a - e) as f64);
            sum += *e as f64;
            sumsq += (*e as f64) * (*e as f64);
            n += 1;
        }
    }
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = sum / n as f64;
    (se / n as f64, sumsq / n as f64 - mean * mean)
}

/// Scale every gradient by one factor so the global L2 norm is at most `max`.
///
/// One factor for all of them: clipping tensor by tensor would rescale some and not others, which
/// changes the DIRECTION of the update, not just its length.
fn clip_global(ctx: &Arc<Context>, grads: &[Tensor], max: f32) -> Vec<Tensor> {
    let mut sq = 0.0f64;
    let host: Vec<Vec<f32>> = grads.iter().map(|g| pollster::block_on(g.to_vec())).collect();
    for h in &host {
        for v in h {
            sq += (*v as f64) * (*v as f64);
        }
    }
    let norm = sq.sqrt() as f32;
    if !norm.is_finite() || norm <= max {
        return grads.to_vec();
    }
    let scale = max / norm;
    grads
        .iter()
        .zip(host.iter())
        .map(|(g, h)| {
            let scaled: Vec<f32> = h.iter().map(|v| v * scale).collect();
            Tensor::from_vec(ctx, &scaled, &g.shape)
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(4000);
    let per_file: usize = flag(&args, "--windows").and_then(|v| v.parse().ok()).unwrap_or(4);
    let out = flag(&args, "--out");
    // Bounds the I/O, not the science: reading every rotating recording means parsing 3.4 GB to
    // keep a few hundred windows. Raised deliberately when the question is whether the vibration
    // corpora were starved rather than rate-limited.
    let max_files: usize = flag(&args, "--max-files").and_then(|v| v.parse().ok()).unwrap_or(40);

    let patcher = Patcher::contiguous(PATCH).unwrap();
    let lr: f32 = flag(&args, "--lr").and_then(|v| v.parse().ok()).unwrap_or(3e-4);
    let clip: f32 = flag(&args, "--clip").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let mut corpora: Vec<Corpus> = Vec::new();
    if let Some(d) = flag(&args, "--hydraulic") {
        corpora.push(load_hydraulic(&d, 400, &patcher));
    }
    if let Some(d) = flag(&args, "--cwru") {
        corpora.push(load_mat("cwru", &d, per_file, 2, &patcher, max_files));
    }
    if let Some(d) = flag(&args, "--rotating") {
        corpora.push(load_mat("rotating", &d, per_file, 4, &patcher, max_files.min(45)));
    }
    if let Some(d) = flag(&args, "--wind") {
        corpora.push(load_mat("wind", &d, per_file, 6, &patcher, max_files.min(10)));
    }
    // EVERY WINDOW IS CHECKED BEFORE ANY OF IT IS TRAINED ON. A single non-finite sample anywhere
    // in four corpora poisons the loss the first time its window is drawn, which surfaces as a
    // divergence hundreds of steps in and looks like an optimizer problem. Cheap to check, and it
    // separates "the data has a hole in it" from "the learning rate is too high" before either
    // costs an hour.
    let mut dropped = 0usize;
    for c in corpora.iter_mut() {
        let ok = |w: &Vec<f32>| w.iter().all(|v| v.is_finite());
        let before = c.train.len() + c.held.len();
        c.train.retain(ok);
        c.held.retain(ok);
        dropped += before - c.train.len() - c.held.len();
    }
    if dropped > 0 {
        println!("  {dropped} windows dropped for non-finite samples");
    }
    corpora.retain(|c| !c.train.is_empty());
    if corpora.is_empty() {
        eprintln!("usage: --hydraulic <dir> --cwru <dir> --rotating <dir> --wind <dir>");
        std::process::exit(2);
    }

    let cfg = EncoderConfig {
        patch_len: PATCH, d_model: 256, n_layers: 5, n_heads: 4, d_ff: 896, latent_dim: 5,
    };
    let params = cfg.params();
    println!("\nUNIVERSAL SENSOR TOKENIZER");
    println!("  {} patch, {} window, {} patches per window", PATCH, WINDOW, WINDOW / PATCH);
    println!("  encoder {} + decoder {} = {} parameters, against the published 9.5M",
             params.total(), cfg.decoder_params(), params.total() + cfg.decoder_params());
    println!("\n  {:<12} {:>10} {:>10} {:>10} {:>10}",
             "corpus", "train", "held out", "files", "skipped");
    println!("  {:-<12} {:->10} {:->10} {:->10} {:->10}", "", "", "", "", "");
    for c in &corpora {
        println!("  {:<12} {:>10} {:>10} {:>10} {:>10}",
                 c.name, c.train.len(), c.held.len(), c.files, c.skipped);
    }

    // The baseline is a property of the DATA, not of the model, so it can be measured on its own.
    // Useful when a training run is already in flight, or when only the baseline has changed.
    // Corpus loading is deterministic given the same flags, so the window counts printed above are
    // what pairs this with a training run — check them before pairing anything.
    if args.iter().any(|a| a == "--baseline-only") {
        println!("\n  MATCHED-BIT-RATE BASELINE ONLY. No model is trained or evaluated here.");
        println!("\n  {:<12} {:>8} {:>12} {:>10}  {}",
                 "corpus", "windows", "MSE", "SNR", "strongest coder");
        println!("  {:-<12} {:->8} {:->12} {:->10}  {:->16}", "", "", "", "", "");
        for c in &corpora {
            let (mse, code) = best_dct_baseline(c.held.iter().flat_map(|b| b.chunks(PATCH)));
            let (mut sum, mut sumsq, mut cnt) = (0.0f64, 0.0f64, 0usize);
            for b in &c.held {
                for &v in b {
                    sum += v as f64;
                    sumsq += (v as f64) * (v as f64);
                    cnt += 1;
                }
            }
            let cf = cnt.max(1) as f64;
            let var = sumsq / cf - (sum / cf) * (sum / cf);
            let snr = 10.0 * (var / mse.max(1e-12)).log10();
            println!("  {:<12} {:>8} {mse:>12.5} {snr:>9.1}dB  {code:?}", c.name, c.held.len());
        }
        println!("\n  15 bits per {PATCH}-sample patch: 7 to name the largest DCT coefficient,");
        println!("  8 to quantize it. The same 15 bits an FSQ code over 32,768 spends.\n");
        return;
    }

    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("no GPU context; this example needs one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);
    let q = Fsq::signal_15bit();
    let enc0 = EncoderWeights::deterministic(&ctx, cfg, 11).unwrap();
    let dec0 = DecoderWeights::deterministic(&ctx, cfg, 11 ^ 0x5DEE).unwrap();
    let n_enc = enc0.params_flat().len();
    let mut weights: Vec<Tensor> =
        enc0.params_flat().into_iter().chain(dec0.params_flat()).collect();
    let mut opt = Adam::new(&weights, lr);

    // ROUND-ROBIN ACROSS CORPORA, one window per step. Sampling proportionally to corpus size
    // would let the largest one set the weights and then be reported as the tokenizer's strength;
    // round-robin makes every corpus contribute the same number of gradients regardless of how
    // many windows it happens to contain.
    let orders: Vec<Vec<usize>> = corpora
        .iter()
        .enumerate()
        .map(|(i, c)| shuffled(c.train.len(), 0xA11CE + i as u64))
        .collect();

    println!("\n  {steps} steps, round-robin across {} corpora, one window per step\n", corpora.len());
    print!("  {:>8}", "step");
    for c in &corpora {
        print!(" {:>12}", c.name);
    }
    println!("     (mean recon MSE over the last 64 windows of each)");
    print!("  {:->8}", "");
    for _ in &corpora {
        print!(" {:->12}", "");
    }
    println!();
    let mut recent: Vec<Vec<f32>> = vec![Vec::new(); corpora.len()];
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for step in 0..steps {
        let ci = step % corpora.len();
        let round = step / corpora.len();
        let idx = orders[ci][round % orders[ci].len()];
        let b = &corpora[ci].train[idx];
        let t = b.len() / PATCH;

        let vars: Vec<Var> = weights.iter().cloned().map(Var::leaf).collect();
        let x = Var::leaf(Tensor::from_vec(&ctx, b, &[t, PATCH]));
        let z = forward_var(&ctx, cfg, &vars[..n_enc], &x).unwrap();
        let zq = straight_through(&ctx, &z, &q);
        let r = decoder_forward_var(&ctx, cfg, &vars[n_enc..], &zq).unwrap();
        let loss = mse(&r, &x);
        loss.backward();
        let l = pollster::block_on(loss.value().to_vec())[0];
        if step == 0 {
            first = l;
        }
        if !l.is_finite() {
            eprintln!("error: loss became {l} at step {step}");
            std::process::exit(1);
        }
        last = l;
        // A RUNNING MEAN PER CORPUS, NOT THE LAST STEP'S LOSS. Round-robin means consecutive steps
        // are different corpora with very different difficulty, so a single step's loss says which
        // corpus it landed on and almost nothing about the training state. The first version of
        // this example printed 0.007 at step 2400 and 0.773 at step 4000 and neither was a fact
        // about convergence.
        recent[ci].push(l);
        if step % (steps / 12).max(1) == 0 || step + 1 == steps {
            print!("  {step:>8}");
            for r in recent.iter_mut() {
                let n = r.len().min(64);
                let m = if n == 0 { f32::NAN } else { r[r.len() - n..].iter().sum::<f32>() / n as f32 };
                print!(" {m:>12.5}");
            }
            println!();
        }
        // GLOBAL-NORM GRADIENT CLIPPING. A five-layer pre-norm tower is deeper than anything else
        // this crate trains, and the first attempt at it diverged to NaN at step 343 — a failure
        // that produces no gradient signal to learn from and simply ends the run. Clipping is the
        // cheap insurance; the norm is computed across ALL parameters at once, because clipping
        // each tensor separately changes the direction of the update and not only its size.
        let grads: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
        let grads = if clip > 0.0 { clip_global(&ctx, &grads, clip) } else { grads };
        opt.step(&mut weights, &grads);
    }
    let _ = last;
    println!("\n  first step {first:.6}");

    // Round-trip the trained encoder through the on-disk format: the weights that tokenize are
    // then weights that could have been loaded from a file.
    let names = EncoderWeights::tensor_names(cfg.n_layers);
    let refs: Vec<(&str, &Tensor)> =
        names.iter().map(|s| s.as_str()).zip(weights[..n_enc].iter()).collect();
    let file = Weights::from_tensors(&refs);
    let digest = file.digest();
    let enc = EncoderWeights::from_weights(&ctx, cfg, &file).unwrap();
    let dec: Vec<Var> = weights[n_enc..].iter().cloned().map(Var::leaf).collect();

    // PER CORPUS, HELD OUT. One pooled number would let an easy corpus carry the hard ones, and
    // the whole claim a universal tokenizer makes is about the hard ones.
    println!("\n  HELD-OUT RECONSTRUCTION, PER CORPUS");
    println!("  recordings the training set never touched.\n");
    println!("  {:<12} {:>8} {:>12} {:>10} {:>12} {:>12}  {}",
             "corpus", "windows", "MSE", "SNR", "1-coef DCT", "codes used", "coder");
    println!("  {:-<12} {:->8} {:->12} {:->10} {:->12} {:->12}", "", "", "", "", "", "");
    let mut all_codes: HashSet<u32> = HashSet::new();
    for c in &corpora {
        let mut codes = HashSet::new();
        let (m, v) = recon(&ctx, cfg, &q, &enc, &dec, &c.held, &mut codes);
        let snr = 10.0 * (v / m.max(1e-12)).log10();
        // The baseline gets the SAME 15 bits per patch and no training, at whichever of the
        // matched-budget value coders is STRONGEST on this corpus — a baseline chosen to be weak
        // would flatter the model.
        let (bmse, code) = best_dct_baseline(c.held.iter().flat_map(|b| b.chunks(PATCH)));
        let bsnr = 10.0 * (v / bmse.max(1e-12)).log10();
        println!("  {:<12} {:>8} {m:>12.5} {snr:>9.1}dB {bsnr:>11.1}dB {:>12}  {code:?}",
                 c.name, c.held.len(), codes.len());
        all_codes.extend(&codes);
    }
    println!("\n  `1-coef DCT` spends the same 15 bits per 128-sample patch with no training:");
    println!("  7 to name the largest DCT coefficient, 8 to quantize it. A corpus where the");
    println!("  tokenizer barely beats it is telling you about the SIGNAL, not the training.");
    println!("\n  {} of {} codes visited across all corpora ({:.1}%)",
             all_codes.len(), q.codebook_size(),
             all_codes.len() as f64 / q.codebook_size() as f64 * 100.0);
    println!("  weights digest {digest}");

    if let Some(path) = out {
        match std::fs::write(&path, file.to_bytes()) {
            Ok(()) => println!("  wrote {path} ({} bytes)", std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)),
            Err(e) => eprintln!("  error writing {path}: {e}"),
        }
    }
    println!();
}
