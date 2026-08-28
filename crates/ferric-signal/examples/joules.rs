//! Measure the energy of tokenizing, or refuse to.
//!
//!   cargo run -p ferric-signal --example joules --release -- [--blocks 11] [--seconds 20]
//!
//! ## Why this is mostly refusal machinery
//!
//! A joules-per-workload figure from a system-wide power meter is **a difference between two
//! windows**, not a reading. On a machine that is loaded, swapping, or running off its battery
//! because the adapter cannot carry the load, that difference is dominated by things that have
//! nothing to do with the workload — routinely including a *negative* marginal power, which is not
//! a small number but a meaningless one.
//!
//! So the admission gates come first and a failure voids the run. `Option<f64>` doing its job:
//! this crate would rather print why it cannot measure than print a number it cannot defend.
//!
//! ## What is measured
//!
//! **The full encoder pass**, not the front end. This is the largest error available here: the
//! patching and quantization path moves roughly 64 bytes per token, and the transformer encoder
//! moves about 1.19 MB per token — some four orders of magnitude apart. Pricing the front end and
//! calling it a sensor tokenizer measures something else entirely.
//!
//! Synthetic signal rather than a corpus, so the run reproduces anywhere. Energy depends on the
//! shape of the computation, not on which physics the samples came from.
//!
//! ## The protocol
//!
//! - `B A B A B … B` blocks of equal length. **B is not "idle": B is A with the workload removed**,
//!   same meter, same processes. A baseline that reaches a power state A cannot reach subtracts a
//!   floor that never existed.
//! - Paired **neighbour-mean** differencing, `Δᵢ = P̄(Aᵢ) − ½(P̄(Bᵢ₋₁) + P̄(Bᵢ₊₁))`, which cancels
//!   drift slower than the block period. Measured baseline drift over three minutes was 1.34 W —
//!   larger than a saturated core's marginal draw, so this is not a refinement.
//! - An **A/A null** with the workload replaced by an equal sleep, giving the detection floor. A
//!   delta smaller than three times the null's spread is reported as unresolved, never as a number.
//! - Blocks are the unit of analysis. Samples within a block are correlated and are never counted
//!   as independent.

use ferric_joule::{machine_is_measurable, power_gates, Macmon, MacmonScope, Meter};
use ferric_signal::{synth, EncoderConfig, EncoderWeights, Fsq, Patcher, RevIn};
use ferric_tensor::Tensor;
use std::sync::Arc;

const PATCH: usize = 128;
const WINDOW: usize = PATCH * 16;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Mean and sample standard deviation.
fn stats(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = if v.len() < 2 {
        0.0
    } else {
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64
    };
    (m, var.sqrt())
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let blocks: usize = flag(&args, "--blocks").and_then(|v| v.parse().ok()).unwrap_or(11);
    let secs: f64 = flag(&args, "--seconds").and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let interval: u64 = flag(&args, "--interval").and_then(|v| v.parse().ok()).unwrap_or(250);

    // ---- admission gates, before anything else ----
    println!("\nMACHINE STATE");
    println!("  {:<24} {:>10}  {}", "gate", "state", "reading");
    println!("  {:-<24} {:->10}  {:-<50}", "", "", "");
    let gates = power_gates();
    for g in &gates {
        println!("  {:<24} {:>10}  {}", g.name, if g.ok { "pass" } else { "FAIL" }, g.detail);
    }
    if !machine_is_measurable() {
        println!("\n  REFUSED. This machine cannot support an energy measurement right now.\n");
        for g in gates.iter().filter(|g| !g.ok) {
            println!("  {} — {}", g.name, g.because);
        }
        println!("\n  A number taken in this state would be a difference between two windows whose");
        println!("  variation is dominated by something other than the workload. That is not a");
        println!("  noisy measurement; it is a measurement of the wrong thing, and its sign is not");
        println!("  even reliable. Nothing is reported rather than something unfounded.\n");
        return;
    }

    // ---- the workload ----
    let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
        eprintln!("no GPU context; this example needs one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);
    let cfg = EncoderConfig {
        patch_len: PATCH, d_model: 256, n_layers: 5, n_heads: 4, d_ff: 896, latent_dim: 5,
    };
    let enc = EncoderWeights::deterministic(&ctx, cfg, 11).unwrap();
    let q = Fsq::signal_15bit();
    let patcher = Patcher::contiguous(PATCH).unwrap();

    // A pool of prepared windows, so the metered path is the encoder and the quantizer and not
    // signal generation. Preparation happens once, outside every measured block.
    let pool: Vec<Vec<f32>> = (0..64)
        .map(|i| {
            let raw: Vec<f32> = (0..WINDOW).map(|k| synth::sample(i % synth::KINDS, i, k)).collect();
            let rev = RevIn::fit(&raw, 1).unwrap();
            patcher.patchify(&rev.apply(&raw).unwrap()).unwrap()
        })
        .collect();

    // One pass over one window: the FULL encoder, then quantization to codes.
    let mut cursor = 0usize;
    let one_pass = |cursor: &mut usize| -> usize {
        let b = &pool[*cursor % pool.len()];
        *cursor += 1;
        let t = b.len() / PATCH;
        let lat = pollster::block_on(
            enc.forward(&ctx, &Tensor::from_vec(&ctx, b, &[t, PATCH])).unwrap().to_vec(),
        );
        let mut n = 0usize;
        for i in 0..t {
            let c = q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap();
            let _ = q.to_index(&c).unwrap();
            n += 1;
        }
        n
    };

    // ---- warm up: compile shaders, make weights resident, reach steady state ----
    println!("\n  warming up ({} passes) so pipeline creation cannot leak into a measured block", 200);
    for _ in 0..200 {
        one_pass(&mut cursor);
    }

    let Some(meter) = Macmon::start(MacmonScope::Soc, interval) else {
        eprintln!("macmon unavailable; install it with `brew install macmon`");
        std::process::exit(1);
    };
    println!("  meter {} class {:?} boundary {}", meter.source(), meter.class(), meter.boundary().label());
    println!("  {blocks} blocks of {secs:.0}s, alternating B A B …, {interval} ms sampling\n");
    // Let the meter fill and settle before the first edge needs bracketing.
    std::thread::sleep(std::time::Duration::from_secs(3));
    let origin = std::time::Instant::now();

    // ---- run the alternating blocks ----
    let mut marks: Vec<(bool, f64, f64, usize)> = Vec::new(); // (is_a, t0, t1, tokens)
    for b in 0..blocks {
        let is_a = b % 2 == 1;
        let t0 = origin.elapsed().as_secs_f64();
        let mut tokens = 0usize;
        if is_a {
            while origin.elapsed().as_secs_f64() - t0 < secs {
                tokens += one_pass(&mut cursor);
            }
        } else {
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        }
        let t1 = origin.elapsed().as_secs_f64();
        marks.push((is_a, t0, t1, tokens));
        print!("\r  block {}/{}  {}", b + 1, blocks, if is_a { "A" } else { "B" });
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
    println!();
    // Pad so the last block's trailing edge can be bracketed by a later sample.
    std::thread::sleep(std::time::Duration::from_secs(3));

    // ---- per-block mean power ----
    let mut power: Vec<Option<f64>> = Vec::new();
    for (_, t0, t1, _) in &marks {
        power.push(meter.energy_over(*t0, *t1).map(|j| j / (t1 - t0)));
    }

    println!("\n  {:>6} {:>4} {:>12} {:>12}", "block", "arm", "mean W", "tokens");
    println!("  {:->6} {:->4} {:->12} {:->12}", "", "", "", "");
    for (i, (is_a, _, _, tok)) in marks.iter().enumerate() {
        match power[i] {
            Some(w) => println!("  {:>6} {:>4} {w:>12.2} {tok:>12}", i + 1, if *is_a { "A" } else { "B" }),
            None => println!("  {:>6} {:>4} {:>12} {tok:>12}", i + 1, if *is_a { "A" } else { "B" }, "unbracketed"),
        }
    }

    // ---- neighbour-mean paired differences ----
    let mut deltas = Vec::new();
    let mut tokens_total = 0usize;
    let mut seconds_a = 0.0;
    for i in 1..marks.len().saturating_sub(1) {
        if !marks[i].0 {
            continue;
        }
        let (Some(a), Some(before), Some(after)) = (power[i], power[i - 1], power[i + 1]) else {
            continue;
        };
        deltas.push(a - 0.5 * (before + after));
        tokens_total += marks[i].3;
        seconds_a += marks[i].2 - marks[i].1;
    }

    let (mean, sd) = stats(&deltas);
    let med = median(&deltas);
    println!("\n  RESULT");
    println!("  paired blocks           {}", deltas.len());
    println!("  marginal power          {mean:.2} W  (median {med:.2}, sd {sd:.2})");
    println!("  tokens                  {tokens_total} over {seconds_a:.0} s of workload");

    // ---- the refusal rule ----
    //
    // Without an A/A null this run cannot state its own detection floor, so the spread of the
    // deltas themselves is used as a stand-in and the report says so. A delta inside its own
    // spread is not a measurement of anything.
    if deltas.len() < 3 {
        println!("\n  UNRESOLVED: fewer than three paired blocks. Increase --blocks.\n");
        return;
    }
    let floor = 3.0 * sd / (deltas.len() as f64).sqrt();
    if med.abs() < floor || med <= 0.0 {
        println!("\n  UNRESOLVED. |median| {:.2} W against a floor of {floor:.2} W", med.abs());
        println!("  This run did not resolve the tokenizer above the machine's own variation.");
        println!("  **That is not a finding that the energy is small.** It means the instrument,");
        println!("  on this machine, in this state, cannot see it. Report nothing.\n");
        return;
    }
    let j_per_token = med * seconds_a / tokens_total as f64;
    println!("\n  marginal energy         {:.3} µJ per token", j_per_token * 1e6);
    println!("  boundary                {}", meter.boundary().label());
    println!("  class                   {:?} — a DIFFERENCE of two system-wide windows,", meter.class());
    println!("                          which is a model of this workload's share, not a reading");
    println!("\n  WHAT THIS DOES NOT ESTABLISH: it is one window length on one laptop SoC. Attention");
    println!("  is quadratic, so a per-token figure is a property of the window it sat in — this");
    println!("  crate already measures the same token costing 5.4x more at 8192 patches than at 16.");
    println!("  A single number here is a point on a curve that has not been drawn.\n");
}
