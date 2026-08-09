//! **Where does going lower stop paying?** The quantization crossover, measured on one device class.
//!
//! One side: SINTEF et al. wired a Joulescope JS110 in series at 2 MHz on a Raspberry Pi 4 and measured
//! **Q8_0 as the energy optimum, with lower bit-widths costing MORE joules**. Llama 3.2 1B went
//! 159.42 J/response at FP16 to 75.85 J at Q8_0, then back UP: Q4 variants averaged 83.95 J and Q3
//! averaged 101.02 J, with Q3 also losing 20-40% accuracy (arXiv:2504.03360, ACM Trans. IoT 2026).
//! The other side: essentially the entire compression layer, racing toward 1.58 bits. Both cannot be
//! right across all hardware, and the crossover as a function of memory bandwidth, SIMD width and
//! dequantisation cost is published nowhere.
//!
//! ## ⚠ Rebuilt 2026-08-08: the first design reported absolute times, and they were not reproducible
//!
//! Repeating the original harness at machine loads of 2.17, 3.42, 3.78 and 3.91 — all of which passed
//! its load gate — gave Q8_0 at **36.22, 12.52, 10.75 and 11.62 ms/token**, a 3.4x spread on identical
//! work. Any ranking read off one such run is a ranking of runs.
//!
//! The structure of the noise is what makes it fixable. It is **bimodal and whole-run**: a process comes
//! up in a ~11 ms regime or a ~37 ms regime and every format inside it moves together, which is a device
//! clock/power state, not kernel quality. So within one process the factor is *common*, and it cancels
//! in a ratio:
//!
//! | run | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | IQ4_XS |
//! |---|---|---|---|---|---|
//! | ~37 ms regime | 1.11 | 1.10 | 1.01 | 1.00 | 1.14 |
//! | ~37 ms regime | 1.00 | 1.03 | 1.03 | 1.00 | 1.05 |
//! | ~11 ms regime | 1.07 | 1.06 | 1.07 | 1.00 | 1.14 |
//!
//! Note what this rules out: **one process per format would have been worse**, not better. It looks like
//! the tidier design, but it gives every format an independent draw of the run-level factor and turns a
//! cancelling common term into per-format noise.
//!
//! So: all formats in one process (the factor cancels), **compare ratios rather than absolute times**,
//! **rotate the order** each repeat so no format keeps the first or last slot, and repeat the whole
//! process several times to see whether the ratios hold. The spread of the ratios is printed, and the
//! run refuses to rank anything when that spread is as large as the differences it would rank.
//!
//! Withdrawn from the previous design: "IQ4_XS is the second-fastest format" and a "2-3x kernel-quality
//! spread between formats". Both were differences between runs read as differences between formats.
//!
//! **Not joules.** No RAPL, no NVIDIA counter, mains power: `ferric_joule::capability_report()` reports
//! nothing measurable, and a joules number from here would be arithmetic wearing a measurement's clothes.
//!
//!   cargo run -p ferric-llama --example quant_crossover --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

/// Same model, same tokenizer, same prompt. Only the quantisation differs.
const FORMATS: &[(&str, &str)] = &[
    ("Q4_K_M", "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("Q5_K_M", "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q5_k_m.gguf"),
    ("Q6_K",   "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q6_k.gguf"),
    ("Q8_0",   "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"),
    // IQ4_XS/IQ4_NL kernels were wired on 2026-08-08 — they existed and were correct, but
    // `QMatrix::block_bytes` omitted ggml type 23 so the loader silently took the f32 dense fallback.
    // 72% of this file's parameters are now packed; both kernels match the CPU dequant to 3.6e-7 on
    // real weights (`iq4_real_weights.rs`).
    ("IQ4_XS", "bartowski_Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-IQ4_XS.gguf"),
];

/// The format every ratio is taken against. Must be present or the run has no baseline.
const BASELINE: &str = "Q8_0";

const DECODE_TOKENS: usize = 24;
const REPS: usize = 5;
/// Independent processes. Each contributes one ratio per format; their spread is the noise floor.
const REPEATS: usize = 4;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() {
    match std::env::var("FERRIC_XOVER_ROT") {
        Ok(r) => pollster::block_on(child(r.parse().expect("rotation"))),
        Err(_) => parent(),
    }
}

/// Measure every format once, in this process, starting at offset `rot`.
///
/// All formats share this process on purpose: the run-level clock factor is common to them here and
/// divides out of the ratios the parent computes. `rot` rotates the visiting order so that across
/// repeats no format is always measured first (cold caches) or last (whatever has accumulated).
async fn child(rot: usize) {
    let home = std::env::var("HOME").unwrap();
    let ctx = Arc::new(Context::new().await.unwrap());

    // Load one model and return a closure-friendly bundle. Kept small so the baseline can stay
    // resident while each candidate is loaded and dropped around it.
    async fn measure(ctx: &Arc<Context>, home: &str, rel: &str) -> Option<(f64, f64)> {
        let path = format!("{home}/.cache/ferric/hub/{rel}");
        let g = GgufFile::open(&path).ok()?;
        let file_mb = std::fs::metadata(&path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
        let embd_mb = g.tensor("token_embd.weight")
            .and_then(|t| ferric_gguf::type_size(t.ggml_type, t.dims.iter().product::<u64>() as usize).ok())
            .unwrap_or(0) as f64 / 1e6;
        let m = Qwen3::load(ctx, &g).ok()?;
        Some((time_decode(&m).await, file_mb - embd_mb))
    }

    // ---- bracketing ----
    //
    // Ratios assume the run-level clock factor is COMMON to everything in the process. The rotated-repeat
    // version showed that assumption failing: most repeats put every format at 1.0-1.14x of Q8_0 and one
    // put every format at ~1.6x, which is the regime switching partway THROUGH a process, not between
    // processes. A single baseline measured once per process cannot see that.
    //
    // So the baseline is measured immediately before and immediately after each candidate, and the
    // candidate is divided by the mean of its two neighbours. Drift slower than one measurement pair
    // cancels; a regime change shows up as a gap between the bracketing pair, which is reported rather
    // than averaged away. This is ordinary bracketing, the same reason a balance is re-zeroed between
    // weighings rather than once at the start.
    let base_rel = FORMATS.iter().find(|f| f.0 == BASELINE).expect("baseline format").1;
    let n = FORMATS.len();
    let mut prev_base = match measure(&ctx, &home, base_rel).await {
        Some((t, mb)) => { println!("RESULT {BASELINE} 1.0 {mb} 0.0"); println!("BASEABS {t}"); t }
        None => return,
    };
    for step in 0..n {
        let i = (step + rot) % n;
        let (name, rel) = FORMATS[i];
        if name == BASELINE { continue; }
        let Some((t, mb)) = measure(&ctx, &home, rel).await else { continue };
        let Some((next_base, _)) = measure(&ctx, &home, base_rel).await else { continue };
        println!("BASEABS {next_base}");
        // Mean of the two brackets, plus how far apart they were: the drift this pair actually saw.
        let base = 0.5 * (prev_base + next_base);
        let drift = (next_base / prev_base).max(prev_base / next_base);
        println!("RESULT {name} {} {mb} {}", t / base, drift - 1.0);
        prev_base = next_base;
    }
}

/// Median decode ms/token over `REPS` timed runs, cache rebuilt each time.
async fn time_decode(m: &Qwen3) -> f64 {
    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;
    let mut c = Cache::new(&m.cfg);
    let mut next = am(&m.forward_cached(&[100u32, 200, 300], &mut c).to_vec().await);
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut c = Cache::new(&m.cfg);
        next = am(&m.forward_cached(&[100u32, 200, 300], &mut c).to_vec().await);
        let t0 = std::time::Instant::now();
        for _ in 0..DECODE_TOKENS {
            next = am(&m.forward_cached(&[next], &mut c).to_vec().await);
        }
        samples.push(t0.elapsed().as_secs_f64() * 1000.0 / DECODE_TOKENS as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[REPS / 2]
}

fn med(v: &mut [f64]) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn parent() {
    println!("Quantisation crossover — Qwen2.5-0.5B, ratios across rotated repeats\n");
    print!("{}", ferric_joule::capability_report());
    if let Some(l) = load_avg() {
        println!("  machine load average: {l:.2}");
        assert!(l < 4.0, "load {l:.2} is too high to time anything. Wait and re-run.");
    }

    let exe = std::env::current_exe().expect("current exe");
    // per format: streamed MB, and one ratio-to-baseline per repeat
    let mut mb: std::collections::HashMap<String, f64> = Default::default();
    // (ratio, bracket drift) so a measurement taken through a clock change can be rejected rather
    // than averaged in. A start-of-run load gate cannot see a machine that gets busy mid-run; the
    // brackets can, because they straddle exactly the interval the candidate was measured in.
    let mut ratios: std::collections::HashMap<String, Vec<(f64, f64)>> = Default::default();
    let mut abs_baseline: Vec<f64> = Vec::new();
    // Gap between the two baselines bracketing each candidate: the drift that pair actually saw.
    let mut drifts: Vec<f64> = Vec::new();

    for rep in 0..REPEATS {
        let out = std::process::Command::new(&exe)
            .env("FERRIC_XOVER_ROT", rep.to_string())
            .output().expect("child failed");
        let s = String::from_utf8_lossy(&out.stdout);
        // The child now divides by its own bracketing baselines, so the parent only collects.
        let mut saw = false;
        for l in s.lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            match f.first() {
                Some(&"RESULT") => {
                    mb.insert(f[1].to_string(), f[3].parse().unwrap());
                    let (r, d): (f64, f64) = (f[2].parse().unwrap(), f[4].parse().unwrap());
                    ratios.entry(f[1].to_string()).or_default().push((r, d));
                    drifts.push(d);
                    saw = true;
                }
                Some(&"BASEABS") => abs_baseline.push(f[1].parse().unwrap()),
                _ => {}
            }
        }
        if !saw { println!("  repeat {rep}: child produced nothing, skipped"); }
    }
    assert!(ratios.len() > 1, "nothing to compare");

    // ---- reject pairs measured through a disturbance ----
    //
    // If the baselines bracketing a candidate disagree by more than this, the clock moved while that
    // candidate was being timed and its ratio is a ratio to a number that was changing. Dropping it is
    // the honest move; keeping it and taking a median is how the first version of this harness produced
    // a confident ordering out of a 3.4x swing.
    const MAX_BRACKET_DRIFT: f64 = 0.15;
    let before: usize = ratios.values().map(|v| v.len()).sum();
    for v in ratios.values_mut() {
        if v.iter().any(|(_, d)| *d <= MAX_BRACKET_DRIFT) {
            v.retain(|(_, d)| *d <= MAX_BRACKET_DRIFT);
        }
    }
    let after: usize = ratios.values().map(|v| v.len()).sum();
    if after < before {
        println!("\n  Rejected {}/{before} measurements whose bracketing baselines disagreed by more",
                 before - after);
        println!("  than {:.0}% — the device clock moved while they were being timed.", 100.0 * MAX_BRACKET_DRIFT);
    }

    // Order the report by the FORMATS table so it reads consistently regardless of rotation.
    let names: Vec<&str> = FORMATS.iter().map(|f| f.0).filter(|n| ratios.contains_key(*n)).collect();

    println!("\n  {REPEATS} independent processes, visiting order rotated each time, {REPS} reps per format.");
    println!("  Absolute {BASELINE} time across those processes: {:.2} – {:.2} ms/token ({:.2}x).",
             abs_baseline.iter().cloned().fold(f64::INFINITY, f64::min),
             abs_baseline.iter().cloned().fold(0.0, f64::max),
             abs_baseline.iter().cloned().fold(0.0, f64::max)
                 / abs_baseline.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  That swing is the device clock state, and it is why the table below is ratios.");
    let worst_drift = drifts.iter().cloned().fold(0.0f64, f64::max);
    let med_drift = { let mut d = drifts.clone(); if d.is_empty() { 0.0 } else { med(&mut d) } };
    println!("  Baseline drift ACROSS each candidate's bracketing pair: median {:.1}%, worst {:.1}%.",
             100.0 * med_drift, 100.0 * worst_drift);
    println!("  A candidate whose brackets disagree badly was measured through a clock change; that is");
    println!("  reported here rather than averaged into a number that looks solid.\n");

    println!("  {:>8} {:>11} {:>12} {:>10} {:>10} {:>10}",
             "format", "MB/token*", "ratio med", "min", "max", "ratio sprd");
    println!("  {:-<66}", "");
    let mut worst_spread = 0.0f64;
    for n in &names {
        let mut r: Vec<f64> = ratios[*n].iter().map(|(x, _)| *x).collect();
        let (lo, hi) = (r.iter().cloned().fold(f64::INFINITY, f64::min),
                        r.iter().cloned().fold(0.0, f64::max));
        let spread = if lo > 0.0 { hi / lo } else { f64::INFINITY };
        if *n != BASELINE { worst_spread = worst_spread.max(spread); }
        println!("  {n:>8} {:>11.1} {:>12.3} {lo:>10.3} {hi:>10.3} {spread:>9.2}x", mb[*n], med(&mut r));
    }
    println!("\n  (* MB/token excludes token_embd — gathered per row, never streamed.)");

    // ---- can this table be read at all? ----
    let rvals = |n: &str| -> Vec<f64> { ratios[n].iter().map(|(x, _)| *x).collect() };
    let meds: Vec<(f64, &str)> = names.iter().map(|n| (med(&mut rvals(n)), *n)).collect();
    let best = meds.iter().cloned().fold((f64::INFINITY, ""), |a, b| if b.0 < a.0 { b } else { a });
    let worst = meds.iter().cloned().fold((0.0, ""), |a, b| if b.0 > a.0 { b } else { a });
    let signal = worst.0 / best.0;
    println!("\n  Between-format signal (slowest/fastest median ratio): {signal:.2}x");
    println!("  Worst within-format spread over identical repeats:    {worst_spread:.2}x");
    if worst_spread >= signal {
        println!("\n  ⛔ NOISE >= EFFECT. Repeating one format varies by {worst_spread:.2}x while the formats");
        println!("  differ by {signal:.2}x, so this cannot rank them and nothing here should be quoted. The");
        println!("  previous version of this harness was in exactly this state while printing a confident");
        println!("  ordering. Fix the drift before reading the table.");
        return;
    }
    println!("  → signal exceeds noise; the ordering is readable.\n");

    println!("  Fastest: {} ({:.3}x). Slowest: {} ({:.3}x).", best.1, best.0, worst.1, worst.0);

    // ---- format vs kernel: bytes are already in the denominator, so what is left is the kernel ----
    let rate = |n: &str| mb[n] / med(&mut rvals(n));
    let best_rate = names.iter().map(|n| rate(n)).fold(0.0f64, f64::max);
    let best_fmt = names.iter().max_by(|a, b| rate(a).partial_cmp(&rate(b)).unwrap()).unwrap();
    println!("\n  Relative bandwidth (MB per unit of {BASELINE}-time) isolates kernel from format. Best is");
    println!("  {best_fmt}; projecting the others onto it is an UPPER BOUND, not a target — a 4-bit codebook");
    println!("  does strictly more dequant work per byte, so equal bandwidth is not available to it.\n");
    println!("    {:>8} {:>12} {:>14} {:>11}", "format", "ratio", "at best rate", "headroom");
    for n in &names {
        let ideal = mb[*n] / best_rate;
        let r = med(&mut rvals(n));
        println!("    {n:>8} {r:>12.3} {ideal:>14.3} {:>10.2}x", r / ideal);
    }
    println!("\n  ⚠ TIME, not joules — no readable power counter here. And this is THIS RUNTIME'S kernels");
    println!("  per format, not the physical crossover: that needs every format comparably tuned, which");
    println!("  the headroom column is how you check.");
}
