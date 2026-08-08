//! **Where does going lower stop paying?** The quantization crossover, measured on one device class.
//!
//! An open empirical question with a decision-relevant answer, and it decides what this runtime should
//! ship as its default.
//!
//! One side: SINTEF et al. wired a Joulescope JS110 in series at 2 MHz on a Raspberry Pi 4 and measured
//! **Q8_0 as the energy optimum, with lower bit-widths costing MORE joules**. Llama 3.2 1B went
//! 159.42 J/response at FP16 to 75.85 J at Q8_0, then back UP: Q4 variants averaged 83.95 J and Q3
//! averaged 101.02 J, with Q3 also losing 20-40% accuracy (arXiv:2504.03360, ACM Trans. IoT 2026).
//!
//! The other side: essentially the entire compression layer, racing toward 1.58 bits.
//!
//! Both cannot be right across all hardware. The crossover as a function of memory bandwidth, SIMD
//! width and dequantization cost is published nowhere.
//!
//! ## What this measures, and what it does not
//!
//! **It does not measure joules.** This machine has no RAPL, no NVIDIA counter, and runs on mains power,
//! so `ferric_joule::capability_report()` correctly reports that nothing is measurable here. Saying so
//! is the point: a joules number from this machine would be arithmetic wearing a measurement's clothes.
//!
//! What it measures is the mechanism underneath the energy result. Decode is weight-streaming bound, so
//! per-token cost tracks **bytes moved** and **dequantisation work**. Lower bit-width cuts the first and
//! raises the second, and the crossover is where the second wins. Time per token at fixed power is a
//! faithful proxy for that trade, and the bytes-per-token column shows which term is moving.
//!
//! To turn this into joules, run it on a metered device. The harness is the same.
//!
//!   cargo run -p ferric-llama --example quant_crossover --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

/// Same model, same tokenizer, same prompt. Only the quantisation differs, which is the only way the
/// comparison means anything.
const FORMATS: &[(&str, &str)] = &[
    ("Q4_K_M", "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("Q5_K_M", "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q5_k_m.gguf"),
    ("Q6_K",   "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q6_k.gguf"),
    ("Q8_0",   "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"),
    // This row used to measure a MISSING kernel and said so. It no longer does, and the history is
    // worth keeping because of how the gap hid: `Iq4XsWeights`/`matmul_iq4_xs` existed and passed
    // their own test, but `QMatrix::block_bytes` did not list ggml type 23, so the loader took the
    // `from_dense` branch and the kernel was unreachable. Nothing errored — the model just ran fatter.
    //
    // Reaching it also required IQ4_NL: this file is 250.5 M params of IQ4_NL against 104.6 M of
    // IQ4_XS, because IQ4_XS needs rows divisible by 256 and n_embd here is 896. Both kernels now run
    // and both match the CPU dequant to 3.6e-7 (see `iq4_real_weights.rs`), so 72% of the model's
    // parameters are packed rather than f32 and this row finally measures the FORMAT.
    ("IQ4_XS", "bartowski_Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-IQ4_XS.gguf"),
];

const DECODE_TOKENS: usize = 24;
const REPS: usize = 5;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    println!("Quantisation crossover — Qwen2.5-0.5B, one device class\n");
    print!("{}", ferric_joule::capability_report());

    if let Some(l) = load_avg() {
        println!("  machine load average: {l:.2}");
        assert!(l < 8.0, "load {l:.2} is too high to time anything; this same harness reported an \
                          8.6x swing on a busy machine earlier today. Wait and re-run.");
    }

    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();

    println!("\n  {:>8} {:>10} {:>13} {:>13} {:>11}", "format", "file MB", "ms/token", "MB/token", "vs Q8_0");
    println!("  {:-<60}", "");

    let mut rows: Vec<(&'static str, f64, f64, f64)> = Vec::new();
    let mut q8_ms = 0.0f64;

    for (name, rel) in FORMATS {
        let path = format!("{home}/.cache/ferric/hub/{rel}");
        let Ok(g) = GgufFile::open(&path) else {
            println!("  {name:>8}  (not present, skipped)");
            continue;
        };
        let file_mb = std::fs::metadata(&path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
        let Ok(m) = Qwen3::load(&ctx, &g) else {
            println!("  {name:>8}  (unsupported by this loader, skipped)");
            continue;
        };
        let vn = m.cfg.n_vocab;
        let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
            .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

        // Warm: first pass compiles pipelines and faults the weights in.
        let mut c = Cache::new(&m.cfg);
        let l = m.forward_cached(&[100u32, 200, 300], &mut c).to_vec().await;
        let mut next = am(&l);

        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward_cached(&[100u32, 200, 300], &mut c).to_vec().await;
            next = am(&l);
            let t0 = std::time::Instant::now();
            for _ in 0..DECODE_TOKENS {
                let l = m.forward_cached(&[next], &mut c).to_vec().await;
                next = am(&l);
            }
            samples.push(t0.elapsed().as_secs_f64() * 1000.0 / DECODE_TOKENS as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = samples[REPS / 2];

        // Bytes the decode path actually streams per token: every weight is read once.
        let mb_per_token = file_mb;
        if *name == "Q8_0" { q8_ms = ms; }
        rows.push((name, file_mb, ms, mb_per_token));
        println!("  {name:>8} {file_mb:>10.1} {ms:>13.2} {mb_per_token:>13.1} {:>10}", "");
    }

    assert!(!rows.is_empty(), "no formats loaded; the comparison measured nothing");

    println!("\n  {:>8} {:>13} {:>13} {:>14}", "format", "ms/token", "vs Q8_0", "MB/ms (eff.)");
    println!("  {:-<52}", "");
    for (name, mb, ms, _) in &rows {
        let rel = if q8_ms > 0.0 { ms / q8_ms } else { f64::NAN };
        println!("  {name:>8} {ms:>13.2} {rel:>12.2}x {:>13.1}", mb / ms);
    }

    // ---- what this actually shows ----
    //
    // The first version of this example concluded "CROSSOVER FOUND" because the smallest format was the
    // slowest. That was wrong, and the way it was wrong is the point.
    let best = rows.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let smallest = rows.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("\n  Fastest per token: {} at {:.2} ms.  Smallest on disk: {} at {:.1} MB.",
             best.0, best.2, smallest.0, smallest.1);

    // The tell: if time were tracking bytes, the ordering by time would match the ordering by size.
    let mut by_size = rows.clone();
    by_size.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut by_time = rows.clone();
    by_time.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let monotonic = by_size.iter().map(|r| r.0).eq(by_time.iter().map(|r| r.0));

    println!("\n  ⚠ THIS DOES NOT ANSWER THE CROSSOVER QUESTION, and saying why is more useful than a");
    println!("  number that looks like it does.\n");
    if !monotonic {
        println!("  Time does not track size here. Ordered by bytes: {}",
                 by_size.iter().map(|r| r.0).collect::<Vec<_>>().join(" < "));
        println!("  Ordered by time:                                 {}",
                 by_time.iter().map(|r| r.0).collect::<Vec<_>>().join(" < "));
        println!("  A smaller format that runs SLOWER than a larger one is not a bytes-versus-dequant");
        println!("  tradeoff. It is a kernel that has not been written or tuned. IQ4_XS has no native");
        println!("  packed kernel in this runtime at all: the loader dequantizes to f32 and runs the");
        println!("  dense path, an 8x memory blow-up, so its column measures a missing kernel.\n");
    }
    println!("  So what this measures is THIS RUNTIME'S KERNEL QUALITY PER FORMAT, which is a real and");
    println!("  useful result, just not the one the example set out to get. The published crossover");
    println!("  question (SINTEF: Q8_0 optimum on a Pi 4, lower widths costing more joules) needs every");
    println!("  format to have a comparably tuned kernel before the comparison means anything. Here it");
    println!("  is answered by whichever kernel happened to get attention.\n");
    println!("  The actionable finding: {} is the fastest format on this device and should be the", best.0);
    println!("  default. {} and the IQ4 family are the kernels worth writing next, and until they are",
             rows.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).map(|r| r.0).unwrap_or("?"));
    println!("  written, no statement about the physical crossover can be made from this runtime.\n");
    println!("  This is the field's own caution landing on us: on identical Snapdragon silicon, two");
    println!("  runtimes differ 13x on the same model from kernel quality alone, against ~1.3x from the");
    println!("  architecture choice. Measure the kernel before theorising about the format.");

    println!("\n  ⚠ This is TIME, not joules. This machine has no readable power counter, and a joules");
    println!("  figure from it would be arithmetic. Time per token at fixed power is a faithful proxy");
    println!("  for the underlying trade (bytes moved vs dequant work) and nothing more. Run the same");
    println!("  harness on a metered device to get the energy answer; ferric-joule is already wired to");
    println!("  report which class the number belongs to.");
}
