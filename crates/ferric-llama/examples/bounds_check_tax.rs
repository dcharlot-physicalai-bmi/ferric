//! **What do WebGPU's mandatory bounds checks cost this runtime?**
//!
//! WebGPU requires every buffer and array access to be bounds-checked. LlamaWeb reported **14% of decode
//! and 23% of prefill** attributable to them, up to 42% on some devices. If that holds here it is one of
//! the largest single-config wins available, and if it does not, that is worth knowing before anyone
//! designs around it.
//!
//! wgpu exposes the switch through `create_shader_module_trusted`. **This needs no fork change** — an
//! earlier note in this workspace assumed one was required. `ferric_core` gates it on
//! `FERRIC_UNCHECKED_SHADERS=1`, native-only, off by default.
//!
//! ## Why this re-executes itself
//!
//! The flag is read once per process and pipelines are compiled from it, so flipping an environment
//! variable mid-run would measure whatever was compiled first. This runs the **same binary twice**, as
//! two child processes differing only in that one variable, and compares. Nothing else changes: same
//! model file, same prompt, same token counts, same code path.
//!
//! ## The check that matters more than the timing
//!
//! Unchecked means an out-of-bounds access is undefined behaviour instead of a clamped read. If any
//! kernel here is *relying* on clamping — reading past the end of a row and getting a harmless
//! in-bounds value — then removing the checks changes the answer. So each child prints a checksum of
//! its logits, and a mismatch is reported as a **correctness failure, not a speed result**. A faster
//! wrong answer is not a win, and this is the only way to tell the two apart.
//!
//!   cargo run -p ferric-llama --example bounds_check_tax --release
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

const MODEL: &str = "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf";
const PREFILL_TOKENS: usize = 256;
const DECODE_TOKENS: usize = 24;
const REPS: usize = 5;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() {
    if std::env::var("FERRIC_BENCH_CHILD").is_ok() {
        pollster::block_on(child());
    } else {
        parent();
    }
}

/// One measurement run, in its own process so the shader-compile flag is fixed for its whole life.
async fn child() {
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/{MODEL}");
    let g = GgufFile::open(&path).expect("model");
    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).expect("load");
    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    let prompt: Vec<u32> = (0..PREFILL_TOKENS).map(|i| ((i * 7919) % 30000 + 100) as u32).collect();

    // Warm: compile every pipeline and fault the weights in before any clock starts.
    {
        let mut c = Cache::new(&m.cfg);
        let _ = m.forward_cached(&prompt, &mut c).to_vec().await;
    }

    let mut prefill = Vec::with_capacity(REPS);
    let mut decode = Vec::with_capacity(REPS);
    let mut checksum = 0f64;
    for _ in 0..REPS {
        let mut c = Cache::new(&m.cfg);
        let t0 = std::time::Instant::now();
        let l = m.forward_cached(&prompt, &mut c).to_vec().await;
        prefill.push(t0.elapsed().as_secs_f64() * 1000.0);

        // Sum of the final logit row: sensitive to any value change anywhere in the forward pass,
        // and stable across runs because decode here is greedy.
        checksum = l[l.len() - vn..].iter().map(|&x| x as f64).sum();
        let mut next = am(&l);

        let t0 = std::time::Instant::now();
        for _ in 0..DECODE_TOKENS {
            let l = m.forward_cached(&[next], &mut c).to_vec().await;
            next = am(&l);
        }
        decode.push(t0.elapsed().as_secs_f64() * 1000.0 / DECODE_TOKENS as f64);
    }
    prefill.sort_by(|a, b| a.partial_cmp(b).unwrap());
    decode.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Machine-readable, because the parent parses it.
    println!("RESULT {} {} {}", prefill[REPS / 2], decode[REPS / 2], checksum);
}

/// Run the child twice, identical except for one environment variable.
fn parent() {
    println!("The bounds-check tax — same binary, two processes, one variable\n");
    print!("{}", ferric_joule::capability_report());
    if let Some(l) = load_avg() {
        println!("  machine load average: {l:.2}");
        assert!(l < 5.0, "load {l:.2} is too high to time anything. Wait and re-run.");
    }

    let exe = std::env::current_exe().expect("current exe");
    let run = |unchecked: bool| -> (f64, f64, f64) {
        let mut cmd = std::process::Command::new(&exe);
        cmd.env("FERRIC_BENCH_CHILD", "1");
        if unchecked { cmd.env("FERRIC_UNCHECKED_SHADERS", "1"); }
        else { cmd.env_remove("FERRIC_UNCHECKED_SHADERS"); }
        let out = cmd.output().expect("child failed to run");
        let s = String::from_utf8_lossy(&out.stdout);
        let line = s.lines().find(|l| l.starts_with("RESULT"))
            .unwrap_or_else(|| panic!("child produced no RESULT line.\nstdout:\n{s}\nstderr:\n{}",
                                      String::from_utf8_lossy(&out.stderr)));
        let f: Vec<f64> = line.split_whitespace().skip(1).map(|x| x.parse().unwrap()).collect();
        (f[0], f[1], f[2])
    };

    println!("\n  running checked (the shipping default) ...");
    let (p_on, d_on, sum_on) = run(false);
    println!("  running unchecked ...");
    let (p_off, d_off, sum_off) = run(true);

    // ---- correctness first: a faster wrong answer is not a result ----
    let rel = if sum_on != 0.0 { (sum_on - sum_off).abs() / sum_on.abs() } else { (sum_on - sum_off).abs() };
    println!("\n  logit checksum  checked {sum_on:.6e}   unchecked {sum_off:.6e}   rel {rel:.3e}");
    assert!(rel < 1e-9,
        "removing bounds checks CHANGED THE ANSWER (rel {rel:.3e}). Some kernel is relying on the clamp \
         to keep an out-of-range index harmless, which is a latent bug the checks were hiding. Fix the \
         indexing before reading any timing below as a speedup.");
    println!("  → identical output, so the timings below compare the same computation.");

    println!("\n  {:>12} {:>12} {:>12} {:>10}", "", "checked", "unchecked", "saved");
    println!("  {:-<50}", "");
    println!("  {:>12} {p_on:>9.2} ms {p_off:>9.2} ms {:>9.1}%", "prefill 256", 100.0 * (p_on - p_off) / p_on);
    println!("  {:>12} {d_on:>9.2} ms {d_off:>9.2} ms {:>9.1}%", "decode/token", 100.0 * (d_on - d_off) / d_on);

    println!("\n  Reference: LlamaWeb reported 14% decode / 23% prefill, up to 42% on some devices.");
    let (dp, pp) = (100.0 * (d_on - d_off) / d_on, 100.0 * (p_on - p_off) / p_on);
    if dp < 3.0 && pp < 3.0 {
        println!("  Measured here: {pp:.1}% prefill, {dp:.1}% decode — NOT reproduced on this device.");
        println!("  Ferric's decode is weight-streaming bound (~525 MB read per token), and a bounds");
        println!("  check costs instructions, not bandwidth. A tax on the term that is not the");
        println!("  bottleneck does not show up in wall clock. Do not ship an unsafe flag for this.");
    } else {
        println!("  Measured here: {pp:.1}% prefill, {dp:.1}% decode.");
        println!("  Real, but it buys undefined behaviour on any out-of-range index. Ship only behind an");
        println!("  explicit flag, native-only, and never as a default — see Context::shader_module.");
    }
    println!("\n  ⚠ TIME, not joules: no readable power counter on this machine. Fewer instructions for");
    println!("  the same memory traffic is a plausible but UNMEASURED energy win here.");
}
