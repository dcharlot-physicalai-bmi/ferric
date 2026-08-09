//! **Does the 12.9x split-KV kernel win move end-to-end tokens/sec?**
//!
//! `ferric-tensor/examples/split_kv.rs` measured the attention kernel alone: 1.19x at S=1024, 5.05x at
//! 4096, **12.86x at 16384**. It also said, in its own closing warning, that this does not settle the
//! question — a kernel speedup only reaches the user in proportion to attention's share of a decode
//! step, and Ferric's decode is otherwise weight-streaming bound (~525 MB read per token, which is why
//! cutting 29% of GPU dispatches moved wall time 0.00 ms).
//!
//! This measures the whole model, so the share is whatever it actually is rather than whatever the
//! arithmetic suggests.
//!
//! ## Why the answer must depend on context length
//!
//! Weight traffic per token is **constant** — every weight is read once regardless of how long the
//! cache is. KV traffic **grows linearly** with the cache. So attention's share of a decode step starts
//! negligible and rises, and the end-to-end speedup has to start near 1.00x and climb. A result that
//! showed a flat speedup across context lengths would mean something is wrong with the harness, not
//! that split-KV is uniformly good.
//!
//! That is the shape this checks for, rather than a single headline number.
//!
//! ## The A/B
//!
//! `decode_splits` reads `FERRIC_SPLITKV` on every call, so both arms run in one process against one
//! loaded model and one warmed set of pipelines. Same weights, same cache contents, same token count;
//! only the split decision differs. The KV cache is rebuilt identically for each arm so neither inherits
//! the other's cache state.
//!
//!   cargo run -p ferric-llama --example splitkv_end_to_end --release
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

const MODEL: &str = "Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf";
/// Context lengths to build before decoding. Chosen to straddle the point where attention stops being
/// a rounding error, and to include one length BELOW the split gate (1024) as a null arm.
const CONTEXTS: &[usize] = &[512, 2048, 8192, 16384];
const DECODE_TOKENS: usize = 16;
const REPS: usize = 3;
/// Prefill is chunked so a 16k prompt does not need a 16k-wide one-pass dispatch.
const CHUNK: usize = 512;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }

async fn run() {
    println!("Split-KV, end to end — does a 12.9x kernel win reach tokens/sec?\n");
    print!("{}", ferric_joule::capability_report());
    if let Some(l) = load_avg() {
        println!("  machine load average: {l:.2}");
        assert!(l < 5.0, "load {l:.2} is too high to time anything. Wait and re-run.");
    }

    let home = std::env::var("HOME").unwrap();
    let g = GgufFile::open(format!("{home}/.cache/ferric/hub/{MODEL}")).expect("model");
    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).expect("load");
    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    // Warm every pipeline on BOTH paths before any clock starts, or the first arm pays for compiling
    // the other one's shaders. Cheap and easy to forget.
    for v in ["1", "4"] {
        std::env::set_var("FERRIC_SPLITKV", v);
        let mut c = Cache::new(&m.cfg);
        let _ = m.forward_cached(&vec![100u32; CHUNK], &mut c).to_vec().await;
        let _ = m.forward_cached(&[200u32], &mut c).to_vec().await;
    }
    std::env::remove_var("FERRIC_SPLITKV");

    println!("\n  {:>8} {:>10} {:>13} {:>13} {:>10} {:>12}",
             "context", "splits", "1 workgroup", "split-KV", "speedup", "attn share");
    println!("  {:-<70}", "");

    for &s in CONTEXTS {
        let prompt: Vec<u32> = (0..s).map(|i| ((i * 7919) % 30000 + 100) as u32).collect();
        // What the heuristic would choose here (mirrors decode_splits with nh = n_head).
        let nh = m.cfg.n_head;
        let splits = if s < 1024 { 1 } else { (s / 512).min(256 / nh.max(1)).max(1) };

        let mut bench = |force: Option<&str>| {
            match force {
                Some(v) => std::env::set_var("FERRIC_SPLITKV", v),
                None => std::env::remove_var("FERRIC_SPLITKV"),
            }
            async {
                let mut best = f64::INFINITY;
                for _ in 0..REPS {
                    // Rebuild the cache each rep so both arms decode from identical state.
                    let mut c = Cache::new(&m.cfg);
                    let mut last = 0u32;
                    for ch in prompt.chunks(CHUNK) {
                        last = am(&m.forward_cached(ch, &mut c).to_vec().await);
                    }
                    let t0 = std::time::Instant::now();
                    for _ in 0..DECODE_TOKENS {
                        last = am(&m.forward_cached(&[last], &mut c).to_vec().await);
                    }
                    best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / DECODE_TOKENS as f64);
                }
                best
            }
        };

        let off = bench(Some("1")).await;
        let on = bench(None).await;
        // Attention's share of the step, derived from the two arms rather than assumed: if the only
        // difference is the attention kernel, the time it saves IS the part of the step that was
        // attention beyond what the split still costs. A lower bound on the share.
        let share = 100.0 * (off - on) / off;
        println!("  {s:>8} {splits:>10} {off:>10.2} ms {on:>10.2} ms {:>9.2}x {:>11.1}%",
                 off / on, share.max(0.0));
    }
    std::env::remove_var("FERRIC_SPLITKV");

    println!("\n  Expected shape: ~1.00x at 512 (below the gate, so both arms run the same kernel and");
    println!("  this row is a NULL CONTROL — a speedup here would mean the harness is measuring noise),");
    println!("  rising with context as KV traffic grows against constant weight traffic.\n");
    println!("  The kernel-level numbers were 1.19x / 5.05x / 12.86x at S=1024 / 4096 / 16384. Whatever");
    println!("  fraction of those survives here is the fraction a user actually receives, and the gap");
    println!("  between the two is the weight streaming that split-KV cannot touch.");
    println!("\n  ⚠ TIME, not joules. Split-KV does the same arithmetic in more workgroups, so it should");
    println!("  cost similar energy for less wall clock; that is a prediction, not a measurement, and");
    println!("  this machine has no readable power counter to settle it.");
}
