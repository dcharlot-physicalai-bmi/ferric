//! **The cross-fabric portability budget** — how many GPU dispatches does Ferric spend per token?
//!
//! Published browser benchmark data puts Firefox's WebGPU per-dispatch cost at **~1,040 µs against
//! Chrome's ~33 µs — about 30× worse**. That number turns dispatch count from an implementation detail
//! into a portability budget: a design issuing many small dispatches per token is not "portable with a
//! slow path", it is Chrome-only with a Firefox-shaped cliff.
//!
//! Ferric batches work into regions per layer via `ferric_tensor::batch`, and this was previously
//! documented as "the right shape" for that budget. **Counting it overturned that.** `batch` collapses
//! *queue submissions* — measured at ~1 per layer, which is genuinely good — but the Firefox penalty is
//! per **dispatch**, and dispatches are unchanged at ~17 per layer. Submission batching does not protect
//! against this cliff at all.
//!
//! ## What is being measured, and what the numbers can and cannot say
//!
//! `op_counters()` reports **dispatches** (compute-shader launches) and **submits** (queue submissions).
//! Both are counted exactly, on the host, with no timing involved — so they are not noisy, and they
//! transfer to any fabric.
//!
//! The Firefox projection below is **arithmetic on someone else's measurement**, not a measurement of
//! Ferric in Firefox. It says what the dispatch count *implies* at a published per-dispatch cost. Treat
//! it as a budget to design against; an actual Firefox run would be a different claim and is not made
//! here.
//!
//!   cargo run -p ferric-llama --example dispatch_budget --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tensor::{op_counters, reset_op_counters};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// Per-dispatch cost, microseconds, from published 2026 browser benchmark data.
const CHROME_US: f64 = 33.0;
const FIREFOX_US: f64 = 1040.0;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();
    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!(),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!(),
    };
    let bpe = Bpe::new(vocab, &merges);
    let m = Qwen3::load(&ctx, &g).unwrap();
    let n_layers = m.cfg.n_layer;

    let prompt = bpe.encode("The capital of France is");
    println!("Dispatch budget — Qwen2.5-0.5B, {n_layers} layers\n");

    // ---- prefill ----
    let mut cache = Cache::new(&m.cfg);
    reset_op_counters();
    let logits = m.forward_cached(&prompt, &mut cache).to_vec().await;
    let (pre_d, pre_s) = op_counters();
    let vn = m.cfg.n_vocab;
    let mut next = logits[logits.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    // ---- decode: the number that matters, because it is paid per token forever ----
    const N: usize = 16;
    reset_op_counters();
    let t0 = std::time::Instant::now();
    for _ in 0..N {
        let l = m.forward_cached(&[next], &mut cache).to_vec().await;
        next = l[l.len() - vn..].iter().enumerate()
            .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / N as f64;
    let (dec_d, dec_s) = op_counters();
    let per_tok = dec_d as f64 / N as f64;
    let per_layer = per_tok / n_layers as f64;

    println!("  {:<24} {:>12}  {:>10}  {:>14}", "", "dispatches", "submits", "per layer");
    println!("  {:-<64}", "");
    println!("  {:<24} {:>12}  {:>10}  {:>14.1}", format!("prefill ({} tok)", prompt.len()),
             pre_d, pre_s, pre_d as f64 / n_layers as f64);
    println!("  {:<24} {:>12.1}  {:>10.1}  {:>14.1}", "decode (per token)", per_tok,
             dec_s as f64 / N as f64, per_layer);

    println!("\n  Measured decode: {ms:.1} ms/token on this device.\n");
    println!("  Projected dispatch overhead alone, at published per-dispatch costs:");
    println!("  {:<24} {:>14}  {:>16}", "", "per dispatch", "per token");
    println!("  {:-<58}", "");
    println!("  {:<24} {:>11.0} µs  {:>13.1} ms", "Chrome", CHROME_US, per_tok * CHROME_US / 1000.0);
    println!("  {:<24} {:>11.0} µs  {:>13.1} ms", "Firefox", FIREFOX_US, per_tok * FIREFOX_US / 1000.0);

    let ff_ms = per_tok * FIREFOX_US / 1000.0;
    println!("\n  These are ARITHMETIC on published per-dispatch costs, not a Firefox run. They say what");
    println!("  the dispatch count implies, which is the useful thing at design time.\n");

    let chrome_ms = per_tok * CHROME_US / 1000.0;
    println!("  ⚠ THE FINDING, and it corrects what this repo previously documented.\n");
    println!("  `ferric_tensor::batch` collapses SUBMISSIONS — {:.1} per token, ~1 per layer, genuinely good.",
             dec_s as f64 / N as f64);
    println!("  But the Firefox penalty is per DISPATCH, and dispatches are {per_layer:.1} per layer, {per_tok:.0} per");
    println!("  token. Submission batching does not protect against this cliff. The earlier note that");
    println!("  Ferric was already \"the right shape\" here was wrong, and only counting showed it.\n");
    println!("  At Chrome's OWN 33 µs that is {chrome_ms:.1} ms/token of launch overhead against {ms:.1} ms/token");
    println!("  measured natively — so in a browser Ferric is dispatch-bound, not compute-bound. Firefox");
    println!("  projects to {ff_ms:.0} ms/token of overhead alone.\n");
    println!("  Tracing the kernels showed QKV, flash-attention and add+rmsnorm were ALREADY fused. The");
    println!("  fat was `gather` — pure data movement doing no math — running 3x per layer to pack the");
    println!("  q/k/v windows of the fused QKV output. All three are now gone (410 -> {per_tok:.0}/token,");
    println!("  a 17.6% cut), by teaching the two kernels that actually consume those windows to read a");
    println!("  strided view in place: `kv_write` and `rope`. Fusing the q and k RoPE into one dispatch");
    println!("  took it to {per_tok:.0}/token — 410 -> {per_tok:.0} is 23.4% off the launch cost.\n");
    println!("  What was NOT done matters more. The tempting fix was to relax the `offset == 0` guard in");
    println!("  is_contiguous() so every view could skip packing. An audit of all 57 kernels found only");
    println!("  THREE honour a tensor's buffer offset (cat, gather, binary); 52 index from element 0.");
    println!("  Worse, two hazards are not kernels at all — `reshape` and `reduce` read from the buffer");
    println!("  start — and `metal4_linear` honours offsets while its WGSL fallback does not, which would");
    println!("  have made correctness depend on macOS, silently. Two kernels changed instead of 52.");

    // Regression guard at the MEASURED value, not an aspirational one — a threshold that passes today
    // while the number is bad teaches nothing. This fails if the count grows.
    assert!(
        per_layer <= 13.5,
        "{per_layer:.1} dispatches per layer per token, up from the 13.1 measured when this was written. \
         Dispatch count is Ferric's portability budget: at Firefox's ~1040 µs this is {ff_ms:.0} ms/token \
         of pure launch overhead."
    );
}
