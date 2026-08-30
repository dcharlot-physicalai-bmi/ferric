//! **Does compute-pass reuse help LLM decode, not just the speech encoder?**
//!
//! Speech gains 1.62x (604 -> 372 ms per encode) because a 42-layer encoder issues ~6300 dispatches,
//! each of which used to open its own `MTLComputeCommandEncoder` (~18 us). Single-token decode issues
//! far fewer dispatches per step, so the win should be smaller — but `batch_throughput`'s own notes
//! put a decode step at ~10.8 ms of which **5.8 ms is host-side command building**, and that is
//! exactly the term a shared pass reduces. Worth measuring rather than assuming either way.
//!
//! ⚠ THE ARMS ARE SEPARATE PROCESS LAUNCHES, not a single-process A/B. `FERRIC_NO_PASS_REUSE` is
//! read once into a `OnceLock` and cannot be flipped mid-run, so the caller invokes this twice.
//! That is the WEAKER design: comparing separate launches on a contended machine is what produced
//! two wrong conclusions earlier in this work (batching "a loss", one encoder "3.3x slower"). Read
//! the two numbers as indicative, and re-run both arms back to back on a quiet machine before
//! quoting a ratio.
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: decode_pass_ab <model.gguf> [steps]");
    let steps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let g = GgufFile::open(path).expect("open gguf");
    let m = Qwen3::load(&ctx, &g).expect("load");
    println!("model: {} layers, vocab {}", m.cfg.n_layer, m.cfg.n_vocab);

    let prompt: Vec<u32> = vec![1, 450, 7483, 310, 3444, 338];      // arbitrary, fixed
    let vn = m.cfg.n_vocab;
    let argmax = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    // One timed decode run: prefill, then `steps` single-token steps. Returns (ms/token, last token).
    let run_once = |label: &str| {
        let mut cache = Cache::new(&m.cfg);
        let mut logits = pollster::block_on(m.forward_cached(&prompt, &mut cache).to_vec());
        let mut tok = argmax(&logits);
        let mut out = Vec::new();
        let t0 = Instant::now();
        for _ in 0..steps {
            logits = pollster::block_on(m.forward_cached(&[tok], &mut cache).to_vec());
            tok = argmax(&logits);
            out.push(tok);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        println!("  {label:<26} {ms:6.2} ms/token   {:6.1} tok/s", 1000.0 / ms);
        out
    };

    // ⚠ The toggle is read once into a OnceLock, so it cannot be flipped mid-process. Each arm is a
    // separate invocation; this binary reports whichever the environment selected, and the caller
    // runs it twice. Stated here because a single-process A/B would be the better design and is NOT
    // what this does.
    let on = std::env::var("FERRIC_NO_PASS_REUSE").is_err();
    let toks = run_once(if on { "pass reuse ON" } else { "pass reuse OFF" });
    println!("  first 8 tokens: {:?}", &toks[..toks.len().min(8)]);
    println!("  (token ids must match between arms — a perf change that alters output is a bug)");
}
