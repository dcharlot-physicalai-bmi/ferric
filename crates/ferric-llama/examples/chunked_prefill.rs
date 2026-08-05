//! **Chunked prefill** — process a long prompt in bounded pieces instead of one quadratic pass.
//!
//! A single-pass prefill materialises a `[T, T]` attention score matrix per head — quadratic in prompt
//! length, which is why kimi-k3-in-c lists chunked prefill as its own highest-value gap.
//!
//! ## What the investigation actually found
//!
//! Ferric's one-pass prefill was **rejected outright above ~862 tokens**, and the cause was not the
//! quadratic attention at all: it was **`swiglu`**, an *elementwise* kernel dispatching `t · n_ff / 64`
//! workgroups. At `n_ff = 4864` that crosses WebGPU's 65,535-per-dimension limit **linearly in prompt
//! length**, long before the quadratic term matters.
//!
//! That is now fixed at the source — `swiglu` folds into a 2-D grid, the same convention several other
//! kernels already used — so single-pass prefill works to 4096 tokens and beyond. Chunking is back to
//! being a *memory* optimisation rather than the only way to run a long prompt.
//!
//! Chunking makes the peak `[C, T]` instead of `[T, T]` for a chunk size `C`, which is flat in the chunk
//! rather than quadratic in the prompt. The cost is that later chunks attend over a longer history —
//! total work is the same, but the *peak* is bounded and controllable.
//!
//! Two things must hold, and as always the second is the one that is easy to lose:
//!
//!   1. peak scratch must actually fall with the chunk size;
//!   2. the logits must be **identical** — chunking changes only how the same attention is scheduled, so
//!      any difference means the mask offset is wrong and the model attended to the wrong span.
//!
//!   cargo run -p ferric-llama --example chunked_prefill --release
use ferric_core::{max_abs_diff, Context};
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// Prefill `tokens` in pieces of at most `chunk`, returning the final logits.
///
/// This is the whole of chunked prefill: feed a slice, let the cache carry the history, repeat. It works
/// because `nn::chunked_attention` handles queries that are the tail of a longer history — without that,
/// every chunk after the first would assert.
async fn prefill_chunked(m: &Qwen3, tokens: &[u32], chunk: usize, cache: &mut Cache) -> Vec<f32> {
    let mut last = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let end = (i + chunk).min(tokens.len());
        last = m.forward_cached(&tokens[i..end], cache).to_vec().await;
        i = end;
    }
    last
}

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

    let corpus = std::fs::read_to_string(".phase0/corpus_real.txt").expect(".phase0/corpus_real.txt");
    let all = bpe.encode(&corpus);
    let nh = m.cfg.n_head;
    let vn = m.cfg.n_vocab;
    let arg = |v: &[f32]| v[v.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;

    println!("Chunked prefill — Qwen2.5-0.5B, {nh} heads\n");

    // ---- 1. the ceiling that WAS here, and what it really was ----
    println!("  One-pass prefill used to be REJECTED above ~862 tokens. The cause was not the quadratic");
    println!("  attention — it was `swiglu`, an ELEMENTWISE kernel dispatching t*n_ff/64 workgroups,");
    println!("  which crosses the 65,535-per-dimension limit LINEARLY in prompt length. Folded into a");
    println!("  2-D grid, so the ceiling is gone; chunking is a memory optimisation again, not a");
    println!("  requirement.\n");

    // ---- 2. identical results where one pass still works ----
    let base: Vec<u32> = all[..768].to_vec();
    let mut c0 = Cache::new(&m.cfg);
    let t0 = std::time::Instant::now();
    let reference = m.forward_cached(&base, &mut c0).to_vec().await;
    let ref_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ref_arg = arg(&reference);

    println!("  768-token prompt (the largest one pass can do here):");
    println!("  {:>10}  {:>10}  {:>13}   {}", "chunk", "wall", "max|Δ|", "argmax");
    println!("  {:-<52}", "");
    println!("  {:>10}  {:>7.0} ms  {:>13}   {ref_arg}", "one pass", ref_ms, "—");
    for chunk in [512usize, 256, 128] {
        let mut c = Cache::new(&m.cfg);
        let t1 = std::time::Instant::now();
        let logits = prefill_chunked(&m, &base, chunk, &mut c).await;
        let ms = t1.elapsed().as_secs_f64() * 1000.0;
        let d = max_abs_diff(&reference[reference.len() - vn..], &logits[logits.len() - vn..]);
        println!("  {:>10}  {:>7.0} ms  {:>13.3e}   {}", chunk, ms, d, arg(&logits));
        assert_eq!(arg(&logits), ref_arg, "chunk {chunk} changed the predicted token");
        assert!(d < 1e-3, "chunk {chunk} perturbed the logits by {d:.3e}");
        assert_eq!(c.pos, base.len(), "chunk {chunk} left the cache at the wrong position");
    }

    // ---- 3. long prompts: one pass now works, and chunking still bounds the peak ----
    println!("\n  Long prompts — one pass now RUNS, and chunking still bounds peak score memory:");
    println!("  {:>8}  {:>11}  {:>11}  {:>16}  {:>10}", "tokens", "one pass", "chunk 256", "peak scores", "max|Δ|");
    println!("  {:-<64}", "");
    for n in [1024usize, 2048, 4096] {
        let p: Vec<u32> = all[..n].to_vec();
        let mut c1 = Cache::new(&m.cfg);
        let ta = std::time::Instant::now();
        let one = m.forward_cached(&p, &mut c1).to_vec().await;
        let one_ms = ta.elapsed().as_secs_f64() * 1000.0;

        let mut c2 = Cache::new(&m.cfg);
        let tb = std::time::Instant::now();
        let ch = prefill_chunked(&m, &p, 256, &mut c2).await;
        let ch_ms = tb.elapsed().as_secs_f64() * 1000.0;

        let d = max_abs_diff(&one[one.len() - vn..], &ch[ch.len() - vn..]);
        assert_eq!(arg(&one), arg(&ch), "chunking changed the prediction at {n} tokens");
        assert_eq!(c2.pos, n, "chunked prefill of {n} left the cache at the wrong position");
        let one_peak = (nh * n * n * 4) as f64 / 1e6;
        let ch_peak = (nh * 256 * n * 4) as f64 / 1e6;
        println!("  {:>8}  {:>8.0} ms  {:>8.0} ms  {:>6.0} -> {:>4.0} MB  {:>10.3e}",
                 n, one_ms, ch_ms, one_peak, ch_peak, d);
    }

    println!("\n  ✅ Chunked prefill gives IDENTICAL predictions at every chunk size and every length,");
    println!("     while cutting peak score memory ~16x at 4096 tokens. It changes only how the same");
    println!("     attention is scheduled, so any difference would mean the mask offset is wrong and the");
    println!("     model attended to the wrong span — which is why argmax is asserted before any timing.");
    println!("\n  The investigation was worth more than the feature: chasing the failure found that the");
    println!("  real ceiling was a LINEAR elementwise kernel, not the quadratic attention everyone");
    println!("  assumes. `run()` now asserts the per-dimension limit with the kernel name, so the next");
    println!("  one is a diagnosis instead of an opaque driver rejection.");
}
