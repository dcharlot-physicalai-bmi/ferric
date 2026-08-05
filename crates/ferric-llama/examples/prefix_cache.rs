//! **Prefix caching: skip prefill for what has already been computed** — and prove it changes nothing.
//!
//! The agent/chat workload SGLang built RadixAttention for: many turns behind one long system prompt.
//! Without a prefix cache each turn re-prefills the whole prompt; with one, the first turn pays and the
//! rest copy its KV.
//!
//! Two things must both hold, and the second is the one that is easy to get wrong:
//!
//!   1. it must be **faster** — otherwise it is complexity for nothing;
//!   2. it must produce **identical logits** — a prefix cache that reuses KV computed from *different*
//!      preceding tokens still emits fluent text, so "it looks fine" proves nothing.
//!
//!   cargo run -p ferric-llama --example prefix_cache --release
use ferric_core::{max_abs_diff, Context};
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::prefix::PrefixCache;
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

const TURNS: usize = 8;

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

    // A long shared system prompt, then a different user turn each time — the agent shape.
    let corpus = std::fs::read_to_string(".phase0/corpus_real.txt").expect(".phase0/corpus_real.txt");
    let all = bpe.encode(&corpus);
    let system: Vec<u32> = all[..512].to_vec();
    let turns: Vec<Vec<u32>> = (0..TURNS)
        .map(|i| {
            let mut t = system.clone();
            t.extend_from_slice(&all[2000 + i * 64..2000 + i * 64 + 24]);
            t
        })
        .collect();

    println!("Prefix caching — Qwen2.5-0.5B, {} turns behind a {}-token shared prompt\n", TURNS, system.len());

    // ---- baseline: every turn prefills everything ----
    let mut base_logits = Vec::new();
    let t0 = std::time::Instant::now();
    for t in &turns {
        let mut c = Cache::new(&m.cfg);
        base_logits.push(m.forward_cached(t, &mut c).to_vec().await);
    }
    let base_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ---- with a prefix cache ----
    let mut pc = PrefixCache::new(8);
    let mut cached_logits = Vec::new();
    let t1 = std::time::Instant::now();
    for t in &turns {
        let (mut c, done) = pc.cache_for(&ctx, &m.cfg, t);
        // Only the un-cached suffix is prefilled. `done` came from the cache, and the KV for it is
        // already installed — feeding those tokens again would double-count them.
        let logits = m.forward_cached(&t[done..], &mut c).to_vec().await;
        cached_logits.push(logits);
        pc.insert(&ctx, t, &c);
    }
    let cached_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ---- correctness first: a faster wrong answer is not a speedup ----
    let vocab_n = m.cfg.n_vocab;
    let mut worst = 0f32;
    for (i, (b, c)) in base_logits.iter().zip(&cached_logits).enumerate() {
        let (bl, cl) = (&b[b.len() - vocab_n..], &c[c.len() - vocab_n..]);
        let d = max_abs_diff(bl, cl);
        worst = worst.max(d);
        let am = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |a, (j, &x)| if x > a.1 { (j, x) } else { a }).0;
        assert_eq!(am(bl), am(cl), "turn {i}: prefix cache changed the predicted token");
    }

    println!("  {:<26} {:>10}  {:>12}  {:>10}", "", "wall", "prefill tokens", "hit rate");
    println!("  {:-<64}", "");
    let base_tok: usize = turns.iter().map(|t| t.len()).sum();
    let saved = pc.tokens_saved as usize;
    println!("  {:<26} {:>8.0} ms  {:>12}  {:>9}", "no prefix cache", base_ms, base_tok, "—");
    println!("  {:<26} {:>8.0} ms  {:>12}  {:>9.0}%", "with prefix cache", cached_ms,
             base_tok - saved, 100.0 * pc.hit_rate());
    println!("\n  {:.2}x faster · {} of {} prefill tokens skipped ({:.0}%)",
             base_ms / cached_ms, saved, base_tok, 100.0 * saved as f64 / base_tok as f64);
    println!("  max|Δ| on final logits: {worst:.3e}   (identical predictions on all {TURNS} turns)");

    assert!(saved > 0, "nothing was reused — the cache never hit");
    assert!(worst < 1e-3, "prefix cache perturbed the logits by {worst:.3e}");
    // The first turn is always a miss, so the ceiling is (TURNS-1)/TURNS of the shared prompt.
    let ceiling = (TURNS - 1) * system.len();
    assert!(saved >= ceiling * 9 / 10, "only {saved} tokens reused of a possible ~{ceiling}");

    println!("\n  ✅ Prefill skipped for the shared prompt, with IDENTICAL predictions. The saving is");
    println!("     compute, not memory: each sequence still holds its own KV, because Ferric's attention");
    println!("     reads a contiguous view. `ferric-kv`'s PagedKv is the memory half and needs a paged");
    println!("     attention kernel — a separate piece of work, and the smaller of the two wins.");
}
