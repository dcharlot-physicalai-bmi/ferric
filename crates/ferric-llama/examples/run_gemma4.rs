//! **Gemma 4 on Ferric** — the correctness run.
//!
//! Takes token ids rather than text on purpose. Gemma 4 ships a new tokenizer (`tokenizer.ggml.model
//! = "gemma4"`: BPE merges with SentencePiece-style whitespace escaping, pre-type `GEMMA4`), and
//! wiring that up is a separable job from proving the *model* is right. Feeding ids isolates the
//! forward pass, so a mismatch here is the architecture and nothing else.
//!
//! Get ids from the reference:
//!
//! ```text
//!   llama-tokenize -m gemma-4-E2B-it-Q8_0.gguf -p "The capital of France is" --ids
//!   → [2, 818, 5279, 529, 7001, 563]
//!
//!   cargo run -p ferric-llama --example run_gemma4 --release -- <model.gguf> 2,818,5279,529,7001,563 [n]
//! ```
//!
//! What a wrong answer looks like: fluent text about something else. Every one of Gemma 4's six
//! departures from Gemma 3 (per-layer embeddings, shared KV, two head widths, the weightless V norm,
//! GELU, no attention scale) degrades output without raising anything, so "it generated words" is not
//! evidence. Compare against `llama-cli --temp 0` on the same ids.
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::gemma4::{Cache, Gemma4};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: run_gemma4 <model.gguf> <comma,separated,ids> [n_gen]");
    let ids: Vec<u32> = a.get(2).expect("token ids")
        .split(',').map(|s| s.trim().parse().expect("id")).collect();
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(path).expect("open");

    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    // Go through the registry rather than assuming: this is the check that would have caught a
    // gemma3 file being fed to the gemma4 loader and vice versa.
    let entry = ferric_llama::arch::resolve(&arch).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(arch, "gemma4", "this loader is for gemma4, got {arch:?} ({})", entry.runtime.label());

    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let eos: Vec<u32> = ["eos_token_id", "eot_token_id"].iter()
        .filter_map(|k| match g.metadata.get(&format!("tokenizer.ggml.{k}")) { Some(Meta::U(v)) => Some(*v as u32), _ => None })
        .collect();

    let t0 = std::time::Instant::now();
    let m = Gemma4::load(&ctx, &g).expect("load gemma4");
    let c = &m.cfg;
    println!("gemma4 · {} blocks, d={}, {} heads x {}/{} (global/swa), {} kv",
             c.n_layer, c.d, c.n_head, c.head_dim, c.head_dim_swa, c.n_kv);
    println!("  KV-owning blocks 0..{} ({} shared); window {}; ple {}; softcap {}",
             c.kv_from_start, c.n_layer - c.kv_from_start, c.window, c.ple, c.final_softcap);
    println!("  sliding blocks read {}, global blocks read {}", c.kv_src(c.n_layer - 1).min(c.kv_from_start - 2), c.kv_from_start - 1);
    println!("  loaded in {:.2?}", t0.elapsed());

    let vn = c.n_vocab;
    let argmax = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;

    let t0 = std::time::Instant::now();
    let mut cache = Cache::new(c);
    let logits = m.forward(&ids, &mut cache).to_vec().await;
    let last = &logits[logits.len() - vn..];
    let bad = last.iter().filter(|v| !v.is_finite()).count();
    let (mn, mx) = last.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("  prefill {} ids in {:.2?} · {bad} non-finite, min {mn:.3} max {mx:.3}", ids.len(), t0.elapsed());
    assert_eq!(bad, 0, "non-finite logits out of prefill");
    // Softcapping at 30 bounds the logits by construction. Blowing past it means the cap was not
    // applied, or was applied to the wrong thing.
    if c.final_softcap > 0.0 {
        assert!(mx <= c.final_softcap * 1.001 && mn >= -c.final_softcap * 1.001,
                "logits [{mn}, {mx}] exceed the softcap {}", c.final_softcap);
    }

    let mut next = argmax(&logits);
    let mut out = Vec::new();
    let t0 = std::time::Instant::now();
    for _ in 0..n_gen {
        if eos.contains(&next) { break; }
        out.push(next);
        next = argmax(&m.forward(&[next], &mut cache).to_vec().await);
    }
    let dt = t0.elapsed();

    // Gemma 4 escapes whitespace SentencePiece-style, so ▁ maps back to a space.
    let text: String = out.iter().map(|&t| tokens[t as usize].replace('\u{2581}', " ")).collect();
    println!("\n--- Ferric ---\n{text}\n");
    println!("{} tokens in {:.2?} ({:.1} ms/tok)", out.len(), dt, dt.as_secs_f64() * 1000.0 / out.len().max(1) as f64);
    println!("ids: {out:?}");
    println!("\nCompare against: llama-cli -m <model> --temp 0 on the same prompt.");
    println!("Fluent text about the wrong subject means one of the six Gemma-4 departures is wrong.");
}
