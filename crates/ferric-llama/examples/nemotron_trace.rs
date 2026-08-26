//! **Diff Nemotron-H's forward against llama.cpp, per op.**
//!   llama-eval-callback -m M -p "Paris" -ngl 0 > ref.log
//!   cargo run -p ferric-llama --example nemotron_trace --release -- M "Paris"
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::nemotron_h::NemotronH;
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: nemotron_trace <model.gguf> <text>");
    let text = a.get(2).map(String::as_str).unwrap_or("Paris");
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");

    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|m| if let Meta::Str(s) = m {
            s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    // Route on tokenizer.ggml.pre — this file says "pixtral", a THIRD variant after gpt2 and qwen2.
    // Hardcoding a family is what cost the BERT port an afternoon.
    let pre = match g.metadata.get("tokenizer.ggml.pre") { Some(Meta::Str(p)) => Some(p.as_str()), _ => None };
    println!("tokenizer.ggml.pre = {pre:?}");
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let bpe = Bpe::new_with_pre(vocab, &merges, Pre::from_gguf(pre));
    let mut ids = bpe.encode(text);
    if matches!(g.metadata.get("tokenizer.ggml.add_bos_token"), Some(Meta::Bool(true))) {
        if let Some(Meta::U(b)) = g.metadata.get("tokenizer.ggml.bos_token_id") { ids.insert(0, *b as u32); }
    }
    println!("  {} tokens: {ids:?}", ids.len());

    let m = NemotronH::load(&ctx, &g).expect("load");
    let (ssm, ffn, attn) = m.cfg.schedule();
    println!("  {ssm} SSM · {ffn} MLP · {attn} attention");
    let (logits, tr) = m.forward_traced(&ids).expect("forward");
    for (name, t) in &tr {
        let v = t.to_vec().await;
        let sum: f32 = v.iter().sum();
        let mx = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        eprintln!("TRACE {name:<22} sum {sum:+.6}  max|v| {mx:.6}");
    }
    let lv = logits.to_vec().await;
    let n = m.cfg.n_vocab;
    assert!(lv.iter().all(|x| x.is_finite()), "logits contain NaN/Inf — the scan or a norm diverged");

    // Greedy generation. Per-block SUMS can cancel — a tensor mostly right can post a sum 30% off,
    // and a tensor badly wrong can post one that matches — so the end-to-end check is whether the
    // text is coherent and matches what the reference produces from the same prompt.
    //
    // No incremental state here: every step re-runs the whole prefix. That is O(T^2) and exactly what
    // an SSM exists to avoid, but carrying conv and scan state across steps is a separate correctness
    // problem, and conflating the two is how a state bug gets attributed to the mixer.
    let steps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(12);
    let mut seq = ids.clone();
    let mut out = String::new();
    for _ in 0..steps {
        let lg = m.forward(&seq).expect("forward").to_vec().await;
        let last = &lg[lg.len() - n..];
        let (best, _) = last.iter().enumerate()
            .fold((0usize, f32::MIN), |acc, (i, &x)| if x > acc.1 { (i, x) } else { acc });
        out.push_str(toks.get(best).map(String::as_str).unwrap_or("?"));
        seq.push(best as u32);
        if matches!(g.metadata.get("tokenizer.ggml.eos_token_id"), Some(Meta::U(e)) if *e as usize == best) { break; }
    }
    println!("  generated: {:?}", out.replace('\u{2581}', " ").replace('Ġ', " "));
}
