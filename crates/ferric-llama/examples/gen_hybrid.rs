//! **Greedy text generation on the hybrid runtime** — the correctness signal for any `qwen35`-family
//! checkpoint, including the plain-GQA MoE ones (`qwen3moe`).
//!
//! `run_bonsai` takes token IDS and `moe_slots` reports one argmax; neither answers "does this model
//! produce sensible text", which is the only question a near-miss architecture fails. A wrong
//! head_dim, a misrouted expert or a dropped shared expert all yield fluent nonsense rather than an
//! error, so the text itself is the test.
//!
//!   cargo run -p ferric-llama --example gen_hybrid --release -- <model.gguf> "prompt" [n]
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen35::{Cache, Qwen35};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: gen_hybrid <model.gguf> [prompt] [n]");
    let prompt = a.get(2).map(|s| s.as_str()).unwrap_or("The capital of France is");
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let g = GgufFile::open(path).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer"),
    };
    let vocab: std::collections::HashMap<String, u32> =
        toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(p, q)| (p.to_string(), q.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);

    let m = Qwen35::load(&ctx, &g).expect("load");
    println!("{} blocks · d={} · {} experts top-{} · head_dim={}",
             m.cfg.n_layer, m.cfg.n_embd, m.cfg.n_expert, m.cfg.n_expert_used, m.cfg.head_dim);

    let mut ids: Vec<u32> = bpe.encode(prompt);
    // Same policy as ferric-serve: an ABSENT add_bos_token with a declared bos_token_id means ADD.
    // Treating absence as denial is what made LFM2.5-8B-A1B answer "France is France is France is".
    if let Some(Meta::U(bos)) = g.metadata.get("tokenizer.ggml.bos_token_id") {
        if !matches!(g.metadata.get("tokenizer.ggml.add_bos_token"), Some(Meta::Bool(false))) {
            ids.insert(0, *bos as u32);
        }
    }
    let n_prompt = ids.len();
    let mut cache = Cache::new(&m.cfg);
    let vn = m.cfg.n_vocab;
    let mut fed = ids.clone();
    for _ in 0..n_gen {
        let out = m.forward_cached(&fed, &mut cache, m.layers.len()).to_vec().await;
        let last = &out[out.len() - vn..];
        // Non-finite logits mean the forward broke outright; say so rather than emitting whatever
        // argmax picks out of NaNs.
        let bad = last.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite logits of {vn}");
        let next = last.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as u32;
        ids.push(next);
        fed = vec![next];
    }
    let text: String = ids[n_prompt..].iter()
        .map(|&i| toks.get(i as usize).cloned().unwrap_or_default().replace('Ġ', " ").replace('Ċ', "\n"))
        .collect();
    println!("\n{prompt}{text}");
}
