//! **Is the SentencePiece path right?** Ferric's `Spm` vs `llama-tokenize`, on a real checkpoint.
//!
//! ⛔ WHY: `encode_piece` appears exactly ONCE in ferric-tokenizer — its own definition. The Spm path
//! has no test and has never been compared to anything, while it serves Gemma 2/3/4, Phi-3.5,
//! TinyLlama and the bge reranker. The BPE side at least had unit tests; this had nothing, and the
//! text->ids conformance gate skips it by design.
//!
//!   cargo run -p ferric-llama --example spm_conformance --release -- <model.gguf> <ref.txt>
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::Spm;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: <model.gguf> <ref.txt>");
    let refp = a.next().expect("usage: <model.gguf> <ref.txt>");
    let g = GgufFile::open(&path).expect("open gguf");
    let get = |k: &str| g.metadata.get(k);
    let model = match get("tokenizer.ggml.model") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };
    let tokens: Vec<String> = match get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(x)) => x.iter().map(|v| if let Meta::Str(s) = v { s.clone() } else { String::new() }).collect(),
        _ => { println!("no tokens"); return; }
    };
    let scores: Vec<f32> = match get("tokenizer.ggml.scores") {
        Some(Meta::Arr(x)) => x.iter().map(|v| if let Meta::F(f) = v { *f as f32 } else { 0.0 }).collect(),
        _ => Vec::new(),
    };
    if scores.is_empty() { println!("model={model}: no scores — not an Spm checkpoint"); return; }
    // ⚠ the same default the server applies; getting this wrong shifts EVERY id by one leading piece
    let add_space_prefix = match get("tokenizer.ggml.add_space_prefix") { Some(Meta::Bool(b)) => *b, _ => true };
    println!("{}  model={model}  {} tokens  add_space_prefix={add_space_prefix}",
             path.rsplit('/').next().unwrap_or(&path), tokens.len());

    // ⛔ types matter: llama.cpp matches USER_DEFINED verbatim BEFORE the merge, and Gemma uses 140
    // of them for whitespace runs. Without the type array those tokens are unreachable.
    let types: Vec<i32> = match get("tokenizer.ggml.token_type") {
        Some(Meta::Arr(x)) => x.iter().map(|v| match v {
            Meta::I(n) => *n as i32, Meta::U(n) => *n as i32, _ => 1 }).collect(),
        _ => Vec::new(),
    };
    let spm = if types.is_empty() { Spm::new(tokens, scores) } else { Spm::with_types(tokens, scores, &types) };
    println!("  USER_DEFINED entries matched verbatim: {}", spm.user_defined_count());
    let txt = std::fs::read_to_string(&refp).expect("ref file");
    let (mut ok, mut n) = (0usize, 0usize);
    let mut misses: Vec<String> = Vec::new();
    for line in txt.lines() {
        let Some((s, ids)) = line.rsplit_once('|') else { continue };
        let want: Vec<u32> = ids.trim().trim_start_matches('[').trim_end_matches(']')
            .split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if want.is_empty() { continue }
        n += 1;
        let got = spm.encode_piece(s, add_space_prefix);
        if got == want { ok += 1 } else if misses.len() < 4 {
            misses.push(format!("{s:?}\n     ferric: {got:?}\n     llama : {want:?}"));
        }
    }
    println!("  {ok}/{n} exact");
    for m in &misses { println!("  {m}"); }
}
