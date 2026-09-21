//! **What is token N, and what TYPE is it?** — for arguing about a tokenizer with evidence.
//!
//! Token types matter more than they look: llama.cpp matches CONTROL and **USER_DEFINED** tokens
//! verbatim BEFORE the SentencePiece merge runs. Gemma's vocabulary uses 140 USER_DEFINED entries,
//! mostly runs of literal spaces ("  ", "   ", …), which a tokenizer that maps every space to `▁`
//! can never reach.
//!
//!   cargo run -p ferric-gguf --release --example tokprobe -- <model.gguf> [id ...]
fn main() {
    let mut a = std::env::args().skip(1);
    let p = a.next().expect("usage: tokprobe <model.gguf> [id ...]");
    let ids: Vec<usize> = a.filter_map(|v| v.parse().ok()).collect();
    let g = ferric_gguf::GgufFile::open(&p).expect("open gguf");
    let (types, toks) = (g.metadata.get("tokenizer.ggml.token_type"), g.metadata.get("tokenizer.ggml.tokens"));
    let (Some(ferric_gguf::Meta::Arr(t)), Some(ferric_gguf::Meta::Arr(v))) = (types, toks) else {
        println!("no token/type arrays in this file"); return;
    };
    // llama.cpp llama_token_type: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED, 6 BYTE
    let name = |x: i64| match x { 1=>"NORMAL",2=>"UNKNOWN",3=>"CONTROL",4=>"USER_DEFINED",5=>"UNUSED",6=>"BYTE",_=>"?" };
    let ty = |i: usize| match t.get(i) {
        Some(ferric_gguf::Meta::I(x)) => *x as i64, Some(ferric_gguf::Meta::U(x)) => *x as i64, _ => -1 };
    for id in &ids {
        if let Some(ferric_gguf::Meta::Str(s)) = v.get(*id) {
            println!("  id {id:>7}  {s:?}  type {} ({})", ty(*id), name(ty(*id)));
        }
    }
    let mut counts = std::collections::BTreeMap::new();
    for i in 0..v.len() { *counts.entry(ty(i)).or_insert(0usize) += 1; }
    println!("vocab {} tokens — {}", v.len(),
             counts.iter().map(|(k, n)| format!("{} {n}", name(*k))).collect::<Vec<_>>().join(", "));
}
