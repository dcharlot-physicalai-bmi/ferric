//! **Compare every `Pre` rule on a prompt FILE**, so newline behaviour is testable.
//!
//! ⛔ WHY A FILE. `llama-tokenize -p` takes a single line, so a corpus built with it is NEWLINE-FREE —
//! and a newline-free corpus cannot see the `laguna` rule at all, whose entire delta from Qwen2 is
//! that a pre-token may never span a '\n'. That blind spot is exactly why Qwen2 scored 20/20 on a
//! laguna checkpoint and looked correct.
//!
//!   cargo run -p ferric-llama --example pretokenizer_file --release -- <model.gguf> <prompt-file> [expected-ids]
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: <model.gguf> <prompt-file> [expected]");
    let pf = a.next().expect("usage: <model.gguf> <prompt-file> [expected]");
    let want: Option<Vec<u32>> = a.next().map(|s| s.trim().trim_start_matches('[').trim_end_matches(']')
        .split(',').filter_map(|v| v.trim().parse().ok()).collect());
    let text = std::fs::read_to_string(&pf).expect("prompt file");
    let g = GgufFile::open(&path).expect("open gguf");
    let get = |k: &str| g.metadata.get(k);
    let declared = match get("tokenizer.ggml.pre") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    let tokens: Vec<String> = match get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(x)) => x.iter().map(|v| if let Meta::Str(s) = v { s.clone() } else { String::new() }).collect(),
        _ => return,
    };
    let merges: Vec<(String, String)> = match get("tokenizer.ggml.merges") {
        Some(Meta::Arr(x)) => x.iter().filter_map(|v| if let Meta::Str(s) = v {
            s.split_once(' ').map(|(l, r)| (l.to_string(), r.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let actual = Pre::from_gguf(Some(&declared));
    println!("{}  pre={declared:?} -> {actual:?}   prompt {:?}",
             path.rsplit('/').next().unwrap_or(&path), text);
    for (name, p) in [("Gpt2", Pre::Gpt2), ("Qwen2", Pre::Qwen2), ("Qwen35", Pre::Qwen35),
                      ("Llama3", Pre::Llama3), ("Laguna", Pre::Laguna), ("Hyv4", Pre::Hyv4)] {
        let ids = Bpe::new_with_pre(vocab.clone(), &merges, p).encode(&text);
        let tag = match (&want, p == actual) {
            (Some(w), _) if *w == ids => "  <= MATCHES llama.cpp",
            (_, true) => "  (current)",
            _ => "",
        };
        println!("  {name:<7} {ids:?}{tag}");
    }
    if let Some(w) = want { println!("  {:<7} {w:?}   <- llama.cpp", "REF"); }
}
