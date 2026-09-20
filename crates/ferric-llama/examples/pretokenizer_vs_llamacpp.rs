//! **Which `Pre` rule reproduces llama.cpp on this checkpoint?**
//!
//! Encodes each probe string under every `Pre` variant and prints the ids, so they can be compared
//! against `llama-tokenize --ids` on the same file. The reference is the outside implementation, not
//! a family resemblance: llama.cpp gives `qwen35` its OWN pre-type (`LLAMA_VOCAB_PRE_TYPE_QWEN35`,
//! llama-vocab.cpp:2223), whose regex is QWEN2's plus `\p{M}` — combining marks counted as letters.
//! That differs from QWEN2 only on scripts with diacritics, which is why ASCII testing misses it.
//!
//!   cargo run -p ferric-llama --example pretokenizer_vs_llamacpp --release -- <model.gguf> <text>...
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: <model.gguf> <text>...");
    let texts: Vec<String> = a.collect();
    let g = GgufFile::open(&path).expect("open gguf");
    let get = |k: &str| g.metadata.get(k);
    let declared = match get("tokenizer.ggml.pre") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };
    println!("pre={declared:?}  currently mapped to {:?}\n", Pre::from_gguf(Some(&declared)));

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
    let variants: [(&str, Pre); 3] = [("Gpt2", Pre::Gpt2), ("Qwen2", Pre::Qwen2), ("Hyv4", Pre::Hyv4)];
    for t in &texts {
        println!("{t:?}");
        for (name, p) in variants {
            let b = Bpe::new_with_pre(vocab.clone(), &merges, p);
            println!("  {name:<6} {:?}", b.encode(t));
        }
    }
}
