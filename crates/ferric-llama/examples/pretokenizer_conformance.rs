//! **Which `Pre` rule reproduces llama.cpp, over a multi-script corpus?**
//!
//! Reads `text|[id, id, ...]` lines produced by `llama-tokenize --ids` and reports, per `Pre`
//! variant, how many strings it reproduces EXACTLY. The reference is the outside implementation.
//!
//!   cargo run -p ferric-llama --example pretokenizer_conformance --release -- <model.gguf> <ref.txt>
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: <model.gguf> <ref.txt>");
    let refp = a.next().expect("usage: <model.gguf> <ref.txt>");
    let g = GgufFile::open(&path).expect("open gguf");
    let get = |k: &str| g.metadata.get(k);
    let declared = match get("tokenizer.ggml.pre") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };

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

    let txt = std::fs::read_to_string(&refp).expect("ref file");
    let cases: Vec<(String, Vec<u32>)> = txt.lines().filter_map(|l| {
        let (s, ids) = l.rsplit_once('|')?;
        let ids: Vec<u32> = ids.trim().trim_start_matches('[').trim_end_matches(']')
            .split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if ids.is_empty() { None } else { Some((s.to_string(), ids)) }
    }).collect();

    println!("model pre={declared:?}  ·  currently mapped to {:?}  ·  {} reference cases\n",
             Pre::from_gguf(Some(&declared)), cases.len());
    let variants: [(&str, Pre); 3] = [("Gpt2 (current)", Pre::Gpt2), ("Qwen2", Pre::Qwen2), ("Hyv4", Pre::Hyv4)];
    let mut best = ("", 0usize);
    for (name, p) in variants {
        let b = Bpe::new_with_pre(vocab.clone(), &merges, p);
        let mut ok = 0usize;
        let mut fails: Vec<&str> = Vec::new();
        for (s, want) in &cases {
            if &b.encode(s) == want { ok += 1 } else if fails.len() < 6 { fails.push(s) }
        }
        if ok > best.1 { best = (name, ok); }
        println!("  {name:<16} {ok:>2}/{} exact", cases.len());
        if !fails.is_empty() { println!("      misses: {}", fails.join(" · ")); }
    }
    println!("\n  closest to the reference: {} ({}/{})", best.0, best.1, cases.len());
}
