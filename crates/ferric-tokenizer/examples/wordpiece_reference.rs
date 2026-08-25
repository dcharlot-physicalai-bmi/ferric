//! **Reference-diff WordPiece against llama.cpp.**
//!   llama-tokenize -m bert.gguf -p TEXT --ids
//!   cargo run -p ferric-tokenizer --example wordpiece_reference --release -- bert.gguf
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::WordPiece;
use std::collections::HashMap;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: wordpiece_reference <bert.gguf>");
    let g = GgufFile::open(mp).expect("open");
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let u = |k: &str, d: u32| match g.metadata.get(k) { Some(Meta::U(v)) => *v as u32, _ => d };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let wp = WordPiece::new(vocab, u("tokenizer.ggml.cls_token_id", 101),
                            u("tokenizer.ggml.seperator_token_id", 102),
                            u("tokenizer.ggml.unknown_token_id", 100));

    // (text, llama.cpp ids). Chosen to cover the cases that separate a real WordPiece from a sketch:
    // subword continuation, punctuation splitting, digit splitting, and casing.
    let cases: &[(&str, &[u32])] = &[
        ("the capital of France is Paris", &[101, 1996, 3007, 1997, 2605, 2003, 3000, 102]),
        ("Halvorsen-Reyes buffer 2265", &[101, 11085, 14550, 5054, 1011, 12576, 17698, 21035, 2629, 102]),
        ("unaffable", &[101, 14477, 20961, 3468, 102]),
    ];
    let mut bad = 0;
    for (t, want) in cases {
        let got = wp.encode(t);
        let ok = got == *want;
        if !ok { bad += 1; }
        println!("{} {:<32} {:?}", if ok { "✅" } else { "❌" }, &t[..t.len().min(30)], got);
        if !ok { println!("     llama.cpp wanted {want:?}"); }
    }
    if bad > 0 { std::process::exit(1); }
    println!("\n  {} / {} identical to llama.cpp", cases.len(), cases.len());
}
