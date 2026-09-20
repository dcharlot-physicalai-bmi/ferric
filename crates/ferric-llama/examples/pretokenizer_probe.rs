//! **Does the pre-tokenizer fall-open change the tokens?**
//!
//! `Pre::from_gguf` maps `qwen2` and the hyv4 regex family, and sends EVERYTHING ELSE to GPT-2 —
//! "which is what the tree did unconditionally before this existed". A survey of the GGUFs on this
//! machine found 16 of 32 declaring a `pre` that falls through: qwen35, default, llama-bpe, lfm2,
//! pixtral, llama4, laguna, deepseek-llm.
//!
//! Falling open is only harmless if the fallback regex agrees with the right one ON REAL TEXT. This
//! encodes the same strings under each rule and prints where they part company.
//!
//!   cargo run -p ferric-llama --example pretokenizer_probe --release -- <model.gguf>
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: pretokenizer_probe <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");
    let get = |k: &str| g.metadata.get(k);
    let declared = match get("tokenizer.ggml.pre") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };
    let arch = match get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };
    println!("{}\n  arch {arch} · tokenizer.ggml.pre {declared:?} · Ferric maps it to {:?}",
             path.rsplit('/').next().unwrap_or(&path), Pre::from_gguf(Some(&declared)));

    let tokens: Vec<String> = match get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => { println!("  no tokens"); return; }
    };
    let merges: Vec<(String, String)> = match get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(l, r)| (l.to_string(), r.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();

    // ⚠ Strings chosen for the constructs the pre-tokenizer rules actually differ on: a hyphen before
    // a capital (the `-Reyes` case the qwen2 mapping was added for), digits, contractions, newlines
    // and repeated spaces. Ordinary prose agrees under every rule, which is why this went unnoticed.
    let probes = ["Hello-Reyes", "the year 2026 and 1999", "don't can't it's",
                  "a\n\nb", "x    y", "CamelCaseWord", "print(x)  # 3.14"];
    let a = Bpe::new_with_pre(vocab.clone(), &merges, Pre::from_gguf(Some(&declared)));
    let b = Bpe::new_with_pre(vocab, &merges, Pre::Qwen2);

    let (mut diff, mut shown) = (0usize, 0usize);
    for p in probes {
        let (x, y) = (a.encode(p), b.encode(p));
        if x != y {
            diff += 1;
            if shown < 5 {
                println!("  {p:?}\n     as mapped ({:?}): {x:?}\n     as Qwen2       : {y:?}", Pre::from_gguf(Some(&declared)));
                shown += 1;
            }
        }
    }
    println!("  -> {diff}/{} probe strings tokenize DIFFERENTLY under the two rules", probes.len());
    if diff == 0 {
        println!("     (for this vocabulary the fall-open is harmless on these constructs)");
    }
}
