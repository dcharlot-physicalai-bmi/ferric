//! Verify the premise behind `ferric_tier::kvstore`'s API: **does BPE actually merge across a byte
//! boundary?**
//!
//! `KvStore::resume` deliberately returns a *byte range* for the un-cached suffix rather than a token
//! slice, because slicing the already-tokenised prompt at a cache boundary is claimed to produce a
//! different token sequence than the model cached. That claim is load-bearing — it is the reason the API
//! is shaped the way it is — so it gets measured against the real Qwen tokenizer rather than asserted.
//!
//!   cargo run -p ferric-llama --example bpe_boundary --release
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;

fn main() {
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();
    let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!(),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!(),
    };
    let bpe = Bpe::new(vocab, &merges);

    let text = "The history of artificial intelligence began in antiquity, with myths and stories of \
artificial beings endowed with intelligence by master craftsmen who built them.";
    let full = bpe.encode(text);

    println!("BPE boundary check — Qwen2.5 tokenizer, {} chars -> {} tokens\n", text.len(), full.len());
    println!("  For each split point: is tokenize(whole) == tokenize(prefix) ++ tokenize(suffix)?\n");

    let (mut clean, mut dirty) = (0usize, 0usize);
    let mut examples = Vec::new();
    for cut in 1..text.len() {
        if !text.is_char_boundary(cut) { continue; }
        let (a, b) = text.split_at(cut);
        let (ta, tb) = (bpe.encode(a), bpe.encode(b));
        let joined: Vec<u32> = ta.iter().chain(tb.iter()).copied().collect();
        if joined == full {
            clean += 1;
        } else {
            dirty += 1;
            if examples.len() < 3 && cut > 20 {
                examples.push((cut, a.chars().rev().take(12).collect::<String>().chars().rev().collect::<String>(),
                               joined.len(), full.len()));
            }
        }
    }

    let total = clean + dirty;
    println!("  split points tested: {total}");
    println!("  concatenation matches the whole-text tokenisation: {clean}  ({:.1}%)", 100.0 * clean as f64 / total as f64);
    println!("  concatenation DIFFERS:                             {dirty}  ({:.1}%)", 100.0 * dirty as f64 / total as f64);
    for (cut, tail, jl, fl) in &examples {
        println!("    e.g. byte {cut} (prefix ends '...{tail}'): {jl} tokens vs {fl}");
    }

    println!();
    if dirty > 0 {
        println!("  ✅ PREMISE CONFIRMED. Splitting at a byte boundary and concatenating the two");
        println!("     tokenisations does NOT in general reproduce the whole-text tokenisation, so a KV");
        println!("     checkpoint resumed from a token slice would feed the model a sequence it never");
        println!("     cached — silently. `KvStore::resume` returning a BYTE RANGE is therefore load-");
        println!("     bearing, not stylistic: it makes that bug inexpressible.");
        let pct = 100.0 * dirty as f64 / total as f64;
        println!("\n     And it is the COMMON case, not an edge case: {pct:.0}% of split points differ.");
        println!("     A resume path that slices tokens is therefore wrong most of the time it is used —");
        println!("     but wrong SILENTLY, degrading output with no error, which is why it survives review.");
    } else {
        println!("  ❌ PREMISE NOT CONFIRMED on this text — every split concatenated cleanly.");
        println!("     The byte-range API is then merely defensive rather than necessary; say so.");
    }
}
