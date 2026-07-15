//! Look up token ids in a GGUF's own vocab (byte-level GPT-2 symbols: "Ġ" = leading space).
use ferric_gguf::{GgufFile, Meta};
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    let toks: Vec<&str> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.as_str() } else { "" }).collect(), _ => vec![] };
    let want: Vec<String> = std::env::args().skip(2).collect();
    let ids: Vec<String> = want.iter().map(|w| {
        match toks.iter().position(|t| t == w) { Some(i) => i.to_string(), None => format!("<MISSING {w:?}>") }
    }).collect();
    for (w, i) in want.iter().zip(&ids) { println!("  {:>12?} → {}", w, i); }
    println!("{}", ids.join(","));
}
