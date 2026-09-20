//! **Does the server tokenize a SentencePiece checkpoint as byte-level BPE?**
//!
//! `ferric-web` routes `tokenizer.ggml.model` in {llama, gemma4, t5} to SentencePiece. `ferric-serve`
//! routes only `llama`. A checkpoint declaring `gemma4` or `t5` therefore gets byte-level BPE in the
//! server and scored greedy merge in the browser — from the same file.
//!
//! The browser's own comment records what that costs: the vocabulary is ▁-convention while
//! `Bpe::encode` byte-maps spaces to GPT-2's Ġ, so every space strands as a lone Ġ token that no
//! merge can join. It was masked at Q8_0, which was robust enough to answer through the garbage.
//!
//! This prints both id sequences from ONE file so the divergence is a number, not an inference.
//!
//!   cargo run -p ferric-llama --example tokenizer_convention_probe --release -- <model.gguf>
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::{Bpe, Pre, Spm};
use std::collections::HashMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: tokenizer_convention_probe <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");
    let m = |k: &str| g.metadata.get(k);
    let model = match m("tokenizer.ggml.model") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };
    let arch = match m("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };
    println!("file  : {path}");
    println!("arch  : {arch}   tokenizer.ggml.model: {model}");

    let tokens: Vec<String> = match m("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => { println!("no token list"); return; }
    };
    let scores: Vec<f32> = match m("tokenizer.ggml.scores") {
        Some(Meta::Arr(a)) => a.iter().map(|x| if let Meta::F(v) = x { *v as f32 } else { 0.0 }).collect(),
        _ => Vec::new(),
    };
    let merges: Vec<(String, String)> = match m("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(l, r)| (l.to_string(), r.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    // ⭐ the convention question, as a measurable property of the vocabulary rather than a guess:
    // SentencePiece pieces carry ▁ (U+2581); byte-level BPE pieces carry Ġ (U+0120).
    let n_sp = tokens.iter().filter(|t| t.starts_with('\u{2581}')).count();
    let n_bl = tokens.iter().filter(|t| t.starts_with('\u{0120}')).count();
    println!("vocab : {} tokens · {} scores · {} merges", tokens.len(), scores.len(), merges.len());
    println!("        {n_sp} start with ▁ ({:.1}%)  ·  {n_bl} start with Ġ ({:.1}%)",
             100.0 * n_sp as f32 / tokens.len() as f32, 100.0 * n_bl as f32 / tokens.len() as f32);

    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let pre = Pre::from_gguf(match m("tokenizer.ggml.pre") { Some(Meta::Str(p)) => Some(p.as_str()), _ => None });
    let bpe = Bpe::new_with_pre(vocab, &merges, pre);          // what ferric-serve builds
    let spm = if scores.is_empty() { None } else { Some(Spm::new(tokens.clone(), scores)) };  // what ferric-web builds

    let prompts = ["The capital of France is", "Hello world", "a b c", "Write a haiku about rust."];
    let (mut tot_b, mut tot_s, mut agree) = (0usize, 0usize, 0usize);
    println!("\n{:<32} {:>22} {:>22}", "prompt", "serve (Bpe)", "web (Spm)");
    for p in prompts {
        let b = bpe.encode(p);
        let s = spm.as_ref().map(|sp| sp.encode_piece(p, true)).unwrap_or_default();
        tot_b += b.len(); tot_s += s.len();
        if b == s { agree += 1; }
        println!("{:<32} {:>22} {:>22}", format!("{:?}", p), format!("{} ids", b.len()), format!("{} ids", s.len()));
        println!("  serve: {:?}", &b[..b.len().min(14)]);
        println!("  web  : {:?}", &s[..s.len().min(14)]);
    }
    println!("\n{agree}/{} prompts agree · serve emits {tot_b} ids where web emits {tot_s}", prompts.len());

    // the stranded-space signature the browser comment names
    if let Some(&lone) = vocab_lookup(&tokens, "\u{0120}").as_ref() {
        let strand: usize = prompts.iter().map(|p| bpe.encode(p).iter().filter(|&&i| i == lone).count()).sum();
        println!("lone Ġ token id {lone} appears {strand} times in the server's output across these prompts");
    }
    if spm.is_none() { println!("⚠ no scores in this file — Spm cannot be built, so the server's choice is forced"); }
}

fn vocab_lookup(tokens: &[String], want: &str) -> Option<u32> {
    tokens.iter().position(|t| t == want).map(|i| i as u32)
}
