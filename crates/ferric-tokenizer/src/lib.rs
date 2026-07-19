//! A pure-Rust byte-level BPE tokenizer (the GPT-2 / RoBERTa / tiktoken family). Byte-level means
//! every input maps losslessly to tokens and back — `decode(encode(x)) == x` for arbitrary bytes.
//! Loads the standard `vocab.json` + `merges.txt` pair. No GPU, compiles clean to wasm32.
//!
//! Pre-tokenization is the full HF SmolLM/GPT-2 ByteLevel scheme — a `Digits(individual)` split
//! followed by the GPT-2 regex (contractions, leading-space attach, multi-space runs, punctuation
//! runs), hand-rolled in `pretokenize()` to match HF `tokenizers` token-for-token. Verified against a
//! reference id-set: `cargo run -p ferric-tokenizer --example verify_tok` (6/6 identical, incl. the
//! edge cases — "don't"/"it's", "3.14"→3·.·1·4, multi-space, punctuation). (Full tiktoken/cl100k —
//! `\p{N}{1,3}` number grouping, case-insensitive contractions — is a separate regex Ferric's target
//! GGUF models, GPT-2-family all, don't use.)

use std::collections::HashMap;

/// GPT-2 byte↔printable-unicode bijection, so raw bytes can live as vocab symbols.
fn byte_to_unicode() -> Vec<char> {
    let mut bs: Vec<u16> = Vec::new();
    bs.extend(b'!' as u16..=b'~' as u16);
    bs.extend(0xA1u16..=0xAC);
    bs.extend(0xAEu16..=0xFF);
    let mut map = vec!['\0'; 256];
    let mut extra = 0u32;
    for b in 0u16..256 {
        if bs.contains(&b) {
            map[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            map[b as usize] = char::from_u32(256 + extra).unwrap();
            extra += 1;
        }
    }
    map
}

/// The 256 base byte-symbols mapped to ids 0..256 — the floor of any byte-level BPE vocab.
pub fn base_byte_vocab() -> HashMap<String, u32> {
    byte_to_unicode().iter().enumerate().map(|(i, &c)| (c.to_string(), i as u32)).collect()
}

pub struct Bpe {
    encoder: HashMap<String, u32>,      // token string → id
    decoder: HashMap<u32, String>,      // id → token string
    ranks: HashMap<(String, String), u32>, // merge pair → rank (lower = merged first)
    b2u: Vec<char>,                     // byte → unicode symbol
    u2b: HashMap<char, u8>,             // inverse
}

impl Bpe {
    /// Build from an in-memory vocab (token→id) and an ordered merges list ("a b" per line).
    pub fn new(vocab: HashMap<String, u32>, merges: &[(String, String)]) -> Bpe {
        let b2u = byte_to_unicode();
        let u2b = b2u.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
        let decoder = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let ranks = merges.iter().enumerate().map(|(i, (a, b))| ((a.clone(), b.clone()), i as u32)).collect();
        Bpe { encoder: vocab, decoder, ranks, b2u, u2b }
    }

    /// Load the standard HF/GPT-2 `vocab.json` + `merges.txt`.
    pub fn from_gpt2(vocab_json: &str, merges_txt: &str) -> Result<Bpe, String> {
        let v: serde_json::Value = serde_json::from_str(vocab_json).map_err(|e| e.to_string())?;
        let vocab: HashMap<String, u32> = v.as_object().ok_or("vocab.json not an object")?
            .iter().map(|(k, val)| (k.clone(), val.as_u64().unwrap() as u32)).collect();
        let merges: Vec<(String, String)> = merges_txt.lines()
            .filter(|l| !l.is_empty() && !l.starts_with("#version"))
            .filter_map(|l| { let mut it = l.split_whitespace(); Some((it.next()?.to_string(), it.next()?.to_string())) })
            .collect();
        Ok(Bpe::new(vocab, &merges))
    }

    /// Load from an HF `tokenizer.json` (the single-file format) — reads `model.vocab` + `model.merges`.
    pub fn from_tokenizer_json(bytes: &[u8]) -> Result<Bpe, String> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let model = &v["model"];
        let vocab: HashMap<String, u32> = model["vocab"].as_object().ok_or("no model.vocab")?
            .iter().map(|(k, val)| (k.clone(), val.as_u64().unwrap() as u32)).collect();
        let merges: Vec<(String, String)> = model["merges"].as_array().ok_or("no model.merges")?.iter().filter_map(|m| {
            if let Some(s) = m.as_str() { let mut it = s.splitn(2, ' '); Some((it.next()?.to_string(), it.next()?.to_string())) }
            else if let Some(a) = m.as_array() { Some((a[0].as_str()?.to_string(), a[1].as_str()?.to_string())) }
            else { None }
        }).collect();
        Ok(Bpe::new(vocab, &merges))
    }

    pub fn vocab_size(&self) -> usize { self.encoder.len() }

    /// Apply BPE to one pre-token's symbols: repeatedly merge the lowest-rank adjacent pair.
    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            // find the best (lowest-rank) adjacent pair
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.is_none_or(|(_, br)| r < br) { best = Some((i, r)); }
                }
            }
            let Some((i, _)) = best else { break };
            symbols[i] = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.remove(i + 1);
        }
        symbols
    }

    /// Encode text → token ids (lossless byte-level).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in pretokenize(text) {
            let symbols: Vec<String> = word.bytes().map(|b| self.b2u[b as usize].to_string()).collect();
            for tok in self.bpe(symbols) {
                // any merged token is in the vocab; base byte-symbols always are
                ids.push(*self.encoder.get(&tok).expect("token missing from vocab"));
            }
        }
        ids
    }

    /// Decode token ids → text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let s: String = ids.iter().map(|id| self.decoder[id].clone()).collect();
        let bytes: Vec<u8> = s.chars().map(|c| self.u2b[&c]).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// The HF SmolLM/GPT-2 pre-tokenizer: a `Digits(individual)` split (each digit isolated) followed by
/// the ByteLevel GPT-2 regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
/// Hand-rolled to match `tokenizers` token-for-token — contractions, leading-space attach, multi-space
/// runs (last space joins the next word), punctuation runs, digits individual.
fn pretokenize(text: &str) -> Vec<String> {
    // Digits step: isolate each digit; group consecutive non-digits.
    let mut frags: Vec<Vec<char>> = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            if !cur.is_empty() { frags.push(std::mem::take(&mut cur)); }
            frags.push(vec![c]);
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() { frags.push(cur); }

    let is_l = |c: char| c.is_alphabetic();
    let is_n = |c: char| c.is_ascii_digit();
    let is_punct = |c: char| !c.is_whitespace() && !is_l(c) && !is_n(c);
    let mut out = Vec::new();
    for f in &frags {
        let n = f.len();
        let mut i = 0;
        while i < n {
            let c = f[i];
            // contractions
            if c == '\'' && i + 1 < n {
                let two: String = f[i + 1..(i + 3).min(n)].iter().collect();
                if two.starts_with("re") || two.starts_with("ve") || two.starts_with("ll") { out.push(f[i..i + 3].iter().collect()); i += 3; continue; }
                if matches!(f[i + 1], 's' | 't' | 'm' | 'd') { out.push(f[i..i + 2].iter().collect()); i += 2; continue; }
            }
            let sp = c == ' ';
            let j = i + sp as usize;
            let cls = |p: &dyn Fn(char) -> bool| j < n && p(f[j]);
            if cls(&is_l) || cls(&is_n) || cls(&is_punct) {
                let pred: &dyn Fn(char) -> bool = if is_l(f[j]) { &is_l } else if is_n(f[j]) { &is_n } else { &is_punct };
                let mut e = j; while e < n && pred(f[e]) { e += 1; }
                out.push(f[i..e].iter().collect()); i = e; continue;
            }
            // whitespace run. The ByteLevel `\s+(?!\S)|\s+` + ` ?\p{L}+` rules attach only a trailing
            // SPACE (0x20) to the following word; other whitespace (\n, \t) is its own run. A single
            // space directly before content is already handled by the `sp` branch above, so the
            // "last char joins next word" split only fires for a genuine ≥2-space run — guarding on
            // `e-1 > i` and `== ' '`. (The old unconditional `i = e-1` looped forever on a bare "\n",
            // where e-1 == i, emitting empty strings until OOM — any text with a newline crashed.)
            let mut e = i; while e < n && f[e].is_whitespace() { e += 1; }
            if e < n && f[e - 1] == ' ' && e - 1 > i {
                out.push(f[i..e - 1].iter().collect()); i = e - 1; // last space joins the next word
            } else {
                out.push(f[i..e].iter().collect()); i = e;
            }
        }
    }
    if out.is_empty() { out.push(String::new()); }
    out
}
