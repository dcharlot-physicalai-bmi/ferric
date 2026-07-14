//! A pure-Rust byte-level BPE tokenizer (the GPT-2 / RoBERTa / tiktoken family). Byte-level means
//! every input maps losslessly to tokens and back — `decode(encode(x)) == x` for arbitrary bytes.
//! Loads the standard `vocab.json` + `merges.txt` pair. No GPU, compiles clean to wasm32.
//!
//! (Pre-tokenization here is a simple whitespace-aware split rather than GPT-2's full regex; it's
//! lossless and correct BPE — exact GPT-2 token-for-token parity would swap in the regex pre-tokenizer.)

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
        for word in split(text) {
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

/// Lossless whitespace-aware pre-tokenizer: keep leading spaces attached to the following word.
fn split(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c == ' ' && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() { out.push(cur); }
    if out.is_empty() { out.push(String::new()); }
    out
}
