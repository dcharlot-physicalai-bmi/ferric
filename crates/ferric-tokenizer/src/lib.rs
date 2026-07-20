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

/// A **SentencePiece** tokenizer (llama.cpp `tokenizer.ggml.model == "llama"` — SPM/BPE-with-scores).
/// This is the Llama-2 / Mistral / **Phi-3** / Gemma family: a scored vocab (no merges list), spaces
/// encoded as `▁` (U+2581), and `<0xXX>` byte-fallback tokens for anything out of vocab. Tokenization
/// is the SPM greedy merge — repeatedly fuse the adjacent pair whose combined token has the highest
/// vocab score — matching llama.cpp's `llm_tokenizer_spm`.
pub struct Spm {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    scores: Vec<f32>,
    byte_tok: Vec<Option<u32>>, // byte value → `<0xXX>` id
}

impl Spm {
    pub fn new(tokens: Vec<String>, scores: Vec<f32>) -> Spm {
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let byte_tok = (0..256u32).map(|b| vocab.get(&format!("<0x{b:02X}>")).copied()).collect();
        Spm { tokens, vocab, scores, byte_tok }
    }
    pub fn vocab_size(&self) -> usize { self.tokens.len() }
    pub fn id_of(&self, s: &str) -> Option<u32> { self.vocab.get(s).copied() }
    fn score_of(&self, s: &str) -> Option<f32> { self.vocab.get(s).map(|&id| self.scores.get(id as usize).copied().unwrap_or(0.0)) }

    /// Encode one raw-text fragment. `prefix` requests SentencePiece's leading-space (▁) — true only for
    /// text at the very start of the sequence (text following a special token gets none).
    pub fn encode_piece(&self, text: &str, prefix: bool) -> Vec<u32> {
        if text.is_empty() { return Vec::new(); }
        // Escape whitespace to ▁ and optionally prepend the leading ▁.
        let mut esc = String::new();
        if prefix { esc.push('\u{2581}'); }
        for c in text.chars() { esc.push(if c == ' ' { '\u{2581}' } else { c }); }
        // Initial symbols = individual UTF-8 chars; greedily merge the highest-score adjacent pair.
        let mut syms: Vec<String> = esc.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, f32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                let merged = format!("{}{}", syms[i], syms[i + 1]);
                if let Some(sc) = self.score_of(&merged) {
                    if best.is_none_or(|(_, bs)| sc > bs) { best = Some((i, sc)); }
                }
            }
            let Some((i, _)) = best else { break };
            let right = syms.remove(i + 1);
            syms[i].push_str(&right);
        }
        // Resegment to ids; anything still out-of-vocab falls back to its raw `<0xXX>` bytes.
        let mut ids = Vec::new();
        for s in &syms {
            if let Some(&id) = self.vocab.get(s) { ids.push(id); }
            else { for b in s.bytes() { if let Some(id) = self.byte_tok[b as usize] { ids.push(id); } } }
        }
        ids
    }

    /// Raw bytes a token represents: `<0xXX>` → that byte, `▁` → space, a bracketed control piece
    /// (`<s>`, `</s>`, `<|user|>`, `<unk>`, …) → None (not literal text — barred under a guided constraint).
    pub fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        let t = self.tokens.get(id as usize)?;
        if t.len() == 6 && t.starts_with("<0x") && t.ends_with('>') {
            return u8::from_str_radix(&t[3..5], 16).ok().map(|b| vec![b]);
        }
        if t.starts_with('<') && t.ends_with('>') && (t.contains('|') || t.ends_with("s>") || matches!(t.as_str(), "<unk>" | "<pad>" | "<mask>")) {
            return None;
        }
        Some(t.replace('\u{2581}', " ").into_bytes())
    }

    /// Decode ids → text (drops control tokens; joins the rest's bytes).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids { if let Some(b) = self.token_bytes(id) { bytes.extend(b); } }
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
