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

/// Which pre-tokenizer regex a checkpoint declares, from GGUF `tokenizer.ggml.pre`.
///
/// This key was read NOWHERE in the tree until 2026-08-21, so every `tokenizer.ggml.model == "gpt2"`
/// file got the GPT-2 ByteLevel scheme — including the entire Qwen family, which declares `qwen2`.
/// The two differ in one rule that matters, and it is measurable against llama.cpp:
///
/// GPT-2 lets only a SPACE lead a word (` ?\p{L}+`). Qwen2 lets **any single non-letter, non-digit,
/// non-newline** character lead it (`[^\r\n\p{L}\p{N}]?\p{L}+`). So `Halvorsen-Reyes` is
/// `Halvorsen` + `-Reyes` under Qwen2 and `Halvorsen` + `-` + `Reyes` under GPT-2 — and the merge
/// `- Re` → `-Re`, rank 67704 in Qwen3's own table, is unreachable in the second case because merges
/// only run WITHIN a chunk. Measured: llama.cpp emits token 67960, Ferric emitted 12 then 693.
///
/// Punctuation that IS preceded by a space is unaffected — the space-punctuation rule claims it
/// first in both schemes — so `the (Reyes) buffer` tokenises identically. The divergence is exactly
/// mid-word punctuation: hyphenated names, apostrophes inside words, dotted identifiers, paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pre {
    /// HF SmolLM/GPT-2 ByteLevel.
    #[default]
    Gpt2,
    /// Qwen2/Qwen3, and anything else declaring `qwen2`.
    Qwen2,
    /// Tencent Hy4 (`hyv4`), and the deepseek3-llm / hunyuan-dense family it shares a regex with.
    ///
    /// Structurally unlike the other two: **no digit rule and no contraction rule**. Its six
    /// alternatives, leftmost-first:
    ///
    /// ```text
    /// [ASCII punct][A-Za-z]+          <- one chunk: `.cfg` and `-Reyes` stay whole
    /// [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    ///  ?[\p{P}\p{S}]+[\r\n]*
    /// \s*[\r\n]+
    /// \s+(?!\S)
    /// \s+
    /// ```
    ///
    /// The first alternative is the distinctive one and it is why this cannot be approximated by
    /// GPT-2: leading punctuation binds to the letters after it, so `foo.bar` is `foo` + `.bar`
    /// where GPT-2 gives `foo` + `.` + `bar`, and the merges reachable inside those chunks differ.
    Hyv4,
}

impl Pre {
    /// Map a GGUF `tokenizer.ggml.pre` value. Unknown values fall back to GPT-2, which is what the
    /// tree did unconditionally before this existed.
    pub fn from_gguf(pre: Option<&str>) -> Pre {
        match pre {
            Some("qwen2") => Pre::Qwen2,
            // llama.cpp maps deepseek3-llm and hunyuan-dense to the identical regex string.
            Some("hyv4") | Some("deepseek3-llm") | Some("hunyuan-dense") => Pre::Hyv4,
            _ => Pre::Gpt2,
        }
    }
}

pub struct Bpe {
    pre: Pre,
    encoder: HashMap<String, u32>,      // token string → id
    decoder: HashMap<u32, String>,      // id → token string
    ranks: HashMap<(String, String), u32>, // merge pair → rank (lower = merged first)
    b2u: Vec<char>,                     // byte → unicode symbol
    u2b: HashMap<char, u8>,             // inverse
}

impl Bpe {
    /// Build from an in-memory vocab (token→id) and an ordered merges list ("a b" per line).
    pub fn new(vocab: HashMap<String, u32>, merges: &[(String, String)]) -> Bpe {
        Bpe::new_with_pre(vocab, merges, Pre::Gpt2)
    }

    /// Build with an explicit pre-tokenizer. See [`Pre`] for why the choice is load-bearing.
    pub fn new_with_pre(vocab: HashMap<String, u32>, merges: &[(String, String)], pre: Pre) -> Bpe {
        let b2u = byte_to_unicode();
        let u2b = b2u.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
        let decoder = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let ranks = merges.iter().enumerate().map(|(i, (a, b))| ((a.clone(), b.clone()), i as u32)).collect();
        Bpe { pre, encoder: vocab, decoder, ranks, b2u, u2b }
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
        for word in pretokenize_with(text, self.pre) {
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
fn pretokenize(text: &str) -> Vec<String> { pretokenize_with(text, Pre::Gpt2) }

/// The `hyv4` / deepseek3-llm / hunyuan-dense splitter: six alternatives, leftmost-first.
///
/// ⚠ **`\p{P}` and `\p{S}` are approximated together** as "not whitespace, not alphabetic, not
/// numeric". The regex only ever uses them as the union `[\p{P}\p{S}]`, so merging them is exact
/// for that class; what the approximation costs is that a numeric-category character which is
/// really punctuation (or vice versa) lands on the wrong side. Rust's `char` exposes no Unicode
/// general category and this crate takes no new dependency, so the alternative is a table of
/// several thousand ranges for a distinction the regex never draws.
///
/// ⛔ **Not verified against llama.cpp's output.** Their splitter is hand-coded precisely because
/// `std::regex` mis-splits runs like `",~"`, and their fork is not built here. The tests below
/// check the properties the regex states, not agreement with a reference — a different claim, and
/// the weaker one.
/// Whether any of the six alternatives matches starting exactly at `j`. Used only to find the end
/// of an unmatched run, so it mirrors the conditions above rather than re-deriving them.
fn matched_here(cs: &[char], j: usize) -> bool {
    let n = cs.len();
    if j >= n { return true }
    let c = cs[j];
    let is_l = |c: char| c.is_alphabetic();
    let is_ps = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    if c.is_whitespace() || is_ps(c) || is_l(c) { return true }
    // a lead char followed by a letter still matches alternative 2
    j + 1 < n && is_l(cs[j + 1]) && c != '\r' && c != '\n'
}

fn pretokenize_hyv4(text: &str) -> Vec<String> {
    const CR: char = '\r';
    const LF: char = '\n';
    let cs: Vec<char> = text.chars().collect();
    let n = cs.len();
    let is_l = |c: char| c.is_alphabetic();
    let is_m = |c: char| c.is_alphabetic() && !c.is_uppercase() && !c.is_lowercase(); // combining-ish
    let is_ps = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    let is_ascii_punct = |c: char| c.is_ascii_punctuation();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i];
        let take = |a: usize, b: usize| -> String { cs[a..b].iter().collect() };

        // 1. [ASCII punct][A-Za-z]+
        if is_ascii_punct(c) && i + 1 < n && cs[i + 1].is_ascii_alphabetic() {
            let mut j = i + 1;
            while j < n && cs[j].is_ascii_alphabetic() { j += 1 }
            out.push(take(i, j)); i = j; continue;
        }
        // 2. An optional single lead char, then a run of letters/marks. The lead class excludes
        //    CR, LF, letters, punctuation and symbols -- so a space or a digit may lead, and
        //    punctuation may not (alternative 1 already claimed the punctuation-then-letters case).
        {
            let lead = c != CR && c != LF && !is_l(c) && !is_ps(c);
            let start = if lead { i + 1 } else { i };
            if start < n && (is_l(cs[start]) || is_m(cs[start])) {
                let mut j = start;
                while j < n && (is_l(cs[j]) || is_m(cs[j])) { j += 1 }
                out.push(take(i, j)); i = j; continue;
            }
        }
        // 3. an optional single space, then a run of punctuation/symbols, then any trailing
        //    newlines.
        {
            let start = if c == ' ' && i + 1 < n && is_ps(cs[i + 1]) { i + 1 } else { i };
            if start < n && is_ps(cs[start]) {
                let mut j = start;
                while j < n && is_ps(cs[j]) { j += 1 }
                while j < n && (cs[j] == CR || cs[j] == LF) { j += 1 }
                out.push(take(i, j)); i = j; continue;
            }
        }
        // 4. a whitespace run that reaches a newline, newlines included.
        if c.is_whitespace() {
            let mut j = i;
            while j < n && cs[j].is_whitespace() && cs[j] != CR && cs[j] != LF { j += 1 }
            if j < n && (cs[j] == CR || cs[j] == LF) {
                while j < n && (cs[j] == CR || cs[j] == LF) { j += 1 }
                out.push(take(i, j)); i = j; continue;
            }
            // 5 and 6 collapse here: a whitespace run either ends the text (5) or is followed by a
            // non-space (6). Both emit the run; the lookahead distinguishes nothing about the output.
            let mut j = i;
            while j < n && cs[j].is_whitespace() { j += 1 }
            out.push(take(i, j)); i = j; continue;
        }

        // ⛔ UNMATCHED TEXT IS ONE CHUNK, NOT ONE CHUNK PER CHARACTER.
        //
        // The six alternatives are NOT exhaustive: a bare digit run matches none of them -- digits
        // are excluded from the punctuation/symbol class, and alternative 2 requires letters after
        // its optional lead. `unicode_regex_split` emits the text between matches as a single
        // chunk, so `abc123` is `abc` + `123`. Emitting per character gave `abc` + `1` + `2` + `3`,
        // which is a different token stream: merges run only WITHIN a chunk, so splitting a number
        // makes every multi-digit merge in the vocabulary unreachable.
        //
        // Found by a test whose expectation was right for the wrong reason -- it asserted `123`
        // stays whole because there is no digit rule, and the bug was that there is no unmatched
        // rule either.
        let mut j = i;
        while j < n && !matched_here(&cs, j) { j += 1 }
        out.push(take(i, j)); i = j;
    }
    out
}

fn pretokenize_with(text: &str, pre: Pre) -> Vec<String> {
    if pre == Pre::Hyv4 { return pretokenize_hyv4(text) }
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
            // Qwen2 alternative 2: `[^\r\n\p{L}\p{N}]?\p{L}+` — ONE non-letter, non-digit,
            // non-newline character may lead a letter run, where GPT-2 admits only a space. Tried
            // before the punctuation-run rule because the regex lists it first, and that ordering is
            // the whole behaviour: `-Reyes` becomes one chunk here and two under GPT-2, which decides
            // whether the merge `- Re` (rank 67704 in Qwen3's table) is reachable at all.
            //
            // A space still falls through to the branch below when what follows is NOT a letter, so
            // ` (Reyes` keeps its ` (` punctuation chunk and tokenises identically under both schemes.
            if pre == Pre::Qwen2 && !is_l(c) && !is_n(c) && c != '\r' && c != '\n'
                && i + 1 < n && is_l(f[i + 1]) {
                let mut e = i + 1;
                while e < n && is_l(f[e]) { e += 1; }
                out.push(f[i..e].iter().collect());
                i = e;
                continue;
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

#[cfg(test)]
mod pre_tests {
    use super::{pretokenize_with, Pre};

    /// The one rule that differs, and the exact string that exposed it.
    ///
    /// Found by diffing against llama.cpp: `Halvorsen-Reyes` produced token 67960 there and 12 then
    /// 693 here, because merges only run WITHIN a pre-token chunk and GPT-2's rule puts the hyphen in
    /// its own chunk — making the merge `- Re` (rank 67704 in Qwen3's own table) unreachable.
    #[test]
    fn qwen2_attaches_one_leading_punctuation_to_a_word_and_gpt2_does_not() {
        assert_eq!(pretokenize_with("Halvorsen-Reyes", Pre::Qwen2), vec!["Halvorsen", "-Reyes"]);
        assert_eq!(pretokenize_with("Halvorsen-Reyes", Pre::Gpt2), vec!["Halvorsen", "-", "Reyes"]);
    }

    /// The control that localised the bug. Punctuation PRECEDED BY A SPACE is claimed by the
    /// space-punctuation rule first in both schemes, so these must stay identical — a "fix" that
    /// changed them would be over-applying the new rule.
    #[test]
    fn punctuation_after_a_space_is_unchanged_by_the_scheme() {
        for t in ["the (Reyes) buffer", "a - b", "end. Next"] {
            assert_eq!(pretokenize_with(t, Pre::Qwen2), pretokenize_with(t, Pre::Gpt2),
                       "space-led punctuation must tokenise identically under both schemes: {t:?}");
        }
    }

    /// Only ONE character may lead, and only when a letter follows it.
    #[test]
    fn the_lead_is_a_single_char_and_needs_a_letter_after_it() {
        // Two punctuation chars: the run rule takes them, no attach.
        assert_eq!(pretokenize_with("a--b", Pre::Qwen2), vec!["a", "--", "b"]);
        // Punctuation with no following letter stays a run.
        assert_eq!(pretokenize_with("a-1", Pre::Qwen2), vec!["a", "-", "1"]);
        // Newlines are excluded by `[^\r\n\p{L}\p{N}]`, so they never attach.
        assert_eq!(pretokenize_with("a\nb", Pre::Qwen2), vec!["a", "\n", "b"]);
    }

    #[test]
    fn the_gguf_key_maps_and_defaults_to_gpt2() {
        assert_eq!(Pre::from_gguf(Some("qwen2")), Pre::Qwen2);
        assert_eq!(Pre::from_gguf(Some("llama-bpe")), Pre::Gpt2, "unknown values keep the old behaviour");
        assert_eq!(Pre::from_gguf(None), Pre::Gpt2, "a file with no key is what the tree assumed for all");
    }
    /// The distinctive rule: leading ASCII punctuation binds to the letters after it. This is what
    /// separates the family from GPT-2, and it decides which merges are reachable at all, since
    /// merges only run WITHIN a chunk.
    #[test]
    fn hyv4_binds_leading_punctuation_to_the_letters_after_it() {
        assert_eq!(pretokenize_with("foo.bar", Pre::Hyv4), vec!["foo", ".bar"]);
        assert_eq!(pretokenize_with("Halvorsen-Reyes", Pre::Hyv4), vec!["Halvorsen", "-Reyes"]);
        // GPT-2 splits the punctuation off on its own; that difference is the point.
        assert_eq!(pretokenize_with("foo.bar", Pre::Gpt2), vec!["foo", ".", "bar"]);
    }

    /// Punctuation NOT followed by letters falls to the punctuation-run rule, and adjacent runs
    /// stay together -- the case llama.cpp hand-codes because `std::regex` mis-splits it.
    #[test]
    fn hyv4_keeps_adjacent_punctuation_runs_together() {
        // `,~b` is NOT one chunk: alternative 1 needs [A-Za-z] immediately after the punctuation,
        // and `~` intervenes, so the punctuation-run rule claims `,~` and `b` starts a new chunk.
        // My first expectation here was wrong; the code was right.
        assert_eq!(pretokenize_with("a,~b", Pre::Hyv4), vec!["a", ",~", "b"]);
        assert_eq!(pretokenize_with("a ,~ b", Pre::Hyv4), vec!["a", " ,~", " b"]);
    }

    /// No digit rule and no contraction rule -- both present in GPT-2 and Qwen2, both absent here.
    /// A digit may LEAD a letter run (it is not excluded from alternative 2's lead class).
    #[test]
    fn hyv4_has_no_digit_or_contraction_rule() {
        assert_eq!(pretokenize_with("abc123", Pre::Hyv4), vec!["abc", "123"]);
        assert_ne!(pretokenize_with("123", Pre::Hyv4), vec!["1", "2", "3"]);
        // GPT-2 isolates every digit; this must not.
        assert_eq!(pretokenize_with("123", Pre::Gpt2), vec!["1", "2", "3"]);
        // `'s` is one chunk under GPT-2's contraction rule and punctuation+letters here -- same
        // string, different rule, and the distinction shows on a contraction GPT-2 does not list.
        assert_eq!(pretokenize_with("it'zz", Pre::Hyv4), vec!["it", "'zz"]);
    }

    /// Every input is reconstructible: the splitter partitions, it never drops or duplicates.
    /// Checked over a spread of shapes because a fall-through that ate a character would otherwise
    /// only show as a quietly wrong token stream.
    #[test]
    fn hyv4_partitions_its_input_exactly() {
        for t in ["", "a", " ", "\n", "\r\n\r\n", "foo.bar baz", "a,~b", "  trailing   ",
                  "mixed 123 ok!!\nnext", "\u{4f60}\u{597d}, world", "-Reyes'quote", "\t\tx"] {
            let parts = pretokenize_with(t, Pre::Hyv4);
            assert_eq!(parts.concat(), *t, "hyv4 split of {t:?} does not reconstruct: {parts:?}");
        }
    }

    #[test]
    fn hyv4_is_selected_by_its_own_name_and_its_regex_family() {
        assert_eq!(Pre::from_gguf(Some("hyv4")), Pre::Hyv4);
        assert_eq!(Pre::from_gguf(Some("deepseek3-llm")), Pre::Hyv4);
        assert_eq!(Pre::from_gguf(Some("hunyuan-dense")), Pre::Hyv4);
        assert_eq!(Pre::from_gguf(Some("something-else")), Pre::Gpt2);
    }

}

/// **WordPiece** — the BERT family's tokenizer (`tokenizer.ggml.model == "bert"`).
///
/// Needed because the small embedding and reranker checkpoints the field actually ships — bge, gte,
/// MiniLM, e5 — are BERT encoders, and none of them could be loaded without this. A 67 MB bge-small
/// replaces a 396 MB decoder-based retriever at the same job.
///
/// Three stages, in llama.cpp's order:
///   1. **clean + lowercase**, dropping control characters,
///   2. **basic split** on whitespace, with punctuation and CJK becoming their own tokens,
///   3. **greedy longest-match** per word against the vocab, continuations prefixed `##`; a word with
///      no match at some position emits `[UNK]` for the WHOLE word, not for the failing piece.
///
/// ⚠ Accent folding is NOT implemented. llama.cpp applies NFD and drops combining marks, so `café`
/// tokenises there as `cafe` and here as whatever the vocab holds for the composed form. English
/// checkpoints are unaffected; this is stated rather than hidden because a wrong tokenisation does not
/// fail, it silently embeds a different string.
pub struct WordPiece {
    vocab: HashMap<String, u32>,
    pub cls: u32,
    pub sep: u32,
    pub unk: u32,
    max_piece: usize,
}

impl WordPiece {
    pub fn new(vocab: HashMap<String, u32>, cls: u32, sep: u32, unk: u32) -> WordPiece {
        // Longest vocab entry in CHARS, less the word-start marker, so the greedy window can
        // still reach the longest real piece.
        let max_piece = vocab.keys().map(|k| k.chars().count()).max().unwrap_or(1);
        WordPiece { vocab, cls, sep, unk, max_piece }
    }

    /// Punctuation as BERT defines it: the four ASCII bands plus anything Unicode calls punctuation.
    /// Each punctuation character becomes its OWN basic token, which is why `Halvorsen-Reyes` splits
    /// at the hyphen where the byte-level BPE families keep it attached.
    fn is_punct(c: char) -> bool {
        let cp = c as u32;
        (33..=47).contains(&cp) || (58..=64).contains(&cp)
            || (91..=96).contains(&cp) || (123..=126).contains(&cp)
            || matches!(cp, 0x2000..=0x206F | 0x3000..=0x303F | 0xFF00..=0xFF65)
    }

    /// CJK gets one token per character, never merged across the boundary.
    fn is_cjk(c: char) -> bool {
        let cp = c as u32;
        matches!(cp, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF
                   | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
    }

    /// Whitespace + punctuation + CJK splitting, lowercased, controls dropped.
    fn basic_split(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        for ch in text.chars() {
            for c in ch.to_lowercase() {
                if c.is_whitespace() {
                    if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                } else if Self::is_punct(c) || Self::is_cjk(c) {
                    if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                    out.push(c.to_string());
                } else if c == '\u{0}' || c == '\u{fffd}' || (c.is_control() && !c.is_whitespace()) {
                    // dropped
                } else {
                    cur.push(c);
                }
            }
        }
        if !cur.is_empty() { out.push(cur); }
        out
    }

    /// Greedy longest-match-first within one word. Returns `None` when any position fails, because
    /// BERT emits `[UNK]` for the entire word rather than keeping the pieces it managed to match.
    fn piece(&self, word: &str) -> Option<Vec<u32>> {
        let ch: Vec<char> = word.chars().collect();
        if ch.len() > 200 { return None; } // BERT's max_input_chars_per_word
        let mut out = Vec::new();
        let mut start = 0;
        while start < ch.len() {
            let mut end = ch.len().min(start + self.max_piece);
            let mut hit = None;
            while end > start {
                let mut s: String = ch[start..end].iter().collect();
                // GGUF stores a WordPiece vocab in the SPM convention, NOT HuggingFace's: the
                // word-INITIAL piece carries `▁` (U+2581) and continuations are bare. HF is the
                // mirror image — bare initial, `##` continuation. Measured in bge-small-en-v1.5:
                // `▁the` is 1996 and `the` is 10760, and `##ffa` does not exist at all while `ffa`
                // does. Reading it the HF way makes almost every word [UNK], which is what the first
                // version of this did.
                if start == 0 { s.insert(0, '\u{2581}'); }
                if let Some(&id) = self.vocab.get(&s) { hit = Some((id, end)); break; }
                end -= 1;
            }
            let (id, e) = hit?;
            out.push(id);
            start = e;
        }
        Some(out)
    }

    /// Encode with the `[CLS] … [SEP]` wrapping the checkpoint was trained on. The pooling position
    /// for a CLS-pooled model is index 0, so dropping the wrapper silently pools the wrong vector.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.cls];
        for w in Self::basic_split(text) {
            match self.piece(&w) {
                Some(p) => ids.extend(p),
                None => ids.push(self.unk),
            }
        }
        ids.push(self.sep);
        ids
    }
}

#[cfg(test)]
mod wordpiece_tests {
    use super::WordPiece;
    use std::collections::HashMap;

    fn wp(entries: &[(&str, u32)]) -> WordPiece {
        let v: HashMap<String, u32> = entries.iter().map(|(s, i)| (s.to_string(), *i)).collect();
        WordPiece::new(v, 101, 102, 100)
    }

    #[test]
    fn the_word_start_marker_is_SPM_not_huggingface() {
        // The defect this replaced: reading the vocab the HuggingFace way (bare initial, `##`
        // continuation) makes almost every word [UNK], because GGUF stores the mirror image.
        // `▁una` + `ffa` + `ble` is how bge-small-en-v1.5 actually holds "unaffable".
        let t = wp(&[("\u{2581}una", 1), ("ffa", 2), ("ble", 3)]);
        assert_eq!(t.encode("unaffable"), vec![101, 1, 2, 3, 102]);
        // The same pieces under the HF convention must NOT resolve — proof the marker is load-bearing.
        let hf = wp(&[("una", 1), ("##ffa", 2), ("##ble", 3)]);
        assert_eq!(hf.encode("unaffable"), vec![101, 100, 102], "HF-style vocab cannot match here");
    }

    #[test]
    fn punctuation_becomes_its_own_word_and_gets_the_marker() {
        // `Halvorsen-Reyes` splits at the hyphen and the hyphen is a word of its own — the opposite
        // of the byte-level BPE families, where a mid-word hyphen ATTACHES to the following word.
        let t = wp(&[("\u{2581}a", 1), ("\u{2581}-", 2), ("\u{2581}b", 3)]);
        assert_eq!(t.encode("a-b"), vec![101, 1, 2, 3, 102]);
    }

    #[test]
    fn an_unmatched_word_is_UNK_as_a_WHOLE_not_piecewise() {
        // BERT discards the pieces it did match. Emitting them would silently change the input the
        // model sees rather than marking it unknown.
        let t = wp(&[("\u{2581}ab", 1), ("\u{2581}zz", 9)]);
        assert_eq!(t.encode("abqq"), vec![101, 100, 102], "partial match must not leak pieces");
        assert_eq!(t.encode("zz"), vec![101, 9, 102]);
    }

    #[test]
    fn casing_is_folded_and_cls_sep_always_wrap() {
        let t = wp(&[("\u{2581}the", 5)]);
        assert_eq!(t.encode("THE"), vec![101, 5, 102]);
        assert_eq!(t.encode(""), vec![101, 102], "an empty string is still CLS+SEP, not empty");
    }
}
