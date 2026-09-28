//! **GBNF grammars** — llama.cpp's grammar format, parsed and matched as llama.cpp does.
//!
//! Parity gap S10 (grammar-constrained decoding, 8.5 of 14 serving peers). GBNF is llama.cpp's own format,
//! so llama.cpp is its authors: this is a line-for-line port of `src/llama-grammar.cpp` (ggml-org/llama.cpp
//! @ 4da6337767f9) — the same flat rule encoding (typed elements with ALT/END markers), the same repetition
//! rewrite (`S{m,n}` → S…S S'(n-m)), the same parse stacks advanced per code point, the same partial-UTF-8
//! rule for a token that ends mid-character, and the same `<[id]>` / `!<[id]>` token elements — so a grammar
//! accepts exactly the strings llama.cpp's accepts. Checked against llama.cpp's own
//! `tests/test-grammar-integration.cpp` cases (tests/fixtures/grammar/, extracted by `extract.py` there).
//!
//! Regex and choice lists (vLLM's `guided_regex` / `guided_choice`) compile INTO this representation
//! (`regex.rs`), so one matcher serves every constraint.
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum Ty { End, Alt, RuleRef, Char, CharNot, CharRngUpper, CharAlt, CharAny, Token, TokenNot }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct El { pub ty: Ty, pub v: u32 }
const fn el(ty: Ty, v: u32) -> El { El { ty, v } }

const MAX_REPETITION_THRESHOLD: u64 = 2000;

/// A parsed grammar: rules by id, and the id of the start symbol.
#[derive(Debug)]
pub struct Grammar { pub rules: Vec<Vec<El>>, pub root: usize, pub symbols: HashMap<String, u32> }

fn is_word(c: u8) -> bool { c.is_ascii_lowercase() || c.is_ascii_uppercase() || c == b'-' || c.is_ascii_digit() }

/// `decode_utf8` on one sequence at `s[p..]`: (code point, bytes used). An invalid lead byte gives (byte, 1),
/// as llama.cpp's single-sequence decoder does for the grammar source.
fn decode_one(s: &[u8], p: usize) -> (u32, usize) {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
    let first = s[p];
    let len = LOOKUP[(first >> 4) as usize];
    let mask = (1u8 << (8 - len.max(1))) - 1;
    let mut v = (first & mask) as u32;
    let mut q = p + 1;
    while q < p + len.max(1) && q < s.len() { v = (v << 6) + (s[q] & 0x3F) as u32; q += 1; }
    (v, q - p)
}

struct Parser<'a> {
    s: &'a [u8],
    rules: Vec<Vec<El>>,
    symbols: HashMap<String, u32>,
    token_of: &'a dyn Fn(&str) -> Option<u32>,
}

type R<T> = Result<T, String>;

impl<'a> Parser<'a> {
    fn at(&self, p: usize) -> u8 { self.s.get(p).copied().unwrap_or(0) }
    fn err(&self, what: &str, p: usize) -> String {
        format!("{what} at {:?}", String::from_utf8_lossy(&self.s[p.min(self.s.len())..(p + 24).min(self.s.len())]))
    }
    fn space(&self, mut p: usize, newline_ok: bool) -> usize {
        loop {
            match self.at(p) {
                b' ' | b'\t' => p += 1,
                b'#' => { while !matches!(self.at(p), 0 | b'\r' | b'\n') { p += 1; } }
                b'\r' | b'\n' if newline_ok => p += 1,
                _ => return p,
            }
        }
    }
    fn name_end(&self, src: usize) -> R<usize> {
        let mut p = src;
        while is_word(self.at(p)) { p += 1; }
        if p == src { return Err(self.err("expecting name", src)); }
        Ok(p)
    }
    fn int_end(&self, src: usize) -> R<usize> {
        let mut p = src;
        while self.at(p).is_ascii_digit() { p += 1; }
        if p == src { return Err(self.err("expecting integer", src)); }
        Ok(p)
    }
    fn hex(&self, src: usize, size: usize) -> R<(u32, usize)> {
        let (mut v, mut p) = (0u32, src);
        while p < src + size {
            let c = self.at(p);
            let d = match c { b'a'..=b'f' => c - b'a' + 10, b'A'..=b'F' => c - b'A' + 10, b'0'..=b'9' => c - b'0', _ => break };
            v = (v << 4) + d as u32;
            p += 1;
        }
        if p != src + size { return Err(self.err(&format!("expecting {size} hex chars"), src)); }
        Ok((v, p))
    }
    fn char_(&self, p: usize) -> R<(u32, usize)> {
        if self.at(p) == b'\\' {
            return match self.at(p + 1) {
                b'x' => self.hex(p + 2, 2), b'u' => self.hex(p + 2, 4), b'U' => self.hex(p + 2, 8),
                b't' => Ok(('\t' as u32, p + 2)), b'r' => Ok(('\r' as u32, p + 2)), b'n' => Ok(('\n' as u32, p + 2)),
                c @ (b'\\' | b'"' | b'[' | b']' | b'-') => Ok((c as u32, p + 2)),
                _ => Err(self.err("unknown escape", p)),
            };
        } else if self.at(p) != 0 {
            let (v, n) = decode_one(self.s, p);
            return Ok((v, p + n));
        }
        Err("unexpected end of input".into())
    }
    fn token(&self, src: usize) -> R<(u32, usize)> {
        let mut p = src;
        if self.at(p) != b'<' { return Err(self.err("expecting '<'", p)); }
        p += 1;
        if self.at(p) == b'[' {
            p += 1;
            let e = self.int_end(p)?;
            let id: u64 = std::str::from_utf8(&self.s[p..e]).unwrap().parse().map_err(|_| self.err("parsed token id is too big", p))?;
            if id > u32::MAX as u64 { return Err(self.err("parsed token id is too big", p)); }
            p = e;
            if self.at(p) != b']' { return Err(self.err("expecting ']'", p)); }
            p += 1;
            if self.at(p) != b'>' { return Err(self.err("expecting '>'", p)); }
            return Ok((id as u32, p + 1));
        }
        while self.at(p) != 0 && self.at(p) != b'>' { p += 1; }
        if self.at(p) != b'>' { return Err(self.err("expecting '>'", p)); }
        p += 1;
        let text = String::from_utf8_lossy(&self.s[src..p]).into_owned();
        let id = (self.token_of)(&text).ok_or_else(|| format!("invalid token '{text}'"))?;
        Ok((id, p))
    }
    fn symbol_id(&mut self, name: &str) -> u32 {
        let next = self.symbols.len() as u32;
        *self.symbols.entry(name.to_string()).or_insert(next)
    }
    fn generate_symbol_id(&mut self, base: &str) -> u32 {
        let next = self.symbols.len() as u32;
        self.symbols.insert(format!("{base}_{next}"), next);
        next
    }
    fn add_rule(&mut self, id: u32, rule: Vec<El>) {
        let id = id as usize;
        if self.rules.len() <= id { self.rules.resize(id + 1, Vec::new()); }
        self.rules[id] = rule;
    }

    fn sequence(&mut self, src: usize, rule_name: &str, rule: &mut Vec<El>, nested: bool) -> R<usize> {
        let mut last_sym_start = rule.len();
        let mut p = src;
        let mut n_prev_rules: u64 = 1;
        // S{m,n} → S … S (m times) S'(n-m); S'(x) ::= S S'(x-1) |; S{m,} → S … S S', S' ::= S S' |
        let repeat = |this: &mut Self, rule: &mut Vec<El>, last_sym_start: usize, n_prev_rules: &mut u64,
                      min: u64, max: u64, p: usize| -> R<()> {
            let no_max = max == u64::MAX;
            if last_sym_start == rule.len() { return Err(this.err("expecting preceding item to */+/?/{", p)); }
            let prev: Vec<El> = rule[last_sym_start..].to_vec();
            let total = if !no_max && max > 0 { max } else if min > 0 { min } else { 1 };
            if n_prev_rules.saturating_mul(total) > MAX_REPETITION_THRESHOLD {
                return Err("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults".into());
            }
            if min == 0 { rule.truncate(last_sym_start); } else { for _ in 1..min { rule.extend_from_slice(&prev); } }
            let mut last_rec = 0u32;
            let n_opt = if no_max { 1 } else { max - min };
            for i in 0..n_opt {
                let mut rec = prev.clone();
                let id = this.generate_symbol_id(rule_name);
                if i > 0 || no_max { rec.push(el(Ty::RuleRef, if no_max { id } else { last_rec })); }
                rec.push(el(Ty::Alt, 0));
                rec.push(el(Ty::End, 0));
                this.add_rule(id, rec);
                last_rec = id;
            }
            if n_opt > 0 { rule.push(el(Ty::RuleRef, last_rec)); }
            *n_prev_rules *= total;
            Ok(())
        };
        while self.at(p) != 0 {
            let c = self.at(p);
            if c == b'"' {
                p += 1;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while self.at(p) != b'"' {
                    if self.at(p) == 0 { return Err("unexpected end of input".into()); }
                    let (v, q) = self.char_(p)?;
                    p = q;
                    rule.push(el(Ty::Char, v));
                }
                p = self.space(p + 1, nested);
            } else if c == b'[' {
                p += 1;
                let mut start = Ty::Char;
                if self.at(p) == b'^' { p += 1; start = Ty::CharNot; }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while self.at(p) != b']' {
                    if self.at(p) == 0 { return Err("unexpected end of input".into()); }
                    let (v, q) = self.char_(p)?;
                    p = q;
                    let ty = if last_sym_start < rule.len() { Ty::CharAlt } else { start };
                    rule.push(el(ty, v));
                    if self.at(p) == b'-' && self.at(p + 1) != b']' {
                        if self.at(p + 1) == 0 { return Err("unexpected end of input".into()); }
                        let (e, q) = self.char_(p + 1)?;
                        p = q;
                        rule.push(el(Ty::CharRngUpper, e));
                    }
                }
                p = self.space(p + 1, nested);
            } else if c == b'<' || c == b'!' {
                let mut ty = Ty::Token;
                if c == b'!' { ty = Ty::TokenNot; p += 1; }
                let (id, e) = self.token(p)?;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(el(ty, id));
                p = self.space(e, nested);
            } else if is_word(c) {
                let e = self.name_end(p)?;
                let name = String::from_utf8_lossy(&self.s[p..e]).into_owned();
                let id = self.symbol_id(&name);
                p = self.space(e, nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(el(Ty::RuleRef, id));
            } else if c == b'(' {
                p = self.space(p + 1, true);
                let before = self.symbols.len() as u64;
                let sub = self.generate_symbol_id(rule_name);
                p = self.alternates(p, rule_name, sub, true)?;
                n_prev_rules = 1.max(self.symbols.len() as u64 - before);
                last_sym_start = rule.len();
                rule.push(el(Ty::RuleRef, sub));
                if self.at(p) != b')' { return Err(self.err("expecting ')'", p)); }
                p = self.space(p + 1, nested);
            } else if c == b'.' {
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(el(Ty::CharAny, 0));
                p = self.space(p + 1, nested);
            } else if c == b'*' {
                p = self.space(p + 1, nested);
                repeat(self, rule, last_sym_start, &mut n_prev_rules, 0, u64::MAX, p)?;
            } else if c == b'+' {
                p = self.space(p + 1, nested);
                repeat(self, rule, last_sym_start, &mut n_prev_rules, 1, u64::MAX, p)?;
            } else if c == b'?' {
                p = self.space(p + 1, nested);
                repeat(self, rule, last_sym_start, &mut n_prev_rules, 0, 1, p)?;
            } else if c == b'{' {
                p = self.space(p + 1, nested);
                if !self.at(p).is_ascii_digit() { return Err(self.err("expecting an int", p)); }
                let e = self.int_end(p)?;
                let min: u64 = std::str::from_utf8(&self.s[p..e]).unwrap().parse().map_err(|_| self.err("bad int", p))?;
                p = self.space(e, nested);
                let mut max = u64::MAX;
                if self.at(p) == b'}' {
                    max = min;
                    p = self.space(p + 1, nested);
                } else if self.at(p) == b',' {
                    p = self.space(p + 1, nested);
                    if self.at(p).is_ascii_digit() {
                        let e = self.int_end(p)?;
                        max = std::str::from_utf8(&self.s[p..e]).unwrap().parse().map_err(|_| self.err("bad int", p))?;
                        p = self.space(e, nested);
                    }
                    if self.at(p) != b'}' { return Err(self.err("expecting '}'", p)); }
                    p = self.space(p + 1, nested);
                } else {
                    return Err(self.err("expecting ','", p));
                }
                if min > MAX_REPETITION_THRESHOLD { return Err("number of repetitions exceeds sane defaults".into()); }
                // Not in llama.cpp: there `{m,n}` with n < m makes `max - min` wrap, and the loop that adds
                // the optional copies then runs ~2^64 times, allocating a rule each time. A JSON schema with
                // minItems > maxItems (or minLength > maxLength) converts to exactly that.
                if max < min { return Err(format!("repetition {{{min},{max}}}: the maximum is below the minimum")); }
                if max != u64::MAX && max > MAX_REPETITION_THRESHOLD { max = u64::MAX; }
                repeat(self, rule, last_sym_start, &mut n_prev_rules, min, max, p)?;
            } else {
                break;
            }
        }
        Ok(p)
    }
    fn alternates(&mut self, src: usize, rule_name: &str, rule_id: u32, nested: bool) -> R<usize> {
        let mut rule = Vec::new();
        let mut p = self.sequence(src, rule_name, &mut rule, nested)?;
        while self.at(p) == b'|' {
            rule.push(el(Ty::Alt, 0));
            p = self.space(p + 1, true);
            p = self.sequence(p, rule_name, &mut rule, nested)?;
        }
        rule.push(el(Ty::End, 0));
        self.add_rule(rule_id, rule);
        Ok(p)
    }
    fn rule(&mut self, src: usize) -> R<usize> {
        let e = self.name_end(src)?;
        let mut p = self.space(e, false);
        let name = String::from_utf8_lossy(&self.s[src..e]).into_owned();
        let id = self.symbol_id(&name);
        if !(self.at(p) == b':' && self.at(p + 1) == b':' && self.at(p + 2) == b'=') { return Err(self.err("expecting ::=", p)); }
        p = self.space(p + 3, true);
        p = self.alternates(p, &name, id, false)?;
        match self.at(p) {
            b'\r' => p += if self.at(p + 1) == b'\n' { 2 } else { 1 },
            b'\n' => p += 1,
            0 => {}
            _ => return Err(self.err("expecting newline or end", p)),
        }
        Ok(self.space(p, true))
    }
}

fn is_end(e: &El) -> bool { matches!(e.ty, Ty::End | Ty::Alt) }

fn left_recursion(rules: &[Vec<El>], i: usize, visited: &mut [bool], in_progress: &mut [bool], may_be_empty: &mut [bool]) -> bool {
    if in_progress[i] { return true; }
    in_progress[i] = true;
    let rule = &rules[i];
    let mut at_start = true;
    for e in rule {
        if is_end(e) {
            if at_start { may_be_empty[i] = true; break; }
            at_start = true;
        } else { at_start = false; }
    }
    let mut recurse = true;
    for e in rule {
        if e.ty == Ty::RuleRef && recurse {
            if left_recursion(rules, e.v as usize, visited, in_progress, may_be_empty) { return true; }
            if !may_be_empty[e.v as usize] { recurse = false; }
        } else if is_end(e) { recurse = true; } else { recurse = false; }
    }
    in_progress[i] = false;
    visited[i] = true;
    false
}

impl Grammar {
    /// Parse `src` with start symbol `root` (llama.cpp's `llama_grammar_init_impl`: parse, every referenced
    /// rule defined, the root present, no left recursion). `token_of` maps a `<text>` token element to its
    /// id (a whole vocabulary token), as llama.cpp tokenizes it.
    pub fn parse(src: &str, root: &str, token_of: &dyn Fn(&str) -> Option<u32>) -> Result<Grammar, String> {
        let mut ps = Parser { s: src.as_bytes(), rules: Vec::new(), symbols: HashMap::new(), token_of };
        let mut p = ps.space(0, true);
        while ps.at(p) != 0 { p = ps.rule(p)?; }
        for rule in &ps.rules {
            if rule.is_empty() { return Err("Undefined rule".into()); }
            for e in rule {
                if e.ty == Ty::RuleRef && (e.v as usize >= ps.rules.len() || ps.rules[e.v as usize].is_empty()) {
                    let name = ps.symbols.iter().find(|(_, v)| **v == e.v).map(|(k, _)| k.clone()).unwrap_or_default();
                    return Err(format!("Undefined rule identifier '{name}'"));
                }
            }
        }
        let root_id = *ps.symbols.get(root).ok_or_else(|| format!("grammar does not contain a {root:?} symbol"))? as usize;
        let n = ps.rules.len();
        let (mut v, mut ip, mut me) = (vec![false; n], vec![false; n], vec![false; n]);
        for i in 0..n {
            if v[i] { continue; }
            if left_recursion(&ps.rules, i, &mut v, &mut ip, &mut me) {
                let name = ps.symbols.iter().find(|(_, x)| **x as usize == i).map(|(k, _)| k.clone()).unwrap_or_default();
                return Err(format!("unsupported grammar, left recursion detected for nonterminal at index {i} ({name})"));
            }
        }
        Ok(Grammar { rules: ps.rules, root: root_id, symbols: ps.symbols })
    }
}

/// A position in a rule: (rule id, element index). A stack's LAST entry is the next element to match.
pub type Pos = (u32, u32);
pub type Stack = Vec<Pos>;

/// Where matching stands: every live parse stack, and a code point begun but not finished by the bytes so
/// far (`n_remain` = continuation bytes still owed; -1 = invalid UTF-8 seen).
#[derive(Clone, Debug)]
pub struct Matcher { g: Arc<Grammar>, stacks: Vec<Stack>, partial: (u32, i8) }

impl Matcher {
    pub fn new(g: Arc<Grammar>) -> Matcher {
        let mut stacks = Vec::new();
        let r = g.root as u32;
        let rule = &g.rules[g.root];
        let mut i = 0usize;
        loop {
            let mut st = Vec::new();
            if !is_end(&rule[i]) { st.push((r, i as u32)); }
            advance(&g, st, &mut stacks);
            while !is_end(&rule[i]) { i += 1; }
            if rule[i].ty == Ty::Alt { i += 1; } else { break; }
        }
        Matcher { g, stacks, partial: (0, 0) }
    }
    fn at(&self, p: Pos) -> El { self.g.rules[p.0 as usize][p.1 as usize] }

    /// A matcher over exactly these stacks (each already advanced to a terminal, or empty), no character
    /// half-written. Stepping bytes treats every stack independently, so a state's mask is the union of its
    /// stacks' masks — which is what lets a mask be cached per stack TOP (`ferric-serve` constrain.rs).
    pub fn from_stacks(g: Arc<Grammar>, stacks: Vec<Stack>) -> Matcher { Matcher { g, stacks, partial: (0, 0) } }
    /// The state of a parse whose stack is `st`, its top not yet expanded to terminals — what matches next
    /// once everything that was above `st` has finished.
    pub fn resume(g: Arc<Grammar>, st: Stack) -> Matcher {
        let mut stacks = Vec::new();
        advance(&g, st, &mut stacks);
        Matcher { g, stacks, partial: (0, 0) }
    }
    pub fn grammar(&self) -> &Arc<Grammar> { &self.g }
    pub fn stacks(&self) -> &[Stack] { &self.stacks }
    /// A code point is begun and not finished.
    pub fn mid_char(&self) -> bool { self.partial.1 != 0 }
    /// Some stack is empty: a parse reached the end of everything these stacks hold. Run from a stack's top
    /// alone, that is the point where the stack below would decide what comes next.
    pub fn reached_bottom(&self) -> bool { self.stacks.iter().any(|s| s.is_empty()) }
    /// Whether the element at `p` is a whole-token element (`<[id]>`, `!<[id]>`), which text never matches.
    pub fn is_token_el(&self, p: Pos) -> bool { matches!(self.at(p).ty, Ty::Token | Ty::TokenNot) }
    /// Every parse is complete (an empty stack exists) and no character is half-written: EOS is legal.
    pub fn can_stop(&self) -> bool { self.partial.1 == 0 && self.stacks.iter().any(|s| s.is_empty()) }
    /// No stack left: the text so far is not a prefix of any sentence of the grammar.
    pub fn dead(&self) -> bool { self.stacks.is_empty() }

    fn accept_chr(&self, chr: u32, out: &mut Vec<Stack>) {
        for st in &self.stacks {
            let Some(&top) = st.last() else { continue };
            let e = self.at(top);
            if matches!(e.ty, Ty::Token | Ty::TokenNot) { continue; }
            let (ok, next) = match_char(&self.g, top, chr);
            if ok {
                let mut ns: Stack = st[..st.len() - 1].to_vec();
                if !is_end(&self.at(next)) { ns.push(next); }
                advance(&self.g, ns, out);
            }
        }
    }

    /// Feed one byte. `false` = no parse survives (or a character that can no longer be completed legally);
    /// the matcher is then dead.
    pub fn step(&mut self, b: u8) -> bool {
        const LOOKUP: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
        let (mut value, mut n_remain) = self.partial;
        if n_remain < 0 { self.stacks.clear(); return false; }
        if n_remain > 0 {
            if (b >> 6) != 2 { self.stacks.clear(); self.partial = (0, -1); return false; }
            value = (value << 6) + (b & 0x3F) as u32;
            n_remain -= 1;
        } else {
            let n = LOOKUP[(b >> 4) as usize] - 1;
            if n < 0 { self.stacks.clear(); self.partial = (0, -1); return false; }
            let mask = (1u8 << (7 - n)) - 1;
            value = (b & mask) as u32;
            n_remain = n;
        }
        if n_remain == 0 {
            let mut out = Vec::with_capacity(self.stacks.len());
            self.accept_chr(value, &mut out);
            self.stacks = out;
            self.partial = (0, 0);
        } else {
            self.partial = (value, n_remain);
            // llama_grammar_reject_candidates: a token ending mid-character survives only if some stack
            // could still take a character this prefix can complete to.
            let alive = self.stacks.iter().any(|st| st.last().is_some_and(|&top| {
                let e = self.at(top);
                !matches!(e.ty, Ty::Token | Ty::TokenNot) && match_partial_char(&self.g, top, value, n_remain)
            }));
            if !alive { self.stacks.clear(); }
        }
        !self.stacks.is_empty()
    }

    /// Feed one whole token that a TOKEN / TOKEN_NOT element may consume: stacks at a token element match
    /// its id; the rest match its text code point by code point (`llama_grammar_accept_token`).
    pub fn step_token(&mut self, id: u32, bytes: &[u8]) -> bool {
        let mut out: Vec<Stack> = Vec::new();
        let mut text = self.clone();
        let mut text_ok = !bytes.is_empty();
        // Stacks sitting at a token element are removed from the text path: a token element never matches
        // characters.
        text.stacks.retain(|st| st.last().is_none_or(|&top| !matches!(self.at(top).ty, Ty::Token | Ty::TokenNot)));
        text.stacks.retain(|st| !st.is_empty());
        for &b in bytes { if !text.step(b) { text_ok = false; break; } }
        for st in &self.stacks {
            let Some(&top) = st.last() else { continue };
            let e = self.at(top);
            let hit = match e.ty { Ty::Token => e.v == id, Ty::TokenNot => e.v != id, _ => continue };
            if hit {
                let mut ns: Stack = st[..st.len() - 1].to_vec();
                let next = (top.0, top.1 + 1);
                if !is_end(&self.at(next)) { ns.push(next); }
                advance(&self.g, ns, &mut out);
            }
        }
        if text_ok {
            for s in text.stacks { if !out.contains(&s) { out.push(s); } }
            self.partial = text.partial;
        } else if !out.is_empty() {
            self.partial = (0, 0);
        }
        self.stacks = out;
        !self.stacks.is_empty()
    }

    /// Whether a stack at a token element takes token `id` whole: `<[id]>`, or `!<[x]>` with x != id.
    pub fn token_allows(&self, id: u32) -> bool {
        self.stacks.iter().any(|st| st.last().is_some_and(|&t| {
            let e = self.at(t);
            match e.ty { Ty::Token => e.v == id, Ty::TokenNot => e.v != id, _ => false }
        }))
    }

    /// Whether any live stack sits at a token element (then a special token may be legal).
    pub fn wants_token(&self) -> bool {
        self.stacks.iter().any(|st| st.last().is_some_and(|&t| matches!(self.at(t).ty, Ty::Token | Ty::TokenNot)))
    }
}

fn match_char(g: &Grammar, mut p: Pos, chr: u32) -> (bool, Pos) {
    let rule = &g.rules[p.0 as usize];
    let first = rule[p.1 as usize];
    let positive = matches!(first.ty, Ty::Char | Ty::CharAny);
    let mut found = false;
    loop {
        let cur = rule[p.1 as usize];
        let nxt = rule.get(p.1 as usize + 1).copied().unwrap_or(el(Ty::End, 0));
        if nxt.ty == Ty::CharRngUpper {
            found = found || (cur.v <= chr && chr <= nxt.v);
            p.1 += 2;
        } else if cur.ty == Ty::CharAny {
            found = true;
            p.1 += 1;
        } else {
            found = found || cur.v == chr;
            p.1 += 1;
        }
        if rule[p.1 as usize].ty != Ty::CharAlt { break; }
    }
    (found == positive, p)
}

fn match_partial_char(g: &Grammar, mut p: Pos, value: u32, n_remain: i8) -> bool {
    let rule = &g.rules[p.0 as usize];
    let positive = matches!(rule[p.1 as usize].ty, Ty::Char | Ty::CharAny);
    if n_remain < 0 || (n_remain == 1 && value < 2) { return false; }
    let n = n_remain as u32;
    let mut low = value << (n * 6);
    let high = low | ((1u32 << (n * 6)) - 1);
    if low == 0 { if n == 2 { low = 1 << 11; } else if n == 3 { low = 1 << 16; } }
    loop {
        let cur = rule[p.1 as usize];
        let nxt = rule.get(p.1 as usize + 1).copied().unwrap_or(el(Ty::End, 0));
        if nxt.ty == Ty::CharRngUpper {
            if cur.v <= high && low <= nxt.v { return positive; }
            p.1 += 2;
        } else if cur.ty == Ty::CharAny {
            return true;
        } else {
            if low <= cur.v && cur.v <= high { return positive; }
            p.1 += 1;
        }
        if rule[p.1 as usize].ty != Ty::CharAlt { break; }
    }
    !positive
}

/// `llama_grammar_advance_stack`: expand rule references at the top until every resulting stack sits at a
/// terminal (character or token element) or is empty (a complete parse). Deduplicated, as llama.cpp does.
fn advance(g: &Grammar, stack: Stack, out: &mut Vec<Stack>) {
    let mut todo = vec![stack];
    let mut seen: BTreeSet<Stack> = BTreeSet::new();
    while let Some(cur) = todo.pop() {
        if !seen.insert(cur.clone()) { continue; }
        let Some(&top) = cur.last() else {
            if !out.contains(&cur) { out.push(cur); }
            continue;
        };
        let e = g.rules[top.0 as usize][top.1 as usize];
        match e.ty {
            Ty::RuleRef => {
                let sub = &g.rules[e.v as usize];
                let mut i = 0usize;
                loop {
                    let mut next: Stack = cur[..cur.len() - 1].to_vec();
                    let after = (top.0, top.1 + 1);
                    if !is_end(&g.rules[after.0 as usize][after.1 as usize]) { next.push(after); }
                    if !is_end(&sub[i]) { next.push((e.v, i as u32)); }
                    todo.push(next);
                    while !is_end(&sub[i]) { i += 1; }
                    if sub[i].ty == Ty::Alt { i += 1; } else { break; }
                }
            }
            Ty::Char | Ty::CharNot | Ty::CharAny | Ty::Token | Ty::TokenNot => {
                if !out.contains(&cur) { out.push(cur); }
            }
            _ => unreachable!("a stack never rests on {:?}", e.ty),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Feed a test string as llama.cpp's harness does: a `token` segment is a whole token; each code point
    /// of a `text` segment is a token of id 0 whose piece is that code point.
    fn matches(g: &Arc<Grammar>, segs: &[Value]) -> bool {
        let mut m = Matcher::new(g.clone());
        for s in segs {
            if let Some(id) = s["token"].as_u64() {
                if !m.step_token(id as u32, &[]) { return false; }
            } else {
                for ch in s["text"].as_str().unwrap().chars() {
                    let mut b = [0u8; 4];
                    if !m.step_token(0, ch.encode_utf8(&mut b).as_bytes()) { return false; }
                }
            }
        }
        m.can_stop()
    }

    /// llama.cpp's own tests/test-grammar-integration.cpp, case for case: every passing string matches,
    /// every failing string does not, every grammar that must fail to build fails.
    #[test]
    fn llama_cpp_grammar_integration_cases() {
        let fx: Value = serde_json::from_str(include_str!("../tests/fixtures/grammar/llama_cpp_integration.json")).unwrap();
        let none = |_: &str| None;
        let (mut n_pass, mut n_fail) = (0, 0);
        for c in fx["cases"].as_array().unwrap() {
            let desc = c["desc"].as_str().unwrap();
            let g = Arc::new(Grammar::parse(c["grammar"].as_str().unwrap(), "root", &none).unwrap_or_else(|e| panic!("{desc}: {e}")));
            for s in c["pass"].as_array().unwrap() { assert!(matches(&g, s.as_array().unwrap()), "{desc}: should match {s}"); n_pass += 1; }
            for s in c["fail"].as_array().unwrap() { assert!(!matches(&g, s.as_array().unwrap()), "{desc}: should NOT match {s}"); n_fail += 1; }
        }
        for f in fx["build_fails"].as_array().unwrap() {
            let r = Grammar::parse(f["grammar"].as_str().unwrap(), f["root"].as_str().unwrap(), &none);
            assert_eq!(r.is_ok(), f["builds"].as_bool().unwrap(), "{}: {:?}", f["test"], r.err());
        }
        assert_eq!((n_pass, n_fail), (70, 69), "the fixture changed size");
    }

    /// `{m,n}` with n < m is refused (llama.cpp wraps `n - m` and loops ~2^64 times, allocating), as is the
    /// grammar a JSON schema with minItems > maxItems converts to; n == m and n > m still build.
    #[test]
    fn a_repetition_whose_maximum_is_below_its_minimum_is_refused() {
        let none = |_: &str| None;
        let e = Grammar::parse("root ::= \"a\"{5,2}", "root", &none).unwrap_err();
        assert!(e.contains("{5,2}"), "{e}");
        assert!(Grammar::parse("root ::= \"a\"{2,2} \"b\"{2,5} \"c\"{0,0}", "root", &none).is_ok());
        let src = crate::json_schema::schema_to_gbnf(&serde_json::json!({"type": "array", "items": {"type": "integer"}, "minItems": 5, "maxItems": 2})).unwrap();
        assert!(Grammar::parse(&src, "root", &none).unwrap_err().contains("{4,1}"), "{src}");
    }

    #[test]
    fn a_token_ending_mid_character_survives_only_if_it_can_complete_legally() {
        let g = Arc::new(Grammar::parse("root ::= [é-ë]+", "root", &|_| None).unwrap());
        let mut m = Matcher::new(g.clone());
        assert!(m.step(0xC3), "é..ë all start with 0xC3");
        assert!(m.step(0xA9) && m.can_stop(), "é");
        let mut m = Matcher::new(g);
        assert!(!m.step(0xE2), "a three-byte lead cannot complete into é..ë");
    }
}
