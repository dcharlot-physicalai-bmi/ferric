//! **Constrained decoding: which constraint a request asks for, and the token mask it implies.**
//!
//! One reader for every spelling peers accept (parity gap S10, 8.5 of 14 serving peers):
//! - OpenAI `response_format` — `json_object`, `json_schema` (the existing JSON / schema guides);
//! - llama-server `grammar` — GBNF;
//! - vLLM `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` (GBNF), and its newer
//!   `structured_outputs: {json | regex | choice | grammar}`; a bare `regex` too.
//! Regex and choice compile to GBNF (`ferric_agent::regex`), so every grammar-shaped constraint runs on the
//! one matcher that is checked against llama.cpp's own grammar tests. Two constraints in one request are a
//! 400 naming both — never one silently ignored.
//!
//! **The mask.** A constraint decides, per step, which tokens may come next. The first version trial-stepped
//! the constraint through every token's bytes, copying its state per token — fine for a JSON state that is a
//! few bytes, not for a grammar's parse stacks over a 152k vocabulary. Tokens share prefixes, so the mask now
//! walks a byte TRIE of the vocabulary: the state is cloned once per trie node that is still alive, and a
//! dead prefix prunes every token under it. Same rule, fewer steps: a token is allowed iff all its bytes step
//! (or, for a grammar waiting on a whole token, iff that token element takes its id); EOS iff the constraint
//! can stop.
//!
//! **A grammar's mask, cached** (after XGrammar's adaptive token-mask cache, Dong et al. 2024,
//! arXiv:2411.15100). Stepping bytes treats each parse stack on its own, so a state's mask is the union of
//! its stacks' masks. For one stack, most tokens are decided by its top few frames alone: walked from a
//! SUFFIX of the stack, a token whose bytes all match is allowed whatever lies below, and one that fails
//! before the suffix is used up is refused whatever lies below. Only a token that runs past the bottom of
//! the suffix (at byte offset o) depends on the rest — and then only through its remainder `token[o..]`,
//! which must match from the state below. A suffix's walk is cached (allowed bitset + each depending
//! token's bottom offsets). The suffix is as short as keeps the depending set small: one frame is enough
//! for a keyword, but a character class inside a group inside a repetition finishes its rule after one
//! character, so a string needs three. A whole stack's mask is cached too, since the same stacks recur
//! step after step. Same answer as the full walk, token for token (`the_cached_mask_equals_the_full_walk`);
//! the cache lives with the grammar's source, so a repeated grammar starts warm.
use crate::Engine;
use ferric_agent::grammar::{Grammar, Matcher, Pos};
use ferric_agent::guide::{Guide, Item, Json, Schema};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What a request constrains its output to, owned (a schema guide borrows its compiled program).
pub(crate) enum Spec { None, Json, Schema(Vec<Item>), Grammar(Arc<Grammar>) }

impl Spec {
    pub fn guide(&self) -> Option<Guide<'_>> {
        match self {
            Spec::None => None,
            Spec::Json => Some(Guide::Json(Json::object())),
            Spec::Schema(p) => Some(Guide::Schema(Schema::new(p))),
            Spec::Grammar(g) => Some(Guide::Grammar(Matcher::new(g.clone()))),
        }
    }
    pub fn is_some(&self) -> bool { !matches!(self, Spec::None) }
}

/// The request fields that carry a constraint (for routing a request to the serial path).
pub(crate) fn asks_for_constraint(req: &Value) -> bool {
    ["grammar", "guided_grammar", "guided_regex", "regex", "guided_choice", "guided_json", "structured_outputs"]
        .iter().any(|k| !req[*k].is_null())
        || !req["response_format"]["type"].is_null()
}

/// Read the request's constraint. `compile` turns GBNF source (start symbol `root`) into a grammar.
pub(crate) fn spec_of(req: &Value, compile: &dyn Fn(&str) -> Result<Arc<Grammar>, String>) -> Result<Spec, String> {
    let so = &req["structured_outputs"];
    let mut found: Vec<(&str, Spec)> = Vec::new();
    let grammar = |src: &str, what: &str| -> Result<Spec, String> {
        compile(src).map(Spec::Grammar).map_err(|e| format!("{what}: {e}"))
    };
    let schema = |s: &Value, what: &str| -> Result<Spec, String> {
        let s = if let Some(t) = s.as_str() { serde_json::from_str::<Value>(t).map_err(|e| format!("{what}: {e}"))? } else { s.clone() };
        Ok(ferric_agent::guide::compile(&s).map(Spec::Schema).unwrap_or(Spec::Json))
    };
    let text = |v: &Value, what: &str| -> Result<String, String> { v.as_str().map(String::from).ok_or_else(|| format!("{what} must be a string")) };
    match req["response_format"]["type"].as_str() {
        None | Some("text") => {}
        Some("json_object") => found.push(("response_format", Spec::Json)),
        Some("json_schema") => found.push(("response_format", schema(&req["response_format"]["json_schema"]["schema"], "response_format.json_schema.schema")?)),
        Some(t) => return Err(format!("response_format.type {t:?}: text, json_object and json_schema are served")),
    }
    for (k, v) in [("grammar", &req["grammar"]), ("guided_grammar", &req["guided_grammar"]), ("structured_outputs.grammar", &so["grammar"])] {
        if !v.is_null() { found.push((k, grammar(&text(v, k)?, k)?)); }
    }
    for (k, v) in [("guided_regex", &req["guided_regex"]), ("regex", &req["regex"]), ("structured_outputs.regex", &so["regex"])] {
        if !v.is_null() { found.push((k, grammar(&ferric_agent::regex::regex_to_gbnf(&text(v, k)?)?, k)?)); }
    }
    for (k, v) in [("guided_choice", &req["guided_choice"]), ("structured_outputs.choice", &so["choice"])] {
        if !v.is_null() {
            let c: Vec<String> = v.as_array().ok_or_else(|| format!("{k} must be an array of strings"))?
                .iter().map(|x| x.as_str().map(String::from).ok_or_else(|| format!("{k} must be an array of strings"))).collect::<Result<_, _>>()?;
            found.push((k, grammar(&ferric_agent::regex::choice_to_gbnf(&c)?, k)?));
        }
    }
    for (k, v) in [("guided_json", &req["guided_json"]), ("structured_outputs.json", &so["json"])] {
        if !v.is_null() { found.push((k, schema(v, k)?)); }
    }
    match found.len() {
        0 => Ok(Spec::None),
        1 => Ok(found.pop().unwrap().1),
        _ => Err(format!("one constraint per request; this one has {}", found.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(" and "))),
    }
}

/// The vocabulary as a byte trie: every token with non-empty text, at the node its bytes spell.
pub(crate) struct Trie { children: Vec<Vec<(u8, u32)>>, ends: Vec<Vec<u32>> }

impl Trie {
    pub fn new(token_bytes: &[Option<Vec<u8>>]) -> Trie {
        let mut t = Trie { children: vec![Vec::new()], ends: vec![Vec::new()] };
        for (id, b) in token_bytes.iter().enumerate() {
            let Some(b) = b.as_ref().filter(|b| !b.is_empty()) else { continue };
            let mut n = 0usize;
            for &c in b {
                n = match t.children[n].iter().find(|(x, _)| *x == c) {
                    Some(&(_, k)) => k as usize,
                    None => {
                        let k = t.children.len();
                        t.children.push(Vec::new());
                        t.ends.push(Vec::new());
                        t.children[n].push((c, k as u32));
                        k
                    }
                };
            }
            t.ends[n].push(id as u32);
        }
        t
    }

    /// Which tokens `g` allows next (EOS handled by the caller).
    pub fn allowed(&self, g: &Guide, n_vocab: usize) -> Vec<bool> {
        let mut ok = vec![false; n_vocab];
        let mut todo: Vec<(u32, Guide)> = vec![(0, g.clone())];
        while let Some((node, st)) = todo.pop() {
            for &(b, child) in &self.children[node as usize] {
                let mut s2 = st.clone();
                if !s2.step(b) { continue; }
                for &t in &self.ends[child as usize] { if (t as usize) < n_vocab { ok[t as usize] = true; } }
                if !self.children[child as usize].is_empty() { todo.push((child, s2)); }
            }
        }
        if g.wants_token() {
            for (i, x) in ok.iter_mut().enumerate() { if !*x && g.token_allowed(i as u32) { *x = true; } }
        }
        ok
    }

    /// One walk from the stack suffix `suf` alone: which tokens match whatever lies below it, and which run
    /// past its bottom (so depend on what does), with the offsets where they did. A token is recorded as
    /// depending only when no path matched all its bytes — a path that did is a path the full stack has too.
    fn suffix_mask(&self, g: &Arc<Grammar>, suf: &[Pos], n_vocab: usize) -> SuffixMask {
        let mut accept = vec![0u64; n_vocab.div_ceil(64)];
        let (mut depends, mut offs) = (Vec::new(), Vec::new());
        // (trie node, state from the suffix alone, byte depth, offsets on this path where the bottom was reached)
        let mut todo = vec![(0u32, Matcher::from_stacks(g.clone(), vec![suf.to_vec()]), 0u16, Vec::<u16>::new())];
        while let Some((node, st, depth, bottoms)) = todo.pop() {
            for &(b, child) in &self.children[node as usize] {
                let mut s2 = st.clone();
                let ok = s2.step(b);
                let mut bottoms = bottoms.clone();
                if ok && !s2.mid_char() && s2.reached_bottom() { bottoms.push(depth + 1); }
                for &t in &self.ends[child as usize] {
                    if (t as usize) >= n_vocab { continue; }
                    if ok { accept[t as usize / 64] |= 1 << (t % 64); }
                    else if !bottoms.is_empty() {
                        depends.push(t);
                        offs.extend(bottoms.iter().map(|&o| o as u32));
                        offs.push(u32::MAX);
                    }
                }
                if !self.children[child as usize].is_empty() && (ok || !bottoms.is_empty()) { todo.push((child, s2, depth + 1, bottoms)); }
            }
        }
        SuffixMask { accept, depends, offs }
    }

    /// One stack's mask: from the shortest suffix whose depending set is small, then those depending tokens
    /// whose remainder the state below the suffix takes.
    fn stack_mask(&self, g: &Arc<Grammar>, st: &[Pos], x: &Masks, n_vocab: usize, token_bytes: &[Option<Vec<u8>>]) -> Vec<u64> {
        let mut k = 1;
        let sm = loop {
            let sm = x.suffix(self, &st[st.len() - k..], n_vocab);
            if sm.depends.len() <= x.limit || k == st.len() { break sm; }
            k += 1;
        };
        let mut w = sm.accept.clone();
        // The whole stack as the suffix: running past its bottom is the end of the grammar, so every
        // depending token is refused (the full walk refuses it the same way).
        if k == st.len() { return w; }
        let below = Matcher::resume(g.clone(), st[..st.len() - k].to_vec());
        let first: Vec<bool> = (0..=255u8).map(|b| below.clone().step(b)).collect();
        let mut o = sm.offs.split(|&v| v == u32::MAX);
        for &t in &sm.depends {
            let at = o.next().unwrap_or(&[]);
            let Some(b) = token_bytes.get(t as usize).and_then(|b| b.as_deref()) else { continue };
            let takes = at.iter().any(|&i| {
                let rest = &b[i as usize..];
                rest.first().is_none_or(|&c| first[c as usize]) && { let mut s = below.clone(); rest.iter().all(|&c| s.step(c)) }
            });
            if takes { w[t as usize / 64] |= 1 << (t % 64); }
        }
        w
    }

    /// The JSON-object guide's mask, cached by state: its state is a few bytes whose equality means "accepts
    /// the same continuations" (`Json`'s `Eq`), and inside a string or between values the same states recur.
    pub fn allowed_json(&self, j: &Json, cache: &mut HashMap<Json, Vec<u64>>, n_vocab: usize) -> Vec<bool> {
        if !cache.contains_key(j) {
            let ok = self.allowed(&Guide::Json(*j), n_vocab);
            let mut w = vec![0u64; n_vocab.div_ceil(64)];
            for (t, _) in ok.iter().enumerate().filter(|(_, x)| **x) { w[t / 64] |= 1 << (t % 64); }
            if cache.len() >= KEPT_STACKS { cache.clear(); }
            cache.insert(*j, w);
        }
        let w = &cache[j];
        (0..n_vocab).map(|t| w[t / 64] >> (t % 64) & 1 == 1).collect()
    }

    /// The mask of grammar state `m` (no character half-written): the OR of its stacks' masks.
    fn allowed_by_stacks(&self, m: &Matcher, x: &Masks, n_vocab: usize, token_bytes: &[Option<Vec<u8>>]) -> Vec<bool> {
        let mut words = vec![0u64; n_vocab.div_ceil(64)];
        for st in m.stacks() {
            let Some(&top) = st.last() else { continue };
            if m.is_token_el(top) { continue; }
            let cached = x.stacks.lock().unwrap().get(st).cloned();
            if cached.is_some() { x.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
            let w = cached.unwrap_or_else(|| {
                let w = Arc::new(self.stack_mask(m.grammar(), st, x, n_vocab, token_bytes));
                let mut c = x.stacks.lock().unwrap();
                if c.len() >= KEPT_STACKS { c.clear(); }
                c.insert(st.clone(), w.clone());
                w
            });
            for (a, b) in words.iter_mut().zip(w.iter()) { *a |= b; }
        }
        let mut ok: Vec<bool> = (0..n_vocab).map(|t| words[t / 64] >> (t % 64) & 1 == 1).collect();
        if m.wants_token() {
            for (i, x) in ok.iter_mut().enumerate() { if !*x && m.token_allows(i as u32) { *x = true; } }
        }
        ok
    }
}

/// A stack suffix's mask: tokens allowed whatever lies below (a bitset), and the tokens that depend on it,
/// each with the byte offsets where it reached the bottom (`offs`: each token's offsets then a u32::MAX,
/// in `depends` order).
pub(crate) struct SuffixMask { accept: Vec<u64>, depends: Vec<u32>, offs: Vec<u32> }

/// A grammar and the masks learned for it so far: per stack suffix, and per whole stack. (Mutexes only so
/// an Engine can move threads.) `limit`: the most depending tokens a suffix may leave before a longer one
/// is tried.
pub(crate) struct Masks {
    g: Arc<Grammar>,
    suffixes: Mutex<HashMap<Vec<Pos>, Arc<SuffixMask>>>,
    stacks: Mutex<HashMap<Vec<Pos>, Arc<Vec<u64>>>>,
    limit: usize,
    /// Whole-stack masks served from the cache.
    hits: std::sync::atomic::AtomicUsize,
}

impl Masks {
    pub fn new(g: Arc<Grammar>, n_vocab: usize) -> Masks {
        Masks { g, suffixes: Mutex::new(HashMap::new()), stacks: Mutex::new(HashMap::new()), limit: (n_vocab / 64).max(256), hits: Default::default() }
    }
    fn suffix(&self, t: &Trie, suf: &[Pos], n_vocab: usize) -> Arc<SuffixMask> {
        if let Some(x) = self.suffixes.lock().unwrap().get(suf) { return x.clone(); }
        let x = Arc::new(t.suffix_mask(&self.g, suf, n_vocab));
        self.suffixes.lock().unwrap().insert(suf.to_vec(), x.clone());
        x
    }
}

/// Whole-stack masks kept per grammar before the table is cleared (each is n_vocab bits).
const KEPT_STACKS: usize = 512;
/// Grammars kept with their masks, by source text (most recent last).
const KEPT_GRAMMARS: usize = 8;

impl Engine {
    /// A GBNF `<text>` token element: a special token spelled exactly so, else text that encodes to one token.
    pub(crate) fn token_of(&self, text: &str) -> Option<u32> {
        if let Some((_, id)) = self.specials.iter().find(|(t, _)| t == text) { return Some(*id); }
        let ids = self.enc(text, false);
        (ids.len() == 1).then(|| ids[0])
    }

    /// The request's constraint, resolved against this model's vocabulary.
    pub(crate) fn constraint(&self, req: &Value) -> Result<Spec, String> { spec_of(req, &|src| self.grammar(src)) }

    /// A grammar from GBNF source — the one already compiled from the same source when it is kept, so its
    /// learned masks come with it.
    fn grammar(&self, src: &str) -> Result<Arc<Grammar>, String> {
        let mut gs = self.grammars.borrow_mut();
        if let Some(i) = gs.iter().position(|(s, _)| s == src) {
            let e = gs.remove(i);
            let g = e.1.g.clone();
            gs.push(e);
            return Ok(g);
        }
        let g = Arc::new(Grammar::parse(src, "root", &|t| self.token_of(t))?);
        if gs.len() >= KEPT_GRAMMARS { gs.remove(0); }
        gs.push((src.to_string(), Arc::new(Masks::new(g.clone(), self.model.n_vocab()))));
        Ok(g)
    }

    /// The allowed-token mask for `g` at this step (EOS by `can_stop`), from the vocabulary trie — through
    /// the mask cache for a kept grammar between characters, else by the full walk.
    pub(crate) fn allowed(&self, g: &Guide) -> Vec<bool> {
        let t = self.trie.get_or_init(|| Trie::new(&self.token_bytes));
        let n = self.model.n_vocab();
        let kept = match g {
            Guide::Grammar(m) if !m.mid_char() =>
                self.grammars.borrow().iter().find(|(_, x)| Arc::ptr_eq(&x.g, m.grammar())).map(|(_, x)| x.clone()),
            _ => None,
        };
        let mut ok = match (g, kept) {
            (Guide::Grammar(m), Some(x)) => t.allowed_by_stacks(m, &x, n, &self.token_bytes),
            (Guide::Json(j), _) => t.allowed_json(j, &mut self.json_masks.borrow_mut(), n),
            _ => t.allowed(g, n),
        };
        let stop = g.can_stop();
        for &e in &self.eos { if (e as usize) < ok.len() { ok[e as usize] = stop; } }
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The trie mask equals trial-stepping every token (the rule it replaces), for JSON and a grammar.
    #[test]
    fn the_trie_mask_equals_stepping_every_token() {
        let vocab: Vec<Option<Vec<u8>>> = ["{", "\"", "a", "ab", "abc", "}", "{\"", ":", "1", "12", " ", "b", "x", "\"a", "yes", "no", "ye"]
            .iter().map(|s| Some(s.as_bytes().to_vec())).chain([None]).collect();
        let t = Trie::new(&vocab);
        let g = Arc::new(Grammar::parse("root ::= (\"yes\" | \"no\" | \"ab\"+ \"c\")", "root", &|_| None).unwrap());
        for guide in [Guide::Json(Json::object()), Guide::Grammar(Matcher::new(g))] {
            let mut state = guide.clone();
            for step in 0..3 {
                let want: Vec<bool> = vocab.iter().map(|b| b.as_ref().is_some_and(|b| { let mut a = state.clone(); b.iter().all(|&c| a.step(c)) })).collect();
                assert_eq!(t.allowed(&state, vocab.len()), want, "step {step}");
                let first = want.iter().position(|&x| x).unwrap();
                state.commit(first as u32, vocab[first].as_deref());
            }
        }
    }

    /// A vocabulary shaped like a byte-level BPE one: every single byte, then every 2-6 byte substring of
    /// sample text in each grammar's language (so tokens cross rule boundaries, as real ones do), including
    /// pieces that cut a multi-byte character.
    fn test_vocab() -> Vec<Option<Vec<u8>>> {
        let samples = [
            "{\"name\": \"Ada\", \"age\": 36, \"tags\": [\"x\", \"y\\n\"], \"ok\": true, \"v\": -1.5e3}\n",
            "[1, 2.0, {\"a\": null}, \"\\u00e9\", false]",
            "int main() {\n  int x = 1;\n  while (x < 10) { x = x + 1; }\n  return x;\n}\n// done\n/* c */",
            "1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. O-O Nf6\n5. exd5 Qxd5+\n",
            "x = (a + b) * 2\ny=z/3\n", "- apples\n- pears\n- café au lait\n",
            "今日は、いい天気ですね。カタカナも。", "Hello, world! It's 3:45 p.m. (really).",
        ];
        let mut set: std::collections::BTreeSet<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for s in samples {
            let b = s.as_bytes();
            for len in 2..=6 { for w in b.windows(len) { set.insert(w.to_vec()); } }
        }
        let mut v: Vec<Option<Vec<u8>>> = set.into_iter().map(Some).collect();
        v.insert(7, None); // a special token: no text, never allowed as text
        v
    }

    /// The cached mask equals the full walk (which equals stepping every token) at every step of seeded
    /// random walks through llama.cpp's sample grammars, regex- and choice-compiled grammars, and a grammar
    /// with token elements — with suffixes of one frame always, of the whole stack always, and chosen by the
    /// depending-set limit (so remainders are resolved at every depth) — and the caches are actually used.
    #[test]
    fn the_cached_mask_equals_the_full_walk() {
        let vocab = test_vocab();
        let n = vocab.len();
        let t = Trie::new(&vocab);
        let tok = |s: &str| vocab.iter().position(|b| b.as_deref() == Some(s.as_bytes())).map(|i| i as u32);
        let mut srcs: Vec<(String, String)> = [
            ("json", include_str!("../tests/fixtures/grammars/json.gbnf")),
            ("json_arr", include_str!("../tests/fixtures/grammars/json_arr.gbnf")),
            ("arithmetic", include_str!("../tests/fixtures/grammars/arithmetic.gbnf")),
            ("c", include_str!("../tests/fixtures/grammars/c.gbnf")),
            ("chess", include_str!("../tests/fixtures/grammars/chess.gbnf")),
            ("list", include_str!("../tests/fixtures/grammars/list.gbnf")),
            ("japanese", include_str!("../tests/fixtures/grammars/japanese.gbnf")),
            ("english", include_str!("../tests/fixtures/grammars/english.gbnf")),
        ].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        srcs.push(("regex".into(), ferric_agent::regex::regex_to_gbnf("(\\d{1,3}\\.){3}\\d{1,3}|[a-z]+@[a-z]+\\.(com|org)").unwrap()));
        srcs.push(("choice".into(), ferric_agent::regex::choice_to_gbnf(&["yes".into(), "no".into(), "maybe not".into()]).unwrap()));
        srcs.push(("tokens".into(), "root ::= \"x\" <[7]> ( \"ab\" | !<[3]> ) \"}\"".into()));
        let limits = [usize::MAX, 0, 16];
        let (mut steps, mut warm, mut lens) = (0usize, 0usize, std::collections::BTreeSet::new());
        for (name, src) in &srcs {
            let g = Arc::new(Grammar::parse(src, "root", &tok).unwrap_or_else(|e| panic!("{name}: {e}")));
            let xs: Vec<Masks> = limits.iter().map(|&l| Masks { limit: l, ..Masks::new(g.clone(), n) }).collect();
            for seed in 0..4u64 {
                let mut rng = 0x9E3779B97F4A7C15u64 ^ seed.wrapping_mul(0xD1B54A32D192ED03);
                let mut state = Guide::Grammar(Matcher::new(g.clone()));
                for _ in 0..40 {
                    let Guide::Grammar(m) = &state else { unreachable!() };
                    let full = t.allowed(&state, n);
                    if !m.mid_char() {
                        for (x, l) in xs.iter().zip(limits) {
                            let cached = t.allowed_by_stacks(m, x, n, &vocab);
                            let diff: Vec<usize> = (0..n).filter(|&i| full[i] != cached[i]).collect();
                            assert!(diff.is_empty(), "{name} seed {seed} step {steps} limit {l}: {} tokens differ, first {:?} (full {}, cached {})",
                                diff.len(), vocab[diff[0]].as_ref().map(|b| String::from_utf8_lossy(b).to_string()), full[diff[0]], cached[diff[0]]);
                        }
                        steps += 1;
                    }
                    let allowed: Vec<usize> = (0..n).filter(|&i| full[i]).collect();
                    if allowed.is_empty() { break; }
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    let pick = allowed[(rng % allowed.len() as u64) as usize];
                    state.commit(pick as u32, vocab[pick].as_deref());
                }
            }
            for k in xs[2].suffixes.lock().unwrap().keys() { lens.insert(k.len()); }
            warm += xs.iter().map(|x| x.hits.load(std::sync::atomic::Ordering::Relaxed)).sum::<usize>();
        }
        assert!(steps > 400, "only {steps} compared steps");
        assert!(warm > steps, "the whole-stack cache is not reused: {warm} hits over {steps} steps x 3 limits");
        assert!(lens.len() >= 3, "the limit never chose a longer suffix: suffix lengths {lens:?}");
    }

    /// The JSON guide's state-keyed cache equals the full walk: one cache shared by seeded random walks, so
    /// states reached by different histories (different stale stack slots) are served from each other.
    #[test]
    fn the_json_state_cache_equals_the_full_walk() {
        let vocab = test_vocab();
        let n = vocab.len();
        let t = Trie::new(&vocab);
        let mut cache = HashMap::new();
        let (mut steps, mut hits) = (0usize, 0usize);
        for seed in 0..12u64 {
            let mut rng = 0x2545F4914F6CDD1Du64 ^ seed.wrapping_mul(0x9E3779B97F4A7C15);
            let mut state = Json::object();
            for _ in 0..50 {
                let hit = cache.contains_key(&state);
                let cached = t.allowed_json(&state, &mut cache, n);
                assert_eq!(cached, t.allowed(&Guide::Json(state), n), "seed {seed} step {steps}");
                steps += 1;
                hits += hit as usize;
                let allowed: Vec<usize> = (0..n).filter(|&i| cached[i]).collect();
                if allowed.is_empty() { break; }
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let pick = allowed[(rng % allowed.len() as u64) as usize];
                for &b in vocab[pick].as_deref().unwrap() { assert!(state.step(b)); }
            }
        }
        assert!(steps > 300 && hits > steps / 3, "{hits} cache hits in {steps} steps");
    }

    /// Cost on a real vocabulary (`FERRIC_BENCH_GGUF=<model.gguf>`, run with --release --ignored): a walk
    /// through json.gbnf, full walk vs per-top cache, cold and warm, asserting equal masks on the way.
    #[test]
    #[ignore]
    fn bench_grammar_mask_on_a_real_vocabulary() {
        use ferric_gguf::{GgufFile, Meta};
        let path = std::env::var("FERRIC_BENCH_GGUF").expect("FERRIC_BENCH_GGUF");
        let g = GgufFile::open(&path).unwrap();
        let Some(Meta::Arr(a)) = g.metadata.get("tokenizer.ggml.tokens") else { panic!("no tokens") };
        let u2b = crate::byte_decoder();
        let vocab: Vec<Option<Vec<u8>>> = a.iter().map(|m| {
            let Meta::Str(t) = m else { return None };
            t.chars().map(|c| u2b.get(&c).copied()).collect::<Option<Vec<u8>>>()
        }).collect();
        let n = vocab.len();
        let t0 = std::time::Instant::now();
        let t = Trie::new(&vocab);
        eprintln!("vocab {n}, trie built in {:?}", t0.elapsed());
        let target_json = "{\"name\": \"Ada Lovelace\", \"born\": 1815, \"fields\": [\"mathematics\", \"computing\"], \"notes\": {\"engine\": \"Analytical\", \"program\": true, \"pages\": 65}}";
        {   // The JSON-object guide (response_format json_object) has no grammar cache: its walk, for scale.
            let mut state = Guide::Json(Json::object());
            let (mut tf, mut k, mut text) = (std::time::Duration::ZERO, 0, Vec::new());
            for _ in 0..60 {
                let a = std::time::Instant::now();
                let full = t.allowed(&state, n);
                tf += a.elapsed();
                k += 1;
                let rest = &target_json.as_bytes()[text.len().min(target_json.len())..];
                let Some(pick) = (0..n).filter(|&i| full[i] && vocab[i].as_deref().is_some_and(|b| rest.starts_with(b)))
                    .max_by_key(|&i| vocab[i].as_ref().unwrap().len()) else { break };
                text.extend_from_slice(vocab[pick].as_deref().unwrap());
                state.commit(pick as u32, vocab[pick].as_deref());
            }
            eprintln!("json_object guide: {k} steps; full walk {:.2} ms/step", tf.as_secs_f64() * 1e3 / k as f64);
        }
        for (name, src, target) in [("json", include_str!("../tests/fixtures/grammars/json.gbnf"), target_json),
                            ("regex", &ferric_agent::regex::regex_to_gbnf("[A-Z][a-z]+ (is|was) [0-9]{1,4}( years)?\\.").unwrap()[..], "Ada was 36 years."),
                            ("c", include_str!("../tests/fixtures/grammars/c.gbnf"), "int main() {\n  int total = 0;\n  for (i = 0; i < 10; i = i + 1) {\n    total = total + i;\n  }\n  return total;\n}\n")] {
            let gr = Arc::new(Grammar::parse(src, "root", &|_| None).unwrap());
            let x = Masks::new(gr.clone(), n);
            for pass in ["cold", "warm"] {
                let mut state = Guide::Grammar(Matcher::new(gr.clone()));
                let (mut tf, mut tc, mut k) = (std::time::Duration::ZERO, std::time::Duration::ZERO, 0);
                let mut rng = 12345u64;
                let mut text = Vec::new();
                for _ in 0..60 {
                    let Guide::Grammar(m) = &state else { unreachable!() };
                    let a = std::time::Instant::now();
                    let full = t.allowed(&state, n);
                    tf += a.elapsed();
                    if !m.mid_char() {
                        let a = std::time::Instant::now();
                        let cached = t.allowed_by_stacks(m, &x, n, &vocab);
                        tc += a.elapsed();
                        assert!(full == cached, "{name}: masks differ");
                        k += 1;
                    }
                    // Steer toward the target: the longest allowed token that continues it, else a random one.
                    let rest = &target.as_bytes()[text.len().min(target.len())..];
                    let pick = (0..n).filter(|&i| full[i] && vocab[i].as_deref().is_some_and(|b| rest.starts_with(b)))
                        .max_by_key(|&i| vocab[i].as_ref().unwrap().len());
                    let pick = match pick {
                        Some(p) => p,
                        None => {
                            let allowed: Vec<usize> = (0..n).filter(|&i| full[i]).collect();
                            if allowed.is_empty() { break; }
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            allowed[(rng % allowed.len() as u64) as usize]
                        }
                    };
                    text.extend_from_slice(vocab[pick].as_deref().unwrap_or(b""));
                    state.commit(pick as u32, vocab[pick].as_deref());
                }
                eprintln!("{name} {pass}: {k} steps; full walk {:.2} ms/step, cached {:.3} ms/step ({} suffixes) — {:?}",
                    tf.as_secs_f64() * 1e3 / k as f64, tc.as_secs_f64() * 1e3 / k as f64, x.suffixes.lock().unwrap().len(),
                    String::from_utf8_lossy(&text).chars().take(80).collect::<String>());
            }
        }
    }

    #[test]
    fn every_spelling_is_read_and_two_at_once_are_refused() {
        let none = |src: &str| Grammar::parse(src, "root", &|_| None).map(Arc::new);
        assert!(matches!(spec_of(&json!({"grammar": "root ::= \"a\""}), &none).unwrap(), Spec::Grammar(_)));
        assert!(matches!(spec_of(&json!({"guided_regex": "[0-9]+"}), &none).unwrap(), Spec::Grammar(_)));
        assert!(matches!(spec_of(&json!({"structured_outputs": {"choice": ["a", "b"]}}), &none).unwrap(), Spec::Grammar(_)));
        assert!(matches!(spec_of(&json!({"response_format": {"type": "json_object"}}), &none).unwrap(), Spec::Json));
        assert!(matches!(spec_of(&json!({"guided_json": {"type": "object", "properties": {"a": {"type": "integer"}}}}), &none).unwrap(), Spec::Schema(_)));
        assert!(matches!(spec_of(&json!({}), &none).unwrap(), Spec::None));
        let e = spec_of(&json!({"grammar": "root ::= \"a\"", "guided_regex": "b"}), &none).err().unwrap();
        assert!(e.contains("grammar") && e.contains("guided_regex"), "{e}");
        assert!(spec_of(&json!({"grammar": "root ::= undefined"}), &none).err().unwrap().contains("Undefined"));
        assert!(spec_of(&json!({"guided_regex": "(a)\\1"}), &none).is_err());
    }
}
