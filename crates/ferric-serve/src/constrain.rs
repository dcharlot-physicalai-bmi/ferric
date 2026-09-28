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
use crate::Engine;
use ferric_agent::grammar::{Grammar, Matcher};
use ferric_agent::guide::{Guide, Item, Json, Schema};
use serde_json::Value;
use std::sync::Arc;

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

/// Read the request's constraint. `token_of` resolves a GBNF `<text>` token element to its id.
pub(crate) fn spec_of(req: &Value, token_of: &dyn Fn(&str) -> Option<u32>) -> Result<Spec, String> {
    let so = &req["structured_outputs"];
    let mut found: Vec<(&str, Spec)> = Vec::new();
    let grammar = |src: &str, what: &str| -> Result<Spec, String> {
        Grammar::parse(src, "root", token_of).map(|g| Spec::Grammar(Arc::new(g))).map_err(|e| format!("{what}: {e}"))
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
}

impl Engine {
    /// A GBNF `<text>` token element: a special token spelled exactly so, else text that encodes to one token.
    pub(crate) fn token_of(&self, text: &str) -> Option<u32> {
        if let Some((_, id)) = self.specials.iter().find(|(t, _)| t == text) { return Some(*id); }
        let ids = self.enc(text, false);
        (ids.len() == 1).then(|| ids[0])
    }

    /// The request's constraint, resolved against this model's vocabulary.
    pub(crate) fn constraint(&self, req: &Value) -> Result<Spec, String> { spec_of(req, &|t| self.token_of(t)) }

    /// The allowed-token mask for `g` at this step (EOS by `can_stop`), from the vocabulary trie.
    pub(crate) fn allowed(&self, g: &Guide) -> Vec<bool> {
        let t = self.trie.get_or_init(|| Trie::new(&self.token_bytes));
        let mut ok = t.allowed(g, self.model.n_vocab());
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

    #[test]
    fn every_spelling_is_read_and_two_at_once_are_refused() {
        let none = |_: &str| None;
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
