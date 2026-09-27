//! Pre-tokenization against the MODEL AUTHORS' tokenizer files, on strings WITH NEWLINES — the case the
//! line-based llama.cpp gate could never check. Fixture: `examples/refgen/pretok_ref.py` (HuggingFace
//! `tokenizers` running each checkpoint's own `tokenizer.json`), pieces recorded as original substrings.
use ferric_tokenizer::{pretokenize_pub, Pre};

#[test]
fn qwen_and_llama3_pretokenize_like_the_authors_tokenizer_files() {
    let txt = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pretok_newlines.json"))
        .expect("fixture");
    let j: serde_json::Value = serde_json::from_str(&txt).expect("json");
    let mut bad = Vec::new();
    let mut total = 0;
    for (name, pre) in [("qwen2", Pre::Qwen2), ("qwen35", Pre::Qwen35), ("llama3", Pre::Llama3)] {
        let rule = &j["rules"][name];
        assert!(rule["cases"].as_array().is_some_and(|c| c.len() > 30), "{name}: the fixture holds too few cases");
        for c in rule["cases"].as_array().unwrap() {
            let text = c["text"].as_str().unwrap();
            let want: Vec<String> = c["pieces"].as_array().unwrap().iter().map(|p| p.as_str().unwrap().to_string()).collect();
            let got = pretokenize_pub(text, pre);
            total += 1;
            if got != want { bad.push(format!("{name} {text:?}\n   authors {want:?}\n   ferric  {got:?}")); }
        }
    }
    // ⚠ and each rule must have cases WITH a newline, or this reverts to the gate it was written to fix.
    // ⛔ Checked on the parsed case texts: the raw file always contains `\n` inside the recorded regexes,
    // so a search of the file text could never fail (found by review before commit).
    for name in ["qwen2", "qwen35", "llama3"] {
        let with_nl = j["rules"][name]["cases"].as_array().unwrap().iter()
            .filter(|c| c["text"].as_str().unwrap().contains('\n')).count();
        assert!(with_nl >= 10, "{name}: only {with_nl} cases contain a newline");
    }
    assert!(bad.is_empty(), "{} of {total} strings pre-tokenize differently:\n{}", bad.len(), bad.join("\n"));
}
