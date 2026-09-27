//! **Render a GGUF's own chat template over a list of conversations** — the Ferric side of
//! `scripts/chat_template_conformance.sh`, which renders the same cases through Hugging Face's
//! `apply_chat_template` environment and compares the strings byte for byte.
//!
//!   render_template <model.gguf> <cases.json>
//!
//! Prints one JSON line per case: `{"i": n, "out": "<rendered>"}` or `{"i": n, "err": "<why>"}`.
use ferric_gguf::{GgufFile, Meta};
use serde_json::{json, Value};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 3, "usage: render_template <model.gguf> <cases.json>");
    let g = GgufFile::open(&a[1]).expect("open gguf");
    let src = match g.metadata.get("tokenizer.chat_template") { Some(Meta::Str(s)) => s.clone(), _ => { eprintln!("no chat template"); std::process::exit(3) } };
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    };
    let tok = |k: &str| match g.metadata.get(k) { Some(Meta::U(v)) => toks.get(*v as usize).cloned().unwrap_or_default(), _ => String::new() };
    let t = match ferric_serve::template::ChatTemplate::compile(&src, &tok("tokenizer.ggml.bos_token_id"), &tok("tokenizer.ggml.eos_token_id")) {
        Ok(t) => t,
        Err(e) => { println!("{}", json!({"compile_err": e})); return; }
    };
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("cases")).expect("cases json");
    for (i, c) in cases.iter().enumerate() {
        let empty = vec![];
        let tools = c["tools"].as_array().map(|v| v.as_slice());
        let kw = c["kwargs"].as_object().cloned().unwrap_or_default();
        let r = t.render(c["messages"].as_array().unwrap_or(&empty), tools, c["add_generation_prompt"].as_bool().unwrap_or(true), &kw);
        println!("{}", match r { Ok(s) => json!({"i": i, "out": s}), Err(e) => json!({"i": i, "err": e}) });
    }
}
