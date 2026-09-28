//! Convert JSON Schemas to GBNF with `ferric_agent::json_schema`, the way llama.cpp's converter is driven for a
//! differential check (tests/fixtures/json_schema/reference/): stdin is a JSON array of schema TEXTS (text, so
//! key order is the document's), stdout a JSON array of `{"ok": grammar}` or `{"err": message}`.
//!
//! `cargo run -p ferric-agent --example json_schema_to_gbnf < schemas.json`
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let texts: Vec<String> = serde_json::from_str(&input).expect("a JSON array of schema texts");
    let out: Vec<serde_json::Value> = texts.iter().map(|t| {
        let r = serde_json::from_str::<serde_json::Value>(t).map_err(|e| format!("bad JSON: {e}"))
            .and_then(|s| ferric_agent::json_schema::schema_to_gbnf(&s));
        match r { Ok(g) => serde_json::json!({"ok": g}), Err(e) => serde_json::json!({"err": e}) }
    }).collect();
    println!("{}", serde_json::to_string(&out).expect("serializes"));
}
