//! Which JSON texts a schema's grammar accepts: `ferric_agent::json_schema` converts each schema to GBNF and
//! the grammar matcher runs every instance through it, byte by byte, to a complete parse. Used to build and
//! explain tests/fixtures/json_schema/semantic.json.gz (make_semantic.py).
//!
//! stdin: `[{"schema": <schema text>, "instances": [<text>, ...]}, ...]`
//! stdout: `[[true, false, ...], ...]`, or `{"err": message}` in place of a schema's list when it does not convert.
//!
//! `cargo run -p ferric-agent --example json_schema_accepts < cases.json`
use ferric_agent::grammar::{Grammar, Matcher};
use serde_json::{json, Value};
use std::io::Read;
use std::sync::Arc;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let cases: Vec<Value> = serde_json::from_str(&input).expect("a JSON array");
    let out: Vec<Value> = cases.iter().map(|c| {
        let g = serde_json::from_str::<Value>(c["schema"].as_str().expect("schema text")).map_err(|e| e.to_string())
            .and_then(|s| ferric_agent::json_schema::schema_to_gbnf(&s))
            .and_then(|src| Grammar::parse(&src, "root", &|_| None));
        match g {
            Err(e) => json!({"err": e}),
            Ok(g) => {
                let g = Arc::new(g);
                json!(c["instances"].as_array().expect("instances").iter().map(|t| {
                    let mut m = Matcher::new(g.clone());
                    t.as_str().expect("instance text").bytes().all(|b| m.step(b)) && m.can_stop()
                }).collect::<Vec<bool>>())
            }
        }
    }).collect();
    println!("{}", serde_json::to_string(&out).expect("serializes"));
}
