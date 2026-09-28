//! `ferric_agent::json_schema` against llama.cpp's own tests (tests/fixtures/json_schema/llama_cpp_cases.json,
//! extracted by extract.py there), and against the `jsonschema` package's verdicts (semantic.json).
use ferric_agent::grammar::{Grammar, Matcher};
use ferric_agent::json_schema::{parse_schema, schema_to_gbnf, Converter, Node};
use serde_json::Value;
use std::sync::Arc;

fn fixture() -> Value { serde_json::from_str(include_str!("fixtures/json_schema/llama_cpp_cases.json")).unwrap() }

fn grammar(src: &str, what: &str) -> Arc<Grammar> {
    let g = Grammar::parse(src, "root", &|_| None).unwrap_or_else(|e| panic!("{what}: the grammar does not parse: {e}\n{src}"));
    assert!(g.symbols.contains_key("root"), "{what}: no root symbol");
    Arc::new(g)
}

/// Feed a string as llama.cpp's grammar harness does, one code point per token, and ask for a complete parse.
fn accepts(g: &Arc<Grammar>, s: &str) -> bool {
    let mut m = Matcher::new(g.clone());
    let mut b = [0u8; 4];
    s.chars().all(|c| m.step_token(0, c.encode_utf8(&mut b).as_bytes())) && m.can_stop()
}

/// tests/test-json-schema-to-grammar.cpp, case for case: every success case prints exactly the expected grammar
/// (the harness compares after trim(); here the whole output must be the trimmed text plus its final newline),
/// which parses with a `root`; every failure case is an error.
#[test]
fn llama_cpp_conversion_cases() {
    let fx = fixture();
    let (mut n_ok, mut n_err) = (0, 0);
    let mut wrong = Vec::new();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let schema: Value = serde_json::from_str(c["schema"].as_str().unwrap()).unwrap();
        let got = schema_to_gbnf(&schema);
        match (c["status"].as_str().unwrap(), got) {
            ("SUCCESS", Ok(g)) => {
                let want = format!("{}\n", c["expected"].as_str().unwrap());
                if g == want { grammar(&g, name); n_ok += 1; } else { wrong.push(format!("{name}:\n--- expected\n{want}--- got\n{g}")); }
            }
            ("SUCCESS", Err(e)) => wrong.push(format!("{name}: expected a grammar, got the error {e}")),
            ("FAILURE", Err(_)) => n_err += 1,
            ("FAILURE", Ok(g)) => wrong.push(format!("{name}: expected an error, got\n{g}")),
            (s, _) => panic!("status {s}"),
        }
    }
    assert!(wrong.is_empty(), "{} of {} cases differ from llama.cpp:\n\n{}", wrong.len(), n_ok + n_err + wrong.len(), wrong.join("\n\n"));
    assert_eq!((n_ok, n_err), (78, 3), "the fixture changed size");
}

/// The file's two hand-written blocks: a document parsed up front converts as its JSON does (recursion
/// included), and a property's `$ref` node converted alone as `root` names the ref rule.
#[test]
fn llama_cpp_extra_cases() {
    let fx = fixture();
    let extra = fx["extra"].as_array().unwrap();
    assert_eq!(extra.len(), 2);
    for c in extra {
        let schema: Value = serde_json::from_str(c["schema"].as_str().unwrap()).unwrap();
        let doc = parse_schema(&schema).unwrap();
        match c["kind"].as_str().unwrap() {
            "document_equals_json" => {
                let mut conv = Converter::new(false, Some(&doc));
                conv.add_schema("root", &doc.root).unwrap();
                let (g, _) = conv.finish().unwrap();
                assert_eq!(g, schema_to_gbnf(&schema).unwrap());
                grammar(&g, "parsed document");
            }
            "property_0_as_root" => {
                let Node::Object { properties, .. } = &doc.root else { panic!("the parameters are an object") };
                let mut conv = Converter::new(false, Some(&doc));
                conv.add_schema("root", &properties[0].schema).unwrap();
                let (g, _) = conv.finish().unwrap();
                assert_eq!(g, format!("{}\n", c["expected"].as_str().unwrap()));
                grammar(&g, "sub-schema $ref");
            }
            k => panic!("kind {k}"),
        }
    }
}

/// tests/test-grammar-integration.cpp's schema cases: the converted grammar matches every passing string and no
/// failing one.
#[test]
fn llama_cpp_schema_matching_cases() {
    let fx = fixture();
    let (mut n_pass, mut n_fail) = (0, 0);
    for c in fx["schema_matching"].as_array().unwrap() {
        let desc = format!("{} {}", c["desc"].as_str().unwrap(), c["schema"].as_str().unwrap());
        let schema: Value = serde_json::from_str(c["schema"].as_str().unwrap()).unwrap();
        let g = grammar(&schema_to_gbnf(&schema).unwrap_or_else(|e| panic!("{desc}: {e}")), &desc);
        for s in c["pass"].as_array().unwrap() {
            assert!(accepts(&g, s.as_str().unwrap()), "{desc}: should match {s}");
            n_pass += 1;
        }
        for s in c["fail"].as_array().unwrap() {
            assert!(!accepts(&g, s.as_str().unwrap()), "{desc}: should NOT match {s}");
            n_fail += 1;
        }
    }
    assert_eq!((n_pass, n_fail), (126, 122), "the fixture changed size");
}
