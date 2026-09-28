//! `ferric_agent::json_schema` against llama.cpp's own tests (tests/fixtures/json_schema/llama_cpp_cases.json,
//! extracted by extract.py there), against llama.cpp's own converter (differential.json.gz) and against the
//! `jsonschema` package's verdicts (semantic.json.gz).
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

/// llama.cpp's own converter, built from its source (fixtures/json_schema/reference/build.sh), run over 1469
/// schemas (make_differential.py: the authors' schemas, integer bounds across magnitudes and signs, lengths,
/// formats, ~90 regex patterns, tricky property names, arrays, tuples, combinators, $refs, 1200 float literals,
/// refused schemas, 350 random compositions): the port prints the same grammar byte for byte, or the same
/// error message. The 3 cases marked `deviation` are llama.cpp splitting a multi-byte character in front of a
/// quantifier (invalid UTF-8); there the port's grammar must differ, keep the character whole and parse.
#[test]
fn llama_cpp_reference_differential() {
    let fx = gunzip("differential.json.gz");
    let (mut same_grammar, mut same_error, mut deviations) = (0, 0, 0);
    let mut wrong = Vec::new();
    for c in fx["cases"].as_array().unwrap() {
        let st = c["schema"].as_str().unwrap();
        let got = serde_json::from_str::<Value>(st).map_err(|e| format!("bad JSON: {e}")).and_then(|s| schema_to_gbnf(&s));
        if c.get("deviation").is_some() {
            let g = got.unwrap_or_else(|e| panic!("{st}: {e}"));
            assert_ne!(Some(g.as_str()), c["ok"].as_str(), "{st}: marked a deviation but equal to llama.cpp");
            assert!(!g.contains('\u{FFFD}'), "{st}: a split character\n{g}");
            grammar(&g, st);
            deviations += 1;
            continue;
        }
        match (got, c["ok"].as_str(), c["err"].as_str()) {
            (Ok(g), Some(want), _) if g == want => same_grammar += 1,
            (Err(e), _, Some(want)) if e == want => same_error += 1,
            (got, ok, err) => wrong.push(format!("[{}] {st}\n  llama.cpp: {:?}\n  ferric:    {got:?}", c["group"], ok.or(err))),
        }
    }
    assert!(wrong.is_empty(), "{} of {} schemas differ from llama.cpp's converter:\n{}", wrong.len(), same_grammar + same_error + wrong.len(),
        wrong.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
    assert_eq!((same_grammar, same_error, deviations), (1409, 57, 3), "the fixture changed size");
}

fn gunzip(name: &str) -> Value {
    use std::io::Read;
    let gz = std::fs::read(format!("{}/tests/fixtures/json_schema/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// What the grammars mean, against the `jsonschema` package (make_semantic.py): for 109 schemas (the authors'
/// and 16 realistic API ones), 8222 JSON texts labelled valid or invalid by jsonschema. The grammar accepts
/// exactly the valid ones, except where the fixture names the llama.cpp departure that makes them differ (each
/// established there by a counterfactual). Such an instance must then disagree; where the counterfactual is a
/// respelling of the same value, the grammar must take the respelled text exactly when jsonschema takes the
/// value.
#[test]
fn semantic_verdicts_match_jsonschema_within_documented_departures() {
    let fx = gunzip("semantic.json.gz");
    let causes = fx["causes"].as_object().unwrap();
    let (mut n, mut agree, mut depart, mut skipped) = (0, 0, 0, 0);
    let mut per_cause: std::collections::BTreeMap<String, usize> = Default::default();
    let mut wrong = Vec::new();
    for s in fx["schemas"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        if s.get("skipped").is_some() { skipped += 1; continue; }
        let schema: Value = serde_json::from_str(s["schema"].as_str().unwrap()).unwrap();
        let g = grammar(&schema_to_gbnf(&schema).unwrap_or_else(|e| panic!("{name}: {e}")), name);
        for i in s["instances"].as_array().unwrap() {
            let (text, valid) = (i["text"].as_str().unwrap(), i["valid"].as_bool().unwrap());
            let got = accepts(&g, text);
            n += 1;
            match i["cause"].as_str() {
                None => if got == valid { agree += 1 } else { wrong.push(format!("[{name}] {text:?}: jsonschema {valid}, grammar {got}")) },
                Some(c) => {
                    for part in c.split('+') { assert!(causes.contains_key(part), "{name}: unknown cause {part}"); }
                    if got == valid { wrong.push(format!("[{name}] {text:?}: marked `{c}` but the grammar agrees with jsonschema ({valid})")); }
                    if let Some(r) = i["respelled"].as_str() && accepts(&g, r) != valid {
                        wrong.push(format!("[{name}] {text:?}: respelled {r:?} still gets {}", !valid));
                    }
                    *per_cause.entry(c.to_string()).or_default() += 1;
                    depart += 1;
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} of {n} instances:\n{}", wrong.len(), wrong.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
    eprintln!("{n} instances: {agree} agree with jsonschema, {depart} depart as documented: {per_cause:?}");
    assert_eq!((n, agree, depart, skipped), (8222, 6999, 1223, 3), "the fixture changed size");
}

/// Guards the port adds where llama.cpp would exhaust the stack: each is an error, never a crash.
#[test]
fn deep_or_cyclic_schemas_are_refused_not_a_stack_overflow() {
    // a $ref cycle inside allOf (llama.cpp's add_component recurses forever): followed once
    let s = serde_json::json!({"allOf": [{"$ref": "#/$defs/a"}], "$defs": {"a": {"allOf": [{"$ref": "#/$defs/a"}, {"properties": {"x": {"type": "integer"}}}]}}});
    let g = schema_to_gbnf(&s).unwrap();
    grammar(&g, "allOf cycle");
    // nesting past MAX_DEPTH, built without the JSON parser's own depth limit
    let mut s = serde_json::json!({"type": "integer"});
    for _ in 0..400 { s = serde_json::json!({"type": "array", "items": s}); }
    let e = schema_to_gbnf(&s).unwrap_err();
    assert!(e.contains("nested more than 256"), "{e}");
    // ...while 250 levels convert and parse (on a test thread's 2 MB stack, in a debug build)
    let mut s = serde_json::json!({"type": "integer"});
    for _ in 0..250 { s = serde_json::json!({"type": "array", "items": s}); }
    grammar(&schema_to_gbnf(&s).unwrap(), "250 nested arrays");
    // a chain of 400 $refs: every hop is a level
    let mut defs = serde_json::Map::new();
    for i in 0..400 { defs.insert(format!("d{i}"), serde_json::json!({"$ref": format!("#/$defs/d{}", i + 1)})); }
    defs.insert("d400".into(), serde_json::json!({"type": "null"}));
    let e = schema_to_gbnf(&serde_json::json!({"$ref": "#/$defs/d0", "$defs": defs})).unwrap_err();
    assert!(e.contains("nested more than 256"), "{e}");
    // a 2000-character property name beside additionalProperties nests the key rule 2000 levels deep
    let long = "k".repeat(2000);
    let e = schema_to_gbnf(&serde_json::json!({"type": "object", "properties": {long: {"type": "integer"}}, "additionalProperties": true})).unwrap_err();
    assert!(e.contains("nests 2000 parentheses"), "{e}");
    // ...while one of 300 characters converts and parses
    let g = schema_to_gbnf(&serde_json::json!({"type": "object", "properties": {"k".repeat(300): {"type": "integer"}}, "additionalProperties": true})).unwrap();
    grammar(&g, "300-character property name");
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
