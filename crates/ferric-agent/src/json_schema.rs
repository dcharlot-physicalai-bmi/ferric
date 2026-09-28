//! **JSON Schema → GBNF**, as llama.cpp converts it, so a schema-constrained request runs on the one verified
//! grammar matcher ([`crate::grammar`]) and its mask cache.
//!
//! A port of llama.cpp's `common/json-schema.cpp` (the schema, read into the subset the converter handles) and
//! `common/json-schema-to-grammar.cpp` (that subset, printed as GBNF), ggml-org/llama.cpp @ 4da6337767f9 —
//! the commit [`crate::grammar`] is ported from. Same rule names, same rule text, same order (rules sorted by
//! name, as its `std::map` prints them), so a schema gives byte for byte the grammar llama.cpp gives. Checked
//! against llama.cpp's own `tests/test-json-schema-to-grammar.cpp` (every expected grammar, every expected
//! failure) and the schema cases of `tests/test-grammar-integration.cpp` (tests/json_schema.rs, fixtures in
//! tests/fixtures/json_schema/).
//!
//! **What it covers** (as llama.cpp does): `$ref` into the same document (`#/$defs/…`, `#/definitions/…`, any
//! JSON pointer), `anyOf` / `oneOf`, `allOf` (object properties merged, enums intersected), `const`, `enum`,
//! `type` (one or a list), `properties` / `required` / `additionalProperties`, `items` / `prefixItems` (a list
//! is a tuple), `minItems` / `maxItems`, string `minLength` / `maxLength` / `pattern` / `format` (`date`,
//! `time`, `date-time`, `uuid`, `uuid1`…`uuid5`), integer `minimum` / `maximum` / `exclusiveMinimum` /
//! `exclusiveMaximum` (a fractional bound rounds inward), and the primitive rules.
//!
//! **What it refuses** (an `Err`, as llama.cpp throws): a `$ref` that is not `#/…` (remote refs are named, never
//! fetched) or does not resolve; a schema that is not an object (so boolean schemas like `items: false`); an
//! unknown or ill-typed `type`; an empty `type` list, `enum`, `anyOf`, `oneOf` or `allOf`; counts that are not
//! non-negative integers, bounds that are not numbers, a `format` or `pattern` that is not a string; a pattern
//! that is not a valid regex (unbalanced parentheses, unterminated class, nothing to repeat, trailing `\`).
//!
//! **What it degrades** (llama.cpp prints a warning; here [`schema_to_gbnf_with_warnings`] returns it): a pattern
//! it cannot translate — not anchored `^…$`, lookaround / named groups / inline flags, `\d` `\w` `\s` and other
//! escapes GBNF has no form for, an anchor inside — accepts any string instead.
//!
//! **What it ignores** (as llama.cpp does, so the grammar accepts more than the schema): number bounds and
//! `multipleOf`, `uniqueItems`, `contains`, `minProperties` / `maxProperties`, `patternProperties`,
//! `propertyNames`, `dependent*`, `if` / `then` / `else`, `not`, `unevaluated*`, other `format`s, and a
//! `required` name with no property; `oneOf` is `anyOf` (no exclusivity); `allOf` keeps only object properties
//! and enum values; beside `$ref`, `anyOf` or `oneOf` every other keyword.
//!
//! **Where the grammar is stricter than the schema** (llama.cpp's choices, for generation): properties come in
//! schema order, required ones first, then optional ones, then additional ones; `additionalProperties` defaults
//! to false once `properties` is given; whitespace is the `space` rule (nothing, one space, or one or two
//! newlines and up to 20 blanks); numbers have at most 16 integral and 16 fractional digits and an exponent
//! without leading zeros; `const` / `enum` values match their compact serialization exactly; a key that
//! additional properties may use must not start like a declared one and stop short (llama.cpp's trie encoding).
//! Repetition limits come from the grammar parser: more than 2000 required items or characters is refused, a
//! maximum above 2000 is not enforced.
//!
//! **Where this port differs from llama.cpp**, each for a reason:
//! - a regex pattern is read per code point, not per byte (llama.cpp splits a multi-byte character in front of a
//!   quantifier, which prints invalid UTF-8 — identical for ASCII patterns);
//! - recursion is bounded: a schema nested more than [`MAX_DEPTH`] levels (counting `$ref` hops) is refused, a
//!   `$ref` cycle inside `allOf` is followed once, and a grammar nested more than [`MAX_GRAMMAR_NESTING`]
//!   parentheses deep is refused (llama.cpp would exhaust the stack; a property name nests one level per
//!   character in the additional-properties key rule);
//! - the key rule and the optional-property chains are built without recursion (same text);
//! - a float in `const` / `enum` prints with the shortest round-trip digits (nlohmann's Grisu2 is shortest in
//!   all but rare cases); serde_json reads `-0` as the float -0.0 where nlohmann reads the integer 0.
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Deepest schema nesting (building and converting, `$ref` hops included) before the schema is refused.
pub const MAX_DEPTH: usize = 256;
/// Deepest parenthesis nesting of a produced grammar before it is refused (the grammar parser recurses per level).
pub const MAX_GRAMMAR_NESTING: usize = 512;

// ------------------------------------------------------------------------------------------------------------
// json-schema.h / json-schema.cpp — the schema, read into the subset the converter handles
// ------------------------------------------------------------------------------------------------------------

/// A string `format` the converter has a rule for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format { None, Uuid, Date, Time, DateTime }

/// One schema node (`common_chat_schema`).
#[derive(Clone, Debug)]
pub enum Node {
    Any,
    /// `{"$ref": "#/..."}`; its target is `Document::refs[ref]`.
    Ref(String),
    /// `anyOf` / `oneOf`, or a `type` list expanded to one alternative per type.
    AnyOf(Vec<Node>),
    AllOf(Vec<Node>),
    Const(Value),
    Enum(Vec<Value>),
    Null,
    Boolean,
    Number,
    /// Inclusive bounds, `exclusive*` folded in; `i64::MIN` / `i64::MAX` when unbounded.
    Integer { minimum: i64, maximum: i64 },
    /// `pattern` empty when absent; `max_length` -1 when unbounded.
    String { pattern: String, format: Format, min_length: i32, max_length: i32 },
    /// `items` is `Any` when absent; `max_items` -1 when unbounded.
    Array { items: Box<Node>, min_items: i32, max_items: i32 },
    Tuple(Vec<Node>),
    /// Properties in schema order; `additional` is `None` when not allowed.
    Object { properties: Vec<Property>, additional: Option<Box<Node>> },
}

#[derive(Clone, Debug)]
pub struct Property { pub name: String, pub schema: Node, pub required: bool }

/// A schema and the `$ref` targets it uses, keyed by the `$ref` string.
#[derive(Clone, Debug)]
pub struct Document { pub root: Node, pub refs: BTreeMap<String, Node> }

fn fail(path: &str, msg: &str) -> String { format!("JSON schema error at {path}: {msg}") }

/// C's `isspace` in the "C" locale, as `strtol` / `strtoull` skip it.
fn c_space(c: char) -> bool { matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r') }

/// `std::stoull`: leading blanks, a sign, decimal digits (a prefix is enough); `None` where it throws.
fn c_stoull(s: &str) -> Option<u64> {
    let s = s.trim_start_matches(c_space);
    let (neg, s) = match s.as_bytes().first() { Some(b'-') => (true, &s[1..]), Some(b'+') => (false, &s[1..]), _ => (false, s) };
    let d = s.bytes().take_while(u8::is_ascii_digit).count();
    if d == 0 { return None; }
    let v: u64 = s[..d].parse().ok()?;
    Some(if neg { v.wrapping_neg() } else { v })
}

/// `std::stoi`: as `c_stoull`, and `None` outside `int`.
fn c_stoi(s: &str) -> Option<i32> {
    let s = s.trim_start_matches(c_space);
    let (neg, s) = match s.as_bytes().first() { Some(b'-') => (true, &s[1..]), Some(b'+') => (false, &s[1..]), _ => (false, s) };
    let d = s.bytes().take_while(u8::is_ascii_digit).count();
    if d == 0 { return None; }
    let v: i64 = s[..d].parse().ok()?;
    i32::try_from(if neg { -v } else { v }).ok()
}

struct Builder<'a> {
    root: &'a Value,
    /// Targets built so far; `None` while one is being built, so a cycle back to it stops there.
    refs: BTreeMap<String, Option<Node>>,
    depth: usize,
}

impl<'a> Builder<'a> {
    /// A count keyword: absent → `def`, else a non-negative integer (read as C++ `get<int>()` reads it).
    fn get_count(schema: &Map<String, Value>, key: &str, path: &str, def: i32) -> Result<i32, String> {
        let Some(v) = schema.get(key) else { return Ok(def) };
        let n = match (v.as_i64(), v.as_u64()) { (Some(i), _) => i as i32, (None, Some(u)) => u as i32, _ => -1 };
        if !(v.is_i64() || v.is_u64()) || n < 0 { return Err(fail(path, &format!("{key} must be a non-negative integer"))); }
        Ok(n)
    }

    /// A bound: an integer as is; a fractional one rounded inwards, towards the integers it still admits.
    fn get_bound(schema: &Map<String, Value>, key: &str, path: &str, round_up: bool) -> Result<i64, String> {
        let v = &schema[key];
        if let Some(i) = v.as_i64() { return Ok(i); }
        if let Some(u) = v.as_u64() { return Ok(u as i64); }
        let Some(d) = v.as_f64() else { return Err(fail(path, &format!("{key} must be a number"))) };
        Ok((if round_up { d.ceil() } else { d.floor() }) as i64)
    }

    fn get_format(schema: &Map<String, Value>, path: &str) -> Result<Format, String> {
        let Some(v) = schema.get("format") else { return Ok(Format::None) };
        let Some(f) = v.as_str() else { return Err(fail(path, "format must be a string")) };
        Ok(match f {
            "date" => Format::Date,
            "time" => Format::Time,
            "date-time" => Format::DateTime,
            "uuid" => Format::Uuid,
            f if f.len() == 5 && f.starts_with("uuid") && (b'1'..=b'5').contains(&f.as_bytes()[4]) => Format::Uuid,
            _ => Format::None,
        })
    }

    /// Follow a `#/a/b/0` pointer from the document root (tokens are not unescaped, as in llama.cpp).
    fn resolve_ref(root: &'a Value, r: &str, path: &str) -> Result<&'a Value, String> {
        let mut target = root;
        for sel in r[1..].split('/').skip(1) {
            if let Some(o) = target.as_object().filter(|o| o.contains_key(sel)) {
                target = &o[sel];
            } else if let Some(a) = target.as_array() {
                let idx = c_stoull(sel).unwrap_or(a.len() as u64);
                if idx >= a.len() as u64 { return Err(fail(path, &format!("cannot resolve $ref {r}, {sel} is out of range"))); }
                target = &a[idx as usize];
            } else {
                return Err(fail(path, &format!("cannot resolve $ref {r}, {sel} not found")));
            }
        }
        Ok(target)
    }

    fn build_ref(&mut self, value: &Value, path: &str) -> Result<Node, String> {
        let Some(r) = value.as_str() else { return Err(fail(path, "$ref must be a string")) };
        if !r.starts_with("#/") {
            return Err(fail(path, &format!("unsupported $ref {r}, only references into the same document are supported")));
        }
        if !self.refs.contains_key(r) {
            // reserve the key first, so that a cycle back to this $ref stops here
            self.refs.insert(r.to_string(), None);
            let target = Self::resolve_ref(self.root, r, path)?;
            let node = self.build_node(target, r)?;
            self.refs.insert(r.to_string(), Some(node));
        }
        Ok(Node::Ref(r.to_string()))
    }

    fn build_alternatives(&mut self, alts: &Value, path: &str) -> Result<Vec<Node>, String> {
        let Some(a) = alts.as_array() else { return Err(fail(path, "must be an array of schemas")) };
        if a.is_empty() { return Err(fail(path, "must not be empty")); }
        a.iter().enumerate().map(|(i, alt)| self.build_node(alt, &format!("{path}/{i}"))).collect()
    }

    fn build_object(&mut self, schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let required: HashSet<&str> = match schema.get("required") {
            Some(Value::Array(r)) => r.iter().filter_map(Value::as_str).collect(),
            _ => HashSet::new(),
        };
        let mut properties = Vec::new();
        if let Some(props) = schema.get("properties") {
            let Some(props) = props.as_object() else { return Err(fail(path, "properties must be an object")) };
            for (name, prop) in props {
                let node = self.build_node(prop, &format!("{path}/properties/{name}"))?;
                properties.push(Property { name: name.clone(), schema: node, required: required.contains(name.as_str()) });
            }
        }
        let additional = match schema.get("additionalProperties") {
            Some(Value::Bool(true)) => Some(Box::new(Node::Any)),
            Some(Value::Bool(false)) => None,
            Some(a @ Value::Object(_)) => Some(Box::new(self.build_node(a, &format!("{path}/additionalProperties"))?)),
            Some(_) => return Err(fail(path, "additionalProperties must be a boolean or a schema")),
            // {"type": "object"} on its own accepts any object
            None if !schema.contains_key("properties") => Some(Box::new(Node::Any)),
            None => None,
        };
        Ok(Node::Object { properties, additional })
    }

    fn build_array(&mut self, schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let items = if schema.contains_key("items") || schema.contains_key("prefixItems") {
            // "items" wins when both are present; a schema instead of an array is the item schema
            let key = if schema.contains_key("items") { "items" } else { "prefixItems" };
            if let Some(list) = schema[key].as_array() {
                let mut t = Vec::with_capacity(list.len());
                for (i, item) in list.iter().enumerate() { t.push(self.build_node(item, &format!("{path}/{key}/{i}"))?); }
                return Ok(Node::Tuple(t));
            }
            self.build_node(&schema[key], &format!("{path}/{key}"))?
        } else {
            Node::Any
        };
        let min_items = Self::get_count(schema, "minItems", path, 0)?;
        let max_items = Self::get_count(schema, "maxItems", path, -1)?;
        Ok(Node::Array { items: Box::new(items), min_items, max_items })
    }

    fn build_string(schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let pattern = match schema.get("pattern") {
            None => String::new(),
            Some(Value::String(p)) => p.clone(),
            Some(_) => return Err(fail(path, "pattern must be a string")),
        };
        let format = Self::get_format(schema, path)?;
        let min_length = Self::get_count(schema, "minLength", path, 0)?;
        let max_length = Self::get_count(schema, "maxLength", path, -1)?;
        Ok(Node::String { pattern, format, min_length, max_length })
    }

    fn build_integer(schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let mut minimum = i64::MIN;
        let mut maximum = i64::MAX;
        if schema.contains_key("minimum") {
            minimum = Self::get_bound(schema, "minimum", path, true)?;
        } else if schema.contains_key("exclusiveMinimum") {
            minimum = Self::get_bound(schema, "exclusiveMinimum", path, false)?.wrapping_add(1);
        }
        if schema.contains_key("maximum") {
            maximum = Self::get_bound(schema, "maximum", path, false)?;
        } else if schema.contains_key("exclusiveMaximum") {
            maximum = Self::get_bound(schema, "exclusiveMaximum", path, true)?.wrapping_sub(1);
        }
        Ok(Node::Integer { minimum, maximum })
    }

    fn build_node(&mut self, schema: &Value, path: &str) -> Result<Node, String> {
        if self.depth >= MAX_DEPTH { return Err(fail(path, &format!("schema nested more than {MAX_DEPTH} levels deep"))); }
        self.depth += 1;
        let r = self.build_node_inner(schema, path);
        self.depth -= 1;
        r
    }

    fn build_node_inner(&mut self, schema: &Value, path: &str) -> Result<Node, String> {
        let Some(obj) = schema.as_object() else { return Err(fail(path, "schema must be an object")) };
        if let Some(r) = obj.get("$ref") { return self.build_ref(r, path); }
        if obj.contains_key("oneOf") || obj.contains_key("anyOf") {
            let key = if obj.contains_key("oneOf") { "oneOf" } else { "anyOf" };
            return Ok(Node::AnyOf(self.build_alternatives(&obj[key], &format!("{path}/{key}"))?));
        }
        let ty = obj.get("type").unwrap_or(&Value::Null);
        if let Value::Array(types) = ty {
            // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
            if types.is_empty() { return Err(fail(path, "type must not be empty")); }
            let mut children = Vec::with_capacity(types.len());
            for (i, t) in types.iter().enumerate() {
                let mut alt = obj.clone();
                alt.insert("type".into(), t.clone());
                children.push(self.build_node(&Value::Object(alt), &format!("{path}/type/{i}"))?);
            }
            return Ok(Node::AnyOf(children));
        }
        if let Some(c) = obj.get("const") { return Ok(Node::Const(c.clone())); }
        if let Some(values) = obj.get("enum") {
            return match values.as_array() {
                Some(v) if !v.is_empty() => Ok(Node::Enum(v.clone())),
                _ => Err(fail(path, "enum must be a non-empty array")),
            };
        }
        let type_name = match ty {
            Value::Null => "",
            Value::String(s) => s.as_str(),
            _ => return Err(fail(path, "type must be a string or an array of strings")),
        };
        let has_properties = obj.contains_key("properties") || obj.get("additionalProperties").is_some_and(|a| *a != Value::Bool(true));

        if type_name.is_empty() {
            // without a type the structural keywords decide, in the same order as the converter
            if has_properties { return self.build_object(obj, path); }
            if let Some(a) = obj.get("allOf") { return Ok(Node::AllOf(self.build_alternatives(a, &format!("{path}/allOf"))?)); }
            if obj.contains_key("items") || obj.contains_key("prefixItems") { return self.build_array(obj, path); }
            if obj.contains_key("pattern") || obj.contains_key("minLength") || obj.contains_key("maxLength")
                || Self::get_format(obj, path)? != Format::None {
                return Self::build_string(obj, path);
            }
            return Ok(Node::Any);
        }
        match type_name {
            "object" => {
                if !has_properties && let Some(a) = obj.get("allOf") {
                    return Ok(Node::AllOf(self.build_alternatives(a, &format!("{path}/allOf"))?));
                }
                self.build_object(obj, path)
            }
            "string" => {
                if let Some(a) = obj.get("allOf") { return Ok(Node::AllOf(self.build_alternatives(a, &format!("{path}/allOf"))?)); }
                Self::build_string(obj, path)
            }
            "array" => self.build_array(obj, path),
            "integer" => Self::build_integer(obj, path),
            "number" => Ok(Node::Number),
            "boolean" => Ok(Node::Boolean),
            "null" => Ok(Node::Null),
            _ => Err(fail(path, &format!("unrecognized type {type_name}"))),
        }
    }
}

/// Read a JSON Schema into the subset the converter handles (`common_chat_schema_from_json`). `Err` names
/// where and why the schema falls outside it.
pub fn parse_schema(schema: &Value) -> Result<Document, String> {
    let mut b = Builder { root: schema, refs: BTreeMap::new(), depth: 0 };
    let root = b.build_node(schema, "#")?;
    let refs = b.refs.into_iter().map(|(k, v)| (k, v.expect("every reserved $ref is built or the build failed"))).collect();
    Ok(Document { root, refs })
}

// ------------------------------------------------------------------------------------------------------------
// json-schema-to-grammar.cpp — the subset, printed as GBNF
// ------------------------------------------------------------------------------------------------------------

/// The whitespace allowed between tokens: nothing, one space, or one or two newlines and up to 20 blanks.
pub const SPACE_RULE: &str = r#"| " " | "\n"{1,2} [ \t]{0,20}"#;

struct BuiltinRule { name: &'static str, content: &'static str, deps: &'static [&'static str] }

const PRIMITIVE_RULES: &[BuiltinRule] = &[
    BuiltinRule { name: "boolean", content: r#"("true" | "false")"#, deps: &[] },
    BuiltinRule { name: "decimal-part", content: "[0-9]{1,16}", deps: &[] },
    BuiltinRule { name: "integral-part", content: "[0] | [1-9] [0-9]{0,15}", deps: &[] },
    BuiltinRule { name: "number", content: r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"#, deps: &["integral-part", "decimal-part"] },
    BuiltinRule { name: "integer", content: r#"("-"? integral-part)"#, deps: &["integral-part"] },
    BuiltinRule { name: "value", content: "object | array | string | number | boolean | null", deps: &["object", "array", "string", "number", "boolean", "null"] },
    BuiltinRule { name: "object", content: r#""{" space ( string ":" space value ("," space string ":" space value)* )? space "}""#, deps: &["string", "value"] },
    BuiltinRule { name: "array", content: r#""[" space ( value ("," space value)* )? space "]""#, deps: &["value"] },
    BuiltinRule { name: "uuid", content: r##""\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"""##, deps: &[] },
    BuiltinRule { name: "char", content: r##"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"##, deps: &[] },
    BuiltinRule { name: "string", content: r##""\"" char* "\"""##, deps: &["char"] },
    BuiltinRule { name: "null", content: r#""null""#, deps: &[] },
];

const STRING_FORMAT_RULES: &[BuiltinRule] = &[
    BuiltinRule { name: "date", content: r#"[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"#, deps: &[] },
    BuiltinRule { name: "time", content: r#"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"#, deps: &[] },
    BuiltinRule { name: "date-time", content: r#"date "T" time"#, deps: &["date", "time"] },
    BuiltinRule { name: "date-string", content: r##""\"" date "\"""##, deps: &["date"] },
    BuiltinRule { name: "time-string", content: r##""\"" time "\"""##, deps: &["time"] },
    BuiltinRule { name: "date-time-string", content: r##""\"" date-time "\"""##, deps: &["date-time"] },
];

fn primitive(name: &str) -> &'static BuiltinRule { PRIMITIVE_RULES.iter().find(|r| r.name == name).expect("a primitive rule") }
fn format_rule(name: &str) -> &'static BuiltinRule { STRING_FORMAT_RULES.iter().find(|r| r.name == name).expect("a format rule") }

fn is_reserved_name(name: &str) -> bool {
    name == "root" || PRIMITIVE_RULES.iter().chain(STRING_FORMAT_RULES).any(|r| r.name == name)
}

/// `regex_replace(name, "[^a-zA-Z0-9-]+", "-")`: each run of other characters becomes one `-`.
fn rule_name_chars(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_run = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' { out.push(c); in_run = false; } else if !in_run { out.push('-'); in_run = true; }
    }
    out
}

/// A GBNF string literal: quoted, with `\r`, `\n`, `"` and `\` escaped (`gbnf_format_literal`).
pub fn format_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len() + 2);
    out.push('"');
    for c in literal.chars() {
        match c { '\r' => out.push_str("\\r"), '\n' => out.push_str("\\n"), '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), c => out.push(c) }
    }
    out.push('"');
    out
}

/// nlohmann's `dump()` with no indent: compact, keys in document order, strings escaped as serde_json escapes
/// them (the same set), numbers as nlohmann prints them.
pub fn dump_json(v: &Value) -> String {
    let mut out = String::new();
    dump_into(v, &mut out);
    out
}

fn dump_into(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() { out.push_str(&i.to_string()) }
            else if let Some(u) = n.as_u64() { out.push_str(&u.to_string()) }
            else { out.push_str(&dump_float(n.as_f64().unwrap_or(f64::NAN))) }
        }
        Value::String(s) => out.push_str(&serde_json::to_string(s).expect("a string serializes")),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() { if i > 0 { out.push(','); } dump_into(x, out); }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 { out.push(','); }
                out.push_str(&serde_json::to_string(k).expect("a string serializes"));
                out.push(':');
                dump_into(x, out);
            }
            out.push('}');
        }
    }
}

/// nlohmann's `dtoa_impl::grisu2` for a finite positive double: decimal digits and an exponent with
/// `value = digits * 10^exponent` — Grisu2's digits, which are the shortest in all but rare cases (it prints
/// 3.9986349430047603e+17 for 3.99863494300476e+17), so they are ported rather than taken from Rust's
/// shortest formatter.
fn grisu2(value: f64) -> (Vec<u8>, i32) {
    #[derive(Clone, Copy)]
    struct Fp { f: u64, e: i32 }
    fn sub(x: Fp, y: Fp) -> Fp { Fp { f: x.f - y.f, e: x.e } }
    fn mul(x: Fp, y: Fp) -> Fp {
        let (u_lo, u_hi, v_lo, v_hi) = (x.f & 0xFFFF_FFFF, x.f >> 32, y.f & 0xFFFF_FFFF, y.f >> 32);
        let (p0, p1, p2, p3) = (u_lo * v_lo, u_lo * v_hi, u_hi * v_lo, u_hi * v_hi);
        let q = (p0 >> 32) + (p1 & 0xFFFF_FFFF) + (p2 & 0xFFFF_FFFF) + (1u64 << 31); // round, ties up
        Fp { f: p3 + (p2 >> 32) + (p1 >> 32) + (q >> 32), e: x.e + y.e + 64 }
    }
    fn normalize(mut x: Fp) -> Fp { while x.f >> 63 == 0 { x.f <<= 1; x.e -= 1; } x }
    fn normalize_to(x: Fp, e: i32) -> Fp { Fp { f: x.f << (x.e - e), e } }
    // compute_boundaries
    const HIDDEN: u64 = 1 << 52;
    let bits = value.to_bits();
    let (be, bf) = (bits >> 52, bits & (HIDDEN - 1));
    let v = if be == 0 { Fp { f: bf, e: -1074 } } else { Fp { f: bf + HIDDEN, e: be as i32 - 1075 } };
    let m_plus = Fp { f: 2 * v.f + 1, e: v.e - 1 };
    let m_minus = if bf == 0 && be > 1 { Fp { f: 4 * v.f - 1, e: v.e - 2 } } else { Fp { f: 2 * v.f - 1, e: v.e - 1 } };
    let w_plus = normalize(m_plus);
    let (w_minus, w) = (normalize_to(m_minus, w_plus.e), normalize(v));
    // get_cached_power_for_binary_exponent
    const CACHED: [(u64, i32, i32); 79] = [
        (0xAB70FE17C79AC6CA, -1060, -300), (0xFF77B1FCBEBCDC4F, -1034, -292), (0xBE5691EF416BD60C, -1007, -284),
        (0x8DD01FAD907FFC3C, -980, -276), (0xD3515C2831559A83, -954, -268), (0x9D71AC8FADA6C9B5, -927, -260),
        (0xEA9C227723EE8BCB, -901, -252), (0xAECC49914078536D, -874, -244), (0x823C12795DB6CE57, -847, -236),
        (0xC21094364DFB5637, -821, -228), (0x9096EA6F3848984F, -794, -220), (0xD77485CB25823AC7, -768, -212),
        (0xA086CFCD97BF97F4, -741, -204), (0xEF340A98172AACE5, -715, -196), (0xB23867FB2A35B28E, -688, -188),
        (0x84C8D4DFD2C63F3B, -661, -180), (0xC5DD44271AD3CDBA, -635, -172), (0x936B9FCEBB25C996, -608, -164),
        (0xDBAC6C247D62A584, -582, -156), (0xA3AB66580D5FDAF6, -555, -148), (0xF3E2F893DEC3F126, -529, -140),
        (0xB5B5ADA8AAFF80B8, -502, -132), (0x87625F056C7C4A8B, -475, -124), (0xC9BCFF6034C13053, -449, -116),
        (0x964E858C91BA2655, -422, -108), (0xDFF9772470297EBD, -396, -100), (0xA6DFBD9FB8E5B88F, -369, -92),
        (0xF8A95FCF88747D94, -343, -84), (0xB94470938FA89BCF, -316, -76), (0x8A08F0F8BF0F156B, -289, -68),
        (0xCDB02555653131B6, -263, -60), (0x993FE2C6D07B7FAC, -236, -52), (0xE45C10C42A2B3B06, -210, -44),
        (0xAA242499697392D3, -183, -36), (0xFD87B5F28300CA0E, -157, -28), (0xBCE5086492111AEB, -130, -20),
        (0x8CBCCC096F5088CC, -103, -12), (0xD1B71758E219652C, -77, -4), (0x9C40000000000000, -50, 4),
        (0xE8D4A51000000000, -24, 12), (0xAD78EBC5AC620000, 3, 20), (0x813F3978F8940984, 30, 28),
        (0xC097CE7BC90715B3, 56, 36), (0x8F7E32CE7BEA5C70, 83, 44), (0xD5D238A4ABE98068, 109, 52),
        (0x9F4F2726179A2245, 136, 60), (0xED63A231D4C4FB27, 162, 68), (0xB0DE65388CC8ADA8, 189, 76),
        (0x83C7088E1AAB65DB, 216, 84), (0xC45D1DF942711D9A, 242, 92), (0x924D692CA61BE758, 269, 100),
        (0xDA01EE641A708DEA, 295, 108), (0xA26DA3999AEF774A, 322, 116), (0xF209787BB47D6B85, 348, 124),
        (0xB454E4A179DD1877, 375, 132), (0x865B86925B9BC5C2, 402, 140), (0xC83553C5C8965D3D, 428, 148),
        (0x952AB45CFA97A0B3, 455, 156), (0xDE469FBD99A05FE3, 481, 164), (0xA59BC234DB398C25, 508, 172),
        (0xF6C69A72A3989F5C, 534, 180), (0xB7DCBF5354E9BECE, 561, 188), (0x88FCF317F22241E2, 588, 196),
        (0xCC20CE9BD35C78A5, 614, 204), (0x98165AF37B2153DF, 641, 212), (0xE2A0B5DC971F303A, 667, 220),
        (0xA8D9D1535CE3B396, 694, 228), (0xFB9B7CD9A4A7443C, 720, 236), (0xBB764C4CA7A44410, 747, 244),
        (0x8BAB8EEFB6409C1A, 774, 252), (0xD01FEF10A657842C, 800, 260), (0x9B10A4E5E9913129, 827, 268),
        (0xE7109BFBA19C0C9D, 853, 276), (0xAC2820D9623BF429, 880, 284), (0x80444B5E7AA7CF85, 907, 292),
        (0xBF21E44003ACDD2D, 933, 300), (0x8E679C2F5E44FF8F, 960, 308), (0xD433179D9C8CB841, 986, 316),
        (0x9E19DB92B4E31BA9, 1013, 324),
    ];
    let f = -60 - w_plus.e - 1;
    let k = (f * 78913) / (1 << 18) + (f > 0) as i32;
    let (cf, ce, ck) = CACHED[((300 + k + 7) / 8) as usize];
    let c = Fp { f: cf, e: ce };
    let (w, w_minus, w_plus) = (mul(w, c), mul(w_minus, c), mul(w_plus, c));
    let m_minus = Fp { f: w_minus.f + 1, e: w_minus.e };
    let m_plus = Fp { f: w_plus.f - 1, e: w_plus.e };
    let mut exp10 = -ck;
    // grisu2_digit_gen
    let mut buf: Vec<u8> = Vec::with_capacity(17);
    let round = |buf: &mut Vec<u8>, dist: u64, delta: u64, mut rest: u64, ten_k: u64| {
        while rest < dist && delta - rest >= ten_k && (rest + ten_k < dist || dist - rest > rest + ten_k - dist) {
            *buf.last_mut().expect("a digit") -= 1;
            rest += ten_k;
        }
    };
    let mut delta = sub(m_plus, m_minus).f;
    let mut dist = sub(m_plus, w).f;
    let one = Fp { f: 1u64 << -m_plus.e, e: m_plus.e };
    let mut p1 = (m_plus.f >> -one.e) as u32;
    let mut p2 = m_plus.f & (one.f - 1);
    let mut pow10: u32 = [1_000_000_000, 100_000_000, 10_000_000, 1_000_000, 100_000, 10_000, 1_000, 100, 10]
        .into_iter().find(|&p| p1 >= p).unwrap_or(1);
    let mut n = pow10.to_string().len() as i32;
    while n > 0 {
        let (d, r) = (p1 / pow10, p1 % pow10);
        buf.push(b'0' + d as u8);
        p1 = r;
        n -= 1;
        let rest = ((p1 as u64) << -one.e) + p2;
        if rest <= delta {
            exp10 += n;
            round(&mut buf, dist, delta, rest, (pow10 as u64) << -one.e);
            return (buf, exp10);
        }
        pow10 /= 10;
    }
    let mut m = 0;
    loop {
        p2 *= 10;
        let (d, r) = (p2 >> -one.e, p2 & (one.f - 1));
        buf.push(b'0' + d as u8);
        p2 = r;
        m += 1;
        delta *= 10;
        dist *= 10;
        if p2 <= delta { break; }
    }
    exp10 -= m;
    round(&mut buf, dist, delta, p2, one.f);
    (buf, exp10)
}

/// nlohmann's `to_chars` for a double: Grisu2's digits, then `format_buffer(.., -4, 15)` — `100.0`, `0.001`,
/// `1e+16`, `1.5e-05`; non-finite as `null` (its serializer's rule).
fn dump_float(x: f64) -> String {
    if !x.is_finite() { return "null".into(); }
    let mut out = String::new();
    let mut v = x;
    if v.is_sign_negative() { out.push('-'); v = -v; }
    if v == 0.0 { out.push_str("0.0"); return out; }
    let (digits, exp10) = grisu2(v);
    let digits = String::from_utf8(digits).expect("digits");
    let k = digits.len() as i32;
    let n = k + exp10; // the decimal point's position in the digits
    let (min_exp, max_exp) = (-4, 15);
    if k <= n && n <= max_exp {
        out.push_str(&digits);
        for _ in k..n { out.push('0'); }
        out.push_str(".0");
    } else if 0 < n && n <= max_exp {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if min_exp < n && n <= 0 {
        out.push_str("0.");
        for _ in 0..-n { out.push('0'); }
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 { out.push('.'); out.push_str(&digits[1..]); }
        let e = n - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        let e = e.unsigned_abs();
        if e < 10 { out.push('0'); }
        out.push_str(&e.to_string());
    }
    out
}

fn build_repetition(item_rule: &str, min_items: i32, max_items: i32, separator_rule: &str) -> String {
    let has_max = max_items != i32::MAX;
    if max_items == 0 { return String::new(); }
    if min_items == 0 && max_items == 1 { return format!("{item_rule}?"); }
    if separator_rule.is_empty() {
        if min_items == 1 && !has_max { return format!("{item_rule}+"); }
        if min_items == 0 && !has_max { return format!("{item_rule}*"); }
        return format!("{item_rule}{{{min_items},{}}}", if has_max { max_items.to_string() } else { String::new() });
    }
    let rest = build_repetition(&format!("({separator_rule} {item_rule})"), if min_items == 0 { 0 } else { min_items - 1 },
        if has_max { max_items - 1 } else { max_items }, "");
    let result = format!("{item_rule} {rest}");
    if min_items == 0 { format!("({result})?") } else { result }
}

fn digit_range(out: &mut String, from: i32, to: i32) {
    out.push('[');
    out.push(from as u8 as char);
    if from != to { out.push('-'); out.push(to as u8 as char); }
    out.push(']');
}

fn more_digits(out: &mut String, min_digits: i32, max_digits: i32) {
    out.push_str("[0-9]");
    if min_digits == max_digits && min_digits == 1 { return; }
    out.push('{');
    out.push_str(&min_digits.to_string());
    if max_digits != min_digits {
        out.push(',');
        if max_digits != i32::MAX { out.push_str(&max_digits.to_string()); }
    }
    out.push('}');
}

/// The decimal strings from `from` to `to`, both of the same length, digit by digit.
fn uniform_range(out: &mut String, from: &[u8], to: &[u8]) {
    let mut i = 0;
    while i < from.len() && i < to.len() && from[i] == to[i] { i += 1; }
    if i > 0 {
        out.push('"');
        out.push_str(std::str::from_utf8(&from[..i]).expect("digits"));
        out.push('"');
    }
    if i < from.len() && i < to.len() {
        if i > 0 { out.push(' '); }
        let sub_len = from.len() - i - 1;
        let (fi, ti) = (from[i] as i32, to[i] as i32);
        if sub_len > 0 {
            let from_sub = &from[i + 1..];
            let to_sub = &to[i + 1..];
            let sub_zeros = vec![b'0'; sub_len];
            let sub_nines = vec![b'9'; sub_len];
            let mut to_reached = false;
            out.push('(');
            if from_sub == &sub_zeros[..] {
                digit_range(out, fi, ti - 1);
                out.push(' ');
                more_digits(out, sub_len as i32, sub_len as i32);
            } else {
                out.push('[');
                out.push(fi as u8 as char);
                out.push_str("] (");
                uniform_range(out, from_sub, &sub_nines);
                out.push(')');
                if fi < ti - 1 {
                    out.push_str(" | ");
                    if to_sub == &sub_nines[..] {
                        digit_range(out, fi + 1, ti);
                        to_reached = true;
                    } else {
                        digit_range(out, fi + 1, ti - 1);
                    }
                    out.push(' ');
                    more_digits(out, sub_len as i32, sub_len as i32);
                }
            }
            if !to_reached {
                out.push_str(" | ");
                digit_range(out, ti, ti);
                out.push(' ');
                uniform_range(out, &sub_zeros, to_sub);
            }
            out.push(')');
        } else {
            out.push('[');
            out.push(fi as u8 as char);
            out.push('-');
            out.push(ti as u8 as char);
            out.push(']');
        }
    }
}

/// The decimal integers in `[min_value, max_value]` as GBNF (`i64::MIN` / `i64::MAX` = unbounded).
fn build_min_max_int(mut min_value: i64, max_value: i64, out: &mut String, decimals_left: i32, top_level: bool) -> Result<(), String> {
    let has_min = min_value != i64::MIN;
    let has_max = max_value != i64::MAX;

    if has_min && has_max {
        if min_value < 0 && max_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(max_value.wrapping_neg(), min_value.wrapping_neg(), out, decimals_left, true)?;
            out.push(')');
            return Ok(());
        }
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(0, min_value.wrapping_neg(), out, decimals_left, true)?;
            out.push_str(") | ");
            min_value = 0;
        }
        let mut min_s = min_value.to_string().into_bytes();
        let max_s = max_value.to_string().into_bytes();
        for digits in min_s.len()..max_s.len() {
            uniform_range(out, &min_s, &vec![b'9'; digits]);
            min_s = [b"1".as_slice(), &vec![b'0'; digits]].concat();
            out.push_str(" | ");
        }
        uniform_range(out, &min_s, &max_s);
        return Ok(());
    }

    let less_decimals = (decimals_left - 1).max(1);

    if has_min {
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(i64::MIN, min_value.wrapping_neg(), out, decimals_left, false)?;
            out.push_str(") | [0] | [1-9] ");
            more_digits(out, 0, decimals_left - 1);
        } else if min_value == 0 {
            if top_level {
                out.push_str("[0] | [1-9] ");
                more_digits(out, 0, less_decimals);
            } else {
                more_digits(out, 1, decimals_left);
            }
        } else if min_value <= 9 {
            let c = b'0' as i32 + min_value as i32;
            let range_start = if top_level { b'1' } else { b'0' } as i32;
            if c > range_start {
                digit_range(out, range_start, c - 1);
                out.push(' ');
                more_digits(out, 1, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, b'9' as i32);
            out.push(' ');
            more_digits(out, 0, less_decimals);
        } else {
            let min_s = min_value.to_string();
            let len = min_s.len() as i32;
            let c = min_s.as_bytes()[0] as i32;
            if c > b'1' as i32 {
                digit_range(out, if top_level { b'1' } else { b'0' } as i32, c - 1);
                out.push(' ');
                more_digits(out, len, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, c);
            out.push_str(" (");
            let rest: i64 = min_s[1..].parse().expect("digits");
            build_min_max_int(rest, i64::MAX, out, less_decimals, false)?;
            out.push(')');
            if c < b'9' as i32 {
                out.push_str(" | ");
                digit_range(out, c + 1, b'9' as i32);
                out.push(' ');
                more_digits(out, len - 1, less_decimals);
            }
        }
        return Ok(());
    }

    if has_max {
        if max_value >= 0 {
            if top_level {
                out.push_str("\"-\" [1-9] ");
                more_digits(out, 0, less_decimals);
                out.push_str(" | ");
            }
            build_min_max_int(0, max_value, out, decimals_left, true)?;
        } else {
            out.push_str("\"-\" (");
            build_min_max_int(max_value.wrapping_neg(), i64::MAX, out, decimals_left, false)?;
            out.push(')');
        }
        return Ok(());
    }

    Err("At least one of min_value or max_value must be set".into())
}

/// Code-point trie of the declared property names (`common_trie`), children in code point order.
struct Trie { nodes: Vec<(BTreeMap<u32, usize>, i32)> }

impl Trie {
    fn new(words: &[String]) -> Trie {
        let mut t = Trie { nodes: vec![(BTreeMap::new(), -1)] };
        let mut n_patterns = 0;
        for w in words {
            let mut cur = 0usize;
            for c in w.chars() {
                cur = match t.nodes[cur].0.get(&(c as u32)) {
                    Some(&k) => k,
                    None => {
                        let k = t.nodes.len();
                        t.nodes.push((BTreeMap::new(), -1));
                        t.nodes[cur].0.insert(c as u32, k);
                        k
                    }
                };
            }
            if t.nodes[cur].1 < 0 { t.nodes[cur].1 = n_patterns; n_patterns += 1; }
        }
        t
    }
}

enum PatternError { Unsupported(String), Invalid(String) }

const MAX_PATTERN_DEPTH: i32 = 100;

fn is_non_literal(c: char) -> bool { matches!(c, '|' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '*' | '+' | '?' | '^' | '$') }
fn escaped_in_regexps_but_not_in_literals(c: char) -> bool {
    matches!(c, '^' | '$' | '.' | '[' | ']' | '(' | ')' | '|' | '{' | '}' | '*' | '+' | '?')
}

/// Length of a GBNF-compatible escape at `pos` (keep in sync with the grammar parser's `char_`), 0 if none.
fn gbnf_escape_length(p: &[char], pos: usize) -> usize {
    if pos + 1 >= p.len() || p[pos] != '\\' { return 0; }
    let n_hex = match p[pos + 1] {
        'x' => 2,
        'u' => 4,
        'U' => 8,
        't' | 'r' | 'n' | '\\' | '"' | '[' | ']' | '-' => return 2,
        _ => return 0,
    };
    if pos + 2 + n_hex > p.len() { return 0; }
    if !p[pos + 2..pos + 2 + n_hex].iter().all(char::is_ascii_hexdigit) { return 0; }
    2 + n_hex
}

/// Where a regex pattern's translation stands (the state `_pattern_to_rule`'s closures share).
struct Pattern { sub: Vec<char>, i: usize, paren_depth: i32, sub_rule_ids: HashMap<String, String> }

fn to_rule(ls: &(String, bool)) -> String { if ls.1 { format!("\"{}\"", ls.0) } else { ls.0.clone() } }

/// Joins a sequence, merging consecutive literals together.
fn join_seq(seq: &[(String, bool)]) -> (String, bool) {
    let mut ret: Vec<(String, bool)> = Vec::new();
    let mut literal = String::new();
    for item in seq {
        if item.1 {
            literal.push_str(&item.0);
        } else {
            if !literal.is_empty() { ret.push((std::mem::take(&mut literal), true)); }
            ret.push(item.clone());
        }
    }
    if !literal.is_empty() { ret.push((literal, true)); }
    (ret.iter().map(to_rule).collect::<Vec<_>>().join(" "), false)
}

/// Converts schema nodes to GBNF rules (`common_chat_schema_converter`). Rules accumulate by name; `finish`
/// prints them sorted by name.
pub struct Converter<'d> {
    dotall: bool,
    doc: Option<&'d Document>,
    rules: BTreeMap<String, String>,
    refs_being_resolved: HashSet<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
    depth: usize,
}

impl<'d> Converter<'d> {
    /// `dotall`: whether a pattern's `.` matches newlines too. `doc`: where `$ref` targets are looked up.
    pub fn new(dotall: bool, doc: Option<&'d Document>) -> Self {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Converter { dotall, doc, rules, refs_being_resolved: HashSet::new(), errors: Vec::new(), warnings: Vec::new(), depth: 0 }
    }

    /// Add rule `name` (made a valid rule name); an existing rule of that name with other text gets a numbered
    /// sibling instead. Returns the name used.
    pub fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc_name = rule_name_chars(name);
        match self.rules.get(&esc_name) {
            None => { self.rules.insert(esc_name.clone(), rule.to_string()); return esc_name; }
            Some(r) if r == rule => return esc_name,
            _ => {}
        }
        let mut i = 0;
        loop {
            let key = format!("{esc_name}{i}");
            match self.rules.get(&key) {
                Some(r) if r != rule => i += 1,
                _ => { self.rules.insert(key.clone(), rule.to_string()); return key; }
            }
        }
    }

    /// The grammar builder's `add_schema`: convert `schema` as rule `name` (`root` is the start symbol).
    pub fn add_schema(&mut self, name: &str, schema: &'d Node) -> Result<String, String> {
        self.visit(schema, if name == "root" { "" } else { name })
    }

    /// The grammar, or every error the conversion met (`check_errors`, then `format_grammar`), with the
    /// warnings about patterns that were widened to any string.
    pub fn finish(self) -> Result<(String, Vec<String>), String> {
        if !self.errors.is_empty() { return Err(format!("JSON schema conversion failed:\n{}", self.errors.join("\n"))); }
        let mut out = String::new();
        for (k, v) in &self.rules {
            out.push_str(k);
            out.push_str(" ::= ");
            out.push_str(v);
            out.push('\n');
        }
        Ok((out, self.warnings))
    }

    fn add_primitive(&mut self, name: &str, rule: &BuiltinRule) -> String {
        let n = self.add_rule(name, rule.content);
        for dep in rule.deps {
            let dep_rule = PRIMITIVE_RULES.iter().chain(STRING_FORMAT_RULES).find(|r| r.name == *dep);
            let Some(dep_rule) = dep_rule else {
                self.errors.push(format!("Rule {dep} not known"));
                continue;
            };
            if !self.rules.contains_key(*dep) { self.add_primitive(dep, dep_rule); }
        }
        n
    }

    fn visit_primitive(&mut self, rule_name: &str, ty: &str) -> String {
        self.add_primitive(if rule_name == "root" { "root" } else { ty }, primitive(ty))
    }

    fn generate_union_rule(&mut self, name: &str, alts: &'d [Node]) -> Result<String, String> {
        let mut rules = Vec::with_capacity(alts.len());
        for (i, alt) in alts.iter().enumerate() {
            rules.push(self.visit(alt, &format!("{name}{}{i}", if name.is_empty() { "alternative-" } else { "-" }))?);
        }
        Ok(rules.join(" | "))
    }

    fn visit_pattern(&mut self, pattern: &str, name: &str) -> String {
        let snapshot = self.rules.clone();
        match self.pattern_to_rule(pattern, name) {
            Ok(r) => r,
            Err(PatternError::Unsupported(e)) => {
                self.rules = snapshot;
                self.warnings.push(format!("pattern {pattern} is not supported ({e}), accepting any string"));
                let s = self.add_primitive("string", primitive("string"));
                self.add_rule(name, &s)
            }
            Err(PatternError::Invalid(e)) => {
                self.rules = snapshot;
                self.errors.push(format!("Invalid pattern {pattern}: {e}"));
                String::new()
            }
        }
    }

    fn pattern_to_rule(&mut self, pattern: &str, name: &str) -> Result<String, PatternError> {
        let chars: Vec<char> = pattern.chars().collect();
        if chars.len() < 2 || chars[0] != '^' || chars[chars.len() - 1] != '$' {
            return Err(PatternError::Unsupported("not anchored with '^' and '$'".into()));
        }
        let mut p = Pattern { sub: chars[1..chars.len() - 1].to_vec(), i: 0, paren_depth: 0, sub_rule_ids: HashMap::new() };
        let rule = to_rule(&self.transform(&mut p, name)?);
        if p.paren_depth != 0 { return Err(PatternError::Invalid("unbalanced parentheses".into())); }
        Ok(self.add_rule(name, &format!("\"\\\"\" ({rule}) \"\\\"\"")))
    }

    fn get_dot(&mut self) -> String {
        let rule = if self.dotall { "[\\U00000000-\\U0010FFFF]" } else { "[^\\x0A\\x0D]" };
        self.add_rule("dot", rule)
    }

    fn transform(&mut self, p: &mut Pattern, name: &str) -> Result<(String, bool), PatternError> {
        use PatternError::{Invalid, Unsupported};
        let length = p.sub.len();
        let mut seq: Vec<(String, bool)> = Vec::new();
        while p.i < length {
            let c = p.sub[p.i];
            if c == '.' {
                let d = self.get_dot();
                seq.push((d, false));
                p.i += 1;
            } else if c == '(' {
                p.i += 1;
                if p.i < length && p.sub[p.i] == '?' {
                    if p.i + 1 < length && p.sub[p.i + 1] == ':' {
                        p.i += 2; // skip "?:" for non-capturing group, treat as regular group
                    } else {
                        // lookaround, named group, inline flags, ...
                        return Err(Unsupported("unsupported group syntax".into()));
                    }
                }
                p.paren_depth += 1;
                if p.paren_depth > MAX_PATTERN_DEPTH { return Err(Unsupported("pattern nesting too deep".into())); }
                let inner = self.transform(p, name)?;
                seq.push((format!("({})", to_rule(&inner)), false));
            } else if c == ')' {
                p.i += 1;
                if p.paren_depth == 0 { return Err(Invalid("unbalanced parentheses".into())); }
                p.paren_depth -= 1;
                return Ok(join_seq(&seq));
            } else if c == '^' || c == '$' {
                return Err(Unsupported("anchor inside the pattern".into()));
            } else if c == '[' {
                let mut square = String::from("[");
                p.i += 1;
                while p.i < length && p.sub[p.i] != ']' {
                    if p.sub[p.i] == '\\' {
                        let n = gbnf_escape_length(&p.sub, p.i);
                        if n == 0 {
                            let s: String = p.sub[p.i..(p.i + 2).min(length)].iter().collect();
                            return Err(Unsupported(format!("unsupported escape in character class: {s}")));
                        }
                        square.extend(&p.sub[p.i..p.i + n]);
                        p.i += n;
                    } else {
                        square.push(p.sub[p.i]);
                        p.i += 1;
                    }
                }
                if p.i >= length { return Err(Invalid("unterminated character class".into())); }
                square.push(']');
                p.i += 1;
                seq.push((square, false));
            } else if c == '|' {
                seq.push(("|".into(), false));
                p.i += 1;
            } else if c == '*' || c == '+' || c == '?' {
                let Some(last) = seq.last_mut() else { return Err(Invalid("nothing to repeat".into())) };
                *last = (format!("{}{c}", to_rule(last)), false);
                p.i += 1;
            } else if c == '{' {
                let mut curly = String::from("{");
                p.i += 1;
                while p.i < length && p.sub[p.i] != '}' {
                    curly.push(p.sub[p.i]);
                    p.i += 1;
                }
                if p.i >= length { return Err(Unsupported("unterminated curly brackets".into())); }
                curly.push('}');
                p.i += 1;
                let nums: Vec<&str> = curly[1..curly.len() - 1].split(',').collect();
                if nums.len() != 1 && nums.len() != 2 { return Err(Unsupported("wrong number of values in curly brackets".into())); }
                let bad = || Unsupported("invalid number in curly brackets".into());
                let (min_times, max_times) = if nums.len() == 1 {
                    let v = c_stoi(nums[0]).ok_or_else(bad)?;
                    (v, v)
                } else {
                    (if nums[0].is_empty() { 0 } else { c_stoi(nums[0]).ok_or_else(bad)? },
                     if nums[1].is_empty() { i32::MAX } else { c_stoi(nums[1]).ok_or_else(bad)? })
                };
                let Some((sub, sub_is_literal)) = seq.last().cloned() else { return Err(Invalid("nothing to repeat".into())) };
                let item = if sub_is_literal {
                    format!("\"{sub}\"")
                } else {
                    if !p.sub_rule_ids.contains_key(&sub) { p.sub_rule_ids.insert(sub.clone(), String::new()); }
                    if p.sub_rule_ids[&sub].is_empty() {
                        let id = self.add_rule(&format!("{name}-{}", p.sub_rule_ids.len()), &sub);
                        p.sub_rule_ids.insert(sub.clone(), id);
                    }
                    p.sub_rule_ids[&sub].clone()
                };
                *seq.last_mut().expect("checked above") = (build_repetition(&item, min_times, max_times, ""), false);
            } else {
                let mut literal = String::new();
                while p.i < length {
                    let ci = p.sub[p.i];
                    if ci == '\\' {
                        if p.i == length - 1 { return Err(Invalid("trailing backslash".into())); }
                        let next = p.sub[p.i + 1];
                        if escaped_in_regexps_but_not_in_literals(next) {
                            p.i += 1;
                            literal.push(p.sub[p.i]);
                            p.i += 1;
                        } else {
                            let n = gbnf_escape_length(&p.sub, p.i);
                            if n == 0 {
                                let s: String = p.sub[p.i..(p.i + 2).min(length)].iter().collect();
                                return Err(Unsupported(format!("unsupported escape: {s}")));
                            }
                            literal.extend(&p.sub[p.i..p.i + n]);
                            p.i += n;
                        }
                    } else if ci == '"' {
                        literal.push_str("\\\"");
                        p.i += 1;
                    } else if !is_non_literal(ci)
                        && (p.i == length - 1 || literal.is_empty() || p.sub[p.i + 1] == '.' || !is_non_literal(p.sub[p.i + 1])) {
                        literal.push(ci);
                        p.i += 1;
                    } else {
                        break;
                    }
                }
                if literal.is_empty() { return Err(Unsupported(format!("unsupported character: {c}"))); }
                seq.push((literal, true));
            }
        }
        Ok(join_seq(&seq))
    }

    /// A rule matching a JSON string that is none of `strings`:
    /// `not_strings(["a"])` → `["] ( [a] char+ | [^"a] char* )? ["]`. Written without recursion (a name nests
    /// one level per character); the text is llama.cpp's, character classes unescaped as there.
    fn not_strings(&mut self, strings: &[String]) -> String {
        let trie = Trie::new(strings);
        let char_rule = self.add_primitive("char", primitive("char"));
        let mut out = String::from("[\"] ( ");
        // One frame per trie node being written: its children in order, the next child to write, the
        // characters written so far.
        type Frame = (Vec<(u32, usize)>, usize, String);
        let children = |n: usize| -> Vec<(u32, usize)> { trie.nodes[n].0.iter().map(|(&c, &k)| (c, k)).collect() };
        let mut stack: Vec<Frame> = vec![(children(0), 0, String::new())];
        while let Some((kids, next, rejects)) = stack.last_mut() {
            if *next < kids.len() {
                let (cpt, child) = kids[*next];
                *next += 1;
                let c = char::from_u32(cpt).expect("a trie of chars");
                rejects.push(c);
                if *next > 1 { out.push_str(" | "); }
                out.push('[');
                out.push(c);
                out.push(']');
                if !trie.nodes[child].0.is_empty() {
                    out.push_str(" (");
                    stack.push((children(child), 0, String::new()));
                } else {
                    out.push(' ');
                    out.push_str(&char_rule);
                    out.push('+');
                }
            } else {
                if !kids.is_empty() {
                    out.push_str(" | [^\"");
                    out.push_str(rejects);
                    out.push_str("] ");
                    out.push_str(&char_rule);
                    out.push('*');
                }
                stack.pop();
                if !stack.is_empty() { out.push(')'); }
            }
        }
        out.push_str(" )");
        if trie.nodes[0].1 < 0 { out.push('?'); }
        out.push_str(" [\"]");
        out
    }

    fn resolve_ref(&mut self, r: &str) -> Result<String, String> {
        let fragment = match r.find('#') { Some(i) => &r[i + 1..], None => r };
        let mut ref_name = format!("ref{}", rule_name_chars(fragment));
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(r) {
            let Some(target) = self.doc.and_then(|d| d.refs.get(r)) else {
                self.errors.push(format!("Unresolved $ref {r}"));
                return Ok(String::new());
            };
            self.refs_being_resolved.insert(r.to_string());
            ref_name = self.visit(target, &ref_name)?;
            self.refs_being_resolved.remove(r);
        }
        Ok(ref_name)
    }

    fn build_object_rule(&mut self, properties: &[(&'d str, &'d Node)], required: &HashSet<&str>, name: &str,
                         additional: Option<&'d Node>) -> Result<String, String> {
        let dash = if name.is_empty() { "" } else { "-" };
        let mut required_props: Vec<&str> = Vec::new();
        let mut optional_props: Vec<&str> = Vec::new();
        let mut prop_kv_rule_names: HashMap<&str, String> = HashMap::new();
        let mut prop_names: Vec<String> = Vec::new();
        for &(prop_name, prop_schema) in properties {
            let prop_rule_name = self.visit(prop_schema, &format!("{name}{dash}{prop_name}"))?;
            let kv = format!("{} space \":\" space {prop_rule_name}", format_literal(&dump_json(&Value::String(prop_name.to_string()))));
            let kv_name = self.add_rule(&format!("{name}{dash}{prop_name}-kv"), &kv);
            prop_kv_rule_names.insert(prop_name, kv_name);
            if required.contains(prop_name) { required_props.push(prop_name); } else { optional_props.push(prop_name); }
            prop_names.push(prop_name.to_string());
        }
        if let Some(additional) = additional {
            let sub_name = format!("{name}{dash}additional");
            let value_rule = if !matches!(additional, Node::Any) {
                self.visit(additional, &format!("{sub_name}-value"))?
            } else {
                self.add_primitive("value", primitive("value"))
            };
            let key_rule = if prop_names.is_empty() {
                self.add_primitive("string", primitive("string"))
            } else {
                let k = self.not_strings(&prop_names);
                self.add_rule(&format!("{sub_name}-k"), &k)
            };
            let kv_rule = self.add_rule(&format!("{sub_name}-kv"), &format!("{key_rule} \":\" space {value_rule}"));
            prop_kv_rule_names.insert("*", kv_rule);
            optional_props.push("*");
        }

        if required_props.is_empty() && optional_props.is_empty() { return Ok("\"{\" space \"}\"".into()); }

        let mut rule = String::from("\"{\" space ");
        for (i, k) in required_props.iter().enumerate() {
            if i > 0 { rule.push_str(" \",\" space "); }
            rule.push_str(&prop_kv_rule_names[k]);
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() { rule.push_str(" \",\" space ( "); }
            // llama.cpp's get_recursive_refs(ks, first_is_optional), without recursion. rest[j] is the rule
            // `<k_{j-1}>-rest` holding get_recursive_refs(ks[j..], true) = `( "," space kv_j )?` then rest[j+1];
            // built from the last one back, the order in which the recursion adds them. The recursion then
            // re-adds the same rules for every later i, which finds them unchanged (add_rule never rewrites a
            // rule), so building each once gives the same text.
            let n = optional_props.len();
            let mut rest: Vec<String> = vec![String::new(); n + 1];
            for j in (1..n).rev() {
                let k = optional_props[j];
                let mut opt = format!("( \",\" space {} ){}", prop_kv_rule_names[k], if k == "*" { "*" } else { "?" });
                if j + 1 < n { opt.push(' '); opt.push_str(&rest[j + 1]); }
                rest[j] = self.add_rule(&format!("{name}{dash}{}-rest", optional_props[j - 1]), &opt);
            }
            for i in 0..n {
                if i > 0 { rule.push_str(" | "); }
                let k = optional_props[i];
                let kv = &prop_kv_rule_names[k];
                rule.push_str(kv);
                if k == "*" { rule.push_str(&format!(" ( \",\" space {kv} )*")); }
                if i + 1 < n { rule.push(' '); rule.push_str(&rest[i + 1]); }
            }
            if !required_props.is_empty() { rule.push_str(" )"); }
            rule.push_str(" )?");
        }

        rule.push_str(" space \"}\"");
        Ok(rule)
    }

    fn visit_all_of(&mut self, children: &'d [Node], name: &str, rule_name: &str) -> Result<String, String> {
        let mut required: HashSet<&str> = HashSet::new();
        let mut properties: Vec<(&'d str, &'d Node)> = Vec::new();
        let mut enum_values: BTreeMap<String, usize> = BTreeMap::new();
        let doc = self.doc;
        // A $ref already being followed is not followed again (llama.cpp recurses forever on a cycle).
        fn add_component<'d>(doc: Option<&'d Document>, comp: &'d Node, is_required: bool, seen: &mut Vec<&'d str>,
                             required: &mut HashSet<&'d str>, properties: &mut Vec<(&'d str, &'d Node)>, enum_values: &mut BTreeMap<String, usize>) {
            match comp {
                Node::Ref(r) => {
                    if seen.contains(&r.as_str()) { return; }
                    if let Some(target) = doc.and_then(|d| d.refs.get(r)) {
                        seen.push(r);
                        add_component(doc, target, is_required, seen, required, properties, enum_values);
                        seen.pop();
                    }
                }
                Node::Object { properties: props, .. } => {
                    for p in props {
                        properties.push((&p.name, &p.schema));
                        if is_required { required.insert(&p.name); }
                    }
                }
                Node::Enum(values) => {
                    for v in values { *enum_values.entry(format_literal(&dump_json(v))).or_insert(0) += 1; }
                }
                _ => {}
            }
        }
        let mut seen = Vec::new();
        for child in children {
            if let Node::AnyOf(alts) = child {
                for alt in alts { add_component(doc, alt, false, &mut seen, &mut required, &mut properties, &mut enum_values); }
            } else {
                add_component(doc, child, true, &mut seen, &mut required, &mut properties, &mut enum_values);
            }
        }
        if !enum_values.is_empty() {
            let inter: Vec<&str> = enum_values.iter().filter(|(_, n)| **n == children.len()).map(|(k, _)| k.as_str()).collect();
            if !inter.is_empty() { return Ok(self.add_rule(rule_name, &format!("({})", inter.join(" | ")))); }
        }
        let rule = self.build_object_rule(&properties, &required, name, None)?;
        Ok(self.add_rule(rule_name, &rule))
    }

    fn visit(&mut self, schema: &'d Node, name: &str) -> Result<String, String> {
        if self.depth >= MAX_DEPTH { return Err(format!("schema nested more than {MAX_DEPTH} levels deep (at rule {name:?})")); }
        self.depth += 1;
        let r = self.visit_inner(schema, name);
        self.depth -= 1;
        r
    }

    fn visit_inner(&mut self, schema: &'d Node, name: &str) -> Result<String, String> {
        let rule_name = if is_reserved_name(name) { format!("{name}-") } else if name.is_empty() { "root".to_string() } else { name.to_string() };
        let sub_name = format!("{name}{}", if name.is_empty() { "" } else { "-" });
        match schema {
            Node::Ref(r) => {
                let target = self.resolve_ref(r)?;
                Ok(self.add_rule(&rule_name, &target))
            }
            Node::AnyOf(children) => {
                let u = self.generate_union_rule(name, children)?;
                Ok(self.add_rule(&rule_name, &u))
            }
            Node::AllOf(children) => self.visit_all_of(children, name, &rule_name),
            Node::Const(v) => Ok(self.add_rule(&rule_name, &format_literal(&dump_json(v)))),
            Node::Enum(values) => {
                let alts: Vec<String> = values.iter().map(|v| format_literal(&dump_json(v))).collect();
                Ok(self.add_rule(&rule_name, &format!("({})", alts.join(" | "))))
            }
            Node::Object { properties, additional } => {
                if properties.is_empty() && matches!(additional.as_deref(), Some(Node::Any)) {
                    let o = self.add_primitive("object", primitive("object"));
                    return Ok(self.add_rule(&rule_name, &o));
                }
                let props: Vec<(&'d str, &'d Node)> = properties.iter().map(|p| (p.name.as_str(), &p.schema)).collect();
                let required: HashSet<&str> = properties.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
                let rule = self.build_object_rule(&props, &required, name, additional.as_deref())?;
                Ok(self.add_rule(&rule_name, &rule))
            }
            Node::Tuple(items) => {
                let mut rule = String::from("\"[\" space ");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { rule.push_str(" \",\" space "); }
                    rule.push_str(&self.visit(item, &format!("{sub_name}tuple-{i}"))?);
                }
                rule.push_str(" space \"]\"");
                Ok(self.add_rule(&rule_name, &rule))
            }
            Node::Array { items, min_items, max_items } => {
                if matches!(**items, Node::Any) && *min_items == 0 && *max_items < 0 {
                    return Ok(self.visit_primitive(&rule_name, "array"));
                }
                let item_rule_name = self.visit(items, &format!("{sub_name}item"))?;
                let max = if *max_items < 0 { i32::MAX } else { *max_items };
                let rep = build_repetition(&item_rule_name, *min_items, max, "\",\" space");
                Ok(self.add_rule(&rule_name, &format!("\"[\" space {rep} space \"]\"")))
            }
            Node::String { pattern, format, min_length, max_length } => {
                if !pattern.is_empty() { return Ok(self.visit_pattern(pattern, &rule_name)); }
                if *format == Format::Uuid { return Ok(self.visit_primitive(&rule_name, "uuid")); }
                if *format != Format::None {
                    let prim = match format { Format::Date => "date-string", Format::Time => "time-string", _ => "date-time-string" };
                    let p = self.add_primitive(prim, format_rule(prim));
                    return Ok(self.add_rule(&rule_name, &p));
                }
                if *min_length > 0 || *max_length >= 0 {
                    let char_rule = self.add_primitive("char", primitive("char"));
                    let max = if *max_length < 0 { i32::MAX } else { *max_length };
                    let rep = build_repetition(&char_rule, *min_length, max, "");
                    return Ok(self.add_rule(&rule_name, &format!("\"\\\"\" {rep} \"\\\"\"")));
                }
                Ok(self.visit_primitive(&rule_name, "string"))
            }
            Node::Integer { minimum, maximum } => {
                if *minimum == i64::MIN && *maximum == i64::MAX { return Ok(self.visit_primitive(&rule_name, "integer")); }
                let mut out = String::from("(");
                build_min_max_int(*minimum, *maximum, &mut out, 16, true)?;
                out.push(')');
                Ok(self.add_rule(&rule_name, &out))
            }
            Node::Number => Ok(self.visit_primitive(&rule_name, "number")),
            Node::Boolean => Ok(self.visit_primitive(&rule_name, "boolean")),
            Node::Null => Ok(self.visit_primitive(&rule_name, "null")),
            Node::Any => {
                let v = self.add_primitive("value", primitive("value"));
                Ok(self.add_rule(&rule_name, &v))
            }
        }
    }
}

/// Deepest parenthesis nesting of GBNF source, outside literals and character classes.
fn grammar_nesting(src: &str) -> usize {
    let (mut depth, mut max, mut in_lit, mut in_class, mut esc) = (0usize, 0usize, false, false, false);
    for c in src.chars() {
        if esc { esc = false; continue; }
        match c {
            '\\' if in_lit || in_class => esc = true,
            '"' if !in_class => in_lit = !in_lit,
            '[' if !in_lit && !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_lit && !in_class => { depth += 1; max = max.max(depth); }
            ')' if !in_lit && !in_class => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    max
}

/// Convert a JSON Schema document to GBNF, start symbol `root`, with the warnings llama.cpp prints about
/// patterns it widened to any string (`json_schema_to_grammar(schema, force_gbnf = true)`).
pub fn schema_to_gbnf_with_warnings(schema: &Value) -> Result<(String, Vec<String>), String> {
    let doc = parse_schema(schema).map_err(|e| format!("JSON schema conversion failed:\n{e}"))?;
    let mut c = Converter::new(false, Some(&doc));
    c.visit(&doc.root, "").map_err(|e| format!("JSON schema conversion failed:\n{e}"))?;
    let (g, warnings) = c.finish()?;
    let nesting = grammar_nesting(&g);
    if nesting > MAX_GRAMMAR_NESTING {
        return Err(format!("JSON schema conversion failed:\nthe grammar nests {nesting} parentheses deep, more than {MAX_GRAMMAR_NESTING}"));
    }
    Ok((g, warnings))
}

/// Convert a JSON Schema document to GBNF (start symbol `root`), as llama.cpp's `json_schema_to_grammar`.
pub fn schema_to_gbnf(schema: &Value) -> Result<String, String> { schema_to_gbnf_with_warnings(schema).map(|(g, _)| g) }
