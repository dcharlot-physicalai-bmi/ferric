//! Tool-calling: advertise OpenAI `tools` to the model in the Hermes/qwen format and parse the
//! model's output back into OpenAI-shaped `tool_calls`. Model-agnostic and wasm-clean.
use serde_json::{json, Value};

/// Hermes/qwen tool system prompt: advertise the tools as `<tools>…</tools>` and ask for
/// `<tool_call>{"name":…,"arguments":…}</tool_call>` back — the format Qwen/Hermes models are trained
/// on and vLLM's `hermes`/`qwen25` parsers expect.
pub fn hermes_prompt(tools: &[Value]) -> String {
    let mut s = String::from("You are a function-calling AI. You are given function signatures inside <tools></tools>. \
        To call a function, emit a JSON object {\"name\": <name>, \"arguments\": <args>} inside <tool_call></tool_call> tags. \
        You may emit multiple <tool_call> blocks. Only call a function when it is needed.\n<tools>\n");
    for t in tools { s.push_str(&t.to_string()); s.push('\n'); }
    s.push_str("</tools>");
    s
}

/// One `{"name":…,"arguments":…}` object → an OpenAI tool_call (arguments as a JSON *string*).
fn push_call(calls: &mut Vec<Value>, v: &Value) {
    let name = v["name"].as_str().unwrap_or("").to_string();
    if name.is_empty() { return; }
    let args = v.get("arguments").map(|a| a.to_string()).unwrap_or_else(|| "{}".into());
    let id = format!("call_{}", calls.len());
    calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}));
}

/// Top-level balanced `{…}` spans in `s` (string-aware, so braces inside JSON strings don't confuse it).
fn balanced_objects(s: &str) -> Vec<&str> {
    let (b, mut out, mut depth, mut start, mut in_str, mut esc) = (s.as_bytes(), Vec::new(), 0i32, 0usize, false, false);
    for (i, &c) in b.iter().enumerate() {
        if in_str { if esc { esc = false; } else if c == b'\\' { esc = true; } else if c == b'"' { in_str = false; } continue; }
        match c {
            b'"' => in_str = true,
            b'{' => { if depth == 0 { start = i; } depth += 1; }
            b'}' => { depth -= 1; if depth == 0 { out.push(&s[start..=i]); } }
            _ => {}
        }
    }
    out
}

/// Parse tool calls from generated text. Pass 1: well-formed `<tool_call>{json}</tool_call>` tags
/// (Hermes emits **multiple concatenated tags**, not an array). Pass 2 (fallback — reasoning models
/// leak/mangle the tags): any balanced `{…}` carrying both `name` and `arguments`; only reached when
/// no clean tag pair parsed, so it won't eat normal content.
pub fn parse_tool_calls(text: &str) -> Vec<Value> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(a) = rest.find("<tool_call>") {
        let after = &rest[a + "<tool_call>".len()..];
        let Some(b) = after.find("</tool_call>") else { break };
        if let Ok(v) = serde_json::from_str::<Value>(after[..b].trim()) { push_call(&mut calls, &v); }
        rest = &after[b + "</tool_call>".len()..];
    }
    if calls.is_empty() {
        for obj in balanced_objects(text) {
            if let Ok(v) = serde_json::from_str::<Value>(obj) {
                if v.get("name").is_some() && v.get("arguments").is_some() { push_call(&mut calls, &v); }
            }
        }
    }
    calls
}

/// **Tool calls in whichever format the model was trained to write** — read from its own chat template,
/// so the parser must follow the model, not the other way round. Detects, in order:
///
/// | family | syntax |
/// |---|---|
/// | Qwen3.5 / Qwen3-Coder / Nemotron | `<tool_call><function=NAME><parameter=P>VALUE</parameter>…</function></tool_call>` |
/// | Hermes / Qwen2.5 | `<tool_call>{"name":…,"arguments":…}</tool_call>` (a doubled `{{…}}` is unwrapped) |
/// | Gemma 4 | `<\|tool_call>call:NAME{key:value,…}<tool_call\|>`, strings in `<\|"\|>…<\|"\|>` |
/// | LFM2 | `<\|tool_call_start\|>[name(a='v', b=1), …]<\|tool_call_end\|>` |
/// | Mistral | `[TOOL_CALLS][{"name":…,"arguments":…}]` or `[TOOL_CALLS]NAME[ARGS]{…}` |
/// | DeepSeek V3 | `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME` + a ```json block |
/// | Llama 3.x | `<\|python_tag\|>{"name":…,"parameters":…}` (several joined by `;`), or the bare object |
///
/// `text` must be the decode WITH special tokens (Gemma's delimiters are specials). `tools` supplies each
/// parameter's JSON Schema type, so an XML or Python-style value becomes the type the tool declared.
/// Returns the text before the first call (what the client shows as `content`) and the OpenAI-shaped
/// calls (`function.arguments` a JSON string).
pub fn parse_tool_calls_any(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
    let ptype = |func: &str, param: &str| -> Option<String> {
        tools.iter().find(|t| t["function"]["name"] == func)
            .and_then(|t| t["function"]["parameters"]["properties"][param]["type"].as_str().map(String::from))
    };
    let typed = |func: &str, param: &str, raw: &str| -> Value {
        match ptype(func, param).as_deref() {
            Some("string") => Value::String(raw.to_string()),
            _ => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())),
        }
    };
    let mut calls: Vec<Value> = Vec::new();
    let add = |calls: &mut Vec<Value>, name: &str, args: Value| {
        if name.is_empty() { return; }
        let id = format!("call_{}", calls.len());
        calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}}));
    };
    let before = |marker: &str| text.find(marker).map(|i| text[..i].trim().to_string()).unwrap_or_default();

    // Qwen3-Coder XML (its <tool_call> wraps <function=…>, not JSON)
    if text.contains("<function=") {
        let mut rest = text;
        while let Some(a) = rest.find("<function=") {
            let after = &rest[a + "<function=".len()..];
            let Some(nend) = after.find('>') else { break };
            let name = after[..nend].trim().to_string();
            let body_end = after.find("</function>").unwrap_or(after.len());
            let body = &after[nend + 1..body_end];
            let mut args = serde_json::Map::new();
            let mut b = body;
            while let Some(pa) = b.find("<parameter=") {
                let pafter = &b[pa + "<parameter=".len()..];
                let Some(pn) = pafter.find('>') else { break };
                let pname = pafter[..pn].trim().to_string();
                let vend = pafter.find("</parameter>").unwrap_or(pafter.len());
                let mut val = &pafter[pn + 1..vend];
                // The template writes the value between newlines; they are framing, not content.
                if let Some(v) = val.strip_prefix('\n') { val = v; }
                if let Some(v) = val.strip_suffix('\n') { val = v; }
                args.insert(pname.clone(), typed(&name, &pname, val));
                b = &pafter[(vend + "</parameter>".len()).min(pafter.len())..];
            }
            add(&mut calls, &name, Value::Object(args));
            rest = &after[(body_end + "</function>".len()).min(after.len())..];
        }
        if !calls.is_empty() {
            let m = if text.contains("<tool_call>") { "<tool_call>" } else { "<function=" };
            return (before(m), calls);
        }
    }
    // Hermes JSON
    if text.contains("<tool_call>") {
        let mut rest = text;
        while let Some(a) = rest.find("<tool_call>") {
            let after = &rest[a + "<tool_call>".len()..];
            let end = after.find("</tool_call>").unwrap_or(after.len());
            let raw = after[..end].trim();
            // A small model sometimes doubles the braces it saw in the template's instructions.
            let v = serde_json::from_str::<Value>(raw).ok().or_else(|| {
                raw.strip_prefix('{').and_then(|r| r.strip_suffix('}')).and_then(|r| serde_json::from_str(r.trim()).ok())
            });
            if let Some(v) = v { add(&mut calls, v["name"].as_str().unwrap_or(""), v.get("arguments").cloned().unwrap_or(json!({}))); }
            rest = &after[(end + "</tool_call>".len()).min(after.len())..];
        }
        if !calls.is_empty() { return (before("<tool_call>"), calls); }
    }
    // Gemma 4
    if let Some(p) = text.find("call:").filter(|&p| text[..p].ends_with("<|tool_call>") || text.contains("<|tool_call>")) {
        let mut rest = &text[p..];
        while let Some(a) = rest.find("call:") {
            let after = &rest[a + "call:".len()..];
            let Some(ob) = after.find('{') else { break };
            let name = after[..ob].trim().to_string();
            let mut i = ob;
            if let Some(v) = gemma_value(after.as_bytes(), &mut i, after) { add(&mut calls, &name, v); }
            rest = &after[i.min(after.len())..];
        }
        if !calls.is_empty() { return (before("<|tool_call>").trim_end_matches("call:").trim().to_string(), calls); }
    }
    // LFM2 Python-style list
    if let Some(a) = text.find("<|tool_call_start|>") {
        let after = &text[a + "<|tool_call_start|>".len()..];
        let body = &after[..after.find("<|tool_call_end|>").unwrap_or(after.len())];
        for (name, args) in python_calls(body) {
            let obj: serde_json::Map<String, Value> = args.into_iter().collect();
            add(&mut calls, &name, Value::Object(obj));
        }
        if !calls.is_empty() { return (before("<|tool_call_start|>"), calls); }
    }
    // Mistral
    if let Some(a) = text.find("[TOOL_CALLS]") {
        let after = text[a + "[TOOL_CALLS]".len()..].trim_start();
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(balanced_array(after).unwrap_or("")) {
            for v in arr { add(&mut calls, v["name"].as_str().unwrap_or(""), v.get("arguments").cloned().unwrap_or(json!({}))); }
        } else {
            let mut rest = &text[a..];
            while let Some(b) = rest.find("[TOOL_CALLS]") {
                let after = &rest[b + "[TOOL_CALLS]".len()..];
                let Some(ai) = after.find("[ARGS]") else { break };
                let name = after[..ai].trim().to_string();
                let objs = balanced_objects(&after[ai + "[ARGS]".len()..]);
                if let Some(o) = objs.first() {
                    if let Ok(v) = serde_json::from_str::<Value>(o) { add(&mut calls, &name, v); }
                }
                rest = &after[ai + "[ARGS]".len()..];
            }
        }
        if !calls.is_empty() { return (before("[TOOL_CALLS]"), calls); }
    }
    // DeepSeek V3
    if text.contains("<｜tool▁call▁begin｜>") {
        for seg in text.split("<｜tool▁call▁begin｜>").skip(1) {
            let seg = seg.split("<｜tool▁call▁end｜>").next().unwrap_or("");
            let Some((_, rest)) = seg.split_once("<｜tool▁sep｜>") else { continue };
            let name = rest.lines().next().unwrap_or("").trim().to_string();
            if let Some(o) = balanced_objects(rest).first() {
                if let Ok(v) = serde_json::from_str::<Value>(o) { add(&mut calls, &name, v); }
            }
        }
        if !calls.is_empty() {
            let m = if text.contains("<｜tool▁calls▁begin｜>") { "<｜tool▁calls▁begin｜>" } else { "<｜tool▁call▁begin｜>" };
            return (before(m), calls);
        }
    }
    // Llama 3.x (and the bare-object fallback for any model): {"name":…, "parameters"|"arguments":…}
    let scope = text.find("<|python_tag|>").map(|a| &text[a + "<|python_tag|>".len()..]).unwrap_or(text);
    for obj in balanced_objects(scope) {
        if let Ok(v) = serde_json::from_str::<Value>(obj) {
            if let (Some(name), Some(args)) = (v["name"].as_str(), v.get("parameters").or_else(|| v.get("arguments"))) {
                add(&mut calls, name, args.clone());
            }
        }
    }
    if !calls.is_empty() {
        let m = if text.contains("<|python_tag|>") { "<|python_tag|>" } else { "{" };
        return (before(m), calls);
    }
    (text.to_string(), calls)
}

/// The first balanced `[…]` at the start of `s` (string-aware).
fn balanced_array(s: &str) -> Option<&str> {
    let b = s.as_bytes();
    if b.first() != Some(&b'[') { return None; }
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, &c) in b.iter().enumerate() {
        if in_str { if esc { esc = false } else if c == b'\\' { esc = true } else if c == b'"' { in_str = false } continue; }
        match c { b'"' => in_str = true, b'[' | b'{' => depth += 1, b']' | b'}' => { depth -= 1; if depth == 0 { return Some(&s[..=i]); } } _ => {} }
    }
    None
}

/// One Gemma-4 argument value starting at `s[*i]`: `{k:v,…}` with bare keys, `[…]`, `<|"|>string<|"|>`,
/// or a bare number / true / false / null.
fn gemma_value(b: &[u8], i: &mut usize, s: &str) -> Option<Value> {
    const Q: &str = "<|\"|>";
    let skip_ws = |i: &mut usize| while *i < b.len() && (b[*i] as char).is_whitespace() { *i += 1 };
    skip_ws(i);
    if s[*i..].starts_with(Q) {
        let start = *i + Q.len();
        let end = s[start..].find(Q).map(|e| start + e)?;
        *i = end + Q.len();
        return Some(Value::String(s[start..end].to_string()));
    }
    match b.get(*i)? {
        b'{' => {
            *i += 1;
            let mut m = serde_json::Map::new();
            loop {
                skip_ws(i);
                if b.get(*i) == Some(&b'}') { *i += 1; return Some(Value::Object(m)); }
                let ks = *i;
                while *i < b.len() && b[*i] != b':' { *i += 1; }
                let key = s[ks..*i].trim().trim_matches('"').to_string();
                *i += 1;
                let v = gemma_value(b, i, s)?;
                m.insert(key, v);
                skip_ws(i);
                match b.get(*i) { Some(b',') => *i += 1, Some(b'}') => { *i += 1; return Some(Value::Object(m)); } _ => return Some(Value::Object(m)) }
            }
        }
        b'[' => {
            *i += 1;
            let mut a = Vec::new();
            loop {
                skip_ws(i);
                if b.get(*i) == Some(&b']') { *i += 1; return Some(Value::Array(a)); }
                a.push(gemma_value(b, i, s)?);
                skip_ws(i);
                match b.get(*i) { Some(b',') => *i += 1, Some(b']') => { *i += 1; return Some(Value::Array(a)); } _ => return Some(Value::Array(a)) }
            }
        }
        _ => {
            let st = *i;
            while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']') { *i += 1; }
            let raw = s[st..*i].trim();
            Some(serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())))
        }
    }
}

/// `name(a='v', b=1, c=True), other()` → [(name, [(a, v), …]), …]. Values: Python-quoted strings, numbers,
/// True/False/None, and JSON for lists and dicts (what LFM2's template writes for mappings).
fn python_calls(body: &str) -> Vec<(String, Vec<(String, Value)>)> {
    let s = body.trim().trim_start_matches('[').trim_end_matches(']');
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        while i < b.len() && (b[i] == b',' || (b[i] as char).is_whitespace()) { i += 1; }
        let ns = i;
        while i < b.len() && b[i] != b'(' { i += 1; }
        if i >= b.len() { break; }
        let name = s[ns..i].trim().to_string();
        i += 1;
        let mut args = Vec::new();
        loop {
            while i < b.len() && (b[i] == b',' || (b[i] as char).is_whitespace()) { i += 1; }
            if i >= b.len() || b[i] == b')' { i += 1; break; }
            let ks = i;
            while i < b.len() && b[i] != b'=' { i += 1; }
            let key = s[ks..i].trim().to_string();
            i += 1;
            let vs = i;
            let (mut depth, mut q) = (0i32, 0u8);
            while i < b.len() {
                let c = b[i];
                if q != 0 { if c == b'\\' { i += 2; continue; } if c == q { q = 0; } i += 1; continue; }
                match c { b'\'' | b'"' => q = c, b'[' | b'{' | b'(' => depth += 1, b']' | b'}' => depth -= 1,
                          b')' if depth == 0 => break, b')' => depth -= 1, b',' if depth == 0 => break, _ => {} }
                i += 1;
            }
            let raw = s[vs..i.min(b.len())].trim();
            let v = if (raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2) || (raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2) {
                Value::String(raw[1..raw.len() - 1].replace("\\'", "'").replace("\\\"", "\"").replace("\\n", "\n").replace("\\\\", "\\"))
            } else {
                match raw { "True" => json!(true), "False" => json!(false), "None" => Value::Null,
                            _ => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())) }
            };
            args.push((key, v));
        }
        out.push((name, args));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clean_tags() {
        let c = parse_tool_calls("<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1}}</tool_call>");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["function"]["name"], "add");
        assert_eq!(c[0]["function"]["arguments"], "{\"a\":1}");
    }
    #[test]
    fn leaked_tag_fallback() {
        // reasoning model wrapped it in <think> and dropped the opening tag
        let c = parse_tool_calls("<think>\n{\"name\": \"get\", \"arguments\": {\"x\": 2}}\n</tool_call>");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["function"]["name"], "get");
    }
    #[test]
    fn no_false_positive() {
        assert!(parse_tool_calls("Just a normal answer with a { brace }.").is_empty());
    }

    fn weather() -> Vec<Value> {
        vec![json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "zip": {"type": "string"}}}}})]
    }
    fn one(text: &str) -> (String, String, Value) {
        let (content, c) = parse_tool_calls_any(text, &weather());
        assert_eq!(c.len(), 1, "one call expected from {text:?}, got {c:?}");
        let args: Value = serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        (content, c[0]["function"]["name"].as_str().unwrap().to_string(), args)
    }

    /// Each family's format, written as that family's OWN chat template writes an assistant tool call.
    #[test]
    fn every_family_parses_to_the_same_call() {
        let want = json!({"city": "Paris", "days": 3});
        // Qwen3.5 / Nemotron XML — a string-typed "12345" must stay a string, an integer becomes a number
        let (_, n, a) = one("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>");
        assert_eq!((n.as_str(), &a), ("get_weather", &want));
        let (_, _, a) = one("<tool_call>\n<function=get_weather>\n<parameter=zip>\n12345\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(a["zip"], "12345", "the schema says string: a numeric-looking value must stay a string");
        // Hermes / Qwen2.5, and the doubled braces a 0.5B model wrote live
        assert_eq!(one("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>").2, want);
        assert_eq!(one("<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}}\n</tool_call>").2, want);
        // Gemma 4
        assert_eq!(one("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>").2, want);
        // LFM2
        assert_eq!(one("<|tool_call_start|>[get_weather(city='Paris', days=3)]<|tool_call_end|>").2, want);
        // Mistral, both generations
        assert_eq!(one("[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}]").2, want);
        assert_eq!(one("[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", \"days\": 3}").2, want);
        // DeepSeek V3
        assert_eq!(one("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\", \"days\": 3}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>").2, want);
        // Llama 3.2, exactly as it came out live (parameters, not arguments; ends in <|eom_id|>)
        let (c, n, a) = one("<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"days\": 3}}<|eom_id|>");
        assert_eq!((c.as_str(), n.as_str(), &a), ("", "get_weather", &want));
    }

    #[test]
    fn text_before_a_call_is_content_and_plain_text_is_no_call() {
        let (c, calls) = parse_tool_calls_any("Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>", &weather());
        assert_eq!((c.as_str(), calls.len()), ("Let me check.", 1));
        let (c, calls) = parse_tool_calls_any("The weather is mild {usually}.", &weather());
        assert!(calls.is_empty());
        assert_eq!(c, "The weather is mild {usually}.");
        let (_, two) = parse_tool_calls_any("<|tool_call_start|>[get_weather(city='A'), get_weather(city='B')]<|tool_call_end|>", &weather());
        assert_eq!(two.len(), 2);
    }
}
