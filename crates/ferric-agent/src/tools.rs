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
}
