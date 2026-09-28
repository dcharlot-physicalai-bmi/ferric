//! **Two more API dialects on the same chat core**: Anthropic's Messages API (`/v1/messages`, offered by
//! 13 of 14 serving peers) and OpenAI's Responses API (`/v1/responses`, 12 of 14; n8n's OpenAI node
//! defaults to it). Each request is translated into the OpenAI chat shape and run through `run_chat`, so
//! a conversation means the same thing whichever door it came through — template, tool formats, reasoning
//! split, stop strings, joules. Images are refused by name, as everywhere else.
use crate::{run_chat, write_json, write_sse_headers, send_sse, mcp, Engine, ChatResult};
use serde_json::{json, Value};
use std::io::Write;
use std::net::TcpStream;

fn bad(stream: &mut TcpStream, anthropic: bool, m: &str) {
    if anthropic {
        write_json(stream, 400, &json!({"type": "error", "error": {"type": "invalid_request_error", "message": m}}))
    } else {
        write_json(stream, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}}))
    }
}

fn rand_id(prefix: &str) -> String {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}{:024x}", n ^ 0x5bd1_e995_9e37_79b9_7f4a_7c15)
}

// ─────────────────────────────── Anthropic Messages ───────────────────────────────

/// Anthropic content (a string or blocks) → the OpenAI messages it stands for. `tool_result` blocks become
/// `tool` messages; an assistant's `tool_use` blocks become its `tool_calls`.
fn anthropic_to_openai(req: &Value) -> Result<Value, String> {
    let mut msgs: Vec<Value> = Vec::new();
    match &req["system"] {
        Value::Null => {}
        Value::String(s) => msgs.push(json!({"role": "system", "content": s})),
        Value::Array(blocks) => {
            let t: Vec<&str> = blocks.iter().filter_map(|b| b["text"].as_str()).collect();
            msgs.push(json!({"role": "system", "content": t.join("\n")}));
        }
        v => return Err(format!("`system` must be a string or text blocks, got {v}")),
    }
    for (i, m) in req["messages"].as_array().ok_or("`messages` is required")?.iter().enumerate() {
        let role = m["role"].as_str().ok_or(format!("messages[{i}].role is required"))?;
        match &m["content"] {
            Value::String(s) => msgs.push(json!({"role": role, "content": s})),
            Value::Array(blocks) => {
                let (mut text, mut calls) = (Vec::new(), Vec::new());
                // An image block keeps its place among the text: the content becomes parts, which the
                // vision path reads (or refuses by name on a model without one).
                let mut parts: Vec<Value> = Vec::new();
                for b in blocks {
                    match b["type"].as_str() {
                        Some("text") => {
                            text.push(b["text"].as_str().unwrap_or("").to_string());
                            parts.push(json!({"type": "text", "text": b["text"]}));
                        }
                        Some("image") => parts.push(b.clone()),
                        Some("tool_use") => calls.push(json!({"id": b["id"], "type": "function",
                            "function": {"name": b["name"], "arguments": b["input"].to_string()}})),
                        Some("tool_result") => {
                            let c = match &b["content"] {
                                Value::String(s) => s.clone(),
                                Value::Array(a) => a.iter().filter_map(|x| x["text"].as_str()).collect::<Vec<_>>().join("\n"),
                                _ => String::new(),
                            };
                            msgs.push(json!({"role": "tool", "tool_call_id": b["tool_use_id"], "content": c}));
                        }
                        Some("thinking") | Some("redacted_thinking") => {} // prior reasoning is not re-fed
                        Some("document") => return Err(format!("messages[{i}] has a `document` block; documents are not read here")),
                        Some(t) => return Err(format!("messages[{i}] has an unknown block type `{t}`")),
                        None => return Err(format!("messages[{i}] has a block without `type`")),
                    }
                }
                let has_image = parts.iter().any(|p| p["type"] == "image");
                if !text.is_empty() || !calls.is_empty() || has_image {
                    let mut o = json!({"role": role, "content": if has_image { Value::Array(parts) } else { json!(text.join("\n")) }});
                    if !calls.is_empty() { o["tool_calls"] = json!(calls); }
                    msgs.push(o);
                }
            }
            v => return Err(format!("messages[{i}].content must be a string or blocks, got {v}")),
        }
    }
    let mut r = json!({"messages": msgs});
    let max = req["max_tokens"].as_u64().ok_or("`max_tokens` is required (Anthropic's API has no default)")?;
    r["max_tokens"] = json!(max);
    // Carried so the model name can select a LoRA adapter (`Engine::gen_opts`).
    for k in ["model", "lora"] { if !req[k].is_null() { r[k] = req[k].clone(); } }
    for (a, o) in [("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop")] {
        if !req[a].is_null() { r[o] = req[a].clone(); }
    }
    if let Some(tools) = req["tools"].as_array() {
        r["tools"] = json!(tools.iter().map(|t| json!({"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}})).collect::<Vec<_>>());
    }
    match req["thinking"]["type"].as_str() {
        Some("enabled") => { r["chat_template_kwargs"] = json!({"enable_thinking": true}); }
        Some("disabled") => { r["chat_template_kwargs"] = json!({"enable_thinking": false}); }
        _ => {}
    }
    Ok(r)
}

fn stop_reason(r: &ChatResult) -> (&'static str, Value) {
    if !r.tool_calls.is_empty() { return ("tool_use", Value::Null); }
    if let Some(s) = &r.stop_seq { return ("stop_sequence", json!(s)); }
    if r.finish == "length" { ("max_tokens", Value::Null) } else { ("end_turn", Value::Null) }
}

fn anthropic_blocks(r: &ChatResult) -> Vec<Value> {
    let mut v = Vec::new();
    if !r.reasoning.is_empty() { v.push(json!({"type": "thinking", "thinking": r.reasoning, "signature": ""})); }
    if !r.text.is_empty() { v.push(json!({"type": "text", "text": r.text})); }
    for c in &r.tool_calls {
        let input: Value = c["function"]["arguments"].as_str().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(json!({}));
        v.push(json!({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": input}));
    }
    v
}

/// A failed write marks the client gone, which stops the generation feeding it (see `crate::mark_peer_gone`).
fn send_event(stream: &mut TcpStream, ev: &str, data: &Value) {
    if stream.write_all(format!("event: {ev}\ndata: {data}\n\n").as_bytes()).is_err() || stream.flush().is_err() { crate::mark_peer_gone(); }
}

pub(crate) fn messages(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, body: &[u8], stream: &mut TcpStream) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad(stream, true, &format!("bad json: {e}")) };
    let r = match anthropic_to_openai(&req) { Ok(r) => r, Err(e) => return bad(stream, true, &e) };
    let empty = vec![];
    if let Err(e) = eng.gen_opts(&r, true).and_then(|_| eng.chat_ids(r["messages"].as_array().unwrap_or(&empty)).map(|_| ())) {
        return bad(stream, true, &e);
    }
    let id = rand_id("msg_");
    let model = req["model"].as_str().map(String::from).unwrap_or_else(|| eng.name.clone());
    if req["stream"].as_bool() != Some(true) {
        return match run_chat(eng, mcps, &r, |_, _, _| {}) {
            Err(e) => bad(stream, true, &e),
            Ok(res) => {
                let (reason, seq) = stop_reason(&res);
                write_json(stream, 200, &json!({"id": id, "type": "message", "role": "assistant", "model": model,
                    "content": anthropic_blocks(&res), "stop_reason": reason, "stop_sequence": seq,
                    "usage": {"input_tokens": res.prompt_tokens, "output_tokens": res.gen_tokens}, "energy": res.energy}))
            }
        };
    }
    // Streaming: message_start, then one block per kind as it begins (thinking → text), deltas, stops.
    write_sse_headers(stream);
    send_event(stream, "message_start", &json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant",
        "model": model, "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null, "usage": {"input_tokens": 0, "output_tokens": 0}}}));
    let mut open: Option<(usize, bool)> = None; // (index, is_thinking)
    let mut next = 0usize;
    let res = run_chat(eng, mcps, &r, |d, _, thinking| {
        if open.map(|(_, t)| t) != Some(thinking) {
            if let Some((i, _)) = open { send_event(stream, "content_block_stop", &json!({"type": "content_block_stop", "index": i})); }
            let block = if thinking { json!({"type": "thinking", "thinking": ""}) } else { json!({"type": "text", "text": ""}) };
            send_event(stream, "content_block_start", &json!({"type": "content_block_start", "index": next, "content_block": block}));
            open = Some((next, thinking));
            next += 1;
        }
        let delta = if thinking { json!({"type": "thinking_delta", "thinking": d}) } else { json!({"type": "text_delta", "text": d}) };
        send_event(stream, "content_block_delta", &json!({"type": "content_block_delta", "index": open.unwrap().0, "delta": delta}));
    });
    if let Some((i, _)) = open { send_event(stream, "content_block_stop", &json!({"type": "content_block_stop", "index": i})); }
    match res {
        Err(e) => send_event(stream, "error", &json!({"type": "error", "error": {"type": "api_error", "message": e}})),
        Ok(res) => {
            // Tool calls are known only at the end; each goes out as a complete tool_use block.
            for c in &res.tool_calls {
                let input: Value = c["function"]["arguments"].as_str().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(json!({}));
                send_event(stream, "content_block_start", &json!({"type": "content_block_start", "index": next,
                    "content_block": {"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": {}}}));
                send_event(stream, "content_block_delta", &json!({"type": "content_block_delta", "index": next,
                    "delta": {"type": "input_json_delta", "partial_json": input.to_string()}}));
                send_event(stream, "content_block_stop", &json!({"type": "content_block_stop", "index": next}));
                next += 1;
            }
            let (reason, seq) = stop_reason(&res);
            send_event(stream, "message_delta", &json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": seq},
                "usage": {"input_tokens": res.prompt_tokens, "output_tokens": res.gen_tokens}, "energy": res.energy}));
        }
    }
    send_event(stream, "message_stop", &json!({"type": "message_stop"}));
}

/// `/v1/messages/count_tokens` — the prompt's size under the model's own template, before sending it.
pub(crate) fn count_tokens(eng: &Engine, body: &[u8], stream: &mut TcpStream) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad(stream, true, &format!("bad json: {e}")) };
    let mut req = req;
    if req["max_tokens"].is_null() { req["max_tokens"] = json!(1); }
    let r = match anthropic_to_openai(&req) { Ok(r) => r, Err(e) => return bad(stream, true, &e) };
    let empty = vec![];
    let tools = r["tools"].as_array().filter(|t| !t.is_empty() && eng.template_handles_tools()).map(|t| t.as_slice());
    match eng.chat_ids_with(r["messages"].as_array().unwrap_or(&empty), tools, &Default::default()) {
        Ok(ids) => write_json(stream, 200, &json!({"input_tokens": ids.len()})),
        Err(e) => bad(stream, true, &e),
    }
}

// ─────────────────────────────── OpenAI Responses ───────────────────────────────

/// Responses kept for `previous_response_id`: (id, the full conversation after it — input and output as
/// chat messages). Bounded; the oldest is forgotten first.
pub(crate) struct ResponseStore(std::sync::Mutex<std::collections::VecDeque<(String, Vec<Value>, Value)>>);
impl ResponseStore {
    pub fn new() -> Self { ResponseStore(std::sync::Mutex::new(Default::default())) }
    fn get(&self, id: &str) -> Option<(Vec<Value>, Value)> {
        self.0.lock().ok()?.iter().find(|(i, _, _)| i == id).map(|(_, m, r)| (m.clone(), r.clone()))
    }
    fn put(&self, id: String, msgs: Vec<Value>, resp: Value) {
        if let Ok(mut q) = self.0.lock() { q.push_back((id, msgs, resp)); while q.len() > 256 { q.pop_front(); } }
    }
}

/// A Responses item's content for a chat message: its text, or — when it carries an `input_image` — its
/// parts, which the vision path reads.
fn item_content(content: &Value) -> Result<Value, String> {
    if let Value::Array(parts) = content {
        if parts.iter().any(|p| p["type"] == "input_image") {
            return parts.iter().map(|p| match p["type"].as_str() {
                Some("input_image") => Ok(p.clone()),
                _ => item_text(&Value::Array(vec![p.clone()])).map(|t| json!({"type": "text", "text": t})),
            }).collect::<Result<Vec<_>, _>>().map(Value::Array);
        }
    }
    item_text(content).map(Value::String)
}

fn item_text(content: &Value) -> Result<String, String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut v = Vec::new();
            for p in parts {
                match p["type"].as_str() {
                    Some("input_text" | "output_text" | "text") => v.push(p["text"].as_str().unwrap_or("").to_string()),
                    Some(t @ ("input_file" | "input_audio")) => return Err(format!("a `{t}` part: files and audio are not read here")),
                    Some("refusal") => v.push(p["refusal"].as_str().unwrap_or("").to_string()),
                    t => return Err(format!("unknown content part {t:?}")),
                }
            }
            Ok(v.join("\n"))
        }
        Value::Null => Ok(String::new()),
        v => Err(format!("content must be a string or parts, got {v}")),
    }
}

/// Responses `input` items → chat messages (appended to what `previous_response_id` carried).
fn responses_to_messages(req: &Value, mut msgs: Vec<Value>) -> Result<Vec<Value>, String> {
    if let Some(ins) = req["instructions"].as_str() {
        msgs.retain(|m| m["role"] != "system");
        msgs.insert(0, json!({"role": "system", "content": ins}));
    }
    match &req["input"] {
        Value::String(s) => msgs.push(json!({"role": "user", "content": s})),
        Value::Array(items) => for it in items {
            match it["type"].as_str() {
                None | Some("message") => {
                    let role = match it["role"].as_str() { Some("developer") => "system", Some(r) => r, None => "user" };
                    msgs.push(json!({"role": role, "content": item_content(&it["content"])?}));
                }
                Some("function_call") => msgs.push(json!({"role": "assistant", "content": "", "tool_calls": [{"id": it["call_id"],
                    "type": "function", "function": {"name": it["name"], "arguments": it["arguments"]}}]})),
                Some("function_call_output") => msgs.push(json!({"role": "tool", "tool_call_id": it["call_id"],
                    "content": it["output"].as_str().map(String::from).unwrap_or_else(|| it["output"].to_string())})),
                Some("reasoning") => {}
                Some(t) => return Err(format!("input item type `{t}` is not supported here")),
            }
        },
        Value::Null => return Err("`input` is required".into()),
        v => return Err(format!("`input` must be a string or items, got {v}")),
    }
    Ok(msgs)
}

fn responses_output(res: &ChatResult) -> Vec<Value> {
    let mut out = Vec::new();
    if !res.reasoning.is_empty() {
        out.push(json!({"type": "reasoning", "id": rand_id("rs_"), "summary": [],
                        "content": [{"type": "reasoning_text", "text": res.reasoning}]}));
    }
    if !res.text.is_empty() || res.tool_calls.is_empty() {
        out.push(json!({"type": "message", "id": rand_id("msg_"), "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": res.text, "annotations": []}]}));
    }
    for c in &res.tool_calls {
        out.push(json!({"type": "function_call", "id": rand_id("fc_"), "call_id": c["id"], "name": c["function"]["name"],
                        "arguments": c["function"]["arguments"], "status": "completed"}));
    }
    out
}

pub(crate) fn responses(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, store: &ResponseStore, body: &[u8], stream: &mut TcpStream) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad(stream, false, &format!("bad json: {e}")) };
    let prior = match req["previous_response_id"].as_str() {
        None => Vec::new(),
        Some(pid) => match store.get(pid) { Some((m, _)) => m, None => return bad(stream, false, &format!("previous_response_id {pid} is not known to this server")) },
    };
    let msgs = match responses_to_messages(&req, prior) { Ok(m) => m, Err(e) => return bad(stream, false, &e) };
    let mut r = json!({"messages": msgs});
    if let Some(n) = req["max_output_tokens"].as_u64() { r["max_tokens"] = json!(n); }
    for k in ["model", "lora"] { if !req[k].is_null() { r[k] = req[k].clone(); } }
    for k in ["temperature", "top_p"] { if !req[k].is_null() { r[k] = req[k].clone(); } }
    if let Some(tools) = req["tools"].as_array() {
        let fns: Vec<Value> = tools.iter().filter(|t| t["type"] == "function").map(|t| json!({"type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}})).collect();
        if fns.len() != tools.len() { return bad(stream, false, "only `function` tools are supported (no built-in web/file search here)"); }
        r["tools"] = json!(fns);
    }
    if req["text"]["format"]["type"] == "json_schema" {
        r["response_format"] = json!({"type": "json_schema", "json_schema": {"schema": req["text"]["format"]["schema"]}});
    } else if req["text"]["format"]["type"] == "json_object" {
        r["response_format"] = json!({"type": "json_object"});
    }
    // `reasoning.effort` is Chat Completions' `reasoning_effort` (see `template_kwargs`).
    if let Some(e) = req["reasoning"]["effort"].as_str() { r["reasoning_effort"] = json!(e); }
    let empty = vec![];
    if let Err(e) = eng.gen_opts(&r, true).and_then(|_| eng.chat_ids(r["messages"].as_array().unwrap_or(&empty)).map(|_| ())) {
        return bad(stream, false, &e);
    }
    let id = rand_id("resp_");
    let model = req["model"].as_str().map(String::from).unwrap_or_else(|| eng.name.clone());
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let envelope = |status: &str, output: Vec<Value>, res: Option<&ChatResult>| {
        let mut v = json!({"id": id, "object": "response", "created_at": created, "status": status, "model": model,
                           "output": output, "incomplete_details": Value::Null, "previous_response_id": req["previous_response_id"]});
        if let Some(res) = res {
            v["usage"] = json!({"input_tokens": res.prompt_tokens, "output_tokens": res.gen_tokens, "total_tokens": res.prompt_tokens + res.gen_tokens});
            v["output_text"] = json!(res.text);
            v["energy"] = res.energy.clone();
            if res.finish == "length" && res.tool_calls.is_empty() {
                v["status"] = json!("incomplete");
                v["incomplete_details"] = json!({"reason": "max_output_tokens"});
            }
        }
        v
    };
    let remember = |res: &ChatResult, v: &Value| {
        let mut conv = msgs.clone();
        let mut a = json!({"role": "assistant", "content": res.text});
        if !res.tool_calls.is_empty() { a["tool_calls"] = json!(res.tool_calls); }
        conv.push(a);
        store.put(id.clone(), conv, v.clone());
    };
    if req["stream"].as_bool() != Some(true) {
        return match run_chat(eng, mcps, &r, |_, _, _| {}) {
            Err(e) => bad(stream, false, &e),
            Ok(res) => { let v = envelope("completed", responses_output(&res), Some(&res)); remember(&res, &v); write_json(stream, 200, &v) }
        };
    }
    write_sse_headers(stream);
    let mut seq = 0u64;
    let mut ev = |stream: &mut TcpStream, t: &str, mut data: Value| {
        data["type"] = json!(t); data["sequence_number"] = json!(seq); seq += 1;
        send_event(stream, t, &data);
    };
    ev(stream, "response.created", json!({"response": envelope("in_progress", vec![], None)}));
    ev(stream, "response.in_progress", json!({"response": envelope("in_progress", vec![], None)}));
    let item_id = rand_id("msg_");
    let mut started = false;
    let res = run_chat(eng, mcps, &r, |d, _, thinking| {
        if thinking { return; } // reasoning arrives in the final response object
        if !started {
            ev(stream, "response.output_item.added", json!({"output_index": 0, "item": {"type": "message", "id": item_id,
                "status": "in_progress", "role": "assistant", "content": []}}));
            ev(stream, "response.content_part.added", json!({"item_id": item_id, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}}));
            started = true;
        }
        ev(stream, "response.output_text.delta", json!({"item_id": item_id, "output_index": 0, "content_index": 0, "delta": d}));
    });
    match res {
        Err(e) => ev(stream, "error", json!({"message": e})),
        Ok(res) => {
            if started {
                ev(stream, "response.output_text.done", json!({"item_id": item_id, "output_index": 0, "content_index": 0, "text": res.text}));
                ev(stream, "response.content_part.done", json!({"item_id": item_id, "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": res.text, "annotations": []}}));
                ev(stream, "response.output_item.done", json!({"output_index": 0, "item": {"type": "message", "id": item_id,
                    "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": res.text, "annotations": []}]}}));
            }
            let v = envelope("completed", responses_output(&res), Some(&res));
            remember(&res, &v);
            ev(stream, "response.completed", json!({"response": v}));
        }
    }
}

pub(crate) fn retrieve(store: &ResponseStore, id: &str, stream: &mut TcpStream) {
    match store.get(id) {
        Some((_, v)) => write_json(stream, 200, &v),
        None => write_json(stream, 404, &json!({"error": {"message": format!("response {id} not found"), "type": "invalid_request_error"}})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_blocks_translate_to_chat_messages() {
        let r = anthropic_to_openai(&json!({"max_tokens": 64, "system": "Be terse.", "stop_sequences": ["\n\n"],
            "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "21C"}]}]})).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be terse."}));
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "toolu_1", "content": "21C"}));
        assert_eq!(r["stop"], json!(["\n\n"]));
        assert_eq!(r["tools"][0]["function"]["parameters"], json!({"type": "object"}));
        assert!(anthropic_to_openai(&json!({"messages": []})).unwrap_err().contains("max_tokens"));
        // An image block keeps its place among the text, as parts the vision path reads; a document is refused.
        let img = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}});
        let r = anthropic_to_openai(&json!({"max_tokens": 1, "messages": [{"role": "user", "content": [img.clone(), {"type": "text", "text": "what is it?"}]}]})).unwrap();
        assert_eq!(r["messages"][0]["content"], json!([img, {"type": "text", "text": "what is it?"}]));
        assert!(anthropic_to_openai(&json!({"max_tokens": 1, "messages": [{"role": "user", "content": [{"type": "document"}]}]})).is_err());
    }

    #[test]
    fn responses_items_and_previous_response_carry_the_conversation() {
        let prior = vec![json!({"role": "user", "content": "Hi"}), json!({"role": "assistant", "content": "Hello."})];
        let m = responses_to_messages(&json!({"instructions": "Be terse.", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "And now?"}]},
            {"type": "function_call_output", "call_id": "c1", "output": "42"}]}), prior).unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be terse."}));
        assert_eq!(m[3], json!({"role": "user", "content": "And now?"}));
        assert_eq!(m[4], json!({"role": "tool", "tool_call_id": "c1", "content": "42"}));
        let img = json!({"type": "input_image", "image_url": "data:image/png;base64,AAAA"});
        let m = responses_to_messages(&json!({"input": [{"role": "user", "content": [{"type": "input_text", "text": "and this?"}, img.clone()]}]}), vec![]).unwrap();
        assert_eq!(m[0]["content"], json!([{"type": "text", "text": "and this?"}, img]), "an input_image stays, as a part");
        assert!(responses_to_messages(&json!({"input": [{"role": "user", "content": [{"type": "input_file"}]}]}), vec![]).is_err());
    }
}
