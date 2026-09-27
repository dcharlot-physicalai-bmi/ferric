//! **The Ollama-native API** — the second dialect local front-ends speak.
//!
//! A parity audit of 13 front-ends (27 Sept 2026) found 10 with an Ollama provider; Home Assistant's core
//! integration reaches a local model ONLY through it. What they call, and implemented here:
//!
//! | route | who needs it |
//! |---|---|
//! | `GET /` (2xx) | AnythingLLM's health check |
//! | `GET /api/version` | Open WebUI's verify button |
//! | `GET /api/tags` — entries carry `name` AND `model`, `modified_at`, `size`, `digest`, `details` | model discovery everywhere; Zed deserialises it strictly, Home Assistant reads `model` |
//! | `POST /api/show` — `capabilities`, `model_info.<arch>.context_length`, `template` | Zed disables a model whose show fails; AnythingLLM falls back to a 4096 context without it |
//! | `GET /api/ps` | Open WebUI's "loaded" badge |
//! | `POST /api/chat` — NDJSON, one object per line, flushed per write; `stream` defaults to TRUE | nearly all; SillyTavern stalls on a buffered stream |
//! | `POST /api/generate` — `raw`, `system`; an empty prompt only loads | SillyTavern, Continue |
//! | `POST /api/embed` (normalised) and legacy `POST /api/embeddings` | RAG in Open WebUI, Cherry Studio, n8n; LobeHub uses the legacy one |
//!
//! Requests are translated into the OpenAI shape and run through the SAME `run_chat` / `Engine::generate`
//! as `/v1`, so the two dialects cannot disagree about a conversation: `options` (`num_predict`,
//! `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `repeat_penalty`, `repeat_last_n`,
//! `presence_penalty`, `frequency_penalty`) map onto `GenOpts`; `format: "json"` or a JSON Schema onto
//! guided decoding; `tools` onto the tool path, with Ollama's `arguments` as an object and an `id` per call
//! (Zed needs it). Images are refused by name. Model management (`pull`, `create`, `delete`, `copy`,
//! `push`) is a 501 that says what to do instead.
//!
//! ⚠ These requests run on the serial path; they do not share a decode batch with `/v1` traffic yet.
use crate::{Engine, ChatResult, run_chat, write_json, mcp};
use serde_json::{json, Value};
use std::io::Write;
use std::net::TcpStream;
use std::time::Instant;

/// The API level this server answers as. Clients gate features on it, so it is a real Ollama release
/// number with a build tag rather than an invented one.
const API_VERSION: &str = "0.12.0-ferric";

/// What `/api/tags`, `/api/show` and `/api/ps` report about one loaded model, gathered once at load.
#[derive(Clone, Debug)]
pub(crate) struct Card {
    pub name: String,
    pub path: String,
    pub arch: String,
    pub params: u64,
    pub size: u64,
    pub quant: String,
    pub modified: std::time::SystemTime,
    pub context: usize,
    pub embedding: bool,
    pub template: String,
    pub dim: usize,
}

fn type_name(t: u32) -> String {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 10 => "Q2_K",
        11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS",
        18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S", 23 => "IQ4_XS", 29 => "IQ1_M",
        30 => "BF16", 34 => "TQ1_0", 35 => "TQ2_0", 39 => "MXFP4", 40 => "NVFP4",
        _ => return format!("type{t}"),
    }.to_string()
}

impl Card {
    /// From the file itself: the parameter count is the sum of every tensor's elements, and the
    /// quantisation is the type holding the most of them (never `general.file_type`, which collides
    /// across forks).
    pub fn from_gguf(name: &str, path: &str, g: &ferric_gguf::GgufFile, embedding: bool, dim: usize, context: usize,
                     template: &str) -> Card {
        let mut by_type: std::collections::HashMap<u32, u64> = Default::default();
        let mut params = 0u64;
        for t in &g.tensors {
            let n: u64 = t.dims.iter().product();
            params += n;
            *by_type.entry(t.ggml_type).or_default() += n;
        }
        let quant = by_type.iter().max_by_key(|(_, n)| **n).map(|(t, _)| type_name(*t)).unwrap_or_default();
        let md = std::fs::metadata(path).ok();
        let arch = match g.metadata.get("general.architecture") { Some(ferric_gguf::Meta::Str(s)) => s.clone(), _ => String::new() };
        Card { name: name.to_string(), path: path.to_string(), arch, params, quant, embedding, dim, context,
               size: md.as_ref().map(|m| m.len()).unwrap_or(0),
               modified: md.and_then(|m| m.modified().ok()).unwrap_or(std::time::UNIX_EPOCH),
               template: template.to_string() }
    }

    /// Ollama names carry a tag; a bare name means `:latest`.
    pub fn tagged(&self) -> String { if self.name.contains(':') { self.name.clone() } else { format!("{}:latest", self.name) } }

    pub fn matches(&self, requested: &str) -> bool {
        let base = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_string();
        base(requested) == base(&self.name)
    }

    fn param_size(&self) -> String {
        let p = self.params as f64;
        if p >= 1e9 { format!("{:.1}B", p / 1e9) } else { format!("{:.0}M", p / 1e6) }
    }

    /// A stable identifier for this file: NOT a content hash (hashing an 18 GB file at load would take
    /// seconds), but a digest of its path, size and modification time, so a changed file changes it.
    fn digest(&self) -> String {
        let secs = self.modified.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let mut out = String::new();
        for salt in 0u64..4 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ salt;
            for b in self.path.bytes().chain(self.size.to_le_bytes()).chain(secs.to_le_bytes()) { h ^= b as u64; h = h.wrapping_mul(0x1000_0000_01b3); }
            out.push_str(&format!("{h:016x}"));
        }
        out
    }

    fn details(&self) -> Value {
        json!({"parent_model": "", "format": "gguf", "family": self.arch, "families": [self.arch],
               "parameter_size": self.param_size(), "quantization_level": self.quant})
    }

    fn tag_entry(&self) -> Value {
        json!({"name": self.tagged(), "model": self.tagged(), "modified_at": rfc3339(self.modified),
               "size": self.size, "digest": self.digest(), "details": self.details()})
    }

    fn capabilities(&self) -> Vec<&'static str> {
        if self.embedding { return vec!["embedding"]; }
        let mut c = vec!["completion"];
        // The tool path renders Hermes-style tool calls; advertise it for a template that knows tools.
        if self.template.contains("tools") { c.push("tools"); }
        c
    }
}

/// RFC 3339 in UTC from a SystemTime, without a date crate (Howard Hinnant's civil-from-days).
pub(crate) fn rfc3339(t: std::time::SystemTime) -> String {
    let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z", rem / 3600, rem % 3600 / 60, rem % 60, d.subsec_nanos())
}

fn now() -> String { rfc3339(std::time::SystemTime::now()) }

fn err(stream: &mut TcpStream, code: u16, m: &str) { write_json(stream, code, &json!({"error": m})) }

fn write_ndjson_headers(stream: &mut TcpStream) {
    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    let _ = stream.flush();
}

/// One NDJSON object per write, flushed — a client reading line by line must never wait on a buffer.
fn send_line(stream: &mut TcpStream, v: &Value) {
    let _ = stream.write_all(format!("{v}\n").as_bytes());
    let _ = stream.flush();
}

/// Ollama `options` + top-level fields → an OpenAI-shaped request `GenOpts` / `run_chat` understand.
fn to_openai(req: &Value) -> Result<Value, String> {
    let o = &req["options"];
    let mut r = json!({});
    match o["num_predict"].as_i64() {
        None | Some(-1) | Some(-2) => {}
        Some(n) if n > 0 => { r["max_tokens"] = json!(n); }
        Some(n) => return Err(format!("options.num_predict {n}: use a positive count, or -1 for unlimited")),
    }
    for k in ["temperature", "top_p", "top_k", "min_p", "seed", "stop", "repeat_penalty", "repeat_last_n",
              "presence_penalty", "frequency_penalty"] {
        if !o[k].is_null() { r[k] = o[k].clone(); }
    }
    match &req["format"] {
        Value::Null => {}
        Value::String(s) if s == "json" => { r["response_format"] = json!({"type": "json_object"}); }
        Value::String(s) if s.is_empty() => {}
        Value::Object(_) => { r["response_format"] = json!({"type": "json_schema", "json_schema": {"schema": req["format"]}}); }
        v => return Err(format!("`format` must be \"json\" or a JSON Schema object, got {v}")),
    }
    if let Some(t) = req["tools"].as_array() { r["tools"] = json!(t); }
    Ok(r)
}

fn durations(t0: Instant, first: Option<Instant>, r: &ChatResult) -> Value {
    let end = Instant::now();
    let first = first.unwrap_or(end);
    json!({"total_duration": (end - t0).as_nanos() as u64, "load_duration": 0u64,
           "prompt_eval_count": r.prompt_tokens, "prompt_eval_duration": (first - t0).as_nanos() as u64,
           "eval_count": r.gen_tokens, "eval_duration": (end - first).as_nanos() as u64})
}

/// OpenAI-shaped tool calls → Ollama's (arguments as an object, an `id` on each).
fn ollama_tool_calls(calls: &[Value]) -> Vec<Value> {
    calls.iter().enumerate().map(|(i, c)| {
        let args = c["function"]["arguments"].as_str().and_then(|s| serde_json::from_str::<Value>(s).ok())
            .unwrap_or_else(|| c["function"]["arguments"].clone());
        json!({"id": c["id"].as_str().map(String::from).unwrap_or_else(|| format!("call_{i}")),
               "function": {"index": i, "name": c["function"]["name"], "arguments": args}})
    }).collect()
}

/// The Ollama routes. Returns whether it recognised the request.
pub(crate) fn handle(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, method: &str, path: &str, body: &[u8],
                     stream: &mut TcpStream) -> bool {
    match (method, path) {
        ("GET", "/") | ("HEAD", "/") => {
            let b = b"ferric-serve is running (OpenAI /v1 and Ollama /api)";
            let _ = stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
            if method == "GET" { let _ = stream.write_all(b); }
            let _ = stream.flush();
        }
        ("GET", "/api/version") => write_json(stream, 200, &json!({"version": API_VERSION})),
        ("GET", "/api/tags") => write_json(stream, 200, &json!({"models": eng.cards().iter().map(Card::tag_entry).collect::<Vec<_>>()})),
        ("GET", "/api/ps") => {
            let models: Vec<Value> = eng.cards().iter().map(|c| {
                let mut e = c.tag_entry();
                e["expires_at"] = json!("2318-01-01T00:00:00Z"); // resident until the server stops
                e["size_vram"] = json!(c.size);
                e
            }).collect();
            write_json(stream, 200, &json!({"models": models}))
        }
        ("POST", "/api/show") => show(eng, body, stream),
        ("POST", "/api/chat") => chat(eng, mcps, body, stream),
        ("POST", "/api/generate") => generate(eng, mcps, body, stream),
        ("POST", "/api/embed") => embed(eng, body, stream, false),
        ("POST", "/api/embeddings") => embed(eng, body, stream, true),
        ("POST", "/api/pull") | ("POST", "/api/push") | ("POST", "/api/create") | ("POST", "/api/copy") | ("DELETE", "/api/delete") =>
            err(stream, 501, "ferric-serve serves the models it was started with; it does not manage a model store. \
                              Start it with the GGUF path or an owner/repo[:file.gguf] Hugging Face reference"),
        _ => return false,
    }
    true
}

fn parse(body: &[u8], stream: &mut TcpStream) -> Option<Value> {
    match serde_json::from_slice(body) { Ok(v) => Some(v), Err(e) => { err(stream, 400, &format!("bad json: {e}")); None } }
}

fn show(eng: &Engine, body: &[u8], stream: &mut TcpStream) {
    let Some(req) = parse(body, stream) else { return };
    let want = req["model"].as_str().or_else(|| req["name"].as_str()).unwrap_or("");
    let cards = eng.cards();
    let Some(c) = cards.iter().find(|c| c.matches(want)).or_else(|| if want.is_empty() { cards.first() } else { None }) else {
        return err(stream, 404, &format!("model '{want}' not found; loaded: {}", cards.iter().map(|c| c.tagged()).collect::<Vec<_>>().join(", ")));
    };
    let mut info = serde_json::Map::new();
    info.insert("general.architecture".into(), json!(c.arch));
    info.insert("general.parameter_count".into(), json!(c.params));
    info.insert(format!("{}.context_length", c.arch), json!(c.context));
    info.insert(format!("{}.embedding_length", c.arch), json!(c.dim));
    write_json(stream, 200, &json!({
        "modelfile": format!("# served by ferric-serve\nFROM {}\n", c.path), "parameters": "",
        "template": c.template, "details": c.details(), "model_info": Value::Object(info),
        "capabilities": c.capabilities(), "modified_at": rfc3339(c.modified),
    }));
}

/// Validate what the OpenAI path would refuse later, BEFORE a 200 goes out, so a bad request is a
/// 400 and not a broken stream: images, the translated options, and the messages themselves.
fn prepare(eng: &Engine, req: &Value, messages: &[Value]) -> Result<Value, String> {
    if req["model"].as_str().is_some_and(|m| eng.cards().iter().any(|c| c.embedding && c.matches(m))) {
        return Err(format!("\"{}\" is an embedding model; it does not generate text", req["model"].as_str().unwrap_or("")));
    }
    for (i, m) in messages.iter().enumerate() {
        if m["images"].as_array().is_some_and(|a| !a.is_empty()) {
            return Err(format!("messages[{i}] carries images; this server feeds text-only prompts"));
        }
    }
    let mut r = to_openai(req)?;
    r["messages"] = json!(messages);
    crate::genopts::GenOpts::from_req(&r, true)?;
    eng.chat_ids(messages)?;
    Ok(r)
}

fn chat(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, body: &[u8], stream: &mut TcpStream) {
    let Some(req) = parse(body, stream) else { return };
    let model = req["model"].as_str().map(String::from).unwrap_or_else(|| eng.name.clone());
    let empty = vec![];
    let messages: Vec<Value> = req["messages"].as_array().unwrap_or(&empty).clone();
    if messages.is_empty() {
        // Ollama's "load the model" request: nothing to generate.
        return write_json(stream, 200, &json!({"model": model, "created_at": now(),
            "message": {"role": "assistant", "content": ""}, "done_reason": "load", "done": true}));
    }
    let r = match prepare(eng, &req, &messages) { Ok(r) => r, Err(e) => return err(stream, 400, &e) };
    let has_tools = r["tools"].as_array().is_some_and(|t| !t.is_empty()) || !mcps.borrow().openai_tools().is_empty();
    // Ollama streams unless told not to. The tool path answers once, as one final line.
    let streaming = req["stream"].as_bool().unwrap_or(true);
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    if streaming {
        write_ndjson_headers(stream);
        let res = run_chat(eng, mcps, &r, |d, _| {
            first.get_or_insert_with(Instant::now);
            send_line(stream, &json!({"model": model, "created_at": now(), "message": {"role": "assistant", "content": d}, "done": false}));
        });
        match res {
            Err(e) => send_line(stream, &json!({"error": e})),
            Ok(res) => {
                let mut msg = json!({"role": "assistant", "content": if has_tools { res.text.clone() } else { String::new() }});
                if !res.tool_calls.is_empty() { msg["tool_calls"] = json!(ollama_tool_calls(&res.tool_calls)); }
                let mut last = json!({"model": model, "created_at": now(), "message": msg,
                                      "done_reason": if res.finish == "length" { "length" } else { "stop" }, "done": true});
                for (k, v) in durations(t0, first, &res).as_object().unwrap() { last[k] = v.clone(); }
                send_line(stream, &last);
            }
        }
        return;
    }
    match run_chat(eng, mcps, &r, |_, _| { first.get_or_insert_with(Instant::now); }) {
        Err(e) => err(stream, 400, &e),
        Ok(res) => {
            let mut msg = json!({"role": "assistant", "content": res.text});
            if !res.tool_calls.is_empty() { msg["tool_calls"] = json!(ollama_tool_calls(&res.tool_calls)); }
            let mut out = json!({"model": model, "created_at": now(), "message": msg,
                                 "done_reason": if res.finish == "length" { "length" } else { "stop" }, "done": true});
            for (k, v) in durations(t0, first, &res).as_object().unwrap() { out[k] = v.clone(); }
            write_json(stream, 200, &out);
        }
    }
}

fn generate(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, body: &[u8], stream: &mut TcpStream) {
    let Some(req) = parse(body, stream) else { return };
    let model = req["model"].as_str().map(String::from).unwrap_or_else(|| eng.name.clone());
    let prompt = req["prompt"].as_str().unwrap_or("");
    if prompt.is_empty() {
        return write_json(stream, 200, &json!({"model": model, "created_at": now(), "response": "", "done_reason": "load", "done": true}));
    }
    if req["suffix"].as_str().is_some_and(|s| !s.is_empty()) {
        return err(stream, 400, "`suffix` (fill-in-the-middle) is not supported by this server");
    }
    if req["images"].as_array().is_some_and(|a| !a.is_empty()) {
        return err(stream, 400, "`images`: this server feeds text-only prompts");
    }
    let raw = req["raw"].as_bool().unwrap_or(false);
    let streaming = req["stream"].as_bool().unwrap_or(true);
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    // `raw` sends the prompt as written; otherwise the model's chat template wraps it (with `system`).
    let run = |on_delta: &mut dyn FnMut(&str)| -> Result<ChatResult, String> {
        if raw {
            let r = to_openai(&req)?;
            let opts = crate::genopts::GenOpts::from_req(&r, false)?;
            let ids = eng.encode_prompt(prompt);
            let max = eng.budget(ids.len(), opts.max_tokens)?;
            let out = eng.generate(&ids, max, &opts, None, |d, _| on_delta(d));
            Ok(ChatResult { text: out.text, tool_calls: vec![], prompt_tokens: out.prompt_tokens, gen_tokens: out.gen_tokens,
                            finish: out.finish, logprobs: vec![] })
        } else {
            let mut messages = Vec::new();
            if let Some(sys) = req["system"].as_str() { messages.push(json!({"role": "system", "content": sys})); }
            messages.push(json!({"role": "user", "content": prompt}));
            let r = prepare(eng, &req, &messages)?;
            run_chat(eng, mcps, &r, |d, _| on_delta(d))
        }
    };
    if streaming {
        // Validate first (a raw prompt has nothing to validate beyond its options).
        if let Err(e) = to_openai(&req).and_then(|r| crate::genopts::GenOpts::from_req(&r, false).map(|_| ())) { return err(stream, 400, &e); }
        write_ndjson_headers(stream);
        let res = run(&mut |d: &str| {
            first.get_or_insert_with(Instant::now);
            send_line(stream, &json!({"model": model, "created_at": now(), "response": d, "done": false}));
        });
        match res {
            Err(e) => send_line(stream, &json!({"error": e})),
            Ok(res) => {
                let mut last = json!({"model": model, "created_at": now(), "response": "",
                                      "done_reason": if res.finish == "length" { "length" } else { "stop" }, "done": true, "context": []});
                for (k, v) in durations(t0, first, &res).as_object().unwrap() { last[k] = v.clone(); }
                send_line(stream, &last);
            }
        }
        return;
    }
    match run(&mut |_d: &str| { first.get_or_insert_with(Instant::now); }) {
        Err(e) => err(stream, 400, &e),
        Ok(res) => {
            let mut out = json!({"model": model, "created_at": now(), "response": res.text,
                                 "done_reason": if res.finish == "length" { "length" } else { "stop" }, "done": true, "context": []});
            for (k, v) in durations(t0, first, &res).as_object().unwrap() { out[k] = v.clone(); }
            write_json(stream, 200, &out);
        }
    }
}

/// `/api/embed` (`input`: string or array; normalised vectors; `truncate` defaults to true) and the legacy
/// `/api/embeddings` (`prompt`: one string → `embedding`). The legacy route returns the same normalised
/// vector: cosine similarity, what every consumer computes, does not depend on the scale.
fn embed(eng: &Engine, body: &[u8], stream: &mut TcpStream, legacy: bool) {
    let Some(req) = parse(body, stream) else { return };
    let inputs: Vec<String> = if legacy {
        match req["prompt"].as_str() { Some(p) => vec![p.to_string()], None => return err(stream, 400, "`prompt` must be a string") }
    } else {
        match &req["input"] {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => {
                let mut v = Vec::with_capacity(a.len());
                for x in a { match x.as_str() { Some(s) => v.push(s.to_string()), None => return err(stream, 400, "`input` must contain only strings") } }
                v
            }
            _ => return err(stream, 400, "`input` must be a string or an array of strings"),
        }
    };
    let t0 = Instant::now();
    let truncate = req["truncate"].as_bool().unwrap_or(true);
    let model = req["model"].as_str().map(|m| m.strip_suffix(":latest").unwrap_or(m).to_string());
    match crate::embed_texts(eng, model.as_deref(), &inputs, truncate) {
        Err(e) => err(stream, 400, &e),
        Ok((vecs, total, name)) => {
            if legacy { return write_json(stream, 200, &json!({"embedding": vecs.into_iter().next().unwrap_or_default()})); }
            write_json(stream, 200, &json!({"model": req["model"].as_str().map(String::from).unwrap_or(name), "embeddings": vecs,
                "total_duration": t0.elapsed().as_nanos() as u64, "load_duration": 0u64, "prompt_eval_count": total}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_formats_known_instants() {
        let t = |s: u64| std::time::UNIX_EPOCH + std::time::Duration::from_secs(s);
        assert_eq!(rfc3339(t(0)), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(rfc3339(t(951_782_400)), "2000-02-29T00:00:00.000000000Z"); // a leap day
        assert_eq!(rfc3339(t(1_790_510_400)), "2026-09-27T12:00:00.000000000Z");
    }

    #[test]
    fn options_translate_and_bad_ones_are_refused() {
        let r = to_openai(&json!({"options": {"num_predict": 12, "temperature": 0.5, "stop": ["x"], "seed": 3}, "format": "json"})).unwrap();
        assert_eq!(r["max_tokens"], 12);
        assert_eq!(r["temperature"], 0.5);
        assert_eq!(r["response_format"]["type"], "json_object");
        assert!(to_openai(&json!({"options": {"num_predict": -1}})).unwrap()["max_tokens"].is_null(), "-1 is unlimited");
        assert!(to_openai(&json!({"options": {"num_predict": -7}})).is_err());
        assert!(to_openai(&json!({"format": 3})).is_err());
        let s = to_openai(&json!({"format": {"type": "object", "properties": {"a": {"type": "string"}}}})).unwrap();
        assert_eq!(s["response_format"]["type"], "json_schema");
    }

    #[test]
    fn tool_calls_carry_object_arguments_and_an_id() {
        let c = ollama_tool_calls(&[json!({"id": "call_7", "function": {"name": "f", "arguments": "{\"a\":1}"}}),
                                    json!({"function": {"name": "g", "arguments": "{}"}})]);
        assert_eq!(c[0]["function"]["arguments"]["a"], 1);
        assert_eq!(c[0]["id"], "call_7");
        assert_eq!(c[1]["id"], "call_1");
    }
}
