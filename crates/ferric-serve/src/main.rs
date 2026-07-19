//! **Ferric OpenAI-compatible server** — the adoption on-ramp. A dependency-light (std TCP + serde)
//! HTTP server exposing `/v1/chat/completions` (streaming + non-streaming), `/v1/completions`,
//! `/v1/models`, and `/health` over the pure-Rust cross-fabric runtime. Any OpenAI client, agent
//! framework (LangChain, LangGraph, the Vercel AI SDK, …), or `curl` points at it unchanged.
//!
//!   cargo run -p ferric-serve --release -- <model.gguf> [--port 8080] [--name my-model]
//!
//! One request at a time (the GPU serializes anyway); continuous batching is the P1 follow-up.
mod mcp;
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

/// GPT-2 byte↔printable-unicode map inverted — turn vocab entries back into raw bytes.
fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new();
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}

struct Engine {
    ctx: Arc<Context>,
    model: Qwen3,
    bpe: Bpe,
    tokens: Vec<String>,
    u2b: HashMap<char, u8>,
    im_start: Option<u32>,
    im_end: Option<u32>,
    bos_id: Option<u32>,
    add_bos: bool,
    eos: Vec<u32>,
    name: String,
    /// Raw bytes each token decodes to (for guided decoding); `None` for special/non-text tokens,
    /// which are disallowed under a constraint (except EOS, handled separately).
    token_bytes: Vec<Option<Vec<u8>>>,
}

impl Engine {
    fn load(path: &str, name: String) -> Engine {
        let ctx = Arc::new(pollster::block_on(Context::new()).unwrap());
        let g = GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e:?}"));
        let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
            _ => panic!("gguf has no tokenizer.ggml.tokens"),
        };
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
            Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m {
                s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string()))
            } else { None }).collect(),
            _ => Vec::new(),
        };
        let bpe = Bpe::new(vocab.clone(), &merges);
        let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
        let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
        let mut eos: Vec<u32> = Vec::new();
        if let Some(Meta::U(v)) = g.metadata.get("tokenizer.ggml.eos_token_id") { eos.push(*v as u32); }
        let im_end = vocab.get("<|im_end|>").copied();
        let im_start = vocab.get("<|im_start|>").copied();
        if let Some(e) = im_end { if !eos.contains(&e) { eos.push(e); } }
        if let Some(&e) = vocab.get("<|endoftext|>") { if !eos.contains(&e) { eos.push(e); } }
        let model = Qwen3::load(&ctx, &g).unwrap_or_else(|e| panic!("load model: {e}"));
        let u2b = byte_decoder();
        // Precompute each token's raw bytes (chars → bytes via u2b). A token containing any char not in
        // the byte map is a special token (e.g. <|im_end|>) → None → disallowed under a constraint.
        let token_bytes: Vec<Option<Vec<u8>>> = tokens.iter().map(|t| {
            let mut b = Vec::with_capacity(t.len());
            for c in t.chars() { match u2b.get(&c) { Some(&x) => b.push(x), None => return None } }
            Some(b)
        }).collect();
        Engine { ctx, model, bpe, tokens, u2b, im_start, im_end, bos_id, add_bos, eos, name, token_bytes }
    }

    fn detok(&self, ids: &[u32]) -> String {
        let s: String = ids.iter().map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| self.u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    }

    /// Build the prompt token stream from OpenAI `messages` using the ChatML template (Qwen/most
    /// GGUF chat models). Special tokens are inserted by id; text is byte-level BPE-encoded between them.
    fn chat_ids(&self, messages: &[Value]) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos { if let Some(b) = self.bos_id { ids.push(b); } }
        let (Some(s), Some(e)) = (self.im_start, self.im_end) else {
            // No ChatML special tokens — fall back to concatenated text.
            let text: String = messages.iter().map(|m| format!("{}: {}\n", m["role"].as_str().unwrap_or("user"), m["content"].as_str().unwrap_or(""))).collect();
            ids.extend(self.bpe.encode(&text));
            return ids;
        };
        for m in messages {
            let role = m["role"].as_str().unwrap_or("user");
            let content = m["content"].as_str().unwrap_or("");
            ids.push(s);
            ids.extend(self.bpe.encode(&format!("{role}\n{content}")));
            ids.push(e);
            ids.extend(self.bpe.encode("\n"));
        }
        ids.push(s);
        ids.extend(self.bpe.encode("assistant\n"));
        ids
    }

    /// Greedy decode (temperature ignored for now — determinism is the point). Calls `on_delta` with
    /// each newly-decoded text fragment for streaming. Returns (full_text, prompt_tokens, gen_tokens).
    fn generate(&self, prompt: &[u32], max_tokens: usize, mut guide: Option<ferric_agent::guide::Guide>, mut on_delta: impl FnMut(&str)) -> (String, usize, usize) {
        let mut cache = Cache::new(&self.model.cfg);
        let n_vocab = self.model.cfg.n_vocab;
        let argmax = |row: &[f32]| (0..n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
        let mut gen: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        for step in 0..max_tokens {
            let input: Vec<u32> = if step == 0 { prompt.to_vec() } else { vec![*gen.last().unwrap()] };
            let logits = self.model.forward_cached(&input, &mut cache);
            let v = pollster::block_on(logits.to_vec());
            let row = &v[v.len() - n_vocab..];
            // Guided decoding: pick the highest-logit token whose bytes keep the JSON valid (EOS only
            // once the value is complete). Most tokens reject on their first byte, so the scan is cheap.
            let next = if let Some(g) = guide.as_ref() {
                let can_stop = g.can_stop();
                let (mut best, mut best_l) = (None, f32::NEG_INFINITY);
                for i in 0..n_vocab {
                    let ok = if self.eos.contains(&(i as u32)) { can_stop }
                        else { match &self.token_bytes[i] { Some(b) if !b.is_empty() => { let mut a = *g; b.iter().all(|&c| a.step(c)) } _ => false } };
                    if ok && row[i] > best_l { best_l = row[i]; best = Some(i as u32); }
                }
                match best { Some(t) => t, None => break }
            } else { argmax(row) };
            if self.eos.contains(&next) { break; }
            if let (Some(g), Some(b)) = (guide.as_mut(), self.token_bytes[next as usize].as_ref()) { for &c in b { g.step(c); } }
            gen.push(next);
            // Re-detok the whole generation and emit only the new suffix (handles multi-byte UTF-8).
            let full = self.detok(&gen);
            if full.len() > emitted.len() && full.is_char_boundary(emitted.len()) {
                let delta = full[emitted.len()..].to_string();
                on_delta(&delta);
                emitted = full;
            }
        }
        (emitted, prompt.len(), gen.len())
    }
}

fn now_unix() -> u64 { 1_700_000_000 } // static stamp (no wall clock needed for the API contract)

/// Resolve a model spec to a local GGUF path. Accepts a local file, or a HuggingFace ref
/// `owner/repo[:file.gguf]` — downloads (and caches under ~/.cache/ferric/hub) via `curl` so
/// `ferric-serve unsloth/Qwen3-0.6B-GGUF` just works with no manual download. curl keeps us dep-light
/// (no reqwest/hf-hub to vendor); a pure-Rust HTTPS client is the follow-up when the vendor tree grows.
fn resolve_model(spec: &str) -> String {
    if std::path::Path::new(spec).exists() { return spec.to_string(); }
    let (repo, file) = match spec.split_once(':') { Some((r, f)) => (r.to_string(), Some(f.to_string())), None => (spec.to_string(), None) };
    if !repo.contains('/') { eprintln!("ferric-serve: '{spec}' is neither a local file nor an HF repo (owner/repo)"); std::process::exit(1); }
    let file = file.unwrap_or_else(|| pick_gguf(&repo));
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = format!("{home}/.cache/ferric/hub/{}", repo.replace('/', "_"));
    std::fs::create_dir_all(&dir).ok();
    let dest = format!("{dir}/{}", file.rsplit('/').next().unwrap_or(&file));
    if std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false) { eprintln!("ferric-serve: cached {dest}"); return dest; }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    eprintln!("ferric-serve: downloading {url}");
    let ok = std::process::Command::new("curl").args(["-L", "-f", "--progress-bar", "-C", "-", "-o", &dest, &url]).status().map(|s| s.success()).unwrap_or(false);
    if !ok { eprintln!("ferric-serve: download failed ({url})"); std::process::exit(1); }
    dest
}

/// Query the HF model API for a repo's file list and pick a GGUF (prefer Q4_K_M, else the first).
fn pick_gguf(repo: &str) -> String {
    let out = std::process::Command::new("curl").args(["-sL", "-f", &format!("https://huggingface.co/api/models/{repo}")]).output();
    let v: Value = out.ok().and_then(|o| serde_json::from_slice(&o.stdout).ok()).unwrap_or(Value::Null);
    let files: Vec<String> = v["siblings"].as_array().map(|a| a.iter().filter_map(|s| s["rfilename"].as_str().map(String::from)).collect()).unwrap_or_default();
    let ggufs: Vec<&String> = files.iter().filter(|f| f.to_lowercase().ends_with(".gguf")).collect();
    let pick = ggufs.iter().find(|f| f.contains("Q4_K_M")).or_else(|| ggufs.first()).cloned();
    match pick { Some(f) => { eprintln!("ferric-serve: picked {f} from {repo}"); f.clone() } None => { eprintln!("ferric-serve: no .gguf found in {repo} (specify owner/repo:file.gguf)"); std::process::exit(1); } }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).unwrap_or_else(|| { eprintln!("usage: ferric-serve <model.gguf> [--port N] [--name S]"); std::process::exit(1); });
    let mut port = 8080u16;
    let mut name = "ferric".to_string();
    let mut mcp_cmds: Vec<(String, String)> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => { port = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(port); i += 2; }
            "--name" => { name = args.get(i + 1).cloned().unwrap_or(name); i += 2; }
            "--mcp" => { if let Some(c) = args.get(i + 1) { mcp_cmds.push(("stdio".to_string(), c.clone())); } i += 2; }
            "--mcp-http" => { if let Some(c) = args.get(i + 1) { mcp_cmds.push(("http".to_string(), c.clone())); } i += 2; }
            _ => i += 1,
        }
    }
    // Connect any configured MCP servers (stdio subprocess or remote Streamable-HTTP) + discover tools.
    let mut mcps = mcp::McpSet::default();
    for (kind, c) in &mcp_cmds {
        let r = if kind == "http" { mcp::Mcp::connect_http(c) } else { mcp::Mcp::connect(c) };
        match r {
            Ok(m) => { eprintln!("ferric-serve: mcp '{}' connected — {} tools: {:?}", m.label, m.tools.len(), m.tools.iter().filter_map(|t| t["name"].as_str()).collect::<Vec<_>>()); mcps.0.push(m); }
            Err(e) => eprintln!("ferric-serve: mcp '{c}' failed: {e}"),
        }
    }
    if args.iter().any(|a| a == "--mcp-test") {
        // Verify the MCP client mechanics: list tools, and call `add(2,3)` if present.
        eprintln!("ferric-serve: --mcp-test, {} tool(s) advertised", mcps.openai_tools().len());
        if mcps.has("add") { eprintln!("  add(2,3) = {:?}", mcps.call("add", &json!({"a": 2, "b": 3}))); }
        return;
    }
    let resolved = resolve_model(path);
    eprintln!("ferric-serve: loading {resolved} …");
    let eng = Engine::load(&resolved, name.clone());
    if args.iter().any(|a| a == "--once") {
        // Smoke test: one chat turn straight through the pipeline, no HTTP.
        let msgs = vec![json!({"role": "user", "content": "Hi"})];
        let (t, p, g) = eng.generate(&eng.chat_ids(&msgs), 16, None, |d| eprint!("{d}"));
        eprintln!("\nferric-serve: --once ok ({p} prompt + {g} gen tokens): {t:?}");
        return;
    }
    let mcps = std::cell::RefCell::new(mcps);
    eprintln!("ferric-serve: {} ({} layers, vocab {}) on {:?}{} — http://127.0.0.1:{port}/v1",
        name, eng.model.cfg.n_layer, eng.model.cfg.n_vocab, eng.ctx.backend,
        if mcps.borrow().0.is_empty() { String::new() } else { format!(" · {} MCP tools", mcps.borrow().openai_tools().len()) });
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| panic!("bind :{port}: {e}"));
    for stream in listener.incoming() {
        if let Ok(s) = stream {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&eng, &mcps, s)));
            if r.is_err() { eprintln!("ferric-serve: handler panicked (recovered)"); }
        }
    }
}

fn handle(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, mut stream: TcpStream) {
    let (method, path, body) = match read_request(&mut stream) { Some(r) => r, None => return };
    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => write_json(&mut stream, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") => write_json(&mut stream, 200, &json!({
            "object": "list",
            "data": [{"id": eng.name, "object": "model", "created": now_unix(), "owned_by": "ferric"}]
        })),
        ("POST", "/v1/chat/completions") => chat(eng, mcps, &mut stream, &body),
        ("POST", "/v1/completions") => completions(eng, &mut stream, &body),
        _ => write_json(&mut stream, 404, &json!({"error": {"message": "not found", "type": "invalid_request_error"}})),
    }
}

fn inject_tools(messages: &mut Vec<Value>, tools: &[Value]) {
    let tp = ferric_agent::tools::hermes_prompt(tools);
    match messages.first_mut() {
        Some(first) if first["role"] == "system" => {
            let merged = format!("{}\n\n{tp}", first["content"].as_str().unwrap_or(""));
            first["content"] = json!(merged);
        }
        _ => messages.insert(0, json!({"role": "system", "content": tp})),
    }
}

fn chat(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, stream: &mut TcpStream, body: &[u8]) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return write_json(stream, 400, &json!({"error": {"message": format!("bad json: {e}")}})) };
    let empty = vec![];
    let mut messages: Vec<Value> = req["messages"].as_array().unwrap_or(&empty).clone();
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
    let streaming = req["stream"].as_bool().unwrap_or(false);
    // Advertised tools = caller's + every connected MCP server's.
    let mut tools = req["tools"].as_array().cloned().unwrap_or_default();
    tools.extend(mcps.borrow().openai_tools());
    let has_tools = !tools.is_empty();
    let id = "chatcmpl-ferric".to_string();

    if has_tools {
        inject_tools(&mut messages, &tools);
        // Server-side agent loop: generate → parse tool_calls → execute the MCP-owned ones and feed
        // results back → repeat. Non-MCP tool calls are returned to the client (standard OpenAI flow).
        let (mut ptok, mut gtok) = (0usize, 0usize);
        let (mut out_text, mut out_calls) = (String::new(), Vec::new());
        for _round in 0..4 {
            let prompt = eng.chat_ids(&messages);
            let (text, p, g) = eng.generate(&prompt, max_tokens, None, |_| {});
            ptok += p; gtok += g;
            let calls = ferric_agent::tools::parse_tool_calls(&text);
            let mcp_calls: Vec<&Value> = calls.iter().filter(|c| mcps.borrow().has(c["function"]["name"].as_str().unwrap_or(""))).collect();
            if mcp_calls.is_empty() { out_text = text; out_calls = calls; break; }
            messages.push(json!({"role": "assistant", "content": text}));
            for c in &mcp_calls {
                let name = c["function"]["name"].as_str().unwrap_or("");
                let args: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or_else(|_| json!({}));
                let result = mcps.borrow_mut().call(name, &args).unwrap_or_else(|e| format!("error: {e}"));
                eprintln!("ferric-serve: mcp call {name}({args}) -> {result}");
                messages.push(json!({"role": "user", "content": format!("<tool_response>\n{{\"name\": \"{name}\", \"content\": {}}}\n</tool_response>", serde_json::to_string(&result).unwrap_or_default())}));
            }
        }
        let (message, finish) = if !out_calls.is_empty() {
            (json!({"role": "assistant", "content": Value::Null, "tool_calls": out_calls}), "tool_calls")
        } else {
            (json!({"role": "assistant", "content": out_text}), "stop")
        };
        return write_json(stream, 200, &json!({
            "id": id, "object": "chat.completion", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok}
        }));
    }

    // No tools → optional guided decoding + streaming.
    let rf = req["response_format"]["type"].as_str().unwrap_or("");
    let sch_prog = if rf == "json_schema" { ferric_agent::guide::compile(&req["response_format"]["json_schema"]["schema"]) } else { None };
    let guide = if let Some(prog) = &sch_prog { Some(ferric_agent::guide::Guide::Schema(ferric_agent::guide::Schema::new(prog))) }
        else if rf == "json_object" || rf == "json_schema" { Some(ferric_agent::guide::Guide::Json(ferric_agent::guide::Json::object())) }
        else { None };
    let prompt = eng.chat_ids(&messages);
    if streaming {
        write_sse_headers(stream);
        send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}]}));
        eng.generate(&prompt, max_tokens, guide, |delta| {
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": Value::Null}]}));
        });
        send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}));
        let _ = stream.write_all(b"data: [DONE]\n\n");
    } else {
        let (text, ptok, gtok) = eng.generate(&prompt, max_tokens, guide, |_| {});
        write_json(stream, 200, &json!({
            "id": id, "object": "chat.completion", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok}
        }));
    }
}

fn completions(eng: &Engine, stream: &mut TcpStream, body: &[u8]) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return write_json(stream, 400, &json!({"error": {"message": format!("bad json: {e}")}})) };
    let prompt_text = req["prompt"].as_str().unwrap_or("");
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
    let mut ids = Vec::new();
    if eng.add_bos { if let Some(b) = eng.bos_id { ids.push(b); } }
    ids.extend(eng.bpe.encode(prompt_text));
    let (text, ptok, gtok) = eng.generate(&ids, max_tokens, None, |_| {});
    write_json(stream, 200, &json!({
        "id": format!("cmpl-ferric-{}", ids.len()), "object": "text_completion", "created": now_unix(), "model": eng.name,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok}
    }));
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
    let peer = stream.try_clone().ok()?;
    let mut reader = BufReader::new(peer);
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 { return None; }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 { break; }
        if h.trim().is_empty() { break; }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") { content_length = v.trim().parse().unwrap_or(0); }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 { reader.read_exact(&mut body).ok()?; }
    Some((method, path, body))
}

fn write_json(stream: &mut TcpStream, status: u16, v: &Value) {
    let body = serde_json::to_vec(v).unwrap_or_default();
    let head = format!("HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        if status == 200 { "OK" } else { "ERR" }, body.len());
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
    let _ = stream.flush();
}

fn write_sse_headers(stream: &mut TcpStream) {
    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    let _ = stream.flush();
}

fn send_sse(stream: &mut TcpStream, v: &Value) {
    let _ = stream.write_all(format!("data: {}\n\n", v).as_bytes());
    let _ = stream.flush();
}
