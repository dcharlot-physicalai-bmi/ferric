//! **Model Context Protocol (MCP) client** — the cross-vendor tool/context standard (Anthropic;
//! adopted by OpenAI/Google/Microsoft). JSON-RPC 2.0 over two transports: **stdio** (spawn a local
//! server subprocess, newline-delimited messages) and **Streamable-HTTP** (POST to a remote endpoint,
//! session via the `Mcp-Session-Id` header, response as JSON or an SSE `data:` line). Handshake
//! `initialize` → `notifications/initialized` → `tools/list`, then `tools/call` on demand. HTTP uses
//! `curl` to stay dependency-light (no reqwest to vendor).
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

enum Transport {
    Stdio { _child: Child, stdin: ChildStdin, reader: BufReader<ChildStdout> },
    Http { url: String, session: Option<String> },
}

pub struct Mcp {
    transport: Transport,
    next_id: u64,
    pub tools: Vec<Value>, // raw MCP tool descriptors (name, description, inputSchema)
    pub label: String,
}

impl Mcp {
    /// Spawn `cmd_line` as a stdio MCP server and complete the handshake + tool discovery.
    pub fn connect(cmd_line: &str) -> Result<Mcp, String> {
        let parts: Vec<&str> = cmd_line.split_whitespace().collect();
        if parts.is_empty() { return Err("empty mcp command".into()); }
        let mut child = Command::new(parts[0]).args(&parts[1..])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
            .spawn().map_err(|e| format!("spawn '{cmd_line}': {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let reader = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        Mcp::handshake(Transport::Stdio { _child: child, stdin, reader }, parts[0].to_string())
    }

    /// Connect to a remote MCP server over Streamable-HTTP.
    pub fn connect_http(url: &str) -> Result<Mcp, String> {
        Mcp::handshake(Transport::Http { url: url.to_string(), session: None }, url.to_string())
    }

    fn handshake(transport: Transport, label: String) -> Result<Mcp, String> {
        let mut m = Mcp { transport, next_id: 0, tools: vec![], label };
        m.request("initialize", json!({
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "ferric-serve", "version": "0.1"}
        }))?;
        m.notify("notifications/initialized", json!({}))?;
        let tl = m.request("tools/list", json!({}))?;
        m.tools = tl["result"]["tools"].as_array().cloned().unwrap_or_default();
        Ok(m)
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.next_id += 1;
        let id = self.next_id;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        match &mut self.transport {
            Transport::Stdio { stdin, reader, .. } => {
                writeln!(stdin, "{msg}").map_err(|e| e.to_string())?;
                stdin.flush().map_err(|e| e.to_string())?;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 { return Err("mcp: server closed".into()); }
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    let v: Value = serde_json::from_str(line).map_err(|e| format!("mcp non-json line ({e}): {line}"))?;
                    if v["id"] == json!(id) {
                        if let Some(err) = v.get("error").filter(|e| !e.is_null()) { return Err(format!("mcp error: {err}")); }
                        return Ok(v);
                    }
                }
            }
            Transport::Http { url, session } => {
                let (v, new_sess) = http_rpc(url, session.as_deref(), &msg.to_string())?;
                if let Some(s) = new_sess { *session = Some(s); }
                let v = v.ok_or("mcp http: empty response")?;
                if let Some(err) = v.get("error").filter(|e| !e.is_null()) { return Err(format!("mcp error: {err}")); }
                Ok(v)
            }
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        match &mut self.transport {
            Transport::Stdio { stdin, .. } => { writeln!(stdin, "{msg}").map_err(|e| e.to_string())?; stdin.flush().map_err(|e| e.to_string()) }
            Transport::Http { url, session } => { let _ = http_rpc(url, session.as_deref(), &msg.to_string())?; Ok(()) }
        }
    }

    /// Call an MCP tool; returns the concatenated text of the result `content` array.
    pub fn call(&mut self, name: &str, args: &Value) -> Result<String, String> {
        let r = self.request("tools/call", json!({"name": name, "arguments": args}))?;
        let text = r["result"]["content"].as_array()
            .map(|a| a.iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
        Ok(text)
    }

    /// This server's tools as OpenAI-style function tool defs (to advertise to the model).
    pub fn openai_tools(&self) -> Vec<Value> {
        self.tools.iter().map(|t| json!({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description").cloned().unwrap_or(Value::Null),
            "parameters": t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"})),
        }})).collect()
    }
}

/// One MCP Streamable-HTTP round-trip via curl. Returns (parsed JSON-RPC response, new session id).
/// The response body is either JSON or an SSE `data: {…}` line; the session id rides the response
/// `Mcp-Session-Id` header (captured on `initialize`, echoed back on later requests).
fn http_rpc(url: &str, session: Option<&str>, body: &str) -> Result<(Option<Value>, Option<String>), String> {
    let mut args: Vec<String> = vec![
        "-s".into(), "-i".into(), "-X".into(), "POST".into(),
        "-H".into(), "content-type: application/json".into(),
        "-H".into(), "accept: application/json, text/event-stream".into(),
    ];
    if let Some(s) = session { args.push("-H".into()); args.push(format!("mcp-session-id: {s}")); }
    args.push("-d".into()); args.push(body.to_string());
    args.push(url.to_string());
    let out = std::process::Command::new("curl").args(&args).output().map_err(|e| format!("curl: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (headers, body) = text.find("\r\n\r\n").map(|i| (&text[..i], &text[i + 4..]))
        .or_else(|| text.find("\n\n").map(|i| (&text[..i], &text[i + 2..])))
        .unwrap_or(("", text.as_ref()));
    let sess = headers.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim().eq_ignore_ascii_case("mcp-session-id").then(|| v.trim().to_string())
    });
    let body = body.trim();
    if body.is_empty() { return Ok((None, sess)); }
    let json_str = if body.starts_with('{') || body.starts_with('[') { body.to_string() }
        else { body.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim).find(|s| s.starts_with('{')).unwrap_or("").to_string() };
    if json_str.is_empty() { return Ok((None, sess)); }
    let v: Value = serde_json::from_str(&json_str).map_err(|e| format!("mcp http bad json ({e}): {json_str}"))?;
    Ok((Some(v), sess))
}

/// A set of connected MCP servers. Looks up which server owns a tool name for routing tool calls.
#[derive(Default)]
pub struct McpSet(pub Vec<Mcp>);
impl McpSet {
    pub fn openai_tools(&self) -> Vec<Value> { self.0.iter().flat_map(|m| m.openai_tools()).collect() }
    pub fn has(&self, name: &str) -> bool { self.0.iter().any(|m| m.tools.iter().any(|t| t["name"] == json!(name))) }
    pub fn call(&mut self, name: &str, args: &Value) -> Result<String, String> {
        for m in &mut self.0 {
            if m.tools.iter().any(|t| t["name"] == json!(name)) { return m.call(name, args); }
        }
        Err(format!("no MCP server owns tool '{name}'"))
    }
}
