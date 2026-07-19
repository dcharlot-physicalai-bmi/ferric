//! **Model Context Protocol (MCP) client** — the cross-vendor tool/context standard (Anthropic;
//! adopted by OpenAI/Google/Microsoft). JSON-RPC 2.0 over a stdio transport (newline-delimited
//! messages): spawn an MCP server subprocess, `initialize` → `notifications/initialized` →
//! `tools/list`, then `tools/call` on demand. Pure-Rust + std only (wasm-clean design; native here).
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct Mcp {
    _child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: u64,
    pub tools: Vec<Value>, // raw MCP tool descriptors (name, description, inputSchema)
    pub label: String,
}

impl Mcp {
    /// Spawn `cmd_line` as an MCP server over stdio and complete the handshake + tool discovery.
    pub fn connect(cmd_line: &str) -> Result<Mcp, String> {
        let parts: Vec<&str> = cmd_line.split_whitespace().collect();
        if parts.is_empty() { return Err("empty mcp command".into()); }
        let mut child = Command::new(parts[0]).args(&parts[1..])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
            .spawn().map_err(|e| format!("spawn '{cmd_line}': {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let reader = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        let mut m = Mcp { _child: child, stdin, reader, next_id: 0, tools: vec![], label: parts[0].to_string() };
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
        writeln!(self.stdin, "{}", json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 { return Err("mcp: server closed".into()); }
            let line = line.trim();
            if line.is_empty() { continue; }
            let v: Value = serde_json::from_str(line).map_err(|e| format!("mcp non-json line ({e}): {line}"))?;
            if v["id"] == json!(id) {
                if let Some(err) = v.get("error").filter(|e| !e.is_null()) { return Err(format!("mcp error: {err}")); }
                return Ok(v);
            }
            // server-initiated request/notification (id mismatch or none) — ignore for this minimal client
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        writeln!(self.stdin, "{}", json!({"jsonrpc": "2.0", "method": method, "params": params})).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())
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
