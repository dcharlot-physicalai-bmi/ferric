//! **In-process use of the serving engine** — the same model loading, chat templates, tokenizers,
//! samplers, constraints, reasoning split and energy attribution as the HTTP server, without a socket.
//! The C ABI (`ferric-ffi`, `libferric`) and through it every language binding (Python, Go, Swift, Zig…)
//! sit on this, so a binding answers exactly what `ferric-serve` answers for the same request.
//!
//! Requests and responses are the OpenAI shapes as JSON values: `chat` takes a `/v1/chat/completions`
//! body and returns a `chat.completion` object (with `energy`); `complete` takes a `/v1/completions` body.
use crate::{mcp, run_chat, Engine};
use serde_json::{json, Value};

pub struct LocalModel { eng: Engine, mcps: std::cell::RefCell<mcp::McpSet> }

fn now_unix() -> u64 { crate::now_unix() }

impl LocalModel {
    /// Load a GGUF (every architecture the registry serves), on the default GPU context.
    pub fn load(path: &str) -> Result<LocalModel, String> {
        let name = crate::models::default_name(std::path::Path::new(path));
        let p = path.to_string();
        // The loader reports a bad file by panicking with the reason; an embedding caller gets it as an error.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(move || Engine::load(&p, name));
        std::panic::set_hook(prev);
        let eng = r.map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "the model failed to load".into()))?;
        Ok(LocalModel { eng, mcps: Default::default() })
    }

    /// The name the server would give this model (the file's stem).
    pub fn name(&self) -> &str { &self.eng.name }

    /// A `/v1/chat/completions` body → a `chat.completion` object. `on_delta(text, is_reasoning)` receives
    /// the answer as it streams (never called for a tool call, whose answer is known only at the end).
    pub fn chat(&self, req: &Value, mut on_delta: impl FnMut(&str, bool)) -> Result<Value, String> {
        if req["n"].as_u64().is_some_and(|n| n > 1) { return Err("`n` > 1 is served over HTTP; ask one choice at a time here".into()); }
        let opts = self.eng.gen_opts(req, true)?;
        let r = run_chat(&self.eng, &self.mcps, req, |d, _, reasoning| on_delta(d, reasoning))?;
        let mut message = if r.tool_calls.is_empty() { json!({"role": "assistant", "content": r.text}) }
                          else { json!({"role": "assistant", "content": Value::Null, "tool_calls": r.tool_calls}) };
        if !r.reasoning.is_empty() { message["reasoning_content"] = json!(r.reasoning); }
        let mut choice = json!({"index": 0, "message": message, "finish_reason": r.finish});
        if opts.logprobs { choice["logprobs"] = crate::logprobs_field(true, &r.logprobs); }
        Ok(json!({"id": "chatcmpl-ferric", "object": "chat.completion", "created": now_unix(), "model": self.eng.name,
                  "choices": [choice],
                  "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.gen_tokens, "total_tokens": r.prompt_tokens + r.gen_tokens},
                  "energy": r.energy}))
    }

    /// A `/v1/completions` body (a string `prompt`) → a `text_completion` object.
    pub fn complete(&self, req: &Value, mut on_delta: impl FnMut(&str)) -> Result<Value, String> {
        if req["n"].as_u64().is_some_and(|n| n > 1) { return Err("`n` > 1 is served over HTTP; ask one choice at a time here".into()); }
        let opts = self.eng.gen_opts(req, false)?;
        let prompt = req["prompt"].as_str().ok_or("`prompt` must be a string")?;
        let mut ids = Vec::new();
        if self.eng.add_bos { if let Some(b) = self.eng.bos_id { ids.push(b); } }
        ids.extend(self.eng.enc(prompt, true));
        let max = self.eng.budget(ids.len(), &opts)?;
        let spec = self.eng.constraint(req)?;
        let out = self.eng.generate(&ids, max, &opts, spec.guide(), |d, _| on_delta(d));
        let mut choice = json!({"index": 0, "text": out.text, "finish_reason": out.finish});
        if opts.logprobs { choice["logprobs"] = crate::logprobs_field(false, &out.logprobs); }
        Ok(json!({"id": "cmpl-ferric", "object": "text_completion", "created": now_unix(), "model": self.eng.name,
                  "choices": [choice],
                  "usage": {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.gen_tokens, "total_tokens": out.prompt_tokens + out.gen_tokens},
                  "energy": out.energy}))
    }
}
