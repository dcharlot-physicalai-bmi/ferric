//! **Ferric C ABI (`libferric`)** — the universal on-ramp. Any language with C FFI drives Ferric through
//! these functions; Zig `@cImport`s the header, Python uses ctypes (`bindings/python`), Go/Swift/Mojo/C#
//! bind the same symbols.
//!
//! The handle is ferric-serve's own engine (`ferric_serve::LocalModel`): every architecture the registry
//! serves, the model's own chat template, its tokenizer, the full sampler set, constrained decoding
//! (JSON Schema, GBNF, regex, choice), reasoning split and per-request energy. So a binding answers exactly
//! what `ferric-serve` answers over HTTP for the same request (`scripts/ffi_check.py` holds it to that).
//!
//! - `ferric_chat` / `ferric_chat_stream`: an OpenAI `/v1/chat/completions` body in, a `chat.completion`
//!   JSON object out (`energy` included); the stream variant calls back per delta.
//! - `ferric_complete`: a `/v1/completions` body in, a `text_completion` object out.
//! - `ferric_generate` / `ferric_generate_json`: the original two calls, now on the same engine (greedy
//!   completion; schema-constrained JSON).
//! Every string returned is released with `ferric_free_string`; the handle with `ferric_free`. An error is
//! returned as `{"error": {"message": ...}}` rather than NULL, so a caller always gets a reason.
use serde_json::{json, Value};
use std::ffi::{c_char, c_int, c_void, CStr, CString};

pub struct FerricHandle { m: ferric_serve::LocalModel }

fn cstr<'a>(p: *const c_char) -> Option<&'a str> { if p.is_null() { None } else { unsafe { CStr::from_ptr(p) }.to_str().ok() } }
fn out(s: String) -> *mut c_char { CString::new(s.replace('\0', "")).unwrap_or_default().into_raw() }
fn err(m: &str) -> *mut c_char { out(json!({"error": {"message": m}}).to_string()) }

/// Streaming callback: `delta` (UTF-8, NUL-terminated, valid only during the call), `is_reasoning` 1 for a
/// thinking model's reasoning, 0 for the answer; `user` is passed through.
pub type FerricDeltaCb = Option<extern "C" fn(delta: *const c_char, is_reasoning: c_int, user: *mut c_void)>;

/// Load a GGUF model. Returns an opaque handle, or NULL on failure (the reason goes to stderr).
#[unsafe(no_mangle)]
pub extern "C" fn ferric_load(model_path: *const c_char) -> *mut FerricHandle {
    let Some(path) = cstr(model_path) else { return std::ptr::null_mut() };
    match ferric_serve::LocalModel::load(path) {
        Ok(m) => Box::into_raw(Box::new(FerricHandle { m })),
        Err(e) => { eprintln!("ferric_load: {e}"); std::ptr::null_mut() }
    }
}

fn chat(h: *mut FerricHandle, request_json: *const c_char, cb: FerricDeltaCb, user: *mut c_void) -> *mut c_char {
    let Some(h) = (unsafe { h.as_ref() }) else { return err("null handle") };
    let req: Value = match cstr(request_json).map(serde_json::from_str) { Some(Ok(v)) => v, Some(Err(e)) => return err(&format!("bad json: {e}")), None => return err("null request") };
    let r = h.m.chat(&req, |d, reasoning| if let Some(f) = cb {
        let c = CString::new(d.replace('\0', "")).unwrap_or_default();
        f(c.as_ptr(), reasoning as c_int, user);
    });
    match r { Ok(v) => out(v.to_string()), Err(e) => err(&e) }
}

/// An OpenAI `/v1/chat/completions` request (JSON) → the `chat.completion` object (JSON). Caller frees.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_chat(h: *mut FerricHandle, request_json: *const c_char) -> *mut c_char {
    chat(h, request_json, None, std::ptr::null_mut())
}

/// `ferric_chat`, calling `cb` with each piece of the answer as it is generated. Returns the final object.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_chat_stream(h: *mut FerricHandle, request_json: *const c_char, cb: FerricDeltaCb, user: *mut c_void) -> *mut c_char {
    chat(h, request_json, cb, user)
}

/// An OpenAI `/v1/completions` request (JSON, a string `prompt`) → the `text_completion` object. Caller frees.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_complete(h: *mut FerricHandle, request_json: *const c_char) -> *mut c_char {
    let Some(h) = (unsafe { h.as_ref() }) else { return err("null handle") };
    let req: Value = match cstr(request_json).map(serde_json::from_str) { Some(Ok(v)) => v, Some(Err(e)) => return err(&format!("bad json: {e}")), None => return err("null request") };
    match h.m.complete(&req, |_| {}) { Ok(v) => out(v.to_string()), Err(e) => err(&e) }
}

fn text_of(v: &Value) -> String { v["choices"][0]["text"].as_str().unwrap_or_default().to_string() }

/// Greedy free-text completion of `prompt` for up to `max_tokens`. Caller frees the result string.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_generate(h: *mut FerricHandle, prompt: *const c_char, max_tokens: u32) -> *mut c_char {
    let (Some(h), Some(p)) = (unsafe { h.as_ref() }, cstr(prompt)) else { return out(String::new()) };
    out(h.m.complete(&json!({"prompt": p, "max_tokens": max_tokens, "temperature": 0}), |_| {}).map(|v| text_of(&v)).unwrap_or_default())
}

/// Schema-constrained generation: output is guaranteed-conformant JSON. `schema` is a JSON-Schema
/// string (empty → any valid JSON object). Caller frees the result string.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_generate_json(h: *mut FerricHandle, prompt: *const c_char, schema: *const c_char, max_tokens: u32) -> *mut c_char {
    let (Some(h), Some(p)) = (unsafe { h.as_ref() }, cstr(prompt)) else { return out(String::new()) };
    let format = match cstr(schema).map(str::trim).filter(|s| !s.is_empty()).map(serde_json::from_str::<Value>) {
        Some(Ok(s)) => json!({"type": "json_schema", "json_schema": {"schema": s}}),
        Some(Err(_)) => return out(String::new()),
        None => json!({"type": "json_object"}),
    };
    out(h.m.complete(&json!({"prompt": p, "max_tokens": max_tokens, "temperature": 0, "response_format": format}), |_| {})
        .map(|v| text_of(&v)).unwrap_or_default())
}

/// Free a string returned by any `ferric_*` call.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_free_string(s: *mut c_char) { if !s.is_null() { unsafe { drop(CString::from_raw(s)); } } }

/// Free a model handle.
#[unsafe(no_mangle)]
pub extern "C" fn ferric_free(h: *mut FerricHandle) { if !h.is_null() { unsafe { drop(Box::from_raw(h)); } } }
