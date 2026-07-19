//! **Ferric C ABI (`libferric`)** — the universal on-ramp. Any language with C FFI drives the
//! pure-Rust cross-fabric runtime through these functions: `ferric_load` a GGUF, then
//! `ferric_generate` (free text) or `ferric_generate_json` (schema-constrained, guaranteed-conformant
//! JSON via guided decoding). Zig `@cImport`s the header directly; Mojo/Go/Java/C#/C++ bind the same.
//! Strings returned by generate must be released with `ferric_free_string`; the handle with `ferric_free`.
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

pub struct FerricHandle {
    ctx: Arc<Context>,
    model: Qwen3,
    bpe: Bpe,
    toks: Vec<String>,
    u2b: HashMap<char, u8>,
    token_bytes: Vec<Option<Vec<u8>>>,
    add_bos: bool,
    bos_id: Option<u32>,
}

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

impl FerricHandle {
    fn load(path: &str) -> Result<FerricHandle, String> {
        let g = GgufFile::open(path).map_err(|e| format!("{e:?}"))?;
        let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
            _ => return Err("no tokens".into()),
        };
        let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
            Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(),
            _ => Vec::new(),
        };
        let bpe = Bpe::new(vocab, &merges);
        let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
        let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
        let u2b = byte_decoder();
        let token_bytes: Vec<Option<Vec<u8>>> = toks.iter().map(|t| {
            let mut b = Vec::with_capacity(t.len());
            for c in t.chars() { match u2b.get(&c) { Some(&x) => b.push(x), None => return None } }
            Some(b)
        }).collect();
        let ctx = Arc::new(pollster::block_on(Context::new()).map_err(|e| format!("{e:?}"))?);
        let model = Qwen3::load(&ctx, &g)?;
        Ok(FerricHandle { ctx, model, bpe, toks, u2b, token_bytes, add_bos, bos_id })
    }

    fn detok(&self, ids: &[u32]) -> String {
        let s: String = ids.iter().map(|&i| self.toks.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| self.u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    }

    fn run(&self, prompt: &str, max_tokens: usize, mut guide: Option<ferric_agent::guide::Guide>) -> String {
        let c = &self.model.cfg;
        let mut ids = self.bpe.encode(prompt);
        if self.add_bos { if let Some(b) = self.bos_id { ids.insert(0, b); } }
        if ids.is_empty() { return String::new(); }
        let eos = |t: u32| t == 151645 || t == 151643;
        let mut cache = Cache::new(c);
        let mut seq = ids.clone();
        for step in 0..max_tokens {
            let logits = if step == 0 { self.model.forward_cached(&ids, &mut cache) } else { self.model.forward_cached(&seq[seq.len() - 1..], &mut cache) };
            let v = pollster::block_on(logits.to_vec());
            let row = &v[v.len() - c.n_vocab..];
            let next = if let Some(g) = guide.as_ref() {
                let can_stop = g.can_stop();
                let (mut best, mut bl) = (None, f32::NEG_INFINITY);
                for i in 0..c.n_vocab {
                    let ok = if eos(i as u32) { can_stop } else { match &self.token_bytes[i] { Some(b) if !b.is_empty() => { let mut a = *g; b.iter().all(|&ch| a.step(ch)) } _ => false } };
                    if ok && row[i] > bl { bl = row[i]; best = Some(i as u32); }
                }
                match best { Some(t) => t, None => break }
            } else { (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32 };
            if eos(next) { break; }
            if let Some(g) = guide.as_mut() { if let Some(b) = &self.token_bytes[next as usize] { for &ch in b { g.step(ch); } } }
            seq.push(next);
        }
        self.detok(&seq[ids.len()..])
    }
}

fn cstr<'a>(p: *const c_char) -> Option<&'a str> { if p.is_null() { None } else { unsafe { CStr::from_ptr(p) }.to_str().ok() } }
fn out(s: String) -> *mut c_char { CString::new(s).unwrap_or_default().into_raw() }

/// Load a GGUF model. Returns an opaque handle, or NULL on failure.
#[no_mangle]
pub extern "C" fn ferric_load(model_path: *const c_char) -> *mut FerricHandle {
    let Some(path) = cstr(model_path) else { return std::ptr::null_mut() };
    match FerricHandle::load(path) { Ok(h) => Box::into_raw(Box::new(h)), Err(_) => std::ptr::null_mut() }
}

/// Greedy free-text completion of `prompt` for up to `max_tokens`. Caller frees the result string.
#[no_mangle]
pub extern "C" fn ferric_generate(h: *mut FerricHandle, prompt: *const c_char, max_tokens: u32) -> *mut c_char {
    let (Some(h), Some(p)) = (unsafe { h.as_ref() }, cstr(prompt)) else { return out(String::new()) };
    out(h.run(p, max_tokens as usize, None))
}

/// Schema-constrained generation: output is guaranteed-conformant JSON. `schema` is a JSON-Schema
/// string (empty → any valid JSON object). Caller frees the result string.
#[no_mangle]
pub extern "C" fn ferric_generate_json(h: *mut FerricHandle, prompt: *const c_char, schema: *const c_char, max_tokens: u32) -> *mut c_char {
    let (Some(h), Some(p)) = (unsafe { h.as_ref() }, cstr(prompt)) else { return out(String::new()) };
    let sch = ferric_agent::guide::compile_str(cstr(schema).unwrap_or(""));
    let guide = match &sch {
        Some(prog) => ferric_agent::guide::Guide::Schema(ferric_agent::guide::Schema::new(prog)),
        None => ferric_agent::guide::Guide::Json(ferric_agent::guide::Json::object()),
    };
    out(h.run(p, max_tokens as usize, Some(guide)))
}

/// Free a string returned by `ferric_generate*`.
#[no_mangle]
pub extern "C" fn ferric_free_string(s: *mut c_char) { if !s.is_null() { unsafe { drop(CString::from_raw(s)); } } }

/// Free a model handle.
#[no_mangle]
pub extern "C" fn ferric_free(h: *mut FerricHandle) { if !h.is_null() { unsafe { drop(Box::from_raw(h)); } } }
