//! **Ferric OpenAI-compatible server** — the adoption on-ramp. A dependency-light (std TCP + serde)
//! HTTP server exposing `/v1/chat/completions` (streaming + non-streaming), `/v1/completions`,
//! `/v1/models`, and `/health` over the pure-Rust cross-fabric runtime. Any OpenAI client, agent
//! framework (LangChain, LangGraph, the Vercel AI SDK, …), or `curl` points at it unchanged.
//!
//!   cargo run -p ferric-serve --release -- <model.gguf> [--port 8080] [--name my-model]
//!
//! **Structured output** (`/v1/chat/completions` only). `response_format` constrains generation
//! *in-runtime* by masking the sampler to the bytes that keep the output a valid JSON prefix — so the
//! model can only emit conformant JSON, deterministically across fabrics (see `ferric_agent::guide`):
//!   - `{"type":"json_object"}` → any single valid JSON object.
//!   - `{"type":"json_schema","json_schema":{"schema": <JSON-Schema>}}` → schema-conformant JSON.
//! Supported schema: objects with `properties` in *declaration* order, `required` (absent ⇒ all
//! required; a subset makes the rest optional & skippable), nesting ≤ 8 and ≤ 32 props per object;
//! `string` (with `minLength`/`maxLength` in Unicode code points), `integer` (with inclusive
//! `minimum`/`maximum` — bounds the value so it can't run away, in object fields AND arrays),
//! `number`, `boolean`, `enum`; and typed arrays of those (`minItems`/`maxItems`). Deeper/wider/
//! unsupported shapes fall back to free-but-valid JSON — never a hard error. `temperature` is honored
//! over the legal-token set (0 = greedy/deterministic). Caveat: float `number` fields aren't
//! magnitude-bounded yet, so a small model can loop digits until `max_tokens` — use an `integer` with
//! bounds where possible, set `maxLength`/`maxItems`, or use adequate `max_tokens`.
//!
//! One request at a time (the GPU serializes anyway); continuous batching is the P1 follow-up.
mod mcp;
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::{qwen3, qwen35};
use ferric_llama::qwen3::Qwen3;
use ferric_llama::qwen35::Qwen35;
use ferric_tensor::Tensor;

/// The loaded model — a dense Qwen3/Llama/Gemma/Phi, or the Qwen3.5/3.6 **GDN-hybrid** (gated delta net
/// + periodic full attention). Both expose a `forward_cached` returning logits, so the generate loop and
/// guided decoding are architecture-agnostic; only the KV/recurrent cache type differs.
enum Model { Dense(Qwen3), Hybrid(Qwen35) }
enum ModelCache { Dense(qwen3::Cache), Hybrid(qwen35::Cache) }
impl Model {
    fn n_vocab(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_vocab, Model::Hybrid(m) => m.cfg.n_vocab } }
    fn n_layer(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_layer, Model::Hybrid(m) => m.cfg.n_layer } }
    fn n_embd(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_embd, Model::Hybrid(m) => m.cfg.n_embd } }
    fn new_cache(&self) -> ModelCache {
        match self { Model::Dense(m) => ModelCache::Dense(qwen3::Cache::new(&m.cfg)), Model::Hybrid(m) => ModelCache::Hybrid(qwen35::Cache::new(&m.cfg)) }
    }
    fn forward_cached(&self, tokens: &[u32], cache: &mut ModelCache) -> Tensor {
        match (self, cache) {
            (Model::Dense(m), ModelCache::Dense(c)) => m.forward_cached(tokens, c),
            (Model::Hybrid(m), ModelCache::Hybrid(c)) => m.forward_cached(tokens, c, m.cfg.n_layer),
            _ => unreachable!("model/cache kind mismatch"),
        }
    }
    fn forward_hidden(&self, ids: &[u32]) -> Tensor {
        match self {
            Model::Dense(m) => m.forward_hidden(ids),
            // Same semantics as the dense path: all layers, then the final norm (what LAST-pooling
            // embedding references pool).
            Model::Hybrid(m) => {
                let mut c = qwen35::Cache::new(&m.cfg);
                m.forward_hidden_cached(ids, &mut c, m.cfg.n_layer).rmsnorm(&m.out_norm, m.cfg.eps)
            }
        }
    }
}
use ferric_tokenizer::{Bpe, Spm};
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
    model: Model,
    bpe: Bpe,
    /// Present for SentencePiece models (`tokenizer.ggml.model == "llama"`: Phi-3 / Mistral / Llama-2 /
    /// Gemma). When set, all text tokenization goes through it instead of the byte-level `bpe`.
    spm: Option<Spm>,
    /// SentencePiece `add_space_prefix` (default true; Gemma sets false): prepend a leading ▁ to the
    /// first text fragment. Wrong value → the first token differs from llama.cpp.
    add_space_prefix: bool,
    tokens: Vec<String>,
    u2b: HashMap<char, u8>,
    im_start: Option<u32>,
    im_end: Option<u32>,
    bos_id: Option<u32>,
    add_bos: bool,
    /// `tokenizer.ggml.eos_token_id` + `add_eos_token` — embedding models (Qwen3-Embedding) append EOS
    /// and pool ITS hidden state (pooling_type=LAST), so `embed` must append it to match the reference.
    eos_id: Option<u32>,
    add_eos: bool,
    eos: Vec<u32>,
    name: String,
    /// Raw bytes each token decodes to (for guided decoding); `None` for special/non-text tokens,
    /// which are disallowed under a constraint (except EOS, handled separately).
    token_bytes: Vec<Option<Vec<u8>>>,
    /// (special-token string, id), longest first — for special-token-aware tokenization of a
    /// rendered chat template (so `<|im_start|>` etc. encode to their id, not BPE'd text).
    specials: Vec<(String, u32)>,
    /// The GGUF `chat_template` string (used only to detect the model's template family).
    template: String,
    /// One-slot prompt-prefix cache (hybrid speculative path): the last request's fed tokens plus
    /// the carried main + draft caches. Multi-turn chat re-sends the whole conversation; when the
    /// new prompt extends the cached tokens, only the new suffix is prefilled. Output is identical
    /// to a full prefill (cached-decode ≡ re-prefill, the invariant `--verify-cache` proves) —
    /// the conversation just stops being re-paid every turn. RefCell: the server is single-threaded.
    prefix: std::cell::RefCell<Option<PrefixSlot>>,
}

/// See `Engine::prefix`. `fed` is exactly the token sequence the main cache has consumed —
/// kept consistent through speculative rollbacks.
struct PrefixSlot { fed: Vec<u32>, cache: qwen35::Cache, mc: qwen35::MtpCache }

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
        // SentencePiece models carry a per-token score array and no merges — detect and build an Spm.
        let spm = match g.metadata.get("tokenizer.ggml.model") {
            Some(Meta::Str(s)) if s == "llama" => {
                let scores: Vec<f32> = match g.metadata.get("tokenizer.ggml.scores") {
                    Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::F(v) = m { *v as f32 } else { 0.0 }).collect(),
                    _ => Vec::new(),
                };
                Some(Spm::new(tokens.clone(), scores))
            }
            _ => None,
        };
        let add_space_prefix = match g.metadata.get("tokenizer.ggml.add_space_prefix") { Some(Meta::Bool(b)) => *b, _ => true };
        let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
        let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
        let eos_id = match g.metadata.get("tokenizer.ggml.eos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
        let add_eos = matches!(g.metadata.get("tokenizer.ggml.add_eos_token"), Some(Meta::Bool(true)));
        let mut eos: Vec<u32> = Vec::new();
        if let Some(e) = eos_id { eos.push(e); }
        let im_end = vocab.get("<|im_end|>").copied();
        let im_start = vocab.get("<|im_start|>").copied();
        if let Some(e) = im_end { if !eos.contains(&e) { eos.push(e); } }
        if let Some(&e) = vocab.get("<|endoftext|>") { if !eos.contains(&e) { eos.push(e); } }
        // Gemma ends a turn with <end_of_turn>; Phi-3 with <|end|> — treat both as stop tokens.
        for t in ["<end_of_turn>", "<|end|>"] { if let Some(&e) = vocab.get(t) { if !eos.contains(&e) { eos.push(e); } } }
        // Dispatch on architecture: the Qwen3.5/3.6 hybrid GGUFs declare `general.architecture = qwen35`
        // (dense FFN) or `qwen35moe` (mixture-of-experts FFN) — both run on the Qwen35 hybrid runtime;
        // everything else (qwen2/qwen3/llama/gemma/phi) is the dense path.
        let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
        let model = if arch.starts_with("qwen35") || arch == "laguna" {
            Model::Hybrid(Qwen35::load(&ctx, &g).unwrap_or_else(|e| panic!("load hybrid model: {e}")))
        } else {
            Model::Dense(Qwen3::load(&ctx, &g).unwrap_or_else(|e| panic!("load model: {e}")))
        };
        let u2b = byte_decoder();
        // Precompute each token's raw bytes (chars → bytes via u2b). A token containing any char not in
        // the byte map is a special token (e.g. <|im_end|>) → None → disallowed under a constraint.
        let token_bytes: Vec<Option<Vec<u8>>> = if let Some(sp) = &spm {
            (0..tokens.len() as u32).map(|i| sp.token_bytes(i)).collect()
        } else {
            tokens.iter().map(|t| {
                let mut b = Vec::with_capacity(t.len());
                for c in t.chars() { match u2b.get(&c) { Some(&x) => b.push(x), None => return None } }
                Some(b)
            }).collect()
        };
        // Special (control) tokens for template-aware tokenization: prefer the GGUF token_type array
        // (3 = CONTROL); else fall back to the reliable `<|…|>` pattern (ChatML/Llama-3 style).
        let ttypes: Vec<i64> = match g.metadata.get("tokenizer.ggml.token_type") {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::I(v) = m { *v } else if let Meta::U(v) = m { *v as i64 } else { 0 }).collect(),
            _ => Vec::new(),
        };
        let mut specials: Vec<(String, u32)> = tokens.iter().enumerate().filter_map(|(i, t)| {
            // Union: token_type CONTROL(3) or USER_DEFINED(4) (Llama-3's <|…|> tokens are 4!), OR the
            // reliable angle-bracket control patterns — so no template's special tokens get BPE'd.
            let is_ctrl = matches!(ttypes.get(i), Some(&3) | Some(&4))
                || (t.starts_with("<|") && t.ends_with("|>"))
                || matches!(t.as_str(), "<s>" | "</s>" | "<bos>" | "<eos>" | "<pad>" | "<unk>" | "<mask>" | "<start_of_turn>" | "<end_of_turn>");
            if is_ctrl && !t.is_empty() { Some((t.clone(), i as u32)) } else { None }
        }).collect();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len())); // longest-match first
        let template = match g.metadata.get("tokenizer.ggml.chat_template") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
        Engine { ctx, model, bpe, spm, add_space_prefix, tokens, u2b, im_start, im_end, bos_id, add_bos, eos_id, add_eos, eos, name, token_bytes, specials, template, prefix: std::cell::RefCell::new(None) }
    }

    /// Tokenize a raw-text fragment through whichever tokenizer this model uses. `at_start` = this is
    /// the first fragment of the sequence → apply SentencePiece's leading-space (gated by the model's
    /// `add_space_prefix`; ignored by byte-level BPE, which encodes spaces directly).
    fn enc(&self, text: &str, at_start: bool) -> Vec<u32> {
        match &self.spm { Some(sp) => sp.encode_piece(text, at_start && self.add_space_prefix), None => self.bpe.encode(text) }
    }

    /// Split `text` on control tokens (longest match) and encode: control tokens → their id, the text
    /// between → byte-level BPE. Lets a rendered chat template carry literal `<|im_start|>` etc.
    fn encode_special(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // find the earliest special-token occurrence
            let mut best: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(pos) = rest.find(s.as_str()) {
                    if best.map(|(bp, _, _)| pos < bp).unwrap_or(true) { best = Some((pos, s, *id)); }
                }
            }
            match best {
                Some((pos, s, id)) => {
                    if pos > 0 { let p = ids.is_empty(); ids.extend(self.enc(&rest[..pos], p)); }
                    ids.push(id);
                    rest = &rest[pos + s.len()..];
                }
                None => { let p = ids.is_empty(); ids.extend(self.enc(rest, p)); break 'outer; }
            }
        }
        ids
    }

    /// Is this control token in the model's vocab?
    fn has(&self, s: &str) -> bool { self.specials.iter().any(|(t, _)| t == s) }

    /// Detect the chat family from the control tokens actually present in the vocab (robust even when
    /// the GGUF omits `tokenizer.ggml.chat_template`).
    fn has_chat_family(&self) -> bool {
        self.has("<|im_start|>") || self.has("<|start_header_id|>") || self.has("<start_of_turn>") || (self.has("<|assistant|>") && self.has("<|end|>"))
    }

    /// Render the chat template to a string (special tokens as literal text), family-detected from the
    /// vocab. Covers ChatML (Qwen/Yi/…), Llama-3, Gemma, Phi-3; else a generic fallback.
    fn render_chat(&self, messages: &[Value]) -> String {
        let m = |v: &Value| (v["role"].as_str().unwrap_or("user").to_string(), v["content"].as_str().unwrap_or("").to_string());
        if self.has("<|start_header_id|>") { // Llama-3
            let mut s = String::from("<|begin_of_text|>");
            for v in messages { let (r, c) = m(v); s.push_str(&format!("<|start_header_id|>{r}<|end_header_id|>\n\n{c}<|eot_id|>")); }
            s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            s
        } else if self.has("<start_of_turn>") { // Gemma (roles user/model, no system → fold into first user)
            let mut s = String::new();
            let mut sys = String::new();
            for v in messages { let (r, c) = m(v);
                if r == "system" { sys = c; continue; }
                let role = if r == "assistant" { "model" } else { "user" };
                let body = if role == "user" && !sys.is_empty() { let b = format!("{sys}\n\n{c}"); sys.clear(); b } else { c };
                s.push_str(&format!("<start_of_turn>{role}\n{body}<end_of_turn>\n"));
            }
            s.push_str("<start_of_turn>model\n");
            s
        } else if self.has("<|assistant|>") && self.has("<|end|>") { // Phi-3
            let mut s = String::new();
            for v in messages { let (r, c) = m(v); s.push_str(&format!("<|{r}|>\n{c}<|end|>\n")); }
            s.push_str("<|assistant|>\n");
            s
        } else { // ChatML (default — Qwen and most GGUF chat models)
            let mut s = String::new();
            for v in messages { let (r, c) = m(v); s.push_str(&format!("<|im_start|>{r}\n{c}<|im_end|>\n")); }
            s.push_str("<|im_start|>assistant\n");
            s
        }
    }

    fn detok(&self, ids: &[u32]) -> String {
        if let Some(sp) = &self.spm { return sp.decode(ids); }
        let s: String = ids.iter().map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| self.u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    }

    /// Build the prompt token stream from OpenAI `messages`: render the model's own chat template
    /// (family-detected from the GGUF) to a string, then tokenize special-token-aware. The template
    /// is self-contained (it carries its own BOS, e.g. Llama-3's `<|begin_of_text|>`), so BOS is not
    /// prepended separately. Byte-identical to the old hardcoded path for ChatML models.
    /// Embed one text → an L2-normalized vector. Runs the transformer, takes the last token's hidden
    /// state (Qwen3-Embedding's last-token pooling, pooling_type=3), and normalizes. Same model code as
    /// generation — this is just the pre-lm_head hidden state, pooled.
    fn embed(&self, text: &str) -> Vec<f32> {
        let n = self.model.n_embd();
        let mut ids = self.enc(text, true);
        // Qwen3-Embedding (add_eos_token) appends EOS and pools ITS hidden state; append it to match.
        if self.add_eos { if let Some(e) = self.eos_id { ids.push(e); } }
        // Empty input: keep the response's vectors equal-length (a zero vector), not a []; some clients
        // build a matrix over a batch and a ragged row breaks them.
        if ids.is_empty() { return vec![0.0; n]; }
        let v = pollster::block_on(self.model.forward_hidden(&ids).to_vec()); // [T·n_embd]
        let t = (v.len() / n).max(1);
        let last = &v[(t - 1) * n..t * n]; // last-token pool (the appended EOS when present)
        let norm = last.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        last.iter().map(|x| x / norm).collect()
    }

    fn chat_ids(&self, messages: &[Value]) -> Vec<u32> {
        if !self.has_chat_family() {
            // No recognized chat family in the vocab → a base model — plain concatenation.
            let text: String = messages.iter().map(|m| format!("{}: {}\n", m["role"].as_str().unwrap_or("user"), m["content"].as_str().unwrap_or(""))).collect();
            let mut ids = Vec::new();
            if self.add_bos { if let Some(b) = self.bos_id { ids.push(b); } }
            ids.extend(self.enc(&text, true));
            return ids;
        }
        let mut ids = self.encode_special(&self.render_chat(messages));
        // SentencePiece templates (Phi-3/Mistral) don't embed BOS; add_bos prepends it. (BPE templates
        // like Llama-3 carry their own <|begin_of_text|>, so guard on it not already leading.)
        if self.spm.is_some() && self.add_bos {
            if let Some(b) = self.bos_id { if ids.first() != Some(&b) { ids.insert(0, b); } }
        }
        ids
    }

    /// Decode. `temperature` 0 → greedy argmax (deterministic — the default). >0 → top-p sampling
    /// with a **fixed-seed** RNG, so even sampled output is reproducible (on-brand for the moat).
    /// Guided decoding always stays argmax (deterministic structured output). Calls `on_delta` per
    /// newly-decoded fragment. Returns (full_text, prompt_tokens, gen_tokens).
    fn generate(&self, prompt: &[u32], max_tokens: usize, temperature: f32, mut guide: Option<ferric_agent::guide::Guide>, mut on_delta: impl FnMut(&str)) -> (String, usize, usize) {
        // Speculative fast path: a hybrid model shipping its own MTP draft block self-drafts.
        // Emits IDENTICAL tokens (drafts are only accepted when they equal what the sampler picks
        // from the true logits, and the fixed-seed RNG advances once per emitted token either way)
        // — the same determinism story, just fewer main-model forwards.
        // Debug: FERRIC_DUMP_IDS=1 prints each request's prompt token ids (replayable in run_bonsai).
        if std::env::var("FERRIC_DUMP_IDS").is_ok() { eprintln!("prompt ids ({}): {:?}", prompt.len(), prompt); }
        if let Model::Hybrid(m) = &self.model {
            // FERRIC_NOSPEC=1 forces the plain loop (A/B + regression escape hatch).
            if m.mtp.is_some() && std::env::var("FERRIC_NOSPEC").is_err() {
                return self.generate_spec(m, prompt, max_tokens, temperature, guide, on_delta);
            }
        }
        let mut cache = self.model.new_cache();
        let n_vocab = self.model.n_vocab();
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D; // deterministic seed → reproducible sampling
        let mut gen: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        for step in 0..max_tokens {
            let input: Vec<u32> = if step == 0 { prompt.to_vec() } else { vec![*gen.last().unwrap()] };
            let logits = self.model.forward_cached(&input, &mut cache);
            let v = pollster::block_on(logits.to_vec());
            let Some(next) = self.select_token(&v[v.len() - n_vocab..], &guide, temperature, &mut rng) else { break };
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

    /// Pick the next token from a row of TRUE model logits: guided decoding masks illegal tokens to
    /// -inf (EOS legal only once the value is complete), then `temperature` is honored over the
    /// legal set — temp 0 stays argmax (deterministic). Returns `None` when the guide leaves no
    /// legal continuation (stop cleanly). Most tokens reject on their first byte, so the scan is cheap.
    fn select_token(&self, row: &[f32], guide: &Option<ferric_agent::guide::Guide>, temperature: f32, rng: &mut u64) -> Option<u32> {
        let n_vocab = row.len();
        let argmax = |r: &[f32]| (0..n_vocab).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap() as u32;
        if let Some(g) = guide.as_ref() {
            let can_stop = g.can_stop();
            let mut masked = vec![f32::NEG_INFINITY; n_vocab];
            let mut any = false;
            for i in 0..n_vocab {
                let ok = if self.eos.contains(&(i as u32)) { can_stop }
                    else { match &self.token_bytes[i] { Some(b) if !b.is_empty() => { let mut a = *g; b.iter().all(|&c| a.step(c)) } _ => false } };
                if ok { masked[i] = row[i]; any = true; }
            }
            if !any { return None; } // no legal continuation (shouldn't happen for a valid schema)
            Some(if temperature > 0.0 { sample_top_p(&masked, temperature, 0.95, rng) } else { argmax(&masked) })
        } else if temperature > 0.0 { Some(sample_top_p(row, temperature, 0.95, rng)) } else { Some(argmax(row)) }
    }

    /// Speculative decoding with the model's own MTP ("nextn") draft block. Every emitted token is
    /// selected by `select_token` from true model logits at a verified position — never from the
    /// draft — and the fixed-seed RNG advances once per emitted token, so output is fully
    /// deterministic (same request → same bytes, every run). It equals the plain loop's output
    /// whenever logit gaps exceed kernel fp-order (measured: 64-token unguided greedy identical);
    /// a restrictive guide mask can leave near-tie candidates where the multi-token verify's
    /// fp-order picks a different (equally legal) token than the single-token path — the same
    /// class of shift as any kernel-fusion change. The draft only decides how many tokens each
    /// main forward yields (~80% acceptance ⇒ ~2 per forward). Rollback on rejection is O(1):
    /// caches are Arc-handle snapshots, never GPU copies.
    fn generate_spec(&self, m: &Qwen35, prompt: &[u32], max_tokens: usize, temperature: f32, mut guide: Option<ferric_agent::guide::Guide>, mut on_delta: impl FnMut(&str)) -> (String, usize, usize) {
        let n_vocab = self.model.n_vocab();
        let argmax = |r: &[f32]| (0..n_vocab).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap() as u32;
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut gen: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        // One-slot prompt-prefix reuse: when this prompt extends the cached conversation, resume
        // its caches and prefill only the new suffix.
        let (mut fed, mut cache, mut mc) = match self.prefix.borrow_mut().take() {
            Some(s) if prompt.len() > s.fed.len() && prompt[..s.fed.len()] == s.fed[..] => (s.fed, s.cache, s.mc),
            _ => {
                let mut mc = qwen35::MtpCache::default();
                mc.pos = 1; // first draft pair (prompt token 1, hidden 0) sits at position 1
                (Vec::new(), qwen35::Cache::new(&m.cfg), mc)
            }
        };
        // Commit one token: advance the guide, then stream the newly-decoded suffix.
        macro_rules! commit {
            ($tok:expr) => {{
                if let (Some(g), Some(b)) = (guide.as_mut(), self.token_bytes[$tok as usize].as_ref()) { for &c in b { g.step(c); } }
                gen.push($tok);
                let full = self.detok(&gen);
                if full.len() > emitted.len() && full.is_char_boundary(emitted.len()) {
                    let delta = full[emitted.len()..].to_string();
                    on_delta(&delta);
                    emitted = full;
                }
            }};
        }
        macro_rules! save_slot {
            () => { *self.prefix.borrow_mut() = Some(PrefixSlot { fed, cache, mc }); };
        }
        // Prompt (or suffix) prefill; the hidden rows also seed the draft block's cache (pairs for
        // the new positions — without them the drafter is blind to the prompt).
        let p0 = fed.len(); // hid row i ↔ absolute position p0 + i
        let suffix: Vec<u32> = prompt[p0..].to_vec();
        let (lg, hid) = m.forward_spec(&suffix, &mut cache, m.cfg.n_layer);
        fed.extend_from_slice(&suffix);
        let v = pollster::block_on(lg.to_vec());
        let first = self.select_token(&v[v.len() - n_vocab..], &guide, temperature, &mut rng);
        let Some(pend0) = first.filter(|t| !self.eos.contains(t)) else { save_slot!(); return (emitted, prompt.len(), 0) };
        commit!(pend0);
        let mut unfed: Vec<u32> = vec![pend0]; // committed tokens the main cache hasn't seen yet
        // Draft pairs resume at the first position the draft cache lacks — but no earlier than the
        // first position whose predecessor hidden we have. A positional gap in the draft cache
        // (possible after an early-EOS request) only costs it context: rope offsets are explicit,
        // and drafts are guesses the main model verifies anyway.
        let start = mc.pos.max(p0 + 1);
        mc.pos = start;
        let mut ptoks: Vec<u32> = prompt[start..].to_vec();
        ptoks.push(pend0);
        let mut phid = hid.narrow(0, start - 1 - p0, prompt.len() - start + 1).contiguous();
        // FERRIC_SPEC_DRAFT=2 → recursively draft a 2nd token per verify (~19% faster; measured d2
        // conditional acceptance ~65-71%). The larger verify `t` flips near-tie logits vs single-token
        // decode a little more often than the 1-token path, so 1-token stays the byte-identical default.
        let draft2 = std::env::var("FERRIC_SPEC_DRAFT").ok().as_deref() == Some("2");
        while gen.len() < max_tokens {
            if draft2 {
                // Draft d1 (advances the real mc past ptoks), then d2 recursively on a throwaway clone.
                let (l1, h1) = m.mtp_forward_h(&ptoks, &phid, &mut mc);
                let d1 = argmax(&pollster::block_on(l1.to_vec())[..n_vocab]);
                let mut probe = mc.clone();
                let l2 = m.mtp_forward_h(&[d1], &h1, &mut probe).0;
                let d2 = argmax(&pollster::block_on(l2.to_vec())[..n_vocab]);
                // Verify [unfed…, d1, d2] — head the last 3 rows (d1-check, d2-check, pend).
                let snap = cache.snapshot();
                let k = unfed.len();
                let toks: Vec<u32> = unfed.iter().copied().chain([d1, d2]).collect();
                let (lg, hid2) = m.forward_spec_k(&toks, &mut cache, m.cfg.n_layer, 3);
                let v = pollster::block_on(lg.to_vec()); // rows: 0=→d1, 1=→d2, 2=→pend
                let t1 = self.select_token(&v[0..n_vocab], &guide, temperature, &mut rng);
                let Some(t1) = t1.filter(|t| !self.eos.contains(t)) else { cache = snap; break; };
                commit!(t1);
                if t1 != d1 || gen.len() >= max_tokens {
                    // Reject both (or stop): discard the forward, re-feed t1 next iter.
                    cache = snap;
                    unfed.push(t1);
                    ptoks = vec![t1];
                    phid = hid2.narrow(0, k - 1, 1).contiguous();
                    continue;
                }
                // d1 accepted — check the 2nd draft against the true token after d1.
                let t2 = self.select_token(&v[n_vocab..2 * n_vocab], &guide, temperature, &mut rng);
                let Some(t2) = t2.filter(|t| !self.eos.contains(t)) else { cache = snap; break; };
                commit!(t2);
                if t2 != d2 || gen.len() >= max_tokens {
                    // Accept d1 only: d2's cache entry is wrong → discard forward, re-feed [d1, t2].
                    cache = snap;
                    unfed = { let mut u = unfed.clone(); u.push(d1); u.push(t2); u };
                    ptoks = vec![d1, t2];
                    phid = hid2.narrow(0, k - 1, 2).contiguous();
                    continue;
                }
                // Accept both: the cache validly holds [unfed…, d1, d2]; emit pend from row 2.
                fed.extend_from_slice(&toks);
                let pend = self.select_token(&v[2 * n_vocab..3 * n_vocab], &guide, temperature, &mut rng);
                let Some(pend) = pend.filter(|t| !self.eos.contains(t)) else { break };
                commit!(pend);
                ptoks = vec![d1, d2, pend];
                phid = hid2.narrow(0, k - 1, 3).contiguous();
                unfed = vec![pend];
                continue;
            }
            // 1. Draft: feed pending pairs (keeps the draft cache aligned), propose one token (argmax
            //    — the draft is a guess; only agreement with the true sampler matters).
            let dlog = m.mtp_forward(&ptoks, &phid, &mut mc);
            let dv = pollster::block_on(dlog.to_vec());
            let d = argmax(&dv[dv.len() - n_vocab..]);
            // 2. Verify: one forward over [unfed…, draft], snapshot first for O(1) rollback.
            let snap = cache.snapshot();
            let k = unfed.len();
            let toks: Vec<u32> = unfed.iter().copied().chain([d]).collect();
            let (lg, hid2) = m.forward_spec(&toks, &mut cache, m.cfg.n_layer);
            // forward_spec heads only the last two positions: row 0 = last unfed (truth), row 1 = draft.
            let v = pollster::block_on(lg.to_vec());
            let truth = self.select_token(&v[0..n_vocab], &guide, temperature, &mut rng);
            let Some(truth) = truth.filter(|t| !self.eos.contains(t)) else {
                cache = snap; // this forward's entries include the unverified draft — discard
                break;
            };
            commit!(truth);
            if truth == d && gen.len() < max_tokens {
                fed.extend_from_slice(&toks); // everything this forward fed is now known-valid
                // Accepted: the draft's own logits row is valid too — take the next token from it.
                let pend = self.select_token(&v[n_vocab..2 * n_vocab], &guide, temperature, &mut rng);
                let Some(pend) = pend.filter(|t| !self.eos.contains(t)) else { break };
                commit!(pend);
                ptoks = vec![d, pend];
                phid = hid2.narrow(0, k - 1, 2).contiguous(); // hiddens at d's and pend's predecessors
                unfed = vec![pend];
            } else if truth == d {
                fed.extend_from_slice(&toks);
                break; // accepted, but max_tokens lands exactly here
            } else {
                // Rejected: the cache holds a wrong entry at the draft's position — roll back.
                // Nothing is wasted: the true token was still learned from this forward.
                cache = snap;
                unfed.push(truth);
                ptoks = vec![truth];
                phid = hid2.narrow(0, k - 1, 1).contiguous(); // hidden at truth's predecessor
            }
        }
        save_slot!();
        (emitted, prompt.len(), gen.len())
    }
}

fn now_unix() -> u64 { 1_700_000_000 } // static stamp (no wall clock needed for the API contract)

/// Top-p (nucleus) sampling from `row` at `temperature`, using a xorshift RNG. Small models loop badly
/// at temperature 0; this makes them usable while staying reproducible (the RNG is fixed-seeded).
fn sample_top_p(row: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> u32 {
    let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = row.iter().map(|&l| ((l - maxl) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    // nucleus: smallest set whose probability mass ≥ top_p
    let (mut cum, mut cut) = (0.0f32, idx.len());
    for (k, &i) in idx.iter().enumerate() { cum += probs[i] / sum; if cum >= top_p { cut = k + 1; break; } }
    // xorshift64 → r in [0, nucleus mass)
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    let r = (*rng >> 11) as f32 / (1u64 << 53) as f32 * cum;
    let (mut acc, mut pick) = (0.0f32, idx[0]);
    for &i in &idx[..cut] { acc += probs[i] / sum; if acc >= r { pick = i; break; } }
    pick as u32
}

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
    if let Some(i) = args.iter().position(|a| a == "--tokenize") {
        // Debug: print the prompt token ids (BOS + first-fragment prefix), to diff against llama-tokenize.
        let text = args.get(i + 1).cloned().unwrap_or_default();
        let mut ids = Vec::new();
        if eng.add_bos { if let Some(b) = eng.bos_id { ids.push(b); } }
        ids.extend(eng.enc(&text, true));
        eprintln!("TOKENS {}: {:?}", ids.len(), ids);
        return;
    }
    if args.iter().any(|a| a == "--once") {
        // Smoke test: one chat turn straight through the pipeline, no HTTP.
        let msgs = vec![json!({"role": "user", "content": "Hi"})];
        let (t, p, g) = eng.generate(&eng.chat_ids(&msgs), 16, 0.0, None, |d| eprint!("{d}"));
        eprintln!("\nferric-serve: --once ok ({p} prompt + {g} gen tokens): {t:?}");
        return;
    }
    let mcps = std::cell::RefCell::new(mcps);
    eprintln!("ferric-serve: {} ({} layers, vocab {}) on {:?}{} — http://127.0.0.1:{port}/v1",
        name, eng.model.n_layer(), eng.model.n_vocab(), eng.ctx.backend,
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
        ("POST", "/v1/embeddings") => embeddings(eng, &mut stream, &body),
        _ => write_json(&mut stream, 404, &json!({"error": {"message": "not found", "type": "invalid_request_error"}})),
    }
}

/// OpenAI-compatible `/v1/embeddings`: `input` is a string or array of strings → L2-normalized vectors.
/// Runs on an embedding model (e.g. Qwen3-Embedding, a Qwen3-arch model with no lm_head).
fn embeddings(eng: &Engine, stream: &mut TcpStream, body: &[u8]) {
    let bad = |stream: &mut TcpStream, m: &str| write_json(stream, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}}));
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad(stream, &format!("bad json: {e}")) };
    let inputs: Vec<String> = match &req["input"] {
        Value::String(s) => vec![s.clone()],
        // Error (don't silently drop) on a non-string element — dropping would misalign every `index`.
        Value::Array(a) => {
            let mut v = Vec::with_capacity(a.len());
            for x in a { match x.as_str() { Some(s) => v.push(s.to_string()), None => return bad(stream, "`input` array must contain only strings") } }
            v
        }
        _ => return bad(stream, "`input` must be a string or array of strings"),
    };
    let mut total = 0usize;
    let data: Vec<Value> = inputs.iter().enumerate().map(|(i, text)| {
        total += eng.enc(text, true).len();
        json!({"object": "embedding", "index": i, "embedding": eng.embed(text)})
    }).collect();
    write_json(stream, 200, &json!({
        "object": "list", "data": data, "model": eng.name,
        "usage": {"prompt_tokens": total, "total_tokens": total}
    }));
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
    let temperature = req["temperature"].as_f64().unwrap_or(0.0) as f32;
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
            let (text, p, g) = eng.generate(&prompt, max_tokens, temperature, None, |_| {});
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
        eng.generate(&prompt, max_tokens, temperature, guide, |delta| {
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": Value::Null}]}));
        });
        send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}));
        let _ = stream.write_all(b"data: [DONE]\n\n");
    } else {
        let (text, ptok, gtok) = eng.generate(&prompt, max_tokens, temperature, guide, |_| {});
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
    let temperature = req["temperature"].as_f64().unwrap_or(0.0) as f32;
    let mut ids = Vec::new();
    if eng.add_bos { if let Some(b) = eng.bos_id { ids.push(b); } }
    ids.extend(eng.enc(prompt_text, true));
    let (text, ptok, gtok) = eng.generate(&ids, max_tokens, temperature, None, |_| {});
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
