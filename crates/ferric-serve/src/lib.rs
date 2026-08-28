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
//! **Continuous batching.** Concurrent requests share one decode step: `forward_batch` advances every
//! in-flight sequence by one token in a single forward, so the weight set is read once for the whole
//! batch instead of once per request. Arrivals join mid-flight and a finished sequence's slot is
//! refilled on the very next step (`ferric_llama::sched::Scheduler` is the policy; `batch.rs` is the
//! transport). `--max-batch N` (default 8), `--no-batch` to force serial. A model whose runtime has no
//! solo-equivalent batched forward — `Model::supports_batching` — falls back to serial automatically,
//! as do guided-decoding and tool-calling requests, which keep the untouched single-request path.
mod mcp;
mod batch;
mod specgate;
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::{qwen3, qwen35};
use ferric_llama::qwen3::Qwen3;
use ferric_llama::qwen35::Qwen35;
use ferric_tensor::Tensor;

/// The loaded model — a dense Qwen3/Llama/Gemma/Phi, or the Qwen3.5/3.6 **GDN-hybrid** (gated delta net
/// + periodic full attention). Both expose a `forward_cached` returning logits, so the generate loop and
/// guided decoding are architecture-agnostic; only the KV/recurrent cache type differs.
pub(crate) enum Model { Dense(Qwen3), Hybrid(Qwen35), Lfm2(ferric_llama::lfm2::Lfm2), Gemma4(ferric_llama::gemma4::Gemma4), DeepSeek2(ferric_llama::deepseek2::DeepSeek2), NemotronH(ferric_llama::nemotron_h::NemotronH) }
pub(crate) enum ModelCache { Dense(qwen3::Cache), Hybrid(qwen35::Cache), Lfm2(ferric_llama::lfm2::Cache), Gemma4(ferric_llama::gemma4::Cache), DeepSeek2(ferric_llama::deepseek2::Cache), NemotronH(ferric_llama::nemotron_h::Cache) }
impl Model {
    fn n_vocab(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_vocab, Model::Hybrid(m) => m.cfg.n_vocab, Model::Lfm2(m) => m.cfg.n_vocab, Model::Gemma4(m) => m.cfg.n_vocab, Model::DeepSeek2(m) => m.cfg.n_vocab, Model::NemotronH(m) => m.cfg.n_vocab } }
    fn n_layer(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_layer, Model::Hybrid(m) => m.cfg.n_layer, Model::Lfm2(m) => m.cfg.n_layer, Model::Gemma4(m) => m.cfg.n_layer, Model::DeepSeek2(m) => m.cfg.n_layer, Model::NemotronH(m) => m.cfg.n_layer } }
    fn n_embd(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_embd, Model::Hybrid(m) => m.cfg.n_embd, Model::Lfm2(m) => m.cfg.d, Model::Gemma4(m) => m.cfg.d, Model::DeepSeek2(m) => m.cfg.d, Model::NemotronH(m) => m.cfg.d } }
    /// **Which runtimes `Engine::load` can build a generative `Model` from.**
    ///
    /// The dispatch consults this before matching, so "the registry says supported" and "the server
    /// can load it" are one list instead of two. They were two, and they drifted: `nemotron_h` sat at
    /// `Status::Verified` — its note claiming it reproduces the reference and generates " Paris." —
    /// while the dispatch arm still panicked "its forward pass is not written yet", stale since the
    /// port landed. Nothing failed, because the test guarding this returned `true` from all eight of
    /// its match arms.
    ///
    /// `Err` is for runtimes that are not generative AT ALL, not for ones nobody has wired yet — a
    /// missing wiring must be a compile error here, which the exhaustive match makes it.
    pub(crate) fn dispatchable(r: ferric_llama::arch::Runtime) -> Result<(), &'static str> {
        use ferric_llama::arch::Runtime as R;
        match r {
            R::Dense | R::Hybrid | R::Lfm2 | R::Gemma4 | R::DeepSeek2 | R::NemotronH => Ok(()),
            R::Bert => Err("a BERT encoder: no KV cache and no LM head, so it cannot serve chat or \
                            completions. Point FERRIC_RERANK_MODEL at it instead"),
            R::Cosmos => Err("loads from safetensors, not GGUF; ferric-serve takes a GGUF"),
            R::Parakeet => Err("a speech recogniser: it takes a WAVEFORM and returns text, and has \
                                no KV cache, no LM head and no token input. Use the parakeet \
                                examples; it cannot serve chat or completions"),
        }
    }

    fn new_cache(&self) -> ModelCache {
        match self {
            Model::Dense(m) => ModelCache::Dense(qwen3::Cache::new(&m.cfg)),
            Model::Hybrid(m) => ModelCache::Hybrid(qwen35::Cache::new(&m.cfg)),
            Model::Lfm2(m) => ModelCache::Lfm2(ferric_llama::lfm2::Cache::new(&m.cfg)),
            Model::Gemma4(m) => ModelCache::Gemma4(ferric_llama::gemma4::Cache::new(&m.cfg)),
            Model::DeepSeek2(m) => ModelCache::DeepSeek2(ferric_llama::deepseek2::Cache::new(&m.cfg)),
            // The only runtime whose cache needs the GPU context; it carries its own, hence
            // `new_cache` on the model rather than `Cache::new` here.
            Model::NemotronH(m) => ModelCache::NemotronH(m.new_cache()),
        }
    }
    fn forward_cached(&self, tokens: &[u32], cache: &mut ModelCache) -> Tensor {
        match (self, cache) {
            (Model::Dense(m), ModelCache::Dense(c)) => m.forward_cached(tokens, c),
            (Model::Hybrid(m), ModelCache::Hybrid(c)) => m.forward_cached(tokens, c, m.cfg.n_layer),
            (Model::Lfm2(m), ModelCache::Lfm2(c)) => m.forward(tokens, c),
            (Model::Gemma4(m), ModelCache::Gemma4(c)) => m.forward(tokens, c),
            (Model::DeepSeek2(m), ModelCache::DeepSeek2(c)) => m.forward(tokens, c),
            // Returns Result because a token id outside the embedding table is a real, reachable
            // input error rather than a bug; the server has no way to recover, so it names it.
            (Model::NemotronH(m), ModelCache::NemotronH(c)) =>
                m.forward_cached(tokens, c).unwrap_or_else(|e| panic!("nemotron_h forward: {e}")),
            _ => unreachable!("model/cache kind mismatch"),
        }
    }
    /// Whether this runtime has a `forward_batch` that is **proven** solo-equivalent — i.e. whether a
    /// batching scheduler is allowed to put it in a batch at all.
    ///
    /// Each runtime carries a **different** per-sequence state — gated-delta-net recurrence (qwen35),
    /// short-conv rolling window (lfm2), shared KV where later blocks read an earlier block's cache
    /// (gemma4), MLA latent KV with asymmetric head widths (deepseek2) — so each needs its own batched
    /// forward *and* its own proof that batching did not cross sequences. `true` here means that proof
    /// exists and is checked in, not that the method compiles.
    ///
    /// ⚠ **This is a claim about the runtime, not about the server.** `ferric-serve` still has no
    /// batched dispatch path: the batched decode loop lives in
    /// `ferric-llama/examples/continuous_batching.rs` and the TCP wiring is unbuilt. Nothing reads this
    /// yet. It is written ahead of that wiring so the wiring cannot be built without confronting which
    /// runtimes are safe to batch, and so `false` here forces **serial fallback** rather than silent
    /// mis-batching. Mis-batching does not error: a batched path that leaked across sequences still
    /// emits fluent text, which is why every port re-runs each sequence solo and compares token ids.
    ///
    /// Written as an exhaustive `match` on purpose — adding a runtime forces a decision here instead
    /// of inheriting a default.
    pub(crate) fn supports_batching(&self) -> bool {
        match self {
            // Dense supports it only when the model does not need scaled or NORM-interleaved rope —
            // rope_at implements neither, so batching such a model silently diverges from solo decode.
            Model::Dense(m) => m.batching_supported(),
            // Routed through the runtime's own predicate for the same reason as Dense: the runtime,
            // not the server, knows which of its checkpoints have a batched path that covers every
            // branch the solo path takes.
            Model::Hybrid(m) => m.batching_supported(),
            // Ported AND adversarially verified token-identical to solo decode: lfm2 at n=2/3/4/8,
            // gemma4 at n=2/3/4 including prompts past the 512 sliding window, deepseek2 at n=2/4 with
            // MLA, DeepSeekMoE routing and YaRN. None of these three has a checkpoint-dependent branch
            // in its batched path, so there is no predicate for them to consult.
            Model::Lfm2(_) | Model::Gemma4(_) | Model::DeepSeek2(_) => true,
            // No `forward_batch` exists for it at all. Mamba-2 carries a per-sequence recurrent
            // state and a conv window, so a batched path is a real port plus its own proof — not a
            // loop — and until both exist this must stay false.
            Model::NemotronH(_) => false,
        }
    }

    /// Batched decode for N sequences, dispatched on the runtime. See `batch::ServeModel::decode`,
    /// which is the only caller; the kind check is `unreachable!` because `ModelCache` is only ever
    /// produced by `new_cache` on this same `Model`.
    fn forward_batch(&self, tokens: &[u32], caches: &mut [&mut ModelCache]) -> Tensor {
        macro_rules! b {
            ($m:expr_2021, $variant:path) => {{
                let mut cs: Vec<_> = caches.iter_mut()
                    .map(|c| match &mut **c { $variant(x) => x, _ => unreachable!("model/cache kind mismatch") })
                    .collect();
                $m.forward_batch(tokens, &mut cs)
            }};
        }
        match self {
            Model::Dense(m) => b!(m, ModelCache::Dense),
            Model::Hybrid(m) => b!(m, ModelCache::Hybrid),
            Model::Lfm2(m) => b!(m, ModelCache::Lfm2),
            Model::Gemma4(m) => b!(m, ModelCache::Gemma4),
            Model::DeepSeek2(m) => b!(m, ModelCache::DeepSeek2),
            // Unreachable via the scheduler, which consults `supports_batching` first; this arm is
            // what makes adding a runtime a compile error rather than a silent wrong answer.
            Model::NemotronH(_) => unreachable!("nemotron_h has no batched path; supports_batching is false"),
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
            Model::Lfm2(m) => {
                let mut c = ferric_llama::lfm2::Cache::new(&m.cfg);
                m.forward_hidden_cached(ids, &mut c)
            }
            Model::Gemma4(m) => {
                let mut c = ferric_llama::gemma4::Cache::new(&m.cfg);
                m.forward_hidden_cached(ids, &mut c)
            }
            Model::DeepSeek2(m) => {
                let mut c = ferric_llama::deepseek2::Cache::new(&m.cfg);
                m.forward_hidden_cached(ids, &mut c)
            }
            // `nemotron_h` exposes no pre-head hidden state: `forward_cached` returns logits and
            // `forward_traced` returns per-op captures, neither of which is "all blocks then the
            // final norm". Refusing names what is missing; returning the logits would silently make
            // every embedding from this model wrong while looking like a vector.
            Model::NemotronH(_) => panic!(
                "nemotron_h cannot produce embeddings: no pre-head hidden-state path exists. It \
                 serves chat and completions; point the embedding endpoint at another model"),
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

pub(crate) struct Engine {
    ctx: Arc<Context>,
    model: Model,
    /// A cross-encoder reranker, loaded SEPARATELY from `FERRIC_RERANK_MODEL`.
    ///
    /// It is a second model, not a mode of the first: rerankers are encoders with a classification
    /// head, they have no KV cache and no LM head, and nothing in `Model` fits them. llama-server
    /// takes the same shape with its own `--reranking` flag. Absent by default, so `/v1/rerank`
    /// answers 400 with what to set rather than pretending the endpoint does not exist.
    reranker: Option<ferric_llama::bert::Reranker>,
    /// Whether speculative decoding pays for itself on this request, learned from observed draft
    /// acceptance. Behind a mutex because the serving path shares one `Engine` across connections and
    /// this is the only mutable state on it — a `Mutex` rather than an atomic because the estimator is
    /// a small map, and rather than a per-request copy because the whole point is to accumulate.
    spec_gate: std::sync::Mutex<specgate::SpecGate>,
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
        // Loaded before the main model so a bad reranker path fails immediately rather than after a
        // multi-GB load, and so the panic names which of the two files was wrong.
        let reranker = std::env::var("FERRIC_RERANK_MODEL").ok().map(|rp| {
            let rg = GgufFile::open(&rp).unwrap_or_else(|e| panic!("open FERRIC_RERANK_MODEL {rp}: {e:?}"));
            ferric_llama::bert::Reranker::load(&ctx, &rg)
                .unwrap_or_else(|e| panic!("load FERRIC_RERANK_MODEL {rp}: {e}"))
        });
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
        // `tokenizer.ggml.pre` names the pre-tokenizer regex and is a SEPARATE question from
        // `tokenizer.ggml.model`. Every Qwen checkpoint declares model=gpt2 and pre=qwen2; reading
        // only the first gave the whole family GPT-2's rule, under which `-Reyes` splits into `-` +
        // `Reyes` and the merge `- Re` becomes unreachable. Diffed against llama.cpp: it emits token
        // 67960 where this emitted 12 then 693. The server serves the same checkpoints as the browser,
        // so it needs the same fix or generation here diverges from generation there.
        let bpe = Bpe::new_with_pre(vocab.clone(), &merges, ferric_tokenizer::Pre::from_gguf(
            match g.metadata.get("tokenizer.ggml.pre") { Some(Meta::Str(p)) => Some(p.as_str()), _ => None }));
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
        // Dispatch through the architecture REGISTRY, which refuses anything this runtime has not been
        // taught. The previous form was `if starts_with("qwen35") … else { Dense }`, and that `else`
        // was a catch-all: a gemma4 / deepseek2 / glm4 / minimax / hunyuan checkpoint loaded as a dense
        // Qwen3, took whichever metadata keys happened to share names, defaulted the rest, and emitted
        // fluent, confident, WRONG text with no error anywhere. Refusing is the feature.
        let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
        let entry = ferric_llama::arch::resolve(&arch).unwrap_or_else(|e| panic!("{e}"));
        if let Err(why) = Model::dispatchable(entry.runtime) { panic!("{path} ({arch}) is {why}"); }
        let model = match entry.runtime {
            ferric_llama::arch::Runtime::Hybrid =>
                Model::Hybrid(Qwen35::load(&ctx, &g).unwrap_or_else(|e| panic!("load hybrid model: {e}"))),
            ferric_llama::arch::Runtime::Dense =>
                Model::Dense(Qwen3::load(&ctx, &g).unwrap_or_else(|e| panic!("load model: {e}"))),
            ferric_llama::arch::Runtime::Lfm2 =>
                Model::Lfm2(ferric_llama::lfm2::Lfm2::load(&ctx, &g).unwrap_or_else(|e| panic!("load lfm2: {e}"))),
            ferric_llama::arch::Runtime::Gemma4 =>
                Model::Gemma4(ferric_llama::gemma4::Gemma4::load(&ctx, &g).unwrap_or_else(|e| panic!("load gemma4: {e}"))),
            ferric_llama::arch::Runtime::DeepSeek2 =>
                Model::DeepSeek2(ferric_llama::deepseek2::DeepSeek2::load(&ctx, &g).unwrap_or_else(|e| panic!("load deepseek2: {e}"))),
            ferric_llama::arch::Runtime::Cosmos =>
                unreachable!("Cosmos is refused by Model::dispatchable before this match"),
            ferric_llama::arch::Runtime::Parakeet =>
                unreachable!("Parakeet is refused by Model::dispatchable before this match"),
            // ⚠ This arm PANICKED with "its forward pass is not written yet" long after the forward
            // pass landed and the registry promoted the row to Status::Verified. The registry and the
            // dispatch are two lists that must agree and nothing made them; the note at arch.rs
            // claimed it generates " Paris." while this line refused to load it at all.
            ferric_llama::arch::Runtime::NemotronH =>
                Model::NemotronH(ferric_llama::nemotron_h::NemotronH::load(&ctx, &g)
                    .unwrap_or_else(|e| panic!("load nemotron_h: {e}"))),
            // Both refusals are already delivered by the `dispatchable` check above, so these arms
            // cannot be reached. They stay because the exhaustive match is what turns "someone added
            // a Runtime" into a compile error rather than a fallthrough.
            ferric_llama::arch::Runtime::Bert =>
                unreachable!("Bert is refused by Model::dispatchable before this match"),
        };
        eprintln!("arch {arch:?} -> {} runtime ({}) — {}",
                  entry.runtime.label(), entry.status.label(), entry.note);
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
        // TWO spellings, and the one this looked for is not the one modern GGUFs use. Converters
        // write `tokenizer.chat_template`; `tokenizer.ggml.chat_template` was the older form. Checked
        // across the checkpoints on this machine: gemma-4-E2B and qwen1.5b both carry ONLY the former,
        // so this lookup found nothing every time and `template` was always empty — the vocab-based
        // family detection below carried the whole feature and its "robust even when the GGUF omits
        // the key" comment described the permanent state rather than a fallback.
        let template = match g.metadata.get("tokenizer.chat_template")
            .or_else(|| g.metadata.get("tokenizer.ggml.chat_template")) {
            Some(Meta::Str(s)) => s.clone(), _ => String::new(),
        };
        // The break-even is `E_draft / E_main`, estimated from shape: one MTP block against the main
        // model's `n_layer`. A structural estimate, not a measurement — see `specgate`.
        let spec_gate = std::sync::Mutex::new(specgate::SpecGate::new(model.n_layer()));
        Engine { ctx, model, bpe, spm, add_space_prefix, tokens, u2b, im_start, im_end, bos_id, add_bos, eos_id, add_eos, eos, name, token_bytes, specials, template, prefix: std::cell::RefCell::new(None), spec_gate, reranker }
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

    /// Whether **this server** may put this engine's requests in a shared decode batch.
    ///
    /// Two conditions, and the second is not the runtime's business:
    ///
    /// 1. `Model::supports_batching()` — the runtime's own claim that its `forward_batch` is proven
    ///    solo-equivalent for the loaded checkpoint. This is that gate's only consumer.
    /// 2. The serial path for this engine must be the plain decode loop. A hybrid shipping its own
    ///    MTP draft block takes `generate_spec` instead, and the batched forward does no drafting —
    ///    so batching it would be comparing against a *different* reference, and "a batched response
    ///    equals a serial one" would stop being a checkable statement. Refuse rather than blur it.
    ///    (`FERRIC_NOSPEC=1` disables drafting, and then this model batches.)
    ///
    /// `FERRIC_NOBATCH=1` forces serial for everything — the A/B escape hatch.
    pub(crate) fn batchable(&self) -> bool {
        if std::env::var("FERRIC_NOBATCH").is_ok() { return false; }
        if !self.model.supports_batching() { return false; }
        if let Model::Hybrid(m) = &self.model {
            if m.mtp.is_some() && std::env::var("FERRIC_NOSPEC").is_err() { return false; }
        }
        true
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
            //
            // Beyond that, `SpecGate` decides. Self-drafting is not free — every step pays the draft
            // forward AND the main forward — so below an acceptance rate of `E_draft / E_main` it
            // spends MORE joules per token than the plain loop. The gate holds a learned acceptance
            // estimate per prompt-length bucket and declines when the request sits below break-even.
            // With no evidence it speculates, which is exactly today's behaviour.
            if m.mtp.is_some() && std::env::var("FERRIC_NOSPEC").is_err() {
                let speculate = {
                    let g = self.spec_gate.lock().expect("spec gate poisoned");
                    let take = g.should_speculate(prompt.len());
                    if std::env::var("FERRIC_SPEC_TRACE").is_ok() {
                        eprintln!("spec gate: prompt {} tok · p(accept)~{:.3} vs break-even {:.3} -> {}",
                                  prompt.len(), g.expected_acceptance(prompt.len()), g.break_even(),
                                  if take { "speculate" } else { "plain" });
                    }
                    take
                };
                if speculate {
                    let (text, p, n, drafted, accepted) =
                        self.generate_spec(m, prompt, max_tokens, temperature, guide, on_delta);
                    // Only ATTEMPTED drafts are recorded. A step that did not draft has no outcome, and
                    // inventing one would let the gate confirm its own decisions.
                    if drafted > 0 {
                        let mut g = self.spec_gate.lock().expect("spec gate poisoned");
                        for i in 0..drafted { g.observe(prompt.len(), i < accepted); }
                    }
                    return (text, p, n);
                }
            }
        }
        let mut cache = self.model.new_cache();
        let n_vocab = self.model.n_vocab();
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D; // deterministic seed → reproducible sampling
        let mut r#gen: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        for step in 0..max_tokens {
            let input: Vec<u32> = if step == 0 { prompt.to_vec() } else { vec![*r#gen.last().unwrap()] };
            let logits = self.model.forward_cached(&input, &mut cache);
            let v = pollster::block_on(logits.to_vec());
            let Some(next) = self.select_token(&v[v.len() - n_vocab..], &guide, temperature, &mut rng) else { break };
            if self.eos.contains(&next) { break; }
            if let (Some(g), Some(b)) = (guide.as_mut(), self.token_bytes[next as usize].as_ref()) { for &c in b { g.step(c); } }
            r#gen.push(next);
            // Re-detok the whole generation and emit only the new suffix (handles multi-byte UTF-8).
            let full = self.detok(&r#gen);
            if full.len() > emitted.len() && full.is_char_boundary(emitted.len()) {
                let delta = full[emitted.len()..].to_string();
                on_delta(&delta);
                emitted = full;
            }
        }
        (emitted, prompt.len(), r#gen.len())
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
    fn generate_spec(&self, m: &Qwen35, prompt: &[u32], max_tokens: usize, temperature: f32, mut guide: Option<ferric_agent::guide::Guide>, mut on_delta: impl FnMut(&str)) -> (String, usize, usize, u32, u32) {
        // Draft accounting, returned so the caller can feed `SpecGate`. The doc above this function
        // asserts "~80% acceptance"; nothing measured it until now, and a gate that decides on an
        // assumed rate is a heuristic wearing a derivation's clothes.
        let (mut drafted, mut accepted) = (0u32, 0u32);
        let n_vocab = self.model.n_vocab();
        let argmax = |r: &[f32]| (0..n_vocab).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap() as u32;
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut r#gen: Vec<u32> = Vec::new();
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
            ($tok:expr_2021) => {{
                if let (Some(g), Some(b)) = (guide.as_mut(), self.token_bytes[$tok as usize].as_ref()) { for &c in b { g.step(c); } }
                r#gen.push($tok);
                let full = self.detok(&r#gen);
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
        let Some(pend0) = first.filter(|t| !self.eos.contains(t)) else { save_slot!(); return (emitted, prompt.len(), 0, drafted, accepted) };
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
        while r#gen.len() < max_tokens {
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
                // Two drafts were proposed this step; each is one observation for the gate. Counted
                // where the decision is ALREADY made rather than re-derived, so the tally cannot drift
                // from the branch it describes.
                drafted += 2;
                if t1 != d1 || r#gen.len() >= max_tokens {
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
                accepted += 1;   // d1 matched
                if t2 != d2 || r#gen.len() >= max_tokens {
                    // Accept d1 only: d2's cache entry is wrong → discard forward, re-feed [d1, t2].
                    cache = snap;
                    unfed = { let mut u = unfed.clone(); u.push(d1); u.push(t2); u };
                    ptoks = vec![d1, t2];
                    phid = hid2.narrow(0, k - 1, 2).contiguous();
                    continue;
                }
                accepted += 1;   // d2 matched too
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
            if truth == d && r#gen.len() < max_tokens {
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
        (emitted, prompt.len(), r#gen.len(), drafted, accepted)
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

/// CLI entry point. Lives in the library so the binary is a two-line shim and everything the
/// server does stays reachable from tests — see the crate docs.
pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).unwrap_or_else(|| { eprintln!("usage: ferric-serve <model.gguf> [--port N] [--name S]"); std::process::exit(1); });
    let mut port = 8080u16;
    let mut name = "ferric".to_string();
    let mut mcp_cmds: Vec<(String, String)> = Vec::new();
    // How many sequences may share one decode step. 8 is a starting point, not a measured optimum:
    // occupancy is what decides the payoff and only the deployment knows its arrival rate.
    let mut max_batch = std::env::var("FERRIC_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(8usize);
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => { port = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(port); i += 2; }
            "--name" => { name = args.get(i + 1).cloned().unwrap_or(name); i += 2; }
            "--max-batch" => { max_batch = args.get(i + 1).and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(max_batch); i += 2; }
            // FIXME: Audit that the environment access only happens in single-threaded code.
            "--no-batch" => { unsafe { std::env::set_var("FERRIC_NOBATCH", "1") }; i += 1; }
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
    let any_mcp_tools = !mcps.borrow().openai_tools().is_empty();
    let batching = eng.batchable();
    eprintln!("ferric-serve: {} ({} layers, vocab {}) on {:?}{} — http://127.0.0.1:{port}/v1",
        name, eng.model.n_layer(), eng.model.n_vocab(), eng.ctx.backend,
        if mcps.borrow().0.is_empty() { String::new() } else { format!(" · {} MCP tools", mcps.borrow().openai_tools().len()) });
    eprintln!("ferric-serve: continuous batching {}",
        if batching { format!("ON, max_batch {max_batch}") } else { "OFF (serial) — this model has no solo-equivalent batched decode, or it was disabled".to_string() });
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| panic!("bind :{port}: {e}"));
    // The batch loop owns the engine on this thread; anything it declines (guided decoding, the tool
    // loop, embeddings, unknown paths) goes to the untouched serial handler below.
    batch::serve_loop(eng, listener, batch::ServeOpts { max_batch, any_mcp_tools },
        |eng, method, path, body, s| {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(eng, &mcps, method, path, body, s)));
            match r {
                Ok(handled) => handled,
                Err(_) => { eprintln!("ferric-serve: handler panicked (recovered)"); true }
            }
        });
}

/// The serial handler: every endpoint the batch loop does not own. Returns whether it recognised the
/// request (`false` → the caller writes a 404).
fn handle(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, method: &str, path: &str, body: &[u8], stream: &mut TcpStream) -> bool {
    match (method, path) {
        ("GET", "/health") => write_json(stream, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") => write_json(stream, 200, &json!({
            "object": "list",
            "data": [{"id": eng.name, "object": "model", "created": now_unix(), "owned_by": "ferric"}]
        })),
        ("POST", "/v1/chat/completions") => chat(eng, mcps, stream, body),
        ("POST", "/v1/completions") => completions(eng, stream, body),
        ("POST", "/v1/embeddings") => embeddings(eng, stream, body),
        ("POST", "/v1/rerank") | ("POST", "/rerank") => rerank(eng, stream, body),
        // Unrecognised: let the caller answer, so there is exactly one 404 writer.
        _ => return false,
    }
    true
}

/// OpenAI-compatible `/v1/embeddings`: `input` is a string or array of strings → L2-normalized vectors.
/// Runs on an embedding model (e.g. Qwen3-Embedding, a Qwen3-arch model with no lm_head).
/// **POST /v1/rerank** — cross-encoder relevance scoring over a shortlist.
///
/// Body: `{"query": str, "documents": [str], "top_n"?: int, "return_documents"?: bool}`.
/// Returns `{"results": [{"index", "relevance_score", "document"?}]}` best-first, the shape Cohere
/// defined and Jina, Voyage and llama-server all followed.
///
/// Scores are RAW logits, matching llama.cpp: ordering is invariant to any monotone squash, and a
/// sigmoid here would make two implementations' numbers incomparable for no gain. Measured against
/// llama-server on bge-reranker-v2-m3: 6.585 vs 6.570 relevant, -8.366 vs -8.361 irrelevant.
fn rerank(eng: &Engine, stream: &mut TcpStream, body: &[u8]) {
    let bad = |stream: &mut TcpStream, m: &str| write_json(stream, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}}));
    let Some(rr) = eng.reranker.as_ref() else {
        return bad(stream, "no reranker loaded: set FERRIC_RERANK_MODEL to a cross-encoder GGUF \
                            (e.g. bge-reranker-v2-m3). A reranker is a SECOND model — an encoder with \
                            a classification head — not a mode of the chat model");
    };
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad(stream, &format!("bad json: {e}")) };
    let Some(query) = req["query"].as_str() else { return bad(stream, "`query` must be a string") };
    // Error rather than skip a non-string: dropping one would misalign every `index` returned, which
    // is the field the caller uses to map results back to its own list.
    let docs: Vec<String> = match &req["documents"] {
        Value::Array(a) => {
            let mut v = Vec::with_capacity(a.len());
            for x in a { match x.as_str() { Some(s) => v.push(s.to_string()), None => return bad(stream, "`documents` must contain only strings") } }
            v
        }
        _ => return bad(stream, "`documents` must be an array of strings"),
    };
    if docs.is_empty() { return bad(stream, "`documents` must not be empty"); }
    let ranked = match pollster::block_on(rr.rank(query, &docs)) {
        Ok(r) => r, Err(e) => return bad(stream, &format!("rerank failed: {e}")),
    };
    let want_docs = req["return_documents"].as_bool().unwrap_or(false);
    let top_n = req["top_n"].as_u64().map(|n| n as usize).unwrap_or(ranked.len()).min(ranked.len());
    let results: Vec<Value> = ranked[..top_n].iter().map(|(i, s)| {
        let mut o = json!({"index": i, "relevance_score": s});
        if want_docs { o["document"] = json!({"text": docs[*i]}); }
        o
    }).collect();
    write_json(stream, 200, &json!({"object": "list", "model": eng.name, "results": results}));
}

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

/// Compile-time check that the engine crosses a thread boundary.
///
/// The continuous-batching wiring's whole shape depends on this. If `Engine` is `Send`, the accept
/// loop can own it behind a mutex; if it is not, the model must stay pinned to one thread and
/// requests must arrive by channel. wgpu's `Device`/`Queue` are `Send + Sync` on native but NOT on
/// wasm32, so this also stops a native-only design from silently breaking the browser build.
///
/// Expressed as a const rather than a `#[test]` because **this crate is binary-only and has no lib
/// target** — none of the server's logic is unit-testable today, which is worth fixing before it
/// grows a scheduler.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<Engine>;
};

#[cfg(test)]
mod tests {
    /// The first unit test this crate has ever been able to have.
    ///
    /// `byte_decoder` inverts GPT-2's byte↔printable-unicode map, and every token this server emits
    /// passes through it. A wrong entry does not error — it silently corrupts one byte value in all
    /// output, which is exactly the failure class that survives "curl it and look".
    #[test]
    fn the_byte_decoder_is_a_bijection_over_all_256_bytes() {
        let m = super::byte_decoder();
        assert_eq!(m.len(), 256, "every byte value must have exactly one printable alias");
        let mut seen = [false; 256];
        for (_, &b) in m.iter() {
            assert!(!seen[b as usize], "byte {b} is produced by two different characters");
            seen[b as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "some byte value has no alias at all");
        // The printable ASCII range must map to itself, or ordinary text decodes to nonsense.
        for b in 0x21u8..=0x7e { assert_eq!(m[&(b as char)], b, "ASCII {b} must be its own alias"); }
    }
}


// ============================================================================================
// Oracle-free self-consistency
// ============================================================================================

/// **What can Ferric prove about a model with no second runtime to check against?**
///
/// `arch.rs` defines `Status::Verified` as "output was compared against the reference
/// implementation and matched". That definition has a hard edge: `laguna` sits one rung below it
/// not because anything is known to be wrong, but because llama.cpp answers
/// `unknown model architecture: laguna`, so there is nothing to diff against. An architecture's
/// confidence ceiling should not be set by another runtime's coverage.
///
/// These checks need only Ferric. Each is a property a correct forward pass satisfies **by
/// construction**, and each is violated by a real class of bug:
///
/// * **determinism** — the same tokens twice must give the same bits. Runs FIRST: if it fails, the
///   other two comparisons are noise and their verdicts mean nothing.
/// * **prefill vs decode** — prefilling T tokens in one call and stepping them one at a time
///   through the cache run genuinely different kernels (batched attention over T rows vs. a
///   single-row step against stored K/V), and must agree. Catches cache offset errors, position
///   bugs, and state leaking between steps.
/// * **prefix causality** — row `k` of a T-token prefill must equal the last row of a `k+1`-token
///   prefill. Position `k` cannot see position `k+1`. Catches mask leaks and residual wiring that
///   carries future state backwards.
///
/// ## Why the verdict is a ratio and not a bit-compare
///
/// The obvious form of these invariants is bit-identity, and that is what this checked first. It
/// FAILED on qwen3 — a `Verified` architecture — by 3.05e-5. The cause is not a bug: `q2_0_split_k`
/// selects the split-K kernel at `rows <= 2` and the flat one above it, so a 1-row decode step and
/// a 6-row prefill sum the same products **in a different order**. Pinning either kernel with
/// `FERRIC_Q2_0_KERNEL` drops the difference to exactly `0e0` on both comparisons — which is the
/// evidence that summation order was the whole of it.
///
/// So the default verdict is relative: `max|Δ| / max|logit|`. The two regimes this must separate
/// are far apart — reordered f32 accumulation lands around 1e-6 relative, while a mask leak or a
/// wrong RoPE base moves a logit by O(1) — so the gate sits at 1e-4, a hundredfold above the noise
/// and four orders below a real defect. `bitwise` is still reported, because on a pinned kernel it
/// is the stronger statement and it does hold.
///
/// ## What this cannot do — measured, not argued
///
/// These prove **self-consistency, not correctness**. That is easy to write and easy to
/// under-weight, so it was tested: `FERRIC_ROPE_NORM=1` forces qwen3 (a NEOX architecture) onto the
/// interleaved RoPE pairing, which rotates the wrong partners — the exact defect that once let
/// `llama` hold a `verified` badge while answering "The capital of France is located in the United
/// States". Under that flag every check still passes, at 1.27e-6, indistinguishable from a healthy
/// run:
///
/// ```text
///   baseline            logits_fnv 183cb6f239f401c6   self-consistent
///   FERRIC_ROPE_NORM=1  logits_fnv 9d74409388e892db   self-consistent
/// ```
///
/// The fingerprints differ, so the fault genuinely took; the verdicts do not, because a wrong
/// rotation applied in prefill and decode alike is wrong *consistently*. (`max_abs_logit` was 24.116
/// in BOTH runs — reading that alone says the flag did nothing, which is why the fingerprint is
/// reported.)
///
/// So the bounded claim is: these rule out the **cache, mask, and position** class of bug — state
/// leaking between steps, a prefill that disagrees with its own decode, a token that can see its
/// future. They say nothing about whether the arithmetic being applied consistently is the RIGHT
/// arithmetic. They cannot promote an architecture to `Verified`, and nothing here should be read as
/// licensing that. What they do is separate "no evidence exists" from "evidence exists and it
/// holds" — which for an architecture no other runtime will load is the whole of the difference.
pub struct SelfCheck {
    pub arch: String,
    pub n_tok: usize,
    pub deterministic: bool,
    /// Agreement within the relative gate — the verdict that tolerates kernel selection.
    pub prefill_matches_decode: bool,
    pub causal: bool,
    /// Agreement to the last bit. True only when one kernel serves both row counts.
    pub bitwise: bool,
    pub max_abs_diff: f32,
    pub max_abs_logit: f32,
    /// FNV over the RAW BITS of the reference prefill's last-token logits.
    ///
    /// Not part of any verdict — it answers a different question, and one that bit this very
    /// session: **did the thing I changed change the model at all?** Two runs under different
    /// settings that report the same fingerprint did the same arithmetic, whatever the setting
    /// claimed to do. `max_abs_logit` is too coarse to serve: a real change can leave the largest
    /// logit untouched, and an ineffective flag can be mistaken for a tolerated one.
    pub logits_fnv: u64,
    /// **Did a deliberately corrupted run actually fail?** Not optional and not env-gated: a check
    /// that has never been seen to fail is not evidence. The first version of this WAS env-gated
    /// and asserted only that the mutated run failed — which it did, while the clean run was
    /// failing identically, so the control passed without the mutation causing anything.
    pub control_ok: bool,
}

impl SelfCheck {
    /// Relative disagreement — the quantity the gate is applied to.
    pub fn rel(&self) -> f32 {
        if self.max_abs_logit > 0.0 { self.max_abs_diff / self.max_abs_logit } else { self.max_abs_diff }
    }
    /// A pass requires the control to have fired. Green with a dead control is not a pass.
    pub fn passed(&self) -> bool {
        self.control_ok && self.deterministic && self.prefill_matches_decode && self.causal
    }
}

/// Gate on relative disagreement. See `SelfCheck` for why this is not zero by default.
const SELFCHECK_REL_GATE: f32 = 1e-4;

/// Run the oracle-free checks against a GGUF, including the mutation control.
pub fn self_check(path: &str) -> SelfCheck {
    let eng = Engine::load(path, "selfcheck".to_string());
    let g = GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e:?}"));
    let arch = match g.metadata.get("general.architecture") {
        Some(Meta::Str(s)) => s.clone(), _ => String::new(),
    };
    let vn = eng.model.n_vocab();

    // Fixed ids rather than text: this must run on an architecture whose tokenizer may be unusual,
    // and the invariants are about the forward pass, not about tokenization.
    let n_tok = 6usize;
    let toks: Vec<u32> = (0..n_tok).map(|i| ((i * 977 + 13) % vn.max(1)) as u32).collect();

    let prefill = |t: &[u32]| -> Vec<f32> {
        let mut c = eng.model.new_cache();
        pollster::block_on(eng.model.forward_cached(t, &mut c).to_vec())
    };

    // ---- 1. determinism (must come first) ----
    let a = prefill(&toks);
    let deterministic = a == prefill(&toks);

    let mut max_abs_diff = 0f32;
    let mut bitwise = true;
    let mut max_abs_logit = 0f32;
    for v in &a { let m = v.abs(); if m > max_abs_logit { max_abs_logit = m; } }
    let mut logits_fnv: u64 = 0xcbf2_9ce4_8422_2325;
    for v in &a[a.len() - vn..] {
        logits_fnv ^= v.to_bits() as u64;
        logits_fnv = logits_fnv.wrapping_mul(0x1000_0000_01b3);
    }

    // NaN-safe by construction: `f32::max` RETURNS THE OTHER OPERAND on NaN, so folding with it
    // would report a maximum difference of zero on a NaN-poisoned run and read as a pass. Bits are
    // compared first; the magnitude is commentary on a difference already detected.
    //
    // A free fn rather than a closure so the control below can call it WITHOUT its result feeding
    // `bitwise` — a deliberately corrupted comparison must not be able to set the real verdict.
    fn worst(x: &[f32], y: &[f32]) -> (f32, bool) {
        if x.len() != y.len() { return (f32::INFINITY, false); }
        let (mut w, mut bit_eq) = (0f32, true);
        for (p, q) in x.iter().zip(y) {
            if p.to_bits() != q.to_bits() { bit_eq = false; }
            let d = (p - q).abs();
            if d.is_nan() { w = f32::INFINITY; } else if d > w { w = d; }
        }
        (w, bit_eq)
    }

    // ---- 2. prefill vs decode ----
    let mut c = eng.model.new_cache();
    let mut stepped: Vec<f32> = Vec::new();
    for t in &toks {
        stepped = pollster::block_on(eng.model.forward_cached(&[*t], &mut c).to_vec());
    }
    let last_prefill = &a[a.len() - vn..];
    let (d_pd, bw_pd) = worst(last_prefill, &stepped[stepped.len() - vn..]);
    bitwise &= bw_pd;
    if d_pd > max_abs_diff { max_abs_diff = d_pd; }

    // ---- 3. prefix causality ----
    let mut d_causal = 0f32;
    for k in 1..n_tok {
        let short = prefill(&toks[..k]);
        let (d, bw) = worst(&short[short.len() - vn..], &a[(k - 1) * vn..k * vn]);
        bitwise &= bw;
        if d > d_causal { d_causal = d; }
    }
    if d_causal > max_abs_diff { max_abs_diff = d_causal; }

    let gate = |d: f32| -> bool {
        if max_abs_logit > 0.0 { d / max_abs_logit <= SELFCHECK_REL_GATE } else { d == 0.0 }
    };
    let prefill_matches_decode = gate(d_pd);
    let causal = gate(d_causal);

    // ---- 4. the control: corrupt one logit and require the gate to REJECT it ----
    //
    // The perturbation is sized to the gate, not to a bit: a one-bit flip is ~1e-7 relative and
    // would sit UNDER a 1e-4 gate, so a bit-flip control would fail to fire and be mistaken for a
    // dead check. It is set ten times the gate — the smallest corruption the verdict claims to
    // catch. Firing here proves the gate rejects what it says it rejects; it does not prove the
    // invariants are sensitive to model bugs, which only a real bug can show.
    let mut corrupted = last_prefill.to_vec();
    corrupted[0] += max_abs_logit * SELFCHECK_REL_GATE * 10.0;
    let (d_ctl, _) = worst(&corrupted, &stepped[stepped.len() - vn..]);
    let control_ok = !gate(d_ctl);

    SelfCheck { arch, n_tok, deterministic, prefill_matches_decode, causal, bitwise,
                max_abs_diff, max_abs_logit, logits_fnv, control_ok }
}

#[cfg(test)]
mod batching_support {
    /// Pins, against the source rather than against a belief, that every runtime the server is willing
    /// to batch actually implements batched decode — and that every runtime it is NOT willing to batch
    /// is refused **deliberately** rather than by omission.
    ///
    /// This guard has now failed twice in its intended way and been rewritten each time, which is the
    /// point of it:
    ///   1. It first looped over the runtimes and asserted nothing — a dead loop that read like
    ///      coverage, the same vacuous-guard failure caught in the rope-type audit.
    ///   2. It then asserted "no runtime but Dense contains forward_batch", which fired correctly the
    ///      moment four ports landed. But that phrasing had a shelf life: once every runtime has the
    ///      method, source text alone can no longer distinguish ported from served, and the assertion
    ///      would have had to be deleted rather than tightened.
    ///
    /// So the invariant is now the *correspondence* between the two lists, which does not expire: a
    /// runtime claimed batchable must have the method, and a runtime that has the method but is not
    /// claimed must appear literally in the refusing arm. The gap between "ported" and "served" is
    /// exactly where a verified port quietly becomes dead code.
    #[test]
    fn every_ported_runtime_has_an_explicit_serving_decision() {
        // ⚠ THE TRAILING `(` IS LOAD-BEARING. Without it the needle is a PREFIX of any renamed
        // method, so `forward_batch_RENAMED` still "contains" it and the check silently passes. That
        // exact mutation defeated the first version of this test.
        const NEEDLE: &str = "pub fn forward_batch(";

        // ⚠ Read ONLY the body of `supports_batching`, never the whole file. The first version of this
        // test searched all of `lib.rs` for arm literals it also declared as `const` two lines above —
        // so `SELF_SRC.contains(ARM)` was satisfied by the test's own source and was true no matter
        // what the function said. Both mutations passed. That is the same failure as the rope-type
        // audit, where the test kept a copy of the predicate it was supposed to be checking.
        let self_src: &str = include_str!("lib.rs");
        let body = {
            let start = self_src.find("fn supports_batching(&self) -> bool {")
                .expect("supports_batching was renamed; this guard reads its body by name");
            let rest = &self_src[start..];
            let end = rest.find("\n    }").expect("unterminated supports_batching body");
            &rest[..end]
        };
        // Proof the slice is the function and not the whole file: the consts below live outside it.
        assert!(!body.contains("const NEEDLE"), "the extracted body swallowed this test's own source");

        // Every runtime is now batchable, so the refusing arm is empty and the interesting invariant
        // is different from what it was an hour ago: it is no longer "which list is it in" but "does
        // every variant appear in SOME arm, and does each one back its claim with a real method".
        const BLANKET_ARM: &str = "Model::Lfm2(_) | Model::Gemma4(_) | Model::DeepSeek2(_) => true";
        assert!(body.contains(BLANKET_ARM),
                "Model::supports_batching's arms changed shape; this guard reads them literally, so \
                 update both together rather than letting the guard go quiet.\nbody was:\n{body}");

        // (runtime source, its Model variant, does it consult its OWN batching_supported predicate)
        for (name, src, variant, asks_runtime) in [
            ("qwen3",     include_str!("../../ferric-llama/src/qwen3.rs"),     "Model::Dense",     true),
            ("qwen35",    include_str!("../../ferric-llama/src/qwen35.rs"),    "Model::Hybrid",    true),
            ("lfm2",      include_str!("../../ferric-llama/src/lfm2.rs"),      "Model::Lfm2",      false),
            ("gemma4",    include_str!("../../ferric-llama/src/gemma4.rs"),    "Model::Gemma4",    false),
            ("deepseek2", include_str!("../../ferric-llama/src/deepseek2.rs"), "Model::DeepSeek2", false),
        ] {
            assert!(src.contains(NEEDLE),
                    "{name} is claimed batchable by Model::supports_batching but has no {NEEDLE} — \
                     either the method was removed or the claim is wrong");
            assert!(body.contains(variant),
                    "{variant} appears in no arm of Model::supports_batching — a runtime must never \
                     fall out of the match entirely");
            if asks_runtime {
                // Its batched path has a checkpoint-dependent branch, so the server must ask the
                // runtime rather than answer for it.
                assert!(body.contains(&format!("{variant}(m) => m.batching_supported()")),
                        "{variant} has a per-checkpoint predicate that supports_batching stopped \
                         consulting; a blanket `true` here would batch a checkpoint the runtime knows \
                         it cannot batch");
                // ⚠ HONEST LABEL: this one is subsumed by the compiler, not independent coverage.
                // Removing the method from the runtime makes `m.batching_supported()` above fail to
                // build, so the mutation never reaches this assertion — verified by running it
                // (`error[E0599]: no method named batching_supported`). Kept only to name the coupling
                // at the point where it matters; the assertion above it is the one that bites.
                assert!(src.contains("pub fn batching_supported(&self) -> bool"),
                        "{name} lost the predicate supports_batching calls");
            }
        }
        // A brand-new runtime cannot slip past either list: Model::supports_batching is an exhaustive
        // match, so the compiler forces a decision before this test ever runs.
    }
    #[test]
    fn every_verified_registry_row_can_actually_be_loaded() {
        // THE TEST THAT WAS MISSING. `nemotron_h` shipped at Status::Verified, with a note claiming
        // it reproduces the reference and generates " Paris.", while this server's dispatch panicked
        // "its forward pass is not written yet". Two lists that had to agree, and nothing made them.
        //
        // The registry's guard for this lived in ferric-llama and could not see ferric-serve, so it
        // asserted the only thing it could reach — that each row NAMES a runtime — with a match whose
        // eight arms all returned `true`. It was `assert!(true)` and it passed for as long as the
        // contradiction existed.
        //
        // `Verified` is the registry's strongest claim. A row carrying it that this server refuses is
        // a lie in the strongest place, so that is what this asserts.
        use crate::Model;
        use ferric_llama::arch::{Status, REGISTRY};
        for a in REGISTRY {
            if a.status != Status::Verified { continue; }
            if let Err(why) = Model::dispatchable(a.runtime) {
                // Non-generative runtimes are legitimately un-loadable HERE, but only the two that
                // are non-generative by nature. Anything else is drift.
                // The refusable set: runtimes that are non-generative BY NATURE. `Parakeet` joined
                // it when speech landed — it takes a waveform, not tokens. Adding a runtime here
                // must be a deliberate edit, which is the whole point: this test went red the moment
                // parakeet was registered, rather than letting a third refusal in unnoticed.
                assert!(matches!(a.runtime, ferric_llama::arch::Runtime::Bert
                                          | ferric_llama::arch::Runtime::Cosmos
                                          | ferric_llama::arch::Runtime::Parakeet),
                        "{} is Status::Verified but this server refuses it: {why}", a.name);
            }
        }
    }

    #[test]
    fn a_runnable_row_is_either_loadable_here_or_non_generative() {
        // The weaker companion, covering `Loads` rows too: the registry may describe work in
        // progress, but a row it calls runnable must not be one this server has simply never wired.
        use crate::Model;
        use ferric_llama::arch::{Runtime, REGISTRY};
        let mut refused: Vec<&str> = Vec::new();
        for a in REGISTRY {
            if !a.status.runnable() { continue; }
            if Model::dispatchable(a.runtime).is_err() { refused.push(a.name); }
        }
        // Exactly the two non-generative runtimes, named — so ADDING a refusal is a test failure
        // rather than a silent loss of support.
        let mut expect: Vec<&str> = REGISTRY.iter()
            .filter(|a| a.status.runnable()
                        && matches!(a.runtime, Runtime::Bert | Runtime::Cosmos | Runtime::Parakeet))
            .map(|a| a.name).collect();
        refused.sort(); expect.sort();
        assert_eq!(refused, expect,
                   "a runnable registry row is refused by this server for a reason other than being \
                    an encoder or a safetensors-only model");
    }

}
