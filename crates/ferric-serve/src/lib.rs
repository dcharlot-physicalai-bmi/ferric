//! **Ferric OpenAI-compatible server** — the adoption on-ramp. A dependency-light (std TCP + serde)
//! HTTP server exposing `/v1/chat/completions` (streaming + non-streaming), `/v1/completions`,
//! `/v1/models`, and `/health` over the pure-Rust cross-fabric runtime. Any OpenAI client, agent
//! framework (LangChain, LangGraph, the Vercel AI SDK, …), or `curl` points at it unchanged.
//!
//!   cargo run -p ferric-serve --release -- [model.gguf] [--port 8080] [--name my-model]
//!
//! **Several models** (`batch::Pool`, `models`). A request's `model` names any GGUF in `FERRIC_MODELS`
//! (default `~/.cache/ferric/hub`) by file stem, `owner/repo[:tag]` or path, and it loads on first use —
//! with no model argument the server answers for the whole directory, as `ollama serve` does. Each resident
//! model keeps its own batch; they share one GPU context and ONE power meter, so concurrent requests on
//! different models split each instant's watts. Ollama's `keep_alive` (and LM Studio's `ttl`) set how long
//! a model stays after its last request (default `--keep-alive 5m`; 0 unloads; negative = until evicted);
//! idle models are unloaded least-recently-used first to fit `--max-models` (3) and 75% of physical memory
//! in model files (`FERRIC_MAX_MEMORY`, GiB). A model named on the command line is never evicted for room
//! and answers requests naming a model nothing on disk matches (llama-server's behaviour); without one, that
//! is a 404 naming what is on disk. `/v1/models` and `/api/tags` list everything loadable, `/api/ps` what is
//! resident. ⚠ Loading runs on the engine thread: other models' generations pause for it (0.3–0.7 s for
//! 0.5–1B models measured). Encoders (`--embed`, `--rerank`, `--asr`) are loaded from flags, not on demand.
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
//! **Request parameters honoured** (chat and completions, serial and batched paths alike — see `genopts`):
//! `max_tokens` / `max_completion_tokens` (absent = until a stop token or the context runs out),
//! `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repeat_penalty` /
//! `repeat_last_n`, `seed`, `stop` (string or array; the text is cut before it), `logprobs` /
//! `top_logprobs`, `stream_options.include_usage`. `finish_reason` is `"length"` at the budget, `"stop"`
//! otherwise. Message `content` may be a string or an array of text parts. `n > 1`, image and audio parts,
//! and malformed values are a 400 naming the field — never silently ignored. `--host` sets the bind
//! address (default 127.0.0.1); CORS preflight is answered.
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
mod genopts;
mod energy;
mod dialects;
mod audio;
mod ollama;
mod models;
mod images;
mod vision;
mod constrain;
mod trace;
mod batchapi;
mod local;
pub use local::LocalModel;
pub mod template;
mod specgate;
use genopts::{GenOpts, Emitter};
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::{qwen3, qwen35};
use ferric_llama::qwen3::Qwen3;
use ferric_llama::qwen35::Qwen35;
use ferric_tensor::Tensor;

/// The loaded model — a dense Qwen3/Llama/Gemma/Phi, or the Qwen3.5/3.6 **GDN-hybrid** (gated delta net
/// + periodic full attention). Both expose a `forward_cached` returning logits, so the generate loop and
/// guided decoding are architecture-agnostic; only the KV/recurrent cache type differs.
pub(crate) enum Model { Dense(Qwen3), Hybrid(Qwen35), Lfm2(ferric_llama::lfm2::Lfm2), Gemma4(ferric_llama::gemma4::Gemma4), DeepSeek2(ferric_llama::deepseek2::DeepSeek2), NemotronH(ferric_llama::nemotron_h::NemotronH), Hyv4(ferric_llama::hyv4::Hyv4) }
pub(crate) enum ModelCache { Dense(qwen3::Cache), Hybrid(qwen35::Cache), Lfm2(ferric_llama::lfm2::Cache), Gemma4(ferric_llama::gemma4::Cache), DeepSeek2(ferric_llama::deepseek2::Cache), NemotronH(ferric_llama::nemotron_h::Cache), Hyv4(ferric_llama::hyv4::Hyv4Cache) }
impl Model {
    fn n_vocab(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_vocab, Model::Hybrid(m) => m.cfg.n_vocab, Model::Lfm2(m) => m.cfg.n_vocab, Model::Gemma4(m) => m.cfg.n_vocab, Model::DeepSeek2(m) => m.cfg.n_vocab, Model::NemotronH(m) => m.cfg.n_vocab, Model::Hyv4(m) => m.cfg.n_vocab } }
    fn n_layer(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_layer, Model::Hybrid(m) => m.cfg.n_layer, Model::Lfm2(m) => m.cfg.n_layer, Model::Gemma4(m) => m.cfg.n_layer, Model::DeepSeek2(m) => m.cfg.n_layer, Model::NemotronH(m) => m.cfg.n_layer, Model::Hyv4(m) => m.cfg.n_layer } }
    fn n_embd(&self) -> usize { match self { Model::Dense(m) => m.cfg.n_embd, Model::Hybrid(m) => m.cfg.n_embd, Model::Lfm2(m) => m.cfg.d, Model::Gemma4(m) => m.cfg.d, Model::DeepSeek2(m) => m.cfg.d, Model::NemotronH(m) => m.cfg.d, Model::Hyv4(m) => m.cfg.d } }
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
            R::Dense | R::Hybrid | R::Lfm2 | R::Gemma4 | R::DeepSeek2 | R::NemotronH | R::Hyv4 => Ok(()),
            R::Bert => Err("a BERT encoder: no KV cache and no LM head, so it cannot serve chat or \
                            completions. Point FERRIC_RERANK_MODEL at it instead"),
            R::ModernBert => Err("a ModernBERT encoder: RoPE, symmetric-band local attention and \
                                  GeGLU, but still no KV cache and no LM head, so it cannot serve \
                                  chat or completions. It is an embedding/decision encoder"),
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
            // Fallible because `CachePolicy::Expanded` is refused rather than substituted; `Latent`
            // is what `Hyv4Cache::new` asks for, so this cannot fail in practice and says so if it does.
            Model::Hyv4(m) => ModelCache::Hyv4(ferric_llama::hyv4::Hyv4Cache::new(m)
                .unwrap_or_else(|e| panic!("hyv4 cache: {e}"))),
        }
    }
    /// Logits for the LAST position only — all a generation step needs. The dense runtime heads just that
    /// row (a 1780-token prefill no longer projects 1780 rows through the vocabulary and reads back 1 GB);
    /// other runtimes compute the full rows and keep the last, as before.
    fn forward_cached_last(&self, tokens: &[u32], cache: &mut ModelCache) -> Tensor {
        if let (Model::Dense(m), ModelCache::Dense(c)) = (self, &mut *cache) { return m.forward_cached_last(tokens, c); }
        let n = self.n_vocab();
        let full = self.forward_cached(tokens, cache);
        let rows = full.numel() / n;
        full.reshape(&[rows, n]).narrow(0, rows - 1, 1).contiguous()
    }

    /// [`Model::forward_cached_last`] as a HOST row `[n_vocab]` — what the sampler reads. The dense runtime
    /// hands it over directly (on the NVIDIA tier that skips a wgpu upload + readback of the row per token,
    /// `Qwen3::forward_cached_last_host`); the others read their tensor back as before.
    fn forward_cached_last_host(&self, tokens: &[u32], cache: &mut ModelCache) -> Vec<f32> {
        if let (Model::Dense(m), ModelCache::Dense(c)) = (self, &mut *cache) { return m.forward_cached_last_host(tokens, c); }
        let v = pollster::block_on(self.forward_cached_last(tokens, cache).to_vec());
        v[v.len() - self.n_vocab()..].to_vec()
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
            (Model::Hyv4(m), ModelCache::Hyv4(c)) => m.decode(tokens, c),
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
            // Ported AND verified token-identical to solo decode at n = 2/3/4, on sequences of
            // DIFFERENT lengths — equal lengths hide the whole class of bug, since borrowing
            // sequence 0's n_past gives every row the right answer when every row is at the same
            // position. `examples/hyv4_synthetic.rs` carries the proof and runs in CI on three
            // adapters; three mutations (rope from sequence 0, top-k offset from sequence 0, n_past
            // never advancing) are all caught by it.
            Model::Hyv4(_) => true,
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
            Model::Hyv4(m) => {
                let mut cs: Vec<_> = caches.iter_mut()
                    .map(|c| match &mut **c { ModelCache::Hyv4(x) => x, _ => unreachable!("model/cache kind mismatch") })
                    .collect();
                m.decode_batch(tokens, &mut cs)
            }
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
            // Same refusal, same reason: `decode` returns logits and no pre-head hidden state is
            // exposed. Returning the logits would make every embedding wrong while looking like a
            // vector, which is the failure this arm exists to prevent.
            Model::Hyv4(_) => panic!(
                "hyv4 cannot produce embeddings: no pre-head hidden-state path is exposed. It \
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

/// The encoder-side models, loaded once from flags and shared by every chat model the server loads.
pub(crate) struct Aux {
    /// A cross-encoder reranker, loaded SEPARATELY from `FERRIC_RERANK_MODEL`.
    ///
    /// It is a second model, not a mode of the first: rerankers are encoders with a classification
    /// head, they have no KV cache and no LM head, and nothing in `Model` fits them. llama-server
    /// takes the same shape with its own `--reranking` flag. Absent by default, so `/v1/rerank`
    /// answers 400 with what to set rather than pretending the endpoint does not exist.
    reranker: Option<ferric_llama::bert::Reranker>,
    /// A sentence embedder (bge, nomic-embed-text, MiniLM…) loaded from `--embed` / `FERRIC_EMBED_MODEL`,
    /// with the name `/v1/models` and `/api/tags` list it under. A chat model's hidden state pooled at its
    /// last token is not what a RAG pipeline means by an embedding; when this is loaded, `/v1/embeddings`
    /// uses it unless the request names the chat model.
    embedder: Option<(String, ferric_llama::bert::Embedder, ollama::Card)>,
    /// A speech recogniser for `/v1/audio/transcriptions`, from `--asr` / FERRIC_ASR_MODEL.
    pub(crate) asr: Option<(String, ferric_llama::parakeet::Parakeet)>,
}

impl Aux {
    fn load(ctx: &Arc<Context>) -> Aux {
        // Loaded before any chat model so a bad path fails immediately rather than after a multi-GB
        // load, and so the panic names which file was wrong.
        let reranker = std::env::var("FERRIC_RERANK_MODEL").ok().map(|rp| {
            let rg = GgufFile::open(&rp).unwrap_or_else(|e| panic!("open FERRIC_RERANK_MODEL {rp}: {e:?}"));
            ferric_llama::bert::Reranker::load(ctx, &rg)
                .unwrap_or_else(|e| panic!("load FERRIC_RERANK_MODEL {rp}: {e}"))
        });
        let embedder = std::env::var("FERRIC_EMBED_MODEL").ok().map(|ep| {
            let eg = GgufFile::open(&ep).unwrap_or_else(|e| panic!("open FERRIC_EMBED_MODEL {ep}: {e:?}"));
            let e = ferric_llama::bert::Embedder::load(ctx, &eg)
                .unwrap_or_else(|e| panic!("load FERRIC_EMBED_MODEL {ep}: {e}"));
            let name = std::path::Path::new(&ep).file_stem().and_then(|s| s.to_str()).unwrap_or("embed").to_string();
            eprintln!("ferric-serve: embedding model {name} ({}, d {}, pooling {}, context {})",
                      e.cfg().arch, e.cfg().d, e.cfg().pooling, e.cfg().n_ctx);
            let card = ollama::Card::from_gguf(&name, &ep, &eg, true, e.cfg().d, e.cfg().n_ctx, "");
            (name, e, card)
        });
        let asr = std::env::var("FERRIC_ASR_MODEL").ok().map(|ap| {
            let ag = GgufFile::open(&ap).unwrap_or_else(|e| panic!("open FERRIC_ASR_MODEL {ap}: {e:?}"));
            let m = ferric_llama::parakeet::Parakeet::load(ctx, &ag).unwrap_or_else(|e| panic!("load FERRIC_ASR_MODEL {ap}: {e}"));
            let name = std::path::Path::new(&ap).file_stem().and_then(|s| s.to_str()).unwrap_or("asr").to_string();
            eprintln!("ferric-serve: speech model {name} ({} Hz) — /v1/audio/transcriptions", m.cfg.sample_rate);
            (name, m)
        });
        Aux { reranker, embedder, asr }
    }
}

/// What every model in the process shares. One GPU context. The encoder-side models. ONE power meter,
/// so that two requests running on DIFFERENT models at the same instant split its watts: `n(t)` in
/// the attribution counts every request in flight, and a meter per model would charge each of them
/// the whole instant. The counters `/metrics` sums, and the Responses store (a `previous_response_id`
/// may continue on another model).
#[derive(Clone)]
pub(crate) struct Shared {
    ctx: Arc<Context>,
    aux: Arc<Aux>,
    energy: Arc<energy::Energy>,
    metrics: Arc<Metrics>,
    responses: Arc<dialects::ResponseStore>,
}

impl Shared {
    pub(crate) fn new() -> Shared {
        let ctx = Arc::new(pollster::block_on(Context::new()).unwrap());
        Shared { aux: Arc::new(Aux::load(&ctx)), ctx, energy: Arc::new(energy::Energy::start()),
                 metrics: Arc::new(Metrics::default()), responses: Arc::new(dialects::ResponseStore::new()) }
    }
}

pub(crate) struct Engine {
    ctx: Arc<Context>,
    model: Model,
    /// The embedder, reranker and speech model — the server's, shared with every other loaded model.
    pub(crate) aux: Arc<Aux>,
    /// What `/api/tags` and `/api/show` report about the chat model.
    card: ollama::Card,
    /// The GGUF's own chat template, compiled — `None` only when it is absent or will not compile, and
    /// then the family heuristic below is the fallback (with a warning at load).
    chat_template: Option<template::ChatTemplate>,
    /// This model's reasoning markers, read from its chat template: `<think>`/`</think>` or Gemma 4's
    /// `<|channel>thought`/`<channel|>`. `None` = not a thinking model.
    reasoning_markers: Option<(String, String)>,
    /// The long-lived power sampler every generation's joules are attributed from (the server's one).
    pub(crate) energy: Arc<energy::Energy>,
    /// Responses kept for the Responses API's `previous_response_id` (shared across models).
    responses: Arc<dialects::ResponseStore>,
    /// Counters for `/metrics` (server-wide).
    pub(crate) metrics: Arc<Metrics>,
    /// **Prompt caching across requests** (dense runtime): the K/V of recent sequences, keyed by their
    /// tokens. A multi-turn chat resends the whole conversation, and every turn used to prefill all of it;
    /// now the longest cached whole-chunk prefix is copied and only the rest is computed. `FERRIC_PREFIX_CACHE`
    /// = sequences kept (default 8; 0 = off).
    prefix_cache: Option<std::cell::RefCell<ferric_llama::prefix::PrefixCache>>,
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
    /// GGUF `<arch>.pooling_type` — 0 NONE, 1 MEAN, 2 CLS, 3 LAST, 4 RANK.
    ///
    /// ⛔ THIS KEY WAS NEVER READ HERE. `/v1/embeddings` hardcoded last-token pooling, which is
    /// right for Qwen3-Embedding (it declares 3) and silently WRONG for the MEAN family — BGE,
    /// E5, GTE, most sentence-transformer exports — where the final position of a
    /// bidirectionally-trained encoder carries no trained meaning. `ferric-web` read it
    /// correctly the whole time; the same rule lived in two crates and one of them was wrong,
    /// which is the `is_spm` defect over again. The rule now lives once, in
    /// `ferric_llama::pooling`, and both front ends call it.
    pooling: Option<u32>,
    eos: Vec<u32>,
    name: String,
    /// Raw bytes each token decodes to (for guided decoding); `None` for special/non-text tokens,
    /// which are disallowed under a constraint (except EOS, handled separately).
    token_bytes: Vec<Option<Vec<u8>>>,
    /// (special-token string, id), longest first — for special-token-aware tokenization of a
    /// rendered chat template (so `<|im_start|>` etc. encode to their id, not BPE'd text).
    specials: Vec<(String, u32)>,
    /// Special tokens after which the authors' tokenizer strips whitespace (`AddedToken(rstrip=True)`). A
    /// GGUF does not record the flag; it is restored from the authors' `tokenizer.json` per family.
    rstrip_after: std::collections::HashSet<u32>,
    /// The GGUF `chat_template` string (used only to detect the model's template family).
    template: String,
    /// One-slot prompt-prefix cache (hybrid speculative path): the last request's fed tokens plus
    /// the carried main + draft caches. Multi-turn chat re-sends the whole conversation; when the
    /// new prompt extends the cached tokens, only the new suffix is prefilled. Output is identical
    /// to a full prefill (cached-decode ≡ re-prefill, the invariant `--verify-cache` proves) —
    /// the conversation just stops being re-paid every turn. RefCell: the server is single-threaded.
    prefix: std::cell::RefCell<Option<PrefixSlot>>,
    /// `<arch>.context_length`: the prompt plus everything generated must fit. A request with no
    /// `max_tokens` generates until a stop token or this limit, as OpenAI, llama-server and Ollama do.
    n_ctx: usize,
    /// The image path, for a vision model served from its authors' checkpoint directory (`vision`).
    vision: Option<vision::Vision>,
    /// LoRA adapters named on the command line (`--lora name=path`), uploaded once: (name, adapter, card).
    adapters: Vec<(String, Arc<ferric_llama::lora::DeviceLora>, ollama::Card)>,
    /// The vocabulary as a byte trie, built on the first constrained request (`constrain`).
    trie: std::cell::OnceCell<constrain::Trie>,
    /// Recently used grammars with the per-top masks learned for them (`constrain`).
    grammars: std::cell::RefCell<Vec<(String, std::sync::Arc<constrain::Masks>)>>,
    /// JSON-object guide masks by guide state (`constrain`).
    json_masks: std::cell::RefCell<HashMap<ferric_agent::guide::Json, Vec<u64>>>,
    /// The model's own recommended sampling (`general.sampling.*` in the GGUF), as request fields.
    sampling_defaults: serde_json::Map<String, Value>,
}

/// What `/metrics` exposes, in Prometheus text format.
#[derive(Default)]
pub(crate) struct Metrics {
    pub requests: std::sync::atomic::AtomicU64,
    pub prompt_tokens: std::sync::atomic::AtomicU64,
    pub gen_tokens: std::sync::atomic::AtomicU64,
    pub cancelled: std::sync::atomic::AtomicU64,
    /// Attributed joules summed (only requests the meter could attribute), and how many were attributed.
    pub joules: std::sync::Mutex<(f64, u64)>,
    pub started: std::sync::OnceLock<std::time::Instant>,
}

impl Metrics {
    pub fn record(&self, prompt: usize, generated: usize, energy: &Value) {
        use std::sync::atomic::Ordering::Relaxed;
        self.requests.fetch_add(1, Relaxed);
        self.prompt_tokens.fetch_add(prompt as u64, Relaxed);
        self.gen_tokens.fetch_add(generated as u64, Relaxed);
        if let (Some(j), Ok(mut g)) = (energy["joules"].as_f64(), self.joules.lock()) { g.0 += j; g.1 += 1; }
    }
}

/// One finished generation.
pub(crate) struct GenOut {
    pub text: String,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// `"stop"` (a stop token, a stop string, or a guide with nowhere left to go) or `"length"`.
    pub finish: &'static str,
    /// Chat-format `logprobs.content` entries, one per generated token, when the request asked.
    pub logprobs: Vec<Value>,
    /// The generated token ids — decoded WITH special tokens for tool-call parsing (Gemma 4's argument
    /// delimiters are specials, which the visible text drops).
    pub ids: Vec<u32>,
    /// Joules attributed to this generation (see `energy`): `{"joules", "joules_per_token", …}`, or
    /// `{"joules": null, "why": …}` when the machine has no meter or the window was unmeasurable.
    pub energy: Value,
    /// The stop string that ended it, if one did.
    pub stop_seq: Option<String>,
}

/// See `Engine::prefix`. `fed` is exactly the token sequence the main cache has consumed —
/// kept consistent through speculative rollbacks.
struct PrefixSlot { fed: Vec<u32>, cache: qwen35::Cache, mc: qwen35::MtpCache }

impl Engine {
    /// Load one model with a fresh context and whatever encoder-side models the environment names.
    fn load(path: &str, name: String) -> Engine { Engine::load_in(&Shared::new(), path, name) }

    /// Load one chat model into the server's shared context, meter and encoder-side models.
    pub(crate) fn load_in(shared: &Shared, path: &str, name: String) -> Engine {
        let ctx = shared.ctx.clone();
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
        // SentencePiece models carry a per-token score array — detect and build an Spm.
        //
        // ⛔ THIS LIST USED TO BE `s == "llama"` ALONE while ferric-web's was `llama | gemma4 | t5`,
        // so the same Gemma-4 file was tokenized two different ways by the two front ends. Measured on
        // `gemma-4-E2B-it-Q8_0.gguf` (52.5% of its 262,144 tokens begin with ▁, exactly 1 begins with
        // Ġ): 0 of 4 prompts agreed, this path emitted 28 ids where the browser emitted 17, and the
        // lone Ġ token 245237 appeared 11 times — `[The, Ġ, capital, Ġ, of, Ġ, France, Ġ, is]`. The
        // model still answered, which is why it survived. Both front ends now call ONE predicate.
        let tok_model = match g.metadata.get("tokenizer.ggml.model") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
        // ⭐ And the name list is backed by a check that reads the VOCABULARY, so the next
        // SentencePiece checkpoint declaring an unlisted name is refused rather than mis-tokenized.
        if !ferric_tokenizer::is_sentencepiece_model(&tok_model) {
            // `load` returns `Engine`, so fail-closed here means refusing to START. That is the
            // right trade: a server that will not boot is a bug report, a server that tokenizes
            // a SentencePiece vocabulary as byte-level BPE is fluent wrong text nobody notices.
            if let Err(e) = ferric_tokenizer::check_byte_level_choice(&tok_model, &tokens) {
                panic!("{e}");
            }
        }
        // ⛔ Say something when the pre-tokenizer is unimplemented. llama.cpp throws here; Ferric
        // falls back to GPT-2, and the four defects found on 2026-09-20 all hid in that silence.
        if let Some(w) = ferric_tokenizer::Pre::fall_open_warning(
            match g.metadata.get("tokenizer.ggml.pre") { Some(Meta::Str(p)) => p.as_str(), _ => "" }) {
            eprintln!("{w}");
        }
        let spm = match g.metadata.get("tokenizer.ggml.model") {
            Some(Meta::Str(s)) if ferric_tokenizer::is_sentencepiece_model(s) => {
                let scores: Vec<f32> = match g.metadata.get("tokenizer.ggml.scores") {
                    Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::F(v) = m { *v as f32 } else { 0.0 }).collect(),
                    _ => Vec::new(),
                };
                // USER_DEFINED entries are matched verbatim — see ferric_gguf::token_types
                Some(Spm::with_types(tokens.clone(), scores, &ferric_gguf::token_types(g.metadata.get("tokenizer.ggml.token_type"))))
            }
            _ => None,
        };
        let add_space_prefix = match g.metadata.get("tokenizer.ggml.add_space_prefix") { Some(Meta::Bool(b)) => *b, _ => true };
        let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
        let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
        let eos_id = match g.metadata.get("tokenizer.ggml.eos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
        let add_eos = matches!(g.metadata.get("tokenizer.ggml.add_eos_token"), Some(Meta::Bool(true)));
        // Architecture-prefixed (`bert.pooling_type`, `qwen3.pooling_type`, …), so it is looked up
        // by SUFFIX rather than by guessing the arch name. One reader serves every family.
        let pooling = ferric_llama::pooling::declared_pooling(g.metadata.iter());
        let mut eos: Vec<u32> = Vec::new();
        if let Some(e) = eos_id { eos.push(e); }
        let im_end = vocab.get("<|im_end|>").copied();
        let im_start = vocab.get("<|im_start|>").copied();
        if let Some(e) = im_end { if !eos.contains(&e) { eos.push(e); } }
        if let Some(&e) = vocab.get("<|endoftext|>") { if !eos.contains(&e) { eos.push(e); } }
        // Gemma ends a turn with <end_of_turn>; Phi-3 with <|end|> — treat both as stop tokens.
        // Llama 3.1+ ends a tool-call message with <|eom_id|> ("end of message", more to come from the tool),
        // which llama.cpp also treats as end-of-generation; without it a tool call ran to max_tokens.
        for t in ["<end_of_turn>", "<|end|>", "<|eom_id|>"] { if let Some(&e) = vocab.get(t) { if !eos.contains(&e) { eos.push(e); } } }
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
            // ⛔ STREAMED, NOT RESIDENT. hyv4's published checkpoint is 213.66 GiB; a resident load
            // of that is killed by the OS with no panic, no error and no exit code — the process
            // simply stops. `Hyv4::load` would do exactly that here. `load_streaming` keeps only a
            // budget of block weights live and rebuilds the rest per token, which is what makes the
            // model runnable on a machine that cannot hold it.
            ferric_llama::arch::Runtime::Hyv4 => {
                let gib: f64 = std::env::var("FERRIC_STREAM_GIB").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(12.0);
                Model::Hyv4(ferric_llama::hyv4::Hyv4::load_streaming(
                        &ctx, path, (gib * 1073741824.0) as u64)
                    .unwrap_or_else(|e| panic!("load hyv4 (streaming, {gib:.1} GiB budget): {e}")))
            }
            ferric_llama::arch::Runtime::Bert =>
                unreachable!("Bert is refused by Model::dispatchable before this match"),
            ferric_llama::arch::Runtime::ModernBert =>
                unreachable!("ModernBert is refused by Model::dispatchable before this match"),
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
        // Absent is not "unlimited": 4096 is a conservative bound for a file that does not say, and the
        // error it produces names the number so a caller can see why.
        let trained = match g.metadata.get(&format!("{arch}.context_length")) { Some(Meta::U(v)) => *v as usize, _ => 4096 };
        let n_ctx = ctx_cap(trained);
        // Phi-3 / 3.5: microsoft/Phi-3.5-mini-instruct's tokenizer.json marks every `<|…|>` added token
        // rstrip=True except `<|endoftext|>`, so "<|user|>\nHi" tokenises as "<|user|>Hi". Without this a
        // two-turn prompt was 36 tokens where the authors' tokenizer gives 26 (checked against HF
        // apply_chat_template). llama.cpp restores the same flag by model name.
        let reasoning_markers = if template.contains("<think>") { Some(("<think>".to_string(), "</think>".to_string())) }
            else if template.contains("<|channel>thought") { Some(("<|channel>thought".to_string(), "<channel|>".to_string())) }
            else { None };
        let rstrip_after: std::collections::HashSet<u32> = if arch == "phi3" {
            specials.iter().filter(|(t, _)| t.starts_with("<|") && t.ends_with("|>") && t != "<|endoftext|>").map(|(_, i)| *i).collect()
        } else { Default::default() };
        let card = ollama::Card::from_gguf(&name, path, &g, false, model.n_embd(), n_ctx, &template);
        let sampling_defaults = genopts::model_sampling_defaults(&g.metadata);
        if !sampling_defaults.is_empty() {
            eprintln!("ferric-serve: sampling defaults from the GGUF (general.sampling.*), used where a request sets none: {}{}",
                serde_json::Value::Object(sampling_defaults.clone()),
                if std::env::var("FERRIC_MODEL_SAMPLING").as_deref() == Ok("0") { " — OFF (FERRIC_MODEL_SAMPLING=0)" } else { "" });
        }
        let tok_str = |id: Option<u32>| id.and_then(|i| tokens.get(i as usize).cloned()).unwrap_or_default();
        let chat_template = if template.is_empty() { None } else {
            match template::ChatTemplate::compile(&template, &tok_str(bos_id), &tok_str(eos_id)) {
                Ok(t) => Some(t),
                Err(e) => { eprintln!("ferric-serve: ⚠ {e}; falling back to the vocabulary-family template"); None }
            }
        };
        Engine { ctx, model, bpe, spm, add_space_prefix, tokens, u2b, im_start, im_end, bos_id, add_bos, eos_id, add_eos, pooling, eos, name, token_bytes, specials, template, prefix: std::cell::RefCell::new(None), spec_gate, aux: shared.aux.clone(), card, chat_template, reasoning_markers, energy: shared.energy.clone(), responses: shared.responses.clone(), metrics: shared.metrics.clone(),
                 prefix_cache: {
                     let n: usize = std::env::var("FERRIC_PREFIX_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
                     (n > 0).then(|| std::cell::RefCell::new(ferric_llama::prefix::PrefixCache::new(n)))
                 },
                 rstrip_after, n_ctx, vision: None, adapters: Vec::new(), trie: Default::default(), grammars: Default::default(), json_masks: Default::default(), sampling_defaults }
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
                // Every text fragment here starts the string or follows a special token, and a
                // SentencePiece model that adds a space prefix adds it to EACH such fragment: the authors'
                // legacy normalizer is `Prepend("▁")` per segment (microsoft/Phi-3.5-mini-instruct's
                // tokenizer.json), and llama.cpp's SPM tokenizer does the same (`is_prev_special`). Only
                // the first got it before, so "<|system|>You" encoded as `You` where the authors write `▁You`.
                Some((pos, s, id)) => {
                    if pos > 0 { ids.extend(self.enc(&rest[..pos], true)); }
                    ids.push(id);
                    rest = &rest[pos + s.len()..];
                    if self.rstrip_after.contains(&id) { rest = rest.trim_start(); }
                }
                None => { ids.extend(self.enc(rest, true)); break 'outer; }
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
        // Content was validated by `chat_ids` (string, null or text parts); an unreadable one is empty here.
        let m = |v: &Value| (v["role"].as_str().unwrap_or("user").to_string(), genopts::content_text(&v["content"]).unwrap_or_default());
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

    /// `detok`, but a special token is written as its literal text — the form a tool-call parser needs.
    pub(crate) fn detok_all(&self, ids: &[u32]) -> String {
        let special: std::collections::HashMap<u32, &str> = self.specials.iter().map(|(t, i)| (*i, t.as_str())).collect();
        let (mut out, mut run) = (String::new(), Vec::new());
        for &i in ids {
            if let Some(t) = special.get(&i) {
                if !run.is_empty() { out.push_str(&self.detok(&run)); run.clear(); }
                out.push_str(t);
            } else { run.push(i); }
        }
        if !run.is_empty() { out.push_str(&self.detok(&run)); }
        out
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
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let n = self.model.n_embd();
        let mut ids = self.enc(text, true);
        // Qwen3-Embedding (add_eos_token) appends EOS and pools ITS hidden state; append it to match.
        if self.add_eos { if let Some(e) = self.eos_id { ids.push(e); } }
        // Empty input: keep the response's vectors equal-length (a zero vector), not a []; some clients
        // build a matrix over a batch and a ragged row breaks them.
        if ids.is_empty() { return Ok(vec![0.0; n]); }
        let v = pollster::block_on(self.model.forward_hidden(&ids).to_vec()); // [T·n_embd]
        let t = (v.len() / n).max(1);
        // ⭐ The checkpoint's declared rule, not a hardcoded one. `unwrap_or(3)` keeps the old
        // behaviour for files that declare nothing — it changes the answer only where the file
        // said so and this endpoint was ignoring it. A type we cannot honour (NONE, RANK) refuses
        // inside `pool`; returning a zero vector here would be a silently-wrong embedding, which is
        // the failure this whole change exists to remove.
        // ⛔ A type we cannot honour (NONE, RANK) REFUSES. It must not fall back here: this function
        // already returns a zero vector to mean "empty input", so reusing it for "wrong pooling"
        // would hand the client a well-formed embedding that ranks arbitrarily — the exact defect
        // this change removes, reintroduced one line below the fix.
        let pooled = ferric_llama::pooling::pool(&v, t, n, self.pooling.unwrap_or(3))?;
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(pooled.iter().map(|x| x / norm).collect())
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

    fn chat_ids(&self, messages: &[Value]) -> Result<Vec<u32>, String> {
        self.chat_ids_with(messages, None, &Default::default())
    }

    /// Whether `tools` go to the model's own template (it reads them) rather than into a Hermes system
    /// prompt of Ferric's making.
    pub(crate) fn template_handles_tools(&self) -> bool { self.chat_template.as_ref().is_some_and(|t| t.handles_tools) }

    /// The prompt for a conversation: the model's OWN chat template when the file carries one (rendered
    /// as Hugging Face renders it — see `template`), with `tools` and `chat_template_kwargs`
    /// (`enable_thinking`, …) passed through; the vocabulary-family heuristic only when it does not.
    pub(crate) fn chat_ids_with(&self, messages: &[Value], tools: Option<&[Value]>,
                                kwargs: &serde_json::Map<String, Value>) -> Result<Vec<u32>, String> {
        // A conversation carrying an image: placed by the vision path, or refused naming why.
        if let Some((ids, _)) = self.vision_prompt(messages, tools, kwargs)? { return Ok(ids); }
        // ⛔ An OpenAI content-part array used to read as "" here (`as_str()` on an array), so the model
        // answered a prompt with the user's words missing. Parts are read, and a part this path cannot
        // feed (an image) is refused by name.
        for (i, m) in messages.iter().enumerate() {
            genopts::content_text(&m["content"]).map_err(|e| format!("messages[{i}]: {e}"))?;
        }
        if let Some(t) = &self.chat_template {
            // Templates read `content` as a string, and tool-call `arguments` as an object (OpenAI clients
            // send a JSON string; vLLM parses it the same way before templating).
            let msgs: Vec<Value> = messages.iter().map(|m| prepare_for_template(m)).collect();
            let text = t.render(&msgs, tools, true, kwargs)?;
            let mut ids = self.encode_special(&text);
            if self.spm.is_some() && self.add_bos {
                if let Some(b) = self.bos_id { if ids.first() != Some(&b) { ids.insert(0, b); } }
            }
            return Ok(ids);
        }
        if !self.has_chat_family() {
            // No recognized chat family in the vocab → a base model — plain concatenation.
            let text: String = messages.iter().map(|m| format!("{}: {}\n", m["role"].as_str().unwrap_or("user"), genopts::content_text(&m["content"]).unwrap_or_default())).collect();
            let mut ids = Vec::new();
            if self.add_bos { if let Some(b) = self.bos_id { ids.push(b); } }
            ids.extend(self.enc(&text, true));
            return Ok(ids);
        }
        let mut ids = self.encode_special(&self.render_chat(messages));
        // SentencePiece templates (Phi-3/Mistral) don't embed BOS; add_bos prepends it. (BPE templates
        // like Llama-3 carry their own <|begin_of_text|>, so guard on it not already leading.)
        if self.spm.is_some() && self.add_bos {
            if let Some(b) = self.bos_id { if ids.first() != Some(&b) { ids.insert(0, b); } }
        }
        Ok(ids)
    }

    /// The prompt, and the image it carries if any — what a generating caller needs (a validating one
    /// needs only `chat_ids_with`).
    /// The prompt a tool-free chat request renders to — the same rules as `run_chat` (`developer` is the
    /// system role; `template_kwargs`), for the batched path.
    pub(crate) fn chat_request_ids(&self, req: &Value) -> Result<Vec<u32>, String> {
        let empty = vec![];
        let mut messages: Vec<Value> = req["messages"].as_array().unwrap_or(&empty).clone();
        for m in messages.iter_mut() { if m["role"] == "developer" { m["role"] = json!("system"); } }
        self.chat_prompt(&messages, None, &template_kwargs(req)).map(|(p, _)| p)
    }

    /// A thinking model's reasoning/answer splitter for a reply to `prompt` (the block may already be open
    /// in the prompt's tail), or None for a model without reasoning markers.
    pub(crate) fn reasoning_split(&self, prompt: &[u32]) -> Option<genopts::ReasoningSplit> {
        self.reasoning_markers.as_ref().map(|(o, c)| {
            let tail = self.detok_all(&prompt[prompt.len().saturating_sub(24)..]);
            let started = tail.rfind(o.as_str()).is_some_and(|a| tail.rfind(c.as_str()).is_none_or(|b| a > b));
            genopts::ReasoningSplit::new(o, c, started)
        })
    }

    pub(crate) fn chat_prompt(&self, messages: &[Value], tools: Option<&[Value]>, kwargs: &serde_json::Map<String, Value>)
        -> Result<(Vec<u32>, Option<Arc<vision::MmInput>>), String>
    {
        if let Some((ids, mm)) = self.vision_prompt(messages, tools, kwargs)? { return Ok((ids, Some(mm))); }
        Ok((self.chat_ids_with(messages, tools, kwargs)?, None))
    }

    /// How many tokens this request may generate: its own limit, capped by what is left of the context.
    /// A prompt that does not fit is an error naming both numbers, never a silent truncation.
    pub(crate) fn budget(&self, prompt_len: usize, o: &GenOpts) -> Result<usize, String> {
        // The request's own window (Ollama's `num_ctx`) can only narrow the server's.
        let (ctx, whose) = match o.num_ctx { Some(c) if c < self.n_ctx => (c, "this request's num_ctx"), _ => (self.n_ctx, "this model's context") };
        if prompt_len >= ctx {
            return Err(format!("the prompt is {prompt_len} tokens and {whose} is {ctx} — shorten the conversation"));
        }
        Ok(o.max_tokens.unwrap_or(usize::MAX).min(ctx - prompt_len))
    }

    /// Every loaded model's card: the chat model, its LoRA adapters, then the embedder.
    pub(crate) fn cards(&self) -> Vec<ollama::Card> {
        let mut v = vec![self.card.clone()];
        v.extend(self.adapters.iter().map(|(_, _, c)| c.clone()));
        if let Some((_, _, c)) = &self.aux.embedder { v.push(c.clone()); }
        v
    }

    /// A raw prompt's ids: BOS when the model adds one, then the text — no chat template.
    pub(crate) fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos { if let Some(b) = self.bos_id { ids.push(b); } }
        ids.extend(self.enc(text, true));
        ids
    }

    /// A fresh cache for `prompt`, seeded from the prefix cache when this is the dense runtime and a
    /// previous sequence shares a whole-chunk prefix. Returns the cache and how many prompt tokens it
    /// already holds. Never the WHOLE prompt: the last token must be fed to produce the next logits.
    pub(crate) fn seeded_cache(&self, prompt: &[u32], lora: &genopts::Lora) -> (ModelCache, usize) {
        if let Model::Dense(m) = &self.model {
            // The adapters go on BEFORE seeding: the prompt cache keys its entries on the selection, and a
            // cache refuses a new selection once it holds tokens.
            let mut c = qwen3::Cache::new(&m.cfg);
            if !lora.0.is_empty() { c.set_adapters(lora.0.clone()).expect("a fresh cache takes any selection"); }
            let n = match (&self.prefix_cache, prompt.len() > 1) {
                (Some(pc), true) => pc.borrow_mut().seed(&self.ctx, &prompt[..prompt.len() - 1], &mut c).map(|h| h.tokens).unwrap_or(0),
                _ => 0,
            };
            return (ModelCache::Dense(c), n);
        }
        (self.model.new_cache(), 0)
    }

    /// A request's generation options, including which LoRA adapters it runs with:
    /// - `model` naming a loaded adapter (vLLM's convention: an adapter is served as a model of its own);
    /// - `lora: [{"id": i | "name": n, "scale": s}]` (llama-server's; `id` is the adapter's place among
    ///   the `--lora` flags). Several SUM, as PEFT does; scale 0 drops one; an unknown one is refused.
    pub(crate) fn gen_opts(&self, req: &Value, chat: bool) -> Result<GenOpts, String> {
        let merged;
        let req = if self.sampling_defaults.is_empty() || std::env::var("FERRIC_MODEL_SAMPLING").as_deref() == Ok("0") { req } else {
            merged = genopts::with_defaults(req, &self.sampling_defaults);
            &merged
        };
        let mut o = GenOpts::from_req(req, chat)?;
        // The reasoning budget (llama.cpp's semantics): the request's `thinking_budget_tokens`, else the
        // server's `--reasoning-budget`; -1 = unlimited. Only a model with reasoning markers has one.
        let budget = match &req["thinking_budget_tokens"] {
            Value::Null => std::env::var("FERRIC_REASONING_BUDGET").ok().and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(-1),
            v => v.as_i64().filter(|&n| n >= -1).ok_or_else(|| format!("`thinking_budget_tokens` must be -1 (unlimited) or a count, got {v}"))?,
        };
        if budget >= 0 {
            let Some((open, close)) = &self.reasoning_markers else {
                return Err("`thinking_budget_tokens`: this model has no reasoning markers, so there is no thinking to budget".into());
            };
            let marker = |s: &str| self.specials.iter().find(|(t, _)| t == s).map(|(_, i)| vec![*i]).unwrap_or_else(|| self.enc(s, false));
            let message = req["reasoning_budget_message"].as_str().map(String::from)
                .or_else(|| std::env::var("FERRIC_REASONING_BUDGET_MESSAGE").ok()).unwrap_or_default();
            let mut forced = if message.is_empty() { Vec::new() } else { self.enc(&message, false) };
            forced.extend(marker(close));
            o.sampling.reasoning_budget = Some(Arc::new(genopts::BudgetCfg { start: marker(open), end: marker(close), forced, budget: budget as i32 }));
        }
        // The tokenizer-dependent sampler inputs, resolved as their reference implementation resolves them:
        // DRY's breakers are the LAST id of "a" + breaker (so a breaker is tokenized as text-final); XTC's
        // specials are the last id of "\n" and EOS; a text logit_bias applies to every token of its text.
        let last_id = |t: &str| self.enc(&format!("a{t}"), false).last().copied();
        if o.sampling.dry_multiplier > 0.0 {
            o.sampling.dry_breakers = o.sampling.dry_breaker_strings.iter().filter_map(|b| last_id(b)).collect();
        }
        if o.sampling.xtc_probability > 0.0 {
            o.sampling.xtc_specials = self.enc("\n", false).last().copied().into_iter().chain(self.eos_id).collect();
        }
        for (text, b) in std::mem::take(&mut o.logit_bias_text) {
            for id in self.enc(&text, false) { o.sampling.logit_bias.push((id, b)); }
        }
        if let Some(&(t, _)) = o.sampling.logit_bias.iter().find(|(t, _)| *t as usize >= self.model.n_vocab()) {
            return Err(format!("logit_bias token {t} is outside the vocabulary ({})", self.model.n_vocab()));
        }
        let find = |n: &str| self.adapters.iter().position(|(a, _, _)| a == n.strip_suffix(":latest").unwrap_or(n));
        let names = || self.adapters.iter().map(|(a, _, _)| a.as_str()).collect::<Vec<_>>().join(", ");
        if let Some(i) = req["model"].as_str().and_then(find) { o.lora.0.push((self.adapters[i].1.clone(), 1.0)); }
        match &req["lora"] {
            Value::Null => {}
            Value::Array(sel) => for (k, x) in sel.iter().enumerate() {
                let i = match (&x["id"], &x["name"]) {
                    (Value::Number(n), _) => n.as_u64().map(|n| n as usize).filter(|&n| n < self.adapters.len())
                        .ok_or_else(|| format!("lora[{k}].id {n}: {} adapter(s) are loaded ({})", self.adapters.len(), names()))?,
                    (_, Value::String(n)) => find(n).ok_or_else(|| format!("lora[{k}].name {n:?} is not loaded; loaded: {}", names()))?,
                    _ => return Err(format!("lora[{k}] needs an `id` or a `name`")),
                };
                let scale = match &x["scale"] { Value::Null => 1.0, v => v.as_f64().ok_or_else(|| format!("lora[{k}].scale must be a number"))? as f32 };
                if scale != 0.0 { o.lora.0.push((self.adapters[i].1.clone(), scale)); }
            },
            v => return Err(format!("`lora` must be an array of {{id|name, scale}}, got {v}")),
        }
        Ok(o)
    }

    /// Upload the `--lora name=path` adapters onto this (dense) model.
    pub(crate) fn load_adapters(&mut self, specs: &[(String, String)]) -> Result<(), String> {
        for (name, path) in specs {
            let Model::Dense(m) = &self.model else { return Err(format!("--lora {name}: LoRA is served on the dense runtime only")); };
            let a = ferric_load::lora::LoraAdapter::open(path).map_err(|e| format!("--lora {name}={path}: {e}"))?;
            let dev = m.upload_lora(&a).map_err(|e| format!("--lora {name}={path}: {e}"))?;
            let mut card = self.card.clone();
            card.name = name.clone();
            card.path = path.clone();
            card.size = walk_size(std::path::Path::new(path));
            eprintln!("ferric-serve: LoRA adapter {name} ({}, base {}) — request it as `model: \"{name}\"` or `lora: [{{\"name\": \"{name}\"}}]`",
                      path, a.base_model.as_deref().unwrap_or("undeclared"));
            self.adapters.push((name.clone(), dev, card));
        }
        Ok(())
    }

    /// Keep `tokens` (exactly the ones this cache has consumed) for later requests to reuse.
    ///
    /// `&mut` because with the NVIDIA tier on (FERRIC_CUDA) the newest K/V rows can live only on the
    /// device; `PrefixCache::insert` brings them into the WGSL store before it copies.
    pub(crate) fn remember(&self, tokens: &[u32], cache: &mut ModelCache) {
        if let (ModelCache::Dense(c), Some(pc)) = (cache, &self.prefix_cache) {
            let n = c.pos.min(tokens.len());
            pc.borrow_mut().insert(&self.ctx, &tokens[..n], c);
        }
    }

    /// A token's text and bytes, for `logprobs`.
    pub(crate) fn piece(&self, tok: u32) -> (String, Vec<u8>) {
        let b = self.token_bytes.get(tok as usize).cloned().flatten().unwrap_or_default();
        (String::from_utf8_lossy(&b).into_owned(), b)
    }

    fn lp_entry(&self, row: &[f32], tok: u32, opts: &GenOpts) -> Value {
        let (lp, alts) = genopts::logprobs_of(row, tok, opts.top_logprobs);
        genopts::logprob_entry(&|t| self.piece(t), tok, lp, &alts)
    }

    /// Decode. `temperature` 0 → greedy argmax (deterministic — the default). >0 → top-p sampling
    /// with a **fixed-seed** RNG, so even sampled output is reproducible (on-brand for the moat).
    /// Guided decoding always stays argmax (deterministic structured output). Calls `on_delta` per
    /// newly-decoded fragment. Returns (full_text, prompt_tokens, gen_tokens).
    fn generate(&self, prompt: &[u32], max_tokens: usize, opts: &GenOpts, guide: Option<ferric_agent::guide::Guide>, on_delta: impl FnMut(&str, &[Value])) -> GenOut {
        let ticket = self.energy.begin();
        PEER_GONE.with(|g| g.set(false));
        let mut on_delta = on_delta;
        let mut first = true;
        let mut out = self.generate_inner(prompt, max_tokens, opts, guide, |d: &str, l: &[Value]| {
            if first && !d.is_empty() { first = false; trace::with(|s| s.first_token()); }
            on_delta(d, l)
        });
        if peer_gone() { self.metrics.cancelled.fetch_add(1, std::sync::atomic::Ordering::Relaxed); trace::with(|s| s.cancelled()); }
        out.energy = self.energy.end(ticket, out.gen_tokens);
        self.metrics.record(out.prompt_tokens, out.gen_tokens, &out.energy);
        trace::with(|s| s.generation(&self.name, out.prompt_tokens, out.gen_tokens, out.finish, &out.energy));
        out
    }

    fn generate_inner(&self, prompt: &[u32], max_tokens: usize, opts: &GenOpts, mut guide: Option<ferric_agent::guide::Guide>, mut on_delta: impl FnMut(&str, &[Value])) -> GenOut {
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
                    let (out, drafted, accepted) =
                        self.generate_spec(m, prompt, max_tokens, opts, guide, on_delta);
                    // Only ATTEMPTED drafts are recorded. A step that did not draft has no outcome, and
                    // inventing one would let the gate confirm its own decisions.
                    if drafted > 0 {
                        let mut g = self.spec_gate.lock().expect("spec gate poisoned");
                        for i in 0..drafted { g.observe(prompt.len(), i < accepted); }
                    }
                    return out;
                }
            }
        }
        // Prompt-lookup speculation (opt-in): dense runtime, f32 cache, no constraint, no image.
        if let (Some(k), true, None, Model::Dense(_)) = (lookup_k(), guide.is_none(), opts.image.as_ref(), &self.model) {
            if crate::qwen3_cache_is_f32() { return self.generate_lookup(prompt, max_tokens, opts, k, on_delta); }
        }
        // ⛔ An image prompt never touches the prefix cache: two images of one size are the SAME ids.
        let mm = opts.image.clone();
        let (mut cache, skip) = if mm.is_some() { (self.model.new_cache(), 0) } else { self.seeded_cache(prompt, &opts.lora) };
        let mut delta = 0i64;
        let n_vocab = self.model.n_vocab();
        let mut rng: u64 = opts.rng; // fixed default seed → reproducible sampling; `seed` overrides it
        let mut r#gen: Vec<u32> = Vec::new();
        let mut em = Emitter::new(&opts.stop);
        let (mut lps, mut lp_sent) = (Vec::new(), 0usize);
        let mut finish = "length";
        for step in 0..max_tokens {
            let input: Vec<u32> = if step == 0 { prompt[skip..].to_vec() } else { vec![*r#gen.last().unwrap()] };
            let v = match (&mm, &mut cache) {
                (Some(mm), ModelCache::Dense(c)) if step == 0 => {
                    let (row, d) = self.vision_prefill(prompt, mm, c).unwrap_or_else(|e| panic!("vision prefill: {e}"));
                    delta = d;
                    row
                }
                // Generated token k-1 sits at position len + delta + (k-1) — see `vision_decode`.
                (Some(_), ModelCache::Dense(c)) => self.vision_decode(input[0], prompt.len() as i64 + delta + step as i64 - 1, c),
                _ => self.model.forward_cached_last_host(&input, &mut cache),
            };
            let row = &v[v.len() - n_vocab..];
            let Some(next) = self.select_token(row, &guide, &opts.sampling, prompt, &r#gen, &mut rng) else { finish = "stop"; break };
            if self.eos.contains(&next) { finish = "stop"; break; }
            if let Some(g) = guide.as_mut() { g.commit(next, self.token_bytes[next as usize].as_deref()); }
            if opts.logprobs { lps.push(self.lp_entry(row, next, opts)); }
            r#gen.push(next);
            // Re-detok the whole generation and release only what is safe (multi-byte UTF-8, stop strings).
            let full = if opts.with_specials { self.detok_all(&r#gen) } else { self.detok(&r#gen) };
            if let Some(d) = em.update(&full) { on_delta(&d, &lps[lp_sent..]); lp_sent = lps.len(); }
            if em.hit_stop { finish = "stop"; break; }
            if peer_gone() { finish = "stop"; break; }
        }
        let fed: Vec<u32> = prompt.iter().chain(r#gen.iter()).copied().collect();
        if mm.is_none() { self.remember(&fed, &mut cache); }
        if let Some(d) = em.flush() { on_delta(&d, &lps[lp_sent..]); }
        GenOut { text: em.text, prompt_tokens: prompt.len(), gen_tokens: r#gen.len(), finish, logprobs: lps, ids: r#gen, energy: Value::Null, stop_seq: em.hit_str.clone() }
    }

    /// **Prompt-lookup decoding** (`FERRIC_LOOKUP=k`): each step feeds the pending token plus up to `k`
    /// tokens that followed the current n-gram earlier in the context, in ONE forward, and reads a token from
    /// every row until the model disagrees with a draft. A draft is kept only if it EQUALS what the sampler
    /// picks from that row, with the RNG advanced once per emitted token as the plain loop advances it — so
    /// the output is the plain loop's, up to the rounding of a multi-row forward against single-row decode.
    /// Rejected drafts leave the cache (`Cache::truncate`). Where an answer copies its input it yields many
    /// tokens per forward — measured on Qwen2.5-0.5B: a verbatim repeat 7.3, a code edit 3.8, free verse 1.1,
    /// answers identical to the plain loop in all 6 runs.
    ///
    /// ⚠ NOT YET FASTER, and opt-in for that reason. A verify forward of 2-9 rows runs the PREFILL matmul
    /// path (~34 ms per forward measured) where a batched decode step takes ~10 ms, so the forwards it saves
    /// cost more each. It pays once a few-row forward reads the weights once, as a decode step does.
    fn generate_lookup(&self, prompt: &[u32], max_tokens: usize, opts: &GenOpts, k: usize, mut on_delta: impl FnMut(&str, &[Value])) -> GenOut {
        let (mut cache, skip) = self.seeded_cache(prompt, &opts.lora);
        let n_vocab = self.model.n_vocab();
        let mut rng: u64 = opts.rng;
        let mut r#gen: Vec<u32> = Vec::new();
        let mut em = Emitter::new(&opts.stop);
        let (mut lps, mut lp_sent) = (Vec::new(), 0usize);
        let mut finish = "length";
        let (mut steps, mut drafted, mut kept) = (1usize, 0usize, 0usize);
        let (t0, mut t_fwd) = (std::time::Instant::now(), 0f64);
        let v = pollster::block_on(self.model.forward_cached_last(&prompt[skip..], &mut cache).to_vec());
        let mut rows: Vec<Vec<f32>> = vec![v[v.len() - n_vocab..].to_vec()];
        let mut draft: Vec<u32> = Vec::new();
        let mut base = match &cache { ModelCache::Dense(c) => c.pos, _ => unreachable!() };
        'run: loop {
            // Emit from each row in turn while the drafts hold. Row i predicts the token after feed[..=i].
            let mut held = 0usize;
            for (i, row) in rows.iter().enumerate() {
                let Some(next) = self.select_token(row, &None, &opts.sampling, prompt, &r#gen, &mut rng) else { finish = "stop"; break 'run };
                if self.eos.contains(&next) { finish = "stop"; break 'run; }
                if opts.logprobs { lps.push(self.lp_entry(row, next, opts)); }
                r#gen.push(next);
                let full = if opts.with_specials { self.detok_all(&r#gen) } else { self.detok(&r#gen) };
                if let Some(d) = em.update(&full) { on_delta(&d, &lps[lp_sent..]); lp_sent = lps.len(); }
                if em.hit_stop || peer_gone() { finish = "stop"; break 'run; }
                if r#gen.len() >= max_tokens { break 'run; }
                if i < draft.len() && next == draft[i] { held += 1; continue; }
                break;
            }
            // Keep what the model confirmed: the pending token and the drafts it agreed with.
            if !draft.is_empty() {
                kept += held;
                if let ModelCache::Dense(c) = &mut cache { c.truncate(base + 1 + held); }
            }
            let ctx: Vec<u32> = prompt.iter().chain(r#gen.iter()).copied().collect();
            draft = lookup_draft(&ctx, k);
            drafted += draft.len();
            let feed: Vec<u32> = std::iter::once(*r#gen.last().unwrap()).chain(draft.iter().copied()).collect();
            base = match &cache { ModelCache::Dense(c) => c.pos, _ => unreachable!() };
            let tf = std::time::Instant::now();
            let v = pollster::block_on(self.model.forward_cached(&feed, &mut cache).to_vec());
            t_fwd += tf.elapsed().as_secs_f64();
            steps += 1;
            rows = v.chunks(n_vocab).map(|r| r.to_vec()).collect();
        }
        // Leave the cache holding exactly what it verified — the prompt and every emitted token but the
        // last (never fed) — so the prompt cache is offered no rejected draft.
        if let ModelCache::Dense(c) = &mut cache {
            let verified = prompt.len() + r#gen.len().saturating_sub(1);
            if c.pos > verified { c.truncate(verified); }
        }
        if std::env::var("FERRIC_LOOKUP_TRACE").is_ok() {
            eprintln!("lookup: {} tokens in {steps} forwards ({:.2} per forward); drafted {drafted}, kept {kept}; {:.3} s total, {:.3} s in verify forwards",
                      r#gen.len(), r#gen.len() as f64 / steps as f64, t0.elapsed().as_secs_f64(), t_fwd);
        }
        let fed: Vec<u32> = prompt.iter().chain(r#gen.iter()).copied().collect();
        self.remember(&fed, &mut cache);
        if let Some(d) = em.flush() { on_delta(&d, &lps[lp_sent..]); }
        GenOut { text: em.text, prompt_tokens: prompt.len(), gen_tokens: r#gen.len(), finish, logprobs: lps, ids: r#gen, energy: Value::Null, stop_seq: em.hit_str.clone() }
    }

    /// Pick the next token from a row of TRUE model logits: guided decoding masks illegal tokens to
    /// -inf (EOS legal only once the value is complete), then `temperature` is honored over the
    /// legal set — temp 0 stays argmax (deterministic). Returns `None` when the guide leaves no
    /// legal continuation (stop cleanly). Most tokens reject on their first byte, so the scan is cheap.
    fn select_token(&self, row: &[f32], guide: &Option<ferric_agent::guide::Guide>, s: &genopts::Sampling, prompt: &[u32], generated: &[u32], rng: &mut u64) -> Option<u32> {
        let n_vocab = row.len();
        // A spent reasoning budget forces its message and end marker, one token per step (every other logit
        // is -inf in llama.cpp's sampler, so sampling can only pick it).
        if let Some(b) = &s.reasoning_budget {
            let complete = |t: u32| self.token_bytes.get(t as usize).and_then(|b| b.as_deref()).is_none_or(genopts::utf8_is_complete);
            if let Some(t) = b.forced(&prompt[prompt.len().saturating_sub(24)..], generated, complete) { return Some(t); }
        }
        if let Some(g) = guide.as_ref() {
            let ok = self.allowed(g);
            let mut masked = vec![f32::NEG_INFINITY; n_vocab];
            let mut any = false;
            for i in 0..n_vocab { if ok[i] { masked[i] = row[i]; any = true; } }
            if !any { return None; } // no legal continuation (shouldn't happen for a valid schema)
            Some(genopts::sample(&masked, s, prompt, generated, rng))
        } else {
            Some(genopts::sample(row, s, prompt, generated, rng))
        }
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
    fn generate_spec(&self, m: &Qwen35, prompt: &[u32], max_tokens: usize, opts: &GenOpts, mut guide: Option<ferric_agent::guide::Guide>, mut on_delta: impl FnMut(&str, &[Value])) -> (GenOut, u32, u32) {
        // Draft accounting, returned so the caller can feed `SpecGate`. The doc above this function
        // asserts "~80% acceptance"; nothing measured it until now, and a gate that decides on an
        // assumed rate is a heuristic wearing a derivation's clothes.
        let (mut drafted, mut accepted) = (0u32, 0u32);
        let n_vocab = self.model.n_vocab();
        let argmax = |r: &[f32]| (0..n_vocab).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap() as u32;
        let mut rng: u64 = opts.rng;
        let mut r#gen: Vec<u32> = Vec::new();
        let mut em = Emitter::new(&opts.stop);
        let (mut lps, mut lp_sent): (Vec<Value>, usize) = (Vec::new(), 0);
        let mut finish = "length";
        let sm = &opts.sampling;
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
        // Commit one token: advance the guide, record its logprob, release what is safe to stream.
        // Evaluates to `true` when a stop string just completed — the caller then stops exactly as it
        // would on a stop token, rolling back any cache entry this step's verify left unconfirmed.
        macro_rules! commit {
            ($tok:expr_2021, $row:expr_2021) => {{
                if let Some(g) = guide.as_mut() { g.commit($tok, self.token_bytes[$tok as usize].as_deref()); }
                if opts.logprobs { lps.push(self.lp_entry($row, $tok, opts)); }
                r#gen.push($tok);
                let full = if opts.with_specials { self.detok_all(&r#gen) } else { self.detok(&r#gen) };
                if let Some(d) = em.update(&full) { on_delta(&d, &lps[lp_sent..]); lp_sent = lps.len(); }
                if em.hit_stop || peer_gone() { finish = "stop"; }
                em.hit_stop || peer_gone()
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
        let row0 = &v[v.len() - n_vocab..];
        let first = self.select_token(row0, &guide, sm, prompt, &r#gen, &mut rng);
        let Some(pend0) = first.filter(|t| !self.eos.contains(t)) else {
            save_slot!();
            return (GenOut { text: String::new(), prompt_tokens: prompt.len(), gen_tokens: 0, finish: "stop", logprobs: lps, ids: Vec::new(), energy: Value::Null, stop_seq: None }, drafted, accepted)
        };
        if commit!(pend0, row0) || max_tokens <= 1 {
            save_slot!();
            if let Some(d) = em.flush() { on_delta(&d, &lps[lp_sent..]); }
            let fin = if em.hit_stop { "stop" } else { "length" };
            return (GenOut { text: em.text, prompt_tokens: prompt.len(), gen_tokens: r#gen.len(), finish: fin, logprobs: lps, ids: r#gen, energy: Value::Null, stop_seq: em.hit_str.clone() }, drafted, accepted)
        }
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
                let t1 = self.select_token(&v[0..n_vocab], &guide, sm, prompt, &r#gen, &mut rng);
                let Some(t1) = t1.filter(|t| !self.eos.contains(t)) else { finish = "stop"; cache = snap; break; };
                if commit!(t1, &v[0..n_vocab]) { cache = snap; break; }
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
                let t2 = self.select_token(&v[n_vocab..2 * n_vocab], &guide, sm, prompt, &r#gen, &mut rng);
                let Some(t2) = t2.filter(|t| !self.eos.contains(t)) else { finish = "stop"; cache = snap; break; };
                if commit!(t2, &v[n_vocab..2 * n_vocab]) { cache = snap; break; }
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
                let pend = self.select_token(&v[2 * n_vocab..3 * n_vocab], &guide, sm, prompt, &r#gen, &mut rng);
                let Some(pend) = pend.filter(|t| !self.eos.contains(t)) else { finish = "stop"; break };
                if commit!(pend, &v[2 * n_vocab..3 * n_vocab]) { break; }
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
            let truth = self.select_token(&v[0..n_vocab], &guide, sm, prompt, &r#gen, &mut rng);
            let Some(truth) = truth.filter(|t| !self.eos.contains(t)) else {
                finish = "stop";
                cache = snap; // this forward's entries include the unverified draft — discard
                break;
            };
            if commit!(truth, &v[0..n_vocab]) { cache = snap; break; }
            if truth == d && r#gen.len() < max_tokens {
                fed.extend_from_slice(&toks); // everything this forward fed is now known-valid
                // Accepted: the draft's own logits row is valid too — take the next token from it.
                let pend = self.select_token(&v[n_vocab..2 * n_vocab], &guide, sm, prompt, &r#gen, &mut rng);
                let Some(pend) = pend.filter(|t| !self.eos.contains(t)) else { finish = "stop"; break };
                if commit!(pend, &v[n_vocab..2 * n_vocab]) { break; }
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
        if let Some(d) = em.flush() { on_delta(&d, &lps[lp_sent..]); }
        (GenOut { text: em.text, prompt_tokens: prompt.len(), gen_tokens: r#gen.len(), finish, logprobs: lps, ids: r#gen, energy: Value::Null, stop_seq: em.hit_str.clone() }, drafted, accepted)
    }
}

/// A message as a chat template expects it: `content` as text, and tool-call `arguments` as an object
/// (OpenAI clients send a JSON string; vLLM parses it the same way before templating). Adds NO key the
/// message did not have.
///
/// ⛔ The first version wrote `m["tool_calls"].as_array_mut()`, and serde_json's `IndexMut` INSERTS a
/// missing key as `null`: every message gained `"tool_calls": null`, Llama-3.2's template read
/// `'tool_calls' in message` as true, and every conversation failed with "cannot calculate length of
/// value of type none". The template-vs-HF harness could not see it (it bypasses this step); comparing
/// the served token ids with HF's `apply_chat_template` did.
fn prepare_for_template(m: &Value) -> Value {
    let mut m = m.clone();
    // ⛔ A message carrying an image keeps its PARTS: the vision template places `<|image_pad|>` where
    // it finds `{"type": "image"}`. Flattening it (the first vision path did) read the part list as text,
    // failed on the image part, and `unwrap_or_default` left "" — the image AND the question gone, and
    // the template placed 0 image tokens. Found by the live check against the authors' prompt ids.
    let has_image = m["content"].as_array().is_some_and(|a| a.iter().any(|p| p["type"] == "image"));
    if m.get("content").is_some() && !has_image {
        m["content"] = json!(genopts::content_text(&m["content"]).unwrap_or_default());
    }
    if let Some(calls) = m.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
        for c in calls {
            if let Some(f) = c.get_mut("function") {
                if let Some(a) = f.get("arguments").and_then(|a| a.as_str()).and_then(|s| serde_json::from_str::<Value>(s).ok()) {
                    f["arguments"] = a;
                }
            }
        }
    }
    m
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
    let dir = format!("{}/{}", models::hub_dir().display(), repo.replace('/', "_"));
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

/// `FERRIC_LOOKUP=k`: draft up to `k` tokens by prompt lookup (off when unset or 0).
fn lookup_k() -> Option<usize> { std::env::var("FERRIC_LOOKUP").ok().and_then(|v| v.parse().ok()).filter(|&k: &usize| k > 0) }

/// Whether dense caches are f32 — prompt lookup truncates rejected drafts, which a block-quantized store cannot.
fn qwen3_cache_is_f32() -> bool { std::env::var("FERRIC_KVQ").map_or(true, |v| v.is_empty() || v == "off" || v == "f32") }

/// The drafts prompt lookup proposes: the tokens that followed the most recent earlier occurrence of the
/// context's last `n` tokens, longest `n` (4 down to 2) first, at most `k` of them. Empty = no match; the
/// step is then a plain one-token decode.
fn lookup_draft(ctx: &[u32], k: usize) -> Vec<u32> {
    for n in (2..=4usize).rev() {
        if ctx.len() <= n { continue; }
        let pat = &ctx[ctx.len() - n..];
        for start in (0..ctx.len() - n).rev() {
            if &ctx[start..start + n] == pat {
                let from = start + n;
                let to = (from + k).min(ctx.len());
                if from < to { return ctx[from..to].to_vec(); }
            }
        }
    }
    Vec::new()
}

/// A file's size, or a directory's (a PEFT adapter is a directory).
fn walk_size(p: &std::path::Path) -> u64 {
    if p.is_dir() { std::fs::read_dir(p).map(|r| r.flatten().map(|e| walk_size(&e.path())).sum()).unwrap_or(0) }
    else { std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) }
}

/// Physical memory in bytes: `hw.memsize` on macOS, `MemTotal` on Linux. `None` elsewhere (no budget).
fn physical_memory() -> Option<u64> {
    if cfg!(target_os = "macos") {
        let o = std::process::Command::new("/usr/sbin/sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        return String::from_utf8_lossy(&o.stdout).trim().parse().ok();
    }
    let m = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = m.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The context the server gives a model: its trained length, or less when `--ctx-size` / FERRIC_CTX_SIZE
/// asks (llama.cpp's `-c`). Never more: past its trained length a model needs a rope scaling it was not
/// given, so a larger value is capped, said once at load.
pub(crate) fn ctx_cap(trained: usize) -> usize {
    match std::env::var("FERRIC_CTX_SIZE").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&c| c > 0) {
        Some(c) if c <= trained => c,
        Some(c) => { eprintln!("ferric-serve: --ctx-size {c} is past this model's trained context {trained}; using {trained}"); trained }
        None => trained,
    }
}

/// CLI entry point. Lives in the library so the binary is a two-line shim and everything the
/// server does stays reachable from tests — see the crate docs.
pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    // The model argument is optional: without one the server answers for every model in its model
    // directory and loads each on its first request, as `ollama serve` does.
    let path: Option<String> = args.get(1).filter(|a| !a.starts_with("--")).cloned();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("usage: ferric-serve [model.gguf | owner/repo[:file.gguf]] [--embed <encoder.gguf>] [--rerank <cross-encoder.gguf>] \
                   [--asr <parakeet.gguf>] [--api-key K] [--host H] [--port N] [--name S] [--max-models N] [--keep-alive 5m]\n\
                   Any other GGUF in FERRIC_MODELS (default ~/.cache/ferric/hub) loads when a request names it.");
        std::process::exit(0);
    }
    let mut port = 8080u16;
    // 127.0.0.1 by default: a model server is not exposed to the network unless someone says so.
    let mut host = "127.0.0.1".to_string();
    // FERRIC_API_KEY or --api-key; unset = no auth (the default for a server bound to localhost).
    let mut api_key: Option<String> = std::env::var("FERRIC_API_KEY").ok().filter(|k| !k.is_empty());
    // Default: the file's stem, the way Ollama and LM Studio name a model — "ferric" said nothing about
    // which model this is, and it collided with nothing only because there was never a second one.
    let mut name = path.as_deref().map(|p| models::default_name(std::path::Path::new(p))).unwrap_or_else(|| "ferric".to_string());
    let mut mcp_cmds: Vec<(String, String)> = Vec::new();
    // How many sequences may share one decode step. 8 is a starting point, not a measured optimum:
    // occupancy is what decides the payoff and only the deployment knows its arrival rate.
    let mut max_batch = std::env::var("FERRIC_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(8usize);
    // Ollama's defaults: three models resident, each kept five minutes after its last request.
    let mut max_models = std::env::var("FERRIC_MAX_LOADED_MODELS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(3usize);
    let mut keep_alive = std::env::var("FERRIC_KEEP_ALIVE").ok().unwrap_or_else(|| "5m".to_string());
    let mut loras: Vec<(String, String)> = Vec::new();
    let mut otlp: Option<String> = None;
    let mut i = if path.is_some() { 2 } else { 1 };
    while i < args.len() {
        match args[i].as_str() {
            "--port" => { port = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(port); i += 2; }
            "--host" => { host = args.get(i + 1).cloned().unwrap_or(host); i += 2; }
            "--api-key" => { api_key = args.get(i + 1).cloned(); i += 2; }
            // FIXME: Audit that the environment access only happens in single-threaded code.
            "--embed" => { if let Some(p) = args.get(i + 1) { unsafe { std::env::set_var("FERRIC_EMBED_MODEL", p) }; } i += 2; }
            "--rerank" => { if let Some(p) = args.get(i + 1) { unsafe { std::env::set_var("FERRIC_RERANK_MODEL", p) }; } i += 2; }
            "--asr" => { if let Some(p) = args.get(i + 1) { unsafe { std::env::set_var("FERRIC_ASR_MODEL", p) }; } i += 2; }
            "--name" => { name = args.get(i + 1).cloned().unwrap_or(name); i += 2; }
            "--max-models" => { max_models = args.get(i + 1).and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(max_models); i += 2; }
            "--keep-alive" => { keep_alive = args.get(i + 1).cloned().unwrap_or(keep_alive); i += 2; }
            // `--reasoning-budget N` (-1 unlimited, 0 = end thinking at once) and the message forced before the
            // end marker when it runs out — llama-server's flags; a request's `thinking_budget_tokens` overrides.
            // FIXME: Audit that the environment access only happens in single-threaded code.
            // `--ctx-size N` (llama.cpp's -c): every model's context, at most its trained length.
            // FIXME: Audit that the environment access only happens in single-threaded code.
            "--ctx-size" | "-c" => { if let Some(v) = args.get(i + 1) { unsafe { std::env::set_var("FERRIC_CTX_SIZE", v) }; } i += 2; }
            "--reasoning-budget" => { if let Some(v) = args.get(i + 1) { unsafe { std::env::set_var("FERRIC_REASONING_BUDGET", v) }; } i += 2; }
            "--reasoning-budget-message" => { if let Some(v) = args.get(i + 1) { unsafe { std::env::set_var("FERRIC_REASONING_BUDGET_MESSAGE", v) }; } i += 2; }
            // `--otlp http://collector:4318`: OpenTelemetry traces (else the OTEL_EXPORTER_OTLP_* variables).
            "--otlp" => { otlp = args.get(i + 1).cloned(); i += 2; }
            // `--lora name=path` (repeatable): a PEFT adapter directory or a llama.cpp GGUF adapter for the
            // model on the command line; `--lora path` names it by its file or directory name.
            "--lora" => {
                if let Some(v) = args.get(i + 1) {
                    let (n, p) = v.split_once('=').map(|(n, p)| (n.to_string(), p.to_string()))
                        .unwrap_or_else(|| (models::default_name(std::path::Path::new(v)), v.clone()));
                    loras.push((n, p));
                }
                i += 2;
            }
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
    match trace::config(&|k| std::env::var(k).ok(), otlp.as_deref()) {
        Ok(cfg) => if let Some(what) = trace::init(cfg) { eprintln!("ferric-serve: {what}"); },
        Err(e) => { eprintln!("ferric-serve: {e}"); std::process::exit(1); }
    }
    let keep_alive = match batch::keep_alive_of(&json!({"keep_alive": keep_alive})) {
        Ok(Some(k)) => k,
        _ => { eprintln!("ferric-serve: --keep-alive {keep_alive:?}: expected a duration like 5m, 1h or 30s (or -1 for until evicted)"); std::process::exit(1); }
    };
    // Weights dominate what a model holds, so model files are budgeted against 75% of physical memory,
    // leaving the rest for KV caches and everything else on the machine. FERRIC_MAX_MEMORY (GiB) overrides.
    let max_bytes = std::env::var("FERRIC_MAX_MEMORY").ok().and_then(|s| s.parse::<f64>().ok()).map(|g| (g * 1073741824.0) as u64)
        .or_else(|| physical_memory().map(|b| b / 4 * 3)).unwrap_or(u64::MAX);
    let shared = Shared::new();
    let _ = shared.metrics.started.set(std::time::Instant::now());
    let mut hub = models::Hub::new(shared.clone());
    let mut initial = Vec::new();
    if path.is_none() && !loras.is_empty() { eprintln!("ferric-serve: --lora adapts the model on the command line; name one"); std::process::exit(1); }
    if let Some(path) = &path {
        let resolved = resolve_model(path);
        eprintln!("ferric-serve: loading {resolved} …");
        let key = std::fs::canonicalize(&resolved).map(|p| p.to_string_lossy().into_owned()).unwrap_or(resolved.clone());
        hub.name(&key, name.clone());
        let bytes = models::model_bytes(std::path::Path::new(&resolved));
        // A model named on the command line that will not load stops the server: a server that will not
        // boot is a bug report; one that boots without the model it was given answers for something else.
        let mut eng = batch::Source::load(&mut hub, &key).unwrap_or_else(|e| { eprintln!("ferric-serve: {e}"); std::process::exit(1) });
        if let Err(e) = eng.load_adapters(&loras) { eprintln!("ferric-serve: {e}"); std::process::exit(1); }
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
            let out = eng.generate(&eng.chat_ids(&msgs).expect("chat ids"), 16, &GenOpts::default(), None, |d, _| eprint!("{d}"));
            eprintln!("\nferric-serve: --once ok ({} prompt + {} gen tokens, {}): {:?}", out.prompt_tokens, out.gen_tokens, out.finish, out.text);
            return;
        }
        eprintln!("ferric-serve: {} ({} layers, vocab {}, context {}) on {:?}",
            name, eng.model.n_layer(), eng.model.n_vocab(), eng.n_ctx, eng.ctx.backend);
        eprintln!("ferric-serve: continuous batching {}",
            if eng.batchable() { format!("ON, max_batch {max_batch}") } else { "OFF (serial) — this model has no solo-equivalent batched decode, or it was disabled".to_string() });
        initial.push(batch::Loaded { key, m: eng, bytes });
    }
    let mcps = std::cell::RefCell::new(mcps);
    let any_mcp_tools = !mcps.borrow().openai_tools().is_empty();
    eprintln!("ferric-serve: http://{host}:{port}/v1 — other models load on request from {} (up to {max_models} resident{}, {}){}",
        models::hub_dir().display(),
        if max_bytes == u64::MAX { String::new() } else { format!(", {:.0} GiB of model files", max_bytes as f64 / 1073741824.0) },
        match keep_alive { Some(d) => format!("kept {} s after last use", d.as_secs()), None => "kept until evicted".to_string() },
        if mcps.borrow().0.is_empty() { String::new() } else { format!(" · {} MCP tools", mcps.borrow().openai_tools().len()) });
    let listener = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|e| panic!("bind {host}:{port}: {e}"));
    // The Batch API sends each line to this server's own endpoint, so a batch runs on the live code path.
    batchapi::init(&host, port, api_key.clone(), max_batch);
    // The batch loop owns the engine on this thread; anything it declines (guided decoding, the tool
    // loop, embeddings, unknown paths) goes to the untouched serial handler below.
    let fallback = !initial.is_empty();
    batch::serve_loop(hub, initial, listener, batch::ServeOpts { max_batch, any_mcp_tools, api_key, max_models, max_bytes, keep_alive, fallback },
        |eng, method, path, body, headers, s| {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(eng, &mcps, method, path, body, headers, s)));
            match r {
                Ok(handled) => handled,
                Err(_) => { eprintln!("ferric-serve: handler panicked (recovered)"); true }
            }
        });
}

/// The serial handler: every endpoint the batch loop does not own. Returns whether it recognised the
/// request (`false` → the caller writes a 404).
fn handle(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, method: &str, path: &str, body: &[u8],
          headers: &[(String, String)], stream: &mut TcpStream) -> bool {
    match (method, path) {
        // `/health`, `/v1/models`, `/metrics`, `/api/tags`, `/api/ps` describe the whole server, not one
        // model, and are answered by the pool before a request reaches a model (batch.rs).
        ("POST", "/v1/chat/completions") => chat(eng, mcps, stream, body),
        ("POST", "/v1/completions") => completions(eng, stream, body),
        ("POST", "/v1/embeddings") => embeddings(eng, stream, body),
        ("POST", "/v1/rerank") | ("POST", "/rerank") => rerank(eng, stream, body),
        ("POST", "/v1/audio/transcriptions") => audio::transcriptions(eng, body, headers, stream),
        ("POST", "/tokenize") | ("POST", "/v1/tokenize") => tokenize(eng, body, stream),
        ("POST", "/detokenize") | ("POST", "/v1/detokenize") => detokenize(eng, body, stream),
        ("POST", "/v1/messages") => dialects::messages(eng, mcps, body, stream),
        ("POST", "/v1/messages/count_tokens") => dialects::count_tokens(eng, body, stream),
        ("POST", "/v1/responses") => dialects::responses(eng, mcps, &eng.responses, body, stream),
        ("GET", p) if p.starts_with("/v1/responses/") => dialects::retrieve(&eng.responses, &p["/v1/responses/".len()..], stream),
        // The Ollama dialect, or unrecognised — then the caller answers, so there is one 404 writer.
        _ => return ollama::handle(eng, mcps, method, path, body, stream),
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
    let Some(rr) = eng.aux.reranker.as_ref() else {
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

/// Embed a batch of texts with whichever model the request means: the dedicated embedder when one is
/// loaded (unless the request names the chat model), else the chat model's own pooled hidden state.
/// Returns (vectors, prompt tokens, model name).
pub(crate) fn embed_texts(eng: &Engine, model: Option<&str>, inputs: &[String], truncate: bool)
    -> Result<(Vec<Vec<f32>>, usize, String), String>
{
    match &eng.aux.embedder {
        Some((name, e, _)) if model != Some(eng.name.as_str()) => {
            let (mut out, mut total) = (Vec::with_capacity(inputs.len()), 0usize);
            for t in inputs {
                let (v, n) = pollster::block_on(e.embed(t, truncate))?;
                out.push(v); total += n;
            }
            Ok((out, total, name.clone()))
        }
        _ => {
            let (mut out, mut total) = (Vec::with_capacity(inputs.len()), 0usize);
            for t in inputs { total += eng.enc(t, true).len(); out.push(eng.embed(t)?); }
            Ok((out, total, eng.name.clone()))
        }
    }
}

/// Little-endian float32 bytes, base64 — what OpenAI's `encoding_format: "base64"` returns (the official
/// Python client asks for it by default and decodes it itself).
fn b64_f32(v: &[f32]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// OpenAI-compatible `/v1/embeddings`: `input` is a string or array of strings → L2-normalized vectors.
fn embeddings(eng: &Engine, stream: &mut TcpStream, body: &[u8]) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad_request(stream, &format!("bad json: {e}")) };
    let inputs: Vec<String> = match &req["input"] {
        Value::String(s) => vec![s.clone()],
        // Error (don't silently drop) on a non-string element — dropping would misalign every `index`.
        Value::Array(a) => {
            let mut v = Vec::with_capacity(a.len());
            for x in a { match x.as_str() { Some(s) => v.push(s.to_string()), None => return bad_request(stream, "`input` array must contain only strings") } }
            v
        }
        _ => return bad_request(stream, "`input` must be a string or array of strings"),
    };
    let b64 = match req["encoding_format"].as_str() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(f) => return bad_request(stream, &format!("`encoding_format` {f:?}: use \"float\" or \"base64\"")),
    };
    // A checkpoint whose pooling this build cannot honour is a 400, not a 200 carrying a vector of the
    // right length and no meaning. The client can act on an error; it cannot act on a cosine score
    // that looks ordinary.
    let (vecs, total, name) = match embed_texts(eng, req["model"].as_str(), &inputs, false) {
        Ok(r) => r, Err(m) => return bad_request(stream, &m),
    };
    if let Some(d) = req["dimensions"].as_u64() {
        // Matryoshka truncation is model-specific (nomic layer-norms before cutting); truncating and
        // renormalising the OpenAI way would return a vector the model was not trained to produce.
        if vecs.first().is_some_and(|v| v.len() != d as usize) {
            return bad_request(stream, &format!("`dimensions` {d}: this model returns {} and does not truncate", vecs[0].len()));
        }
    }
    let data: Vec<Value> = vecs.iter().enumerate().map(|(i, e)| json!({"object": "embedding", "index": i,
        "embedding": if b64 { json!(b64_f32(e)) } else { json!(e) }})).collect();
    write_json(stream, 200, &json!({
        "object": "list", "data": data, "model": name,
        "usage": {"prompt_tokens": total, "total_tokens": total}
    }));
}

/// `/tokenize` in both shapes clients use: llama-server's `{"content", "add_special", "with_pieces"}` and
/// vLLM's `{"prompt" | "messages", "add_special_tokens"}` — a chat is tokenised through the model's own
/// template, exactly as a chat request would be.
fn tokenize(eng: &Engine, body: &[u8], stream: &mut TcpStream) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad_request(stream, &format!("bad json: {e}")) };
    let ids = if let Some(msgs) = req["messages"].as_array() {
        match eng.chat_ids(msgs) { Ok(i) => i, Err(e) => return bad_request(stream, &e) }
    } else {
        let Some(text) = req["content"].as_str().or_else(|| req["prompt"].as_str()) else {
            return bad_request(stream, "give `content` (llama-server), `prompt` or `messages` (vLLM)")
        };
        let special = req["add_special"].as_bool().or_else(|| req["add_special_tokens"].as_bool()).unwrap_or(false);
        let mut ids = if special { Vec::new() } else { eng.encode_special(text) };
        if special { ids = eng.encode_prompt(text); }
        ids
    };
    let mut v = json!({"tokens": ids, "count": ids.len(), "max_model_len": eng.n_ctx});
    if req["with_pieces"].as_bool() == Some(true) {
        v["tokens"] = json!(ids.iter().map(|&i| json!({"id": i, "piece": eng.detok_all(&[i])})).collect::<Vec<_>>());
    }
    write_json(stream, 200, &v);
}

fn detokenize(eng: &Engine, body: &[u8], stream: &mut TcpStream) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad_request(stream, &format!("bad json: {e}")) };
    let Some(ids) = req["tokens"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|i| i as u32)).collect::<Vec<u32>>()) else {
        return bad_request(stream, "`tokens` must be an array of ids")
    };
    if let Some(&bad) = ids.iter().find(|&&i| i as usize >= eng.tokens.len()) { return bad_request(stream, &format!("token id {bad} is outside the vocabulary")); }
    let text = eng.detok_all(&ids);
    write_json(stream, 200, &json!({"content": text, "prompt": text}));
}

/// Prometheus text format. `ferric_energy_joules_total` is the attributed joules of every request the meter
/// could attribute — the counter no serving peer exports. Server-wide: the counters are shared by every
/// loaded model, and `ferric_model_info` / `ferric_context_length` carry one line per loaded model.
pub(crate) fn metrics_body(m: &Metrics, meter: bool, models: &[(String, String, usize)]) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let (j, jn) = m.joules.lock().map(|g| *g).unwrap_or((0.0, 0));
    let up = m.started.get().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
    let esc = |x: &str| x.replace('\\', "\\\\").replace('"', "\\\"");
    let mut body = format!(
"# HELP ferric_requests_total Generations completed.\n# TYPE ferric_requests_total counter\nferric_requests_total {}\n\
# HELP ferric_prompt_tokens_total Prompt tokens processed.\n# TYPE ferric_prompt_tokens_total counter\nferric_prompt_tokens_total {}\n\
# HELP ferric_generation_tokens_total Tokens generated.\n# TYPE ferric_generation_tokens_total counter\nferric_generation_tokens_total {}\n\
# HELP ferric_requests_cancelled_total Generations stopped because the client disconnected.\n# TYPE ferric_requests_cancelled_total counter\nferric_requests_cancelled_total {}\n\
# HELP ferric_energy_joules_total Joules attributed to requests (accelerator rails, idle-subtracted; derived).\n# TYPE ferric_energy_joules_total counter\nferric_energy_joules_total {:.3}\n\
# HELP ferric_energy_attributed_requests_total Requests whose joules the meter could attribute.\n# TYPE ferric_energy_attributed_requests_total counter\nferric_energy_attributed_requests_total {}\n\
# HELP ferric_energy_meter_available Whether a power meter is running.\n# TYPE ferric_energy_meter_available gauge\nferric_energy_meter_available {}\n\
# HELP ferric_uptime_seconds Seconds since the server started.\n# TYPE ferric_uptime_seconds gauge\nferric_uptime_seconds {:.1}\n\
# HELP ferric_models_loaded Chat models resident now.\n# TYPE ferric_models_loaded gauge\nferric_models_loaded {}\n\
# HELP ferric_model_info A loaded chat model.\n# TYPE ferric_model_info gauge\n",
        m.requests.load(Relaxed), m.prompt_tokens.load(Relaxed), m.gen_tokens.load(Relaxed), m.cancelled.load(Relaxed),
        j, jn, meter as u8, up, models.len());
    for (name, arch, _) in models { body.push_str(&format!("ferric_model_info{{model=\"{}\",arch=\"{}\"}} 1\n", esc(name), esc(arch))); }
    body.push_str("# HELP ferric_context_length A loaded model's context.\n# TYPE ferric_context_length gauge\n");
    for (name, _, n) in models { body.push_str(&format!("ferric_context_length{{model=\"{}\"}} {n}\n", esc(name))); }
    body
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

fn bad_request(stream: &mut TcpStream, m: &str) {
    write_json(stream, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}}))
}

/// The `logprobs` field of a choice: chat format, or completions' parallel arrays.
fn logprobs_field(chat: bool, lps: &[Value]) -> Value {
    if chat { return json!({"content": lps}); }
    let mut offset = 0usize;
    let mut offs = Vec::with_capacity(lps.len());
    for e in lps { offs.push(offset); offset += e["token"].as_str().map(str::len).unwrap_or(0); }
    json!({
        "tokens": lps.iter().map(|e| e["token"].clone()).collect::<Vec<_>>(),
        "token_logprobs": lps.iter().map(|e| e["logprob"].clone()).collect::<Vec<_>>(),
        "top_logprobs": lps.iter().map(|e| {
            let mut m = serde_json::Map::new();
            for a in e["top_logprobs"].as_array().into_iter().flatten() {
                if let Some(t) = a["token"].as_str() { m.insert(t.to_string(), a["logprob"].clone()); }
            }
            Value::Object(m)
        }).collect::<Vec<_>>(),
        "text_offset": offs,
    })
}

/// One chat turn, whatever API it arrived through.
pub(crate) struct ChatResult {
    pub text: String,
    /// The model's reasoning, apart from the answer (`reasoning_content`); empty for non-thinking models.
    pub reasoning: String,
    /// OpenAI-shaped `tool_calls` (`function.arguments` a JSON string); empty unless the model called one.
    pub tool_calls: Vec<Value>,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub finish: &'static str,
    pub logprobs: Vec<Value>,
    /// Joules for the whole turn (every tool-loop round summed).
    pub energy: Value,
    pub stop_seq: Option<String>,
}

/// Several generations' energy as one: joules summed, the rest from the last.
fn sum_energy(parts: &[Value]) -> Value {
    let Some(last) = parts.last() else { return Value::Null };
    if parts.len() == 1 { return last.clone(); }
    let mut v = last.clone();
    for k in ["joules", "window_joules", "seconds"] {
        let xs: Vec<f64> = parts.iter().filter_map(|p| p[k].as_f64()).collect();
        v[k] = if xs.len() == parts.len() { json!((xs.iter().sum::<f64>() * 1000.0).round() / 1000.0) } else { Value::Null };
    }
    v["rounds"] = json!(parts.len());
    v
}

/// What the chat template is rendered with besides the messages: `chat_template_kwargs`, plus OpenAI's
/// `reasoning_effort` (passed to templates that read it — Bonsai 2, gpt-oss — and, when the request did
/// not set `enable_thinking`, "none"/"minimal" turn thinking off in templates that read that instead).
/// One reader for the serial and batched paths, so the same request renders the same prompt on both.
pub(crate) fn template_kwargs(req: &Value) -> serde_json::Map<String, Value> {
    let mut k = req["chat_template_kwargs"].as_object().cloned().unwrap_or_default();
    if let Some(e) = req["reasoning_effort"].as_str() {
        k.entry("reasoning_effort").or_insert_with(|| json!(e));
        k.entry("enable_thinking").or_insert_with(|| json!(e != "none" && e != "minimal"));
    }
    k
}

/// **The chat core both API dialects share** (OpenAI `/v1/chat/completions`, Ollama `/api/chat`), so they
/// cannot disagree about what a conversation means. `req` is OpenAI-shaped. `on_delta` receives the
/// streamed text and its logprob entries; it is not called on the tool path, whose answer is only known
/// once the model has finished (a tool call is parsed from the whole output).
pub(crate) fn run_chat(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, req: &Value,
                       mut on_delta: impl FnMut(&str, &[Value], bool)) -> Result<ChatResult, String> {
    let empty = vec![];
    let mut messages: Vec<Value> = req["messages"].as_array().unwrap_or(&empty).clone();
    // `developer` is OpenAI's newer name for the system role (Cursor sends it).
    for m in messages.iter_mut() { if m["role"] == "developer" { m["role"] = json!("system"); } }
    let mut opts = eng.gen_opts(req, true)?;
    opts.with_specials = eng.reasoning_markers.is_some();
    // A thinking model's reasoning is split from its answer; the block may already be open in the prompt.
    let splitter = |prompt: &[u32]| eng.reasoning_split(prompt);
    // Advertised tools = caller's + every connected MCP server's.
    let mut tools = req["tools"].as_array().cloned().unwrap_or_default();
    tools.extend(mcps.borrow().openai_tools());

    let kwargs = template_kwargs(req);
    let via_template = eng.template_handles_tools();
    let tools_arg = (via_template && !tools.is_empty()).then_some(tools.as_slice());
    if !tools.is_empty() {
        if !via_template { inject_tools(&mut messages, &tools); }
        // Server-side agent loop: generate → parse tool_calls → execute the MCP-owned ones and feed
        // results back → repeat. Non-MCP tool calls are returned to the client (standard OpenAI flow).
        let (mut ptok, mut gtok) = (0usize, 0usize);
        let mut energies: Vec<Value> = Vec::new();
        for _round in 0..4 {
            let (prompt, image) = eng.chat_prompt(&messages, tools_arg, &kwargs)?;
            opts.image = image;
            let max = eng.budget(prompt.len(), &opts)?;
            let out = eng.generate(&prompt, max, &opts, None, |_, _| {});
            ptok += out.prompt_tokens; gtok += out.gen_tokens;
            energies.push(out.energy.clone());
            // Reasoning first (a thinking model reasons before it calls), then the answer part is parsed:
            // each model writes tool calls in its own template's format, in the decode that keeps special
            // tokens, with the tools' schemas for argument types.
            let raw = eng.detok_all(&out.ids);
            let (reasoning, answer) = match splitter(&prompt) {
                Some(mut sp) => { sp.push(&raw); sp.finish(); (sp.reasoning, sp.content) }
                None => (String::new(), raw),
            };
            let (before_call, calls) = ferric_agent::tools::parse_tool_calls_any(&answer, &tools);
            let mcp_calls: Vec<&Value> = calls.iter().filter(|c| mcps.borrow().has(c["function"]["name"].as_str().unwrap_or(""))).collect();
            if mcp_calls.is_empty() {
                let finish = if calls.is_empty() { out.finish } else { "tool_calls" };
                // With a call, `content` is the visible text before it (often empty); without, the answer.
                let text = if calls.is_empty() { if reasoning.is_empty() { out.text } else { answer.trim().to_string() } } else {
                    let special: Vec<&str> = eng.specials.iter().map(|(t, _)| t.as_str()).collect();
                    let mut t = before_call;
                    for sp in special { t = t.replace(sp, ""); }
                    t.trim().to_string()
                };
                return Ok(ChatResult { text, reasoning, tool_calls: calls, prompt_tokens: ptok, gen_tokens: gtok, finish, logprobs: Vec::new(), energy: sum_energy(&energies), stop_seq: out.stop_seq.clone() });
            }
            messages.push(json!({"role": "assistant", "content": out.text}));
            for c in &mcp_calls {
                let name = c["function"]["name"].as_str().unwrap_or("");
                let args: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or_else(|_| json!({}));
                let result = mcps.borrow_mut().call(name, &args).unwrap_or_else(|e| format!("error: {e}"));
                eprintln!("ferric-serve: mcp call {name}({args}) -> {result}");
                messages.push(json!({"role": "user", "content": format!("<tool_response>\n{{\"name\": \"{name}\", \"content\": {}}}\n</tool_response>", serde_json::to_string(&result).unwrap_or_default())}));
            }
        }
        return Err("the tool loop did not settle in 4 rounds".into());
    }

    // No tools → optional constrained decoding (JSON, schema, GBNF, regex, choice — see `constrain`).
    let spec = eng.constraint(req)?;
    let guide = spec.guide();
    let (prompt, image) = eng.chat_prompt(&messages, None, &kwargs)?;
    opts.image = image;
    let max = eng.budget(prompt.len(), &opts)?;
    let mut sp = splitter(&prompt);
    let out = eng.generate(&prompt, max, &opts, guide, |d, l| match sp.as_mut() {
        Some(sp) => {
            let (r, c) = sp.push(d);
            if !r.is_empty() { on_delta(&r, &[], true); }
            if !c.is_empty() { on_delta(&c, l, false); }
        }
        None => on_delta(d, l, false),
    });
    let (text, reasoning) = match sp.as_mut() {
        Some(sp) => {
            let (r, c) = sp.finish();
            if !r.is_empty() { on_delta(&r, &[], true); }
            if !c.is_empty() { on_delta(&c, &[], false); }
            (sp.content.trim_end().to_string(), sp.reasoning.trim_end().to_string())
        }
        None => (out.text, String::new()),
    };
    Ok(ChatResult { text, reasoning, tool_calls: Vec::new(), prompt_tokens: out.prompt_tokens, gen_tokens: out.gen_tokens,
                    finish: out.finish, logprobs: out.logprobs, energy: out.energy, stop_seq: out.stop_seq })
}

fn chat(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, stream: &mut TcpStream, body: &[u8]) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad_request(stream, &format!("bad json: {e}")) };
    // Validate before any header goes out, so a bad request is a 400 and not a broken stream.
    let opts = match eng.gen_opts(&req, true) { Ok(o) => o, Err(e) => return bad_request(stream, &e) };
    let empty = vec![];
    if let Err(e) = eng.chat_ids(req["messages"].as_array().unwrap_or(&empty)) { return bad_request(stream, &e); }
    // The constraint too (a schema the converter refuses, a GBNF that does not parse): a stream's headers go
    // out before generation resolves it. The compiled grammar is kept, so generation finds it again.
    if constrain::asks_for_constraint(&req) && let Err(e) = eng.constraint(&req) { return bad_request(stream, &e); }
    let has_tools = req["tools"].as_array().is_some_and(|t| !t.is_empty()) || !mcps.borrow().openai_tools().is_empty();
    if opts.n > 1 { return chat_n(eng, mcps, stream, &req, opts.n, has_tools, opts.logprobs); }
    let id = "chatcmpl-ferric";
    let streaming = req["stream"].as_bool().unwrap_or(false);
    let usage = |r: &ChatResult| json!({"prompt_tokens": r.prompt_tokens, "completion_tokens": r.gen_tokens, "total_tokens": r.prompt_tokens + r.gen_tokens});
    if streaming && has_tools {
        // A tool call is parsed from the whole output, so the answer exists only at the end — but a client
        // that asked for a stream must still get SSE (a JSON body breaks it). The tool calls go out as one
        // delta, with the index/id/type/function shape streaming clients accumulate.
        let r = run_chat(eng, mcps, &req, |_, _, _| {});
        write_sse_headers(stream);
        let chunk = |delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(),
            "model": eng.name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        send_sse(stream, &chunk(json!({"role": "assistant"}), Value::Null));
        match &r {
            Err(e) => { send_sse(stream, &json!({"error": {"message": e, "type": "server_error"}})); }
            Ok(r) if !r.tool_calls.is_empty() => {
                let calls: Vec<Value> = r.tool_calls.iter().enumerate().map(|(i, c)| json!({"index": i,
                    "id": c["id"].as_str().map(String::from).unwrap_or_else(|| format!("call_{i}")), "type": "function",
                    "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}})).collect();
                send_sse(stream, &chunk(json!({"tool_calls": calls}), Value::Null));
            }
            Ok(r) => { send_sse(stream, &chunk(json!({"content": r.text}), Value::Null)); }
        }
        let finish = r.as_ref().map(|r| r.finish).unwrap_or("stop");
        let mut last = chunk(json!({}), json!(finish));
        if let Ok(r) = &r { last["energy"] = r.energy.clone(); }
        send_sse(stream, &last);
        if let (Ok(r), true) = (&r, req["stream_options"]["include_usage"].as_bool() == Some(true)) {
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name, "choices": [], "usage": usage(r)}));
        }
        let _ = stream.write_all(b"data: [DONE]\n\n");
        return;
    }
    if streaming {
        write_sse_headers(stream);
        send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}]}));
        let r = run_chat(eng, mcps, &req, |delta, lps, reasoning| {
            let mut ch = json!({"index": 0, "delta": {(if reasoning { "reasoning_content" } else { "content" }): delta}, "finish_reason": Value::Null});
            if opts.logprobs && !reasoning { ch["logprobs"] = json!({"content": lps}); }
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name, "choices": [ch]}));
        });
        let (finish, u) = match &r { Ok(r) => (r.finish, Some(usage(r))), Err(_) => ("stop", None) };
        if let Err(e) = &r { send_sse(stream, &json!({"error": {"message": e, "type": "server_error"}})); }
        let mut last = json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]});
        if let Ok(r) = &r { last["energy"] = r.energy.clone(); }
        send_sse(stream, &last);
        if let (Some(u), true) = (u, req["stream_options"]["include_usage"].as_bool() == Some(true)) {
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name, "choices": [], "usage": u}));
        }
        let _ = stream.write_all(b"data: [DONE]\n\n");
        return;
    }
    match run_chat(eng, mcps, &req, |_, _, _| {}) {
        Err(e) => bad_request(stream, &e),
        Ok(r) => {
            let mut message = if r.tool_calls.is_empty() { json!({"role": "assistant", "content": r.text}) }
                          else { json!({"role": "assistant", "content": Value::Null, "tool_calls": r.tool_calls}) };
            if !r.reasoning.is_empty() { message["reasoning_content"] = json!(r.reasoning); }
            let mut choice = json!({"index": 0, "message": message, "finish_reason": r.finish});
            if opts.logprobs { choice["logprobs"] = logprobs_field(true, &r.logprobs); }
            write_json(stream, 200, &json!({"id": id, "object": "chat.completion", "created": now_unix(), "model": eng.name,
                "choices": [choice], "usage": usage(&r), "energy": r.energy}));
        }
    }
}

/// **n > 1 choices** of one chat prompt (parity gap S13): each choice is the request for that index
/// (`genopts::choice_request`), generated one after another; after the first, the prompt cache holds the
/// prompt, so a later choice prefills almost nothing. Streaming sends each choice's chunks under its own
/// `index`, which OpenAI's protocol allows. Usage counts the prompt once and every choice's tokens.
fn chat_n(eng: &Engine, mcps: &std::cell::RefCell<mcp::McpSet>, stream: &mut TcpStream, req: &Value, n: usize, has_tools: bool, logprobs: bool) {
    if has_tools { return bad_request(stream, "`n` > 1 with tools is not served: a tool round-trip is one conversation"); }
    let id = "chatcmpl-ferric";
    let streaming = req["stream"].as_bool().unwrap_or(false);
    let chunk = |i: usize, delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(),
        "model": eng.name, "choices": [{"index": i, "delta": delta, "finish_reason": finish}]});
    if streaming { write_sse_headers(stream); }
    let (mut choices, mut energies, mut ptok, mut gtok) = (Vec::new(), Vec::new(), 0usize, 0usize);
    for i in 0..n {
        let r_i = genopts::choice_request(req, i);
        if streaming { send_sse(stream, &chunk(i, json!({"role": "assistant"}), Value::Null)); }
        let r = run_chat(eng, mcps, &r_i, |delta, lps, reasoning| {
            if !streaming { return; }
            let mut ch = json!({"index": i, "delta": {(if reasoning { "reasoning_content" } else { "content" }): delta}, "finish_reason": Value::Null});
            if logprobs && !reasoning { ch["logprobs"] = json!({"content": lps}); }
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name, "choices": [ch]}));
        });
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                if streaming { send_sse(stream, &json!({"error": {"message": e, "type": "server_error"}})); let _ = stream.write_all(b"data: [DONE]\n\n"); }
                else { bad_request(stream, &e); }
                return;
            }
        };
        ptok = r.prompt_tokens;
        gtok += r.gen_tokens;
        energies.push(r.energy.clone());
        if streaming {
            send_sse(stream, &chunk(i, json!({}), json!(r.finish)));
        } else {
            let mut message = json!({"role": "assistant", "content": r.text});
            if !r.reasoning.is_empty() { message["reasoning_content"] = json!(r.reasoning); }
            let mut choice = json!({"index": i, "message": message, "finish_reason": r.finish});
            if logprobs { choice["logprobs"] = logprobs_field(true, &r.logprobs); }
            choices.push(choice);
        }
        if peer_gone() { return; }
    }
    let usage = json!({"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok});
    let energy = sum_energy(&energies);
    if streaming {
        if req["stream_options"]["include_usage"].as_bool() == Some(true) {
            send_sse(stream, &json!({"id": id, "object": "chat.completion.chunk", "created": now_unix(), "model": eng.name, "choices": [], "usage": usage, "energy": energy}));
        }
        let _ = stream.write_all(b"data: [DONE]\n\n");
    } else {
        write_json(stream, 200, &json!({"id": id, "object": "chat.completion", "created": now_unix(), "model": eng.name,
            "choices": choices, "usage": usage, "energy": energy}));
    }
}

fn completions(eng: &Engine, stream: &mut TcpStream, body: &[u8]) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad_request(stream, &format!("bad json: {e}")) };
    let Some(prompt_text) = req["prompt"].as_str() else {
        return bad_request(stream, "`prompt` must be a string (arrays of prompts and token arrays are not accepted here)")
    };
    let opts = match eng.gen_opts(&req, false) { Ok(o) => o, Err(e) => return bad_request(stream, &e) };
    // llama-server's `/completion` takes `grammar`; vLLM's completions take `guided_*` — both here.
    let spec = match eng.constraint(&req) { Ok(s) => s, Err(e) => return bad_request(stream, &e) };
    let mut ids = Vec::new();
    if eng.add_bos { if let Some(b) = eng.bos_id { ids.push(b); } }
    ids.extend(eng.enc(prompt_text, true));
    let max = match eng.budget(ids.len(), &opts) { Ok(m) => m, Err(e) => return bad_request(stream, &e) };
    let cid = format!("cmpl-ferric-{}", ids.len());
    if req["stream"].as_bool() == Some(true) && opts.n > 1 {
        return bad_request(stream, "`n` > 1 with a streamed completion is not served; stream n = 1 or ask without `stream`");
    }
    if req["stream"].as_bool() == Some(true) {
        write_sse_headers(stream);
        let out = eng.generate(&ids, max, &opts, spec.guide(), |delta, lps| {
            let mut ch = json!({"index": 0, "text": delta, "finish_reason": Value::Null});
            if opts.logprobs { ch["logprobs"] = logprobs_field(false, lps); }
            send_sse(stream, &json!({"id": cid, "object": "text_completion", "created": now_unix(), "model": eng.name, "choices": [ch]}));
        });
        send_sse(stream, &json!({"id": cid, "object": "text_completion", "created": now_unix(), "model": eng.name,
            "choices": [{"index": 0, "text": "", "finish_reason": out.finish}], "energy": out.energy}));
        if req["stream_options"]["include_usage"].as_bool() == Some(true) {
            send_sse(stream, &json!({"id": cid, "object": "text_completion", "created": now_unix(), "model": eng.name, "choices": [],
                "usage": {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.gen_tokens, "total_tokens": out.prompt_tokens + out.gen_tokens}}));
        }
        let _ = stream.write_all(b"data: [DONE]\n\n");
        return;
    }
    // n > 1: each choice with its own request (`genopts::choice_request`); the prompt cache makes every
    // choice after the first nearly prefill-free.
    let (mut choices, mut energies, mut gtok, mut ptok) = (Vec::new(), Vec::new(), 0usize, 0usize);
    for i in 0..opts.n {
        let o_i = if i == 0 { opts.clone() } else {
            match eng.gen_opts(&genopts::choice_request(&req, i), false) { Ok(o) => o, Err(e) => return bad_request(stream, &e) }
        };
        let out = eng.generate(&ids, max, &o_i, spec.guide(), |_, _| {});
        let mut choice = json!({"index": i, "text": out.text, "finish_reason": out.finish});
        if opts.logprobs { choice["logprobs"] = logprobs_field(false, &out.logprobs); }
        choices.push(choice);
        energies.push(out.energy);
        ptok = out.prompt_tokens;
        gtok += out.gen_tokens;
        if peer_gone() { return; }
    }
    write_json(stream, 200, &json!({
        "id": cid, "object": "text_completion", "created": now_unix(), "model": eng.name,
        "choices": choices,
        "usage": {"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok},
        "energy": if energies.len() == 1 { energies.pop().unwrap() } else { sum_energy(&energies) }
    }));
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>, Vec<(String, String)>)> {
    let peer = stream.try_clone().ok()?;
    let mut reader = BufReader::new(peer);
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 { return None; }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    // The query string is not part of the route: `/health?x=1` used to 404.
    let path = parts.next()?.split('?').next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    let mut headers: Vec<(String, String)> = Vec::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 { break; }
        if h.trim().is_empty() { break; }
        if let Some((k, v)) = h.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_string());
            if k == "content-length" { content_length = v.parse().unwrap_or(0); }
            headers.push((k, v));
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 { reader.read_exact(&mut body).ok()?; }
    Some((method, path, body, headers))
}

fn write_json(stream: &mut TcpStream, status: u16, v: &Value) {
    trace::with(|s| s.status(status));
    let body = serde_json::to_vec(v).unwrap_or_default();
    let head = format!("HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        if status == 200 { "OK" } else { "ERR" }, body.len());
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
    let _ = stream.flush();
}

/// CORS preflight. A browser front-end (a web chat UI calling this server directly) sends OPTIONS before
/// every JSON POST; answering it 404 made every such request fail before it was sent.
pub(crate) fn write_preflight(stream: &mut TcpStream) {
    let _ = stream.write_all(b"HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    let _ = stream.flush();
}

fn write_sse_headers(stream: &mut TcpStream) {
    trace::with(|s| s.status(200));
    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    let _ = stream.flush();
}

/// Returns false when the client has gone (the write failed) — the caller can stop generating for it.
fn send_sse(stream: &mut TcpStream, v: &Value) -> bool {
    let ok = stream.write_all(format!("data: {}\n\n", v).as_bytes()).is_ok() && stream.flush().is_ok();
    if !ok { mark_peer_gone(); }
    ok
}

thread_local! {
    /// A stream write on this (the engine) thread failed: the client of the generation running here has
    /// gone. Every serial streaming route writes through `send_sse` / `ollama::send_line` /
    /// `dialects::send_event`, and `generate` stops at the next token when this is set, so none of them
    /// has to thread a cancel flag of its own. Cleared when a generation starts.
    ///
    /// ⛔ Found by the CLI's live test: Ctrl-C closed an `/api/chat` stream at 4 s, and the server went on
    /// to generate all 800 requested tokens with `ferric_requests_cancelled_total` still 0 — the NDJSON
    /// writer ignored its write errors. The batched path had this (`Gen::gone`, 3f6089f); no serial one did.
    static PEER_GONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) fn mark_peer_gone() { PEER_GONE.with(|g| g.set(true)); }
fn peer_gone() -> bool { PEER_GONE.with(|g| g.get()) }

/// Whether the peer has closed the connection, without consuming anything it sent.
pub(crate) fn peer_closed(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() { return false; }
    let mut b = [0u8; 1];
    let closed = matches!(stream.peek(&mut b), Ok(0));
    let _ = stream.set_nonblocking(false);
    closed
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
    /// `encoding_format: "base64"` is what the official OpenAI Python client asks for by default; a
    /// wrong encoder returns vectors that decode to different floats with no error.
    #[test]
    fn prompt_lookup_drafts_what_followed_the_longest_recent_match() {
        use super::lookup_draft;
        // "... 5 6 7 8 9 ... 5 6 7" → the 3-gram 5 6 7 was followed by 8 9 before.
        assert_eq!(lookup_draft(&[1, 5, 6, 7, 8, 9, 2, 3, 5, 6, 7], 4), vec![8, 9, 2, 3]);
        // The MOST RECENT occurrence wins: 4 4 was followed by 1 first, then by 2.
        assert_eq!(lookup_draft(&[4, 4, 1, 4, 4, 2, 9, 4, 4], 1), vec![2]);
        assert!(lookup_draft(&[1, 2, 3, 4], 8).is_empty(), "no repeated n-gram, no draft");
        assert_eq!(lookup_draft(&[7, 8, 7, 8], 8), vec![7, 8], "never drafts past the end of the context");
    }

    #[test]
    fn preparing_a_message_for_the_template_adds_no_key() {
        let m = serde_json::json!({"role": "user", "content": [{"type": "text", "text": "hi"}]});
        let p = super::prepare_for_template(&m);
        assert_eq!(p, serde_json::json!({"role": "user", "content": "hi"}));
        assert!(p.get("tool_calls").is_none(), "a missing key must stay missing, or `'tool_calls' in message` is true");
        let tc = serde_json::json!({"role": "assistant", "content": null,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]});
        assert_eq!(super::prepare_for_template(&tc)["tool_calls"][0]["function"]["arguments"]["a"], 1);
        // An image message keeps its parts — the template places the image where it finds one.
        let img = serde_json::json!({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what is it?"}]});
        assert_eq!(super::prepare_for_template(&img), img);
    }

    #[test]
    fn base64_embeddings_are_little_endian_f32() {
        // 1.0f32 = 00 00 80 3F; -2.0f32 = 00 00 00 C0  →  "AACAPwAAAMA="
        assert_eq!(super::b64_f32(&[1.0, -2.0]), "AACAPwAAAMA=");
        // one float: 4 bytes → 8 characters with "==" padding
        assert_eq!(super::b64_f32(&[0.0]), "AAAAAA==");
    }

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
        // ⚠ PER-RUNTIME needle. It was a single `pub fn forward_batch(` for every row, and that
        // silently stopped covering anything the moment a runtime named its batched entry point
        // differently: hyv4's is `pub fn decode_batch(`, because its solo entry point is `decode`,
        // not `forward`. A guard whose needle cannot match a runtime it is checking reports success
        // about a method it never looked for.

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

        // ⚠ Not every runtime is batchable — `Model::NemotronH(_) => false` — so the invariant is
        // "does every variant appear in SOME arm, and does each one that claims `true` back the
        // claim with a real method". (This comment previously said the refusing arm was empty; it
        // has not been since nemotron_h landed.)
        const BLANKET_ARM: &str = "Model::Lfm2(_) | Model::Gemma4(_) | Model::DeepSeek2(_) => true";
        assert!(body.contains(BLANKET_ARM),
                "Model::supports_batching's arms changed shape; this guard reads them literally, so \
                 update both together rather than letting the guard go quiet.\nbody was:\n{body}");

        // (runtime source, its Model variant, its batched entry point, does it consult its OWN
        //  batching_supported predicate)
        for (name, src, variant, needle, asks_runtime) in [
            ("qwen3",     include_str!("../../ferric-llama/src/qwen3.rs"),     "Model::Dense",     "pub fn forward_batch(", true),
            ("qwen35",    include_str!("../../ferric-llama/src/qwen35.rs"),    "Model::Hybrid",    "pub fn forward_batch(", true),
            ("lfm2",      include_str!("../../ferric-llama/src/lfm2.rs"),      "Model::Lfm2",      "pub fn forward_batch(", false),
            ("gemma4",    include_str!("../../ferric-llama/src/gemma4.rs"),    "Model::Gemma4",    "pub fn forward_batch(", false),
            ("deepseek2", include_str!("../../ferric-llama/src/deepseek2.rs"), "Model::DeepSeek2", "pub fn forward_batch(", false),
            ("hyv4",      include_str!("../../ferric-llama/src/hyv4.rs"),      "Model::Hyv4",      "pub fn decode_batch(",  false),
        ] {
            // ⚠ HONEST LABEL, verified by running the mutation: this is SUBSUMED BY THE COMPILER
            // for every row whose dispatch arm calls the method — renaming `decode_batch` fails with
            // `error[E0599]: no method named decode_batch`, so the mutation never reaches here. What
            // it still catches is the case the compiler cannot see: a variant that claims `true` in
            // supports_batching while its forward_batch arm is `unreachable!` and calls nothing.
            assert!(src.contains(needle),
                    "{name} is claimed batchable by Model::supports_batching but has no {needle} — \
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
                // it when speech landed — it takes a waveform, not tokens. `ModernBert` joined it on
                // 2026-09-22: a second encoder family, structurally unlike `Bert` but with the same
                // answer — no KV cache, no LM head. Adding a runtime here must be a deliberate edit,
                // which is the whole point: this test went red the moment each was registered,
                // rather than letting a refusal in unnoticed.
                assert!(matches!(a.runtime, ferric_llama::arch::Runtime::Bert
                                          | ferric_llama::arch::Runtime::ModernBert
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
                        && matches!(a.runtime, Runtime::Bert | Runtime::ModernBert
                                               | Runtime::Cosmos | Runtime::Parakeet))
            .map(|a| a.name).collect();
        refused.sort(); expect.sort();
        assert_eq!(refused, expect,
                   "a runnable registry row is refused by this server for a reason other than being \
                    an encoder or a safetensors-only model");
    }

}
