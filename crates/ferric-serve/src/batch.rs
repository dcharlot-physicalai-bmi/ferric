//! **Continuous batching, wired into the transport.**
//!
//! `ferric_llama::sched::Scheduler` is the policy (unit-tested without a GPU) and
//! `ferric-llama/examples/continuous_batching.rs` is the engine loop proven on real weights. This is
//! the part between them that was missing: the accept/dispatch path, so that concurrent HTTP
//! requests actually land in the same `forward_batch` instead of queueing behind one another.
//!
//! ## Shape of the thing
//!
//! ```text
//!   accept thread ──┬─ reader thread ─┐
//!                   ├─ reader thread ─┼──▶ Inbox (Mutex + Condvar) ──▶ ENGINE THREAD
//!                   └─ reader thread ─┘                                 Scheduler
//!                                                                       prefill (solo)
//!                                                                       decode  (batched)
//!                                                                       write responses
//! ```
//!
//! The model never crosses a thread. One reader thread per connection exists so a slow client cannot
//! stall the accept loop, but it only parses HTTP; the socket is handed to the engine once a whole
//! request is in hand. That leaves exactly one thread touching the GPU, which is what the runtime
//! wants, and it is why the design does not need `Engine: Sync` (it holds a `RefCell`).
//!
//! **The inbox is drained on every engine step**, not once per batch. That is the entire difference
//! between continuous and static batching: a request arriving while four sequences are mid-decode is
//! admitted on the next step into whatever slot is free.
//!
//! ## Browser (wasm32)
//!
//! This module is native-only *by construction*, and not because of the mutex. `wgpu`'s `Device` and
//! `Queue` are `Send + Sync` on native but **not** on `wasm32`, and there is no `TcpListener` in a
//! browser at all. A browser build of the same feature needs a different transport and a different
//! ownership story: the model pinned to one worker thread, requests arriving over a `postMessage`
//! channel rather than a socket, and the scheduler stepped from that worker's own event loop. The
//! policy (`sched::Scheduler`) and the per-step loop below are transport-agnostic and would port; the
//! `Inbox` and `serve_loop`'s accept plumbing would not. `ferric-serve` is not in the wasm build
//! (`scripts/fabric-ci.sh` builds only `-p ferric-web` for `wasm32-unknown-unknown`).
//!
//! ## What is deliberately NOT batched
//!
//! Guided decoding (`response_format`) and the server-side tool/MCP agent loop keep the **existing
//! serial code path, unchanged**. That is a limitation with a reason: `guide::Guide<'a>` borrows the
//! compiled schema program, so carrying one inside a long-lived in-flight sequence is a
//! self-referential lifetime, and the tool loop is multi-round (generate → call → generate) rather
//! than a single generation. Both still work exactly as before; they just do not share a batch, and
//! they block the batch while they run. Keeping them on the untouched path also means structured
//! output cannot regress as a side effect of this change.

use crate::{Engine, ModelCache, write_json, write_sse_headers, write_preflight, send_sse, now_unix, read_request};
use crate::genopts::{Emitter, GenOpts, Sampling};
use ferric_llama::sched::{Done, Scheduler, SeqId};
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};

/// Everything the batching loop needs from a model.
///
/// Extracted as a trait for one reason, and it is not abstraction for its own sake: without it the
/// dispatch path can only be tested with a GPU and a multi-hundred-megabyte checkpoint, which means
/// in practice it is tested by hand and never in CI. The test double below is a deterministic
/// stand-in whose output *changes* if the loop misaligns a sequence with its state — so the tests
/// exercise the real `serve_loop`, the real `Scheduler`, real sockets and real HTTP, and mock only
/// the matmul.
pub(crate) trait ServeModel {
    /// Per-sequence state — the KV/recurrent cache. Lives outside the scheduler, keyed by sequence.
    type State;

    fn name(&self) -> &str;
    fn n_vocab(&self) -> usize;

    /// May this model's sequences share a `forward_batch`? `false` forces the serial fallback: the
    /// loop below runs with `max_batch == 1` and never calls `decode` with more than one sequence.
    fn can_batch(&self) -> bool;

    /// Tokenize a chat request's `messages`. `Err` = a message this path cannot feed (a 400).
    fn encode_chat(&self, messages: &[Value]) -> Result<Vec<u32>, String>;
    /// Tokenize a `/v1/completions` prompt string.
    fn encode_text(&self, text: &str) -> Vec<u32>;

    /// Start a sequence: a fresh state (or one seeded from the prompt cache) and how many of the
    /// prompt's tokens it already holds — never all of them, since the last must be fed to give logits.
    fn begin(&self, prompt: &[u32], opts: &GenOpts) -> (Self::State, usize);

    /// A request's generation options. The default reads only the request; a model with LoRA adapters
    /// also resolves which of them it selects (`Engine::gen_opts`), so every batched row keeps its own.
    fn gen_opts(&self, req: &Value, chat: bool) -> Result<GenOpts, String> { GenOpts::from_req(req, chat) }

    /// Feed the next run of prompt tokens. `last` = they end the prompt: return the **last** row of
    /// logits, the row the first sampled token comes from.
    ///
    /// Prefill is per-sequence on purpose: `forward_batch` is the DECODE step (one token per
    /// sequence), so a newly admitted request builds its cache alone and joins the batch from its
    /// second token onward — fed in chunks while others decode (see `step`).
    fn feed(&self, state: &mut Self::State, toks: &[u32], last: bool) -> Option<Vec<f32>>;

    /// One decode step for N sequences: `toks[i]` advances `states[i]`. Returns N × `n_vocab`
    /// logits, row `i` belonging to sequence `i`.
    fn decode(&self, toks: &[u32], states: &mut [&mut Self::State]) -> Vec<f32>;

    /// Sample one token from one row. `None` = stop with nothing further emitted.
    fn pick(&self, row: &[f32], s: &Sampling, prompt: &[u32], generated: &[u32], rng: &mut u64) -> Option<u32>;
    fn is_stop(&self, tok: u32) -> bool;
    fn text_of(&self, ids: &[u32]) -> String;
    /// A token's text and bytes, for `logprobs`.
    fn piece(&self, tok: u32) -> (String, Vec<u8>);
    /// Tokens this request may generate given its prompt: its own `max_tokens`, capped by the context.
    fn budget(&self, prompt_len: usize, want: Option<usize>) -> Result<usize, String>;
    /// This model generates better on its own serial loop than in the scheduler — a hybrid with an MTP
    /// draft block, whose `generate_spec` (and one-slot prefix cache) the batched step cannot run.
    fn serial_generation(&self) -> bool { false }
    /// Open a request's energy window (see `energy`); the default meters nothing.
    fn energy_begin(&self) -> Option<crate::energy::Ticket> { None }
    /// Count a generation abandoned by its client.
    fn cancelled(&self) {}
    /// Offer a finished sequence's cache for prompt caching (`tokens` = prompt + generated).
    fn remember(&self, _tokens: &[u32], _state: &mut Self::State) {}
    /// Count a finished generation.
    fn record(&self, _prompt: usize, _gen: usize, _energy: &Value) {}
    /// Close it and attribute its joules.
    fn energy_end(&self, _t: Option<crate::energy::Ticket>, _tokens: usize) -> Value { Value::Null }
    /// `(name, Ollama /api/tags entry)` for everything this model answers for: itself first, then any
    /// encoder-side model the server shares with it (an embedder). A request naming any of them routes here.
    fn cards(&self) -> Vec<(String, Value)> {
        vec![(self.name().to_string(), json!({"name": self.name(), "model": self.name(), "size": 0}))]
    }
    /// The context the model serves (for `/api/ps`'s `context_length`); 0 = unknown.
    fn context(&self) -> usize { 0 }
}

/// One HTTP request, parsed off its socket by a reader thread and handed to the engine thread.
pub(crate) struct Job {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    /// Request headers, names lower-cased (Content-Type for multipart, Authorization / x-api-key).
    pub headers: Vec<(String, String)>,
    pub stream: TcpStream,
}

/// Reader-threads → engine-thread handoff. `closed` is set when the listener stops yielding.
pub(crate) struct Inbox {
    q: Mutex<(VecDeque<Job>, bool)>,
    cv: Condvar,
}

impl Inbox {
    pub fn new() -> Inbox { Inbox { q: Mutex::new((VecDeque::new(), false)), cv: Condvar::new() } }

    pub fn push(&self, j: Job) {
        let mut g = self.q.lock().unwrap();
        g.0.push_back(j);
        self.cv.notify_all();
    }

    pub fn close(&self) {
        let mut g = self.q.lock().unwrap();
        g.1 = true;
        self.cv.notify_all();
    }

    /// Take everything queued **right now**, without waiting. Called on every engine step; this is
    /// what makes the batching continuous rather than static.
    pub fn drain(&self) -> Vec<Job> {
        let mut g = self.q.lock().unwrap();
        g.0.drain(..).collect()
    }

    /// Block until a job arrives or `deadline` passes (then an empty batch). `None` = closed and empty.
    pub fn wait_until(&self, deadline: std::time::Instant) -> Option<Vec<Job>> {
        let mut g = self.q.lock().unwrap();
        while g.0.is_empty() && !g.1 {
            let now = std::time::Instant::now();
            if now >= deadline { return Some(Vec::new()); }
            g = self.cv.wait_timeout(g, deadline - now).unwrap().0;
        }
        if g.0.is_empty() { return None; }
        Some(g.0.drain(..).collect())
    }

    /// Block until at least one job arrives. `None` = the listener closed and nothing is left.
    /// Only ever called when the engine has no in-flight work, so it can never stall a live batch.
    pub fn wait(&self) -> Option<Vec<Job>> {
        let mut g = self.q.lock().unwrap();
        while g.0.is_empty() && !g.1 { g = self.cv.wait(g).unwrap(); }
        if g.0.is_empty() { return None; }
        Some(g.0.drain(..).collect())
    }
}

/// One in-flight generation. The scheduler owns the *slot*; this owns everything the response needs.
struct Gen<S> {
    id: SeqId,
    stream: TcpStream,
    streaming: bool,
    /// `/v1/chat/completions` (true) vs `/v1/completions` (false) — decides the response envelope.
    chat: bool,
    opts: GenOpts,
    /// Per-sequence RNG, seeded identically to the serial path so sampled output is reproducible
    /// **and** independent of what else happens to be in the batch. A shared RNG would make a
    /// request's output depend on its neighbours, which is exactly the cross-sequence coupling the
    /// batched forward is verified not to have.
    rng: u64,
    prompt: Vec<u32>,
    r#gen: Vec<u32>,
    em: Emitter,
    logprobs: Vec<Value>,
    lp_sent: usize,
    include_usage: bool,
    /// Energy window, opened when the sequence is admitted (prefill) and closed when it retires.
    ticket: Option<crate::energy::Ticket>,
    /// The client went away: stop generating for it and free the slot on the next step.
    gone: bool,
    state: Option<S>,
    /// Prompt tokens its state holds so far (chunked prefill), and whether its first token is out —
    /// only then does it join the decode batch.
    fed: usize,
    ready: bool,
    /// The token to feed on the next decode step.
    next: u32,
}

impl<S> Gen<S> {
    /// Commit one token: record it (and its logprob), and stream whatever is safe to release. The same
    /// `Emitter` as the serial `Engine::generate`, so multi-byte UTF-8 and stop strings are handled one
    /// way. Returns `true` when a stop string just completed — the sequence must retire.
    fn commit<M: ServeModel<State = S>>(&mut self, m: &M, tok: u32, row: &[f32]) -> bool {
        if self.opts.logprobs {
            let (lp, alts) = crate::genopts::logprobs_of(row, tok, self.opts.top_logprobs);
            self.logprobs.push(crate::genopts::logprob_entry(&|t| m.piece(t), tok, lp, &alts));
        }
        self.r#gen.push(tok);
        if let Some(d) = self.em.update(&m.text_of(&self.r#gen)) { self.send_delta(m, &d); }
        self.em.hit_stop
    }

    fn send_delta<M: ServeModel<State = S>>(&mut self, m: &M, delta: &str) {
        if self.streaming {
            let mut ch = json!({"index": 0, "delta": {"content": delta}, "finish_reason": Value::Null});
            if self.opts.logprobs { ch["logprobs"] = json!({"content": &self.logprobs[self.lp_sent..]}); }
            if !send_sse(&mut self.stream, &json!({
                "id": "chatcmpl-ferric", "object": "chat.completion.chunk", "created": now_unix(),
                "model": m.name(), "choices": [ch]})) { self.gone = true; }
        }
        self.lp_sent = self.logprobs.len();
    }
}

/// Server knobs the loop needs that the model does not know about.
pub(crate) struct ServeOpts {
    pub max_batch: usize,
    /// Any MCP server is connected, so every chat request advertises tools and must take the
    /// multi-round serial agent path.
    pub any_mcp_tools: bool,
    /// `--api-key`: when set, every request but `/health` and CORS preflight must present it as
    /// `Authorization: Bearer <key>` or `x-api-key: <key>` (the OpenAI and Anthropic spellings).
    pub api_key: Option<String>,
    /// Most chat models resident at once (`--max-models`, FERRIC_MAX_LOADED_MODELS; Ollama's default is 3).
    pub max_models: usize,
    /// Most bytes of model files resident at once (FERRIC_MAX_MEMORY, GiB): weights dominate what a model
    /// holds, so the file size is the estimate. KV caches are NOT counted — see `Pool::room_for`.
    pub max_bytes: u64,
    /// How long a model loaded for a request stays after its last one (`--keep-alive`, FERRIC_KEEP_ALIVE;
    /// Ollama's default 5 m). `None` = until evicted for room. A model named on the command line stays.
    pub keep_alive: Option<std::time::Duration>,
    /// A model name nothing on disk matches goes to the default model instead of a 404 — the
    /// llama-server behaviour single-model deployments rely on (clients that send `gpt-4o` to a local
    /// server). On when the server was started with a model; the response's `model` names who answered.
    pub fallback: bool,
}

fn authorized(headers: &[(String, String)], key: &str) -> bool {
    headers.iter().any(|(k, v)| (k == "authorization" && v.strip_prefix("Bearer ").map(str::trim) == Some(key)) || (k == "x-api-key" && v.trim() == key))
}

/// Requests the batch loop declines, and hands to the untouched serial path. See the module docs.
fn must_run_serial(req: &Value, chat: bool, opts: &ServeOpts) -> bool {
    // A constraint masks every step (`constrain`); the batched step has no mask.
    if crate::constrain::asks_for_constraint(req) { return true; }
    // Streaming completions is served by the serial path's SSE writer only for chat; completions
    // here are non-streaming, so a streaming completions request is declined rather than answered as
    // one JSON blob it did not ask for.
    if !chat && req["stream"].as_bool() == Some(true) { return true; }
    if chat && (opts.any_mcp_tools || req["tools"].as_array().is_some_and(|t| !t.is_empty())) { return true; }
    // An image is spliced into its own prefill with its own positions (`vision`); it never shares a batch.
    if chat && req["messages"].as_array().is_some_and(|ms| ms.iter().any(|m|
        m["images"].as_array().is_some_and(|a| !a.is_empty())
        || m["content"].as_array().is_some_and(|ps| ps.iter().any(|p| matches!(p["type"].as_str(), Some("image_url" | "input_image" | "image"))))))
    { return true; }
    false
}

/// The server. Owns every loaded model on this thread; the listener is drained by a spawned accept
/// thread. `serial` handles everything the batch loop declines (guided decoding, the tool loop,
/// embeddings, unknown paths) and returns whether it recognised the request. `initial` = the models
/// named on the command line; `src` loads any other model a request names.
pub(crate) fn serve_loop<S: Source>(
    src: S,
    initial: Vec<Loaded<S::M>>,
    listener: TcpListener,
    opts: ServeOpts,
    mut serial: impl FnMut(&S::M, &str, &str, &[u8], &[(String, String)], &mut TcpStream) -> bool,
) {
    let inbox = Arc::new(Inbox::new());
    {
        let ib = inbox.clone();
        std::thread::spawn(move || {
            for s in listener.incoming() {
                let Ok(s) = s else { continue };
                let ib2 = ib.clone();
                // One reader per connection: parsing must not be able to block the accept loop, and
                // the engine must never block on a socket that has not finished sending its body.
                std::thread::spawn(move || {
                    let mut s = s;
                    if let Some((method, path, body, headers)) = read_request(&mut s) {
                        ib2.push(Job { method, path, body, headers, stream: s });
                    }
                });
            }
            ib.close();
        });
    }

    let mut pool = Pool { src, slots: Vec::new() };
    for l in initial { pool.slots.push(Slot::new(l, None, true, opts.max_batch)); }
    // Requests for a model with no room yet: every resident model is mid-generation. Retried each step.
    let mut pending: VecDeque<Job> = VecDeque::new();

    loop {
        // Block only when nothing is in flight; otherwise take whatever has arrived and keep stepping.
        // This is the continuous-batching admission point. An idle wait still wakes for the next
        // keep-alive expiry, so an idle model is unloaded on time rather than at the next request.
        let busy = pool.slots.iter().any(Slot::busy);
        let jobs = if busy || !pending.is_empty() {
            inbox.drain()
        } else {
            match pool.next_expiry() {
                Some(t) => match inbox.wait_until(t) { Some(j) => j, None => return },
                None => match inbox.wait() { Some(j) => j, None => return },
            }
        };
        let mut todo = std::mem::take(&mut pending);
        todo.extend(jobs);
        for j in todo { if let Some(j) = pool.route(j, &opts, &mut serial) { pending.push_back(j); } }
        // One step per busy model per turn: two models generating at once interleave their steps on
        // the one GPU thread rather than one starving the other.
        for s in pool.slots.iter_mut() {
            if s.busy() { step(&s.m, &mut s.sched, &mut s.gens); s.last = std::time::Instant::now(); }
        }
        pool.expire();
    }
}

/// Dispatch one parsed request to ONE model: generation into its scheduler, everything else to the
/// serial handler.
fn route<M: ServeModel>(
    m: &M,
    sched: &mut Scheduler,
    gens: &mut Vec<Gen<M::State>>,
    mut j: Job,
    opts: &ServeOpts,
    serial: &mut impl FnMut(&M, &str, &str, &[u8], &[(String, String)], &mut TcpStream) -> bool,
) {
    let chat = j.path == "/v1/chat/completions";
    let is_gen = chat || j.path == "/v1/completions";
    if j.method == "POST" && is_gen {
        let bad = |s: &mut TcpStream, m: &str| write_json(s, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}}));
        let req: Value = match serde_json::from_slice(&j.body) {
            Ok(v) => v,
            Err(e) => return bad(&mut j.stream, &format!("bad json: {e}")),
        };
        if must_run_serial(&req, chat, opts) || m.serial_generation() {
            if !serial(m, &j.method, &j.path, &j.body, &j.headers, &mut j.stream) {
                write_json(&mut j.stream, 404, &json!({"error": {"message": "not found", "type": "invalid_request_error"}}));
            }
            return;
        }
        let gopts = match m.gen_opts(&req, chat) { Ok(o) => o, Err(e) => return bad(&mut j.stream, &e) };
        let prompt = if chat {
            let empty = vec![];
            match m.encode_chat(req["messages"].as_array().unwrap_or(&empty)) { Ok(p) => p, Err(e) => return bad(&mut j.stream, &e) }
        } else {
            match req["prompt"].as_str() {
                Some(p) => m.encode_text(p),
                None => return bad(&mut j.stream, "`prompt` must be a string (arrays of prompts and token arrays are not accepted here)"),
            }
        };
        // Same debug hook as the serial path: the ids the model will actually see, replayable elsewhere.
        if std::env::var("FERRIC_DUMP_IDS").is_ok() { eprintln!("prompt ids ({}): {:?}", prompt.len(), prompt); }
        let max_tokens = match m.budget(prompt.len(), gopts.max_tokens) { Ok(n) => n, Err(e) => return bad(&mut j.stream, &e) };
        let streaming = chat && req["stream"].as_bool().unwrap_or(false);
        if streaming {
            write_sse_headers(&mut j.stream);
            send_sse(&mut j.stream, &json!({
                "id": "chatcmpl-ferric", "object": "chat.completion.chunk", "created": now_unix(),
                "model": m.name(),
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}]}));
        }
        let id = sched.submit(prompt.clone(), max_tokens);
        gens.push(Gen {
            id, stream: j.stream, streaming, chat,
            // Same seed as `Engine::generate` (the fixed default, or the request's `seed`), per sequence.
            rng: gopts.rng,
            em: Emitter::new(&gopts.stop),
            include_usage: req["stream_options"]["include_usage"].as_bool() == Some(true),
            opts: gopts,
            prompt, r#gen: Vec::new(), logprobs: Vec::new(), lp_sent: 0, ticket: None, gone: false, state: None, fed: 0, ready: false, next: 0,
        });
        return;
    }
    if !serial(m, &j.method, &j.path, &j.body, &j.headers, &mut j.stream) {
        write_json(&mut j.stream, 404, &json!({"error": {"message": "not found", "type": "invalid_request_error"}}));
    }
}

// ---------------------------------------------------------------------------------------------
// Several models: the pool
// ---------------------------------------------------------------------------------------------

/// Where models come from when a request names one that is not loaded (`models::Hub` for real files).
pub(crate) trait Source {
    type M: ServeModel;
    /// The name a request used → the key of a loadable model (its canonical path) and its size in
    /// bytes, or why nothing matches (a 404 carrying that text).
    fn resolve(&mut self, spec: &str) -> Result<(String, u64), String>;
    /// Build it. Runs on the engine thread, so every other model's generations pause for the load.
    fn load(&mut self, key: &str) -> Result<Self::M, String>;
    /// Every model that could be served, loaded or not: `(name, Ollama /api/tags entry)`.
    fn available(&mut self) -> Vec<(String, Value)>;
    /// `/api/show` for a model that is not loaded, from its file alone.
    fn show(&mut self, _key: &str) -> Option<Value> { None }
    /// `/metrics` for the server, given the keys of the loaded models.
    fn metrics(&mut self, _loaded: &[&str]) -> Option<String> { None }
}

/// A model already built, handed to the pool (the ones named on the command line).
pub(crate) struct Loaded<M> { pub key: String, pub m: M, pub bytes: u64 }

/// One resident model: its own scheduler and in-flight sequences, so its batch never mixes with
/// another model's (a batch is one weight set read once).
struct Slot<M: ServeModel> {
    key: String,
    m: M,
    bytes: u64,
    sched: Scheduler,
    gens: Vec<Gen<M::State>>,
    /// `None` = until evicted for room.
    keep_alive: Option<std::time::Duration>,
    /// When it last did anything: routed a request, or stepped. Keep-alive counts from here, so a
    /// long generation is not unloaded mid-answer and the clock starts when it finishes.
    last: std::time::Instant,
    /// Named on the command line: the default for requests that name no model, and never unloaded to
    /// make room for another (only an explicit `keep_alive: 0` unloads it). Found live: with three
    /// models on demand, the fourth evicted the command-line model, and the fallback for an unknown
    /// name then went to whichever model happened to be used last.
    startup: bool,
}

impl<M: ServeModel> Slot<M> {
    fn new(l: Loaded<M>, keep_alive: Option<std::time::Duration>, startup: bool, max_batch: usize) -> Slot<M> {
        // A model that cannot batch runs the SAME loop with one slot. There is no second code path to
        // rot: the fallback differs only in `max_batch` and in `decode` taking the solo forward.
        let max_batch = if l.m.can_batch() { max_batch.max(1) } else { 1 };
        Slot { key: l.key, m: l.m, bytes: l.bytes, sched: Scheduler::new(max_batch), gens: Vec::new(),
               keep_alive, last: std::time::Instant::now(), startup }
    }
    fn busy(&self) -> bool { !self.gens.is_empty() }
    fn expires(&self) -> Option<std::time::Instant> { self.keep_alive.map(|d| self.last + d) }
    fn answers_to(&self, spec: &str) -> bool {
        let base = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_ascii_lowercase();
        self.key == spec || self.m.cards().iter().any(|(n, _)| base(n) == base(spec))
    }
}

/// Ollama's `keep_alive` (and LM Studio's `ttl`, seconds): `Ok(None)` = not given; `Ok(Some(None))` =
/// forever (a negative value); `Ok(Some(Some(d)))` = unload `d` after the last request (0 = at once).
/// A number is seconds; a string is a Go duration (`"5m"`, `"1h30m"`, `"250ms"`, `"-1"`), or bare seconds.
pub(crate) fn keep_alive_of(req: &Value) -> Result<Option<Option<std::time::Duration>>, String> {
    let secs = |x: f64| if x < 0.0 { None } else { Some(std::time::Duration::from_secs_f64(x)) };
    let v = if !req["keep_alive"].is_null() { &req["keep_alive"] } else { &req["ttl"] };
    match v {
        Value::Null => Ok(None),
        Value::Number(n) => Ok(Some(secs(n.as_f64().unwrap_or(0.0)))),
        Value::String(s) => {
            let t = s.trim();
            if let Ok(x) = t.parse::<f64>() { return Ok(Some(secs(x))); }
            let (neg, mut rest) = match t.strip_prefix('-') { Some(r) => (true, r), None => (false, t.strip_prefix('+').unwrap_or(t)) };
            let mut total = 0f64;
            if rest.is_empty() { return Err(format!("keep_alive {s:?}: expected a duration like \"5m\" or seconds")); }
            while !rest.is_empty() {
                let n = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
                let x: f64 = rest[..n].parse().map_err(|_| format!("keep_alive {s:?}: expected a duration like \"5m\" or seconds"))?;
                rest = &rest[n..];
                let u = rest.find(|c: char| c.is_ascii_digit() || c == '.').unwrap_or(rest.len());
                let scale = match &rest[..u] { "ns" => 1e-9, "us" | "µs" | "μs" => 1e-6, "ms" => 1e-3, "s" => 1.0, "m" => 60.0, "h" => 3600.0,
                    unit => return Err(format!("keep_alive {s:?}: unknown unit {unit:?} (ns, us, ms, s, m, h)")) };
                total += x * scale;
                rest = &rest[u..];
            }
            Ok(Some(if neg { None } else { secs(total) }))
        }
        v => Err(format!("keep_alive must be a number of seconds or a duration string, got {v}")),
    }
}

struct Pool<S: Source> { src: S, slots: Vec<Slot<S::M>> }

/// Why a request could not be given a model: the status and the message.
type Refusal = (u16, String);

impl<S: Source> Pool<S> {
    fn next_expiry(&self) -> Option<std::time::Instant> {
        self.slots.iter().filter(|s| !s.busy()).filter_map(Slot::expires).min()
    }

    /// Unload every idle model whose keep-alive has run out.
    fn expire(&mut self) {
        let now = std::time::Instant::now();
        self.slots.retain(|s| {
            let stay = s.busy() || s.expires().is_none_or(|t| t > now);
            if !stay { eprintln!("ferric-serve: unloaded {} (keep-alive elapsed)", s.m.name()); }
            stay
        });
    }

    /// The model for a request that names none: the command-line model, else the one used last.
    fn default_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.startup)
            .or_else(|| self.slots.iter().enumerate().max_by_key(|(_, s)| s.last).map(|(i, _)| i))
    }

    /// Unload idle models, least recently used first, until `bytes` more fit in both budgets.
    /// `Ok(false)` = not yet: what would have to go is mid-generation. Command-line models never go.
    ///
    /// ⚠ The memory budget counts model FILES. A KV cache grows with each sequence's context and is not
    /// in it; `max_bytes` defaults to 75% of physical memory to leave that room, not to measure it.
    fn room_for(&mut self, bytes: u64, opts: &ServeOpts) -> Result<bool, Refusal> {
        if bytes > opts.max_bytes {
            return Err((507, format!("this model's files are {:.1} GiB; the memory budget is {:.1} GiB (FERRIC_MAX_MEMORY)",
                                     bytes as f64 / 1073741824.0, opts.max_bytes as f64 / 1073741824.0)));
        }
        loop {
            let used: u64 = self.slots.iter().map(|s| s.bytes).sum();
            if self.slots.len() < opts.max_models.max(1) && used + bytes <= opts.max_bytes { return Ok(true); }
            let lru = self.slots.iter().enumerate().filter(|(_, s)| !s.busy() && !s.startup).min_by_key(|(_, s)| s.last).map(|(i, _)| i);
            match lru {
                Some(i) => { let s = self.slots.remove(i); eprintln!("ferric-serve: unloaded {} to make room", s.m.name()); }
                // Waiting only helps if a model that CAN go is busy now; otherwise the request would be
                // deferred forever, and the loop would spin retrying it.
                None if self.slots.iter().any(|s| s.busy() && !s.startup) => return Ok(false),
                None => return Err((503, format!("no room: the models named on the command line fill the budget \
                                                   ({} resident, --max-models {}); raise it or unload one with keep_alive 0",
                                                  self.slots.len(), opts.max_models))),
            }
        }
    }

    /// The slot for a named model, loading it if it is on disk. `Ok(None)` = wait for room.
    fn slot_for(&mut self, spec: Option<&str>, opts: &ServeOpts) -> Result<Option<usize>, Refusal> {
        let Some(spec) = spec else {
            return self.default_slot().map(Some).ok_or((400, "no model is loaded: name one in `model`".to_string()));
        };
        if let Some(i) = self.slots.iter().position(|s| s.answers_to(spec)) { return Ok(Some(i)); }
        let (key, bytes) = match self.src.resolve(spec) {
            Ok(k) => k,
            Err(why) => return match self.default_slot() {
                Some(i) if opts.fallback => Ok(Some(i)),
                _ => Err((404, why)),
            },
        };
        if let Some(i) = self.slots.iter().position(|s| s.key == key) { return Ok(Some(i)); }
        if !self.room_for(bytes, opts)? { return Ok(None); }
        let m = self.src.load(&key).map_err(|e| (500, format!("loading {spec}: {e}")))?;
        self.slots.push(Slot::new(Loaded { key, m, bytes }, opts.keep_alive, false, opts.max_batch));
        Ok(Some(self.slots.len() - 1))
    }

    /// Every model this server can answer for — loaded ones first — deduplicated by name.
    fn listing(&mut self) -> Vec<(String, Value, bool)> {
        let mut out: Vec<(String, Value, bool)> = Vec::new();
        for s in &self.slots { for (n, e) in s.m.cards() { if !out.iter().any(|o| o.0 == n) { out.push((n, e, true)); } } }
        for (n, e) in self.src.available() { if !out.iter().any(|o| o.0 == n) { out.push((n, e, false)); } }
        out
    }

    fn ps(&self) -> Value {
        let mut seen: Vec<String> = Vec::new();
        let mut models = Vec::new();
        for s in &self.slots {
            for (k, (n, mut e)) in s.m.cards().into_iter().enumerate() {
                if seen.contains(&n) { continue; }
                seen.push(n);
                // Ollama's spelling of "until evicted": a date no client will reach.
                e["expires_at"] = json!(match s.keep_alive {
                    None => "2318-01-01T00:00:00Z".to_string(),
                    Some(d) => crate::ollama::rfc3339(std::time::SystemTime::now() + (s.last + d).saturating_duration_since(std::time::Instant::now())),
                });
                e["size_vram"] = e["size"].clone();
                // The model's own card (the first); a shared embedder's context is not this model's.
                if k == 0 && s.m.context() > 0 { e["context_length"] = json!(s.m.context()); }
                models.push(e);
            }
        }
        json!({"models": models})
    }

    /// Answer one request: the endpoints about the server itself here, everything else on its model.
    /// Returns the job when it must wait for room.
    fn route(&mut self, mut j: Job, opts: &ServeOpts,
             serial: &mut impl FnMut(&S::M, &str, &str, &[u8], &[(String, String)], &mut TcpStream) -> bool) -> Option<Job> {
        if j.method == "OPTIONS" { write_preflight(&mut j.stream); return None; }
        if let Some(key) = &opts.api_key {
            if j.path != "/health" && !authorized(&j.headers, key) {
                write_json(&mut j.stream, 401, &json!({"error": {"message": "missing or wrong API key: send Authorization: Bearer <key> or x-api-key", "type": "authentication_error"}}));
                return None;
            }
        }
        let ollama = j.path.starts_with("/api/");
        let refuse = |s: &mut TcpStream, code: u16, m: &str| if ollama { write_json(s, code, &json!({"error": m})) }
            else { write_json(s, code, &json!({"error": {"message": m, "type": "invalid_request_error", "code": if code == 404 { "model_not_found" } else { "invalid_request" }}})) };
        match (j.method.as_str(), j.path.as_str()) {
            ("GET", "/health") => { write_json(&mut j.stream, 200, &json!({"status": "ok"})); return None; }
            ("GET", "/") | ("HEAD", "/") => {
                use std::io::Write;
                let b = b"ferric-serve is running (OpenAI /v1 and Ollama /api)";
                let _ = j.stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
                if j.method == "GET" { let _ = j.stream.write_all(b); }
                let _ = j.stream.flush();
                return None;
            }
            ("GET", "/api/version") => { write_json(&mut j.stream, 200, &json!({"version": crate::ollama::API_VERSION})); return None; }
            ("GET", "/api/tags") => {
                let models: Vec<Value> = self.listing().into_iter().map(|(_, e, _)| e).collect();
                write_json(&mut j.stream, 200, &json!({"models": models}));
                return None;
            }
            ("GET", "/api/ps") => { let v = self.ps(); write_json(&mut j.stream, 200, &v); return None; }
            ("GET", "/v1/models") => {
                // `loaded` is not OpenAI's; LM Studio's model list carries the same fact, and a picker can
                // show which answer at once and which will load first.
                let data: Vec<Value> = self.listing().into_iter().map(|(n, _, loaded)|
                    json!({"id": n, "object": "model", "created": now_unix(), "owned_by": "ferric", "loaded": loaded})).collect();
                write_json(&mut j.stream, 200, &json!({"object": "list", "data": data}));
                return None;
            }
            ("GET", "/metrics") => {
                let keys: Vec<&str> = self.slots.iter().map(|s| s.key.as_str()).collect();
                if let Some(body) = self.src.metrics(&keys) {
                    use std::io::Write;
                    let _ = j.stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes());
                    let _ = j.stream.write_all(body.as_bytes());
                    let _ = j.stream.flush();
                    return None;
                }
            }
            _ => {}
        }
        let req: Option<Value> = if j.method == "POST" { serde_json::from_slice(&j.body).ok() } else { None };
        let spec: Option<String> = req.as_ref().and_then(|r| r["model"].as_str().or_else(|| if j.path == "/api/show" { r["name"].as_str() } else { None }))
            .filter(|s| !s.is_empty()).map(String::from);
        let keep = match req.as_ref().map(keep_alive_of) {
            Some(Err(e)) => { refuse(&mut j.stream, 400, &e); return None; }
            Some(Ok(k)) => k,
            None => None,
        };
        // Ollama's load and unload requests: a generate with no prompt, a chat with no messages.
        let load_only = req.as_ref().is_some_and(|r| match j.path.as_str() {
            "/api/generate" => r["prompt"].as_str().is_none_or(str::is_empty) && r["images"].is_null(),
            "/api/chat" => r["messages"].as_array().is_none_or(|m| m.is_empty()),
            _ => false,
        });
        let answer_load = |s: &mut TcpStream, model: &str, reason: &str, chat: bool| {
            let mut v = json!({"model": model, "created_at": crate::ollama::now(), "done": true, "done_reason": reason});
            if chat { v["message"] = json!({"role": "assistant", "content": ""}); } else { v["response"] = json!(""); }
            write_json(s, 200, &v);
        };
        if load_only && keep == Some(Some(std::time::Duration::ZERO)) {
            let name = spec.clone().unwrap_or_default();
            if let Some(i) = spec.as_deref().and_then(|sp| self.slots.iter().position(|s| s.answers_to(sp))) {
                if self.slots[i].busy() { self.slots[i].keep_alive = Some(std::time::Duration::ZERO); }
                else { let s = self.slots.remove(i); eprintln!("ferric-serve: unloaded {} (keep_alive 0)", s.m.name()); }
            }
            answer_load(&mut j.stream, &name, "unload", j.path == "/api/chat");
            return None;
        }
        if j.path == "/api/show" {
            if let Some(sp) = spec.as_deref().filter(|sp| !self.slots.iter().any(|s| s.answers_to(sp))) {
                match self.src.resolve(sp).map(|(k, _)| self.src.show(&k)) {
                    Ok(Some(v)) => write_json(&mut j.stream, 200, &v),
                    Ok(None) => refuse(&mut j.stream, 404, &format!("model '{sp}' has no readable header")),
                    Err(why) => refuse(&mut j.stream, 404, &why),
                }
                return None;
            }
        }
        let i = match self.slot_for(spec.as_deref(), opts) {
            Ok(Some(i)) => i,
            Ok(None) => return Some(j),
            Err((code, msg)) => { refuse(&mut j.stream, code, &msg); return None; }
        };
        let s = &mut self.slots[i];
        if let Some(k) = keep { s.keep_alive = k; }
        s.last = std::time::Instant::now();
        if load_only {
            let name = s.m.name().to_string();
            answer_load(&mut j.stream, &name, "load", j.path == "/api/chat");
            return None;
        }
        route(&s.m, &mut s.sched, &mut s.gens, j, opts, serial);
        None
    }
}

/// Prompt tokens fed per step while other sequences decode (`FERRIC_PREFILL_CHUNK`, default 512 — llama.cpp's
/// default micro-batch). Smaller = smoother streams for everyone else, larger = the new request's first token
/// sooner.
fn prefill_chunk() -> usize {
    std::env::var("FERRIC_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n > 0).unwrap_or(512)
}

/// One scheduler step: admit + prefill new arrivals, decode everything running in ONE forward,
/// then flush whatever retired.
fn step<M: ServeModel>(m: &M, sched: &mut Scheduler, gens: &mut Vec<Gen<M::State>>) {
    let batch = sched.step_batch();
    let live: HashSet<SeqId> = batch.iter().copied().collect();

    // --- admission: prefill alone, because `decode` is the DECODE step ---
    //
    // ⭐ CHUNKED while anything is decoding: at most `chunk` prompt tokens per step, so a long prompt
    // arriving costs every in-flight stream one chunk's latency per token instead of its whole prefill
    // (a 4k-token prompt at ~500 tok/s froze every stream ~8 s). With nothing decoding there is no one to
    // stall, and the prompt goes in whole — one forward, the old path exactly.
    let decoding = gens.iter().any(|g| g.ready && live.contains(&g.id));
    let mut budget = if decoding { prefill_chunk() } else { usize::MAX };
    let mut first: Vec<(SeqId, u32, bool)> = Vec::new();
    for g in gens.iter_mut() {
        if !live.contains(&g.id) || g.ready { continue; }
        if budget == 0 { break; }
        if g.state.is_none() {
            g.ticket = m.energy_begin();
            let (st, skip) = m.begin(&g.prompt, &g.opts);
            g.state = Some(st);
            g.fed = skip;
        }
        let to = g.prompt.len().min(g.fed.saturating_add(budget));
        let last = to == g.prompt.len();
        let Some(st) = g.state.as_mut() else { continue };
        let row = m.feed(st, &g.prompt[g.fed..to], last);
        budget -= to - g.fed;
        g.fed = to;
        let Some(row) = row else { continue };
        g.ready = true;
        match m.pick(&row, &g.opts.sampling, &g.prompt, &g.r#gen, &mut g.rng) {
            Some(t) if !m.is_stop(t) => { g.next = t; let hit = g.commit(m, t, &row); first.push((g.id, t, hit)); }
            // Stop token (or a dead guide) on the very first sampled token: the serial path emits
            // nothing at all in that case, so neither does this one.
            _ => first.push((g.id, 0, true)),
        }
    }
    for (id, t, stop) in first { sched.record(id, t, stop); }

    // --- decode: one weight read serves every running sequence ---
    let running: HashSet<SeqId> = sched.running().iter().map(|s| s.id).collect();
    let mut idxs: Vec<usize> = Vec::new();
    let mut toks: Vec<u32> = Vec::new();
    let logits = {
        // `states` borrows into `gens`; the block scopes those borrows so the sampling pass below
        // can take `gens` mutably again.
        let mut states: Vec<&mut M::State> = Vec::new();
        for (i, g) in gens.iter_mut().enumerate() {
            // A sequence still mid-prefill has no token to feed yet.
            if !running.contains(&g.id) || !g.ready { continue; }
            let t = g.next;
            let Some(st) = g.state.as_mut() else { continue };
            idxs.push(i);
            toks.push(t);
            states.push(st);
        }
        if states.is_empty() { Vec::new() } else { m.decode(&toks, &mut states) }
    };
    if !logits.is_empty() {
        let nv = m.n_vocab();
        assert_eq!(logits.len(), idxs.len() * nv,
                   "decode returned {} logits for {} sequences × {nv} vocab — a row/sequence \
                    misalignment here returns fluent text for the WRONG request", logits.len(), idxs.len());
        let mut rec: Vec<(SeqId, u32, bool)> = Vec::new();
        for (row, &i) in idxs.iter().enumerate() {
            let g = &mut gens[i];
            let lrow = &logits[row * nv..(row + 1) * nv];
            match m.pick(lrow, &g.opts.sampling, &g.prompt, &g.r#gen, &mut g.rng) {
                Some(t) if !m.is_stop(t) => {
                    g.next = t;
                    let hit = g.commit(m, t, lrow);
                    // A client that disconnected (a failed stream write, or a closed socket seen every 16
                    // tokens for a non-streaming one) stops holding a batch slot.
                    if !g.gone && !g.streaming && g.r#gen.len() % 16 == 0 && crate::peer_closed(&g.stream) { g.gone = true; }
                    rec.push((g.id, t, hit || g.gone));
                }
                _ => rec.push((g.id, 0, true)),
            }
        }
        for (id, t, stop) in rec { sched.record(id, t, stop); }
    }

    // --- retirement: free the slot's state and answer the client on the SAME step it finished ---
    for (id, why) in sched.take_retired() {
        let Some(pos) = gens.iter().position(|g| g.id == id) else { continue };
        let g = gens.remove(pos);
        finish(m, g, why);
    }
}

/// Write the final response for a retired sequence and drop its socket. `finish_reason` is `"length"`
/// when the scheduler retired it at its token budget and `"stop"` for a stop token or stop string — the
/// same rule as the serial path, so a batched response stays indistinguishable from a serial one.
fn finish<M: ServeModel>(m: &M, mut g: Gen<M::State>, why: Done) {
    if let Some(st) = g.state.as_mut() {
        let fed: Vec<u32> = g.prompt.iter().chain(g.r#gen.iter()).copied().collect();
        m.remember(&fed, st);
    }
    if g.gone {
        m.cancelled();
        let _ = m.energy_end(g.ticket.take(), g.r#gen.len());
        return;
    }
    let reason = if g.em.hit_stop || !matches!(why, Done::Length) { "stop" } else { "length" };
    if let Some(d) = g.em.flush() { g.send_delta(m, &d); }
    let (ptok, gtok) = (g.prompt.len(), g.r#gen.len());
    let usage = json!({"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok});
    let energy = m.energy_end(g.ticket.take(), gtok);
    m.record(ptok, gtok, &energy);
    if g.streaming {
        send_sse(&mut g.stream, &json!({
            "id": "chatcmpl-ferric", "object": "chat.completion.chunk", "created": now_unix(),
            "model": m.name(),
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}], "energy": energy}));
        if g.include_usage {
            send_sse(&mut g.stream, &json!({"id": "chatcmpl-ferric", "object": "chat.completion.chunk",
                "created": now_unix(), "model": m.name(), "choices": [], "usage": usage}));
        }
        use std::io::Write;
        let _ = g.stream.write_all(b"data: [DONE]\n\n");
        let _ = g.stream.flush();
        return;
    }
    let mut choice = if g.chat {
        json!({"index": 0, "message": {"role": "assistant", "content": g.em.text}, "finish_reason": reason})
    } else {
        json!({"index": 0, "text": g.em.text, "finish_reason": reason})
    };
    if g.opts.logprobs { choice["logprobs"] = crate::logprobs_field(g.chat, &g.logprobs); }
    let body = if g.chat {
        json!({"id": "chatcmpl-ferric", "object": "chat.completion", "created": now_unix(), "model": m.name(),
               "choices": [choice], "usage": usage, "energy": energy})
    } else {
        json!({"id": format!("cmpl-ferric-{ptok}"), "object": "text_completion", "created": now_unix(), "model": m.name(),
               "choices": [choice], "usage": usage, "energy": energy})
    };
    write_json(&mut g.stream, 200, &body);
}

// ---------------------------------------------------------------------------------------------
// The real model
// ---------------------------------------------------------------------------------------------

impl ServeModel for Engine {
    type State = ModelCache;

    fn name(&self) -> &str { &self.name }
    fn n_vocab(&self) -> usize { self.model.n_vocab() }
    fn can_batch(&self) -> bool { self.batchable() }
    fn encode_chat(&self, messages: &[Value]) -> Result<Vec<u32>, String> { self.chat_ids(messages) }

    fn encode_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos { if let Some(b) = self.bos_id { ids.push(b); } }
        ids.extend(self.enc(text, true));
        ids
    }

    fn begin(&self, prompt: &[u32], opts: &GenOpts) -> (ModelCache, usize) { self.seeded_cache(prompt, &opts.lora) }
    fn gen_opts(&self, req: &Value, chat: bool) -> Result<GenOpts, String> { Engine::gen_opts(self, req, chat) }

    /// A chunk continues the cache exactly as a prompt-cache hit does (the suffix after a seeded prefix),
    /// so chunked and whole prefill feed the same tokens through the same cached forward.
    fn feed(&self, c: &mut ModelCache, toks: &[u32], last: bool) -> Option<Vec<f32>> {
        let v = pollster::block_on(self.model.forward_cached_last(toks, c).to_vec());
        let nv = self.model.n_vocab();
        last.then(|| v[v.len() - nv..].to_vec())
    }

    fn decode(&self, toks: &[u32], states: &mut [&mut ModelCache]) -> Vec<f32> {
        let nv = self.model.n_vocab();
        if !self.batchable() {
            // Serial fallback. `forward_batch` PANICS on a runtime whose batched path is not
            // solo-equivalent, so the fallback must reach the solo forward, not call the batched one
            // with N=1.
            let mut out = Vec::with_capacity(toks.len() * nv);
            for (i, &t) in toks.iter().enumerate() {
                let v = pollster::block_on(self.model.forward_cached(&[t], &mut *states[i]).to_vec());
                out.extend_from_slice(&v[v.len() - nv..]);
            }
            return out;
        }
        pollster::block_on(self.model.forward_batch(toks, states).to_vec())
    }

    fn pick(&self, row: &[f32], s: &Sampling, prompt: &[u32], generated: &[u32], rng: &mut u64) -> Option<u32> {
        self.select_token(row, &None, s, prompt, generated, rng)
    }

    fn is_stop(&self, tok: u32) -> bool { self.eos.contains(&tok) }
    fn text_of(&self, ids: &[u32]) -> String { self.detok(ids) }
    fn piece(&self, tok: u32) -> (String, Vec<u8>) { Engine::piece(self, tok) }
    fn budget(&self, prompt_len: usize, want: Option<usize>) -> Result<usize, String> { Engine::budget(self, prompt_len, want) }
    /// ⛔ Wiring batching sent plain requests on an MTP hybrid through the scheduler, which cannot
    /// draft — so speculative decoding, its energy gate and the one-slot prefix cache ran only for
    /// `response_format` and tool requests. Such a model keeps its own serial loop.
    fn serial_generation(&self) -> bool {
        (matches!(&self.model, crate::Model::Hybrid(m) if m.mtp.is_some()) && std::env::var("FERRIC_NOSPEC").is_err())
            // Prompt lookup (FERRIC_LOOKUP) speculates on the serial loop; a dense model that asked for it
            // takes that loop for every request, trading batching for tokens per forward.
            || (matches!(&self.model, crate::Model::Dense(_)) && crate::lookup_k().is_some() && crate::qwen3_cache_is_f32())
    }
    fn energy_begin(&self) -> Option<crate::energy::Ticket> { self.energy.begin() }
    fn cancelled(&self) { self.metrics.cancelled.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    fn remember(&self, tokens: &[u32], state: &mut ModelCache) { Engine::remember(self, tokens, state) }
    fn record(&self, prompt: usize, generated: usize, energy: &Value) { self.metrics.record(prompt, generated, energy) }
    fn energy_end(&self, t: Option<crate::energy::Ticket>, tokens: usize) -> Value { self.energy.end(t, tokens) }
    fn cards(&self) -> Vec<(String, Value)> { Engine::cards(self).iter().map(|c| (c.name.clone(), c.tag_entry())).collect() }
    fn context(&self) -> usize { self.n_ctx }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// A deterministic stand-in for a transformer.
    ///
    /// The point of it is **discrimination, not realism**: a sequence's next token is a hash of its
    /// OWN accumulated state, so if the loop ever pairs `toks[i]` with `states[j]`, or samples row
    /// `i` for sequence `j`, the emitted token stream changes. A mock whose output did not depend on
    /// per-sequence history would pass while the dispatch was crossing sequences — the exact
    /// "input distribution that cannot distinguish right from wrong" failure that has bitten this
    /// project before, and the reason the mock is a hash rather than a constant.
    struct Mock {
        batchable: bool,
        /// Applied once per `decode` call, regardless of batch width — which is the physical claim
        /// batching rests on: one weight read serves the whole batch.
        step_delay: Duration,
        /// Observability for the tests: how wide the widest single decode call was, and how many
        /// decode calls happened in total.
        widest: Arc<AtomicUsize>,
        calls: Arc<AtomicUsize>,
        /// If set, the sequence stops when it would emit this token.
        stop: u32,
        /// Which model this is. The salt enters every hash, so two models answer the same prompt
        /// differently and a request run on the wrong one is visible in its text.
        name: String,
        salt: u32,
        /// Prefill cost per prompt token (the physical fact chunking trades on), and chunks fed so far.
        prefill_per_token: Duration,
        prefills: Arc<AtomicUsize>,
    }

    /// A sequence's whole visible history. Keeping the full history (rather than a rolling hash)
    /// makes a cross-sequence leak change the output at the very next token instead of eventually.
    #[derive(Clone)]
    struct MockState { fed: Vec<u32> }

    fn hash(v: &[u32]) -> u32 { hash_salted(v, 0) }

    fn hash_salted(v: &[u32], salt: u32) -> u32 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ salt as u64;
        for &x in v { h ^= x as u64; h = h.wrapping_mul(0x1000_0000_01b3); }
        (h % 4096) as u32 + 1 // never 0, so `stop: 0` means "never stop"
    }

    impl Mock {
        fn new(batchable: bool) -> Mock {
            Mock { batchable, step_delay: Duration::ZERO,
                   widest: Arc::new(AtomicUsize::new(0)), calls: Arc::new(AtomicUsize::new(0)), stop: 0,
                   name: "mock".to_string(), salt: 0, prefill_per_token: Duration::ZERO, prefills: Arc::new(AtomicUsize::new(0)) }
        }
    }

    impl ServeModel for Mock {
        type State = MockState;
        fn name(&self) -> &str { &self.name }
        fn n_vocab(&self) -> usize { 8192 }
        fn can_batch(&self) -> bool { self.batchable }
        fn encode_chat(&self, messages: &[Value]) -> Result<Vec<u32>, String> {
            let mut out = Vec::new();
            for v in messages { out.extend(self.encode_text(&crate::genopts::content_text(&v["content"])?)); }
            Ok(out)
        }
        fn encode_text(&self, text: &str) -> Vec<u32> { text.bytes().map(|b| b as u32).collect() }
        fn begin(&self, _prompt: &[u32], _o: &GenOpts) -> (MockState, usize) { (MockState { fed: Vec::new() }, 0) }
        fn feed(&self, st: &mut MockState, toks: &[u32], last: bool) -> Option<Vec<f32>> {
            self.prefills.fetch_add(1, Ordering::SeqCst);
            if !self.prefill_per_token.is_zero() { std::thread::sleep(self.prefill_per_token * toks.len() as u32); }
            st.fed.extend_from_slice(toks);
            last.then(|| onehot(hash_salted(&st.fed, self.salt), self.n_vocab()))
        }
        fn decode(&self, toks: &[u32], states: &mut [&mut MockState]) -> Vec<f32> {
            assert_eq!(toks.len(), states.len(), "one token per sequence");
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.widest.fetch_max(toks.len(), Ordering::SeqCst);
            if !self.step_delay.is_zero() { std::thread::sleep(self.step_delay); }
            let mut out = Vec::with_capacity(toks.len() * self.n_vocab());
            for (i, &t) in toks.iter().enumerate() {
                states[i].fed.push(t);
                out.extend_from_slice(&onehot(hash_salted(&states[i].fed, self.salt), self.n_vocab()));
            }
            out
        }
        fn pick(&self, row: &[f32], _s: &Sampling, _p: &[u32], _g: &[u32], _rng: &mut u64) -> Option<u32> {
            Some(row.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32)
        }
        fn is_stop(&self, tok: u32) -> bool { self.stop != 0 && tok == self.stop }
        fn text_of(&self, ids: &[u32]) -> String {
            ids.iter().map(|i| format!("{i},")).collect()
        }
        fn piece(&self, tok: u32) -> (String, Vec<u8>) { let t = format!("{tok},"); (t.clone(), t.into_bytes()) }
        fn budget(&self, _p: usize, want: Option<usize>) -> Result<usize, String> { Ok(want.unwrap_or(256)) }
    }

    fn onehot(i: u32, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        v[i as usize % n] = 1.0;
        v
    }

    /// Start a server on an ephemeral port and return its address. The thread is deliberately not
    /// joined: `serve_loop` is a server and never returns, and the test process reaps it on exit.
    fn spawn(m: Mock, max_batch: usize) -> String {
        spawn_pool(Models::new(&[]), vec![Loaded { key: "mock".into(), m, bytes: 0 }], opts(max_batch))
    }

    fn opts(max_batch: usize) -> ServeOpts {
        ServeOpts { max_batch, any_mcp_tools: false, api_key: None, max_models: 3, max_bytes: u64::MAX, keep_alive: None, fallback: true }
    }

    fn spawn_pool(src: Models, initial: Vec<Loaded<Mock>>, o: ServeOpts) -> String {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = l.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            serve_loop(src, initial, l, o, |_m, _me, _p, _b, _h, _s| false);
        });
        addr
    }

    /// A model directory of mocks: `(name, bytes)`. Every load is counted. `broken` fails to load.
    struct Models { have: Vec<(String, u64)>, loads: Arc<AtomicUsize>, step_delay: Duration }

    impl Models {
        fn new(have: &[(&str, u64)]) -> Models {
            Models { have: have.iter().map(|(n, b)| (n.to_string(), *b)).collect(), loads: Arc::new(AtomicUsize::new(0)), step_delay: Duration::ZERO }
        }
    }

    fn mock_named(name: &str, delay: Duration) -> Mock {
        let mut m = Mock::new(true);
        m.name = name.to_string();
        m.salt = hash(&name.bytes().map(|b| b as u32).collect::<Vec<_>>());
        m.step_delay = delay;
        m
    }

    impl Source for Models {
        type M = Mock;
        fn resolve(&mut self, spec: &str) -> Result<(String, u64), String> {
            let want = spec.strip_suffix(":latest").unwrap_or(spec);
            self.have.iter().find(|(n, _)| n == want).cloned().ok_or(format!("model '{spec}' not found"))
        }
        fn load(&mut self, key: &str) -> Result<Mock, String> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            if key == "broken" { return Err("not a model".into()); }
            Ok(mock_named(key, self.step_delay))
        }
        fn available(&mut self) -> Vec<(String, Value)> {
            self.have.iter().map(|(n, b)| (n.clone(), json!({"name": format!("{n}:latest"), "model": format!("{n}:latest"), "size": b}))).collect()
        }
    }

    /// What model `name` answers to `prompt` on its own: the reference a pooled answer must equal.
    fn solo(name: &str, prompt: &str, n: usize) -> String {
        let addr = spawn_pool(Models::new(&[]), vec![Loaded { key: name.into(), m: mock_named(name, Duration::ZERO), bytes: 0 }], opts(1));
        text_of(&post(&addr, "/v1/completions", &json!({"prompt": prompt, "max_tokens": n}).to_string()))
    }

    fn get(addr: &str, path: &str) -> Value {
        let (code, b) = request(addr, "GET", path, "");
        assert_eq!(code, 200, "GET {path}: {b}");
        serde_json::from_str(&b).unwrap()
    }

    fn loaded(addr: &str) -> Vec<String> {
        let mut v: Vec<String> = get(addr, "/api/ps")["models"].as_array().unwrap().iter()
            .map(|m| m["name"].as_str().unwrap().trim_end_matches(":latest").to_string()).collect();
        v.sort();
        v
    }

    /// A real HTTP POST over a real socket. Returns the response body.
    fn post(addr: &str, path: &str, body: &str) -> String {
        let mut s = TcpStream::connect(addr).expect("connect");
        // A server that never answers must FAIL the test, not hang it (and CI with it).
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let req = format!("POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}", body.len());
        s.write_all(req.as_bytes()).unwrap();
        s.flush().unwrap();
        let mut r = BufReader::new(s);
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap() == 0 { break; }
            if line.trim().is_empty() { break; }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") { len = v.trim().parse().unwrap_or(0); }
        }
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf).unwrap();
        String::from_utf8_lossy(&buf).into_owned()
    }

    fn text_of(resp: &str) -> String {
        let v: Value = serde_json::from_str(resp).unwrap_or_else(|e| panic!("bad response {resp:?}: {e}"));
        v["choices"][0]["text"].as_str().or_else(|| v["choices"][0]["message"]["content"].as_str())
            .unwrap_or_else(|| panic!("no text in {resp}")).to_string()
    }

    /// A raw request; returns (status code, body).
    fn request(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
        let mut s = TcpStream::connect(addr).expect("connect");
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let req = format!("{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}", body.len());
        s.write_all(req.as_bytes()).unwrap();
        s.flush().unwrap();
        let mut r = BufReader::new(s);
        let mut status = String::new();
        r.read_line(&mut status).unwrap();
        let code: u16 = status.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap() == 0 { break; }
            if line.trim().is_empty() { break; }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") { len = v.trim().parse().unwrap_or(0); }
        }
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf).unwrap();
        (code, String::from_utf8_lossy(&buf).into_owned())
    }

    fn finish_of(resp: &str) -> String {
        let v: Value = serde_json::from_str(resp).unwrap();
        v["choices"][0]["finish_reason"].as_str().unwrap_or("").to_string()
    }

    /// `finish_reason` says WHY it stopped: `"length"` at the token budget, `"stop"` on a stop token.
    /// It was `"stop"` for both, so a client could not tell a complete answer from a truncated one.
    #[test]
    fn finish_reason_distinguishes_the_budget_from_a_stop_token() {
        let addr = spawn(Mock::new(true), 4);
        let r = post(&addr, "/v1/completions", &json!({"prompt": "alpha", "max_tokens": 5}).to_string());
        assert_eq!(finish_of(&r), "length");
        assert_eq!(text_of(&r).matches(',').count(), 5);
        // Stop on the third token the mock would emit.
        let third: u32 = text_of(&r).split(',').nth(2).unwrap().parse().unwrap();
        let mut m = Mock::new(true);
        m.stop = third;
        let addr = spawn(m, 4);
        let r = post(&addr, "/v1/completions", &json!({"prompt": "alpha", "max_tokens": 5}).to_string());
        assert_eq!(finish_of(&r), "stop");
        assert_eq!(text_of(&r).matches(',').count(), 2, "the stop token itself is not emitted");
    }

    /// A stop STRING cuts the text before it and ends the sequence, and the batched path cuts exactly
    /// where the serial one does.
    #[test]
    fn a_stop_string_ends_generation_identically_batched_and_serial() {
        let full = text_of(&post(&spawn(Mock::new(false), 1), "/v1/completions",
                                 &json!({"prompt": "bravo bravo", "max_tokens": 8}).to_string()));
        let toks: Vec<&str> = full.split(',').collect();
        let stop = format!("{},{}", toks[3], toks[4]); // spans a token boundary
        let body = json!({"prompt": "bravo bravo", "max_tokens": 8, "stop": [stop]}).to_string();
        let serial = post(&spawn(Mock::new(false), 1), "/v1/completions", &body);
        let batched = post(&spawn(Mock::new(true), 4), "/v1/completions", &body);
        // Where it FIRST occurs — the mock can repeat a token, so that may be earlier than index 3.
        let want = full[..full.find(&stop).expect("the stop string occurs in the full text")].to_string();
        assert!(!want.is_empty() && want.len() < full.len(), "the cut must land strictly inside the text");
        assert_eq!(text_of(&serial), want, "text must end right before the stop string");
        assert_eq!(text_of(&batched), text_of(&serial));
        assert_eq!(finish_of(&serial), "stop");
        assert_eq!(finish_of(&batched), "stop");
    }

    /// Content-part arrays are read (they were silently dropped — the model saw an empty message), an
    /// image part is refused by name, and a parameter this server cannot honour is a 400, not ignored.
    #[test]
    fn content_parts_are_read_and_unsupported_requests_are_refused() {
        let addr = spawn(Mock::new(true), 4);
        let plain = post(&addr, "/v1/chat/completions",
                         &json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}).to_string());
        let parts = post(&addr, "/v1/chat/completions",
                         &json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], "max_tokens": 4}).to_string());
        assert_eq!(text_of(&parts), text_of(&plain), "a one-part array must mean the same as the string");
        let empty = post(&addr, "/v1/chat/completions",
                         &json!({"messages": [{"role": "user", "content": ""}], "max_tokens": 4}).to_string());
        assert_ne!(text_of(&parts), text_of(&empty), "the parts must reach the model, not read as empty");
        // ⛔ An image never enters a batch: the batched prefill has no image to splice, so a vision model
        // would answer fluently about nothing. Every spelling goes to the serial path (vision, or its refusal).
        let o = ServeOpts { max_batch: 4, any_mcp_tools: false, api_key: None, max_models: 3, max_bytes: u64::MAX, keep_alive: None, fallback: true };
        for m in [json!({"role": "user", "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:x"}}]}),
                  json!({"role": "user", "content": [{"type": "input_image", "image_url": "data:x"}]}),
                  json!({"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}]}),
                  json!({"role": "user", "content": "what is this", "images": ["AAAA"]})] {
            assert!(must_run_serial(&json!({"messages": [m.clone()]}), true, &o), "batched: {m}");
        }
        assert!(!must_run_serial(&json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}), true, &o));
        let (c, b) = request(&addr, "POST", "/v1/completions", &json!({"prompt": "x", "n": 3}).to_string());
        assert_eq!(c, 400, "n=3 must be refused, not answered with one choice: {b}");
    }

    /// A client that disconnects mid-stream frees its batch slot. With ONE slot and a 5000-token request
    /// abandoned after its first chunk, the next request would otherwise wait out ~10 s of decode steps
    /// generated for nobody.
    #[test]
    fn a_disconnected_client_frees_its_slot() {
        let mut m = Mock::new(true);
        m.step_delay = Duration::from_millis(2);
        let addr = spawn(m, 1);
        {
            let mut s = TcpStream::connect(&addr).unwrap();
            let body = json!({"messages": [{"role": "user", "content": "long"}], "max_tokens": 5000, "stream": true}).to_string();
            s.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
            let mut first = [0u8; 256];
            let _ = s.read(&mut first);
        } // dropped: the client is gone
        let t0 = Instant::now();
        let r = post(&addr, "/v1/completions", &json!({"prompt": "next", "max_tokens": 4}).to_string());
        assert_eq!(text_of(&r).matches(',').count(), 4);
        assert!(t0.elapsed() < Duration::from_secs(3),
                "the next request waited {:?}: the abandoned one kept its slot", t0.elapsed());
    }

    /// A browser front-end sends OPTIONS before every JSON POST. It used to 404.
    #[test]
    fn cors_preflight_is_answered() {
        let addr = spawn(Mock::new(true), 4);
        let (c, _) = request(&addr, "OPTIONS", "/v1/chat/completions", "");
        assert_eq!(c, 204);
    }

    const PROMPTS: [&str; 4] = ["alpha", "bravo bravo", "c", "delta echo foxtrot"];

    fn fire_concurrently(addr: &str, max_tokens: usize) -> Vec<(String, Duration)> {
        let t0 = Instant::now();
        let handles: Vec<_> = PROMPTS.iter().map(|p| {
            let addr = addr.to_string();
            let p = p.to_string();
            std::thread::spawn(move || {
                let body = json!({"prompt": p, "max_tokens": max_tokens}).to_string();
                let r = post(&addr, "/v1/completions", &body);
                (text_of(&r), t0.elapsed())
            })
        }).collect();
        handles.into_iter().map(|h| h.join().expect("client thread")).collect()
    }

    /// **The correctness bar.** A batched response must be token-identical to a serial one for the
    /// same prompt, because a batched path that crossed sequences returns fluent text and no error.
    ///
    /// Both sides run the real `serve_loop` over real sockets; the only difference is
    /// `batchable()`, which is the gate `Model::supports_batching` feeds. The serial side is the
    /// reference, and it is a *different execution schedule*, not a copy of the batched result.
    #[test]
    fn batched_responses_are_identical_to_serial_ones() {
        let serial_addr = spawn(Mock::new(false), 8);
        let batched = Mock::new(true);
        let widest = batched.widest.clone();
        let batched_addr = spawn(batched, 4);

        // Serial reference: one at a time, so nothing can share a batch even in principle.
        let serial: Vec<String> = PROMPTS.iter().map(|p| {
            text_of(&post(&serial_addr, "/v1/completions", &json!({"prompt": p, "max_tokens": 12}).to_string()))
        }).collect();

        let got = fire_concurrently(&batched_addr, 12);
        let batched_texts: Vec<String> = got.into_iter().map(|(t, _)| t).collect();

        assert!(widest.load(Ordering::SeqCst) > 1,
                "the four concurrent requests never shared a decode call, so this test compared the \
                 serial path against itself and proved nothing");
        assert_eq!(batched_texts, serial,
                   "batching changed a response — it must be a scheduling change only");
        // Guard against the whole thing being trivially equal (e.g. every response empty).
        assert!(serial.iter().all(|s| s.matches(',').count() == 12),
                "each response must actually carry its 12 generated tokens: {serial:?}");
        assert!(serial[0] != serial[1] && serial[1] != serial[2],
                "different prompts must give different outputs, or this test cannot see a mix-up");
    }

    /// **The serialisation signature, and its absence.**
    ///
    /// Four concurrent requests against a server that queues produce a staircase: each completes one
    /// whole generation after the previous one, and the model is entered 4·N times. Against a server
    /// that batches, all four are advanced by the SAME decode call, so the model is entered ~N times
    /// and the four finish together.
    ///
    /// The primary assertions are on call counts and batch width, not on the clock — this machine is
    /// shared with three other build agents and wall-time is noise. The clock is checked only with a
    /// margin wide enough that it cannot flip on load.
    #[test]
    fn concurrent_requests_interleave_instead_of_queueing() {
        const N: usize = 24;

        let mut sm = Mock::new(false);
        sm.step_delay = Duration::from_millis(2);
        let serial_calls = sm.calls.clone();
        let serial_widest = sm.widest.clone();
        let serial_addr = spawn(sm, 8);

        let mut bm = Mock::new(true);
        bm.step_delay = Duration::from_millis(2);
        let batched_calls = bm.calls.clone();
        let batched_widest = bm.widest.clone();
        let batched_addr = spawn(bm, 4);

        let s = fire_concurrently(&serial_addr, N);
        let b = fire_concurrently(&batched_addr, N);

        let (sc, bc) = (serial_calls.load(Ordering::SeqCst), batched_calls.load(Ordering::SeqCst));
        let (sw, bw) = (serial_widest.load(Ordering::SeqCst), batched_widest.load(Ordering::SeqCst));

        // Each request needs N-1 decode steps (its first token comes from prefill).
        assert_eq!(sw, 1, "the serial fallback must never put two sequences in one forward");
        assert_eq!(sc, 4 * (N - 1), "serial: every sequence pays its own forward, {sc} calls");
        assert_eq!(bw, PROMPTS.len(),
                   "all {} concurrent requests must reach ONE decode call; widest seen was {bw}",
                   PROMPTS.len());
        assert!(bc <= 2 * (N - 1),
                "batched: {bc} decode calls for 4 requests of {N} tokens — that is queueing, not \
                 sharing (a fully shared batch is {} calls)", N - 1);

        // The staircase, stated on the clock. Serialised completions are spread by a whole
        // generation each; shared ones land together.
        let spread = |v: &[(String, Duration)]| {
            let mx = v.iter().map(|(_, d)| *d).max().unwrap();
            let mn = v.iter().map(|(_, d)| *d).min().unwrap();
            (mx, mn, mx - mn)
        };
        let (s_last, _, s_spread) = spread(&s);
        let (b_last, _, b_spread) = spread(&b);
        eprintln!("serial : last={s_last:?} spread={s_spread:?} calls={sc} widest={sw}");
        eprintln!("batched: last={b_last:?} spread={b_spread:?} calls={bc} widest={bw}");
        assert!(b_spread * 3 < s_spread,
                "batched completions are still spread like a queue: batched {b_spread:?} vs serial {s_spread:?}");
    }

    /// A slot freed by a finished sequence is taken by a request that arrives LATER — the property
    /// that separates continuous batching from static batching. With `max_batch` 2 and four
    /// requests, static batching would run 2, drain, then run 2; continuous batching refills.
    #[test]
    fn a_late_request_joins_a_batch_already_in_flight() {
        let mut m = Mock::new(true);
        m.step_delay = Duration::from_millis(3);
        let widest = m.widest.clone();
        let addr = spawn(m, 2);

        // Two long requests occupy both slots.
        let a = addr.clone();
        let long = std::thread::spawn(move || {
            let h: Vec<_> = (0..2).map(|i| {
                let a = a.clone();
                std::thread::spawn(move || post(&a, "/v1/completions", &json!({"prompt": format!("long{i}"), "max_tokens": 60}).to_string()))
            }).collect();
            h.into_iter().map(|x| x.join().unwrap()).collect::<Vec<_>>()
        });
        std::thread::sleep(Duration::from_millis(40)); // both are mid-decode by now
        let late = post(&addr, "/v1/completions", &json!({"prompt": "late", "max_tokens": 4}).to_string());
        let longs = long.join().unwrap();

        assert_eq!(widest.load(Ordering::SeqCst), 2, "the two long requests must have shared a batch");
        assert_eq!(text_of(&late).matches(',').count(), 4, "the late request must have completed");
        for l in &longs { assert_eq!(text_of(l).matches(',').count(), 60, "a long request was cut short"); }
    }

    /// `Engine::batchable` is the ONLY consumer of `Model::supports_batching`, and the gate is
    /// useless if it stops being consulted. A hardcoded `true` there would compile, serve fluent
    /// text, and silently batch a runtime whose batched path is not solo-equivalent.
    ///
    /// This reads the function's body by name — never the whole file — for the same reason the
    /// `batching_support` guard does: a test that searches a file for a literal it also declares is
    /// a tautology. Deleting the call and returning `true` is a mutation that COMPILES, so this
    /// assertion is not subsumed by the compiler.
    // ---- several models ----------------------------------------------------------------------

    #[test]
    fn a_long_prompt_is_prefilled_in_chunks_while_others_stream() {
        // 1 ms per prompt token: a 2,000-token prompt prefilled whole freezes every stream for ~2 s.
        let mk = || { let mut m = Mock::new(true); m.step_delay = Duration::from_millis(2); m.prefill_per_token = Duration::from_millis(1); m };
        let long: String = (0..2000).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let want = { let addr = spawn(mk(), 4); text_of(&post(&addr, "/v1/completions", &json!({"prompt": long, "max_tokens": 6}).to_string())) };
        let m = mk();
        let prefills = m.prefills.clone();
        let addr = spawn(m, 4);
        // A streams; its deltas are timestamped as they arrive.
        let a = {
            let addr = addr.clone();
            std::thread::spawn(move || {
                let mut s = TcpStream::connect(&addr).unwrap();
                s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
                let body = json!({"messages": [{"role": "user", "content": "go"}], "max_tokens": 700, "stream": true}).to_string();
                s.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
                let mut times = Vec::new();
                for line in BufReader::new(s).lines() {
                    let Ok(line) = line else { break };
                    if line.starts_with("data: {") { times.push(Instant::now()); }
                    if line == "data: [DONE]" { break; }
                }
                times
            })
        };
        std::thread::sleep(Duration::from_millis(200)); // A is decoding
        let before = prefills.load(Ordering::SeqCst);
        let got = text_of(&post(&addr, "/v1/completions", &json!({"prompt": long, "max_tokens": 6}).to_string()));
        let chunks = prefills.load(Ordering::SeqCst) - before;
        let times = a.join().unwrap();
        let gap = times.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert_eq!(got, want, "a prompt fed in chunks must answer exactly as one fed whole");
        assert!(chunks >= 4, "2,000 tokens at 512 per step is 4 chunks, saw {chunks}");
        assert!(gap < Duration::from_millis(1000), "the stream froze for {gap:?} while the long prompt prefilled");
    }

    #[test]
    fn keep_alive_reads_ollamas_spellings() {
        use std::time::Duration as D;
        let k = |v: Value| keep_alive_of(&json!({"keep_alive": v}));
        assert_eq!(k(json!("5m")), Ok(Some(Some(D::from_secs(300)))));
        assert_eq!(k(json!("1h30m")), Ok(Some(Some(D::from_secs(5400)))));
        assert_eq!(k(json!("250ms")), Ok(Some(Some(D::from_millis(250)))));
        assert_eq!(k(json!(0)), Ok(Some(Some(D::ZERO))));
        assert_eq!(k(json!("0")), Ok(Some(Some(D::ZERO))));
        assert_eq!(k(json!(-1)), Ok(Some(None)), "negative = until evicted");
        assert_eq!(k(json!("-1m")), Ok(Some(None)));
        assert_eq!(k(json!(90)), Ok(Some(Some(D::from_secs(90)))));
        assert_eq!(keep_alive_of(&json!({"ttl": 60})), Ok(Some(Some(D::from_secs(60)))), "LM Studio's ttl");
        assert_eq!(keep_alive_of(&json!({})), Ok(None));
        assert!(k(json!("5x")).unwrap_err().contains("unknown unit"));
        assert!(k(json!("soon")).is_err());
        assert!(k(json!([1])).is_err());
    }

    #[test]
    fn requests_run_on_the_model_they_name_even_interleaved() {
        let addr = spawn_pool(Models::new(&[("alpha", 1), ("beta", 1)]), vec![], ServeOpts { fallback: false, ..opts(4) });
        let (a, b) = (solo("alpha", "same prompt", 24), solo("beta", "same prompt", 24));
        assert_ne!(a, b, "the two mocks must answer differently or this test cannot see a misroute");
        let hs: Vec<_> = (0..8).map(|i| {
            let addr = addr.clone();
            let name = if i % 2 == 0 { "alpha" } else { "beta" };
            std::thread::spawn(move || (name, post(&addr, "/v1/completions", &json!({"model": name, "prompt": "same prompt", "max_tokens": 24}).to_string())))
        }).collect();
        for h in hs {
            let (name, r) = h.join().unwrap();
            let v: Value = serde_json::from_str(&r).unwrap();
            assert_eq!(v["model"], json!(name), "{r}");
            assert_eq!(text_of(&r), if name == "alpha" { a.clone() } else { b.clone() }, "{name} answered with another model's text");
        }
        assert_eq!(loaded(&addr), vec!["alpha", "beta"]);
    }

    #[test]
    fn an_unknown_model_is_a_404_unless_the_server_was_started_with_one() {
        let bare = spawn_pool(Models::new(&[("alpha", 1)]), vec![], ServeOpts { fallback: false, ..opts(4) });
        let (code, body) = request(&bare, "POST", "/v1/completions", &json!({"model": "gpt-4o", "prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(code, 404, "{body}");
        assert!(body.contains("model_not_found") && body.contains("gpt-4o"), "{body}");
        let (code, body) = request(&bare, "POST", "/api/generate", &json!({"model": "gpt-4o", "prompt": "x"}).to_string());
        assert_eq!(code, 404);
        assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_string(), "the Ollama dialect's error is a string: {body}");
        let (code, _) = request(&bare, "POST", "/v1/completions", &json!({"prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(code, 400, "no model named and none loaded");
        // With a model loaded there IS something to fall back to; a server started without one still refuses.
        post(&bare, "/v1/completions", &json!({"model": "alpha", "prompt": "x", "max_tokens": 1}).to_string());
        let (code, body) = request(&bare, "POST", "/v1/completions", &json!({"model": "gpt-4o", "prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(code, 404, "an unknown name was answered by the loaded model: {body}");
        // Started with a model: an unmatched name is answered by it, and the response says who answered.
        let started = spawn_pool(Models::new(&[]), vec![Loaded { key: "mock".into(), m: Mock::new(true), bytes: 0 }], opts(4));
        let r = post(&started, "/v1/completions", &json!({"model": "gpt-4o", "prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(serde_json::from_str::<Value>(&r).unwrap()["model"], json!("mock"));
    }

    #[test]
    fn ollama_loads_on_an_empty_request_and_unloads_on_keep_alive_zero() {
        let src = Models::new(&[("alpha", 1)]);
        let loads = src.loads.clone();
        let addr = spawn_pool(src, vec![], ServeOpts { fallback: false, ..opts(4) });
        assert!(loaded(&addr).is_empty());
        let v: Value = serde_json::from_str(&post(&addr, "/api/generate", &json!({"model": "alpha"}).to_string())).unwrap();
        assert_eq!((v["done_reason"].as_str(), v["done"].as_bool()), (Some("load"), Some(true)));
        assert_eq!(loaded(&addr), vec!["alpha"]);
        let v: Value = serde_json::from_str(&post(&addr, "/api/chat", &json!({"model": "alpha:latest", "messages": []}).to_string())).unwrap();
        assert_eq!(v["done_reason"], json!("load"));
        assert_eq!(loads.load(Ordering::SeqCst), 1, "an already-loaded model is not loaded again");
        let v: Value = serde_json::from_str(&post(&addr, "/api/generate", &json!({"model": "alpha", "keep_alive": 0}).to_string())).unwrap();
        assert_eq!(v["done_reason"], json!("unload"));
        assert!(loaded(&addr).is_empty(), "keep_alive 0 unloads");
    }

    #[test]
    fn an_idle_model_is_unloaded_when_its_keep_alive_runs_out() {
        let addr = spawn_pool(Models::new(&[("alpha", 1), ("beta", 1)]), vec![], ServeOpts { fallback: false, ..opts(4) });
        let t0 = Instant::now();
        post(&addr, "/v1/completions", &json!({"model": "alpha", "prompt": "x", "max_tokens": 2, "keep_alive": "400ms"}).to_string());
        post(&addr, "/v1/completions", &json!({"model": "beta", "prompt": "x", "max_tokens": 2, "keep_alive": -1}).to_string());
        assert_eq!(loaded(&addr), vec!["alpha", "beta"]);
        let ps = get(&addr, "/api/ps");
        assert!(ps["models"].as_array().unwrap().iter().any(|m| m["expires_at"].as_str().is_some_and(|e| e.starts_with("2318"))), "{ps}");
        // Nothing arrives while the keep-alive runs out, so only the idle wait's own deadline can unload it:
        // a request would wake the loop and hide a wait that never woke (polling here did exactly that).
        // `/api/ps` is answered BEFORE the loop's expiry pass, so this one look cannot trigger it either.
        std::thread::sleep(Duration::from_millis(1200));
        assert_eq!(loaded(&addr), vec!["beta"], "alpha outlived its keep-alive with the server idle");
        assert!(t0.elapsed() >= Duration::from_millis(400));
    }

    #[test]
    fn a_command_line_model_is_never_evicted_for_room() {
        let addr = spawn_pool(Models::new(&[("alpha", 1), ("beta", 1)]), vec![Loaded { key: "mock".into(), m: Mock::new(true), bytes: 1 }],
                              ServeOpts { max_models: 2, ..opts(4) });
        for m in ["alpha", "mock", "beta"] {
            post(&addr, "/v1/completions", &json!({"model": m, "prompt": "x", "max_tokens": 1}).to_string());
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(loaded(&addr), vec!["beta", "mock"], "the command-line model must stay; alpha was the one to go");
        let r = post(&addr, "/v1/completions", &json!({"model": "gpt-4o", "prompt": "x", "max_tokens": 1}).to_string());
        assert_eq!(serde_json::from_str::<Value>(&r).unwrap()["model"], json!("mock"), "the fallback is the command-line model");
        // Only command-line models stand in the way and nothing is generating: refuse, don't defer forever.
        let full = spawn_pool(Models::new(&[("alpha", 1)]), vec![Loaded { key: "mock".into(), m: Mock::new(true), bytes: 1 }],
                              ServeOpts { max_models: 1, ..opts(4) });
        let (code, body) = request(&full, "POST", "/v1/completions", &json!({"model": "alpha", "prompt": "x", "max_tokens": 1}).to_string());
        assert_eq!(code, 503, "{body}");
    }

    #[test]
    fn the_least_recently_used_idle_model_makes_room() {
        let addr = spawn_pool(Models::new(&[("alpha", 1), ("beta", 1), ("gamma", 1)]), vec![], ServeOpts { fallback: false, max_models: 2, ..opts(4) });
        for m in ["alpha", "beta", "alpha"] {
            post(&addr, "/v1/completions", &json!({"model": m, "prompt": "x", "max_tokens": 2}).to_string());
            std::thread::sleep(Duration::from_millis(5));
        }
        post(&addr, "/v1/completions", &json!({"model": "gamma", "prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(loaded(&addr), vec!["alpha", "gamma"], "beta was the least recently used");
    }

    #[test]
    fn a_generating_model_is_never_evicted_the_new_request_waits_for_it() {
        let mut src = Models::new(&[("alpha", 1), ("beta", 1)]);
        src.step_delay = Duration::from_millis(3);
        let addr = spawn_pool(src, vec![], ServeOpts { fallback: false, max_models: 1, ..opts(4) });
        let want = solo("alpha", "long one", 120);
        let a = { let addr = addr.clone(); std::thread::spawn(move || post(&addr, "/v1/completions", &json!({"model": "alpha", "prompt": "long one", "max_tokens": 120}).to_string())) };
        std::thread::sleep(Duration::from_millis(60));
        let rb = post(&addr, "/v1/completions", &json!({"model": "beta", "prompt": "x", "max_tokens": 3}).to_string());
        assert_eq!(text_of(&a.join().unwrap()), want, "alpha's answer was cut short or corrupted by the load of beta");
        assert_eq!(text_of(&rb).matches(',').count(), 3);
        assert_eq!(loaded(&addr), vec!["beta"]);
    }

    #[test]
    fn a_model_over_the_memory_budget_or_failing_to_load_is_refused_and_the_server_lives() {
        let addr = spawn_pool(Models::new(&[("alpha", 1), ("huge", 1000), ("broken", 1)]), vec![], ServeOpts { fallback: false, max_bytes: 100, ..opts(4) });
        let (code, body) = request(&addr, "POST", "/v1/completions", &json!({"model": "huge", "prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(code, 507, "{body}");
        let (code, body) = request(&addr, "POST", "/v1/completions", &json!({"model": "broken", "prompt": "x", "max_tokens": 2}).to_string());
        assert_eq!(code, 500, "{body}");
        assert!(body.contains("not a model"), "{body}");
        assert_eq!(text_of(&post(&addr, "/v1/completions", &json!({"model": "alpha", "prompt": "x", "max_tokens": 2}).to_string())).matches(',').count(), 2);
    }

    #[test]
    fn model_lists_show_everything_on_disk_and_say_what_is_loaded() {
        let addr = spawn_pool(Models::new(&[("alpha", 1), ("beta", 1)]), vec![], ServeOpts { fallback: false, ..opts(4) });
        post(&addr, "/v1/completions", &json!({"model": "beta", "prompt": "x", "max_tokens": 1}).to_string());
        let m = get(&addr, "/v1/models");
        let ids: Vec<(String, bool)> = m["data"].as_array().unwrap().iter().map(|d| (d["id"].as_str().unwrap().to_string(), d["loaded"].as_bool().unwrap())).collect();
        assert_eq!(ids, vec![("beta".to_string(), true), ("alpha".to_string(), false)]);
        let tags = get(&addr, "/api/tags");
        assert_eq!(tags["models"].as_array().unwrap().len(), 2, "{tags}");
    }

    #[test]
    fn engine_batchable_actually_consults_the_runtime_gate() {
        let src: &str = include_str!("lib.rs");
        let start = src.find("pub(crate) fn batchable(&self) -> bool {")
            .expect("Engine::batchable was renamed; this guard reads its body by name");
        let rest = &src[start..];
        let end = rest.find("\n    }").expect("unterminated batchable body");
        let body = &rest[..end];
        assert!(!body.contains("fn engine_batchable_actually"), "the extracted body swallowed this test");
        assert!(body.contains("self.model.supports_batching()"),
                "Engine::batchable stopped asking Model::supports_batching, so the per-runtime \
                 batching gate has no consumer again.\nbody was:\n{body}");
    }
}
