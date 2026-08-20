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

use crate::{Engine, ModelCache, write_json, write_sse_headers, send_sse, now_unix, read_request};
use ferric_llama::sched::{Scheduler, SeqId};
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

    /// Tokenize a chat request's `messages`.
    fn encode_chat(&self, messages: &[Value]) -> Vec<u32>;
    /// Tokenize a `/v1/completions` prompt string.
    fn encode_text(&self, text: &str) -> Vec<u32>;

    /// Prefill a prompt into a fresh state. Returns the state and the **last** row of logits — the
    /// row the first sampled token comes from.
    ///
    /// Prefill is per-sequence on purpose: `forward_batch` is the DECODE step (one token per
    /// sequence), so a newly admitted request builds its cache alone and joins the batch from its
    /// second token onward. Chunked prefill interleaved into the decode batch is the next lever and
    /// is not attempted here.
    fn prefill(&self, prompt: &[u32]) -> (Self::State, Vec<f32>);

    /// One decode step for N sequences: `toks[i]` advances `states[i]`. Returns N × `n_vocab`
    /// logits, row `i` belonging to sequence `i`.
    fn decode(&self, toks: &[u32], states: &mut [&mut Self::State]) -> Vec<f32>;

    /// Sample one token from one row. `None` = stop with nothing further emitted.
    fn pick(&self, row: &[f32], temperature: f32, rng: &mut u64) -> Option<u32>;
    fn is_stop(&self, tok: u32) -> bool;
    fn text_of(&self, ids: &[u32]) -> String;
}

/// One HTTP request, parsed off its socket by a reader thread and handed to the engine thread.
pub(crate) struct Job {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
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
    temperature: f32,
    /// Per-sequence RNG, seeded identically to the serial path so sampled output is reproducible
    /// **and** independent of what else happens to be in the batch. A shared RNG would make a
    /// request's output depend on its neighbours, which is exactly the cross-sequence coupling the
    /// batched forward is verified not to have.
    rng: u64,
    prompt: Vec<u32>,
    gen: Vec<u32>,
    emitted: String,
    state: Option<S>,
    /// The token to feed on the next decode step.
    next: u32,
}

impl<S> Gen<S> {
    /// Commit one token: record it, and stream the newly-decoded suffix if the client is streaming.
    /// Byte-for-byte the same delta logic as the serial `Engine::generate`, including the
    /// char-boundary guard that keeps multi-byte UTF-8 from being split across SSE frames.
    fn commit<M: ServeModel<State = S>>(&mut self, m: &M, tok: u32) {
        self.gen.push(tok);
        let full = m.text_of(&self.gen);
        if full.len() > self.emitted.len() && full.is_char_boundary(self.emitted.len()) {
            let delta = full[self.emitted.len()..].to_string();
            if self.streaming {
                send_sse(&mut self.stream, &json!({
                    "id": "chatcmpl-ferric", "object": "chat.completion.chunk", "created": now_unix(),
                    "model": m.name(),
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": Value::Null}]}));
            }
            self.emitted = full;
        }
    }
}

/// Server knobs the loop needs that the model does not know about.
pub(crate) struct ServeOpts {
    pub max_batch: usize,
    /// Any MCP server is connected, so every chat request advertises tools and must take the
    /// multi-round serial agent path.
    pub any_mcp_tools: bool,
}

/// Requests the batch loop declines, and hands to the untouched serial path. See the module docs.
fn must_run_serial(req: &Value, chat: bool, opts: &ServeOpts) -> bool {
    if !req["response_format"]["type"].is_null() { return true; }
    if chat && (opts.any_mcp_tools || req["tools"].as_array().is_some_and(|t| !t.is_empty())) { return true; }
    false
}

/// The server. Owns the model on this thread forever; the listener is drained by a spawned accept
/// thread. `serial` handles everything the batch loop declines (guided decoding, the tool loop,
/// embeddings, unknown paths) and returns whether it recognised the request.
pub(crate) fn serve_loop<M: ServeModel>(
    m: M,
    listener: TcpListener,
    opts: ServeOpts,
    mut serial: impl FnMut(&M, &str, &str, &[u8], &mut TcpStream) -> bool,
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
                    if let Some((method, path, body)) = read_request(&mut s) {
                        ib2.push(Job { method, path, body, stream: s });
                    }
                });
            }
            ib.close();
        });
    }

    // A model that cannot batch runs the SAME loop with one slot. There is no second code path to
    // rot: the fallback differs only in `max_batch` and in `decode` taking the solo forward.
    let max_batch = if m.can_batch() { opts.max_batch.max(1) } else { 1 };
    let mut sched = Scheduler::new(max_batch);
    let mut gens: Vec<Gen<M::State>> = Vec::new();

    loop {
        // Block only when there is nothing in flight; otherwise take whatever has arrived and keep
        // stepping. This is the continuous-batching admission point.
        let jobs = if gens.is_empty() {
            match inbox.wait() { Some(j) => j, None => return }
        } else {
            inbox.drain()
        };
        for j in jobs { route(&m, &mut sched, &mut gens, j, &opts, &mut serial); }
        if gens.is_empty() { continue; }

        step(&m, &mut sched, &mut gens);
    }
}

/// Dispatch one parsed request: fast endpoints inline, generation into the scheduler, everything
/// else to the serial handler.
fn route<M: ServeModel>(
    m: &M,
    sched: &mut Scheduler,
    gens: &mut Vec<Gen<M::State>>,
    mut j: Job,
    opts: &ServeOpts,
    serial: &mut impl FnMut(&M, &str, &str, &[u8], &mut TcpStream) -> bool,
) {
    let chat = j.path == "/v1/chat/completions";
    let is_gen = chat || j.path == "/v1/completions";
    if j.method == "POST" && is_gen {
        let req: Value = match serde_json::from_slice(&j.body) {
            Ok(v) => v,
            Err(e) => return write_json(&mut j.stream, 400, &json!({"error": {"message": format!("bad json: {e}")}})),
        };
        if must_run_serial(&req, chat, opts) {
            if !serial(m, &j.method, &j.path, &j.body, &mut j.stream) {
                write_json(&mut j.stream, 404, &json!({"error": {"message": "not found", "type": "invalid_request_error"}}));
            }
            return;
        }
        let prompt = if chat {
            let empty = vec![];
            m.encode_chat(req["messages"].as_array().unwrap_or(&empty))
        } else {
            m.encode_text(req["prompt"].as_str().unwrap_or(""))
        };
        let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
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
            temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
            // Same fixed seed as `Engine::generate`, per sequence.
            rng: 0x2545_F491_4F6C_DD1D,
            prompt, gen: Vec::new(), emitted: String::new(), state: None, next: 0,
        });
        return;
    }
    if !serial(m, &j.method, &j.path, &j.body, &mut j.stream) {
        write_json(&mut j.stream, 404, &json!({"error": {"message": "not found", "type": "invalid_request_error"}}));
    }
}

/// One scheduler step: admit + prefill new arrivals, decode everything running in ONE forward,
/// then flush whatever retired.
fn step<M: ServeModel>(m: &M, sched: &mut Scheduler, gens: &mut Vec<Gen<M::State>>) {
    let batch = sched.step_batch();
    let live: HashSet<SeqId> = batch.iter().copied().collect();

    // --- admission: prefill alone, because `decode` is the DECODE step ---
    let mut first: Vec<(SeqId, u32, bool)> = Vec::new();
    for g in gens.iter_mut() {
        if !live.contains(&g.id) || g.state.is_some() { continue; }
        let (st, row) = m.prefill(&g.prompt);
        g.state = Some(st);
        match m.pick(&row, g.temperature, &mut g.rng) {
            Some(t) if !m.is_stop(t) => { g.next = t; g.commit(m, t); first.push((g.id, t, false)); }
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
            if !running.contains(&g.id) { continue; }
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
            match m.pick(&logits[row * nv..(row + 1) * nv], g.temperature, &mut g.rng) {
                Some(t) if !m.is_stop(t) => { g.next = t; g.commit(m, t); rec.push((g.id, t, false)); }
                _ => rec.push((g.id, 0, true)),
            }
        }
        for (id, t, stop) in rec { sched.record(id, t, stop); }
    }

    // --- retirement: free the slot's state and answer the client on the SAME step it finished ---
    for (id, _why) in sched.take_retired() {
        let Some(pos) = gens.iter().position(|g| g.id == id) else { continue };
        let g = gens.remove(pos);
        finish(m, g);
    }
}

/// Write the final response for a retired sequence and drop its socket.
///
/// ⚠ `finish_reason` is hardcoded `"stop"` because **the serial path hardcodes it too**
/// (`chat`/`completions` in `lib.rs`). It is wrong for a length-limited generation, which OpenAI
/// spells `"length"` — but the property under test here is that a batched response is
/// indistinguishable from a serial one, so this path mirrors the existing behaviour rather than
/// diverging from it. Fixing it is a one-line change in three places and belongs in its own commit,
/// where the change is visible instead of hidden inside a batching diff.
fn finish<M: ServeModel>(m: &M, mut g: Gen<M::State>) {
    let (ptok, gtok) = (g.prompt.len(), g.gen.len());
    if g.streaming {
        send_sse(&mut g.stream, &json!({
            "id": "chatcmpl-ferric", "object": "chat.completion.chunk", "created": now_unix(),
            "model": m.name(),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}));
        use std::io::Write;
        let _ = g.stream.write_all(b"data: [DONE]\n\n");
        let _ = g.stream.flush();
        return;
    }
    let body = if g.chat {
        json!({
            "id": "chatcmpl-ferric", "object": "chat.completion", "created": now_unix(), "model": m.name(),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": g.emitted}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok}
        })
    } else {
        json!({
            "id": format!("cmpl-ferric-{ptok}"), "object": "text_completion", "created": now_unix(), "model": m.name(),
            "choices": [{"index": 0, "text": g.emitted, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": gtok, "total_tokens": ptok + gtok}
        })
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
    fn encode_chat(&self, messages: &[Value]) -> Vec<u32> { self.chat_ids(messages) }

    fn encode_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos { if let Some(b) = self.bos_id { ids.push(b); } }
        ids.extend(self.enc(text, true));
        ids
    }

    fn prefill(&self, prompt: &[u32]) -> (ModelCache, Vec<f32>) {
        let mut c = self.model.new_cache();
        let v = pollster::block_on(self.model.forward_cached(prompt, &mut c).to_vec());
        let nv = self.model.n_vocab();
        let row = v[v.len() - nv..].to_vec();
        (c, row)
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

    fn pick(&self, row: &[f32], temperature: f32, rng: &mut u64) -> Option<u32> {
        self.select_token(row, &None, temperature, rng)
    }

    fn is_stop(&self, tok: u32) -> bool { self.eos.contains(&tok) }
    fn text_of(&self, ids: &[u32]) -> String { self.detok(ids) }
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
    }

    /// A sequence's whole visible history. Keeping the full history (rather than a rolling hash)
    /// makes a cross-sequence leak change the output at the very next token instead of eventually.
    #[derive(Clone)]
    struct MockState { fed: Vec<u32> }

    fn hash(v: &[u32]) -> u32 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in v { h ^= x as u64; h = h.wrapping_mul(0x1000_0000_01b3); }
        (h % 4096) as u32 + 1 // never 0, so `stop: 0` means "never stop"
    }

    impl Mock {
        fn new(batchable: bool) -> Mock {
            Mock { batchable, step_delay: Duration::ZERO,
                   widest: Arc::new(AtomicUsize::new(0)), calls: Arc::new(AtomicUsize::new(0)), stop: 0 }
        }
    }

    impl ServeModel for Mock {
        type State = MockState;
        fn name(&self) -> &str { "mock" }
        fn n_vocab(&self) -> usize { 8192 }
        fn can_batch(&self) -> bool { self.batchable }
        fn encode_chat(&self, messages: &[Value]) -> Vec<u32> {
            messages.iter().flat_map(|v| self.encode_text(v["content"].as_str().unwrap_or(""))).collect()
        }
        fn encode_text(&self, text: &str) -> Vec<u32> { text.bytes().map(|b| b as u32).collect() }
        fn prefill(&self, prompt: &[u32]) -> (MockState, Vec<f32>) {
            let st = MockState { fed: prompt.to_vec() };
            (st.clone(), onehot(hash(&st.fed), self.n_vocab()))
        }
        fn decode(&self, toks: &[u32], states: &mut [&mut MockState]) -> Vec<f32> {
            assert_eq!(toks.len(), states.len(), "one token per sequence");
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.widest.fetch_max(toks.len(), Ordering::SeqCst);
            if !self.step_delay.is_zero() { std::thread::sleep(self.step_delay); }
            let mut out = Vec::with_capacity(toks.len() * self.n_vocab());
            for (i, &t) in toks.iter().enumerate() {
                states[i].fed.push(t);
                out.extend_from_slice(&onehot(hash(&states[i].fed), self.n_vocab()));
            }
            out
        }
        fn pick(&self, row: &[f32], _t: f32, _rng: &mut u64) -> Option<u32> {
            Some(row.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32)
        }
        fn is_stop(&self, tok: u32) -> bool { self.stop != 0 && tok == self.stop }
        fn text_of(&self, ids: &[u32]) -> String {
            ids.iter().map(|i| format!("{i},")).collect()
        }
    }

    fn onehot(i: u32, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        v[i as usize % n] = 1.0;
        v
    }

    /// Start a server on an ephemeral port and return its address. The thread is deliberately not
    /// joined: `serve_loop` is a server and never returns, and the test process reaps it on exit.
    fn spawn(m: Mock, max_batch: usize) -> String {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = l.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            serve_loop(m, l, ServeOpts { max_batch, any_mcp_tools: false }, |_m, _me, _p, _b, _s| false);
        });
        addr
    }

    /// A real HTTP POST over a real socket. Returns the response body.
    fn post(addr: &str, path: &str, body: &str) -> String {
        let mut s = TcpStream::connect(addr).expect("connect");
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
