//! **Continuous batching on a real model** — the scheduler from `ferric-kv` driving Qwen2.5-0.5B.
//!
//! `ferric-kv`'s scheduler is pure bookkeeping and its unit tests run in microseconds. This is the other
//! half: the same scheduler deciding what a real model runs, so the claim is about tokens rather than
//! integers.
//!
//! ## The comparison that matters
//!
//! **Static batching** forms a batch, runs it to completion, and only then forms the next. A short request
//! that arrives behind a long one waits for the long one — its latency is the *batch's* length, not its
//! own.
//!
//! **Continuous batching** re-forms the batch every step from whatever is runnable. The short request
//! retires the moment it is done and a queued one takes its slot immediately.
//!
//! Both are measured here on the same requests with the same model, and the thing being compared is
//! **per-request completion latency**, not throughput — throughput can look identical while every
//! individual request is slow, which is exactly the failure continuous batching fixes.
//!
//! The outputs must also be identical: scheduling changes *when* a token is computed, never *what* it is.
//! A scheduler that reorders KV would still emit fluent text, so the tokens are compared directly.
//!
//!   cargo run -p ferric-llama --example continuous_batching --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_kv::{Idle, PagedKv, Request, Scheduler};
use ferric_llama::qwen3::{Cache, Cfg, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// Requests with deliberately uneven lengths — an even workload cannot show the difference.
const WORK: &[(u64, &str, usize)] = &[
    (1, "Write a long detailed explanation of how a computer works, starting from", 48),
    (2, "The capital of France is", 3),
    (3, "Two plus two equals", 3),
    (4, "Once upon a time in a distant kingdom there lived", 40),
    (5, "The sky is", 3),
    (6, "Water boils at", 3),
];

fn argmax(v: &[f32], n_vocab: usize) -> u32 {
    let last = &v[v.len() - n_vocab..];
    last.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32
}

/// Advance one sequence by a single token. Shared by both schedulers so the *only* difference measured
/// is the order work is issued in.
async fn decode_one(m: &Qwen3, cache: &mut Cache, tok: u32, n_vocab: usize) -> u32 {
    let logits = m.forward_cached(&[tok], cache).to_vec().await;
    argmax(&logits, n_vocab)
}

/// Static batching: run the whole batch to completion before admitting anything else.
///
/// Modelled honestly — every request in the batch is stepped together, and none retires until the
/// longest is done. That "none retires early" is the entire cost, and it is why a 3-token request
/// behind a 48-token one waits 48 steps.
async fn run_static(m: &Qwen3, cfg: &Cfg, prompts: &[(u64, Vec<u32>, usize)], batch: usize)
    -> Vec<(u64, usize, Vec<u32>)>
{
    let n_vocab = cfg.n_vocab;
    let mut out = Vec::new();
    let mut step = 0usize;
    for group in prompts.chunks(batch) {
        let mut caches: Vec<Cache> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        let mut gen: Vec<Vec<u32>> = Vec::new();
        for (_, toks, _) in group {
            let mut c = Cache::new(cfg);
            let l = m.forward_cached(toks, &mut c).to_vec().await;
            next.push(argmax(&l, n_vocab));
            caches.push(c);
            gen.push(Vec::new());
        }
        let longest = group.iter().map(|(_, _, n)| *n).max().unwrap_or(0);
        for _ in 0..longest {
            step += 1;
            for i in 0..group.len() {
                // Still stepped even once done — that is what "the batch runs to completion" means.
                if gen[i].len() < group[i].2 {
                    gen[i].push(next[i]);
                    next[i] = decode_one(m, &mut caches[i], next[i], n_vocab).await;
                }
            }
        }
        // Nothing retires until the whole group is finished.
        for (i, (id, _, _)) in group.iter().enumerate() {
            out.push((*id, step, gen[i].clone()));
        }
    }
    out
}

/// Continuous batching: `ferric_kv::Scheduler` decides, and every decode in a step runs in ONE forward.
///
/// This is the pairing that makes both halves pay. The scheduler removes head-of-line blocking — a short
/// request retires the step it finishes rather than waiting for the batch. `forward_batch` then executes
/// the whole step's decodes in a single pass, so the ~525 MB of weights are read once for the step
/// instead of once per sequence. Without the second half the scheduler improves latency and nothing
/// else: `examples/batch_throughput.rs` measured that at exactly 1.00x scaling.
///
/// Prefills stay separate — they carry different token counts per sequence, so they do not stack into a
/// uniform `[N, d]`. Only the decodes batch, which is where the steady-state cost lives anyway.
async fn run_continuous(m: &Qwen3, cfg: &Cfg, prompts: &[(u64, Vec<u32>, usize)], max_seqs: usize)
    -> Vec<(u64, usize, Vec<u32>)>
{
    let n_vocab = cfg.n_vocab;
    let mut s = Scheduler::new(PagedKv::new(4096, 16), max_seqs, 4096);
    for (id, toks, n) in prompts {
        s.submit(Request::new(*id, toks.clone(), *n));
    }

    // One live sequence's model state. Held in a slab rather than a map because a step needs many
    // `&mut Cache` at once, and a HashMap cannot hand out disjoint mutable borrows by key.
    struct Slot { seq: u64, cache: Cache, next: u32, gen: Vec<u32> }
    let mut slots: Vec<Slot> = Vec::new();
    let mut out = Vec::new();
    let mut step = 0usize;

    loop {
        let batch = match s.step() {
            Ok(b) => b,
            Err(Idle::Empty) => break,
            Err(Idle::Starved) => panic!("scheduler starved with {} waiting", s.waiting()),
        };
        step += 1;
        let seq_of: HashMap<u64, u64> = s.running().iter().map(|r| (r.req.id, r.seq)).collect();

        // ---- prefills: one call each, since token counts differ ----
        for &(seq, start, n) in &batch.work {
            let running = s.running().iter().find(|r| r.seq == seq).expect("batch names a running seq");
            // A decode is encoded as start == tokens.len(); a prefill as start == filled, strictly below.
            // Inferring from `n == 1` would misread a one-token prefill chunk as a decode.
            if start == running.req.tokens.len() { continue }
            let toks = running.req.tokens[start..start + n].to_vec();
            if !slots.iter().any(|sl| sl.seq == seq) {
                slots.push(Slot { seq, cache: Cache::new(cfg), next: 0, gen: Vec::new() });
            }
            let sl = slots.iter_mut().find(|sl| sl.seq == seq).unwrap();
            let l = m.forward_cached(&toks, &mut sl.cache).to_vec().await;
            sl.next = argmax(&l, n_vocab);
        }

        // ---- decodes: ALL of them in one forward pass ----
        let decoding: Vec<u64> = batch.work.iter()
            .filter(|&&(seq, start, _)| s.running().iter().any(|r| r.seq == seq && start == r.req.tokens.len()))
            .map(|&(seq, _, _)| seq)
            .collect();

        let mut produced: Vec<(u64, u32)> = Vec::new();
        if !decoding.is_empty() {
            // One `iter_mut` yields disjoint `&mut` for exactly the slots decoding this step.
            let mut picked: Vec<&mut Slot> = slots.iter_mut().filter(|sl| decoding.contains(&sl.seq)).collect();
            let toks: Vec<u32> = picked.iter().map(|sl| sl.next).collect();
            for (sl, &t) in picked.iter_mut().zip(&toks) { sl.gen.push(t); produced.push((sl.seq, t)); }
            let mut caches: Vec<&mut Cache> = picked.iter_mut().map(|sl| &mut sl.cache).collect();
            let logits = m.forward_batch(&toks, &mut caches).to_vec().await;
            for (i, sl) in picked.iter_mut().enumerate() {
                sl.next = argmax(&logits[i * n_vocab..(i + 1) * n_vocab], n_vocab);
            }
        }

        for done in s.complete(&batch, &produced) {
            let seq = seq_of.get(&done.id).copied().expect("a finished request was running this step");
            let gen = slots.iter().position(|sl| sl.seq == seq).map(|i| slots.remove(i).gen).unwrap_or_default();
            out.push((done.id, step, gen));
        }
        if step > 10_000 { panic!("continuous batching did not terminate"); }
    }
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();
    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!(),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!(),
    };
    let bpe = Bpe::new(vocab, &merges);
    let m = Qwen3::load(&ctx, &g).unwrap();

    let prompts: Vec<(u64, Vec<u32>, usize)> =
        WORK.iter().map(|(id, p, n)| (*id, bpe.encode(p), *n)).collect();

    println!("Continuous batching — Qwen2.5-0.5B, {} requests of uneven length\n", prompts.len());
    println!("  {:>4}  {:>7}  {}", "id", "tokens", "prompt");
    for (id, t, n) in &prompts {
        println!("  {id:>4}  {n:>7}  {:.46}", WORK.iter().find(|w| w.0 == *id).unwrap().1);
        let _ = t;
    }

    let t0 = std::time::Instant::now();
    let stat = run_static(&m, &m.cfg, &prompts, 3).await;
    let static_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let cont = run_continuous(&m, &m.cfg, &prompts, 8).await;
    let cont_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("\n  Completion step per request (lower is better for that request):");
    println!("  {:>4}  {:>10}  {:>12}  {:>10}", "id", "requested", "static step", "cont step");
    println!("  {:-<44}", "");
    let mut static_wait = 0usize;
    let mut cont_wait = 0usize;
    for (id, _, n) in &prompts {
        let sst = stat.iter().find(|(i, _, _)| i == id).map(|(_, s, _)| *s).unwrap_or(0);
        let cst = cont.iter().find(|(i, _, _)| i == id).map(|(_, s, _)| *s).unwrap_or(0);
        static_wait += sst;
        cont_wait += cst;
        println!("  {id:>4}  {n:>10}  {sst:>12}  {cst:>10}");
    }

    println!("\n  {:<28} {:>10}  {:>16}", "", "wall", "total wait (steps)");
    println!("  {:-<58}", "");
    println!("  {:<28} {:>7.0} ms  {:>16}", "static batching", static_ms, static_wait);
    println!("  {:<28} {:>7.0} ms  {:>16}", "continuous batching", cont_ms, cont_wait);

    assert_eq!(cont.len(), prompts.len(), "continuous batching lost a request");
    assert_eq!(stat.len(), prompts.len(), "static batching lost a request");

    // Correctness before speed: scheduling changes WHEN a token is computed, never WHAT it is. A
    // scheduler that mismatched KV to sequence would still emit fluent text, so compare the tokens.
    for (id, _, n) in &prompts {
        let sg = &stat.iter().find(|(i, _, _)| i == id).unwrap().2;
        let cg = &cont.iter().find(|(i, _, _)| i == id).unwrap().2;
        assert_eq!(sg.len(), *n, "request {id}: static produced {} of {n} tokens", sg.len());
        assert_eq!(sg, cg, "request {id}: scheduling changed the output — KV is mismatched to sequence");
    }
    println!("\n  Generated tokens are IDENTICAL under both schedulers for all {} requests.", prompts.len());
    assert!(
        cont_wait < static_wait,
        "continuous batching did not reduce total wait ({cont_wait} vs {static_wait}) — \
         short requests are still waiting behind long ones"
    );

    let short_static: usize = prompts.iter().filter(|(_, _, n)| *n <= 3)
        .filter_map(|(id, _, _)| stat.iter().find(|(i, _, _)| i == id).map(|(_, s, _)| *s)).sum();
    let short_cont: usize = prompts.iter().filter(|(_, _, n)| *n <= 3)
        .filter_map(|(id, _, _)| cont.iter().find(|(i, _, _)| i == id).map(|(_, s, _)| *s)).sum();

    println!("\n  ✅ Two wins, and they come from two different pieces.\n");
    println!("     LATENCY: the 3-token requests wait {short_static} steps under static batching and {short_cont} under");
    println!("     continuous — {:.1}x less. That is the scheduler: a request retires the step it finishes",
             short_static as f64 / short_cont.max(1) as f64);
    println!("     instead of waiting for the longest in its batch.\n");
    println!("     THROUGHPUT: {:.2}x on wall clock, and that is NOT the scheduler — it is `forward_batch`",
             static_ms / cont_ms);
    println!("     executing a whole step's decodes in one pass, so the ~525 MB of weights are read once");
    println!("     per step rather than once per sequence. Scheduling alone gave exactly 1.00x scaling");
    println!("     (examples/batch_throughput.rs); it decides WHAT runs but cannot amortise anything.\n");
    println!("     Both halves are needed, and neither substitutes for the other.");
}
