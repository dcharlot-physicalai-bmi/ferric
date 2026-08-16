//! **Continuous batching, end to end on real weights.**
//!
//! `sched::Scheduler` is the policy and is unit-tested without a GPU. This is the proof that the
//! policy actually drives the model: sequences are admitted, prefilled, decoded in one batched
//! forward, and — the part that matters — **a finished sequence's slot is refilled by a waiting
//! request on the very next step**, mid-flight.
//!
//! ```text
//!   cargo run -p ferric-llama --example continuous_batching --release -- <model.gguf> [max_batch]
//! ```
//!
//! ## The correctness property
//!
//! Every sequence's output must be **token-identical** to running it alone. Batching is a scheduling
//! change, not a semantic one, and a batched path that leaked across sequences would still emit
//! fluent text — so this checks each sequence against a solo run rather than eyeballing the output.
//!
//! ## Why prefill is separate
//!
//! `forward_batch` takes exactly one token per sequence: it is the decode step. A newly admitted
//! request has a prompt, so its cache is built by a solo `forward_cached` on admission and it joins
//! the batch from its second token onward. That split is why a long prompt arriving mid-flight does
//! not stall the sequences already decoding — the industry answer beyond this is chunked prefill
//! interleaved into the decode batch, which Ferric has the pieces for but does not schedule yet.
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::sched::{Done, Scheduler, SeqId};
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: continuous_batching <model.gguf> [max_batch]");
    let max_batch: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(path).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    let vn = m.cfg.n_vocab;
    let argmax = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;

    // Five requests, deliberately uneven budgets so slots free at different times — a uniform batch
    // would never exercise refill and would pass while static batching was silently in use.
    let reqs: Vec<(Vec<u32>, usize)> = vec![
        (vec![785, 6722, 315, 9625, 374], 6),   // "The capital of France is"
        (vec![785, 8489, 374], 3),
        (vec![7679, 220, 17, 488, 220, 17, 284], 4),
        (vec![785, 7160, 25283], 2),
        (vec![14990, 1879], 5),
    ];

    let mut sched = Scheduler::new(max_batch);
    let mut caches: HashMap<SeqId, Cache> = HashMap::new();
    let mut out: HashMap<SeqId, Vec<u32>> = HashMap::new();
    let mut order: Vec<SeqId> = Vec::new();
    for (p, n) in &reqs { let id = sched.submit(p.clone(), *n); order.push(id); }

    println!("continuous batching · max_batch {max_batch} · {} requests", reqs.len());
    let t0 = std::time::Instant::now();
    let mut steps = 0usize;
    let mut peak = 0usize;

    while !sched.is_idle() {
        let batch = sched.step_batch();
        if batch.is_empty() { break; }
        peak = peak.max(batch.len());

        // Admission: a new sequence prefills solo, because forward_batch is the DECODE step.
        // Its first sampled token comes from that prefill, so it enters the batch already advanced.
        let mut first: Vec<(SeqId, u32)> = Vec::new();
        for &id in &batch {
            if caches.contains_key(&id) { continue; }
            let idx = order.iter().position(|x| *x == id).expect("known id");
            let mut c = Cache::new(&m.cfg);
            let logits = m.forward_cached(&reqs[idx].0, &mut c).to_vec().await;
            caches.insert(id, c);
            first.push((id, argmax(&logits)));
        }
        for (id, tok) in first {
            out.entry(id).or_default().push(tok);
            sched.record(id, tok, false);
        }

        // Everything still running after admission decodes together: ONE weight read for all of them.
        let live: Vec<SeqId> = sched.running().iter().map(|s| s.id).collect();
        if !live.is_empty() {
            let toks: Vec<u32> = sched.running().iter().map(|s| s.pending[0]).collect();
            // Borrow the caches for exactly this batch. Disjoint keys, so the split is safe.
            let mut owned: Vec<(SeqId, Cache)> = live.iter()
                .map(|id| (*id, caches.remove(id).expect("cache for a running seq"))).collect();
            let logits = {
                let mut refs: Vec<&mut Cache> = owned.iter_mut().map(|(_, c)| c).collect();
                m.forward_batch(&toks, &mut refs).to_vec().await
            };
            for (row, (id, c)) in owned.into_iter().enumerate() {
                caches.insert(id, c);
                let tok = argmax(&logits[row * vn..(row + 1) * vn]);
                out.entry(id).or_default().push(tok);
                sched.record(id, tok, false);
            }
            steps += 1;
        }

        // Free the caches of anything that retired this step — the drain is what prevents a leak.
        for (id, why) in sched.take_retired() {
            caches.remove(&id);
            println!("  seq {} retired: {why:?} after {} tokens", id.0, out.get(&id).map_or(0, |v| v.len()));
        }
    }

    println!("\n{} batched steps, peak batch {peak}, {:.2?} total", steps, t0.elapsed());
    println!("admitted {} finished {} · caches left {} (must be 0)", sched.admitted, sched.finished, caches.len());
    assert_eq!(caches.len(), 0, "a retired sequence leaked its KV cache");

    // THE correctness check: batching must not change any sequence's output.
    println!("\nverifying each sequence against a solo run:");
    let mut mismatch = 0;
    for (i, id) in order.iter().enumerate() {
        let (prompt, budget) = &reqs[i];
        let mut c = Cache::new(&m.cfg);
        let mut solo = Vec::new();
        let mut next = argmax(&m.forward_cached(prompt, &mut c).to_vec().await);
        for _ in 0..*budget {
            solo.push(next);
            next = argmax(&m.forward_cached(&[next], &mut c).to_vec().await);
        }
        let got = out.get(id).cloned().unwrap_or_default();
        let ok = got == solo;
        if !ok { mismatch += 1; }
        println!("  seq {}: {} batched={got:?}", id.0, if ok { "OK  " } else { "DIFF" });
        if !ok { println!("          solo   ={solo:?}"); }
    }
    assert_eq!(mismatch, 0, "batching changed {mismatch} sequence(s) — it must be a scheduling change only");
    println!("\nall sequences token-identical to solo decode.");
}
