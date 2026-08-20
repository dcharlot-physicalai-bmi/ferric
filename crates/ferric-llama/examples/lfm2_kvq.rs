//! **LFM2 cached decode under `FERRIC_KVQ`** — what the quantized KV cache costs and what it saves.
//!
//! This exists because the obvious instrument was the wrong one. `run_lfm2_gguf` re-runs the **full
//! prefix every step** and never builds a [`Cache`], so `FERRIC_KVQ` cannot affect it: all three
//! formats produced byte-identical text there, which reads like "the formats are accurate" and
//! actually means "nothing under test ran". An example that does not touch the cache cannot say
//! anything about the cache.
//!
//! So: prefill once into a real `Cache`, decode one token at a time from it, and report the token ids
//! **and** the live cache bytes. Both are needed. The ids alone cannot distinguish an accurate format
//! from an inert switch — only the byte count proves the quantized store is the one being written.
//!
//! LFM2 is the interesting case for this. Only the attention blocks carry KV — 6 of 16 on the 1.2B —
//! and the conv blocks' `l_cache-1` window is bounded by `l_cache` rather than by context, so it is
//! never quantized and never counted here. Quantization therefore reaches a smaller share of this
//! model's state than it does on a dense transformer, which is worth seeing rather than assuming.
//!
//!   cargo run -p ferric-llama --example lfm2_kvq --release -- <model.gguf> "<prompt>" [n]
//!   FERRIC_KVQ=q8_0 cargo run ... (same, with a quantized cache)
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::lfm2::{Cache, Lfm2};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    // argv only, no default: a hardcoded checkpoint hid a live divergence in this tree for a day.
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: lfm2_kvq <model.gguf> \"<prompt>\" [n_gen]");
    let prompt = a.get(2).map(String::as_str).unwrap_or("The capital of France is");
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(path).expect("open");
    let m = Lfm2::load(&ctx, &g).expect("load");

    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer.ggml.tokens"),
    };
    // A REAL BPE, not a longest-match sketch. The first version of this example rolled its own
    // greedy matcher; it produced ids the model had never seen in that order, saturated to token 1098
    // ("TheTheThe..."), and every FERRIC_KVQ setting printed the same degenerate stream. That reads
    // like "the formats are equivalent" and means "the input could not distinguish anything".
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m {
            s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string()))
        } else { None }).collect(),
        _ => panic!("no tokenizer.ggml.merges in {path}"),
    };
    let bpe = Bpe::new(vocab, &merges);
    let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
    let mut ids = bpe.encode(prompt);
    if add_bos { if let Some(b) = bos_id { ids.insert(0, b); } }
    // Bound-check before the embed slice does it with an opaque byte offset. A tokenizer that emits an
    // id outside the table panics 200 lines away as "range start index N out of range", which says
    // nothing about the cause.
    assert!(ids.iter().all(|&t| (t as usize) < m.cfg.n_vocab),
            "tokenizer produced ids outside the {}-entry vocabulary: {:?}",
            m.cfg.n_vocab, ids.iter().filter(|&&t| (t as usize) >= m.cfg.n_vocab).collect::<Vec<_>>());
    println!("prompt ids: {ids:?}  (vocab {}, tokens table {})", m.cfg.n_vocab, toks.len());

    let n_attn = m.cfg.kv.iter().filter(|&&k| k > 0).count();
    let fmt = Cache::new(&m.cfg).kvq_fmt();
    println!("model {path}");
    println!("LFM2 · {} blocks ({} conv + {n_attn} attn) · d={}", m.cfg.n_layer, m.cfg.n_layer - n_attn, m.cfg.d);
    println!("FERRIC_KVQ -> {}", fmt.map(|f| f.name().to_string()).unwrap_or_else(|| "f32 (off)".into()));

    let mut cache = Cache::new(&m.cfg);
    let mut logits = m.forward(&ids, &mut cache);
    let mut out: Vec<u32> = Vec::with_capacity(n_gen);
    let nv = m.cfg.n_vocab;
    for _ in 0..n_gen {
        // ⚠ THE LAST ROW, not the whole tensor. Prefill returns `[t, n_vocab]`, so an argmax over the
        // flattened buffer ranges up to `t * n_vocab` and yields an id far outside the vocabulary —
        // which then panics inside `embed` as an opaque byte-offset slice error 200 lines away. Decode
        // steps return `[1, n_vocab]`, where the two happen to agree, so this only bites on the first
        // step after a multi-token prefill.
        let all = logits.to_vec().await;
        let row = &all[all.len() - nv..];
        let next = row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32)
            .expect("non-empty logits");
        out.push(next);
        logits = m.forward(&[next], &mut cache);
    }

    let (q, f32b) = (cache.kv_bytes(), cache.kv_f32_bytes());
    println!("ids: {out:?}");
    println!("text: {}", out.iter().map(|&t| toks[t as usize].replace('\u{2581}', " ")).collect::<String>().trim_end());
    // The byte count is the half that cannot be faked by an accurate format. If the switch were inert
    // these two would be equal, whatever the ids said.
    let live = cache.kv_live_bytes();
    // TWO ratios, because one of them is routinely misread. `allocated` is what the device holds and
    // includes doubling slack; `live` is what the format buys. Reporting only the first makes q8_0
    // look like ~1.9x when the format is 3.76x, and reporting only the second overstates what the
    // process actually frees.
    println!("kv cache: allocated {q} B, live {live} B, f32 {f32b} B");
    println!("  allocated vs f32 live: {:.2}x  (includes growth slack; both stores double)", f32b as f64 / q.max(1) as f64);
    println!("  live vs f32 live:      {:.2}x  (the format ratio)", f32b as f64 / live.max(1) as f64);
    // A saturated stream makes the id comparison vacuous, so refuse to report one. This is the guard
    // the first version of this example needed and did not have.
    let distinct: std::collections::BTreeSet<u32> = out.iter().copied().collect();
    assert!(distinct.len() > 1,
            "every generated token is {:?} — the model is saturated and these ids cannot distinguish \
             one KV format from another, whatever they look like. Fix the prompt or the tokenizer \
             before reading anything into the output above.", out.first());
    assert!(fmt.is_none() || q < f32b,
            "FERRIC_KVQ={:?} but the cache is not smaller than f32 — the switch did not reach the store",
            fmt.map(|f| f.name()));
}
