//! **Run a transformer whose weights are not resident — and get the same tokens.**
//!
//! The dense half of the streaming capability. `moe_streaming` showed a real MoE delivering identical
//! bytes under eviction; this shows a whole model *generating*, with its layer weights fetched per visit
//! from a budget far smaller than the weight set, and asserts the **token ids are identical** at every
//! budget.
//!
//! Token identity is the strongest form of the claim. Byte-identity of delivered weights implies it, but
//! asserting it end-to-end is what rules out everything in between — a stale slot, a mis-sliced tensor, a
//! layer built from the wrong run.
//!
//!   cargo run -p ferric-llama --example stream_generate --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::stream::layer_runs;
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

const GEN: usize = 12;

/// Greedy decode, so the token stream is a deterministic function of the weights. Sampling would make
/// "identical output" a statement about an RNG rather than about the tier.
async fn generate(m: &Qwen3, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut cache = Cache::new(&m.cfg);
    let mut ids = prompt.to_vec();
    let mut out = Vec::new();
    let mut fed = 0usize;
    for _ in 0..n {
        let logits = m.forward_cached(&ids[fed..], &mut cache).to_vec().await;
        cache.pos = ids.len();
        fed = ids.len();
        let v = m.cfg.n_vocab;
        let last = &logits[logits.len() - v..];
        let (mut best, mut bv) = (0u32, f32::MIN);
        for (i, &x) in last.iter().enumerate() {
            if x > bv { bv = x; best = i as u32; }
        }
        ids.push(best);
        out.push(best);
    }
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();

    let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!(),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!(),
    };
    let bpe = Bpe::new(vocab, &merges);
    let prompt = bpe.encode("The capital of France is");

    let runs = layer_runs(&g).unwrap();
    let total: u64 = runs.iter().map(|r| r.bytes).sum();
    let biggest = runs.iter().map(|r| r.bytes).max().unwrap();
    println!("Streamed generation — Qwen2.5-0.5B, {} layers, {:.1} MB of layer weights\n",
             runs.len(), total as f64 / 1e6);

    // ---- reference: fully resident ----
    let resident = Qwen3::load(&ctx, &g).unwrap();
    let ref_ids = generate(&resident, &prompt, GEN).await;
    drop(resident);
    println!("  {:>12}  {:>8}  {:>9}  {:>10}   {}", "budget", "pinned", "hit rate", "rebuilds", "generated token ids");
    println!("  {:-<96}", "");
    println!("  {:>12}  {:>8}  {:>9}  {:>10}   {:?}", "resident", "24/24", "100.0%", 0, &ref_ids[..GEN.min(8)]);

    // ---- the ladder ----
    let mut streamed_any = false;
    for budget in [biggest + 4096, total / 8, total / 4, total / 2] {
        let m = match Qwen3::load_streaming(&ctx, &path, budget) {
            Ok(m) => m,
            Err(e) => { println!("  {:>9.1} MB  rejected: {e}", budget as f64 / 1e6); continue; }
        };
        let ids = generate(&m, &prompt, GEN).await;
        let (plan, st, rebuilds) = {
            let s = m.stream.as_ref().unwrap();
            (s.plan().clone(), s.stats(), s.rebuilds.get())
        };
        assert_eq!(
            ids, ref_ids,
            "PLACEMENT CHANGED THE OUTPUT at a {:.1} MB budget — the memory budget must decide where \
             weights come from, never what the model says", budget as f64 / 1e6
        );
        if plan.npin < runs.len() { streamed_any = true; }
        println!("  {:>9.1} MB  {:>4}/{:<3}  {:>8.1}%  {:>10}   {:?}",
                 budget as f64 / 1e6, plan.npin, runs.len(), 100.0 * st.hit_rate(), rebuilds, &ids[..GEN.min(8)]);
    }

    // Anti-vacuity: identical tokens are trivially true if every budget happened to pin everything.
    assert!(streamed_any, "no budget actually streamed — the ladder never left the resident case");

    println!("\n  ✅ IDENTICAL TOKEN IDS at every budget, down to one that holds {:.0}% of the layer weights.",
             100.0 * (biggest + 4096) as f64 / total as f64);
    println!("     Not merely identical delivered bytes — identical OUTPUT, which is what rules out a");
    println!("     stale slot, a mis-sliced tensor, or a layer built from the wrong run.");
    println!("\n     The saving is memory and the cost is bandwidth: a streamed layer is rebuilt on every");
    println!("     visit, so this path is slower than resident by design. It runs when resident cannot.");
}
