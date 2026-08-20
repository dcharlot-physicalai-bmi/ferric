//! **The radix prefix tree, wired to a real runtime** — and proved to change nothing.
//!
//! `examples/prefix_cache.rs` shows the *linear* case: one long system prompt, many turns behind it.
//! This shows the case that needs a tree — an **agent loop**, which branches. One root context, two
//! tool-call branches off it, several leaf turns off each branch, and the branches are visited
//! interleaved rather than in order, so at any moment more than one live prefix must be cached. A
//! single-slot cache serves the most recent branch and re-prefills the other every time it comes back.
//!
//! Two things must both hold, and the second is the one that is easy to get wrong:
//!
//!   1. the tree must actually **find** the shared prefixes — otherwise it is a data structure for
//!      nothing;
//!   2. every request must produce **identical logits** to prefilling it from scratch. KV forked from
//!      the wrong branch point still yields fluent text, so "the output looks fine" proves nothing.
//!
//! Note what this file does *not* do: it does not touch `qwen3.rs`. The whole runtime-side contract is
//! [`fork`] below — `KvBuf::clone_prefix` per layer plus `cache.pos`. That is the entire integration
//! surface, which is the point of keeping the tree free of the model.
//!
//! ```text
//!   cargo run -p ferric-llama --release --example prefix_tree -- <model.gguf> <corpus.txt>
//! ```

use ferric_core::{max_abs_diff, Context};
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::prefix::{EntryId, PrefixTree, Reuse};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// **The one operation a runtime must supply.** A new cache holding a copy of `src`'s first `rows`
/// positions, with `pos` set to match.
///
/// `KvBuf::clone_prefix` copies rather than aliasing, and it has to: `KvBuf::append` writes in place
/// through an `Arc<wgpu::Buffer>`, so two caches sharing one buffer would overwrite each other's
/// history while both kept reading plausible-looking KV. Copy-on-write would need paged attention.
///
/// `pos` is not bookkeeping — it is the RoPE position of the next token. Forking the KV and leaving
/// `pos` at 0 rotates the continuation as if it were the start of the sequence, which is a silent
/// wrong answer rather than a crash.
fn fork(ctx: &Arc<Context>, model: &Qwen3, src: &Cache, rows: usize) -> Cache {
    let mut c = Cache::new(&model.cfg);
    c.set_layers(
        src.layers().iter()
            .map(|(k, v)| (k.clone_prefix(ctx, rows), v.clone_prefix(ctx, rows)))
            .collect(),
    );
    c.pos = rows;
    c
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <model.gguf> <corpus.txt> [capacity_tokens]", args[0]);
        eprintln!("  both paths are required — a hardcoded model path once hid a live divergence here");
        std::process::exit(2);
    }
    let (model_path, corpus_path) = (&args[1], &args[2]);
    let capacity: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1200);

    let ctx = Arc::new(Context::new().await.expect("no GPU adapter"));
    let g = GgufFile::open(model_path).unwrap_or_else(|e| panic!("open {model_path}: {e}"));

    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("{model_path} has no tokenizer.ggml.tokens"),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!("{model_path} has no tokenizer.ggml.merges"),
    };
    let bpe = Bpe::new(vocab, &merges);
    let model = Qwen3::load(&ctx, &g).expect("load");

    // ---- an agent-shaped workload: real text, branching, visited out of order ----
    let corpus = std::fs::read_to_string(corpus_path).unwrap_or_else(|e| panic!("read {corpus_path}: {e}"));
    let all = bpe.encode(&corpus);
    assert!(all.len() > 6000, "{corpus_path} tokenizes to {} tokens; need >6000", all.len());
    let take = |from: usize, n: usize| all[from..from + n].to_vec();

    let root = take(0, 192);                                  // the system prompt / task context
    let mut branch = Vec::new();                              // two tool-call branches off it
    for b in 0..2 {
        let mut t = root.clone();
        t.extend(take(1000 + b * 64, 48));
        branch.push(t);
    }
    // Leaves alternate between the branches, so both must be cached at once. This is exactly the
    // access pattern a single-slot cache cannot serve: it thrashes, one branch evicting the other.
    let mut requests: Vec<Vec<u32>> = Vec::new();
    for i in 0..8 {
        let mut t = branch[i % 2].clone();
        t.extend(take(3000 + i * 40, 24));
        requests.push(t);
    }

    println!("Radix prefix tree — {} requests, {}-token root, 2 branches, visited alternately",
             requests.len(), root.len());
    println!("model {model_path}\ncorpus {corpus_path}\ncapacity {capacity} tokens\n");

    // ---- baseline: every request prefills everything ----
    let mut base: Vec<Vec<f32>> = Vec::new();
    for r in &requests {
        let mut c = Cache::new(&model.cfg);
        base.push(model.forward_cached(r, &mut c).to_vec().await);
    }

    // ---- with the tree ----
    let mut tree = PrefixTree::new(capacity);
    let mut store: HashMap<EntryId, Cache> = HashMap::new();
    let mut cached: Vec<Vec<f32>> = Vec::new();
    let mut prefilled = 0usize;
    let mut forks = 0usize;
    let mut adoptable = 0usize;

    for r in &requests {
        let m = tree.match_prefix(r);
        // At least one token must go through the forward or there are no logits, so a request that is
        // wholly cached still runs its final token. Clamping the FORK to match is what keeps this
        // correct: fork 3 rows and feed 2 tokens and the last position is computed twice.
        let rows = m.map(|m| m.len.min(r.len() - 1)).unwrap_or(0);
        let mut c = match m {
            Some(m) => {
                // `plan` would offer `Reuse::Adopt` when the match is the WHOLE entry and nobody else
                // holds it — the "continue this session" case, where the sequence appends to the
                // adopted buffers instead of copying them. This example re-derives logits for the
                // final token, so `rows` is clamped strictly below the entry length and adoption
                // never applies. Counted rather than assumed, so the claim stays honest.
                if matches!(tree.plan(&m), Reuse::Adopt { .. }) && rows == m.entry_len { adoptable += 1; }
                forks += 1;
                let src = store.get(&m.entry).expect("a matched entry is still in the store");
                fork(&ctx, &model, src, rows)
            }
            None => Cache::new(&model.cfg),
        };
        if let Some(m) = m { tree.release(m.entry); }

        prefilled += r.len() - rows;
        cached.push(model.forward_cached(&r[rows..], &mut c).to_vec().await);

        if let Some(id) = tree.insert(r) {
            store.insert(id, c);
            tree.release(id);
        }
        for gone in tree.take_evicted() {
            store.remove(&gone).expect("the tree evicted an entry the store never had");
        }
    }
    assert_eq!(tree.pins(), 0, "a pin leaked — the cache would silently stop evicting");
    assert_eq!(store.len(), tree.len(), "the store and the index disagree about what is live");

    // ---- correctness first: a cheaper wrong answer is not a saving ----
    let vocab_n = model.cfg.n_vocab;
    let mut worst = 0f32;
    for (i, (b, c)) in base.iter().zip(&cached).enumerate() {
        let (bl, cl) = (&b[b.len() - vocab_n..], &c[c.len() - vocab_n..]);
        worst = worst.max(max_abs_diff(bl, cl));
        assert_eq!(argmax(bl), argmax(cl), "request {i}: the forked prefix changed the predicted token");
    }

    let total: usize = requests.iter().map(|r| r.len()).sum();
    println!("  {:<28} {:>10}", "prefill tokens, baseline", total);
    println!("  {:<28} {:>10}", "prefill tokens, with tree", prefilled);
    println!("  {:<28} {:>10}", "reused", total - prefilled);
    println!("  {:<28} {:>9.0}%", "hit rate", 100.0 * tree.hit_rate());
    println!("  {:<28} {:>10}", "forks / adoptable", format!("{forks} / {adoptable}"));
    println!("  {:<28} {:>10}", "live entries", tree.len());
    println!("  {:<28} {:>10}", "tree nodes", tree.nodes());
    println!("  {:<28} {:>10}", "tokens held / capacity", format!("{} / {}", tree.tokens_used(), capacity));
    println!("  {:<28} {:>10}", "evictions", tree.evictions);
    println!("\n  max|Δ| on final logits: {worst:.3e}");

    assert!(worst < 1e-3, "the forked prefix perturbed the logits by {worst:.3e}");
    assert!(prefilled < total, "nothing was reused — the tree never hit");
    // Requests 2..n all sit behind a branch that was visited two requests earlier, so each should
    // reuse at least the root plus its branch tail.
    let floor = (requests.len() - 2) * branch[0].len();
    assert!(total - prefilled >= floor,
            "reused {} tokens of a possible ~{floor} — the tree is not holding both branches at once",
            total - prefilled);
    assert!(tree.nodes() <= 2 * tree.len(), "the structure decayed out of radix form");

    println!("\n  ✅ Both branches stayed cached simultaneously and every prediction was IDENTICAL.");
    println!("     The whole runtime-side contract is `fork` at the top of this file: clone_prefix per");
    println!("     layer plus cache.pos. No edit to qwen3.rs, and nothing model-specific in the tree.");
}
