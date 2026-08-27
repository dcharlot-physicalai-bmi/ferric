//! **What does the shipped expert cache actually achieve on a real modern MoE, and how much is left?**
//!
//! `route_predictability` established that the exploitable structure in this checkpoint's routing is
//! TEMPORAL — same layer, previous token, 41.0% against a 3.1% floor — which is exactly what
//! `ferric_tier::ExpertCache`'s hotness-LFU bets on. That says the policy is aimed correctly. It does
//! not say how well it does, and "aimed correctly" is not a hit rate.
//!
//! So this replays a real captured routing trace through three policies at a sweep of cache sizes:
//!
//! * **`ExpertCache`** — the actual shipped type, not a reimplementation of it. A simulation of a
//!   policy measures the simulation; the number here is only about Ferric if Ferric's own code
//!   produces it.
//! * **LRU** — the reference the crate's docs compare against ("36.24%, dead flat from 8 GB to
//!   64 GB" on a vendor trace).
//! * **Belady** — the offline optimum, evicting whichever resident expert is next used furthest in
//!   the future. Unimplementable online; it is the ceiling, and the distance to it is the headroom
//!   any smarter policy is competing for.
//!
//! ## ⛔ THE RESULT: the shipped policy is right below the floor and WRONG above it
//!
//! Qwen3.6-35B-A3B, 12 MoE layers, 9,888 accesses over 103 tokens, working-set floor 96:
//!
//! ```text
//!    capacity         LRU     shipped      Belady    gap closed
//!           9        0.0%        0.7%        7.9%            9%
//!          24        0.0%        6.3%       21.1%           30%
//!          48        0.0%       12.7%       35.3%           36%
//!          96       23.2%       21.5%       48.6%           -7%
//!         192       40.4%       33.2%       58.9%          -39%
//!         384       51.9%       47.0%       67.7%          -31%
//!         768       63.4%       62.6%       75.7%           -7%
//! ```
//!
//! **Below the floor the crate's argument holds exactly.** LRU is *0.0%* — not low, ZERO, the cyclic
//! collapse `lib.rs` predicts — while hotness-LFU scores 6.3% and 12.7%. Frequency-primary is
//! precisely what rescues the sub-working-set regime, and this is that claim reproduced on real
//! weights rather than cited from a vendor trace.
//!
//! ⛔ **Above the floor it LOSES to plain LRU at every capacity**, by up to 7.2 points (33.2% vs
//! 40.4% at 192). "Gap closed" goes negative: the shipped policy is not merely failing to beat LRU,
//! it is behind it.
//!
//! ## Why, and it follows from the other measurement
//!
//! `route_predictability` found the exploitable structure here is TEMPORAL — same layer, previous
//! token, 41.0%. Now count the frequency signal: 12 layers x 256 experts = 3,072 cells, 103 tokens x
//! 8 = 824 uses per layer, so a given (layer, expert) is touched about **3 times** across the whole
//! trace. Three samples do not estimate a frequency. Hotness-LFU is therefore ranking mostly on
//! noise and using recency only to break ties — it has frequency as the primary key and recency as
//! the secondary, on a workload where recency is the signal that exists.
//!
//! ⭐ So the two measurements agree on a mechanism: **this router's usage is flat enough that
//! frequency carries little information, and the cache is keyed on the weaker of the two axes.**
//! Below the working set that does not matter, because ANY retention beats a cyclic LRU's zero.
//! Above it, it costs up to 7 points.
//!
//! ⚠ **What this does NOT say.** One checkpoint, 9,888 accesses against the vendor trace's 100,096,
//! and the routing was captured during PREFILL then replayed token-major, which models decode. The
//! crate's cited "LRU 36.24%, dead flat 8→64 GB" is a much larger model whose working set exceeded
//! every budget tested — i.e. entirely the regime where hotness-LFU wins here. Both results can be
//! true at once, and the honest reading is that the right policy depends on which side of the
//! working-set floor a deployment sits, which nothing currently decides at runtime.
//!
//! ## Registered before running
//!
//! The crate's docs cite a 25.5-point LRU→Belady gap on someone else's trace. If that gap is small
//! here, the policy question is settled and effort belongs elsewhere (capacity, or the fabric split).
//! If it is large, there is something left to win and hotness-LFU's share of it says whether
//! frequency was the right bet.
//!
//! ⚠ **The floor is structural, not a tuning preference.** `expert.rs` measures that a cache holding
//! fewer than `n_layers × top_k` entries scores ~0 regardless of policy, because iterating layers
//! every token makes the combined (layer, expert) access cyclic — the same pathology that makes LRU
//! worthless for layer streaming. The sweep therefore straddles that number instead of starting
//! above it, or it would report a policy difference where only capacity matters.
//!
//!   FERRIC_ROUTE_TRACE=1 FERRIC_MAX_LAYERS=12 cargo run -p ferric-llama --example expert_cache_sim --release -- <model.gguf>
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen35::{route_trace, Ffn, Qwen35};
use ferric_tier::{Backing, ExpertCache, SliceBacking};
use std::collections::HashMap;
use std::sync::Arc;

const PROMPTS: &[&str] = &[
    "The capital of France is Paris, and the river that runs through it",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot =",
    "In 1687 Newton published the Principia, which set out three laws of",
    "SELECT customer_id, SUM(total) FROM orders WHERE created_at >",
    "Die Katze sitzt auf dem Tisch und schaut aus dem Fenster, weil",
    "The mitochondrion generates most of the cell's supply of adenosine",
];

/// One access: which expert of which layer, and which token it belongs to. The token boundary
/// matters — `ExpertCache` ages hotness per token, so replaying without it measures a policy that
/// never decays.
#[derive(Clone, Copy)]
struct Access { token: usize, layer: u32, expert: u32 }

/// LRU over `(layer, expert)`. Written out rather than reused because the crate deliberately does
/// NOT ship an expert LRU — its docs explain why — and comparing against the policy it rejected is
/// the point.
fn lru_hits(seq: &[Access], cap: usize) -> (u64, u64) {
    let (mut hits, mut gets) = (0u64, 0u64);
    let mut resident: Vec<(u32, u32)> = Vec::with_capacity(cap);
    let mut clock: HashMap<(u32, u32), u64> = HashMap::new();
    let mut t = 0u64;
    for a in seq {
        gets += 1; t += 1;
        let key = (a.layer, a.expert);
        if resident.contains(&key) { hits += 1; clock.insert(key, t); continue }
        if resident.len() == cap {
            // Evict least-recently-used.
            let victim = resident.iter().copied()
                .min_by_key(|k| *clock.get(k).unwrap_or(&0)).expect("non-empty");
            resident.retain(|k| *k != victim);
        }
        resident.push(key);
        clock.insert(key, t);
    }
    (hits, gets)
}

/// Belady's optimum: evict whichever resident key is next used furthest ahead (or never again).
/// Offline and unimplementable; the ceiling.
fn belady_hits(seq: &[Access], cap: usize) -> (u64, u64) {
    // next_use[i] = index of the next access to the same key after i, or usize::MAX.
    let mut next_use = vec![usize::MAX; seq.len()];
    let mut last: HashMap<(u32, u32), usize> = HashMap::new();
    for i in (0..seq.len()).rev() {
        let key = (seq[i].layer, seq[i].expert);
        if let Some(&nx) = last.get(&key) { next_use[i] = nx }
        last.insert(key, i);
    }
    let (mut hits, mut gets) = (0u64, 0u64);
    let mut resident: HashMap<(u32, u32), usize> = HashMap::new(); // key -> its next use
    for (i, a) in seq.iter().enumerate() {
        gets += 1;
        let key = (a.layer, a.expert);
        if resident.contains_key(&key) {
            hits += 1;
            resident.insert(key, next_use[i]);
            continue;
        }
        if resident.len() == cap {
            let victim = *resident.iter().max_by_key(|&(_, nx)| *nx).expect("non-empty").0;
            resident.remove(&victim);
        }
        resident.insert(key, next_use[i]);
    }
    (hits, gets)
}

/// The SHIPPED policy, driven exactly as a runtime would drive it.
fn shipped_hits(seq: &[Access], cap: usize, n_layers: u32, n_experts: u32, k: usize,
                slot: usize, backing: &dyn Backing) -> Option<(u64, u64)> {
    let mut c = ExpertCache::new(n_layers, n_experts, slot, cap, k).ok()?;
    let mut tok = usize::MAX;
    let mut i = 0usize;
    while i < seq.len() {
        if seq[i].token != tok {
            if tok != usize::MAX { c.end_token(); }
            tok = seq[i].token;
        }
        // `note_selected` is documented as once per (token, layer) BEFORE the gets — it bumps
        // hotness and protects the step's working set from eviction. Calling it per-get instead
        // would inflate hotness k-fold and measure a policy nobody ships.
        let layer = seq[i].layer;
        let mut group = Vec::with_capacity(k);
        let mut j = i;
        while j < seq.len() && seq[j].token == tok && seq[j].layer == layer {
            group.push(seq[j].expert); j += 1;
        }
        c.note_selected(layer, &group);
        for &e in &group {
            let off = (layer as u64 * n_experts as u64 + e as u64) * slot as u64;
            c.get(layer, e, off, backing).ok()?;
        }
        i = j;
    }
    let s = c.stats();
    Some((s.hits, s.gets))
}

fn main() { pollster::block_on(run()); }

async fn run() {
    assert!(route_trace::enabled(), "set FERRIC_ROUTE_TRACE=1 or the trace is empty");
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| format!("{}/.cache/ferric/qwen3.6-35b-a3b-q4km.gguf",
                                   std::env::var("HOME").unwrap()));
    let g = GgufFile::open(&path).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));

    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer"),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(p, q)| (p.to_string(), q.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);

    let m = Qwen35::load(&ctx, &g).expect("load");
    let moe: Vec<usize> = (0..m.layers.len()).filter(|&i| matches!(m.layers[i].ffn, Ffn::Moe(_))).collect();
    let (k, ne) = (m.cfg.n_expert_used, m.cfg.n_expert as u32);
    println!("Expert cache on a real routing trace — {path}");
    println!("  {} blocks, {} MoE, {ne} experts, top-{k}\n", m.layers.len(), moe.len());

    // ---- capture ---------------------------------------------------------------------------
    let mut seq: Vec<Access> = Vec::new();
    let mut token_base = 0usize;
    for p in PROMPTS {
        let ids: Vec<u32> = bpe.encode(p);
        let _ = route_trace::take();
        let mut cache = ferric_llama::qwen35::Cache::new(&m.cfg);
        let _ = m.forward_cached(&ids, &mut cache, m.layers.len());
        let rec = route_trace::take();
        assert_eq!(rec.len(), moe.len(), "recorded {} MoE invocations for {} layers", rec.len(), moe.len());
        // Layer-major on the wire, token-major in the access order — a runtime touches every layer
        // of ONE token before moving on, and replaying it layer-major would turn a cyclic access
        // pattern into a sequential one and flatter every policy.
        let mut per: Vec<Vec<(u32, Vec<u32>)>> = vec![Vec::new(); ids.len()];
        for (li, t) in rec.iter().enumerate() {
            let v = t.to_vec().await;
            for tok in 0..ids.len() {
                let e: Vec<u32> = (0..k).map(|j| v[tok * 2 * k + k + j] as u32).collect();
                assert!(e.iter().all(|&x| x < ne), "expert id out of range: {e:?}");
                per[tok].push((moe[li] as u32, e));
            }
        }
        for (ti, layers) in per.into_iter().enumerate() {
            for (layer, experts) in layers {
                for e in experts { seq.push(Access { token: token_base + ti, layer, expert: e }); }
            }
        }
        token_base += ids.len();
    }
    let n_layers = m.layers.len() as u32;
    println!("  trace: {} accesses over {token_base} tokens ({} layers x top-{k})",
             seq.len(), moe.len(), );
    assert_eq!(seq.len(), token_base * moe.len() * k, "trace length disagrees with the geometry");

    // ---- replay ----------------------------------------------------------------------------
    const SLOT: usize = 64; // hit rate is what is being measured; real bytes are applied after
    let backing = SliceBacking::new(vec![0u8; n_layers as usize * ne as usize * SLOT]);
    let floor = moe.len() * k;
    println!("  working-set floor (n_layers x top_k) = {floor} entries\n");
    println!("  {:>9}  {:>10}  {:>10}  {:>10}  {:>12}", "capacity", "LRU", "shipped", "Belady", "gap closed");
    println!("  {:-<60}", "");

    let mut caps: Vec<usize> = vec![k + 1, floor / 4, floor / 2, floor, floor * 2, floor * 4, floor * 8];
    caps.retain(|&c| c >= k + 1);
    caps.sort_unstable(); caps.dedup();
    for cap in caps {
        let (lh, lg) = lru_hits(&seq, cap);
        let (bh, _) = belady_hits(&seq, cap);
        let sh = shipped_hits(&seq, cap, n_layers, ne, k, SLOT, &backing);
        let (l, b) = (lh as f64 / lg as f64, bh as f64 / lg as f64);
        match sh {
            Some((s, sg)) => {
                assert_eq!(sg, lg, "the policies saw different access counts");
                let s = s as f64 / sg as f64;
                let closed = if b - l > 1e-9 { 100.0 * (s - l) / (b - l) } else { f64::NAN };
                println!("  {cap:>9}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>11.0}%",
                         100.0 * l, 100.0 * s, 100.0 * b, closed);
            }
            None => println!("  {cap:>9}  {:>9.1}%  {:>10}  {:>9.1}%  {:>12}",
                             100.0 * l, "refused", 100.0 * b, "-"),
        }
    }

    println!("\n  'gap closed' is the shipped policy's share of the LRU→Belady headroom: 0% means it");
    println!("  is LRU, 100% means it is clairvoyant. Belady is offline and unreachable; it is the");
    println!("  ceiling that says whether a better ONLINE policy is worth writing at all.");
    println!("\n  ⚠ {} accesses from prefill over {} prompts on {} of the checkpoint's layers.",
             seq.len(), PROMPTS.len(), moe.len());
    println!("     The crate's cited 36.24%/25.5-point figures come from a 100,096-request vendor");
    println!("     trace — a different model and ~10x the accesses, so these are not comparable");
    println!("     numbers, they are the same question asked of this checkpoint.");
}
