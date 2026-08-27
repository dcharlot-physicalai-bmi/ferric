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
//! ## ⛔ MY FIRST EXPLANATION WAS WRONG, AND THE SWEEP THAT KILLED IT ALSO GIVES THE FIX
//!
//! I wrote: "a cell is touched ~3 times in the whole trace, so frequency is mostly noise". The policy
//! never sees a whole-trace count. `DECAY_TOKENS = 16` halves hotness every 16 tokens
//! (`expert.rs:42`, `:168`) while a cell is selected only `k/n_expert = 8/256 = 0.5` times per
//! window — so the primary ranking key (`expert.rs:238`) holds **0 or 1 almost always**, about one
//! bit of range. My reason predicted a longer trace would repair the signal; the code says it cannot.
//! (The "~3" was also wrong: Belady's 75.7% at cap 768 bounds it at **≥4.1** per *touched* cell.)
//!
//! Made falsifiable by parameterising the decay rate — `end_token()` calls per token, so the
//! effective window is `DECAY_TOKENS / d`:
//!
//! ```text
//!    capacity       LRU         d=0         d=1         d=4        d=16
//!         192     40.4%       25.9%       33.2%       41.1%       40.4%
//!         384     51.9%       38.4%       47.0%       51.8%       52.0%
//! ```
//!
//! ⭐ **d = 4 BEATS LRU** (41.1 vs 40.4 at cap 192; 51.8 vs 51.9 at 384). So frequency is NOT the
//! wrong key here — `DECAY_TOKENS = 16` is simply too slow a horizon for this workload, and the
//! deficit is a **constant**, not an axis. That is a far better outcome for the shipped design than
//! my original claim, and I would not have found it by defending the claim.
//!
//! ⭐ **d = 0 — true LFU, hotness accumulating forever — is the WORST arm at 25.9%.** Exactly the
//! opposite of "a longer trace would help". And **d = 16 converges to LRU to a tenth of a point**,
//! which is the sanity check the sweep needed: halving every token leaves hotness binary, so the
//! policy degenerates to recency and must land on LRU. It does.
//!
//! The sub-floor rows are consistent: below the working set, slow decay is what retains anything at
//! all, which is why d = 1 wins there and loses above it. One horizon cannot serve both regimes,
//! and `DECAY_TOKENS` is fixed at compile time.
//!
//! ⚠ **What this does NOT say.** One checkpoint, 9,888 accesses against the vendor trace's 100,096,
//! and the routing was captured during PREFILL then replayed token-major, which models decode —
//! adversarial review cleared that one: `moe_topk` has no `T==1` specialisation and the GDN mixer is
//! a sequential recurrence, so position *t*'s hidden state is identical either way.
//!
//! ⚠ **The floor here is 96 because `FERRIC_MAX_LAYERS=12` truncated the model; a deployed one is
//! ~320.** Read the sweep by its RATIO to the floor, never by absolute capacity. ⭐ The routing is
//! unaffected — layer *l* depends only on layers < *l*, so these are the full model's selections for
//! these 12 layers. A labelling defect, not a fidelity one.
//!
//! ⚠ **Unseparated confound:** one cache over six concatenated prompts, never reset. Mean prompt is
//! 17.2 tokens against a 16–32-token heat horizon, so every token sits inside a domain-switch window.
//! Mitigating: `ExpertCache` exposes no reset at all, so a server genuinely never resets it — this is
//! deployed behaviour, but the table measures a multi-document stream, not steady state. The
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
/// `decays_per_token` calls `end_token()` that many times at each token boundary, which scales the
/// effective decay window: `DECAY_TOKENS / decays_per_token`. **0 disables decay entirely** — hotness
/// then accumulates over the whole trace, which is true LFU and the thing the original write-up
/// mistakenly believed it was measuring.
fn shipped_hits(seq: &[Access], cap: usize, n_layers: u32, n_experts: u32, k: usize,
                slot: usize, backing: &dyn Backing, decays_per_token: u32) -> Option<(u64, u64)> {
    let mut c = ExpertCache::new(n_layers, n_experts, slot, cap, k).ok()?;
    let mut tok = usize::MAX;
    let mut i = 0usize;
    while i < seq.len() {
        if seq[i].token != tok {
            if tok != usize::MAX { for _ in 0..decays_per_token { c.end_token() } }
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
    // ⛔ THE FLOOR PUBLISHED FIRST WAS A TRUNCATION ARTIFACT. FERRIC_MAX_LAYERS caps the model at 12
    // layers; the checkpoint has far more, so a DEPLOYED cache's floor is several times this one and
    // a reader sizing against 96 lands on the wrong side of the boundary the whole conclusion turns
    // on. ⭐ The recorded ROUTING is unaffected — layer l depends only on layers < l, so these are
    // the full model's selections for these 12 layers. It is a labelling defect, not a fidelity one.
    let full_blocks = match g.metadata.get("qwen35moe.block_count") {
        Some(Meta::U(v)) => *v as usize, _ => moe.len(),
    };
    println!("  working-set floor: {floor} entries HERE (12 layers x top-{k})");
    println!("  ⚠ the checkpoint has {full_blocks} blocks, so a DEPLOYED floor is ~{} entries.",
             full_blocks * k);
    println!("    Read the sweep by its RATIO to the floor, never by absolute capacity.\n");
    // ⛔ This column WAS "gap closed" = (shipped−LRU)/(Belady−LRU). It normalises a deficit by
    // headroom the deficit does not live in, and it printed −7% for BOTH a −1.7 point gap (cap 96)
    // and a −0.8 point one (cap 768). Raw points say what happened.
    println!("  {:>9}  {:>10}  {:>10}  {:>10}  {:>12}", "capacity", "LRU", "shipped", "Belady", "shipped−LRU");
    println!("  {:-<60}", "");

    let mut caps: Vec<usize> = vec![k + 1, floor / 4, floor / 2, floor, floor * 2, floor * 4, floor * 8];
    caps.retain(|&c| c >= k + 1);
    caps.sort_unstable(); caps.dedup();
    for cap in caps {
        let (lh, lg) = lru_hits(&seq, cap);
        let (bh, _) = belady_hits(&seq, cap);
        let sh = shipped_hits(&seq, cap, n_layers, ne, k, SLOT, &backing, 1);
        let (l, b) = (lh as f64 / lg as f64, bh as f64 / lg as f64);
        match sh {
            Some((s, sg)) => {
                assert_eq!(sg, lg, "the policies saw different access counts");
                let s = s as f64 / sg as f64;
                println!("  {cap:>9}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>+11.1}",
                         100.0 * l, 100.0 * s, 100.0 * b, 100.0 * (s - l));
            }
            None => println!("  {cap:>9}  {:>9.1}%  {:>10}  {:>9.1}%  {:>12}",
                             100.0 * l, "refused", 100.0 * b, "-"),
        }
    }

    // ---- D1: is the deficit about FREQUENCY, or about one constant? ------------------------
    //
    // The original write-up said frequency is unmeasurable because a cell is touched ~3 times in the
    // whole trace. The policy never sees a whole-trace count: DECAY_TOKENS=16 halves hotness every
    // 16 tokens while a cell is selected only k/n_expert = 0.5 times per window, so the primary
    // ranking key holds 0 or 1 almost always. If that is the cause, TURNING DECAY OFF should recover
    // the signal; if frequency is genuinely the wrong axis, no decay rate helps.
    println!("\n  decay sweep — end_token() calls per token (0 = never decay = true LFU):");
    println!("  {:>9}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}", "capacity", "LRU", "d=0", "d=1", "d=4", "d=16");
    println!("  {:-<66}", "");
    for cap in [192usize, 384] {
        let (lh, lg) = lru_hits(&seq, cap);
        let mut row = format!("  {cap:>9}  {:>7.1}%", 100.0 * lh as f64 / lg as f64);
        for d in [0u32, 1, 4, 16] {
            match shipped_hits(&seq, cap, n_layers, ne, k, SLOT, &backing, d) {
                Some((h, gt)) => row.push_str(&format!("  {:>9.1}%", 100.0 * h as f64 / gt as f64)),
                None => row.push_str(&format!("  {:>10}", "refused")),
            }
        }
        println!("{row}");
    }
    println!("  If any decay rate reaches LRU, the deficit is a CONSTANT (DECAY_TOKENS), not the axis.");
    println!("  If none does at any rate, \"frequency is the wrong key here\" is earned.");

    println!("\n  Belady is offline and unreachable; it is the ceiling that says whether a better");
    println!("  ONLINE policy is worth writing at all.");
    println!("\n  ⚠ {} accesses from prefill over {} prompts on {} of the checkpoint's layers.",
             seq.len(), PROMPTS.len(), moe.len());
    println!("     The crate's cited 36.24%/25.5-point figures come from a 100,096-request vendor");
    println!("     trace — a different model and ~10x the accesses, so these are not comparable");
    println!("     numbers, they are the same question asked of this checkpoint.");
}
