//! **Placement invariance, enforced.**
//!
//! kimi-k3-in-c advertises "byte-identical output at every budget" between 8 GB and 224 GB. colibri says
//! "placement only ever decides speed". Both are true of those systems, and both are claims in a README —
//! kimi-k3 does check its emitted token ids across a ladder, but nothing in any of the three projects
//! asserts the *weight-delivery* invariant itself.
//!
//! This file makes it a test. A full read trace is replayed at a ladder of memory budgets from "almost
//! nothing is resident" to "everything is resident", and every byte delivered must be identical at every
//! rung. If a future optimisation ever makes a cache return something subtly different — a stale slot, a
//! partially-overwritten buffer, an off-by-one in the ring — this fails, loudly, in CI, with no checkpoint
//! and no GPU required.
//!
//! Two guards against a vacuous pass, both learned from kimi-k3's own harness (whose ladder script
//! refuses to compare an unreadable result file, because comparing empty against empty "confirms"
//! identical output at every rung):
//!
//!   1. the ladder must actually span tiers — some rung must stream and some rung must be fully resident;
//!   2. the trace must be non-trivial — a checksum of the delivered bytes must be non-zero and must
//!      differ between distinct weights, or "all identical" is satisfied by returning nothing.

use ferric_tier::{Backing, ExpertCache, LayerCache, LayerDesc, Tier, TierError};
use std::cell::RefCell;

/// Deterministic synthetic checkpoint. Byte `i` is a pure function of its absolute offset, so any correct
/// cache must deliver identical bytes for identical ranges — which is exactly the invariant under test.
struct Checkpoint {
    reads: RefCell<u64>,
    bytes: RefCell<u64>,
}
impl Checkpoint {
    fn new() -> Self { Self { reads: RefCell::new(0), bytes: RefCell::new(0) } }
    #[inline]
    fn byte_at(off: u64) -> u8 {
        // A cheap mixer, so neighbouring offsets are not trivially similar and a slot boundary error
        // shows up as a mismatch rather than a coincidence.
        let x = off.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((x >> 29) ^ x) as u8
    }
}
impl Backing for Checkpoint {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        *self.reads.borrow_mut() += 1;
        *self.bytes.borrow_mut() += dst.len() as u64;
        for (i, b) in dst.iter_mut().enumerate() {
            *b = Checkpoint::byte_at(offset + i as u64);
        }
        Ok(())
    }
}

fn fnv1a(seed: u64, data: &[u8]) -> u64 {
    let mut h = seed;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

const N_LAYERS: usize = 40;
const LAYER_BYTES: u64 = 4096;
/// Layer 0 is deliberately the outlier, as it is in every real frontier MoE (a dense wide FFN at the
/// front). This is what makes exact-size pinning and streaming-only ring sizing matter.
const LAYER0_BYTES: u64 = 4096 * 3;

fn layer_shape() -> Vec<LayerDesc> {
    let mut v = Vec::with_capacity(N_LAYERS);
    let mut off = 0u64;
    v.push(LayerDesc { offset: off, bytes: LAYER0_BYTES });
    off += LAYER0_BYTES;
    for _ in 1..N_LAYERS {
        v.push(LayerDesc { offset: off, bytes: LAYER_BYTES });
        off += LAYER_BYTES;
    }
    v
}

/// Replay a fixed cyclic layer walk and return (checksum of every delivered byte, hit rate, tiers seen).
fn run_layer_ladder(budget: u64) -> (u64, f64, Vec<Tier>) {
    let ck = Checkpoint::new();
    let mut cache = LayerCache::with_budget(layer_shape(), budget, 0, 64)
        .unwrap_or_else(|e| panic!("budget {budget} rejected: {e}"));
    // Pin at startup, as a real deployment does. Without this the cold first token counts every pinned
    // layer as a miss, and the measured rate is a startup artifact rather than the policy's steady state.
    cache.prefill(&ck).unwrap();
    let mut sum = 0xcbf2_9ce4_8422_2325u64;
    let mut tiers = Vec::new();
    for _token in 0..4 {
        for l in 0..N_LAYERS as u32 {
            let (bytes, tier) = cache.bind(l, &ck).unwrap();
            sum = fnv1a(sum, bytes);
            tiers.push(tier);
        }
    }
    (sum, cache.stats().hit_rate(), tiers)
}

#[test]
fn layer_bytes_are_identical_at_every_budget() {
    // A ladder from "one layer barely fits" to "everything is resident", mirroring kimi-k3's 8 GB → 224 GB
    // memory ladder in miniature.
    let ladder: Vec<u64> = vec![
        LAYER0_BYTES + 512,   // floor: essentially everything streams
        24 * 1024,
        40 * 1024,
        64 * 1024,
        96 * 1024,
        128 * 1024,
        160 * 1024,
        1 << 22,              // ceiling: everything is pinned
    ];

    let mut reference: Option<u64> = None;
    let mut rates = Vec::new();
    let mut any_streamed = false;
    let mut any_fully_resident = false;

    for &budget in &ladder {
        let (sum, rate, tiers) = run_layer_ladder(budget);
        match reference {
            None => reference = Some(sum),
            Some(r) => assert_eq!(
                sum, r,
                "PLACEMENT CHANGED RESULTS: budget {budget} delivered different bytes than the first rung. \
                 The memory budget must decide WHERE bytes come from, never WHAT they are."
            ),
        }
        rates.push((budget, rate));
        if tiers.iter().any(|t| *t == Tier::Backing) { any_streamed = true; }
        if tiers.iter().all(|t| *t == Tier::Pinned) { any_fully_resident = true; }
    }

    // --- anti-vacuity ---
    assert!(any_streamed, "no rung actually streamed — the ladder never exercised the miss path");
    assert!(any_fully_resident, "no rung was fully resident — the ladder never exercised the pinned path");
    let sum = reference.unwrap();
    assert_ne!(sum, 0xcbf2_9ce4_8422_2325u64, "checksum unchanged: no bytes were ever delivered");

    // The invariant is byte-identity; the *payoff* is that more memory buys more hits. Assert that too,
    // so a cache that trivially satisfies invariance by never caching anything cannot pass.
    let (_, lo) = rates[0];
    let (_, hi) = rates[rates.len() - 1];
    assert!(hi > lo, "hit rate did not improve with budget ({lo:.3} -> {hi:.3})");
    assert!(
        (hi - 1.0).abs() < 1e-9,
        "a fully-resident rung must be a pure cache hit in steady state, got {hi:.3}"
    );
}

#[test]
fn expert_bytes_are_identical_at_every_capacity() {
    const N_LAYERS_E: u32 = 6;
    const N_EXPERTS: u32 = 48;
    const SLOT: usize = 256;
    const TOP_K: usize = 4;
    let off = |l: u32, e: u32| (l as u64 * N_EXPERTS as u64 + e as u64) * SLOT as u64;

    // A fixed, reproducible routing trace with genuine reuse and a skewed head, so cache policy is
    // actually exercised rather than degenerating into a scan.
    let mut trace: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut s = 0x1234_5678u64;
    for _tok in 0..60 {
        for l in 0..N_LAYERS_E {
            let mut sel = Vec::with_capacity(TOP_K);
            while sel.len() < TOP_K {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Bias toward a hot head: two thirds of picks land in the first 8 experts.
                let e = if (s >> 60) % 3 != 0 { ((s >> 20) % 8) as u32 } else { ((s >> 20) % N_EXPERTS as u64) as u32 };
                if !sel.contains(&e) { sel.push(e); }
            }
            trace.push((l, sel));
        }
    }

    let replay = |capacity: usize| -> (u64, f64, u64) {
        let ck = Checkpoint::new();
        let mut c = ExpertCache::new(N_LAYERS_E, N_EXPERTS, SLOT, capacity, TOP_K).unwrap();
        let mut sum = 0xcbf2_9ce4_8422_2325u64;
        for (i, (l, sel)) in trace.iter().enumerate() {
            c.note_selected(*l, sel);
            for &e in sel {
                let (bytes, _) = c.get(*l, e, off(*l, e), &ck).unwrap();
                sum = fnv1a(sum, bytes);
            }
            if (i + 1) % N_LAYERS_E as usize == 0 { c.end_token(); }
        }
        (sum, c.stats().hit_rate(), c.stats().evictions)
    };

    let capacities = [TOP_K + 1, 8, 16, 32, 64, 128, (N_LAYERS_E * N_EXPERTS) as usize];
    let mut reference: Option<u64> = None;
    let mut first_rate = 0.0;
    let mut last_rate = 0.0;
    let mut saw_eviction = false;
    let mut saw_no_eviction = false;

    for (i, &cap) in capacities.iter().enumerate() {
        let (sum, rate, evictions) = replay(cap);
        match reference {
            None => { reference = Some(sum); first_rate = rate; }
            Some(r) => assert_eq!(
                sum, r,
                "PLACEMENT CHANGED RESULTS: capacity {cap} delivered different expert bytes than capacity {}",
                capacities[0]
            ),
        }
        if i == capacities.len() - 1 { last_rate = rate; }
        if evictions > 0 { saw_eviction = true; } else { saw_no_eviction = true; }
    }

    assert!(saw_eviction, "no capacity evicted anything — the ladder never exercised the policy");
    assert!(saw_no_eviction, "every capacity evicted — the ladder never reached full residency");
    assert!(last_rate > first_rate, "hit rate did not improve with capacity ({first_rate:.3} -> {last_rate:.3})");
}

/// The two policies must not be interchangeable, and this is the measurement that proves it.
///
/// Recency-only eviction on a cyclic walk is the pathology [`LayerCache`] exists to avoid: it scores
/// **exactly zero** when the working set exceeds capacity, at any capacity below the layer count. The
/// pinned prefix on the identical trace scores `npin/n`. If this test ever stops showing a large gap,
/// someone has replaced the layer policy with an LRU and the hit rate has quietly gone to nothing.
#[test]
fn a_recency_policy_would_score_zero_on_the_cyclic_walk() {
    // Simulate pure LRU over the same cyclic access pattern, policy only, no I/O.
    fn lru_cyclic_hit_rate(n_layers: usize, capacity: usize, tokens: usize) -> f64 {
        let mut slots: Vec<Option<usize>> = vec![None; capacity];
        let mut used: Vec<u64> = vec![0; capacity];
        let (mut clock, mut hits, mut total) = (0u64, 0u64, 0u64);
        for _ in 0..tokens {
            for l in 0..n_layers {
                clock += 1;
                total += 1;
                if let Some(s) = slots.iter().position(|x| *x == Some(l)) {
                    hits += 1;
                    used[s] = clock;
                } else {
                    let v = (0..capacity).min_by_key(|&i| used[i]).unwrap();
                    slots[v] = Some(l);
                    used[v] = clock;
                }
            }
        }
        hits as f64 / total as f64
    }

    let n = 40;
    for capacity in [4usize, 10, 20, 30, 39] {
        let lru = lru_cyclic_hit_rate(n, capacity, 6);
        assert!(
            lru < 1e-9,
            "LRU scored {lru:.4} at capacity {capacity}/{n}; the cyclic pathology is supposed to be total"
        );
    }
    // Capacity == n_layers is the only case LRU survives, and it is the case where no policy is needed.
    assert!(lru_cyclic_hit_rate(n, n, 6) > 0.8);

    // The shipped policy on the same shape and a comparable budget.
    let budget = 20 * LAYER_BYTES + LAYER0_BYTES;
    let (_, rate, _) = run_layer_ladder(budget);
    assert!(
        rate > 0.4,
        "pinned prefix scored {rate:.3}; it should be ~npin/n, not the LRU's zero"
    );
}
