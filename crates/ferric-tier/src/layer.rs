//! Layer streaming: **pinned prefix + ring. Not an LRU.**
//!
//! A transformer walks its layers cyclically — 0, 1, ... N-1, 0, 1, ... — once per token, forever. That
//! is the exact pathological case for LRU: with fewer slots than layers, the walk arrives back at layer 0
//! precisely when layer 0 has become the least-recently-used entry and was evicted to make room. The hit
//! rate is **zero, and stays zero no matter how much memory you add**. Buying RAM does not help, which
//! makes it a genuinely nasty bug: the cache looks correct, reports plausible statistics, and never works.
//!
//! Pinning a prefix instead gives a hit rate of exactly `npin/n_layers`, deterministically, with every
//! additional byte buying its fair share.
//!
//! The corollary is worth stating because it is easy to get backwards: **this policy is right for layers
//! precisely because their access order is fixed, and wrong for experts, whose access is data-dependent.**
//! See [`crate::ExpertCache`] for the other half.

use crate::{plan::RING_SLOTS, Backing, LayerDesc, LayerPlan, Tier, TierError};

/// Observability. Reported so a caller can size a budget — never so it can branch on placement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerStats {
    pub binds: u64,
    pub pinned_hits: u64,
    pub ring_hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
}

impl LayerStats {
    pub fn hit_rate(&self) -> f64 {
        if self.binds == 0 { return 0.0; }
        (self.pinned_hits + self.ring_hits) as f64 / self.binds as f64
    }
}

/// One ring slot's occupancy.
///
/// Three states, not two. `Loading` exists because the two-state version has a real corruption bug: see
/// [`LayerCache::bind`]. It is meaningful even single-threaded, since a failed read must not leave the
/// slot claiming to hold anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Empty,
    Loading,
    Holds(u32),
}

/// Pinned-prefix + ring cache over a layer-major backing store.
#[derive(Debug)]
pub struct LayerCache {
    plan: LayerPlan,
    layers: Vec<LayerDesc>,
    /// Exact-size buffers for `0..npin`. Exact rather than uniform because one outlier layer would
    /// otherwise set a slot size that every other layer underfills — roughly half the pinned budget.
    pinned: Vec<Vec<u8>>,
    pinned_filled: Vec<bool>,
    ring: Vec<Vec<u8>>,
    ring_state: Vec<SlotState>,
    next_slot: usize,
    stats: LayerStats,
}

impl LayerCache {
    /// Build a cache from a resolved plan. Allocates the pinned buffers and the ring up front so a
    /// mid-token allocation failure is impossible; contents are filled on first touch.
    pub fn new(plan: LayerPlan, layers: Vec<LayerDesc>) -> Self {
        let pinned: Vec<Vec<u8>> =
            layers[..plan.npin].iter().map(|l| vec![0u8; l.bytes as usize]).collect();
        let pinned_filled = vec![false; plan.npin];
        let ring: Vec<Vec<u8>> = (0..RING_SLOTS).map(|_| vec![0u8; plan.ring_slot as usize]).collect();
        let ring_state = vec![SlotState::Empty; RING_SLOTS as usize];
        Self { plan, layers, pinned, pinned_filled, ring, ring_state, next_slot: 0, stats: LayerStats::default() }
    }

    /// Plan a budget and build in one step.
    pub fn with_budget(layers: Vec<LayerDesc>, budget: u64, widen: u64, align: u64) -> Result<Self, TierError> {
        let plan = crate::plan_layers(&layers, budget, widen, align);
        // Refuse up front rather than fail mid-token. A plan that does not fit cannot bind even a single
        // layer, and reporting that as a budget error at construction beats reporting it as a read
        // failure once per layer per token.
        if !plan.fits(budget) {
            return Err(TierError::BudgetTooSmall { need: plan.spent, have: budget });
        }
        Ok(Self::new(plan, layers))
    }

    pub fn plan(&self) -> &LayerPlan { &self.plan }
    pub fn stats(&self) -> LayerStats { self.stats }

    /// Bind a layer, returning its bytes and where they came from.
    ///
    /// The returned slice is **byte-identical regardless of tier** — that is the invariant this whole
    /// crate exists to hold, and `tests/placement_invariance.rs` asserts it across a dozen budgets.
    ///
    /// # The corruption trap
    ///
    /// A slot is marked `Loading` **before** the read and only marked `Holds(layer)` **after the read
    /// succeeds**. Doing it the other way round — registering the layer first, or leaving the old tag in
    /// place on failure — produces the worst class of bug in a streaming engine: a partial read overwrites
    /// the front of the buffer, the slot still claims to hold the *previous* layer, and the next bind of
    /// that previous layer counts a **HIT**, skips the read, and returns weights that are half one layer
    /// and half another. No error is raised; the model simply emits plausible, wrong tokens.
    pub fn bind(&mut self, layer: u32, backing: &dyn Backing) -> Result<(&[u8], Tier), TierError> {
        let li = layer as usize;
        let desc = *self
            .layers
            .get(li)
            .ok_or(TierError::OutOfRange(crate::WeightId::layer(layer)))?;
        self.stats.binds += 1;

        // --- pinned prefix: permanent once filled ---
        if li < self.plan.npin {
            if !self.pinned_filled[li] {
                backing.read_at(desc.offset, &mut self.pinned[li])?;
                self.pinned_filled[li] = true;
                self.stats.bytes_read += desc.bytes;
                self.stats.misses += 1;
            } else {
                self.stats.pinned_hits += 1;
            }
            return Ok((&self.pinned[li], Tier::Pinned));
        }

        // --- ring: hit? ---
        if let Some(s) = self.ring_state.iter().position(|st| *st == SlotState::Holds(layer)) {
            self.stats.ring_hits += 1;
            return Ok((&self.ring[s][..desc.bytes as usize], Tier::Cached));
        }

        // --- ring: miss. Choose a victim, invalidate it, then read. ---
        let s = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.ring.len();
        self.ring_state[s] = SlotState::Loading; // BEFORE the read. See the doc comment above.
        let n = desc.bytes as usize;
        backing.read_at(desc.offset, &mut self.ring[s][..n])?; // `?` leaves the slot Loading, i.e. invalid
        self.ring_state[s] = SlotState::Holds(layer); // only now is the slot truthful
        self.stats.misses += 1;
        self.stats.bytes_read += desc.bytes;
        Ok((&self.ring[s][..n], Tier::Backing))
    }

    /// Eagerly fill the pinned prefix. Optional — `bind` fills lazily — but doing it up front moves the
    /// cost to startup where it is visible, instead of onto the first token where it looks like latency.
    pub fn prefill(&mut self, backing: &dyn Backing) -> Result<(), TierError> {
        for li in 0..self.plan.npin {
            if !self.pinned_filled[li] {
                backing.read_at(self.layers[li].offset, &mut self.pinned[li])?;
                self.pinned_filled[li] = true;
                self.stats.bytes_read += self.layers[li].bytes;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Deterministic synthetic store: byte `i` of the file is `(i * 31 + 7) as u8`. Content is a pure
    /// function of offset, so any two reads of the same range must agree — which is what lets the tests
    /// below check placement-invariance without a checkpoint.
    struct Synth {
        reads: RefCell<u64>,
    }
    impl Synth {
        fn new() -> Self { Self { reads: RefCell::new(0) } }
        fn byte_at(off: u64) -> u8 { (off.wrapping_mul(31).wrapping_add(7) & 0xff) as u8 }
    }
    impl Backing for Synth {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
            *self.reads.borrow_mut() += 1;
            for (i, b) in dst.iter_mut().enumerate() {
                *b = Synth::byte_at(offset + i as u64);
            }
            Ok(())
        }
    }

    /// A store that fails partway through, to prove a failed read cannot leave a slot claiming a layer.
    struct Failing {
        fail_after: RefCell<u32>,
    }
    impl Backing for Failing {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
            let mut n = self.fail_after.borrow_mut();
            if *n == 0 {
                // Scribble on the buffer first, exactly as a real partial read would.
                for b in dst.iter_mut() { *b = 0xAA; }
                return Err(TierError::ShortRead { want: dst.len(), got: dst.len() / 2 });
            }
            *n -= 1;
            for (i, b) in dst.iter_mut().enumerate() { *b = Synth::byte_at(offset + i as u64); }
            Ok(())
        }
    }

    fn shape(n: usize, bytes: u64) -> Vec<LayerDesc> {
        (0..n).map(|i| LayerDesc { offset: i as u64 * bytes, bytes }).collect()
    }

    #[test]
    fn cyclic_access_does_not_collapse_to_zero_hits() {
        // THE regression this policy exists for. An LRU with fewer slots than layers scores exactly 0 on
        // this workload; a pinned prefix scores npin/n.
        let layers = shape(93, 1000);
        let budget = 40 * 1000 + 1000; // room for ~40 pinned layers plus one ring slot
        let mut c = LayerCache::with_budget(layers, budget, 0, 64).unwrap();
        let b = Synth::new();
        // Prefill so the measurement is steady-state. Without it the cold first token counts every pinned
        // layer as a miss and drags the rate below the plan's promise for reasons that have nothing to do
        // with the policy under test.
        c.prefill(&b).unwrap();
        for _tok in 0..5 {
            for l in 0..93u32 {
                c.bind(l, &b).unwrap();
            }
        }
        let hr = c.stats().hit_rate();
        assert!(hr > 0.35, "cyclic hit rate collapsed to {hr:.3} — this is the LRU pathology");
        // Exactly the plan's promise, not merely close: a pinned prefix's hit rate is arithmetic, not an
        // empirical property. If these ever diverge, the policy is not doing what the plan says.
        assert!(
            (hr - c.plan().hit_rate()).abs() < 1e-9,
            "measured {hr:.6} vs planned {:.6} — a deterministic policy must hit its stated rate",
            c.plan().hit_rate()
        );
    }

    #[test]
    fn bytes_are_identical_whatever_tier_serves_them() {
        let layers = shape(20, 512);
        let b = Synth::new();
        // Tiny budget: almost everything streams. Huge budget: everything is pinned.
        let mut small = LayerCache::with_budget(layers.clone(), 1600, 0, 64).unwrap();
        let mut large = LayerCache::with_budget(layers.clone(), 1 << 20, 0, 64).unwrap();
        for l in 0..20u32 {
            let (a, ta) = small.bind(l, &b).unwrap();
            let a = a.to_vec();
            let (z, tz) = large.bind(l, &b).unwrap();
            assert_eq!(a, z, "layer {l} differs between budgets — placement changed RESULTS");
            assert_ne!((ta, tz), (Tier::Pinned, Tier::Backing), "test did not actually exercise two tiers");
        }
        assert_eq!(large.plan().npin, 20, "large budget should pin everything");
    }

    #[test]
    fn failed_read_never_leaves_a_slot_claiming_a_layer() {
        // The silent-corruption trap: if a failed read left the slot tagged, the NEXT bind of the layer
        // that was previously in the slot would count a hit and return half-overwritten bytes.
        let layers = shape(8, 256);
        let mut c = LayerCache::with_budget(layers, 600, 0, 64).unwrap();
        let good = Synth::new();
        let npin = c.plan().npin;
        let streamed: u32 = npin as u32; // first layer that actually uses the ring

        // Warm the ring with a streamed layer.
        let (want, _) = c.bind(streamed, &good).unwrap();
        let want = want.to_vec();

        // Now fail a read of a DIFFERENT streamed layer into that same slot.
        let bad = Failing { fail_after: RefCell::new(0) };
        assert!(c.bind(streamed + 1, &bad).is_err(), "read should have failed");

        // Re-binding the original layer must re-read, not serve the scribbled buffer.
        let (got, tier) = c.bind(streamed, &good).unwrap();
        assert_eq!(got, &want[..], "served corrupt bytes after a failed read into the same slot");
        assert_eq!(tier, Tier::Backing, "should have re-read, not counted a hit on a poisoned slot");
    }

    #[test]
    fn budget_that_cannot_hold_one_layer_is_refused_up_front() {
        let layers = shape(4, 4096);
        let e = LayerCache::with_budget(layers, 100, 0, 64).unwrap_err();
        assert!(matches!(e, TierError::BudgetTooSmall { .. }), "got {e:?}");
    }

    #[test]
    fn pinned_layers_are_read_exactly_once() {
        let layers = shape(6, 128);
        let mut c = LayerCache::with_budget(layers, 1 << 16, 0, 64).unwrap();
        let b = Synth::new();
        for _ in 0..10 {
            for l in 0..6u32 { c.bind(l, &b).unwrap(); }
        }
        assert_eq!(*b.reads.borrow(), 6, "pinned layers must be read once, not per token");
    }
}
