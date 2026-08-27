//! Expert caching: hotness-LFU with an LRU tiebreak.
//!
//! The opposite situation from [`crate::LayerCache`]. Expert access is **data-dependent** — which experts
//! fire depends on the token — so recency and frequency genuinely predict reuse and a real cache policy
//! earns its keep.
//!
//! ## Why pure LRU underperforms here, and by how much
//!
//! Frontier MoE routers are increasingly trained to *flatten* expert usage (Kimi K3 uses Quantile
//! Balancing explicitly for this). Flat usage is precisely what LRU cannot exploit. Measured on a shipped
//! 100,096-request trace: LRU sits at **36.24%, dead flat from 8 GB to 64 GB**, while Belady's optimum
//! climbs 39% → 62% over the same range. **A 25.5-point gap at identical memory** — so on this workload
//! policy is worth far more than capacity, which is the reverse of the usual intuition.
//!
//! Frequency-primary with recency as the tiebreak is the cheap approximation of that: it keeps an expert
//! that fires often but not recently, which is exactly the entry pure LRU throws away.
//!
//! ## Decay, and why a "pin" is softer than it looks
//!
//! ## Sizing: below one token's working set, no policy helps
//!
//! Measured on a real MoE (`ferric-llama/examples/moe_streaming.rs`): a cache spanning 6 layers scored a
//! **0.0% hit rate at both 7 and 10 entries**, and 75% at 48. Iterating layers 0..N every token makes the
//! combined `(layer, expert)` access **cyclic**, which is the same pathology that makes an LRU worthless
//! for layer streaming — an entry is evicted before the walk comes back to it.
//!
//! So the floor is not a tuning preference: an expert cache must hold at least `n_layers × top_k` entries
//! to score at all. Below that the policy is irrelevant and only the size matters; above it, policy is
//! what closes the Belady gap. [`ExpertCache::new`] enforces only the per-STEP minimum (`top_k + 1`),
//! because it cannot know the layer count — the caller owns this one.
//!
//! Hotness is halved every [`DECAY_TOKENS`] tokens so an expert that was hot during one phase of a
//! generation does not hold a slot forever. This has a consequence worth stating because the systems that
//! ship it do not: seeding hotness from a startup popularity list produces a pin that **decays to nothing
//! within a few dozen tokens**. It is a warm start, not a reservation. If you need a true reservation, use
//! [`ExpertCache::pin`], which is exempt from both decay and eviction.

use crate::{Backing, Tier, TierError, WeightId};

/// Tokens between hotness halvings. Small enough that stale heat fades within a generation, large enough
/// that a genuinely hot expert survives a lull.
pub const DECAY_TOKENS: u64 = 16;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertStats {
    pub gets: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
}

impl ExpertStats {
    pub fn hit_rate(&self) -> f64 {
        if self.gets == 0 { return 0.0; }
        self.hits as f64 / self.gets as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Empty,
    /// Reserved for a read in progress. A slot in this state is **not a valid cache entry** and is never
    /// a hit and never a victim. Without this third state, a failed or concurrent read can publish a slot
    /// whose tag and contents disagree.
    Loading,
    Holds { layer: u32, expert: u32 },
}

/// Fixed-capacity expert cache with uniform slots.
///
/// Uniform because every routed expert in a well-formed checkpoint is the same size; an off-class expert
/// (a mixed-precision "boosted" layer) is refused rather than silently given a slot it does not fit.
#[derive(Debug)]
pub struct ExpertCache {
    n_layers: u32,
    n_experts: u32,
    slot_bytes: usize,
    buffers: Vec<Vec<u8>>,
    slots: Vec<Slot>,
    last_used: Vec<u64>,
    /// Dense `(layer, expert) -> slot`. Direct-indexed, no hashing: the shape is known and small, and an
    /// O(1) array index beats a hash lookup on the hot path.
    index: Vec<Option<u32>>,
    hotness: Vec<u32>,
    pinned: Vec<bool>,
    /// Experts protected for the step currently executing. Evicting one of these would drop a weight that
    /// is still being multiplied.
    protected: Vec<(u32, u32)>,
    clock: u64,
    tokens: u64,
    stats: ExpertStats,
}

impl ExpertCache {
    /// `capacity` is the number of resident experts. Refuses a capacity that cannot hold one step's
    /// working set: `top_k` experts fire together, and a cache smaller than that would evict an expert
    /// that the current step is still using. One spare slot on top keeps a miss from evicting a peer.
    pub fn new(
        n_layers: u32,
        n_experts: u32,
        slot_bytes: usize,
        capacity: usize,
        top_k: usize,
    ) -> Result<Self, TierError> {
        if capacity < top_k + 1 {
            return Err(TierError::BudgetTooSmall {
                need: ((top_k + 1) * slot_bytes) as u64,
                have: (capacity * slot_bytes) as u64,
            });
        }
        let cells = (n_layers as usize) * (n_experts as usize);
        Ok(Self {
            n_layers,
            n_experts,
            slot_bytes,
            buffers: (0..capacity).map(|_| vec![0u8; slot_bytes]).collect(),
            slots: vec![Slot::Empty; capacity],
            last_used: vec![0; capacity],
            index: vec![None; cells],
            hotness: vec![0; cells],
            pinned: vec![false; capacity],
            protected: Vec::with_capacity(top_k),
            clock: 0,
            tokens: 0,
            stats: ExpertStats::default(),
        })
    }

    pub fn stats(&self) -> ExpertStats { self.stats }
    pub fn capacity(&self) -> usize { self.slots.len() }

    #[inline]
    fn cell(&self, layer: u32, expert: u32) -> Result<usize, TierError> {
        if layer >= self.n_layers || expert >= self.n_experts {
            return Err(TierError::OutOfRange(WeightId::expert(layer, expert)));
        }
        Ok(layer as usize * self.n_experts as usize + expert as usize)
    }

    /// Declare the experts this step will use: bumps their hotness and protects them from eviction for
    /// the duration of the step. Call once per (token, layer) before the `get` calls.
    pub fn note_selected(&mut self, layer: u32, experts: &[u32]) {
        self.protected.clear();
        for &e in experts {
            if let Ok(c) = self.cell(layer, e) {
                self.hotness[c] = self.hotness[c].saturating_add(1);
                self.protected.push((layer, e));
            }
        }
    }

    /// Seed hotness from a startup popularity list — a warm start, not a reservation. See the module
    /// docs: decay erodes this within a few dozen tokens. Use [`Self::pin`] for something permanent.
    pub fn seed_hotness(&mut self, entries: &[(u32, u32, u32)]) {
        for &(layer, expert, heat) in entries {
            if let Ok(c) = self.cell(layer, expert) {
                self.hotness[c] = self.hotness[c].saturating_add(heat);
            }
        }
    }

    /// End of a decode step: age the hotness table. Halving (rather than subtracting) keeps the ordering
    /// of long-lived hot entries intact while letting a one-off spike fade geometrically.
    pub fn end_token(&mut self) {
        self.tokens += 1;
        if self.tokens % DECAY_TOKENS == 0 {
            for h in self.hotness.iter_mut() { *h >>= 1; }
        }
    }

    /// Make an already-resident expert permanently exempt from eviction and decay.
    pub fn pin(&mut self, layer: u32, expert: u32) -> Result<bool, TierError> {
        let c = self.cell(layer, expert)?;
        match self.index[c] {
            Some(s) => { self.pinned[s as usize] = true; Ok(true) }
            None => Ok(false),
        }
    }

    /// Fetch one expert. Returns bytes identical regardless of which tier served them.
    ///
    /// `offset` is where this expert lives in the backing store; the caller owns that mapping because it
    /// is checkpoint-format-specific and this crate is deliberately format-agnostic.
    pub fn get(
        &mut self,
        layer: u32,
        expert: u32,
        offset: u64,
        backing: &dyn Backing,
    ) -> Result<(&[u8], Tier), TierError> {
        let c = self.cell(layer, expert)?;
        self.stats.gets += 1;
        self.clock += 1;

        if let Some(s) = self.index[c] {
            let s = s as usize;
            self.last_used[s] = self.clock;
            self.stats.hits += 1;
            let tier = if self.pinned[s] { Tier::Pinned } else { Tier::Cached };
            return Ok((&self.buffers[s][..], tier));
        }

        let victim = self.pick_victim()?;
        // Drop the old occupant's index entry BEFORE the read, so a failure cannot leave a cell pointing
        // at a slot whose contents have been overwritten. Same trap as LayerCache::bind.
        if let Slot::Holds { layer: ol, expert: oe } = self.slots[victim] {
            let oc = ol as usize * self.n_experts as usize + oe as usize;
            self.index[oc] = None;
            self.stats.evictions += 1;
        }
        self.slots[victim] = Slot::Loading;
        backing.read_at(offset, &mut self.buffers[victim][..])?; // on `?`, the slot stays Loading = invalid

        self.slots[victim] = Slot::Holds { layer, expert };
        self.index[c] = Some(victim as u32);
        self.last_used[victim] = self.clock;
        self.stats.misses += 1;
        self.stats.bytes_read += self.slot_bytes as u64;
        Ok((&self.buffers[victim][..], Tier::Backing))
    }

    /// **Decide where an expert lives, moving no bytes.** The GPU counterpart of [`Self::get`].
    ///
    /// `get` copies into a host buffer this cache owns, which is the wrong shape for a weight that
    /// must end up in a GPU slab — it would hold every expert twice. `place` runs the same residency
    /// and eviction logic and returns the SLOT, so a runtime can `write_rows` a fetched expert into
    /// that row of its slab. This is the call that was missing between this crate and the tensor
    /// layer, and the reason `ExpertCache` had no runtime consumer.
    ///
    /// Returns `(slot, was_miss)`. On a miss the slot is marked [`Slot::Loading`] — never a hit and
    /// never a victim — and the caller **must** call [`Self::commit`] once the bytes are in place, or
    /// [`Self::abort`] if the fetch failed. That is the same discipline `get` follows on its error
    /// path: a slot whose tag and contents could disagree is never published.
    ///
    /// ⚠ Placing the same expert twice without committing is an ERROR, not a second allocation.
    /// Without that check a caller retrying a failed fetch would burn a fresh slot each attempt and
    /// quietly shrink the cache to nothing.
    pub fn place(&mut self, layer: u32, expert: u32) -> Result<(usize, bool), TierError> {
        let c = self.cell(layer, expert)?;
        self.stats.gets += 1;
        self.clock += 1;

        if let Some(s) = self.index[c] {
            let s = s as usize;
            // A hit needs BOTH the index entry and a slot that actually holds this expert. An index
            // pointing at a `Loading` slot means a place is outstanding, not that the expert is here.
            if self.slots[s] == (Slot::Holds { layer, expert }) {
                self.last_used[s] = self.clock;
                self.stats.hits += 1;
                return Ok((s, false));
            }
            return Err(TierError::Io(format!(
                "expert ({layer}, {expert}) is already placed in slot {s} and not yet committed;                  placing again would allocate a second slot and leak the first")));
        }

        let victim = self.pick_victim()?;
        // Drop the old occupant's index entry BEFORE the slot changes state, so a failed fill cannot
        // leave a cell pointing at a row whose contents have been overwritten.
        if let Slot::Holds { layer: ol, expert: oe } = self.slots[victim] {
            let oc = ol as usize * self.n_experts as usize + oe as usize;
            self.index[oc] = None;
            self.stats.evictions += 1;
        }
        self.slots[victim] = Slot::Loading;
        // Point the cell at the reserved slot so a duplicate `place` is detected rather than served.
        self.index[c] = Some(victim as u32);
        Ok((victim, true))
    }

    /// Publish a slot filled after [`Self::place`]. Until this runs the slot is not a valid entry.
    pub fn commit(&mut self, slot: usize, layer: u32, expert: u32) -> Result<(), TierError> {
        let c = self.cell(layer, expert)?;
        if self.slots.get(slot) != Some(&Slot::Loading) {
            return Err(TierError::Io(format!(
                "commit on slot {slot}, which is not awaiting a fill — commit without a matching                  place would publish whatever bytes happen to be there")));
        }
        if self.index[c] != Some(slot as u32) {
            return Err(TierError::Io(format!(
                "commit of ({layer}, {expert}) into slot {slot}, which was reserved for something                  else — publishing here would give the slot a tag its contents do not match")));
        }
        self.slots[slot] = Slot::Holds { layer, expert };
        self.last_used[slot] = self.clock;
        self.stats.misses += 1;
        self.stats.bytes_read += self.slot_bytes as u64;
        Ok(())
    }

    /// Release a slot reserved by [`Self::place`] whose fill failed.
    ///
    /// ⚠ Without this a failed fetch strands a `Loading` slot forever — never a hit, never a victim —
    /// so the cache silently shrinks by one every time a read fails.
    pub fn abort(&mut self, slot: usize, layer: u32, expert: u32) -> Result<(), TierError> {
        let c = self.cell(layer, expert)?;
        if self.slots.get(slot) != Some(&Slot::Loading) {
            return Err(TierError::Io(format!("abort on slot {slot}, which is not awaiting a fill")));
        }
        // ⛔ THE TAG MUST MATCH, for the same reason `commit` checks it. The first version freed the
        // slot on ANY tag and only cleared the index when it happened to match — so aborting under
        // the wrong expert emptied a slot while the RIGHT expert's cell still pointed at it. The
        // next `place` for that expert then reported a hit on an Empty slot, i.e. served whatever
        // bytes were there. Freeing is destructive, so it needs the same authority as publishing.
        if self.index[c] != Some(slot as u32) {
            return Err(TierError::Io(format!(
                "abort of ({layer}, {expert}) on slot {slot}, which was reserved for something else \
                 — freeing it would strand the reserving expert's index entry")));
        }
        self.index[c] = None;
        self.slots[slot] = Slot::Empty;
        Ok(())
    }

    /// Victim selection: **lowest hotness wins; ties broken by oldest use.**
    ///
    /// Frequency-primary is the whole point — see the module docs on the 25.5-point Belady gap that pure
    /// recency leaves on the table under a usage-flattening router.
    fn pick_victim(&self) -> Result<usize, TierError> {
        let mut best: Option<(u32, u64, usize)> = None;
        for (s, slot) in self.slots.iter().enumerate() {
            match slot {
                // An empty slot is free real estate; take it immediately.
                Slot::Empty => return Ok(s),
                // Never evict a read in progress, a pin, or an expert this step is still using.
                Slot::Loading => continue,
                Slot::Holds { layer, expert } => {
                    if self.pinned[s] || self.protected.contains(&(*layer, *expert)) { continue; }
                    let c = *layer as usize * self.n_experts as usize + *expert as usize;
                    let key = (self.hotness[c], self.last_used[s], s);
                    if best.is_none_or(|b| (key.0, key.1) < (b.0, b.1)) {
                        best = Some(key);
                    }
                }
            }
        }
        best.map(|(_, _, s)| s).ok_or(TierError::BudgetTooSmall {
            need: ((self.protected.len() + 1) * self.slot_bytes) as u64,
            have: (self.slots.len() * self.slot_bytes) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;


    /// `place` must reuse the slot a resident expert already occupies, without evicting anything.
    #[test]
    fn placing_a_resident_expert_returns_its_slot_and_evicts_nothing() {
        let mut c = ExpertCache::new(2, 8, 64, 5, 2).unwrap();
        c.note_selected(0, &[3]);
        let (slot, miss) = c.place(0, 3).unwrap();
        assert!(miss, "first placement must be a miss");
        c.commit(slot, 0, 3).unwrap();
        let (again, miss2) = c.place(0, 3).unwrap();
        assert_eq!(again, slot, "a resident expert moved slots");
        assert!(!miss2, "a resident expert reported a miss");
        assert_eq!(c.stats().evictions, 0);
    }

    /// ⛔ Until `commit`, the expert is NOT resident. A `place` that published immediately would
    /// hand out a slot whose bytes the caller has not written yet — the tag and contents disagree,
    /// which is exactly what `Slot::Loading` exists to prevent.
    #[test]
    fn an_uncommitted_placement_is_not_a_hit_and_is_not_a_victim() {
        let mut c = ExpertCache::new(1, 8, 64, 3, 2).unwrap();
        let (s0, _) = c.place(0, 1).unwrap();
        // Same expert again: an error, not a second slot. Retrying a failed fetch must not leak.
        assert!(c.place(0, 1).is_err(), "a duplicate place allocated a second slot");
        // A different expert must not be given the reserved slot.
        let (s1, _) = c.place(0, 2).unwrap();
        assert_ne!(s1, s0, "a Loading slot was handed out as a victim");
        c.commit(s1, 0, 2).unwrap();
        // And the reserved slot is still not a valid entry.
        assert!(c.place(0, 2).unwrap().0 == s1, "committed expert lost its slot");
    }

    /// A failed fill must give the slot back, or the cache shrinks by one on every failure.
    #[test]
    fn abort_returns_a_reserved_slot_to_the_pool() {
        let mut c = ExpertCache::new(1, 8, 64, 3, 2).unwrap();
        let (s, _) = c.place(0, 1).unwrap();
        c.abort(s, 0, 1).unwrap();
        // The slot is reusable, and the expert is not resident.
        let (s2, miss) = c.place(0, 1).unwrap();
        assert!(miss, "an aborted expert reported as resident");
        assert_eq!(s2, s, "the aborted slot was not reused");
    }

    /// commit must refuse a slot nobody reserved, and refuse to publish under the wrong tag.
    #[test]
    fn commit_refuses_anything_it_did_not_reserve() {
        let mut c = ExpertCache::new(1, 8, 64, 4, 2).unwrap();
        assert!(c.commit(0, 0, 1).is_err(), "committed a slot with no matching place");
        let (s, _) = c.place(0, 1).unwrap();
        assert!(c.commit(s, 0, 7).is_err(), "published slot under an expert it was not reserved for");
        // ⛔ THIS LINE WAS `assert!(... .is_err() || true)` — vacuous, and written precisely because
        // I did not know the semantics. It hid a real bug: abort freed the slot on ANY tag while
        // only clearing the index on a match, stranding the reserving expert's cell pointing at an
        // Empty slot. When you are unsure what a call should do, that is the moment to decide, not
        // to write an assertion that cannot fail.
        assert!(c.abort(s, 0, 7).is_err(), "aborted a slot reserved for a different expert");
        // And the reservation must have survived both refusals intact.
        c.commit(s, 0, 1).unwrap();
        assert_eq!(c.place(0, 1).unwrap(), (s, false), "the reservation did not survive the refusals");
    }

    /// `place` must honour the protected working set exactly as `get` does, or a step can evict an
    /// expert it is still using — the failure `note_selected` exists to prevent.
    #[test]
    fn place_will_not_evict_this_steps_own_experts() {
        let mut c = ExpertCache::new(1, 8, 64, 3, 2).unwrap();
        c.note_selected(0, &[1, 2]);
        let (a, _) = c.place(0, 1).unwrap(); c.commit(a, 0, 1).unwrap();
        let (b, _) = c.place(0, 2).unwrap(); c.commit(b, 0, 2).unwrap();
        // Third expert, capacity 3: fits. Fourth would have to evict a protected one.
        let (d, _) = c.place(0, 3).unwrap(); c.commit(d, 0, 3).unwrap();
        c.note_selected(0, &[1, 2]);
        // Only slot `d` is unprotected, so it must be the victim.
        let (v, miss) = c.place(0, 4).unwrap();
        assert!(miss);
        assert_eq!(v, d, "evicted a protected expert instead of the unprotected one");
    }

    struct Synth { reads: RefCell<u64> }
    impl Synth {
        fn new() -> Self { Self { reads: RefCell::new(0) } }
    }
    impl Backing for Synth {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
            *self.reads.borrow_mut() += 1;
            for (i, b) in dst.iter_mut().enumerate() {
                *b = (offset.wrapping_add(i as u64).wrapping_mul(31).wrapping_add(7) & 0xff) as u8;
            }
            Ok(())
        }
    }
    fn off(layer: u32, expert: u32) -> u64 { (layer as u64 * 1024 + expert as u64) * 64 }

    #[test]
    fn capacity_below_one_step_working_set_is_refused() {
        let e = ExpertCache::new(4, 32, 64, 4, 6).unwrap_err();
        assert!(matches!(e, TierError::BudgetTooSmall { .. }), "got {e:?}");
        assert!(ExpertCache::new(4, 32, 64, 7, 6).is_ok(), "top_k + 1 must be accepted");
    }

    #[test]
    fn frequency_beats_recency_when_usage_is_skewed() {
        // A hot expert that has not been touched *most recently* must survive a parade of one-shot
        // experts. Pure LRU evicts it; that is the behaviour this policy exists to avoid.
        let b = Synth::new();
        let mut c = ExpertCache::new(1, 64, 64, 4, 2).unwrap();
        for _ in 0..10 {
            c.note_selected(0, &[0]);
            c.get(0, 0, off(0, 0), &b).unwrap(); // expert 0 is genuinely hot
        }
        for e in 10..20u32 {
            c.note_selected(0, &[e]);
            c.get(0, e, off(0, e), &b).unwrap(); // a parade of cold one-shots
        }
        c.note_selected(0, &[0]);
        let (_, tier) = c.get(0, 0, off(0, 0), &b).unwrap();
        assert_ne!(tier, Tier::Backing, "hot expert was evicted by cold one-shots — this is LRU behaviour");
    }

    #[test]
    fn protected_experts_are_never_evicted_mid_step() {
        // A cache that evicts an expert the current step is still multiplying returns garbage. Capacity is
        // exactly top_k + 1 here, so the policy has no slack to hide behind.
        let b = Synth::new();
        let mut c = ExpertCache::new(1, 64, 64, 5, 4).unwrap();
        let sel = [1u32, 2, 3, 4];
        c.note_selected(0, &sel);
        for &e in &sel { c.get(0, e, off(0, e), &b).unwrap(); }
        for &e in &sel {
            let (_, tier) = c.get(0, e, off(0, e), &b).unwrap();
            assert_ne!(tier, Tier::Backing, "expert {e} was evicted while still in the working set");
        }
    }

    #[test]
    fn bytes_are_identical_whatever_tier_serves_them() {
        let b = Synth::new();
        let mut tiny = ExpertCache::new(2, 16, 64, 3, 2).unwrap();
        let mut big = ExpertCache::new(2, 16, 64, 32, 2).unwrap();
        for layer in 0..2u32 {
            for e in 0..16u32 {
                tiny.note_selected(layer, &[e]);
                big.note_selected(layer, &[e]);
                let a = tiny.get(layer, e, off(layer, e), &b).unwrap().0.to_vec();
                let z = big.get(layer, e, off(layer, e), &b).unwrap().0.to_vec();
                assert_eq!(a, z, "expert ({layer},{e}) differs between budgets");
            }
        }
        assert!(tiny.stats().evictions > 0, "tiny cache should have evicted; test proves nothing otherwise");
        assert_eq!(big.stats().evictions, 0, "big cache should not have evicted");
    }

    #[test]
    fn pins_survive_decay_but_seeded_hotness_does_not() {
        // The honest distinction: a seeded popularity list is a warm start that decays; a pin is a
        // reservation. Systems that ship the former sometimes describe it as the latter.
        let b = Synth::new();
        let mut c = ExpertCache::new(1, 64, 64, 4, 2).unwrap();
        c.seed_hotness(&[(0, 7, 6000)]);
        c.note_selected(0, &[7]);
        c.get(0, 7, off(0, 7), &b).unwrap();
        c.pin(0, 7).unwrap();
        // 6000 halves to 0 in 13 decays; run well past that, with pressure.
        for t in 0..400u32 {
            let e = 20 + (t % 30);
            c.note_selected(0, &[e]);
            c.get(0, e, off(0, e), &b).unwrap();
            c.end_token();
        }
        c.note_selected(0, &[7]);
        let (_, tier) = c.get(0, 7, off(0, 7), &b).unwrap();
        assert_eq!(tier, Tier::Pinned, "an explicit pin must outlive hotness decay");
    }

    #[test]
    fn failed_read_does_not_publish_a_slot() {
        struct Boom;
        impl Backing for Boom {
            fn read_at(&self, _o: u64, dst: &mut [u8]) -> Result<(), TierError> {
                for b in dst.iter_mut() { *b = 0xAA; }
                Err(TierError::Io("disk fell over".into()))
            }
        }
        let good = Synth::new();
        let mut c = ExpertCache::new(1, 16, 64, 3, 2).unwrap();
        c.note_selected(0, &[1]);
        c.get(0, 1, off(0, 1), &good).unwrap();
        let want = c.get(0, 1, off(0, 1), &good).unwrap().0.to_vec();

        c.note_selected(0, &[2]);
        assert!(c.get(0, 2, off(0, 2), &Boom).is_err());
        // The failed expert must not be readable as a hit...
        c.note_selected(0, &[2]);
        let (_, tier) = c.get(0, 2, off(0, 2), &good).unwrap();
        assert_eq!(tier, Tier::Backing, "a failed read published a slot");
        // ...and the untouched expert must still be correct.
        c.note_selected(0, &[1]);
        assert_eq!(c.get(0, 1, off(0, 1), &good).unwrap().0, &want[..]);
    }
}
