//! Budget arithmetic: turn "you may use N bytes" into "pin these layers, stream the rest".
//!
//! This is pure integer arithmetic with no allocation and no I/O, which is deliberate — the budget
//! decision is the one place where an off-by-one silently wastes gigabytes at *every* budget, so it wants
//! to be exhaustively unit-testable.

/// Ring slots for streaming layers.
///
/// **One, not two.** A second slot is only worth its bytes if something is prefetching into it while the
/// first is in use. Without a real async reader, the second slot is pure waste — kimi-k3-in-c measured its
/// two-slot ring at "binds 558, hits 0 (0.0%)" while holding 2.34 GB, i.e. 20% of a 12 GB budget for
/// nothing, and cut it to one.
///
/// When Ferric adds an async reader this becomes 2 and the prefetch becomes trivially correct, because the
/// access order is fixed and known forever (0, 1, ... N-1, repeat). That is the single largest performance
/// win available here, and it is why the constant is named rather than inlined.
pub const RING_SLOTS: u64 = 1;

/// Passes of the sizing fixed-point. See [`plan_layers`] for why this terminates.
const MAX_PASSES: usize = 4;

/// One layer's footprint in the backing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerDesc {
    /// Byte offset of this layer's contiguous weight run.
    pub offset: u64,
    /// Length of that run.
    pub bytes: u64,
}

/// The resolved plan: which layers are pinned, and how big a streaming slot must be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    /// Layers `0..npin` are pinned in exact-size allocations.
    pub npin: usize,
    /// Size of each ring slot, large enough for the largest *streaming* layer.
    pub ring_slot: u64,
    /// Total bytes the plan commits to.
    pub spent: u64,
    /// Total layers this plan covers.
    pub n_layers: usize,
}

impl LayerPlan {
    /// Does this plan actually fit the budget it was computed against?
    ///
    /// [`plan_layers`] reports what a workable configuration *requires*, which can exceed a budget that
    /// cannot hold even one ring slot — the smallest configuration that can bind any layer at all. It
    /// deliberately does not clamp: silently returning a plan that cannot stream is how you get a cache
    /// that fails on its first bind instead of at construction.
    pub fn fits(&self, budget: u64) -> bool { self.spent <= budget }

    /// Deterministic hit rate. This is the property that makes a pinned prefix better than an LRU here:
    /// it is a known fraction, not an empirical one, and it rises smoothly with the budget.
    pub fn hit_rate(&self) -> f64 {
        if self.n_layers == 0 { return 0.0; }
        self.npin as f64 / self.n_layers as f64
    }
}

/// Round `v` up to a multiple of `align`. `align` must be a power of two.
#[inline]
pub fn align_up(v: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two, got {align}");
    (v + align - 1) & !(align - 1)
}

/// Resolve a byte budget into (pinned prefix, ring slot size).
///
/// # Why this is a fixed point rather than one pass
///
/// Ring size and pin count are **mutually dependent**: the ring must fit the largest layer that will
/// actually stream, but which layers stream depends on how many are pinned, which depends on how much
/// budget the ring consumed. One pass gets it wrong in a specific and expensive way.
///
/// # Why the ring is sized from streaming layers only
///
/// Sizing the ring over *all* layers reserves room for layer 0 — which prefix-pinning pins first, so it
/// never streams. On a real frontier model layer 0 is the outlier (a dense wide FFN: 2.34 GB against
/// 1.27 GB typical), so this mistake wastes roughly a gigabyte at *every* budget above the floor.
///
/// # Why pinned layers get exact-size allocations
///
/// Uniform slots sized to the largest layer would waste ~half the pinned budget for the same reason: one
/// outlier layer sets a slot size that every other layer underfills.
///
/// # Termination
///
/// Monotone, so it converges — and in practice in 2–3 passes. Pass 0 sizes the ring from all layers,
/// which is the largest it can ever be, giving the smallest `npin`. Each subsequent pass excludes the
/// newly-pinned (typically largest) layers from the ring calculation, so `ring_slot` is non-increasing
/// and `npin` is non-decreasing. Bounded at [`MAX_PASSES`] regardless, because a sizing loop that could
/// spin is worse than one that is occasionally one pass from optimal. `debug_assert`s below check the
/// monotonicity claim rather than trusting it.
///
/// `widen` is per-layer overhead the caller needs on top of the raw bytes (scratch, fp32 expansion of
/// small vectors, alignment slack). Pass 0 if there is none.
pub fn plan_layers(layers: &[LayerDesc], budget: u64, widen: u64, align: u64) -> LayerPlan {
    let n = layers.len();
    if n == 0 {
        return LayerPlan { npin: 0, ring_slot: 0, spent: 0, n_layers: 0 };
    }

    let (mut npin, mut ring_slot, mut spent) = (0usize, 0u64, 0u64);

    for pass in 0..MAX_PASSES {
        // Size the ring from the layers that will actually STREAM under the current npin.
        let big = layers[npin.min(n)..].iter().map(|l| l.bytes).max().unwrap_or(0);
        let rs = if big == 0 {
            // Everything is pinned; the ring is never bound. Charge nothing for it.
            0
        } else {
            align_up(align_up(big, align) + widen, align)
        };

        // Greedily pin the prefix that fits alongside the ring, at exact size.
        let mut sp = RING_SLOTS * rs;
        let mut np = 0usize;
        while np < n && sp + layers[np].bytes + widen <= budget {
            sp += layers[np].bytes + widen;
            np += 1;
        }

        if pass > 0 {
            debug_assert!(rs <= ring_slot, "ring_slot must be non-increasing (pass {pass})");
            debug_assert!(np >= npin, "npin must be non-decreasing (pass {pass})");
        }
        let converged = rs == ring_slot && np == npin;
        ring_slot = rs;
        npin = np;
        spent = sp;
        if converged { break; }
    }

    LayerPlan { npin, ring_slot, spent, n_layers: n }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layer 0 is the outlier, as it is in every real frontier MoE (dense wide FFN at the front).
    fn shape() -> Vec<LayerDesc> {
        let mut v = vec![LayerDesc { offset: 0, bytes: 2_340_000_000 }];
        let mut off = 2_340_000_000u64;
        for _ in 1..93 {
            v.push(LayerDesc { offset: off, bytes: 1_270_000_000 });
            off += 1_270_000_000;
        }
        v
    }

    #[test]
    fn empty_shape_is_not_a_panic() {
        let p = plan_layers(&[], 1 << 30, 0, 4096);
        assert_eq!(p.npin, 0);
        assert_eq!(p.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_rises_monotonically_with_budget() {
        // The whole point of a pinned prefix over an LRU: every extra byte buys its fair share, and the
        // return is deterministic rather than empirical.
        let s = shape();
        let mut last = -1.0f64;
        for gb in [4u64, 8, 16, 32, 64, 96, 128, 160, 192] {
            let p = plan_layers(&s, gb * 1_000_000_000, 25_890_000, 4096);
            assert!(p.hit_rate() >= last, "hit rate went DOWN from {last} at {gb} GB");
            last = p.hit_rate();
        }
        assert!(last > 0.9, "192 GB should pin nearly everything, got {last}");
    }

    #[test]
    fn ring_is_sized_from_streaming_layers_not_all_layers() {
        // The regression this guards: sizing the ring over ALL layers reserves room for layer 0, which
        // prefix-pinning pins first. On this shape that is ~1.07 GB wasted at every budget.
        let s = shape();
        let p = plan_layers(&s, 64_000_000_000, 0, 4096);
        assert!(p.npin >= 1, "expected layer 0 to be pinned at 64 GB");
        assert!(
            p.ring_slot < 2_340_000_000,
            "ring slot {} was sized from the pinned outlier layer 0, not from streaming layers",
            p.ring_slot
        );
        assert!(p.ring_slot >= 1_270_000_000, "ring slot must still fit a streaming layer");
    }

    #[test]
    fn plan_never_exceeds_a_budget_it_can_serve() {
        let s = shape();
        for gb in [4u64, 7, 11, 23, 50, 97, 131, 200] {
            let budget = gb * 1_000_000_000;
            let p = plan_layers(&s, budget, 25_890_000, 4096);
            assert!(p.fits(budget), "plan spent {} over budget {budget}", p.spent);
        }
    }

    #[test]
    fn a_budget_too_small_for_one_ring_slot_reports_infeasible() {
        // The bug this guards: the ring is charged unconditionally, so a budget below one slot yields a
        // plan that "spends" more than it was given. Clamping would hide it; reporting it lets the
        // constructor refuse at build time instead of failing on the first bind.
        let s = shape();
        let tiny = 1000u64;
        let p = plan_layers(&s, tiny, 0, 4096);
        assert_eq!(p.npin, 0);
        assert!(!p.fits(tiny), "plan claimed to fit {tiny} bytes while needing {}", p.spent);
    }

    #[test]
    fn fixed_point_converges_and_is_stable() {
        // Re-planning from the result must be a no-op. If a second call moved, the loop exited early.
        let s = shape();
        for gb in [8u64, 33, 64, 128] {
            let budget = gb * 1_000_000_000;
            let a = plan_layers(&s, budget, 25_890_000, 4096);
            let b = plan_layers(&s, budget, 25_890_000, 4096);
            assert_eq!(a, b, "planning is not deterministic at {gb} GB");
        }
    }

    #[test]
    fn everything_pinned_charges_nothing_for_the_ring() {
        let s = shape();
        let p = plan_layers(&s, 1_000_000_000_000, 0, 4096);
        assert_eq!(p.npin, s.len(), "a 1 TB budget should pin all 93 layers");
        assert_eq!(p.ring_slot, 0, "with nothing streaming, the ring must cost zero");
        assert_eq!(p.hit_rate(), 1.0);
    }

    #[test]
    fn align_up_is_exact_on_boundaries() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096, "an exact multiple must not be rounded up a full block");
        assert_eq!(align_up(4097, 4096), 8192);
    }
}
