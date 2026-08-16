//! **When does recursing over a document beat pasting it?** — in joules.
//!
//! Recursive Language Models (Zhang et al., MIT OAS; `github.com/alexzhang13/rlm`, MIT, 5.5k stars)
//! swap `llm.completion` for `rlm.completion`: the context is not pasted into the prompt but bound to
//! a variable in a live Python session, and the model writes code to survey it — `llm_query` for a
//! flat sub-call, `rlm_query` to recurse, bounded by `max_concurrent_subcalls`.
//!
//! The paradigm's own honest boundary is stated qualitatively everywhere it is discussed:
//! *recursion multiplies latency by construction, and earns its cost only when the input is too big
//! to hold sharply or forks naturally into slices — if your document fits comfortably, paste it.*
//!
//! **That is a break-even, and nobody computes it.** This module does, in joules, because the shape of
//! the answer is not intuitive:
//!
//! ## Why splitting wins, and by exactly how much
//!
//! Prefill is **quadratic** in context: `E_prefill(m) = a·m + b·m²/2` ([`crate::compaction::StepCost`],
//! whose coefficients are measured, not assumed). Split `n` tokens into `k` slices and each costs
//! `a·(n/k) + b·(n/k)²/2`, so `k` of them cost
//!
//! ```text
//!     a·n + b·n²/(2k)
//! ```
//!
//! The **linear term is unchanged** — every token is still read once, and no amount of decomposition
//! avoids that. The **quadratic term falls by exactly `k`**. So recursion buys down attention, and
//! only attention:
//!
//! ```text
//!     saving = (b·n²/2)·(1 − 1/k)
//! ```
//!
//! Against that it pays orchestration: reconnaissance, per-slice generation, and a synthesis pass over
//! the collected answers. Those are real and mostly *linear*, which is why the trade flips with `n`
//! rather than being universally good or bad. Small documents: orchestration dominates, paste. Large
//! documents: the quadratic term dominates, recurse. The crossover is computable and is
//! [`Plan::break_even_tokens`].
//!
//! ## Context rot is the other half of the argument
//!
//! Chroma's *Context Rot* study (`github.com/chroma-core/context-rot`) documents that model
//! performance degrades as input grows **even on trivial tasks** — models do not process context
//! uniformly. Put beside the arithmetic above, long context is paying a *quadratically growing* energy
//! bill for *falling* quality past some length. This module prices only the energy half; the quality
//! half is why the break-even is a floor on the case for recursion rather than the whole of it.

use crate::compaction::StepCost;

/// One recursive decomposition, priced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Total context tokens the task must cover.
    pub n: usize,
    /// Number of slices the root fans out into. `k = 1` is the paste baseline.
    pub k: usize,
    /// Tokens the root reads to decide how to slice (a listing, a size, a sample). Small by design —
    /// the root surveys rather than reads.
    pub recon: usize,
    /// Tokens each sub-call generates back to the root.
    pub per_slice_out: usize,
    /// Tokens the final answer generates.
    pub answer_out: usize,
}

/// What a decomposition costs against pasting the same context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    pub paste: f64,
    pub recurse: f64,
    /// Joules saved by recursing. **Negative means paste**, and it is returned unclamped.
    pub saved: f64,
}

impl Plan {
    pub fn worth_it(&self) -> bool { self.saved > 0.0 }
    /// Saving as a fraction of the paste cost. `NaN` when pasting is free.
    pub fn fraction(&self) -> f64 {
        if self.paste <= 0.0 { f64::NAN } else { self.saved / self.paste }
    }
}

/// Price recursion against pasting, on measured per-token energy.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    pub cost: StepCost,
}

impl Model {
    pub fn new(cost: StepCost) -> Model { Model { cost } }

    /// Cost of pasting: read all `n`, then generate the answer at full context.
    pub fn paste(&self, s: &Shape) -> f64 {
        self.cost.prefill(s.n) + self.cost.step(s.n) * s.answer_out as f64
    }

    /// Cost of recursing: reconnaissance, `k` slices in parallel, then synthesis over their answers.
    ///
    /// Parallel sub-calls do not reduce **energy** — only latency. Every slice's joules are still
    /// spent, which is exactly the accounting error a wall-clock view of this paradigm invites, and is
    /// the same failure [`crate::ladder`] exists to prevent for routing.
    pub fn recurse(&self, s: &Shape) -> f64 {
        if s.k <= 1 { return self.paste(s); }
        let slice = s.n / s.k;
        // The root surveys: reads a little, and emits the code that drives the fan-out.
        let recon = self.cost.prefill(s.recon) + self.cost.step(s.recon) * s.recon as f64;
        // Each child prefills its own slice and generates a summary of it.
        let children = s.k as f64
            * (self.cost.prefill(slice) + self.cost.step(slice) * s.per_slice_out as f64);
        // Synthesis reads the k answers and writes the final one.
        let collected = s.k * s.per_slice_out;
        let synth = self.cost.prefill(collected) + self.cost.step(collected) * s.answer_out as f64;
        recon + children + synth
    }

    pub fn plan(&self, s: &Shape) -> Plan {
        let (p, r) = (self.paste(s), self.recurse(s));
        Plan { paste: p, recurse: r, saved: p - r }
    }

    /// The context length at which recursion starts to pay, holding the rest of the shape fixed.
    ///
    /// Returns `None` when recursion never wins at any length — a real outcome when `k` is small or
    /// orchestration is heavy, and one worth reporting rather than searching past.
    pub fn break_even_tokens(&self, s: &Shape, max_n: usize) -> Option<usize> {
        if s.k <= 1 { return None; }
        // Monotone in n once the quadratic term dominates, so a binary search is exact enough and
        // cannot loop; the guard below rejects the case where it never crosses.
        let mut probe = *s;
        probe.n = max_n;
        if !self.plan(&probe).worth_it() { return None; }
        let (mut lo, mut hi) = (1usize, max_n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            probe.n = mid;
            if self.plan(&probe).worth_it() { hi = mid } else { lo = mid + 1 }
        }
        Some(lo)
    }

    /// The `k` that minimises joules for this context, searched over `2..=max_k`.
    ///
    /// Not unbounded: every extra slice adds a fixed orchestration cost, so the curve has a floor and
    /// then rises. Reporting the argmin rather than "more slices is better" is the point.
    pub fn best_k(&self, s: &Shape, max_k: usize) -> (usize, f64) {
        let mut best = (1usize, self.paste(s));
        for k in 2..=max_k {
            let mut probe = *s;
            probe.k = k;
            let e = self.recurse(&probe);
            if e < best.1 { best = (k, e); }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, Class, Reading};

    fn reading(j: f64) -> Reading {
        Reading { joules: j, seconds: 1.0, class: Class::Measured, source: "test", boundary: Boundary::DEVICE }
    }
    /// a = 1.0 J/token, b = 0.001 J/token-of-context.
    fn model() -> Model {
        Model::new(StepCost::fit((1_000, &reading(2.0)), (10_000, &reading(11.0))).unwrap())
    }
    fn shape(n: usize, k: usize) -> Shape {
        Shape { n, k, recon: 200, per_slice_out: 300, answer_out: 500 }
    }

    #[test]
    fn a_small_document_should_be_pasted() {
        // The paradigm's own advice, now with a number behind it: nobody hands a librarian one page.
        let m = model();
        let p = m.plan(&shape(2_000, 8));
        assert!(!p.worth_it(), "recursing a 2k document saved {} J", p.saved);
    }

    #[test]
    fn a_large_document_should_be_recursed() {
        let m = model();
        let p = m.plan(&shape(400_000, 8));
        assert!(p.worth_it());
        assert!(p.fraction() > 0.5, "expected a large win, got {:.1}%", 100.0 * p.fraction());
    }

    #[test]
    fn splitting_divides_the_quadratic_term_by_k_and_leaves_the_linear_one_alone() {
        // THE mechanism. Every token is still read once — decomposition cannot avoid that — so only
        // attention is bought down. A model that claimed to reduce the linear term would be claiming
        // not to read the input.
        let m = model();
        let (n, k) = (100_000usize, 10usize);
        let whole = m.cost.prefill(n);
        let split = k as f64 * m.cost.prefill(n / k);
        let (a, b) = (m.cost.a, m.cost.b);
        let linear = a * n as f64;
        assert!((whole - (linear + b * (n as f64).powi(2) / 2.0)).abs() < 1e-3);
        assert!((split - (linear + b * (n as f64).powi(2) / (2.0 * k as f64))).abs() < 1e-3,
                "split prefill should be linear + quadratic/k");
        assert!(split < whole);
    }

    #[test]
    fn there_is_a_crossover_and_it_is_findable() {
        let m = model();
        let s = shape(0, 8);
        let n = m.break_even_tokens(&s, 1_000_000).expect("should cross");
        // Straddle it: just below must lose, just above must win.
        let mut lo = s; lo.n = n.saturating_sub(1);
        let mut hi = s; hi.n = n + 1;
        assert!(!m.plan(&lo).worth_it(), "at {} recursion should still lose", lo.n);
        assert!(m.plan(&hi).worth_it(), "at {} recursion should win", hi.n);
    }

    #[test]
    fn parallel_sub_calls_do_not_reduce_energy() {
        // Latency is not energy. Eight slices run at once still burn eight slices' joules, and a
        // wall-clock view of this paradigm hides that — the same error `ladder` exists to prevent.
        let m = model();
        let s = shape(200_000, 8);
        let slice = s.n / s.k;
        let children = s.k as f64 * (m.cost.prefill(slice) + m.cost.step(slice) * s.per_slice_out as f64);
        assert!(m.recurse(&s) > children, "the total must include every child, not the slowest one");
    }

    #[test]
    fn more_slices_is_not_monotonically_better() {
        // Each slice adds fixed orchestration, so the curve bottoms out and climbs. Treating "more
        // recursion" as strictly better is how a paradigm gets applied past the point it helps.
        let m = model();
        let s = shape(50_000, 2);
        let (k, e) = m.best_k(&s, 512);
        assert!(k > 1, "some decomposition should beat pasting at 50k");
        let mut way_past = s; way_past.k = 512;
        assert!(m.recurse(&way_past) > e, "512 slices should be worse than the optimum at k={k}");
    }

    #[test]
    fn k_of_one_is_exactly_the_paste_baseline() {
        // The no-op arm must be the baseline itself, not an approximation of it, or every saving is
        // measured against a slightly different thing than the alternative actually is.
        let m = model();
        let s = shape(30_000, 1);
        assert_eq!(m.recurse(&s), m.paste(&s));
        assert_eq!(m.plan(&s).saved, 0.0);
    }

    #[test]
    fn a_decomposition_that_never_wins_reports_none_rather_than_a_number() {
        // Heavy orchestration against a 2-way split: there is no length at which this pays, and
        // returning some large n would invite deploying it there.
        let m = model();
        let s = Shape { n: 0, k: 2, recon: 100_000, per_slice_out: 50_000, answer_out: 1_000 };
        assert_eq!(m.break_even_tokens(&s, 100_000), None);
    }
}
