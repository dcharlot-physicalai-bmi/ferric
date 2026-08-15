//! **When is compacting a conversation worth the energy it costs?**
//!
//! Agent harnesses compact long sessions: summarize an older span into one node and drop the rest.
//! DeepSeek Harness (MIT, 2026-08-13) makes this a first-class capability seam, and its default
//! backend triggers at `thresholdRatio = 0.8` of the routed context window, retaining
//! `retainRatio = 0.16`. Claude Code, Codex and the rest use the same shape.
//!
//! That is a **capacity** trigger. It answers *will this fit* — and it has to, because a harness sees
//! an opaque provider across a network and prices context with a deliberate heuristic (Harness'
//! `tokenMeter`: "four characters per token plus structural overhead", explicitly settings-free).
//!
//! It does not answer *is this worth it*, and that is a different question with a different answer.
//! Compaction is not free:
//!
//! ```text
//!   pay now:    one summarization call, plus a re-prefill — the surface changed, so the KV prefix
//!               after the replacement point is invalidated
//!   save later: every subsequent step attends over a shorter cache
//! ```
//!
//! So compaction pays for itself only after enough further steps. Compact a session that ends two
//! turns later and you burned energy to save nothing. **A runtime can compute that break-even; a
//! harness cannot**, because the terms are prefill and attention energy, which only the thing running
//! the model can measure. That is the whole of this module.
//!
//! ## The cost model
//!
//! Per decode step at context length `n`, energy splits into a part that does not depend on context
//! and a part that does:
//!
//! ```text
//!   E_step(n) = a + b·n
//! ```
//!
//! `a` is weights, projections and FFN — paid per token regardless of history. `b` is attention over
//! `n` cached keys. Prefilling `m` tokens pays `a` per token and attention against everything before
//! it, so `E_prefill(m) = a·m + b·m²/2`.
//!
//! Both `a` and `b` are **measured on the deployment**, never assumed — see [`StepCost::fit`], which
//! recovers them from two readings at different context lengths. A model fitted from guesses would
//! optimise a model of the machine rather than the machine.
//!
//! ## Two regimes, and saying which one you are in
//!
//! ```text
//!   capacity:  n + next_turn > context_window   ->  compact, cost is irrelevant, there is no choice
//!   cost:      otherwise                        ->  compact only if expected remaining steps > k*
//! ```
//!
//! Conflating them is how a policy ends up looking principled while being neither. [`Decision`] names
//! the regime it is in.

use crate::Reading;

/// Measured per-step energy as a function of context length: `E(n) = a + b·n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCost {
    /// Joules per token independent of context: weights, projections, FFN.
    pub a: f64,
    /// Joules per token of cached context: the attention term.
    pub b: f64,
    /// Prefill energy per token relative to decode. See [`StepCost::with_prefill_ratio`]; 1.0 is the
    /// conservative default and overstates prefill.
    pub prefill_ratio: f64,
}

/// Why a cost model could not be built. Each is a case where proceeding would produce a confident
/// number with nothing behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// Both samples were taken at the same context length, so the two terms cannot be separated.
    Degenerate,
    /// A sample was non-finite or non-positive.
    BadSample,
    /// The fit implies attention gets *cheaper* with more context, which means the samples are noise
    /// rather than signal.
    NegativeAttention,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::Degenerate => write!(f, "both samples at the same context length: a and b are not separable"),
            FitError::BadSample => write!(f, "a sample was non-finite or non-positive"),
            FitError::NegativeAttention => write!(f, "fit implies negative attention cost — the samples are noise, not signal"),
        }
    }
}

impl std::error::Error for FitError {}

impl StepCost {
    /// Recover `a` and `b` from two measured decode steps at different context lengths.
    ///
    /// Two points, two unknowns. Deliberately not a least-squares fit over many points: the failure
    /// this guards against is a *fabricated* model, and two honest readings beat a smooth curve
    /// through numbers nobody took.
    pub fn fit(short: (usize, &Reading), long: (usize, &Reading)) -> Result<StepCost, FitError> {
        let (n1, r1) = (short.0 as f64, short.1.joules);
        let (n2, r2) = (long.0 as f64, long.1.joules);
        if !r1.is_finite() || !r2.is_finite() || r1 <= 0.0 || r2 <= 0.0 { return Err(FitError::BadSample); }
        if (n2 - n1).abs() < f64::EPSILON { return Err(FitError::Degenerate); }
        let b = (r2 - r1) / (n2 - n1);
        if b < 0.0 { return Err(FitError::NegativeAttention); }
        let a = r1 - b * n1;
        if a < 0.0 { return Err(FitError::BadSample); }
        Ok(StepCost { a, b, prefill_ratio: 1.0 })
    }

    /// Energy of one decode step at context length `n`.
    pub fn step(&self, n: usize) -> f64 { self.a + self.b * n as f64 }

    /// Energy of prefilling `m` tokens from empty: `r·(a·m + b·m²/2)`, where `r` is
    /// [`StepCost::prefill_ratio`].
    ///
    /// The quadratic term is why long prompts are expensive and why re-prefilling after a compaction
    /// is not a rounding error.
    pub fn prefill(&self, m: usize) -> f64 {
        let m = m as f64;
        self.prefill_ratio * (self.a * m + self.b * m * m / 2.0)
    }

    /// Set the measured prefill-to-decode energy ratio per token.
    ///
    /// **Prefill and decode are not the same cost per token.** Decode is memory-bound — one token
    /// re-reads the whole weight set — while prefill is compute-bound and batches many tokens against
    /// one weight read, so prefill costs substantially *less* energy per token on real hardware.
    ///
    /// [`StepCost::fit`] measures **decode**, because that is what a step is. Reusing those
    /// coefficients for prefill unchanged (ratio 1.0, the default) therefore **overstates** the
    /// re-prefill term. That error is one-sided and its direction is worth stating plainly: it makes
    /// this policy *conservative*, compacting less often than optimal, never more. Measure the ratio
    /// on the deployment to remove it.
    pub fn with_prefill_ratio(self, r: f64) -> StepCost {
        assert!(r.is_finite() && r > 0.0, "prefill ratio must be finite and positive, got {r}");
        StepCost { prefill_ratio: r, ..self }
    }
}

/// What compacting this session would cost and save.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Context length now.
    pub n: usize,
    /// Context length after compaction: summary + retained tail.
    pub retained: usize,
    /// Joules to perform the compaction.
    pub cost: f64,
    /// Joules saved per subsequent decode step.
    pub per_step_saving: f64,
}

impl Plan {
    /// Steps that must still be taken for compaction to break even.
    ///
    /// `INFINITY` when compaction saves nothing per step — which happens when the "compacted" context
    /// is not actually shorter, and is a real configuration rather than a hypothetical.
    pub fn break_even_steps(&self) -> f64 {
        if self.per_step_saving <= 0.0 { f64::INFINITY } else { self.cost / self.per_step_saving }
    }

    /// Net joules if the session runs `steps` more decode steps after compacting. Negative means
    /// compacting lost.
    pub fn net_at(&self, steps: usize) -> f64 {
        self.per_step_saving * steps as f64 - self.cost
    }
}

/// The regime a compaction decision is in, and what it decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    /// The next turn would not fit. Compaction is forced and the cost is not a consideration.
    Capacity { plan: Plan, overflow_by: usize },
    /// Worth it: the session is expected to run past the break-even.
    Worth { plan: Plan, break_even: f64, expected: f64 },
    /// Not worth it: compacting now would spend more than it recovers.
    Wasteful { plan: Plan, break_even: f64, expected: f64 },
}

impl Decision {
    pub fn should_compact(&self) -> bool {
        matches!(self, Decision::Capacity { .. } | Decision::Worth { .. })
    }
    pub fn plan(&self) -> &Plan {
        match self { Decision::Capacity { plan, .. } | Decision::Worth { plan, .. } | Decision::Wasteful { plan, .. } => plan }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Decision::Capacity { .. } => "capacity",
            Decision::Worth { .. } => "worth",
            Decision::Wasteful { .. } => "wasteful",
        }
    }
}

/// A compaction policy that decides on measured joules rather than a context-window fraction.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub cost: StepCost,
    /// Hard capacity of the route. Capacity always wins over cost.
    pub context_window: usize,
    /// Tokens the next turn is expected to add before another decision point. Used only for the
    /// capacity test.
    pub headroom: usize,
}

impl Policy {
    pub fn new(cost: StepCost, context_window: usize, headroom: usize) -> Policy {
        Policy { cost, context_window, headroom }
    }

    /// Price one candidate compaction.
    ///
    /// `summary_tokens` is what the summary will occupy, `retained_tail` the recent span kept
    /// verbatim, `summary_output` the tokens the summarization call generates (it is a generation, so
    /// it is paid at the *current* context length, which is what makes summarizing a long session
    /// expensive).
    pub fn plan(&self, n: usize, summary_tokens: usize, retained_tail: usize, summary_output: usize) -> Plan {
        let retained = summary_tokens + retained_tail;
        // Generating the summary happens against the full current context. The span being summarized
        // is already in the cache, so it is not re-prefilled to read it — matching how harnesses
        // replay the prefix to reuse the provider's KV cache.
        let summarize = self.cost.step(n) * summary_output as f64;
        // The surface changed, so everything from the replacement point on must be prefilled again.
        let reprefill = self.cost.prefill(retained);
        Plan {
            n,
            retained,
            cost: summarize + reprefill,
            // Per later step we attend over `retained` instead of `n`.
            per_step_saving: self.cost.step(n) - self.cost.step(retained),
        }
    }

    /// Decide, naming the regime.
    ///
    /// `expected_steps` is how many more decode steps the session is expected to take. It is a
    /// forecast and is treated as one: it only selects between [`Decision::Worth`] and
    /// [`Decision::Wasteful`], and never overrides capacity.
    pub fn decide(&self, n: usize, summary_tokens: usize, retained_tail: usize,
                  summary_output: usize, expected_steps: f64) -> Decision {
        let plan = self.plan(n, summary_tokens, retained_tail, summary_output);
        if n + self.headroom > self.context_window {
            return Decision::Capacity { plan, overflow_by: (n + self.headroom) - self.context_window };
        }
        let break_even = plan.break_even_steps();
        if expected_steps > break_even {
            Decision::Worth { plan, break_even, expected: expected_steps }
        } else {
            Decision::Wasteful { plan, break_even, expected: expected_steps }
        }
    }

    /// What a fixed-ratio policy would do, for comparison. The industry default is 0.8.
    ///
    /// Provided so a deployment can measure its own trigger against this one on the same session
    /// rather than take the claim on faith.
    pub fn ratio_trigger(&self, n: usize, threshold_ratio: f64) -> bool {
        n as f64 >= self.context_window as f64 * threshold_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, Class};

    fn reading(j: f64) -> Reading {
        Reading { joules: j, seconds: 1.0, class: Class::Measured, source: "test", boundary: Boundary::DEVICE }
    }

    /// a = 1.0 J/token, b = 0.001 J/token-of-context.
    fn cost() -> StepCost {
        StepCost::fit((1_000, &reading(2.0)), (10_000, &reading(11.0))).unwrap()
    }

    #[test]
    fn the_cost_model_is_fitted_from_readings_not_assumed() {
        let c = cost();
        assert!((c.a - 1.0).abs() < 1e-9, "a = {}", c.a);
        assert!((c.b - 0.001).abs() < 1e-12, "b = {}", c.b);
        // And it reproduces the samples it was fitted from.
        assert!((c.step(1_000) - 2.0).abs() < 1e-9);
        assert!((c.step(10_000) - 11.0).abs() < 1e-9);
    }

    #[test]
    fn two_samples_at_one_length_cannot_separate_the_terms() {
        // A degenerate fit would silently attribute everything to `a`, making attention free and
        // compaction always look worthless.
        assert_eq!(StepCost::fit((5_000, &reading(6.0)), (5_000, &reading(6.0))).unwrap_err(), FitError::Degenerate);
    }

    #[test]
    fn a_fit_implying_cheaper_attention_with_more_context_is_refused() {
        // Physically impossible, so it is measurement noise. Accepting it yields a negative `b`, which
        // makes long contexts look CHEAPER and inverts every subsequent decision.
        assert_eq!(StepCost::fit((1_000, &reading(11.0)), (10_000, &reading(2.0))).unwrap_err(),
                   FitError::NegativeAttention);
    }

    #[test]
    fn compacting_a_session_that_is_about_to_end_loses() {
        // THE case the industry's 0.8-of-window trigger gets wrong. Pressure says compact; the session
        // ends three steps later and the compaction never earns its cost back.
        let p = Policy::new(cost(), 128_000, 2_000);
        // 0.8 x 128k = 102.4k, so the session has to be ABOVE that for the ratio trigger to fire.
        let n = 105_000;
        let d = p.decide(n, 1_000, 20_000, 500, 3.0);
        assert_eq!(d.label(), "wasteful");
        assert!(!d.should_compact());
        assert!(d.plan().net_at(3) < 0.0, "net {} should be a loss", d.plan().net_at(3));
        // ...and the ratio trigger WOULD have fired, which is the point of the comparison.
        assert!(p.ratio_trigger(n, 0.8), "this test needs the industry default to fire");
    }

    #[test]
    fn compacting_a_session_that_keeps_going_wins() {
        let p = Policy::new(cost(), 128_000, 2_000);
        let d = p.decide(100_000, 1_000, 20_000, 500, 5_000.0);
        assert_eq!(d.label(), "worth");
        assert!(d.should_compact());
        assert!(d.plan().net_at(5_000) > 0.0);
    }

    #[test]
    fn the_break_even_is_where_the_two_verdicts_meet() {
        // The decision must be continuous in expected_steps: no gap, no overlap.
        let p = Policy::new(cost(), 128_000, 2_000);
        let k = p.plan(100_000, 1_000, 20_000, 500).break_even_steps();
        assert!(k.is_finite() && k > 0.0, "break-even {k}");
        assert_eq!(p.decide(100_000, 1_000, 20_000, 500, k * 1.01).label(), "worth");
        assert_eq!(p.decide(100_000, 1_000, 20_000, 500, k * 0.99).label(), "wasteful");
        // Exactly at the break-even the net is zero, so it is not "worth" it.
        assert!(p.plan(100_000, 1_000, 20_000, 500).net_at(k as usize).abs() < p.plan(100_000, 1_000, 20_000, 500).per_step_saving);
    }

    #[test]
    fn capacity_overrides_cost_and_says_so() {
        // When the next turn does not fit there is no decision to make, and a policy that reported
        // "wasteful" here would be correct about the joules and useless about the situation.
        let p = Policy::new(cost(), 128_000, 2_000);
        let d = p.decide(127_000, 1_000, 20_000, 500, 1.0);
        assert_eq!(d.label(), "capacity");
        assert!(d.should_compact(), "an overflow must compact regardless of cost");
        match d {
            Decision::Capacity { overflow_by, .. } => assert_eq!(overflow_by, 1_000),
            _ => panic!("expected the capacity regime"),
        }
    }

    #[test]
    fn a_compaction_that_does_not_shrink_never_breaks_even() {
        // `retained >= n` is a real misconfiguration (a generous retain ratio on a short session).
        // It must report INFINITY rather than a small positive number that looks actionable.
        let p = Policy::new(cost(), 128_000, 2_000);
        let plan = p.plan(10_000, 4_000, 8_000, 500);
        assert!(plan.retained > plan.n, "test setup: retention must exceed the current context");
        assert!(plan.per_step_saving <= 0.0);
        assert_eq!(plan.break_even_steps(), f64::INFINITY);
        assert!(plan.net_at(1_000_000) < 0.0, "it can never win, at any horizon");
    }

    #[test]
    fn re_prefill_is_charged_because_the_prefix_is_invalidated() {
        // Compaction replaces a span, so the cached prefix after that point is gone. A model that
        // counted only the summarization call would understate the cost and compact far too eagerly.
        let p = Policy::new(cost(), 128_000, 2_000);
        let plan = p.plan(100_000, 1_000, 20_000, 500);
        let summarize_only = p.cost.step(100_000) * 500.0;
        assert!(plan.cost > summarize_only, "re-prefill was not charged");
        assert!((plan.cost - summarize_only - p.cost.prefill(21_000)).abs() < 1e-6);
    }

    #[test]
    fn treating_prefill_as_expensive_as_decode_only_ever_compacts_less() {
        // The default ratio of 1.0 reuses DECODE coefficients for PREFILL, which overstates prefill
        // because prefill batches many tokens against one weight read while decode re-reads the
        // weights per token. Pin the DIRECTION of that error: measuring the real (cheaper) ratio can
        // only lower the cost and hence the break-even, never raise it. A policy whose known error
        // pushed the other way would compact too eagerly, which is the expensive mistake.
        let conservative = Policy::new(cost(), 128_000, 2_000);
        let measured = Policy::new(cost().with_prefill_ratio(0.25), 128_000, 2_000);

        let k_cons = conservative.plan(100_000, 1_000, 20_000, 500).break_even_steps();
        let k_meas = measured.plan(100_000, 1_000, 20_000, 500).break_even_steps();
        assert!(k_meas < k_cons, "measured break-even {k_meas} should be below conservative {k_cons}");

        // The per-step saving is a decode quantity and must NOT move with the prefill ratio.
        assert_eq!(conservative.plan(100_000, 1_000, 20_000, 500).per_step_saving,
                   measured.plan(100_000, 1_000, 20_000, 500).per_step_saving);

        // And there is a real session in the gap: one the conservative policy declines and the
        // measured one takes. That is what "conservative" costs in practice.
        let between = (k_meas + k_cons) / 2.0;
        assert!(!conservative.decide(100_000, 1_000, 20_000, 500, between).should_compact());
        assert!(measured.decide(100_000, 1_000, 20_000, 500, between).should_compact());
    }

    #[test]
    fn the_ratio_trigger_and_the_cost_trigger_genuinely_disagree() {
        // If they always agreed this module would be decoration. Exhibit one session where the
        // industry default fires and the energy answer is "do not", and one of the reverse.
        let p = Policy::new(cost(), 128_000, 2_000);

        // Ratio fires, cost says no: near the window but about to finish.
        assert!(p.ratio_trigger(105_000, 0.8));
        assert!(!p.decide(105_000, 1_000, 20_000, 500, 2.0).should_compact());

        // Ratio does not fire, cost says yes: only 40% of the window, but thousands of steps to go,
        // so the attention term dominates and shortening it early pays.
        assert!(!p.ratio_trigger(50_000, 0.8));
        let d = p.decide(50_000, 1_000, 10_000, 500, 100_000.0);
        assert_eq!(d.label(), "worth");
        assert!(d.should_compact(), "a long-running session should compact well before 80%");
    }
}
