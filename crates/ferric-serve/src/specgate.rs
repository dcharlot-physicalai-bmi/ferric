//! **Should this request speculate at all?** — the energy break-even, learned rather than assumed.
//!
//! `generate` speculates whenever the model ships an MTP draft block. That is the right default and
//! the wrong invariant: self-drafting is not free. Each step pays the draft forward AND the main
//! forward, and only yields more than one token when the draft is accepted. So there is an acceptance
//! rate below which speculating spends MORE joules per token than not speculating, and nothing in the
//! serving path was checking which side of it a given request sat on.
//!
//! ## The break-even, derived
//!
//! Let `E_m` be the main forward's energy, `E_d` the draft's, and `a` the acceptance rate.
//!
//! ```text
//!   plain:  1 token      per  E_m
//!   spec:   (1 + a) tokens per (E_m + E_d)
//!
//!   spec wins  <=>  (1 + a) / (E_m + E_d)  >  1 / E_m
//!              <=>  a * E_m  >  E_d
//!              <=>  a  >  E_d / E_m
//! ```
//!
//! That is the same shape as [`ferric_joule::Profile`]'s rule for trying a rung at all — `p_k >
//! E_k / E(k+1)` — which is why the router's machinery applies here rather than a bespoke heuristic.
//!
//! ## Where the ratio comes from, and what it is NOT
//!
//! `E_d / E_m` is estimated **structurally**: the MTP draft is one transformer block plus the output
//! head, against the main model's `n_layer` blocks plus the same head. This is a shape argument, not a
//! measurement — no RAPL, no powermetrics, no mains meter. It is used because it is available on every
//! machine and because being roughly right about the ratio matters far less than being roughly right
//! about the acceptance rate, which IS measured. `docs/sota-feature-matrix.md` records the distinction.
//!
//! ## Cold start
//!
//! With no observations the estimator sits at its prior and the gate speculates — today's behaviour,
//! unchanged. It only ever declines once it has watched drafts actually get rejected for requests like
//! this one.
use ferric_joule::{prompt_len_bucket, OnlineRate};

/// Decides whether to speculate, and learns from what happened.
pub(crate) struct SpecGate {
    rate: OnlineRate<fn(&usize) -> u32>,
    /// `E_draft / E_main`, the acceptance rate above which speculating is worth its energy.
    break_even: f64,
}

impl SpecGate {
    /// `n_layer` is the main model's block count; the draft is one block.
    pub(crate) fn new(n_layer: usize) -> SpecGate {
        // One draft block against `n_layer` main blocks. Both pay the output head, so the head is
        // omitted from both sides rather than guessed at — including it would shrink the ratio and
        // make the gate MORE eager, which is the direction that costs energy when wrong.
        let break_even = 1.0 / n_layer.max(1) as f64;
        // `prompt_len_bucket` takes usize by value; the predictor buckets by reference, so this
        // adapts rather than casts (a cast between those two fn types is not a valid coercion).
        fn by_ref(n: &usize) -> u32 { prompt_len_bucket(*n) }
        SpecGate { rate: OnlineRate::new(by_ref as fn(&usize) -> u32), break_even }
    }

    /// The acceptance rate above which speculation pays for itself.
    pub(crate) fn break_even(&self) -> f64 { self.break_even }

    /// Whether to speculate for a prompt of this length.
    pub(crate) fn should_speculate(&self, prompt_len: usize) -> bool {
        self.expected_acceptance(prompt_len) > self.break_even
    }

    /// The current estimate — exposed so a caller can log WHY, not just what.
    pub(crate) fn expected_acceptance(&self, prompt_len: usize) -> f64 {
        // Rung 0 is "the draft was accepted"; the profile shape the router wants is two-rung, and only
        // rung 0 is ever observed here because rung 1 (the main forward) cannot fail.
        self.rate.p_success_at(&prompt_len, 0)
    }

    /// Record one draft outcome. Call once per draft ATTEMPTED, never for a step that did not draft —
    /// a skipped draft has no outcome, and inventing one lets the gate confirm its own decisions.
    pub(crate) fn observe(&mut self, prompt_len: usize, accepted: bool) {
        self.rate.observe(&prompt_len, 0, accepted);
    }
}
