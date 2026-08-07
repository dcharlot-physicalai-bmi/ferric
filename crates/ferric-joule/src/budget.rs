//! **Decoding under a joule ceiling.** The largest unowned lever in the field.
//!
//! The evidence is third-party and settled. Autoregressive decode is **86 to 97%** of inference energy
//! on edge vision-language models; each output token costs **11 to 39x** an input token; deleting every
//! visual token saves at most 10% while controlling generation length saves up to **97%**
//! (arXiv:2607.09520). Reasoning modes cost 13x to 25x for the same job, close to linear in output
//! tokens (arXiv:2509.20241, arXiv:2601.22076).
//!
//! Across a 91-group survey of everyone working on AI energy, **nobody builds a technique around it.**
//! The compression layer races to 1.58 bits for gains the same survey measured at 1.4 to 3x, while a 97%
//! lever sits unattended.
//!
//! ## Why this needs a certificate and not just a cap
//!
//! Truncating generation to save energy is trivial and dishonest. A capped answer that is presented as a
//! complete one has not saved energy, it has moved the cost onto whoever reads it and then asks again.
//! The retry is not free: one measured agentic run spent 62.4% of its energy on a failed attempt before
//! the successful retry (arXiv:2605.22883).
//!
//! So a budget here never silently truncates. Every generation ends with a [`Stop`] that says which of
//! four things happened, and [`Outcome::honest`] is false whenever the answer was cut short without the
//! caller being told. The whole point is that the saving is auditable rather than asserted.
//!
//! ## The rule the caller supplies
//!
//! This crate cannot know whether an answer is adequate, so it does not pretend to. The caller supplies
//! an [`Adequacy`] test, and the budget's job is to spend joules until either the test passes, the
//! ceiling is reached, or the generator stops on its own. That division is deliberate: adequacy is
//! domain knowledge, accounting is not.

use crate::{Meter, Reading, measure};

/// Why generation ended. Every one of these is reported; none is silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The generator emitted its own end token. Nothing was cut.
    Natural,
    /// The caller's adequacy test passed, so further tokens would have been paid for and discarded.
    /// This is the case the lever exists to create.
    Adequate,
    /// The joule ceiling was reached. The answer is INCOMPLETE and the caller must be told.
    Exhausted,
    /// The token ceiling was reached. Also incomplete.
    TokenLimit,
}

impl Stop {
    /// Whether the answer is complete on its own terms.
    ///
    /// `Exhausted` and `TokenLimit` are not. Reporting an energy saving from a truncated answer without
    /// saying it was truncated is the failure this type exists to make impossible.
    pub fn complete(self) -> bool {
        matches!(self, Stop::Natural | Stop::Adequate)
    }
    pub fn label(self) -> &'static str {
        match self {
            Stop::Natural => "natural",
            Stop::Adequate => "adequate",
            Stop::Exhausted => "budget exhausted (INCOMPLETE)",
            Stop::TokenLimit => "token limit (INCOMPLETE)",
        }
    }
}

/// The caller's judgement of whether generation so far is good enough to stop.
///
/// Called after each token. Return `true` to stop early, which is the entire saving.
pub trait Adequacy {
    fn adequate(&mut self, tokens_so_far: usize) -> bool;
}

/// Never stop early. The honest default: with no adequacy rule, a budget can only cap, and capping is
/// not a saving.
pub struct RunToCompletion;
impl Adequacy for RunToCompletion {
    fn adequate(&mut self, _: usize) -> bool { false }
}

impl<F: FnMut(usize) -> bool> Adequacy for F {
    fn adequate(&mut self, n: usize) -> bool { self(n) }
}

/// What one budgeted generation actually cost and produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub stop: Stop,
    pub tokens: usize,
    /// Energy spent on the whole generation, measured.
    pub reading: Option<Reading>,
    /// The ceiling that was set, for the record.
    pub ceiling_joules: f64,
}

impl Outcome {
    /// Joules per output token, the unit that matters for the decode lever.
    pub fn per_token(&self) -> Option<f64> {
        let r = self.reading.as_ref()?;
        (self.tokens > 0).then(|| r.joules / self.tokens as f64)
    }

    /// Fraction of the ceiling actually spent.
    pub fn utilisation(&self) -> Option<f64> {
        let r = self.reading.as_ref()?;
        (self.ceiling_joules > 0.0).then(|| r.joules / self.ceiling_joules)
    }

    /// Whether this outcome may be reported as a saving without further qualification.
    ///
    /// False when the answer was cut short. A truncated answer has not saved energy; it has deferred the
    /// cost to a retry, and the retry is where the field's measured 62.4% waste comes from.
    pub fn honest(&self) -> bool { self.stop.complete() }
}

/// Generate under a joule ceiling, stopping on adequacy, exhaustion, or the generator's own end.
///
/// `step` produces one token and returns `false` when the generator is finished on its own terms. The
/// meter is read every `check_every` tokens rather than every token, because sampling a power counter
/// costs real time (~10 ms for `nvidia-smi`) and doing it per token would make the instrument the
/// dominant cost. That interval is a knob rather than a constant because the right value depends on the
/// meter, and picking it wrong silently changes what is being measured.
pub struct Budget {
    pub ceiling_joules: f64,
    pub max_tokens: usize,
    pub check_every: usize,
}

impl Budget {
    pub fn new(ceiling_joules: f64) -> Self {
        Self { ceiling_joules, max_tokens: usize::MAX, check_every: 8 }
    }
    pub fn max_tokens(mut self, n: usize) -> Self { self.max_tokens = n; self }
    pub fn check_every(mut self, n: usize) -> Self { self.check_every = n.max(1); self }

    /// Run a budgeted generation.
    pub fn run<M: Meter>(
        &self,
        meter: &M,
        mut step: impl FnMut() -> bool,
        mut adequacy: impl Adequacy,
    ) -> Outcome {
        let start = meter.read_joules();
        let t0 = std::time::Instant::now();
        let mut tokens = 0usize;
        let mut stop = Stop::Natural;

        loop {
            if tokens >= self.max_tokens { stop = Stop::TokenLimit; break; }
            if !step() { stop = Stop::Natural; break; }
            tokens += 1;

            if adequacy.adequate(tokens) { stop = Stop::Adequate; break; }

            if tokens % self.check_every == 0 {
                if let (Some(a), Some(b)) = (start, meter.read_joules()) {
                    if b - a >= self.ceiling_joules { stop = Stop::Exhausted; break; }
                }
            }
        }

        let reading = match (start, meter.read_joules()) {
            (Some(a), Some(b)) if b >= a => Some(Reading {
                joules: b - a,
                seconds: t0.elapsed().as_secs_f64(),
                class: meter.class(),
                source: meter.source(),
                boundary: meter.boundary(),
            }),
            _ => None,
        };

        Outcome { stop, tokens, reading, ceiling_joules: self.ceiling_joules }
    }
}

/// Aggregate outcomes over a workload, keeping truncation visible.
#[derive(Debug, Clone, Default)]
pub struct Budgeted {
    pub outcomes: Vec<Outcome>,
}

impl Budgeted {
    pub fn push(&mut self, o: Outcome) { self.outcomes.push(o); }
    pub fn joules(&self) -> f64 {
        self.outcomes.iter().filter_map(|o| o.reading.as_ref()).map(|r| r.joules).sum()
    }
    pub fn tokens(&self) -> usize { self.outcomes.iter().map(|o| o.tokens).sum() }

    /// Generations that ended complete. This is the denominator for any claim.
    pub fn complete(&self) -> usize { self.outcomes.iter().filter(|o| o.honest()).count() }

    /// Generations cut short by a ceiling. **A saving computed while this is nonzero is not a saving**,
    /// it is a deferral, and `claim` refuses it.
    pub fn truncated(&self) -> usize { self.outcomes.len() - self.complete() }

    /// Joules per complete generation.
    pub fn per_complete(&self) -> f64 {
        let c = self.complete();
        if c == 0 { f64::NAN } else { self.joules() / c as f64 }
    }

    /// Whether these outcomes may back an energy claim, and why not when they may not.
    pub fn claim(&self) -> Result<(), String> {
        if self.outcomes.is_empty() { return Err("no generations recorded".into()); }
        if self.complete() == 0 {
            return Err("every generation was cut short: that is a cap, not a saving".into());
        }
        if self.truncated() > 0 {
            return Err(format!(
                "{} of {} generations were cut short by a ceiling. Their cost moved to whoever asks again, \
                 and the retry is where the field's measured 62.4% waste comes from. Report the truncation \
                 rate alongside the saving or lower the ceiling until it is zero.",
                self.truncated(), self.outcomes.len()));
        }
        Ok(())
    }

    /// How generations ended, which is the readout that tells you whether the ceiling is set right.
    pub fn stop_mix(&self) -> [(Stop, usize); 4] {
        let count = |s: Stop| self.outcomes.iter().filter(|o| o.stop == s).count();
        [(Stop::Natural, count(Stop::Natural)), (Stop::Adequate, count(Stop::Adequate)),
         (Stop::Exhausted, count(Stop::Exhausted)), (Stop::TokenLimit, count(Stop::TokenLimit))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, Class};

    struct Step { j: std::cell::Cell<f64>, per: f64 }
    impl Step {
        fn new(per: f64) -> Self { Self { j: std::cell::Cell::new(0.0), per } }
    }
    impl Meter for Step {
        fn read_joules(&self) -> Option<f64> {
            let v = self.j.get() + self.per;
            self.j.set(v);
            Some(v)
        }
        fn class(&self) -> Class { Class::Measured }
        fn source(&self) -> &'static str { "step" }
        fn boundary(&self) -> Boundary { Boundary::DEVICE }
    }

    #[test]
    fn stopping_on_adequacy_is_the_saving_and_is_complete() {
        // The case the lever exists for: the answer was good enough at token 10, so tokens 11..200 were
        // never paid for. Nothing was cut, so this is honestly claimable.
        let m = Step::new(1.0);
        let b = Budget::new(1e9).max_tokens(200);
        let o = b.run(&m, || true, |n: usize| n >= 10);
        assert_eq!(o.stop, Stop::Adequate);
        assert_eq!(o.tokens, 10);
        assert!(o.honest(), "an adequate stop must be honest");
    }

    #[test]
    fn a_ceiling_that_bites_produces_an_INCOMPLETE_outcome() {
        // Truncation is allowed, but it is never silent. This is the whole contract.
        let m = Step::new(10.0);
        let b = Budget::new(25.0).max_tokens(1000).check_every(1);
        let o = b.run(&m, || true, RunToCompletion);
        assert_eq!(o.stop, Stop::Exhausted);
        assert!(!o.honest(), "a truncated answer was reported as complete");
        assert!(!o.stop.complete());
    }

    #[test]
    fn a_workload_of_only_truncations_is_a_cap_not_a_saving() {
        let m = Step::new(10.0);
        let mut w = Budgeted::default();
        for _ in 0..5 {
            w.push(Budget::new(15.0).check_every(1).run(&m, || true, RunToCompletion));
        }
        assert_eq!(w.complete(), 0);
        assert!(w.claim().unwrap_err().contains("cap, not a saving"));
    }

    #[test]
    fn a_partially_truncated_workload_must_report_its_truncation_rate() {
        // The subtle failure: 90% complete looks like a clean 90% and the 10% deferred cost vanishes.
        let m = Step::new(1.0);
        let mut w = Budgeted::default();
        for i in 0..10 {
            let generous = i < 9;
            let b = Budget::new(if generous { 1e9 } else { 2.0 }).max_tokens(50).check_every(1);
            w.push(b.run(&m, || true, |n: usize| generous && n >= 5));
        }
        assert_eq!(w.complete(), 9);
        assert_eq!(w.truncated(), 1);
        let e = w.claim().unwrap_err();
        assert!(e.contains("1 of 10"), "{e}");
        assert!(e.contains("62.4%"), "the reason must cite why deferral is not free");
    }

    #[test]
    fn a_generator_that_finishes_on_its_own_is_natural_and_complete() {
        let m = Step::new(1.0);
        let mut left = 4;
        let o = Budget::new(1e9).run(&m, || { left -= 1; left > 0 }, RunToCompletion);
        assert_eq!(o.stop, Stop::Natural);
        assert_eq!(o.tokens, 3);
        assert!(o.honest());
    }

    #[test]
    fn joules_per_output_token_is_reported() {
        // The unit for this lever, since each output token costs 11-39x an input token.
        let m = Step::new(2.0);
        let o = Budget::new(1e9).max_tokens(10).run(&m, || true, RunToCompletion);
        assert_eq!(o.stop, Stop::TokenLimit);
        assert_eq!(o.tokens, 10);
        assert!(o.per_token().unwrap() > 0.0);
    }

    #[test]
    fn the_stop_mix_shows_whether_the_ceiling_is_set_right() {
        // All-Exhausted means the ceiling is too tight; all-Natural means it never binds and is
        // therefore buying nothing. The mix is how a caller tunes it.
        let m = Step::new(1.0);
        let mut w = Budgeted::default();
        w.push(Budget::new(1e9).run(&m, || { false }, RunToCompletion));
        w.push(Budget::new(1e9).max_tokens(3).run(&m, || true, |n: usize| n >= 2));
        let mix = w.stop_mix();
        assert_eq!(mix[0].1, 1, "expected one natural stop");
        assert_eq!(mix[1].1, 1, "expected one adequate stop");
    }

    #[test]
    fn checking_the_meter_every_token_is_not_the_default() {
        // Sampling a power counter costs ~10 ms on nvidia-smi. Per-token checking would make the
        // instrument the dominant cost and quietly change what is being measured.
        assert_eq!(Budget::new(1.0).check_every, 8);
        assert_eq!(Budget::new(1.0).check_every(0).check_every, 1, "check_every must never be zero");
    }
}
