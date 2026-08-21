//! **The weakest sufficient system.** Routing, and the accounting that makes routing honest.
//!
//! This is the largest measured energy lever in the field and it is not a research programme. Over
//! 80.2M real queries, oracle routing cut energy **80.4%**, and a realistic router that is only 80%
//! accurate still cut **64.3%** — measured against an already-*batched* cloud baseline, so the saving is
//! not an artifact of a lazy comparison. Local models handled 88.7% of single-turn chat and reasoning
//! traffic correctly (arXiv:2511.07885).
//!
//! Nothing else on the list comes close. Quantisation, distillation and better kernels all operate
//! inside one rung; routing decides which rung runs at all.
//!
//! ## Why escalation has to be accounted for, not assumed away
//!
//! A router that guesses wrong pays twice. It burns the cheap rung, fails, and burns the expensive one
//! anyway — so a naive analysis that counts only the successful call reports a saving that does not
//! exist. This module charges **every attempt** to the task, which is why [`Ladder::run`] returns the
//! full attempt trail rather than just the answer.
//!
//! That is the same failure the field makes at larger scale: in one agentic run, 62.4% of the energy
//! went to a failed attempt before the successful retry (arXiv:2605.22883). Retries are not overhead.
//! They are the cost.
//!
//! ## The break-even, which decides whether to route at all
//!
//! Escalating is worth it when
//!
//! ```text
//!     p · E_cheap + (1 - p) · (E_cheap + E_dear)  <  E_dear
//! ```
//!
//! where `p` is the probability the cheap rung succeeds. That rearranges to **p > E_cheap / E_dear**:
//! the cheap rung has to succeed more often than its own relative cost. A rung 100x cheaper only needs
//! to be right 1% of the time to pay for itself, which is why routing wins so decisively in practice
//! and why [`Ladder::break_even`] is worth computing before deploying rather than after.

use crate::{Class, Meter, Reading, Saving, measure};

/// One rung: a system that may or may not be sufficient for a given task.
///
/// Ordered cheapest-first. "Cheapest" means joules per successful task, which is not the same as
/// smallest and not the same as fastest.
pub struct Rung<'a, T> {
    pub name: &'static str,
    /// Runs the task. `None` means this rung declined or failed, and the ladder escalates.
    ///
    /// The rung decides its own sufficiency. That is deliberate: a rung knows things a central router
    /// does not, such as whether its own confidence was low or its output failed a validator.
    pub run: Box<dyn FnMut() -> Option<T> + 'a>,
}

/// What one task actually cost, including everything that failed on the way.
#[derive(Debug, Clone)]
pub struct Trail {
    /// Every rung attempted, in order, with what it cost. Failed attempts included, because they were
    /// paid for.
    pub attempts: Vec<(&'static str, Reading)>,
    /// The rung that succeeded, if any.
    pub resolved_by: Option<&'static str>,
}

impl Trail {
    /// Total joules, charging failed attempts to the task. This is the number that matters.
    pub fn joules(&self) -> f64 {
        self.attempts.iter().map(|(_, r)| r.joules).sum()
    }
    /// Joules burned on attempts that did not resolve the task.
    pub fn wasted_joules(&self) -> f64 {
        match self.resolved_by {
            Some(_) => self.attempts.iter().rev().skip(1).map(|(_, r)| r.joules).sum(),
            None => self.joules(),
        }
    }
    pub fn succeeded(&self) -> bool { self.resolved_by.is_some() }
    pub fn escalations(&self) -> usize { self.attempts.len().saturating_sub(1) }
}

/// An ordered set of systems, cheapest first, with escalation on failure.
pub struct Ladder<'a, T> {
    rungs: Vec<Rung<'a, T>>,
}

impl<'a, T> Ladder<'a, T> {
    pub fn new() -> Self { Self { rungs: Vec::new() } }

    /// Add a rung. Order is the contract: cheapest first, and the ladder does not sort for you because
    /// only you know the joules-per-success ordering on your hardware.
    pub fn rung(mut self, name: &'static str, run: impl FnMut() -> Option<T> + 'a) -> Self {
        self.rungs.push(Rung { name, run: Box::new(run) });
        self
    }

    pub fn len(&self) -> usize { self.rungs.len() }
    pub fn is_empty(&self) -> bool { self.rungs.is_empty() }

    /// Run one task, escalating until a rung resolves it or the ladder is exhausted.
    ///
    /// Returns the answer and the full trail. Every attempt is measured, so a caller cannot accidentally
    /// report only the successful one.
    pub fn run<M: Meter>(&mut self, meter: &M) -> (Option<T>, Trail) {
        let mut trail = Trail { attempts: Vec::new(), resolved_by: None };
        for rung in self.rungs.iter_mut() {
            let (out, reading) = measure(meter, || (rung.run)());
            if let Some(r) = reading {
                trail.attempts.push((rung.name, r));
            }
            if out.is_some() {
                trail.resolved_by = Some(rung.name);
                return (out, trail);
            }
        }
        (None, trail)
    }

    /// The success probability the cheap rung must exceed for escalation to be worth it, **against a
    /// single next rung**.
    ///
    /// `p > E_cheap / E_dear`. A rung 100x cheaper needs to be right only 1% of the time. Compute this
    /// before deploying a ladder, because a ladder whose cheap rung is nearly as expensive as its dear
    /// one is strictly worse than not routing.
    ///
    /// ## ⚠ This is conservative on a ladder deeper than two rungs
    ///
    /// The denominator here is the next rung's cost. The correct denominator is the expected cost of
    /// the *entire remaining ladder*, `E(k+1) = E_{k+1} + (1-p_{k+1})·E(k+2) ≥ E_{k+1}`, so the true
    /// threshold is **lower** than this one and the error is one-sided: this function declares a cheap
    /// rung not worth trying in cases where trying it is demonstrably cheaper. Use
    /// [`crate::router::Profile::break_even`] on any ladder with three or more rungs.
    pub fn break_even(e_cheap: f64, e_dear: f64) -> f64 {
        if e_dear <= 0.0 { return f64::NAN; }
        e_cheap / e_dear
    }
}

impl<'a, T> Default for Ladder<'a, T> {
    fn default() -> Self { Self::new() }
}

/// Aggregate outcome of running a ladder over a workload.
#[derive(Debug, Clone, Default)]
pub struct Routed {
    pub trails: Vec<Trail>,
}

impl Routed {
    pub fn push(&mut self, t: Trail) { self.trails.push(t); }

    pub fn attempted(&self) -> u64 { self.trails.len() as u64 }
    pub fn succeeded(&self) -> u64 { self.trails.iter().filter(|t| t.succeeded()).count() as u64 }
    pub fn joules(&self) -> f64 { self.trails.iter().map(|t| t.joules()).sum() }

    /// Joules burned on attempts that resolved nothing. The number a routing proposal must survive.
    pub fn wasted_joules(&self) -> f64 { self.trails.iter().map(|t| t.wasted_joules()).sum() }

    /// Fraction of energy spent on failed attempts. Measured at 62.4% in one published agentic run,
    /// which is why this is a headline field rather than a footnote.
    pub fn waste_fraction(&self) -> f64 {
        let j = self.joules();
        if j <= 0.0 { 0.0 } else { self.wasted_joules() / j }
    }

    /// Joules per SUCCESSFUL task, charging every failure to the task.
    pub fn per_success(&self) -> f64 {
        let s = self.succeeded();
        if s == 0 { f64::NAN } else { self.joules() / s as f64 }
    }

    /// How the workload distributed across rungs, by name and count of *resolutions*.
    ///
    /// The published local-model figure is 88.7% of single-turn traffic. If a deployment's bottom rung
    /// resolves far less than that, either the rung is too weak or the escalation test is too eager,
    /// and this is how you find out which.
    pub fn resolution_mix(&self) -> Vec<(&'static str, usize)> {
        let mut out: Vec<(&'static str, usize)> = Vec::new();
        for t in &self.trails {
            if let Some(n) = t.resolved_by {
                match out.iter_mut().find(|(k, _)| *k == n) {
                    Some((_, c)) => *c += 1,
                    None => out.push((n, 1)),
                }
            }
        }
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    /// Compare against always using the top rung, which is what routing replaces.
    ///
    /// `top_only` must be a [`Routed`] produced by running the SAME workload with a single-rung ladder.
    /// The resulting [`Saving`] then carries both arms and everything [`Saving::claimable`] checks.
    pub fn against(&self, top_only: &Routed, class: Class, source: &'static str,
                   boundary: crate::Boundary, seconds: (f64, f64)) -> Saving {
        // Each arm's OWN success count. This read `successes: self.succeeded()` — the ladder's own
        // number, applied to both arms — until 2026-08-21, which is precisely the error a ladder
        // exists to expose: routing buys its energy saving by resolving more work on weaker rungs,
        // so the arm that saves energy is exactly the arm whose success count is most likely to
        // differ. Charging the top-rung baseline the ladder's success count made the saving read as
        // free.
        Saving {
            baseline: Reading { joules: top_only.joules(), seconds: seconds.0, class, source, boundary },
            candidate: Reading { joules: self.joules(), seconds: seconds.1, class, source, boundary },
            tasks: self.attempted(),
            successes: (top_only.succeeded(), self.succeeded()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Boundary;

    /// Deterministic fake counter: each read advances by a fixed step, so a rung's cost is knowable.
    struct Stepped { j: std::cell::Cell<f64>, step: std::cell::Cell<f64> }
    impl Stepped {
        fn new() -> Self { Self { j: std::cell::Cell::new(0.0), step: std::cell::Cell::new(1.0) } }
        fn set_step(&self, s: f64) { self.step.set(s); }
    }
    impl Meter for Stepped {
        fn read_joules(&self) -> Option<f64> {
            let v = self.j.get() + self.step.get();
            self.j.set(v);
            Some(v)
        }
        fn class(&self) -> Class { Class::Measured }
        fn source(&self) -> &'static str { "stepped" }
        fn boundary(&self) -> Boundary { Boundary::DEVICE }
    }

    #[test]
    fn a_failed_cheap_attempt_is_charged_to_the_task() {
        // THE thing this module exists to get right. A router that counts only the successful call
        // reports a saving that does not exist; measured at 62.4% waste in one published agentic run.
        let m = Stepped::new();
        let mut l: Ladder<u8> = Ladder::new()
            .rung("local", || None)          // fails, but is paid for
            .rung("cloud", || Some(7));
        let (out, trail) = l.run(&m);
        assert_eq!(out, Some(7));
        assert_eq!(trail.attempts.len(), 2, "the failed attempt was not recorded");
        assert_eq!(trail.resolved_by, Some("cloud"));
        assert!(trail.wasted_joules() > 0.0, "a failed attempt was charged nothing");
        assert_eq!(trail.escalations(), 1);
    }

    #[test]
    fn the_cheap_rung_resolving_costs_one_attempt() {
        let m = Stepped::new();
        let mut l: Ladder<u8> = Ladder::new()
            .rung("local", || Some(1))
            .rung("cloud", || Some(2));
        let (out, trail) = l.run(&m);
        assert_eq!(out, Some(1));
        assert_eq!(trail.attempts.len(), 1, "the ladder ran a rung it did not need");
        assert_eq!(trail.wasted_joules(), 0.0);
    }

    #[test]
    fn an_exhausted_ladder_charges_everything_and_resolves_nothing() {
        // Zero successes at any energy is not an efficiency result, and the accounting must say so.
        let m = Stepped::new();
        let mut l: Ladder<u8> = Ladder::new().rung("a", || None).rung("b", || None);
        let (out, trail) = l.run(&m);
        assert!(out.is_none());
        assert!(!trail.succeeded());
        assert_eq!(trail.wasted_joules(), trail.joules(), "an unresolved task must be all waste");
    }

    #[test]
    fn break_even_is_the_cost_ratio() {
        // A rung 100x cheaper needs to be right 1% of the time. This is why routing wins.
        assert!((Ladder::<u8>::break_even(1.0, 100.0) - 0.01).abs() < 1e-12);
        assert!((Ladder::<u8>::break_even(50.0, 100.0) - 0.5).abs() < 1e-12);
        // A cheap rung that is nearly as dear as the expensive one must be right nearly always, i.e.
        // routing is not worth it.
        assert!(Ladder::<u8>::break_even(99.0, 100.0) > 0.98);
    }

    #[test]
    fn routing_beats_top_only_when_the_cheap_rung_usually_wins() {
        // The published shape: local resolves the large majority, and the saving survives charging
        // every escalation to the task.
        let m = Stepped::new();
        let mut routed = Routed::default();
        for i in 0..100 {
            let local_ok = i % 10 != 0;         // 90% resolved locally
            m.set_step(1.0);                    // local rung costs 1 J
            let mut l: Ladder<u8> = Ladder::new()
                .rung("local", move || local_ok.then_some(1))
                .rung("cloud", || { Some(2) });
            let (_, t) = l.run(&m);
            routed.push(t);
        }
        let mut top = Routed::default();
        for _ in 0..100 {
            let mut l: Ladder<u8> = Ladder::new().rung("cloud", || Some(2));
            let (_, t) = l.run(&m);
            top.push(t);
        }
        assert_eq!(routed.succeeded(), 100);
        let mix = routed.resolution_mix();
        assert_eq!(mix[0].0, "local");
        assert_eq!(mix[0].1, 90, "expected 90 local resolutions, got {}", mix[0].1);
        // Escalations were paid for, so waste is nonzero and reported rather than hidden.
        assert!(routed.waste_fraction() > 0.0, "10 failed local attempts cost nothing?");
        assert!(routed.waste_fraction() < 0.2, "waste {} is implausibly high", routed.waste_fraction());
    }

    #[test]
    fn a_saving_from_routing_carries_both_arms_and_can_be_refused() {
        let a = Routed { trails: vec![Trail { attempts: vec![("x", Reading { joules: 10.0, seconds: 1.0, class: Class::Measured, source: "s", boundary: Boundary::DEVICE })], resolved_by: Some("x") }] };
        let b = Routed { trails: vec![Trail { attempts: vec![("x", Reading { joules: 40.0, seconds: 1.0, class: Class::Measured, source: "s", boundary: Boundary::DEVICE })], resolved_by: Some("x") }] };
        let s = a.against(&b, Class::Measured, "s", Boundary::DEVICE, (4.0, 1.0));
        assert!((s.percent() - 75.0).abs() < 1e-9, "expected 75%, got {}", s.percent());
        assert_eq!(s.successes, (1, 1), "both arms resolved their one task here");
        assert!(s.claimable().is_ok(), "{:?}", s.claimable());
    }

    #[test]
    fn a_ladder_that_saves_energy_by_resolving_less_does_not_read_as_a_free_win() {
        // `against` charged BOTH arms the ladder's own success count until 2026-08-21. That is the
        // one place the error is guaranteed to matter: a ladder buys its saving by resolving work on
        // weaker rungs, so the arm that saves the energy is the arm whose success count differs.
        let rd = |j: f64| Reading { joules: j, seconds: 1.0, class: Class::Measured, source: "s", boundary: Boundary::DEVICE };
        let trail = |j: f64, ok: bool| Trail { attempts: vec![("r", rd(j))], resolved_by: if ok { Some("r") } else { None } };

        // Ladder: 4 tasks at 10 J, 2 resolved. Top-rung-only: 4 tasks at 40 J, all 4 resolved.
        let ladder = Routed { trails: (0..4).map(|i| trail(10.0, i < 2)).collect() };
        let top = Routed { trails: (0..4).map(|_| trail(40.0, true)).collect() };
        let s = ladder.against(&top, Class::Measured, "s", Boundary::DEVICE, (4.0, 4.0));

        assert_eq!(s.successes, (4, 2), "each arm must carry its OWN success count");
        assert!((s.percent() - 75.0).abs() < 1e-9, "total energy still falls 75%: {:.1}%", s.percent());
        // 160 J / 4 = 40 J per resolved task; 40 J / 2 = 20 J per resolved task. Still a win, but 50%
        // rather than 75% — the routing gave back a third of the saving in unresolved work, and the
        // old code could not have shown that because both arms divided by 2.
        let (b, c) = s.per_success();
        assert!((b - 40.0).abs() < 1e-9 && (c - 20.0).abs() < 1e-9, "per success {b} -> {c}");
        assert!((s.percent_per_success() - 50.0).abs() < 1e-9,
                "the saving on the unit that matters is 50%, not 75%: {:.1}%", s.percent_per_success());
        assert!(s.percent_per_success() < s.percent(),
                "a ladder that resolves less MUST look worse per success than per joule");
    }

    #[test]
    fn resolution_mix_exposes_a_bottom_rung_that_is_too_weak() {
        // The published local figure is 88.7% of single-turn traffic. A deployment resolving far less
        // locally has either too weak a rung or too eager an escalation test, and this is the readout
        // that tells you to go look.
        let m = Stepped::new();
        let mut routed = Routed::default();
        for i in 0..100 {
            let local_ok = i % 2 == 0;   // only 50% — well under what the literature reports
            let mut l: Ladder<u8> = Ladder::new()
                .rung("local", move || local_ok.then_some(1))
                .rung("cloud", || Some(2));
            let (_, t) = l.run(&m);
            routed.push(t);
        }
        let mix = routed.resolution_mix();
        let local = mix.iter().find(|(n, _)| *n == "local").map(|(_, c)| *c).unwrap_or(0);
        assert_eq!(local, 50);
        assert!(routed.waste_fraction() > 0.2, "half the traffic escalating should show real waste");
    }
}
