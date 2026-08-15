//! **Predictive routing**: choose which rung to *start* on, instead of always starting at the bottom.
//!
//! [`crate::ladder`] implements escalation — try cheap, promote on failure. That is one routing
//! algorithm out of several, and it is the one that always pays the cheap rung even when the cheap
//! rung was never going to work. This module implements the other half: a router that reads the
//! request first and skips rungs it predicts will fail.
//!
//! ## Why this exists as a separate thing
//!
//! NVIDIA's Switchyard (Rust, Apache-2.0) ships `escalation` alongside `llm_class`, `stage`, `rand`,
//! `affinity` and a prefill router that reads residual-stream signals to predict per-model accuracy.
//! OpenRouter does the same job across providers. Both optimise **cost and latency**. Neither routes
//! on energy — `grep -icE "energy|joule|watt|power"` over the entire Switchyard tree returns 0.
//!
//! Routing on dollars is the correct objective for a datacenter API broker, and this module does not
//! claim otherwise. It occupies the axis they leave empty: the objective here is **joules per
//! successful task**, charging every failed attempt to the task, which is the number that decides
//! whether a workload can run on a device at all.
//!
//! ## The decision rule, derived rather than tuned
//!
//! Let rungs be ordered cheapest-first with per-attempt energies `E_k` and success probabilities
//! `p_k`. Starting at rung `k` and escalating on failure costs, in expectation,
//!
//! ```text
//!     E(k) = E_k + (1 - p_k) · E(k+1),        E(n) = 0
//! ```
//!
//! The router picks `argmin_k E(k)`. Rearranging `E(k) < E(k+1)` gives the condition for trying rung
//! `k` at all:
//!
//! ```text
//!     p_k  >  E_k / E(k+1)
//! ```
//!
//! ### ⚠ This corrects [`crate::ladder::Ladder::break_even`]
//!
//! That function computes `E_cheap / E_dear` against the **next rung's** cost. The correct denominator
//! is `E(k+1)`, the expected cost of the *entire remaining ladder*, and since
//! `E(k+1) = E_{k+1} + (1-p_{k+1})·E(k+2) ≥ E_{k+1}`, the true threshold is **lower**. The two-rung
//! formula is therefore conservative: it declares a cheap rung not worth trying in cases where it
//! demonstrably is. The error is one-sided and grows with ladder depth. See
//! [`tests::the_two_rung_break_even_is_conservative_on_a_deep_ladder`].
//!
//! ## Prediction is a liability until it is calibrated
//!
//! A predictive router that is confidently wrong is *worse than no router*: it skips the rung that
//! would have worked, pays the expensive one, and the accounting shows a saving only if you forget to
//! charge the skip. The published figures make the stakes explicit — oracle routing cut energy 80.4%
//! over 80.2M queries, but a router that is 80% accurate cut 64.3% (arXiv:2511.07885). The gap between
//! those two numbers is the predictor's error, and it is the only thing a routing proposal is really
//! about.
//!
//! So [`Router`] tracks its own [`Calibration`] and [`Router::regret`] reports joules spent above what
//! an oracle would have spent. A saving reported without those is not reported here.

use crate::ladder::Trail;
use crate::Reading;

/// What one rung costs and how often it works — measured, not assumed.
///
/// `joules` is the mean energy of one *attempt*, success or failure. It must come from real
/// [`Reading`]s: a routing plan built on nameplate arithmetic optimises a model of the machine rather
/// than the machine.
#[derive(Debug, Clone)]
pub struct RungProfile {
    pub name: &'static str,
    /// Mean joules per attempt at this rung.
    pub joules: f64,
    /// Number of attempts the mean is drawn from. Zero means "never measured", and
    /// [`Profile::from_readings`] refuses to build a plan from it.
    pub attempts: u64,
}

impl RungProfile {
    /// Mean cost of a set of measured attempts.
    pub fn from_readings(name: &'static str, rs: &[Reading]) -> RungProfile {
        let n = rs.len() as u64;
        let j = if n == 0 { 0.0 } else { rs.iter().map(|r| r.joules).sum::<f64>() / n as f64 };
        RungProfile { name, joules: j, attempts: n }
    }
}

/// Why a plan could not be produced. Every variant is a case where guessing would produce a route
/// that runs and wastes energy rather than one that fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// No rungs at all.
    Empty,
    /// A rung has never been measured, so its cost is unknown rather than zero.
    Unmeasured(&'static str),
    /// A rung reported non-positive or non-finite energy.
    BadEnergy(&'static str),
    /// A predicted probability was outside `[0, 1]` or not finite.
    BadProbability(&'static str),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Empty => write!(f, "no rungs to route between"),
            PlanError::Unmeasured(n) => write!(f, "rung {n:?} has no measured attempts"),
            PlanError::BadEnergy(n) => write!(f, "rung {n:?} reported non-positive or non-finite joules"),
            PlanError::BadProbability(n) => write!(f, "rung {n:?} was given a probability outside [0,1]"),
        }
    }
}

impl std::error::Error for PlanError {}

/// An energy-ordered set of rungs. Construction validates; nothing downstream has to re-check.
#[derive(Debug, Clone)]
pub struct Profile {
    rungs: Vec<RungProfile>,
}

impl Profile {
    /// Build from measured rungs. Rungs are **sorted cheapest-first here** rather than trusting the
    /// caller's order, because the decision rule's correctness depends on the ordering and a
    /// mis-ordered ladder produces a plan that is valid-looking and wrong.
    pub fn new(mut rungs: Vec<RungProfile>) -> Result<Profile, PlanError> {
        if rungs.is_empty() { return Err(PlanError::Empty); }
        for r in &rungs {
            if r.attempts == 0 { return Err(PlanError::Unmeasured(r.name)); }
            if !r.joules.is_finite() || r.joules <= 0.0 { return Err(PlanError::BadEnergy(r.name)); }
        }
        rungs.sort_by(|a, b| a.joules.partial_cmp(&b.joules).expect("finite, checked above"));
        Ok(Profile { rungs })
    }

    pub fn len(&self) -> usize { self.rungs.len() }
    pub fn is_empty(&self) -> bool { self.rungs.is_empty() }
    pub fn names(&self) -> Vec<&'static str> { self.rungs.iter().map(|r| r.name).collect() }
    pub fn joules(&self, k: usize) -> f64 { self.rungs[k].joules }

    /// Expected joules of starting at each rung and escalating on failure: `E(k)` for every `k`,
    /// plus the terminal `E(n) = 0`. Returned in full because the *shape* of this vector is what tells
    /// you whether a ladder is worth having at all.
    ///
    /// `p[k]` is the probability rung `k` resolves the task.
    pub fn expected_costs(&self, p: &[f64]) -> Result<Vec<f64>, PlanError> {
        for (k, r) in self.rungs.iter().enumerate() {
            let pk = p.get(k).copied().unwrap_or(0.0);
            if !pk.is_finite() || !(0.0..=1.0).contains(&pk) {
                return Err(PlanError::BadProbability(r.name));
            }
        }
        let n = self.rungs.len();
        let mut e = vec![0.0; n + 1];
        // Backwards: E(k) needs E(k+1). Nothing after the last rung, so E(n) = 0 — an exhausted
        // ladder resolves nothing, and that is accounted as total waste by `Trail`, not as free.
        for k in (0..n).rev() {
            let pk = p.get(k).copied().unwrap_or(0.0);
            e[k] = self.rungs[k].joules + (1.0 - pk) * e[k + 1];
        }
        Ok(e)
    }

    /// The rung to start on, and what starting there is expected to cost.
    ///
    /// Ties go to the **cheaper** rung: at equal expected energy, the one that risks less per attempt
    /// is the safer place to be wrong.
    pub fn plan(&self, p: &[f64]) -> Result<Plan, PlanError> {
        let e = self.expected_costs(p)?;
        let mut best = 0usize;
        for k in 1..self.rungs.len() {
            if e[k] < e[best] { best = k; }
        }
        Ok(Plan {
            start: best,
            start_name: self.rungs[best].name,
            expected_joules: e[best],
            escalation_only_joules: e[0],
            skipped: self.rungs[..best].iter().map(|r| r.name).collect(),
        })
    }

    /// The success probability rung `k` must exceed for starting there to beat skipping it.
    ///
    /// `p_k > E_k / E(k+1)`, with `E(k+1)` the expected cost of the whole remaining ladder. Use this
    /// rather than [`crate::ladder::Ladder::break_even`], which uses only the next rung and is
    /// therefore conservative — see the module docs.
    pub fn break_even(&self, k: usize, p: &[f64]) -> Result<f64, PlanError> {
        let e = self.expected_costs(p)?;
        let rest = e[k + 1];
        Ok(if rest <= 0.0 { f64::INFINITY } else { self.rungs[k].joules / rest })
    }
}

/// The routing decision for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Index into the energy-sorted profile.
    pub start: usize,
    pub start_name: &'static str,
    /// Expected joules of this plan, charging escalation.
    pub expected_joules: f64,
    /// What pure escalation (always start at the bottom) would have been expected to cost. Reporting
    /// both is the point: a predictive router that cannot beat this number should be turned off.
    pub escalation_only_joules: f64,
    /// Rungs the prediction chose to skip. Named, because a skip is a claim.
    pub skipped: Vec<&'static str>,
}

impl Plan {
    /// Expected joules saved against pure escalation. Negative means the prediction made it worse,
    /// which is a real outcome and is not clamped away.
    pub fn expected_saving(&self) -> f64 { self.escalation_only_joules - self.expected_joules }
}

/// Predicts, per rung, the chance it resolves a given request.
///
/// Deliberately a trait with no built-in features: the useful signals are domain-specific (prompt
/// length, tool-call presence, a classifier's logit, Switchyard's residual-stream probe) and baking in
/// a feature set would make the module wrong everywhere it did not fit.
pub trait Predictor<R: ?Sized> {
    /// One probability per rung, in the profile's energy-sorted order.
    fn p_success(&self, request: &R, profile: &Profile) -> Vec<f64>;
}

/// The trivial predictor: every rung is equally likely to work. Reduces the router to pure
/// escalation, and exists so that "no predictor" is a *measurable arm* rather than a different code
/// path. Switchyard ships `noop` for the same reason.
pub struct Uniform(pub f64);

impl<R: ?Sized> Predictor<R> for Uniform {
    fn p_success(&self, _r: &R, profile: &Profile) -> Vec<f64> {
        vec![self.0.clamp(0.0, 1.0); profile.len()]
    }
}

/// How well the predictor's probabilities match reality.
///
/// A router is only as good as this. Brier score is used rather than accuracy because a router
/// consumes *probabilities*, not labels: a predictor that says 0.51 and one that says 0.99 produce
/// different plans and identical accuracy.
#[derive(Debug, Clone, Default)]
pub struct Calibration {
    /// `(predicted, actually_succeeded)` for every attempt actually made.
    obs: Vec<(f64, bool)>,
}

impl Calibration {
    pub fn record(&mut self, predicted: f64, succeeded: bool) { self.obs.push((predicted, succeeded)); }
    pub fn n(&self) -> usize { self.obs.len() }

    /// Mean squared error between predicted probability and outcome. 0 is perfect, 0.25 is what you
    /// get by always saying 0.5, and **above 0.25 means the predictor is worse than saying nothing**.
    pub fn brier(&self) -> Option<f64> {
        if self.obs.is_empty() { return None; }
        let s: f64 = self.obs.iter().map(|(p, y)| { let d = p - if *y { 1.0 } else { 0.0 }; d * d }).sum();
        Some(s / self.obs.len() as f64)
    }

    /// Mean predicted probability minus observed success rate. Positive means **overconfident**, which
    /// is the direction that costs energy: it skips cheap rungs that would have worked.
    pub fn bias(&self) -> Option<f64> {
        if self.obs.is_empty() { return None; }
        let n = self.obs.len() as f64;
        let mp: f64 = self.obs.iter().map(|(p, _)| p).sum::<f64>() / n;
        let mo: f64 = self.obs.iter().filter(|(_, y)| *y).count() as f64 / n;
        Some(mp - mo)
    }

    /// Whether the predictor is good enough to be worth consulting. A router whose Brier score is no
    /// better than a coin flip should be replaced by escalation, and this says so rather than leaving
    /// it to a judgement call.
    pub fn beats_guessing(&self) -> bool {
        self.brier().is_some_and(|b| b < 0.25)
    }
}

/// Runs a predictive route and keeps the accounting that makes the result reportable.
///
/// Holds no models and no rung closures — it takes the profile, produces a [`Plan`], and consumes the
/// [`Trail`] the caller produced by executing it. That split exists so this is testable without a GPU
/// and so a caller cannot hand it a trail it did not actually run.
#[derive(Debug, Clone)]
pub struct Router {
    profile: Profile,
    cal: Calibration,
    /// Joules actually spent, across every routed task.
    spent: f64,
    /// Joules an oracle would have spent: straight to the rung that ended up resolving it, once.
    oracle: f64,
    /// Joules pure escalation would have spent on the same outcomes.
    escalation: f64,
    tasks: u64,
    resolved: u64,
}

impl Router {
    pub fn new(profile: Profile) -> Router {
        Router { profile, cal: Calibration::default(), spent: 0.0, oracle: 0.0, escalation: 0.0, tasks: 0, resolved: 0 }
    }

    pub fn profile(&self) -> &Profile { &self.profile }
    pub fn calibration(&self) -> &Calibration { &self.cal }

    /// Plan a request.
    pub fn plan<R: ?Sized>(&self, req: &R, pred: &dyn Predictor<R>) -> Result<Plan, PlanError> {
        let p = pred.p_success(req, &self.profile);
        self.profile.plan(&p)
    }

    /// Record what a routed task actually cost.
    ///
    /// `trail` is the real execution. `predicted` is what the predictor said about each rung that was
    /// actually attempted, so calibration is scored on decisions that were made rather than on
    /// counterfactuals.
    pub fn observe(&mut self, trail: &Trail, predicted: &[(f64, &'static str)]) {
        self.tasks += 1;
        self.spent += trail.joules();
        if trail.succeeded() { self.resolved += 1; }

        for (p, name) in predicted {
            let attempted = trail.attempts.iter().any(|(n, _)| n == name);
            if attempted {
                self.cal.record(*p, trail.resolved_by == Some(*name));
            }
        }

        // An oracle pays exactly once, at the rung that resolved it. If nothing resolved it, the
        // oracle pays nothing — there was no right answer to jump to, and pretending otherwise would
        // flatter the router by inflating the baseline.
        if let Some(win) = trail.resolved_by {
            if let Some(k) = self.profile.rungs.iter().position(|r| r.name == win) {
                self.oracle += self.profile.rungs[k].joules;
                // Pure escalation would have paid every rung from the bottom up to the winner.
                self.escalation += self.profile.rungs[..=k].iter().map(|r| r.joules).sum::<f64>();
            }
        } else {
            self.escalation += self.profile.rungs.iter().map(|r| r.joules).sum::<f64>();
        }
    }

    pub fn tasks(&self) -> u64 { self.tasks }
    pub fn resolved(&self) -> u64 { self.resolved }
    pub fn joules(&self) -> f64 { self.spent }

    /// Joules per successful task, charging every failed attempt. The headline number.
    pub fn per_success(&self) -> f64 {
        if self.resolved == 0 { f64::NAN } else { self.spent / self.resolved as f64 }
    }

    /// Joules spent above what an oracle would have spent. **This is the cost of the predictor being
    /// imperfect**, and it is never negative for a correct router — you cannot beat knowing the answer.
    pub fn regret(&self) -> f64 { self.spent - self.oracle }

    /// Regret as a fraction of the oracle's spend. The published gap between oracle routing (80.4%
    /// saved) and a realistic 80%-accurate router (64.3% saved) is this quantity.
    pub fn regret_fraction(&self) -> f64 {
        if self.oracle <= 0.0 { f64::NAN } else { self.regret() / self.oracle }
    }

    /// Joules saved against pure escalation on the same tasks. **Negative is a real answer** and means
    /// the predictor is doing harm; it is returned rather than clamped so it cannot be reported away.
    pub fn saved_against_escalation(&self) -> f64 { self.escalation - self.spent }

    /// What an oracle would have spent — the floor no router can beat.
    pub fn oracle_joules(&self) -> f64 { self.oracle }
    /// What pure escalation would have spent — the ceiling a router must beat to justify itself.
    pub fn escalation_joules(&self) -> f64 { self.escalation }

    /// Whether this router has earned the right to be deployed: it beat escalation **and** its
    /// predictor beat guessing. Both, because either alone can be true while the router is useless.
    pub fn worth_deploying(&self) -> bool {
        self.saved_against_escalation() > 0.0 && self.cal.beats_guessing()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, Class};

    fn prof(rs: &[(&'static str, f64)]) -> Profile {
        Profile::new(rs.iter().map(|(n, j)| RungProfile { name: n, joules: *j, attempts: 10 }).collect()).unwrap()
    }

    fn reading(j: f64) -> Reading {
        Reading { joules: j, seconds: 1.0, class: Class::Measured, source: "test", boundary: Boundary::DEVICE }
    }

    fn trail(attempts: &[(&'static str, f64)], won: Option<&'static str>) -> Trail {
        Trail {
            attempts: attempts.iter().map(|(n, j)| (*n, reading(*j))).collect(),
            resolved_by: won,
        }
    }

    #[test]
    fn a_hopeless_cheap_rung_is_skipped() {
        // The whole point. Escalation pays `local` every time; a predictor that knows it will fail
        // should route past it.
        let p = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let plan = p.plan(&[0.0, 1.0]).unwrap();
        assert_eq!(plan.start_name, "cloud");
        assert_eq!(plan.skipped, ["local"]);
        // Escalation would pay 1 + 100; the plan pays 100.
        assert!((plan.escalation_only_joules - 101.0).abs() < 1e-9);
        assert!((plan.expected_joules - 100.0).abs() < 1e-9);
        assert!((plan.expected_saving() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_cheap_rung_that_usually_works_is_kept() {
        let p = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let plan = p.plan(&[0.9, 1.0]).unwrap();
        assert_eq!(plan.start_name, "local");
        assert!(plan.skipped.is_empty());
        // 1 + 0.1*100 = 11
        assert!((plan.expected_joules - 11.0).abs() < 1e-9, "{}", plan.expected_joules);
    }

    #[test]
    fn a_rung_100x_cheaper_is_worth_trying_at_2_percent() {
        // The published shape: a very cheap rung needs to be right only rarely to pay for itself.
        let p = prof(&[("tiny", 1.0), ("big", 100.0)]);
        assert_eq!(p.plan(&[0.02, 1.0]).unwrap().start_name, "tiny");
        assert_eq!(p.plan(&[0.005, 1.0]).unwrap().start_name, "big");
    }

    #[test]
    fn the_two_rung_break_even_is_conservative_on_a_deep_ladder() {
        // ⚠ THE CORRECTION. `ladder::break_even` compares a rung against the NEXT rung. The right
        // denominator is the expected cost of the whole REMAINING ladder, which is larger, so the true
        // threshold is lower and the shipped formula refuses cheap rungs that are in fact worth trying.
        let p = prof(&[("a", 10.0), ("b", 12.0), ("c", 1000.0)]);
        let probs = [0.5, 0.1, 1.0];

        let two_rung = crate::ladder::Ladder::<u8>::break_even(10.0, 12.0);  // 0.833…
        let true_be = p.break_even(0, &probs).unwrap();
        assert!(true_be < two_rung, "true {true_be} should be below two-rung {two_rung}");

        // And the gap contains real decisions: p = 0.5 sits between them, so the two-rung rule says
        // "skip a" while the correct rule says "try a" — and the correct rule is right.
        assert!(probs[0] < two_rung, "test setup: p must be below the two-rung threshold");
        assert!(probs[0] > true_be, "test setup: p must be above the true threshold");
        let plan = p.plan(&probs).unwrap();
        assert_eq!(plan.start_name, "a", "the deep ladder makes trying the cheap rung correct");

        // Verify by direct expectation, not by trusting the planner:
        //   E(2) = 1000;  E(1) = 12 + 0.9*1000 = 912;  E(0) = 10 + 0.5*912 = 466
        let e = p.expected_costs(&probs).unwrap();
        assert!((e[2] - 1000.0).abs() < 1e-9);
        assert!((e[1] - 912.0).abs() < 1e-9);
        assert!((e[0] - 466.0).abs() < 1e-9);
        assert!(e[0] < e[1], "starting at the cheap rung must actually be cheaper here");
    }

    #[test]
    fn an_unmeasured_rung_is_refused_rather_than_treated_as_free() {
        // A rung with no measurements has unknown cost, not zero cost. Zero would make it the
        // planner's favourite and route all traffic to the thing nobody has profiled.
        let e = Profile::new(vec![RungProfile { name: "new", joules: 5.0, attempts: 0 }]).unwrap_err();
        assert_eq!(e, PlanError::Unmeasured("new"));
    }

    #[test]
    fn rungs_are_sorted_by_energy_not_by_caller_order() {
        // The decision rule assumes cheapest-first. Trusting a mis-ordered caller produces a plan that
        // looks valid and routes backwards.
        let p = prof(&[("dear", 100.0), ("cheap", 1.0)]);
        assert_eq!(p.names(), ["cheap", "dear"]);
    }

    #[test]
    fn a_nonsense_probability_is_refused() {
        let p = prof(&[("a", 1.0), ("b", 2.0)]);
        assert_eq!(p.plan(&[1.5, 1.0]).unwrap_err(), PlanError::BadProbability("a"));
        assert_eq!(p.plan(&[f64::NAN, 1.0]).unwrap_err(), PlanError::BadProbability("a"));
    }

    #[test]
    fn regret_is_the_price_of_being_wrong_and_is_never_negative() {
        let p = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let mut r = Router::new(p);
        // Predictor said local would fail; it would actually have succeeded. We skipped it and paid
        // the cloud. Oracle would have paid 1.
        r.observe(&trail(&[("cloud", 100.0)], Some("cloud")), &[(1.0, "cloud")]);
        assert!((r.oracle_joules() - 100.0).abs() < 1e-9);
        assert!(r.regret() >= 0.0, "regret {} went negative", r.regret());
    }

    #[test]
    fn skipping_a_rung_that_would_have_worked_shows_up_as_lost_energy() {
        // THE failure mode of predictive routing. Escalation would have resolved at `local` for 1 J.
        // The router skipped to `cloud` for 100 J. The accounting must show this as a LOSS, not hide
        // it behind "we avoided an escalation".
        let p = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let mut r = Router::new(p);
        for _ in 0..10 {
            r.observe(&trail(&[("cloud", 100.0)], Some("cloud")), &[(0.9, "cloud")]);
        }
        // Escalation on these outcomes pays 1 + 100 each; we paid 100 each. So against escalation we
        // "saved" 10 J — but that is only because the winner is recorded as cloud.
        assert!((r.saved_against_escalation() - 10.0).abs() < 1e-9);
        // ...and regret against the oracle is zero here because cloud genuinely resolved it.
        assert!(r.regret().abs() < 1e-9);
    }

    #[test]
    fn a_router_that_makes_things_worse_reports_a_negative_saving() {
        // Not clamped. A predictive router can lose to escalation and the number must say so.
        let p = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let mut r = Router::new(p);
        // The router burned local AND cloud, but escalation on a local win would have paid only 1.
        r.observe(&trail(&[("local", 1.0), ("cloud", 100.0)], Some("local")), &[(0.5, "local")]);
        assert!(r.saved_against_escalation() < 0.0,
                "spent {} vs escalation {}", r.joules(), r.escalation_joules());
    }

    #[test]
    fn brier_above_a_quarter_means_worse_than_saying_nothing() {
        let mut c = Calibration::default();
        for _ in 0..10 { c.record(0.9, false); }   // confidently wrong
        assert!(c.brier().unwrap() > 0.25);
        assert!(!c.beats_guessing());
        assert!(c.bias().unwrap() > 0.0, "confidently-wrong must read as OVERconfident");

        let mut good = Calibration::default();
        for _ in 0..9 { good.record(0.9, true); }
        good.record(0.9, false);
        assert!(good.brier().unwrap() < 0.25);
        assert!(good.beats_guessing());
    }

    #[test]
    fn deployment_needs_both_a_saving_and_a_calibrated_predictor() {
        // Either alone can be true while the router is useless: a saving can come from a lucky
        // workload, and calibration can be fine on a ladder where routing does not pay.
        let p = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let mut r = Router::new(p);
        for _ in 0..20 { r.observe(&trail(&[("cloud", 100.0)], Some("cloud")), &[(0.95, "cloud")]); }
        assert!(r.saved_against_escalation() > 0.0);
        assert!(r.calibration().beats_guessing());
        assert!(r.worth_deploying());

        let p2 = prof(&[("local", 1.0), ("cloud", 100.0)]);
        let mut bad = Router::new(p2);
        for _ in 0..20 { bad.observe(&trail(&[("cloud", 100.0)], Some("cloud")), &[(0.05, "cloud")]); }
        assert!(!bad.calibration().beats_guessing(), "an anti-calibrated predictor must not deploy");
        assert!(!bad.worth_deploying());
    }

    #[test]
    fn uniform_predictor_reproduces_plain_escalation() {
        // "No predictor" has to be a measurable arm, not a separate code path, or you cannot tell
        // whether the prediction is doing anything.
        let p = prof(&[("a", 1.0), ("b", 10.0), ("c", 100.0)]);
        let r = Router::new(p);
        let plan = r.plan("anything", &Uniform(0.5)).unwrap();
        assert_eq!(plan.start_name, "a", "uniform belief should start at the bottom");
        assert!(plan.skipped.is_empty());
        assert!((plan.expected_saving()).abs() < 1e-12, "escalation cannot beat itself");
    }

    #[test]
    fn an_exhausted_ladder_charges_escalation_the_whole_stack() {
        // If nothing resolved, escalation would have paid every rung. The oracle pays nothing, because
        // there was no right answer to jump to — inflating the oracle here would flatter the router.
        let p = prof(&[("a", 1.0), ("b", 10.0)]);
        let mut r = Router::new(p);
        r.observe(&trail(&[("a", 1.0), ("b", 10.0)], None), &[(0.5, "a")]);
        assert_eq!(r.resolved(), 0);
        assert!((r.escalation_joules() - 11.0).abs() < 1e-9);
        assert!((r.oracle_joules() - 0.0).abs() < 1e-9);
        assert!(r.per_success().is_nan(), "zero successes is not an efficiency result");
    }

    #[test]
    fn profiles_come_from_readings_not_from_guesses() {
        let rs = [reading(2.0), reading(4.0)];
        let rp = RungProfile::from_readings("m", &rs);
        assert_eq!(rp.attempts, 2);
        assert!((rp.joules - 3.0).abs() < 1e-9);
        // An empty set does not silently become a free rung.
        assert_eq!(RungProfile::from_readings("m", &[]).attempts, 0);
    }
}
