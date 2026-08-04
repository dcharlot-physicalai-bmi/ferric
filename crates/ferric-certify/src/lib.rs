//! # ferric-certify — a certificate is a proof, or it is nothing
//!
//! Branch-and-bound verification over continuous domains: prove a property holds **everywhere** in a
//! region, or return a concrete counterexample.
//!
//! This is the Institute's central claim in library form — *the certificate is the agency*. A learned
//! controller that usually works is a demo; one that comes with a machine-checkable proof of the region
//! it is safe in is a component. The difference is entirely in whether the proof is real.
//!
//! ## Why this crate exists
//!
//! The certificate is a founding concept and it existed only as example code: **27 example files**
//! referenced certification and **zero library files** implemented it, with at least four independent
//! reimplementations of interval arithmetic among them.
//!
//! All four used plain round-to-nearest `f64` arithmetic, which is **unsound** for interval bounds — it
//! can narrow an interval and so assert a tighter bound than the arithmetic justifies. Measured: summing
//! `0.1` a million times, the round-to-nearest lower bound sits 8.2e-6 *above* the correctly-rounded one.
//! In practice the margins involved were far larger than the drift, so those results were very likely
//! right — but "very likely right" is not what a certificate claims. [`Iv`] rounds outward.
//!
//! ## The shape of a result
//!
//! Verification returns one of three things, and the third is why this is useful rather than merely
//! reassuring:
//!
//! - **Certified** — the property provably holds over the whole domain.
//! - **Refuted** — a witness point where it provably fails.
//! - **Unknown** — the search hit its depth budget with boxes still undecided, and the region is
//!   reported.
//!
//! `Unknown` is deliberately not folded into either answer. A verifier that reports failure when it
//! merely ran out of budget teaches its user to distrust it; one that reports success is dangerous.
//!
//! ## When NOT to use this crate
//!
//! Naive interval arithmetic is the general tool, not the best one. Ferric's Taylor+CROWN verifier
//! (`ferric-tensor/examples/ebm_cert_verify.rs`) computes a second-order model with an exact centre
//! gradient, and is far **tighter** than replacing its arithmetic with [`Iv`] would be — converting it
//! here would be a downgrade, not a promotion.
//!
//! For a verifier like that, the right response to the rounding concern is a **measured soundness
//! margin** rather than a rewrite. That certificate was swept: it survives a margin of 1e-6 and fails at
//! 1e-5, with a converged worst bound of −3.499e-6, against an accumulated round-to-nearest error on the
//! order of 1e-13 — roughly seven orders of magnitude of headroom. Its published result stands, and it
//! now defaults to a 1e-9 margin so the robustness is structural rather than incidental.
//!
//! Use this crate when the property is expressed directly in arithmetic over a box. Use a Taylor model
//! when the bounds need to be tight enough to converge, and give it a margin.
//!
//! ## A counterexample is the useful output
//!
//! When a candidate fails, the witness is a real point in the state space where the condition breaks —
//! which is exactly what a learner needs in order to fix it. The discover-then-verify loop that Ferric's
//! examples run is: propose a candidate, verify it, feed the counterexamples back as training points,
//! repeat. That loop only works if the witness is genuine, which is another way of saying it only works
//! if the arithmetic is sound.

#![forbid(unsafe_code)]

mod interval;
pub use interval::Iv;

/// What a verification concluded.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Provably holds everywhere in the domain.
    Certified { boxes_examined: u64, max_depth_reached: u32 },
    /// Provably fails at this point.
    Refuted { witness: Vec<f64>, boxes_examined: u64 },
    /// Ran out of subdivision budget with boxes still undecided.
    ///
    /// Distinct from `Refuted` on purpose: "I could not decide" and "it is false" are different claims,
    /// and collapsing them either teaches the user to ignore failures or hides real ones.
    Unknown { undecided: Vec<Vec<Iv>>, boxes_examined: u64 },
}

impl Outcome {
    pub fn is_certified(&self) -> bool { matches!(self, Outcome::Certified { .. }) }
    pub fn witness(&self) -> Option<&[f64]> {
        match self { Outcome::Refuted { witness, .. } => Some(witness), _ => None }
    }
}

/// Per-box verdict from a condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Local {
    /// Holds throughout this box.
    Holds,
    /// Fails throughout this box — a genuine refutation, and subdividing cannot rescue it.
    Fails,
    /// The bounds are too loose to decide; subdividing may help.
    Indeterminate,
    /// Outside the region of interest (an excluded ball around an equilibrium, say). Not a failure.
    Excluded,
}

/// A property to verify over a box.
///
/// The implementation must be **sound**: `Holds` may be returned only when the property is true for every
/// point in the box, and `Fails` only when it is false for every point. When in doubt, return
/// `Indeterminate` — the search will subdivide, which costs time and never correctness.
pub trait Condition {
    fn eval(&self, region: &[Iv]) -> Local;
}

impl<F: Fn(&[Iv]) -> Local> Condition for F {
    fn eval(&self, region: &[Iv]) -> Local { self(region) }
}

/// Search limits. Present so a verifier that cannot decide says so rather than running forever.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Maximum bisection depth per box.
    pub max_depth: u32,
    /// Ceiling on total boxes examined.
    pub max_boxes: u64,
}

impl Default for Budget {
    fn default() -> Self { Self { max_depth: 12, max_boxes: 2_000_000 } }
}

/// Verify `cond` over `domain` by adaptive bisection.
///
/// Splits the widest axis, which keeps boxes from degenerating into slabs — bisecting a fixed axis makes
/// one dimension exponentially thin while another stays wide, and the wide one is usually what is
/// blocking a decision.
pub fn verify(cond: &impl Condition, domain: &[Iv], budget: Budget) -> Outcome {
    let mut stack: Vec<(Vec<Iv>, u32)> = vec![(domain.to_vec(), 0)];
    let mut examined = 0u64;
    let mut deepest = 0u32;
    let mut undecided: Vec<Vec<Iv>> = Vec::new();

    while let Some((region, depth)) = stack.pop() {
        examined += 1;
        deepest = deepest.max(depth);
        if examined > budget.max_boxes {
            undecided.push(region);
            undecided.extend(stack.into_iter().map(|(r, _)| r));
            return Outcome::Unknown { undecided, boxes_examined: examined };
        }

        match cond.eval(&region) {
            Local::Holds | Local::Excluded => continue,
            Local::Fails => {
                return Outcome::Refuted {
                    witness: region.iter().map(|iv| iv.mid()).collect(),
                    boxes_examined: examined,
                };
            }
            Local::Indeterminate => {
                if depth >= budget.max_depth {
                    undecided.push(region);
                    continue;
                }
                // Split the widest axis.
                let (ax, _) = region
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.width().total_cmp(&b.1.width()))
                    .expect("empty domain");
                let (lo, hi) = region[ax].bisect();
                let mut a = region.clone();
                let mut b = region;
                a[ax] = lo;
                b[ax] = hi;
                stack.push((a, depth + 1));
                stack.push((b, depth + 1));
            }
        }
    }

    if undecided.is_empty() {
        Outcome::Certified { boxes_examined: examined, max_depth_reached: deepest }
    } else {
        Outcome::Unknown { undecided, boxes_examined: examined }
    }
}

/// Verify over a grid of top-level boxes, collecting **every** failing region rather than stopping at
/// the first.
///
/// A learner improving a candidate wants all the counterexamples it can get per round; returning one at a
/// time makes the discover-verify loop take as many rounds as there are bad regions.
pub fn verify_grid(
    cond: &impl Condition,
    domain: &[Iv],
    splits: usize,
    budget: Budget,
) -> (Vec<Vec<f64>>, Vec<Vec<Iv>>, u64) {
    let mut witnesses = Vec::new();
    let mut undecided = Vec::new();
    let mut examined = 0u64;
    for cell in grid(domain, splits) {
        match verify(cond, &cell, budget) {
            Outcome::Certified { boxes_examined, .. } => examined += boxes_examined,
            Outcome::Refuted { witness, boxes_examined } => {
                witnesses.push(witness);
                examined += boxes_examined;
            }
            Outcome::Unknown { undecided: u, boxes_examined } => {
                undecided.extend(u);
                examined += boxes_examined;
            }
        }
    }
    (witnesses, undecided, examined)
}

/// Split `domain` into `splits^d` cells.
fn grid(domain: &[Iv], splits: usize) -> Vec<Vec<Iv>> {
    let splits = splits.max(1);
    let mut cells: Vec<Vec<Iv>> = vec![Vec::new()];
    for ax in domain {
        let step = ax.width() / splits as f64;
        let mut next = Vec::with_capacity(cells.len() * splits);
        for c in &cells {
            for k in 0..splits {
                let mut c2 = c.clone();
                let lo = ax.lo + step * k as f64;
                c2.push(Iv::new(lo, if k + 1 == splits { ax.hi } else { lo + step }));
                next.push(c2);
            }
        }
        cells = next;
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `x² + y² > 0` outside a small ball around the origin — certifiable.
    fn positive_outside_ball(r: f64) -> impl Fn(&[Iv]) -> Local {
        move |b: &[Iv]| {
            let (x, y) = (b[0], b[1]);
            // Excluded region: entirely inside the ball.
            if x.lo > -r && x.hi < r && y.lo > -r && y.hi < r { return Local::Excluded; }
            let v = x.sq().add(y.sq());
            if v.is_positive() { Local::Holds } else { Local::Indeterminate }
        }
    }

    #[test]
    fn certifies_a_true_property() {
        let dom = vec![Iv::new(-2.0, 2.0), Iv::new(-2.0, 2.0)];
        let out = verify(&positive_outside_ball(0.25), &dom, Budget::default());
        assert!(out.is_certified(), "{out:?}");
    }

    #[test]
    fn refutes_a_false_property_with_a_real_witness() {
        // x > 1 over [-2, 2] is false, and the witness must be a point where it genuinely fails —
        // otherwise a learner fed these counterexamples trains on noise.
        let cond = |b: &[Iv]| {
            let x = b[0];
            if x.lo > 1.0 { Local::Holds } else if x.hi <= 1.0 { Local::Fails } else { Local::Indeterminate }
        };
        let out = verify(&cond, &[Iv::new(-2.0, 2.0)], Budget::default());
        let w = out.witness().expect("should refute").to_vec();
        assert!(w[0] <= 1.0, "witness {w:?} does not actually violate the property");
    }

    #[test]
    fn unknown_is_not_reported_as_refuted() {
        // The distinction that makes a verifier usable. A condition that never decides must come back
        // Unknown; folding it into Refuted would teach a user to ignore failures.
        let always_unsure = |_: &[Iv]| Local::Indeterminate;
        let out = verify(&always_unsure, &[Iv::new(0.0, 1.0)], Budget { max_depth: 3, max_boxes: 1000 });
        match out {
            Outcome::Unknown { undecided, .. } => assert!(!undecided.is_empty()),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn the_box_budget_terminates_a_hopeless_search() {
        let always_unsure = |_: &[Iv]| Local::Indeterminate;
        let out = verify(&always_unsure, &[Iv::new(0.0, 1.0)], Budget { max_depth: 60, max_boxes: 500 });
        match out {
            Outcome::Unknown { boxes_examined, .. } => assert!(boxes_examined <= 501),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn a_genuinely_strict_lyapunov_function_certifies() {
        // Linear stable system: xdot = -x, ydot = -y, with V = x^2 + y^2.
        // Vdot = -2x^2 - 2y^2, strictly negative everywhere except the origin.
        let cond = |b: &[Iv]| {
            let (x, y) = (b[0], b[1]);
            if x.lo > -0.2 && x.hi < 0.2 && y.lo > -0.2 && y.hi < 0.2 { return Local::Excluded; }
            let vdot = x.sq().add(y.sq()).scale(-2.0);
            if vdot.is_negative() { Local::Holds } else { Local::Indeterminate }
        };
        let dom = vec![Iv::new(-1.0, 1.0), Iv::new(-1.0, 1.0)];
        let (w, undecided, _) = verify_grid(&cond, &dom, 8, Budget { max_depth: 12, max_boxes: 200_000 });
        assert!(w.is_empty(), "refuted a valid Lyapunov function: {:?}", &w[..w.len().min(3)]);
        assert!(undecided.is_empty(), "{} boxes undecided on an easy case", undecided.len());
    }

    #[test]
    fn the_verifier_declines_to_certify_a_merely_semi_definite_candidate() {
        // This test exists because the verifier corrected me.
        //
        // I first wrote it as "the canonical damped-pendulum Lyapunov function certifies", using
        // V = x^2 + y^2 with xdot = y, ydot = -sin(x) - c*y. It came back with 23,024 undecided boxes,
        // which I initially read as the interval bounds being too loose. They are not. Working it out:
        //
        //     Vdot = 2y(x - sin x) - 2c*y^2
        //
        // is exactly ZERO on the whole y = 0 axis, so V is negative SEMI-definite, not strictly
        // decreasing — asymptotic stability there needs LaSalle, not this V. The tool was right and the
        // premise was wrong.
        //
        // So the property under test is that it does NOT certify: a verifier that certifies a
        // semi-definite candidate as strictly decreasing is worse than no verifier.
        let c = 1.0;
        let cond = move |b: &[Iv]| {
            let (x, y) = (b[0], b[1]);
            if x.lo > -0.35 && x.hi < 0.35 && y.lo > -0.35 && y.hi < 0.35 { return Local::Excluded; }
            let vdot = y.mul(x.sub(x.sin())).scale(2.0).sub(y.sq().scale(2.0 * c));
            if vdot.is_negative() { Local::Holds } else { Local::Indeterminate }
        };
        let dom = vec![Iv::new(-1.0, 1.0), Iv::new(-1.0, 1.0)];
        let out = verify(&cond, &dom, Budget { max_depth: 10, max_boxes: 200_000 });
        assert!(!out.is_certified(), "certified a candidate that is zero along an entire axis");

        // And specifically: a box straddling y = 0 away from the origin must stay undecided, because
        // Vdot really does touch zero there.
        let on_axis = [Iv::new(0.6, 0.7), Iv::new(-0.01, 0.01)];
        assert_eq!(cond(&on_axis), Local::Indeterminate,
                   "claimed a decision on a box where Vdot is genuinely zero");
    }

    #[test]
    fn grid_cells_tile_the_domain_exactly() {
        let dom = vec![Iv::new(-1.0, 1.0), Iv::new(0.0, 4.0)];
        let cells = grid(&dom, 4);
        assert_eq!(cells.len(), 16);
        let area: f64 = cells.iter().map(|c| c[0].width() * c[1].width()).sum();
        assert!((area - 8.0).abs() < 1e-12, "cells cover {area}, domain is 8.0");
        // And the corners are exact, so no sliver of the domain goes unverified.
        assert!(cells.iter().any(|c| c[0].lo == -1.0 && c[1].lo == 0.0));
        assert!(cells.iter().any(|c| c[0].hi == 1.0 && c[1].hi == 4.0));
    }
}
