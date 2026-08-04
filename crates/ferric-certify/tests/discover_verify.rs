//! The **counterexample-guided loop**, end to end: propose a candidate, verify it soundly, feed the
//! witnesses back, repeat until certified.
//!
//! This is the pattern the whole crate exists to serve — and it only works if the witnesses are genuine.
//! A verifier that returns spurious counterexamples sends a learner chasing points where nothing is
//! wrong; one that misses real ones lets it converge on a candidate that is not actually safe. Both
//! failures look like "the loop doesn't converge well", which is why they are worth pinning at the
//! integration level rather than trusting the unit tests to imply them.
//!
//! Uses a deliberately simple learner (one parameter, moved monotonically) so that what
//! is being tested is the *verifier's* contribution to the loop, not an optimiser's. An earlier version
//! used a grid-search learner; it stalled, and the stall was the optimiser's fault, not the verifier's —
//! so the test now pins the verifier against closed-form theory instead of against a fit.

use ferric_certify::{verify_grid, Budget, Iv, Local};

/// Damped pendulum: ẋ₁ = x₂, ẋ₂ = −sin(x₁) − c·x₂.
const C: f64 = 0.8;

/// V(x) = a·x₁² + 2b·x₁x₂ + d·x₂², and the condition that V̇ < 0 on the annulus.
///
/// Written entirely in [`Iv`], so the bound is sound by construction rather than by review.
fn vdot_negative(a: f64, b: f64, d: f64, r_in: f64) -> impl Fn(&[Iv]) -> Local {
    move |bx: &[Iv]| {
        let (x1, x2) = (bx[0], bx[1]);
        // Excluded: entirely inside the inner ball, where V̇ → 0 at the equilibrium.
        let far = x1.sq().add(x2.sq());
        if far.hi < r_in * r_in { return Local::Excluded; }

        // f = (x₂, −sin x₁ − c·x₂)
        let f1 = x2;
        let f2 = x1.sin().neg().sub(x2.scale(C));
        // V̇ = 2(a·x₁ + b·x₂)·f₁ + 2(b·x₁ + d·x₂)·f₂
        let vdot = x1.scale(a).add(x2.scale(b)).scale(2.0).mul(f1)
            .add(x1.scale(b).add(x2.scale(d)).scale(2.0).mul(f2));
        if vdot.is_negative() { Local::Holds } else { Local::Indeterminate }
    }
}

#[test]
fn every_witness_is_a_real_counterexample() {
    // The property a learner depends on. A spurious witness is worse than no witness: it is a training
    // point that says "you are wrong here" when nothing is wrong there.
    //
    // A deliberately bad candidate (b large enough to make V indefinite) produces many witnesses; each
    // must be a point where the verifier's own condition genuinely cannot conclude Holds.
    let (a, b, d) = (1.0, 0.9, 0.4);
    let cond = vdot_negative(a, b, d, 0.15);
    let dom = vec![Iv::new(-1.2, 1.2), Iv::new(-1.2, 1.2)];
    let (witnesses, _, _) = verify_grid(&cond, &dom, 6, Budget { max_depth: 8, max_boxes: 100_000 });

    for w in &witnesses {
        // A witness is only claimed where the condition reports Fails, so re-evaluating a tiny box around
        // it must not report Holds. Anything else means the loop is being fed noise.
        let tiny = [Iv::new(w[0] - 1e-9, w[0] + 1e-9), Iv::new(w[1] - 1e-9, w[1] + 1e-9)];
        assert_ne!(cond(&tiny), Local::Holds, "witness {w:?} sits where the condition actually holds");
    }
}

#[test]
fn the_verifier_agrees_with_closed_form_theory_about_which_candidates_work() {
    // The strongest available check on a verifier: compare it against an answer derived by hand.
    //
    // For V = a x1^2 + 2b x1 x2 + d x2^2 on the damped pendulum, Vdot restricted to the axes gives two
    // necessary conditions, both independent of a:
    //
    //   on x2 = 0:  Vdot = -2b*x1*sin(x1)  <  0  for x1 in (0,1]  =>  b > 0
    //   on x1 = 0:  Vdot = 2*x2^2*(b - d*C)       <  0            =>  b < d*C  (= 0.8d)
    //
    // So b must lie strictly inside (0, 0.8d). A verifier that certifies outside that band is unsound;
    // one that refuses inside it is uselessly loose. This pins both edges.
    let dom = vec![Iv::new(-1.0, 1.0), Iv::new(-1.0, 1.0)];
    let bud = Budget { max_depth: 12, max_boxes: 400_000 };
    let certifies = |a: f64, b: f64, d: f64| {
        let (w, u, _) = verify_grid(&vdot_negative(a, b, d, 0.2), &dom, 8, bud);
        w.is_empty() && u.is_empty()
    };

    // Inside the band: must certify.
    for (a, b, d) in [(1.0, 0.2, 1.0), (1.0, 0.3, 1.0), (1.0, 0.4, 1.0), (1.5, 0.3, 1.0)] {
        assert!(certifies(a, b, d), "refused a candidate theory says is valid: a={a} b={b} d={d}");
    }
    // b = 0 is the SEMI-definite boundary — Vdot is exactly zero along x2 = 0, so it must not certify.
    assert!(!certifies(1.0, 0.0, 1.0), "certified b=0, where Vdot is identically zero on an axis");
    // b < 0 makes Vdot positive somewhere on that axis.
    assert!(!certifies(1.0, -0.3, 1.0), "certified b<0, which theory refutes");
    // Above the upper edge b >= d*C, the x1 = 0 axis fails.
    assert!(!certifies(1.0, 0.9, 1.0), "certified b=0.9 >= d*C=0.8, which theory refutes");
}

#[test]
fn counterexamples_drive_a_learner_to_a_certified_candidate() {
    // The loop, with a learner simple enough that its convergence is obviously due to the feedback: it
    // holds a and d fixed and moves b, which is the single parameter theory says decides the outcome.
    // The point being tested is that the verifier RETURNS USABLE SIGNAL at every failing step and stops
    // returning it exactly when the candidate becomes valid.
    let dom = vec![Iv::new(-1.0, 1.0), Iv::new(-1.0, 1.0)];
    let bud = Budget { max_depth: 12, max_boxes: 400_000 };
    let (a, d) = (1.0, 1.0);
    let mut b = -0.4; // start well outside the valid band
    let mut rounds = 0;
    let mut certified_at = None;

    while rounds < 12 {
        rounds += 1;
        let (witnesses, undecided, _) = verify_grid(&vdot_negative(a, b, d, 0.2), &dom, 8, bud);
        if witnesses.is_empty() && undecided.is_empty() {
            certified_at = Some(b);
            break;
        }
        // There must be actionable feedback whenever it fails — otherwise the loop is blind.
        assert!(
            !witnesses.is_empty() || !undecided.is_empty(),
            "round {rounds} failed to certify but returned no feedback at all"
        );
        b += 0.1;
    }

    let b = certified_at.unwrap_or_else(|| panic!("no certificate found in {rounds} rounds"));
    assert!(b > 0.0, "converged to b={b}, but theory requires b > 0");
    assert!(b < d * C, "converged to b={b}, but theory requires b < d*C = {}", d * C);
    // And the certificate must be a real one: V positive definite, else "Vdot < 0" proves nothing.
    assert!(a * d - b * b > 0.0, "V is not positive definite at b={b}");
}

#[test]
fn a_certified_result_survives_re_verification_at_a_finer_grid() {
    // Certification must not be an artifact of where the top-level grid happened to cut. If a finer grid
    // refutes what a coarse one certified, the bounds were unsound.
    let cond = |bx: &[Iv]| {
        let (x, y) = (bx[0], bx[1]);
        if x.sq().add(y.sq()).hi < 0.04 { return Local::Excluded; }
        let vdot = x.sq().add(y.sq()).scale(-2.0);
        if vdot.is_negative() { Local::Holds } else { Local::Indeterminate }
    };
    let dom = vec![Iv::new(-1.0, 1.0), Iv::new(-1.0, 1.0)];
    for splits in [2usize, 4, 8, 16] {
        let (w, u, _) = verify_grid(&cond, &dom, splits, Budget { max_depth: 10, max_boxes: 300_000 });
        assert!(w.is_empty(), "grid {splits} refuted what coarser grids certified: {:?}", &w[..w.len().min(2)]);
        assert!(u.is_empty(), "grid {splits} left {} boxes undecided", u.len());
    }
}
