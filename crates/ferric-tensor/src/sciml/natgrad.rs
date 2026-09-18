//! **Gauss-Newton natural-gradient steps for physics-informed nets.**
//!
//! Adam and L-BFGS descend in the *parameter* metric. For a least-squares PINN loss `½‖r(θ)‖²` the
//! natural descent direction in the metric induced by the residual map is the Gauss-Newton step
//! `δθ = (JᵀJ)⁺ Jᵀ r`, with `J = ∂r/∂θ` at the collocation points. This is the mechanism behind the
//! accuracy gap reported by Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy Natural
//! Gradient Descent* (2302.13163): a preconditioner built from the problem's own operator reaches errors
//! orders of magnitude below first-order methods on the *same* network, in tens of iterations rather than
//! tens of thousands.
//!
//! ⚠ **Named precisely.** What is implemented here is the Gauss-Newton natural gradient — the Gramian of
//! the **residual** map in `L²`. The paper's headline variant uses the PDE's **energy** (an `H¹`-type)
//! inner product instead, which gives a different Gramian; that is the follow-on, not this. Both are
//! natural gradients, in different metrics.
//!
//! The cost is honest and unavoidable: reverse mode yields one *row* of `J` per pass, so a step costs `N`
//! backward passes over a graph that already contains the residual's own second derivatives. It pays
//! because the step count collapses.

use crate::{grad, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// Solve a symmetric positive-definite system `(A + λI) x = b` in `f64` by Cholesky, in place: `a` is
/// overwritten with the factor and `b` with the solution. Returns `false` if the factorisation hits a
/// non-positive pivot, which is the caller's signal to raise `λ`.
pub fn solve_spd(a: &mut [f64], b: &mut [f64], p: usize, lambda: f64) -> bool {
    assert_eq!(a.len(), p * p);
    assert_eq!(b.len(), p);
    for i in 0..p {
        a[i * p + i] += lambda;
    }
    for j in 0..p {
        let mut d = a[j * p + j];
        for k in 0..j {
            d -= a[j * p + k] * a[j * p + k];
        }
        // NaN must be refused too, which `d <= 0.0` alone would let through
        if d.is_nan() || d <= 0.0 {
            return false;
        }
        let ljj = d.sqrt();
        a[j * p + j] = ljj;
        for i in j + 1..p {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            a[i * p + j] = s / ljj;
        }
    }
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * b[k];
        }
        b[i] = s / a[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in i + 1..p {
            s -= a[k * p + i] * b[k];
        }
        b[i] = s / a[i * p + i];
    }
    true
}

/// The residual Jacobian `J = ∂r/∂θ`, row-major `[n, p]`, where `r` is `[n, 1]`.
///
/// Reverse mode gives one row per pass: seeding the backward pass with `e_i` yields `∂r_i/∂θ`. The
/// residual graph is built once by the caller and reused for all `n` passes — `grad()` is functional and
/// does not consume it, unlike `backward()`.
pub async fn residual_jacobian(ctx: &Arc<Context>, residual: &Var, pv: &[Var]) -> (Vec<f64>, usize, usize) {
    let n = residual.value().shape.iter().product::<usize>();
    let p: usize = pv.iter().map(|v| v.value().numel()).sum();
    let mut j = vec![0.0f64; n * p];
    let mut seed = vec![0.0f32; n];
    for i in 0..n {
        seed[i] = 1.0;
        let row = grad(residual, pv, Some(&Var::leaf(Tensor::from_vec(ctx, &seed, &residual.value().shape))));
        seed[i] = 0.0;
        let mut off = 0usize;
        for g in &row {
            for v in g.value().to_vec().await {
                j[i * p + off] = v as f64;
                off += 1;
            }
        }
        debug_assert_eq!(off, p);
    }
    (j, n, p)
}

/// The Gauss-Newton natural-gradient step `δθ` for `½‖r‖²`, as one flat vector in [`super::flatten`]
/// order. `lambda` is **relative**: the Tikhonov term added is `lambda · tr(JᵀJ)/p`, so it carries no
/// units and does not have to be retuned when the residual is rescaled. Returns `None` if the Gramian
/// stays indefinite after raising the regularisation, which for a rank-deficient `J` means the caller
/// should take more collocation points than parameters.
pub async fn gauss_newton_step(ctx: &Arc<Context>, residual: &Var, pv: &[Var], lambda: f64) -> Option<Vec<f32>> {
    let (jac, n, p) = residual_jacobian(ctx, residual, pv).await;
    let rv: Vec<f64> = residual.value().to_vec().await.iter().map(|&v| v as f64).collect();
    // A = JᵀJ, g = Jᵀ r
    let mut a = vec![0.0f64; p * p];
    let mut g = vec![0.0f64; p];
    for i in 0..n {
        let row = &jac[i * p..(i + 1) * p];
        for (c, &vc) in row.iter().enumerate() {
            if vc == 0.0 {
                continue;
            }
            g[c] += vc * rv[i];
            let ac = &mut a[c * p..(c + 1) * p];
            for (d, &vd) in row.iter().enumerate() {
                ac[d] += vc * vd;
            }
        }
    }
    let scale = (0..p).map(|i| a[i * p + i]).sum::<f64>() / p as f64;
    if scale.is_nan() || scale <= 0.0 {
        return None;
    }
    let mut rel = lambda;
    for _ in 0..12 {
        let (mut ai, mut bi) = (a.clone(), g.clone());
        if solve_spd(&mut ai, &mut bi, p, rel * scale) {
            return Some(bi.iter().map(|&v| v as f32).collect());
        }
        rel *= 10.0;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::util::{leaf, rel_l2, u01};
    use crate::sciml::{deriv, Act, Mlp};

    /// The Cholesky solve against a system whose answer is known by construction, and against its own
    /// residual computed from a kept copy of the matrix — `a` is destroyed by the factorisation, so a
    /// check that reused it would be checking `L` against itself.
    #[test]
    fn the_spd_solve_returns_the_solution_it_was_given() {
        let p = 7usize;
        // A = MᵀM + 2I is symmetric positive definite for any M
        let m: Vec<f64> = (0..p * p).map(|i| u01(i as u32, 11) as f64 - 0.5).collect();
        let mut a = vec![0.0f64; p * p];
        for r in 0..p {
            for c in 0..p {
                let mut s = 0.0;
                for k in 0..p {
                    s += m[k * p + r] * m[k * p + c];
                }
                a[r * p + c] = s + if r == c { 2.0 } else { 0.0 };
            }
        }
        let want: Vec<f64> = (0..p).map(|i| u01(i as u32 + 100, 13) as f64 - 0.5).collect();
        let mut b: Vec<f64> = (0..p)
            .map(|r| (0..p).map(|c| a[r * p + c] * want[c]).sum())
            .collect();
        let kept = a.clone();
        assert!(solve_spd(&mut a, &mut b, p, 0.0), "MᵀM + 2I must factorise");
        let err = b.iter().zip(&want).map(|(x, w)| (x - w).abs()).fold(0.0f64, f64::max);
        // and independently: does it satisfy the original system?
        let resid = (0..p)
            .map(|r| ((0..p).map(|c| kept[r * p + c] * b[c]).sum::<f64>() - (0..p).map(|c| kept[r * p + c] * want[c]).sum::<f64>()).abs())
            .fold(0.0f64, f64::max);
        eprintln!("  SPD solve, p={p}: worst |x − x*| {err:.2e}, worst residual against the kept matrix {resid:.2e}");
        assert!(err < 1e-9, "the solve must recover the known solution: {err:.2e}");
        assert!(resid < 1e-9, "and satisfy the system it was given: {resid:.2e}");
        // a non-SPD matrix must be refused, not silently mis-solved
        let mut bad = vec![0.0f64; 4];
        bad[0] = 1.0;
        bad[3] = -1.0; // diag(1, −1) as a 2×2
        let mut rhs = vec![1.0f64, 1.0];
        assert!(!solve_spd(&mut bad, &mut rhs, 2, 0.0), "an indefinite matrix must be refused");
    }


    /// ⭐⭐ **The same network, the same collocation points, the same problem — and Gauss-Newton reaches
    /// an accuracy Adam does not, in tens of steps rather than tens of thousands.** 1-D Poisson
    /// `−u'' = f` on `(0,1)` with `u* = sin(πx) + ½sin(3πx)`, boundary conditions hard-constrained by
    /// `x(1−x)` so the loss is purely the interior residual and both arms optimise exactly the same
    /// objective. The only difference is the metric the step is taken in.
    ///
    /// The Gauss-Newton arm takes a short Adam warm-up first: from a random initialisation the Gramian is
    /// near-singular and the step is meaningless, which is a real property of the method and not a
    /// weakness of this fixture.
    #[ignore = "trains two arms of a 1-D PINN on the GPU (~13 min); run with -- --ignored"]
    #[test]
    fn gauss_newton_reaches_an_accuracy_adam_does_not_on_the_same_net() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let pi = std::f32::consts::PI;
            let (nc, warm, gn_steps, adam_steps) = (300usize, 1500u32, 30usize, 20000u32);
            // interior collocation points, and the closed-form solution and forcing
            let xs: Vec<f32> = (0..nc).map(|i| (i as f32 + 0.5) / nc as f32).collect();
            let fv: Vec<f32> = xs.iter().map(|&x| pi * pi * (pi * x).sin() + 4.5 * pi * pi * (3.0 * pi * x).sin()).collect();
            let ustar: Vec<f32> = xs.iter().map(|&x| (pi * x).sin() + 0.5 * (3.0 * pi * x).sin()).collect();
            let fvar = leaf(&ctx, &fv, &[nc, 1]);
            let ones = leaf(&ctx, &vec![1.0f32; nc], &[nc, 1]);

            let net = Mlp::new(&ctx, &[1, 12, 12, 1], 3);
            let shapes: Vec<Vec<usize>> = net.params.iter().map(|t| t.shape.clone()).collect();
            // u = x(1−x)·M(x): the boundary conditions are exact, so the loss is the interior residual alone
            let build = |wp: &[Tensor]| {
                let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
                let x = leaf(&ctx, &xs, &[nc, 1]);
                let u = x.mul(&ones.sub(&x)).mul(&Mlp::forward_act(&pv, &x, Act::Tanh));
                let r = deriv(&deriv(&u, &x), &x).add(&fvar); // −u'' = f  ⇒  u'' + f = 0
                (pv, u, r)
            };
            let err_of = |wp: &[Tensor]| {
                let (_, u, _) = build(wp);
                u
            };

            // ---- arm A: Adam alone, given far more steps ----
            let mut wa = net.params.clone();
            let mut adam = crate::Adam::new(&wa, 3e-3);
            for ep in 0..adam_steps {
                if ep == adam_steps * 3 / 4 {
                    adam = crate::Adam::new(&wa, 3e-4);
                }
                let (pv, _, r) = build(&wa);
                let loss = r.mul(&r).mean_all();
                crate::sciml::util::step(&ctx, &loss, &pv, &mut wa, &mut adam).await;
            }
            let rel_adam = rel_l2(&err_of(&wa).value().to_vec().await, &ustar);

            // ---- arm B: the same Adam for `warm` steps, then Gauss-Newton ----
            let mut wb = net.params.clone();
            let mut adam = crate::Adam::new(&wb, 3e-3);
            for _ in 0..warm {
                let (pv, _, r) = build(&wb);
                let loss = r.mul(&r).mean_all();
                crate::sciml::util::step(&ctx, &loss, &pv, &mut wb, &mut adam).await;
            }
            let rel_warm = rel_l2(&err_of(&wb).value().to_vec().await, &ustar);
            let sq = |v: &[f32]| v.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>();
            let mut lambda = 1e-8f64;
            let t0 = std::time::Instant::now();
            for it in 0..gn_steps {
                let flat = crate::sciml::flatten(&wb).await;
                let (pv, _, r) = build(&wb);
                let f_now = sq(&r.value().to_vec().await);
                let Some(delta) = gauss_newton_step(&ctx, &r, &pv, lambda).await else {
                    eprintln!("    step {it}: the Gramian stayed indefinite; stopping");
                    break;
                };
                // backtracking on the same objective the step was derived from
                let mut accepted = false;
                let mut alpha = 1.0f32;
                for _ in 0..12 {
                    let trial: Vec<f32> = flat.iter().zip(&delta).map(|(&w, &d)| w - alpha * d).collect();
                    let cand = crate::sciml::unflatten(&ctx, &trial, &shapes);
                    let (_, _, rc) = build(&cand);
                    if sq(&rc.value().to_vec().await) < f_now {
                        wb = cand;
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if !accepted {
                    lambda *= 10.0;
                    if lambda > 1.0 {
                        eprintln!("    step {it}: no step reduced the residual even at λ = {lambda:.0e}; stopping");
                        break;
                    }
                }
            }
            let secs = t0.elapsed().as_secs_f64();
            let rel_gn = rel_l2(&err_of(&wb).value().to_vec().await, &ustar);
            eprintln!(
                "  1-D Poisson, [1,12,12,1] tanh ({} params), {nc} collocation points:\n    Adam {adam_steps} steps:                      rel-L2 {rel_adam:.3e}\n    Adam {warm} steps:                       rel-L2 {rel_warm:.3e}\n    + {gn_steps} Gauss-Newton steps ({:.0} s):     rel-L2 {rel_gn:.3e}   ({:.0}x better than Adam)",
                net.params.iter().map(|t| t.numel()).sum::<usize>(),
                secs,
                rel_adam / rel_gn
            );
            // ⛔ The premise is NOT "Adam is bad" — it reaches 1.4e-4 here, which is a respectable PINN
            // accuracy. The premise is that Adam has PLATEAUED: 13x more steps (1500 → 20000) bought only
            // 1.7x, so the remaining error is not a step-count problem and cannot be optimised away by
            // running the same method longer. That is what makes the 119x from 30 Gauss-Newton steps a
            // statement about the metric rather than about budget. (A first version asserted
            // `rel_adam > 1e-3` and failed on Adam doing well — the wrong thing to require.)
            assert!(
                rel_adam > rel_warm / 5.0,
                "Adam must have plateaued, or its arm is merely undertrained and this ranks budgets: {rel_warm:.3e} at {warm} steps vs {rel_adam:.3e} at {adam_steps}"
            );
            assert!(rel_gn < rel_adam / 10.0, "Gauss-Newton must beat converged Adam by at least an order of magnitude: {rel_gn:.3e} vs {rel_adam:.3e}");
            assert!(rel_gn < rel_warm / 10.0, "and it must be the Gauss-Newton steps doing it, not the warm-up: {rel_gn:.3e} vs {rel_warm:.3e}");
        });
    }

    /// ⭐ **The residual Jacobian against central differences in every parameter.** Each row comes from a
    /// seeded reverse pass through a graph that already contains the residual's own second derivative, so
    /// a mistake here is a plausible-looking Gauss-Newton step that descends the wrong direction. The
    /// oracle perturbs the parameters themselves and re-evaluates the residual — it shares nothing with
    /// the code under test but the forward pass.
    #[test]
    fn the_residual_jacobian_matches_finite_differences_in_every_parameter() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let net = Mlp::new(&ctx, &[1, 4, 1], 5);
            let shapes: Vec<Vec<usize>> = net.params.iter().map(|t| t.shape.clone()).collect();
            let xs: Vec<f32> = vec![0.15, 0.4, 0.62, 0.88];
            let n = xs.len();
            // r = u'' − u, which needs the second derivative of the net in x
            let resid = |flat: &[f32]| {
                let wp = crate::sciml::unflatten(&ctx, flat, &shapes);
                let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
                let x = leaf(&ctx, &xs, &[n, 1]);
                let u = Mlp::forward_act(&pv, &x, Act::Tanh);
                let uxx = deriv(&deriv(&u, &x), &x);
                (pv, uxx.sub(&u))
            };
            let flat0 = crate::sciml::flatten(&net.params).await;
            let (pv, r) = resid(&flat0);
            let (jac, nn, p) = residual_jacobian(&ctx, &r, &pv).await;
            assert_eq!((nn, p), (n, flat0.len()));
            let eps = 1e-3f32;
            let mut worst = 0.0f32;
            let mut worst_at = (0usize, 0usize);
            for c in 0..p {
                let (mut up, mut dn) = (flat0.clone(), flat0.clone());
                up[c] += eps;
                dn[c] -= eps;
                let hi = resid(&up).1.value().to_vec().await;
                let lo = resid(&dn).1.value().to_vec().await;
                for i in 0..n {
                    let fd = (hi[i] - lo[i]) / (2.0 * eps);
                    let got = jac[i * p + c] as f32;
                    let e = (fd - got).abs();
                    assert!(e < 3e-2 * (1.0 + fd.abs()), "J[{i}][{c}] = {got} but finite differences say {fd}");
                    if e > worst {
                        worst = e;
                        worst_at = (i, c);
                    }
                }
            }
            eprintln!("  residual Jacobian {n}×{p} through a second derivative: worst |J − finite difference| {worst:.2e} at {worst_at:?}");
        });
    }
}
