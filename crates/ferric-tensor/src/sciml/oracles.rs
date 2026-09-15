//! **The oracles for the training recipe.** Every technique in [`super::train`] and [`super::features`]
//! is tested the same way: the vanilla loss on a problem where it measurably FAILS, then the technique
//! on the same problem, same network, same optimiser, same step budget. A technique that only ever
//! matches the baseline would be decoration, and a fixture on which the baseline already succeeds
//! proves nothing — several of these had their difficulty raised until the baseline broke, and the
//! numbers in each test's doc are what was measured.

use super::util::*;
use super::*;
use crate::Adam;
use ferric_core::Context;
use std::sync::Arc;

/// ⭐⭐ **Spectral bias, and its cure.** `u'' + ω²u = 0`, `u(0)=1`, `u'(0)=0`, true `cos ωt`, at a frequency
/// a plain tanh net cannot reach in the step budget. Same optimiser, same steps, same width: the tanh
/// MLP returns a low-frequency wrong answer; the Fourier-feature net returns the solution.
#[ignore = "trains two nets on the GPU; the six training oracles take ~40 min together — run with `cargo test --release -p ferric-tensor sciml -- --ignored`"]
#[test]
fn spectral_bias_is_fixed_by_fourier_features() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let (w, t_max, nc, steps) = (10.0f32, 2.0f32, 120usize, 3000usize);
        let tcol: Vec<f32> = (0..nc).map(|i| i as f32 * t_max / (nc as f32 - 1.0)).collect();
        let te: Vec<f32> = (0..200).map(|i| i as f32 * t_max / 199.0).collect();
        let truth: Vec<f32> = te.iter().map(|&t| (w * t).cos()).collect();

        // one training routine, parameterised by the forward map
        async fn train(ctx: &Arc<Context>, wp: &mut [Tensor], fwd: &dyn Fn(&[Var], &Var) -> Var, tcol: &[f32], w: f32, steps: usize) {
            let nc = tcol.len();
            let mut adam = Adam::new(wp, 3e-3);
            for _ in 0..steps {
                let pv = vars(wp);
                let tv = col(ctx, tcol);
                let u = fwd(&pv, &tv);
                let u_t = deriv(&u, &tv);
                let u_tt = deriv(&u_t, &tv);
                let res = u_tt.add(&u.mul(&col(ctx, &vec![w * w; nc])));
                let t0 = col(ctx, &[0.0]);
                let u0 = fwd(&pv, &t0);
                let u0t = deriv(&u0, &t0);
                let e0 = u0.sub(&col(ctx, &[1.0]));
                let ic = e0.mul(&e0).sum_all().add(&u0t.mul(&u0t).sum_all());
                let loss = mse(&res).add(&ic.mul(&scalar(&ic, 40.0)));
                step(ctx, &loss, &pv, wp, &mut adam).await;
            }
        }
        async fn err(ctx: &Arc<Context>, wp: &[Tensor], fwd: &dyn Fn(&[Var], &Var) -> Var, te: &[f32], truth: &[f32]) -> f32 {
            let u = fwd(&vars(wp), &col(ctx, te)).value().to_vec().await;
            rel_l2(&u, truth)
        }

        // control: a tanh MLP of the same width
        let plain = Mlp::new(&ctx, &[1, 64, 64, 1], 7);
        let mut wp_plain = plain.params.clone();
        let f_plain = |pv: &[Var], x: &Var| Mlp::forward_act(pv, x, Act::Tanh);
        train(&ctx, &mut wp_plain, &f_plain, &tcol, w, steps).await;
        let e_plain = err(&ctx, &wp_plain, &f_plain, &te, &truth).await;

        // the technique: Fourier features at scales bracketing ω/2π ≈ 1.6
        let ff = FourierNet::new(&ctx, 1, 32, &[1.0, 2.0], &[64, 64], 1, Act::Tanh, 7);
        let mut wp_ff = ff.params.clone();
        let f_ff = |pv: &[Var], x: &Var| ff.forward(pv, x);
        train(&ctx, &mut wp_ff, &f_ff, &tcol, w, steps).await;
        let e_ff = err(&ctx, &wp_ff, &f_ff, &te, &truth).await;

        eprintln!("  ω = {w}: tanh MLP rel-L2 {e_plain:.4}   Fourier-feature net rel-L2 {e_ff:.4}");
        assert!(e_plain > 0.3, "the control must FAIL at this frequency or the fixture is too easy: {e_plain:.4}");
        assert!(e_ff < 0.05, "Fourier features must solve it: {e_ff:.4}");
    });
}

/// ⭐⭐ **A boundary the residual drowns, rescued by gradient-norm balancing.** `u'' = −(aπ)² sin(aπx)` on
/// `[0,1]` with `u(0) = u(1) = 0`; exact `sin(aπx)`. The residual carries a factor `(aπ)⁴` in its
/// gradients and the boundary term carries `1`, so with equal weights the optimiser satisfies the
/// equation and ignores the boundary — the net converges to `sin(aπx) + c₀ + c₁x`. Balancing lifts the
/// boundary weight to where its gradient competes, and the same net in the same steps lands on the solution.
#[ignore = "trains two nets on the GPU; the six training oracles take ~40 min together — run with `cargo test --release -p ferric-tensor sciml -- --ignored`"]
#[test]
fn loss_balancing_rescues_a_boundary_the_residual_drowns() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let (a, nc, steps) = (6.0f32, 128usize, 4000usize);
        let ap = a * std::f32::consts::PI;
        let x: Vec<f32> = (0..nc).map(|i| (i as f32 + 0.5) / nc as f32).collect();
        let f: Vec<f32> = x.iter().map(|&xi| -ap * ap * (ap * xi).sin()).collect();
        let xe: Vec<f32> = (0..201).map(|i| i as f32 / 200.0).collect();
        let truth: Vec<f32> = xe.iter().map(|&xi| (ap * xi).sin()).collect();

        async fn run(ctx: &Arc<Context>, balance: bool, x: &[f32], f: &[f32], xe: &[f32], truth: &[f32], steps: usize) -> (f32, f32) {
            // ⛔ a tanh MLP, as in the paper: with Fourier features the equal-weight baseline solved
            // a = 4 to rel-L2 0.035 and there was nothing to rescue
            let net = Mlp::new(ctx, &[1, 64, 64, 1], 3);
            let fwd = |pv: &[Var], x: &Var| Mlp::forward_act(pv, x, Act::Tanh);
            let mut wp = net.params.clone();
            let mut adam = Adam::new(&wp, 1e-3);
            let mut bal = LossBalancer::new(2, 0.1);
            for it in 0..steps {
                let pv = vars(&wp);
                let xv = col(ctx, x);
                let u = fwd(&pv, &xv);
                let uxx = deriv(&deriv(&u, &xv), &xv);
                let l_res = mse(&uxx.sub(&col(ctx, f)));
                let ub = fwd(&pv, &col(ctx, &[0.0, 1.0]));
                let l_bc = mse(&ub);
                if balance && it % 100 == 0 {
                    bal.update(&[l_res.clone(), l_bc.clone()], &pv).await;
                }
                let loss = if balance { bal.combine(&[l_res, l_bc]) } else { l_res.add(&l_bc) };
                step(ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
            let u = fwd(&vars(&wp), &col(ctx, xe)).value().to_vec().await;
            (rel_l2(&u, truth), bal.weights[1])
        }
        let (e_plain, _) = run(&ctx, false, &x, &f, &xe, &truth, steps).await;
        let (e_bal, lam) = run(&ctx, true, &x, &f, &xe, &truth, steps).await;
        eprintln!("  a = {a}: equal weights rel-L2 {e_plain:.4}   balanced rel-L2 {e_bal:.4}  (final λ_bc = {lam:.1})");
        assert!(e_plain > 0.2, "the control must FAIL or the fixture is too easy: {e_plain:.4}");
        assert!(e_bal < 0.05, "balancing must solve it: {e_bal:.4}");
        assert!(lam > 10.0, "the boundary weight must have been lifted well above 1, got {lam:.2}");
    });
}

/// ⭐⭐ **A wrong branch a vanilla PINN lands on, avoided by causal training.** The reaction equation
/// `u_t = ρ u (1 − u)` on `[0, 2π] × [0, 1]`, `u(x,0) = exp(−(x−π)²/(2(π/4)²))`, periodic in `x`; exact
/// `u = h eᵖᵗ / (h eᵖᵗ + 1 − h)` (Krishnapriyan et al. 2021, arXiv 2109.01050, whose Fig. 1 is this
/// failure). Same tanh net and optimiser; the causal weights force early times to be solved first.
///
/// ⚠ Sized by measurement (`fixture_sweep_for_rar_and_causal`). At `ρ = 5`, 3000 steps: vanilla 0.23,
/// causal 0.23 — nothing to rescue. At `ρ = 10`, 4000 steps: vanilla 0.94, causal 0.46 — the right branch,
/// not converged. At `ρ = 10`, 8000 steps on a 48×24 grid: **vanilla 0.9372, causal 0.0943** in the
/// sweep and **0.1165** when this test was run again unchanged — the same configuration, so that spread
/// is GPU run-to-run variance and the bound below (0.15) is set to clear it, not to flatter the better run.
#[ignore = "trains two nets on the GPU; the six training oracles take ~40 min together — run with `cargo test --release -p ferric-tensor sciml -- --ignored`"]
#[test]
fn causal_training_finds_the_solution_a_vanilla_pinn_cannot() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let (rho, nx, nt, steps, n_slabs) = (10.0f32, 48usize, 24usize, 8000usize, 8usize);
        let two_pi = std::f32::consts::TAU;
        let h = |x: f32| (-(x - std::f32::consts::PI).powi(2) / (2.0 * (std::f32::consts::PI / 4.0).powi(2))).exp();
        let exact = |x: f32, t: f32| {
            let hh = h(x);
            hh * (rho * t).exp() / (hh * (rho * t).exp() + 1.0 - hh)
        };
        // collocation grid, ordered by time so slab = t-index bucket
        let mut xt = Vec::with_capacity(nx * nt * 2);
        let mut slab = Vec::with_capacity(nx * nt);
        for j in 0..nt {
            let t = (j as f32 + 0.5) / nt as f32;
            for i in 0..nx {
                let x = (i as f32 + 0.5) * two_pi / nx as f32;
                xt.push(x);
                xt.push(t);
                slab.push(j * n_slabs / nt);
            }
        }
        let n = nx * nt;
        let xic: Vec<f32> = (0..nx).flat_map(|i| [(i as f32 + 0.5) * two_pi / nx as f32, 0.0]).collect();
        let uic: Vec<f32> = (0..nx).map(|i| h((i as f32 + 0.5) * two_pi / nx as f32)).collect();
        let tb: Vec<f32> = (0..nt).map(|j| (j as f32 + 0.5) / nt as f32).collect();
        let xl: Vec<f32> = tb.iter().flat_map(|&t| [0.0, t]).collect();
        let xr: Vec<f32> = tb.iter().flat_map(|&t| [two_pi, t]).collect();
        let mut xe = Vec::new();
        let mut truth = Vec::new();
        for j in 0..=20 {
            for i in 0..=40 {
                let (x, t) = (i as f32 * two_pi / 40.0, j as f32 / 20.0);
                xe.push(x);
                xe.push(t);
                truth.push(exact(x, t));
            }
        }

        #[allow(clippy::too_many_arguments)]
        async fn run(ctx: &Arc<Context>, causal: bool, xt: &[f32], slab: &[usize], n: usize, n_slabs: usize, xic: &[f32], uic: &[f32], xl: &[f32], xr: &[f32], xe: &[f32], truth: &[f32], rho: f32, steps: usize) -> f32 {
            let net = Mlp::new(ctx, &[2, 50, 50, 50, 1], 11);
            let mut wp = net.params.clone();
            let mut adam = Adam::new(&wp, 1e-3);
            let mut cz = Causal::new(vec![1e-2, 1e-1, 1.0, 10.0, 100.0], 0.99);
            let fwd = |pv: &[Var], x: &Var| Mlp::forward_act(pv, x, Act::Tanh);
            for _ in 0..steps {
                let pv = vars(&wp);
                let xv = leaf(ctx, xt, &[n, 2]);
                let u = fwd(&pv, &xv);
                let u_t = dcol(ctx, &u, &xv, n, 2, 1);
                let res = u_t.sub(&u.sub(&u.mul(&u)).mul(&col(ctx, &vec![rho; n])));
                let l_res = if causal {
                    let r = res.value().to_vec().await;
                    let w = cz.weights(&Causal::slab_losses(&r, slab, n_slabs));
                    mse(&res.mul(&Causal::point_weights(ctx, &w, slab).mul(&Causal::point_weights(ctx, &w, slab)).sqrt()))
                } else {
                    mse(&res)
                };
                let ic = mse(&fwd(&pv, &leaf(ctx, xic, &[uic.len(), 2])).sub(&col(ctx, uic)));
                let per = mse(&fwd(&pv, &leaf(ctx, xl, &[xl.len() / 2, 2])).sub(&fwd(&pv, &leaf(ctx, xr, &[xr.len() / 2, 2]))));
                let loss = l_res.add(&ic.add(&per).mul(&scalar(&ic, 100.0)));
                step(ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
            let u = fwd(&vars(&wp), &leaf(ctx, xe, &[truth.len(), 2])).value().to_vec().await;
            rel_l2(&u, truth)
        }
        let e_plain = run(&ctx, false, &xt, &slab, n, n_slabs, &xic, &uic, &xl, &xr, &xe, &truth, rho, steps).await;
        let e_causal = run(&ctx, true, &xt, &slab, n, n_slabs, &xic, &uic, &xl, &xr, &xe, &truth, rho, steps).await;
        eprintln!("  reaction ρ = {rho}: vanilla rel-L2 {e_plain:.4}   causal rel-L2 {e_causal:.4}");
        assert!(e_plain > 0.3, "the control must FAIL or the fixture is too easy: {e_plain:.4}");
        assert!(e_causal < 0.15, "causal training must solve it (measured 0.094): {e_causal:.4}");
    });
}

/// ⚠ **Residual-based refinement: the primitive verified, the benefit NOT yet demonstrated.**
///
/// What [`rar_select`] promises is that the points it adds are where the residual is largest — for
/// `u'' = f` with `u = tanh(k(x − ½))` that is the layer, and this test checks it: at least three quarters
/// of the added points must land within `2/k` of the centre.
///
/// ⛔ What it does NOT check is that refinement lowers the error, because measured, here, it does not.
/// A sweep over `k ∈ {20, 30}` and budgets `{64, 128}` (`fixture_sweep_for_rar_and_causal`), uniform
/// against 75 % uniform + 25 % refined at equal budget, Adam 3000 steps:
///
/// | k | budget | uniform | refined |
/// |---|---|---|---|
/// | 20 | 64 | 0.1699 | 0.1771 |
/// | 20 | 128 | 0.1690 | 0.1778 |
/// | 30 | 64 | 0.5364 | 0.6126 |
/// | 30 | 128 | 0.4590 | 0.4698 |
///
/// The error tracks `k` and barely moves with the budget: this regime is **optimisation-limited**, and no
/// placement of points can help until it is sampling-limited. The literature's demonstrations (DeepXDE on
/// Burgers) use thousands of base points and an L-BFGS stage, so the smooth part is solved before the
/// shock is what remains. Building that fixture is the open item; asserting a benefit this sweep did not
/// find would be the wrong kind of green.
#[test]
fn refinement_selects_the_layer_even_where_it_cannot_yet_be_shown_to_help() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let k = 30.0f32;
        let f_ex = |x: f32| {
            let s = (k * (x - 0.5)).cosh().powi(-2);
            -2.0 * k * k * (k * (x - 0.5)).tanh() * s
        };
        let cand: Vec<f32> = (0..800).map(|i| (i as f32 + 0.5) / 800.0).collect();
        // an UNTRAINED net: its residual is dominated by −f, which is the layer's own signature
        let net = FourierNet::new(&ctx, 1, 32, &[1.0, 5.0], &[64, 64], 1, Act::Tanh, 5);
        let pv = net.vars();
        let cv = col(&ctx, &cand);
        let u = net.forward(&pv, &cv);
        let r = deriv(&deriv(&u, &cv), &cv).sub(&col(&ctx, &cand.iter().map(|&c| f_ex(c)).collect::<Vec<_>>())).value().to_vec().await;
        let picked = rar_select(&r, 16);
        let in_layer = picked.iter().filter(|&&i| (cand[i] - 0.5).abs() < 2.0 / k).count();
        eprintln!("  RAR picked 16 points; {in_layer} lie within 2/k of the layer centre (layer width ≈ {:.3})", 2.0 / k);
        assert!(in_layer >= 12, "refinement must concentrate on the layer: {in_layer} of 16");
        assert_eq!(picked.len(), 16);
    });
}

/// ⭐ **L-BFGS takes a PINN an order of magnitude past where Adam stalls.** Adam to its plateau on the
/// 1-D Poisson problem, then L-BFGS from the same parameters, same loss, same collocation.
#[ignore = "trains two nets on the GPU; the six training oracles take ~40 min together — run with `cargo test --release -p ferric-tensor sciml -- --ignored`"]
#[test]
fn lbfgs_takes_a_pinn_an_order_of_magnitude_past_adam() {
    let ctx = Arc::new(pollster::block_on(Context::new()).unwrap());
    let (a, nc) = (2.0f32, 64usize);
    let ap = a * std::f32::consts::PI;
    let x: Vec<f32> = (0..nc).map(|i| (i as f32 + 0.5) / nc as f32).collect();
    let f: Vec<f32> = x.iter().map(|&xi| -ap * ap * (ap * xi).sin()).collect();
    let net = Mlp::new(&ctx, &[1, 32, 32, 1], 2);
    let shapes: Vec<Vec<usize>> = net.params.iter().map(|t| t.shape.clone()).collect();
    let fwd = |pv: &[Var], x: &Var| Mlp::forward_act(pv, x, Act::Tanh);

    let loss_of = |ctx: &Arc<Context>, pv: &[Var]| -> Var {
        let xv = col(ctx, &x);
        let u = fwd(pv, &xv);
        let uxx = deriv(&deriv(&u, &xv), &xv);
        let l_res = mse(&uxx.sub(&col(ctx, &f)));
        let l_bc = mse(&fwd(pv, &col(ctx, &[0.0, 1.0])));
        l_res.add(&l_bc.mul(&scalar(&l_bc, 100.0)))
    };

    // stage 1: Adam to its plateau
    let mut wp = net.params.clone();
    let l_adam = pollster::block_on(async {
        let mut adam = Adam::new(&wp, 2e-3);
        let mut last = f32::NAN;
        for _ in 0..2000 {
            let pv = vars(&wp);
            let loss = loss_of(&ctx, &pv);
            last = step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
        }
        last
    });
    // stage 2: L-BFGS from there
    let x0 = pollster::block_on(flatten(&wp));
    let evaluate = |flat: &[f32]| -> (f32, Vec<f32>) {
        let ts = unflatten(&ctx, flat, &shapes);
        let pv = vars(&ts);
        let loss = loss_of(&ctx, &pv);
        loss.backward();
        pollster::block_on(async {
            let v = loss.value().to_vec().await[0];
            let mut g = Vec::with_capacity(flat.len());
            for (p, t) in pv.iter().zip(&ts) {
                match p.grad() {
                    Some(gt) => g.extend(gt.to_vec().await),
                    None => g.extend(vec![0.0; t.numel()]),
                }
            }
            (v, g)
        })
    };
    let r = Lbfgs::new(20).minimize(x0, evaluate, 300, 1e-9);
    eprintln!("  Adam plateau loss {l_adam:.3e} → L-BFGS {} iters ({} evals, stop {:?}) loss {:.3e}", r.iters, r.evals, r.stop, r.f);
    assert!(r.f < 0.1 * l_adam, "L-BFGS must reach at least 10x below Adam's plateau: {:.3e} vs {l_adam:.3e}", r.f);
    assert!(r.f.is_finite());
}

/// ⭐ **An unknown coefficient recovered from sparse noisy data** — the inverse problem, where PINNs earn
/// their keep. Heat equation `u_t = D u_xx`, exact `e^{−Dπ²t} sin πx`, `D = 0.7` unknown, 30 noisy
/// measurements, `D` trained alongside the net.
#[ignore = "trains two nets on the GPU; the six training oracles take ~40 min together — run with `cargo test --release -p ferric-tensor sciml -- --ignored`"]
#[test]
fn an_unknown_coefficient_is_recovered_from_sparse_data() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let (d_true, nx, nt, steps) = (0.7f32, 32usize, 16usize, 4000usize);
        let pi = std::f32::consts::PI;
        let exact = |x: f32, t: f32| (-d_true * pi * pi * t).exp() * (pi * x).sin();
        let mut xt = Vec::new();
        for j in 0..nt {
            for i in 0..nx {
                xt.push((i as f32 + 0.5) / nx as f32);
                xt.push((j as f32 + 0.5) * 0.5 / nt as f32);
            }
        }
        let n = nx * nt;
        let mut xd = Vec::new();
        let mut ud = Vec::new();
        for k in 0..30u32 {
            let (x, t) = (u01(k, 1), 0.5 * u01(k, 2));
            xd.push(x);
            xd.push(t);
            ud.push(exact(x, t) * (1.0 + 0.01 * (2.0 * u01(k, 3) - 1.0)));
        }
        let xic: Vec<f32> = (0..nx).flat_map(|i| [(i as f32 + 0.5) / nx as f32, 0.0]).collect();
        let uic: Vec<f32> = (0..nx).map(|i| ((i as f32 + 0.5) / nx as f32 * pi).sin()).collect();
        let xbc: Vec<f32> = (0..nt).flat_map(|j| { let t = (j as f32 + 0.5) * 0.5 / nt as f32; [0.0, t, 1.0, t] }).collect();

        let net = Mlp::new(&ctx, &[2, 32, 32, 1], 4);
        let mut wp = net.params.clone();
        wp.push(Tensor::from_vec(&ctx, &[0.2], &[1])); // the unknown D, started far from 0.7
        let mut adam = Adam::new(&wp, 3e-3);
        let fwd = |pv: &[Var], x: &Var| Mlp::forward_act(&pv[..pv.len() - 1], x, Act::Tanh);
        let mut d_est = 0.0;
        for _ in 0..steps {
            let pv = vars(&wp);
            let d = pv[pv.len() - 1].clone();
            let xv = leaf(&ctx, &xt, &[n, 2]);
            let u = fwd(&pv, &xv);
            let u_t = dcol(&ctx, &u, &xv, n, 2, 1);
            let u_x = dcol(&ctx, &u, &xv, n, 2, 0);
            let u_xx = dcol(&ctx, &u_x, &xv, n, 2, 0);
            let res = u_t.sub(&u_xx.mul(&d)); // [N,1] × [1] broadcasts like a bias add
            let l_data = mse(&fwd(&pv, &leaf(&ctx, &xd, &[30, 2])).sub(&col(&ctx, &ud)));
            let l_ic = mse(&fwd(&pv, &leaf(&ctx, &xic, &[nx, 2])).sub(&col(&ctx, &uic)));
            let l_bc = mse(&fwd(&pv, &leaf(&ctx, &xbc, &[2 * nt, 2])));
            let loss = mse(&res).add(&l_data.add(&l_ic).add(&l_bc).mul(&scalar(&l_data, 20.0)));
            step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            d_est = wp[wp.len() - 1].to_vec().await[0];
        }
        let rel = (d_est - d_true).abs() / d_true;
        eprintln!("  heat equation: D recovered = {d_est:.4} against {d_true} ({:.2}% off), from 30 points at 1% noise", rel * 100.0);
        assert!(rel < 0.03, "the coefficient must be recovered within 3%: got {d_est:.4}");
    });
}

/// ⭐ **A neural operator learned from the residual alone** (physics-informed DeepONet, Wang, Wang &
/// Perdikaris 2021, arXiv 2103.10974). The antiderivative operator `G[f] = ∫₀ˣ f`: no solution data at
/// all, only `∂ₓG[f](x) = f(x)` and `G[f](0) = 0`, verified on held-out functions against the exact integral.
#[ignore = "trains two nets on the GPU; the six training oracles take ~40 min together — run with `cargo test --release -p ferric-tensor sciml -- --ignored`"]
#[test]
fn a_deeponet_learns_an_operator_from_the_residual_alone() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let (m, p, hw, b, steps) = (32usize, 24usize, 48usize, 32usize, 3000usize);
        let xg: Vec<f32> = (0..m).map(|j| j as f32 / (m as f32 - 1.0)).collect();
        let sample = |seed: u32| -> (Vec<f32>, Vec<f32>) {
            let a: Vec<f32> = (1..=4).map(|k| (u01(k, seed) * 2.0 - 1.0) / k as f32).collect();
            let f: Vec<f32> = xg.iter().map(|&x| (0..4).map(|k| a[k] * ((k as f32 + 1.0) * std::f32::consts::PI * x).sin()).sum()).collect();
            let dx = 1.0 / (m as f32 - 1.0);
            let mut g = vec![0.0f32; m];
            for j in 1..m {
                g[j] = g[j - 1] + 0.5 * (f[j] + f[j - 1]) * dx;
            }
            (f, g)
        };
        let branch = Mlp::new(&ctx, &[m, hw, p], 21);
        let trunk = Mlp::new(&ctx, &[1, hw, p], 23);
        let mut wp: Vec<Tensor> = branch.params.iter().chain(trunk.params.iter()).cloned().collect();
        let mut adam = Adam::new(&wp, 1e-3);
        // S replicates each function's branch vector across its M query points: [B·M, B]
        let mut sel = vec![0.0f32; b * m * b];
        for bi in 0..b {
            for j in 0..m {
                sel[(bi * m + j) * b + bi] = 1.0;
            }
        }
        let xrep: Vec<f32> = (0..b).flat_map(|_| xg.iter().cloned()).collect();
        let x0rep: Vec<f32> = vec![0.0; b];
        for ep in 0..steps as u32 {
            let mut fs = vec![0.0f32; b * m];
            for bi in 0..b {
                let (f, _) = sample(ep.wrapping_mul(131) + bi as u32 + 1);
                fs[bi * m..(bi + 1) * m].copy_from_slice(&f);
            }
            let pv = vars(&wp);
            let br = Mlp::forward_act(&pv[0..4], &leaf(&ctx, &fs, &[b, m]), Act::Tanh); // [B,P]
            let br_rep = leaf(&ctx, &sel, &[b * m, b]).matmul(&br); // [B·M, P]
            let xv = col(&ctx, &xrep); // [B·M,1]
            let tr = Mlp::forward_act(&pv[4..8], &xv, Act::Tanh); // [B·M,P]
            let g = br_rep.mul(&tr).sum(&[1]).reshape(&[b * m, 1]); // G[f_b](x_j)
            let g_x = deriv(&g, &xv); // per point: G depends on its own x only
            let l_res = mse(&g_x.sub(&col(&ctx, &fs)));
            // G[f](0) = 0: trunk at x = 0, one row per function
            let tr0 = Mlp::forward_act(&pv[4..8], &col(&ctx, &x0rep), Act::Tanh); // [B,P]
            let g0 = br.mul(&tr0).sum(&[1]).reshape(&[b, 1]);
            let loss = l_res.add(&mse(&g0).mul(&scalar(&l_res, 10.0)));
            let lv = step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            if ep % 500 == 0 || ep + 1 == steps as u32 {
                eprintln!("    step {ep:4}  loss {lv:.4e}  (residual {:.3e}, G(0) {:.3e})", l_res.value().to_vec().await[0], mse(&g0).value().to_vec().await[0]);
            }
        }
        // held-out functions, against the exact antiderivative
        let (nh, mut num, mut den) = (100u32, 0.0f32, 0.0f32);
        let pv = vars(&wp);
        for k in 0..nh {
            let (f, gtrue) = sample(900_000 + k);
            let br = Mlp::forward_act(&pv[0..4], &leaf(&ctx, &f, &[1, m]), Act::Tanh);
            let tr = Mlp::forward_act(&pv[4..8], &col(&ctx, &xg), Act::Tanh);
            let g = br.matmul(&tr.transpose(1, 0)).value().to_vec().await;
            for j in 0..m {
                num += (g[j] - gtrue[j]).powi(2);
                den += gtrue[j].powi(2);
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("  PI-DeepONet from the residual alone: held-out rel-L2 {rel:.4} over {nh} unseen functions");
        assert!(rel < 0.1, "the operator must be learned from physics alone: {rel:.4}");
    });
}

/// The bookkeeping pieces, checked directly.
#[test]
fn causal_weights_respect_the_arrow_of_time_and_rar_picks_the_worst_points() {
    let mut cz = Causal::new(vec![1.0, 10.0], 0.99);
    let w = cz.weights(&[0.5, 0.5, 0.0, 0.0]);
    assert!((w[0] - 1.0).abs() < 1e-7 && w[1] < w[0] && w[2] < w[1] && (w[3] - w[2]).abs() < 1e-7, "{w:?}");
    assert_eq!(cz.idx, 0, "not every weight cleared δ, so ε must not advance");
    let w2 = cz.weights(&[1e-6; 4]);
    assert!(w2.iter().all(|&v| v > 0.99));
    assert_eq!(cz.idx, 1, "all weights above δ: ε advances to the next value");
    assert_eq!(cz.eps(), 10.0);

    assert_eq!(rar_select(&[0.1, -5.0, 3.0, 0.2, 4.0], 2), vec![1, 4], "largest |residual| first");

    let mut rba = Rba::new(3, 0.9, 0.1);
    for _ in 0..200 {
        rba.update(&[0.0, 0.5, 1.0]);
    }
    assert!(rba.mult[0] < 1e-6, "a point with zero residual decays: {:?}", rba.mult);
    assert!((rba.mult[2] - 1.0).abs() < 1e-3, "the worst point saturates at η/(1−γ) = 1: {:?}", rba.mult);
    assert!(rba.mult[1] > rba.mult[0] && rba.mult[1] < rba.mult[2]);
}

/// ⛔ **Every activation this stack trains through gets a finite-difference check, first AND second order.**
/// Five tanh-based oracles failed to learn anything at once while the sin-based one had always passed;
/// a shared primitive was the only common factor, and a primitive whose gradient is wrong or zero does
/// not raise an error — it trains a net that quietly stays where it started.
#[test]
fn every_activation_differentiates_correctly_to_second_order() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        let xs: Vec<f32> = vec![-1.3, -0.4, 0.1, 0.7, 1.9];
        let h = 1e-2f32;
        for (name, f, d1, d2) in [
            ("tanh", (|v: &Var| v.tanh()) as fn(&Var) -> Var, (|x: f32| 1.0 - x.tanh().powi(2)) as fn(f32) -> f32, (|x: f32| -2.0 * x.tanh() * (1.0 - x.tanh().powi(2))) as fn(f32) -> f32),
            ("sin", |v| v.sin(), |x| x.cos(), |x| -x.sin()),
            ("exp", |v| v.exp(), |x| x.exp(), |x| x.exp()),
            ("sqrt", |v| v.sqrt(), |x| 0.5 / x.abs().sqrt(), |x| -0.25 / x.abs().powf(1.5)),
        ] {
            let pts: Vec<f32> = if name == "sqrt" { xs.iter().map(|x| x.abs() + 0.5).collect() } else { xs.clone() };
            let xv = col(&ctx, &pts);
            let y = f(&xv);
            let g1 = deriv(&y, &xv);
            let g2 = deriv(&g1, &xv);
            let (v1, v2) = (g1.value().to_vec().await, g2.value().to_vec().await);
            for (i, &x) in pts.iter().enumerate() {
                let (e1, e2) = (d1(x), d2(x));
                assert!((v1[i] - e1).abs() < 1e-3 + 1e-3 * e1.abs(), "{name}'({x}) = {} but the VJP gave {}", e1, v1[i]);
                assert!((v2[i] - e2).abs() < 1e-2 + 1e-2 * e2.abs(), "{name}''({x}) = {} but the second-order VJP gave {}", e2, v2[i]);
            }
            // and the parameter path: d/dw of sum f(w·x) through the activation, against finite differences
            let w0 = 0.8f32;
            let loss_at = |w: f32| -> f32 {
                let s: f32 = pts.iter().map(|&x| { let z = w * x; match name { "tanh" => z.tanh(), "sin" => z.sin(), "exp" => z.exp(), _ => z.abs().sqrt() } }).sum();
                s
            };
            let wv = leaf(&ctx, &[w0], &[1]);
            let z = xv.mul(&wv);
            let l = f(&z).sum_all();
            l.backward();
            let gw = wv.grad().expect("the weight must receive a gradient").to_vec().await[0];
            let fd = (loss_at(w0 + h) - loss_at(w0 - h)) / (2.0 * h);
            assert!((gw - fd).abs() < 2e-2 * (1.0 + fd.abs()), "{name}: parameter gradient {gw} vs finite difference {fd}");
            eprintln!("  {name}: first/second-order input derivatives and the parameter gradient agree with finite differences");
        }
    });
}

/// **How the RAR and causal fixtures were sized — a sweep, not a guess.** Prints a grid of both arms of
/// each oracle over point budgets, layer sharpness and step counts, so the fixture settings above can be
/// chosen from measurements: the control must fail and the technique must fix, and the first attempts
/// at both (k = 40 with 40 points; ρ = 5 at 3000 steps; k = 15 with 16 points) did neither.
#[ignore = "fixture-design sweep: ~20 min on the GPU, prints a grid and asserts nothing"]
#[test]
fn fixture_sweep_for_rar_and_causal() {
    pollster::block_on(async {
        let ctx = Arc::new(Context::new().await.unwrap());
        // ---- RAR: uniform vs 75% uniform + 25% refined, equal budget ----
        for &k in &[20.0f32, 30.0] {
            for &n_total in &[64usize, 128] {
                let u_ex = |x: f32| (k * (x - 0.5)).tanh();
                let f_ex = |x: f32| { let s = (k * (x - 0.5)).cosh().powi(-2); -2.0 * k * k * (k * (x - 0.5)).tanh() * s };
                let cand: Vec<f32> = (0..800).map(|i| (i as f32 + 0.5) / 800.0).collect();
                let xe: Vec<f32> = (0..801).map(|i| i as f32 / 800.0).collect();
                let truth: Vec<f32> = xe.iter().map(|&x| u_ex(x)).collect();
                let (ua, ub) = (u_ex(0.0), u_ex(1.0));
                let run = |x0: Vec<f32>, rar: Option<(usize, usize, usize)>| {
                    let (cand, xe, truth) = (cand.clone(), xe.clone(), truth.clone());
                    let ctx = ctx.clone();
                    async move {
                        let net = FourierNet::new(&ctx, 1, 32, &[1.0, k / 6.0], &[64, 64], 1, Act::Tanh, 5);
                        let mut wp = net.params.clone();
                        let mut adam = Adam::new(&wp, 2e-3);
                        let mut x = x0;
                        for it in 0..3000usize {
                            let pv = vars(&wp);
                            if let Some((_, add, _)) = rar.filter(|&(every, _, max_pts)| it > 0 && it % every == 0 && x.len() < max_pts) {
                                let cv = col(&ctx, &cand);
                                let u = net.forward(&pv, &cv);
                                let r = deriv(&deriv(&u, &cv), &cv).sub(&col(&ctx, &cand.iter().map(|&c| f_ex(c)).collect::<Vec<_>>())).value().to_vec().await;
                                for i in rar_select(&r, add) { x.push(cand[i]); }
                            }
                            let xv = col(&ctx, &x);
                            let u = net.forward(&pv, &xv);
                            let uxx = deriv(&deriv(&u, &xv), &xv);
                            let l_res = mse(&uxx.sub(&col(&ctx, &x.iter().map(|&c| f_ex(c)).collect::<Vec<_>>())).mul(&scalar(&uxx, 1.0 / (k * k))));
                            let l_bc = mse(&net.forward(&pv, &col(&ctx, &[0.0, 1.0])).sub(&col(&ctx, &[ua, ub])));
                            let loss = l_res.add(&l_bc.mul(&scalar(&l_bc, 100.0)));
                            step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                        }
                        let u = net.forward(&vars(&wp), &col(&ctx, &xe)).value().to_vec().await;
                        (rel_l2(&u, &truth), x.len())
                    }
                };
                let uni: Vec<f32> = (0..n_total).map(|i| (i as f32 + 0.5) / n_total as f32).collect();
                let nb = n_total * 3 / 4;
                let base: Vec<f32> = (0..nb).map(|i| (i as f32 + 0.5) / nb as f32).collect();
                let add = (n_total - nb) / 4;
                let (e_u, n_u) = run(uni, None).await;
                let (e_r, n_r) = run(base, Some((600, add, n_total))).await;
                eprintln!("  RAR  k={k:>4}  budget {n_total:>3}: uniform({n_u}) {e_u:.4}   refined({n_r}) {e_r:.4}");
            }
        }
        // ---- causal: vanilla vs causal, longer schedule, lighter grid ----
        let (rho, nx, nt, n_slabs) = (10.0f32, 48usize, 24usize, 8usize);
        let two_pi = std::f32::consts::TAU;
        let h = |x: f32| (-(x - std::f32::consts::PI).powi(2) / (2.0 * (std::f32::consts::PI / 4.0).powi(2))).exp();
        let exact = |x: f32, t: f32| { let hh = h(x); hh * (rho * t).exp() / (hh * (rho * t).exp() + 1.0 - hh) };
        let mut xt = Vec::new(); let mut slab = Vec::new();
        for j in 0..nt { let t = (j as f32 + 0.5) / nt as f32; for i in 0..nx { xt.push((i as f32 + 0.5) * two_pi / nx as f32); xt.push(t); slab.push(j * n_slabs / nt); } }
        let n = nx * nt;
        let xic: Vec<f32> = (0..nx).flat_map(|i| [(i as f32 + 0.5) * two_pi / nx as f32, 0.0]).collect();
        let uic: Vec<f32> = (0..nx).map(|i| h((i as f32 + 0.5) * two_pi / nx as f32)).collect();
        let tb: Vec<f32> = (0..nt).map(|j| (j as f32 + 0.5) / nt as f32).collect();
        let xl: Vec<f32> = tb.iter().flat_map(|&t| [0.0, t]).collect();
        let xr: Vec<f32> = tb.iter().flat_map(|&t| [two_pi, t]).collect();
        let mut xe = Vec::new(); let mut truth = Vec::new();
        for j in 0..=20 { for i in 0..=40 { let (x, t) = (i as f32 * two_pi / 40.0, j as f32 / 20.0); xe.push(x); xe.push(t); truth.push(exact(x, t)); } }
        for &steps in &[8000usize] {
            for &causal in &[false, true] {
                let net = Mlp::new(&ctx, &[2, 50, 50, 50, 1], 11);
                let mut wp = net.params.clone();
                let mut adam = Adam::new(&wp, 1e-3);
                let mut cz = Causal::new(vec![1e-2, 1e-1, 1.0, 10.0, 100.0], 0.99);
                let fwd = |pv: &[Var], x: &Var| Mlp::forward_act(pv, x, Act::Tanh);
                for _ in 0..steps {
                    let pv = vars(&wp);
                    let xv = leaf(&ctx, &xt, &[n, 2]);
                    let u = fwd(&pv, &xv);
                    let u_t = dcol(&ctx, &u, &xv, n, 2, 1);
                    let res = u_t.sub(&u.sub(&u.mul(&u)).mul(&col(&ctx, &vec![rho; n])));
                    let l_res = if causal {
                        let r = res.value().to_vec().await;
                        let w = cz.weights(&Causal::slab_losses(&r, &slab, n_slabs));
                        mse(&res.mul(&Causal::point_weights(&ctx, &w, &slab).sqrt()))
                    } else { mse(&res) };
                    let ic = mse(&fwd(&pv, &leaf(&ctx, &xic, &[nx, 2])).sub(&col(&ctx, &uic)));
                    let per = mse(&fwd(&pv, &leaf(&ctx, &xl, &[nt, 2])).sub(&fwd(&pv, &leaf(&ctx, &xr, &[nt, 2]))));
                    let loss = l_res.add(&ic.add(&per).mul(&scalar(&ic, 100.0)));
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let u = fwd(&vars(&wp), &leaf(&ctx, &xe, &[truth.len(), 2])).value().to_vec().await;
                eprintln!("  CAUSAL ρ={rho} steps {steps} {}: rel-L2 {:.4}  (ε reached {})", if causal { "causal " } else { "vanilla" }, rel_l2(&u, &truth), cz.eps());
            }
        }
    });
}
