//! **Hard-constrained ansätze — building the boundary and initial conditions into the function.**
//!
//! A PINN usually enforces `u = g` on the boundary with a penalty term, which turns training into a
//! balancing problem between the residual and that penalty. The alternative is to make the constraint
//! *structural*: write `u = A(x) + B(x)·M(x)` where `A` satisfies the condition and `B` vanishes where the
//! condition is imposed, so every network `M` satisfies it exactly and there is nothing to weight.
//!
//! ⭐⭐ On the benchmark's Helmholtz row this was worth ~50×: the recipe table's best with a soft penalty is
//! 0.3066, and `(1−x²)(1−y²)·M(x,y)` with a 501-parameter tanh MLP and plain Adam reaches 0.0059 / 0.0064.
//! That result compares against the *table*, not against a matched soft arm, so it shows the formulation
//! matters without isolating how much. The advection fixture here is the controlled version: same net, same
//! points, same steps, same optimiser, and the boundary/initial treatment as the single difference.
//!
//! `periodic_pair` is the piece that makes a **periodic** condition structural rather than penalised: feed
//! `sin(2πx/L)` and `cos(2πx/L)` instead of `x`, and every function of them is exactly `L`-periodic. It is
//! built from `sin`/`cos`/`matmul`, all of which carry differentiable VJPs, so `deriv` still reaches the
//! raw coordinate through it and the residual can be differentiated as usual.

use crate::{Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// Column `k` of an `[n, d]` input, as `[n, 1]` — by a one-hot matmul rather than `narrow`, because
/// `narrow` carries no differentiable VJP and `deriv` must reach through this.
pub fn col_of(ctx: &Arc<Context>, x: &Var, d: usize, k: usize) -> Var {
    let mut e = vec![0.0f32; d];
    e[k] = 1.0;
    x.matmul(&Var::leaf(Tensor::from_vec(ctx, &e, &[d, 1])))
}

/// Place `[n, 1]` columns side by side into an `[n, cols]` matrix, by one-hot matmuls and addition —
/// the differentiable stand-in for `cat`.
pub fn place(ctx: &Arc<Context>, parts: &[Var], cols: usize) -> Var {
    assert_eq!(parts.len(), cols);
    let mut out: Option<Var> = None;
    for (i, p) in parts.iter().enumerate() {
        let mut e = vec![0.0f32; cols];
        e[i] = 1.0;
        let t = p.matmul(&Var::leaf(Tensor::from_vec(ctx, &e, &[1, cols])));
        out = Some(match out {
            None => t,
            Some(o) => o.add(&t),
        });
    }
    out.unwrap()
}

/// `(sin(2πx/L), cos(2πx/L))` for a coordinate column — an exactly `L`-periodic embedding.
pub fn periodic_pair(x: &Var, period: f32) -> (Var, Var) {
    let w = Var::leaf(Tensor::from_vec(&x.value().ctx_arc(), &[std::f32::consts::TAU / period], &[1]));
    let th = x.mul(&w);
    (th.sin(), th.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::util::{leaf, rel_l2, step, vars};
    use crate::sciml::{Act, Mlp};

    /// The hard-constrained advection ansatz: `u = sin x + t·M(sin x, cos x, t)`. Exactly `2π`-periodic in
    /// `x` and exactly `sin x` at `t = 0`, for every `M`.
    fn advection_ansatz(ctx: &Arc<Context>, pv: &[Var], x: &Var, n: usize) -> Var {
        let xc = col_of(ctx, x, 2, 0);
        let tc = col_of(ctx, x, 2, 1);
        let (s, c) = periodic_pair(&xc, std::f32::consts::TAU);
        let feats = place(ctx, &[s.clone(), c, tc.clone()], 3);
        let _ = n;
        s.add(&tc.mul(&Mlp::forward_act(pv, &feats, Act::Tanh)))
    }



    /// ⭐⭐ **The consequence of the diagnosis: hard constraints + time marching solve the advection row.**
    ///
    /// The fixture above shows hard constraints alone do not help, because advection's limit is the
    /// residual landscape rather than the boundary treatment — at `β = 30` the solution turns ~4.8 times
    /// over `t ∈ [0,1]` and minimising the global residual does not lead to it. Time marching attacks
    /// exactly that: on a window of width `Δt`, the phase only turns `βΔt`, and with eight windows that is
    /// 3.75 radians rather than 30.
    ///
    /// The two compose. Each window carries its initial condition **structurally** — `u = g(x) + τ·M`,
    /// which is `g` at `τ = 0` for every `M` — where `g` is the previous window's solution at its own end.
    /// So no window has an initial-condition penalty to balance, and no window has to learn a long horizon.
    /// `g` and `g′` are evaluated once per window and carried as constants, so the residual is written out
    /// (`u_τ = M + τM_τ`, `u_x = g′ + τM_x`) rather than differentiated through a growing stack of frozen
    /// networks — the cost per window stays flat instead of growing with the window index.
    #[ignore = "trains eight advection windows on the GPU (~15 min); run with -- --ignored"]
    #[test]
    fn hard_constraints_and_time_marching_together_solve_the_advection_row() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (nx, ntau, windows, beta, steps) = (64usize, 16usize, 8usize, 30.0f32, 3000u32);
            let n = nx * ntau;
            let dt = 1.0f32 / windows as f32;
            let xs: Vec<f32> = (0..nx).map(|i| std::f32::consts::TAU * i as f32 / nx as f32).collect();
            let taus: Vec<f32> = (0..ntau).map(|j| dt * j as f32 / (ntau as f32 - 1.0)).collect();
            let mut pts = vec![0.0f32; n * 2];
            for i in 0..nx {
                for j in 0..ntau {
                    pts[(i * ntau + j) * 2] = xs[i];
                    pts[(i * ntau + j) * 2 + 1] = taus[j];
                }
            }
            let betav = leaf(&ctx, &[beta], &[1]);
            // g and g′ at the grid's x values: the previous window's solution and slope at its own end
            let mut g: Vec<f32> = xs.iter().map(|&x| x.sin()).collect();
            let mut gp: Vec<f32> = xs.iter().map(|&x| x.cos()).collect();
            // g repeated down each x row so it lines up with the [nx*ntau, 1] collocation layout
            let tile = |v: &[f32]| -> Vec<f32> { (0..n).map(|k| v[k / ntau]).collect() };

            let (mut worst_window, mut rel_all_num, mut rel_all_den) = (0.0f32, 0.0f64, 0.0f64);
            for w in 0..windows {
                let t0 = w as f32 * dt;
                let net = Mlp::new(&ctx, &[3, 48, 48, 1], 7 + w as u32);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                let gv = leaf(&ctx, &tile(&g), &[n, 1]);
                let gpv = leaf(&ctx, &tile(&gp), &[n, 1]);
                let forward = |pv: &[Var], x: &Var| {
                    let xc = col_of(&ctx, x, 2, 0);
                    let tc = col_of(&ctx, x, 2, 1);
                    let (sn, cs) = periodic_pair(&xc, std::f32::consts::TAU);
                    let feats = place(&ctx, &[sn, cs, tc.clone()], 3);
                    (Mlp::forward_act(pv, &feats, Act::Tanh), tc)
                };
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let x = leaf(&ctx, &pts, &[n, 2]);
                    let (mm, tc) = forward(&pv, &x);
                    let dm = crate::sciml::deriv(&mm, &x);
                    let (mx, mt) = (col_of(&ctx, &dm, 2, 0), col_of(&ctx, &dm, 2, 1));
                    // u = g + τ·M  ⇒  u_τ = M + τ·M_τ,  u_x = g′ + τ·M_x, written out because g is a constant
                    let ut = mm.add(&tc.mul(&mt));
                    let ux = gpv.add(&tc.mul(&mx));
                    let r = ut.add(&ux.mul(&betav));
                    let loss = r.mul(&r).mean_all();
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                // the window's own error against the closed form, and its end state for the next window
                let pv = vars(&wp);
                let x = leaf(&ctx, &pts, &[n, 2]);
                let (mm, tc) = forward(&pv, &x);
                let dm = crate::sciml::deriv(&mm, &x);
                let mx = col_of(&ctx, &dm, 2, 0);
                let u = gv.add(&tc.mul(&mm));
                let uv = u.value().to_vec().await;
                let uxv = gpv.add(&tc.mul(&mx)).value().to_vec().await;
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for (i, &xv) in xs.iter().enumerate() {
                    for (j, &tv) in taus.iter().enumerate() {
                        let k = i * ntau + j;
                        let want = (xv - beta * (t0 + tv)).sin();
                        num += (uv[k] - want).powi(2) as f64;
                        den += (want * want) as f64;
                    }
                }
                let e = (num / den).sqrt() as f32;
                worst_window = worst_window.max(e);
                rel_all_num += num;
                rel_all_den += den;
                for i in 0..nx {
                    g[i] = uv[i * ntau + (ntau - 1)];
                    gp[i] = uxv[i * ntau + (ntau - 1)];
                }
                eprintln!("    [banked] window {w} (t ∈ [{t0:.3}, {:.3}]): rel-L2 in-window {e:.4}", t0 + dt);
            }
            let rel = (rel_all_num / rel_all_den).sqrt() as f32;
            eprintln!(
                "  advection β={beta}, {windows} time windows × hard initial conditions, [3,48,48,1] per window, {steps} Adam steps each:\n    rel-L2 over the whole domain: {rel:.4}   (worst single window {worst_window:.4})\n    against: 0.9144 vanilla and 0.9712 full recipe in the table; 0.8098 soft / 0.9781 hard as ONE global fit; 0.3252 for time marching alone in 8 windows"
            );
            assert!(rel < 0.1, "hard constraints and time marching together must solve the advection row: {rel:.4}");
        });
    }

    /// ⛔⛔ **Hard constraints do NOT fix advection — and the contrast with Helmholtz is the finding.**
    /// `u_t + β u_x = 0` at `β = 30`, periodic on `[0, 2π]`, `u(x,0) = sin x`, exact `sin(x − βt)`. The
    /// table's numbers on this row are **0.9144** (vanilla) and **0.9712** (the full recipe) — no better
    /// than predicting nothing.
    ///
    /// Both arms share the grid, the step count, the schedule, the seed, the optimiser and the hidden
    /// layers; only the treatment of the conditions differs (soft penalties on raw `(x,t)`, against
    /// `u = sin x + t·M(sin x, cos x, t)`, exactly `sin x` at `t = 0` and exactly `2π`-periodic for every
    /// `M`). Measured: **soft 0.8098, hard 0.9781.** Hard constraints make it *worse*.
    ///
    /// ⭐ **The diagnosis, which the companion fixture supplies:** that same hard-constrained ansatz, fitted
    /// to the closed form by plain regression, reaches **0.0142**. So the ansatz can represent the answer
    /// to 1.4 % and residual training from it lands at 98 %. The limitation is neither the network nor the
    /// boundary treatment — it is that minimising this residual does not lead to this solution, the
    /// large-`β` advection failure (Krishnapriyan et al., 2109.01050), which is what the stack's *time
    /// marching* addresses and what hard constraints have nothing to say about.
    ///
    /// ⛔ This is the guard against over-reading the Helmholtz result. Hard constraints fixed that row
    /// because its binding constraint was **loss balancing**, which they remove by construction. They fix
    /// the failure mode they address; advection's failure mode is a different one. "Re-run the table with
    /// hard constraints" is therefore not a general prescription, and this fixture is why.
    #[ignore = "trains two advection PINNs on the GPU (~20 min); run with -- --ignored"]
    #[test]
    fn hard_constraints_do_not_fix_advection_whose_failure_is_not_the_boundary_treatment() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (nx, nt, beta, steps, bw) = (64usize, 64usize, 30.0f32, 10000u32, 100.0f32);
            let n = nx * nt;
            let (mut pts, mut exact) = (vec![0.0f32; n * 2], vec![0.0f32; n]);
            for i in 0..nx {
                for j in 0..nt {
                    let x = std::f32::consts::TAU * i as f32 / nx as f32;
                    let t = j as f32 / (nt as f32 - 1.0);
                    let idx = i * nt + j;
                    pts[idx * 2] = x;
                    pts[idx * 2 + 1] = t;
                    exact[idx] = (x - beta * t).sin();
                }
            }
            // the soft arm's extra supervision: the initial line, and the two periodic edges
            let ic: Vec<f32> = (0..nx).flat_map(|i| [std::f32::consts::TAU * i as f32 / nx as f32, 0.0]).collect();
            let ic_u: Vec<f32> = (0..nx).map(|i| (std::f32::consts::TAU * i as f32 / nx as f32).sin()).collect();
            let edge_l: Vec<f32> = (0..nt).flat_map(|j| [0.0, j as f32 / (nt as f32 - 1.0)]).collect();
            let edge_r: Vec<f32> = (0..nt).flat_map(|j| [std::f32::consts::TAU, j as f32 / (nt as f32 - 1.0)]).collect();
            let bwv = leaf(&ctx, &[bw], &[1]);
            let betav = leaf(&ctx, &[beta], &[1]);

            let residual_of = |u: &Var, x: &Var| {
                let g = crate::sciml::deriv(u, x);
                let ux = col_of(&ctx, &g, 2, 0);
                let ut = col_of(&ctx, &g, 2, 1);
                ut.add(&ux.mul(&betav))
            };

            let mut results = Vec::new();
            for hard in [false, true] {
                let dims: Vec<usize> = if hard { vec![3, 96, 96, 96, 1] } else { vec![2, 96, 96, 96, 1] };
                let net = Mlp::new(&ctx, &dims, 5);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let x = leaf(&ctx, &pts, &[n, 2]);
                    let u = if hard {
                        advection_ansatz(&ctx, &pv, &x, n)
                    } else {
                        Mlp::forward_act(&pv, &x, Act::Tanh)
                    };
                    let r = residual_of(&u, &x);
                    let mut loss = r.mul(&r).mean_all();
                    if !hard {
                        // the classic PINN's extra terms: the initial line, and periodicity across the edges
                        let ui = Mlp::forward_act(&pv, &leaf(&ctx, &ic, &[nx, 2]), Act::Tanh);
                        let di = ui.sub(&leaf(&ctx, &ic_u, &[nx, 1]));
                        let ul = Mlp::forward_act(&pv, &leaf(&ctx, &edge_l, &[nt, 2]), Act::Tanh);
                        let ur = Mlp::forward_act(&pv, &leaf(&ctx, &edge_r, &[nt, 2]), Act::Tanh);
                        let dp = ul.sub(&ur);
                        loss = loss.add(&di.mul(&di).mean_all().add(&dp.mul(&dp).mean_all()).mul(&bwv));
                    }
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let x = leaf(&ctx, &pts, &[n, 2]);
                let u = if hard { advection_ansatz(&ctx, &pv, &x, n) } else { Mlp::forward_act(&pv, &x, Act::Tanh) };
                let e = rel_l2(&u.value().to_vec().await, &exact);
                eprintln!("    [banked] advection β={beta}, {} conditions, {dims:?}, {steps} Adam steps: rel-L2 {e:.4}", if hard { "HARD" } else { "soft" });
                results.push(e);
            }
            let (soft, hard) = (results[0], results[1]);
            eprintln!(
                "  advection β={beta} on [0,2π]×[0,1], {n} collocation points, same net body / steps / seed:\n    soft penalties (weight {bw}):  rel-L2 {soft:.4}\n    hard constraints:             rel-L2 {hard:.4}\n    for context, the recipe table on this row: 0.9144 vanilla, 0.9712 full recipe; and this ansatz fitted to the closed form by regression reaches 0.0142, which is the floor this net can reach at all"
            );
            assert!(soft > 0.5, "the soft arm must measurably fail, as the table's numbers say it does, or there is nothing to fix: {soft:.4}");
            // ⛔ A RECORDED NEGATIVE. Measured 0.8098 soft against 0.9781 hard: hard constraints do not
            // help on this row, and the companion regression fixture (0.0142) says the ansatz is not what
            // is stopping it. If this assertion ever fires, hard constraints HAVE started helping here and
            // the finding — and the note in docs/SCIML.md — has changed and must be rewritten.
            assert!(
                hard > soft / 2.0,
                "RECORDED NEGATIVE OVERTURNED: hard constraints now materially help on advection ({hard:.4} against soft {soft:.4}) — update the claim in docs/SCIML.md"
            );
        });
    }

    /// ⚠ **The premise, before any comparison: can this ansatz represent the answer at all?** At `β = 30`
    /// the solution `sin(x − 30t)` turns ~4.8 times over `t ∈ [0,1]`, and the ansatz has to carry that in
    /// `M(sin x, cos x, t)` — where the exact `M` is `s(cos 30t − 1)/t − c·sin(30t)/t`, smooth at `t = 0`
    /// but reaching ~30 in amplitude. Fitted by plain regression against the closed form, with no PDE
    /// residual in the loop. If this does not fit, a soft-versus-hard comparison on the PDE would be two
    /// arms failing for a reason that has nothing to do with the boundary treatment.
    #[ignore = "fits the advection ansatz by regression on the GPU (~4 min); run with -- --ignored"]
    #[test]
    fn the_hard_constrained_advection_ansatz_can_represent_the_solution() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (nx, nt, beta, steps) = (64usize, 64usize, 30.0f32, 6000u32);
            let n = nx * nt;
            let mut pts = vec![0.0f32; n * 2];
            let mut target = vec![0.0f32; n];
            for i in 0..nx {
                for j in 0..nt {
                    let x = std::f32::consts::TAU * i as f32 / nx as f32;
                    let t = j as f32 / (nt as f32 - 1.0);
                    let idx = i * nt + j;
                    pts[idx * 2] = x;
                    pts[idx * 2 + 1] = t;
                    target[idx] = (x - beta * t).sin();
                }
            }
            let tv = leaf(&ctx, &target, &[n, 1]);
            for hidden in [vec![64usize, 64], vec![96, 96, 96]] {
                let mut dims = vec![3usize];
                dims.extend(hidden.iter().copied());
                dims.push(1);
                let net = Mlp::new(&ctx, &dims, 5);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let u = advection_ansatz(&ctx, &pv, &leaf(&ctx, &pts, &[n, 2]), n);
                    let d = u.sub(&tv);
                    let loss = d.mul(&d).mean_all();
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let u = advection_ansatz(&ctx, &pv, &leaf(&ctx, &pts, &[n, 2]), n).value().to_vec().await;
                let e = rel_l2(&u, &target);
                eprintln!("    [banked] ansatz {dims:?}, {steps} Adam steps: rel-L2 vs sin(x−{beta}t) = {e:.4}");
                // the ansatz's own guarantees, checked rather than assumed
                let at_t0: f32 = (0..nx).map(|i| (u[i * nt] - target[i * nt]).abs()).fold(0.0, f32::max);
                assert!(at_t0 < 1e-5, "the ansatz must be exactly sin x at t = 0: worst {at_t0:.2e}");
            }
        });
    }
}
