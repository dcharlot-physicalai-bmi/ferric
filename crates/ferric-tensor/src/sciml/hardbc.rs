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


    /// The harness hook itself, on a toy problem: the wrapper is applied to the network's output, and a
    /// problem that supplies one is **refused** by time marching rather than silently solved wrong.
    #[test]
    fn the_harness_applies_a_hard_constraint_and_refuses_to_march_one() {
        pollster::block_on(async {
            use crate::sciml::harness::{run_time_marched, Problem, Recipe};
            struct Toy;
            impl Problem for Toy {
                fn name(&self) -> &str { "toy" }
                fn dim(&self) -> usize { 2 }
                fn lo(&self) -> Vec<f64> { vec![-1.0, 0.0] }
                fn hi(&self) -> Vec<f64> { vec![1.0, 1.0] }
                fn residual(&self, _c: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, _n: usize) -> Var { fwd(x) }
                fn boundary_constraints(&self, _c: &Arc<Context>, _f: &dyn Fn(&Var) -> Var, _n: usize, _s: u32) -> Vec<Var> { vec![] }
                fn time_axis(&self) -> Option<usize> { Some(1) }
                fn reference(&self, _x: &[f64]) -> f64 { 0.0 }
                fn hard_constraint(&self, ctx: &Arc<Context>, x: &Var, raw: &Var) -> Option<Var> {
                    let xc = col_of(ctx, x, 2, 0);
                    let one = Var::leaf(Tensor::from_vec(ctx, &[1.0f32], &[1]));
                    Some(one.sub(&xc.mul(&xc)).mul(raw))
                }
            }
            let ctx = Arc::new(Context::new().await.unwrap());
            let t = Toy;
            // the wrapper multiplies by (1−x²), so the constrained field is exactly zero at x = ±1
            let x = leaf(&ctx, &[1.0, 0.5, 0.0, 0.5, -1.0, 0.5], &[3, 2]);
            let raw = leaf(&ctx, &[7.0, 7.0, 7.0], &[3, 1]);
            let u = t.hard_constraint(&ctx, &x, &raw).expect("Toy supplies one").value().to_vec().await;
            assert!(u[0].abs() < 1e-6 && u[2].abs() < 1e-6, "must vanish at x = ±1: {u:?}");
            assert!((u[1] - 7.0).abs() < 1e-5, "and be the raw output where the factor is 1: {u:?}");
            // ⛔ and marching must refuse it: the constraint fixes t = 0, which window two must not satisfy
            let colloc: Vec<f32> = (0..64).flat_map(|i| [(i % 8) as f32 / 8.0 - 0.5, (i / 8) as f32 / 8.0]).collect();
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_time_marched(&ctx, &t, &Recipe::vanilla(), &colloc, 2, 4, 1)
            }));
            assert!(caught.is_err(), "run_time_marched must refuse a problem with a stationary hard constraint");
        });
    }




    /// The **marched** hard-constraint hook, on a toy problem: window zero takes the problem's own
    /// `initial_value`, later windows take the previous window's *solution* (recursively, not its raw
    /// network), and the wrapper is exact at each window's start time.
    ///
    /// ⛔ A first version of `run_time_marched`'s wiring evaluated the previous window's RAW network as the
    /// start state. Every assertion still passed and advection scored **123.5** — two orders of magnitude
    /// worse than predicting nothing. This fixture is the cheap check that would have caught it: the
    /// constrained field at `t = t₀` must equal the start state exactly, which a raw-network start state
    /// does not.
    #[test]
    fn the_harness_can_march_a_hard_constrained_problem() {
        pollster::block_on(async {
            use crate::sciml::harness::Problem;
            struct Toy;
            impl Problem for Toy {
                fn name(&self) -> &str { "toy-marched" }
                fn dim(&self) -> usize { 2 }
                fn lo(&self) -> Vec<f64> { vec![0.0, 0.0] }
                fn hi(&self) -> Vec<f64> { vec![1.0, 1.0] }
                fn residual(&self, _c: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, _n: usize) -> Var { fwd(x) }
                fn boundary_constraints(&self, _c: &Arc<Context>, _f: &dyn Fn(&Var) -> Var, _n: usize, _s: u32) -> Vec<Var> { vec![] }
                fn time_axis(&self) -> Option<usize> { Some(1) }
                fn reference(&self, _x: &[f64]) -> f64 { 0.0 }
                fn initial_value(&self, ctx: &Arc<Context>, x: &Var) -> Option<Var> {
                    Some(col_of(ctx, x, 2, 0).sin())
                }
                fn marched_constraint(&self, ctx: &Arc<Context>, x: &Var, t0: f64, start: &dyn Fn(&Var) -> Var, raw: &Var) -> Option<Var> {
                    let xc = col_of(ctx, x, 2, 0);
                    let tc = col_of(ctx, x, 2, 1);
                    let t0v = Var::leaf(Tensor::from_vec(ctx, &[t0 as f32], &[1]));
                    let at_start = place(ctx, &[xc, tc.sub(&tc).add(&t0v)], 2);
                    Some(start(&at_start).add(&tc.sub(&t0v).mul(raw)))
                }
            }
            let ctx = Arc::new(Context::new().await.unwrap());
            let t = Toy;
            // at t = t₀ the wrapper must be EXACTLY the start state, whatever the raw output is
            let t0 = 0.25f64;
            let xs = vec![0.3f32, t0 as f32, 0.7, t0 as f32, 1.1, t0 as f32];
            let x = leaf(&ctx, &xs, &[3, 2]);
            let raw = leaf(&ctx, &[9.0, -4.0, 2.5], &[3, 1]);
            let start = |xx: &Var| col_of(&ctx, xx, 2, 0).mul(&Var::leaf(Tensor::from_vec(&ctx, &[2.0f32], &[1])));
            let u = t.marched_constraint(&ctx, &x, t0, &start, &raw).expect("Toy supplies one").value().to_vec().await;
            for (i, &xv) in [0.3f32, 0.7, 1.1].iter().enumerate() {
                assert!((u[i] - 2.0 * xv).abs() < 1e-5, "at t = t₀ the field must be the start state: got {} want {}", u[i], 2.0 * xv);
            }
            // and away from t₀ it must move with the raw output
            let x2 = leaf(&ctx, &[0.3f32, 0.75, 0.7, 0.75], &[2, 2]);
            let raw2 = leaf(&ctx, &[1.0f32, 1.0], &[2, 1]);
            let u2 = t.marched_constraint(&ctx, &x2, t0, &start, &raw2).expect("some").value().to_vec().await;
            assert!((u2[0] - (2.0 * 0.3 + 0.5)).abs() < 1e-5, "u = g + (t−t₀)·raw: got {}", u2[0]);
            eprintln!("  marched hook: exact at t₀ on 3 points, and g + (t−t₀)·raw away from it");
        });
    }

    /// ⛔ **Reproducing the number that says enabling the hook under the harness's own recipes is worse.**
    /// `docs/SCIML.md` quotes vanilla 6.3903 and full 0.8154 for helmholtz with `(1−x²)(1−y²)·M` turned on.
    /// That measurement was originally a one-off — the hook was enabled on `Helmholtz`, run, and removed —
    /// which left a load-bearing figure in the docs with no fixture behind it. This is the fixture: a
    /// test-local wrapper that delegates every part of the problem and adds only the constraint.
    #[ignore = "two harness trainings on the GPU (~25 min); run with -- --ignored"]
    #[test]
    fn enabling_the_hard_constraint_under_the_harness_recipes_is_worse_on_helmholtz() {
        pollster::block_on(async {
            use crate::sciml::harness::{run, Helmholtz, Problem, Recipe};
            use crate::sciml::util::box_points;
            struct Hard(Helmholtz);
            impl Problem for Hard {
                fn name(&self) -> &str { "helmholtz2d-hard" }
                fn dim(&self) -> usize { self.0.dim() }
                fn lo(&self) -> Vec<f64> { self.0.lo() }
                fn hi(&self) -> Vec<f64> { self.0.hi() }
                fn residual(&self, c: &Arc<Context>, f: &dyn Fn(&Var) -> Var, x: &Var, n: usize) -> Var { self.0.residual(c, f, x, n) }
                fn boundary_constraints(&self, c: &Arc<Context>, f: &dyn Fn(&Var) -> Var, n: usize, s: u32) -> Vec<Var> { self.0.boundary_constraints(c, f, n, s) }
                fn reference(&self, x: &[f64]) -> f64 { self.0.reference(x) }
                fn scales(&self) -> Vec<Vec<f32>> { self.0.scales() }
                fn hard_constraint(&self, ctx: &Arc<Context>, x: &Var, raw: &Var) -> Option<Var> {
                    let (xc, yc) = (col_of(ctx, x, 2, 0), col_of(ctx, x, 2, 1));
                    let one = Var::leaf(Tensor::from_vec(ctx, &[1.0f32], &[1]));
                    Some(one.sub(&xc.mul(&xc)).mul(&one.sub(&yc.mul(&yc))).mul(raw))
                }
            }
            let ctx = Arc::new(Context::new().await.unwrap());
            let p = Hard(Helmholtz { a1: 1.0, a2: 4.0, k: 1.0 });
            let colloc = box_points(&p.lo(), &p.hi(), 2000, 7);
            let v = run(&ctx, &p, &Recipe::vanilla(), &colloc, 51, 1);
            eprintln!("    [banked] harness + hard constraint, vanilla: rel-L2 {:.4} (soft was 0.4766)", v.rel_l2);
            let f = run(&ctx, &p, &Recipe::full(), &colloc, 51, 1);
            eprintln!("    [banked] harness + hard constraint, full:    rel-L2 {:.4} (soft was 0.3066)", f.rel_l2);
            eprintln!("  helmholtz with the hook on, under the harness's own recipes and 2000 RANDOM points:\n    vanilla {:.4} against 0.4766 soft;  full {:.4} against 0.3066 soft", v.rel_l2, f.rel_l2);
            // ⛔ A RECORDED NEGATIVE, as in the advection fixture: if this ever fires, enabling the hook
            // under these recipes has started helping and docs/SCIML.md must be rewritten.
            assert!(
                v.rel_l2 > 0.4766 && f.rel_l2 > 0.3066,
                "RECORDED NEGATIVE OVERTURNED: the hook now helps under the harness recipes (vanilla {:.4}, full {:.4}) — update docs/SCIML.md",
                v.rel_l2, f.rel_l2
            );
        });
    }

    /// ⭐ **The matched helmholtz comparison the row never had.** Advection and burgers were settled with
    /// arms that shared everything but the condition treatment; helmholtz was only ever compared against
    /// the *recipe table*, which differs in net, point count and formulation at once. Then the harness hook
    /// measured the same constraint as **worse** under the harness's recipes (vanilla 6.3903, full 0.8154),
    /// which makes "hard constraints fixed helmholtz" an attribution resting on nothing controlled.
    ///
    /// So: same `[2,20,20,1]` tanh net, same 900 cell-centred interior points, same 10,000 Adam steps and
    /// schedule, same seed. Only the boundary treatment differs — a penalty on the four edges, against
    /// `(1−x²)(1−y²)·M` which is exactly zero on all of them.
    #[ignore = "trains two helmholtz PINNs on the GPU (~30 min); run with -- --ignored"]
    #[test]
    fn soft_against_hard_conditions_on_the_helmholtz_row() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (m, steps, bw) = (30usize, 10000u32, 100.0f32);
            let (a1, a2, k) = (1.0f64, 4.0, 1.0);
            let n = m * m;
            let (mut pts, mut qv, mut ustar) = (vec![0.0f32; n * 2], vec![0.0f32; n], vec![0.0f32; n]);
            for i in 0..m {
                for j in 0..m {
                    let x = -1.0 + 2.0 * (i as f64 + 0.5) / m as f64;
                    let y = -1.0 + 2.0 * (j as f64 + 0.5) / m as f64;
                    let (u, q) = crate::sciml::bench::helmholtz(x, y, a1, a2, k);
                    let idx = i * m + j;
                    pts[idx * 2] = x as f32;
                    pts[idx * 2 + 1] = y as f32;
                    qv[idx] = q as f32;
                    ustar[idx] = u as f32;
                }
            }
            // the soft arm's boundary supervision: the four edges of [−1,1]²
            let mut edges = Vec::new();
            for i in 0..m {
                let v = -1.0 + 2.0 * (i as f32 + 0.5) / m as f32;
                edges.extend([-1.0, v, 1.0, v, v, -1.0, v, 1.0]);
            }
            let nb = edges.len() / 2;
            let qvar = leaf(&ctx, &qv, &[n, 1]);
            let ones = leaf(&ctx, &vec![1.0f32; n], &[n, 1]);
            let kk = leaf(&ctx, &[(k * k) as f32], &[1]);
            let bwv = leaf(&ctx, &[bw], &[1]);

            let mut results = Vec::new();
            for hard in [false, true] {
                let net = Mlp::new(&ctx, &[2, 20, 20, 1], 11);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                let field = |pv: &[Var], x: &Var| -> Var {
                    let raw = Mlp::forward_act(pv, x, Act::Tanh);
                    if !hard {
                        return raw;
                    }
                    let (xc, yc) = (col_of(&ctx, x, 2, 0), col_of(&ctx, x, 2, 1));
                    let onec = Var::leaf(Tensor::from_vec(&ctx, &[1.0f32], &[1]));
                    onec.sub(&xc.mul(&xc)).mul(&onec.sub(&yc.mul(&yc))).mul(&raw)
                };
                let resid = |pv: &[Var]| -> Var {
                    let x = leaf(&ctx, &pts, &[n, 2]);
                    let u = field(pv, &x);
                    let g = crate::sciml::deriv(&u, &x);
                    let mut lap: Option<Var> = None;
                    for c in 0..2 {
                        let gc = col_of(&ctx, &g, 2, c);
                        let t = col_of(&ctx, &crate::sciml::deriv(&gc, &x), 2, c);
                        lap = Some(match lap { None => t, Some(l) => l.add(&t) });
                    }
                    lap.unwrap().add(&u.mul(&kk)).sub(&qvar)
                };
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let r = resid(&pv);
                    let mut loss = r.mul(&r).mean_all();
                    if !hard {
                        let ub = field(&pv, &leaf(&ctx, &edges, &[nb, 2]));
                        loss = loss.add(&ub.mul(&ub).mean_all().mul(&bwv));
                    }
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let u = field(&pv, &leaf(&ctx, &pts, &[n, 2])).value().to_vec().await;
                let e = rel_l2(&u, &ustar);
                eprintln!("    [banked] helmholtz a=(1,4) k=1, {} conditions, {steps} Adam steps: rel-L2 {e:.4}", if hard { "HARD" } else { "soft" });
                let _ = &ones;
                results.push(e);
            }
            let (soft, hard) = (results[0], results[1]);
            eprintln!(
                "  helmholtz a=(1,4) k=1 on [−1,1]², {n} points, [2,20,20,1] tanh, {steps} Adam steps, same net/points/steps/seed:\n    soft boundary penalty (weight {bw}):  rel-L2 {soft:.4}\n    hard boundary constraint:             rel-L2 {hard:.4}\n    for context: 0.3066 is the recipe table's best on this row, with a different net and 2000 random points"
            );
            assert!(soft.is_finite() && hard.is_finite(), "both arms must produce a number: soft {soft:.4} hard {hard:.4}");
        });
    }

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




    /// The hard-constrained Burgers ansatz: `u = −sin(πx) + t(1−x²)·M(x,t)`. Exactly `−sin(πx)` at `t = 0`
    /// and exactly zero at `x = ±1`, for every `M` — both conditions structural.
    fn burgers_ansatz(ctx: &Arc<Context>, pv: &[Var], x: &Var) -> Var {
        let xc = col_of(ctx, x, 2, 0);
        let tc = col_of(ctx, x, 2, 1);
        let pi = Var::leaf(Tensor::from_vec(&x.value().ctx_arc(), &[std::f32::consts::PI], &[1]));
        let one = Var::leaf(Tensor::from_vec(&x.value().ctx_arc(), &[1.0f32], &[1]));
        let ic = xc.mul(&pi).sin().neg();
        let bc = one.sub(&xc.mul(&xc));
        ic.add(&tc.mul(&bc).mul(&Mlp::forward_act(pv, x, Act::Tanh)))
    }

    /// ⚠ **Diagnosing the burgers row before building anything for it — and the suspect was wrong.** At
    /// `ν = 0.01/π ≈ 0.0032` the solution steepens into a shock at `x = 0`, so spectral bias looked like
    /// the obvious limit. It is not: fitted to the Cole–Hopf reference by plain regression with no PDE
    /// residual in the loop, the ansatz reaches **0.0050**, and a much larger net reaches 0.0056 — no
    /// better. The shock is genuinely steep on this grid (the reference jumps **0.615** between
    /// neighbouring `x` at `t = 1`) and the network represents it anyway.
    ///
    /// So representation is not what limits this row, and the table's 0.2079 is ~40× worse than what the
    /// same ansatz can express. That leaves the two failure modes the other rows turned on — loss
    /// balancing, or the residual landscape — which the companion fixture separates.
    ///
    /// ⚠ The grid resolves the shock only marginally: `Δx = 2/256 = 0.0078` against a width of ~0.0032, so
    /// these numbers measure the ansatz on *this sampling*, which is the same sampling the PDE arms would
    /// use. A finer grid is a different question and is not answered here.
    #[ignore = "fits the burgers ansatz by regression on the GPU (~6 min); run with -- --ignored"]
    #[test]
    fn representation_is_not_what_limits_the_burgers_row() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (nx, nt, steps) = (256usize, 64usize, 6000u32);
            let nu = 0.01 / std::f64::consts::PI;
            let n = nx * nt;
            let (mut pts, mut target) = (vec![0.0f32; n * 2], vec![0.0f32; n]);
            for i in 0..nx {
                for j in 0..nt {
                    let x = -1.0 + 2.0 * (i as f64 + 0.5) / nx as f64;
                    let t = j as f64 / (nt as f64 - 1.0);
                    let k = i * nt + j;
                    pts[k * 2] = x as f32;
                    pts[k * 2 + 1] = t as f32;
                    target[k] = crate::sciml::bench::burgers(x, t, nu) as f32;
                }
            }
            // how steep does the reference actually get on this grid? the premise, in a number
            let mut worst_jump = 0.0f32;
            for i in 1..nx {
                worst_jump = worst_jump.max((target[i * nt + (nt - 1)] - target[(i - 1) * nt + (nt - 1)]).abs());
            }
            eprintln!("    [banked] reference at t=1 on Δx={:.4}: largest neighbour jump {worst_jump:.3}", 2.0 / nx as f32);
            let tv = leaf(&ctx, &target, &[n, 1]);
            for dims in [vec![2usize, 96, 96, 96, 1], vec![2, 128, 128, 128, 128, 1]] {
                let net = Mlp::new(&ctx, &dims, 5);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let u = burgers_ansatz(&ctx, &pv, &leaf(&ctx, &pts, &[n, 2]));
                    let d = u.sub(&tv);
                    let loss = d.mul(&d).mean_all();
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let uv = burgers_ansatz(&ctx, &pv, &leaf(&ctx, &pts, &[n, 2])).value().to_vec().await;
                let e = rel_l2(&uv, &target);
                // the ansatz's own guarantees, checked rather than assumed
                let at_t0 = (0..nx).map(|i| (uv[i * nt] - target[i * nt]).abs()).fold(0.0f32, f32::max);
                let at_edge = (0..nt).map(|j| uv[j].abs().max(uv[(nx - 1) * nt + j].abs())).fold(0.0f32, f32::max);
                eprintln!("    [banked] burgers ansatz {dims:?}, {steps} Adam steps: rel-L2 {e:.4} (t=0 exact to {at_t0:.1e}, |u| at the nearest x to ±1 is {at_edge:.3})");
                assert!(at_t0 < 1e-4, "the ansatz must be exactly −sin(πx) at t = 0: worst {at_t0:.2e}");
            }
        });
    }



    /// ⭐ **Testing the burgers prediction.** The diagnosis says burgers is residual-landscape-limited like
    /// advection, so marching should be its fix too. The machinery is heavier: the residual needs `u_xx`,
    /// so each window must carry `g″` as well as `g` and `g′`, and a second derivative accumulated from
    /// window to window is the obvious place for this to fall apart. With
    /// `u = g(x) + τ(1−x²)·M(x,τ)` and `b = 1−x²`:
    ///
    /// ```text
    /// u_τ  = b(M + τ M_τ)
    /// u_x  = g′  + τ(−2x·M + b·M_x)
    /// u_xx = g″  + τ(−2M − 4x·M_x + b·M_xx)
    /// ```
    ///
    /// The boundary stays exact across every window for free: `u(±1) = g(±1)` because `b(±1) = 0`, and
    /// `g` starts at `−sin(πx)`, which is already zero there.
    #[ignore = "trains eight burgers windows on the GPU (~25 min); run with -- --ignored"]
    #[test]
    fn marching_with_hard_conditions_on_the_burgers_row() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (nx, ntau, windows, steps) = (128usize, 16usize, 8usize, 3000u32);
            let nu = 0.01 / std::f64::consts::PI;
            let n = nx * ntau;
            let dt = 1.0f32 / windows as f32;
            let xs: Vec<f32> = (0..nx).map(|i| -1.0 + 2.0 * (i as f32 + 0.5) / nx as f32).collect();
            let taus: Vec<f32> = (0..ntau).map(|j| dt * j as f32 / (ntau as f32 - 1.0)).collect();
            let mut pts = vec![0.0f32; n * 2];
            for (i, &xv) in xs.iter().enumerate() {
                for (j, &tv) in taus.iter().enumerate() {
                    pts[(i * ntau + j) * 2] = xv;
                    pts[(i * ntau + j) * 2 + 1] = tv;
                }
            }
            let tile = |v: &[f32]| -> Vec<f32> { (0..n).map(|k| v[k / ntau]).collect() };
            let bx: Vec<f32> = (0..n).map(|k| 1.0 - xs[k / ntau] * xs[k / ntau]).collect();
            let bpx: Vec<f32> = (0..n).map(|k| -2.0 * xs[k / ntau]).collect();
            let bv = leaf(&ctx, &bx, &[n, 1]);
            let bpv = leaf(&ctx, &bpx, &[n, 1]);
            let two = leaf(&ctx, &[2.0f32], &[1]);
            let nuv = leaf(&ctx, &[nu as f32], &[1]);
            let pi = std::f32::consts::PI;
            let mut g: Vec<f32> = xs.iter().map(|&x| -(pi * x).sin()).collect();
            let mut gp: Vec<f32> = xs.iter().map(|&x| -pi * (pi * x).cos()).collect();
            let mut gpp: Vec<f32> = xs.iter().map(|&x| pi * pi * (pi * x).sin()).collect();

            let (mut num_all, mut den_all, mut worst) = (0.0f64, 0.0f64, 0.0f32);
            for w in 0..windows {
                let t0 = w as f32 * dt;
                let net = Mlp::new(&ctx, &[2, 64, 64, 1], 13 + w as u32);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                let gv = leaf(&ctx, &tile(&g), &[n, 1]);
                let gpv = leaf(&ctx, &tile(&gp), &[n, 1]);
                let gppv = leaf(&ctx, &tile(&gpp), &[n, 1]);
                let parts = |pv: &[Var], x: &Var| {
                    let tc = col_of(&ctx, x, 2, 1);
                    let mm = Mlp::forward_act(pv, x, Act::Tanh);
                    let d1 = crate::sciml::deriv(&mm, x);
                    let mx = col_of(&ctx, &d1, 2, 0);
                    let mt = col_of(&ctx, &d1, 2, 1);
                    let mxx = col_of(&ctx, &crate::sciml::deriv(&mx, x), 2, 0);
                    let u = gv.add(&tc.mul(&bv).mul(&mm));
                    let ut = bv.mul(&mm.add(&tc.mul(&mt)));
                    let ux = gpv.add(&tc.mul(&bpv.mul(&mm).add(&bv.mul(&mx))));
                    let uxx = gppv.add(&tc.mul(
                        &mm.mul(&two).neg().add(&bpv.mul(&mx).mul(&two)).add(&bv.mul(&mxx)),
                    ));
                    (u, ut, ux, uxx)
                };
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let x = leaf(&ctx, &pts, &[n, 2]);
                    let (u, ut, ux, uxx) = parts(&pv, &x);
                    let r = ut.add(&u.mul(&ux)).sub(&uxx.mul(&nuv));
                    let loss = r.mul(&r).mean_all();
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let x = leaf(&ctx, &pts, &[n, 2]);
                let (u, _, ux, uxx) = parts(&pv, &x);
                let (uv, uxv, uxxv) = (u.value().to_vec().await, ux.value().to_vec().await, uxx.value().to_vec().await);
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for (i, &xv) in xs.iter().enumerate() {
                    for (j, &tv) in taus.iter().enumerate() {
                        let want = crate::sciml::bench::burgers(xv as f64, (t0 + tv) as f64, nu) as f32;
                        num += (uv[i * ntau + j] - want).powi(2) as f64;
                        den += (want * want) as f64;
                    }
                }
                let e = (num / den).sqrt() as f32;
                worst = worst.max(e);
                num_all += num;
                den_all += den;
                for i in 0..nx {
                    let k = i * ntau + (ntau - 1);
                    g[i] = uv[k];
                    gp[i] = uxv[k];
                    gpp[i] = uxxv[k];
                }
                eprintln!("    [banked] burgers window {w} (t ∈ [{t0:.3}, {:.3}]): rel-L2 in-window {e:.4}", t0 + dt);
            }
            let rel = (num_all / den_all).sqrt() as f32;
            eprintln!(
                "  burgers ν=0.01/π, {windows} time windows × hard conditions, [2,64,64,1] per window, {steps} Adam steps each:\n    rel-L2 over the whole domain: {rel:.4}   (worst single window {worst:.4})\n    against: 0.2079 vanilla and 0.2434 full recipe in the table; 0.1980 soft / 0.3957 hard as ONE global fit; and 0.0050 for the ansatz regressed onto the reference"
            );
            assert!(rel < 0.1, "marching must fix the burgers row if the residual-landscape diagnosis is right: {rel:.4}");
        });
    }

    /// ⭐ **Which failure mode is it, then?** The regression fixture rules representation out, leaving the
    /// two the other rows turned on: the loss balancing a soft boundary penalty forces (Helmholtz's), or
    /// the residual landscape (advection's). Hard constraints separate them — they remove the first
    /// entirely and do nothing about the second, so this arm pair is the test.
    ///
    /// Both arms share the grid, steps, schedule, seed, optimiser and hidden layers; only the treatment of
    /// the initial and boundary conditions differs. ⚠ The PDE grid is coarser than the regression one
    /// (`Δx = 0.0156` against a shock width of ~0.0032) because the residual needs `u_xx` at every point;
    /// both arms carry that identically, so it bounds the achievable error for both rather than favouring
    /// either.
    #[ignore = "trains two burgers PINNs on the GPU (~25 min); run with -- --ignored"]
    #[test]
    fn soft_against_hard_conditions_on_the_burgers_row() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (nx, nt, steps, bw) = (128usize, 32usize, 8000u32, 100.0f32);
            let nu = 0.01 / std::f64::consts::PI;
            let n = nx * nt;
            let (mut pts, mut exact) = (vec![0.0f32; n * 2], vec![0.0f32; n]);
            for i in 0..nx {
                for j in 0..nt {
                    let x = -1.0 + 2.0 * (i as f64 + 0.5) / nx as f64;
                    let t = j as f64 / (nt as f64 - 1.0);
                    let k = i * nt + j;
                    pts[k * 2] = x as f32;
                    pts[k * 2 + 1] = t as f32;
                    exact[k] = crate::sciml::bench::burgers(x, t, nu) as f32;
                }
            }
            let ic: Vec<f32> = (0..nx).flat_map(|i| [-1.0 + 2.0 * (i as f32 + 0.5) / nx as f32, 0.0]).collect();
            let ic_u: Vec<f32> = (0..nx).map(|i| -((std::f32::consts::PI * (-1.0 + 2.0 * (i as f32 + 0.5) / nx as f32)).sin())).collect();
            let edges: Vec<f32> = (0..nt).flat_map(|j| [-1.0, j as f32 / (nt as f32 - 1.0), 1.0, j as f32 / (nt as f32 - 1.0)]).collect();
            let nuv = leaf(&ctx, &[nu as f32], &[1]);
            let bwv = leaf(&ctx, &[bw], &[1]);

            let mut results = Vec::new();
            for hard in [false, true] {
                let net = Mlp::new(&ctx, &[2, 96, 96, 96, 1], 5);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 3e-3);
                for ep in 0..steps {
                    if ep == steps * 3 / 4 {
                        adam = crate::Adam::new(&wp, 3e-4);
                    }
                    let pv = vars(&wp);
                    let x = leaf(&ctx, &pts, &[n, 2]);
                    let u = if hard { burgers_ansatz(&ctx, &pv, &x) } else { Mlp::forward_act(&pv, &x, Act::Tanh) };
                    let g = crate::sciml::deriv(&u, &x);
                    let ux = col_of(&ctx, &g, 2, 0);
                    let ut = col_of(&ctx, &g, 2, 1);
                    let uxx = col_of(&ctx, &crate::sciml::deriv(&ux, &x), 2, 0);
                    let r = ut.add(&u.mul(&ux)).sub(&uxx.mul(&nuv));
                    let mut loss = r.mul(&r).mean_all();
                    if !hard {
                        let ui = Mlp::forward_act(&pv, &leaf(&ctx, &ic, &[nx, 2]), Act::Tanh);
                        let di = ui.sub(&leaf(&ctx, &ic_u, &[nx, 1]));
                        let ub = Mlp::forward_act(&pv, &leaf(&ctx, &edges, &[2 * nt, 2]), Act::Tanh);
                        loss = loss.add(&di.mul(&di).mean_all().add(&ub.mul(&ub).mean_all()).mul(&bwv));
                    }
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let x = leaf(&ctx, &pts, &[n, 2]);
                let u = if hard { burgers_ansatz(&ctx, &pv, &x) } else { Mlp::forward_act(&pv, &x, Act::Tanh) };
                let e = rel_l2(&u.value().to_vec().await, &exact);
                eprintln!("    [banked] burgers ν=0.01/π, {} conditions, {steps} Adam steps: rel-L2 {e:.4}", if hard { "HARD" } else { "soft" });
                results.push(e);
            }
            let (soft, hard) = (results[0], results[1]);
            eprintln!(
                "  burgers ν=0.01/π on [−1,1]×[0,1], {n} collocation points, same net / steps / seed:\n    soft conditions (weight {bw}):  rel-L2 {soft:.4}\n    hard conditions:                rel-L2 {hard:.4}\n    for context: 0.2079 vanilla and 0.2434 full recipe in the table; and this ansatz regresses onto the reference at 0.0050"
            );
            assert!(hard < 0.5 && soft < 0.9, "both arms must produce something, or this ranks nothing: soft {soft:.4} hard {hard:.4}");
        });
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
