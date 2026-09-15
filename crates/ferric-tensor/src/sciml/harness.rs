//! **The benchmark harness** — the field's standard problems ([`super::bench`]) solved by a named recipe,
//! scored against an independent reference, so a number from this stack means the same thing as one from
//! PINNacle or jinns. A [`Problem`] supplies its residual, its constraints and its reference; a
//! [`Recipe`] names the net, the optimiser stages and the loss handling; [`run`] trains and scores.

use super::bench;
use super::util::*;
use super::{flatten, mse, scalar, unflatten, Act, Causal, FourierNet, Lbfgs, LossBalancer, Mlp, NtkBalancer};
use crate::{Adam, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;
use std::time::Instant;

/// A PDE benchmark: its box domain, PDE residual, constraint losses and reference solution.
pub trait Problem {
    fn name(&self) -> &str;
    /// Input dimension (space and time coordinates together).
    fn dim(&self) -> usize;
    fn lo(&self) -> Vec<f64>;
    fn hi(&self) -> Vec<f64>;
    /// PDE residual at the `[N, dim]` points `x`, as `[N, 1]`, for the network `fwd`.
    fn residual(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, n: usize) -> Var;
    /// Boundary / initial / periodic constraint RESIDUALS, each `[·, 1]`, sampled with `n` points each.
    ///
    /// ⛔ Residual vectors, not a summed loss: an NTK trace is a property of the per-point Jacobian, and
    /// summing first destroys it. The harness takes the mean square of each for the loss.
    fn constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var>;
    /// Index of the time coordinate, if the problem is time-dependent — the axis causal weighting slabs.
    fn time_axis(&self) -> Option<usize> {
        None
    }
    fn reference(&self, x: &[f64]) -> f64;
    /// Fourier-feature scales that suit the problem's frequency content.
    fn scales(&self) -> Vec<f32> {
        vec![1.0, 3.0]
    }
}

/// Which network the recipe trains.
#[derive(Clone, Debug)]
pub enum Net {
    TanhMlp { hidden: Vec<usize> },
    Fourier { m_per_scale: usize, hidden: Vec<usize> },
}

/// How the residual and constraint terms are weighted against each other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Weighting {
    /// One fixed multiplier on every constraint term.
    Fixed(f32),
    /// Gradient-norm balancing — equalises how hard each term PULLS (arXiv 2001.04536).
    GradNorm,
    /// NTK-trace balancing — equalises how fast each term's error DECAYS (arXiv 2007.14527).
    Ntk,
}

/// A named training recipe.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub name: &'static str,
    pub net: Net,
    pub adam_steps: usize,
    pub lr: f32,
    pub lbfgs_iters: usize,
    pub weighting: Weighting,
    /// Causal weighting in time with this many slabs (arXiv 2203.07404). Requires `Problem::time_axis`.
    pub causal_slabs: Option<usize>,
}

impl Recipe {
    /// The classic: tanh MLP, Adam, fixed boundary weight.
    pub fn vanilla() -> Self {
        Recipe { name: "vanilla", net: Net::TanhMlp { hidden: vec![64, 64, 64] }, adam_steps: 4000, lr: 1e-3, lbfgs_iters: 0, weighting: Weighting::Fixed(10.0), causal_slabs: None }
    }
    /// Fourier features + gradient-norm balancing + Adam then strong-Wolfe L-BFGS.
    pub fn full() -> Self {
        Recipe { name: "fourier+balance+lbfgs", net: Net::Fourier { m_per_scale: 32, hidden: vec![64, 64] }, adam_steps: 4000, lr: 1e-3, lbfgs_iters: 500, weighting: Weighting::GradNorm, causal_slabs: None }
    }
    /// The full recipe with NTK-trace weighting instead of gradient-norm — what jaxpi/PirateNets run, and
    /// what the Helmholtz row of the benchmark table names as missing.
    pub fn ntk() -> Self {
        Recipe { name: "fourier+ntk+lbfgs", weighting: Weighting::Ntk, ..Self::full() }
    }
    /// The full recipe plus causal weighting in time — for the problems where a vanilla PINN fits a later
    /// time first and lands on a wrong branch (advection at beta = 30, the reaction equation).
    pub fn causal(slabs: usize) -> Self {
        Recipe { name: "fourier+causal+lbfgs", causal_slabs: Some(slabs), ..Self::full() }
    }
}

/// What a run produced.
#[derive(Clone, Debug)]
pub struct Report {
    pub problem: String,
    pub recipe: &'static str,
    pub n_colloc: usize,
    pub rel_l2: f32,
    pub loss_after_adam: f32,
    pub loss_final: f32,
    pub secs: f64,
}

enum NetImpl {
    Mlp(Mlp),
    Fourier(FourierNet),
}

/// Train `recipe` on `problem` with `colloc` collocation points (`[N, dim]` flattened) and score it on a
/// `grid_m`-per-axis grid against the reference. Synchronous; the L-BFGS stage blocks on readbacks.
pub fn run(ctx: &Arc<Context>, problem: &dyn Problem, recipe: &Recipe, colloc: &[f32], grid_m: usize, seed: u32) -> Report {
    let t0 = Instant::now();
    let d = problem.dim();
    let n = colloc.len() / d;
    let net = match &recipe.net {
        Net::TanhMlp { hidden } => {
            let mut dims = vec![d];
            dims.extend(hidden);
            dims.push(1);
            NetImpl::Mlp(Mlp::new(ctx, &dims, seed))
        }
        Net::Fourier { m_per_scale, hidden } => NetImpl::Fourier(FourierNet::new(ctx, d, *m_per_scale, &problem.scales(), hidden, 1, Act::Tanh, seed)),
    };
    let params0: Vec<Tensor> = match &net {
        NetImpl::Mlp(m) => m.params.clone(),
        NetImpl::Fourier(f) => f.params.clone(),
    };
    let forward = |pv: &[Var], x: &Var| -> Var {
        match &net {
            NetImpl::Mlp(_) => Mlp::forward_act(pv, x, Act::Tanh),
            NetImpl::Fourier(f) => f.forward(pv, x),
        }
    };
    // per-point residual and constraint residual VECTORS; the loss takes their mean squares
    let terms = |pv: &[Var], it: u32| -> Vec<Var> {
        let fwd = |x: &Var| forward(pv, x);
        let xv = leaf(ctx, colloc, &[n, d]);
        let mut v = vec![problem.residual(ctx, &fwd, &xv, n)];
        v.extend(problem.constraints(ctx, &fwd, 200, seed.wrapping_add(it)));
        v
    };
    // causal slab index of each collocation point, from the problem's time axis
    let slab: Option<Vec<usize>> = recipe.causal_slabs.and_then(|k| {
        problem.time_axis().map(|ax| {
            let (lo, hi) = (problem.lo()[ax], problem.hi()[ax]);
            colloc
                .chunks(d)
                .map(|p| {
                    let f = ((p[ax] as f64 - lo) / (hi - lo)).clamp(0.0, 0.999);
                    (f * k as f64) as usize
                })
                .collect()
        })
    });

    let mut wp = params0;
    let n_terms = 1 + problem.constraints(ctx, &|x: &Var| forward(&vars(&wp), x), 8, 0).len();
    let mut bal = LossBalancer::new(n_terms, 0.1);
    let mut ntk = NtkBalancer::new(n_terms, 4, 0.5);
    let mut causal = recipe.causal_slabs.map(|_| Causal::new(vec![1e-2, 1e-1, 1.0, 10.0, 100.0], 0.99));
    let loss_after_adam = pollster::block_on(async {
        let mut adam = Adam::new(&wp, recipe.lr);
        let mut last = f32::NAN;
        for it in 0..recipe.adam_steps {
            let pv = vars(&wp);
            let mut t = terms(&pv, it as u32);
            // causal weighting multiplies the RESIDUAL term's points by sqrt(w) before the mean square
            if let (Some(cz), Some(sl), Some(k)) = (causal.as_mut(), slab.as_ref(), recipe.causal_slabs) {
                let r = t[0].value().to_vec().await;
                let w = cz.weights(&Causal::slab_losses(&r, sl, k));
                t[0] = t[0].mul(&Causal::point_weights(ctx, &w, sl).sqrt());
            }
            let loss = match recipe.weighting {
                Weighting::Fixed(w) => {
                    let mut l = mse(&t[0]);
                    for c in &t[1..] {
                        l = l.add(&mse(c).mul(&scalar(c, w)));
                    }
                    l
                }
                Weighting::GradNorm => {
                    let ls: Vec<Var> = t.iter().map(mse).collect();
                    if it % 100 == 0 {
                        bal.update(&ls, &pv).await;
                    }
                    bal.combine(&ls)
                }
                Weighting::Ntk => {
                    if it % 100 == 0 {
                        ntk.update(ctx, &t, &pv, it as u32);
                    }
                    ntk.combine(&t)
                }
            };
            last = step(ctx, &loss, &pv, &mut wp, &mut adam).await;
        }
        last
    });
    let mut loss_final = loss_after_adam;
    if recipe.lbfgs_iters > 0 {
        let shapes: Vec<Vec<usize>> = wp.iter().map(|t| t.shape.clone()).collect();
        let x0 = pollster::block_on(flatten(&wp));
        let w_final: Vec<f32> = match recipe.weighting {
            Weighting::Fixed(w) => std::iter::once(1.0).chain(std::iter::repeat_n(w, n_terms - 1)).collect(),
            Weighting::GradNorm => bal.weights.clone(),
            Weighting::Ntk => ntk.weights.clone(),
        };
        let evaluate = |flat: &[f32]| -> (f32, Vec<f32>) {
            let ts = unflatten(ctx, flat, &shapes);
            let pv = vars(&ts);
            let t = terms(&pv, 0);
            let mut loss = mse(&t[0]).mul(&scalar(&t[0], w_final[0]));
            for (c, &w) in t[1..].iter().zip(&w_final[1..]) {
                loss = loss.add(&mse(c).mul(&scalar(c, w)));
            }
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
        let r = Lbfgs::new(20).minimize(x0, evaluate, recipe.lbfgs_iters, 1e-9);
        loss_final = r.f;
        wp = unflatten(ctx, &r.x, &shapes);
    }
    // score on the grid
    let (lo, hi) = (problem.lo(), problem.hi());
    let g = box_grid(&lo, &hi, grid_m);
    let ng = g.len() / d;
    let pred = pollster::block_on(async { forward(&vars(&wp), &leaf(ctx, &g, &[ng, d])).value().to_vec().await });
    let truth: Vec<f32> = g.chunks(d).map(|p| problem.reference(&p.iter().map(|&v| v as f64).collect::<Vec<_>>()) as f32).collect();
    Report { problem: problem.name().to_string(), recipe: recipe.name, n_colloc: n, rel_l2: rel_l2(&pred, &truth), loss_after_adam, loss_final, secs: t0.elapsed().as_secs_f64() }
}

// ---------------------------------------------------------------- the problems ------------------

/// Burgers, `ν = 0.01/π`: the PINNacle / Raissi setting, with the shock at `x = 0`.
pub struct Burgers {
    pub nu: f64,
}
impl Problem for Burgers {
    fn name(&self) -> &str {
        "burgers1d"
    }
    fn dim(&self) -> usize {
        2
    }
    fn lo(&self) -> Vec<f64> {
        vec![-1.0, 0.0]
    }
    fn hi(&self) -> Vec<f64> {
        vec![1.0, 1.0]
    }
    fn residual(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, n: usize) -> Var {
        let u = fwd(x);
        let u_t = dcol(ctx, &u, x, n, 2, 1);
        let u_x = dcol(ctx, &u, x, n, 2, 0);
        let u_xx = dcol(ctx, &u_x, x, n, 2, 0);
        u_t.add(&u.mul(&u_x)).sub(&u_xx.mul(&scalar(&u_xx, self.nu as f32)))
    }
    fn constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let t: Vec<f32> = (0..n).map(|i| u01(i as u32, seed)).collect();
        let xs: Vec<f32> = (0..n).map(|i| 2.0 * u01(i as u32, seed ^ 0x5151) - 1.0).collect();
        let bl: Vec<f32> = t.iter().flat_map(|&t| [-1.0, t]).collect();
        let br: Vec<f32> = t.iter().flat_map(|&t| [1.0, t]).collect();
        let ic: Vec<f32> = xs.iter().flat_map(|&x| [x, 0.0]).collect();
        let u0: Vec<f32> = xs.iter().map(|&x| -(std::f32::consts::PI * x).sin()).collect();
        vec![fwd(&leaf(ctx, &bl, &[n, 2])), fwd(&leaf(ctx, &br, &[n, 2])), fwd(&leaf(ctx, &ic, &[n, 2])).sub(&col(ctx, &u0))]
    }
    fn time_axis(&self) -> Option<usize> {
        Some(1)
    }
    fn reference(&self, x: &[f64]) -> f64 {
        bench::burgers(x[0], x[1], self.nu)
    }
    fn scales(&self) -> Vec<f32> {
        vec![1.0, 4.0]
    }
}

/// Helmholtz on `[−1,1]²` with the manufactured `sin(a₁πx) sin(a₂πy)`.
pub struct Helmholtz {
    pub a1: f64,
    pub a2: f64,
    pub k: f64,
}
impl Problem for Helmholtz {
    fn name(&self) -> &str {
        "helmholtz2d"
    }
    fn dim(&self) -> usize {
        2
    }
    fn lo(&self) -> Vec<f64> {
        vec![-1.0, -1.0]
    }
    fn hi(&self) -> Vec<f64> {
        vec![1.0, 1.0]
    }
    fn residual(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, n: usize) -> Var {
        let u = fwd(x);
        let ux = dcol(ctx, &u, x, n, 2, 0);
        let uy = dcol(ctx, &u, x, n, 2, 1);
        let lap = dcol(ctx, &ux, x, n, 2, 0).add(&dcol(ctx, &uy, x, n, 2, 1));
        // q at the collocation points comes from the reference's own forcing
        let pts = pollster::block_on(x.value().to_vec());
        let q: Vec<f32> = pts.chunks(2).map(|p| bench::helmholtz(p[0] as f64, p[1] as f64, self.a1, self.a2, self.k).1 as f32).collect();
        lap.add(&u.mul(&scalar(&u, (self.k * self.k) as f32))).sub(&col(ctx, &q))
    }
    fn constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let s: Vec<f32> = (0..n).map(|i| 2.0 * u01(i as u32, seed) - 1.0).collect();
        let mut b = Vec::with_capacity(8 * n);
        for &v in &s {
            b.extend([-1.0, v, 1.0, v, v, -1.0, v, 1.0]);
        }
        vec![fwd(&leaf(ctx, &b, &[4 * n, 2]))]
    }
    fn reference(&self, x: &[f64]) -> f64 {
        bench::helmholtz(x[0], x[1], self.a1, self.a2, self.k).0
    }
    fn scales(&self) -> Vec<f32> {
        vec![1.0, 2.0]
    }
}

/// Heat equation on `[0,1] × [0,1]`.
pub struct Heat;
impl Problem for Heat {
    fn name(&self) -> &str {
        "heat1d"
    }
    fn dim(&self) -> usize {
        2
    }
    fn lo(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }
    fn hi(&self) -> Vec<f64> {
        vec![1.0, 1.0]
    }
    fn residual(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, n: usize) -> Var {
        let u = fwd(x);
        let u_t = dcol(ctx, &u, x, n, 2, 1);
        let u_x = dcol(ctx, &u, x, n, 2, 0);
        u_t.sub(&dcol(ctx, &u_x, x, n, 2, 0))
    }
    fn constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let t: Vec<f32> = (0..n).map(|i| u01(i as u32, seed)).collect();
        let xs: Vec<f32> = (0..n).map(|i| u01(i as u32, seed ^ 0x77)).collect();
        let b: Vec<f32> = t.iter().flat_map(|&t| [0.0, t, 1.0, t]).collect();
        let ic: Vec<f32> = xs.iter().flat_map(|&x| [x, 0.0]).collect();
        let u0: Vec<f32> = xs.iter().map(|&x| (std::f32::consts::PI * x).sin()).collect();
        vec![fwd(&leaf(ctx, &b, &[2 * n, 2])), fwd(&leaf(ctx, &ic, &[n, 2])).sub(&col(ctx, &u0))]
    }
    fn time_axis(&self) -> Option<usize> {
        Some(1)
    }
    fn reference(&self, x: &[f64]) -> f64 {
        bench::heat(x[0], x[1])
    }
    /// The solution is `sin πx`: half a cycle per unit, so scales near `0.5`; the default `[1, 3]`
    /// measured 7× WORSE than a plain tanh net here.
    fn scales(&self) -> Vec<f32> {
        vec![0.5, 1.0]
    }
}

/// Advection with speed `β`, periodic on `[0, 2π] × [0, 1]`.
pub struct Advection {
    pub beta: f64,
}
impl Problem for Advection {
    fn name(&self) -> &str {
        "advection1d"
    }
    fn dim(&self) -> usize {
        2
    }
    fn lo(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }
    fn hi(&self) -> Vec<f64> {
        vec![std::f64::consts::TAU, 1.0]
    }
    fn residual(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, x: &Var, n: usize) -> Var {
        let u = fwd(x);
        let u_t = dcol(ctx, &u, x, n, 2, 1);
        let u_x = dcol(ctx, &u, x, n, 2, 0);
        u_t.add(&u_x.mul(&scalar(&u_x, self.beta as f32)))
    }
    fn constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let t: Vec<f32> = (0..n).map(|i| u01(i as u32, seed)).collect();
        let xs: Vec<f32> = (0..n).map(|i| std::f32::consts::TAU * u01(i as u32, seed ^ 0x99)).collect();
        let l: Vec<f32> = t.iter().flat_map(|&t| [0.0, t]).collect();
        let r: Vec<f32> = t.iter().flat_map(|&t| [std::f32::consts::TAU, t]).collect();
        let ic: Vec<f32> = xs.iter().flat_map(|&x| [x, 0.0]).collect();
        let u0: Vec<f32> = xs.iter().map(|&x| x.sin()).collect();
        vec![fwd(&leaf(ctx, &l, &[n, 2])).sub(&fwd(&leaf(ctx, &r, &[n, 2]))), fwd(&leaf(ctx, &ic, &[n, 2])).sub(&col(ctx, &u0))]
    }
    fn time_axis(&self) -> Option<usize> {
        Some(1)
    }
    fn reference(&self, x: &[f64]) -> f64 {
        bench::advection(x[0], x[1], self.beta)
    }
    fn scales(&self) -> Vec<f32> {
        vec![1.0, (self.beta / 6.0) as f32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::{rar_select, train};

    fn ctx() -> Arc<Context> {
        Arc::new(pollster::block_on(Context::new()).unwrap())
    }

    /// One benchmark problem, both recipes, printed as a table row and sanity-checked. The bars are
    /// deliberately loose until the table has been measured: a benchmark harness that asserts the
    /// result it hopes for is not a benchmark. Measured rows are recorded in `docs/SCIML.md`.
    ///
    /// ⛔ The first version ran all four problems in one test — eight trainings, ~50 min — and timed out
    /// at 58 min sharing the GPU with another run, having produced two rows: heat vanilla 0.045 vs full
    /// 0.318 (the recipe's default Fourier scales `[1, 3]` are wrong for a solution at frequency π, and
    /// the harness is what made that visible), and Helmholtz `a = (1, 4)` 0.477 vs 0.478 — unsolved by
    /// either at 4000 steps, where the source paper uses ten times as many.
    fn bench_one(p: &dyn Problem, colloc_n: usize, seed: u32) -> (Report, Report) {
        let ctx = ctx();
        let colloc = box_points(&p.lo(), &p.hi(), colloc_n, 7);
        let v = run(&ctx, p, &Recipe::vanilla(), &colloc, 51, seed);
        let f = run(&ctx, p, &Recipe::full(), &colloc, 51, seed);
        eprintln!("  {:<12} vanilla {:.4} ({:.0}s)   full {:.4} ({:.0}s)   full losses {:.1e}->{:.1e}", p.name(), v.rel_l2, v.secs, f.rel_l2, f.secs, f.loss_after_adam, f.loss_final);
        // ⚠ A benchmark row is a MEASUREMENT, not a verdict: an unsolved problem is a legitimate row
        // (advection at β = 30 came out 0.91 / 1.08 — Krishnapriyan's failure case, the one causal
        // training exists for — and a "better than zero" bar here turned that measurement into a
        // failed test). The only assertion is that the score is a number; the verdict is printed.
        for r in [&v, &f] {
            assert!(r.rel_l2.is_finite(), "{}: {} produced a non-finite score", p.name(), r.recipe);
            let verdict = if r.rel_l2 < 0.05 { "solved" } else if r.rel_l2 < 1.0 { "partial" } else { "UNSOLVED (no better than zero)" };
            eprintln!("    {:<24} {:<8}", r.recipe, verdict);
        }
        (v, f)
    }

    #[ignore = "two trainings on the GPU (~12 min); run with -- --ignored"]
    #[test]
    fn benchmark_heat() {
        bench_one(&Heat, 2000, 1);
    }
    #[ignore = "two trainings on the GPU (~12 min); run with -- --ignored"]
    #[test]
    fn benchmark_helmholtz() {
        bench_one(&Helmholtz { a1: 1.0, a2: 4.0, k: 1.0 }, 2000, 1);
    }

    /// One problem against a named list of recipes — for the rows where the table says a specific piece
    /// is missing. Prints; asserts only that each score is a number.
    fn compare(p: &dyn Problem, recipes: &[Recipe], colloc_n: usize, seed: u32) -> Vec<Report> {
        let ctx = ctx();
        let colloc = box_points(&p.lo(), &p.hi(), colloc_n, 7);
        recipes
            .iter()
            .map(|r| {
                let rep = run(&ctx, p, r, &colloc, 51, seed);
                eprintln!("  {:<12} {:<24} rel-L2 {:.4}  ({:.0}s, loss {:.1e} -> {:.1e})", p.name(), rep.recipe, rep.rel_l2, rep.secs, rep.loss_after_adam, rep.loss_final);
                assert!(rep.rel_l2.is_finite(), "{} produced a non-finite score", rep.recipe);
                rep
            })
            .collect()
    }

    /// ⭐ **Helmholtz, where the table said NTK weighting was the missing piece** — gradient-norm against
    /// NTK-trace weighting, same net, same steps, same points.
    #[ignore = "two trainings on the GPU (~14 min); run with -- --ignored"]
    #[test]
    fn helmholtz_gradnorm_against_ntk_weighting() {
        compare(&Helmholtz { a1: 1.0, a2: 4.0, k: 1.0 }, &[Recipe::full(), Recipe::ntk()], 2000, 1);
    }

    /// ⭐ **Advection at β = 30, where the table said causal weighting was the missing piece** — the full
    /// recipe against the same recipe with causal weighting in time.
    #[ignore = "two trainings on the GPU (~10 min); run with -- --ignored"]
    #[test]
    fn advection_with_and_without_causal_weighting() {
        compare(&Advection { beta: 30.0 }, &[Recipe::full(), Recipe::causal(16)], 2000, 1);
    }
    #[ignore = "two trainings on the GPU (~12 min); run with -- --ignored"]
    #[test]
    fn benchmark_burgers() {
        bench_one(&Burgers { nu: 0.01 / std::f64::consts::PI }, 2000, 1);
    }
    #[ignore = "two trainings on the GPU (~12 min); run with -- --ignored"]
    #[test]
    fn benchmark_advection() {
        bench_one(&Advection { beta: 30.0 }, 2000, 1);
    }

    /// ⭐ **Refinement, in the regime where it can be shown to help.** Burgers with its shock, both arms
    /// Fourier features + Adam + L-BFGS, equal budget: 2540 uniform against 2000 uniform + 540 refined —
    /// DeepXDE's own numbers for this problem (Lu et al. 2021, §3.2).
    #[ignore = "two Burgers solves on the GPU (~15 min); run with -- --ignored"]
    #[test]
    fn refinement_beats_uniform_sampling_on_the_burgers_shock_at_equal_budget() {
        let ctx = ctx();
        let p = Burgers { nu: 0.01 / std::f64::consts::PI };
        let (lo, hi) = (p.lo(), p.hi());
        let recipe = Recipe::full();
        // uniform arm
        let uni = box_points(&lo, &hi, 2540, 7);
        let r_uni = run(&ctx, &p, &recipe, &uni, 51, 1);
        // refined arm: a warm-up on the base set, then residual-selected additions, then the full recipe
        let mut pts = box_points(&lo, &hi, 2000, 7);
        let cand = box_points(&lo, &hi, 20000, 99);
        {
            let net = FourierNet::new(&ctx, 2, 32, &p.scales(), &[64, 64], 1, Act::Tanh, 1);
            let mut wp = net.params.clone();
            pollster::block_on(async {
                let mut adam = Adam::new(&wp, 1e-3);
                // ⚠ 1500, not 3000: this warm-up exists only to produce a residual field good enough to
                // SELECT points from — the scoring runs are the two full-recipe solves below, and at
                // 3000 the whole test exceeded a 40-minute budget twice.
                for it in 0..1500usize {
                    let pv = vars(&wp);
                    let fwd = |x: &Var| net.forward(&pv, x);
                    if it > 0 && it % 250 == 0 && pts.len() / 2 < 2540 {
                        let nc = cand.len() / 2;
                        let cv = leaf(&ctx, &cand, &[nc, 2]);
                        let r = p.residual(&ctx, &fwd, &cv, nc).value().to_vec().await;
                        for i in rar_select(&r, 108) { // 5 batches at 500..2500 = 540, DeepXDE's count
                            pts.extend([cand[2 * i], cand[2 * i + 1]]);
                        }
                    }
                    let n = pts.len() / 2;
                    let xv = leaf(&ctx, &pts, &[n, 2]);
                    let mut loss = train::mse(&p.residual(&ctx, &fwd, &xv, n));
                    for c in p.constraints(&ctx, &fwd, 200, it as u32) {
                        loss = loss.add(&train::mse(&c));
                    }
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
            });
        }
        let r_rar = run(&ctx, &p, &recipe, &pts, 51, 1);
        eprintln!("  burgers: uniform({}) rel-L2 {:.4}   refined({}) rel-L2 {:.4}", r_uni.n_colloc, r_uni.rel_l2, r_rar.n_colloc, r_rar.rel_l2);
        assert_eq!(r_uni.n_colloc, r_rar.n_colloc, "equal budget or it is not a comparison");
        assert!(r_rar.rel_l2 < r_uni.rel_l2, "refinement must beat uniform on the shock: {:.4} vs {:.4}", r_rar.rel_l2, r_uni.rel_l2);
    }

    /// ⭐⭐ **The certificate, on a trained PINN.** Poisson on the unit square (the `pinn_poisson2d` setting):
    /// the bound computed from the net's own residual and boundary mismatch must sit ABOVE the true error
    /// against the manufactured solution, and not absurdly so.
    #[ignore = "trains one PINN on the GPU (~5 min); run with -- --ignored"]
    #[test]
    fn the_certificate_bounds_a_trained_poisson_pinn_from_its_residual_alone() {
        use crate::sciml::certify::{elliptic_l2_bound, grid_l2};
        let ctx = ctx();
        let pi = std::f32::consts::PI;
        let net = FourierNet::new(&ctx, 2, 32, &[1.0, 2.0], &[64, 64], 1, Act::Tanh, 1);
        let mut wp = net.params.clone();
        let colloc = box_points(&[0.0, 0.0], &[1.0, 1.0], 2000, 7);
        let n = 2000;
        let f: Vec<f32> = colloc.chunks(2).map(|p| -2.0 * pi * pi * (pi * p[0]).sin() * (pi * p[1]).sin()).collect();
        let residual = |pv: &[Var], x: &Var, nn: usize, fv: &[f32]| -> Var {
            let u = net.forward(pv, x);
            let ux = dcol(&ctx, &u, x, nn, 2, 0);
            let uy = dcol(&ctx, &u, x, nn, 2, 1);
            dcol(&ctx, &ux, x, nn, 2, 0).add(&dcol(&ctx, &uy, x, nn, 2, 1)).sub(&col(&ctx, fv))
        };
        pollster::block_on(async {
            let mut adam = Adam::new(&wp, 1e-3);
            let mut bal = LossBalancer::new(2, 0.1);
            for it in 0..4000usize {
                let pv = vars(&wp);
                let xv = leaf(&ctx, &colloc, &[n, 2]);
                let l_res = mse(&residual(&pv, &xv, n, &f));
                let s: Vec<f32> = (0..200).map(|i| u01(i, it as u32)).collect();
                let mut b = Vec::new();
                for &v in &s {
                    b.extend([0.0, v, 1.0, v, v, 0.0, v, 1.0]);
                }
                let l_bc = mse(&net.forward(&pv, &leaf(&ctx, &b, &[800, 2])));
                if it % 100 == 0 {
                    bal.update(&[l_res.clone(), l_bc.clone()], &pv).await;
                }
                let loss = bal.combine(&[l_res, l_bc]);
                step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
        });
        // the certificate's inputs, from the net alone
        let m = 101usize;
        let g = box_grid(&[0.0, 0.0], &[1.0, 1.0], m);
        let ng = m * m;
        let fg: Vec<f32> = g.chunks(2).map(|p| -2.0 * pi * pi * (pi * p[0]).sin() * (pi * p[1]).sin()).collect();
        let pv = vars(&wp);
        let (r, u) = pollster::block_on(async {
            let xv = leaf(&ctx, &g, &[ng, 2]);
            (residual(&pv, &xv, ng, &fg).value().to_vec().await, net.forward(&pv, &xv).value().to_vec().await)
        });
        let r_l2 = grid_l2(&r, m, 2, 1.0);
        let mut bmax = 0.0f32;
        for i in 0..m {
            for &(x, y) in &[(0usize, i), (m - 1, i), (i, 0), (i, m - 1)] {
                bmax = bmax.max(u[x * m + y].abs());
            }
        }
        let bound = elliptic_l2_bound(r_l2, bmax as f64, 2.0 * (pi as f64).powi(2), 0.0, 1.0).expect("Poisson is coercive");
        let truth: Vec<f32> = g.chunks(2).map(|p| (pi * p[0]).sin() * (pi * p[1]).sin()).collect();
        let e: Vec<f32> = u.iter().zip(&truth).map(|(a, b)| a - b).collect();
        let e_l2 = grid_l2(&e, m, 2, 1.0);
        eprintln!("  Poisson PINN: ‖residual‖ = {r_l2:.3e}, boundary max = {bmax:.3e}  ⇒  certificate ‖e‖ ≤ {bound:.3e};  true ‖e‖ = {e_l2:.3e}  ({:.1}x)", bound / e_l2);
        assert!(bound >= e_l2, "the certificate must be SOUND: bound {bound:.3e} < true error {e_l2:.3e}");
        assert!(bound < 100.0 * e_l2, "and not vacuous: {:.1}x the true error", bound / e_l2);
    }
}
