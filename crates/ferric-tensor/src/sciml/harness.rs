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
    /// Constraint residuals that hold at **every** time — spatial boundaries, periodicity — each `[·, 1]`.
    ///
    /// ⛔ Residual vectors, not a summed loss: an NTK trace is a property of the per-point Jacobian, and
    /// summing first destroys it. The harness takes the mean square of each for the loss.
    ///
    /// ⚠ Separated from the initial condition because **time marching replaces the initial condition and
    /// keeps the boundaries**. A single `constraints()` cannot express that: window two must satisfy the
    /// same walls as window one but is handed its start state by window one, not by `t = 0`.
    fn boundary_constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var>;
    /// The `t = 0` initial condition, if the problem has one. `None` for a steady problem.
    fn initial_constraint(&self, _ctx: &Arc<Context>, _fwd: &dyn Fn(&Var) -> Var, _n: usize, _seed: u32) -> Option<Var> {
        None
    }
    /// Every constraint: the boundaries plus the initial condition. What ordinary (unwindowed) training uses.
    fn constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let mut v = self.boundary_constraints(ctx, fwd, n, seed);
        v.extend(self.initial_constraint(ctx, fwd, n, seed));
        v
    }
    /// Points on the time slice `t`, for handing one window's end state to the next. `None` unless the
    /// problem is time-dependent.
    fn time_slice(&self, _n: usize, _t: f64, _seed: u32) -> Option<Vec<f32>> {
        None
    }
    /// Index of the time coordinate, if the problem is time-dependent — the axis causal weighting slabs.
    fn time_axis(&self) -> Option<usize> {
        None
    }
    /// Wrap the raw network output so the problem's conditions hold **by construction** — `u = A(x) +
    /// B(x)·M(x)` with `A` satisfying the condition and `B` vanishing where it is imposed. Returning
    /// `Some` tells [`run`] to drop every penalty term, because there is nothing left to weight.
    ///
    /// ⭐ On the benchmark's helmholtz row this is worth ~50× (0.3066 with a soft penalty and the full
    /// recipe, against 0.0059 with `(1−x²)(1−y²)·M` and plain Adam). ⚠ It is not a general win: on
    /// advection and burgers it makes things *worse*, because their binding constraint is the residual
    /// landscape rather than the balancing this removes. See `sciml::hardbc`.
    ///
    /// ⛔ A problem that supplies this must not be handed to [`run_time_marched`]: the constraint bakes in
    /// the `t = 0` initial condition, which is exactly what window two must NOT satisfy. `run_time_marched`
    /// refuses such a problem rather than silently solving the wrong thing.
    fn hard_constraint(&self, _ctx: &Arc<Context>, _x: &Var, _raw: &Var) -> Option<Var> {
        None
    }
    /// The initial state as a **function** of the input, for hard-constrained marching — window zero's
    /// start state. `None` unless the problem supports [`Problem::marched_constraint`].
    fn initial_value(&self, _ctx: &Arc<Context>, _x: &Var) -> Option<Var> {
        None
    }
    /// Wrap the raw output for one time-marched window so its **start state holds by construction**:
    /// `u = g(x) + (t − t₀)·B(x)·M(x,t)`, where `g` is the window's start state and `B` vanishes wherever a
    /// spatial condition is imposed. Returning `Some` tells [`run_time_marched`] to drop the start-state
    /// penalty, because there is nothing left to weight; a problem whose `B` also handles the walls can
    /// return an empty [`Problem::boundary_constraints`] and have no penalty terms at all.
    ///
    /// `start` evaluates the window's start state at any points — the problem's own
    /// [`Problem::initial_value`] in window zero, the previous window's frozen network after that. Its
    /// graph is differentiable, so `deriv` reaches through it and no derivative of `g` has to be carried
    /// by hand.
    ///
    /// ⭐ This is what solves the advection and burgers rows: `sciml::hardbc` measures **0.0251** and
    /// **0.0142** with it, against 0.9712 and 0.2434 for the best recipe with soft conditions.
    fn marched_constraint(&self, _ctx: &Arc<Context>, _x: &Var, _t0: f64, _start: &dyn Fn(&Var) -> Var, _raw: &Var) -> Option<Var> {
        None
    }
    fn reference(&self, x: &[f64]) -> f64;
    /// Fourier-feature scales that suit the problem's frequency content, per group and per INPUT AXIS
    /// (cycles per unit). One `σ` per axis, not one per group: an anisotropic solution — a travelling
    /// wave, anything with very different space and time frequencies — cannot be covered by a single
    /// number. See [`FourierNet::new_anisotropic`](super::features::FourierNet::new_anisotropic).
    fn scales(&self) -> Vec<Vec<f32>> {
        vec![vec![1.0; self.dim()], vec![3.0; self.dim()]]
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

impl NetImpl {
    fn params(&self) -> Vec<Tensor> {
        match self {
            NetImpl::Mlp(m) => m.params.clone(),
            NetImpl::Fourier(f) => f.params.clone(),
        }
    }
    fn forward(&self, pv: &[Var], x: &Var) -> Var {
        match self {
            NetImpl::Mlp(_) => Mlp::forward_act(pv, x, Act::Tanh),
            NetImpl::Fourier(f) => f.forward(pv, x),
        }
    }
}

fn build_net(ctx: &Arc<Context>, problem: &dyn Problem, recipe: &Recipe, d: usize, seed: u32) -> NetImpl {
    match &recipe.net {
        Net::TanhMlp { hidden } => {
            let mut dims = vec![d];
            dims.extend(hidden);
            dims.push(1);
            NetImpl::Mlp(Mlp::new(ctx, &dims, seed))
        }
        Net::Fourier { m_per_scale, hidden } => NetImpl::Fourier(FourierNet::new_anisotropic(ctx, *m_per_scale, &problem.scales(), hidden, 1, Act::Tanh, seed)),
    }
}

/// Train `recipe` on `problem` with `colloc` collocation points (`[N, dim]` flattened) and score it on a
/// `grid_m`-per-axis grid against the reference. Synchronous; the L-BFGS stage blocks on readbacks.
pub fn run(ctx: &Arc<Context>, problem: &dyn Problem, recipe: &Recipe, colloc: &[f32], grid_m: usize, seed: u32) -> Report {
    let t0 = Instant::now();
    let d = problem.dim();
    let n = colloc.len() / d;
    let net = build_net(ctx, problem, recipe, d, seed);
    let params0: Vec<Tensor> = net.params();
    let forward = |pv: &[Var], x: &Var| -> Var {
        let raw = net.forward(pv, x);
        problem.hard_constraint(ctx, x, &raw).unwrap_or(raw)
    };
    // does this problem impose its conditions structurally? asked once, on a single probe point, so the
    // flag and the wrapper above cannot disagree about it
    let hard = {
        let px = leaf(ctx, &colloc[..d], &[1, d]);
        let pv = vars(&params0);
        let raw = net.forward(&pv, &px);
        problem.hard_constraint(ctx, &px, &raw).is_some()
    };
    // per-point residual and constraint residual VECTORS; the loss takes their mean squares
    let terms = |pv: &[Var], it: u32| -> Vec<Var> {
        let fwd = |x: &Var| forward(pv, x);
        let xv = leaf(ctx, colloc, &[n, d]);
        let mut v = vec![problem.residual(ctx, &fwd, &xv, n)];
        if !hard {
            v.extend(problem.constraints(ctx, &fwd, 200, seed.wrapping_add(it)));
        }
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
    let n_terms = if hard { 1 } else { 1 + problem.constraints(ctx, &|x: &Var| forward(&vars(&wp), x), 8, 0).len() };
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

/// The start state for time-marched window `w`, as a differentiable function of the input.
///
/// ⛔ This is **recursive**, and it has to be. Window `w`'s start state is window `w−1`'s *solution* at the
/// boundary time — which is itself `g_{w−1}(x) + (t − t_{w−1})·raw_{w−1}`, not `raw_{w−1}`. A first version
/// evaluated the previous window's RAW network here; the start state was then wrong from window one onward,
/// each window inherited the error and amplified it, and advection scored **123.5** where soft marching
/// scores 0.3252. The tell was that it was worse than predicting nothing, by two orders of magnitude.
///
/// Cost is `O(w)` network evaluations at window `w`, which is the price of not carrying `g′` and `g″` by
/// hand; `deriv` reaches through the whole chain because every frozen parameter is a constant leaf.
#[allow(clippy::too_many_arguments)]
fn marched_start(
    ctx: &Arc<Context>,
    problem: &dyn Problem,
    recipe: &Recipe,
    d: usize,
    seed: u32,
    history: &[(Vec<Tensor>, f64)],
    w: usize,
    x: &Var,
) -> Var {
    if w == 0 {
        return problem
            .initial_value(ctx, x)
            .expect("a marched hard constraint needs Problem::initial_value for window zero");
    }
    let (pp, t_prev) = &history[w - 1];
    let pnet = build_net(ctx, problem, recipe, d, seed.wrapping_add((w - 1) as u32));
    let raw = pnet.forward(&vars(pp), x);
    let start = |xx: &Var| marched_start(ctx, problem, recipe, d, seed, history, w - 1, xx);
    problem.marched_constraint(ctx, x, *t_prev, &start, &raw).unwrap_or(raw)
}

/// **Time-marched training** (Krishnapriyan et al. 2021, arXiv 2109.01050 §5, "seq2seq") — the piece the
/// advection row of the benchmark table names, and a training *strategy* rather than a loss term.
///
/// A PINN asked to fit `sin(x − 30t)` over the whole of `t ∈ [0, 1]` must represent five wavelengths of a
/// travelling wave at once, and every recipe here failed at it (0.91–1.08, barely better than predicting
/// zero). Marching splits the horizon into `k` windows and solves them in order: window `w` sees only
/// `t ∈ [t_w, t_{w+1}]`, keeps the problem's spatial boundary conditions, and takes its **initial
/// condition from window `w−1`'s trained network**, evaluated on the slice `t = t_w` and frozen.
///
/// ⛔ **Each window keeps its own network, and that is not an optimisation.** Warm-starting one network
/// through the windows would end with a network that fits only the last one — the earlier windows are
/// trained away, which is exactly the catastrophic forgetting marching exists to avoid. Scoring therefore
/// evaluates each point with the network that owns its time slab.
pub fn run_time_marched(ctx: &Arc<Context>, problem: &dyn Problem, recipe: &Recipe, colloc: &[f32], windows: usize, grid_m: usize, seed: u32) -> Report {
    let t0 = Instant::now();
    let d = problem.dim();
    let ax = problem.time_axis().expect("time marching needs a time axis");
    let (lo, hi) = (problem.lo()[ax], problem.hi()[ax]);
    assert!(windows >= 1);
    // ⛔ A stationary hard constraint bakes in the t = 0 initial condition, which is exactly what window
    // two must NOT satisfy — it is handed its start state by window one. Silently solving that would give
    // a plausible-looking wrong answer, so refuse instead. (Marched hard constraints are a real technique
    // and they work — `sciml::hardbc` measures 0.0251 on advection and 0.0142 on burgers — but they need
    // the previous window's g, g′ and g″ threaded through this loop, which this harness does not do.)
    {
        let px = leaf(ctx, &colloc[..d], &[1, d]);
        let net = build_net(ctx, problem, recipe, d, seed);
        let raw = net.forward(&vars(&net.params()), &px);
        assert!(
            problem.hard_constraint(ctx, &px, &raw).is_none(),
            "run_time_marched cannot take a problem with a stationary hard constraint: it fixes u at t = 0, \
             which window two must not satisfy. See sciml::hardbc for the marched form."
        );
    }
    let edge = |w: usize| lo + (hi - lo) * w as f64 / windows as f64;

    let mut per_window: Vec<Vec<Tensor>> = Vec::with_capacity(windows);
    let mut prev: Option<(Vec<Tensor>, Vec<f32>, Vec<f32>)> = None; // params, slice points, frozen values
    let mut history: Vec<(Vec<Tensor>, f64)> = Vec::with_capacity(windows); // params and start time per window
    for w in 0..windows {
        let (a, b) = (edge(w), edge(w + 1));
        let pts: Vec<f32> = colloc
            .chunks(d)
            .filter(|p| (p[ax] as f64) >= a && (p[ax] as f64) <= b)
            .flatten()
            .copied()
            .collect();
        let n = pts.len() / d;
        assert!(n > 0, "window {w} has no collocation points");
        let net = build_net(ctx, problem, recipe, d, seed.wrapping_add(w as u32));
        let mut wp = net.params();
        // the window's start state as a differentiable function of x — the previous window's SOLUTION,
        // recursively (see `marched_start`), not its raw network
        let start_at = |x: &Var| -> Var { marched_start(ctx, problem, recipe, d, seed, &history, w, x) };
        let hard_marched = {
            let px = leaf(ctx, &pts[..d], &[1, d]);
            let raw = net.forward(&vars(&wp), &px);
            problem.marched_constraint(ctx, &px, a, &start_at, &raw).is_some()
        };
        let fwd_with = |pv: &[Var], x: &Var| -> Var {
            let raw = net.forward(pv, x);
            if hard_marched {
                problem.marched_constraint(ctx, x, a, &start_at, &raw).unwrap_or(raw)
            } else {
                raw
            }
        };
        pollster::block_on(async {
            let mut adam = Adam::new(&wp, recipe.lr);
            for it in 0..recipe.adam_steps {
                let pv = vars(&wp);
                let fwd = |x: &Var| fwd_with(&pv, x);
                let xv = leaf(ctx, &pts, &[n, d]);
                let mut loss = mse(&problem.residual(ctx, &fwd, &xv, n));
                for c in problem.boundary_constraints(ctx, &fwd, 200, seed.wrapping_add(it as u32)) {
                    loss = loss.add(&mse(&c).mul(&scalar(&c, 10.0)));
                }
                // the start state: structural if the problem supplies a marched constraint, otherwise a
                // penalty — the problem's own IC in window 0, the previous window's net after that
                if !hard_marched {
                    let start = match &prev {
                        None => problem.initial_constraint(ctx, &fwd, 200, seed.wrapping_add(it as u32)),
                        Some((_, sl, vals)) => {
                            let m = sl.len() / d;
                            Some(fwd(&leaf(ctx, sl, &[m, d])).sub(&leaf(ctx, vals, &[m, 1])))
                        }
                    };
                    if let Some(st) = start {
                        loss = loss.add(&mse(&st).mul(&scalar(&st, 100.0)));
                    }
                }
                step(ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
        });
        // hand this window's end state to the next, frozen
        if w + 1 < windows {
            let sl = problem.time_slice(400, b, seed.wrapping_add(7717 + w as u32)).expect("a time-dependent problem must give a slice");
            let m = sl.len() / d;
            let vals = pollster::block_on(async { fwd_with(&vars(&wp), &leaf(ctx, &sl, &[m, d])).value().to_vec().await });
            prev = Some((wp.clone(), sl, vals));
        }
        history.push((wp.clone(), a));
        per_window.push(wp);
    }

    // score: every grid point evaluated by the network that owns its slab
    let g = box_grid(&problem.lo(), &problem.hi(), grid_m);
    let ng = g.len() / d;
    let mut pred = vec![0.0f32; ng];
    for (w, wp) in per_window.iter().enumerate() {
        let (a, b) = (edge(w), edge(w + 1));
        let idx: Vec<usize> = (0..ng).filter(|&i| { let t = g[i * d + ax] as f64; t >= a && (t <= b || w + 1 == windows) && (w == 0 || t > a) }).collect();
        if idx.is_empty() {
            continue;
        }
        let sub: Vec<f32> = idx.iter().flat_map(|&i| g[i * d..(i + 1) * d].to_vec()).collect();
        let net = build_net(ctx, problem, recipe, d, seed.wrapping_add(w as u32));
        // ⛔ scored through the SAME wrapper it was trained through: scoring the raw network of a
        // hard-constrained run would grade a different function than the one that was fitted
        let start_at = |x: &Var| -> Var { marched_start(ctx, problem, recipe, d, seed, &history, w, x) };
        let out = pollster::block_on(async {
            let xv = leaf(ctx, &sub, &[idx.len(), d]);
            let raw = net.forward(&vars(wp), &xv);
            let u = problem.marched_constraint(ctx, &xv, a, &start_at, &raw).unwrap_or(raw);
            u.value().to_vec().await
        });
        for (k, &i) in idx.iter().enumerate() {
            pred[i] = out[k];
        }
    }
    let truth: Vec<f32> = g.chunks(d).map(|p| problem.reference(&p.iter().map(|&v| v as f64).collect::<Vec<_>>()) as f32).collect();
    Report { problem: problem.name().to_string(), recipe: "time-marched", n_colloc: colloc.len() / d, rel_l2: rel_l2(&pred, &truth), loss_after_adam: f32::NAN, loss_final: f32::NAN, secs: t0.elapsed().as_secs_f64() }
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
    fn boundary_constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let t: Vec<f32> = (0..n).map(|i| u01(i as u32, seed)).collect();
        let bl: Vec<f32> = t.iter().flat_map(|&t| [-1.0, t]).collect();
        let br: Vec<f32> = t.iter().flat_map(|&t| [1.0, t]).collect();
        vec![fwd(&leaf(ctx, &bl, &[n, 2])), fwd(&leaf(ctx, &br, &[n, 2]))]
    }
    fn initial_constraint(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Option<Var> {
        let xs: Vec<f32> = (0..n).map(|i| 2.0 * u01(i as u32, seed ^ 0x5151) - 1.0).collect();
        let ic: Vec<f32> = xs.iter().flat_map(|&x| [x, 0.0]).collect();
        let u0: Vec<f32> = xs.iter().map(|&x| -(std::f32::consts::PI * x).sin()).collect();
        Some(fwd(&leaf(ctx, &ic, &[n, 2])).sub(&col(ctx, &u0)))
    }
    fn time_slice(&self, n: usize, t: f64, seed: u32) -> Option<Vec<f32>> {
        Some((0..n).flat_map(|i| [2.0 * u01(i as u32, seed) - 1.0, t as f32]).collect())
    }
    fn time_axis(&self) -> Option<usize> {
        Some(1)
    }
    fn reference(&self, x: &[f64]) -> f64 {
        bench::burgers(x[0], x[1], self.nu)
    }
    /// `−sin πx` initially, steepening to a shock of width ~ν at `x = 0`: low in `t`, and a spread in `x`.
    fn scales(&self) -> Vec<Vec<f32>> {
        vec![vec![0.5, 0.5], vec![4.0, 1.0]]
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
    fn boundary_constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let s: Vec<f32> = (0..n).map(|i| 2.0 * u01(i as u32, seed) - 1.0).collect();
        let mut b = Vec::with_capacity(8 * n);
        for &v in &s {
            b.extend([-1.0, v, 1.0, v, v, -1.0, v, 1.0]);
        }
        vec![fwd(&leaf(ctx, &b, &[4 * n, 2]))]
    }
    // ⛔ Helmholtz does NOT supply a hard constraint here, though `sciml::hardbc` measures 0.0059 for one.
    // Turning `(1−x²)(1−y²)·M` on under THIS harness's recipes was measured and is WORSE: vanilla 6.3903
    // against 0.4766, full 0.8154 against 0.3066. The fixture's 0.0059 comes from the whole configuration
    // — cell-centred interior points, a 501-parameter tanh net, 15000 Adam steps at 3e-3 — and not from
    // the constraint alone. The recipes here sample 2000 random points (some arbitrarily close to the
    // edge, where the constraint's own factor vanishes), run 4000 steps at 1e-3, and give a Fourier net
    // scales chosen for `u` rather than for `u/(1−x²)(1−y²)`. The hook is available; enabling it is a
    // retuning job, not a switch.
    fn reference(&self, x: &[f64]) -> f64 {
        bench::helmholtz(x[0], x[1], self.a1, self.a2, self.k).0
    }
    /// `sin(a₁πx) sin(a₂πy)`: `a₁/2` and `a₂/2` cycles per unit on the two axes.
    fn scales(&self) -> Vec<Vec<f32>> {
        vec![vec![(self.a1 / 2.0) as f32, (self.a2 / 2.0) as f32], vec![self.a1 as f32, self.a2 as f32]]
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
    fn boundary_constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let t: Vec<f32> = (0..n).map(|i| u01(i as u32, seed)).collect();
        let b: Vec<f32> = t.iter().flat_map(|&t| [0.0, t, 1.0, t]).collect();
        vec![fwd(&leaf(ctx, &b, &[2 * n, 2]))]
    }
    fn initial_constraint(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Option<Var> {
        let xs: Vec<f32> = (0..n).map(|i| u01(i as u32, seed ^ 0x77)).collect();
        let ic: Vec<f32> = xs.iter().flat_map(|&x| [x, 0.0]).collect();
        let u0: Vec<f32> = xs.iter().map(|&x| (std::f32::consts::PI * x).sin()).collect();
        Some(fwd(&leaf(ctx, &ic, &[n, 2])).sub(&col(ctx, &u0)))
    }
    fn time_slice(&self, n: usize, t: f64, seed: u32) -> Option<Vec<f32>> {
        Some((0..n).flat_map(|i| [u01(i as u32, seed), t as f32]).collect())
    }
    fn time_axis(&self) -> Option<usize> {
        Some(1)
    }
    fn reference(&self, x: &[f64]) -> f64 {
        bench::heat(x[0], x[1])
    }
    /// `e^{−π²t} sin πx`: half a cycle per unit in `x`, and a decay in `t` slower still. The default
    /// isotropic `[1, 3]` measured 7× WORSE than a plain tanh net here.
    fn scales(&self) -> Vec<Vec<f32>> {
        vec![vec![0.5, 0.25], vec![1.0, 0.5]]
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
    fn boundary_constraints(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Vec<Var> {
        let t: Vec<f32> = (0..n).map(|i| u01(i as u32, seed)).collect();
        let l: Vec<f32> = t.iter().flat_map(|&t| [0.0, t]).collect();
        let r: Vec<f32> = t.iter().flat_map(|&t| [std::f32::consts::TAU, t]).collect();
        vec![fwd(&leaf(ctx, &l, &[n, 2])).sub(&fwd(&leaf(ctx, &r, &[n, 2])))]
    }
    fn initial_constraint(&self, ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, n: usize, seed: u32) -> Option<Var> {
        let xs: Vec<f32> = (0..n).map(|i| std::f32::consts::TAU * u01(i as u32, seed ^ 0x99)).collect();
        let ic: Vec<f32> = xs.iter().flat_map(|&x| [x, 0.0]).collect();
        let u0: Vec<f32> = xs.iter().map(|&x| x.sin()).collect();
        Some(fwd(&leaf(ctx, &ic, &[n, 2])).sub(&col(ctx, &u0)))
    }
    fn time_slice(&self, n: usize, t: f64, seed: u32) -> Option<Vec<f32>> {
        Some((0..n).flat_map(|i| [std::f32::consts::TAU * u01(i as u32, seed), t as f32]).collect())
    }
    fn time_axis(&self) -> Option<usize> {
        Some(1)
    }
    // ⛔ Advection does NOT supply `initial_value` / `marched_constraint` here, though `sciml::hardbc`
    // measures 0.0251 with marched hard conditions. Wiring them into THIS harness was measured and is
    // WORSE: 1.4835 against 0.3252 for soft marching, on the same 8 windows. The same verdict as the
    // stationary hook on helmholtz, for the same reason — the fixture's number comes from its whole
    // configuration (a fixed cell-centred grid shared by every window, a periodic sin/cos embedding that
    // makes the walls structural too, `g` frozen as data with analytic derivatives), while the harness
    // filters random points per window and inherits the previous window's error through a chain of eight
    // networks that the hard constraint forces it to match exactly. The hook is available and exercised by
    // `hardbc::the_harness_can_march_a_hard_constrained_problem`; turning it on for a benchmark row is a
    // retuning job, not a switch.
    fn reference(&self, x: &[f64]) -> f64 {
        bench::advection(x[0], x[1], self.beta)
    }
    /// ⭐ The exact solution `sin(x − βt)` is a plane wave with wavevector `(1, β)` rad/unit, i.e.
    /// `(1/2π, β/2π)` cycles per unit — which is `(0.16, 4.8)` at `β = 30`. A single `σ` cannot be both,
    /// and the isotropic recipe scored 1.08 here, worse than predicting zero.
    fn scales(&self) -> Vec<Vec<f32>> {
        let tau = std::f32::consts::TAU;
        vec![vec![1.0 / tau, self.beta as f32 / tau], vec![2.0 / tau, 2.0 * self.beta as f32 / tau]]
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
        let reports = recipes
            .iter()
            .map(|r| {
                let rep = run(&ctx, p, r, &colloc, 51, seed);
                let verdict = if rep.rel_l2 < 0.05 { "solved" } else if rep.rel_l2 < 1.0 { "partial" } else { "UNSOLVED (worse than zero)" };
                eprintln!("  {:<12} {:<24} rel-L2 {:.4}  {:<26} ({:.0}s, loss {:.1e} -> {:.1e})", p.name(), rep.recipe, rep.rel_l2, verdict, rep.secs, rep.loss_after_adam, rep.loss_final);
                assert!(rep.rel_l2.is_finite(), "{} produced a non-finite score", rep.recipe);
                rep
            })
            .collect::<Vec<_>>();
        if reports.iter().all(|r| r.rel_l2 >= 0.2) {
            eprintln!("  ⚠ every arm failed on {} — this comparison does not rank the recipes", p.name());
        }
        reports
    }

    /// ⚠ **Helmholtz, where the table said NTK weighting was the missing piece — and the comparison does
    /// NOT discriminate.** Gradient-norm 0.4781, NTK 0.5183: NTK is marginally worse, and both are far
    /// from solved at 4000 steps. Two failing arms cannot rank two techniques, which is the same rule
    /// that sent the balancing and causal fixtures in `oracles.rs` back to be re-sized. What this row
    /// says is that NTK weighting does not rescue Helmholtz at THIS budget; the paper's own result uses
    /// ten times the steps. Making one arm succeed first is the open work.
    ///
    /// ⭐ The technique-level evidence lives in `oracles.rs`, where each piece has a fixture its control
    /// measurably fails. A bundled recipe on an arbitrary PDE is a different, weaker claim.
    #[ignore = "two trainings on the GPU (~14 min); run with -- --ignored"]
    #[test]
    fn helmholtz_gradnorm_against_ntk_weighting() {
        compare(&Helmholtz { a1: 1.0, a2: 4.0, k: 1.0 }, &[Recipe::full(), Recipe::ntk()], 2000, 1);
    }

    /// ⭐⭐ **Advection at β = 30 by TIME MARCHING** — the piece the table named after weighting failed.
    /// One global fit against eight windows solved in order, each with its own network, each handed its
    /// start state by the previous one. Same net size, same per-window step budget.
    #[ignore = "nine trainings on the GPU (~20 min); run with -- --ignored"]
    #[test]
    fn advection_by_time_marching_against_one_global_fit() {
        let ctx = ctx();
        let p = Advection { beta: 30.0 };
        let colloc = box_points(&p.lo(), &p.hi(), 4000, 7);
        let one = run(&ctx, &p, &Recipe::full(), &colloc, 51, 1);
        let marched = Recipe { adam_steps: 1500, lbfgs_iters: 0, ..Recipe::full() };
        let many = run_time_marched(&ctx, &p, &marched, &colloc, 8, 51, 1);
        eprintln!("  advection β=30: one global fit {:.4} ({:.0}s)   8 windows marched {:.4} ({:.0}s)", one.rel_l2, one.secs, many.rel_l2, many.secs);
        assert!(one.rel_l2.is_finite() && many.rel_l2.is_finite());
        let verdict = |r: f32| if r < 0.05 { "solved" } else if r < 1.0 { "partial" } else { "UNSOLVED" };
        eprintln!("    global {} / marched {}", verdict(one.rel_l2), verdict(many.rel_l2));
    }

    /// ⚠ **Advection at β = 30, where the table said causal weighting was the missing piece — and again
    /// the comparison does NOT discriminate.** 1.0815 without, 1.0935 with: both arms are worse than
    /// predicting zero, so neither is solving anything and the ordering is noise. Note what this does
    /// NOT contradict — `causal_training_finds_the_solution_a_vanilla_pinn_cannot` shows the same
    /// machinery taking the reaction equation from 0.937 to 0.094, with a control that fails. Causal
    /// weighting works where the arrow of time is the binding constraint; at β = 30 with these Fourier
    /// scales something upstream is failing first, and finding it is the open work.
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
            let net = FourierNet::new_anisotropic(&ctx, 32, &p.scales(), &[64, 64], 1, Act::Tanh, 1);
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
