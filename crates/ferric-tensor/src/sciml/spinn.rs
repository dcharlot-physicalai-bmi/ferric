//! **Separable PINN (SPINN)** — Cho et al., *Separable Physics-Informed Neural Networks* (2306.15969).
//!
//! A dense PINN on a `d`-dimensional tensor grid of `N` points per axis evaluates its network at `N^d`
//! collocation points. A separable one factors the field into a rank-`r` sum of products of **one-dimensional**
//! networks,
//!
//! ```text
//! u(x₁, …, x_d) = Σ_{j=1}^{r} Π_{i=1}^{d} f_i(x_i)_j
//! ```
//!
//! so the same `N^d` grid costs `d·N` network evaluations, not `N^d` — at `d = 3, N = 32` that is 96 against
//! 32,768. The residual still sees every grid point, because the product is formed as a tensor contraction
//! *after* the networks have run.
//!
//! ⚠ The saving is real and so is its price: a rank-`r` product ansatz can only represent a field of
//! separation rank ≤ `r`. That is a modelling choice, not a free lunch, and
//! `raising_the_separable_rank_monotonically_improves_the_fit` measures what it could of that, and says
//! plainly what it could not separate.

use super::{Act, Mlp};
use crate::{grad, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// **∂y/∂x for a map that is elementwise in a one-dimensional input** — `x` is `[N, 1]`, `y` is `[N, r]`,
/// and row `n` of `y` depends only on `x[n]`. Returns `[N, r]`.
///
/// Reverse mode gives one *row* of a Jacobian per pass, so the obvious route costs `r` backward passes —
/// one per output column. This is the reverse-over-reverse JVP instead: differentiate `uᵀ y` with respect
/// to `x` (giving `Jᵀu`, a graph that is linear in the cotangent `u`), then differentiate *that* with
/// respect to `u`. Two passes, whatever `r` is. `u` is zeros because only the graph's dependence on it
/// matters, not its value.
///
/// This is the piece that makes a separable PINN affordable here: the fabric has no forward mode, which is
/// what the original uses (`jax.jvp`) for exactly this reason.
pub fn jvp_1d(y: &Var, x: &Var) -> Var {
    let sh = y.value().shape.clone();
    assert_eq!(sh.len(), 2, "jvp_1d wants y as [N, r]");
    assert_eq!(x.value().shape, vec![sh[0], 1], "jvp_1d wants x as [N, 1]");
    let u = Var::leaf(Tensor::zeros(&x.value().ctx_arc(), &sh));
    let jtu = grad(&y.mul(&u).sum_all(), core::slice::from_ref(x), None).remove(0);
    grad(&jtu.sum_all(), core::slice::from_ref(&u), None).remove(0)
}

/// A rank-`r` separable field on `d` axes: one 1-D network per axis, combined by a tensor contraction.
pub struct Spinn {
    pub dim: usize,
    pub rank: usize,
    pub act: Act,
    /// All axes' parameters, concatenated; axis `k` owns `per_axis` of them starting at `k * per_axis`.
    pub params: Vec<Tensor>,
    pub per_axis: usize,
}

impl Spinn {
    /// `dim` axes, each a `[1, hidden, …, rank]` MLP with `layers` hidden layers.
    pub fn new(ctx: &Arc<Context>, dim: usize, hidden: usize, layers: usize, rank: usize, act: Act, seed: u32) -> Self {
        assert!(dim >= 1 && rank >= 1 && layers >= 1);
        let mut dims = vec![1usize];
        dims.extend(std::iter::repeat_n(hidden, layers));
        dims.push(rank);
        let mut params = Vec::new();
        let mut per_axis = 0;
        for k in 0..dim {
            let net = Mlp::new(ctx, &dims, seed.wrapping_add(1 + k as u32 * 131));
            per_axis = net.params.len();
            params.extend(net.params);
        }
        Spinn { dim, rank, act, params, per_axis }
    }

    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }

    /// Per-axis features `[N_k, rank]` for the axis coordinates `xs[k]` (each `[N_k, 1]`).
    pub fn features(&self, pv: &[Var], xs: &[Var]) -> Vec<Var> {
        assert_eq!(xs.len(), self.dim, "one coordinate column per axis");
        (0..self.dim)
            .map(|k| Mlp::forward_act(&pv[k * self.per_axis..(k + 1) * self.per_axis], &xs[k], self.act))
            .collect()
    }

    /// Contract per-axis features into the field on the full tensor grid, shape `[N_0, N_1, …, N_{d−1}]`.
    /// This is where the `N^d` points appear — after the networks have run, not before.
    pub fn combine(feats: &[Var]) -> Var {
        let r = feats[0].value().shape[1];
        let mut grid: Vec<usize> = vec![feats[0].value().shape[0]];
        let mut acc = feats[0].clone();
        for f in &feats[1..] {
            let p = acc.value().shape[0];
            let nk = f.value().shape[0];
            acc = acc.reshape(&[p, 1, r]).mul(&f.reshape(&[1, nk, r])).reshape(&[p * nk, r]);
            grid.push(nk);
        }
        acc.sum(&[1]).reshape(&grid)
    }

    /// The field's derivative along axis `k`: the same contraction with that axis's features replaced by
    /// their derivative, since `∂/∂x_k Σ_j Π_i f_i = Σ_j f_k′ Π_{i≠k} f_i`.
    pub fn combine_with(feats: &[Var], k: usize, replacement: &Var) -> Var {
        let mut v = feats.to_vec();
        v[k] = replacement.clone();
        Self::combine(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::util::{dcol, leaf, rel_l2, step, u01, vars};

    /// ⭐ **The reverse-over-reverse JVP against central differences**, to second order. If this is wrong
    /// every separable residual is wrong, and nothing downstream would say so — a PDE residual built on a
    /// bad derivative still trains, to the wrong thing.
    #[test]
    fn the_one_dimensional_jvp_matches_finite_differences_to_second_order() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, r) = (7usize, 5usize);
            let net = Mlp::new(&ctx, &[1, 9, r], 4);
            let xs: Vec<f32> = (0..n).map(|i| -0.8 + 1.7 * u01(i as u32, 12)).collect();
            let eval = |v: &[f32], order: usize| {
                let pv = net.vars();
                let x = leaf(&ctx, v, &[v.len(), 1]);
                let y = Mlp::forward_act(&pv, &x, Act::Tanh);
                match order {
                    0 => y,
                    1 => jvp_1d(&y, &x),
                    _ => jvp_1d(&jvp_1d(&y, &x), &x),
                }
            };
            let eps = 1e-3f32;
            for order in [1usize, 2] {
                let got = eval(&xs, order).value().to_vec().await;
                let (mut up, mut dn) = (xs.clone(), xs.clone());
                let mut worst = 0.0f32;
                for i in 0..n {
                    up[i] += eps;
                    dn[i] -= eps;
                    let (hi, lo) = (eval(&up, order - 1).value().to_vec().await, eval(&dn, order - 1).value().to_vec().await);
                    up[i] = xs[i];
                    dn[i] = xs[i];
                    for j in 0..r {
                        let fd = (hi[i * r + j] - lo[i * r + j]) / (2.0 * eps);
                        let e = (fd - got[i * r + j]).abs();
                        assert!(e < 2e-2 * (1.0 + fd.abs()), "order {order}, point {i}, column {j}: jvp {} vs finite difference {fd}", got[i * r + j]);
                        worst = worst.max(e);
                    }
                }
                eprintln!("  jvp_1d order {order} on a [1,9,{r}] tanh net: worst |jvp − finite difference| {worst:.2e}");
            }
        });
    }


    /// The manufactured 3-D Poisson problem: `−Δu = f` on the unit cube with `u = 0` on the boundary,
    /// `u* = Σ_m a_m sin(mπx) sin(mπy) sin(mπz)`. Computed in Rust from the closed form — it touches no
    /// network. Returns `(u*, f)` on the `n³` grid, row-major.
    fn poisson3d(grid: &[f32], a: &[f64; 2]) -> (Vec<f32>, Vec<f32>) {
        let n = grid.len();
        let pi = std::f64::consts::PI;
        let (mut u, mut f) = (vec![0.0f64; n * n * n], vec![0.0f64; n * n * n]);
        for (m, &am) in a.iter().enumerate() {
            let mp = (m + 1) as f64 * pi;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let s = (mp * grid[i] as f64).sin() * (mp * grid[j] as f64).sin() * (mp * grid[k] as f64).sin();
                        let idx = (i * n + j) * n + k;
                        u[idx] += am * s;
                        f[idx] += 3.0 * mp * mp * am * s; // −Δ[sin·sin·sin] = 3(mπ)² sin·sin·sin
                    }
                }
            }
        }
        (u.iter().map(|&v| v as f32).collect(), f.iter().map(|&v| v as f32).collect())
    }

    /// ⭐⭐ **The structural claim, measured on its own: a separable residual costs less per step.** One
    /// graph build plus one `backward()` of the same 3-D Poisson residual on the same `n³` grid, with no
    /// training in the loop — so this measures the cost of the formulation and nothing else. Both arms
    /// carry the same hard boundary constraint `Π x_k(1−x_k)`, which is itself separable.
    ///
    /// ⚠ This test deliberately makes **no accuracy claim**. A cost ratio and a quality ratio are separate
    /// questions and a single fixture cannot answer both; the accuracy of the separable arm is asserted by
    /// `a_separable_pinn_solves_3d_poisson`, against ground truth rather than against the other arm.
    #[ignore = "a wall-clock measurement on the GPU (~2 min); run with -- --ignored --nocapture"]
    #[test]
    fn a_separable_residual_costs_less_per_step_than_a_dense_one() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let iters = 20;
            let mut ratios = Vec::new();
            for n in [8usize, 16, 32] {
                let m = n * n * n;
                let grid: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
                let (_, fv) = poisson3d(&grid, &[1.0, 0.5]);
                let one = leaf(&ctx, &vec![1.0f32; n], &[n, 1]);
                let net = Spinn::new(&ctx, 3, 32, 2, 16, Act::Tanh, 7);
                let sep_step = || {
                    let pv = net.vars();
                    let xs: Vec<Var> = (0..3).map(|_| leaf(&ctx, &grid, &[n, 1])).collect();
                    let raw = net.features(&pv, &xs);
                    let feats: Vec<Var> = raw.iter().zip(&xs).map(|(f, x)| f.mul(&x.mul(&one.sub(x)))).collect();
                    let mut lap: Option<Var> = None;
                    for k in 0..3 {
                        let d2 = jvp_1d(&jvp_1d(&feats[k], &xs[k]), &xs[k]);
                        let t = Spinn::combine_with(&feats, k, &d2);
                        lap = Some(match lap { None => t, Some(l) => l.add(&t) });
                    }
                    let r = lap.unwrap().reshape(&[m, 1]).add(&leaf(&ctx, &fv, &[m, 1]));
                    r.mul(&r).mean_all()
                };
                let mut pts = vec![0.0f32; m * 3];
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            let idx = (i * n + j) * n + k;
                            pts[idx * 3] = grid[i];
                            pts[idx * 3 + 1] = grid[j];
                            pts[idx * 3 + 2] = grid[k];
                        }
                    }
                }
                let dense = Mlp::new(&ctx, &[3, 40, 40, 40, 1], 7);
                let ones_m = leaf(&ctx, &vec![1.0f32; m], &[m, 1]);
                let dense_step = || {
                    let pv = dense.vars();
                    let x = leaf(&ctx, &pts, &[m, 3]);
                    let mut bc = ones_m.clone();
                    for k in 0..3 {
                        let mut e = vec![0.0f32; 3];
                        e[k] = 1.0;
                        let xk = x.matmul(&leaf(&ctx, &e, &[3, 1]));
                        bc = bc.mul(&xk.mul(&ones_m.sub(&xk)));
                    }
                    let u = bc.mul(&Mlp::forward_act(&pv, &x, Act::Tanh));
                    let g = crate::sciml::deriv(&u, &x);
                    let mut lap: Option<Var> = None;
                    for k in 0..3 {
                        let mut mask = vec![0.0f32; m * 3];
                        for i in 0..m { mask[i * 3 + k] = 1.0; }
                        let gk = g.mul(&leaf(&ctx, &mask, &[m, 3])).sum(&[1]).reshape(&[m, 1]);
                        let t = dcol(&ctx, &gk, &x, m, 3, k);
                        lap = Some(match lap { None => t, Some(l) => l.add(&t) });
                    }
                    let r = lap.unwrap().add(&leaf(&ctx, &fv, &[m, 1]));
                    r.mul(&r).mean_all()
                };
                // warm up, then time graph-build + backward
                for f in [0usize, 1] {
                    let l = if f == 0 { sep_step() } else { dense_step() };
                    l.backward();
                    let _ = l.value().to_vec().await;
                }
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let l = sep_step();
                    l.backward();
                    let _ = l.value().to_vec().await;
                }
                let ts = t0.elapsed().as_secs_f64() / iters as f64;
                let t1 = std::time::Instant::now();
                for _ in 0..iters {
                    let l = dense_step();
                    l.backward();
                    let _ = l.value().to_vec().await;
                }
                let td = t1.elapsed().as_secs_f64() / iters as f64;
                eprintln!(
                    "  n={n:3} ({m:6} grid points): separable {:7.1} ms/step from {:4} network rows   dense {:7.1} ms/step from {m:6} rows   {:.1}x cheaper (rows {:.0}x fewer)",
                    ts * 1e3, 3 * n, td * 1e3, td / ts, m as f64 / (3 * n) as f64
                );
                ratios.push(td / ts);
            }
            eprintln!("  ratios (dense/separable) across the sweep: {ratios:?}");
            // ⚠ Measured on an IDLE machine: 0.99x, 2.81x, 11.4x — separable 61.6/93.5/114.3 ms against
            // dense 61.2/262.5/1299.0 ms. The separable arm's cost grows slowly with the grid, the dense
            // arm's tracks it. At n = 8 both sit on the dispatch floor and the 21x fewer network rows buy
            // nothing, so the advantage is asserted as one that APPEARS with scale, not as a blanket win.
            // ⛔ An earlier run of this same test, with other cargo jobs sharing the GPU, read
            // 0.97x/0.90x/2.82x and would have put the crossover a full octave too high. Timing here is
            // only meaningful on an otherwise idle machine.
            assert!(ratios[2] > 2.0, "at the largest grid the separable residual must be materially cheaper: {ratios:?}");
            assert!(ratios[2] > 2.0 * ratios[0], "and the advantage must be one that APPEARS with scale: {ratios:?}");
        });
    }

    /// ⭐⭐ **The separable PINN solves 3-D Poisson**, against the closed-form solution — not against
    /// another arm. `16³ = 4096` collocation points reached from `3 × 16 = 48` network rows per step.
    #[ignore = "trains a separable 3-D PINN on the GPU (~8 min); run with -- --ignored"]
    #[test]
    fn a_separable_pinn_solves_3d_poisson() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, rank, steps) = (16usize, 16usize, 2500u32);
            let m = n * n * n;
            let grid: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
            let (ustar, fv) = poisson3d(&grid, &[1.0, 0.5]);
            let one = leaf(&ctx, &vec![1.0f32; n], &[n, 1]);
            let net = Spinn::new(&ctx, 3, 32, 2, rank, Act::Tanh, 7);
            let mut wp = net.params.clone();
            let mut adam = crate::Adam::new(&wp, 3e-3);
            let field = |pv: &[Var]| {
                let xs: Vec<Var> = (0..3).map(|_| leaf(&ctx, &grid, &[n, 1])).collect();
                let raw = net.features(pv, &xs);
                // the hard boundary constraint is itself separable, so it rides inside each axis
                let feats: Vec<Var> = raw.iter().zip(&xs).map(|(f, x)| f.mul(&x.mul(&one.sub(x)))).collect();
                let mut lap: Option<Var> = None;
                for k in 0..3 {
                    let d2 = jvp_1d(&jvp_1d(&feats[k], &xs[k]), &xs[k]);
                    let t = Spinn::combine_with(&feats, k, &d2);
                    lap = Some(match lap { None => t, Some(l) => l.add(&t) });
                }
                (Spinn::combine(&feats), lap.unwrap())
            };
            for ep in 0..steps {
                if ep == steps * 3 / 4 {
                    adam = crate::Adam::new(&wp, 3e-4);
                }
                let pv = vars(&wp);
                let (_, lap) = field(&pv);
                let r = lap.reshape(&[m, 1]).add(&leaf(&ctx, &fv, &[m, 1]));
                let loss = r.mul(&r).mean_all();
                step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
            let (u, _) = field(&vars(&wp));
            let e = rel_l2(&u.value().to_vec().await, &ustar);
            eprintln!("  separable PINN, {n}³ = {m} collocation points from {} network rows/step, rank {rank}: rel-L2 vs the exact solution {e:.4}", 3 * n);
            assert!(e < 0.01, "the separable PINN must solve 3-D Poisson: {e:.4}");
        });
    }

    /// ⚠ **Raising the separable rank monotonically improves the fit** — the ansatz's price, as far as
    /// this could be measured. A rank-`r` sum of products represents a field of separation rank `R` only
    /// when `r ≥ R`; the target here is five shifted-Legendre products scaled to equal energy, so its
    /// separation rank is exactly 5, and it is fitted by plain regression at four ranks with everything
    /// else held fixed.
    ///
    /// ⛔ **What this fixture does NOT show.** The clean signature of a rank limit would be an error that
    /// falls to near zero at `r = R` and then flattens. It does not: measured 0.862 / 0.703 / 0.348 /
    /// 0.289 at ranks 1 / 2 / 5 / 12, still falling past the field's own rank. So rank is *not* cleanly
    /// separated from optimisation difficulty here — fitting a sum of products is a nonconvex tensor
    /// factorisation, and it is plainly not being solved to optimality at any rank. Two earlier targets
    /// failed worse and for different reasons: `sin(mπ·)` up to `m = 5` stalled at 0.185 because the 1-D
    /// tanh nets could not fit `sin(5πx)` (spectral bias, not rank), and Legendre terms weighted `1/(q+1)`
    /// left ranks 1 and 2 scoring *identically* 0.0990 because the `q = 0` term carried nearly all the
    /// energy. What survives is the monotone dependence on rank, which is evidence that rank binds; the
    /// claim is kept to that.
    #[ignore = "fits four separable ansätze on the GPU (~11 min); run with -- --ignored"]
    #[test]
    fn raising_the_separable_rank_monotonically_improves_the_fit() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, target_rank, steps) = (10usize, 5usize, 2500u32);
            let m = n * n * n;
            let grid: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
            // shifted Legendre P_q(2x−1), q = 0..4: linearly independent, and all low-frequency
            let leg = |q: usize, x: f32| -> f64 {
                let t = 2.0 * x as f64 - 1.0;
                match q {
                    0 => 1.0,
                    1 => t,
                    2 => (3.0 * t * t - 1.0) / 2.0,
                    3 => (5.0 * t * t * t - 3.0 * t) / 2.0,
                    _ => (35.0 * t.powi(4) - 30.0 * t * t + 3.0) / 8.0,
                }
            };
            let mut target = vec![0.0f64; m];
            for q in 0..target_rank {
                // equal ENERGY per term: ‖P_q‖² = 1/(2q+1) on [0,1], so the triple product carries
                // (2q+1)^{-3/2} without this. ⛔ With amplitudes 1/(q+1) the q = 0 term dominated and rank 1
                // and rank 2 both scored 0.0990 — the sweep measured nothing.
                let amp = ((2 * q + 1) as f64).powf(1.5);
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            target[(i * n + j) * n + k] += amp * leg(q, grid[i]) * leg(q, grid[j]) * leg(q, grid[k]);
                        }
                    }
                }
            }
            let tv: Vec<f32> = target.iter().map(|&v| v as f32).collect();
            let tvar = leaf(&ctx, &tv, &[m, 1]);
            let mut errs = Vec::new();
            for &r in &[1usize, 2, 5, 12] {
                let net = Spinn::new(&ctx, 3, 32, 2, r, Act::Tanh, 3);
                let mut wp = net.params.clone();
                let mut adam = crate::Adam::new(&wp, 5e-3);
                for ep in 0..steps {
                    if ep == steps * 2 / 3 {
                        adam = crate::Adam::new(&wp, 5e-4);
                    }
                    let pv = vars(&wp);
                    let xs: Vec<Var> = (0..3).map(|_| leaf(&ctx, &grid, &[n, 1])).collect();
                    let u = Spinn::combine(&net.features(&pv, &xs)).reshape(&[m, 1]);
                    let d = u.sub(&tvar);
                    let loss = d.mul(&d).mean_all();
                    step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                }
                let pv = vars(&wp);
                let xs: Vec<Var> = (0..3).map(|_| leaf(&ctx, &grid, &[n, 1])).collect();
                let u = Spinn::combine(&net.features(&pv, &xs)).reshape(&[m, 1]).value().to_vec().await;
                let e = rel_l2(&u, &tv);
                eprintln!("  fitting a separation-rank-{target_rank} field with a rank-{r} ansatz: rel-L2 {e:.4}");
                errs.push(e);
            }
            assert!(errs[0] > 0.5, "a rank-1 ansatz must visibly fail on a rank-{target_rank} field, or this measures nothing: {:.4}", errs[0]);
            assert!(errs.windows(2).all(|w| w[1] < w[0]), "the fit must improve at every rank increase: {errs:?}");
            assert!(errs[2] < 0.5 * errs[0], "reaching the field's own rank must help substantially: {errs:?}");
        });
    }

    /// ⛔ The cheap route to the same derivative is `r` reverse passes with one-hot seeds. That is what the
    /// JVP must reproduce — a second, independent computation of the same quantity, so a shared mistake in
    /// the shapes cannot make both agree.
    #[test]
    fn the_jvp_agrees_with_one_reverse_pass_per_column() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, r) = (6usize, 4usize);
            let net = Mlp::new(&ctx, &[1, 8, r], 9);
            let xv: Vec<f32> = (0..n).map(|i| -1.0 + 2.0 * u01(i as u32, 77)).collect();
            let pv = net.vars();
            let x = leaf(&ctx, &xv, &[n, 1]);
            let y = Mlp::forward_act(&pv, &x, Act::Tanh);
            let fast = jvp_1d(&y, &x).value().to_vec().await;
            let mut slow = vec![0.0f32; n * r];
            for j in 0..r {
                let mut seed = vec![0.0f32; n * r];
                for i in 0..n {
                    seed[i * r + j] = 1.0;
                }
                let pv = net.vars();
                let x = leaf(&ctx, &xv, &[n, 1]);
                let y = Mlp::forward_act(&pv, &x, Act::Tanh);
                let g = grad(&y, core::slice::from_ref(&x), Some(&leaf(&ctx, &seed, &[n, r]))).remove(0).value().to_vec().await;
                for i in 0..n {
                    slow[i * r + j] = g[i];
                }
            }
            let e = rel_l2(&fast, &slow);
            eprintln!("  one double-backward vs {r} seeded reverse passes: rel-L2 {e:.2e}");
            assert!(e < 1e-5, "the two routes to ∂y/∂x must agree: {e:.2e}");
        });
    }

    /// The contraction builds the field the definition says it does: `u = Σ_j Π_i f_i(x_i)_j`, checked
    /// against the same sum written as explicit loops in Rust over the read-back features.
    #[test]
    fn the_contraction_is_the_sum_of_products_it_claims_to_be() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (dim, rank) = (3usize, 4usize);
            let ns = [3usize, 4, 5];
            let net = Spinn::new(&ctx, dim, 8, 1, rank, Act::Tanh, 21);
            let pv = net.vars();
            let xs: Vec<Var> = (0..dim)
                .map(|k| {
                    let v: Vec<f32> = (0..ns[k]).map(|i| u01((k * 17 + i) as u32, 5) - 0.5).collect();
                    leaf(&ctx, &v, &[ns[k], 1])
                })
                .collect();
            let feats = net.features(&pv, &xs);
            let u = Spinn::combine(&feats);
            assert_eq!(u.value().shape, ns.to_vec(), "the field lives on the full tensor grid");
            let mut fv = Vec::new();
            for f in &feats {
                fv.push(f.value().to_vec().await);
            }
            let got = u.value().to_vec().await;
            let mut worst = 0.0f32;
            for i in 0..ns[0] {
                for j in 0..ns[1] {
                    for k in 0..ns[2] {
                        let mut want = 0.0f32;
                        for q in 0..rank {
                            want += fv[0][i * rank + q] * fv[1][j * rank + q] * fv[2][k * rank + q];
                        }
                        let idx = (i * ns[1] + j) * ns[2] + k;
                        worst = worst.max((got[idx] - want).abs());
                    }
                }
            }
            eprintln!("  {dim}-D rank-{rank} contraction on a {}×{}×{} grid from {} network rows: worst |tensor − loop| {worst:.2e}", ns[0], ns[1], ns[2], ns.iter().sum::<usize>());
            assert!(worst < 1e-5, "the contraction must be the sum of products: {worst:.2e}");
        });
    }
}
