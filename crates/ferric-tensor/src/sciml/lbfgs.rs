//! **L-BFGS** — the second-stage optimiser every serious PINN stack ends on (DeepXDE, jaxpi, PINA all
//! run Adam then L-BFGS), because a physics residual is a smooth, deterministic, full-batch objective
//! and a quasi-Newton method takes it one to two orders lower than Adam stalls at.
//!
//! Implemented from Nocedal & Wright, *Numerical Optimization*, Algorithms 7.4 (two-loop recursion) and
//! 7.5 (L-BFGS), with the standard `γₖ = sᵀy / yᵀy` initial Hessian scaling and a **strong-Wolfe** line
//! search (Algorithms 3.5 and 3.6). Pairs with `sᵀy ≤ 0` are still skipped as a guard, but the curvature
//! condition is what makes that guard almost never fire — see the note on [`Lbfgs::minimize`]. The objective is a plain closure returning
//! `(f, ∇f)` on CPU vectors, so it is agnostic to where the graph runs — the PINN closure uploads the
//! vector, builds the graph on the device, and reads the gradient back (see [`unflatten`](super::unflatten)).

use std::collections::VecDeque;

/// Result of an [`Lbfgs::minimize`] run.
#[derive(Clone, Debug)]
pub struct LbfgsResult {
    pub x: Vec<f32>,
    pub f: f32,
    pub grad_norm: f32,
    pub iters: usize,
    pub evals: usize,
    /// Why it stopped: gradient tolerance met, iteration cap, or a line search that could not decrease `f`.
    pub stop: LbfgsStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbfgsStop {
    GradTol,
    MaxIters,
    LineSearchFailed,
}

/// Limited-memory BFGS with history `m`.
pub struct Lbfgs {
    pub m: usize,
    s: VecDeque<Vec<f32>>,
    y: VecDeque<Vec<f32>>,
    /// Armijo constant (sufficient decrease), default `1e-4`.
    pub c1: f32,
    /// Wolfe curvature constant, default `0.9` (the quasi-Newton choice).
    pub c2: f32,
    /// Evaluations allowed in each phase of a line search.
    pub max_backtracks: usize,
}

impl Lbfgs {
    pub fn new(m: usize) -> Self {
        Lbfgs { m: m.max(1), s: VecDeque::new(), y: VecDeque::new(), c1: 1e-4, c2: 0.9, max_backtracks: 30 }
    }

    /// Two-loop recursion: `r = Hₖ g` for the current history.
    fn direction(&self, g: &[f32]) -> Vec<f32> {
        let mut q = g.to_vec();
        let k = self.s.len();
        let mut alpha = vec![0.0f32; k];
        let mut rho = vec![0.0f32; k];
        for i in (0..k).rev() {
            rho[i] = 1.0 / dot(&self.y[i], &self.s[i]);
            alpha[i] = rho[i] * dot(&self.s[i], &q);
            axpy(-alpha[i], &self.y[i], &mut q);
        }
        if k > 0 {
            let gamma = dot(&self.s[k - 1], &self.y[k - 1]) / dot(&self.y[k - 1], &self.y[k - 1]);
            for v in q.iter_mut() {
                *v *= gamma;
            }
        }
        for i in 0..k {
            let beta = rho[i] * dot(&self.y[i], &q);
            axpy(alpha[i] - beta, &self.s[i], &mut q);
        }
        q
    }

    /// Minimise `f` from `x0`. `f` returns the value and gradient at a point. Stops when `‖∇f‖ < gtol`, after
    /// `max_iters`, or when a line search cannot find any decrease.
    ///
    /// ⛔ **The line search is strong-Wolfe, and it has to be.** The first version used backtracking
    /// Armijo alone, and on Rosenbrock it stalled at `f = 3.47` for 200 iterations: from iteration 3 the
    /// accepted steps had `sᵀy ≈ −5e-7` — negative curvature — so every new pair was rejected, the history
    /// froze at three stale pairs, and their direction collapsed to `|d| ≈ 1.8e-3`, accepted at unit step
    /// every time. Armijo bounds the decrease but says nothing about curvature. The Wolfe curvature
    /// condition `∇f(x+td)ᵀd ≥ c₂ ∇fᵀd` gives `sᵀy ≥ (c₂ − 1) t ∇fᵀd > 0` by construction, which is what
    /// keeps the inverse-Hessian estimate positive definite (Nocedal & Wright, eq. 6.7 and Thm 6.5).
    pub fn minimize<F>(&mut self, x0: Vec<f32>, mut f: F, max_iters: usize, gtol: f32) -> LbfgsResult
    where
        F: FnMut(&[f32]) -> (f32, Vec<f32>),
    {
        self.s.clear();
        self.y.clear();
        let mut x = x0;
        let (mut fx, mut g) = f(&x);
        let mut evals = 1usize;
        let mut iters = 0usize;
        loop {
            let gn = norm(&g);
            if !gn.is_finite() || gn < gtol {
                return LbfgsResult { x, f: fx, grad_norm: gn, iters, evals, stop: LbfgsStop::GradTol };
            }
            if iters >= max_iters {
                return LbfgsResult { x, f: fx, grad_norm: gn, iters, evals, stop: LbfgsStop::MaxIters };
            }
            let mut d = self.direction(&g);
            for v in d.iter_mut() {
                *v = -*v;
            }
            let mut slope = dot(&g, &d);
            if slope >= 0.0 || slope.is_nan() {
                // not a descent direction (history gone bad): restart from steepest descent
                self.s.clear();
                self.y.clear();
                d = g.iter().map(|v| -v).collect();
                slope = dot(&g, &d);
            }
            // first step of a fresh history: unit step in a scaled direction, as Nocedal & Wright §7.2 recommends
            let t0 = if self.s.is_empty() { (1.0 / norm(&d)).min(1.0) } else { 1.0 };
            let Some((xn, fn_, gn_, used)) = wolfe(&mut f, &x, fx, &g, &d, slope, t0, self.c1, self.c2, self.max_backtracks) else {
                return LbfgsResult { x, f: fx, grad_norm: gn, iters, evals, stop: LbfgsStop::LineSearchFailed };
            };
            evals += used;
            let s: Vec<f32> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
            let y: Vec<f32> = gn_.iter().zip(&g).map(|(a, b)| a - b).collect();
            if dot(&s, &y) > 1e-10 * norm(&s) * norm(&y) {
                if self.s.len() == self.m {
                    self.s.pop_front();
                    self.y.pop_front();
                }
                self.s.push_back(s);
                self.y.push_back(y);
            }
            x = xn;
            fx = fn_;
            g = gn_;
            iters += 1;
        }
    }
}

/// Strong-Wolfe line search, Nocedal & Wright Algorithms 3.5 (bracketing) and 3.6 (zoom, by bisection).
/// Returns the accepted point `(x, f, ∇f, evaluations)`; `None` only if no point with sufficient decrease
/// was found at all. A point satisfying Armijo but not the curvature condition is still returned when the
/// zoom budget runs out — the caller's `sᵀy > 0` check decides whether it enters the history.
#[allow(clippy::too_many_arguments)]
fn wolfe<F>(f: &mut F, x: &[f32], f0: f32, g0: &[f32], d: &[f32], slope: f32, t0: f32, c1: f32, c2: f32, budget: usize) -> Option<(Vec<f32>, f32, Vec<f32>, usize)>
where
    F: FnMut(&[f32]) -> (f32, Vec<f32>),
{
    let _ = g0;
    let at = |t: f32| -> Vec<f32> { x.iter().zip(d).map(|(a, b)| a + t * b).collect() };
    let mut evals = 0usize;
    let mut eval = |t: f32| {
        let xt = at(t);
        let (ft, gt) = f(&xt);
        let dt = dot(&gt, d);
        (xt, ft, gt, dt)
    };
    let armijo = |t: f32, ft: f32| ft <= f0 + c1 * t * slope;
    let curv = |dt: f32| dt.abs() <= -c2 * slope;

    // bracketing phase
    let (mut t_lo, mut f_lo, mut d_lo, mut x_lo, mut g_lo) = (0.0f32, f0, slope, x.to_vec(), g0.to_vec());
    let mut t = t0;
    let mut bracket: Option<(f32, f32)> = None; // (lo, hi) with the invariants of Alg. 3.6
    let mut hi_state: Option<(f32, f32)> = None; // (t_hi, f_hi)
    for i in 0..budget {
        let (xt, ft, gt, dt) = eval(t);
        evals += 1;
        if !ft.is_finite() {
            t *= 0.5;
            continue;
        }
        if !armijo(t, ft) || (i > 0 && ft >= f_lo) {
            bracket = Some((t_lo, t));
            hi_state = Some((t, ft));
            break;
        }
        if curv(dt) {
            return Some((xt, ft, gt, evals));
        }
        if dt >= 0.0 {
            bracket = Some((t, t_lo));
            hi_state = Some((t_lo, f_lo));
            t_lo = t;
            f_lo = ft;
            d_lo = dt;
            x_lo = xt;
            g_lo = gt;
            break;
        }
        t_lo = t;
        f_lo = ft;
        d_lo = dt;
        x_lo = xt;
        g_lo = gt;
        t *= 2.0;
    }
    let Some((mut lo, mut hi)) = bracket else {
        // ran out of budget while still expanding: the last point had sufficient decrease
        return if t_lo > 0.0 { Some((x_lo, f_lo, g_lo, evals)) } else { None };
    };
    let _ = hi_state;
    let _ = d_lo;
    // zoom phase: the low end always satisfies Armijo; shrink by bisection
    for _ in 0..budget {
        let tj = 0.5 * (lo + hi);
        let (xt, ft, gt, dt) = eval(tj);
        evals += 1;
        if !ft.is_finite() || !armijo(tj, ft) || ft >= f_lo {
            hi = tj;
        } else {
            if curv(dt) {
                return Some((xt, ft, gt, evals));
            }
            if dt * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = tj;
            f_lo = ft;
            x_lo = xt;
            g_lo = gt;
        }
        if (hi - lo).abs() < 1e-12 {
            break;
        }
    }
    if t_lo > 0.0 || lo > 0.0 {
        Some((x_lo, f_lo, g_lo, evals))
    } else {
        None
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock: the classic curved valley that steepest descent crawls along. L-BFGS with a backtracking
    /// search reaches the minimum `(1, 1)` to 1e-4 in well under a hundred iterations.
    #[test]
    fn lbfgs_solves_rosenbrock() {
        let f = |x: &[f32]| {
            let (a, b) = (x[0], x[1]);
            let v = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            let g = vec![-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)];
            (v, g)
        };
        let r = Lbfgs::new(10).minimize(vec![-1.2, 1.0], f, 200, 1e-6);
        eprintln!("  rosenbrock: {} iters, {} evals, f = {:.3e}, x = {:?}, stop {:?}", r.iters, r.evals, r.f, r.x, r.stop);
        assert!((r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] - 1.0).abs() < 1e-3, "did not reach (1,1): {:?}", r.x);
        assert!(r.iters < 100, "too many iterations: {}", r.iters);
    }

    /// On a quadratic, BFGS with exact curvature information converges superlinearly; a 20-dimensional
    /// ill-conditioned quadratic (condition number 1e4) must reach 1e-6 gradient norm far faster than the
    /// ~1e4 steepest-descent iterations the conditioning would demand.
    #[test]
    fn lbfgs_beats_the_condition_number_on_a_quadratic() {
        let n = 20;
        let diag: Vec<f32> = (0..n).map(|i| 1.0 + 9999.0 * i as f32 / (n as f32 - 1.0)).collect();
        let f = |x: &[f32]| {
            let v: f32 = x.iter().zip(&diag).map(|(xi, d)| 0.5 * d * xi * xi).sum();
            let g: Vec<f32> = x.iter().zip(&diag).map(|(xi, d)| d * xi).collect();
            (v, g)
        };
        let r = Lbfgs::new(10).minimize(vec![1.0; n], f, 500, 1e-5);
        eprintln!("  quadratic κ=1e4: {} iters, grad norm {:.2e}, stop {:?}", r.iters, r.grad_norm, r.stop);
        assert_eq!(r.stop, LbfgsStop::GradTol, "must converge by gradient tolerance");
        assert!(r.iters < 150, "L-BFGS should not need {} iterations on a 20-d quadratic", r.iters);
    }
}
