//! **NTK loss balancing** (Wang, Yu & Perdikaris 2022, arXiv 2007.14527) — the weighting the Helmholtz
//! benchmark row names as missing, and the one jaxpi/PirateNets use by default.
//!
//! Gradient-norm balancing ([`LossBalancer`](super::LossBalancer)) equalises how hard each term PULLS.
//! The NTK view asks a different question: how fast does each term's error DECAY? Under gradient flow the
//! residual of term `i` decays at a rate set by the eigenvalues of its neural tangent kernel
//! `K_ii = J_i J_iᵀ` (with `J_i = ∂r_i/∂θ`), so a term whose kernel is small converges slowly no matter
//! how large its gradient is. The paper's fix weights each term by the inverse of its kernel trace:
//!
//! ```text
//! λ_i = (Σ_j tr K_jj) / tr K_ii
//! ```
//!
//! ⛔ **The trace is estimated, not assembled, and that is the whole reason this is affordable here.**
//! `tr K_ii = ‖J_i‖_F²` would need one backward pass per collocation point (jaxpi gets them from JAX's
//! `jacrev` + `vmap`; this fabric has neither). Hutchinson's identity gives it in a handful of passes: for
//! a Rademacher vector `v`,
//!
//! ```text
//! E‖∇_θ (vᵀ r_i)‖² = E[vᵀ J_i J_iᵀ v] = tr(J_i J_iᵀ) = tr K_ii
//! ```
//!
//! so a few probes of a scalar backward pass estimate the trace. `the_hutchinson_estimator_converges_to_the_exact_ntk_trace`
//! checks it against a closed form on a linear model, where `tr K = ‖A‖_F²` exactly.

use super::train::scalar;
use super::util::u01;
use crate::{grad, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// NTK-trace balancing over a set of residual VECTORS (not scalar losses — the trace is a property of the
/// per-point Jacobian, which a summed loss has already destroyed).
pub struct NtkBalancer {
    pub weights: Vec<f32>,
    /// Hutchinson probes per term per update. Four is enough for the ratio; the estimator's variance falls
    /// as `1/√probes` and the weights are used as a ratio of sums, which cancels much of it.
    pub probes: usize,
    pub alpha: f32,
    /// Denominator floor as a fraction of the largest trace — the same guard [`LossBalancer`] needs, for
    /// the same reason: a converged term's trace goes to zero and an unfloored ratio goes to infinity.
    pub floor: f32,
}

impl NtkBalancer {
    pub fn new(n_terms: usize, probes: usize, alpha: f32) -> Self {
        assert!(n_terms >= 1 && probes >= 1);
        NtkBalancer { weights: vec![1.0; n_terms], probes, alpha, floor: 1e-6 }
    }

    /// One Hutchinson sample of `tr K` for a residual vector `r` (`[N, 1]`) with respect to `params`.
    fn trace_probe(ctx: &Arc<Context>, r: &Var, params: &[Var], seed: u32) -> f64 {
        let n = r.value().shape.iter().product::<usize>();
        let v: Vec<f32> = (0..n).map(|i| if u01(i as u32, seed) < 0.5 { -1.0 } else { 1.0 }).collect();
        let s = r.mul(&Var::leaf(Tensor::from_vec(ctx, &v, &r.value().shape))).sum_all();
        let gs = grad(&s, params, None);
        let mut acc = 0.0f64;
        for g in &gs {
            let gv = pollster::block_on(g.value().to_vec());
            for x in gv {
                acc += (x as f64) * (x as f64);
            }
        }
        acc
    }

    /// Estimated NTK traces of each residual vector, in order.
    pub fn traces(&self, ctx: &Arc<Context>, residuals: &[Var], params: &[Var], seed: u32) -> Vec<f64> {
        residuals
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut acc = 0.0;
                for p in 0..self.probes {
                    acc += Self::trace_probe(ctx, r, params, seed.wrapping_add((i * 131 + p * 7717) as u32));
                }
                acc / self.probes as f64
            })
            .collect()
    }

    /// Re-estimate the weights: `λ_i = (Σ_j tr K_jj) / tr K_ii`, smoothed by `alpha`.
    pub fn update(&mut self, ctx: &Arc<Context>, residuals: &[Var], params: &[Var], seed: u32) {
        assert_eq!(residuals.len(), self.weights.len());
        let tr = self.traces(ctx, residuals, params, seed);
        let total: f64 = tr.iter().sum();
        let biggest = tr.iter().cloned().fold(0.0f64, f64::max);
        if !(total.is_finite() && total > 0.0) {
            return;
        }
        for (w, &t) in self.weights.iter_mut().zip(&tr) {
            let hat = (total / t.max(self.floor as f64 * biggest)) as f32;
            *w = (1.0 - self.alpha) * *w + self.alpha * hat;
        }
    }

    /// `Σ_i λ_i · mean(r_i²)` — the weighted loss.
    pub fn combine(&self, residuals: &[Var]) -> Var {
        let mut total = residuals[0].mul(&residuals[0]).mean_all().mul(&scalar(&residuals[0], self.weights[0]));
        for (r, &w) in residuals.iter().zip(&self.weights).skip(1) {
            total = total.add(&r.mul(r).mean_all().mul(&scalar(r, w)));
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::util::{leaf, u01 as u};

    /// ⭐ **The estimator is checked against a closed form.** For a residual linear in the parameters,
    /// `r = A θ`, the Jacobian IS `A` and `tr K = tr(A Aᵀ) = ‖A‖_F²` exactly — so the Hutchinson estimate
    /// has a number to converge to, and the test watches it converge as probes are added rather than
    /// asserting one lucky draw.
    #[test]
    fn the_hutchinson_estimator_converges_to_the_exact_ntk_trace() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, p) = (64usize, 24usize);
            let a: Vec<f32> = (0..n * p).map(|i| u(i as u32, 11) - 0.5).collect();
            let exact: f64 = a.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let theta = vec![Var::leaf(Tensor::from_vec(&ctx, &vec![0.3f32; p], &[p, 1]))];
            let r = leaf(&ctx, &a, &[n, p]).matmul(&theta[0]);

            let mut prev = f64::INFINITY;
            let mut rows = Vec::new();
            for &probes in &[1usize, 4, 16, 64] {
                let b = NtkBalancer::new(1, probes, 1.0);
                let est = b.traces(&ctx, core::slice::from_ref(&r), &theta, 7)[0];
                let rel = (est - exact).abs() / exact;
                rows.push(format!("{probes} probes {est:.3} ({:+.1}%)", 100.0 * (est / exact - 1.0)));
                if probes == 64 {
                    assert!(rel < 0.15, "64 probes should be within 15% of ‖A‖_F² = {exact:.3}, got {est:.3}");
                }
                prev = prev.min(rel);
            }
            eprintln!("  NTK trace, exact ‖A‖_F² = {exact:.3}: {}", rows.join(", "));
            assert!(prev < 0.15, "the estimator must approach the exact trace");
        });
    }

    /// The weights are the paper's ratio, and a vanished trace cannot send one to infinity.
    #[test]
    fn the_weights_are_the_inverse_trace_ratio_and_a_dead_term_is_floored() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            // two residuals with deliberately different scales: r2 = 0.01 · r1 ⇒ tr K2 = 1e-4 · tr K1
            let p = 8usize;
            let theta = vec![Var::leaf(Tensor::from_vec(&ctx, &vec![0.5f32; p], &[p, 1]))];
            let a: Vec<f32> = (0..32 * p).map(|i| u(i as u32, 3) - 0.5).collect();
            let big = leaf(&ctx, &a, &[32, p]).matmul(&theta[0]);
            let small = big.mul(&scalar(&big, 0.01));
            let mut b = NtkBalancer::new(2, 32, 1.0);
            b.update(&ctx, &[big.clone(), small.clone()], &theta, 5);
            eprintln!("  NTK weights for traces differing 1e4x: {:?}", b.weights);
            assert!(b.weights[1] / b.weights[0] > 100.0, "the slow term must get far more weight: {:?}", b.weights);
            // a term with no gradient at all is floored rather than infinite
            let dead = Var::leaf(Tensor::zeros(&ctx, &[32, 1]));
            let mut b2 = NtkBalancer::new(2, 4, 1.0);
            b2.update(&ctx, &[big, dead], &theta, 5);
            assert!(b2.weights[1].is_finite() && b2.weights[1] <= 1.0 / b2.floor + 1.0, "a dead term must be floored, got {}", b2.weights[1]);
        });
    }
}
