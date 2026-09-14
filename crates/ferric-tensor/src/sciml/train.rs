//! **The physics-informed training recipe** — the pieces that separate a PINN that works from one that
//! returns a plausible flat line. Each is implemented from its paper and each ships with a test in which
//! the vanilla loss measurably fails and the technique measurably fixes it; a technique that only ever
//! matches the baseline is decoration.
//!
//! | piece | source | what it fixes |
//! |---|---|---|
//! | [`LossBalancer`] | Wang, Teng & Perdikaris 2021, arXiv 2001.04536 | boundary terms whose gradients are dwarfed by the residual's, so the net satisfies the PDE and ignores the boundary |
//! | [`Rba`] | Anagnostopoulos et al. 2023, arXiv 2307.00379 | collocation points the optimiser has given up on, by re-weighting each point by its own running residual |
//! | [`Causal`] | Wang, Sankaran & Perdikaris 2022, arXiv 2203.07404 | time-dependent problems where the net fits later times before earlier ones and lands on a wrong branch |
//! | [`rar_select`] | Lu et al. 2021 (DeepXDE), arXiv 1907.04502 | sharp features that a fixed collocation set never samples |
//! | [`flatten`] / [`unflatten`] | — | moving parameters to and from the CPU vector an [`Lbfgs`](super::Lbfgs) step needs |

use crate::{grad, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// Mean of squares of a `Var` — the loss form every term below takes.
pub fn mse(v: &Var) -> Var {
    v.mul(v).mean_all()
}

/// A scalar constant as a `[1]` `Var` on the same device as `like`.
pub fn scalar(like: &Var, c: f32) -> Var {
    Var::leaf(Tensor::from_vec(&like.value().ctx_arc(), &[c], &[1]))
}

/// **Gradient-norm loss balancing** ("learning-rate annealing", arXiv 2001.04536 §3).
///
/// With loss `L = L_r + Σᵢ λᵢ Lᵢ`, the residual term's parameter gradients are typically orders larger
/// than the boundary terms', so plain gradient descent moves only along the residual. The fix reads the
/// gradients and sets each `λᵢ` so the terms pull comparably:
///
/// ```text
/// λ̂ᵢ = max|∇θ L_r| / mean|∇θ Lᵢ|,     λᵢ ← (1 − α) λᵢ + α λ̂ᵢ
/// ```
///
/// Term `0` is the reference (the residual) and keeps weight `1`. Call [`LossBalancer::update`] every
/// few hundred steps — it costs one extra gradient pass per term — and [`LossBalancer::combine`] every step.
///
/// ⚠ One safeguard the paper does not have: the denominator is floored at `floor × max|∇L_r|`, so a
/// term whose gradient has vanished cannot send its weight to infinity. See the note in `update`.
pub struct LossBalancer {
    pub weights: Vec<f32>,
    pub alpha: f32,
    /// Denominator floor as a fraction of the reference gradient; caps every weight at `1/floor`. Default `1e-4`.
    pub floor: f32,
}

impl LossBalancer {
    pub fn new(n_terms: usize, alpha: f32) -> Self {
        assert!(n_terms >= 1);
        LossBalancer { weights: vec![1.0; n_terms], alpha, floor: 1e-4 }
    }

    /// Re-estimate the weights from the current gradients of each term with respect to `params`.
    pub async fn update(&mut self, terms: &[Var], params: &[Var]) {
        assert_eq!(terms.len(), self.weights.len());
        let mut max_ref = 0.0f32;
        let mut means = vec![0.0f32; terms.len()];
        for (i, t) in terms.iter().enumerate() {
            let gs = grad(t, params, None);
            let mut sum = 0.0f32;
            let mut n = 0usize;
            let mut mx = 0.0f32;
            for g in &gs {
                for v in g.value().to_vec().await {
                    let a = v.abs();
                    sum += a;
                    mx = mx.max(a);
                    n += 1;
                }
            }
            means[i] = if n > 0 { sum / n as f32 } else { 0.0 };
            if i == 0 {
                max_ref = mx;
            }
        }
        for (i, mean_i) in means.iter().enumerate().skip(1) {
            if max_ref.is_finite() && max_ref > 0.0 {
                // ⛔ The paper's ratio has no floor, and it needs one. Once a boundary term is satisfied
                // to float precision its gradient is ~0, the ratio is unbounded, and measured on the 1-D
                // Poisson oracle the weight ran to 1.4e12 — at which point any boundary deviation of 1e-8
                // produces an O(1e4) gradient and training is wrecked (rel-L2 0.93 against 0.03 unweighted).
                // Flooring the denominator at `floor` × the reference caps the weight at `1/floor`, which
                // is still four orders of balancing and cannot detonate.
                let denom = mean_i.max(self.floor * max_ref);
                let hat = max_ref / denom;
                self.weights[i] = (1.0 - self.alpha) * self.weights[i] + self.alpha * hat;
            }
        }
    }

    /// `Σᵢ λᵢ · termsᵢ` as a graph node.
    pub fn combine(&self, terms: &[Var]) -> Var {
        let mut total = terms[0].mul(&scalar(&terms[0], self.weights[0]));
        for (t, &w) in terms.iter().zip(&self.weights).skip(1) {
            total = total.add(&t.mul(&scalar(t, w)));
        }
        total
    }
}

/// **Residual-based attention** (arXiv 2307.00379): a per-collocation-point multiplier that grows where
/// the residual stays large, so the loss `mean((α ⊙ r)²)` keeps pushing on the points the optimiser has
/// neglected. `αᵢ ← γ αᵢ + η |rᵢ| / max|r|`, bounded by `η / (1 − γ)`.
pub struct Rba {
    pub mult: Vec<f32>,
    pub gamma: f32,
    pub eta: f32,
}

impl Rba {
    pub fn new(n_points: usize, gamma: f32, eta: f32) -> Self {
        Rba { mult: vec![1.0; n_points], gamma, eta }
    }

    /// Update from the current residual magnitudes (read back once per step or per few steps).
    pub fn update(&mut self, residual: &[f32]) {
        let mx = residual.iter().fold(0.0f32, |m, r| m.max(r.abs())).max(1e-12);
        for (a, r) in self.mult.iter_mut().zip(residual) {
            *a = self.gamma * *a + self.eta * r.abs() / mx;
        }
    }

    /// The multipliers as an `[N, 1]` `Var` (a constant — no gradient flows into it).
    pub fn as_var(&self, ctx: &Arc<Context>) -> Var {
        Var::leaf(Tensor::from_vec(ctx, &self.mult, &[self.mult.len(), 1]))
    }
}

/// **Causal training** (arXiv 2203.07404): weight the residual loss of time slab `i` by
/// `wᵢ = exp(−ε Σ_{k<i} L_k)`, so a later slab only counts once the earlier ones are solved. The net is
/// then forced to respect the arrow of time instead of fitting a convenient wrong branch late and pinning
/// early times to it.
///
/// `ε` is annealed upward through `eps_schedule` — each value is used until every `wᵢ` exceeds `delta`
/// (the paper uses `δ = 0.99`), which is the signal that the whole horizon is being trained at once.
pub struct Causal {
    pub eps_schedule: Vec<f32>,
    pub idx: usize,
    pub delta: f32,
}

impl Causal {
    pub fn new(eps_schedule: Vec<f32>, delta: f32) -> Self {
        assert!(!eps_schedule.is_empty());
        Causal { eps_schedule, idx: 0, delta }
    }

    pub fn eps(&self) -> f32 {
        self.eps_schedule[self.idx]
    }

    /// Weights for slabs given their current losses (in time order), and advance the schedule if the
    /// smallest weight has cleared `delta`. Returns the weights (constants: no gradient through them).
    pub fn weights(&mut self, slab_losses: &[f32]) -> Vec<f32> {
        let eps = self.eps();
        let mut acc = 0.0f32;
        let mut w = Vec::with_capacity(slab_losses.len());
        for &l in slab_losses {
            w.push((-eps * acc).exp());
            acc += l;
        }
        if w.iter().cloned().fold(1.0f32, f32::min) > self.delta && self.idx + 1 < self.eps_schedule.len() {
            self.idx += 1;
        }
        w
    }

    /// Per-slab mean-square losses from per-point residuals and each point's slab index.
    pub fn slab_losses(residual: &[f32], slab: &[usize], n_slabs: usize) -> Vec<f32> {
        let mut sum = vec![0.0f32; n_slabs];
        let mut cnt = vec![0usize; n_slabs];
        for (r, &s) in residual.iter().zip(slab) {
            sum[s] += r * r;
            cnt[s] += 1;
        }
        sum.iter().zip(&cnt).map(|(s, &c)| if c > 0 { s / c as f32 } else { 0.0 }).collect()
    }

    /// Expand slab weights to a per-point `[N, 1]` constant `Var`.
    pub fn point_weights(ctx: &Arc<Context>, w: &[f32], slab: &[usize]) -> Var {
        let v: Vec<f32> = slab.iter().map(|&s| w[s]).collect();
        Var::leaf(Tensor::from_vec(ctx, &v, &[v.len(), 1]))
    }
}

/// **Residual-based adaptive refinement** (arXiv 1907.04502 §2.4): the indices of the `k` candidate
/// points with the largest residual magnitude, to be added to the collocation set. Call on a dense
/// candidate cloud every few hundred steps; the sharp feature the fixed set was missing is where the
/// residual is large.
pub fn rar_select(residual_abs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..residual_abs.len()).collect();
    idx.sort_by(|&a, &b| residual_abs[b].abs().partial_cmp(&residual_abs[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Read every parameter tensor back into one CPU vector, in order.
pub async fn flatten(params: &[Tensor]) -> Vec<f32> {
    let mut out = Vec::new();
    for p in params {
        out.extend(p.to_vec().await);
    }
    out
}

/// Rebuild parameter tensors of the given shapes from one CPU vector (the inverse of [`flatten`]).
pub fn unflatten(ctx: &Arc<Context>, flat: &[f32], shapes: &[Vec<usize>]) -> Vec<Tensor> {
    let mut off = 0usize;
    let mut out = Vec::with_capacity(shapes.len());
    for s in shapes {
        let n: usize = s.iter().product();
        out.push(Tensor::from_vec(ctx, &flat[off..off + n], s));
        off += n;
    }
    assert_eq!(off, flat.len(), "flat vector does not match the shapes");
    out
}
