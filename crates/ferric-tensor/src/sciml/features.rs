//! **Fourier-feature networks** — the standard cure for spectral bias in physics-informed nets.
//!
//! A coordinate MLP learns low frequencies first and high ones slowly or never (the neural-tangent-kernel
//! view: Tancik et al. 2020, arXiv 2006.10739). Mapping the input through fixed random sinusoids
//! `γ(x) = [sin(Bx), cos(Bx)]`, `B ~ N(0, σ²)`, moves the kernel's bandwidth to `σ` and lets the net fit
//! a target of that scale in ordinary training time. For PINNs this is the difference between solving
//! `u'' + ω²u = 0` at ω = 8 and returning a flat line — measured in `spectral_bias_is_fixed_by_fourier_features`.
//!
//! Several scales at once (Wang, Wang & Perdikaris 2021, arXiv 2012.10047) let one net cover a target
//! with both slow and fast content. ⚠ That paper's MsFFN keeps one hidden tower per scale and merges at the
//! output; this is the single-tower variant that concatenates the scales' features into one first layer —
//! the same idea, fewer parameters, and what DeepXDE's `msffn` reduces to when the towers share weights.

use super::{randn, Act};
use crate::{Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// A tanh (or other) MLP whose first layer reads Fourier features of the input instead of the input.
///
/// `sin(xB)` and `cos(xB)` each get their own slice of the first weight matrix, which is algebraically the
/// concatenated feature vector times one matrix — written this way so the graph uses only `matmul`,
/// `sin`, `cos` and `add`, every one of which carries a second-order VJP on this fabric. (A `cat` in the
/// path would be the first op whose double-derivative this stack has not had to trust.)
pub struct FourierNet {
    /// `[d_in, m]`, frequencies with `2π` already folded in; fixed, not trained.
    pub b: Tensor,
    /// `[W1_sin, W1_cos, b1, W2, b2, …]` — trained.
    pub params: Vec<Tensor>,
    pub act: Act,
}

impl FourierNet {
    /// `d_in` inputs → `m_per_scale` random frequencies per entry of `scales` (each `σ`) → `hidden` layers
    /// of the given widths → `d_out`. Deterministic in `seed`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &Arc<Context>, d_in: usize, m_per_scale: usize, scales: &[f32], hidden: &[usize], d_out: usize, act: Act, seed: u32) -> Self {
        assert!(!scales.is_empty() && !hidden.is_empty(), "need at least one scale and one hidden layer");
        let m = m_per_scale * scales.len();
        // B: column block s holds m_per_scale frequencies ~ N(0, σ_s²), times 2π
        let mut bvec = vec![0.0f32; d_in * m];
        for (s, &sigma) in scales.iter().enumerate() {
            let block = randn(d_in * m_per_scale, seed.wrapping_add(1000 + s as u32), sigma * std::f32::consts::TAU);
            for i in 0..d_in {
                for j in 0..m_per_scale {
                    bvec[i * m + s * m_per_scale + j] = block[i * m_per_scale + j];
                }
            }
        }
        let b = Tensor::from_vec(ctx, &bvec, &[d_in, m]);
        let mut params = Vec::new();
        let h0 = hidden[0];
        let sc = (1.0 / (2 * m) as f32).sqrt();
        params.push(Tensor::from_vec(ctx, &randn(m * h0, seed.wrapping_add(1), sc), &[m, h0]));
        params.push(Tensor::from_vec(ctx, &randn(m * h0, seed.wrapping_add(2), sc), &[m, h0]));
        params.push(Tensor::zeros(ctx, &[h0]));
        let mut dims: Vec<usize> = hidden.to_vec();
        dims.push(d_out);
        for l in 0..dims.len() - 1 {
            let (fi, fo) = (dims[l], dims[l + 1]);
            params.push(Tensor::from_vec(ctx, &randn(fi * fo, seed.wrapping_add(10 + l as u32), (1.0 / fi as f32).sqrt()), &[fi, fo]));
            params.push(Tensor::zeros(ctx, &[fo]));
        }
        FourierNet { b, params, act }
    }

    /// Fresh `Var` leaves of the trainable parameters (rebuild each step; the tape is consumed by `backward`).
    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }

    /// Forward pass on `x` (`[N, d_in]`) with parameter `Var`s `pv` from [`FourierNet::vars`].
    pub fn forward(&self, pv: &[Var], x: &Var) -> Var {
        let z = x.matmul(&Var::leaf(self.b.clone()));
        let mut h = z.sin().matmul(&pv[0]).add(&z.cos().matmul(&pv[1])).add(&pv[2]);
        h = act(&h, self.act);
        let nl = (pv.len() - 3) / 2;
        for l in 0..nl {
            h = h.matmul(&pv[3 + 2 * l]).add(&pv[4 + 2 * l]);
            if l + 1 < nl {
                h = act(&h, self.act);
            }
        }
        h
    }
}

pub(super) fn act(h: &Var, a: Act) -> Var {
    match a {
        Act::Relu => h.relu(),
        Act::Tanh => h.tanh(),
        Act::Sin => h.sin(),
    }
}
