//! Scientific-ML building blocks on the fabric — the reusable core behind the PINN / neural-operator
//! examples (`pinn_siren`, `pinn_poisson2d`, `deeponet`), lifted out of the examples so they are a
//! *library*, not one-offs. A [`Siren`] (sin-activation MLP) — the natural physics-informed network,
//! since sine represents a function AND its derivatives well — plus [`deriv`], a differentiable d/dx
//! built on the second-order `grad()`: `u' = deriv(u, x)`, `u'' = deriv(u', x)`. Train through the tape
//! with any optimizer (e.g. `Adam`); everything runs GPU-native on the wgpu device. Pure Rust — the seed
//! of a pure-Rust PINN/operator stack, a niche no existing (Python/Julia) library occupies.

use crate::{grad, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// A SIREN: a multilayer perceptron with **sine** activations on the hidden layers and a linear output.
/// `dims` is the full layer spec, e.g. `[1, 32, 32, 1]` (scalar in → scalar out) for a 1-D PINN, or
/// `[2, 40, 40, 1]` for a 2-D field. Parameters are owned `Tensor`s laid out `[W1, b1, W2, b2, …]`; wrap
/// them fresh as `Var`s each training step with [`Siren::vars`] and run [`Siren::forward`].
pub struct Siren {
    pub params: Vec<Tensor>,
}

impl Siren {
    /// SIREN-initialized network on `ctx`. The first layer is scaled up (the SIREN ω₀ trick) with a random
    /// phase bias so the sines span varied frequencies; hidden layers use the √(6/fan_in) SIREN scale; the
    /// output layer is small. Deterministic in `seed`.
    pub fn new(ctx: &Arc<Context>, dims: &[usize], seed: u32) -> Self {
        assert!(dims.len() >= 2, "need at least an input and output layer");
        let nl = dims.len() - 1;
        let mut params = Vec::with_capacity(nl * 2);
        for l in 0..nl {
            let (fi, fo) = (dims[l], dims[l + 1]);
            let sc = if l == 0 { 2.4 } else if l + 1 == nl { (1.0 / fi as f32).sqrt() } else { (6.0 / fi as f32).sqrt() };
            params.push(Tensor::from_vec(ctx, &randn(fi * fo, seed.wrapping_add(l as u32 * 7 + 1), sc), &[fi, fo]));
            let b = if l == 0 { randn(fo, seed.wrapping_add(99), 2.0) } else { vec![0.0; fo] };
            params.push(Tensor::from_vec(ctx, &b, &[fo]));
        }
        Siren { params }
    }

    /// Fresh `Var` leaves of the current parameters — rebuild these each training step (the tape is
    /// consumed by `backward()`), read gradients back with `Var::grad`, and update with an optimizer.
    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }

    /// Forward pass on input `x` (`[N, dims[0]]`): sine on every hidden layer, linear output. `pv` are the
    /// parameter `Var`s from [`Siren::vars`].
    pub fn forward(pv: &[Var], x: &Var) -> Var {
        let nl = pv.len() / 2;
        let mut h = x.clone();
        for l in 0..nl {
            h = h.matmul(&pv[2 * l]).add(&pv[2 * l + 1]);
            if l + 1 < nl {
                h = h.sin();
            }
        }
        h
    }
}

/// A plain MLP with ReLU hidden activations and a linear output — the branch/trunk network for neural
/// operators (e.g. DeepONet). Same layout and usage as [`Siren`]: params `[W1, b1, …]`, [`Mlp::vars`] +
/// [`Mlp::forward`]. (PINNs want the smooth [`Siren`]; operators, learned from data, are fine with ReLU.)
pub struct Mlp {
    pub params: Vec<Tensor>,
}

impl Mlp {
    /// Xavier-ish-initialized MLP on `ctx` with layer spec `dims` (e.g. `[40, 48, 32]`).
    pub fn new(ctx: &Arc<Context>, dims: &[usize], seed: u32) -> Self {
        assert!(dims.len() >= 2, "need at least an input and output layer");
        let nl = dims.len() - 1;
        let mut params = Vec::with_capacity(nl * 2);
        for l in 0..nl {
            let (fi, fo) = (dims[l], dims[l + 1]);
            params.push(Tensor::from_vec(ctx, &randn(fi * fo, seed.wrapping_add(l as u32 * 7 + 1), (1.0 / fi as f32).sqrt()), &[fi, fo]));
            params.push(Tensor::zeros(ctx, &[fo]));
        }
        Mlp { params }
    }
    /// Fresh `Var` leaves of the current parameters (rebuild each training step).
    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }
    /// Forward pass on `x` (`[N, dims[0]]`): ReLU on hidden layers, linear output.
    pub fn forward(pv: &[Var], x: &Var) -> Var {
        let nl = pv.len() / 2;
        let mut h = x.clone();
        for l in 0..nl {
            h = h.matmul(&pv[2 * l]).add(&pv[2 * l + 1]);
            if l + 1 < nl {
                h = h.relu();
            }
        }
        h
    }
}

/// Differentiable derivative of a batched scalar field `y` (`[N,1]`) with respect to its input `x`,
/// returned as a `Var` so it can be differentiated **again** — the crux of PINNs: `u' = deriv(u, x)`,
/// `u'' = deriv(u', x)`. Seeds the reverse pass with ones over the batch, which for an elementwise map
/// `x_i → y_i` yields the per-point derivative. Every op on the path must carry a VJP (all the primitives
/// used here do), so arbitrary orders compose.
pub fn deriv(y: &Var, x: &Var) -> Var {
    grad(&y.sum_all(), &[x.clone()], None).remove(0)
}

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let a = (h32((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) % 1_000_000 + 1) as f32 / 1_000_000.0;
            let b = (h32((i as u32).wrapping_mul(2654435761).wrapping_add(seed).wrapping_add(9973)) % 1_000_000 + 1) as f32 / 1_000_000.0;
            ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Adam;

    /// Library-level proof: a physics-informed net solves u''+ω²u=0, u(0)=1, u'(0)=0 (true cos ωt) from the
    /// RESIDUAL alone — built entirely from `Siren` + `deriv` on the fabric — and converges below 0.1.
    #[test]
    fn pinn_harmonic_oscillator_converges() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (w, t_max, nc) = (2.0f32, 3.0f32, 40usize);
            let net = Siren::new(&ctx, &[1, 32, 32, 1], 1);
            let mut wp = net.params.clone();
            let mut adam = Adam::new(&wp, 3e-3);
            let tcol: Vec<f32> = (0..nc).map(|i| i as f32 * t_max / (nc as f32 - 1.0)).collect();
            for _ in 0..2500 {
                let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
                let tv = Var::leaf(Tensor::from_vec(&ctx, &tcol, &[nc, 1]));
                let u = Siren::forward(&pv, &tv);
                let u_t = deriv(&u, &tv);
                let u_tt = deriv(&u_t, &tv);
                let res = u_tt.add(&u.mul(&Var::leaf(Tensor::from_vec(&ctx, &vec![w * w; nc], &[nc, 1]))));
                let loss_res = res.mul(&res).mean_all();
                let t0 = Var::leaf(Tensor::from_vec(&ctx, &[0.0], &[1, 1]));
                let u0 = Siren::forward(&pv, &t0);
                let u0t = deriv(&u0, &t0);
                let e0 = u0.sub(&Var::leaf(Tensor::from_vec(&ctx, &[1.0], &[1, 1])));
                let loss_ic = e0.mul(&e0).sum_all().add(&u0t.mul(&u0t).sum_all());
                let loss = loss_res.add(&loss_ic.mul(&Var::leaf(Tensor::from_vec(&ctx, &[40.0], &[1]))));
                loss.backward();
                let g: Vec<Tensor> = pv.iter().zip(&wp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::zeros(&ctx, &t.shape))).collect();
                adam.step(&mut wp, &g);
            }
            let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
            let te: Vec<f32> = (0..100).map(|i| i as f32 * t_max / 99.0).collect();
            let ue = Siren::forward(&pv, &Var::leaf(Tensor::from_vec(&ctx, &te, &[100, 1]))).value().to_vec().await;
            let max_err = te.iter().zip(&ue).map(|(&t, &u)| (u - (w * t).cos()).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 0.1, "PINN did not converge: max_err = {max_err}");
        });
    }
}
