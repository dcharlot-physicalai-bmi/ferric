//! Small helpers shared by the oracles, the harness and the certificate: deterministic sampling, `Var`
//! constructors, the column-of-a-gradient extractor, and relative-L2.

use super::deriv;
use crate::{Adam, Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

pub fn h32(mut h: u32) -> u32 {
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    h ^= h >> 16;
    h
}
/// Deterministic uniform in `(0, 1]` from an index and a seed.
pub fn u01(i: u32, s: u32) -> f32 {
    (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0
}
pub fn leaf(ctx: &Arc<Context>, v: &[f32], shape: &[usize]) -> Var {
    Var::leaf(Tensor::from_vec(ctx, v, shape))
}
pub fn col(ctx: &Arc<Context>, v: &[f32]) -> Var {
    leaf(ctx, v, &[v.len(), 1])
}
pub fn vars(wp: &[Tensor]) -> Vec<Var> {
    wp.iter().map(|t| Var::leaf(t.clone())).collect()
}
/// One Adam step on `loss`; returns the loss value.
pub async fn step(ctx: &Arc<Context>, loss: &Var, pv: &[Var], wp: &mut [Tensor], adam: &mut Adam) -> f32 {
    loss.backward();
    let g: Vec<Tensor> = pv.iter().zip(wp.iter()).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::zeros(ctx, &t.shape))).collect();
    adam.step(wp, &g);
    loss.value().to_vec().await[0]
}
/// Column `k` of the gradient of a scalar field `u` (`[N,1]`) with respect to a `[N,d]` input, as `[N,1]`.
pub fn dcol(ctx: &Arc<Context>, u: &Var, x: &Var, n: usize, d: usize, k: usize) -> Var {
    let g = deriv(u, x);
    let mut m = vec![0.0f32; n * d];
    for i in 0..n {
        m[i * d + k] = 1.0;
    }
    g.mul(&leaf(ctx, &m, &[n, d])).sum(&[1]).reshape(&[n, 1])
}
pub fn rel_l2(pred: &[f32], truth: &[f32]) -> f32 {
    let num: f32 = pred.iter().zip(truth).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f32 = truth.iter().map(|b| b * b).sum();
    (num / den.max(1e-30)).sqrt()
}
/// `n` hash-uniform points in the box `lo..hi` (dimension `lo.len()`), flattened row-major.
pub fn box_points(lo: &[f64], hi: &[f64], n: usize, seed: u32) -> Vec<f32> {
    let d = lo.len();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        for a in 0..d {
            let u = u01((i * d + a) as u32, seed) as f64;
            out.push((lo[a] + (hi[a] - lo[a]) * u) as f32);
        }
    }
    out
}
/// A regular grid over the box, `m` points per axis, flattened row-major (last axis fastest).
pub fn box_grid(lo: &[f64], hi: &[f64], m: usize) -> Vec<f32> {
    let d = lo.len();
    let total = m.pow(d as u32);
    let mut out = Vec::with_capacity(total * d);
    for idx in 0..total {
        let mut rem = idx;
        let mut coords = vec![0.0f32; d];
        for a in (0..d).rev() {
            let i = rem % m;
            rem /= m;
            coords[a] = (lo[a] + (hi[a] - lo[a]) * i as f64 / (m as f64 - 1.0)) as f32;
        }
        out.extend(coords);
    }
    out
}
