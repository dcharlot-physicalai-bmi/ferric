//! **The multi-channel Fourier neural operator layer — a full `w × w` complex matrix per mode.**
//!
//! [`operators::SpectralConv2d`](super::operators::SpectralConv2d) is *channel-diagonal*: one complex
//! scalar per retained mode. That is exactly the form of a scalar constant-coefficient solution operator,
//! which is why its weights can be read against a closed form — and exactly what cannot represent an
//! operator that **couples channels**. The FNO of Li et al. (2010.08895) is the general case: at each
//! retained wavevector `k` the layer applies a learned complex matrix `R(k) ∈ ℂ^{w×w}` to the channel
//! vector of the spectrum. This module is that layer, plus the surrounding network (lift → `L` Fourier
//! layers with a pointwise skip → project).
//!
//! The fabric has no complex dtype and no FFT, so as in the channel-diagonal layer the transform is the
//! Kronecker DFT applied by matmul and real/imaginary parts ride as separate real tensors. The per-mode
//! matrices are **not** a loop over modes: the retained spectrum is compressed to `[B, w, M]`, transposed
//! to mode-major `[M, B, w]`, and multiplied by `R` of shape `[M, w, w]` as one batched GEMM.
//!
//! ⚠ Two parameter groups here are *structurally* non-identifiable and are masked to zero rather than
//! left to drift (they would otherwise sit at their random initialisation and read like learned values —
//! the failure this stack already hit once, see the conjugate-symmetry note in `operators`):
//! * the imaginary part at a **self-mirrored** mode (`k = −k mod n`), whose spectrum is real for a real
//!   field and whose imaginary output is discarded by the symmetry expansion;
//! * the off-diagonal blocks when the layer is built with [`SpectralConvMulti::new_diagonal`].

use crate::{Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// **The separable 2-D DFT** — two 1-D transforms instead of one Kronecker matrix.
///
/// `F₂ = F₁ ⊗ F₁` applied as a single `n² × n²` matmul costs `n⁴` per field and needs an `n² × n²`
/// matrix: 1.07 GB at `n = 128`, times four for the real/imaginary forward and inverse pair. Because the
/// transform is separable it is instead two `n × n` matmuls, one along each axis with a transpose between
/// — `2n³` per field from a 65 kB matrix. Counting the actual dispatches, a forward-plus-inverse pair is
/// **12 matmuls of `n³` against 4 of `n⁴`**, an arithmetic ratio of `n/3`; the matrix memory ratio is the
/// blunt one, `n²` — 16,384× at `n = 128`, the difference between a transform that fits and one that does
/// not. ⚠ Three times the dispatches means the small-`n` end is not automatically faster in wall-clock;
/// `the_separable_transform_against_the_kronecker_one_in_wall_clock` measures it rather than assuming it.
///
/// This is not the FFT — that would take it to `n² log n` and needs a butterfly primitive the fabric does
/// not have (the gather/scatter it would be built from carry no differentiable VJP). It is the whole of
/// the asymptotic gain available from existing differentiable ops, and it is exact: matmul, transpose and
/// reshape only, so the layer stays second-order differentiable.
/// Which of the two exact plans a [`Dft2`] runs. They agree to fp precision; they differ only in how the
/// work is divided, and therefore in where each one wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dft2Plan {
    /// One `n² × n²` matmul per part: 4 dispatches for a forward-plus-inverse pair, `n⁴` each.
    Kronecker,
    /// Two `n × n` matmuls per part with a transpose between: 12 dispatches, `n³` each.
    Separable,
}

pub struct Dft2 {
    pub n: usize,
    pub plan: Dft2Plan,
    /// Forward `[n, n]` (separable) or `[n², n²]` (Kronecker), real and imaginary.
    fr: Tensor,
    fi: Tensor,
    /// Inverse, with the normalisation folded in — `1/n` per pass for the separable plan, `1/n²` once
    /// for the Kronecker one.
    cr: Tensor,
    ci: Tensor,
}

/// Above this grid size the separable plan is chosen. ⚠ Set from measurement, not from the flop count:
/// at batch 48 on Metal a forward-plus-inverse pair runs Kronecker/separable at 1.61/4.51 ms (n=16),
/// 3.46/6.49 ms (n=32) and 15.14/4.18 ms (n=64). The separable plan does `n/3` the arithmetic but three
/// times the dispatches, so it *loses* by 2.8× at n=16 and wins by 3.6× at n=64.
pub const SEPARABLE_ABOVE: usize = 32;

impl Dft2 {
    /// The plan chosen by [`SEPARABLE_ABOVE`] — what a caller should use unless it is measuring.
    pub fn new(ctx: &Arc<Context>, n: usize) -> Self {
        Self::with_plan(ctx, n, if n > SEPARABLE_ABOVE { Dft2Plan::Separable } else { Dft2Plan::Kronecker })
    }

    pub fn with_plan(ctx: &Arc<Context>, n: usize, plan: Dft2Plan) -> Self {
        let tau = std::f64::consts::TAU;
        let w1 = |j: usize, k: usize| -> (f64, f64) {
            let th = -tau * ((j * k) % n) as f64 / n as f64;
            (th.cos(), th.sin())
        };
        let (fr, fi, cr, ci, shape) = match plan {
            Dft2Plan::Separable => {
                let (mut fr, mut fi, mut cr, mut ci) = (vec![0.0f32; n * n], vec![0.0f32; n * n], vec![0.0f32; n * n], vec![0.0f32; n * n]);
                for j in 0..n {
                    for k in 0..n {
                        let (c, sn) = w1(j, k);
                        fr[j * n + k] = c as f32;
                        fi[j * n + k] = sn as f32;
                        // the inverse is the conjugate transpose over n; indexed [k, j] so it contracts k
                        cr[k * n + j] = (c / n as f64) as f32;
                        ci[k * n + j] = (-sn / n as f64) as f32;
                    }
                }
                (fr, fi, cr, ci, vec![n, n])
            }
            Dft2Plan::Kronecker => {
                let nn = n * n;
                let (mut fr, mut fi, mut cr, mut ci) = (vec![0.0f32; nn * nn], vec![0.0f32; nn * nn], vec![0.0f32; nn * nn], vec![0.0f32; nn * nn]);
                for j1 in 0..n {
                    for j2 in 0..n {
                        for k1 in 0..n {
                            for k2 in 0..n {
                                let ((a, b), (c, d)) = (w1(j1, k1), w1(j2, k2));
                                let (re, im) = (a * c - b * d, a * d + b * c);
                                let (row, col) = (j1 * n + j2, k1 * n + k2);
                                fr[row * nn + col] = re as f32;
                                fi[row * nn + col] = im as f32;
                                cr[col * nn + row] = (re / nn as f64) as f32;
                                ci[col * nn + row] = (-im / nn as f64) as f32;
                            }
                        }
                    }
                }
                (fr, fi, cr, ci, vec![nn, nn])
            }
        };
        Dft2 {
            n,
            plan,
            fr: Tensor::from_vec(ctx, &fr, &shape),
            fi: Tensor::from_vec(ctx, &fi, &shape),
            cr: Tensor::from_vec(ctx, &cr, &shape),
            ci: Tensor::from_vec(ctx, &ci, &shape),
        }
    }

    /// Bytes the two plans need for their matrices, at grid size `n` — `(Kronecker, separable)`.
    pub fn matrix_bytes(n: usize) -> (usize, usize) {
        (4 * (n * n) * (n * n) * 4, 4 * (n * n) * 4)
    }

    /// Forward transform of a real field `[…, n, n]`, returning `(Re X̂, Im X̂)` of the same shape.
    pub fn forward(&self, x: &Var) -> (Var, Var) {
        let r = x.value().rank();
        assert!(r >= 2 && x.value().shape[r - 1] == self.n && x.value().shape[r - 2] == self.n, "Dft2 wants […, n, n]");
        let (fr, fi) = (Var::leaf(self.fr.clone()), Var::leaf(self.fi.clone()));
        if self.plan == Dft2Plan::Kronecker {
            let (flat, back) = self.flatten(x);
            return (flat.matmul(&fr).reshape(&back), flat.matmul(&fi).reshape(&back));
        }
        // along the last axis (j2 → k2); the input is real, so no cross terms yet
        let (ar, ai) = (x.matmul(&fr), x.matmul(&fi));
        // along the other axis: transpose so j1 is last, transform, transpose back
        let (tr, ti) = (ar.transpose(r - 2, r - 1), ai.transpose(r - 2, r - 1));
        let br = tr.matmul(&fr).sub(&ti.matmul(&fi));
        let bi = tr.matmul(&fi).add(&ti.matmul(&fr));
        (br.transpose(r - 2, r - 1), bi.transpose(r - 2, r - 1))
    }

    /// Inverse transform, returning the **real part** — the field, for a conjugate-symmetric spectrum.
    pub fn inverse(&self, re: &Var, im: &Var) -> Var {
        let r = re.value().rank();
        let (cr, ci) = (Var::leaf(self.cr.clone()), Var::leaf(self.ci.clone()));
        if self.plan == Dft2Plan::Kronecker {
            let (fre, back) = self.flatten(re);
            let (fim, _) = self.flatten(im);
            return fre.matmul(&cr).sub(&fim.matmul(&ci)).reshape(&back);
        }
        let ar = re.matmul(&cr).sub(&im.matmul(&ci));
        let ai = re.matmul(&ci).add(&im.matmul(&cr));
        let (tr, ti) = (ar.transpose(r - 2, r - 1), ai.transpose(r - 2, r - 1));
        // only the real part survives, so the imaginary half of the second pass is never formed
        tr.matmul(&cr).sub(&ti.matmul(&ci)).transpose(r - 2, r - 1)
    }

    /// `[…, n, n]` → `[…, n²]` for the Kronecker plan, with the shape to restore afterwards.
    fn flatten(&self, x: &Var) -> (Var, Vec<usize>) {
        let back = x.value().shape.clone();
        let mut flat = back[..back.len() - 2].to_vec();
        flat.push(self.n * self.n);
        (x.reshape(&flat), back)
    }
}

/// Multi-channel 2-D spectral convolution on an `n × n` periodic grid with `width` channels, keeping
/// `modes` frequencies per axis. Parameters are `[Rr, Ri]`, each `[M, width, width]` in **[out, in]** order
/// (`R[k][c][d]` is output channel `c`'s share of input channel `d`), indexed by the
/// half-spectrum representatives in [`SpectralConvMulti::reps`] and expanded to the full spectrum by
/// conjugate symmetry.
pub struct SpectralConvMulti {
    pub n: usize,
    pub modes: usize,
    pub width: usize,
    /// Half-spectrum representatives `(k1, k2)` of the retained modes — the mode axis of `Rr`/`Ri`.
    pub reps: Vec<(usize, usize)>,
    /// The separable transform — `2n³` per field from two `n × n` matrices, not `n⁴` from an `n² × n²` one.
    pub dft: Dft2,
    /// `[n², M]`: picks each representative's column out of the full spectrum.
    sel: Tensor,
    /// `[M, n²]`: places a representative's value at its mode and its mirror (`pr`), or with the sign
    /// flipped at the mirror (`pi`) — conjugate symmetry, so a real input gives a real output.
    pr: Tensor,
    pi: Tensor,
    /// `[M, width, width]` 0/1 masks applied to `Ri` and to both parts: zero wherever a weight cannot be
    /// identified from data (see the module note).
    imask: Tensor,
    dmask: Tensor,
    /// `[Rr, Ri]`, each `[M, width, width]`.
    pub params: Vec<Tensor>,
}

impl SpectralConvMulti {
    /// Full multi-channel layer.
    pub fn new(ctx: &Arc<Context>, n: usize, modes: usize, width: usize, seed: u32) -> Self {
        Self::build(ctx, n, modes, width, seed, false)
    }

    /// The same layer with every off-diagonal channel block masked to zero — the channel-diagonal
    /// ablation, identical in every other respect, which is what makes it a fair baseline.
    pub fn new_diagonal(ctx: &Arc<Context>, n: usize, modes: usize, width: usize, seed: u32) -> Self {
        Self::build(ctx, n, modes, width, seed, true)
    }

    fn build(ctx: &Arc<Context>, n: usize, modes: usize, width: usize, seed: u32, diag: bool) -> Self {
        let nn = n * n;
        let keep = |k: usize| k < modes || n - k < modes;
        let mirror = |k: usize| (n - k) % n;
        let mut reps: Vec<(usize, usize)> = Vec::new();
        for k1 in 0..n {
            for k2 in 0..n {
                if keep(k1) && keep(k2) && (k1, k2) <= (mirror(k1), mirror(k2)) {
                    reps.push((k1, k2));
                }
            }
        }
        let m = reps.len();
        let (mut sel, mut pr, mut pi) = (vec![0.0f32; nn * m], vec![0.0f32; m * nn], vec![0.0f32; m * nn]);
        let ww = width * width;
        let (mut imask, mut dmask) = (vec![1.0f32; m * ww], vec![1.0f32; m * ww]);
        for (r, &(k1, k2)) in reps.iter().enumerate() {
            let (i, j) = (k1 * n + k2, mirror(k1) * n + mirror(k2));
            sel[i * m + r] = 1.0;
            pr[r * nn + i] = 1.0;
            if i != j {
                pr[r * nn + j] = 1.0;
                pi[r * nn + i] = 1.0;
                pi[r * nn + j] = -1.0;
            } else {
                for e in &mut imask[r * ww..(r + 1) * ww] {
                    *e = 0.0;
                }
            }
            if diag {
                for a in 0..width {
                    for b in 0..width {
                        if a != b {
                            dmask[r * ww + a * width + b] = 0.0;
                        }
                    }
                }
            }
        }
        // scale the init by 1/width so a wide layer does not start with an exploding channel sum
        let sc = 0.2 / width as f32;
        let init = |off: u32| -> Vec<f32> {
            (0..m * ww).map(|i| sc * (super::util::u01(i as u32 + off, seed) - 0.5)).collect()
        };
        SpectralConvMulti {
            n,
            modes,
            width,
            reps,
            dft: Dft2::new(ctx, n),
            sel: Tensor::from_vec(ctx, &sel, &[nn, m]),
            pr: Tensor::from_vec(ctx, &pr, &[m, nn]),
            pi: Tensor::from_vec(ctx, &pi, &[m, nn]),
            imask: Tensor::from_vec(ctx, &imask, &[m, width, width]),
            dmask: Tensor::from_vec(ctx, &dmask, &[m, width, width]),
            params: vec![
                Tensor::from_vec(ctx, &init(0), &[m, width, width]),
                Tensor::from_vec(ctx, &init(7919), &[m, width, width]),
            ],
        }
    }

    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }

    /// The identifiable weights: `Rr` and `Ri` with the structural masks applied. Read the layer's
    /// learned per-mode matrices through this, never off the raw parameters — a masked-out entry keeps
    /// whatever it was initialised to.
    pub fn effective(&self, pv: &[Var]) -> (Var, Var) {
        let d = Var::leaf(self.dmask.clone());
        (pv[0].mul(&d), pv[1].mul(&d).mul(&Var::leaf(self.imask.clone())))
    }

    /// `y = Re F⁻¹( R(k) · F x )` for a batch of `width`-channel fields `x` of shape `[B, width, n²]`.
    pub fn forward(&self, pv: &[Var], x: &Var) -> Var {
        let nn = self.n * self.n;
        let b = x.value().shape[0];
        assert_eq!(x.value().shape, vec![b, self.width, nn], "SpectralConvMulti wants [B, width, n²]");
        // ⛔ `R` is stored and read as [out, in] — the way an operator matrix is written, and the way the
        // oracle compares it to a closed form. The batched GEMM below contracts `x`'s channel axis against
        // R's LAST axis, so it needs Rᵀ; storing the transpose instead would silently compare the learned
        // matrix to the transpose of the truth, which an identity or diagonal check can never reveal.
        let (rr, ri) = self.effective(pv);
        let (rr, ri) = (rr.transpose(1, 2), ri.transpose(1, 2));
        // forward transform along both grid axes, then keep only the representative modes
        let (xr, xi) = self.dft.forward(&x.reshape(&[b, self.width, self.n, self.n]));
        let (xr, xi) = (xr.reshape(&[b, self.width, nn]), xi.reshape(&[b, self.width, nn]));
        let sel = Var::leaf(self.sel.clone());
        // [B, w, M] → mode-major [M, B, w] so the per-mode matrices are one batched GEMM
        let to_modes = |t: Var| t.matmul(&sel).transpose(0, 2).transpose(1, 2);
        let (ar, ai) = (to_modes(xr), to_modes(xi));
        let (yr, yi) = (
            ar.matmul(&rr).sub(&ai.matmul(&ri)),
            ar.matmul(&ri).add(&ai.matmul(&rr)),
        );
        // back to [B, w, M], then expand to the full spectrum with conjugate symmetry
        let un = |t: Var| t.transpose(0, 1).transpose(1, 2);
        let sh = [b, self.width, self.n, self.n];
        let fr_ = un(yr).matmul(&Var::leaf(self.pr.clone())).reshape(&sh);
        let fi_ = un(yi).matmul(&Var::leaf(self.pi.clone())).reshape(&sh);
        self.dft.inverse(&fr_, &fi_).reshape(&[b, self.width, nn])
    }
}

/// **The FNO network**: lift `d_in → width` pointwise, `layers` Fourier layers
/// (`h ← σ(W h + b + SpectralConvMulti(h))`, no activation on the last), project `width → d_out`
/// pointwise. The pointwise skip `W` is what carries the frequencies the spectral layer truncates; the
/// activation between layers is what makes the whole map **nonlinear**, which is the only reason a stack
/// of these can represent an operator a single spectral layer cannot.
///
/// Fields are `[B, n², d]` — grid axis in the middle, channels last, the layout data arrives in; the
/// layer transposes into channel-major internally.
pub struct Fno2d {
    pub n: usize,
    pub width: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub act: super::Act,
    /// One spectral layer per Fourier layer; their `params` are spliced into [`Fno2d::params`].
    pub spectral: Vec<SpectralConvMulti>,
    /// `[P_lift, b_lift, (Rr, Ri, W, b) × layers, P_proj, b_proj]`.
    pub params: Vec<Tensor>,
    /// When set, the activations are skipped and the network is exactly linear — the ablation that must
    /// fail on a nonlinear operator.
    pub linear: bool,
}

impl Fno2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &Arc<Context>, n: usize, modes: usize, width: usize, layers: usize, d_in: usize, d_out: usize, act: super::Act, seed: u32) -> Self {
        Self::build(ctx, n, modes, width, layers, d_in, d_out, act, seed, false)
    }

    /// The same network with every activation removed: same shape, same parameter count, exactly linear.
    #[allow(clippy::too_many_arguments)]
    pub fn new_linear(ctx: &Arc<Context>, n: usize, modes: usize, width: usize, layers: usize, d_in: usize, d_out: usize, seed: u32) -> Self {
        Self::build(ctx, n, modes, width, layers, d_in, d_out, super::Act::Tanh, seed, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(ctx: &Arc<Context>, n: usize, modes: usize, width: usize, layers: usize, d_in: usize, d_out: usize, act: super::Act, seed: u32, linear: bool) -> Self {
        assert!(layers >= 1, "an FNO needs at least one Fourier layer");
        let rn = |cnt: usize, s: u32, sc: f32| super::randn(cnt, seed.wrapping_add(s), sc);
        let mut params = vec![
            Tensor::from_vec(ctx, &rn(d_in * width, 1, (1.0 / d_in as f32).sqrt()), &[d_in, width]),
            Tensor::zeros(ctx, &[width]),
        ];
        let mut spectral = Vec::with_capacity(layers);
        for l in 0..layers {
            let sc = SpectralConvMulti::new(ctx, n, modes, width, seed.wrapping_add(100 + l as u32));
            params.push(sc.params[0].clone());
            params.push(sc.params[1].clone());
            params.push(Tensor::from_vec(ctx, &rn(width * width, 40 + l as u32, (1.0 / width as f32).sqrt()), &[width, width]));
            params.push(Tensor::zeros(ctx, &[width]));
            spectral.push(sc);
        }
        params.push(Tensor::from_vec(ctx, &rn(width * d_out, 9, (1.0 / width as f32).sqrt()), &[width, d_out]));
        params.push(Tensor::zeros(ctx, &[d_out]));
        Fno2d { n, width, d_in, d_out, act, spectral, params, linear }
    }

    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }

    /// Forward pass on `x` of shape `[B, n², d_in]`; returns `[B, n², d_out]`.
    pub fn forward(&self, pv: &[Var], x: &Var) -> Var {
        let nl = self.spectral.len();
        let mut h = x.matmul(&pv[0]).add(&pv[1]);
        for (l, sc) in self.spectral.iter().enumerate() {
            let p = 2 + 4 * l;
            let s = sc.forward(&pv[p..p + 2], &h.transpose(1, 2)).transpose(1, 2);
            h = h.matmul(&pv[p + 2]).add(&pv[p + 3]).add(&s);
            if !self.linear && l + 1 < nl {
                h = super::features::act(&h, self.act);
            }
        }
        h.matmul(&pv[2 + 4 * nl]).add(&pv[3 + 4 * nl])
    }
}

/// The per-mode matrix of the reference channel-coupling operator used by the oracles:
/// `M(k) = g(k)·A + i·s(k)·B` on the unit torus, with `g(k) = 1/(1+|k|)` even in `k` and
/// `s(k) = sin(2π k₁/n)` odd in `k₁`, so `M(−k) = conj(M(k))` and the operator maps real fields to real
/// fields. `A` couples the two channels; `B` is the phase-shifting part. Returned as `(real, imag)`.
pub fn coupling_operator(n: usize, k1: usize, k2: usize) -> ([[f64; 2]; 2], [[f64; 2]; 2]) {
    let sg = |k: usize| -> f64 { if k <= n / 2 { k as f64 } else { k as f64 - n as f64 } };
    let (q1, q2) = (sg(k1), sg(k2));
    let g = 1.0 / (1.0 + (q1 * q1 + q2 * q2).sqrt());
    let s = (std::f64::consts::TAU * q1 / n as f64).sin();
    let a = [[1.0, 0.6], [-0.4, 0.8]];
    let b = [[0.3, -0.5], [0.2, 0.35]];
    let sc = |f: f64, m: [[f64; 2]; 2]| [[f * m[0][0], f * m[0][1]], [f * m[1][0], f * m[1][1]]];
    (sc(g, a), sc(s, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::operators::SpectralConv2d;
    use crate::sciml::util::{leaf, rel_l2, step, u01, vars};

    fn leaf_t(t: &Tensor) -> Var { Var::leaf(t.clone()) }
    use crate::Adam;

    /// A two-channel sample of the reference operator, built **in real space** as an explicit sum of
    /// sinusoids with closed-form transformed coefficients — it never touches the layer, its DFT matrices
    /// or its parameters, so it is an independent oracle for both. Returns `(x, y)`, each `[2, n²]`.
    fn sample(n: usize, reps: &[(usize, usize)], seed: u32) -> (Vec<f32>, Vec<f32>) {
        let nn = n * n;
        let tau = std::f64::consts::TAU;
        let (mut x, mut y) = (vec![0.0f64; 2 * nn], vec![0.0f64; 2 * nn]);
        for (r, &(k1, k2)) in reps.iter().enumerate() {
            let selfmir = (k1, k2) == ((n - k1) % n, (n - k2) % n);
            let mut ar = [0.0f64; 2];
            let mut ai = [0.0f64; 2];
            for c in 0..2 {
                ar[c] = u01(4 * r as u32 + c as u32, seed) as f64 - 0.5;
                ai[c] = if selfmir { 0.0 } else { u01(4 * r as u32 + c as u32 + 2, seed) as f64 - 0.5 };
            }
            let (mr, mi) = coupling_operator(n, k1, k2);
            // (br + i bi) = (mr + i mi)(ar + i ai)
            let mut br = [0.0f64; 2];
            let mut bi = [0.0f64; 2];
            for c in 0..2 {
                for d in 0..2 {
                    br[c] += mr[c][d] * ar[d] - mi[c][d] * ai[d];
                    bi[c] += mr[c][d] * ai[d] + mi[c][d] * ar[d];
                }
            }
            for j1 in 0..n {
                for j2 in 0..n {
                    let th = tau * ((k1 * j1 + k2 * j2) % n) as f64 / n as f64;
                    let (ct, st) = (th.cos(), th.sin());
                    for c in 0..2 {
                        x[c * nn + j1 * n + j2] += ar[c] * ct - ai[c] * st;
                        y[c * nn + j1 * n + j2] += br[c] * ct - bi[c] * st;
                    }
                }
            }
        }
        (x.iter().map(|&v| v as f32).collect(), y.iter().map(|&v| v as f32).collect())
    }


    /// The separable transform against the **Kronecker** one, which the channel-diagonal layer already
    /// verifies two independent ways (`F⁻¹F = I`, and its learned weights matching the Poisson Green's
    /// function). Both the forward pair and the inverse are compared, so an error in either pass shows.
    #[test]
    fn the_separable_transform_agrees_with_the_kronecker_form() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            for n in [8usize, 16] {
                let nn = n * n;
                let kron = SpectralConv2d::new(&ctx, n, n, 1); // all modes kept; we only use its matrices
                // ⛔ named explicitly: `Dft2::new` picks by `n`, so at these sizes it would hand back the
                // Kronecker plan and this test would compare that plan to itself
                let sep = Dft2::with_plan(&ctx, n, Dft2Plan::Separable);
                assert_eq!(sep.plan, Dft2Plan::Separable, "this test must exercise the separable plan");
                let b = 3usize;
                let xv: Vec<f32> = (0..b * nn).map(|i| u01(i as u32, 31) - 0.5).collect();
                let xflat = leaf(&ctx, &xv, &[b, nn]);
                let kr = xflat.matmul(&leaf_t(&kron.fr));
                let ki = xflat.matmul(&leaf_t(&kron.fi));
                let (sr, si) = sep.forward(&leaf(&ctx, &xv, &[b, n, n]));
                let er = rel_l2(&sr.value().to_vec().await, &kr.value().to_vec().await);
                let ei = rel_l2(&si.value().to_vec().await, &ki.value().to_vec().await);
                // inverse: feed both the SAME spectrum, taken from the Kronecker forward
                let kinv = kr.matmul(&leaf_t(&kron.cr)).sub(&ki.matmul(&leaf_t(&kron.ci)));
                let sinv = sep.inverse(&kr.reshape(&[b, n, n]), &ki.reshape(&[b, n, n]));
                let ev = rel_l2(&sinv.value().to_vec().await, &kinv.value().to_vec().await);
                let rt = rel_l2(&sep.inverse(&sr, &si).value().to_vec().await, &xv);
                eprintln!("  n={n}: separable vs Kronecker — forward Re {er:.2e}, Im {ei:.2e}, inverse {ev:.2e}; round trip {rt:.2e}");
                for (e, what) in [(er, "forward real"), (ei, "forward imaginary"), (ev, "inverse"), (rt, "round trip")] {
                    assert!(e < 1e-4, "n={n}: the separable transform must match the Kronecker form on the {what}: {e:.2e}");
                }
            }
        });
    }

    /// The point of the separable form: it runs at a resolution where the Kronecker matrices do not fit.
    #[test]
    fn the_separable_transform_runs_where_the_kronecker_matrices_do_not_fit() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let n = 128usize;
            let (kron_bytes, sep_bytes) = Dft2::matrix_bytes(n);
            let sep = Dft2::new(&ctx, n);
            assert_eq!(sep.plan, Dft2Plan::Separable, "at n={n} the default plan must be the separable one");
            let xv: Vec<f32> = (0..n * n).map(|i| u01(i as u32, 77) - 0.5).collect();
            let x = leaf(&ctx, &xv, &[1, n, n]);
            let (re, im) = sep.forward(&x);
            let back = sep.inverse(&re, &im).value().to_vec().await;
            let e = rel_l2(&back, &xv);
            eprintln!(
                "  n={n}: round trip {e:.2e}; DFT matrices would be {:.2} GB as Kronecker, are {:.1} kB separable ({}x)",
                kron_bytes as f64 / 1e9,
                sep_bytes as f64 / 1e3,
                kron_bytes / sep_bytes
            );
            assert!(e < 1e-3, "the separable transform must invert itself at n={n}: {e:.2e}");
            assert!(kron_bytes / sep_bytes >= 1000, "the whole point is the memory ratio: {}x", kron_bytes / sep_bytes);
        });
    }

    /// With `R(k) = I` at every mode the multi-channel layer must be the identity — the check that the
    /// compress / mode-major transpose / expand plumbing puts every channel and mode back where it came
    /// from. A single transposed axis or a mis-ordered `sel`/`pr` pair breaks this.
    #[test]
    fn the_layer_with_identity_weights_is_the_identity() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, w, b) = (8usize, 3usize, 2usize);
            let sc = SpectralConvMulti::new(&ctx, n, n, w, 1);
            let m = sc.reps.len();
            let mut rr = vec![0.0f32; m * w * w];
            for r in 0..m {
                for c in 0..w {
                    rr[r * w * w + c * w + c] = 1.0;
                }
            }
            let pv = vec![leaf(&ctx, &rr, &[m, w, w]), leaf(&ctx, &vec![0.0; m * w * w], &[m, w, w])];
            let x: Vec<f32> = (0..b * w * n * n).map(|i| u01(i as u32, 5) - 0.5).collect();
            let y = sc.forward(&pv, &leaf(&ctx, &x, &[b, w, n * n])).value().to_vec().await;
            let err = rel_l2(&y, &x);
            eprintln!("  identity weights, 8×8 × 3 channels: rel-L2 {err:.2e}");
            assert!(err < 1e-5, "R = I must be the identity map: {err:.2e}");
        });
    }

    /// At width 1 the multi-channel layer **is** the channel-diagonal layer: same reps, same weights,
    /// same output. This pins the two implementations to each other, so the closed-form Green's-function
    /// check already proved for `SpectralConv2d` transfers to this layer's width-1 case.
    #[test]
    fn width_one_reproduces_the_channel_diagonal_layer() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, modes, b) = (8usize, 3usize, 3usize);
            let diag = SpectralConv2d::new(&ctx, n, modes, 11);
            let multi = SpectralConvMulti::new(&ctx, n, modes, 1, 11);
            assert_eq!(diag.reps, multi.reps, "the two layers must enumerate the same half-spectrum");
            // ⚠ at this `n` both sides run the Kronecker plan, so an exact 0 here is agreement of the
            // layer logic, NOT of the transforms; the transforms are compared in
            // `the_separable_transform_agrees_with_the_kronecker_form`
            assert_eq!(multi.dft.plan, Dft2Plan::Kronecker, "expected the small-n default");
            let nr = diag.reps.len();
            let wr: Vec<f32> = (0..nr).map(|i| u01(i as u32, 21) - 0.5).collect();
            // the imaginary weight of a self-mirrored mode is masked in the multi-channel layer and
            // expands to nothing in the diagonal one, so both must ignore whatever is placed there
            let wi: Vec<f32> = (0..nr).map(|i| u01(i as u32, 22) - 0.5).collect();
            let x: Vec<f32> = (0..b * n * n).map(|i| u01(i as u32, 23) - 0.5).collect();
            let yd = diag
                .forward(&[leaf(&ctx, &wr, &[1, nr]), leaf(&ctx, &wi, &[1, nr])], &leaf(&ctx, &x, &[b, n * n]))
                .value()
                .to_vec()
                .await;
            let ym = multi
                .forward(&[leaf(&ctx, &wr, &[nr, 1, 1]), leaf(&ctx, &wi, &[nr, 1, 1])], &leaf(&ctx, &x, &[b, 1, n * n]))
                .value()
                .to_vec()
                .await;
            let err = rel_l2(&ym, &yd);
            eprintln!("  width-1 multi vs channel-diagonal: rel-L2 {err:.2e}");
            assert!(err < 1e-5, "the two layers must agree at width 1: {err:.2e}");
        });
    }


    /// The layer **with the reference weights installed** must be the reference operator — no training in
    /// the loop. This is what localises a disagreement between the closed-form sample and the layer's
    /// per-mode convention: if it holds, a trained weight that differs from `M(k)` is a training or
    /// read-out fault; if it fails, the conventions differ and no amount of training would reconcile them.
    #[test]
    fn the_layer_with_the_reference_weights_is_the_reference_operator() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, modes, w) = (16usize, 4usize, 2usize);
            let nn = n * n;
            let sc = SpectralConvMulti::new(&ctx, n, modes, w, 1);
            let m = sc.reps.len();
            let (mut rr, mut ri) = (vec![0.0f32; m * w * w], vec![0.0f32; m * w * w]);
            for (r, &(k1, k2)) in sc.reps.iter().enumerate() {
                let (mr, mi) = coupling_operator(n, k1, k2);
                for c in 0..w {
                    for d in 0..w {
                        rr[r * w * w + c * w + d] = mr[c][d] as f32;
                        ri[r * w * w + c * w + d] = mi[c][d] as f32;
                    }
                }
            }
            let pv = vec![leaf(&ctx, &rr, &[m, w, w]), leaf(&ctx, &ri, &[m, w, w])];
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for k in 0..4u32 {
                let (x, y) = sample(n, &sc.reps, k + 1);
                xs.extend_from_slice(&x);
                ys.extend_from_slice(&y);
            }
            let p = sc.forward(&pv, &leaf(&ctx, &xs, &[4, w, nn])).value().to_vec().await;
            let err = rel_l2(&p, &ys);
            // and mode by mode, on a field carrying that mode alone — a whole-field norm can hide a
            // single bad mode under the modes that dominate it
            let (mut worst, mut worst_at) = (0.0f32, (0usize, 0usize));
            for i in 0..m {
                let (x1, y1) = sample(n, &sc.reps[i..i + 1], 7);
                let p1 = sc.forward(&pv, &leaf(&ctx, &x1, &[1, w, nn])).value().to_vec().await;
                let e = rel_l2(&p1, &y1);
                if e > worst {
                    worst = e;
                    worst_at = sc.reps[i];
                }
            }
            // where does it disagree? report the per-mode ratio the two conventions imply
            eprintln!("  reference weights installed: rel-L2 {err:.3e}; worst single mode {worst:.3e} at {worst_at:?}");
            assert!(err < 1e-4, "the layer's per-mode convention must match the closed form: {err:.3e}");
            assert!(worst < 1e-4, "mode {worst_at:?} alone disagrees with the closed form: {worst:.3e}");
        });
    }


    /// The **layer** — not just the transform — at a grid the Kronecker form could not have held, checked
    /// the same way as at `n = 16`: reference weights installed, output against the closed form. This is
    /// what the separable transform buys, stated as a working resolution rather than a flop count.
    #[test]
    fn the_layer_runs_at_a_resolution_the_kronecker_form_could_not_hold() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, modes, w) = (64usize, 8usize, 2usize);
            let nn = n * n;
            let sc = SpectralConvMulti::new(&ctx, n, modes, w, 1);
            assert_eq!(sc.dft.plan, Dft2Plan::Separable, "at n={n} the layer must have taken the separable plan");
            let m = sc.reps.len();
            let (mut rr, mut ri) = (vec![0.0f32; m * w * w], vec![0.0f32; m * w * w]);
            for (r, &(k1, k2)) in sc.reps.iter().enumerate() {
                let (mr, mi) = coupling_operator(n, k1, k2);
                for c in 0..w {
                    for d in 0..w {
                        rr[r * w * w + c * w + d] = mr[c][d] as f32;
                        ri[r * w * w + c * w + d] = mi[c][d] as f32;
                    }
                }
            }
            let pv = vec![leaf(&ctx, &rr, &[m, w, w]), leaf(&ctx, &ri, &[m, w, w])];
            let (x, y) = sample(n, &sc.reps, 5);
            let p = sc.forward(&pv, &leaf(&ctx, &x, &[1, w, nn])).value().to_vec().await;
            let e = rel_l2(&p, &y);
            let (kron_bytes, sep_bytes) = Dft2::matrix_bytes(n);
            eprintln!(
                "  n={n}, {m} retained modes, {w} channels: rel-L2 vs the closed form {e:.3e}; transform matrices {:.1} MB as Kronecker, {:.1} kB separable",
                kron_bytes as f64 / 1e6,
                sep_bytes as f64 / 1e3
            );
            assert!(e < 1e-4, "the layer must stay exact at n={n}: {e:.3e}");
            // and it still differentiates — the shapes changed, the tape did not
            let loss = sc.forward(&pv, &leaf(&ctx, &x, &[1, w, nn])).mul(&leaf(&ctx, &x, &[1, w, nn])).sum_all();
            loss.backward();
            let g = pv[0].grad().expect("the per-mode weights must receive a gradient at this resolution");
            assert_eq!(g.shape, vec![m, w, w], "gradient shape");
        });
    }


    /// ⚠ **Flops are not wall-clock.** The separable form does `12` matmuls of `n³` where the Kronecker
    /// form does `4` of `n⁴` — an arithmetic ratio of `n/3`, but three times the dispatches. On a GPU with
    /// per-dispatch overhead the small-`n` end can go either way, so this measures it instead of asserting
    /// it. Run with `-- --ignored --nocapture`; it prints, and only asserts the thing that cannot flip:
    /// that the advantage grows with `n`.
    #[ignore = "a wall-clock measurement, not a correctness check; run with -- --ignored --nocapture"]
    #[test]
    fn the_separable_transform_against_the_kronecker_one_in_wall_clock() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (b, iters) = (48usize, 20);
            let mut ratios = Vec::new();
            for n in [16usize, 32, 64] {
                let nn = n * n;
                let kron = SpectralConv2d::new(&ctx, n, n, 1);
                let sep = Dft2::with_plan(&ctx, n, Dft2Plan::Separable);
                let xv: Vec<f32> = (0..b * nn).map(|i| u01(i as u32, 3) - 0.5).collect();
                let (xf, xg) = (leaf(&ctx, &xv, &[b, nn]), leaf(&ctx, &xv, &[b, n, n]));
                let (fr, fi, cr, ci) = (leaf_t(&kron.fr), leaf_t(&kron.fi), leaf_t(&kron.cr), leaf_t(&kron.ci));
                let kron_once = |_: usize| {
                    let (a, c) = (xf.matmul(&fr), xf.matmul(&fi));
                    a.matmul(&cr).sub(&c.matmul(&ci))
                };
                let sep_once = |_: usize| {
                    let (a, c) = sep.forward(&xg);
                    sep.inverse(&a, &c)
                };
                // warm up both (kernel selection and any autotuning happen on first use)
                let _ = kron_once(0).value().to_vec().await;
                let _ = sep_once(0).value().to_vec().await;
                let t0 = std::time::Instant::now();
                for i in 0..iters { let _ = kron_once(i).value().to_vec().await; }
                let tk = t0.elapsed().as_secs_f64() / iters as f64;
                let t1 = std::time::Instant::now();
                for i in 0..iters { let _ = sep_once(i).value().to_vec().await; }
                let ts = t1.elapsed().as_secs_f64() / iters as f64;
                let (kb, sb) = Dft2::matrix_bytes(n);
                eprintln!(
                    "  n={n:3} (batch {b}): Kronecker {:8.2} ms   separable {:8.2} ms   speedup {:5.2}x   matrices {:8.1} MB vs {:6.1} kB",
                    tk * 1e3, ts * 1e3, tk / ts, kb as f64 / 1e6, sb as f64 / 1e3
                );
                ratios.push(tk / ts);
            }
            assert!(
                ratios[2] > ratios[0],
                "the separable form's advantage must grow with n (that is the whole claim): {ratios:?}"
            );
        });
    }

    /// ⭐⭐ **It learns a channel-coupling operator and its per-mode matrices ARE `M(k)`.** The reference
    /// operator mixes the two channels (`A` has off-diagonal 0.6 and −0.4), so a channel-diagonal layer
    /// cannot represent it at all. Both arms train on the same data, same steps, same optimiser; the
    /// diagonal arm is this same layer with its off-diagonal blocks masked, so the only difference is the
    /// channel coupling. The learned matrices are then read against the closed form entry by entry —
    /// a good loss is not evidence that the weights mean anything (see `operators`).
    #[ignore = "trains two multi-channel spectral operators on the GPU (~2 min); run with -- --ignored"]
    #[test]
    fn a_multi_channel_layer_recovers_the_per_mode_matrix_and_the_diagonal_one_cannot() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, modes, w, b) = (16usize, 4usize, 2usize, 24usize);
            let (nn, steps, decay) = (n * n, 4000u32, 2000u32);
            let train = |diag: bool| {
                let ctx = ctx.clone();
                async move {
                    let sc = if diag {
                        SpectralConvMulti::new_diagonal(&ctx, n, modes, w, 3)
                    } else {
                        SpectralConvMulti::new(&ctx, n, modes, w, 3)
                    };
                    let mut wp = sc.params.clone();
                    let mut adam = Adam::new(&wp, 3e-3);
                    for ep in 0..steps + decay {
                        if ep == steps {
                            adam = Adam::new(&wp, 3e-4);
                        }
                        let (mut xs, mut ys) = (vec![0.0f32; b * w * nn], vec![0.0f32; b * w * nn]);
                        for bi in 0..b {
                            let (x, y) = sample(n, &sc.reps, ep.wrapping_mul(97) + bi as u32 + 1);
                            xs[bi * w * nn..(bi + 1) * w * nn].copy_from_slice(&x);
                            ys[bi * w * nn..(bi + 1) * w * nn].copy_from_slice(&y);
                        }
                        let pv = vars(&wp);
                        let d = sc.forward(&pv, &leaf(&ctx, &xs, &[b, w, nn])).sub(&leaf(&ctx, &ys, &[b, w, nn]));
                        let loss = d.mul(&d).mean_all();
                        step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                    }
                    // held out
                    let pv = vars(&wp);
                    let (mut num, mut den) = (0.0f32, 0.0f32);
                    for k in 0..40u32 {
                        let (x, y) = sample(n, &sc.reps, 800_000 + k);
                        let p = sc.forward(&pv, &leaf(&ctx, &x, &[1, w, nn])).value().to_vec().await;
                        for j in 0..w * nn {
                            num += (p[j] - y[j]).powi(2);
                            den += y[j].powi(2);
                        }
                    }
                    let (er, ei) = sc.effective(&pv);
                    (sc, (num / den).sqrt(), er.value().to_vec().await, ei.value().to_vec().await)
                }
            };
            let (_, rel_diag, _, _) = train(true).await;
            let (sc, rel_multi, rr, ri) = train(false).await;
            // the learned per-mode matrices against the closed form, entry by entry
            let (mut worst, mut worst_at) = (0.0f64, (0usize, 0usize));
            let mut scale = 0.0f64;
            for (r, &(k1, k2)) in sc.reps.iter().enumerate() {
                let (mr, mi) = coupling_operator(n, k1, k2);
                for c in 0..w {
                    for d in 0..w {
                        let i = r * w * w + c * w + d;
                        let selfmir = (k1, k2) == ((n - k1) % n, (n - k2) % n);
                        let want_i = if selfmir { 0.0 } else { mi[c][d] };
                        let e = (rr[i] as f64 - mr[c][d]).abs().max((ri[i] as f64 - want_i).abs());
                        scale = scale.max(mr[c][d].abs().max(want_i.abs()));
                        if e > worst {
                            worst = e;
                            worst_at = (k1, k2);
                        }
                    }
                }
            }
            eprintln!("  multi-channel: held-out rel-L2 {rel_multi:.4}; channel-diagonal ablation {rel_diag:.4}");
            eprintln!("  worst absolute deviation of a learned matrix entry from M(k): {worst:.3e} at k={worst_at:?} (entries span ±{scale:.3})");
            // measured 0.0001 held-out and 1.24e-4 worst entry against entries spanning ±1.0
            assert!(rel_multi < 0.002, "the multi-channel layer must learn the operator: {rel_multi:.4}");
            assert!(
                rel_diag > 10.0 * rel_multi,
                "the ablation must FAIL, or this fixture proves nothing about channel mixing: diagonal {rel_diag:.4} vs multi {rel_multi:.4}"
            );
            assert!(worst < 1e-3, "the weights must BE M(k): worst entry off by {worst:.3e} at k={worst_at:?}");
        });
    }

    /// A sample of the nonlinear reference operator `f ↦ u = s + s²`, where `s` is the band-limited forcing
    /// `f` smoothed by the Poisson-type multiplier `4/|k|²` (scaled so `s` is O(1), not the Green's function's
    /// `1/(4π²|k|²)`). Built in real space from explicit sinusoids with
    /// the multiplier applied in closed form per mode, and the square taken pointwise — it touches
    /// neither the network nor its DFT matrices. Returns `(f, s, u)`, each `[n²]`.
    fn nonlinear_sample(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (nn, tau) = (n * n, std::f64::consts::TAU);
        let (mut f, mut s) = (vec![0.0f64; nn], vec![0.0f64; nn]);
        let mut c = 0u32;
        for k1 in -3i32..=3 {
            for k2 in -3i32..=3 {
                if k1 == 0 && k2 == 0 {
                    continue;
                }
                let a = 0.3 * (u01(c, seed) as f64 - 0.5);
                let ph = tau * u01(c + 5000, seed) as f64;
                c += 1;
                let kk = (k1 * k1 + k2 * k2) as f64;
                for i in 0..n {
                    for j in 0..n {
                        let th = tau * (k1 as f64 * i as f64 + k2 as f64 * j as f64) / n as f64 + ph;
                        let v = a * th.cos();
                        f[i * n + j] += v;
                        s[i * n + j] += 4.0 * v / kk;
                    }
                }
            }
        }
        let u: Vec<f32> = s.iter().map(|&v| (v + v * v) as f32).collect();
        (f.iter().map(|&v| v as f32).collect(), s.iter().map(|&v| v as f32).collect(), u)
    }

    /// ⭐⭐ **The FNO network learns a nonlinear operator, and the same network without its activations
    /// cannot.** The target is `f ↦ s + s²` with `s` the Poisson solve — a linear map composed with a
    /// pointwise square. The ablation is this network with `linear: true`: identical shape, identical
    /// parameter count, identical data and schedule, activations removed. A linear map cannot produce the
    /// quadratic term, so its error is floored by that term's share of the target — which the fixture
    /// **measures and asserts** rather than assuming, because a quadratic term too small to matter would
    /// make the comparison vacuous.
    #[ignore = "trains two FNO networks on the GPU (~3 min); run with -- --ignored"]
    #[test]
    fn the_fno_learns_a_nonlinear_operator_and_the_linear_ablation_cannot() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, modes, w, layers, b) = (16usize, 4usize, 8usize, 2usize, 16usize);
            let (nn, steps, decay) = (n * n, 3000u32, 1500u32);
            // the fixture's premise, in numbers: the quadratic term must carry real weight in the target
            let (mut e2, mut eu) = (0.0f64, 0.0f64);
            for k in 0..40u32 {
                let (_, s, u) = nonlinear_sample(n, k + 1);
                for j in 0..nn {
                    e2 += (s[j] as f64).powi(4);
                    eu += (u[j] as f64).powi(2);
                }
            }
            let share = (e2 / eu).sqrt();
            eprintln!("  target f ↦ s + s²: ‖s²‖ / ‖u‖ = {share:.3}");
            assert!(share > 0.3, "the quadratic term must dominate enough for the ablation to be forced to fail: {share:.3}");

            let train = |linear: bool| {
                let ctx = ctx.clone();
                async move {
                    let net = if linear {
                        Fno2d::new_linear(&ctx, n, modes, w, layers, 1, 1, 5)
                    } else {
                        Fno2d::new(&ctx, n, modes, w, layers, 1, 1, crate::sciml::Act::Tanh, 5)
                    };
                    let mut wp = net.params.clone();
                    let mut adam = Adam::new(&wp, 3e-3);
                    for ep in 0..steps + decay {
                        if ep == steps {
                            adam = Adam::new(&wp, 3e-4);
                        }
                        let (mut fs, mut us) = (vec![0.0f32; b * nn], vec![0.0f32; b * nn]);
                        for bi in 0..b {
                            let (f, _, u) = nonlinear_sample(n, ep.wrapping_mul(131) + bi as u32 + 1);
                            fs[bi * nn..(bi + 1) * nn].copy_from_slice(&f);
                            us[bi * nn..(bi + 1) * nn].copy_from_slice(&u);
                        }
                        let pv = vars(&wp);
                        let d = net.forward(&pv, &leaf(&ctx, &fs, &[b, nn, 1])).sub(&leaf(&ctx, &us, &[b, nn, 1]));
                        let loss = d.mul(&d).mean_all();
                        step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
                    }
                    let pv = vars(&wp);
                    let (mut num, mut den) = (0.0f32, 0.0f32);
                    for k in 0..40u32 {
                        let (f, _, u) = nonlinear_sample(n, 900_000 + k);
                        let p = net.forward(&pv, &leaf(&ctx, &f, &[1, nn, 1])).value().to_vec().await;
                        for j in 0..nn {
                            num += (p[j] - u[j]).powi(2);
                            den += u[j].powi(2);
                        }
                    }
                    (num / den).sqrt()
                }
            };
            let rel_lin = train(true).await;
            let rel_nl = train(false).await;
            eprintln!("  FNO held-out rel-L2 {rel_nl:.4}; linear ablation {rel_lin:.4}");
            // measured: FNO 0.0248, linear ablation 0.5999, with the quadratic term carrying 73 % of the target
            assert!(rel_nl < 0.04, "the FNO must learn the nonlinear operator: {rel_nl:.4}");
            assert!(rel_lin > 3.0 * rel_nl, "the ablation must fail, or the activations are not what did the work: linear {rel_lin:.4} vs FNO {rel_nl:.4}");
        });
    }
}
