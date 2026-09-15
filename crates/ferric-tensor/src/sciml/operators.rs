//! **A 2-D spectral convolution — the Fourier-neural-operator layer on a periodic grid.**
//!
//! This fabric has no complex dtype and no FFT, so the transform is the discrete Fourier matrix
//! applied by matmul, exactly as the 1-D `fno` example does, lifted to two dimensions by the Kronecker
//! product: a field on an `n × n` grid is a row of length `n²`, and `F₂ = F₁ ⊗ F₁` is an `n² × n²`
//! matrix. At `n = 16` that is a 256 × 256 matmul per transform — trivial on the device, and the
//! asymptotic loss against an FFT (`O(n⁴)` against `O(n² log n)`) is irrelevant at operator-learning
//! resolutions. Real and imaginary parts are carried as separate real tensors.
//!
//! The layer keeps the `modes × modes` lowest frequencies in each axis (positive and negative) and
//! multiplies each by a learned complex weight — a **channel-diagonal** spectral convolution, the form in
//! which the solution operator of a constant-coefficient linear PDE is exactly representable, and the
//! reason the oracle can check the learned weights against a closed form. ⚠ The full multi-channel FNO
//! (a `w × w` complex matrix per mode plus a pointwise channel mixing) is the follow-on, not this layer.

use crate::{Tensor, Var};
use ferric_core::Context;
use std::sync::Arc;

/// Channel-diagonal 2-D spectral convolution on an `n × n` periodic grid, keeping `modes` frequencies
/// (`|k| < modes` in each axis). Parameters: `[Rr, Ri]`, one complex weight per **half-spectrum** retained
/// mode, expanded to the full spectrum with conjugate symmetry so the output of a real field is real and
/// every weight is identifiable.
pub struct SpectralConv2d {
    pub n: usize,
    pub modes: usize,
    /// Forward DFT (Kronecker), real and imaginary parts, `[n², n²]`.
    pub fr: Tensor,
    pub fi: Tensor,
    /// Inverse DFT with the `1/n²` normalisation folded in, real and imaginary parts, `[n², n²]`.
    pub cr: Tensor,
    pub ci: Tensor,
    /// Half-spectrum representatives of the retained modes as `(k1, k2)`; the layer's parameters live here.
    pub reps: Vec<(usize, usize)>,
    /// Expansion from half-spectrum parameters to the full spectrum, `[reps, n²]`: `pr` places each
    /// weight at its mode and its mirror, `pi` places it with a sign flip at the mirror (conjugate symmetry).
    pub pr: Tensor,
    pub pi: Tensor,
    /// `[Rr, Ri]`, each `[1, reps]`.
    pub params: Vec<Tensor>,
}

impl SpectralConv2d {
    pub fn new(ctx: &Arc<Context>, n: usize, modes: usize, seed: u32) -> Self {
        let nn = n * n;
        let tau = std::f64::consts::TAU;
        // 1-D DFT entries e^{-2πi jk/n}
        let w1 = |j: usize, k: usize| -> (f64, f64) {
            let th = -tau * ((j * k) % n) as f64 / n as f64;
            (th.cos(), th.sin())
        };
        let mut fr = vec![0.0f32; nn * nn];
        let mut fi = vec![0.0f32; nn * nn];
        let mut cr = vec![0.0f32; nn * nn];
        let mut ci = vec![0.0f32; nn * nn];
        for j1 in 0..n {
            for j2 in 0..n {
                for k1 in 0..n {
                    for k2 in 0..n {
                        let (a, b) = w1(j1, k1);
                        let (c, d) = w1(j2, k2);
                        // (a + ib)(c + id)
                        let (re, im) = (a * c - b * d, a * d + b * c);
                        let row = j1 * n + j2;
                        let colk = k1 * n + k2;
                        fr[row * nn + colk] = re as f32;
                        fi[row * nn + colk] = im as f32;
                        // inverse: conjugate transpose / n², laid out so that Y = Ŷ · C
                        cr[colk * nn + row] = (re / nn as f64) as f32;
                        ci[colk * nn + row] = (-im / nn as f64) as f32;
                    }
                }
            }
        }
        // ⛔ A real field's spectrum is conjugate-symmetric, so a layer with independent weights at k and
        // −k can only identify R(k) + conj(R(−k)): the first version learned the operator to 1.4 % and
        // its weights matched nothing (k = (1,1): 0.0433 against 0.0127). Parameters therefore live on a
        // half-spectrum and are expanded with the symmetry built in.
        let keep = |k: usize| k < modes || n - k < modes;
        let mirror = |k: usize| (n - k) % n;
        let mut reps: Vec<(usize, usize)> = Vec::new();
        for k1 in 0..n {
            for k2 in 0..n {
                if !(keep(k1) && keep(k2)) {
                    continue;
                }
                let m = (mirror(k1), mirror(k2));
                if (k1, k2) <= m {
                    reps.push((k1, k2));
                }
            }
        }
        let nr = reps.len();
        let mut pr = vec![0.0f32; nr * nn];
        let mut pi = vec![0.0f32; nr * nn];
        for (r, &(k1, k2)) in reps.iter().enumerate() {
            let (m1, m2) = (mirror(k1), mirror(k2));
            let (i, j) = (k1 * n + k2, m1 * n + m2);
            pr[r * nn + i] = 1.0;
            if i != j {
                pr[r * nn + j] = 1.0;
                pi[r * nn + i] = 1.0;
                pi[r * nn + j] = -1.0;
            } // a self-mirrored mode must be real: its imaginary weight expands to nothing
        }
        let init: Vec<f32> = (0..nr).map(|i| 0.1 * (super::util::u01(i as u32, seed) - 0.5)).collect();
        SpectralConv2d {
            n,
            modes,
            fr: Tensor::from_vec(ctx, &fr, &[nn, nn]),
            fi: Tensor::from_vec(ctx, &fi, &[nn, nn]),
            cr: Tensor::from_vec(ctx, &cr, &[nn, nn]),
            ci: Tensor::from_vec(ctx, &ci, &[nn, nn]),
            reps,
            pr: Tensor::from_vec(ctx, &pr, &[nr, nn]),
            pi: Tensor::from_vec(ctx, &pi, &[nr, nn]),
            params: vec![Tensor::from_vec(ctx, &init, &[1, nr]), Tensor::zeros(ctx, &[1, nr])],
        }
    }

    pub fn vars(&self) -> Vec<Var> {
        self.params.iter().map(|t| Var::leaf(t.clone())).collect()
    }

    /// `y = Re F⁻¹( R ⊙ F x )` for a batch of fields `x` (`[B, n²]`, row-major grids). `pv = [Rr, Ri]`.
    pub fn forward(&self, pv: &[Var], x: &Var) -> Var {
        let (fr, fi, cr, ci) = (Var::leaf(self.fr.clone()), Var::leaf(self.fi.clone()), Var::leaf(self.cr.clone()), Var::leaf(self.ci.clone()));
        let (rr, ri) = (pv[0].matmul(&Var::leaf(self.pr.clone())), pv[1].matmul(&Var::leaf(self.pi.clone())));
        let (xr, xi) = (x.matmul(&fr), x.matmul(&fi));
        // (xr + i xi)(rr + i ri): the `[1, n²]` weights broadcast over the batch
        let yr = xr.mul(&rr).sub(&xi.mul(&ri));
        let yi = xr.mul(&ri).add(&xi.mul(&rr));
        // Re[(yr + i yi)(cr + i ci)] = yr cr − yi ci
        yr.matmul(&cr).sub(&yi.matmul(&ci))
    }
}

/// Multiplier of the exact 2-D Poisson solution operator `−Δu = f` (periodic, mean zero) at integer
/// wavenumbers `(k1, k2)` on the unit torus: `1 / (4π² |k|²)`, `0` at `k = 0`.
pub fn poisson_multiplier(n: usize, k1: usize, k2: usize) -> f64 {
    let s = |k: usize| -> f64 { if k <= n / 2 { k as f64 } else { k as f64 - n as f64 } };
    let kk = s(k1) * s(k1) + s(k2) * s(k2);
    if kk == 0.0 { 0.0 } else { 1.0 / (4.0 * std::f64::consts::PI.powi(2) * kk) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::util::{leaf, rel_l2, step, u01, vars};
    use crate::Adam;

    /// The transform is a transform: `F⁻¹ F x = x` on random fields, through the Kronecker matrices.
    #[test]
    fn the_kronecker_dft_inverts_itself() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let n = 8;
            let sc = SpectralConv2d::new(&ctx, n, n, 1); // all modes kept
            let x: Vec<f32> = (0..3 * n * n).map(|i| u01(i as u32, 5) - 0.5).collect();
            // with R = 1 + 0i the layer is the identity
            let nr = sc.reps.len();
            let pv = vec![leaf(&ctx, &vec![1.0; nr], &[1, nr]), leaf(&ctx, &vec![0.0; nr], &[1, nr])];
            let y = sc.forward(&pv, &leaf(&ctx, &x, &[3, n * n])).value().to_vec().await;
            let err = rel_l2(&y, &x);
            eprintln!("  F⁻¹F on 8×8 random fields: rel-L2 {err:.2e}");
            assert!(err < 1e-5, "the inverse must undo the forward transform: {err:.2e}");
        });
    }

    /// ⭐⭐ **It learns the 2-D Poisson solution operator and its weights ARE the Green's function.** Random
    /// band-limited forcings on the torus, exact solutions by the spectral formula; after training on
    /// `f ↦ u` pairs, held-out error is small and the learned multiplier at every retained mode matches
    /// `1/(4π²|k|²)` — the interpretable check that the layer learned the operator, not a lookup table.
    #[ignore = "trains a spectral operator on the GPU (~3 min); run with -- --ignored"]
    #[test]
    fn a_2d_spectral_operator_recovers_the_poisson_greens_function() {
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (n, modes, b, steps) = (16usize, 4usize, 32usize, 5000usize);
            let nn = n * n;
            let tau = std::f32::consts::TAU;
            // a forcing with modes |k| ≤ 3 and its exact periodic solution
            let sample = |seed: u32| -> (Vec<f32>, Vec<f32>) {
                let mut f = vec![0.0f32; nn];
                let mut u = vec![0.0f32; nn];
                let mut c = 0u32;
                for k1 in -3i32..=3 {
                    for k2 in -3i32..=3 {
                        if k1 == 0 && k2 == 0 { continue; }
                        let a = u01(c, seed) - 0.5;
                        let ph = tau * u01(c + 1000, seed);
                        c += 1;
                        let kk = (k1 * k1 + k2 * k2) as f32;
                        for i in 0..n {
                            for j in 0..n {
                                let x = i as f32 / n as f32;
                                let y = j as f32 / n as f32;
                                let v = a * (tau * (k1 as f32 * x + k2 as f32 * y) + ph).cos();
                                f[i * n + j] += v;
                                u[i * n + j] += v / (tau * tau * kk);
                            }
                        }
                    }
                }
                (f, u)
            };
            let sc = SpectralConv2d::new(&ctx, n, modes, 3);
            let mut wp = sc.params.clone();
            // ⚠ two stages: the high retained modes carry multipliers ~20x smaller than the low ones and
            // sit at Adam's noise floor at 3e-3 (worst mode 12% off, and worse with MORE steps), so the
            // second stage decays the rate
            let mut adam = Adam::new(&wp, 3e-3);
            for ep in 0..(steps + 2000) as u32 {
                if ep == steps as u32 {
                    adam = Adam::new(&wp, 3e-4);
                }
                let mut fs = vec![0.0f32; b * nn];
                let mut us = vec![0.0f32; b * nn];
                for bi in 0..b {
                    let (f, u) = sample(ep.wrapping_mul(97) + bi as u32 + 1);
                    fs[bi * nn..(bi + 1) * nn].copy_from_slice(&f);
                    us[bi * nn..(bi + 1) * nn].copy_from_slice(&u);
                }
                let pv = vars(&wp);
                let pred = sc.forward(&pv, &leaf(&ctx, &fs, &[b, nn]));
                let d = pred.sub(&leaf(&ctx, &us, &[b, nn]));
                let loss = d.mul(&d).mean_all();
                step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
            // held-out
            let (mut num, mut den) = (0.0f32, 0.0f32);
            let pv = vars(&wp);
            for k in 0..50u32 {
                let (f, u) = sample(700_000 + k);
                let p = sc.forward(&pv, &leaf(&ctx, &f, &[1, nn])).value().to_vec().await;
                for j in 0..nn { num += (p[j] - u[j]).powi(2); den += u[j].powi(2); }
            }
            let rel = (num / den).sqrt();
            // the learned half-spectrum weights against the Green's function, mode by mode
            let rr = wp[0].to_vec().await;
            let ri = wp[1].to_vec().await;
            let mut worst = 0.0f64;
            let mut worst_mode = (0usize, 0usize);
            let mut shown = Vec::new();
            for (r, &(k1, k2)) in sc.reps.iter().enumerate() {
                if k1 == 0 && k2 == 0 { continue; }
                let want = poisson_multiplier(n, k1, k2);
                let got = rr[r] as f64;
                let dev = ((got - want).abs() / want).max(ri[r].abs() as f64 / want);
                if dev > worst { worst = dev; worst_mode = (k1, k2); }
                if shown.len() < 4 { shown.push(format!("k=({k1},{k2}) {got:.5} vs {want:.5}")); }
            }
            eprintln!("  2-D spectral operator: held-out rel-L2 {rel:.4}; learned multipliers vs 1/(4π²|k|²): {} … worst relative deviation {worst:.3e} at k={worst_mode:?}", shown.join(", "));
            assert!(rel < 0.02, "the operator must be learned: {rel:.4}");
            // ⚠ the low modes match to five figures; the worst deviation sits on a high retained mode whose
            // multiplier is ~20x smaller and whose share of the training signal is smallest — measured
            // 7.3e-2 at 3000 steps, so the bar is 0.1 rather than a number the low modes would flatter
            assert!(worst < 0.1, "the weights must BE the Green's function: worst deviation {worst:.3e} at k={worst_mode:?}");
        });
    }
}
