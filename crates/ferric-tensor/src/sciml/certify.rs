//! **A computable a-posteriori error certificate for elliptic PDEs** — the piece none of the incumbent
//! libraries ship. A trained network's residual is the one thing it can always compute about itself; for
//! a well-posed linear problem that residual bounds the error the network does not know.
//!
//! For `−Δe − k²e = r` in `Ω` with `e = g` on `∂Ω` (the error of an approximation to `−Δu − k²u = f`,
//! `u = u_b` on the boundary — Poisson at `k = 0`, Helmholtz otherwise), split `e = e₁ + e₂`: `e₁` has the
//! residual and zero boundary data, `e₂` is the boundary correction with zero residual. Then
//!
//! ```text
//! ‖e₁‖_L² ≤ ‖r‖_L² / (λ₁ − k²)                (spectral: λ₁ the first Dirichlet eigenvalue of −Δ on Ω)
//! ‖e₂‖_L² ≤ |Ω|^{1/2} · ‖g‖_L∞(∂Ω)             (maximum principle, valid while λ₁ − k² > 0)
//! ‖e‖_L²  ≤ the sum.
//! ```
//!
//! Both inputs come from the network alone — the residual by quadrature over the domain, the boundary
//! mismatch by evaluation on the boundary — so the certificate is computable with no reference solution.
//! It is **sound** (never below the true error) and, on the first eigenfunction, **tight** (equal to it):
//! `the_bound_is_sharp_on_the_first_eigenfunction` checks both. The bound's own honesty caveat is the
//! quadrature of `‖r‖`: a grid too coarse to see a residual spike under-reports it. Use a fine grid.
//!
//! Scope, stated: elliptic, linear, `k² < λ₁`. This is the domain of validity of the Mishra–Molinaro
//! error-versus-residual results (arXiv 2006.16144); hyperbolic and nonlinear problems have no bound of
//! this form here, and none is claimed.

/// The certificate. `lambda1` is the first Dirichlet eigenvalue of `−Δ` on the domain (`2π²` on the unit
/// square, `π²/2` on `[−1,1]²`, `d·π²` on the unit cube); `k2` is the Helmholtz `k²` (zero for Poisson);
/// `area` is `|Ω|`. `None` if `k² ≥ λ₁`, where the operator is not coercive and the bound does not hold.
pub fn elliptic_l2_bound(residual_l2: f64, boundary_max: f64, lambda1: f64, k2: f64, area: f64) -> Option<f64> {
    let gap = lambda1 - k2;
    if gap <= 0.0 || gap.is_nan() || !residual_l2.is_finite() || !boundary_max.is_finite() {
        return None;
    }
    Some(residual_l2 / gap + area.sqrt() * boundary_max)
}

/// `L²` norm over a regular grid by the composite trapezoid rule, `m` points per axis over a box of
/// volume `vol`; `values` is row-major as produced by [`super::util::box_grid`].
pub fn grid_l2(values: &[f32], m: usize, dim: usize, vol: f64) -> f64 {
    let total = m.pow(dim as u32);
    assert_eq!(values.len(), total);
    let mut acc = 0.0f64;
    for (idx, &v) in values.iter().enumerate() {
        let mut w = 1.0f64;
        let mut rem = idx;
        for _ in 0..dim {
            let i = rem % m;
            rem /= m;
            if i == 0 || i + 1 == m {
                w *= 0.5;
            }
        }
        acc += w * (v as f64) * (v as f64);
    }
    let cell = vol / ((m as f64 - 1.0).powi(dim as i32));
    (acc * cell).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sciml::util::box_grid;

    /// ⭐ On `e = ε sin πx sin πy` the residual is `2π²e`, so the bound `‖r‖/λ₁` with `λ₁ = 2π²` is EXACTLY
    /// `‖e‖` — the first eigenfunction is the extremal case, and a bound that were slack there would be
    /// slack everywhere. Sound and tight, checked by quadrature on the same grid the PINN oracle uses.
    #[test]
    fn the_bound_is_sharp_on_the_first_eigenfunction() {
        let pi = std::f64::consts::PI;
        let (m, eps) = (101usize, 0.3f64);
        let g = box_grid(&[0.0, 0.0], &[1.0, 1.0], m);
        let e: Vec<f32> = g.chunks(2).map(|p| (eps * (pi * p[0] as f64).sin() * (pi * p[1] as f64).sin()) as f32).collect();
        let r: Vec<f32> = e.iter().map(|&v| (2.0 * pi * pi) as f32 * v).collect();
        let (e_l2, r_l2) = (grid_l2(&e, m, 2, 1.0), grid_l2(&r, m, 2, 1.0));
        let bound = elliptic_l2_bound(r_l2, 0.0, 2.0 * pi * pi, 0.0, 1.0).expect("coercive");
        eprintln!("  first eigenfunction: ‖e‖ = {e_l2:.6}, bound = {bound:.6} (exact ‖e‖ = ε/2 = {:.6})", eps / 2.0);
        // ⚠ On the extremal case the bound EQUALS the error, so the two sides differ only by the f32
        // rounding of the test's own data (measured: 0.14999999558 against 0.15000000011, 3e-8). The
        // comparison is made at single precision; the PINN oracle in `harness` has real slack to test.
        assert!(bound >= e_l2 * (1.0 - 1e-6), "the bound must not sit below the error: {bound} < {e_l2}");
        assert!((bound - e_l2).abs() < 1e-3 * e_l2, "and on the extremal case it must be tight: {bound} vs {e_l2}");
        assert!((e_l2 - eps / 2.0).abs() < 1e-3 * eps, "quadrature check: {e_l2} vs {}", eps / 2.0);
    }

    /// A boundary mismatch alone (harmonic error) is bounded by the maximum principle: `e = ε x` on the
    /// unit square has zero residual and `‖e‖_∞(∂Ω) = ε`, so the bound is `ε`, above the true `ε/√3`.
    #[test]
    fn a_boundary_mismatch_is_bounded_by_the_maximum_principle_and_a_bad_k_is_refused() {
        let eps = 0.2f64;
        let bound = elliptic_l2_bound(0.0, eps, 2.0 * std::f64::consts::PI.powi(2), 0.0, 1.0).unwrap();
        let true_l2 = eps / 3f64.sqrt();
        assert!(bound >= true_l2 && bound < 2.0 * true_l2, "{bound} vs {true_l2}");
        assert!(elliptic_l2_bound(1.0, 0.0, 2.0, 2.0, 1.0).is_none(), "k² = λ₁ is not coercive");
        assert!(elliptic_l2_bound(1.0, 0.0, 2.0, 5.0, 1.0).is_none());
    }
}
