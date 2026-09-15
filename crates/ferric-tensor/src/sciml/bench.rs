//! **Reference solutions for the field's standard PDE benchmarks** — the ground truth the recipe is
//! measured against, so this stack's numbers are comparable to PINNacle / PDEBench / jinns rather than
//! to fixtures of its own choosing.
//!
//! Every reference here is INDEPENDENT of any network and is itself tested: a closed form is checked
//! against the PDE by finite differences, and the one that is an integral (Burgers, via Cole–Hopf) is
//! checked the same way plus against its initial condition. A reference that has not been checked is a
//! second unknown, and comparing an unknown to an unknown is how a benchmark reports whatever you hoped.
//!
//! | problem | equation | reference |
//! |---|---|---|
//! | [`burgers`] | `u_t + u u_x = ν u_xx`, `ν = 0.01/π`, `u(x,0) = −sin πx`, `u(±1,t) = 0` | Cole–Hopf integral, Hermite-Gauss form (Basdevant et al. 1986) |
//! | [`helmholtz`] | `Δu + k²u = q` on `[−1,1]²`, `u = 0` on the boundary | manufactured `sin(a₁πx) sin(a₂πy)` (Wang, Teng & Perdikaris 2021) |
//! | [`heat`] | `u_t = u_xx` on `[0,1]`, `u(0,t) = u(1,t) = 0` | `e^{−π²t} sin πx` |
//! | [`advection`] | `u_t + β u_x = 0`, periodic on `[0,2π]`, `u(x,0) = sin x` | `sin(x − βt)` (Krishnapriyan et al. 2021's hard case at β = 30) |

/// Burgers' equation `u_t + u u_x = ν u_xx` on `x ∈ [−1, 1]`, `u(x,0) = −sin πx`, `u(±1, t) = 0`, by the
/// Cole–Hopf transformation: `u = −2ν φ_x / φ`, `φ = ∫ exp(−(x−η)²/(4νt) − (1/2ν)∫₀^η u₀)dη`. With this
/// initial condition `∫₀^η u₀ = (cos πη − 1)/π`, and the substitution `η = x − 2√(νt) z` puts the Gaussian
/// weight at `e^{−z²}`, which the trapezoid rule on `z ∈ [−8, 8]` integrates to machine precision. The
/// exponent is shifted by its maximum before exponentiating so `e^{1/ν}`-sized terms do not overflow.
///
/// At `t = 0` the integrand is a delta at `η = x` and the initial condition is returned exactly.
pub fn burgers(x: f64, t: f64, nu: f64) -> f64 {
    if t <= 0.0 {
        return -(std::f64::consts::PI * x).sin();
    }
    let n = 4001usize;
    let (zlo, zhi) = (-8.0f64, 8.0f64);
    let dz = (zhi - zlo) / (n as f64 - 1.0);
    let s = 2.0 * (nu * t).sqrt();
    // exponent(z) = −z² + (1 − cos(π η)) / (2νπ),  η = x − s z
    let expo = |z: f64| {
        let eta = x - s * z;
        -z * z + (1.0 - (std::f64::consts::PI * eta).cos()) / (2.0 * nu * std::f64::consts::PI)
    };
    let mut emax = f64::NEG_INFINITY;
    for i in 0..n {
        emax = emax.max(expo(zlo + i as f64 * dz));
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..n {
        let z = zlo + i as f64 * dz;
        let w = if i == 0 || i + 1 == n { 0.5 } else { 1.0 };
        let e = (expo(z) - emax).exp() * w;
        // (x − η)/t = s z / t
        num += (s * z / t) * e;
        den += e;
    }
    num / den
}

/// Helmholtz `Δu + k²u = q` on `[−1,1]²` with `u = 0` on the boundary and the manufactured solution
/// `u = sin(a₁πx) sin(a₂πy)`, for which `q = (k² − (a₁² + a₂²)π²) u`. Returns `(u, q)`.
pub fn helmholtz(x: f64, y: f64, a1: f64, a2: f64, k: f64) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let u = (a1 * pi * x).sin() * (a2 * pi * y).sin();
    (u, (k * k - (a1 * a1 + a2 * a2) * pi * pi) * u)
}

/// Heat equation `u_t = u_xx` on `[0,1]` with `u(0,t) = u(1,t) = 0` and `u(x,0) = sin πx`.
pub fn heat(x: f64, t: f64) -> f64 {
    let pi = std::f64::consts::PI;
    (-pi * pi * t).exp() * (pi * x).sin()
}

/// Advection `u_t + β u_x = 0`, periodic on `[0, 2π]`, `u(x,0) = sin x`.
pub fn advection(x: f64, t: f64, beta: f64) -> f64 {
    (x - beta * t).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central differences of a closed form must satisfy its PDE: the residual is what the PINNs are
    /// later scored on, so the reference has to be a solution of it to well below the score threshold.
    fn fd_residual_1d(u: &dyn Fn(f64, f64) -> f64, x: f64, t: f64, h: f64, res: &dyn Fn(f64, f64, f64, f64, f64) -> f64) -> f64 {
        let (u0, ut, ux, uxx) = (
            u(x, t),
            (u(x, t + h) - u(x, t - h)) / (2.0 * h),
            (u(x + h, t) - u(x - h, t)) / (2.0 * h),
            (u(x + h, t) - 2.0 * u(x, t) + u(x - h, t)) / (h * h),
        );
        res(u0, ut, ux, uxx, x)
    }

    /// ⭐ **Burgers by Cole–Hopf is a solution of Burgers.** Checked by finite differences at points on
    /// both sides of the forming shock and at three times, plus the initial condition and the boundary
    /// values. `ν = 0.01/π` is the PINNacle / Raissi setting; the shock steepens to a width ~ν by `t = 1`,
    /// so the differencing step must be far below that and the check stays away from `x = 0` at late times.
    #[test]
    fn cole_hopf_satisfies_burgers_its_initial_condition_and_its_boundaries() {
        let nu = 0.01 / std::f64::consts::PI;
        let u = |x: f64, t: f64| burgers(x, t, nu);
        // initial condition
        for &x in &[-0.9, -0.3, 0.2, 0.7] {
            assert!((u(x, 0.0) + (std::f64::consts::PI * x).sin()).abs() < 1e-12);
            assert!((u(x, 1e-4) + (std::f64::consts::PI * x).sin()).abs() < 2e-3, "the solution must start from −sin πx: {}", u(x, 1e-4));
        }
        // boundaries stay at zero (the initial datum is odd and vanishes at ±1)
        for &t in &[0.1, 0.5, 1.0] {
            assert!(u(-1.0, t).abs() < 1e-9 && u(1.0, t).abs() < 1e-9, "u(±1,{t}) = {} {}", u(-1.0, t), u(1.0, t));
        }
        // the PDE, away from the shock
        let h = 1e-4;
        let mut worst = 0.0f64;
        for &t in &[0.1, 0.4, 0.8] {
            for &x in &[-0.7, -0.35, 0.3, 0.65] {
                let r = fd_residual_1d(&u, x, t, h, &|u0, ut, ux, uxx, _| ut + u0 * ux - nu * uxx);
                worst = worst.max(r.abs());
            }
        }
        eprintln!("  Cole–Hopf Burgers: worst finite-difference PDE residual {worst:.2e} (shock height at t=1: {:.4})", u(-0.02, 1.0) - u(0.02, 1.0));
        assert!(worst < 5e-3, "the reference must satisfy Burgers: residual {worst:.3e}");
        // the shock is there: a steep drop across x = 0 at t = 1
        assert!(u(-0.05, 1.0) > 0.5 && u(0.05, 1.0) < -0.5, "the shock must have formed by t = 1: {} {}", u(-0.05, 1.0), u(0.05, 1.0));
    }

    #[test]
    fn the_closed_forms_satisfy_their_equations() {
        let pi = std::f64::consts::PI;
        // heat: u_t − u_xx = 0
        let r = fd_residual_1d(&heat, 0.37, 0.2, 1e-4, &|_, ut, _, uxx, _| ut - uxx);
        assert!(r.abs() < 1e-5, "heat residual {r:.2e}");
        assert!((heat(0.0, 0.3)).abs() < 1e-15 && (heat(1.0, 0.3)).abs() < 1e-12);
        // advection: u_t + β u_x = 0, periodic
        let beta = 30.0;
        let adv = |x: f64, t: f64| advection(x, t, beta);
        let r = fd_residual_1d(&adv, 1.1, 0.4, 1e-5, &|_, ut, ux, _, _| ut + beta * ux);
        assert!(r.abs() < 1e-4, "advection residual {r:.2e}");
        assert!((adv(0.0, 0.5) - adv(2.0 * pi, 0.5)).abs() < 1e-12, "periodic");
        // helmholtz: Δu + k²u − q = 0 by 2-D differences
        let (a1, a2, k) = (1.0, 4.0, 1.0);
        let (x, y, h) = (0.3, -0.45, 1e-4);
        let uf = |x: f64, y: f64| helmholtz(x, y, a1, a2, k).0;
        let lap = (uf(x + h, y) - 2.0 * uf(x, y) + uf(x - h, y)) / (h * h) + (uf(x, y + h) - 2.0 * uf(x, y) + uf(x, y - h)) / (h * h);
        let (u, q) = helmholtz(x, y, a1, a2, k);
        assert!((lap + k * k * u - q).abs() < 1e-4, "helmholtz residual {:.2e}", lap + k * k * u - q);
        assert!(uf(1.0, 0.2).abs() < 1e-12 && uf(0.2, -1.0).abs() < 1e-12, "zero on the boundary");
    }
}
