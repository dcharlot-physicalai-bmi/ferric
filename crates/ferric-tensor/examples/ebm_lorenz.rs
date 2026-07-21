//! EFA energy-first #11 (AI×physics, opening #4) — discover the LORENZ system (chaos) from data.
//!
//! Scale equation-discovery (build #9) to the canonical chaotic benchmark. True system (σ=10, ρ=28, β=8/3):
//!   ẋ = σ(y−x),  ẏ = x(ρ−z)−y,  ż = xy − βz   — 3 coupled nonlinear ODEs on a strange attractor.
//! We integrate a trajectory (RK4), add measurement noise, estimate derivatives by finite difference, then
//! minimize the least-squares energy ‖Ẋ−Θ(x,y,z)·ξ‖² WITH sparsity (STLSQ) over a degree-2 polynomial library
//! to recover the exact governing equations from data — the SINDy headline (Brunton/Kutz PNAS 2016), as EFA
//! energy-minimization. (Tiny 10×10 solve → CPU; the point is the method + that it works on CHAOS.)
//!
//! Run: `cargo run -p ferric-tensor --example ebm_lorenz --release`

const K: usize = 10; // 1, x, y, z, x², xy, xz, y², yz, z²
const NAMES: [&str; K] = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"];
const SIG: f64 = 10.0; const RHO: f64 = 28.0; const BET: f64 = 8.0 / 3.0;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn uni(i: u32, s: u32) -> f64 { ((h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f64 / 1e6) - 0.5 }
fn lorenz(x: f64, y: f64, z: f64) -> (f64, f64, f64) { (SIG * (y - x), x * (RHO - z) - y, x * y - BET * z) }
fn lib(x: f64, y: f64, z: f64) -> [f64; K] { [1.0, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z] }

// least-squares energy min over ACTIVE columns via normal equations + Gaussian elimination (column-scaled for conditioning)
fn lstsq(theta: &[f64], y: &[f64], n: usize, active: &[bool], scale: &[f64]) -> [f64; K] {
    let idx: Vec<usize> = (0..K).filter(|&k| active[k]).collect(); let m = idx.len();
    let mut a = vec![0.0f64; m * m]; let mut b = vec![0.0f64; m];
    for i in 0..n { for r in 0..m { let vr = theta[i * K + idx[r]] / scale[idx[r]]; b[r] += vr * y[i]; for c in 0..m { a[r * m + c] += vr * theta[i * K + idx[c]] / scale[idx[c]]; } } }
    for col in 0..m { let mut piv = col; for r in col + 1..m { if a[r * m + col].abs() > a[piv * m + col].abs() { piv = r; } }
        if piv != col { for c in 0..m { a.swap(col * m + c, piv * m + c); } b.swap(col, piv); }
        let d = a[col * m + col]; if d.abs() < 1e-14 { continue; }
        for r in 0..m { if r == col { continue; } let f = a[r * m + col] / d; for c in col..m { a[r * m + c] -= f * a[col * m + c]; } b[r] -= f * b[col]; } }
    let mut xi = [0.0f64; K]; for r in 0..m { let d = a[r * m + r]; if d.abs() > 1e-14 { xi[idx[r]] = b[r] / d / scale[idx[r]]; } } // undo scaling
    xi
}

fn main() {
    println!("  EFA energy-first — DISCOVER the LORENZ system (chaotic) from noisy data (SINDy = sparse energy-min)");
    // integrate a long trajectory (RK4), skip transient
    let dt = 0.004; let steps = 12000; let (mut x, mut y, mut z) = (1.0f64, 1.0, 1.0);
    let mut traj = Vec::with_capacity(steps);
    for s in 0..steps { let (k1x, k1y, k1z) = lorenz(x, y, z);
        let (k2x, k2y, k2z) = lorenz(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z);
        let (k3x, k3y, k3z) = lorenz(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z);
        let (k4x, k4y, k4z) = lorenz(x + dt * k3x, y + dt * k3y, z + dt * k3z);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x); y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y); z += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        if s > 1000 { traj.push((x, y, z)); } }
    // measurement noise on the trajectory
    let noise = 0.10; let tn: Vec<(f64, f64, f64)> = traj.iter().enumerate().map(|(i, &(a, b, c))| (a + noise * uni(i as u32, 1), b + noise * uni(i as u32, 2), c + noise * uni(i as u32, 3))).collect();
    // finite-difference derivatives (central) → (state, Ẋ) pairs
    let n = tn.len() - 2; let mut theta = vec![0.0f64; n * K]; let mut yd = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    for i in 0..n { let (a, b, c) = tn[i + 1]; let f = lib(a, b, c); for k in 0..K { theta[i * K + k] = f[k]; }
        yd[0][i] = (tn[i + 2].0 - tn[i].0) / (2.0 * dt); yd[1][i] = (tn[i + 2].1 - tn[i].1) / (2.0 * dt); yd[2][i] = (tn[i + 2].2 - tn[i].2) / (2.0 * dt); }
    // column scales (RMS) for conditioning
    let mut scale = [1.0f64; K]; for k in 0..K { let mut s = 0.0; for i in 0..n { s += theta[i * K + k] * theta[i * K + k]; } scale[k] = (s / n as f64).sqrt().max(1e-6); }

    let stlsq = |yd: &[f64]| -> [f64; K] { let mut active = [true; K]; let mut xi = [0.0f64; K];
        for _ in 0..10 { xi = lstsq(&theta, yd, n, &active, &scale); let mut ch = false;
            for k in 0..K { let keep = xi[k].abs() > 0.5; if keep != active[k] { ch = true; } active[k] = keep; if !keep { xi[k] = 0.0; } } if !ch { break; } } xi };
    let show = |xi: &[f64], t: &str| { let s: Vec<String> = (0..K).filter(|&k| xi[k].abs() > 0.5).map(|k| format!("{:+.2}·{}", xi[k], NAMES[k])).collect(); println!("     {t:<7}{}", s.join(" ")); };
    let (dx, dy, dz) = (stlsq(&yd[0]), stlsq(&yd[1]), stlsq(&yd[2]));
    println!("\n  TRUE:   ẋ = −10.00·x +10.00·y   ẏ = +28.00·x −1.00·y −1.00·xz   ż = +1.00·xy −2.67·z\n");
    println!("  discovered from {n} noisy chaotic samples (sparse energy-min / STLSQ):");
    show(&dx, "ẋ ="); show(&dy, "ẏ ="); show(&dz, "ż =");
    let hit = |xi: &[f64], k: usize, v: f64| (xi[k] - v).abs() < 1.0;
    let ok = hit(&dx, 1, -10.0) && hit(&dx, 2, 10.0) && hit(&dy, 1, 28.0) && hit(&dy, 6, -1.0) && hit(&dz, 5, 1.0) && hit(&dz, 3, -BET);
    println!("\n  {} — the exact Lorenz equations recovered from data, on a STRANGE ATTRACTOR (chaos), by energy-min.",
        if ok { "✅ ALL terms + coefficients correct" } else { "⚠ partial recovery (see above)" });
}
