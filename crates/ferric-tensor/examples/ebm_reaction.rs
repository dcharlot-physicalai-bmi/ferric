//! EFA energy-first #16 (AI×physics/bio, opening #4) — discover a REACTION–DIFFUSION PDE from field data.
//!
//! Extend equation-discovery beyond mechanics/fluids into mathematical biology. Target = Fisher–KPP:
//!   u_t = u_xx + u − u²   (diffusion + logistic reaction; population spread, traveling fronts, patterns).
//! We simulate the field, estimate u_t/u_x/u_xx by finite differences (+ noise), and minimize the
//! least-squares energy ‖u_t − Θ(u, u_x, u_xx)·ξ‖² WITH sparsity (STLSQ) over a library that mixes REACTION
//! terms (u, u², u³) and TRANSPORT terms (u_x, u_xx, u·u_x) → recover the governing reaction–diffusion PDE.
//! A different PDE class (parabolic reaction–diffusion) from Burgers — breadth for the physics-discovery seam.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_reaction --release`

const NX: usize = 128;
const K: usize = 7; // 1, u, u², u³, u_x, u_xx, u·u_x
const NAMES: [&str; K] = ["1", "u", "u^2", "u^3", "u_x", "u_xx", "u·u_x"];

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn uni(i: u32, s: u32) -> f64 { ((h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f64 / 1e6) - 0.5 }

fn lstsq(theta: &[f64], y: &[f64], n: usize, active: &[bool], scale: &[f64]) -> [f64; K] {
    let idx: Vec<usize> = (0..K).filter(|&k| active[k]).collect(); let m = idx.len();
    let mut a = vec![0.0f64; m * m]; let mut b = vec![0.0f64; m];
    for i in 0..n { for r in 0..m { let vr = theta[i * K + idx[r]] / scale[idx[r]]; b[r] += vr * y[i]; for c in 0..m { a[r * m + c] += vr * theta[i * K + idx[c]] / scale[idx[c]]; } } }
    for col in 0..m { let mut piv = col; for r in col + 1..m { if a[r * m + col].abs() > a[piv * m + col].abs() { piv = r; } }
        if piv != col { for c in 0..m { a.swap(col * m + c, piv * m + c); } b.swap(col, piv); }
        let d = a[col * m + col]; if d.abs() < 1e-14 { continue; }
        for r in 0..m { if r == col { continue; } let f = a[r * m + col] / d; for c in col..m { a[r * m + c] -= f * a[col * m + c]; } b[r] -= f * b[col]; } }
    let mut xi = [0.0f64; K]; for r in 0..m { let d = a[r * m + r]; if d.abs() > 1e-14 { xi[idx[r]] = b[r] / d / scale[idx[r]]; } }
    xi
}

fn main() {
    println!("  EFA energy-first — DISCOVER a REACTION–DIFFUSION PDE (Fisher–KPP u_t=u_xx+u−u²) from field data");
    let dx = 20.0 / NX as f64; let dt = 0.002; let nt = 1200;
    // initial condition: spatially-structured field so u, u_xx, u² all vary (not a uniform state / pure front)
    let mut u = vec![0.0f64; NX];
    for i in 0..NX { let x = i as f64 * dx; u[i] = 0.45 + 0.25 * (0.7 * x).sin() + 0.12 * (1.9 * x + 1.0).sin(); }
    let mut field = Vec::with_capacity(nt); field.push(u.clone());
    for _ in 1..nt { let mut un = vec![0.0f64; NX];
        for i in 0..NX { let ip = (i + 1) % NX; let im = (i + NX - 1) % NX; let uxx = (u[ip] - 2.0 * u[i] + u[im]) / (dx * dx);
            un[i] = u[i] + dt * (uxx + u[i] * (1.0 - u[i])); }
        u = un; field.push(u.clone()); }
    // sample (x,t) interior; finite-diff u_t/u_x/u_xx + 1% noise; build library
    let mut theta = Vec::new(); let mut ut = Vec::new(); let mut n = 0usize;
    for t in (4..nt - 4).step_by(4) { for i in (0..NX).step_by(2) {
        let ip = (i + 1) % NX; let im = (i + NX - 1) % NX; let uu = field[t][i];
        let ux = (field[t][ip] - field[t][im]) / (2.0 * dx); let uxx = (field[t][ip] - 2.0 * uu + field[t][im]) / (dx * dx);
        let utv = (field[t + 1][i] - field[t - 1][i]) / (2.0 * dt); let nz = 1.0 + 0.01 * uni((t * NX + i) as u32, 1);
        let lib = [1.0, uu, uu * uu, uu * uu * uu, ux, uxx, uu * ux];
        for k in 0..K { theta.push(lib[k] * nz); } ut.push(utv * nz); n += 1;
    } }
    let mut scale = [1.0f64; K]; for k in 0..K { let mut s = 0.0; for i in 0..n { s += theta[i * K + k] * theta[i * K + k]; } scale[k] = (s / n as f64).sqrt().max(1e-6); }
    let mut active = [true; K]; let mut xi = [0.0f64; K];
    for _ in 0..12 { xi = lstsq(&theta, &ut, n, &active, &scale); let mut ch = false;
        for k in 0..K { let keep = xi[k].abs() > 0.05; if keep != active[k] { ch = true; } active[k] = keep; if !keep { xi[k] = 0.0; } } if !ch { break; } }

    println!("\n  TRUE:       u_t = +1.00·u_xx +1.00·u −1.00·u²   (diffusion + logistic reaction)\n");
    let terms: Vec<String> = (0..K).filter(|&k| xi[k].abs() > 0.05).map(|k| format!("{:+.3}·{}", xi[k], NAMES[k])).collect();
    println!("  discovered from {n} spatiotemporal samples (sparse energy-min / STLSQ):");
    println!("     u_t = {}", terms.join(" "));
    let hit = |k: usize, v: f64| (xi[k] - v).abs() < 0.1;
    let ok = hit(5, 1.0) && hit(1, 1.0) && hit(2, -1.0) && (0..K).filter(|&k| xi[k].abs() > 0.05).count() == 3;
    println!("\n  {} — the reaction–diffusion PDE (logistic growth + spatial spread) recovered from FIELD data by energy-min.",
        if ok { "✅ exact: u_xx + u − u², correct support + coefficients" } else { "⚠ partial recovery (see above)" });
}
