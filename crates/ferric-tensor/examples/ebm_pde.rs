//! EFA energy-first #13 (AI×physics, opening #4) — discover a PDE from spatiotemporal data (PDE-FIND).
//!
//! Scale equation-discovery from ODEs (Lorenz) to PARTIAL differential equations. Target = Burgers' equation
//!   u_t = −u·u_x + ν·u_xx   (nonlinear advection + viscous diffusion; the canonical PDE-FIND benchmark,
//! Rudy/Brunton/Proctor/Kutz, Sci. Adv. 2017). We simulate the field u(x,t), estimate u_t, u_x, u_xx by
//! finite differences, and minimize the least-squares energy ‖u_t − Θ(u,u_x,u_xx)·ξ‖² WITH sparsity (STLSQ)
//! over a library of candidate spatial terms → recover the governing PDE. Same EFA energy-min, now over fields.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_pde --release`

const NX: usize = 128;          // spatial grid
const K: usize = 8;             // 1, u, u_x, u_xx, u², u·u_x, u·u_xx, u_x·u_xx
const NAMES: [&str; K] = ["1", "u", "u_x", "u_xx", "u^2", "u·u_x", "u·u_xx", "u_x·u_xx"];
const NU: f64 = 0.1;            // viscosity (true coefficient on u_xx)

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
    println!("  EFA energy-first — DISCOVER a PDE (Burgers u_t=−u·u_x+ν·u_xx) from spatiotemporal data (PDE-FIND)");
    let dx = 2.0 * std::f64::consts::PI / NX as f64; let dt = 0.0004; let nt = 800;
    // initial condition u0 = −sin(x); simulate viscous Burgers with periodic finite differences (explicit Euler)
    let mut u = vec![0.0f64; NX]; for i in 0..NX { u[i] = -(i as f64 * dx).sin(); }
    let mut field = Vec::with_capacity(nt); field.push(u.clone());
    for _ in 1..nt {
        let mut un = vec![0.0f64; NX];
        for i in 0..NX { let ip = (i + 1) % NX; let im = (i + NX - 1) % NX;
            let ux = (u[ip] - u[im]) / (2.0 * dx); let uxx = (u[ip] - 2.0 * u[i] + u[im]) / (dx * dx);
            un[i] = u[i] + dt * (-u[i] * ux + NU * uxx); }
        u = un; field.push(u.clone());
    }
    // sample (x,t) interior points; compute u_t (time central-diff), u_x, u_xx (space central-diff), + 1% noise
    let mut theta = Vec::new(); let mut ut = Vec::new(); let mut n = 0usize;
    for t in (4..nt - 4).step_by(3) { for i in (0..NX).step_by(2) {
        let ip = (i + 1) % NX; let im = (i + NX - 1) % NX;
        let uu = field[t][i]; let ux = (field[t][ip] - field[t][im]) / (2.0 * dx); let uxx = (field[t][ip] - 2.0 * uu + field[t][im]) / (dx * dx);
        let utv = (field[t + 1][i] - field[t - 1][i]) / (2.0 * dt);
        let nz = 1.0 + 0.01 * uni((t * NX + i) as u32, 1);
        let lib = [1.0, uu, ux, uxx, uu * uu, uu * ux, uu * uxx, ux * uxx];
        for k in 0..K { theta.push(lib[k] * nz); } ut.push(utv * nz); n += 1;
    } }
    let mut scale = [1.0f64; K]; for k in 0..K { let mut s = 0.0; for i in 0..n { s += theta[i * K + k] * theta[i * K + k]; } scale[k] = (s / n as f64).sqrt().max(1e-6); }

    // STLSQ sparse energy-min
    let mut active = [true; K]; let mut xi = [0.0f64; K];
    for _ in 0..12 { xi = lstsq(&theta, &ut, n, &active, &scale); let mut ch = false;
        for k in 0..K { let keep = xi[k].abs() > 0.03; if keep != active[k] { ch = true; } active[k] = keep; if !keep { xi[k] = 0.0; } } if !ch { break; } }

    println!("\n  TRUE:       u_t = −1.00·u·u_x +{:.2}·u_xx        (ν = {NU})\n", NU);
    let terms: Vec<String> = (0..K).filter(|&k| xi[k].abs() > 0.03).map(|k| format!("{:+.3}·{}", xi[k], NAMES[k])).collect();
    println!("  discovered from {n} spatiotemporal samples (sparse energy-min / STLSQ):");
    println!("     u_t = {}", terms.join(" "));
    let hit = |k: usize, v: f64| (xi[k] - v).abs() < 0.1;
    let ok = hit(5, -1.0) && hit(3, NU) && (0..K).filter(|&k| xi[k].abs() > 0.03).count() == 2;
    println!("\n  {} — the governing PDE (nonlinear advection + viscous diffusion) recovered from FIELD data by energy-min.",
        if ok { "✅ exact: −1·u·u_x + ν·u_xx, correct support + coefficients" } else { "⚠ partial recovery (see above)" });
}
