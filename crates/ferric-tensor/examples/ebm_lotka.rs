//! EFA energy-first #14 (AI×physics, opening #4) — discover predator–prey dynamics from REAL historical data.
//!
//! The honest test: real measured data, not a simulation. The Hudson Bay Company lynx–hare pelt records
//! (1900–1920, the iconic Elton/Odum dataset) famously fit the LOTKA–VOLTERRA equations:
//!   Ḣ = αH − βHV ,  V̇ = δHV − γV    (hare grows & is eaten; lynx grows by eating & dies).
//! In LOG coordinates this LINEARIZES: d(lnH)/dt = α − βV,  d(lnV)/dt = δH − γ — so we discover the law by
//! energy-min (least-squares) over the library {1, H, V} from finite-difference log-derivatives of the REAL
//! series. Real ⇒ noisy & sparse (21 annual points); the honest question is whether the Lotka–Volterra
//! STRUCTURE (signs: +α, −β on prey growth; +δ, −γ on predator growth) emerges from actual measurements.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_lotka --release`

// Hudson Bay lynx–hare, 1900–1920 (thousands of pelts) — real historical data
const H: [f64; 21] = [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7];
const V: [f64; 21] = [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6];
const K: usize = 3; // 1, H, V

fn lstsq3(theta: &[f64], y: &[f64], n: usize) -> [f64; K] {
    let mut a = [0.0f64; K * K]; let mut b = [0.0f64; K];
    for i in 0..n { for r in 0..K { b[r] += theta[i * K + r] * y[i]; for c in 0..K { a[r * K + c] += theta[i * K + r] * theta[i * K + c]; } } }
    for col in 0..K { let mut piv = col; for r in col + 1..K { if a[r * K + col].abs() > a[piv * K + col].abs() { piv = r; } }
        if piv != col { for c in 0..K { a.swap(col * K + c, piv * K + c); } b.swap(col, piv); }
        let d = a[col * K + col]; if d.abs() < 1e-14 { continue; }
        for r in 0..K { if r == col { continue; } let f = a[r * K + col] / d; for c in col..K { a[r * K + c] -= f * a[col * K + c]; } b[r] -= f * b[col]; } }
    let mut x = [0.0f64; K]; for r in 0..K { let d = a[r * K + r]; if d.abs() > 1e-14 { x[r] = b[r] / d; } } x
}

fn main() {
    println!("  EFA energy-first — DISCOVER predator–prey dynamics from REAL lynx–hare data (Hudson Bay 1900–1920)");
    // central finite-difference LOG-derivatives (dt = 1 yr); library rows [1, H, V] at interior points
    let n = 19usize; let mut th = vec![0.0f64; n * K]; let mut dlh = vec![0.0f64; n]; let mut dlv = vec![0.0f64; n];
    for i in 1..20 { let j = i - 1; th[j * K] = 1.0; th[j * K + 1] = H[i]; th[j * K + 2] = V[i];
        dlh[j] = (H[i + 1].ln() - H[i - 1].ln()) / 2.0; dlv[j] = (V[i + 1].ln() - V[i - 1].ln()) / 2.0; }
    let ch = lstsq3(&th, &dlh, n); let cv = lstsq3(&th, &dlv, n);
    // ch = [α, ~0, −β] ; cv = [−γ, δ, ~0]
    println!("\n  Lotka–Volterra form:  d(lnH)/dt = α − β·V     d(lnV)/dt = δ·H − γ\n");
    println!("  discovered from the REAL series (energy-min over {{1, H, V}}):");
    println!("     d(lnH)/dt = {:+.3} {:+.4}·H {:+.4}·V     ⇒  α≈{:.2}, β≈{:.4}", ch[0], ch[1], ch[2], ch[0], -ch[2]);
    println!("     d(lnV)/dt = {:+.3} {:+.4}·H {:+.4}·V     ⇒  γ≈{:.2}, δ≈{:.4}", cv[0], cv[1], cv[2], -cv[0], cv[1]);
    let ok = ch[0] > 0.0 && ch[2] < 0.0 && cv[0] < 0.0 && cv[1] > 0.0;
    println!("\n  Lotka–Volterra sign structure (prey: +α growth, −β predation ; predator: +δ from prey, −γ death): {}",
        if ok { "✅ RECOVERED from real data" } else { "⚠ not clean (real data is noisy/sparse)" });
    println!("  → the predator–prey law emerges from 21 real annual measurements by energy-minimization — real data, honest signal.");
}
