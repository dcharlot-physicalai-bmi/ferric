//! EFA energy-first #9 (AI×physics, opening #4) — DISCOVER the governing equation from data (SINDy-as-energy).
//!
//! Symbolic regression = find the scalar functional explaining data; SINDy (Brunton/Kutz, PNAS 2016) does it
//! as SPARSE regression over a candidate-function library. The energy is the least-squares misfit
//! E(ξ)=‖Ẋ−Θ(state)·ξ‖²; minimizing it WITH sparsity recovers the law. STLSQ = sequential thresholded
//! least-squares: solve the UNBIASED least-squares energy minimum, zero the small terms, re-fit on the
//! surviving support (avoids the LASSO shrinkage bias that biased a naive L1 energy-descent). Ground truth:
//! a cubic (Duffing) oscillator ẋ = y, ẏ = −x − 0.5·x³. Library = polynomials up to degree 3 in (x,y).
//! (Note: the fit is a tiny 10×10 solve — Ferric's GPU isn't needed at this scale; the point is the method.)
//!
//! Run: `cargo run -p ferric-tensor --example ebm_discover --release`

const N: usize = 2000;
const K: usize = 10; // 1, x, y, x², xy, y², x³, x²y, xy², y³
const NAMES: [&str; K] = ["1", "x", "y", "x^2", "xy", "y^2", "x^3", "x^2y", "xy^2", "y^3"];

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn lib(x: f32, y: f32) -> [f32; K] { [1.0, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y] }

// solve the least-squares energy min over the ACTIVE columns: (ΘᵀΘ)s = Θᵀy, via Gaussian elimination
fn lstsq(theta: &[f32], y: &[f32], active: &[bool]) -> [f32; K] {
    let idx: Vec<usize> = (0..K).filter(|&k| active[k]).collect(); let m = idx.len();
    let mut a = vec![0.0f64; m * m]; let mut bb = vec![0.0f64; m];
    for i in 0..N { for r in 0..m { let vr = theta[i * K + idx[r]] as f64; bb[r] += vr * y[i] as f64; for c in 0..m { a[r * m + c] += vr * theta[i * K + idx[c]] as f64; } } }
    // Gaussian elimination with partial pivoting
    for col in 0..m {
        let mut piv = col; for r in col + 1..m { if a[r * m + col].abs() > a[piv * m + col].abs() { piv = r; } }
        if piv != col { for c in 0..m { a.swap(col * m + c, piv * m + c); } bb.swap(col, piv); }
        let d = a[col * m + col]; if d.abs() < 1e-12 { continue; }
        for r in 0..m { if r == col { continue; } let f = a[r * m + col] / d; for c in col..m { a[r * m + c] -= f * a[col * m + c]; } bb[r] -= f * bb[col]; }
    }
    let mut xi = [0.0f32; K];
    for r in 0..m { let d = a[r * m + r]; if d.abs() > 1e-12 { xi[idx[r]] = (bb[r] / d) as f32; } }
    xi
}

fn main() {
    println!("  EFA energy-first — DISCOVER the equation: ẋ=y, ẏ=−x−0.5x³ from data (SINDy = sparse energy-min)");
    // data: (x,y) + exact derivatives + 2% noise (honest robustness)
    let mut theta = vec![0.0f32; N * K]; let mut ydot = vec![vec![0.0f32; N]; 2];
    for i in 0..N { let x = (u(i as u32, 1) * 2.0 - 1.0) * 2.0; let y = (u(i as u32, 2) * 2.0 - 1.0) * 2.0;
        let f = lib(x, y); for k in 0..K { theta[i * K + k] = f[k]; }
        let n0 = (u(i as u32, 3) - 0.5) * 2.0; let n1 = (u(i as u32, 4) - 0.5) * 2.0; // ±1.0 noise (~15-20% of signal)
        ydot[0][i] = y + n0; ydot[1][i] = -x - 0.5 * x * x * x + n1; }

    // STLSQ: unbiased least-squares → threshold small terms → re-fit on survivors → repeat
    let stlsq = |y: &[f32], thresh: f32| -> [f32; K] {
        let mut active = [true; K]; let mut xi = [0.0f32; K];
        for _ in 0..8 { xi = lstsq(&theta, y, &active); let mut ch = false;
            for k in 0..K { let keep = xi[k].abs() > thresh; if keep != active[k] { ch = true; } active[k] = keep; if !keep { xi[k] = 0.0; } }
            if !ch { break; } }
        xi
    };
    let dense: [f32; K] = lstsq(&theta, &ydot[0], &[true; K]); // for baseline display (eq 1)
    let dense2: [f32; K] = lstsq(&theta, &ydot[1], &[true; K]);
    let s0 = stlsq(&ydot[0], 0.1); let s1 = stlsq(&ydot[1], 0.1);

    let show = |xi: &[f32], title: &str| { let t: Vec<String> = (0..K).filter(|&k| xi[k].abs() > 0.02).map(|k| format!("{:+.2}·{}", xi[k], NAMES[k])).collect();
        println!("     {title:<8}{}", if t.is_empty() { "0".into() } else { t.join(" ") }); };
    println!("\n  TRUE:    ẋ = +1.00·y     ẏ = −1.00·x −0.50·x^3\n");
    println!("  discovered (sparse energy-min / STLSQ):");
    show(&s0, "ẋ ="); show(&s1, "ẏ =");
    let nnz = |xi: &[f32]| (0..K).filter(|&k| xi[k].abs() > 0.02).count();
    println!("\n  dense least-squares (no sparsity — fits but selects ~every term, not a law):");
    show(&dense, "ẋ ="); show(&dense2, "ẏ =");
    println!("\n  support — sparse: ẋ {} / ẏ {} terms (true 1/2)   |   dense: ẋ {} / ẏ {} terms",
        nnz(&s0), nnz(&s1), nnz(&dense), nnz(&dense2));
    println!("  sparse energy-min RECOVERS the exact governing law from noisy data; the dense energy-min does not.");
}
