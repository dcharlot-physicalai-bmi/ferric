//! EFA energy-first #50 — the SMT-CERTIFIED ternary Lyapunov energy, ported to the pure-Rust stack.
//!
//! `ebm_lyapunov.rs` (#49) earned a grid-verified certificate and said honestly: "NOT an SMT/dReal proof."
//! This experiment ships that missing proof. The weights embedded below are the EXACT artifact that dReal
//! 4.21 (δ-complete SMT, δ=5e-3) certified on 2026-07-23 — synthesized counterexample-guided in the
//! Neural-Lyapunov-Control lineage (Chang, Roohi & Gao, NeurIPS 2019; Gao, Kong & Clarke, CADE 2013) on the
//! reversed Van der Pol oscillator, whose region of attraction is NON-CONVEX (bounded by an unstable limit
//! cycle) — the regime where a quadratic energy provably cannot win:
//!   · best quadratic  V=xᵀP₀x : dReal-certified to R=1.1, REFUTED at R=1.2 (concrete counterexample);
//!   · this ternary net        : dReal-CERTIFIED on the annulus 0.15 ≤ ‖x‖ ≤ 1.3 (fresh re-verify of the
//!     saved snapshot: CERT, 66 s) — +18% radius / ~40% area beyond any provable quadratic.
//! V(x) = xᵀP₀x + Σⱼ w₂ⱼ·tanh(s·(Tⱼ·x) + b₁ⱼ) − v₀,  T ∈ {−1,0,+1}⁸ˣ² with 8/16 nonzeros — and every
//! row of T is ±e₁ or ±e₂, so the "matrix multiply" is literally select-negate: pre-scale (s·x₁, s·x₂)
//! ONCE (2 multiplies) and the whole hidden layer is selects, sign flips, and adds. The 1.58-bit structure
//! that deletes the MAC also deletes it inside the certificate's own energy.
//!
//! What this program does, in dependency-free f64 Rust (compiles unchanged to wasm32 — the browser story):
//!   1. CROSS-VERIFY: reproduces the Python/numpy reference values of the certified V to ≤1e-8 — the
//!      artifact evaluated on the Rust fabric is the artifact dReal proved.
//!   2. CERTIFIED DESCENT: rolls out trajectories from the certified annulus and measures the property the
//!      SMT proof guarantees — V strictly decreases at every in-annulus step — plus convergence to the ball.
//! HONEST scope: 2D benchmark; certificate is dReal's, on the annulus, at δ=5e-3 / decrease-margin 1e-3;
//! R=1.4 has genuine counterexamples; the synthesis METHOD is Chang & Gao's — our legs are the ternary
//! weights, the quadratic-anchored init, and the train-stricter-than-verify margin design. Artifact + ops
//! notes: bmi-concept/research/certificate-toolchain/ (certified_R1.3.npz re-verified fresh before export).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_ternary_cert --release`

const DT: f64 = 0.02;
const H: usize = 8;
// ---- the dReal-certified artifact (certified_R1.3.npz), embedded exactly ----
const P0: [[f64; 2]; 2] = [
    [76.2755359693619, -25.51278060966837],
    [-25.51278060966837, 51.27806096683688],
];
const S: f64 = 3.4601597785949707; // absmean scale: W1 = S * T, exactly
const T: [[i8; 2]; H] = [[-1, 0], [0, -1], [1, 0], [0, -1], [0, -1], [-1, 0], [0, 1], [1, 0]];
const B1: [f64; H] = [
    -2.3375589847564697, -2.102060556411743, -2.708591938018799, -2.0922701358795166,
    -2.0625295639038086, 1.6640418767929077, -2.6110639572143555, -2.7082889080047607,
];
const W2: [f64; H] = [
    -7.458496570587158, 3.7638297080993652, -4.320842266082764, 3.884138584136963,
    3.7923171520233154, 1.5532889366149902, 7.402223110198975, -4.4193596839904785,
];
/// v0 = tanh(b1)·w2 exactly as the dReal worker computed it (float32 accumulation) — embedded so the
/// Rust evaluation is bit-faithful to the certified expression, not a re-derivation of it.
const V0: f64 = -0.9857823848724365;

/// The certified energy. The ternary layer is multiply-free: pre-scale the two coordinates once,
/// then every hidden unit is a coordinate SELECT, an optional NEGATE, and an ADD — no matmul.
fn v(x: [f64; 2]) -> f64 {
    let quad = x[0] * (P0[0][0] * x[0] + P0[0][1] * x[1]) + x[1] * (P0[1][0] * x[0] + P0[1][1] * x[1]);
    let sx = [S * x[0], S * x[1]]; // the layer's only two multiplies
    let mut nl = 0.0;
    for j in 0..H {
        // Tⱼ·x by select/negate (each row of T is ±e₁ or ±e₂; 0-rows would contribute b₁ only)
        let pre = if T[j][0] != 0 { if T[j][0] > 0 { sx[0] } else { -sx[0] } }
                  else if T[j][1] != 0 { if T[j][1] > 0 { sx[1] } else { -sx[1] } }
                  else { 0.0 };
        nl += W2[j] * (pre + B1[j]).tanh();
    }
    quad + nl - V0
}

/// Reversed Van der Pol, forward Euler — the exact discrete system dReal certified.
fn step(x: [f64; 2]) -> [f64; 2] {
    [x[0] - DT * x[1], x[1] + DT * (x[0] + (x[0] * x[0] - 1.0) * x[1])]
}

fn main() {
    println!("EFA #50 — dReal-certified ternary Lyapunov energy on the pure-Rust fabric");
    println!("  ternary T: 8/16 nonzeros; every row ±e1/±e2 -> hidden layer = 2 muls + selects/adds\n");

    // 1 · cross-verify against the numpy reference of the certified artifact
    let pts: [[f64; 2]; 5] = [[1.2, 0.0], [0.0, 1.25], [-0.9, 0.7], [0.5, -1.1], [1.0, 0.8]];
    let refs: [f64; 5] = [90.516210288982, 94.033887217396, 112.661121676941, 127.420234038918, 59.401118604465];
    let mut worst = 0.0f64;
    for (p, r) in pts.iter().zip(refs.iter()) {
        let d = (v(*p) - r).abs();
        if d > worst { worst = d; }
    }
    let mut x = [1.25, 0.3];
    let (mut v1, mut v100, mut v400) = (0.0, 0.0, 0.0);
    for i in 0..400 {
        x = step(x);
        let vv = v(x);
        if i == 0 { v1 = vv } else if i == 99 { v100 = vv } else if i == 399 { v400 = vv }
    }
    let tref = [83.511396715931, 18.249347248844, -0.00502281083];
    let tworst = (v1 - tref[0]).abs().max((v100 - tref[1]).abs()).max((v400 - tref[2]).abs());
    println!("1 · CROSS-VERIFY vs numpy reference: worst point-err {:.2e}, worst trajectory-err {:.2e}  -> {}",
        worst, tworst, if worst < 1e-8 && tworst < 1e-8 { "MATCH (same artifact, same numbers)" } else { "MISMATCH" });
    assert!(worst < 1e-8 && tworst < 1e-8, "cross-language verification failed");

    // 2 · the certified property, measured: V strictly decreases on the annulus 0.15..1.3
    let (r0, rr) = (0.15f64, 1.3f64);
    let mut lcg: u64 = 0x2026_0723;
    let mut rnd = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (lcg >> 11) as f64 / (1u64 << 53) as f64 };
    let (mut steps_in, mut descents, mut reached, n) = (0u64, 0u64, 0u32, 200u32);
    for _ in 0..n {
        // rejection-sample a start in the certified annulus
        let mut x = loop {
            let c = [rnd() * 2.6 - 1.3, rnd() * 2.6 - 1.3];
            let r2 = c[0] * c[0] + c[1] * c[1];
            if r2 >= r0 * r0 && r2 <= rr * rr { break c; }
        };
        let mut vprev = v(x);
        let mut hit = false;
        for _ in 0..1500 {
            x = step(x);
            let r2 = x[0] * x[0] + x[1] * x[1];
            let vn = v(x);
            if r2 >= r0 * r0 && r2 <= rr * rr {
                steps_in += 1;
                if vn < vprev { descents += 1; }
            }
            if r2 < r0 * r0 { hit = true; break; }
            vprev = vn;
        }
        if hit { reached += 1; }
    }
    println!("2 · CERTIFIED DESCENT: {}/{} in-annulus steps strictly decreased V ({}%); {}/{} trajectories reached the inner ball",
        descents, steps_in, if steps_in > 0 { 100 * descents / steps_in } else { 0 }, reached, n);
    assert!(descents == steps_in && reached == n, "certified property violated at runtime");

    println!("\nPASS — the SMT-certified ternary energy runs, matches, and descends on the pure-Rust fabric.");
    println!("scope: 2D benchmark; certificate = dReal δ=5e-3 on the annulus (R=1.4 refuted); method lineage Chang/Gao.");
}
