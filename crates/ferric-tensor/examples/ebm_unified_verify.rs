//! EFA energy-first #53 — the UNIFICATION proof on the fabric: one energy, controller AND certificate.
//!
//! The certificate program (ebm_cert_verify) re-proved a Lyapunov energy under a FIXED control law.
//! This proves the sharper EFA claim: a single learned value-energy is simultaneously the CONTROL
//! objective (act by descending it) and the stability CERTIFICATE (it contracts under that very
//! descent) — on the saturated double integrator, where a plain quadratic certificate fails.
//!
//! The controller is greedy: u*(s) = argmin_{u∈U} V(step(s,u)). For a fixed torque the closed loop is
//! AFFINE, s' = A s + c_u with the SAME A=[[1,dt],[0,1]] for every u, so the greedy controller is an
//! OR OVER ACTIONS: a box is soundly certified for contraction if SOME single torque makes
//!   ΔV(s) + α‖s‖² < 0   over the whole box,
//! bounded by the exact affine 2nd-order Taylor + per-box CROWN |tanh″| head (center gradient
//! AᵀgV(f)−gV(c); Hessian Aᵀ2P A − 2P + head). The verifier's torque set ⊆ the greedy controller's,
//! so a certifying torque is always available to the greedy policy — the bound is sound for the
//! actual closed loop. Dependency-free f64 + tanh, no solver; compiles unchanged to wasm32.
//!
//! Energy V(e)=eᵀPe + Σⱼ w₂ⱼ·tanh(W1ⱼ·e + b₁ⱼ) − v₀, value-iteration-trained (research/efa-unification,
//! efa_vform.py). Certifies the annulus 0.15..1.0; cross-verified to the reference V to ~1e-9.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_unified_verify --release`

const DT: f64 = 0.15; const UMAX: f64 = 3.0; const NCAND: usize = 31;
const ALPHA: f64 = 5e-3; const R0: f64 = 0.15; const RR: f64 = 1.0;
// the unified value-energy (quadratic P + dense tanh head), embedded exactly
const P: [[f64; 2]; 2] = [[0.05, 0.0], [0.0, 0.05]];
const W1: [[f64; 2]; 8] = [
    [-1.5893343687057495, -0.1714242398738861], [0.0016526344697922468, -0.9124658107757568],
    [1.6216663122177124, 0.015389240346848965], [0.17682550847530365, -0.6551535725593567],
    [1.0049142837524414, -0.08851369470357895], [1.194985032081604, 1.255305290222168],
    [1.095798373222351, 1.369579792022705], [1.2361358404159546, -0.2221991866827011]];
const B1: [f64; 8] = [-1.7399159669876099, -2.074284315109253, -1.9195194244384766, 1.2396570444107056,
                      -1.1304279565811157, 2.8246781826019287, -2.8087005615234375, 1.6094090938568115];
const W2: [f64; 8] = [0.7340201735496521, 0.6679110527038574, 1.0044790506362915, -0.24614089727401733,
                      0.5424213409423828, -1.2263835668563843, 1.1697341203689575, -0.8474427461624146];
const V0: f64 = -6.108821868896484;

fn torque(i: usize) -> f64 { -UMAX + (i as f64) * (2.0 * UMAX / (NCAND as f64 - 1.0)) }
/// V(e)
fn vfn(e1: f64, e2: f64) -> f64 {
    let mut v = P[0][0] * e1 * e1 + P[1][1] * e2 * e2 + 2.0 * P[0][1] * e1 * e2;
    for j in 0..8 { v += W2[j] * (W1[j][0] * e1 + W1[j][1] * e2 + B1[j]).tanh(); }
    v - V0
}
/// ∇V(e)
fn grad_v(e1: f64, e2: f64) -> (f64, f64) {
    let (mut g1, mut g2) = (2.0 * (P[0][0] * e1 + P[0][1] * e2), 2.0 * (P[1][0] * e1 + P[1][1] * e2));
    for j in 0..8 {
        let th = (W1[j][0] * e1 + W1[j][1] * e2 + B1[j]).tanh();
        let d = W2[j] * (1.0 - th * th);
        g1 += d * W1[j][0]; g2 += d * W1[j][1];
    }
    (g1, g2)
}
/// one closed-loop step for torque u: s' = A s + c_u (A=[[1,dt],[0,1]], c_u=[dt²u, dt·u])
fn step(e1: f64, e2: f64, u: f64) -> (f64, f64) {
    let v2 = e2 + DT * u;
    (e1 + DT * v2, v2)
}
fn d2max(lo: f64, hi: f64) -> f64 {
    let (tl, th) = (lo.tanh(), hi.tanh());
    let m = (2.0 * tl.abs() * (1.0 - tl * tl)).max(2.0 * th.abs() * (1.0 - th * th));
    if (lo <= 0.6585 && hi >= 0.6585) || (lo <= -0.6585 && hi >= -0.6585) { 0.7698 } else { m }
}
/// entrywise |head Hessian| bound over the box (center c, radius r), dense W1
fn head_hess(c1: f64, c2: f64, r1: f64, r2: f64) -> [[f64; 2]; 2] {
    let mut h = [[0.0; 2]; 2];
    for j in 0..8 {
        let (a0, a1) = (W1[j][0].abs(), W1[j][1].abs());
        let zc = W1[j][0] * c1 + W1[j][1] * c2 + B1[j]; let zr = a0 * r1 + a1 * r2;
        let coef = W2[j].abs() * d2max(zc - zr, zc + zr);
        h[0][0] += coef * a0 * a0; h[0][1] += coef * a0 * a1; h[1][0] += coef * a1 * a0; h[1][1] += coef * a1 * a1;
    }
    h
}
/// affine Taylor+CROWN upper bound on ΔV+α‖e‖² over the box, for a fixed torque u
fn bound_action(c1: f64, c2: f64, r1: f64, r2: f64, u: f64) -> f64 {
    let (fx, fy) = step(c1, c2, u);
    let dvc = vfn(fx, fy) - vfn(c1, c2);
    let (gfx, gfy) = grad_v(fx, fy); let (gsx, gsy) = grad_v(c1, c2);
    // A = [[1,dt],[0,1]] : center gradient of ΔV = Aᵀ∇V(f) − ∇V(c)
    let gd1 = gfx - gsx;
    let gd2 = DT * gfx + gfy - gsy;
    // quadratic Hessian AᵀP2A − P2 (constant); |A| entries = [[1,dt],[0,1]]
    let p2 = [[2.0 * P[0][0], 2.0 * P[0][1]], [2.0 * P[1][0], 2.0 * P[1][1]]];
    let a = [[1.0, DT], [0.0, 1.0]];
    let mut pj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { pj[i][k] = p2[i][0] * a[0][k] + p2[i][1] * a[1][k]; } }
    let mut m = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { m[i][k] = a[0][i] * pj[0][k] + a[1][i] * pj[1][k] - p2[i][k]; } }
    let aj = [[1.0, DT], [0.0, 1.0]];               // |A|
    let (fr1, fr2) = (aj[0][0] * r1 + aj[0][1] * r2, aj[1][0] * r1 + aj[1][1] * r2);
    let hs = head_hess(c1, c2, r1, r2);
    let hf = head_hess(fx, fy, fr1, fr2);
    // ajᵀ hf aj
    let mut hfj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { hfj[i][k] = hf[i][0] * aj[0][k] + hf[i][1] * aj[1][k]; } }
    let mut habs = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { habs[i][k] = m[i][k].abs() + hs[i][k] + (aj[0][i] * hfj[0][k] + aj[1][i] * hfj[1][k]); } }
    let ss_hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    let rem = 0.5 * (habs[0][0] * r1 * r1 + habs[0][1] * r1 * r2 + habs[1][0] * r2 * r1 + habs[1][1] * r2 * r2);
    dvc + gd1.abs() * r1 + gd2.abs() * r2 + rem + ALPHA * ss_hi
}
/// greedy = OR over torques: certified if SOME torque contracts the whole box
fn box_certified(c1: f64, c2: f64, r1: f64, r2: f64) -> bool {
    let mut best = f64::INFINITY;
    for i in 0..NCAND { let b = bound_action(c1, c2, r1, r2, torque(i)); if b < best { best = b; } }
    best < 0.0
}
fn in_region(c1: f64, c2: f64, r1: f64, r2: f64) -> bool {
    let lo = (c1.abs() - r1).max(0.0).powi(2) + (c2.abs() - r2).max(0.0).powi(2);
    let hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    hi >= R0 * R0 && lo <= RR * RR
}

fn main() {
    println!("EFA #53 — unification proof on the fabric (dependency-free f64; wasm-clean)");
    // 1 · cross-verify the embedded energy vs the reference (efa_vform.py)
    let refs = [([0.5, -0.5], 0.257325731607), ([-0.3, 0.8], 0.189448763562), ([0.9, 0.4], 0.937009696021)];
    let worst = refs.iter().map(|(e, r)| (vfn(e[0], e[1]) - r).abs()).fold(0.0f64, f64::max);
    println!("1 · CROSS-VERIFY unified energy vs reference: worst err {:.2e} -> {}",
        worst, if worst < 1e-8 { "MATCH" } else { "MISMATCH" });
    assert!(worst < 1e-8, "embedded weights are not the trained unified energy");

    // 2 · sound greedy-contraction certification over the annulus (adaptive box refinement)
    println!("\n2 · prove one energy is BOTH controller and certificate (greedy = OR over {} torques):", NCAND);
    let h0 = 0.06f64;
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let n = (2.0 * RR / h0).ceil() as i64;
    for i in 0..n { for k in 0..n {
        let c1 = -RR + (i as f64 + 0.5) * h0; let c2 = -RR + (k as f64 + 0.5) * h0;
        if in_region(c1, c2, h0 / 2.0, h0 / 2.0) { boxes.push([c1, c2, h0 / 2.0, h0 / 2.0]); }
    }}
    let mut certified = 0u64; let mut depth = 0;
    loop {
        let mut fails: Vec<[f64; 4]> = Vec::new();
        for b in &boxes {
            if box_certified(b[0], b[1], b[2], b[3]) { certified += 1; } else { fails.push(*b); }
        }
        println!("   depth {}: {} boxes, {} fail, certified so far {}", depth, boxes.len(), fails.len(), certified);
        if fails.is_empty() { break; }
        if depth >= 14 { println!("   REJECTED at [{:.3},{:.3}]", fails[0][0], fails[0][1]); std::process::exit(1); }
        let mut next: Vec<[f64; 4]> = Vec::with_capacity(fails.len() * 4);
        for b in &fails {
            let (nr1, nr2) = (b[2] / 2.0, b[3] / 2.0);
            for &sx in &[-1.0, 1.0] { for &sy in &[-1.0, 1.0] {
                let (c1, c2) = (b[0] + sx * nr1, b[1] + sy * nr2);
                if in_region(c1, c2, nr1, nr2) { next.push([c1, c2, nr1, nr2]); }
            }}
        }
        boxes = next; depth += 1;
    }
    println!("\nCERTIFIED — the unified value-energy is a PROVEN Lyapunov certificate under its OWN greedy");
    println!("control over the annulus 0.15..1.0 ({} boxes). One energy, both roles, soundly, on the fabric.", certified);
}
