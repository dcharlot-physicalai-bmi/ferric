//! EFA energy-first #54 — the unification proof on a NONLINEAR, WEAK-TORQUE plant, on the fabric.
//!
//! ebm_unified_verify proved one energy = controller + certificate on the (affine) double integrator.
//! This proves the sharper, harder case: the torque-underactuated pendulum in the WEAK-TORQUE regime
//! (u_max=5 < gravity-at-horizontal=10, so it cannot lift the pole directly). There, an energy trained
//! by value iteration is NOT 1-step-certifiable (control needs lookahead) — but an energy synthesized
//! DIRECTLY as a control-Lyapunov function IS: its greedy is a 100% controller AND a sound certificate.
//! (research/efa-unification, efa_pendulum_weaktorque_clf.py; A1, robust across seeds.) This ports that
//! sound proof to dependency-free f64 Rust, so it re-proves on the fabric; compiles unchanged to wasm32.
//!
//! Nonlinear closed loop for a fixed torque u:  s' = f(s,u),
//!   f0 = s0 + A0·s1,   f1 = (1−DMP)·s1 + BG·sin(s0) + BU·u.
//! The Jacobian J(s)=[[1,A0],[BG·cos s0, 1−DMP]] is STATE-dependent but action-INDEPENDENT, so the greedy
//! controller is still an OR OVER TORQUES: a box is certified if SOME torque makes ΔV+α‖s‖²<0 over it.
//! Per torque: 2nd-order Taylor at the box center + interval |Hess G| bound that accounts for (a) the
//! state-dependent |J| (cos over the box), and (b) the extra dynamics-Hessian term ∂²f1/∂s0² = −BG·sin s0,
//! which the affine plant did not have. Adaptive box refinement. Verifier torque set ⊆ controller's -> sound.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_wtclf_verify --release`

const A0: f64 = 0.08; const BG: f64 = 0.05; const DMP: f64 = 0.002; const BU: f64 = 0.005;
const UMAX: f64 = 5.0; const NCAND: usize = 31;
const ALPHA: f64 = 5e-3; const R0: f64 = 0.15; const RR: f64 = 0.35;   // weak-torque basin annulus
const HH: usize = 12;
// the DIRECT control-Lyapunov energy (quadratic P diag + dense tanh head), embedded exactly
const P00: f64 = 0.05000000074505806; const P11: f64 = 0.05000000074505806;
const V0: f64 = -1.224698543548584;
const W1: [[f64; 2]; HH] = [
    [-3.1025891304016113, -2.0950419902801514], [-0.3347717821598053, -0.8919756412506104],
    [0.6075884103775024, 0.07458163052797318], [-0.013516340404748917, -3.0233681201934814],
    [0.39023053646087646, -0.25354301929473877], [1.0057071447372437, -0.03833085671067238],
    [0.058683693408966064, 2.1846842765808105], [-0.04432548210024834, 0.00786590576171875],
    [2.0521111488342285, 2.813333749771118], [1.952376365661621, 0.26965293288230896],
    [0.3949240744113922, -0.23951223492622375], [-0.30748090147972107, -3.1440351009368896]];
const B1: [f64; HH] = [0.736440122127533, 0.751457154750824, 0.5404043197631836, 0.7823678255081177,
    -0.25804927945137024, 0.7141858339309692, -0.7642426490783691, -0.4624954164028168,
    0.5647906064987183, -0.5173326730728149, 0.2026713788509369, -0.8166104555130005];
const W2: [f64; HH] = [-0.17059792578220367, -0.13181152939796448, 0.015476273372769356, -0.3895718455314636,
    0.049851737916469574, 0.024852115660905838, 0.4077019691467285, -0.004935368429869413,
    -0.29746684432029724, 0.0815349742770195, 0.020873045548796654, 0.5085486173629761];

fn torque(i: usize) -> f64 { -UMAX + (i as f64) * (2.0 * UMAX / (NCAND as f64 - 1.0)) }
fn vfn(s1: f64, s2: f64) -> f64 {
    let mut v = P00 * s1 * s1 + P11 * s2 * s2;
    for j in 0..HH { v += W2[j] * (W1[j][0] * s1 + W1[j][1] * s2 + B1[j]).tanh(); }
    v - V0
}
fn grad_v(s1: f64, s2: f64) -> (f64, f64) {
    let (mut g1, mut g2) = (2.0 * P00 * s1, 2.0 * P11 * s2);
    for j in 0..HH {
        let th = (W1[j][0] * s1 + W1[j][1] * s2 + B1[j]).tanh();
        let d = W2[j] * (1.0 - th * th);
        g1 += d * W1[j][0]; g2 += d * W1[j][1];
    }
    (g1, g2)
}
fn step(s1: f64, s2: f64, u: f64) -> (f64, f64) {
    (s1 + A0 * s2, (1.0 - DMP) * s2 + BG * s1.sin() + BU * u)
}
fn d2max(lo: f64, hi: f64) -> f64 {
    let (tl, th) = (lo.tanh(), hi.tanh());
    let m = (2.0 * tl.abs() * (1.0 - tl * tl)).max(2.0 * th.abs() * (1.0 - th * th));
    if (lo <= 0.6585 && hi >= 0.6585) || (lo <= -0.6585 && hi >= -0.6585) { 0.7698 } else { m }
}
fn head_hess(c1: f64, c2: f64, r1: f64, r2: f64) -> [[f64; 2]; 2] {
    let mut h = [[0.0; 2]; 2];
    for j in 0..HH {
        let (a0, a1) = (W1[j][0].abs(), W1[j][1].abs());
        let zc = W1[j][0] * c1 + W1[j][1] * c2 + B1[j]; let zr = a0 * r1 + a1 * r2;
        let coef = W2[j].abs() * d2max(zc - zr, zc + zr);
        h[0][0] += coef * a0 * a0; h[0][1] += coef * a0 * a1; h[1][0] += coef * a1 * a0; h[1][1] += coef * a1 * a1;
    }
    h
}
/// nonlinear 2nd-order Taylor+CROWN upper bound on ΔV+α‖s‖² over the box, for a fixed torque u
fn bound_action(c1: f64, c2: f64, r1: f64, r2: f64, u: f64) -> f64 {
    let (fx, fy) = step(c1, c2, u);
    let gc = vfn(fx, fy) - vfn(c1, c2) + ALPHA * (c1 * c1 + c2 * c2);
    let (gfx, gfy) = grad_v(fx, fy); let (gsx, gsy) = grad_v(c1, c2);
    // center gradient of G = J(c)ᵀ∇V(f) − ∇V(c) + 2α c ;  J=[[1,A0],[BG·cos c1, 1−DMP]]
    let gd1 = gfx + BG * c1.cos() * gfy - gsx + 2.0 * ALPHA * c1;
    let gd2 = A0 * gfx + (1.0 - DMP) * gfy - gsy + 2.0 * ALPHA * c2;
    // enclosure of f(box): f0 affine; f1 via Lipschitz bound on sin (|sin'|<=1)
    let rf1 = r1 + A0.abs() * r2;
    let rf2 = (1.0 - DMP) * r2 + BG * r1;
    let (s0lo, s0hi) = (c1 - r1, c1 + r1);
    let cmax = if s0lo <= 0.0 && s0hi >= 0.0 { 1.0 } else { s0lo.cos().abs().max(s0hi.cos().abs()) };
    let smax = s0lo.sin().abs().max(s0hi.sin().abs());
    let ajj = [[1.0, A0.abs()], [BG * cmax, (1.0 - DMP).abs()]];   // entrywise |J| over box
    let p2 = [[2.0 * P00, 0.0], [0.0, 2.0 * P11]];
    let hs = head_hess(c1, c2, r1, r2);
    let hf = head_hess(fx, fy, rf1, rf2);
    let hvb = [[p2[0][0] + hs[0][0], hs[0][1]], [hs[1][0], p2[1][1] + hs[1][1]]];   // |Hess V| over box
    let hvf = [[p2[0][0] + hf[0][0], hf[0][1]], [hf[1][0], p2[1][1] + hf[1][1]]];   // |Hess V| over enclosure
    // ajjᵀ · hvf · ajj
    let mut hj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { hj[i][k] = hvf[i][0] * ajj[0][k] + hvf[i][1] * ajj[1][k]; } }
    let mut term1 = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { term1[i][k] = ajj[0][i] * hj[0][k] + ajj[1][i] * hj[1][k]; } }
    // |dV/dy1| over enclosure F : quadratic part + (tanh'∈[0,1]) sum |w2_j W1[j][1]|
    let mut gy1 = 2.0 * P11 * (fy.abs() + rf2);
    for j in 0..HH { gy1 += (W2[j] * W1[j][1]).abs(); }
    let mut habs = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { habs[i][k] = term1[i][k] + hvb[i][k]; } }
    habs[0][0] += gy1 * BG * smax;                                 // dynamics-Hessian term ∂²f1/∂s0² = −BG·sin s0
    habs[0][0] += 2.0 * ALPHA; habs[1][1] += 2.0 * ALPHA;          // Hess of α‖s‖²
    let rem = 0.5 * (habs[0][0] * r1 * r1 + habs[0][1] * r1 * r2 + habs[1][0] * r2 * r1 + habs[1][1] * r2 * r2);
    gc + gd1.abs() * r1 + gd2.abs() * r2 + rem
}
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
    println!("EFA #54 — nonlinear weak-torque unification proof on the fabric (dependency-free f64; wasm-clean)");
    // 1 · cross-verify the embedded CLF energy vs the reference (efa_pendulum_certify.py)
    let refs = [([0.2, -0.1], 0.05013136309498911), ([-0.15, 0.2], 0.14468887802739028), ([0.25, 0.15], 0.24859221753292549)];
    let worst = refs.iter().map(|(s, r)| (vfn(s[0], s[1]) - r).abs()).fold(0.0f64, f64::max);
    println!("1 · CROSS-VERIFY CLF energy vs reference: worst err {:.2e} -> {}",
        worst, if worst < 1e-9 { "MATCH" } else { "MISMATCH" });
    assert!(worst < 1e-9, "embedded weights are not the trained weak-torque CLF energy");

    // 2 · sound greedy-contraction certification over the weak-torque basin annulus (adaptive refinement)
    println!("\n2 · prove the CLF energy is BOTH controller and certificate (greedy = OR over {} torques):", NCAND);
    let h0 = 0.04f64;
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let n = (2.0 * RR / h0).ceil() as i64;
    for i in 0..n { for k in 0..n {
        let c1 = -RR + (i as f64 + 0.5) * h0; let c2 = -RR + (k as f64 + 0.5) * h0;
        if in_region(c1, c2, h0 / 2.0, h0 / 2.0) { boxes.push([c1, c2, h0 / 2.0, h0 / 2.0]); }
    }}
    let mut depth = 0;
    let final_boxes: u64 = loop {
        let mut fails: Vec<[f64; 4]> = Vec::new(); let mut cert = 0u64;
        for b in &boxes {
            if box_certified(b[0], b[1], b[2], b[3]) { cert += 1; } else { fails.push(*b); }
        }
        println!("   depth {}: {} boxes, {} fail, {} certified", depth, boxes.len(), fails.len(), cert);
        if fails.is_empty() { break boxes.len() as u64; }
        if depth >= 8 { println!("   REJECTED at [{:.3},{:.3}]", fails[0][0], fails[0][1]); std::process::exit(1); }
        let mut next: Vec<[f64; 4]> = Vec::with_capacity(fails.len() * 4);
        for b in &fails {
            let (nr1, nr2) = (b[2] / 2.0, b[3] / 2.0);
            for &sx in &[-1.0, 1.0] { for &sy in &[-1.0, 1.0] {
                let (c1, c2) = (b[0] + sx * nr1, b[1] + sy * nr2);
                if in_region(c1, c2, nr1, nr2) { next.push([c1, c2, nr1, nr2]); }
            }}
        }
        boxes = next; depth += 1;
    };
    println!("\nCERTIFIED — on the NONLINEAR weak-torque pendulum, the directly-synthesized CLF energy is a PROVEN");
    println!("Lyapunov certificate under its OWN greedy control ({} boxes at the final depth). One energy, both", final_boxes);
    println!("roles, soundly, where value iteration cannot — on the fabric.");
}
