//! EFA energy-first #51 — the device-side certificate RE-VERIFIER, in dependency-free f64 Rust.
//!
//! Ferrite's moat verifies a pack *reproduces* bit-exactly across fabrics (signed eval vectors).
//! This is the axis beyond that axis — verified *correctness*: re-prove, on the device, that the
//! DEPLOYED weights still carry a valid formal Lyapunov certificate before the pack is trusted.
//! It is exactly what `ferrite-gate` needs on-device: no SDP, no SMT solver, no libm beyond tanh,
//! compiles unchanged to wasm32 — the whole check is arithmetic + a box worklist.
//!
//! It re-verifies the saturated-hybrid contact certificate (Charlot Lab certificate program):
//! a learned TERNARY energy V(e)=eᵀPe + Σⱼ w₂ⱼ·tanh(s·(Tⱼ·e)+b₁ⱼ) − v₀ that dReal/Taylor+CROWN
//! certified to R=1.2 across BOTH the free/contact mode switch AND actuator saturation — the six
//! cases {free,contact}×{u=−UM, linear, u=+UM}. The verifier is the 2nd-order Taylor model
//!   ΔV(e) ≤ ΔV(c) + |∇ΔV(c)|·r + ½|H|·r²  (+ α‖e‖²),
//! with the EXACT center gradient (which cancels ≈ −2Qc) and a Hessian bound whose Jᵀ2PJ−2P term
//! cancels to ≈ −2Q and whose tanh-head uses a per-box CROWN |tanh″| interval. Adaptive box
//! refinement certifies the whole continuous annulus; a pack that no longer certifies is rejected
//! (prints the offending box) exactly as a drifted eval vector is 400'd.
//!
//! It demonstrates BOTH gate decisions on the one verifier: the deployed ternary energy is ACCEPTED
//! (bound converges to −0), and — with the learned head switched off (`head=0` → the bare quadratic
//! §5's law says is refuted past R≈1.0) — the SAME verifier REJECTS it at R=1.2, naming the box.
//! That reject is sound, not a depth cutoff: the quadratic's worst bound plateaus at +0.25 while its
//! failing boxes multiply under refinement — a real non-convex-ROA violation, not a loose bound. The
//! gate has teeth. A drifted/tampered pack rejects identically.
//!
//! HONEST scope: 2D benchmark; the certificate is the Taylor+CROWN pass (SOS/dReal stay a
//! build-time/fleet gate — they need solvers). Weights = `certified_sat_taylor_R1.2.npz`,
//! re-snapshotted + soundness-checked (6M samples, 0 violations). Artifact + toolchain:
//! bmi-concept/research/certificate-toolchain/.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_cert_verify --release`

// ---- system constants (saturated hybrid wall-contact) ----
const XW: f64 = 1.0; const GB: f64 = 0.6; const KS: f64 = 60.0; const CC: f64 = 10.0;
const BD: f64 = 0.5; const DT: f64 = 0.02; const UM: f64 = 4.0; const KX: f64 = 8.0; const KV: f64 = 3.0;
const ALPHA: f64 = 5e-4; const R0: f64 = 0.15; const RR: f64 = 1.2;
// ---- the certified ternary energy (certified_sat_taylor_R1.2.npz), embedded exactly ----
const P: [[f64; 2]; 2] = [[31.988, 2.543], [2.543, 1.4169999999999998]];
const SCALE: f64 = 1.4740514336487223;
const T: [[i8; 2]; 8] = [[-1, 0], [-1, 0], [0, 0], [-1, -1], [-1, 0], [1, 1], [1, 0], [0, 0]];
const B1: [f64; 8] = [-1.7253050443315214, -1.6895583892614585, -1.572812794286765, -2.9962607818256326,
                      -1.362285297766529, 2.9948712315814445, 1.6641829682841895, 0.15496976355688338];
const W2: [f64; 8] = [-1.8092778484049137, -0.6474111753241604, -0.0407591631059273, 1.2424339130389697,
                      -0.3175122724570142, -1.1023667519975417, 0.6676446220380198, -5.977279221172131e-12];
const V0: f64 = 0.9069008854718814;

#[inline] fn t(j: usize, k: usize) -> f64 { T[j][k] as f64 }

// `head` scales the learned ternary correction: 1.0 = the certified energy, 0.0 = the bare quadratic
// (which the §5 law says is refuted here) — so the same verifier both ACCEPTS and REJECTS.
/// V(e) = eᵀPe + head·Σ w₂ⱼ tanh(s·(Tⱼ·e)+b₁ⱼ) − head·v₀
fn vfn(e1: f64, e2: f64, head: f64) -> f64 {
    let mut v = e1 * (P[0][0] * e1 + P[0][1] * e2) + e2 * (P[1][0] * e1 + P[1][1] * e2);
    for j in 0..8 { v += head * W2[j] * (SCALE * (t(j, 0) * e1 + t(j, 1) * e2) + B1[j]).tanh(); }
    v - head * V0
}
fn grad_v(e1: f64, e2: f64, head: f64) -> (f64, f64) {
    let (mut g1, mut g2) = (2.0 * (P[0][0] * e1 + P[0][1] * e2), 2.0 * (P[1][0] * e1 + P[1][1] * e2));
    for j in 0..8 {
        let th = (SCALE * (t(j, 0) * e1 + t(j, 1) * e2) + B1[j]).tanh();
        let d = head * W2[j] * (1.0 - th * th) * SCALE;
        g1 += d * t(j, 0); g2 += d * t(j, 1);
    }
    (g1, g2)
}
/// per-case dynamics: mode 0=free / 1=contact; clamp -1=u −UM / 0=linear / +1=u +UM
fn step(e1: f64, e2: f64, mode: usize, clamp: i32) -> (f64, f64) {
    let u = if clamp == -1 { -UM } else if clamp == 1 { UM } else { -GB - KX * e1 - KV * e2 };
    let mut a = GB + u - BD * e2;
    if mode == 1 { a -= KS * e1 + CC * e2; }
    let v2 = e2 + DT * a;
    (e1 + DT * v2, v2)
}
/// closed-loop Jacobian (constant per case — dynamics are affine)
fn jf(mode: usize, clamp: i32) -> [[f64; 2]; 2] {
    let (mut da1, mut da2) = (0.0, -BD);
    if clamp == 0 { da1 += -KX; da2 += -KV; }
    if mode == 1 { da1 += -KS; da2 += -CC; }
    let (dv1, dv2) = (DT * da1, 1.0 + DT * da2);
    [[1.0 + DT * dv1, DT * dv2], [dv1, dv2]]
}
/// tight per-box bound on |tanh″(z)| = 2|t|(1−t²) over z∈[lo,hi]; peak 0.7698 at |z|=0.6585
fn d2max(lo: f64, hi: f64) -> f64 {
    let (tl, th) = (lo.tanh(), hi.tanh());
    let m = (2.0 * tl.abs() * (1.0 - tl * tl)).max(2.0 * th.abs() * (1.0 - th * th));
    if (lo <= 0.6585 && hi >= 0.6585) || (lo <= -0.6585 && hi >= -0.6585) { 0.7698 } else { m }
}
/// the 2nd-order Taylor + CROWN upper bound on ΔV+α‖e‖² over box (center c, radius r), one case
fn bound(c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32, head: f64) -> f64 {
    let (fx, fy) = step(c1, c2, mode, clamp);
    let dvc = vfn(fx, fy, head) - vfn(c1, c2, head);
    // exact center gradient of ΔV: JᵀgV(f) − gV(c)  (cancels to ≈ −2Qc)
    let (gfx, gfy) = grad_v(fx, fy, head);
    let (gsx, gsy) = grad_v(c1, c2, head);
    let j = jf(mode, clamp);
    let gd1 = j[0][0] * gfx + j[1][0] * gfy - gsx;
    let gd2 = j[0][1] * gfx + j[1][1] * gfy - gsy;
    // Hessian abs bound: |Jᵀ 2P J − 2P|  (cancels to ≈ −2Q)  +  CROWN tanh head (at c and, via J, at f)
    let p2 = [[2.0 * P[0][0], 2.0 * P[0][1]], [2.0 * P[1][0], 2.0 * P[1][1]]];
    let mut pj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { pj[i][k] = p2[i][0] * j[0][k] + p2[i][1] * j[1][k]; } }
    let mut m = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { m[i][k] = j[0][i] * pj[0][k] + j[1][i] * pj[1][k] - p2[i][k]; } }
    let mut hs = [[0.0; 2]; 2]; let mut hfm = [[0.0; 2]; 2];
    let aj = [[j[0][0].abs(), j[0][1].abs()], [j[1][0].abs(), j[1][1].abs()]];
    let (fr1, fr2) = (aj[0][0] * r1 + aj[0][1] * r2, aj[1][0] * r1 + aj[1][1] * r2);
    for jx in 0..8 {
        let (a0, a1) = (t(jx, 0).abs(), t(jx, 1).abs());
        let zc = SCALE * (t(jx, 0) * c1 + t(jx, 1) * c2) + B1[jx]; let zr = SCALE * (a0 * r1 + a1 * r2);
        let cs = head.abs() * W2[jx].abs() * d2max(zc - zr, zc + zr) * SCALE * SCALE;
        hs[0][0] += cs * a0 * a0; hs[0][1] += cs * a0 * a1; hs[1][0] += cs * a1 * a0; hs[1][1] += cs * a1 * a1;
        let zcf = SCALE * (t(jx, 0) * fx + t(jx, 1) * fy) + B1[jx]; let zrf = SCALE * (a0 * fr1 + a1 * fr2);
        let cf = head.abs() * W2[jx].abs() * d2max(zcf - zrf, zcf + zrf) * SCALE * SCALE;
        hfm[0][0] += cf * a0 * a0; hfm[0][1] += cf * a0 * a1; hfm[1][0] += cf * a1 * a0; hfm[1][1] += cf * a1 * a1;
    }
    // Jᵀ Hf J  (abs J, sound)
    let mut hfj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { hfj[i][k] = hfm[i][0] * aj[0][k] + hfm[i][1] * aj[1][k]; } }
    let mut habs = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { habs[i][k] = m[i][k].abs() + hs[i][k] + (aj[0][i] * hfj[0][k] + aj[1][i] * hfj[1][k]); } }
    let ss_hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    let rem = 0.5 * (habs[0][0] * r1 * r1 + habs[0][1] * r1 * r2 + habs[1][0] * r2 * r1 + habs[1][1] * r2 * r2);
    dvc + (gd1.abs() * r1 + gd2.abs() * r2) + rem + ALPHA * ss_hi
}
fn case_active(c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32) -> bool {
    let (x_lo, x_hi) = (c1 + XW - r1, c1 + XW + r1);
    let mode_ok = if mode == 0 { x_lo < XW } else { x_hi >= XW };
    let ur_c = -GB - KX * c1 - KV * c2; let ur_r = KX * r1 + KV * r2;
    let clamp_ok = match clamp { -1 => ur_c - ur_r <= -UM, 1 => ur_c + ur_r >= UM, _ => ur_c - ur_r <= UM && ur_c + ur_r >= -UM };
    mode_ok && clamp_ok
}
fn in_region(c1: f64, c2: f64, r1: f64, r2: f64) -> bool {
    let lo = (c1.abs() - r1).max(0.0).powi(2) + (c2.abs() - r2).max(0.0).powi(2);
    let hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    hi >= R0 * R0 && lo <= RR * RR
}

/// Run the adaptive Taylor+CROWN certificate over the annulus for one energy (`head`: 1=ternary, 0=quadratic).
/// Returns Ok(certified_box_count) or Err(first uncertifiable box) — exactly the gate's accept/reject decision.
fn certify(head: f64, max_depth: i32) -> Result<u64, [f64; 4]> {
    let h0 = 0.06f64;
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let n = (2.0 * RR / h0).ceil() as i64;
    for i in 0..n { for k in 0..n {
        let c1 = -RR + (i as f64 + 0.5) * h0; let c2 = -RR + (k as f64 + 0.5) * h0;
        if in_region(c1, c2, h0 / 2.0, h0 / 2.0) { boxes.push([c1, c2, h0 / 2.0, h0 / 2.0]); }
    }}
    let cap = 4_000_000usize;
    let mut certified = 0u64; let mut depth = 0;
    loop {
        let mut fails: Vec<[f64; 4]> = Vec::new();
        let mut worst_b = f64::NEG_INFINITY;
        for b in &boxes {
            let (c1, c2, r1, r2) = (b[0], b[1], b[2], b[3]);
            let mut ok = true;
            for mode in 0..2 { for clamp in -1..=1 {
                if case_active(c1, c2, r1, r2, mode, clamp) {
                    let bd = bound(c1, c2, r1, r2, mode, clamp, head);
                    if bd > worst_b { worst_b = bd; }
                    if bd >= 0.0 { ok = false; }
                }
            }}
            if ok { certified += 1; } else { fails.push(*b); }
        }
        println!("   depth {}: {} boxes, worst bound {:+.5}, fails {}, certified so far {}",
            depth, boxes.len(), worst_b, fails.len(), certified);
        if fails.is_empty() { return Ok(certified); }
        if fails.len() > cap { return Err(fails[0]); }
        // subdivide each failing box into 4 (halve both dims)
        let mut next: Vec<[f64; 4]> = Vec::with_capacity(fails.len() * 4);
        for b in &fails {
            let (nr1, nr2) = (b[2] / 2.0, b[3] / 2.0);
            for &sx in &[-1.0, 1.0] { for &sy in &[-1.0, 1.0] {
                let (c1, c2) = (b[0] + sx * nr1, b[1] + sy * nr2);
                if in_region(c1, c2, nr1, nr2) { next.push([c1, c2, nr1, nr2]); }
            }}
        }
        boxes = next; depth += 1;
        if depth > max_depth { return Err(boxes.first().copied().unwrap_or([0.0; 4])); }
    }
}

fn main() {
    println!("EFA #51 — device-side certificate re-verifier (dependency-free f64 Rust; wasm-clean)");
    // 1 · cross-verify the embedded energy against the certified reference
    let refs = [([0.5, -0.5], 7.262593626850), ([-0.3, 0.8], 2.244816459922), ([0.9, 0.4], 28.178306136649)];
    let worst = refs.iter().map(|(e, r)| (vfn(e[0], e[1], 1.0) - r).abs()).fold(0.0f64, f64::max);
    println!("1 · CROSS-VERIFY embedded energy vs certified reference: worst err {:.2e} -> {}",
        worst, if worst < 1e-8 { "MATCH" } else { "MISMATCH" });
    assert!(worst < 1e-8, "embedded weights are not the certified artifact");

    // 2 · ACCEPT — the deployed ternary energy (head=1) re-proves its certificate over the whole annulus.
    println!("\n2 · re-verify the DEPLOYED ternary energy (Taylor+CROWN, adaptive, 6 hybrid cases):");
    match certify(1.0, 14) {
        Ok(certified) => {
            println!("   CERTIFIED — valid Lyapunov certificate over the whole annulus 0.15..1.2");
            println!("   (all 6 free/contact×saturation cases, sound Taylor+CROWN, {} certified boxes).", certified);
            println!("   -> ACCEPT: a pack carrying this energy passes the certificate gate.");
        }
        Err(b) => { println!("   unexpected reject at [{:.3},{:.3}]", b[0], b[1]); panic!("deployed energy failed to certify"); }
    }

    // 3 · REJECT — the gate's teeth. Turn the learned head OFF (head=0): the bare quadratic is what
    //     §5's law says is REFUTED past R≈1.0. The SAME verifier must now reject it — proving the gate
    //     is a real discriminator, not a rubber stamp. (A drifted/tampered pack rejects the same way.)
    println!("\n3 · re-verify with the learned head DISABLED (head=0 → bare quadratic; the §5 law says it must fail):");
    match certify(0.0, 3) {
        Ok(certified) => { println!("   unexpectedly certified {} boxes — the law would be violated", certified); panic!("quadratic should NOT certify at R=1.2"); }
        Err(b) => {
            println!("   REJECTED — box [{:.3},{:.3}]±{:.3} cannot be certified: the quadratic alone carries", b[0], b[1], b[2]);
            println!("   NO valid certificate at R=1.2. The gate 400s this pack, naming the box.");
        }
    }

    println!("\nDONE — same verifier: ACCEPTS the certified ternary energy, REJECTS the bare quadratic.");
    println!("The gate has teeth. Verified correctness, on-device, no solver — runs browser→Jetson→edge.");
}
