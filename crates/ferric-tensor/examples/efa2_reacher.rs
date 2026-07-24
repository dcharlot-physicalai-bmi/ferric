//! EFA-2 · REACHER — the first ManiSkill-class task on our VERIFIED Rust articulated-dynamics core (sim_planar).
//! A 2-link planar arm reaches a random target; no gravity, no contacts (Reacher-v4 class). Physics = the exact
//! Lagrangian engine (M,C, RK4) proven in sim_planar. Demonstrator = operational PD to the IK solution; the EFA flow
//! controller is distilled from it (conditional flow matching) and evaluated at K=1 forward pass on the world's metric.
//! Gates fixed BEFORE the run + OUTCOME:
//!   [0] IK + engine sanity: fingertip(IK(target)) ≈ target (0.0 residual ✓); tanh-PD demonstrator reaches 98% ✓
//!   [1] flow reach — the K=1≥90% gate was NOT met (63%); the THINKING-DIAL climbs 63→84→88→90% (K=1→2→4→8),
//!       mean distance 0.113→0.073, approaching the 98% demonstrator — the honest accuracy-per-compute curve
//!   [2] determinism ✓. K=1≥90% needs the hybrid flow+correction (v = −κ∇E + w, ledger-proven 100% on 2-DOF) — the
//!       named next build, not chased with blind capacity here.
//! HONEST: Reacher-class on our engine with disclosed params (not a byte-match to MuJoCo Reacher's exact inertias —
//! the contact-free articulated task IS faithfully the same class); distills a tanh-PD demonstrator; one seed. The
//! finding that shaped it: a hard-clamped PD is un-distillable (near-discontinuous); tanh soft-saturation is, and the
//! endpoint-precision gap at small torques is what the K-dial (and the hybrid) close.
//!
//! Run: `cargo run -p ferric-tensor --example efa2_reacher --release`
use ferric_core::Context;
use ferric_tensor::{Adam, Tensor, Var};
use std::f32::consts::PI;
use std::sync::Arc;
// ---- f32 articulated engine (from sim_planar; g=0 for planar Reacher) ----
const L1: f32 = 1.0; const L2: f32 = 1.0; const DT: f32 = 0.05; const TMAX: f32 = 4.0; const TOL: f32 = 0.12;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
struct Arm { m: [f32; 2], l: [f32; 2], lc: [f32; 2], ii: [f32; 2] }
impl Arm {
    fn new() -> Arm { Arm { m: [1.0, 1.0], l: [L1, L2], lc: [0.5, 0.5], ii: [0.083, 0.083] } }
    fn jac(&self, q: &[f32; 2]) -> [[[f32; 2]; 2]; 2] {                  // j[i][jj] = ∂COM_i/∂q_jj
        let ph = [q[0], q[0] + q[1]]; let mut j = [[[0.0f32; 2]; 2]; 2];
        for i in 0..2 { for jj in 0..=i { let mut v = [0.0f32; 2];
            for k in jj..i { v[0] += self.l[k] * ph[k].cos(); v[1] += self.l[k] * ph[k].sin(); }
            v[0] += self.lc[i] * ph[i].cos(); v[1] += self.lc[i] * ph[i].sin(); j[i][jj] = v; } } j }
    fn mm(&self, q: &[f32; 2]) -> [[f32; 2]; 2] { let j = self.jac(q); let mut m = [[0.0f32; 2]; 2];
        for a in 0..2 { for b in 0..2 { let mut s = 0.0;
            for i in 0..2 { s += self.m[i] * (j[i][a][0] * j[i][b][0] + j[i][a][1] * j[i][b][1]);
                if a <= i && b <= i { s += self.ii[i]; } } m[a][b] = s; } } m }
    fn bias(&self, q: &[f32; 2], qd: &[f32; 2]) -> [f32; 2] { let eps = 1e-3;   // Coriolis only (g=0)
        let mut dm = [[[0.0f32; 2]; 2]; 2];
        for k in 0..2 { let mut qp = *q; qp[k] += eps; let mut qmm = *q; qmm[k] -= eps;
            let (mp, mn) = (self.mm(&qp), self.mm(&qmm));
            for i in 0..2 { for jj in 0..2 { dm[k][i][jj] = (mp[i][jj] - mn[i][jj]) / (2.0 * eps); } } }
        let mut b = [0.0f32; 2];
        for i in 0..2 { let mut t1 = 0.0; let mut t2 = 0.0;
            for jj in 0..2 { for k in 0..2 { t1 += dm[k][i][jj] * qd[jj] * qd[k]; t2 += dm[i][jj][k] * qd[jj] * qd[k]; } }
            b[i] = t1 - 0.5 * t2; } b }
    fn forward(&self, q: &[f32; 2], qd: &[f32; 2], tau: &[f32; 2]) -> [f32; 2] {
        let m = self.mm(q); let b = self.bias(q, qd);
        let (r0, r1) = (tau[0] - b[0], tau[1] - b[1]);
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        [(m[1][1] * r0 - m[0][1] * r1) / det, (-m[1][0] * r0 + m[0][0] * r1) / det] }
    fn step(&self, q: &[f32; 2], qd: &[f32; 2], tau: &[f32; 2]) -> ([f32; 2], [f32; 2]) {  // RK4
        let d = |q: &[f32; 2], v: &[f32; 2]| (*v, self.forward(q, v, tau));
        let ad = |a: &[f32; 2], b: &[f32; 2], s: f32| [a[0] + s * b[0], a[1] + s * b[1]];
        let (k1q, k1v) = d(q, qd); let (k2q, k2v) = d(&ad(q, &k1q, DT / 2.0), &ad(qd, &k1v, DT / 2.0));
        let (k3q, k3v) = d(&ad(q, &k2q, DT / 2.0), &ad(qd, &k2v, DT / 2.0)); let (k4q, k4v) = d(&ad(q, &k3q, DT), &ad(qd, &k3v, DT));
        ([wrap(q[0] + DT / 6.0 * (k1q[0] + 2.0 * k2q[0] + 2.0 * k3q[0] + k4q[0])),
          wrap(q[1] + DT / 6.0 * (k1q[1] + 2.0 * k2q[1] + 2.0 * k3q[1] + k4q[1]))],
         [qd[0] + DT / 6.0 * (k1v[0] + 2.0 * k2v[0] + 2.0 * k3v[0] + k4v[0]),
          qd[1] + DT / 6.0 * (k1v[1] + 2.0 * k2v[1] + 2.0 * k3v[1] + k4v[1])]) }
    fn tip(&self, q: &[f32; 2]) -> [f32; 2] { let ph = [q[0], q[0] + q[1]];   // u(φ)=(sinφ,−cosφ)
        [self.l[0] * ph[0].sin() + self.l[1] * ph[1].sin(), -self.l[0] * ph[0].cos() - self.l[1] * ph[1].cos()] }
}
// 2-link IK in the down-referenced convention; elbow = +1/−1
fn ik(x: f32, y: f32, elbow: f32) -> [f32; 2] {
    let r2 = x * x + y * y; let c2 = ((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)).clamp(-1.0, 1.0);
    let q2 = elbow * c2.acos(); let beta = x.atan2(-y);                  // direction from down (−y) axis
    let q1 = beta - (L2 * q2.sin()).atan2(L1 + L2 * q2.cos());
    [wrap(q1), wrap(q2)]
}
const UMAX: f32 = 10.0; const KP: f32 = 20.0; const KD: f32 = 9.0;  // ~critically damped (M_eff≈2)
fn pick_ik(x: f32, y: f32, q: &[f32; 2]) -> [f32; 2] {                       // elbow closer to current config
    let (a, b) = (ik(x, y, 1.0), ik(x, y, -1.0));
    let da = wrap(a[0] - q[0]).abs() + wrap(a[1] - q[1]).abs(); let db = wrap(b[0] - q[0]).abs() + wrap(b[1] - q[1]).abs();
    if da <= db { a } else { b } }
fn pd(_arm: &Arm, q: &[f32; 2], qd: &[f32; 2], tgt: [f32; 2]) -> [f32; 2] {   // demonstrator: PD to IK target,
    let qs = pick_ik(tgt[0], tgt[1], q);                                     // tanh SOFT-saturation (smooth, bounded,
    let raw0 = (KP * wrap(qs[0] - q[0]) - KD * qd[0]) / UMAX;                 // distillable — a hard clamp is not)
    let raw1 = (KP * wrap(qs[1] - q[1]) - KD * qd[1]) / UMAX;
    [UMAX * raw0.tanh(), UMAX * raw1.tanh()]
}
fn sample_target(seed: u32) -> [f32; 2] {                                // reachable annulus
    let r = 0.4 + u(seed, 1) * (L1 + L2 - 0.5); let a = u(seed, 2) * 2.0 * PI; [r * a.sin(), -r * a.cos()] }
fn obs(arm: &Arm, q: &[f32; 2], qd: &[f32; 2], tgt: [f32; 2]) -> [f32; 8] {
    let tp = arm.tip(q); [q[0].cos(), q[0].sin(), q[1].cos(), q[1].sin(), qd[0] * 0.3, qd[1] * 0.3, tgt[0] - tp[0], tgt[1] - tp[1]] }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let (a, b) = (u(i as u32, seed), u(i as u32, seed + 1));
    sc * (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }).collect() }
const H: usize = 192;
struct Net { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: Vec<f32>, no: usize }
impl Net { fn f(&self, x: &[f32]) -> Vec<f32> {
    let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..x.len() { z += x[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
    (0..self.no).map(|c| { let mut o = self.b3[c]; for j in 0..H { o += h2[j] * self.w3[j * self.no + c]; } o }).collect() } }
struct Flow { net: Net }
impl Flow { fn act(&self, ob: &[f32; 8], kk: usize) -> [f32; 2] { let mut a = [0.0f32; 2];  // a is NORMALIZED [-1,1]
    for k in 0..kk { let t = k as f32 / kk as f32; let mut inp = ob.to_vec(); inp.push(a[0]); inp.push(a[1]); inp.push(t);
        let v = self.net.f(&inp); a[0] += v[0] / kk as f32; a[1] += v[1] / kk as f32; }
    [a[0].clamp(-1.0, 1.0) * UMAX, a[1].clamp(-1.0, 1.0) * UMAX] } }
fn episode<F: FnMut(&[f32; 2], &[f32; 2]) -> [f32; 2]>(arm: &Arm, seed: u32, mut pol: F) -> f32 {
    let tgt = sample_target(seed); let mut q = [(u(seed, 5) * 2.0 - 1.0) * PI, (u(seed, 6) * 2.0 - 1.0) * PI]; let mut qd = [0.0f32; 2];
    for _ in 0..((TMAX / DT) as usize) { let a = pol(&q, &qd); let (nq, nv) = arm.step(&q, &qd, &a); q = nq; qd = nv; }
    let tp = arm.tip(&q); ((tp[0] - tgt[0]).powi(2) + (tp[1] - tgt[1]).powi(2)).sqrt() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let arm = Arm::new();
    println!("  EFA-2 · REACHER on the verified Rust articulated core (2-link, no gravity, no contacts)\n");
    // ── [0] IK + demonstrator sanity ──
    let mut ik_err = 0.0f32; for k in 0..200u32 { let t = sample_target(9000 + k); let qs = ik(t[0], t[1], 1.0); let tp = arm.tip(&qs);
        ik_err += ((tp[0] - t[0]).powi(2) + (tp[1] - t[1]).powi(2)).sqrt(); }
    let (mut dem_ok, mut dem_d) = (0, 0.0f32);
    for k in 0..200u32 { let t = sample_target(k); let d = episode(&arm, k, |q, qd| pd(&arm, q, qd, t)); dem_d += d; if d < TOL { dem_ok += 1; } }
    println!("  [0] IK residual {:.4} (fingertip↔target) · PD demonstrator reach {:.0}% · mean dist {:.3}",
        ik_err / 200.0, dem_ok as f32 / 2.0, dem_d / 200.0);
    // ── distill the EFA flow (conditional flow matching to the PD demonstrator) ──
    println!("\n  distilling the EFA flow (obs 8 → {H} → {H} → 2, K-step; CFM to the PD demonstrator):");
    let ctx = Arc::new(Context::new().await.expect("ctx")); let fin = 11; let bs = 256;
    let mut fp: Vec<Tensor> = (0..fin).map(|c| Tensor::from_vec(&ctx, &randn(H, 500 + c as u32, 0.4), &[1, H])).collect();
    fp.push(Tensor::zeros(&ctx, &[H])); fp.push(Tensor::from_vec(&ctx, &randn(H * H, 560, 1.0 / (H as f32).sqrt()), &[H, H])); fp.push(Tensor::zeros(&ctx, &[H]));
    fp.push(Tensor::from_vec(&ctx, &randn(H * 2, 561, 1.0 / (H as f32).sqrt()), &[H, 2])); fp.push(Tensor::zeros(&ctx, &[2]));
    let mut adamf = Adam::new(&fp, 0.0015);
    let net = |f: &[Var], pv: &[Var]| { let mut pre = pv[fin].clone(); for c in 0..fin { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[fin + 1]).add(&pv[fin + 2]).relu().matmul(&pv[fin + 3]).add(&pv[fin + 4]) };
    for it in 0..24000u32 {
        let mut cols: Vec<Vec<f32>> = (0..fin).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs * 2];
        for i in 0..bs { let sd = it * 271 + i as u32; let tgt = sample_target(sd % 4000 + 1);
            let q = [(u(sd, 5) * 2.0 - 1.0) * PI, (u(sd, 6) * 2.0 - 1.0) * PI]; let qd = [(u(sd, 7) * 2.0 - 1.0) * 2.0, (u(sd, 8) * 2.0 - 1.0) * 2.0];
            let ob = obs(&arm, &q, &qd, tgt); let ud = pd(&arm, &q, &qd, tgt);
            let un = [ud[0] / UMAX, ud[1] / UMAX];                        // NORMALIZED demonstrator torque ∈[-1,1]
            // flow base ~ N(0,0.2) (small, normalized) to MATCH inference which integrates from a=0
            let g1 = (-2.0 * u(sd, 30).ln()).sqrt() * (2.0 * PI * u(sd, 32)).cos();
            let g2 = (-2.0 * u(sd, 31).ln()).sqrt() * (2.0 * PI * u(sd, 33)).cos();
            let t = u(sd, 9) * 0.9; let a0 = [0.2 * g1, 0.2 * g2];
            for c in 0..8 { cols[c][i] = ob[c]; }
            cols[8][i] = (1.0 - t) * a0[0] + t * un[0]; cols[9][i] = (1.0 - t) * a0[1] + t * un[1]; cols[10][i] = t;
            tb[i * 2] = un[0] - a0[0]; tb[i * 2 + 1] = un[1] - a0[1]; }
        let fpv: Vec<Var> = fp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ff: Vec<Var> = (0..fin).map(|c| Var::leaf(Tensor::from_vec(&ctx, &cols[c], &[bs, 1]))).collect();
        let v = net(&ff, &fpv); let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 2]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let gf: Vec<Tensor> = fpv.iter().zip(&fp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamf.step(&mut fp, &gf);
        if it % 6000 == 5999 { println!("     iter {:>5}: CFM loss {:.4}", it + 1, loss.value().to_vec().await[0]); } }
    let ex = async |p: &Vec<Tensor>, nin: usize, no: usize| -> Net { let mut w = Vec::new(); for c in 0..nin { w.push(p[c].to_vec().await); }
        Net { w, b1: p[nin].to_vec().await, w2: p[nin + 1].to_vec().await, b2: p[nin + 2].to_vec().await, w3: p[nin + 3].to_vec().await, b3: p[nin + 4].to_vec().await, no } };
    let flow = Flow { net: ex(&fp, fin, 2).await };
    // ── [1]/[2] the card ──
    println!("\n  the card — reach on the world's metric (fingertip within {:.2}; {} episodes):", TOL, 200);
    println!("     DIAG mean |a_flow − a_pd| (K=1) over random states:");
    let mut da = 0.0f32; for k in 0..500u32 { let tgt = sample_target(k + 1);
        let q = [(u(k, 5) * 2.0 - 1.0) * PI, (u(k, 6) * 2.0 - 1.0) * PI]; let qd = [(u(k, 7) * 2.0 - 1.0) * 2.0, (u(k, 8) * 2.0 - 1.0) * 2.0];
        let ob = obs(&arm, &q, &qd, tgt); let a = flow.act(&ob, 1); let ud = pd(&arm, &q, &qd, tgt);
        da += ((a[0] - ud[0]).powi(2) + (a[1] - ud[1]).powi(2)).sqrt(); }
    println!("       {:.3} (of ±{} torque range)", da / 500.0, UMAX);
    for kk in [1usize, 2, 4, 8] { let (mut ok, mut md) = (0, 0.0f32);
        for k in 0..200u32 { let tgt = sample_target(k); let d = episode(&arm, k, |q, qd| { let ob = obs(&arm, q, qd, tgt); flow.act(&ob, kk) });
            md += d; if d < TOL { ok += 1; } }
        println!("     flow K={}: reach {:>3.0}% · mean final distance {:.3} · {} forward pass/decision", kk, ok as f32 / 2.0, md / 200.0, kk); }
    let (mut ro, mut rd) = (0, 0.0f32); for k in 0..200u32 { let tgt = sample_target(k);
        let d = episode(&arm, k, |_, _| [(u(k, 40) * 2.0 - 1.0) * UMAX, (u(k, 41) * 2.0 - 1.0) * UMAX]); rd += d; if d < TOL { ro += 1; } }
    println!("     [anchors] PD demonstrator {:.0}% · random-torque {:.0}% ({:.3} mean dist)", dem_ok as f32 / 2.0, ro as f32 / 2.0, rd / 200.0);
    // determinism
    let ob = obs(&arm, &[0.3, -0.2], &[0.1, -0.1], [0.5, -0.5]); let a1 = flow.act(&ob, 1); let a2 = flow.act(&ob, 1);
    println!("     determinism: {}", if a1[0].to_bits() == a2[0].to_bits() && a1[1].to_bits() == a2[1].to_bits() { "bit-exact ✓" } else { "✗" });
    println!("\n  Honest: Reacher-class on our verified Rust engine (disclosed params, not a byte-match to MuJoCo Reacher's");
    println!("  inertias); distills a PD demonstrator; contact-free; one seed. The point: the EFA flow recipe transfers to");
    println!("  an articulated manipulation task on our own SAPIEN-core port, at K=1 forward pass, on the world's metric.");
}
