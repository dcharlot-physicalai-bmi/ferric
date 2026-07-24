//! EFA-2 · REACHER, HYBRID controller — closing the K=1 gap with v = −κ∇ₐE + w (the ledger recipe, ebm_hflow).
//! The plain flow reached only 63% at K=1 (endpoint-precision gap). The hybrid gives the single step a correctly
//! oriented gradient: a scalar potential E(obs,a,t) whose action-gradient −κ∇ₐE points toward the demonstrator action
//! (a bowl minimized at u*), plus a small penalized correction w(obs,a,t) that absorbs the residual the scalar can't
//! express. Trained jointly by conditional flow matching with λ‖w‖² (2nd-order autograd: ∇ₐE is inside the loss).
//! Same verified Rust engine, same tanh-PD demonstrator (98%), same eval. Gate: flow-K=1 was 63%; target ≥90% at K=1.
//! HONEST: distills the tanh-PD demonstrator; contact-free Reacher-class on our engine; one seed. Records the field
//! split (how much of the velocity flows through ∇E vs the correction) — the true "energy-firstness" number.
//!
//! Run: `cargo run -p ferric-tensor --example efa2_reacher_hyb --release`
use ferric_core::Context;
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::f32::consts::PI;
use std::sync::Arc;
const L1: f32 = 1.0; const L2: f32 = 1.0; const DT: f32 = 0.05; const TMAX: f32 = 4.0; const TOL: f32 = 0.12;
const UMAX: f32 = 10.0; const KP: f32 = 20.0; const KD: f32 = 9.0; const H: usize = 160; const KAPPA: f32 = 2.0; const LAMBDA: f32 = 0.1;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
struct Arm { m: [f32; 2], l: [f32; 2], lc: [f32; 2], ii: [f32; 2] }
impl Arm {
    fn new() -> Arm { Arm { m: [1.0, 1.0], l: [L1, L2], lc: [0.5, 0.5], ii: [0.083, 0.083] } }
    fn jac(&self, q: &[f32; 2]) -> [[[f32; 2]; 2]; 2] { let ph = [q[0], q[0] + q[1]]; let mut j = [[[0.0f32; 2]; 2]; 2];
        for i in 0..2 { for jj in 0..=i { let mut v = [0.0f32; 2];
            for k in jj..i { v[0] += self.l[k] * ph[k].cos(); v[1] += self.l[k] * ph[k].sin(); }
            v[0] += self.lc[i] * ph[i].cos(); v[1] += self.lc[i] * ph[i].sin(); j[i][jj] = v; } } j }
    fn mm(&self, q: &[f32; 2]) -> [[f32; 2]; 2] { let j = self.jac(q); let mut m = [[0.0f32; 2]; 2];
        for a in 0..2 { for b in 0..2 { let mut s = 0.0;
            for i in 0..2 { s += self.m[i] * (j[i][a][0] * j[i][b][0] + j[i][a][1] * j[i][b][1]);
                if a <= i && b <= i { s += self.ii[i]; } } m[a][b] = s; } } m }
    fn bias(&self, q: &[f32; 2], qd: &[f32; 2]) -> [f32; 2] { let eps = 1e-3; let mut dm = [[[0.0f32; 2]; 2]; 2];
        for k in 0..2 { let mut qp = *q; qp[k] += eps; let mut qmm = *q; qmm[k] -= eps; let (mp, mn) = (self.mm(&qp), self.mm(&qmm));
            for i in 0..2 { for jj in 0..2 { dm[k][i][jj] = (mp[i][jj] - mn[i][jj]) / (2.0 * eps); } } }
        let mut b = [0.0f32; 2]; for i in 0..2 { let mut t1 = 0.0; let mut t2 = 0.0;
            for jj in 0..2 { for k in 0..2 { t1 += dm[k][i][jj] * qd[jj] * qd[k]; t2 += dm[i][jj][k] * qd[jj] * qd[k]; } } b[i] = t1 - 0.5 * t2; } b }
    fn forward(&self, q: &[f32; 2], qd: &[f32; 2], tau: &[f32; 2]) -> [f32; 2] { let m = self.mm(q); let b = self.bias(q, qd);
        let (r0, r1) = (tau[0] - b[0], tau[1] - b[1]); let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        [(m[1][1] * r0 - m[0][1] * r1) / det, (-m[1][0] * r0 + m[0][0] * r1) / det] }
    fn step(&self, q: &[f32; 2], qd: &[f32; 2], tau: &[f32; 2]) -> ([f32; 2], [f32; 2]) {
        let d = |q: &[f32; 2], v: &[f32; 2]| (*v, self.forward(q, v, tau));
        let ad = |a: &[f32; 2], b: &[f32; 2], s: f32| [a[0] + s * b[0], a[1] + s * b[1]];
        let (k1q, k1v) = d(q, qd); let (k2q, k2v) = d(&ad(q, &k1q, DT / 2.0), &ad(qd, &k1v, DT / 2.0));
        let (k3q, k3v) = d(&ad(q, &k2q, DT / 2.0), &ad(qd, &k2v, DT / 2.0)); let (k4q, k4v) = d(&ad(q, &k3q, DT), &ad(qd, &k3v, DT));
        ([wrap(q[0] + DT / 6.0 * (k1q[0] + 2.0 * k2q[0] + 2.0 * k3q[0] + k4q[0])), wrap(q[1] + DT / 6.0 * (k1q[1] + 2.0 * k2q[1] + 2.0 * k3q[1] + k4q[1]))],
         [qd[0] + DT / 6.0 * (k1v[0] + 2.0 * k2v[0] + 2.0 * k3v[0] + k4v[0]), qd[1] + DT / 6.0 * (k1v[1] + 2.0 * k2v[1] + 2.0 * k3v[1] + k4v[1])]) }
    fn tip(&self, q: &[f32; 2]) -> [f32; 2] { let ph = [q[0], q[0] + q[1]];
        [self.l[0] * ph[0].sin() + self.l[1] * ph[1].sin(), -self.l[0] * ph[0].cos() - self.l[1] * ph[1].cos()] }
}
fn ik(x: f32, y: f32, elbow: f32) -> [f32; 2] { let r2 = x * x + y * y; let c2 = ((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)).clamp(-1.0, 1.0);
    let q2 = elbow * c2.acos(); let beta = x.atan2(-y); let q1 = beta - (L2 * q2.sin()).atan2(L1 + L2 * q2.cos()); [wrap(q1), wrap(q2)] }
fn pick_ik(x: f32, y: f32, q: &[f32; 2]) -> [f32; 2] { let (a, b) = (ik(x, y, 1.0), ik(x, y, -1.0));
    let da = wrap(a[0] - q[0]).abs() + wrap(a[1] - q[1]).abs(); let db = wrap(b[0] - q[0]).abs() + wrap(b[1] - q[1]).abs(); if da <= db { a } else { b } }
fn pd(q: &[f32; 2], qd: &[f32; 2], tgt: [f32; 2]) -> [f32; 2] { let qs = pick_ik(tgt[0], tgt[1], q);
    [UMAX * ((KP * wrap(qs[0] - q[0]) - KD * qd[0]) / UMAX).tanh(), UMAX * ((KP * wrap(qs[1] - q[1]) - KD * qd[1]) / UMAX).tanh()] }
fn sample_target(seed: u32) -> [f32; 2] { let r = 0.4 + u(seed, 1) * (L1 + L2 - 0.5); let a = u(seed, 2) * 2.0 * PI; [r * a.sin(), -r * a.cos()] }
fn obs(arm: &Arm, q: &[f32; 2], qd: &[f32; 2], tgt: [f32; 2]) -> [f32; 8] { let tp = arm.tip(q);
    [q[0].cos(), q[0].sin(), q[1].cos(), q[1].sin(), qd[0] * 0.3, qd[1] * 0.3, tgt[0] - tp[0], tgt[1] - tp[1]] }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let (a, b) = (u(i as u32, seed), u(i as u32, seed + 1));
    sc * (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }).collect() }
// CPU nets
struct Net { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: Vec<f32>, no: usize }
impl Net { fn f(&self, x: &[f32]) -> Vec<f32> {
    let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..x.len() { z += x[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
    (0..self.no).map(|c| { let mut o = self.b3[c]; for j in 0..H { o += h2[j] * self.w3[j * self.no + c]; } o }).collect() } }
// hybrid velocity v = −κ∇ₐE + w; ∇ₐE via central finite difference at inference
fn hyb_v(pot: &Net, cor: &Net, ob: &[f32; 8], a: [f32; 2], t: f32) -> [f32; 2] {
    let feat = |aa: [f32; 2]| { let mut f = ob.to_vec(); f.push(aa[0]); f.push(aa[1]); f.push(t); f };
    let e = 1e-3;
    let ga = (pot.f(&feat([a[0] + e, a[1]]))[0] - pot.f(&feat([a[0] - e, a[1]]))[0]) / (2.0 * e);
    let gb = (pot.f(&feat([a[0], a[1] + e]))[0] - pot.f(&feat([a[0], a[1] - e]))[0]) / (2.0 * e);
    let w = cor.f(&feat(a)); [-KAPPA * ga + w[0], -KAPPA * gb + w[1]] }
fn act_hybrid(pot: &Net, cor: &Net, ob: &[f32; 8], kk: usize) -> [f32; 2] { let mut a = [0.0f32; 2];
    for k in 0..kk { let t = k as f32 / kk as f32; let v = hyb_v(pot, cor, ob, a, t); a[0] += v[0] / kk as f32; a[1] += v[1] / kk as f32; }
    [a[0].clamp(-1.0, 1.0) * UMAX, a[1].clamp(-1.0, 1.0) * UMAX] }
fn episode<F: FnMut(&[f32; 2], &[f32; 2]) -> [f32; 2]>(arm: &Arm, seed: u32, mut pol: F) -> f32 {
    let tgt = sample_target(seed); let mut q = [(u(seed, 5) * 2.0 - 1.0) * PI, (u(seed, 6) * 2.0 - 1.0) * PI]; let mut qd = [0.0f32; 2];
    for _ in 0..((TMAX / DT) as usize) { let a = pol(&q, &qd); let (nq, nv) = arm.step(&q, &qd, &a); q = nq; qd = nv; }
    let tp = arm.tip(&q); ((tp[0] - tgt[0]).powi(2) + (tp[1] - tgt[1]).powi(2)).sqrt() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let arm = Arm::new();
    println!("  EFA-2 · REACHER · HYBRID v = −κ∇ₐE + w (closing the K=1 gap; κ={KAPPA}, λ={LAMBDA}, H={H})\n");
    let ctx = Arc::new(Context::new().await.expect("ctx")); let nin = 11; let bs = 256;
    // potential params: nin rank-1 [1,H] + b1 + W2 + b2 + W3[H,1] + b3[1]
    let mkp = |seed: u32, no: usize| -> Vec<Tensor> { let mut p: Vec<Tensor> = (0..nin).map(|c| Tensor::from_vec(&ctx, &randn(H, seed + c as u32, 0.4), &[1, H])).collect();
        p.push(Tensor::zeros(&ctx, &[H])); p.push(Tensor::from_vec(&ctx, &randn(H * H, seed + 60, 1.0 / (H as f32).sqrt()), &[H, H])); p.push(Tensor::zeros(&ctx, &[H]));
        p.push(Tensor::from_vec(&ctx, &randn(H * no, seed + 61, 1.0 / (H as f32).sqrt()), &[H, no])); p.push(Tensor::zeros(&ctx, &[no])); p };
    let mut pp = mkp(500, 1); let mut wp = mkp(700, 2);
    let mut adamp = Adam::new(&pp, 0.0012); let mut adamw = Adam::new(&wp, 0.0012);
    let netf = |f: &[Var], pv: &[Var], no: usize| { let mut pre = pv[nin].clone(); for c in 0..nin { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[nin + 1]).add(&pv[nin + 2]).relu().matmul(&pv[nin + 3]).add(&pv[nin + 4 + no - 1 - (no - 1)]) };  // b3 at nin+4
    println!("  distilling the hybrid (potential ∇E + penalized correction w; 2nd-order autograd; CFM to tanh-PD):");
    for it in 0..18000u32 {
        let mut cols: Vec<Vec<f32>> = (0..nin).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs * 2];
        for i in 0..bs { let sd = it * 271 + i as u32; let tgt = sample_target(sd % 4000 + 1);
            let q = [(u(sd, 5) * 2.0 - 1.0) * PI, (u(sd, 6) * 2.0 - 1.0) * PI]; let qd = [(u(sd, 7) * 2.0 - 1.0) * 2.0, (u(sd, 8) * 2.0 - 1.0) * 2.0];
            let ob = obs(&arm, &q, &qd, tgt); let ud = pd(&q, &qd, tgt); let un = [ud[0] / UMAX, ud[1] / UMAX];
            let g1 = (-2.0 * u(sd, 30).ln()).sqrt() * (2.0 * PI * u(sd, 32)).cos(); let g2 = (-2.0 * u(sd, 31).ln()).sqrt() * (2.0 * PI * u(sd, 33)).cos();
            let t = u(sd, 9) * 0.9; let a0 = [0.2 * g1, 0.2 * g2];
            for c in 0..8 { cols[c][i] = ob[c]; }
            cols[8][i] = (1.0 - t) * a0[0] + t * un[0]; cols[9][i] = (1.0 - t) * a0[1] + t * un[1]; cols[10][i] = t;
            tb[i * 2] = un[0] - a0[0]; tb[i * 2 + 1] = un[1] - a0[1]; }
        let ppv: Vec<Var> = pp.iter().map(|t| Var::leaf(t.clone())).collect();
        let wpv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        // action columns as grad-tracked leaves
        let a1v = Var::leaf(Tensor::from_vec(&ctx, &cols[8], &[bs, 1])); let a2v = Var::leaf(Tensor::from_vec(&ctx, &cols[9], &[bs, 1]));
        let mut feat: Vec<Var> = (0..8).map(|c| Var::leaf(Tensor::from_vec(&ctx, &cols[c], &[bs, 1]))).collect();
        feat.push(a1v.clone()); feat.push(a2v.clone()); feat.push(Var::leaf(Tensor::from_vec(&ctx, &cols[10], &[bs, 1])));
        let e = netf(&feat, &ppv, 1);                                    // potential [bs,1]
        let ge = grad(&e.sum_all(), &[a1v.clone(), a2v.clone()], None);  // ∇ₐE per sample (differentiable)
        let w = netf(&feat, &wpv, 2);                                    // correction [bs,2]
        let w1 = w.matmul(&Var::leaf(Tensor::from_vec(&ctx, &[1.0, 0.0], &[2, 1]))); let w2 = w.matmul(&Var::leaf(Tensor::from_vec(&ctx, &[0.0, 1.0], &[2, 1])));
        let kap = Var::leaf(Tensor::from_vec(&ctx, &[KAPPA], &[1]));
        let v1 = w1.sub(&ge[0].mul(&kap)); let v2 = w2.sub(&ge[1].mul(&kap));   // v = −κ∇E + w
        let t1 = Var::leaf(Tensor::from_vec(&ctx, &(0..bs).map(|i| tb[i * 2]).collect::<Vec<_>>(), &[bs, 1]));
        let t2 = Var::leaf(Tensor::from_vec(&ctx, &(0..bs).map(|i| tb[i * 2 + 1]).collect::<Vec<_>>(), &[bs, 1]));
        let d1 = v1.sub(&t1); let d2 = v2.sub(&t2);
        let lam = Var::leaf(Tensor::from_vec(&ctx, &[LAMBDA], &[1]));
        let loss = d1.mul(&d1).add(&d2.mul(&d2)).mean_all().add(&w1.mul(&w1).add(&w2.mul(&w2)).mean_all().mul(&lam));
        loss.backward();
        let gp: Vec<Tensor> = ppv.iter().zip(&pp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        let gw: Vec<Tensor> = wpv.iter().zip(&wp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamp.step(&mut pp, &gp); adamw.step(&mut wp, &gw);
        if it % 4500 == 4499 { println!("     iter {:>5}: loss {:.4}", it + 1, loss.value().to_vec().await[0]); }
    }
    let ex = async |p: &Vec<Tensor>, no: usize| -> Net { let mut w = Vec::new(); for c in 0..nin { w.push(p[c].to_vec().await); }
        Net { w, b1: p[nin].to_vec().await, w2: p[nin + 1].to_vec().await, b2: p[nin + 2].to_vec().await, w3: p[nin + 3].to_vec().await, b3: p[nin + 4].to_vec().await, no } };
    let pot = ex(&pp, 1).await; let cor = ex(&wp, 2).await;
    // field split: mean |κ∇E| vs mean |w| over random states (the energy-firstness number)
    let (mut me, mut mw) = (0.0f32, 0.0f32); for k in 0..400u32 { let tgt = sample_target(k + 1);
        let q = [(u(k, 5) * 2.0 - 1.0) * PI, (u(k, 6) * 2.0 - 1.0) * PI]; let qd = [(u(k, 7) * 2.0 - 1.0) * 2.0, (u(k, 8) * 2.0 - 1.0) * 2.0];
        let ob = obs(&arm, &q, &qd, tgt); let feat = |aa: [f32; 2]| { let mut f = ob.to_vec(); f.push(aa[0]); f.push(aa[1]); f.push(0.0); f };
        let e = 1e-3; let ga = (pot.f(&feat([e, 0.0]))[0] - pot.f(&feat([-e, 0.0]))[0]) / (2.0 * e);
        let gb = (pot.f(&feat([0.0, e]))[0] - pot.f(&feat([0.0, -e]))[0]) / (2.0 * e); let w = cor.f(&feat([0.0, 0.0]));
        me += KAPPA * (ga * ga + gb * gb).sqrt(); mw += (w[0] * w[0] + w[1] * w[1]).sqrt(); }
    println!("\n  field split: mean |κ∇E| : mean |w| = {:.2} : {:.2}  ⇒ potential carries {:.0}% of the velocity", me / 400.0, mw / 400.0, me / (me + mw) * 100.0);
    println!("\n  the card — reach (fingertip < {:.2}; 200 episodes) · flow-only was K=1 63% → K=8 90%:", TOL);
    for kk in [1usize, 2, 4] { let (mut ok, mut md) = (0, 0.0f32);
        for k in 0..200u32 { let tgt = sample_target(k); let d = episode(&arm, k, |q, qd| { let ob = obs(&arm, q, qd, tgt); act_hybrid(&pot, &cor, &ob, kk) });
            md += d; if d < TOL { ok += 1; } }
        println!("     hybrid K={}: reach {:>3.0}% · mean final distance {:.3} · {} fwd pass/decision", kk, ok as f32 / 2.0, md / 200.0, kk); }
    let ob = obs(&arm, &[0.3, -0.2], &[0.1, -0.1], [0.5, -0.5]); let a1 = act_hybrid(&pot, &cor, &ob, 1); let a2 = act_hybrid(&pot, &cor, &ob, 1);
    println!("     determinism: {}", if a1[0].to_bits() == a2[0].to_bits() && a1[1].to_bits() == a2[1].to_bits() { "bit-exact ✓" } else { "✗" });
    println!("\n  Honest: distills the tanh-PD demonstrator; contact-free Reacher on the verified engine; one seed.");
    println!("  The hybrid gives the K=1 step a correctly-oriented ∇E gradient; the field split reports true energy-firstness.");
}
