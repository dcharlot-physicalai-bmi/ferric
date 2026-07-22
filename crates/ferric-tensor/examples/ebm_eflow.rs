//! EFA energy-first #57 — ENERGY-FIRST flow: v = −∇ₐE. ONE scalar potential that ACTUATES (descend) AND VERIFIES (score).
//!
//! ebm_flow2 fixed multi-DOF actuation with flow-matching but used a PLAIN velocity net — not energy-first. This closes
//! that honestly: parameterize the flow velocity as the NEGATIVE ACTION-GRADIENT of a scalar potential Eθ(s,a,t),
//! v = −∇ₐEθ, and train the same conditional-flow-matching target (v ≈ u*−a₀) by SINGLE-STEP gradient matching
//! (score-matching style, NOT the brittle multi-step BPTT). Then: (1) actuation = integrate a ← a − (1/K)∇ₐEθ (literally
//! energy descent along the flow), and (2) the SAME potential at t=1 is the VERIFY energy (low Eθ(s,·,1) = valid action,
//! since the flow converges to u* there). One potential, two readings — the honest "energy-first" the plain velocity net
//! was not. Tested on the 2-DOF arm. HONEST: distilled demonstrator, small body; the point is unifying actuate+verify in −∇E.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_eflow --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const G5: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const G9: [f32; 9] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
const TG: [(f32, f32); 4] = [(0.8, -0.8), (-1.0, 0.6), (0.5, 1.0), (-0.6, -0.9)];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(s: [f32; 4], u1: f32, u2: f32) -> [f32; 4] {
    let (t1, t2, o1, o2) = (s[0], s[1], s[2], s[3]); let (c1, c2) = (u1.clamp(-UMAX, UMAX), u2.clamp(-UMAX, UMAX));
    let no1 = o1 + DT * (-t1.sin() - 0.05 * o1 + CPL * (t2 - t1).sin() + c1);
    let no2 = o2 + DT * (-t2.sin() - 0.05 * o2 + CPL * (t1 - t2).sin() + c2);
    [wrap(t1 + DT * no1), wrap(t2 + DT * no2), no1, no2]
}
fn cost(s: [f32; 4], g: (f32, f32), u1: f32, u2: f32) -> f32 { wrap(s[0] - g.0).powi(2) + wrap(s[1] - g.1).powi(2) + 0.05 * (s[2] * s[2] + s[3] * s[3]) + 0.01 * (u1 * u1 + u2 * u2) }
fn feat8(s: [f32; 4], g: (f32, f32)) -> [f32; 8] { let (d1, d2) = (s[0] - g.0, s[1] - g.1); [d1.cos(), d1.sin(), s[2], d2.cos(), d2.sin(), s[3], s[0].sin(), s[1].sin()] }

struct Vn { w: [Vec<f32>; 8], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Vn {
    fn eval(&self, s: [f32; 4], g: (f32, f32)) -> f32 { let f = feat8(s, g);
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..8 { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
    fn ustar(&self, s: [f32; 4], g: (f32, f32), grid: &[f32]) -> (f32, f32) { let mut bu = (0.0, 0.0); let mut be = f32::MAX;
        for &u1 in grid { for &u2 in grid { let ns = step(s, u1, u2); let q = (cost(s, g, u1, u2)) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = (u1, u2); } } } bu }
}
// scalar potential Eθ(s,a,t): [8 state, a1, a2, t] → scalar. relu hidden + LINEAR output — no exp anywhere
// (the naive softplus exp→log overflowed f32 under the large gradient targets and NaN'd; relu/linear is the stable net,
// same family that trained fine as the plain velocity field). Targets scaled by κ; integration un-scales.
const KAPPA: f32 = 2.0;
struct Ef { w: [Vec<f32>; 11], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Ef {
    fn e(&self, s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, t: f32) -> f32 { let mut f = [0.0f32; 11]; let ff = feat8(s, g); for c in 0..8 { f[c] = ff[c]; } f[8] = a1; f[9] = a2; f[10] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..11 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } o }
    // EXACT action-gradient: analytic backprop through the relu net (replaces finite differences)
    fn grad_a(&self, s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, t: f32) -> (f32, f32) {
        let mut f = [0.0f32; 11]; let ff = feat8(s, g); for c in 0..8 { f[c] = ff[c]; } f[8] = a1; f[9] = a2; f[10] = t;
        let mut h1 = [0.0f32; H]; let mut m1 = [false; H];
        for j in 0..H { let mut z = self.b1[j]; for c in 0..11 { z += f[c] * self.w[c][j]; } m1[j] = z > 0.0; h1[j] = z.max(0.0); }
        let mut m2 = [false; H];
        for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } m2[j] = z > 0.0; }
        let mut d2 = [0.0f32; H]; for j in 0..H { if m2[j] { d2[j] = self.w3[j]; } }
        let mut d1 = [0.0f32; H]; for k in 0..H { if m1[k] { let mut z = 0.0; for j in 0..H { z += self.w2[k * H + j] * d2[j]; } d1[k] = z; } }
        let (mut g1, mut g2) = (0.0f32, 0.0f32); for j in 0..H { g1 += self.w[8][j] * d1[j]; g2 += self.w[9][j] * d1[j]; }
        (g1, g2) }
    // actuation: integrate a ← a − (κ/K)·∇ₐE  (energy descent along the flow; κ un-scales the trained gradient) from a=0
    fn act(&self, s: [f32; 4], g: (f32, f32), k: usize) -> (f32, f32) { let (mut a1, mut a2) = (0.0f32, 0.0f32);
        for i in 0..k { let t = i as f32 / k as f32; let (g1, g2) = self.grad_a(s, g, a1, a2, t); a1 -= KAPPA * g1 / k as f32; a2 -= KAPPA * g2 / k as f32; }
        (a1.clamp(-UMAX, UMAX), a2.clamp(-UMAX, UMAX)) }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — ENERGY-FIRST flow: v=−∇ₐE. ONE potential ACTUATES (descend) + VERIFIES (score). 2-DOF arm.\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    let sp = |z: Var, ov: &Var| z.exp().add(ov).log();
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[8].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[9]).add(&pv[10]), ov).matmul(&pv[11]).add(&pv[12]), ov) };
    // ---- Stage 1: FVI V → demonstrator u* ----
    let mut p: Vec<Tensor> = (0..8).map(|c| Tensor::from_vec(&ctx, &randn(H, 22 + c as u32, 0.5), &[1, H])).collect();
    p.push(Tensor::zeros(&ctx, &[H])); p.push(Tensor::from_vec(&ctx, &randn(H * H, 40, 1.0 / (H as f32).sqrt()), &[H, H])); p.push(Tensor::zeros(&ctx, &[H]));
    p.push(Tensor::from_vec(&ctx, &randn(H, 41, 1.0 / (H as f32).sqrt()), &[H, 1])); p.push(Tensor::zeros(&ctx, &[1]));
    let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002); let gg = &G5; let ga = gg.len() * gg.len();
    for it in 0..16000 {
        let mut fc: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs]).collect();
        let mut nf: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs * ga]).collect(); let mut cst = vec![0.0f32; bs * ga];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * 3.0, (u(sd, 4) * 2.0 - 1.0) * 3.0];
            let g = ((u(sd, 5) * 2.0 - 1.0) * 1.2, (u(sd, 6) * 2.0 - 1.0) * 1.2); let f = feat8(s, g); for c in 0..8 { fc[c][i] = f[c]; }
            let mut a = 0; for &u1 in gg { for &u2 in gg { let ns = step(s, u1, u2); let nff = feat8(ns, g); for c in 0..8 { nf[c][i * ga + a] = nff[c]; } cst[i * ga + a] = cost(s, g, u1, u2); a += 1; } } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = vnet(&(0..8).map(|c| l(&nf[c], bs * ga)).collect::<Vec<_>>(), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for a in 0..ga { let q = cst[i * ga + a] * DT + GAMMA * et[i * ga + a]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = vnet(&(0..8).map(|c| l(&fc[c], bs)).collect::<Vec<_>>(), &pv, &ov); let d = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let mut wv: [Vec<f32>; 8] = Default::default(); for c in 0..8 { wv[c] = p[c].to_vec().await; }
    let vn = Vn { w: wv, b1: p[8].to_vec().await, w2: p[9].to_vec().await, b2: p[10].to_vec().await, w3: p[11].to_vec().await, b3: p[12].to_vec().await[0] };

    // ---- Stage 2: train the scalar potential Eθ(s,a,t) so v=−∇ₐEθ matches the flow target (single-step gradient matching) ----
    // relu hidden + LINEAR output (no exp → no overflow); targets scaled by 1/κ for headroom
    let enet = |f: &[Var], a1: &Var, a2: &Var, tt: &Var, pv: &[Var], _ov: &Var| {
        let mut pre = pv[11].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre = pre.add(&a1.matmul(&pv[8])).add(&a2.matmul(&pv[9])).add(&tt.matmul(&pv[10]));
        pre.relu().matmul(&pv[12]).add(&pv[13]).relu().matmul(&pv[14]).add(&pv[15]) };
    let mut q: Vec<Tensor> = (0..11).map(|c| Tensor::from_vec(&ctx, &randn(H, 60 + c as u32, 0.4), &[1, H])).collect();
    q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H * H, 80, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(&ctx, &[H]));
    q.push(Tensor::from_vec(&ctx, &randn(H, 81, 1.0 / (H as f32).sqrt()), &[H, 1])); q.push(Tensor::zeros(&ctx, &[1]));
    let mut adamq = Adam::new(&q, 0.0015);
    for it in 0..12000 {
        let mut fc: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs]).collect();
        let (mut at1, mut at2, mut tt, mut g1t, mut g2t) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        for i in 0..bs { let sd = it as u32 * 13 + i as u32;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * 3.0, (u(sd, 4) * 2.0 - 1.0) * 3.0];
            let g = ((u(sd, 5) * 2.0 - 1.0) * 1.2, (u(sd, 6) * 2.0 - 1.0) * 1.2); let us = vn.ustar(s, g, &G9);
            let a01 = (u(sd, 7) * 2.0 - 1.0) * 3.0; let a02 = (u(sd, 8) * 2.0 - 1.0) * 3.0; let t = u(sd, 9) * 0.9;   // cap t: the target field steepens ∝1/(1−t); the K-grid never reaches t=1
            let ff = feat8(s, g); for c in 0..8 { fc[c][i] = ff[c]; }
            at1[i] = (1.0 - t) * a01 + t * us.0; at2[i] = (1.0 - t) * a02 + t * us.1; tt[i] = t;
            g1t[i] = -(us.0 - a01) / KAPPA; g2t[i] = -(us.1 - a02) / KAPPA;   // want −κ∇ₐE = (u*−a0) ⇒ ∇ₐE target = −(u*−a0)/κ
        }
        let ov = Var::leaf(one.clone()); let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect();
        let fv: Vec<Var> = (0..8).map(|c| Var::leaf(Tensor::from_vec(&ctx, &fc[c], &[bs, 1]))).collect();
        let a1v = Var::leaf(Tensor::from_vec(&ctx, &at1, &[bs, 1])); let a2v = Var::leaf(Tensor::from_vec(&ctx, &at2, &[bs, 1])); let tv = Var::leaf(Tensor::from_vec(&ctx, &tt, &[bs, 1]));
        let e = enet(&fv, &a1v, &a2v, &tv, &pv, &ov);
        let gr = grad(&e.sum_all(), &[a1v.clone(), a2v.clone()], None);              // ∇ₐE (differentiable in weights)
        let d1 = gr[0].sub(&Var::leaf(Tensor::from_vec(&ctx, &g1t, &[bs, 1]))); let d2 = gr[1].sub(&Var::leaf(Tensor::from_vec(&ctx, &g2t, &[bs, 1])));
        let loss = d1.mul(&d1).add(&d2.mul(&d2)).mean_all(); loss.backward();                  // backprop THROUGH ∇ₐE into E-weights (2nd order, single-step)
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamq.step(&mut q, &g);
    }
    let mut ew: [Vec<f32>; 11] = Default::default(); for c in 0..11 { ew[c] = q[c].to_vec().await; }
    let ef = Ef { w: ew, b1: q[11].to_vec().await, w2: q[12].to_vec().await, b2: q[13].to_vec().await, w3: q[14].to_vec().await, b3: q[15].to_vec().await[0] };

    // ---- Stage 3: read the ONE potential two ways ----
    let nep = 40usize; let n = nep * TG.len();
    let mut inits: Vec<[f32; 4]> = vec![]; let mut goals: Vec<(f32, f32)> = vec![];
    for (gi, &g) in TG.iter().enumerate() { for e in 0..nep { let sd = (gi * nep + e) as u32; inits.push([(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, 0.0, 0.0]); goals.push(g); } }
    let reached = |s: [f32; 4], g: (f32, f32)| wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && s[2].abs() < 0.7 && s[3].abs() < 0.7;
    let run_disc = |grid: &[f32]| -> f32 { let mut r = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..260 { let uu = vn.ustar(s, g, grid); s = step(s, uu.0, uu.1); } if reached(s, g) { r += 1; } } r as f32 / n as f32 * 100.0 };
    let rf = run_disc(&G9);
    // (a) ACTUATION: integrate v=−∇ₐE
    let run_act = |k: usize| -> f32 { let mut r = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..260 { let (u1, u2) = ef.act(s, g, k); s = step(s, u1, u2); } if reached(s, g) { r += 1; } } r as f32 / n as f32 * 100.0 };
    // (b) VERIFY from the SAME potential at t=1: is E(s,u*,1) < E(s,bad,1)?
    let (mut vg, mut vt) = (0, 0); for k in 0..3000 { let s = [(u(k as u32, 41) * 2.0 - 1.0) * PI, (u(k as u32, 42) * 2.0 - 1.0) * PI, (u(k as u32, 43) * 2.0 - 1.0) * 3.0, (u(k as u32, 44) * 2.0 - 1.0) * 3.0];
        let g = ((u(k as u32, 45) * 2.0 - 1.0) * 1.2, (u(k as u32, 46) * 2.0 - 1.0) * 1.2); let us = vn.ustar(s, g, &G5); let bad = ((u(k as u32, 47) * 2.0 - 1.0) * UMAX, (u(k as u32, 48) * 2.0 - 1.0) * UMAX);
        vt += 1; if ef.e(s, g, us.0, us.1, 1.0) < ef.e(s, g, bad.0, bad.1, 1.0) { vg += 1; } }
    let (mut dm, mut de) = (0.0f32, 0.0f32); for i in 0..n { let a = ef.act(inits[i], goals[i], 2); let us = vn.ustar(inits[i], goals[i], &G5); dm += (a.0 * a.0 + a.1 * a.1).sqrt(); de += ((a.0 - us.0).powi(2) + (a.1 - us.1).powi(2)).sqrt(); }

    println!("  ONE scalar potential Eθ(s,a,t); v=−∇ₐEθ. DIAG: K=2 descend-action mean|u|={:.2}, mean|u−u*|={:.2}\n", dm / n as f32, de / n as f32);
    println!("     reading                                        result");
    println!("     discrete argmin (demonstrator), fine 9×9       {:>4.0}% reach   (81 evals/decision)", rf);
    for k in [1usize, 2, 4, 8] { println!("     ACTUATE — descend v=−∇ₐE, K={:<2}                  {:>4.0}% reach   ({} grad evals)", k, run_act(k), k); }
    println!("     VERIFY  — same Eθ(·,1) ranks good action < bad  {:>4.1}%", vg as f32 / vt as f32 * 100.0);
    println!("\n  If ACTUATE reaches AND VERIFY is high from the SAME potential, EFA has a literally energy-first unified object:");
    println!("  the action is energy descent (v=−∇ₐE) and validity is the same energy's value — flow-matching, done energy-first.");
    println!("  HONEST: distilled demonstrator, small body, FD action-gradient at eval (exact via autograd in training).");
}
