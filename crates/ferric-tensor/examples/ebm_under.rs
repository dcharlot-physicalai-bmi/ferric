//! EFA energy-first #62 — UNDERACTUATED 3-joint chain: 2 torques, passive third joint, goals REACHABLE by construction.
//!
//! The honest underactuated variant: joint 3 has NO torque — it can only be driven through the coupling. Arbitrary joint
//! triples are not reachable, so goals are built to be: (g1,g2) free, and g3 SOLVED from the passive joint's equilibrium
//! condition sin(g3) = CPL·sin(g2−g3) (fixed-point solve). The controller must steer the actuated joints so the coupling
//! parks the passive joint at its equilibrium — genuine underactuated control. Pipeline = the proven corrected stack
//! (FVI value HV=128 → two-stage teacher over the 2-D torque grid, 25+9=34 evals/decision → flow distill, 2-output).
//! Reported: overall reach (ALL THREE joints incl. the passive) + the passive joint's final |θ3−g3| specifically.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_under --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const HV: usize = 128; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const G5: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const TGP: [(f32, f32); 4] = [(0.8, -0.6), (-0.7, 0.5), (0.5, 0.9), (-0.5, -0.8)];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
// UNDERACTUATED: torque on joints 1,2 only; joint 3 passive (driven by coupling alone)
fn step(s: [f32; 6], u1: f32, u2: f32) -> [f32; 6] {
    let (t1, t2, t3, o1, o2, o3) = (s[0], s[1], s[2], s[3], s[4], s[5]);
    let (c1, c2) = (u1.clamp(-UMAX, UMAX), u2.clamp(-UMAX, UMAX));
    let no1 = o1 + DT * (-t1.sin() - 0.05 * o1 + CPL * (t2 - t1).sin() + c1);
    let no2 = o2 + DT * (-t2.sin() - 0.05 * o2 + CPL * (t1 - t2).sin() + CPL * (t3 - t2).sin() + c2);
    let no3 = o3 + DT * (-t3.sin() - 0.05 * o3 + CPL * (t2 - t3).sin());
    [wrap(t1 + DT * no1), wrap(t2 + DT * no2), wrap(t3 + DT * no3), no1, no2, no3]
}
// reachable goal: g3 solved from the passive equilibrium sin(g3) = CPL·sin(g2−g3)
fn g3_of(g2: f32) -> f32 { let mut g3 = 0.0f32; for _ in 0..60 { g3 = (CPL * (g2 - g3).sin()).asin(); } g3 }
fn cost(s: [f32; 6], g: (f32, f32, f32), u1: f32, u2: f32) -> f32 {
    wrap(s[0] - g.0).powi(2) + wrap(s[1] - g.1).powi(2) + wrap(s[2] - g.2).powi(2) + 0.05 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]) + 0.01 * (u1 * u1 + u2 * u2)
}
const NF: usize = 12;
fn feat(s: [f32; 6], g: (f32, f32, f32)) -> [f32; NF] { let (d1, d2, d3) = (s[0] - g.0, s[1] - g.1, s[2] - g.2);
    [d1.cos(), d1.sin(), s[3], d2.cos(), d2.sin(), s[4], d3.cos(), d3.sin(), s[5], s[0].sin(), s[1].sin(), s[2].sin()] }

struct Vn { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Vn {
    fn eval(&self, s: [f32; 6], g: (f32, f32, f32)) -> f32 { let f = feat(s, g);
        let mut h1 = [0.0f32; HV]; for j in 0..HV { let mut z = self.b1[j]; for c in 0..NF { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; HV]; for j in 0..HV { let mut z = self.b2[j]; for k in 0..HV { z += h1[k] * self.w2[k * HV + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..HV { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
    fn ustar(&self, s: [f32; 6], g: (f32, f32, f32)) -> (f32, f32) { let mut bu = (0.0f32, 0.0f32); let mut be = f32::MAX;
        for &u1 in &G5 { for &u2 in &G5 { let ns = step(s, u1, u2); let q = cost(s, g, u1, u2) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = (u1, u2); } } }
        let base = bu;
        for &d1 in &[-0.75f32, 0.0, 0.75] { for &d2 in &[-0.75f32, 0.0, 0.75] {
            let uu = ((base.0 + d1).clamp(-UMAX, UMAX), (base.1 + d2).clamp(-UMAX, UMAX));
            let ns = step(s, uu.0, uu.1); let q = cost(s, g, uu.0, uu.1) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = uu; } } }
        bu }
}
struct Fl { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: [f32; 2] }
impl Fl {
    fn vel(&self, s: [f32; 6], g: (f32, f32, f32), a1: f32, a2: f32, t: f32) -> (f32, f32) { let mut f = [0.0f32; 15]; let ff = feat(s, g); for c in 0..NF { f[c] = ff[c]; } f[12] = a1; f[13] = a2; f[14] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..15 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let (mut o1, mut o2) = (self.b3[0], self.b3[1]); for j in 0..H { o1 += h2[j] * self.w3[j * 2]; o2 += h2[j] * self.w3[j * 2 + 1]; } (o1, o2) }
    fn act(&self, s: [f32; 6], g: (f32, f32, f32), k: usize) -> (f32, f32) { let (mut a1, mut a2) = (0.0f32, 0.0f32);
        for i in 0..k { let t = i as f32 / k as f32; let (v1, v2) = self.vel(s, g, a1, a2, t); a1 += v1 / k as f32; a2 += v2 / k as f32; }
        (a1.clamp(-UMAX, UMAX), a2.clamp(-UMAX, UMAX)) }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — UNDERACTUATED 3-joint chain (2 torques, passive 3rd joint), goals reachable by construction\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 192usize;
    let sp = |z: Var, ov: &Var| z.exp().add(ov).log();
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[NF].clone(); for c in 0..NF { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[NF + 1]).add(&pv[NF + 2]), ov).matmul(&pv[NF + 3]).add(&pv[NF + 4]), ov) };
    // ---- FVI over the 2-D torque grid (25 actions), goals with g3 solved ----
    let mut p: Vec<Tensor> = (0..NF).map(|c| Tensor::from_vec(&ctx, &randn(HV, 22 + c as u32, 0.45), &[1, HV])).collect();
    p.push(Tensor::zeros(&ctx, &[HV])); p.push(Tensor::from_vec(&ctx, &randn(HV * HV, 40, 1.0 / (HV as f32).sqrt()), &[HV, HV])); p.push(Tensor::zeros(&ctx, &[HV]));
    p.push(Tensor::from_vec(&ctx, &randn(HV, 41, 1.0 / (HV as f32).sqrt()), &[HV, 1])); p.push(Tensor::zeros(&ctx, &[1]));
    let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002); let ga = 25usize;
    for it in 0..18000 {
        let mut fc: Vec<Vec<f32>> = (0..NF).map(|_| vec![0.0f32; bs]).collect();
        let mut nf: Vec<Vec<f32>> = (0..NF).map(|_| vec![0.0f32; bs * ga]).collect(); let mut cst = vec![0.0f32; bs * ga];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * PI, (u(sd, 4) * 2.0 - 1.0) * 3.0, (u(sd, 5) * 2.0 - 1.0) * 3.0, (u(sd, 6) * 2.0 - 1.0) * 3.0];
            let (g1, g2) = ((u(sd, 10) * 2.0 - 1.0) * 1.0, (u(sd, 11) * 2.0 - 1.0) * 1.0); let g = (g1, g2, g3_of(g2));
            let f = feat(s, g); for c in 0..NF { fc[c][i] = f[c]; }
            let mut a = 0; for &u1 in &G5 { for &u2 in &G5 { let ns = step(s, u1, u2); let nff = feat(ns, g); for c in 0..NF { nf[c][i * ga + a] = nff[c]; } cst[i * ga + a] = cost(s, g, u1, u2); a += 1; } } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = vnet(&(0..NF).map(|c| l(&nf[c], bs * ga)).collect::<Vec<_>>(), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for a in 0..ga { let q = cst[i * ga + a] * DT + GAMMA * et[i * ga + a]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = vnet(&(0..NF).map(|c| l(&fc[c], bs)).collect::<Vec<_>>(), &pv, &ov); let d = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let mut vw: Vec<Vec<f32>> = Vec::new(); for c in 0..NF { vw.push(p[c].to_vec().await); }
    let vn = Vn { w: vw, b1: p[NF].to_vec().await, w2: p[NF + 1].to_vec().await, b2: p[NF + 2].to_vec().await, w3: p[NF + 3].to_vec().await, b3: p[NF + 4].to_vec().await[0] };

    // ---- flow distill (2-output) ----
    let fnet = |f: &[Var], pv: &[Var]| { let mut pre = pv[15].clone(); for c in 0..15 { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[16]).add(&pv[17]).relu().matmul(&pv[18]).add(&pv[19]) };
    let mut q: Vec<Tensor> = (0..15).map(|c| Tensor::from_vec(&ctx, &randn(H, 60 + c as u32, 0.4), &[1, H])).collect();
    q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H * H, 80, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(&ctx, &[H]));
    q.push(Tensor::from_vec(&ctx, &randn(H * 2, 81, 1.0 / (H as f32).sqrt()), &[H, 2])); q.push(Tensor::zeros(&ctx, &[2]));
    let mut adamq = Adam::new(&q, 0.002); let fbs = 128usize;
    for it in 0..6000 {
        let mut fc: Vec<Vec<f32>> = (0..15).map(|_| vec![0.0f32; fbs]).collect(); let mut tb = vec![0.0f32; fbs * 2];
        for i in 0..fbs { let sd = it as u32 * 13 + i as u32;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * PI, (u(sd, 4) * 2.0 - 1.0) * 3.0, (u(sd, 5) * 2.0 - 1.0) * 3.0, (u(sd, 6) * 2.0 - 1.0) * 3.0];
            let (g1, g2) = ((u(sd, 10) * 2.0 - 1.0) * 1.0, (u(sd, 11) * 2.0 - 1.0) * 1.0); let g = (g1, g2, g3_of(g2));
            let us = vn.ustar(s, g); let t = u(sd, 9) * 0.9;
            let a0 = ((u(sd, 7) * 2.0 - 1.0) * 3.0, (u(sd, 8) * 2.0 - 1.0) * 3.0);
            let ff = feat(s, g); for c in 0..NF { fc[c][i] = ff[c]; }
            fc[12][i] = (1.0 - t) * a0.0 + t * us.0; fc[13][i] = (1.0 - t) * a0.1 + t * us.1; fc[14][i] = t;
            tb[i * 2] = us.0 - a0.0; tb[i * 2 + 1] = us.1 - a0.1; }
        let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect();
        let fv: Vec<Var> = (0..15).map(|c| Var::leaf(Tensor::from_vec(&ctx, &fc[c], &[fbs, 1]))).collect();
        let v = fnet(&fv, &pv); let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[fbs, 2]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamq.step(&mut q, &g);
    }
    let fb2 = q[19].to_vec().await;
    let mut fw: Vec<Vec<f32>> = Vec::new(); for c in 0..15 { fw.push(q[c].to_vec().await); }
    let fl = Fl { w: fw, b1: q[15].to_vec().await, w2: q[16].to_vec().await, b2: q[17].to_vec().await, w3: q[18].to_vec().await, b3: [fb2[0], fb2[1]] };

    // ---- eval: teacher vs flow; overall reach + passive-joint tracking ----
    let nep = 30usize; let n = nep * TGP.len();
    let mut inits: Vec<[f32; 6]> = vec![]; let mut goals: Vec<(f32, f32, f32)> = vec![];
    for (gi, &(g1, g2)) in TGP.iter().enumerate() { let g = (g1, g2, g3_of(g2));
        for e in 0..nep { let sd = (gi * nep + e) as u32;
            inits.push([(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, (u(900 + sd, 9) * 2.0 - 1.0) * PI, 0.0, 0.0, 0.0]); goals.push(g); } }
    let reached = |s: [f32; 6], g: (f32, f32, f32)| wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && wrap(s[2] - g.2).abs() < 0.35 && s[3].abs() < 0.7 && s[4].abs() < 0.7 && s[5].abs() < 0.7;
    println!("  goals: (g1,g2) free, g3 = passive equilibrium of the coupling: {}", TGP.iter().map(|&(a, b)| format!("({:.1},{:.1}→{:.2})", a, b, g3_of(b))).collect::<Vec<_>>().join(" "));
    println!("\n     controller                        reach     passive |θ3−g3| (mean)   evals/decision");
    let (mut rd, mut p3) = (0, 0.0f32); for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..300 { let uu = vn.ustar(s, g); s = step(s, uu.0, uu.1); } if reached(s, g) { rd += 1; } p3 += wrap(s[2] - g.2).abs(); }
    println!("     discrete two-stage argmin          {:>4.0}%        {:.3} rad                34", rd as f32 / n as f32 * 100.0, p3 / n as f32);
    for k in [1usize, 2, 4] { let (mut rr, mut pp) = (0, 0.0f32); for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..300 { let a = fl.act(s, g, k); s = step(s, a.0, a.1); } if reached(s, g) { rr += 1; } pp += wrap(s[2] - g.2).abs(); }
        println!("     flow-matching, K={:<2} fwd passes    {:>4.0}%        {:.3} rad                {:>2}", k, rr as f32 / n as f32 * 100.0, pp / n as f32, k); }
    println!("\n  The passive joint has NO torque — it is parked at its equilibrium purely through the coupling. Reach requires all");
    println!("  THREE joints (incl. passive) within 0.35 rad / 0.7 rad·s. HONEST: reachable-by-construction goals; one seed.");
}
