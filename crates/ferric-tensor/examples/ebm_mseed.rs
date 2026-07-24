//! EFA energy-first #61 — MULTI-SEED robustness of the 3-DOF headline (flow 100% @ K=1 vs teacher 57% @ 152 evals).
//!
//! The 3-DOF resolution (ebm_arm3c) was one seed. This reruns the full pipeline (FVI value HV=128 → two-stage teacher →
//! flow distill → eval) at 3 independent seeds, varying init, sampling, AND eval-episode seeds, and reports per-seed rows
//! + min/mean/max. The 2-DOF flow headline already has independent replications (flow2, hflow, lsweep λ≤0.1 all trained
//! fresh and hit 100%); the 3-DOF number is the one that needed this. HONEST: 3 seeds is robustness-lite, not a
//! statistical study; recipe slightly lightened (18k FVI, 6k distill) to fit three runs.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_mseed --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const HV: usize = 128; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const G5: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const TG: [(f32, f32, f32); 4] = [(0.8, -0.6, 0.5), (-0.7, 0.5, -0.6), (0.5, 0.9, -0.4), (-0.5, -0.8, 0.7)];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(s: [f32; 6], uu: [f32; 3]) -> [f32; 6] {
    let (t1, t2, t3, o1, o2, o3) = (s[0], s[1], s[2], s[3], s[4], s[5]);
    let c: Vec<f32> = uu.iter().map(|x| x.clamp(-UMAX, UMAX)).collect();
    let no1 = o1 + DT * (-t1.sin() - 0.05 * o1 + CPL * (t2 - t1).sin() + c[0]);
    let no2 = o2 + DT * (-t2.sin() - 0.05 * o2 + CPL * (t1 - t2).sin() + CPL * (t3 - t2).sin() + c[1]);
    let no3 = o3 + DT * (-t3.sin() - 0.05 * o3 + CPL * (t2 - t3).sin() + c[2]);
    [wrap(t1 + DT * no1), wrap(t2 + DT * no2), wrap(t3 + DT * no3), no1, no2, no3]
}
fn cost(s: [f32; 6], g: (f32, f32, f32), uu: [f32; 3]) -> f32 {
    wrap(s[0] - g.0).powi(2) + wrap(s[1] - g.1).powi(2) + wrap(s[2] - g.2).powi(2) + 0.05 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]) + 0.01 * (uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2])
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
    fn ustar(&self, s: [f32; 6], g: (f32, f32, f32)) -> [f32; 3] { let mut bu = [0.0f32; 3]; let mut be = f32::MAX;
        for &u1 in &G5 { for &u2 in &G5 { for &u3 in &G5 { let uu = [u1, u2, u3]; let ns = step(s, uu); let q = cost(s, g, uu) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = uu; } } } }
        let base = bu;
        for &d1 in &[-0.75f32, 0.0, 0.75] { for &d2 in &[-0.75f32, 0.0, 0.75] { for &d3 in &[-0.75f32, 0.0, 0.75] {
            let uu = [(base[0] + d1).clamp(-UMAX, UMAX), (base[1] + d2).clamp(-UMAX, UMAX), (base[2] + d3).clamp(-UMAX, UMAX)];
            let ns = step(s, uu); let q = cost(s, g, uu) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = uu; } } } }
        bu }
}
struct Fl { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: [f32; 3] }
impl Fl {
    fn vel(&self, s: [f32; 6], g: (f32, f32, f32), a: [f32; 3], t: f32) -> [f32; 3] { let mut f = [0.0f32; 16]; let ff = feat(s, g); for c in 0..NF { f[c] = ff[c]; } f[12] = a[0]; f[13] = a[1]; f[14] = a[2]; f[15] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..16 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = [self.b3[0], self.b3[1], self.b3[2]]; for j in 0..H { for c in 0..3 { o[c] += h2[j] * self.w3[j * 3 + c]; } } o }
    fn act(&self, s: [f32; 6], g: (f32, f32, f32), k: usize) -> [f32; 3] { let mut a = [0.0f32; 3];
        for i in 0..k { let t = i as f32 / k as f32; let v = self.vel(s, g, a, t); for c in 0..3 { a[c] += v[c] / k as f32; } }
        [a[0].clamp(-UMAX, UMAX), a[1].clamp(-UMAX, UMAX), a[2].clamp(-UMAX, UMAX)] }
}

async fn run_seed(ctx: &Arc<ferric_core::Context>, soff: u32) -> (f32, f32, f32) {
    let one = Tensor::from_vec(ctx, &[1.0], &[1]); let bs = 160usize;
    let sp = |z: Var, ov: &Var| z.exp().add(ov).log();
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[NF].clone(); for c in 0..NF { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[NF + 1]).add(&pv[NF + 2]), ov).matmul(&pv[NF + 3]).add(&pv[NF + 4]), ov) };
    let mut p: Vec<Tensor> = (0..NF).map(|c| Tensor::from_vec(ctx, &randn(HV, 22 + soff + c as u32, 0.45), &[1, HV])).collect();
    p.push(Tensor::zeros(ctx, &[HV])); p.push(Tensor::from_vec(ctx, &randn(HV * HV, 40 + soff, 1.0 / (HV as f32).sqrt()), &[HV, HV])); p.push(Tensor::zeros(ctx, &[HV]));
    p.push(Tensor::from_vec(ctx, &randn(HV, 41 + soff, 1.0 / (HV as f32).sqrt()), &[HV, 1])); p.push(Tensor::zeros(ctx, &[1]));
    let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002); let ga = 125usize;
    for it in 0..18000 {
        let mut fc: Vec<Vec<f32>> = (0..NF).map(|_| vec![0.0f32; bs]).collect();
        let mut nf: Vec<Vec<f32>> = (0..NF).map(|_| vec![0.0f32; bs * ga]).collect(); let mut cst = vec![0.0f32; bs * ga];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32 + soff * 1000;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * PI, (u(sd, 4) * 2.0 - 1.0) * 3.0, (u(sd, 5) * 2.0 - 1.0) * 3.0, (u(sd, 6) * 2.0 - 1.0) * 3.0];
            let g = ((u(sd, 10) * 2.0 - 1.0) * 1.0, (u(sd, 11) * 2.0 - 1.0) * 1.0, (u(sd, 12) * 2.0 - 1.0) * 1.0);
            let f = feat(s, g); for c in 0..NF { fc[c][i] = f[c]; }
            let mut a = 0; for &u1 in &G5 { for &u2 in &G5 { for &u3 in &G5 { let uu = [u1, u2, u3]; let ns = step(s, uu); let nff = feat(ns, g); for c in 0..NF { nf[c][i * ga + a] = nff[c]; } cst[i * ga + a] = cost(s, g, uu); a += 1; } } } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = vnet(&(0..NF).map(|c| l(&nf[c], bs * ga)).collect::<Vec<_>>(), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for a in 0..ga { let q = cst[i * ga + a] * DT + GAMMA * et[i * ga + a]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = vnet(&(0..NF).map(|c| l(&fc[c], bs)).collect::<Vec<_>>(), &pv, &ov); let d = e.sub(&Var::leaf(Tensor::from_vec(ctx, &target, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let mut vw: Vec<Vec<f32>> = Vec::new(); for c in 0..NF { vw.push(p[c].to_vec().await); }
    let vn = Vn { w: vw, b1: p[NF].to_vec().await, w2: p[NF + 1].to_vec().await, b2: p[NF + 2].to_vec().await, w3: p[NF + 3].to_vec().await, b3: p[NF + 4].to_vec().await[0] };

    let fnet = |f: &[Var], pv: &[Var]| { let mut pre = pv[16].clone(); for c in 0..16 { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[17]).add(&pv[18]).relu().matmul(&pv[19]).add(&pv[20]) };
    let mut q: Vec<Tensor> = (0..16).map(|c| Tensor::from_vec(ctx, &randn(H, 60 + soff + c as u32, 0.4), &[1, H])).collect();
    q.push(Tensor::zeros(ctx, &[H])); q.push(Tensor::from_vec(ctx, &randn(H * H, 80 + soff, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(ctx, &[H]));
    q.push(Tensor::from_vec(ctx, &randn(H * 3, 81 + soff, 1.0 / (H as f32).sqrt()), &[H, 3])); q.push(Tensor::zeros(ctx, &[3]));
    let mut adamq = Adam::new(&q, 0.002); let fbs = 128usize;
    for it in 0..6000 {
        let mut fc: Vec<Vec<f32>> = (0..16).map(|_| vec![0.0f32; fbs]).collect(); let mut tb = vec![0.0f32; fbs * 3];
        for i in 0..fbs { let sd = it as u32 * 13 + i as u32 + soff * 1000;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * PI, (u(sd, 4) * 2.0 - 1.0) * 3.0, (u(sd, 5) * 2.0 - 1.0) * 3.0, (u(sd, 6) * 2.0 - 1.0) * 3.0];
            let g = ((u(sd, 10) * 2.0 - 1.0) * 1.0, (u(sd, 11) * 2.0 - 1.0) * 1.0, (u(sd, 12) * 2.0 - 1.0) * 1.0);
            let us = vn.ustar(s, g); let t = u(sd, 9) * 0.9;
            let a0 = [(u(sd, 7) * 2.0 - 1.0) * 3.0, (u(sd, 8) * 2.0 - 1.0) * 3.0, (u(sd, 14) * 2.0 - 1.0) * 3.0];
            let ff = feat(s, g); for c in 0..NF { fc[c][i] = ff[c]; }
            for c in 0..3 { fc[12 + c][i] = (1.0 - t) * a0[c] + t * us[c]; tb[i * 3 + c] = us[c] - a0[c]; } fc[15][i] = t; }
        let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect();
        let fv: Vec<Var> = (0..16).map(|c| Var::leaf(Tensor::from_vec(ctx, &fc[c], &[fbs, 1]))).collect();
        let v = fnet(&fv, &pv); let d = v.sub(&Var::leaf(Tensor::from_vec(ctx, &tb, &[fbs, 3]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamq.step(&mut q, &g);
    }
    let fb3 = q[20].to_vec().await;
    let mut fw: Vec<Vec<f32>> = Vec::new(); for c in 0..16 { fw.push(q[c].to_vec().await); }
    let fl = Fl { w: fw, b1: q[16].to_vec().await, w2: q[17].to_vec().await, b2: q[18].to_vec().await, w3: q[19].to_vec().await, b3: [fb3[0], fb3[1], fb3[2]] };

    let nep = 30usize; let n = nep * TG.len();
    let mut inits: Vec<[f32; 6]> = vec![]; let mut goals: Vec<(f32, f32, f32)> = vec![];
    for (gi, &g) in TG.iter().enumerate() { for e in 0..nep { let sd = (gi * nep + e) as u32 + soff * 7777;
        inits.push([(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, (u(900 + sd, 9) * 2.0 - 1.0) * PI, 0.0, 0.0, 0.0]); goals.push(g); } }
    let reached = |s: [f32; 6], g: (f32, f32, f32)| wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && wrap(s[2] - g.2).abs() < 0.35 && s[3].abs() < 0.7 && s[4].abs() < 0.7 && s[5].abs() < 0.7;
    let mut rd = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..300 { let uu = vn.ustar(s, g); s = step(s, uu); } if reached(s, g) { rd += 1; } }
    let mut r1 = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..300 { let a = fl.act(s, g, 1); s = step(s, a); } if reached(s, g) { r1 += 1; } }
    let mut r2 = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..300 { let a = fl.act(s, g, 2); s = step(s, a); } if reached(s, g) { r2 += 1; } }
    (rd as f32 / n as f32 * 100.0, r1 as f32 / n as f32 * 100.0, r2 as f32 / n as f32 * 100.0)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — MULTI-SEED robustness of the 3-DOF headline (teacher @152 evals vs flow @K fwd passes)\n");
    println!("     seed   teacher(152ev)   flow K=1   flow K=2");
    let (mut td, mut f1, mut f2) = (vec![], vec![], vec![]);
    for (si, &soff) in [0u32, 31, 62].iter().enumerate() {
        let (a, b, c) = run_seed(&ctx, soff).await;
        println!("     {:>4}       {:>4.0}%          {:>4.0}%      {:>4.0}%", si, a, b, c);
        td.push(a); f1.push(b); f2.push(c);
    }
    let stats = |v: &Vec<f32>| (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().sum::<f32>() / v.len() as f32, v.iter().cloned().fold(f32::MIN, f32::max));
    let (tmin, tmean, tmax) = stats(&td); let (amin, amean, amax) = stats(&f1); let (bmin, bmean, bmax) = stats(&f2);
    println!("\n     teacher: min/mean/max = {:.0}/{:.0}/{:.0}%   flow K=1: {:.0}/{:.0}/{:.0}%   flow K=2: {:.0}/{:.0}/{:.0}%", tmin, tmean, tmax, amin, amean, amax, bmin, bmean, bmax);
    println!("\n  HONEST: 3 seeds = robustness-lite; recipe lightened (18k FVI, 6k distill) vs the flagship run to fit three trainings.");
    println!("  2-DOF flow headline already replicated across flow2/hflow/lsweep independent trainings (all 100%).");
}
