//! EFA energy-first #54 — CORRECTED actuation: a FLOW-MATCHING policy on the 2-DOF arm where IBC/descent got 0%.
//!
//! The 2026 frontier check (docs/FRONTIER-CHECK-2026.md) found my multi-DOF failures were re-deriving Implicit Behavior
//! Cloning (IBC iterative energy descent over actions), which is known-failing on manipulation (0.21 vs 0.88 Diffusion
//! Policy). The field's fix = flow-matching / one-or-two-step policies (SSCP 2506.21427, MIP 2512.01809): NO iterative
//! energy descent, NO BPTT — just regress a velocity field vθ(s,a,t) that flows noise→action, then integrate 1–2 steps
//! of FORWARD PASSES. (Per Energy Matching 2504.10612 the velocity field IS −∇ of a scalar potential, so this stays
//! energy-first; it's just the training recipe that works.) Test on the SAME coupled 2-DOF arm where descent got 0%: if
//! flow-matching reaches ~discrete at K=1–2 forward passes, the recipe was the cripple, not the body.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_flow2 --release`
use ferric_tensor::{Adam, Tensor, Var};
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

// flow velocity field vθ(s, a, t): inputs [8 state, a1, a2, t] (11) → [v1, v2].  relu hidden, LINEAR 2-D output.
struct Fn2 { w: [Vec<f32>; 11], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: [f32; 2] }
impl Fn2 {
    fn vel(&self, s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, t: f32) -> (f32, f32) {
        let mut f = [0.0f32; 11]; let ff = feat8(s, g); for c in 0..8 { f[c] = ff[c]; } f[8] = a1; f[9] = a2; f[10] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..11 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let (mut o1, mut o2) = (self.b3[0], self.b3[1]); for j in 0..H { o1 += h2[j] * self.w3[j * 2]; o2 += h2[j] * self.w3[j * 2 + 1]; } (o1, o2) }
    // integrate the flow from a=0 for K Euler steps → action
    fn act(&self, s: [f32; 4], g: (f32, f32), k: usize) -> (f32, f32) {
        let (mut a1, mut a2) = (0.0f32, 0.0f32);
        for i in 0..k { let t = i as f32 / k as f32; let (v1, v2) = self.vel(s, g, a1, a2, t); a1 += v1 / k as f32; a2 += v2 / k as f32; }
        (a1.clamp(-UMAX, UMAX), a2.clamp(-UMAX, UMAX)) }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — CORRECTED actuation: FLOW-MATCHING policy on the 2-DOF arm (where IBC/descent got 0%)\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    let sp_of = |z: Var, ov: &Var| z.exp().add(ov).log();
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[8].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); }
        sp_of(sp_of(sp_of(pre, ov).matmul(&pv[9]).add(&pv[10]), ov).matmul(&pv[11]).add(&pv[12]), ov) };
    // ---- Stage 1: FVI V (discrete demonstrator) ----
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
        let nfv: Vec<Var> = (0..8).map(|c| l(&nf[c], bs * ga)).collect();
        let et = vnet(&nfv, &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for a in 0..ga { let q = cst[i * ga + a] * DT + GAMMA * et[i * ga + a]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let fv: Vec<Var> = (0..8).map(|c| l(&fc[c], bs)).collect();
        let e = vnet(&fv, &pv, &ov); let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let mut wv: [Vec<f32>; 8] = Default::default(); for c in 0..8 { wv[c] = p[c].to_vec().await; }
    let vn = Vn { w: wv, b1: p[8].to_vec().await, w2: p[9].to_vec().await, b2: p[10].to_vec().await, w3: p[11].to_vec().await, b3: p[12].to_vec().await[0] };

    // ---- Stage 2: CONDITIONAL FLOW MATCHING — regress velocity field vθ(s, a_t, t) → (u* − a0), a_t=(1−t)a0+t·u* ----
    // 11 rank-1 input weights (8 state + a1 + a2 + t), b1, W2, b2, W3[H,2], b3[2]. relu hidden, linear output.
    let flownet = |f: &[Var], pv: &[Var], _ov: &Var| { let mut pre = pv[11].clone(); for c in 0..11 { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[12]).add(&pv[13]).relu().matmul(&pv[14]).add(&pv[15]) };
    let mut q: Vec<Tensor> = (0..11).map(|c| Tensor::from_vec(&ctx, &randn(H, 60 + c as u32, 0.4), &[1, H])).collect();
    q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H * H, 80, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(&ctx, &[H]));
    q.push(Tensor::from_vec(&ctx, &randn(H * 2, 81, 1.0 / (H as f32).sqrt()), &[H, 2])); q.push(Tensor::zeros(&ctx, &[2]));
    let mut adamq = Adam::new(&q, 0.002);
    for it in 0..12000 {
        let mut fc: Vec<Vec<f32>> = (0..11).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs * 2];
        for i in 0..bs { let sd = it as u32 * 13 + i as u32;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * 3.0, (u(sd, 4) * 2.0 - 1.0) * 3.0];
            let g = ((u(sd, 5) * 2.0 - 1.0) * 1.2, (u(sd, 6) * 2.0 - 1.0) * 1.2); let us = vn.ustar(s, g, &G9);
            let a01 = (u(sd, 7) * 2.0 - 1.0) * 3.0; let a02 = (u(sd, 8) * 2.0 - 1.0) * 3.0; let t = u(sd, 9);   // a0 ~ noise, t ~ U[0,1]
            let at1 = (1.0 - t) * a01 + t * us.0; let at2 = (1.0 - t) * a02 + t * us.1;                        // flow point a_t
            let ff = feat8(s, g); for c in 0..8 { fc[c][i] = ff[c]; } fc[8][i] = at1; fc[9][i] = at2; fc[10][i] = t;
            tb[i * 2] = us.0 - a01; tb[i * 2 + 1] = us.1 - a02;                                                // target velocity = u* − a0
        }
        let ov = Var::leaf(one.clone()); let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect();
        let fv: Vec<Var> = (0..11).map(|c| Var::leaf(Tensor::from_vec(&ctx, &fc[c], &[bs, 1]))).collect();
        let v = flownet(&fv, &pv, &ov); let diff = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 2])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamq.step(&mut q, &g);
    }
    let mut fw: [Vec<f32>; 11] = Default::default(); for c in 0..11 { fw[c] = q[c].to_vec().await; }
    let fb3 = q[15].to_vec().await; let fnet = Fn2 { w: fw, b1: q[11].to_vec().await, w2: q[12].to_vec().await, b2: q[13].to_vec().await, w3: q[14].to_vec().await, b3: [fb3[0], fb3[1]] };

    // ---- Stage 3: eval — discrete argmin (25/81 evals) vs flow-matching K forward passes (no grad) ----
    let nep = 40usize; let n = nep * TG.len();
    let mut inits: Vec<[f32; 4]> = vec![]; let mut goals: Vec<(f32, f32)> = vec![];
    for (gi, &g) in TG.iter().enumerate() { for e in 0..nep { let sd = (gi * nep + e) as u32; inits.push([(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, 0.0, 0.0]); goals.push(g); } }
    let reached = |s: [f32; 4], g: (f32, f32)| wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && s[2].abs() < 0.7 && s[3].abs() < 0.7;
    let run_discrete = |grid: &[f32]| -> f32 { let mut r = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..260 { let uu = vn.ustar(s, g, grid); s = step(s, uu.0, uu.1); } if reached(s, g) { r += 1; } } r as f32 / n as f32 * 100.0 };
    let run_flow = |k: usize| -> f32 { let mut r = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i]; for _ in 0..260 { let (u1, u2) = fnet.act(s, g, k); s = step(s, u1, u2); } if reached(s, g) { r += 1; } } r as f32 / n as f32 * 100.0 };
    // diagnostic: flow action vs u*
    let (mut mag, mut err) = (0.0f32, 0.0f32); for i in 0..n { let a = fnet.act(inits[i], goals[i], 2); let us = vn.ustar(inits[i], goals[i], &G5); mag += (a.0 * a.0 + a.1 * a.1).sqrt(); err += ((a.0 - us.0).powi(2) + (a.1 - us.1).powi(2)).sqrt(); }

    println!("  2-DOF coupled arm. flow-matching policy = integrate learned velocity field K forward passes (NO grad, NO BPTT).");
    println!("  DIAG: K=2 flow action mean|u|={:.2}, mean|u−u*|={:.2}\n", mag / n as f32, err / n as f32);
    println!("     controller                         reach     evals/decision   note");
    println!("     discrete argmin, coarse 5×5         {:>4.0}%          25          Gᵈ (exponential in DOF)", run_discrete(&G5));
    println!("     discrete argmin, fine   9×9         {:>4.0}%          81", run_discrete(&G9));
    for k in [1usize, 2, 4] { println!("     flow-matching, K={:<2} forward passes  {:>4.0}%          {:>2}          K forward passes (constant in DOF)", k, run_flow(k), k); }
    println!("\n  If flow-matching REACHES (vs IBC-descent's 0% on this same arm), the frontier check is validated: the recipe was");
    println!("  the cripple, not the body — and K forward passes ≪ Gᵈ argmin evals is the real perf-per-watt edge on multi-DOF.");
    println!("  HONEST: distills the discrete demonstrator (win = works where IBC didn't, + constant eval budget); energy kept separately for verify/likelihood.");
}
