//! EFA energy-first #52 — SCALE the proof: 2-DOF coupled arm, where continuous energy descent COMPOUNDS over discrete.
//!
//! On the 1-D pendulum, continuous descent only tied discrete argmin (5 action-evals is cheap). The edge is DIMENSIONAL:
//! discrete argmin costs Gᵈ action-evals per decision (exponential in DOF), continuous descent costs K grad-evals
//! regardless. Here d=2 (two torques), so a discrete grid is 5×5=25 or 9×9=81 evals, while descent is still K. We build
//! a coupled 2-link arm, distill the discrete controller into a descendable action-energy E(state,u₁,u₂,goal), and plot
//! reach% AND evals-per-decision for discrete (coarse 25 / fine 81) vs continuous K-step descent. The perf-per-watt
//! (reach-per-eval) should now favor continuous — the 1-D "modest" win compounding as predicted. HONEST: still a small
//! body; the point is the SCALING of the eval budget, shown, not asserted.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_arm2 --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const G5: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];                 // coarse per-joint action grid (5×5 = 25)
const G9: [f32; 9] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]; // fine grid (9×9 = 81)
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

// value V(state,goal) — 8 features
struct Vn { w: [Vec<f32>; 8], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Vn {
    fn eval(&self, s: [f32; 4], g: (f32, f32)) -> f32 { let f = feat8(s, g);
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..8 { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
    fn ustar(&self, s: [f32; 4], g: (f32, f32), grid: &[f32]) -> (f32, f32) { let mut bu = (0.0, 0.0); let mut be = f32::MAX;
        for &u1 in grid { for &u2 in grid { let ns = step(s, u1, u2); let q = (cost(s, g, u1, u2)) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = (u1, u2); } } } bu }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — SCALE: 2-DOF coupled arm; does continuous descent COMPOUND over discrete (eval budget)?\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    let sp_of = |z: Var, ov: &Var| z.exp().add(ov).log();
    // V-net: 8 rank-1 input weights, b1, W2, b2, W3, b3
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[8].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); }
        sp_of(sp_of(sp_of(pre, ov).matmul(&pv[9]).add(&pv[10]), ov).matmul(&pv[11]).add(&pv[12]), ov) };
    let mut p: Vec<Tensor> = (0..8).map(|c| Tensor::from_vec(&ctx, &randn(H, 22 + c as u32, 0.5), &[1, H])).collect();
    p.push(Tensor::zeros(&ctx, &[H])); p.push(Tensor::from_vec(&ctx, &randn(H * H, 40, 1.0 / (H as f32).sqrt()), &[H, H])); p.push(Tensor::zeros(&ctx, &[H]));
    p.push(Tensor::from_vec(&ctx, &randn(H, 41, 1.0 / (H as f32).sqrt()), &[H, 1])); p.push(Tensor::zeros(&ctx, &[1]));
    let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002);
    // ---- Stage 1: FVI train V with the coarse 5×5 discrete grid ----
    let gg = &G5; let ga = gg.len() * gg.len();
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
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let fv: Vec<Var> = (0..8).map(|c| l(&fc[c], bs)).collect();
        let e = vnet(&fv, &pv, &ov); let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let mut wv: [Vec<f32>; 8] = Default::default(); for c in 0..8 { wv[c] = p[c].to_vec().await; }
    let vn = Vn { w: wv, b1: p[8].to_vec().await, w2: p[9].to_vec().await, b2: p[10].to_vec().await, w3: p[11].to_vec().await, b3: p[12].to_vec().await[0] };

    // ---- Stage 2: distill discrete u*(fine grid) into a descendable action-energy E(state,u1,u2,goal) ----
    // features: 8 state + [u1,u2,u1²,u2²] = 12; target bowl (u1−u1*)²+(u2−u2*)²
    let enet = |f: &[Var], u1: &Var, u2: &Var, u1s: &Var, u2s: &Var, pv: &[Var], ov: &Var| {
        let mut pre = pv[12].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre = pre.add(&u1.matmul(&pv[8])).add(&u2.matmul(&pv[9])).add(&u1s.matmul(&pv[10])).add(&u2s.matmul(&pv[11]));
        sp_of(sp_of(sp_of(pre, ov).matmul(&pv[13]).add(&pv[14]), ov).matmul(&pv[15]).add(&pv[16]), ov) };
    let mut q: Vec<Tensor> = (0..12).map(|c| Tensor::from_vec(&ctx, &randn(H, 50 + c as u32, 0.5), &[1, H])).collect();
    q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H * H, 70, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(&ctx, &[H]));
    q.push(Tensor::from_vec(&ctx, &randn(H, 71, 1.0 / (H as f32).sqrt()), &[H, 1])); q.push(Tensor::zeros(&ctx, &[1]));
    let mut adamq = Adam::new(&q, 0.002);
    for it in 0..16000 {
        let mut fc: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs]).collect();
        let (mut u1b, mut u2b, mut u1s, mut u2s, mut tb) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        for i in 0..bs { let sd = it as u32 * 13 + i as u32;
            let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * 3.0, (u(sd, 4) * 2.0 - 1.0) * 3.0];
            let g = ((u(sd, 5) * 2.0 - 1.0) * 1.2, (u(sd, 6) * 2.0 - 1.0) * 1.2); let f = feat8(s, g); for c in 0..8 { fc[c][i] = f[c]; }
            let ua = (u(sd, 7) * 2.0 - 1.0) * UMAX; let ub = (u(sd, 8) * 2.0 - 1.0) * UMAX; let us = vn.ustar(s, g, &G5);
            u1b[i] = ua; u2b[i] = ub; u1s[i] = ua * ua; u2s[i] = ub * ub; tb[i] = (ua - us.0).powi(2) + (ub - us.1).powi(2); }
        let l = |v: &[f32]| Var::leaf(Tensor::from_vec(&ctx, v, &[bs, 1])); let ov = Var::leaf(one.clone());
        let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect(); let fv: Vec<Var> = (0..8).map(|c| l(&fc[c])).collect();
        let e = enet(&fv, &l(&u1b), &l(&u2b), &l(&u1s), &l(&u2s), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 1]))); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamq.step(&mut q, &g);
    }
    let qv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());

    // ---- Stage 3: eval reach% + evals/decision — discrete coarse(25)/fine(81) vs continuous K-step descent ----
    let nep = 40usize; let n = nep * TG.len();
    let mut inits: Vec<[f32; 4]> = vec![]; let mut goals: Vec<(f32, f32)> = vec![];
    for (gi, &g) in TG.iter().enumerate() { for e in 0..nep { let sd = (gi * nep + e) as u32; inits.push([(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, 0.0, 0.0]); goals.push(g); } }
    let run_discrete = |grid: &[f32]| -> f32 { let mut reach = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i];
        for t in 0..260 { let uu = vn.ustar(s, g, grid); s = step(s, uu.0, uu.1); if t >= 220 && (wrap(s[0] - g.0).abs() > 0.35 || wrap(s[1] - g.1).abs() > 0.35 || s[2].abs() > 0.7 || s[3].abs() > 0.7) { reach = reach; } }
        if wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && s[2].abs() < 0.7 && s[3].abs() < 0.7 { reach += 1; } } reach as f32 / n as f32 * 100.0 };
    let rc = run_discrete(&G5); let rf = run_discrete(&G9);

    // continuous K-step descent over (u1,u2)
    let alpha = 0.3f32; let ks = [1usize, 2, 4, 8]; let mut cres: Vec<(usize, f32)> = vec![];
    for &kk in &ks {
        let mut s: Vec<[f32; 4]> = inits.clone(); let mut up: Vec<(f32, f32)> = vec![(0.0, 0.0); n]; let mut reach = vec![true; n];
        for t in 0..260 {
            let mut fc: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; n]).collect();
            for i in 0..n { let f = feat8(s[i], goals[i]); for c in 0..8 { fc[c][i] = f[c]; } }
            let fv: Vec<Var> = (0..8).map(|c| Var::leaf(Tensor::from_vec(&ctx, &fc[c], &[n, 1]))).collect();
            let (mut cu1, mut cu2): (Vec<f32>, Vec<f32>) = (up.iter().map(|x| x.0).collect(), up.iter().map(|x| x.1).collect());
            for _ in 0..kk {
                let u1v = Var::leaf(Tensor::from_vec(&ctx, &cu1, &[n, 1])); let u2v = Var::leaf(Tensor::from_vec(&ctx, &cu2, &[n, 1]));
                let u1s = u1v.mul(&u1v); let u2s = u2v.mul(&u2v);   // u² IN-GRAPH so ∂E/∂u flows through the bowl curvature
                let e = enet(&fv, &u1v, &u2v, &u1s, &u2s, &qv, &ov);
                let gd = grad(&e.sum_all(), &[u1v.clone(), u2v.clone()], None);
                let d1 = gd[0].value().to_vec().await; let d2 = gd[1].value().to_vec().await;
                for i in 0..n { cu1[i] = (cu1[i] - alpha * d1[i]).clamp(-UMAX, UMAX); cu2[i] = (cu2[i] - alpha * d2[i]).clamp(-UMAX, UMAX); }
            }
            for i in 0..n { s[i] = step(s[i], cu1[i], cu2[i]); up[i] = (cu1[i], cu2[i]);
                if t >= 220 && !(wrap(s[i][0] - goals[i].0).abs() < 0.35 && wrap(s[i][1] - goals[i].1).abs() < 0.35 && s[i][2].abs() < 0.7 && s[i][3].abs() < 0.7) { reach[i] = false; } }
        }
        cres.push((kk, reach.iter().filter(|&&b| b).count() as f32 / n as f32 * 100.0));
    }

    println!("  2-DOF coupled arm (2 torques). eval: {} episodes × {} goals.\n", nep, TG.len());
    println!("     controller                         reach     evals/decision   note");
    println!("     discrete argmin, coarse 5×5         {:>4.0}%          25          Gᵈ evals — exponential in DOF", rc);
    println!("     discrete argmin, fine   9×9         {:>4.0}%          81          refine grid → 81 evals", rf);
    for &(k, r) in &cres { println!("     continuous descent, K={:<2}            {:>4.0}%          {:>2}          K grad-evals — CONSTANT in DOF", k, r, k); }
    println!("\n  Reading: on 2-DOF, discrete argmin pays 25–81 action-evals/decision; continuous descent pays only K.");
    println!("  If continuous matches discrete-fine at K≪81, the perf-per-watt (reach-per-eval) edge that was ~0 on 1-D has COMPOUNDED.");
    println!("  HONEST: still a small body + distilled controller; the demonstrated quantity is the EVAL-BUDGET SCALING (Gᵈ vs K), not new capability.");
}
