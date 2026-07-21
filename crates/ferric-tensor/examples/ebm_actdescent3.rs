//! EFA energy-first #50 — INGEST EBT-Policy, Phase 2: an action-energy SHAPED FOR DESCENT (the correct mechanism).
//!
//! Phases 0 & 1 proved the lesson: descending an FVI/Bellman energy over a continuous torque STALLS (0-4% reach) —
//! a one-step-lookahead value is nearly flat in the action (∝ DT), so ∂E/∂u is tiny. EBT-Policy's edge is NOT
//! "descend any energy"; it is an energy TRAINED so its minimum sits at the good action, with real gradients. Here we
//! distill the working discrete controller's action u*(s,g) into a descendable action-energy E(s,u,g) (trained toward a
//! bowl minimized at u*), then control by K-step descent of E over u. ∂E/∂u now points at u*, so descent converges and
//! K becomes the real accuracy-vs-joules dial. HONEST: on a 1-D torque this distills the discrete controller (the win is
//! smoothness + a continuous, K-tunable controller, not new capability); the compounding payoff is multi-joint (scale-next).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_actdescent3 --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 64; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const TESTG: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }

// value V(state,goal), 5 features — trains the discrete controller (the "demonstrator")
struct Vn { wrc: Vec<f32>, wrs: Vec<f32>, wo: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Vn {
    fn eval(&self, th: f32, om: f32, g: f32) -> f32 {
        let d = th - g; let (rc, rs, c, s) = (d.cos(), d.sin(), th.cos(), th.sin());
        let mut h1 = [0.0f32; H]; for j in 0..H { let p = self.b1[j] + rc * self.wrc[j] + rs * self.wrs[j] + om * self.wo[j] + c * self.wc[j] + s * self.ws[j]; h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn ustar(&self, th: f32, om: f32, g: f32) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let q = 0.01 * uu * uu * DT + GAMMA * self.eval(nt, no, g); if q < be { be = q; bu = uu; } } bu }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — INGEST EBT-Policy Phase 2: action-energy SHAPED FOR DESCENT (min at the good action)\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    let enet5 = |rc: &Var, rs: &Var, om: &Var, cth: &Var, sth: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let pre = rc.matmul(&pv[0]).add(&rs.matmul(&pv[1])).add(&om.matmul(&pv[2])).add(&cth.matmul(&pv[3])).add(&sth.matmul(&pv[4])).add(&pv[5]);
        sp(sp(sp(pre).matmul(&pv[6]).add(&pv[7])).matmul(&pv[8]).add(&pv[9]))
    };
    // ---- Stage 1: FVI train V(state,goal) → discrete controller u*(s,g) ----
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 23, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 24, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 26, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002);
    for it in 0..14000 {
        let (mut rc, mut rs, mut om, mut ct, mut st) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        let (mut nrc, mut nrs, mut nom, mut nct, mut nst) = (vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5]); let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om0 = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0;
            let d = th - g; rc[i] = d.cos(); rs[i] = d.sin(); om[i] = om0; ct[i] = th.cos(); st[i] = th.sin();
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om0, uu); let dd = nt - g; nrc[i * 5 + ai] = dd.cos(); nrs[i * 5 + ai] = dd.sin(); nom[i * 5 + ai] = no; nct[i * 5 + ai] = nt.cos(); nst[i * 5 + ai] = nt.sin();
                cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om0 * om0 + 0.01 * uu * uu; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
        let tvv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet5(&l(&nrc, bs * 5), &l(&nrs, bs * 5), &l(&nom, bs * 5), &l(&nct, bs * 5), &l(&nst, bs * 5), &tvv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet5(&l(&rc, bs), &l(&rs, bs), &l(&om, bs), &l(&ct, bs), &l(&st, bs), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let vn = Vn { wrc: p[0].to_vec().await, wrs: p[1].to_vec().await, wo: p[2].to_vec().await, wc: p[3].to_vec().await, ws: p[4].to_vec().await, b1: p[5].to_vec().await, w2: p[6].to_vec().await, b2: p[7].to_vec().await, w3: p[8].to_vec().await, b3: p[9].to_vec().await[0] };

    // ---- Stage 2: distill u*(s,g) into a descendable action-energy E(s,u,g) → bowl (u−u*)² (7 features incl u,u²) ----
    let enet7 = |rc: &Var, rs: &Var, om: &Var, cth: &Var, sth: &Var, uu: &Var, u2: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let pre = rc.matmul(&pv[0]).add(&rs.matmul(&pv[1])).add(&om.matmul(&pv[2])).add(&cth.matmul(&pv[3])).add(&sth.matmul(&pv[4])).add(&uu.matmul(&pv[5])).add(&u2.matmul(&pv[6])).add(&pv[7]);
        sp(sp(sp(pre).matmul(&pv[8]).add(&pv[9])).matmul(&pv[10]).add(&pv[11]))
    };
    let mut q = vec![
        Tensor::from_vec(&ctx, &randn(H, 32, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 33, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 34, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 35, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 36, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 37, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 38, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 39, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]), Tensor::from_vec(&ctx, &randn(H, 40, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adamq = Adam::new(&q, 0.002);
    for it in 0..12000 {
        let (mut rc, mut rs, mut om, mut ct, mut st, mut uq, mut u2) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]); let mut tb = vec![0.0f32; bs];
        for i in 0..bs { let sd = it as u32 * 13 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om0 = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0; let ua = (u(sd, 5) * 2.0 - 1.0) * UMAX;
            let us = vn.ustar(th, om0, g); let d = th - g; rc[i] = d.cos(); rs[i] = d.sin(); om[i] = om0; ct[i] = th.cos(); st[i] = th.sin(); uq[i] = ua; u2[i] = ua * ua;
            tb[i] = (ua - us) * (ua - us); }   // bowl minimized at u* — the descendable target energy
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1])); let ov = Var::leaf(one.clone());
        let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet7(&l(&rc, bs), &l(&rs, bs), &l(&om, bs), &l(&ct, bs), &l(&st, bs), &l(&uq, bs), &l(&u2, bs), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 1]))); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamq.step(&mut q, &g);
    }
    let qv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());

    // ---- Stage 3: eval reach%/smoothness — discrete baseline (greedy on V) vs K-step descent of E over u ----
    let nep = 48usize; let ng = TESTG.len(); let n = nep * ng; let alpha = 0.35f32; let steps = 220usize;
    let ks = [0usize, 1, 2, 4, 8, 16]; let mut results: Vec<(usize, f32, f32)> = vec![];
    for &kdesc in &ks {
        let mut th = vec![0.0f32; n]; let mut om = vec![0.0f32; n]; let mut gg = vec![0.0f32; n]; let mut uprev = vec![0.0f32; n];
        for gi in 0..ng { for e in 0..nep { let idx = gi * nep + e; th[idx] = (u(900 + idx as u32, 7) * 2.0 - 1.0) * PI; gg[idx] = TESTG[gi]; } }
        let (mut reach, mut tv, mut tvn) = (vec![true; n], 0.0f32, 0.0f32);
        for t in 0..steps {
            let mut uu = vec![0.0f32; n];
            if kdesc == 0 { for i in 0..n { uu[i] = vn.ustar(th[i], om[i], gg[i]); } }
            else {
                let mut ucur = uprev.clone();
                let (rcc, rss, omm, ctt, stt): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) = (0..n).map(|i| { let d = th[i] - gg[i]; (d.cos(), d.sin(), om[i], th[i].cos(), th[i].sin()) }).fold((vec![], vec![], vec![], vec![], vec![]), |mut a, x| { a.0.push(x.0); a.1.push(x.1); a.2.push(x.2); a.3.push(x.3); a.4.push(x.4); a });
                let rcv = Var::leaf(Tensor::from_vec(&ctx, &rcc, &[n, 1])); let rsv = Var::leaf(Tensor::from_vec(&ctx, &rss, &[n, 1])); let omv = Var::leaf(Tensor::from_vec(&ctx, &omm, &[n, 1])); let ctv = Var::leaf(Tensor::from_vec(&ctx, &ctt, &[n, 1])); let stv = Var::leaf(Tensor::from_vec(&ctx, &stt, &[n, 1]));
                for _ in 0..kdesc {
                    let uv = Var::leaf(Tensor::from_vec(&ctx, &ucur, &[n, 1])); let u2v = uv.mul(&uv);   // u² IN-GRAPH (correct ∂E/∂u)
                    let qval = enet7(&rcv, &rsv, &omv, &ctv, &stv, &uv, &u2v, &qv, &ov);
                    let gu = grad(&qval.sum_all(), &[uv.clone()], None);
                    let du = gu[0].value().to_vec().await;
                    for i in 0..n { ucur[i] = (ucur[i] - alpha * du[i]).clamp(-UMAX, UMAX); }
                }
                uu = ucur;
            }
            for i in 0..n { tv += (uu[i] - uprev[i]).abs(); tvn += 1.0; let (nt, no) = step(th[i], om[i], uu[i]); th[i] = nt; om[i] = no; uprev[i] = uu[i];
                if t >= steps - 40 && !(wrap(th[i] - gg[i]).abs() < 0.3 && om[i].abs() < 0.6) { reach[i] = false; } }
        }
        results.push((kdesc, reach.iter().filter(|&&b| b).count() as f32 / n as f32 * 100.0, tv / tvn));
    }
    println!("  action-energy E(s,u,g) distilled from the discrete controller u* (bowl at u*); control = K-step descent of E over u.");
    println!("  eval: {} episodes × {} goals; reach = within 0.3 rad / 0.6 rad·s over a settled window.\n", nep, ng);
    println!("     K (descent steps)     control-reach   action-smoothness (mean|Δu|)   note");
    for &(k, r, tv) in &results {
        if k == 0 { println!("     0  (discrete greedy)     {:>4.0}%              {:.3}                 baseline — discrete argmin (the demonstrator)", r, tv); }
        else { println!("     {:>2} (continuous descent)   {:>4.0}%              {:.3}                 {} energy-grad evals/decision", k, r, tv, k); } }
    println!("\n  Reading: with the energy SHAPED for descent, continuous control should now MATCH the discrete baseline and RISE with K —");
    println!("  the real accuracy-vs-joules curve. This is EBT-Policy's mechanism done right (energy trained for descent, not a Bellman value).");
    println!("  HONEST: 1-D torque distills the discrete controller (win = smooth, continuous, K-tunable); the compounding edge is multi-joint.");
}
