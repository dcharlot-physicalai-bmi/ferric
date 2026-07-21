//! EFA energy-first #48 — INGEST EBT-Policy, Phase 1: a DEDICATED continuous-action energy Q(s,u,g) descended over u.
//!
//! Phase 0 (ebm_actdescent) proved you cannot descend a 1-step VALUE over a continuous torque — ∂(next)/∂u ∝ DT=0.05
//! makes the gradient tiny and descent under-steps (1-3% reach). Fix: put the action u DIRECTLY in the energy input
//! (columns u, u²), so ∂Q/∂u is O(1). Train Q(state,u,goal) by FVI with the action SAMPLED CONTINUOUSLY and the
//! Bellman target V(s')=min over a u'-grid of Q_tgt(s',u'); at inference, control = K-step gradient descent of Q over u.
//! This is EBT-Policy's mechanism on EFA's own energy — K (descent steps) becomes the literal joules-per-task knob.
//! HONEST: on a 1-D torque the continuous edge over 5 discrete torques is modest; the point is that descent now WORKS
//! and K is a real efficiency dial. The exponential payoff is multi-joint (scale-next).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_actdescent2 --release`
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
// grid of candidate next-actions for the Bellman min (9 pts) and for the discrete baseline eval
const GRID: [f32; 9] = [-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0];

// action-energy Q(state,u,goal) — softplus MLP over [cos(θ−g),sin(θ−g),ω,cosθ,sinθ,u,u²]
struct Q { wrc: Vec<f32>, wrs: Vec<f32>, wo: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, wu: Vec<f32>, wu2: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Q {
    fn eval(&self, th: f32, om: f32, g: f32, uu: f32) -> f32 {
        let d = th - g; let (rc, rs, c, s) = (d.cos(), d.sin(), th.cos(), th.sin());
        let mut h1 = [0.0f32; H];
        for j in 0..H { let p = self.b1[j] + rc * self.wrc[j] + rs * self.wrs[j] + om * self.wo[j] + c * self.wc[j] + s * self.ws[j] + uu * self.wu[j] + uu * uu * self.wu2[j]; h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn greedy(&self, th: f32, om: f32, g: f32) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let q = self.eval(th, om, g, uu); if q < be { be = q; bu = uu; } } bu }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — INGEST EBT-Policy Phase 1: dedicated action-energy Q(s,u,g), descended over continuous u\n");
    // params: 7 rank-1 input weights [1,H] (rc,rs,om,cth,sth,u,u²), b1[H], W2[H,H], b2[H], W3[H,1], b3[1]
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 23, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 24, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 26, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 29, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 30, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut tgt = p.clone();
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    let enet = |rc: &Var, rs: &Var, om: &Var, cth: &Var, sth: &Var, uu: &Var, u2: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let pre = rc.matmul(&pv[0]).add(&rs.matmul(&pv[1])).add(&om.matmul(&pv[2])).add(&cth.matmul(&pv[3])).add(&sth.matmul(&pv[4])).add(&uu.matmul(&pv[5])).add(&u2.matmul(&pv[6])).add(&pv[7]);
        sp(sp(sp(pre).matmul(&pv[8]).add(&pv[9])).matmul(&pv[10]).add(&pv[11]))
    };
    let gpts = GRID.len();
    for it in 0..16000 {
        // sample (θ,ω,g,u); build current features + next-state features over the u'-grid for the Bellman min
        let (mut rc, mut rs, mut om, mut ct, mut st, mut uu_, mut u2_) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        let mut cst = vec![0.0f32; bs];
        let (mut nrc, mut nrs, mut nom, mut nct, mut nst, mut nu, mut nu2) = (vec![0.0f32; bs * gpts], vec![0.0f32; bs * gpts], vec![0.0f32; bs * gpts], vec![0.0f32; bs * gpts], vec![0.0f32; bs * gpts], vec![0.0f32; bs * gpts], vec![0.0f32; bs * gpts]);
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om0 = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0; let ua = (u(sd, 5) * 2.0 - 1.0) * UMAX;
            let d = th - g; rc[i] = d.cos(); rs[i] = d.sin(); om[i] = om0; ct[i] = th.cos(); st[i] = th.sin(); uu_[i] = ua; u2_[i] = ua * ua;
            cst[i] = wrap(th - g).powi(2) + 0.05 * om0 * om0 + 0.01 * ua * ua;
            let (nt, no) = step(th, om0, ua); let dd = nt - g; let (drc, drs, dct, dst) = (dd.cos(), dd.sin(), nt.cos(), nt.sin());
            for (j, &up) in GRID.iter().enumerate() { let idx = i * gpts + j; nrc[idx] = drc; nrs[idx] = drs; nom[idx] = no; nct[idx] = dct; nst[idx] = dst; nu[idx] = up; nu2[idx] = up * up; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let qn = enet(&l(&nrc, bs * gpts), &l(&nrs, bs * gpts), &l(&nom, bs * gpts), &l(&nct, bs * gpts), &l(&nst, bs * gpts), &l(&nu, bs * gpts), &l(&nu2, bs * gpts), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for j in 0..gpts { if qn[i * gpts + j] < m { m = qn[i * gpts + j]; } } target[i] = cst[i] * DT + GAMMA * m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&l(&rc, bs), &l(&rs, bs), &l(&om, bs), &l(&ct, bs), &l(&st, bs), &l(&uu_, bs), &l(&u2_, bs), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
    }
    let qq = Q { wrc: p[0].to_vec().await, wrs: p[1].to_vec().await, wo: p[2].to_vec().await, wc: p[3].to_vec().await, ws: p[4].to_vec().await, wu: p[5].to_vec().await, wu2: p[6].to_vec().await, b1: p[7].to_vec().await, w2: p[8].to_vec().await, b2: p[9].to_vec().await, w3: p[10].to_vec().await, b3: p[11].to_vec().await[0] };
    let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());

    let nep = 48usize; let ng = TESTG.len(); let n = nep * ng; let alpha = 0.3f32; let steps = 220usize;
    let ks = [0usize, 1, 2, 4, 8, 16]; let mut results: Vec<(usize, f32, f32)> = vec![];
    for &kdesc in &ks {
        let mut th = vec![0.0f32; n]; let mut om = vec![0.0f32; n]; let mut gg = vec![0.0f32; n]; let mut uprev = vec![0.0f32; n];
        for gi in 0..ng { for e in 0..nep { let idx = gi * nep + e; th[idx] = (u(900 + idx as u32, 7) * 2.0 - 1.0) * PI; gg[idx] = TESTG[gi]; } }
        let (mut reach, mut tv, mut tvn) = (vec![true; n], 0.0f32, 0.0f32);
        for t in 0..steps {
            let mut uu = vec![0.0f32; n];
            if kdesc == 0 { for i in 0..n { uu[i] = qq.greedy(th[i], om[i], gg[i]); } }        // discrete-greedy baseline
            else {
                let mut ucur = uprev.clone();                                                    // warm-start
                // per-episode state columns are constant across descent steps; only u,u² change
                let (rcc, rss, omm, ctt, stt): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) =
                    (0..n).map(|i| { let d = th[i] - gg[i]; (d.cos(), d.sin(), om[i], th[i].cos(), th[i].sin()) }).fold((vec![], vec![], vec![], vec![], vec![]), |mut a, x| { a.0.push(x.0); a.1.push(x.1); a.2.push(x.2); a.3.push(x.3); a.4.push(x.4); a });
                let rcv = Var::leaf(Tensor::from_vec(&ctx, &rcc, &[n, 1])); let rsv = Var::leaf(Tensor::from_vec(&ctx, &rss, &[n, 1])); let omv = Var::leaf(Tensor::from_vec(&ctx, &omm, &[n, 1]));
                let ctv = Var::leaf(Tensor::from_vec(&ctx, &ctt, &[n, 1])); let stv = Var::leaf(Tensor::from_vec(&ctx, &stt, &[n, 1]));
                for _ in 0..kdesc {
                    let u2: Vec<f32> = ucur.iter().map(|&x| x * x).collect();
                    let uv = Var::leaf(Tensor::from_vec(&ctx, &ucur, &[n, 1])); let u2v = Var::leaf(Tensor::from_vec(&ctx, &u2, &[n, 1]));
                    let qval = enet(&rcv, &rsv, &omv, &ctv, &stv, &uv, &u2v, &pv, &ov);
                    let gu = grad(&qval.sum_all(), &[uv.clone()], None);                          // ∂Q/∂u, O(1) — u is a direct input
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

    println!("  dedicated action-energy Q(s,u,g) trained (FVI, u sampled continuously). Control = K-step descent of Q over u.");
    println!("  eval: {} episodes × {} goals; reach = within 0.3 rad / 0.6 rad·s over a settled window.\n", nep, ng);
    println!("     K (descent steps)     control-reach   action-smoothness (mean|Δu|)   note");
    for &(k, r, tv) in &results {
        if k == 0 { println!("     0  (discrete greedy)     {:>4.0}%              {:.3}                 baseline — the OLD primitive (5 evals, bang-bang)", r, tv); }
        else { println!("     {:>2} (continuous descent)   {:>4.0}%              {:.3}                 {} energy-grad evals/decision", k, r, tv, k); } }
    println!("\n  Reading: with u a DIRECT input, ∂Q/∂u is O(1) — descent should now control (vs Phase-0's DT-starved 1-3%).");
    println!("  reach% vs K = the accuracy-vs-joules curve; continuous descent should match the discrete baseline with smoother action.");
}
