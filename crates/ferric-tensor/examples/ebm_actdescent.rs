//! EFA energy-first #47 — INGEST EBT-Policy: continuous multi-step energy DESCENT over actions (Step 1 of the ingestion roadmap).
//!
//! EFA's "control by descent" was NOT descent — it was a 1-step argmin over 5 discrete torques of a Bellman value.
//! This ingests EBT-Policy's actual mechanism: inference = K steps of gradient descent on the energy over a CONTINUOUS
//! action, using Ferric's autograd for ∂Q/∂u through the pendulum dynamics put IN-GRAPH (the energy is periodic in θ via
//! cos/sin, so no wrap is needed for the gradient). K (descent steps) becomes the literal joules-per-task knob: we plot
//! CONTROL reach% vs K ∈ {0,1,2,4,8,16} against the discrete-greedy baseline — the accuracy-vs-joules curve on a body.
//! Phase-0 de-risk spike: descend a value energy V(state,goal) trained by ordinary FVI; Q(s,u)=0.01u²·DT + γ·V(step(s,u)).
//! HONEST: on a 1-D torque the continuous win over 5 discrete torques is inherently modest — the exponential payoff is
//! multi-joint only, which this body does not show. The point here is that descent WORKS and K is a real efficiency dial.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_actdescent --release`
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

// value energy V(state,goal) — softplus MLP over relative features [cos(θ−g),sin(θ−g),ω,cosθ,sinθ]
struct V { wrc: Vec<f32>, wrs: Vec<f32>, wo: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl V {
    fn eval(&self, th: f32, om: f32, g: f32) -> f32 {
        let d = th - g; let (rc, rs, c, s) = (d.cos(), d.sin(), th.cos(), th.sin());
        let mut h1 = [0.0f32; H];
        for j in 0..H { let p = self.b1[j] + rc * self.wrc[j] + rs * self.wrs[j] + om * self.wo[j] + c * self.wc[j] + s * self.ws[j]; h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn qcost(&self, th: f32, om: f32, g: f32, uu: f32) -> f32 { let (nt, no) = step(th, om, uu); 0.01 * uu * uu * DT + GAMMA * self.eval(nt, no, g) }
    fn greedy(&self, th: f32, om: f32, g: f32) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let q = self.qcost(th, om, g, uu); if q < be { be = q; bu = uu; } } bu }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — INGEST EBT-Policy: continuous multi-step energy DESCENT over actions (K = the joules knob)\n");
    // params: 5 rank-1 input weights [1,H], b1[H], W2[H,H], b2[H], W3[H,1], b3[1]
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 23, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 24, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 26, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut tgt = p.clone();
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    // enet over feature columns given as [n,1] leaves rc,rs,om,cth,sth
    let enet = |rc: &Var, rs: &Var, om: &Var, cth: &Var, sth: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let pre = rc.matmul(&pv[0]).add(&rs.matmul(&pv[1])).add(&om.matmul(&pv[2])).add(&cth.matmul(&pv[3])).add(&sth.matmul(&pv[4])).add(&pv[5]);
        sp(sp(sp(pre).matmul(&pv[6]).add(&pv[7])).matmul(&pv[8]).add(&pv[9]))
    };
    // ---- FVI train V(state,goal), discrete-action Bellman (the baseline value the descent will use) ----
    for it in 0..14000 {
        let (mut rc, mut rs, mut om, mut ct, mut st) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        let (mut nrc, mut nrs, mut nom, mut nct, mut nst) = (vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5]); let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om0 = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0;
            let d = th - g; rc[i] = d.cos(); rs[i] = d.sin(); om[i] = om0; ct[i] = th.cos(); st[i] = th.sin();
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om0, uu); let dd = nt - g;
                nrc[i * 5 + ai] = dd.cos(); nrs[i * 5 + ai] = dd.sin(); nom[i * 5 + ai] = no; nct[i * 5 + ai] = nt.cos(); nst[i * 5 + ai] = nt.sin();
                cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om0 * om0 + 0.01 * uu * uu; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet(&l(&nrc, bs * 5), &l(&nrs, bs * 5), &l(&nom, bs * 5), &l(&nct, bs * 5), &l(&nst, bs * 5), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&l(&rc, bs), &l(&rs, bs), &l(&om, bs), &l(&ct, bs), &l(&st, bs), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
    }
    let vv = V { wrc: p[0].to_vec().await, wrs: p[1].to_vec().await, wo: p[2].to_vec().await, wc: p[3].to_vec().await, ws: p[4].to_vec().await, b1: p[5].to_vec().await, w2: p[6].to_vec().await, b2: p[7].to_vec().await, w3: p[8].to_vec().await, b3: p[9].to_vec().await[0] };
    let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());

    // ---- EVAL: reach%, action smoothness (TV), avg steps — for K-step CONTINUOUS descent vs K=0 discrete-greedy baseline ----
    // Batched over N episodes × |TESTG| goals. Each control step: K descent iterations of u ← clamp(u − α·∂Q/∂u).
    let nep = 48usize; let ng = TESTG.len(); let n = nep * ng; let alpha = 0.4f32; let steps = 220usize;
    let dtv = Var::leaf(Tensor::from_vec(&ctx, &[DT], &[1])); let dt2v = Var::leaf(Tensor::from_vec(&ctx, &[DT * DT], &[1]));
    let acv = Var::leaf(Tensor::from_vec(&ctx, &[0.01 * DT], &[1])); let gav = Var::leaf(Tensor::from_vec(&ctx, &[GAMMA], &[1]));
    let ks = [0usize, 1, 2, 4, 8, 16]; let mut results: Vec<(usize, f32, f32)> = vec![];
    for &kdesc in &ks {
        let mut th = vec![0.0f32; n]; let mut om = vec![0.0f32; n]; let mut gg = vec![0.0f32; n]; let mut uprev = vec![0.0f32; n];
        for gi in 0..ng { for e in 0..nep { let idx = gi * nep + e; th[idx] = (u(900 + idx as u32, 7) * 2.0 - 1.0) * PI; gg[idx] = TESTG[gi]; } }
        let (mut reach, mut tv, mut tvn) = (vec![true; n], 0.0f32, 0.0f32);
        for t in 0..steps {
            let mut uu = vec![0.0f32; n];
            if kdesc == 0 { for i in 0..n { uu[i] = vv.greedy(th[i], om[i], gg[i]); } }       // discrete-greedy baseline (the OLD primitive)
            else {
                let mut ucur = uprev.clone();                                                   // warm-start from last applied torque
                let gv = Var::leaf(Tensor::from_vec(&ctx, &gg, &[n, 1]));
                let c1: Vec<f32> = (0..n).map(|i| om[i] + DT * (-th[i].sin() - 0.05 * om[i])).collect();  // u-independent part of no
                let c2: Vec<f32> = (0..n).map(|i| th[i] + DT * c1[i]).collect();                          // u-independent part of nθ
                let c1v = Var::leaf(Tensor::from_vec(&ctx, &c1, &[n, 1])); let c2v = Var::leaf(Tensor::from_vec(&ctx, &c2, &[n, 1]));
                for _ in 0..kdesc {
                    let uv = Var::leaf(Tensor::from_vec(&ctx, &ucur, &[n, 1]));
                    let no = c1v.add(&uv.mul(&dtv));                                             // no = c1 + DT·u
                    let nth = c2v.add(&uv.mul(&dt2v));                                           // nθ = c2 + DT²·u  (periodic energy ⇒ no wrap)
                    let dth = nth.sub(&gv);
                    let vval = enet(&dth.cos(), &dth.sin(), &no, &nth.cos(), &nth.sin(), &pv, &ov);
                    let q = uv.mul(&uv).mul(&acv).add(&vval.mul(&gav));
                    let gu = grad(&q.sum_all(), &[uv.clone()], None);                            // per-episode ∂Q/∂u
                    let du = gu[0].value().to_vec().await;
                    for i in 0..n { ucur[i] = (ucur[i] - alpha * du[i]).clamp(-UMAX, UMAX); }
                }
                uu = ucur;
            }
            for i in 0..n { tv += (uu[i] - uprev[i]).abs(); tvn += 1.0; let (nt, no) = step(th[i], om[i], uu[i]); th[i] = nt; om[i] = no; uprev[i] = uu[i];
                if t >= steps - 40 && !(wrap(th[i] - gg[i]).abs() < 0.3 && om[i].abs() < 0.6) { reach[i] = false; } }
        }
        let r = reach.iter().filter(|&&b| b).count() as f32 / n as f32 * 100.0;
        results.push((kdesc, r, tv / tvn));
    }

    println!("  goal-conditioned value energy trained (FVI). Control = K-step descent of Q(s,u)=0.01u²·DT+γ·V(step(s,u)) over continuous u.");
    println!("  eval: {} episodes × {} goals; reach = within 0.3 rad / 0.6 rad·s over a settled window.\n", nep, ng);
    println!("     K (descent steps)     control-reach   action-smoothness (mean|Δu|)   note");
    for &(k, r, tv) in &results {
        if k == 0 { println!("     0  (discrete greedy)     {:>4.0}%              {:.3}                 baseline — the OLD primitive (5 evals, bang-bang)", r, tv); }
        else { println!("     {:>2} (continuous descent)   {:>4.0}%              {:.3}                 {} energy-grad evals/decision", k, r, tv, k); } }
    println!("\n  Reading: reach% vs K is the accuracy-vs-JOULES curve on a body — K is the literal effort/energy dial.");
    println!("  Continuous descent should match/beat the discrete baseline with far smoother action (lower |Δu|), and improve with K.");
    println!("  HONEST: on a 1-D torque the continuous edge over 5 discrete torques is modest; the payoff compounds on multi-joint bodies (scale-next).");
}
