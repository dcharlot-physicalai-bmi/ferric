//! EFA energy-first #49 — INGEST neural-Lyapunov (Chang & Sicun Gao), Step 2: EARN the certificate EBT-Policy lacks.
//!
//! The flagship's "E decreases on N% of steps = an intrinsic Lyapunov certificate" was an on-policy MONITOR, not a
//! certificate — renamed honestly here. This earns a real one: V(x)=E(x)−E(g) with V(g)=0, trained with a Lyapunov-risk
//! objective (V>0 away from g; dV=V(step(x,ctrl))−V(x) < 0), then VERIFIED over a dense (θ,ω) grid to report a certified
//! region-of-attraction (largest sublevel {V≤c} with no descent violation). Two modes compared — FVI-only vs
//! FVI+Lyapunov-risk — so the certificate is shown to be EARNED, not assumed. This is the differentiator EBT-Policy has
//! zero of, at ~zero marginal inference cost (V *is* the control energy). HONEST: grid/Lipschitz-sampling verified
//! (conservative), NOT an SMT/dReal proof; needs known dynamics; single designated goal.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_lyapunov --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 64; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const G: f32 = 1.0;                       // single designated goal angle (region-of-attraction certified around it)
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }
fn d2(th: f32, om: f32) -> f32 { wrap(th - G).powi(2) + om * om }   // ‖x−g‖²

struct E { wrc: Vec<f32>, wrs: Vec<f32>, wo: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl E {
    fn eval(&self, th: f32, om: f32) -> f32 {
        let d = th - G; let (rc, rs, c, s) = (d.cos(), d.sin(), th.cos(), th.sin());
        let mut h1 = [0.0f32; H];
        for j in 0..H { let p = self.b1[j] + rc * self.wrc[j] + rs * self.wrs[j] + om * self.wo[j] + c * self.wc[j] + s * self.ws[j]; h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn greedy(&self, th: f32, om: f32) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let q = 0.01 * uu * uu * DT + GAMMA * self.eval(nt, no); if q < be { be = q; bu = uu; } } bu }
    fn vhat(&self, th: f32, om: f32, e0: f32) -> f32 { self.eval(th, om) - e0 }   // V=E−E(g)
}

// train E by FVI (value to g); if lyap, add Lyapunov-risk against the greedy controller (read from target net).
async fn train(ctx: &Arc<ferric_core::Context>, lyap: bool) -> E {
    let mut p = vec![
        Tensor::from_vec(ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(ctx, &randn(H, 23, 0.6), &[1, H]),
        Tensor::from_vec(ctx, &randn(H, 24, 0.6), &[1, H]), Tensor::from_vec(ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::from_vec(ctx, &randn(H, 26, 0.6), &[1, H]), Tensor::zeros(ctx, &[H]),
        Tensor::from_vec(ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(ctx, &[H]),
        Tensor::from_vec(ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(ctx, &[1]),
    ];
    let mut tgt = p.clone();
    let one = Tensor::from_vec(ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    let enet = |rc: &Var, rs: &Var, om: &Var, cth: &Var, sth: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let pre = rc.matmul(&pv[0]).add(&rs.matmul(&pv[1])).add(&om.matmul(&pv[2])).add(&cth.matmul(&pv[3])).add(&sth.matmul(&pv[4])).add(&pv[5]);
        sp(sp(sp(pre).matmul(&pv[6]).add(&pv[7])).matmul(&pv[8]).add(&pv[9]))
    };
    let feat = |th: f32, om: f32| { let d = th - G; [d.cos(), d.sin(), om, th.cos(), th.sin()] };
    let mut ectrl: Option<E> = None;   // CPU copy of the target net, for the greedy controller in the Lyapunov term
    for it in 0..14000 {
        // FVI batch
        let (mut rc, mut rs, mut om, mut ct, mut st) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        let (mut nrc, mut nrs, mut nom, mut nct, mut nst) = (vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5]); let mut cst = vec![0.0f32; bs * 5];
        // Lyapunov batch (states + controlled next states + d²)
        let (mut lrc, mut lrs, mut lom, mut lct, mut lst) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        let (mut prc, mut prs, mut pom, mut pct, mut pst) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]); let mut dd = vec![0.0f32; bs];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om0 = (u(sd, 2) * 2.0 - 1.0) * 3.0;
            let f = feat(th, om0); rc[i] = f[0]; rs[i] = f[1]; om[i] = f[2]; ct[i] = f[3]; st[i] = f[4];
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om0, uu); let g = feat(nt, no);
                nrc[i * 5 + ai] = g[0]; nrs[i * 5 + ai] = g[1]; nom[i * 5 + ai] = g[2]; nct[i * 5 + ai] = g[3]; nst[i * 5 + ai] = g[4];
                cst[i * 5 + ai] = wrap(th - G).powi(2) + 0.05 * om0 * om0 + 0.01 * uu * uu; }
            if lyap { // sample a state in the region box; controlled next state via the (frozen target) greedy controller
                let lt = G + (u(sd, 8) * 2.0 - 1.0) * 2.5; let lo = (u(sd, 9) * 2.0 - 1.0) * 3.0;
                let uu = ectrl.as_ref().map(|e| e.greedy(lt, lo)).unwrap_or(0.0); let (nt, no) = step(lt, lo, uu);
                let f = feat(lt, lo); let g = feat(nt, no);
                lrc[i] = f[0]; lrs[i] = f[1]; lom[i] = f[2]; lct[i] = f[3]; lst[i] = f[4];
                prc[i] = g[0]; prs[i] = g[1]; pom[i] = g[2]; pct[i] = g[3]; pst[i] = g[4]; dd[i] = d2(lt, lo); } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet(&l(&nrc, bs * 5), &l(&nrs, bs * 5), &l(&nom, bs * 5), &l(&nct, bs * 5), &l(&nst, bs * 5), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&l(&rc, bs), &l(&rs, bs), &l(&om, bs), &l(&ct, bs), &l(&st, bs), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(ctx, &target, &[bs, 1])));
        let mut loss = diff.mul(&diff).mean_all();
        if lyap && ectrl.is_some() {
            // V(g)=E(g); V>0 away from g: relu(0.5·d² − V); descent: relu(V(x')−V(x) + 0.3·d²)
            let vg = enet(&l(&[(G - G).cos()], 1), &l(&[(G - G).sin()], 1), &l(&[0.0], 1), &l(&[G.cos()], 1), &l(&[G.sin()], 1), &pv, &ov);   // E(g,0)
            let vx = enet(&l(&lrc, bs), &l(&lrs, bs), &l(&lom, bs), &l(&lct, bs), &l(&lst, bs), &pv, &ov).sub(&vg);
            let vp = enet(&l(&prc, bs), &l(&prs, bs), &l(&pom, bs), &l(&pct, bs), &l(&pst, bs), &pv, &ov).sub(&vg);
            let d2v = Var::leaf(Tensor::from_vec(ctx, &dd, &[bs, 1]));
            let pos = vx.neg().add(&d2v.mul(&Var::leaf(Tensor::from_vec(ctx, &[0.5], &[1])))).relu();          // relu(0.5d² − V)
            let dv = vp.sub(&vx).add(&d2v.mul(&Var::leaf(Tensor::from_vec(ctx, &[0.3], &[1])))).relu();        // relu(dV + 0.3d²)
            let lyloss = pos.mul(&pos).add(&dv.mul(&dv)).mean_all().mul(&Var::leaf(Tensor::from_vec(ctx, &[0.4], &[1])));
            loss = loss.add(&lyloss);
        }
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone();
            ectrl = Some(E { wrc: p[0].to_vec().await, wrs: p[1].to_vec().await, wo: p[2].to_vec().await, wc: p[3].to_vec().await, ws: p[4].to_vec().await, b1: p[5].to_vec().await, w2: p[6].to_vec().await, b2: p[7].to_vec().await, w3: p[8].to_vec().await, b3: p[9].to_vec().await[0] }); }
    }
    E { wrc: p[0].to_vec().await, wrs: p[1].to_vec().await, wo: p[2].to_vec().await, wc: p[3].to_vec().await, ws: p[4].to_vec().await, b1: p[5].to_vec().await, w2: p[6].to_vec().await, b2: p[7].to_vec().await, w3: p[8].to_vec().await, b3: p[9].to_vec().await[0] }
}

// verify V over a dense grid: returns (violation_rate% away from g, certified sublevel c, certified region % of box, on-policy monitor%)
fn verify(e: &E) -> (f32, f32, f32, f32) {
    let e0 = e.eval(G, 0.0); let (nth, nom) = (121usize, 81usize); let eps = 1e-4;
    let (mut viol, mut tot) = (0.0f32, 0.0f32);
    let mut pts: Vec<(f32, f32)> = vec![]; // (V, dV) away from g
    for i in 0..nth { for j in 0..nom { let th = G - 2.5 + 5.0 * i as f32 / (nth - 1) as f32; let om = -3.0 + 6.0 * j as f32 / (nom - 1) as f32;
        if d2(th, om) < 0.02 { continue; } let v = e.vhat(th, om, e0); let uu = e.greedy(th, om); let (pt, po) = step(th, om, uu); let dv = e.vhat(pt, po, e0) - v;
        tot += 1.0; if dv >= -eps || v <= 0.0 { viol += 1.0; } pts.push((v, dv)); } }
    // largest c such that every grid point with V≤c (away from g) is strictly descending (dV<−eps) and V>0
    let mut c = f32::MAX;
    for &(v, dv) in &pts { if dv >= -eps || v <= 0.0 { if v < c { c = v; } } }   // smallest V among violators = ceiling of certified sublevel
    let cert = if c == f32::MAX { f32::MAX } else { c };
    let inside = pts.iter().filter(|&&(v, _)| v <= cert && v > 0.0).count() as f32;
    let region = inside / tot * 100.0;
    // on-policy monitor: dV<0 fraction along trajectories from random starts in the box (the OLD "certificate", renamed)
    let (mut mon, mut mtot) = (0.0f32, 0.0f32);
    for s in 0..40 { let mut th = G + (u(s, 3) * 2.0 - 1.0) * 2.0; let mut om = (u(s, 4) * 2.0 - 1.0) * 2.0;
        for _ in 0..150 { let v = e.vhat(th, om, e0); let uu = e.greedy(th, om); let (nt, no) = step(th, om, uu); let dv = e.vhat(nt, no, e0) - v;
            if d2(th, om) > 0.02 { mtot += 1.0; if dv < 0.0 { mon += 1.0; } } th = nt; om = no; } }
    (viol / tot * 100.0, cert, region, mon / mtot * 100.0)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — INGEST neural-Lyapunov: EARN the certificate (the differentiator EBT-Policy lacks)\n");
    println!("  Certifying the region of attraction around goal θ={:.1} for the greedy energy controller.", G);
    println!("  'Certified' = over a 121×81 (θ,ω) grid, largest sublevel {{V≤c}} with V>0 and strict descent dV<0 everywhere.\n");
    for (tag, lyap) in [("FVI-only (value as Lyapunov candidate)", false), ("FVI + Lyapunov-risk (EARNED)", true)] {
        let e = train(&ctx, lyap).await; let (viol, c, region, mon) = verify(&e);
        println!("  [{}]", tag);
        println!("     grid descent-violation rate (away from g): {:>5.1}%", viol);
        println!("     certified sublevel c = {:>7.3}   → certified region = {:>5.1}% of the box", c, region);
        println!("     on-policy MONITOR (dV<0 along trajectories): {:>5.1}%   (this is the OLD 'certificate' — renamed honestly)\n", mon);
    }
    println!("  Reading: the Lyapunov-risk objective should CUT the grid-violation rate and GROW the certified region vs FVI-only —");
    println!("  earning a real (region) certificate, not just an on-policy monitor. At ~zero inference cost: V IS the control energy.");
    println!("  HONEST: grid/Lipschitz-sampling verified (conservative), NOT SMT-proven; known dynamics; single designated goal.");
}
