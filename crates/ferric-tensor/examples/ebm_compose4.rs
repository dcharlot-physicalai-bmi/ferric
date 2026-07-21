//! EFA energy-first #43 — RELIABLE compositional language via the RIGHT (MULTIPLICATIVE) structure.
//!
//! ebm_compose3 showed additive composition made held-out WORSE — because the task's composition is MULTIPLICATIVE
//! (goal angle = direction_sign × distance_magnitude). This matches the structure to the task, with no trig-in-graph:
//! learn direction → a SIGN scalar a(dir), distance → a MAGNITUDE scalar b(dist), compose by MULTIPLICATION
//! gθ = a·b (an in-graph op), then feed the angular offset δ = θ − gθ to the energy. Multiplicative composition of
//! independently-learned factors should obey UNSEEN (direction,distance) commands. Trained on 4/6 combos; 2 held-out.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_compose4 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const DIST: [f32; 3] = [0.5, 1.5, 2.5];
const TRAIN: [(usize, usize); 4] = [(0, 0), (0, 2), (1, 0), (1, 1)];
const HELD: [(usize, usize); 2] = [(0, 1), (1, 2)];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }
fn goal(dir: usize, dist: usize) -> f32 { (if dir == 0 { -1.0 } else { 1.0 }) * DIST[dist] }

struct En { da: Vec<f32>, db: Vec<f32>, wd: Vec<f32>, wo: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl En {
    fn eval(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 {
        let gth = self.da[dir] * self.db[dist];                       // MULTIPLICATIVE composition: sign × magnitude
        let delta = th - gth;                                         // angular offset to the composed goal
        let (c, s) = (th.cos(), th.sin());
        let mut h1 = [0.0f32; H];
        for j in 0..H { let p = self.b1[j] + delta * self.wd[j] + om * self.wo[j] + c * self.wc[j] + s * self.ws[j]; h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn greedy(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let e = self.eval(nt, no, dir, dist); if e < be { be = e; bu = uu; } } bu }
    fn assess(&self, dir: usize, dist: usize) -> (f32, f32) {
        let g = goal(dir, dist);
        let mut bth = 0.0; let mut be = f32::MAX; for gi in 0..361 { let th = -PI + gi as f32 / 360.0 * 2.0 * PI; let en = self.eval(th, 0.0, dir, dist); if en < be { be = en; bth = th; } }
        let mut reach = 0; let nn = 60; for i in 0..nn { let mut th = (u(700 + (dir * 200 + dist * 60 + i) as u32, 1) * 2.0 - 1.0) * PI; let mut om = 0.0f32;
            for _ in 0..200 { let uu = self.greedy(th, om, dir, dist); let (nt, no) = step(th, om, uu); th = nt; om = no; if wrap(th - g).abs() < 0.2 && om.abs() < 0.5 { break; } }
            if wrap(th - g).abs() < 0.3 && om.abs() < 0.6 { reach += 1; } }
        (wrap(bth - g).abs(), reach as f32 / nn as f32 * 100.0)
    }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — RELIABLE compositional language via MULTIPLICATIVE composition (sign × magnitude)\n");
    // params: dir_sign[2,1], dist_mag[3,1], then rank-1 input weights for (δ,ω,cosθ,sinθ), b1[H], W2, b2, W3, b3
    let mk = || vec![
        Tensor::from_vec(&ctx, &[-0.5, 0.5], &[2, 1]), Tensor::from_vec(&ctx, &[1.0, 1.5, 2.0], &[3, 1]), // init near sign/magnitude
        Tensor::from_vec(&ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 23, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 24, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut p = mk(); let mut tgt = p.clone();
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    let enet = |th: &Var, om: &Var, cth: &Var, sth: &Var, ohd: &Var, ohm: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let a = ohd.matmul(&pv[0]); let b = ohm.matmul(&pv[1]);       // sign a(dir), magnitude b(dist)
        let gth = a.mul(&b);                                          // MULTIPLICATIVE composition [n,1]
        let delta = th.sub(&gth);                                     // angular offset
        let pre = delta.matmul(&pv[2]).add(&om.matmul(&pv[3])).add(&cth.matmul(&pv[4])).add(&sth.matmul(&pv[5])).add(&pv[6]);
        sp(sp(sp(pre).matmul(&pv[7]).add(&pv[8])).matmul(&pv[9]).add(&pv[10]))
    };
    for it in 0..14000 {
        let (mut th_, mut om_, mut c_, mut s_) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]); let mut ohd = vec![0.0f32; bs * 2]; let mut ohm = vec![0.0f32; bs * 3];
        let (mut nth, mut nom, mut nc, mut ns) = (vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5]); let mut nohd = vec![0.0f32; bs * 5 * 2]; let mut nohm = vec![0.0f32; bs * 5 * 3]; let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let (dir, dist) = TRAIN[(u(sd, 4) * 4.0) as usize % 4]; let g = goal(dir, dist);
            let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0;
            th_[i] = th; om_[i] = om; c_[i] = th.cos(); s_[i] = th.sin(); ohd[i * 2 + dir] = 1.0; ohm[i * 3 + dist] = 1.0;
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om, uu); nth[i * 5 + ai] = nt; nom[i * 5 + ai] = no; nc[i * 5 + ai] = nt.cos(); ns[i * 5 + ai] = nt.sin();
                nohd[(i * 5 + ai) * 2 + dir] = 1.0; nohm[(i * 5 + ai) * 3 + dist] = 1.0; cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om * om + 0.01 * uu * uu; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1])); let l2 = |v: &[f32], r: usize, c: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, c]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet(&l(&nth, bs * 5), &l(&nom, bs * 5), &l(&nc, bs * 5), &l(&ns, bs * 5), &l2(&nohd, bs * 5, 2), &l2(&nohm, bs * 5, 3), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&l(&th_, bs), &l(&om_, bs), &l(&c_, bs), &l(&s_, bs), &l2(&ohd, bs, 2), &l2(&ohm, bs, 3), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
    }
    let e = En { da: p[0].to_vec().await, db: p[1].to_vec().await, wd: p[2].to_vec().await, wo: p[3].to_vec().await, wc: p[4].to_vec().await, ws: p[5].to_vec().await, b1: p[6].to_vec().await, w2: p[7].to_vec().await, b2: p[8].to_vec().await, w3: p[9].to_vec().await, b3: p[10].to_vec().await[0] };

    println!("  learned factors:  direction signs a(left,right) = ({:+.2}, {:+.2});  distance mags b(near,mid,far) = ({:.2}, {:.2}, {:.2})", e.da[0], e.da[1], e.db[0], e.db[1], e.db[2]);
    println!("  goal angle composed MULTIPLICATIVELY as a(dir)·b(dist).  trained on 4 combos; 2 HELD-OUT never seen.\n");
    println!("     command                 goal    energy-min err   control-reach   split");
    let names = |d: usize, m: usize| format!("{}·{}", if d == 0 { "left " } else { "right" }, ["near", "mid ", "far "][m]);
    let (mut tr_r, mut tr_c, mut tn) = (0.0f32, 0.0f32, 0.0f32); let (mut he_r, mut he_c) = (0.0f32, 0.0f32);
    for dir in 0..2 { for dist in 0..3 { let (rr, cc) = e.assess(dir, dist); let held = HELD.contains(&(dir, dist));
        println!("     {}   {:>5.1}    {:>6.3} rad     {:>4.0}%          {}", names(dir, dist), goal(dir, dist), rr, cc, if held { "HELD-OUT" } else { "train" });
        if held { he_r += rr; he_c += cc; } else { tr_r += rr; tr_c += cc; tn += 1.0; } } }
    println!("\n  TRAIN    avg energy-min err {:.3} rad, control {:.0}%", tr_r / tn, tr_c / tn);
    println!("  HELD-OUT avg energy-min err {:.3} rad, control {:.0}%   ← never trained on these commands", he_r / 2.0, he_c / 2.0);
    println!("\n  If BOTH held-out commands now compose, the RIGHT (multiplicative) structure made compositional language");
    println!("  reliable — the lever was matching the inductive bias to the task's generative form, not more scale.");
}
