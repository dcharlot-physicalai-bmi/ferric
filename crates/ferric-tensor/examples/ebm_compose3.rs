//! EFA energy-first #42 — RELIABLE compositional language via STRUCTURED composition (fixing the fragile 1/2).
//!
//! ebm_compose2 composed direction×distance for 1 of 2 held-out commands — fragile, because the factor embeddings
//! entangled with the direction they were trained beside. The fix is structure-as-differentiation: decode each
//! factor into a learned GOAL CONTRIBUTION (a 2-vector), COMPOSE them additively into a goal vector g = g_dir + g_dist,
//! then build the energy on the state RELATIVE to that composed goal (cos(θ−g), sin(θ−g)) — the same relative
//! structuring that took control 50→89%. Compositionality now lives in the factor→goal decoder (each factor's meaning
//! learned once, added), and control gets the well-shaped relative energy. Trained on 4/6 combos, tested on 2 held-out.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_compose3 --release`
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

// CPU evaluator: composes g = g_dir + g_dist (2-vecs), then energy on state RELATIVE to g. Params gathered from tensors.
struct En { ed: Vec<f32>, em: Vec<f32>, wrc: Vec<f32>, wrs: Vec<f32>, wom: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl En {
    fn eval(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 {
        let gx = self.ed[dir * 2] + self.em[dist * 2]; let gy = self.ed[dir * 2 + 1] + self.em[dist * 2 + 1]; // composed goal vector
        let (c, s) = (th.cos(), th.sin());
        let rc = c * gx + s * gy; let rs = s * gx - c * gy; // state relative to the composed goal
        let mut h1 = [0.0f32; H];
        for j in 0..H { let p = self.b1[j] + rc * self.wrc[j] + rs * self.wrs[j] + om * self.wom[j] + c * self.wc[j] + s * self.ws[j]; h1[j] = (p.exp() + 1.0).ln(); }
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
    println!("  EFA energy-first — RELIABLE compositional language (structured composition: factors → goal vector → relative energy)\n");
    // params: dir_emb[2,2], dist_emb[3,2], then 5 rank-1 input weights [1,H] for (rc,rs,ω,cosθ,sinθ), b1[H], W2, b2, W3, b3
    let mk = || vec![
        Tensor::from_vec(&ctx, &randn(2 * 2, 20, 0.6), &[2, 2]), Tensor::from_vec(&ctx, &randn(3 * 2, 21, 0.6), &[3, 2]),
        Tensor::from_vec(&ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 23, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 24, 0.6), &[1, H]),
        Tensor::from_vec(&ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 26, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut p = mk(); let mut tgt = p.clone();
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    let selx = Tensor::from_vec(&ctx, &[1.0, 0.0], &[2, 1]); let sely = Tensor::from_vec(&ctx, &[0.0, 1.0], &[2, 1]);
    // energy over (state, direction-onehot, distance-onehot). state cols cth,sth,om given as [n,1] leaves.
    let enet = |cth: &Var, sth: &Var, om: &Var, ohd: &Var, ohm: &Var, pv: &[Var], ov: &Var, sx: &Var, sy: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let gv = ohd.matmul(&pv[0]).add(&ohm.matmul(&pv[1]));          // composed goal vector [n,2]
        let gx = gv.matmul(sx); let gy = gv.matmul(sy);               // [n,1] each
        let rc = cth.mul(&gx).add(&sth.mul(&gy));                     // cos(θ−g)·|g|
        let rs = sth.mul(&gx).sub(&cth.mul(&gy));                     // sin(θ−g)·|g|
        let pre = rc.matmul(&pv[2]).add(&rs.matmul(&pv[3])).add(&om.matmul(&pv[4])).add(&cth.matmul(&pv[5])).add(&sth.matmul(&pv[6])).add(&pv[7]);
        sp(sp(sp(pre).matmul(&pv[8]).add(&pv[9])).matmul(&pv[10]).add(&pv[11]))
    };
    for it in 0..14000 {
        let (mut cc, mut ss, mut oo) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]); let mut ohd = vec![0.0f32; bs * 2]; let mut ohm = vec![0.0f32; bs * 3];
        let (mut ncc, mut nss, mut noo) = (vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5]); let mut nohd = vec![0.0f32; bs * 5 * 2]; let mut nohm = vec![0.0f32; bs * 5 * 3]; let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let (dir, dist) = TRAIN[(u(sd, 4) * 4.0) as usize % 4]; let g = goal(dir, dist);
            let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0;
            cc[i] = th.cos(); ss[i] = th.sin(); oo[i] = om; ohd[i * 2 + dir] = 1.0; ohm[i * 3 + dist] = 1.0;
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om, uu); ncc[i * 5 + ai] = nt.cos(); nss[i * 5 + ai] = nt.sin(); noo[i * 5 + ai] = no;
                nohd[(i * 5 + ai) * 2 + dir] = 1.0; nohm[(i * 5 + ai) * 3 + dist] = 1.0; cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om * om + 0.01 * uu * uu; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1])); let l2 = |v: &[f32], r: usize, c: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, c]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let sxv = Var::leaf(selx.clone()); let syv = Var::leaf(sely.clone());
        let et = enet(&l(&ncc, bs * 5), &l(&nss, bs * 5), &l(&noo, bs * 5), &l2(&nohd, bs * 5, 2), &l2(&nohm, bs * 5, 3), &tv, &ov, &sxv, &syv).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&l(&cc, bs), &l(&ss, bs), &l(&oo, bs), &l2(&ohd, bs, 2), &l2(&ohm, bs, 3), &pv, &ov, &sxv, &syv);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
    }
    let e = En { ed: p[0].to_vec().await, em: p[1].to_vec().await, wrc: p[2].to_vec().await, wrs: p[3].to_vec().await, wom: p[4].to_vec().await, wc: p[5].to_vec().await, ws: p[6].to_vec().await, b1: p[7].to_vec().await, w2: p[8].to_vec().await, b2: p[9].to_vec().await, w3: p[10].to_vec().await, b3: p[11].to_vec().await[0] };

    println!("  command = DIRECTION × DISTANCE, composed into a goal vector.  trained on 4 combos; 2 HELD-OUT never seen.");
    println!("     command                 goal    energy-min err   control-reach   split");
    let names = |d: usize, m: usize| format!("{}·{}", if d == 0 { "left " } else { "right" }, ["near", "mid ", "far "][m]);
    let (mut tr_r, mut tr_c, mut tn) = (0.0f32, 0.0f32, 0.0f32); let (mut he_r, mut he_c) = (0.0f32, 0.0f32);
    for dir in 0..2 { for dist in 0..3 { let (rr, cc) = e.assess(dir, dist); let held = HELD.contains(&(dir, dist));
        println!("     {}   {:>5.1}    {:>6.3} rad     {:>4.0}%          {}", names(dir, dist), goal(dir, dist), rr, cc, if held { "HELD-OUT" } else { "train" });
        if held { he_r += rr; he_c += cc; } else { tr_r += rr; tr_c += cc; tn += 1.0; } } }
    println!("\n  TRAIN    avg energy-min err {:.3} rad, control {:.0}%", tr_r / tn, tr_c / tn);
    println!("  HELD-OUT avg energy-min err {:.3} rad, control {:.0}%   ← never trained on these commands", he_r / 2.0, he_c / 2.0);
    println!("\n  If BOTH held-out commands now compose (low err, high control), structured composition made compositional");
    println!("  language RELIABLE — factors decoded once, composed into a goal, controlled by a well-shaped energy.");
}
