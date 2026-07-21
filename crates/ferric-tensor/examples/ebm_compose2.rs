//! EFA energy-first #41 — COMPOSITIONAL language: does the energy COMPOSE factors to obey UNSEEN commands?
//!
//! ebm_lang decoded 6 atomic symbols. Real language COMPOSES. Here an instruction is two factors — DIRECTION
//! (left/right) × DISTANCE (near/mid/far) → a goal angle. The energy sees TWO learned embeddings (one per factor)
//! and is trained on only 4 of the 6 (direction,distance) combinations. We then test the 2 HELD-OUT combinations
//! it never saw. If the energy places the correct goal-attractor and controls to it for the held-out commands, it
//! has COMPOSED the factors — genuine compositional generalization, not a lookup table. One structured energy;
//! compositional language; a body. Learned by fitted value iteration (score-first, no partition function).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_compose2 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const D: usize = 8; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const DIST: [f32; 3] = [0.5, 1.5, 2.5]; // near / mid / far
// train on 4 (dir,dist) combos; HOLD OUT (left,mid) and (right,far) — each factor value still appears elsewhere in training
const TRAIN: [(usize, usize); 4] = [(0, 0), (0, 2), (1, 0), (1, 1)];
const HELD: [(usize, usize); 2] = [(0, 1), (1, 2)];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }
fn goal(dir: usize, dist: usize) -> f32 { (if dir == 0 { -1.0 } else { 1.0 }) * DIST[dist] } // direction × distance
fn sfeat(th: f32, om: f32) -> [f32; 3] { [th.cos(), th.sin(), om] }

struct En { ws: Vec<f32>, wd: Vec<f32>, wm: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32, ed: Vec<f32>, em: Vec<f32> }
impl En {
    fn eval(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 {
        let sf = sfeat(th, om); let mut h1 = [0.0f32; H];
        for j in 0..H { let mut p = self.b1[j];
            for k in 0..3 { p += sf[k] * self.ws[k * H + j]; }
            for k in 0..D { p += self.ed[dir * D + k] * self.wd[k * H + j] + self.em[dist * D + k] * self.wm[k * H + j]; }
            h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn greedy(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let e = self.eval(nt, no, dir, dist); if e < be { be = e; bu = uu; } } bu }
    // (remember_err, control_reach%) for a given (dir,dist) command
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
    println!("  EFA energy-first — COMPOSITIONAL language: compose DIRECTION × DISTANCE, obey UNSEEN commands\n");
    let mk = || vec![
        Tensor::from_vec(&ctx, &randn(3 * H, 10, 0.6), &[3, H]),
        Tensor::from_vec(&ctx, &randn(D * H, 11, 0.6), &[D, H]), Tensor::from_vec(&ctx, &randn(D * H, 15, 0.6), &[D, H]),
        Tensor::zeros(&ctx, &[H]), Tensor::from_vec(&ctx, &randn(H * H, 12, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 13, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
        Tensor::from_vec(&ctx, &randn(2 * D, 14, 0.5), &[2, D]), Tensor::from_vec(&ctx, &randn(3 * D, 16, 0.5), &[3, D]),
    ];
    let mut p = mk(); let mut tgt = p.clone();
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    let enet = |sf: &Var, ohd: &Var, ohm: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let ied = ohd.matmul(&pv[8]); let iem = ohm.matmul(&pv[9]);
        let h1 = sp(sf.matmul(&pv[0]).add(&ied.matmul(&pv[1])).add(&iem.matmul(&pv[2])).add(&pv[3]));
        sp(sp(h1.matmul(&pv[4]).add(&pv[5])).matmul(&pv[6]).add(&pv[7]))
    };
    for it in 0..14000 {
        let mut sfc = vec![0.0f32; bs * 3]; let mut ohdc = vec![0.0f32; bs * 2]; let mut ohmc = vec![0.0f32; bs * 3];
        let mut sfn = vec![0.0f32; bs * 5 * 3]; let mut ohdn = vec![0.0f32; bs * 5 * 2]; let mut ohmn = vec![0.0f32; bs * 5 * 3]; let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32;
            let (dir, dist) = TRAIN[(u(sd, 4) * 4.0) as usize % 4]; let g = goal(dir, dist); // TRAIN combos only
            let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0;
            let f = sfeat(th, om); for k in 0..3 { sfc[i * 3 + k] = f[k]; } ohdc[i * 2 + dir] = 1.0; ohmc[i * 3 + dist] = 1.0;
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om, uu); let nf = sfeat(nt, no);
                for k in 0..3 { sfn[(i * 5 + ai) * 3 + k] = nf[k]; } ohdn[(i * 5 + ai) * 2 + dir] = 1.0; ohmn[(i * 5 + ai) * 3 + dist] = 1.0;
                let _ = nt; cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om * om + 0.01 * uu * uu; } }
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet(&Var::leaf(Tensor::from_vec(&ctx, &sfn, &[bs * 5, 3])), &Var::leaf(Tensor::from_vec(&ctx, &ohdn, &[bs * 5, 2])), &Var::leaf(Tensor::from_vec(&ctx, &ohmn, &[bs * 5, 3])), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs];
        for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&Var::leaf(Tensor::from_vec(&ctx, &sfc, &[bs, 3])), &Var::leaf(Tensor::from_vec(&ctx, &ohdc, &[bs, 2])), &Var::leaf(Tensor::from_vec(&ctx, &ohmc, &[bs, 3])), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
    }
    let e = En { ws: p[0].to_vec().await, wd: p[1].to_vec().await, wm: p[2].to_vec().await, b1: p[3].to_vec().await, w2: p[4].to_vec().await, b2: p[5].to_vec().await, w3: p[6].to_vec().await, b3: p[7].to_vec().await[0], ed: p[8].to_vec().await, em: p[9].to_vec().await };

    println!("  command = DIRECTION × DISTANCE.  trained on 4 combos; the 2 HELD-OUT were never seen.");
    println!("     command                 goal    energy-min err   control-reach   split");
    let names = |d: usize, m: usize| format!("{}·{}", if d == 0 { "left " } else { "right" }, ["near", "mid ", "far "][m]);
    let (mut tr_r, mut tr_c, mut tn) = (0.0f32, 0.0f32, 0.0f32); let (mut he_r, mut he_c) = (0.0f32, 0.0f32);
    for dir in 0..2 { for dist in 0..3 { let (rr, cc) = e.assess(dir, dist); let held = HELD.contains(&(dir, dist));
        println!("     {}   {:>5.1}    {:>6.3} rad     {:>4.0}%          {}", names(dir, dist), goal(dir, dist), rr, cc, if held { "HELD-OUT" } else { "train" });
        if held { he_r += rr; he_c += cc; } else { tr_r += rr; tr_c += cc; tn += 1.0; } } }
    println!("\n  TRAIN    avg energy-min err {:.3} rad, control {:.0}%", tr_r / tn, tr_c / tn);
    println!("  HELD-OUT avg energy-min err {:.3} rad, control {:.0}%   ← never trained on these commands", he_r / 2.0, he_c / 2.0);
    println!("\n  If the HELD-OUT commands place the right goal-attractor and control to it, the energy COMPOSED direction");
    println!("  and distance it never saw combined — compositional language in one structured energy, on a body.");
}
