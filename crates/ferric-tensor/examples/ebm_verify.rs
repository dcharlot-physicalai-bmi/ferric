//! EFA energy-first #2 — ENERGY-AS-VERIFIER best-of-N (native System-2, robust where Langevin was fragile).
//!
//! A learned energy E(x,c) IS a verifier: low = valid. So instead of DESCENDING the composed energy (Langevin,
//! which we saw is sampler-fragile — energy-balance breaks composition), just GENERATE N candidates and SELECT
//! the min-energy one — no MCMC, nothing to destabilize. This is EBT self-verification (best-of-N by energy,
//! +10–14%) and the generator–verifier gap (verifying is easier than generating): the verifier is NATIVE to
//! the model, needs no separately-trained reward model. We reuse the 4 composed concept-energies and ask:
//! does best-of-N-by-energy (a) beat random selection, (b) improve with N (test-time verification compute),
//! and (c) FIX the fragile conjunctions (left∧top, the 4-way) that Langevin composition collapsed on?
//!
//! Run: `cargo run -p ferric-tensor --example ebm_verify --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const DOM: f32 = 2.5;
const H: usize = 64;
const NC: usize = 4;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn sat(c: usize, x0: f32, x1: f32) -> bool { match c { 0 => x0 < 0.0, 1 => x1 > 0.0, 2 => x0 * x0 + x1 * x1 < 2.0, _ => x0 * x0 + x1 * x1 > 0.5 } }
fn gen_concept(c: usize, m: usize, seed: u32) -> Vec<f32> {
    let mut o = Vec::with_capacity(m * 2); let mut s = seed;
    while o.len() < m * 2 { let x0 = (u(s, 1) * 2.0 - 1.0) * DOM; let x1 = (u(s, 2) * 2.0 - 1.0) * DOM; s = s.wrapping_add(2749); if sat(c, x0, x1) { o.push(x0); o.push(x1); } }
    o
}
fn gen_uniform(m: usize, seed: u32) -> Vec<f32> { (0..m * 2).map(|i| (u(i as u32, seed) * 2.0 - 1.0) * DOM).collect() }

fn energy_v(xv: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| -> Var { z.exp().add(one).log() };
    let h2 = sp(sp(xv.matmul(&p[0]).add(&p[1])).matmul(&p[2]).add(&p[3]));
    h2.matmul(&p[4]).add(&p[5])
}
// tensor forward (no autograd) — verification only scores candidates, no gradient needed
fn energy_t(x: &Tensor, p: &[Tensor], one: &Tensor) -> Tensor {
    let sp = |z: Tensor| -> Tensor { z.exp().add(one).log() };
    let h2 = sp(sp(x.matmul(&p[0]).add(&p[1])).matmul(&p[2]).add(&p[3]));
    h2.matmul(&p[4]).add(&p[5])
}
// best-of-N: T trials, each draws N uniform candidates, selects argmin(summed energy), checks conjunction
async fn bestof(ctx: &Arc<ferric_core::Context>, cps: &[Vec<Tensor>], one: &Tensor, subset: &[usize], nn: usize, t: usize) -> f32 {
    let x = gen_uniform(t * nn, 777 + nn as u32 * 13 + subset.iter().sum::<usize>() as u32 * 131);
    let xt = Tensor::from_vec(ctx, &x, &[t * nn, 2]);
    let mut e = vec![0.0f32; t * nn];
    for &c in subset { let ec = energy_t(&xt, &cps[c], one).to_vec().await; for i in 0..t * nn { e[i] += ec[i]; } }
    let mut ok = 0;
    for tr in 0..t {
        let (mut bi, mut be) = (0usize, f32::MAX);
        for j in 0..nn { let idx = tr * nn + j; if e[idx] < be { be = e[idx]; bi = idx; } }
        let (x0, x1) = (x[bi * 2], x[bi * 2 + 1]);
        if subset.iter().all(|&c| sat(c, x0, x1)) { ok += 1; }
    }
    ok as f32 / t as f32
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — ENERGY-AS-VERIFIER best-of-N (2D, {NC} concepts)");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);

    // train the same 4 concept energies (contrastive, uniform negatives)
    let mut cps: Vec<Vec<Tensor>> = Vec::new();
    let bs = 512usize; let reg = Tensor::from_vec(&ctx, &[0.1], &[1]);
    for c in 0..NC {
        let mut p = vec![
            Tensor::from_vec(&ctx, &randn(2 * H, 100 + c as u32, 1.0 / 1.5), &[2, H]), Tensor::zeros(&ctx, &[H]),
            Tensor::from_vec(&ctx, &randn(H * H, 200 + c as u32, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
            Tensor::from_vec(&ctx, &randn(H, 300 + c as u32, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
        ];
        let mut adam = Adam::new(&p, 0.002);
        for step in 0..2500 {
            let pos = gen_concept(c, bs, step as u32 * 17 + 1 + c as u32 * 7919);
            let neg = gen_uniform(bs, step as u32 * 29 + 5 + c as u32 * 104729);
            let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
            let ep = energy_v(&Var::leaf(Tensor::from_vec(&ctx, &pos, &[bs, 2])), &pv, &ov);
            let en = energy_v(&Var::leaf(Tensor::from_vec(&ctx, &neg, &[bs, 2])), &pv, &ov);
            let loss = ep.mean_all().sub(&en.mean_all()).add(&ep.mul(&ep).mean_all().add(&en.mul(&en).mean_all()).mul(&Var::leaf(reg.clone())));
            loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); adam.step(&mut p, &g);
        }
        cps.push(p);
    }

    let names = ["left", "top", "disk", "ring-out"];
    println!("\n  best-of-N by energy — conjunction accuracy vs N (N=1 = random pick = region base-rate), 400 trials:");
    println!("     {:<28} {:>5} {:>5} {:>5} {:>6} {:>6}", "conjunction", "N=1", "N=4", "N=16", "N=64", "N=256");
    for subset in [vec![0usize, 1], vec![0, 2], vec![2, 3], vec![0, 1, 2], vec![0, 1, 2, 3]] {
        let nm: Vec<&str> = subset.iter().map(|&c| names[c]).collect();
        let mut r = Vec::new();
        for &nn in &[1usize, 4, 16, 64, 256] { r.push(bestof(&ctx, &cps, &one, &subset, nn, 400).await); }
        println!("     {:<28} {:>4.0}% {:>4.0}% {:>4.0}% {:>5.0}% {:>5.0}%", nm.join(" ∧ "), r[0] * 100.0, r[1] * 100.0, r[2] * 100.0, r[3] * 100.0, r[4] * 100.0);
    }
    println!("\n  best-of-N rises with N (test-time verification compute) and — unlike Langevin — DOESN'T collapse on the");
    println!("  fragile conjunctions (left∧top, the 4-way): verifying is more robust than sampling the composed landscape.");
}
