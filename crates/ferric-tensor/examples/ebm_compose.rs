//! EFA energy-first → ZERO-SHOT composition by energy summation (the thing transformers can't do).
//!
//! An EBM defines p(x|c) ∝ e^{−E(x|c)}, so independent concept-energies combine analytically: the AND of
//! concepts is p ∝ e^{−Σ E_i} — literally SUM the energies, then sample the joint minimum by energy descent
//! (Langevin). Because each E_i is an independent constraint learned on its OWN disjoint data, summing them
//! builds a landscape whose minima lie in an intersection the model NEVER saw jointly → zero-shot logical
//! composition, no retraining. A feedforward net conditioned on a concept-id has no mechanism for a novel
//! conjunction; the EBM composes it for free. (Du & Mordatch 1903.08689; Du/Li/Mordatch 2004.06030.)
//!
//! We train 4 concept-energies on disjoint 2D data, then measure zero-shot conjunction accuracy for 2/3/4-way
//! combos — vs a no-composition ablation (sample one concept only) — and the THINKING axis (accuracy vs #
//! energy-descent steps K), which a single feed-forward pass structurally cannot have. Uses Ferric's
//! input-gradient autograd: energy descent on the SAMPLE, the same mechanism as EFA's planner, over concepts.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_compose --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const DOM: f32 = 2.5;
const H: usize = 64;
const NC: usize = 4;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn sat(c: usize, x0: f32, x1: f32) -> bool {
    match c { 0 => x0 < 0.0, 1 => x1 > 0.0, 2 => x0 * x0 + x1 * x1 < 2.0, _ => x0 * x0 + x1 * x1 > 0.5 }
}
fn gen_concept(c: usize, m: usize, seed: u32) -> Vec<f32> {
    let mut o = Vec::with_capacity(m * 2); let mut s = seed;
    while o.len() < m * 2 { let x0 = (u(s, 1) * 2.0 - 1.0) * DOM; let x1 = (u(s, 2) * 2.0 - 1.0) * DOM; s = s.wrapping_add(2749); if sat(c, x0, x1) { o.push(x0); o.push(x1); } }
    o
}
fn gen_uniform(m: usize, seed: u32) -> Vec<f32> { (0..m * 2).map(|i| (u(i as u32, seed) * 2.0 - 1.0) * DOM).collect() }
// negatives = the concept's COMPLEMENT (harder negatives → crisp decision boundary; fixes soft half-plane energies)
fn gen_complement(c: usize, m: usize, seed: u32) -> Vec<f32> {
    let mut o = Vec::with_capacity(m * 2); let mut s = seed;
    while o.len() < m * 2 { let x0 = (u(s, 1) * 2.0 - 1.0) * DOM; let x1 = (u(s, 2) * 2.0 - 1.0) * DOM; s = s.wrapping_add(2749); if !sat(c, x0, x1) { o.push(x0); o.push(x1); } }
    o
}

// softplus MLP energy E(x): 2 → H → H → 1 (softplus = log(1+e^z), smooth grads for Langevin)
fn energy(xv: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| -> Var { z.exp().add(one).log() };
    let h1 = sp(xv.matmul(&p[0]).add(&p[1]));
    let h2 = sp(h1.matmul(&p[2]).add(&p[3]));
    h2.matmul(&p[4]).add(&p[5])
}

// sample from the SUM of a subset of concept-energies via Langevin (energy descent on the sample)
async fn sample(ctx: &Arc<ferric_core::Context>, cps: &[Vec<Tensor>], one: &Tensor, subset: &[usize], k: usize, b: usize) -> Vec<f32> {
    let mut x = gen_uniform(b, 424242);
    let (alpha, sig0) = (0.10f32, 0.10f32);
    for step in 0..k {
        let xv = Var::leaf(Tensor::from_vec(ctx, &x, &[b, 2]));
        let ov = Var::leaf(one.clone());
        let mut e: Option<Var> = None;
        for &c in subset { let pv: Vec<Var> = cps[c].iter().map(|t| Var::leaf(t.clone())).collect();
            let ec = energy(&xv, &pv, &ov); e = Some(match e { None => ec, Some(pe) => pe.add(&ec) }); }
        e.unwrap().sum(&[0]).sum(&[1]).backward(); // sum over batch → per-sample grad in xv.grad()
        let g = xv.grad().unwrap().to_vec().await;
        let sig = sig0 * (1.0 - step as f32 / k as f32);
        let nz = randn(b * 2, step as u32 * 131 + 9, 1.0);
        for i in 0..b * 2 { x[i] = (x[i] - alpha * g[i] + sig * nz[i]).clamp(-DOM, DOM); }
    }
    x
}

fn acc(x: &[f32], subset: &[usize]) -> (f32, Vec<f32>) {
    let b = x.len() / 2; let mut all = 0; let mut per = vec![0f32; subset.len()];
    for i in 0..b { let (x0, x1) = (x[i * 2], x[i * 2 + 1]); let mut ok = true;
        for (j, &c) in subset.iter().enumerate() { if sat(c, x0, x1) { per[j] += 1.0; } else { ok = false; } }
        if ok { all += 1; } }
    (all as f32 / b as f32, per.iter().map(|v| v / b as f32).collect())
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — ZERO-SHOT composition by energy summation (2D, {NC} concepts, softplus-MLP energies)");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);

    // ---- train NC concept energies on DISJOINT data (contrastive: E low on concept, high on uniform) ----
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
            // uniform negatives keep the GLOBAL energy landscape balanced across concepts — critical for the
            // summed (composed) landscape to keep its minimum at the intersection. Sharper boundary negatives
            // (concept complement) were tried and BROKE composition (energy-balance is finicky — the classic
            // EBM fragility): 3-way conjunction collapsed 73%→0% and thinking inverted. Uniform wins here.
            let neg = gen_uniform(bs, step as u32 * 29 + 5 + c as u32 * 104729);
            let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
            let ov = Var::leaf(one.clone());
            let ep = energy(&Var::leaf(Tensor::from_vec(&ctx, &pos, &[bs, 2])), &pv, &ov);
            let en = energy(&Var::leaf(Tensor::from_vec(&ctx, &neg, &[bs, 2])), &pv, &ov);
            let loss = ep.mean_all().sub(&en.mean_all()).add(&ep.mul(&ep).mean_all().add(&en.mul(&en).mean_all()).mul(&Var::leaf(reg.clone())));
            loss.backward();
            let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); adam.step(&mut p, &g);
        }
        cps.push(p);
    }

    let names = ["left", "top", "disk", "ring-out"];
    println!("\n  zero-shot conjunction accuracy (K=200 energy-descent steps, 400 samples):");
    for subset in [vec![0usize, 1], vec![0, 2], vec![2, 3], vec![0, 1, 2], vec![0, 1, 2, 3]] {
        let x = sample(&ctx, &cps, &one, &subset, 200, 400).await; let (a, per) = acc(&x, &subset);
        let nm: Vec<&str> = subset.iter().map(|&c| names[c]).collect();
        println!("     {:<28} all {:>4.0}%   (per-concept {:?})", nm.join(" ∧ "), a * 100.0, per.iter().map(|v| (v * 100.0) as i32).collect::<Vec<_>>());
    }

    let x1 = sample(&ctx, &cps, &one, &[0usize], 200, 400).await; let (a1, _) = acc(&x1, &[0, 1, 2]);
    println!("\n  ablation — sample ONE concept (left) only, scored on left∧top∧disk: {:>4.0}%  (composition is what buys the conjunction)", a1 * 100.0);

    println!("\n  THINKING — left∧top∧disk accuracy vs energy-descent steps K:");
    for &k in &[5usize, 20, 50, 100, 200] { let x = sample(&ctx, &cps, &one, &[0, 1, 2], k, 400).await; let (a, _) = acc(&x, &[0, 1, 2]); println!("     K={:<4}  {:>4.0}%", k, a * 100.0); }
    println!("\n  high zero-shot conjunction + rising-with-K → energy summation composes novel goals the model never");
    println!("  saw jointly, and thinks longer to satisfy them — a structural edge feed-forward/AR models lack.");
}
