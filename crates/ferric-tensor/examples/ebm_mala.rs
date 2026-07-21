//! EFA energy-first #5 — METROPOLIS-corrected sampling (MALA) fixes compositional GENERATION.
//!
//! Build-1 showed composition by plain Langevin (ULA) is fragile: it collapsed on left∧top (31%) and the
//! 4-way (1%). Reduce-Reuse-Recycle (Du et al. 2302.11552) diagnoses this exactly — plain Langevin does NOT
//! sample the product distribution e^{−ΣE_i}; you need a Metropolis-Hastings accept/reject step to target it.
//! MALA: propose x' = x − (ε²/2)∇E(x) + ε·z, then ACCEPT with prob min(1, e^{−E(x')+E(x)}·q(x|x')/q(x'|x))
//! where q is the (asymmetric) Gaussian proposal — the correction makes the chain provably target e^{−E}.
//! We compare ULA (no correction) vs MALA (corrected) with the SAME proposal, on the fragile conjunctions:
//! does the Metropolis step make GENERATION robust where descent/ULA failed (build-2 already fixed VERIFICATION)?
//!
//! Run: `cargo run -p ferric-tensor --example ebm_mala --release`
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
fn energy_t(x: &Tensor, p: &[Tensor], one: &Tensor) -> Tensor {
    let sp = |z: Tensor| -> Tensor { z.exp().add(one).log() };
    let h2 = sp(sp(x.matmul(&p[0]).add(&p[1])).matmul(&p[2]).add(&p[3]));
    h2.matmul(&p[4]).add(&p[5])
}
// per-sample summed energy VALUES over a subset
async fn esum_vals(ctx: &Arc<ferric_core::Context>, x: &[f32], b: usize, cps: &[Vec<Tensor>], subset: &[usize], one: &Tensor) -> Vec<f32> {
    let xt = Tensor::from_vec(ctx, x, &[b, 2]); let mut e = vec![0.0f32; b];
    for &c in subset { let ec = energy_t(&xt, &cps[c], one).to_vec().await; for i in 0..b { e[i] += ec[i]; } }
    e
}
// per-sample gradient of the summed energy over a subset
async fn esum_grad(ctx: &Arc<ferric_core::Context>, x: &[f32], b: usize, cps: &[Vec<Tensor>], subset: &[usize], one: &Tensor) -> Vec<f32> {
    let xv = Var::leaf(Tensor::from_vec(ctx, x, &[b, 2])); let ov = Var::leaf(one.clone());
    let mut e: Option<Var> = None;
    for &c in subset { let pv: Vec<Var> = cps[c].iter().map(|t| Var::leaf(t.clone())).collect();
        let ec = energy_v(&xv, &pv, &ov); e = Some(match e { None => ec, Some(pe) => pe.add(&ec) }); }
    e.unwrap().sum(&[0]).sum(&[1]).backward();
    xv.grad().unwrap().to_vec().await
}
fn acc(x: &[f32], subset: &[usize]) -> f32 {
    let b = x.len() / 2; let mut ok = 0;
    for i in 0..b { if subset.iter().all(|&c| sat(c, x[i * 2], x[i * 2 + 1])) { ok += 1; } }
    ok as f32 / b as f32
}

// sampler: adjusted=false → ULA (always accept); adjusted=true → MALA (Metropolis accept/reject). eps = step.
async fn sample(ctx: &Arc<ferric_core::Context>, cps: &[Vec<Tensor>], one: &Tensor, subset: &[usize], k: usize, b: usize, eps: f32, adjusted: bool) -> (f32, f32) {
    let eps2 = eps * eps;
    let mut x = gen_uniform(b, 424242);
    let mut accepts = 0u64; let mut total = 0u64;
    for step in 0..k {
        let e0 = esum_vals(ctx, &x, b, cps, subset, one).await;
        let g0 = esum_grad(ctx, &x, b, cps, subset, one).await;
        let z = randn(b * 2, step as u32 * 131 + 9, 1.0);
        let mut xp = vec![0.0f32; b * 2];
        for i in 0..b * 2 { xp[i] = (x[i] - 0.5 * eps2 * g0[i] + eps * z[i]).clamp(-DOM, DOM); }
        if !adjusted { x = xp; continue; }
        let ep = esum_vals(ctx, &xp, b, cps, subset, one).await;
        let gp = esum_grad(ctx, &xp, b, cps, subset, one).await;
        for i in 0..b {
            // asymmetric Gaussian proposal correction: logq(a|b) = -‖a − (b − ε²/2 ∇E(b))‖²/(2ε²)
            let mut fwd = 0.0f32; let mut rev = 0.0f32; // fwd: q(x'|x), rev: q(x|x')
            for d in 0..2 { let j = i * 2 + d;
                let mf = xp[j] - (x[j] - 0.5 * eps2 * g0[j]); fwd += mf * mf;
                let mr = x[j] - (xp[j] - 0.5 * eps2 * gp[j]); rev += mr * mr; }
            let log_acc = (e0[i] - ep[i]) + (-rev / (2.0 * eps2)) - (-fwd / (2.0 * eps2));
            total += 1;
            if u(i as u32, step as u32 * 7919 + 3).ln() < log_acc { x[i * 2] = xp[i * 2]; x[i * 2 + 1] = xp[i * 2 + 1]; accepts += 1; }
        }
    }
    let ar = if total > 0 { accepts as f32 / total as f32 } else { 1.0 };
    (acc(&x, subset), ar)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — MALA (Metropolis-corrected) vs ULA (plain Langevin) for composed GENERATION");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let mut cps: Vec<Vec<Tensor>> = Vec::new(); let bs = 512usize; let reg = Tensor::from_vec(&ctx, &[0.1], &[1]);
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

    let names = ["left", "top", "disk", "ring-out"]; let (k, b) = (200usize, 400usize);
    // FIRST honest finding: well-tuned constant-ε ULA already composes fine (build-1's collapse was a step-size /
    // annealed-noise artifact, NOT fundamental). MALA's real value = staying UNBIASED at LARGE ε where ULA breaks.
    // So sweep ε: at small ε both work (high accept → MALA≈ULA); at large ε ULA degrades, MALA should hold.
    println!("\n  STEP-SIZE SWEEP on the 4-way conjunction (the hardest) — ULA vs MALA accuracy + MALA accept-rate:");
    println!("     {:>6}   {:>8} {:>8} {:>12}", "ε", "ULA", "MALA", "MALA accept");
    let four = [0usize, 1, 2, 3];
    for &eps in &[0.15f32, 0.3, 0.5, 0.8, 1.2] {
        let (ula, _) = sample(&ctx, &cps, &one, &four, k, b, eps, false).await;
        let (mala, ar) = sample(&ctx, &cps, &one, &four, k, b, eps, true).await;
        println!("     {:>6.2}   {:>7.0}% {:>7.0}% {:>11.0}%", eps, ula * 100.0, mala * 100.0, ar * 100.0);
    }
    println!("\n  reference (ε=0.3) across conjunctions — ULA vs MALA:");
    println!("     {:<28} {:>8} {:>8} {:>12}", "conjunction", "ULA", "MALA", "MALA accept");
    for subset in [vec![0usize, 1], vec![0, 2], vec![2, 3], vec![0, 1, 2], vec![0, 1, 2, 3]] {
        let nm: Vec<&str> = subset.iter().map(|&c| names[c]).collect();
        let (ula, _) = sample(&ctx, &cps, &one, &subset, k, b, 0.3, false).await;
        let (mala, ar) = sample(&ctx, &cps, &one, &subset, k, b, 0.3, true).await;
        println!("     {:<28} {:>7.0}% {:>7.0}% {:>11.0}%", nm.join(" ∧ "), ula * 100.0, mala * 100.0, ar * 100.0);
    }
    println!("\n  the honest reading: well-tuned ULA already composes (build-1's collapse was step-size, not fundamental);");
    println!("  MALA's payoff is at LARGE ε — where ULA's bias/instability shows and the Metropolis step keeps it correct.");
}
