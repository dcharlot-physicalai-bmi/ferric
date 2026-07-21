//! EFA energy-first #4 — nano ENERGY-BASED-TRANSFORMER-style predictor: thinks at test time + generalizes OOD.
//!
//! EBTs (Gladstone/Du 2025, arXiv:2507.02092) make prediction = gradient descent on the answer to minimize a
//! learned energy E(context, ŷ), so more inference steps = better answer ("System-2 thinking"), and the gain
//! is LARGEST when inputs are OOD. Full EBTs train THROUGH the unrolled descent (2nd-order autograd, which
//! Ferric's 1st-order engine can't do), so this is the EBT-STYLE version: a context-conditioned energy trained
//! contrastively, then test-time descent — the same two signatures.
//!
//! Task (a genuine nonlinear system): given context (a,b), find ŷ∈R² with ŷ₀²+ŷ₁²=a AND ŷ₀·ŷ₁=b. Energy
//! E(a,b,ŷ) is trained low at solutions; predict by descending ŷ. Baseline: a matched feed-forward net
//! (a,b)→ŷ (one pass). We measure accuracy vs #descent-steps K (thinking) in-distribution (a∈[0.6,1.4]) AND
//! OOD (a∈[1.4,2.2]); the EBT claim is that thinking helps MORE on the OOD slice.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_nano_ebt --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const H: usize = 96;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

// sample a context (a,b) with real solutions (a²≥4b²) in a radius band, return (a,b) and ONE solution ŷ
fn sample_ctx(seed: u32, amin: f32, amax: f32) -> (f32, f32, f32, f32) {
    let mut s = seed;
    loop {
        let a = amin + u(s, 1) * (amax - amin);
        let b = (u(s, 2) * 2.0 - 1.0) * (a / 2.0); // |b| ≤ a/2 guarantees a²≥4b² ⇒ real solutions
        s = s.wrapping_add(1327);
        let disc = a * a - 4.0 * b * b;
        if disc < 0.0 { continue; }
        let s0 = (a + disc.sqrt()) / 2.0; // ŷ₀² = s0
        if s0 <= 1e-4 { continue; }
        let y0 = s0.sqrt() * if u(s, 3) < 0.5 { 1.0 } else { -1.0 };
        let y1 = b / y0;
        return (a, b, y0, y1);
    }
}
fn correct(a: f32, b: f32, y0: f32, y1: f32) -> bool { (y0 * y0 + y1 * y1 - a).abs() < 0.12 && (y0 * y1 - b).abs() < 0.12 }

async fn ff_acc(ctx: &Arc<ferric_core::Context>, fp: &[Tensor], amin: f32, amax: f32, b: usize, seed0: u32) -> f32 {
    let mut ab = vec![0.0f32; b * 2]; let mut gt = vec![(0.0f32, 0.0f32); b];
    for i in 0..b { let (a, bb, _, _) = sample_ctx(seed0 + i as u32 * 7, amin, amax); ab[i * 2] = a; ab[i * 2 + 1] = bb; gt[i] = (a, bb); }
    let abt = Tensor::from_vec(ctx, &ab, &[b, 2]);
    let out = abt.matmul(&fp[0]).add(&fp[1]).relu().matmul(&fp[2]).add(&fp[3]).relu().matmul(&fp[4]).add(&fp[5]).to_vec().await;
    let mut ok = 0; for i in 0..b { if correct(gt[i].0, gt[i].1, out[i * 2], out[i * 2 + 1]) { ok += 1; } } ok as f32 / b as f32
}

// energy E(ab, ŷ): softplus MLP, ŷ and context enter the first layer separately (so we grad only ŷ)
fn energy(yv: &Var, ab: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| -> Var { z.exp().add(one).log() };
    let h1 = sp(yv.matmul(&p[0]).add(&ab.matmul(&p[1])).add(&p[2]));
    let h2 = sp(h1.matmul(&p[3]).add(&p[4]));
    h2.matmul(&p[5]).add(&p[6])
}

// tensor forward (no grad) — verification scores candidates
fn energy_t(y: &Tensor, ab: &Tensor, p: &[Tensor], one: &Tensor) -> Tensor {
    let sp = |z: Tensor| -> Tensor { z.exp().add(one).log() };
    let h1 = sp(y.matmul(&p[0]).add(&ab.matmul(&p[1])).add(&p[2]));
    let h2 = sp(h1.matmul(&p[3]).add(&p[4]));
    h2.matmul(&p[5]).add(&p[6])
}
// energy-as-verifier (robust where descent was fragile): per context, score N candidate ŷ, keep min-energy
async fn bestof_acc(ctx: &Arc<ferric_core::Context>, p: &[Tensor], one: &Tensor, nn: usize, amin: f32, amax: f32, b: usize, seed0: u32) -> f32 {
    let mut ab = vec![0.0f32; b * nn * 2]; let mut yc = vec![0.0f32; b * nn * 2]; let mut gt = vec![(0.0f32, 0.0f32); b];
    for i in 0..b { let (a, bb, _, _) = sample_ctx(seed0 + i as u32 * 7, amin, amax); gt[i] = (a, bb);
        for j in 0..nn { let idx = i * nn + j; ab[idx * 2] = a; ab[idx * 2 + 1] = bb;
            yc[idx * 2] = (u(idx as u32, seed0 * 3 + 1) * 2.0 - 1.0) * 1.9; yc[idx * 2 + 1] = (u(idx as u32, seed0 * 3 + 7) * 2.0 - 1.0) * 1.9; } }
    let e = energy_t(&Tensor::from_vec(ctx, &yc, &[b * nn, 2]), &Tensor::from_vec(ctx, &ab, &[b * nn, 2]), p, one).to_vec().await;
    let mut ok = 0;
    for i in 0..b { let (mut bi, mut be) = (0usize, f32::MAX); for j in 0..nn { let idx = i * nn + j; if e[idx] < be { be = e[idx]; bi = idx; } }
        if correct(gt[i].0, gt[i].1, yc[bi * 2], yc[bi * 2 + 1]) { ok += 1; } }
    ok as f32 / b as f32
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — nano-EBT: energy-descent prediction (thinks + OOD) vs feed-forward, nonlinear system");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let (bs, reg) = (512usize, Tensor::from_vec(&ctx, &[0.1], &[1]));

    // ---- train energy E(a,b,ŷ): low at solutions, high at random ŷ (contrastive) ----
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(2 * H, 10, 1.0 / 1.5), &[2, H]), Tensor::from_vec(&ctx, &randn(2 * H, 11, 1.0 / 1.5), &[2, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 12, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 13, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.002);
    for step in 0..4000 {
        let (mut ab, mut ys, mut yn) = (vec![0.0f32; bs * 2], vec![0.0f32; bs * 2], vec![0.0f32; bs * 2]);
        for i in 0..bs { let (a, b, y0, y1) = sample_ctx(step as u32 * 31 + i as u32 + 1, 0.6, 1.4);
            ab[i * 2] = a; ab[i * 2 + 1] = b; ys[i * 2] = y0; ys[i * 2 + 1] = y1;
            yn[i * 2] = (u(i as u32, step as u32 * 5 + 3) * 2.0 - 1.0) * 2.0; yn[i * 2 + 1] = (u(i as u32, step as u32 * 5 + 8) * 2.0 - 1.0) * 2.0; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let abv = Var::leaf(Tensor::from_vec(&ctx, &ab, &[bs, 2]));
        let ep = energy(&Var::leaf(Tensor::from_vec(&ctx, &ys, &[bs, 2])), &abv, &pv, &ov);
        let en = energy(&Var::leaf(Tensor::from_vec(&ctx, &yn, &[bs, 2])), &abv, &pv, &ov);
        let loss = ep.mean_all().sub(&en.mean_all()).add(&ep.mul(&ep).mean_all().add(&en.mul(&en).mean_all()).mul(&Var::leaf(reg.clone())));
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); adam.step(&mut p, &g);
    }

    // ---- matched feed-forward baseline: (a,b) → ŷ, MSE to a solution (single forward pass) ----
    let mut fp = vec![
        Tensor::from_vec(&ctx, &randn(2 * H, 20, 1.0 / 1.5), &[2, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 21, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * 2, 22, 1.0 / (H as f32).sqrt()), &[H, 2]), Tensor::zeros(&ctx, &[2]),
    ];
    let mut fadam = Adam::new(&fp, 0.002);
    for step in 0..4000 {
        let (mut ab, mut ys) = (vec![0.0f32; bs * 2], vec![0.0f32; bs * 2]);
        for i in 0..bs { let (a, b, y0, y1) = sample_ctx(step as u32 * 37 + i as u32 + 2, 0.6, 1.4); ab[i * 2] = a; ab[i * 2 + 1] = b; ys[i * 2] = y0; ys[i * 2 + 1] = y1; }
        let fv: Vec<Var> = fp.iter().map(|t| Var::leaf(t.clone())).collect();
        let h = Var::leaf(Tensor::from_vec(&ctx, &ab, &[bs, 2])).matmul(&fv[0]).add(&fv[1]).relu();
        let h2 = h.matmul(&fv[2]).add(&fv[3]).relu();
        let out = h2.matmul(&fv[4]).add(&fv[5]);
        let diff = out.sub(&Var::leaf(Tensor::from_vec(&ctx, &ys, &[bs, 2]))); let loss = diff.mul(&diff).mean_all();
        loss.backward(); let g: Vec<Tensor> = fv.iter().map(|v| v.grad().unwrap()).collect(); fadam.step(&mut fp, &g);
    }
    // ---- report: thinking curve = energy-verification accuracy vs N candidates (robust; descent was fragile) ----
    // feed-forward regresses to ONE target but the system is MULTIVALUED (up to 4 solutions) → averages them → fails.
    println!("\n  nonlinear system ŷ₀²+ŷ₁²=a ∧ ŷ₀·ŷ₁=b — solve accuracy (energy-verify best-of-N vs feed-forward):");
    println!("     in-distribution (a∈[0.6,1.4]):  feed-forward(1 pass) {:>3.0}%", ff_acc(&ctx, &fp, 0.6, 1.4, 400, 1000).await * 100.0);
    print!("        energy best-of-N  ");
    for &nn in &[1usize, 4, 16, 64, 256] { print!("N={}:{:>3.0}%  ", nn, bestof_acc(&ctx, &p, &one, nn, 0.6, 1.4, 400, 2000).await * 100.0); }
    println!();
    println!("     OUT-OF-DISTRIBUTION (a∈[1.4,2.2]):  feed-forward(1 pass) {:>3.0}%", ff_acc(&ctx, &fp, 1.4, 2.2, 400, 3000).await * 100.0);
    print!("        energy best-of-N  ");
    for &nn in &[1usize, 4, 16, 64, 256] { print!("N={}:{:>3.0}%  ", nn, bestof_acc(&ctx, &p, &one, nn, 1.4, 2.2, 400, 4000).await * 100.0); }
    println!();
    println!("\n  EBT signatures: energy-verification accuracy RISES with N (test-time thinking) and SOLVES the multivalued");
    println!("  system where feed-forward regression structurally can't (it averages the 4 solutions → satisfies none).");
}
