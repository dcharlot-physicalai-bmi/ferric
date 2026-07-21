//! EFA energy-first #12 (AI×physics, opening #4) — DISCOVER a conservation law from trajectory data.
//!
//! Not the dynamics (HNN) nor the governing equation (SINDy) — the INVARIANT: learn a scalar I(state) that
//! stays CONSTANT along trajectories (AI-Poincaré, Liu & Tegmark 2020). A quantity is conserved iff its
//! gradient is orthogonal to the flow: ∇I·f = 0. So minimize the conservation residual (∇I·f)² with a
//! ‖∇I‖≈1 normalization (else the trivial I=const wins). Nonlinear PENDULUM: θ̈=−sinθ, flow f=(ω,−sinθ);
//! true invariant = the NONLINEAR energy E = ½ω² − cosθ (∇E=(sinθ,ω), ∇E·f = ω·sinθ − sinθ·ω = 0 ✓). We
//! never tell the model the form — it must discover it. Finite-difference ∇I keeps it 1st-order trainable.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_conserve --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const HH: usize = 64; const EPS: f32 = 0.03;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn e_true(th: f32, w: f32) -> f32 { 0.5 * w * w - th.cos() }

fn inet(s: &Var, p: &[Var], one: &Var) -> Var { let sp = |z: Var| z.exp().add(one).log(); let h = sp(s.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }
fn inet_t(s: &Tensor, p: &[Tensor], one: &Tensor) -> Tensor { let sp = |z: Tensor| z.exp().add(one).log(); let h = sp(s.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — DISCOVER a conservation law: learn I(θ,ω) conserved along pendulum trajectories");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 512usize; let i2e = Tensor::from_vec(&ctx, &[1.0 / (2.0 * EPS)], &[1]);
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(2 * HH, 1, 1.0 / 1.5), &[2, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 2, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH, 3, 1.0 / (HH as f32).sqrt()), &[HH, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.002); let (one_c, beta) = (Tensor::from_vec(&ctx, &[1.0], &[1]), Tensor::from_vec(&ctx, &[1.0], &[1]));
    for _step in 0..6000 {
        let mut th = vec![0.0f32; bs]; let mut w = vec![0.0f32; bs];
        for i in 0..bs { th[i] = (u(i as u32, _step as u32 * 3 + 1) * 2.0 - 1.0) * 3.1416; w[i] = (u(i as u32, _step as u32 * 3 + 2) * 2.0 - 1.0) * 2.5; }
        let mk = |dth: f32, dw: f32| -> Tensor { let mut v = vec![0.0f32; bs * 2]; for i in 0..bs { v[i * 2] = th[i] + dth; v[i * 2 + 1] = w[i] + dw; } Tensor::from_vec(&ctx, &v, &[bs, 2]) };
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let i2 = Var::leaf(i2e.clone());
        let git = inet(&Var::leaf(mk(EPS, 0.0)), &pv, &ov).sub(&inet(&Var::leaf(mk(-EPS, 0.0)), &pv, &ov)).mul(&i2); // ∂I/∂θ
        let giw = inet(&Var::leaf(mk(0.0, EPS)), &pv, &ov).sub(&inet(&Var::leaf(mk(0.0, -EPS)), &pv, &ov)).mul(&i2); // ∂I/∂ω
        // flow f = (ω, −sinθ); conservation residual ∇I·f = git·ω + giw·(−sinθ)
        let fw = Var::leaf(Tensor::from_vec(&ctx, &w, &[bs, 1]));
        let fs = Var::leaf(Tensor::from_vec(&ctx, &th.iter().map(|t| -t.sin()).collect::<Vec<_>>(), &[bs, 1]));
        let resid = git.mul(&fw).add(&giw.mul(&fs));
        let gnorm = git.mul(&git).add(&giw.mul(&giw)); // ‖∇I‖²
        // loss = mean(residual²) + β·(mean‖∇I‖² − 1)²  (normalization avoids the trivial I=const)
        let norm_pen = gnorm.mean_all().sub(&Var::leaf(one_c.clone())); let norm_pen = norm_pen.mul(&norm_pen);
        let loss = resid.mul(&resid).mean_all().add(&norm_pen.mul(&Var::leaf(beta.clone())));
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); adam.step(&mut p, &g);
    }

    // ---- eval: (1) I ≈ function of E? correlation over a grid. (2) conserved along trajectories? ----
    let m = 3000usize; let mut grid = vec![0.0f32; m * 2]; let mut et = vec![0.0f32; m];
    for i in 0..m { let th = (u(i as u32, 71) * 2.0 - 1.0) * 3.1416; let w = (u(i as u32, 72) * 2.0 - 1.0) * 2.5; grid[i * 2] = th; grid[i * 2 + 1] = w; et[i] = e_true(th, w); }
    let iv = inet_t(&Tensor::from_vec(&ctx, &grid, &[m, 2]), &p, &one).to_vec().await;
    let corr = |a: &[f32], b: &[f32]| -> f32 { let n = a.len() as f32; let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let mut cov = 0.0; let mut va = 0.0; let mut vb = 0.0; for i in 0..a.len() { cov += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma).powi(2); vb += (b[i] - mb).powi(2); } cov / (va.sqrt() * vb.sqrt() + 1e-9) };
    println!("\n  |corr(discovered I, true E=½ω²−cosθ)| over the phase plane: {:.3}  (1.0 = I is a function of the true invariant)", corr(&iv, &et).abs());

    // conservation along trajectories: integrate 24 pendulum orbits, measure how flat I stays vs its spread across orbits
    let dt = 0.01; let steps = 400; let ntraj = 24usize;
    let mut all_i: Vec<f32> = Vec::new(); let mut within = 0.0f32;
    for tr in 0..ntraj {
        let mut th = (u(tr as u32, 5) * 2.0 - 1.0) * 3.0; let mut w = (u(tr as u32, 6) * 2.0 - 1.0) * 2.2; let mut traj = Vec::with_capacity(steps);
        for _ in 0..steps { traj.push((th, w)); let a = -th.sin(); w += dt * a; th += dt * w; } // semi-implicit
        let flat: Vec<f32> = traj.iter().flat_map(|&(a, b)| [a, b]).collect();
        let ivt = inet_t(&Tensor::from_vec(&ctx, &flat, &[steps, 2]), &p, &one).to_vec().await;
        let mean = ivt.iter().sum::<f32>() / steps as f32; let sd = (ivt.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / steps as f32).sqrt();
        within += sd; all_i.push(mean);
    }
    within /= ntraj as f32;
    let gm = all_i.iter().sum::<f32>() / ntraj as f32; let across = (all_i.iter().map(|v| (v - gm).powi(2)).sum::<f32>() / ntraj as f32).sqrt();
    println!("  conservation: within-trajectory std {:.4}  vs  across-trajectory std {:.4}  → ratio {:.3}", within, across, within / (across + 1e-9));
    println!("\n  small within/across ratio + high corr → I is CONSTANT along each orbit yet DISTINGUISHES orbits = a genuine");
    println!("  discovered conserved quantity (the nonlinear pendulum energy), found from trajectories without its form.");
}
