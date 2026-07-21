//! EFA energy-first #17 (AI×physics, opening #1) — energy-conserving FIELD surrogate, EXACT gradients.
//!
//! Build #15 (`ebm_field.rs`) was an honest NEGATIVE: an energy-conserving surrogate (learn U(q), force
//! F=−∇U) conserved WORSE than a naive force net, because it used FINITE-DIFFERENCE gradients — which cap
//! both the potential fit and the rollout accuracy. Now that Ferric has SECOND-ORDER autograd, we can train
//! and roll with EXACT gradients: F = −∂U_ψ/∂q via `grad()`, differentiable in ψ (a loss on the force is a
//! second-order gradient in the weights). Same 12-mass nonlinear lattice, same naive baseline. Question:
//! does exact-gradient training FLIP the negative — does the energy surrogate now conserve better?
//!
//! Run: `cargo run -p ferric-tensor --example ebm_field2 --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const N: usize = 12; const HH: usize = 96; const BETA: f32 = 0.5;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn diffs(q: &[f32]) -> Vec<f32> { let mut d = vec![0.0f32; N + 1]; for i in 0..=N { let a = if i == 0 { 0.0 } else { q[i - 1] }; let b = if i == N { 0.0 } else { q[i] }; d[i] = b - a; } d }
fn u_true(q: &[f32]) -> f32 { diffs(q).iter().map(|&d| 0.5 * d * d + 0.25 * BETA * d * d * d * d).sum() }
fn force_true(q: &[f32]) -> Vec<f32> { let d = diffs(q); (0..N).map(|i| { let fr = d[i + 1] + BETA * d[i + 1].powi(3); let fl = d[i] + BETA * d[i].powi(3); fr - fl }).collect() }

fn pot(q: &Var, p: &[Var], one: &Var) -> Var { let sp = |z: Var| z.exp().add(one).log(); let h = sp(q.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }

// EXACT force F = −∂U_ψ/∂q via second-order-capable grad(): returns [b,N]
async fn surr_force(ctx: &Arc<ferric_core::Context>, wp: &[Tensor], one: &Tensor, q: &[f32], b: usize) -> Vec<f32> {
    let qv = Var::leaf(Tensor::from_vec(ctx, q, &[b, N]));
    let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
    let usum = pot(&qv, &pv, &Var::leaf(one.clone())).sum_all();
    let g = grad(&usum, &[qv], None).remove(0).value().to_vec().await; // ∂ΣU/∂q = per-sample ∂U/∂q
    g.iter().map(|x| -x).collect()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — energy-conserving FIELD surrogate with EXACT gradients ({N}-mass lattice)");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 128usize;

    // ---- learn U_ψ(q); train F=−∂U_ψ/∂q (EXACT, via grad, differentiable in ψ = 2nd order) to true force ----
    let mut wp = vec![
        Tensor::from_vec(&ctx, &randn(N * HH, 1, 1.0 / (N as f32).sqrt()), &[N, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 2, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH, 3, 1.0 / (HH as f32).sqrt()), &[HH, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut wadam = Adam::new(&wp, 0.002);
    for step in 0..4000 {
        let mut q = vec![0.0f32; bs * N]; let mut ft = vec![0.0f32; bs * N];
        for b in 0..bs { let row: Vec<f32> = (0..N).map(|i| (u((b * N + i) as u32, step as u32 * 3 + 1) * 2.0 - 1.0) * 0.7).collect();
            let f = force_true(&row); for i in 0..N { q[b * N + i] = row[i]; ft[b * N + i] = f[i]; } }
        let qv = Var::leaf(Tensor::from_vec(&ctx, &q, &[bs, N]));
        let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        let usum = pot(&qv, &pv, &Var::leaf(one.clone())).sum_all();
        let dudq = grad(&usum, &[qv], None).remove(0);           // ∂U/∂q, differentiable in pv
        let fpred = dudq.neg();                                   // F = −∂U/∂q
        let diff = fpred.sub(&Var::leaf(Tensor::from_vec(&ctx, &ft, &[bs, N])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();                                          // ∂loss/∂ψ — SECOND order
        let g: Vec<Tensor> = pv.iter().zip(&wp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        wadam.step(&mut wp, &g);
    }

    // ---- naive baseline: MLP q → force directly ----
    let mut np = vec![
        Tensor::from_vec(&ctx, &randn(N * HH, 10, 1.0 / (N as f32).sqrt()), &[N, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 11, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * N, 12, 1.0 / (HH as f32).sqrt()), &[HH, N]), Tensor::zeros(&ctx, &[N]),
    ];
    let mut nadam = Adam::new(&np, 0.002);
    for step in 0..4000 {
        let mut q = vec![0.0f32; bs * N]; let mut ft = vec![0.0f32; bs * N];
        for b in 0..bs { let row: Vec<f32> = (0..N).map(|i| (u((b * N + i) as u32, step as u32 * 7 + 5) * 2.0 - 1.0) * 0.7).collect();
            let f = force_true(&row); for i in 0..N { q[b * N + i] = row[i]; ft[b * N + i] = f[i]; } }
        let pv: Vec<Var> = np.iter().map(|t| Var::leaf(t.clone())).collect();
        let out = Var::leaf(Tensor::from_vec(&ctx, &q, &[bs, N])).matmul(&pv[0]).add(&pv[1]).relu().matmul(&pv[2]).add(&pv[3]).relu().matmul(&pv[4]).add(&pv[5]);
        let diff = out.sub(&Var::leaf(Tensor::from_vec(&ctx, &ft, &[bs, N]))); let loss = diff.mul(&diff).mean_all();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); nadam.step(&mut np, &g);
    }
    let naive_force = |q: &Tensor| -> Tensor { q.matmul(&np[0]).add(&np[1]).relu().matmul(&np[2]).add(&np[3]).relu().matmul(&np[4]).add(&np[5]) };

    // ---- long symplectic rollout; measure true energy drift ----
    let b = 16usize; let steps = 3000usize; let dt = 0.02f32;
    let mut q = vec![0.0f32; b * N]; for bb in 0..b { for i in 0..N { q[bb * N + i] = (u((bb * N + i) as u32, 42) * 2.0 - 1.0) * 0.5; } }
    let q0 = q.clone(); let p0 = vec![0.0f32; b * N];
    let energy = |q: &[f32], p: &[f32]| -> f32 { (0..b).map(|bb| { let ke: f32 = p[bb * N..(bb + 1) * N].iter().map(|v| 0.5 * v * v).sum(); ke + u_true(&q[bb * N..(bb + 1) * N]) }).sum::<f32>() / b as f32 };
    let e0 = energy(&q0, &p0);
    // surrogate (exact-gradient force), semi-implicit symplectic
    let (mut q, mut p) = (q0.clone(), p0.clone());
    for _ in 0..steps { let f = surr_force(&ctx, &wp, &one, &q, b).await; for j in 0..b * N { p[j] += dt * f[j]; } for j in 0..b * N { q[j] += dt * p[j]; } }
    let e_surr = energy(&q, &p);
    // naive
    let (mut q, mut p) = (q0.clone(), p0.clone());
    for _ in 0..steps { let f = naive_force(&Tensor::from_vec(&ctx, &q, &[b, N])).to_vec().await; for j in 0..b * N { p[j] += dt * f[j]; } for j in 0..b * N { q[j] += dt * p[j]; } }
    let e_naive = energy(&q, &p);
    println!("\n  true energy over a {steps}-step field rollout (should stay {e0:.3}):");
    println!("     {:<40} drift {:+.1}%", "energy surrogate (EXACT ∂U/∂q, 2nd-order)", (e_surr - e0) / e0 * 100.0);
    println!("     {:<40} drift {:+.1}%", "naive force MLP (baseline)", (e_naive - e0) / e0 * 100.0);
    let flipped = (e_surr - e0).abs() < (e_naive - e0).abs();
    println!("\n  {}", if flipped { "✅ FLIPPED — exact gradients make the energy surrogate conserve better than the naive net" } else { "⚠ still not better — see analysis (the negative was more than just finite-diff)" });
}
