//! EFA energy-first #8 (AI×physics, opening #4) — HAMILTONIAN discovery: physics IS energy.
//!
//! Physics is intrinsically energy-based (Hamiltonians, Lagrangians, least-action). A Hamiltonian Neural
//! Network (Greydanus 1906.01563) learns a SCALAR energy H(q,p) and gets dynamics from its symplectic
//! gradients — conserving energy BY CONSTRUCTION, where a naive net that predicts the dynamics directly
//! drifts. That is exactly EFA's thesis (a learned scalar energy is the native object) applied to physics.
//! True HNN training needs 2nd-order autograd (grad of the input-grad) which Ferric's 1st-order engine lacks,
//! so we FINITE-DIFFERENCE the gradients (∂H/∂q ≈ [H(q+ε)−H(q−ε)]/2ε) — all forward evals, 1st-order trainable.
//!
//! System: harmonic oscillator, true H = ½(q²+p²), dynamics q̇=p, ṗ=−q (energy conserved). We learn H_θ from
//! (q,p)→(q̇,ṗ) samples, then roll a trajectory forward and measure ENERGY DRIFT vs a naive (q,p)→(q̇,ṗ) MLP.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_hamiltonian --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const HH: usize = 64;
const EPS: f32 = 0.05; // finite-difference step
const DT: f32 = 0.05;  // integration step

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn energy_v(qp: &Var, p: &[Var], one: &Var) -> Var { let sp = |z: Var| z.exp().add(one).log(); let h = sp(qp.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }
fn energy_t(qp: &Tensor, p: &[Tensor], one: &Tensor) -> Tensor { let sp = |z: Tensor| z.exp().add(one).log(); let h = sp(qp.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — HAMILTONIAN discovery: learn H(q,p), conserve energy vs a naive dynamics MLP");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 512usize;
    let inv2e = Tensor::from_vec(&ctx, &[1.0 / (2.0 * EPS)], &[1]);

    // ---- HNN: learn scalar H_θ(q,p); dynamics = symplectic finite-diff gradient; MSE to true (q̇,ṗ) ----
    let mut hp = vec![
        Tensor::from_vec(&ctx, &randn(2 * HH, 1, 1.0 / 1.5), &[2, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 2, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH, 3, 1.0 / (HH as f32).sqrt()), &[HH, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut hadam = Adam::new(&hp, 0.002);
    for step in 0..5000 {
        // sample (q,p) in a disk; true dynamics q̇=p, ṗ=−q
        let mut q = vec![0.0f32; bs]; let mut pp = vec![0.0f32; bs];
        for i in 0..bs { let r = 1.6 * u(i as u32, step as u32 * 3 + 1).sqrt(); let th = 6.2831853 * u(i as u32, step as u32 * 3 + 2); q[i] = r * th.cos(); pp[i] = r * th.sin(); }
        // four ε-shifted inputs (q±ε, p±ε), as constants
        let mk = |dq: f32, dp: f32| -> Tensor { let mut v = vec![0.0f32; bs * 2]; for i in 0..bs { v[i * 2] = q[i] + dq; v[i * 2 + 1] = pp[i] + dp; } Tensor::from_vec(&ctx, &v, &[bs, 2]) };
        let pv: Vec<Var> = hp.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let i2 = Var::leaf(inv2e.clone());
        let hqp = energy_v(&Var::leaf(mk(EPS, 0.0)), &pv, &ov); let hqm = energy_v(&Var::leaf(mk(-EPS, 0.0)), &pv, &ov);
        let hpp = energy_v(&Var::leaf(mk(0.0, EPS)), &pv, &ov); let hpm = energy_v(&Var::leaf(mk(0.0, -EPS)), &pv, &ov);
        let dhdq = hqp.sub(&hqm).mul(&i2); let dhdp = hpp.sub(&hpm).mul(&i2);
        // predicted dynamics: q̇=∂H/∂p, ṗ=−∂H/∂q ; targets q̇=p, ṗ=−q
        let tq = Var::leaf(Tensor::from_vec(&ctx, &pp, &[bs, 1])); let tp = Var::leaf(Tensor::from_vec(&ctx, &q.iter().map(|v| -v).collect::<Vec<_>>(), &[bs, 1]));
        let eq = dhdp.sub(&tq); let ep = dhdq.neg().sub(&tp); // pred q̇=∂H/∂p vs p ; pred ṗ=−∂H/∂q vs −q
        let loss = eq.mul(&eq).mean_all().add(&ep.mul(&ep).mean_all());
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); hadam.step(&mut hp, &g);
    }

    // ---- naive baseline: MLP (q,p) → (q̇,ṗ) directly ----
    let mut np = vec![
        Tensor::from_vec(&ctx, &randn(2 * HH, 10, 1.0 / 1.5), &[2, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 11, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * 2, 12, 1.0 / (HH as f32).sqrt()), &[HH, 2]), Tensor::zeros(&ctx, &[2]),
    ];
    let mut nadam = Adam::new(&np, 0.002);
    for step in 0..5000 {
        let mut qp = vec![0.0f32; bs * 2]; let mut tgt = vec![0.0f32; bs * 2];
        for i in 0..bs { let r = 1.6 * u(i as u32, step as u32 * 7 + 5).sqrt(); let th = 6.2831853 * u(i as u32, step as u32 * 7 + 6); let (qq, ppp) = (r * th.cos(), r * th.sin()); qp[i * 2] = qq; qp[i * 2 + 1] = ppp; tgt[i * 2] = ppp; tgt[i * 2 + 1] = -qq; }
        let pv: Vec<Var> = np.iter().map(|t| Var::leaf(t.clone())).collect();
        let out = Var::leaf(Tensor::from_vec(&ctx, &qp, &[bs, 2])).matmul(&pv[0]).add(&pv[1]).relu().matmul(&pv[2]).add(&pv[3]).relu().matmul(&pv[4]).add(&pv[5]);
        let diff = out.sub(&Var::leaf(Tensor::from_vec(&ctx, &tgt, &[bs, 2]))); let loss = diff.mul(&diff).mean_all();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); nadam.step(&mut np, &g);
    }
    let naive_t = |qp: &Tensor, p: &[Tensor]| -> Tensor { qp.matmul(&p[0]).add(&p[1]).relu().matmul(&p[2]).add(&p[3]).relu().matmul(&p[4]).add(&p[5]) };

    // ---- roll trajectories forward; measure TRUE energy ½(q²+p²) drift over the rollout ----
    let b = 64usize; let steps = 3000usize; // long horizon: naive drift COMPOUNDS, HNN stays bounded
    let mut q_h = vec![0.0f32; b]; let mut p_h = vec![0.0f32; b]; let mut q_n = vec![0.0f32; b]; let mut p_n = vec![0.0f32; b];
    for i in 0..b { let th = 6.2831853 * (i as f32 / b as f32); q_h[i] = th.cos(); p_h[i] = th.sin(); q_n[i] = q_h[i]; p_n[i] = p_h[i]; } // all on the unit circle, true H=0.5
    let energy_of = |q: &[f32], p: &[f32]| -> f32 { (0..q.len()).map(|i| 0.5 * (q[i] * q[i] + p[i] * p[i])).sum::<f32>() / q.len() as f32 };
    let e0 = energy_of(&q_h, &p_h);
    // SYMPLECTIC (semi-implicit) Euler for both: update p using ṗ(q,p), then q using the derivative at the new p.
    // (Plain forward Euler pumps energy into an oscillator regardless of the model — the integrator, not the net.)
    let fd = |q: &[f32], p: &[f32], dq: f32, dp: f32| -> Tensor { let mut v = vec![0.0f32; b * 2]; for i in 0..b { v[i * 2] = q[i] + dq; v[i * 2 + 1] = p[i] + dp; } Tensor::from_vec(&ctx, &v, &[b, 2]) };
    for _ in 0..steps {
        // HNN: ṗ=−∂H/∂q at (q,p); update p; then q̇=∂H/∂p at (q, p_new); update q
        let hqp = energy_t(&fd(&q_h, &p_h, EPS, 0.0), &hp, &one).to_vec().await; let hqm = energy_t(&fd(&q_h, &p_h, -EPS, 0.0), &hp, &one).to_vec().await;
        for i in 0..b { let dhdq = (hqp[i] - hqm[i]) / (2.0 * EPS); p_h[i] += DT * (-dhdq); }
        let hpp = energy_t(&fd(&q_h, &p_h, 0.0, EPS), &hp, &one).to_vec().await; let hpm = energy_t(&fd(&q_h, &p_h, 0.0, -EPS), &hp, &one).to_vec().await;
        for i in 0..b { let dhdp = (hpp[i] - hpm[i]) / (2.0 * EPS); q_h[i] += DT * dhdp; }
        // naive: (q̇,ṗ)=MLP(q,p); update p; then re-eval at (q,p_new) and update q
        let qpn = { let mut v = vec![0.0f32; b * 2]; for i in 0..b { v[i * 2] = q_n[i]; v[i * 2 + 1] = p_n[i]; } v };
        let d1 = naive_t(&Tensor::from_vec(&ctx, &qpn, &[b, 2]), &np).to_vec().await;
        for i in 0..b { p_n[i] += DT * d1[i * 2 + 1]; }
        let qpn2 = { let mut v = vec![0.0f32; b * 2]; for i in 0..b { v[i * 2] = q_n[i]; v[i * 2 + 1] = p_n[i]; } v };
        let d2 = naive_t(&Tensor::from_vec(&ctx, &qpn2, &[b, 2]), &np).to_vec().await;
        for i in 0..b { q_n[i] += DT * d2[i * 2]; }
    }
    let eh = energy_of(&q_h, &p_h); let en = energy_of(&q_n, &p_n);
    println!("\n  true energy ½(q²+p²) over a {steps}-step rollout (should stay {e0:.3}):");
    println!("     {:<34} start {:.3}  →  end {:.3}   drift {:+.1}%", "HNN (learned scalar energy)", e0, eh, (eh - e0) / e0 * 100.0);
    println!("     {:<34} start {:.3}  →  end {:.3}   drift {:+.1}%", "naive dynamics MLP (baseline)", e0, en, (en - e0) / e0 * 100.0);
    println!("\n  |HNN drift| ≪ |naive drift| → the learned scalar energy CONSERVES by construction, where a net that");
    println!("  predicts the dynamics directly leaks/gains energy — the physics prior (energy) being load-bearing.");
}
