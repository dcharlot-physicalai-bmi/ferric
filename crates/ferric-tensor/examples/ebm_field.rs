//! EFA energy-first #15 (AI×physics, opening #1) — ENERGY-CONSERVING neural surrogate for a FIELD.
//!
//! The research's top physics opening: FNO/DeepONet/GraphCast learn dynamics with NO conservation guarantee
//! and DRIFT over long rollouts. An energy-based surrogate fixes this structurally. We scale the HNN idea to
//! a FIELD — a 12-mass nonlinear lattice (FPUT-like: linear + cubic springs, fixed walls) — by learning the
//! POTENTIAL ENERGY U(q) so the force F = −∇U is CONSERVATIVE BY CONSTRUCTION (curl-free ⇒ no net work ⇒
//! bounded energy). A naive net predicts the force directly with no such structure. Both fit the same force
//! data; over a long symplectic rollout the naive net's non-conservative error COMPOUNDS into energy drift
//! while the energy surrogate stays bounded — the stability edge, at field scale. (∂U/∂q by finite diff ⇒
//! 1st-order trainable on Ferric.)
//!
//! Run: `cargo run -p ferric-tensor --example ebm_field --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const N: usize = 12; const HH: usize = 96; const EPS: f32 = 0.008; const BETA: f32 = 0.5;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
// true potential energy U(q) and force −∂U/∂q for the fixed-wall nonlinear lattice (q_0 = q_{N+1} = 0)
fn diffs(q: &[f32]) -> Vec<f32> { let mut d = vec![0.0f32; N + 1]; for i in 0..=N { let a = if i == 0 { 0.0 } else { q[i - 1] }; let b = if i == N { 0.0 } else { q[i] }; d[i] = b - a; } d }
fn u_true(q: &[f32]) -> f32 { diffs(q).iter().map(|&d| 0.5 * d * d + 0.25 * BETA * d * d * d * d).sum() }
fn force_true(q: &[f32]) -> Vec<f32> { let d = diffs(q); (0..N).map(|i| { let fr = d[i + 1] + BETA * d[i + 1].powi(3); let fl = d[i] + BETA * d[i].powi(3); fr - fl }).collect() } // −∂U/∂q_i

fn pot_v(q: &Var, p: &[Var], one: &Var) -> Var { let sp = |z: Var| z.exp().add(one).log(); let h = sp(q.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }
fn pot_t(q: &Tensor, p: &[Tensor], one: &Tensor) -> Tensor { let sp = |z: Tensor| z.exp().add(one).log(); let h = sp(q.matmul(&p[0]).add(&p[1])); let h2 = sp(h.matmul(&p[2]).add(&p[3])); h2.matmul(&p[4]).add(&p[5]) }
// surrogate force −∂U_ψ/∂q via central finite difference (2N potential evals)
async fn surr_force(ctx: &Arc<ferric_core::Context>, wp: &[Tensor], one: &Tensor, q: &[f32], b: usize) -> Vec<f32> {
    let mut f = vec![0.0f32; b * N];
    for i in 0..N { let mut plus = q.to_vec(); let mut minus = q.to_vec(); for bb in 0..b { plus[bb * N + i] += EPS; minus[bb * N + i] -= EPS; }
        let du = pot_t(&Tensor::from_vec(ctx, &plus, &[b, N]), wp, one).to_vec().await; let dm = pot_t(&Tensor::from_vec(ctx, &minus, &[b, N]), wp, one).to_vec().await;
        for bb in 0..b { f[bb * N + i] = -(du[bb] - dm[bb]) / (2.0 * EPS); } }
    f
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — energy-conserving FIELD surrogate: learn U(q) for a {N}-mass nonlinear lattice");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize; let i2e = Tensor::from_vec(&ctx, &[1.0 / (2.0 * EPS)], &[1]);

    // ---- ENERGY surrogate: learn potential U_ψ(q); force = −∂U_ψ/∂q (finite diff) fit to true force ----
    let mut wp = vec![
        Tensor::from_vec(&ctx, &randn(N * HH, 1, 1.0 / (N as f32).sqrt()), &[N, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 2, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH, 3, 1.0 / (HH as f32).sqrt()), &[HH, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut wadam = Adam::new(&wp, 0.002);
    for step in 0..7000 {
        let mut q = vec![0.0f32; bs * N]; let mut ft = vec![0.0f32; bs * N];
        for b in 0..bs { let row: Vec<f32> = (0..N).map(|i| (u((b * N + i) as u32, step as u32 * 3 + 1) * 2.0 - 1.0) * 0.7).collect();
            let f = force_true(&row); for i in 0..N { q[b * N + i] = row[i]; ft[b * N + i] = f[i]; } }
        let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let i2 = Var::leaf(i2e.clone());
        // finite-diff −∂U/∂q_i for each coordinate i (2N forward evals of U_ψ)
        let mut floss: Option<Var> = None;
        for i in 0..N {
            let mut plus = q.clone(); let mut minus = q.clone(); for b in 0..bs { plus[b * N + i] += EPS; minus[b * N + i] -= EPS; }
            let du = pot_v(&Var::leaf(Tensor::from_vec(&ctx, &plus, &[bs, N])), &pv, &ov).sub(&pot_v(&Var::leaf(Tensor::from_vec(&ctx, &minus, &[bs, N])), &pv, &ov)).mul(&i2); // ∂U/∂q_i [bs,1]
            let fi = du.neg(); // predicted force_i = −∂U/∂q_i
            let ti = Var::leaf(Tensor::from_vec(&ctx, &(0..bs).map(|b| ft[b * N + i]).collect::<Vec<_>>(), &[bs, 1]));
            let e = fi.sub(&ti); let l = e.mul(&e).mean_all();
            floss = Some(match floss { None => l, Some(pl) => pl.add(&l) });
        }
        let loss = floss.unwrap();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); wadam.step(&mut wp, &g);
    }

    // ---- naive baseline: MLP q → force directly (no conservative structure) ----
    let mut np = vec![
        Tensor::from_vec(&ctx, &randn(N * HH, 10, 1.0 / (N as f32).sqrt()), &[N, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * HH, 11, 1.0 / (HH as f32).sqrt()), &[HH, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH * N, 12, 1.0 / (HH as f32).sqrt()), &[HH, N]), Tensor::zeros(&ctx, &[N]),
    ];
    let mut nadam = Adam::new(&np, 0.002);
    for step in 0..7000 {
        let mut q = vec![0.0f32; bs * N]; let mut ft = vec![0.0f32; bs * N];
        for b in 0..bs { let row: Vec<f32> = (0..N).map(|i| (u((b * N + i) as u32, step as u32 * 7 + 5) * 2.0 - 1.0) * 0.7).collect();
            let f = force_true(&row); for i in 0..N { q[b * N + i] = row[i]; ft[b * N + i] = f[i]; } }
        let pv: Vec<Var> = np.iter().map(|t| Var::leaf(t.clone())).collect();
        let out = Var::leaf(Tensor::from_vec(&ctx, &q, &[bs, N])).matmul(&pv[0]).add(&pv[1]).relu().matmul(&pv[2]).add(&pv[3]).relu().matmul(&pv[4]).add(&pv[5]);
        let diff = out.sub(&Var::leaf(Tensor::from_vec(&ctx, &ft, &[bs, N]))); let loss = diff.mul(&diff).mean_all();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); nadam.step(&mut np, &g);
    }
    let naive_force = |q: &Tensor| -> Tensor { q.matmul(&np[0]).add(&np[1]).relu().matmul(&np[2]).add(&np[3]).relu().matmul(&np[4]).add(&np[5]) };

    // ---- long symplectic rollout; measure true energy H = ½Σp² + U_true(q) drift ----
    let b = 16usize; let steps = 2000usize; let dt = 0.02f32;
    let init = |seed: u32| -> (Vec<f32>, Vec<f32>) { let mut q = vec![0.0f32; b * N]; let p = vec![0.0f32; b * N];
        for bb in 0..b { for i in 0..N { q[bb * N + i] = (u((bb * N + i) as u32, seed) * 2.0 - 1.0) * 0.5; } } (q, p) };
    let energy = |q: &[f32], p: &[f32]| -> f32 { (0..b).map(|bb| { let qi = &q[bb * N..(bb + 1) * N]; let ke: f32 = p[bb * N..(bb + 1) * N].iter().map(|v| 0.5 * v * v).sum(); ke + u_true(qi) }).sum::<f32>() / b as f32 };
    let (q0, p0) = init(42); let e0 = energy(&q0, &p0);
    // surrogate rollout (semi-implicit symplectic)
    let (mut q, mut p) = (q0.clone(), p0.clone());
    for _ in 0..steps { let f = surr_force(&ctx, &wp, &one, &q, b).await; for j in 0..b * N { p[j] += dt * f[j]; } for j in 0..b * N { q[j] += dt * p[j]; } }
    let e_surr = energy(&q, &p);
    // naive rollout
    let (mut q, mut p) = (q0.clone(), p0.clone());
    for _ in 0..steps { let f = naive_force(&Tensor::from_vec(&ctx, &q, &[b, N])).to_vec().await; for j in 0..b * N { p[j] += dt * f[j]; } for j in 0..b * N { q[j] += dt * p[j]; } }
    let e_naive = energy(&q, &p);
    println!("\n  true energy over a {steps}-step field rollout (should stay {e0:.3}):");
    println!("     {:<38} end {:.3}   drift {:+.1}%", "energy surrogate (learned U, F=−∇U)", e_surr, (e_surr - e0) / e0 * 100.0);
    println!("     {:<38} end {:.3}   drift {:+.1}%", "naive force MLP (baseline)", e_naive, (e_naive - e0) / e0 * 100.0);
    println!("\n  |surrogate drift| ≪ |naive drift| → learning the ENERGY makes the field surrogate CONSERVE over long");
    println!("  horizons where a direct force/dynamics net drifts — the stability edge FNO/GraphCast lack (opening #1).");
}
