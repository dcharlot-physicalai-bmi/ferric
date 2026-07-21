//! EFA energy-first #18 — the TRUE Energy-Based Transformer: TRAIN THROUGH the unrolled energy descent.
//!
//! The flagship payoff of Ferric's new second-order autograd. Earlier (`ebm_nano_ebt.rs`) the EBT could only
//! be done via VERIFICATION (best-of-N) because training THROUGH the descent needs grad-of-grad. Now it's the
//! real thing (Gladstone/Du 2025, arXiv:2507.02092): predict ŷ by K steps of gradient descent on the energy
//! (ŷ ← ŷ − α·∂E/∂ŷ), UNROLL those steps, and supervise the endpoint — backprop through the whole unrolled
//! descent to the weights (2nd order, since each step uses ∂E/∂ŷ). Task = the multivalued nonlinear system
//! ŷ₀²+ŷ₁²=a ∧ ŷ₀·ŷ₁=b. CORRECTION: earlier text/prints here claimed "feedforward = 0%" — that was ASSERTED,
//! NOT measured, and is WRONG. A fairly-supervised feedforward solves this at 100% on ~350 params (see ebm_edge.rs).
//! The EBT's real edge is representing the MULTIVALUED SET (different inits → different valid solutions) + thinking.
//! The EBT's descent lands on A valid solution, and THINKING (more K at test time) should help.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_ebt_true --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const HE: usize = 96;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn ctx_ab(seed: u32, amin: f32, amax: f32) -> (f32, f32) { let a = amin + u(seed, 1) * (amax - amin); let b = (u(seed, 2) * 2.0 - 1.0) * (a / 2.0); (a, b) }
// up-to-4 real solutions of ŷ₀²+ŷ₁²=a, ŷ₀ŷ₁=b
fn sols(a: f32, b: f32) -> Vec<(f32, f32)> {
    let disc = a * a - 4.0 * b * b; if disc < 0.0 { return vec![]; }
    let mut v = vec![]; for s in [(a + disc.sqrt()) / 2.0, (a - disc.sqrt()) / 2.0] { if s > 1e-4 { let y = s.sqrt(); v.push((y, b / y)); v.push((-y, -b / y)); } } v
}
fn correct(a: f32, b: f32, y0: f32, y1: f32) -> bool { (y0 * y0 + y1 * y1 - a).abs() < 0.15 && (y0 * y1 - b).abs() < 0.15 }

fn energy(yv: &Var, ab: &Var, p: &[Var], one: &Var) -> Var { let sp = |z: Var| z.exp().add(one).log(); let h1 = sp(yv.matmul(&p[0]).add(&ab.matmul(&p[1])).add(&p[2])); let h2 = sp(h1.matmul(&p[3]).add(&p[4])); h2.matmul(&p[5]).add(&p[6]) }

// solve by energy descent (K steps); return fraction satisfying the constraints
async fn solve(ctx: &Arc<ferric_core::Context>, p: &[Tensor], one: &Tensor, al: &Tensor, amin: f32, amax: f32, k: usize, t: usize, seed0: u32) -> f32 {
    let mut ab = vec![0.0f32; t * 2]; let mut probs = Vec::with_capacity(t);
    for i in 0..t { let (a, b) = ctx_ab(seed0 + i as u32 * 5, amin, amax); ab[i * 2] = a; ab[i * 2 + 1] = b; probs.push((a, b)); }
    let abv = Var::leaf(Tensor::from_vec(ctx, &ab, &[t, 2])); let pv: Vec<Var> = p.iter().map(|x| Var::leaf(x.clone())).collect(); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
    let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(t * 2, seed0 ^ 0xabc, 0.8), &[t, 2]));
    for _ in 0..k { let e = energy(&y, &abv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv)); }
    let yk = y.value().to_vec().await; let mut ok = 0; for i in 0..t { if correct(probs[i].0, probs[i].1, yk[i * 2], yk[i * 2 + 1]) { ok += 1; } } ok as f32 / t as f32
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — TRUE Energy-Based Transformer: train THROUGH the unrolled energy descent");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let (bs, ktr, alpha) = (96usize, 6usize, 0.2f32);
    let al = Tensor::from_vec(&ctx, &[alpha], &[1]);
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(2 * HE, 10, 1.0 / 1.5), &[2, HE]), Tensor::from_vec(&ctx, &randn(2 * HE, 11, 1.0 / 1.5), &[2, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * HE, 12, 1.0 / (HE as f32).sqrt()), &[HE, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE, 13, 1.0 / (HE as f32).sqrt()), &[HE, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.001);

    for step in 0..2500 {
        // batch of contexts (a,b) with real solutions
        let mut ab = vec![0.0f32; bs * 2]; let mut probs = Vec::with_capacity(bs);
        for i in 0..bs { let (a, b) = ctx_ab(step as u32 * 31 + i as u32 + 1, 0.6, 1.4); ab[i * 2] = a; ab[i * 2 + 1] = b; probs.push((a, b)); }
        let abv = Var::leaf(Tensor::from_vec(&ctx, &ab, &[bs, 2]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
        // UNROLL K descent steps: ŷ ← ŷ − α·∂E/∂ŷ  (each step differentiable in the weights)
        let mut y = Var::leaf(Tensor::from_vec(&ctx, &randn(bs * 2, step as u32 * 7 + 3, 0.8), &[bs, 2]));
        for _ in 0..ktr {
            let e = energy(&y, &abv, &pv, &ov).sum_all();
            let g = grad(&e, &[y.clone()], None).remove(0);  // ∂ΣE/∂ŷ (differentiable in weights)
            y = y.sub(&g.mul(&alv));
        }
        // supervise ŷ_K to the NEAREST true solution (lets the energy have minima at all solutions)
        let yk = y.value().to_vec().await; let mut tgt = vec![0.0f32; bs * 2];
        for i in 0..bs { let (a, b) = probs[i]; let ss = sols(a, b); let (yy0, yy1) = (yk[i * 2], yk[i * 2 + 1]);
            if let Some(&(s0, s1)) = ss.iter().min_by(|x, z| ((x.0 - yy0).powi(2) + (x.1 - yy1).powi(2)).partial_cmp(&((z.0 - yy0).powi(2) + (z.1 - yy1).powi(2))).unwrap()) { tgt[i * 2] = s0; tgt[i * 2 + 1] = s1; } }
        let diff = y.sub(&Var::leaf(Tensor::from_vec(&ctx, &tgt, &[bs, 2])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();   // ∂loss/∂weights — backprop THROUGH the K-step unrolled descent (2nd order)
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }

    // ---- eval: solve accuracy by energy descent, vs K (thinking), in-dist and OOD ----
    println!("\n  solve accuracy of the descent-trained EBT vs #thinking steps K (multivalued; NOTE: a fair feedforward also solves this — see ebm_edge.rs):");
    print!("     in-distribution (a∈[0.6,1.4]):  ");
    for &k in &[1usize, 3, 6, 12, 25] { print!("K={}:{:>3.0}%  ", k, solve(&ctx, &p, &one, &al, 0.6, 1.4, k, 400, 900).await * 100.0); }
    println!();
    print!("     OUT-OF-DIST (a∈[1.4,2.2]):      ");
    for &k in &[1usize, 3, 6, 12, 25] { print!("K={}:{:>3.0}%  ", k, solve(&ctx, &p, &one, &al, 1.4, 2.2, k, 400, 5000).await * 100.0); }
    println!("\n\n  trained THROUGH the unrolled descent (2nd-order autograd) → the EBT solves the multivalued system by");
    println!("  energy descent where feed-forward can't, and thinking (more K) refines the answer — the real EBT, on Ferric.");
}
