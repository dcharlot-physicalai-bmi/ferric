//! EFA energy-first #7 (AI-for-math, opening #2) — energy VERIFIER for algebra: select the correct answer.
//!
//! The USAMO-5% / FrontierMath selection gap is: models generate a correct answer but can't PICK it. A learned
//! energy verifier does best-of-N selection natively (build-2/6). Here we take it to real (parameterized)
//! algebra: solve a·x + b = c for x∈{0..9}. The verifier E(a,b,c,x) must learn the MULTIPLICATIVE relation
//! a·x+b==c (harder than a fixed sum — a·x is a product of inputs) and GENERALIZE across problems. We test
//! best-of-N selection accuracy vs N, in-distribution (a∈1..3) AND out-of-distribution (a∈4..6, unseen larger
//! coefficients) — the distinctive EBM claim (build-3): a verifier that generalizes where a memorized map won't.
//! Verifier = a valid/invalid classifier (build-6 lesson: robust; a contrastive scalar energy needs care).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_math_verify --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const HE: usize = 128;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
// a problem: pick a∈[amin,amax], x∈0..9, b∈0..9 → c = a·x + b. Returns (a,b,c,x_correct).
fn problem(seed: u32, amin: usize, amax: usize) -> (usize, usize, usize, usize) {
    let a = amin + (u(seed, 1) * (amax - amin + 1) as f32) as usize % (amax - amin + 1);
    let x = (u(seed, 2) * 10.0) as usize % 10; let b = (u(seed, 3) * 10.0) as usize % 10;
    (a, b, a * x + b, x)
}
fn feats(a: usize, b: usize, c: usize, x: usize) -> [f32; 4] { [a as f32 / 6.0, b as f32 / 9.0, c as f32 / 70.0, x as f32 / 9.0] }

// best-of-N: per problem, N candidate x∈0..9, verifier picks argmax(valid-logit); returns (bestofN, oracle, random)
async fn eval(ctx: &Arc<ferric_core::Context>, p: &[Tensor], amin: usize, amax: usize, nn: usize, trials: usize, seed0: u32) -> (f32, f32, f32) {
    let mut inp = vec![0.0f32; trials * nn * 4]; let mut probs = Vec::with_capacity(trials); let mut cand = vec![0usize; trials * nn];
    for tr in 0..trials { let (a, b, c, x) = problem(seed0 + tr as u32 * 7, amin, amax); probs.push((a, b, c, x));
        for j in 0..nn { let xx = (u((tr * nn + j) as u32, seed0 * 3 + 1) * 10.0) as usize % 10; cand[tr * nn + j] = xx; let f = feats(a, b, c, xx); for k in 0..4 { inp[(tr * nn + j) * 4 + k] = f[k]; } } }
    let lg = Tensor::from_vec(ctx, &inp, &[trials * nn, 4]).matmul(&p[0]).add(&p[1]).relu().matmul(&p[2]).add(&p[3]).relu().matmul(&p[4]).add(&p[5]).to_vec().await; // [trials*nn, 2]
    let (mut bon, mut orc, mut rnd) = (0usize, 0usize, 0usize);
    for tr in 0..trials { let (a, b, c, _) = probs[tr];
        let (mut bi, mut bv) = (tr * nn, f32::MIN); let mut any = false;
        for j in 0..nn { let idx = tr * nn + j; let valid_logit = lg[idx * 2] - lg[idx * 2 + 1]; if valid_logit > bv { bv = valid_logit; bi = idx; } if a * cand[idx] + b == c { any = true; } }
        if a * cand[bi] + b == c { bon += 1; } if any { orc += 1; } if a * cand[tr * nn] + b == c { rnd += 1; }
    }
    (bon as f32 / trials as f32, orc as f32 / trials as f32, rnd as f32 / trials as f32)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — algebra VERIFIER: solve a·x+b=c by best-of-N energy selection (train a∈1..3)");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let eps = Tensor::from_vec(&ctx, &[1e-6], &[1]);
    let bs = 512usize;

    // verifier = valid/invalid classifier over [a,b,c,x]: 4 → HE → HE → 2, softmax-CE
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(4 * HE, 1, 1.0 / 2.0), &[4, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * HE, 2, 1.0 / (HE as f32).sqrt()), &[HE, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * 2, 3, 1.0 / (HE as f32).sqrt()), &[HE, 2]), Tensor::zeros(&ctx, &[2]),
    ];
    let mut adam = Adam::new(&p, 0.002);
    let clf_v = |xv: &Var, p: &[Var]| -> Var { let h = xv.matmul(&p[0]).add(&p[1]).relu(); let h2 = h.matmul(&p[2]).add(&p[3]).relu(); h2.matmul(&p[4]).add(&p[5]) };
    for step in 0..6000 {
        // half positives (correct x), half negatives (wrong x) — same problems, a∈1..3
        let mut inp = vec![0.0f32; bs * 4]; let mut lab = vec![0.0f32; bs * 2];
        for i in 0..bs {
            let (a, b, c, x) = problem(step as u32 * 31 + i as u32 + 1, 1, 3);
            let pos = i % 2 == 0;
            let xx = if pos { x } else { let mut w = (u(i as u32, step as u32 * 5 + 7) * 10.0) as usize % 10; if w == x { w = (w + 1) % 10; } w };
            let f = feats(a, b, c, xx); for k in 0..4 { inp[i * 4 + k] = f[k]; }
            lab[i * 2 + if pos { 0 } else { 1 }] = 1.0;
        }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let logits = clf_v(&Var::leaf(Tensor::from_vec(&ctx, &inp, &[bs, 4])), &pv);
        let logp = logits.softmax(1).add(&Var::leaf(eps.clone())).log();
        let loss = Var::leaf(Tensor::from_vec(&ctx, &lab, &[bs, 2])).mul(&logp).mean_all().neg();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); adam.step(&mut p, &g);
    }
    for (label, amin, amax, s) in [("in-distribution (a∈1..3)", 1usize, 3usize, 900u32), ("OUT-OF-DISTRIBUTION (a∈4..6, unseen)", 4, 6, 5000)] {
        println!("\n  {label}:");
        println!("     {:>4}  {:>11}  {:>12}  {:>9}", "N", "energy-BoN", "oracle(any)", "random");
        for &nn in &[1usize, 2, 4, 8, 16] { let (bon, orc, rnd) = eval(&ctx, &p, amin, amax, nn, 400, s + nn as u32 * 13).await; println!("     {:>4}  {:>10.0}%  {:>11.0}%  {:>8.0}%", nn, bon * 100.0, orc * 100.0, rnd * 100.0); }
    }
    println!("\n  energy-BoN tracking the oracle IN-dist → verifier selects the correct algebra answer; if it also holds");
    println!("  OOD (a∈4..6, never trained) → the learned verifier GENERALIZES the a·x+b=c relation, not a memorized map.");
}
