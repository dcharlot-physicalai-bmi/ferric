//! EFA energy-first #19 — DOES THINKING SCALE WITH DIFFICULTY? The EBT's signature claim, measured.
//!
//! `ebm_ebt_true.rs` proved the true Energy-Based Transformer works on a 2-variable multivalued system
//! (train THROUGH an unrolled energy descent, 2nd-order autograd). The flagship EBT claim (Gladstone/Du 2025,
//! arXiv:2507.02092, Fig 7) is stronger: **a model that predicts by thinking should need MORE thinking on
//! HARDER problems** — adaptive test-time compute, which a fixed feed-forward pass structurally cannot do.
//!
//! Test it by scaling difficulty with a knob D (chain length). The task generalizes the 2-var system to a
//! CHAIN of D coupled variables with D−1 links:   ŷ_i² + ŷ_{i+1}² = a_i   ∧   ŷ_i·ŷ_{i+1} = b_i   (i=0..D−2).
//! Each link is multivalued (feed-forward regression averages the branches → 0%); a longer chain is a harder
//! joint constraint-satisfaction problem. We train one descent-trained EBT per D and read accuracy vs the
//! number of thinking steps K. Prediction under test: bigger D needs bigger K to reach the same accuracy.
//!
//! Ground truth: sample a valid ŷ*, DEFINE the a_i,b_i from it (so ≥1 solution always exists); ±ŷ* are both
//! valid, so supervise the descent endpoint to the nearer of {ŷ*, −ŷ*}; SCORE by constraint satisfaction.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_ebt_scale --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

// sample a valid ŷ* (each coord in ±[0.4,1.1]), then DEFINE the link constraints from it → context = [a_0..a_{D-2}, b_0..b_{D-2}]
fn problem(d: usize, seed: u32, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let ys: Vec<f32> = (0..d).map(|i| { let m = 0.4 + u(seed, i as u32 * 3 + 1) * (0.7 * scale); if u(seed, i as u32 * 3 + 2) > 0.5 { m } else { -m } }).collect();
    let mut ctx = Vec::with_capacity(2 * (d - 1));
    for i in 0..d - 1 { ctx.push(ys[i] * ys[i] + ys[i + 1] * ys[i + 1]); }
    for i in 0..d - 1 { ctx.push(ys[i] * ys[i + 1]); }
    (ctx, ys)
}
fn correct(d: usize, ctx: &[f32], y: &[f32]) -> bool {
    for i in 0..d - 1 {
        if (y[i] * y[i] + y[i + 1] * y[i + 1] - ctx[i]).abs() > 0.15 { return false; }
        if (y[i] * y[i + 1] - ctx[d - 1 + i]).abs() > 0.15 { return false; }
    }
    true
}

fn energy(yv: &Var, cx: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| z.exp().add(one).log();
    let h1 = sp(yv.matmul(&p[0]).add(&cx.matmul(&p[1])).add(&p[2]));
    let h2 = sp(h1.matmul(&p[3]).add(&p[4]));
    h2.matmul(&p[5]).add(&p[6])
}

// solve a fresh test set by K descent steps; return constraint-satisfaction fraction
async fn solve(ctx: &Arc<ferric_core::Context>, d: usize, p: &[Tensor], one: &Tensor, al: &Tensor, scale: f32, k: usize, t: usize, seed0: u32) -> f32 {
    let mut cxf = vec![0.0f32; t * 2 * (d - 1)]; let mut probs = Vec::with_capacity(t);
    for i in 0..t { let (c, _) = problem(d, seed0 + i as u32 * 7, scale); for (j, v) in c.iter().enumerate() { cxf[i * 2 * (d - 1) + j] = *v; } probs.push(c); }
    let cxv = Var::leaf(Tensor::from_vec(ctx, &cxf, &[t, 2 * (d - 1)]));
    let pv: Vec<Var> = p.iter().map(|x| Var::leaf(x.clone())).collect(); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
    let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(t * d, seed0 ^ 0xabc, 0.8), &[t, d]));
    for _ in 0..k { let e = energy(&y, &cxv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv)); }
    let yk = y.value().to_vec().await;
    let mut ok = 0; for i in 0..t { if correct(d, &probs[i], &yk[i * d..(i + 1) * d]) { ok += 1; } }
    ok as f32 / t as f32
}

// train one descent-trained EBT for chain length d at hidden width he; return its parameters
async fn train(ctx: &Arc<ferric_core::Context>, d: usize, he: usize, one: &Tensor) -> Vec<Tensor> {
    let cd = 2 * (d - 1);
    let mut p = vec![
        Tensor::from_vec(ctx, &randn(d * he, 10 + d as u32, 1.0 / (d as f32).sqrt()), &[d, he]),
        Tensor::from_vec(ctx, &randn(cd * he, 11 + d as u32, 1.0 / (cd as f32).sqrt()), &[cd, he]), Tensor::zeros(ctx, &[he]),
        Tensor::from_vec(ctx, &randn(he * he, 12 + d as u32, 1.0 / (he as f32).sqrt()), &[he, he]), Tensor::zeros(ctx, &[he]),
        Tensor::from_vec(ctx, &randn(he, 13 + d as u32, 1.0 / (he as f32).sqrt()), &[he, 1]), Tensor::zeros(ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.001);
    let bs = 96usize;
    for step in 0..3600 {
        let mut cxf = vec![0.0f32; bs * cd]; let mut stars = vec![0.0f32; bs * d];
        for i in 0..bs { let (c, ys) = problem(d, step as u32 * 131 + i as u32 * 7 + 1, 1.0);
            for (j, v) in c.iter().enumerate() { cxf[i * cd + j] = *v; } for (j, v) in ys.iter().enumerate() { stars[i * d + j] = *v; } }
        let cxv = Var::leaf(Tensor::from_vec(ctx, &cxf, &[bs, cd]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        // EBT stabilizers (Gladstone/Du 2025): RANDOMIZE step-count K and step-size α per batch, + Langevin noise.
        // This is the reported "biggest generalizer" — trains the energy to be descendable across thinking budgets.
        let ktr = 3 + (h32(step as u32 ^ 0x51ec) % 8) as usize;                       // K ∈ 3..10
        let a_step = 0.12 + (h32(step as u32 ^ 0xa17c) % 1000) as f32 / 1000.0 * 0.16; // α ∈ 0.12..0.28
        let alv = Var::leaf(Tensor::from_vec(ctx, &[a_step], &[1]));
        // UNROLL K descent steps ŷ ← ŷ − α·∂E/∂ŷ + noise (each differentiable in the weights → 2nd order)
        let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(bs * d, step as u32 * 17 + 3, 0.8), &[bs, d]));
        for si in 0..ktr {
            let e = energy(&y, &cxv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0);
            y = y.sub(&g.mul(&alv)).add(&Var::leaf(Tensor::from_vec(ctx, &randn(bs * d, step as u32 * 977 + si as u32 + 1, 0.02), &[bs, d])));
        }
        // supervise ŷ_K to the nearer of the guaranteed valid pair {ŷ*, −ŷ*}
        let yk = y.value().to_vec().await; let mut tgt = vec![0.0f32; bs * d];
        for i in 0..bs {
            let (mut dp, mut dn) = (0.0f32, 0.0f32);
            for j in 0..d { let s = stars[i * d + j]; dp += (yk[i * d + j] - s).powi(2); dn += (yk[i * d + j] + s).powi(2); }
            let sgn = if dp <= dn { 1.0 } else { -1.0 };
            for j in 0..d { tgt[i * d + j] = sgn * stars[i * d + j]; }
        }
        let diff = y.sub(&Var::leaf(Tensor::from_vec(ctx, &tgt, &[bs, d])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();  // backprop THROUGH the unrolled descent to the weights (2nd order)
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    p
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — IS THE D≥6 CEILING CAPACITY? scale model width at the hardest chain length");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let al = Tensor::from_vec(&ctx, &[0.2], &[1]);
    let ks = [3usize, 6, 12, 25];
    // Fix the hardest cases (D=6, D=8) and sweep hidden width. If the plateau RISES with width,
    // the earlier collapse was a capacity ceiling ("capacity floor, then compute"), not a fundamental limit.
    println!("\n  plateau solve accuracy (%) vs hidden width he   (feed-forward ≈ 0%; earlier HE=128 collapse: D6=12%):");
    print!("    D \\ he    "); for he in [128usize, 256, 512] { print!("he={:<5}", he); } println!("  (best over K∈{{3,6,12,25}})");
    for &d in &[6usize] {
        print!("    D={} ({} links) ", d, d - 1);
        for &he in &[128usize, 256, 512] {
            let p = train(&ctx, d, he, &one).await;
            let mut best = 0.0f32;
            for &k in &ks { let a = solve(&ctx, d, &p, &one, &al, 1.0, k, 400, 3000 + d as u32 * 1000 + he as u32).await * 100.0; if a > best { best = a; } }
            print!("{:>5.0} ", best);
        }
        println!();
    }
    println!("\n  If the plateau rises with width, the multivalued-chain ceiling is CAPACITY — bigger energy net,");
    println!("  more valid minima represented. That prices the limit: 'capacity floor, then compute', measured.");
}
