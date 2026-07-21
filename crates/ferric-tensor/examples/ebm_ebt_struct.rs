//! EFA energy-first #20 — STRUCTURE vs SCALE: does a weight-shared local energy beat brute width?
//!
//! `ebm_ebt_scale.rs` found the true-EBT ceiling on a chain of D coupled multivalued links is CAPACITY — a
//! generic MLP energy needed ever-more width and still only reached 42% at D=6. But the chain is built from ONE
//! repeated primitive (every link is the same law ŷ_i²+ŷ_{i+1}²=a_i ∧ ŷ_iŷ_{i+1}=b_i). The energy-based right
//! answer is COMPOSITION: an energy that is a SUM of per-link local energies with SHARED weights —
//!     E(ŷ, ctx) = Σ_i g_θ(ŷ_i, ŷ_{i+1}, a_i, b_i)
//! Its parameter count is INDEPENDENT of D, so if it generalizes it solves arbitrary-length chains at fixed,
//! tiny capacity — structure beating scale. This is the EBM compositionality thesis, tested at the architecture
//! level, trained THROUGH the unrolled descent (2nd-order autograd) exactly like the generic EBT.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_ebt_struct --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const H: usize = 64; // shared local-energy hidden width — the SAME tiny net for every D

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn problem(d: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ys: Vec<f32> = (0..d).map(|i| { let m = 0.4 + u(seed, i as u32 * 3 + 1) * 0.7; if u(seed, i as u32 * 3 + 2) > 0.5 { m } else { -m } }).collect();
    let a: Vec<f32> = (0..d - 1).map(|i| ys[i] * ys[i] + ys[i + 1] * ys[i + 1]).collect();
    let b: Vec<f32> = (0..d - 1).map(|i| ys[i] * ys[i + 1]).collect();
    (a, b, ys)
}
fn correct(d: usize, a: &[f32], b: &[f32], y: &[f32]) -> bool {
    for i in 0..d - 1 { if (y[i] * y[i] + y[i + 1] * y[i + 1] - a[i]).abs() > 0.15 { return false; } if (y[i] * y[i + 1] - b[i]).abs() > 0.15 { return false; } }
    true
}

// left/right neighbor selector matrices: yL = ŷ@L picks cols 0..D-2, yR = ŷ@R picks cols 1..D-1
fn selectors(ctx: &Arc<ferric_core::Context>, d: usize) -> (Tensor, Tensor) {
    let (mut l, mut r) = (vec![0.0f32; d * (d - 1)], vec![0.0f32; d * (d - 1)]);
    for i in 0..d - 1 { l[i * (d - 1) + i] = 1.0; r[(i + 1) * (d - 1) + i] = 1.0; }
    (Tensor::from_vec(ctx, &l, &[d, d - 1]), Tensor::from_vec(ctx, &r, &[d, d - 1]))
}

// STRUCTURED energy: sum of shared local link-energies. g = [w_yL,w_yR,w_a,w_b (each [1,H]), b1 [H], W2 [H,1], b2 [1]]
fn senergy(y: &Var, af: &Var, bf: &Var, l: &Var, r: &Var, g: &[Var], one: &Var, d: usize, bs: usize) -> Var {
    let sp = |z: Var| z.exp().add(one).log();
    let m = bs * (d - 1);
    let yl = y.matmul(l).reshape(&[m, 1]);
    let yr = y.matmul(r).reshape(&[m, 1]);
    let a = af.reshape(&[m, 1]);
    let b = bf.reshape(&[m, 1]);
    let pre = yl.matmul(&g[0]).add(&yr.matmul(&g[1])).add(&a.matmul(&g[2])).add(&b.matmul(&g[3])).add(&g[4]);
    sp(pre).matmul(&g[5]).add(&g[6]) // [m,1] per-link energies; sum_all → total (links only touch their own ŷ)
}

async fn solve(ctx: &Arc<ferric_core::Context>, d: usize, g: &[Tensor], l: &Tensor, r: &Tensor, one: &Tensor, al: &Tensor, k: usize, t: usize, seed0: u32) -> f32 {
    let (mut af, mut bf) = (vec![0.0f32; t * (d - 1)], vec![0.0f32; t * (d - 1)]); let mut probs = Vec::with_capacity(t);
    for i in 0..t { let (a, b, _) = problem(d, seed0 + i as u32 * 7); for j in 0..d - 1 { af[i * (d - 1) + j] = a[j]; bf[i * (d - 1) + j] = b[j]; } probs.push((a, b)); }
    let afv = Var::leaf(Tensor::from_vec(ctx, &af, &[t, d - 1])); let bfv = Var::leaf(Tensor::from_vec(ctx, &bf, &[t, d - 1]));
    let lv = Var::leaf(l.clone()); let rv = Var::leaf(r.clone()); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
    let gv: Vec<Var> = g.iter().map(|x| Var::leaf(x.clone())).collect();
    let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(t * d, seed0 ^ 0xabc, 0.8), &[t, d]));
    for _ in 0..k { let e = senergy(&y, &afv, &bfv, &lv, &rv, &gv, &ov, d, t).sum_all(); let gr = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&gr.mul(&alv)); }
    let yk = y.value().to_vec().await;
    let mut ok = 0; for i in 0..t { if correct(d, &probs[i].0, &probs[i].1, &yk[i * d..(i + 1) * d]) { ok += 1; } } ok as f32 / t as f32
}

async fn train(ctx: &Arc<ferric_core::Context>, d: usize, l: &Tensor, r: &Tensor, one: &Tensor) -> Vec<Tensor> {
    let mut g = vec![
        Tensor::from_vec(ctx, &randn(H, 20, 0.7), &[1, H]), Tensor::from_vec(ctx, &randn(H, 21, 0.7), &[1, H]),
        Tensor::from_vec(ctx, &randn(H, 22, 0.7), &[1, H]), Tensor::from_vec(ctx, &randn(H, 23, 0.7), &[1, H]),
        Tensor::zeros(ctx, &[H]), Tensor::from_vec(ctx, &randn(H, 24, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(ctx, &[1]),
    ];
    let mut adam = Adam::new(&g, 0.002);
    let bs = 96usize;
    for step in 0..3000 {
        let (mut af, mut bf, mut stars) = (vec![0.0f32; bs * (d - 1)], vec![0.0f32; bs * (d - 1)], vec![0.0f32; bs * d]);
        for i in 0..bs { let (a, b, ys) = problem(d, step as u32 * 131 + i as u32 * 7 + 1);
            for j in 0..d - 1 { af[i * (d - 1) + j] = a[j]; bf[i * (d - 1) + j] = b[j]; } for j in 0..d { stars[i * d + j] = ys[j]; } }
        let afv = Var::leaf(Tensor::from_vec(ctx, &af, &[bs, d - 1])); let bfv = Var::leaf(Tensor::from_vec(ctx, &bf, &[bs, d - 1]));
        let lv = Var::leaf(l.clone()); let rv = Var::leaf(r.clone()); let ov = Var::leaf(one.clone());
        let gv: Vec<Var> = g.iter().map(|t| Var::leaf(t.clone())).collect();
        let ktr = d + (h32(step as u32 ^ 0x51ec) % (2 * d) as u32) as usize; // FAIR: K ∝ D (local descent needs ~D steps to propagate along the chain)
        let a_step = 0.12 + (h32(step as u32 ^ 0xa17c) % 1000) as f32 / 1000.0 * 0.16;
        let alv = Var::leaf(Tensor::from_vec(ctx, &[a_step], &[1]));
        let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(bs * d, step as u32 * 17 + 3, 0.8), &[bs, d]));
        for si in 0..ktr {
            let e = senergy(&y, &afv, &bfv, &lv, &rv, &gv, &ov, d, bs).sum_all();
            let gr = grad(&e, &[y.clone()], None).remove(0);
            y = y.sub(&gr.mul(&alv)).add(&Var::leaf(Tensor::from_vec(ctx, &randn(bs * d, step as u32 * 977 + si as u32 + 1, 0.02), &[bs, d])));
        }
        let yk = y.value().to_vec().await; let mut tgt = vec![0.0f32; bs * d];
        for i in 0..bs { let (mut dp, mut dn) = (0.0f32, 0.0f32);
            for j in 0..d { let s = stars[i * d + j]; dp += (yk[i * d + j] - s).powi(2); dn += (yk[i * d + j] + s).powi(2); }
            let sgn = if dp <= dn { 1.0 } else { -1.0 }; for j in 0..d { tgt[i * d + j] = sgn * stars[i * d + j]; } }
        let diff = y.sub(&Var::leaf(Tensor::from_vec(ctx, &tgt, &[bs, d])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();
        let gg: Vec<Tensor> = gv.iter().zip(&g).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut g, &gg);
    }
    g
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — STRUCTURE vs SCALE, FAIR RE-TEST: weight-shared local energy with K ∝ D descent budget");
    println!("  (the first run used a FIXED K≤25 which cannot propagate constraints along a long chain — a possible design artifact)");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let al = Tensor::from_vec(&ctx, &[0.2], &[1]);
    println!("\n  solve accuracy (%) vs D, sweeping the thinking budget K in MULTIPLES of D   (fixed-K≤25 first run: D2/4/6/8/12 = 90/8/2/0/0):");
    print!("    D            K=D    K=2D   K=4D   K=8D   K=16D  K=32D   | best\n");
    for &d in &[6usize, 8] {
        let (l, r) = selectors(&ctx, d);
        let g = train(&ctx, d, &l, &r, &one).await;
        print!("    D={:<2} ({:>2} links) ", d, d - 1);
        let mut best = 0.0f32;
        for m in [1usize, 2, 4, 8, 16, 32] { let k = (m * d).max(1); let a = solve(&ctx, d, &g, &l, &r, &one, &al, k, 400, 7000 + d as u32 * 100 + m as u32).await * 100.0; if a > best { best = a; } print!("{:>5.0} ", a); }
        println!("  | {:.0}", best);
    }
    println!("\n  If accuracy now HOLDS as D grows (where fixed-K≤25 collapsed), the earlier 'structure is worse' was a");
    println!("  BUDGET artifact — local energy needs thinking-time ∝ problem size. If it STILL collapses, the negative is real.");
}
