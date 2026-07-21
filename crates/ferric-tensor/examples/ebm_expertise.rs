//! EFA energy-first #37 — INTELLIGENCE = ECONOMY OF EFFORT (Krakauer): does the compute to SOLVE drop with expertise?
//!
//! Krakauer (SFI): "intelligence is making hard problems easy"; experts LOOK LESS — they recode the problem into a
//! representation where the answer is a reflex (system-2 → system-1). If EFA is on the right track, then as a
//! descent-trained Energy-Based Transformer LEARNS, the energy landscape should recode so that fewer descent steps
//! K (= less compute = fewer watts) are needed for the SAME capability. We measure the accuracy-vs-K curve at
//! training checkpoints on the multivalued system ŷ₀²+ŷ₁²=a ∧ ŷ₀ŷ₁=b. Prediction: the K needed to reach high
//! accuracy FALLS as training proceeds — expertise = economy of effort, measured in compute.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_expertise --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;
const HE: usize = 96;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn ctx_ab(seed: u32) -> (f32, f32) { let a = 0.6 + u(seed, 1) * 0.8; let b = (u(seed, 2) * 2.0 - 1.0) * (a / 2.0); (a, b) }
fn sols(a: f32, b: f32) -> Vec<(f32, f32)> { let d = a * a - 4.0 * b * b; if d < 0.0 { return vec![]; } let mut v = vec![]; for s in [(a + d.sqrt()) / 2.0, (a - d.sqrt()) / 2.0] { if s > 1e-4 { let y = s.sqrt(); v.push((y, b / y)); v.push((-y, -b / y)); } } v }
fn correct(a: f32, b: f32, y0: f32, y1: f32) -> bool { (y0 * y0 + y1 * y1 - a).abs() < 0.15 && (y0 * y1 - b).abs() < 0.15 }
fn nearest(a: f32, b: f32, y0: f32, y1: f32) -> (f32, f32) { let ss = sols(a, b); if ss.is_empty() { return (0.0, 0.0); } *ss.iter().min_by(|x, z| ((x.0 - y0).powi(2) + (x.1 - y1).powi(2)).partial_cmp(&((z.0 - y0).powi(2) + (z.1 - y1).powi(2))).unwrap()).unwrap() }
fn energy(yv: &Var, ab: &Var, p: &[Var], one: &Var) -> Var { let sp = |z: Var| z.exp().add(one).log(); let h1 = sp(yv.matmul(&p[0]).add(&ab.matmul(&p[1])).add(&p[2])); let h2 = sp(h1.matmul(&p[3]).add(&p[4])); h2.matmul(&p[5]).add(&p[6]) }
async fn solve(ctx: &Arc<ferric_core::Context>, p: &[Tensor], one: &Tensor, al: &Tensor, k: usize, t: usize, seed0: u32) -> f32 {
    let mut ab = vec![0.0f32; t * 2]; let mut probs = Vec::new();
    for i in 0..t { let (a, b) = ctx_ab(seed0 + i as u32 * 5); ab[i * 2] = a; ab[i * 2 + 1] = b; probs.push((a, b)); }
    let abv = Var::leaf(Tensor::from_vec(ctx, &ab, &[t, 2])); let pv: Vec<Var> = p.iter().map(|x| Var::leaf(x.clone())).collect(); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
    let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(t * 2, seed0 ^ 0xabc, 0.8), &[t, 2]));
    for _ in 0..k { let e = energy(&y, &abv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv)); }
    let yk = y.value().to_vec().await; let mut ok = 0; for i in 0..t { if correct(probs[i].0, probs[i].1, yk[i * 2], yk[i * 2 + 1]) { ok += 1; } } ok as f32 / t as f32
}
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — INTELLIGENCE = ECONOMY OF EFFORT: does the compute-to-solve drop with expertise?\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let al = Tensor::from_vec(&ctx, &[0.2], &[1]);
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(2 * HE, 10, 1.0 / 1.5), &[2, HE]), Tensor::from_vec(&ctx, &randn(2 * HE, 11, 1.0 / 1.5), &[2, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * HE, 12, 1.0 / (HE as f32).sqrt()), &[HE, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE, 13, 1.0 / (HE as f32).sqrt()), &[HE, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.001); let (bs, ktr) = (96usize, 6usize);
    let ks = [1usize, 2, 4, 8, 16, 32];
    let checkpoints = [60usize, 200, 600, 1400, 2600];
    println!("  accuracy (%) vs #descent steps K, at TRAINING CHECKPOINTS (K* = steps to first reach 90%):");
    print!("    train step   "); for k in ks { print!("K={:<4}", k); } println!("  | K* (watts to solve)");
    let mut ci = 0;
    for step in 0..2601 {
        if ci < checkpoints.len() && step == checkpoints[ci] {
            print!("    {:<9}    ", step); let mut kstar = String::from("—");
            for &k in &ks { let acc = solve(&ctx, &p, &one, &al, k, 400, 900).await * 100.0; print!("{:>4.0} ", acc); if kstar == "—" && acc >= 90.0 { kstar = format!("{}", k); } }
            println!("  | {}", kstar); ci += 1;
        }
        // one training step (train THROUGH the unrolled descent, 2nd order)
        let mut ab = vec![0.0f32; bs * 2]; let mut probs = Vec::new();
        for i in 0..bs { let (a, b) = ctx_ab(step as u32 * 31 + i as u32 + 1); ab[i * 2] = a; ab[i * 2 + 1] = b; probs.push((a, b)); }
        let abv = Var::leaf(Tensor::from_vec(&ctx, &ab, &[bs, 2]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
        let mut y = Var::leaf(Tensor::from_vec(&ctx, &randn(bs * 2, step as u32 * 7 + 3, 0.8), &[bs, 2]));
        for _ in 0..ktr { let e = energy(&y, &abv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv)); }
        let yk = y.value().to_vec().await; let mut tgt = vec![0.0f32; bs * 2];
        for i in 0..bs { let (s0, s1) = nearest(probs[i].0, probs[i].1, yk[i * 2], yk[i * 2 + 1]); tgt[i * 2] = s0; tgt[i * 2 + 1] = s1; }
        let diff = y.sub(&Var::leaf(Tensor::from_vec(&ctx, &tgt, &[bs, 2])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    println!("\n  If K* (the descent steps to solve) FALLS as training proceeds, the landscape has recoded so the hard");
    println!("  problem became easy — expertise = economy of effort = fewer watts for the same capability (Krakauer).");
}
