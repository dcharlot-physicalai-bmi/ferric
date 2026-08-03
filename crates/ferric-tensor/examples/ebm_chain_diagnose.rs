//! EFA energy-first — WHAT IS THE MULTIVALUED-CHAIN CEILING, REALLY? A decomposition, not an assertion.
//!
//! `ebm_ebt_scale.rs` found that a descent-trained EBT solves a chain of D coupled multivalued links well at D≤4 and
//! collapses by D=6, and that widening the energy net lifts the D=6 plateau (13→25→42% at he=128/256/512). From those
//! three points it concluded "the ceiling is CAPACITY, not fundamental." That conclusion is plausible but it was
//! INFERRED, not measured — and this arc has already been burned once by an asserted-not-measured claim (the
//! "feed-forward = 0%" line in ebm_ebt_true, which turned out to be 100% when someone finally ran it).
//!
//! So measure it. The reported accuracy is GLOBAL: `correct()` demands every link hold at once. That single number
//! cannot distinguish three very different worlds, which have three different fixes:
//!
//!   (1) REPRESENTATION  — the per-link solve quality itself degrades as D grows. Fix: capacity (width/depth).
//!   (2) COMPOUNDING     — per-link quality stays high, but a chain needs (D−1) independent successes, so global
//!                          accuracy decays like p^(D−1) no matter how good p is. Fix: NOT width — you need a
//!                          different inference structure (message passing, sequential decoding), because holding
//!                          p fixed and growing D is exponentially hopeless.
//!   (3) BASIN SELECTION — failures are CORRELATED: descent commits to a globally wrong branch and misses almost
//!                          every link at once. Fix: the sampler/init (restarts, annealing, best-of-N), not capacity.
//!
//! The decisive signal is the DISTRIBUTION of how many links each attempt satisfies. Independent compounding gives a
//! smooth binomial-looking spread around p(D−1). Basin selection gives a BIMODAL all-or-nothing histogram. We report
//! per-link rate, the compounding prediction p^(D−1), the measured global rate, and the full histogram, so the reader
//! can see which world we are in instead of taking our word for it.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_chain_diagnose --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn problem(d: usize, seed: u32, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let ys: Vec<f32> = (0..d).map(|i| { let m = 0.4 + u(seed, i as u32 * 3 + 1) * (0.7 * scale); if u(seed, i as u32 * 3 + 2) > 0.5 { m } else { -m } }).collect();
    let mut ctx = Vec::with_capacity(2 * (d - 1));
    for i in 0..d - 1 { ctx.push(ys[i] * ys[i] + ys[i + 1] * ys[i + 1]); }
    for i in 0..d - 1 { ctx.push(ys[i] * ys[i + 1]); }
    (ctx, ys)
}
// how many of the (d-1) links this candidate satisfies  (the global metric is links_ok == d-1)
fn links_ok(d: usize, cx: &[f32], y: &[f32]) -> usize {
    let mut n = 0;
    for i in 0..d - 1 {
        if (y[i] * y[i] + y[i + 1] * y[i + 1] - cx[i]).abs() <= 0.15 && (y[i] * y[i + 1] - cx[d - 1 + i]).abs() <= 0.15 { n += 1; }
    }
    n
}

fn energy(yv: &Var, cx: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| z.exp().add(one).log();
    let h1 = sp(yv.matmul(&p[0]).add(&cx.matmul(&p[1])).add(&p[2]));
    let h2 = sp(h1.matmul(&p[3]).add(&p[4]));
    h2.matmul(&p[5]).add(&p[6])
}

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
    for step in 0..320 {
        let mut cxf = vec![0.0f32; bs * cd]; let mut stars = vec![0.0f32; bs * d];
        for i in 0..bs { let (c, ys) = problem(d, step as u32 * 131 + i as u32 * 7 + 1, 1.0);
            for (j, v) in c.iter().enumerate() { cxf[i * cd + j] = *v; } for (j, v) in ys.iter().enumerate() { stars[i * d + j] = *v; } }
        let cxv = Var::leaf(Tensor::from_vec(ctx, &cxf, &[bs, cd]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let ktr = 3 + (h32(step as u32 ^ 0x51ec) % 8) as usize;
        let a_step = 0.12 + (h32(step as u32 ^ 0xa17c) % 1000) as f32 / 1000.0 * 0.16;
        let alv = Var::leaf(Tensor::from_vec(ctx, &[a_step], &[1]));
        let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(bs * d, step as u32 * 17 + 3, 0.8), &[bs, d]));
        for si in 0..ktr {
            let e = energy(&y, &cxv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0);
            y = y.sub(&g.mul(&alv)).add(&Var::leaf(Tensor::from_vec(ctx, &randn(bs * d, step as u32 * 977 + si as u32 + 1, 0.02), &[bs, d])));
        }
        let yk = y.value().to_vec().await; let mut tgt = vec![0.0f32; bs * d];
        for i in 0..bs {
            let (mut dp, mut dn) = (0.0f32, 0.0f32);
            for j in 0..d { let s = stars[i * d + j]; dp += (yk[i * d + j] - s).powi(2); dn += (yk[i * d + j] + s).powi(2); }
            let sgn = if dp <= dn { 1.0 } else { -1.0 };
            for j in 0..d { tgt[i * d + j] = sgn * stars[i * d + j]; }
        }
        let diff = y.sub(&Var::leaf(Tensor::from_vec(ctx, &tgt, &[bs, d])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    p
}

// solve a test set and return (global_rate, per_link_rate, histogram of #links satisfied)
async fn diagnose(ctx: &Arc<ferric_core::Context>, d: usize, p: &[Tensor], one: &Tensor, k: usize, t: usize, seed0: u32) -> (f32, f32, Vec<usize>) {
    let cd = 2 * (d - 1);
    let mut cxf = vec![0.0f32; t * cd]; let mut probs = Vec::with_capacity(t);
    for i in 0..t { let (c, _) = problem(d, seed0 + i as u32 * 7, 1.0); for (j, v) in c.iter().enumerate() { cxf[i * cd + j] = *v; } probs.push(c); }
    let cxv = Var::leaf(Tensor::from_vec(ctx, &cxf, &[t, cd]));
    let pv: Vec<Var> = p.iter().map(|x| Var::leaf(x.clone())).collect(); let ov = Var::leaf(one.clone());
    let alv = Var::leaf(Tensor::from_vec(ctx, &[0.2f32], &[1]));
    let mut y = Var::leaf(Tensor::from_vec(ctx, &randn(t * d, seed0 ^ 0xabc, 0.8), &[t, d]));
    for _ in 0..k { let e = energy(&y, &cxv, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv)); }
    let yk = y.value().to_vec().await;
    let mut hist = vec![0usize; d];                      // index = #links satisfied, 0..=d-1
    let mut tot_links = 0usize;
    for i in 0..t { let n = links_ok(d, &probs[i], &yk[i * d..(i + 1) * d]); hist[n] += 1; tot_links += n; }
    let global = hist[d - 1] as f32 / t as f32;
    let per_link = tot_links as f32 / (t * (d - 1)) as f32;
    (global, per_link, hist)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let one = Tensor::from_vec(&ctx, &[1.0f32], &[1]);
    println!("WHAT IS THE MULTIVALUED-CHAIN CEILING? Decomposing a global metric that hides three different failures.\n");
    println!("  A chain of D coupled multivalued links; a solve counts only if ALL D−1 links hold. We report the");
    println!("  per-link rate p, the INDEPENDENT-COMPOUNDING prediction p^(D−1), the measured global rate, and the");
    println!("  histogram of how many links each attempt satisfied. Which of the three worlds we are in is visible.\n");

    let he = 256usize; let k = 12usize; let t = 400usize;
    println!("  [width he={he}, thinking K={k}, {t} problems]");
    println!("    {:>3} {:>8} {:>11} {:>16} {:>10}   {}", "D", "links", "per-link p", "p^(D−1) predict", "GLOBAL", "histogram of #links satisfied (0 → D−1)");
    for &d in &[2usize, 3, 4, 5, 6, 7] {
        let p = train(&ctx, d, he, &one).await;
        let (g, pl, hist) = diagnose(&ctx, d, &p, &one, k, t, 4000 + d as u32 * 91).await;
        let pred = pl.powi(d as i32 - 1) * 100.0;
        let hs: Vec<String> = hist.iter().map(|&c| format!("{:.0}", 100.0 * c as f32 / t as f32)).collect();
        println!("    {:>3} {:>8} {:>10.1}% {:>15.1}% {:>9.1}%   [{}]", d, d - 1, pl * 100.0, pred, g * 100.0, hs.join(" "));
    }

    println!("\n  HOW TO READ THE HISTOGRAM: each bucket is the % of attempts satisfying that many links, from 0 on the");
    println!("  left to all D−1 on the right. A smooth hump in the middle = independent per-link failures compounding.");
    println!("  Mass piled at BOTH ends = correlated, all-or-nothing basin selection. Mass sliding left as D grows with");
    println!("  the right tail thinning = per-link representation degrading.\n");

    println!("  [does the per-link rate itself depend on capacity, or only the global rate?]");
    println!("    {:>3} {:>6} {:>12} {:>12} {:>16}", "D", "he", "per-link p", "GLOBAL", "p^(D−1) predict");
    for &d in &[6usize] {
        for &hw in &[128usize, 512] {
            let p = train(&ctx, d, hw, &one).await;
            let (g, pl, _) = diagnose(&ctx, d, &p, &one, k, t, 7000 + d as u32 * 91 + hw as u32).await;
            println!("    {:>3} {:>6} {:>11.1}% {:>11.1}% {:>15.1}%", d, hw, pl * 100.0, g * 100.0, pl.powi(d as i32 - 1) * 100.0);
        }
    }
    println!("\n  If widening lifts the GLOBAL rate mainly by lifting p, the ceiling is representation and 'capacity' was");
    println!("  the right word. If p is already high and flat while GLOBAL still collapses, the ceiling is COMPOUNDING —");
    println!("  and no affordable width fixes it, because p^(D−1) punishes chain length exponentially. That distinction");
    println!("  decides whether the next rung is a bigger energy net or a different inference structure.");
}
