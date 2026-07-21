//! EFA energy-first #27 — EMERGENT COMMITMENT: does the model LEARN to be bistable, instead of it being imposed?
//!
//! v3 showed reanalysis is a thinking-resolvable cost — but the double-well (commitment) was HAND-IMPOSED. The
//! deeper question: does commitment EMERGE from learning? v1 showed that a free energy on a plain integrate-the-
//! evidence task just becomes a SMOOTH integrator (no commitment). So emergence needs a REASON: a task where
//! committing is optimal. Here early tokens are reliable SIGNAL and late tokens are same-magnitude NOISE — and
//! because token embeddings are value-based (NO position info), the only way to trust-early-and-resist-late is
//! temporal HYSTERESIS. We make the well depth A a LEARNED parameter (init ≈0, i.e. a smooth integrator) and ask:
//! does training GROW it? Emergence is confirmed if (a) A rises from ~0, and (b) forcing A=0 at test HURTS accuracy
//! (commitment is load-bearing), and beaten by a fair A-frozen-at-0 model trained from scratch.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_garden4 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const DH: usize = 8; const T: usize = 6; const EARLY: usize = 3;
const EV: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

// task: early tokens (0..EARLY) are the reliable SIGNAL; label = sign(early sum). late tokens are same-value NOISE.
// early sum forced nonzero. returns token indices [T] and label.
fn item(seed: u32) -> (Vec<usize>, usize) {
    let mut e = vec![0usize; T]; let mut es = 0.0f32; let mut tries = 0;
    loop {
        es = 0.0; for t in 0..EARLY { let k = (u(seed.wrapping_add(tries * 131), t as u32 + 1) * 4.0) as usize % 4; e[t] = k; es += EV[k]; }
        if es.abs() > 0.5 { break; } tries += 1; if tries > 20 { es = 1.0; e[0] = 3; break; }
    }
    for t in EARLY..T { e[t] = (u(seed.wrapping_add(999), t as u32 + 1) * 4.0) as usize % 4; } // late = same-value noise
    (e, if es > 0.0 { 1 } else { 0 })
}

// tp = [emb[4,DH], wg[DH,1], wr[1,1], br[1], aparam[1]]. A = softplus(aparam). a0=true forces A=0 (smooth ablation).
fn forward(ctx: &Arc<ferric_core::Context>, tp: &[Var], toks: &Vec<Vec<usize>>, bs: usize, k: usize, a0: bool) -> Var {
    let (emb, wg, wr, br) = (&tp[0], &tp[1], &tp[2], &tp[3]);
    let one = Var::leaf(Tensor::from_vec(ctx, &[1.0], &[1]));
    let al = Var::leaf(Tensor::from_vec(ctx, &[0.1], &[1]));
    let four = if a0 { Var::leaf(Tensor::from_vec(ctx, &[0.0], &[1])) }
               else { tp[4].exp().add(&one).log().mul(&Var::leaf(Tensor::from_vec(ctx, &[4.0], &[1]))) }; // 4·softplus(aparam)
    let mut d = Var::leaf(Tensor::zeros(ctx, &[bs, 1]));
    let mut h = Var::leaf(Tensor::zeros(ctx, &[bs, DH]));
    for t in 0..T {
        let mut oh = vec![0.0f32; bs * 4]; for i in 0..bs { oh[i * 4 + toks[t][i]] = 1.0; }
        h = h.add(&Var::leaf(Tensor::from_vec(ctx, &oh, &[bs, 4])).matmul(emb));
        let tilt = h.matmul(wg);
        for _ in 0..k {
            let dsq = d.mul(&d).sub(&one);
            let force = d.mul(&dsq).mul(&four).sub(&tilt); // 4A·d(d²−1) − tilt
            d = d.sub(&force.mul(&al));
        }
    }
    d.mul(wr).add(br)
}

async fn train(ctx: &Arc<ferric_core::Context>, learn_a: bool, log: bool, a_init: f32) -> (Vec<Tensor>, Vec<f32>) {
    let mut tp = vec![
        Tensor::from_vec(ctx, &randn(4 * DH, 13, 0.5), &[4, DH]),
        Tensor::from_vec(ctx, &randn(DH, 14, 0.4), &[DH, 1]),
        Tensor::from_vec(ctx, &[3.0], &[1, 1]), Tensor::zeros(ctx, &[1]),
        Tensor::from_vec(ctx, &[a_init], &[1]), // aparam init (softplus → well depth A)
    ];
    let ntrain = if learn_a { 5 } else { 4 }; // freeze aparam if !learn_a (stays A≈0.05≈0)
    let mut adam = Adam::new(&tp[..ntrain], 0.004);
    let bs = 64usize; let mut atraj = Vec::new();
    for step in 0..2200 {
        let mut toks = vec![vec![0usize; bs]; T]; let mut lbl = vec![0.0f32; bs];
        for i in 0..bs { let (e, l) = item(step as u32 * 977 + i as u32 * 7 + 1); for t in 0..T { toks[t][i] = e[t]; } lbl[i] = l as f32; }
        let ktr = 3 + (h32(step as u32 ^ 0x51ec) % 5) as usize;
        let tpv: Vec<Var> = tp.iter().map(|t| Var::leaf(t.clone())).collect();
        let logit = forward(ctx, &tpv, &toks, bs, ktr, false);
        let one = Var::leaf(Tensor::from_vec(ctx, &[1.0], &[1]));
        let pr = one.div(&one.add(&logit.neg().exp()));
        let eps = Var::leaf(Tensor::from_vec(ctx, &vec![1e-6; bs], &[bs, 1]));
        let yv = Var::leaf(Tensor::from_vec(ctx, &lbl, &[bs, 1]));
        let loss = yv.mul(&pr.add(&eps).log()).add(&one.sub(&yv).mul(&one.sub(&pr).add(&eps).log())).mean_all().neg();
        loss.backward();
        let g: Vec<Tensor> = (0..ntrain).map(|i| tpv[i].grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; tp[i].numel()], &tp[i].shape))).collect();
        let mut head: Vec<Tensor> = tp[..ntrain].to_vec();
        adam.step(&mut head, &g);
        for i in 0..ntrain { tp[i] = head[i].clone(); }
        if log && step % 550 == 0 { let a = tp[4].to_vec().await[0]; atraj.push((a.exp()).ln_1p()); }
    }
    let a = tp[4].to_vec().await[0]; atraj.push((a.exp()).ln_1p());
    (tp, atraj)
}

async fn acc(ctx: &Arc<ferric_core::Context>, tp: &[Tensor], k: usize, a0: bool, seed0: u32) -> f32 {
    let n = 500usize; let mut toks = vec![vec![0usize; n]; T]; let mut lbl = vec![0usize; n];
    for i in 0..n { let (e, l) = item(seed0 + i as u32 * 7); for t in 0..T { toks[t][i] = e[t]; } lbl[i] = l; }
    let tpv: Vec<Var> = tp.iter().map(|t| Var::leaf(t.clone())).collect();
    let lv = forward(ctx, &tpv, &toks, n, k, a0).value().to_vec().await;
    let mut ok = 0; for i in 0..n { if (if lv[i] > 0.0 { 1 } else { 0 }) == lbl[i] { ok += 1; } } ok as f32 / n as f32 * 100.0
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — EMERGENT COMMITMENT: does the model LEARN to be bistable (well depth A), unforced?");
    println!("  task: early {} tokens = reliable signal (label=sign), late {} = same-value NOISE; value-based emb (no position)\n", EARLY, T - EARLY);

    let (tp, atraj) = train(&ctx, true, true, -3.0).await;      // COLD start: A≈0.05 (does commitment emerge?)
    let (tpw, atrajw) = train(&ctx, true, true, -0.2).await;    // WARM start: A≈0.6 (does a real well PERSIST / help?)
    let (tp0, _) = train(&ctx, false, false, -3.0).await;       // A frozen ≈0 (pure smooth integrator baseline)

    print!("  COLD-start well-depth A (init 0.05):  "); for a in &atraj { print!("{:.3} → ", a); } println!("(final) — did commitment EMERGE?");
    print!("  WARM-start well-depth A (init 0.60):  "); for a in &atrajw { print!("{:.3} → ", a); } println!("(final) — did a real well PERSIST (=beneficial) or DECAY (=useless)?");
    let af = *atraj.last().unwrap(); let afw = *atrajw.last().unwrap();

    // PERFORMANCE PER UNIT COMPUTE (K = descent steps = energy): accuracy vs thinking budget
    println!("\n  PERFORMANCE-PER-COMPUTE (K = descent steps = the energy axis; NOT tokens): accuracy (%) vs K");
    print!("      model                        "); for k in [1usize, 2, 4, 8, 16] { print!("K={:<4}", k); } println!();
    print!("      cold-start (final A={:.2})         ", af); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tp, k, false, 60000).await); } println!();
    print!("      warm-start (final A={:.2})         ", afw); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tpw, k, false, 60000).await); } println!();
    print!("      smooth baseline (A frozen ≈0)    "); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tp0, k, false, 60000).await); } println!();
    print!("      warm-start, commitment ablated   "); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tpw, k, true, 60000).await); } println!();
    println!("\n  Emergence is confirmed if A grew well above 0.05, the emergent model beats the smooth baseline,");
    println!("  and ablating its commitment (A=0) collapses it. Read the win as PERFORMANCE PER WATT: more");
    println!("  capability at equal/less compute — the metric that matters for on-device physical AI, not tokens.");
}
