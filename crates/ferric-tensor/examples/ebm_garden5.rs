//! EFA energy-first #28 — EMERGENT COMMITMENT under STRONG pressure: give the well a real reason to exist.
//!
//! garden4 nulled: on an early-reliable / weak-late-noise task, a smooth integrator already scored ~91%, so there
//! was no pressure for a commitment well (and a warm-started well DECAYED). This sharpens the pressure: the late
//! tokens are STRONG (±2) and LONG (4 of them), so a smooth integrator is dragged OFF the reliable early signal —
//! and because embeddings are value-based (no position cue), the ONLY way to trust-early-and-resist-late is
//! temporal HYSTERESIS (a bistable well). If commitment ever emerges from learning, it should be here. Cold-start
//! learnable A (does it grow?) + warm-start control (does an imposed well persist=beneficial, or decay=useless?).
//! Read as performance-per-watt: accuracy vs K = capability vs compute; a committed strategy that holds the answer
//! cheaply would be an energy-efficient inference strategy the model discovered on its own.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_garden5 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const DH: usize = 8; const T: usize = 6; const EARLY: usize = 2;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

// STRONG pressure: early 2 tokens = reliable +2·dir (label=dir); late 4 tokens = STRONG ±2 noise (drags a smooth solver).
// tokens are indices into values {−2,−1,+1,+2}; strong tokens use idx 0(−2) / 3(+2).
fn item(seed: u32) -> (Vec<usize>, usize) {
    let dir = if u(seed, 1) > 0.5 { 1.0 } else { -1.0 };
    let mut e = vec![0usize; T];
    for t in 0..EARLY { e[t] = if dir > 0.0 { 3 } else { 0 }; }               // +2·dir reliable signal
    for t in EARLY..T { e[t] = if u(seed.wrapping_add(777), t as u32 + 1) > 0.5 { 3 } else { 0 }; } // ±2 strong noise
    (e, if dir > 0.0 { 1 } else { 0 })
}

// tp = [emb[4,DH], wg[DH,1], wr[1,1], br[1], aparam[1]]. A = softplus(aparam). a0 forces A=0.
fn forward(ctx: &Arc<ferric_core::Context>, tp: &[Var], toks: &Vec<Vec<usize>>, bs: usize, k: usize, a0: bool) -> Var {
    let (emb, wg, wr, br) = (&tp[0], &tp[1], &tp[2], &tp[3]);
    let one = Var::leaf(Tensor::from_vec(ctx, &[1.0], &[1]));
    let al = Var::leaf(Tensor::from_vec(ctx, &[0.1], &[1]));
    let four = if a0 { Var::leaf(Tensor::from_vec(ctx, &[0.0], &[1])) }
               else { tp[4].exp().add(&one).log().mul(&Var::leaf(Tensor::from_vec(ctx, &[4.0], &[1]))) };
    let mut d = Var::leaf(Tensor::zeros(ctx, &[bs, 1]));
    let mut h = Var::leaf(Tensor::zeros(ctx, &[bs, DH]));
    for t in 0..T {
        let mut oh = vec![0.0f32; bs * 4]; for i in 0..bs { oh[i * 4 + toks[t][i]] = 1.0; }
        h = h.add(&Var::leaf(Tensor::from_vec(ctx, &oh, &[bs, 4])).matmul(emb));
        let tilt = h.matmul(wg);
        for _ in 0..k {
            let dsq = d.mul(&d).sub(&one);
            let force = d.mul(&dsq).mul(&four).sub(&tilt);
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
        Tensor::from_vec(ctx, &[a_init], &[1]),
    ];
    let ntrain = if learn_a { 5 } else { 4 };
    let mut adam = Adam::new(&tp[..ntrain], 0.004);
    let bs = 64usize; let mut atraj = Vec::new();
    for step in 0..2600 {
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
        if log && step % 650 == 0 { let a = tp[4].to_vec().await[0]; atraj.push((a.exp()).ln_1p()); }
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
    println!("  EFA energy-first — EMERGENT COMMITMENT under STRONG pressure (early reliable, late STRONG ±2 noise)");
    // baseline: what does a pure smooth integrator score? (if already high, no pressure)
    let (tp, atraj) = train(&ctx, true, true, -3.0).await;      // COLD: A≈0.05
    let (tpw, atrajw) = train(&ctx, true, true, -0.2).await;    // WARM: A≈0.6
    let (tp0, _) = train(&ctx, false, false, -3.0).await;       // A frozen ≈0 (smooth baseline)

    print!("  COLD-start A (init 0.05):  "); for a in &atraj { print!("{:.3} → ", a); } println!("(final)");
    print!("  WARM-start A (init 0.60):  "); for a in &atrajw { print!("{:.3} → ", a); } println!("(final)");
    let (af, afw) = (*atraj.last().unwrap(), *atrajw.last().unwrap());
    println!("\n  PERFORMANCE-PER-COMPUTE (K = descent steps = energy axis): accuracy (%) vs K");
    print!("      model                        "); for k in [1usize, 2, 4, 8, 16] { print!("K={:<4}", k); } println!();
    print!("      cold-start (final A={:.2})         ", af); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tp, k, false, 60000).await); } println!();
    print!("      warm-start (final A={:.2})         ", afw); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tpw, k, false, 60000).await); } println!();
    print!("      smooth baseline (A frozen ≈0)    "); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tp0, k, false, 60000).await); } println!();
    print!("      warm-start, commitment ablated   "); for k in [1usize, 2, 4, 8, 16] { print!("{:>4.0} ", acc(&ctx, &tpw, k, true, 60000).await); } println!();
    println!("\n  Strong late noise SHOULD drag a smooth integrator off the early signal. If A now grows / persists");
    println!("  and beats the smooth baseline, commitment finally emerges because it earns its watts. If it still");
    println!("  nulls (A decays, all rows equal), commitment does NOT emerge in this architecture — an honest bound.");
}
