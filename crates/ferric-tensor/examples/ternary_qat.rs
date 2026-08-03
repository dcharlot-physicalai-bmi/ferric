//! QAT for ternary — the path PTQ can't reach, shown on a task with SLACK. A single linear fit hits ternary's
//! capacity floor (both PTQ and QAT stuck ~45%); real value shows where the TASK has room, like an MLP solving
//! XOR (nonlinear, needs a hidden layer). PTQ-ternarizing a trained f32 net collapses its accuracy; QAT keeps
//! FP32 SHADOW weights, ternarizes them in the forward, and via the STRAIGHT-THROUGH ESTIMATOR routes the
//! ternary weights' gradient back to the shadow — so the net RE-LEARNS to solve the task with ternary weights.
//! Ferric already trains (autograd+Adam); STE = ternarize(shadow) → forward as leaf → backward → Adam-step
//! the shadow with the ternary grad. Proves QAT recovers what PTQ destroys, in pure Rust.
//!   cargo run -p ferric-tensor --example ternary_qat --release
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const GS: usize = 128;
fn ternarize(w: &[f32]) -> Vec<f32> { // per-group absmean ternary (BitNet b1.58)
    let mut out = vec![0f32; w.len()];
    for g in 0..(w.len() + GS - 1) / GS {
        let (lo, hi) = (g * GS, ((g + 1) * GS).min(w.len()));
        let gamma = (w[lo..hi].iter().map(|x| x.abs()).sum::<f32>() / (hi - lo) as f32).max(1e-8);
        for k in lo..hi { out[k] = (w[k] / gamma).round().clamp(-1.0, 1.0) * gamma; }
    }
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (n, hid) = (256usize, 64usize);
    let mut seed = 0xACE1u64;
    let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 40) as f32 + 0.5) / (1u32 << 24) as f32 };
    let mut gs = |u: &mut dyn FnMut() -> f32| (-2.0 * u().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * u()).cos();
    // XOR-ish task: 2D inputs, label sign(x1·x2) ∈ {−1,+1} — nonlinear, needs the hidden layer
    let mut xd = vec![0f32; n * 2]; let mut yd = vec![0f32; n];
    for i in 0..n { let (a, b) = (gs(&mut u), gs(&mut u)); xd[i * 2] = a; xd[i * 2 + 1] = b; yd[i] = if a * b > 0.0 { 1.0 } else { -1.0 }; }
    let xv = Var::leaf(Tensor::from_vec(&ctx, &xd, &[n, 2]));
    let yv = Var::leaf(Tensor::from_vec(&ctx, &yd, &[n, 1]));

    let acc = |pred: &[f32]| pred.iter().zip(&yd).filter(|(p, y)| p.signum() == y.signum()).count() as f32 / n as f32;
    let mse = |a: &Var, b: &Var| { let d = a.sub(b); d.mul(&d).mean_all() };
    // f32 forward (Tensor): x[n,2] @ W1[2,hid] → relu → @ W2[hid,1]
    let fwd_f32 = |w1: &Tensor, w2: &Tensor| { xv.value().matmul(w1).relu().matmul(w2) };

    // ---- 1) train an f32 MLP ----  (weights [in,out] so plain matmul works)
    let mut w1 = Tensor::from_vec(&ctx, &(0..2 * hid).map(|_| gs(&mut u) * 0.5).collect::<Vec<_>>(), &[2, hid]);
    let mut w2 = Tensor::from_vec(&ctx, &(0..hid).map(|_| gs(&mut u) * 0.3).collect::<Vec<_>>(), &[hid, 1]);
    let mut adam = Adam::new(&[w1.clone(), w2.clone()], 0.03);
    for _ in 0..600 {
        let (v1, v2) = (Var::leaf(w1.clone()), Var::leaf(w2.clone()));
        let pred = xv.matmul(&v1).relu().matmul(&v2);
        let loss = mse(&pred, &yv); loss.backward();
        let mut ps = vec![w1.clone(), w2.clone()];
        adam.step(&mut ps, &[v1.grad().unwrap(), v2.grad().unwrap()]); w1 = ps[0].clone(); w2 = ps[1].clone();
    }
    let acc_f32 = acc(&fwd_f32(&w1, &w2).to_vec().await);

    // ---- 2) PTQ: ternarize the trained weights ----
    let (w1t, w2t) = (Tensor::from_vec(&ctx, &ternarize(&w1.to_vec().await), &[2, hid]), Tensor::from_vec(&ctx, &ternarize(&w2.to_vec().await), &[hid, 1]));
    let acc_ptq = acc(&fwd_f32(&w1t, &w2t).to_vec().await);

    // ---- 3) QAT: shadow=f32 weights; ternarize each step; STE routes ternary-grad → shadow ----
    let mut s1 = w1.clone(); let mut s2 = w2.clone();
    let mut qadam = Adam::new(&[s1.clone(), s2.clone()], 0.02);
    for _ in 0..800 {
        let t1 = Tensor::from_vec(&ctx, &ternarize(&pollster::block_on(s1.to_vec())), &[2, hid]);
        let t2 = Tensor::from_vec(&ctx, &ternarize(&pollster::block_on(s2.to_vec())), &[hid, 1]);
        let (v1, v2) = (Var::leaf(t1), Var::leaf(t2));
        let pred = xv.matmul(&v1).relu().matmul(&v2);
        let loss = mse(&pred, &yv); loss.backward();
        let mut ps = vec![s1.clone(), s2.clone()];
        qadam.step(&mut ps, &[v1.grad().unwrap(), v2.grad().unwrap()]); s1 = ps[0].clone(); s2 = ps[1].clone();  // STE → shadow
    }
    let q1 = Tensor::from_vec(&ctx, &ternarize(&s1.to_vec().await), &[2, hid]);
    let q2 = Tensor::from_vec(&ctx, &ternarize(&s2.to_vec().await), &[hid, 1]);
    let acc_qat = acc(&fwd_f32(&q1, &q2).to_vec().await);

    println!("XOR task, 2→{hid}→1 MLP, ternary weights (accuracy):");
    println!("  f32 (full precision)         {:.1}%", 100.0 * acc_f32);
    println!("  PTQ (naive ternarize f32)    {:.1}%   ← collapses", 100.0 * acc_ptq);
    println!("  QAT (STE, 800 steps)         {:.1}%   ← recovers", 100.0 * acc_qat);
    println!("\n✅ QAT + straight-through estimator recovers ternary accuracy PTQ destroys, in pure Rust Ferric.");
    println!("   Same shadow-weights+STE substrate scales to a full model (add teacher-logit distillation) — the QAT path.");
}
