//! EFA energy-first #21 — the PREDICTIVE-CODING LANGUAGE TWIN: does inference EFFORT track surprisal?
//!
//! Human reading-time ∝ surprisal (Hale/Levy): the harder a word is to predict, the longer the brain dwells.
//! Predictive coding says the cortex predicts the next input and spends effort proportional to prediction error.
//! An energy-based LM predicts the next token by DESCENDING an energy E(context, ŷ) — so it, too, can spend
//! variable inference effort per token. A feed-forward LM cannot: one fixed pass, constant compute per token
//! (it can REPORT difficulty via output entropy, but cannot ALLOCATE compute to it). The falsifiable claim:
//! the EBM's descent EFFORT correlates with the TRUE surprisal of the context — the language twin of our
//! "thinking scales with difficulty" result, and the computational form of reading-time-∝-surprisal.
//!
//! Controlled test: a synthetic order-2 language over V=6 tokens whose per-context next-token distribution has
//! a KNOWN entropy gradient (some histories near-deterministic → low surprisal; some near-uniform → high). We
//! train an energy-based LM (predict = K steps of descent on the logits, 2nd-order autograd), then measure the
//! Pearson correlation between descent EFFORT (integrated energy "work"; steps-to-settle) and true entropy.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_lm_surprisal --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const V: usize = 6;          // vocabulary
const NC: usize = V * V;     // order-2 contexts (c1,c2)
const HE: usize = 96;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

// true generative process: each context (c1,c2) has a peaked-preference softened by a temperature τ that rises
// with (c1+c2) → a smooth entropy gradient from near-deterministic (low surprisal) to near-uniform (high).
fn true_dist(c1: usize, c2: usize) -> Vec<f32> {
    let tau = 0.18 + (c1 + c2) as f32 / (2.0 * (V - 1) as f32) * 2.4; // 0.18 (peaked) .. 2.58 (flat)
    let logits: Vec<f32> = (0..V).map(|k| { let s = (c1 * 7 + c2 * 13 + 3) as u32; (u(s, k as u32 + 1) * 2.0 - 1.0) * 2.5 }).collect();
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = logits.iter().map(|x| ((x - m) / tau).exp()).collect();
    let z: f32 = ex.iter().sum(); ex.iter().map(|x| x / z).collect()
}
fn entropy(p: &[f32]) -> f32 { -p.iter().map(|&x| if x > 1e-9 { x * x.ln() } else { 0.0 }).sum::<f32>() }

fn energy(cx: &Var, yv: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| z.exp().add(one).log();
    let h1 = sp(cx.matmul(&p[0]).add(&yv.matmul(&p[1])).add(&p[2]));
    let h2 = sp(h1.matmul(&p[3]).add(&p[4]));
    h2.matmul(&p[5]).add(&p[6])
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32; let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
    let mut sxy = 0.0; let mut sx = 0.0; let mut sy = 0.0;
    for i in 0..x.len() { let dx = x[i] - mx; let dy = y[i] - my; sxy += dx * dy; sx += dx * dx; sy += dy * dy; }
    if sx * sy < 1e-12 { 0.0 } else { sxy / (sx * sy).sqrt() }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — PREDICTIVE-CODING LANGUAGE TWIN: does inference effort track surprisal?");

    // build all NC contexts: one-hot(c1)+one-hot(c2) [NC, 2V] and true distributions
    let mut cxf = vec![0.0f32; NC * 2 * V]; let mut tp = vec![0.0f32; NC * V]; let mut hctx = vec![0.0f32; NC];
    for c1 in 0..V { for c2 in 0..V { let i = c1 * V + c2; cxf[i * 2 * V + c1] = 1.0; cxf[i * 2 * V + V + c2] = 1.0;
        let d = true_dist(c1, c2); for k in 0..V { tp[i * V + k] = d[k]; } hctx[i] = entropy(&d); } }

    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(2 * V * HE, 10, 1.0 / (2.0 * V as f32).sqrt()), &[2 * V, HE]),
        Tensor::from_vec(&ctx, &randn(V * HE, 11, 1.0 / (V as f32).sqrt()), &[V, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * HE, 12, 1.0 / (HE as f32).sqrt()), &[HE, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE, 13, 1.0 / (HE as f32).sqrt()), &[HE, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.002);
    let cxv0 = Var::leaf(Tensor::from_vec(&ctx, &cxf, &[NC, 2 * V]));
    let tpv = Var::leaf(Tensor::from_vec(&ctx, &tp, &[NC, V]));

    // train: predict = K descent steps on the logits ŷ; supervise softmax(ŷ_K) to the TRUE distribution (CE)
    for step in 0..3000 {
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let ktr = 3 + (h32(step as u32 ^ 0x51ec) % 8) as usize;
        let a_step = 0.15 + (h32(step as u32 ^ 0xa17c) % 1000) as f32 / 1000.0 * 0.20;
        let alv = Var::leaf(Tensor::from_vec(&ctx, &[a_step], &[1]));
        let mut y = Var::leaf(Tensor::from_vec(&ctx, &randn(NC * V, step as u32 * 17 + 3, 0.5), &[NC, V]));
        for _ in 0..ktr { let e = energy(&cxv0, &y, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv)); }
        let logp = y.softmax(1).add(&Var::leaf(Tensor::from_vec(&ctx, &vec![1e-9; NC * V], &[NC, V]))).log();
        let loss = tpv.mul(&logp).sum_all().neg(); // cross-entropy Σ −p_true·log p_model
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }

    // ---- measure: run a long descent per context, record the energy trajectory, extract effort ----
    let kmax = 40usize; let alpha = 0.25f32;
    let al = Tensor::from_vec(&ctx, &[alpha], &[1]);
    let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let alv = Var::leaf(al.clone());
    let mut y = Var::leaf(Tensor::from_vec(&ctx, &randn(NC * V, 777, 0.5), &[NC, V]));
    let mut etraj: Vec<Vec<f32>> = Vec::with_capacity(kmax + 1); // per-step per-context energy
    for _ in 0..=kmax {
        let epc = energy(&cxv0, &y, &pv, &ov).value().to_vec().await; // [NC]
        etraj.push(epc);
        let e = energy(&cxv0, &y, &pv, &ov).sum_all(); let g = grad(&e, &[y.clone()], None).remove(0); y = y.sub(&g.mul(&alv));
    }
    // model fit: KL(true‖model) per context (sanity that the LM learned the distribution)
    let pfin = y.softmax(1).value().to_vec().await;
    let mut work = vec![0.0f32; NC]; let mut steps95 = vec![0.0f32; NC]; let mut efinal = vec![0.0f32; NC]; let mut kl = vec![0.0f32; NC];
    for c in 0..NC {
        let e0 = etraj[0][c]; let ef = etraj[kmax][c]; efinal[c] = ef;
        let drop = (e0 - ef).max(1e-6);
        let mut w = 0.0; for k in 0..=kmax { w += (etraj[k][c] - ef).max(0.0); } work[c] = w;              // integrated descent "work"
        let mut s = kmax as f32; for k in 0..=kmax { if e0 - etraj[k][c] >= 0.95 * drop { s = k as f32; break; } } steps95[c] = s;
        let mut d = 0.0; for k in 0..V { let pt = tp[c * V + k]; let pm = pfin[c * V + k].max(1e-9); if pt > 1e-9 { d += pt * (pt / pm).ln(); } } kl[c] = d;
    }
    let mean_kl = kl.iter().sum::<f32>() / NC as f32;

    println!("\n  synthetic order-2 language: V={} tokens, {} contexts, true entropy spread {:.2}..{:.2} nats",
        V, NC, hctx.iter().cloned().fold(f32::MAX, f32::min), hctx.iter().cloned().fold(f32::MIN, f32::max));
    println!("  model fit: mean KL(true‖model) = {:.4} nats  (→0 = the EBM learned the true distribution)\n", mean_kl);
    println!("  DOES INFERENCE EFFORT TRACK SURPRISAL?  Pearson ρ across the {} contexts vs true entropy:", NC);
    println!("     integrated descent work  ρ = {:+.3}", pearson(&work, &hctx));
    println!("     steps-to-settle (95%)    ρ = {:+.3}", pearson(&steps95, &hctx));
    println!("     final (minimum) energy   ρ = {:+.3}", pearson(&efinal, &hctx));
    println!("\n  A feed-forward LM spends CONSTANT compute per token — its effort cannot correlate with anything.");
    println!("  If ρ(effort, surprisal) > 0, the energy-based LM allocates inference like reading-time does.");
}
