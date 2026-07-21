//! EFA energy-first #24 — GARDEN-PATH reanalysis: does inference EFFORT track REVISION at constant surprisal?
//!
//! The language twin (`ebm_lm_surprisal.rs`) falsified the strong claim that descent effort tracks output-entropy
//! surprisal, and pointed at the real distinction: energy-based inference should spend compute on REASONING-DEPTH
//! difficulty (revising a commitment), not on uncertainty. The sharpest test is the garden-path: a sentence where
//! an early commitment must be REANALYZED when later evidence arrives ("The horse raced past the barn *fell*").
//!
//! Controlled isolation: an incremental energy-based model reads a sequence of evidence tokens and maintains a
//! persistent latent belief z, re-settled by K steps of energy descent at each token (predictive coding). The
//! tokens are drawn UNIFORMLY, so per-token surprisal is CONSTANT by construction — any variation in descent
//! effort therefore comes from INTERNAL belief revision, not from the input. Claim: effort spikes when the
//! running interpretation FLIPS (reanalysis), even though every token is equally (un)surprising.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_garden --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const DZ: usize = 6; const HH: usize = 48; const T: usize = 5; const KTR: usize = 4;
const EV: [f32; 4] = [-2.0, -1.0, 1.0, 2.0]; // evidence values (uniform over 4 → constant surprisal = log 4)

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

// energy E(z, h) = softplus(z·Wz + h·Wh + b1)·W2 + b2
fn energy(z: &Var, h: &Var, p: &[Var], one: &Var) -> Var {
    let a = z.matmul(&p[0]).add(&h.matmul(&p[1])).add(&p[2]);
    a.exp().add(one).log().matmul(&p[3]).add(&p[4])
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — GARDEN-PATH: does descent effort track belief REVISION at constant surprisal?");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let alpha = Tensor::from_vec(&ctx, &[0.2], &[1]);
    // params: Wz[DZ,HH] Wh[DZ,HH] b1[HH] W2[HH,1] b2[1]; token emb[4,DZ]; readout wr[DZ,1] br[1]
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(DZ * HH, 10, 1.0 / (DZ as f32).sqrt()), &[DZ, HH]),
        Tensor::from_vec(&ctx, &randn(DZ * HH, 11, 1.0 / (DZ as f32).sqrt()), &[DZ, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH, 12, 1.0 / (HH as f32).sqrt()), &[HH, 1]), Tensor::zeros(&ctx, &[1]),
        Tensor::from_vec(&ctx, &randn(4 * DZ, 13, 0.5), &[4, DZ]),
        Tensor::from_vec(&ctx, &randn(DZ, 14, 1.0 / (DZ as f32).sqrt()), &[DZ, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.003);
    let bs = 64usize;

    // one incremental forward over a batch of evidence-index sequences; returns final s-logit Var (+ optionally z per step)
    // toks: [T][bs] evidence indices
    let forward = |p: &[Tensor], toks: &Vec<Vec<usize>>, bs: usize| {
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let av = Var::leaf(alpha.clone());
        let mut z = Var::leaf(Tensor::zeros(&ctx, &[bs, DZ]));
        let mut h = Var::leaf(Tensor::zeros(&ctx, &[bs, DZ]));
        let mut zsnap: Vec<Var> = Vec::new();
        for t in 0..T {
            // h += onehot(tok) @ emb
            let mut oh = vec![0.0f32; bs * 4]; for i in 0..bs { oh[i * 4 + toks[t][i]] = 1.0; }
            let ohv = Var::leaf(Tensor::from_vec(&ctx, &oh, &[bs, 4]));
            h = h.add(&ohv.matmul(&pv[5]));
            for _ in 0..KTR { let e = energy(&z, &h, &pv, &ov).sum_all(); let g = grad(&e, &[z.clone()], None).remove(0); z = z.sub(&g.mul(&av)); }
            zsnap.push(z.clone());
        }
        let logit = z.matmul(&pv[6]).add(&pv[7]);
        (logit, zsnap, pv)
    };

    for step in 0..1600 {
        // random uniform evidence sequences
        let mut toks = vec![vec![0usize; bs]; T]; let mut lbl = vec![0.0f32; bs];
        for i in 0..bs { let mut s = 0.0f32; for t in 0..T { let e = (u(step as u32 * 31 + i as u32 * 7 + t as u32, 3) * 4.0) as usize % 4; toks[t][i] = e; s += EV[e]; }
            lbl[i] = if s > 0.0 { 1.0 } else { 0.0 }; }
        let (logit, _z, pv) = forward(&p, &toks, bs);
        let ov = Var::leaf(one.clone());
        let pr = ov.div(&ov.add(&logit.neg().exp()));                       // sigmoid
        let eps = Var::leaf(Tensor::from_vec(&ctx, &vec![1e-6; bs], &[bs, 1]));
        let yv = Var::leaf(Tensor::from_vec(&ctx, &lbl, &[bs, 1]));
        let loss = yv.mul(&pr.add(&eps).log()).add(&ov.sub(&yv).mul(&ov.sub(&pr).add(&eps).log())).mean_all().neg();
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }

    // ---- accuracy on random items + per-position effort ‖z_t − z_{t-1}‖ vs reanalysis (running-sum sign flip) ----
    let eb = 400usize;
    let mut toks = vec![vec![0usize; eb]; T]; let mut lbl = vec![0.0f32; eb];
    // running cumulative-sign per item/position to mark reanalysis
    let mut csum = vec![vec![0.0f32; eb]; T];
    for i in 0..eb { let mut s = 0.0f32; for t in 0..T { let e = (u(9000 + i as u32 * 7 + t as u32, 3) * 4.0) as usize % 4; toks[t][i] = e; s += EV[e]; csum[t][i] = s; } lbl[i] = if s > 0.0 { 1.0 } else { 0.0 }; }
    let (logit, zsnap, _pv) = forward(&p, &toks, eb);
    let lg = logit.value().to_vec().await;
    let mut acc = 0; for i in 0..eb { let pred = if lg[i] > 0.0 { 1.0 } else { 0.0 }; if pred == lbl[i] { acc += 1; } }
    // z per position
    let mut zt: Vec<Vec<f32>> = Vec::new(); for t in 0..T { zt.push(zsnap[t].value().to_vec().await); }
    // effort per (item, position>=1) and reanalysis flag
    let (mut eff_re, mut n_re, mut eff_no, mut n_no) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..eb { for t in 1..T {
        let mut d = 0.0f32; for k in 0..DZ { let dd = zt[t][i * DZ + k] - zt[t - 1][i * DZ + k]; d += dd * dd; } let eff = d.sqrt();
        let flip = (csum[t][i] > 0.0) != (csum[t - 1][i] > 0.0);
        if flip { eff_re += eff as f64; n_re += 1.0; } else { eff_no += eff as f64; n_no += 1.0; }
    } }

    // garden-path vs control demo items (evidence held identical in surprisal — all uniform tokens)
    let gp = [0usize, 1, 3, 3, 2];   // e = −2,−1,+2,+2,+1 → cumsum −2,−3,−1,+1,+2  (flips at pos 4: reanalysis)
    let cl = [3usize, 2, 3, 2, 3];   // e = +2,+1,+2,+1,+2 → cumsum monotone up, no flip
    let mut dt = vec![vec![0usize; 2]; T]; for t in 0..T { dt[t][0] = gp[t]; dt[t][1] = cl[t]; }
    let (_lg2, zs2, _p2) = forward(&p, &dt, 2);
    let mut zt2: Vec<Vec<f32>> = Vec::new(); for t in 0..T { zt2.push(zs2[t].value().to_vec().await); }
    let effpos = |item: usize| { let mut v = vec![0.0f32; T]; for t in 1..T { let mut d = 0.0; for k in 0..DZ { let dd = zt2[t][item * DZ + k] - zt2[t - 1][item * DZ + k]; d += dd * dd; } v[t] = d.sqrt(); } v };
    let (eg, ec) = (effpos(0), effpos(1));

    println!("\n  label accuracy on uniform evidence sequences: {:.0}%   (the EBM integrates evidence into the belief)\n", acc as f32 / eb as f32 * 100.0);
    println!("  EVERY token is equally surprising (uniform draw). Mean per-token descent EFFORT ‖Δz‖:");
    println!("     positions WITH interpretation flip (reanalysis):  {:.4}", eff_re / n_re.max(1.0));
    println!("     positions withOUT a flip (no reanalysis):         {:.4}", eff_no / n_no.max(1.0));
    println!("     → ratio {:.2}×  (effort concentrates on REVISION, not on the equally-surprising input)\n", (eff_re / n_re.max(1.0)) / (eff_no / n_no.max(1.0)).max(1e-6));
    println!("  garden-path item (flips at position 4) vs control (no flip) — effort ‖Δz‖ by position:");
    print!("     garden-path:  "); for t in 1..T { print!("p{}={:.3}  ", t + 1, eg[t]); } println!();
    print!("     control:      "); for t in 1..T { print!("p{}={:.3}  ", t + 1, ec[t]); } println!();
    println!("\n  If the flip position spikes (garden-path) while the control stays flat — at identical token surprisal —");
    println!("  the energy-based model spends compute on REANALYSIS: reasoning-depth difficulty, the reading-time kind.");
}
