//! EFA energy-first #26 — GARDEN-PATH v3: a BISTABLE-PRIOR model, the design that lets the claim be tested.
//!
//! v1 was circular (smooth integrator, no commitment); v2 didn't train (per-position bilevel objective collapsed).
//! v3 fixes both at the root:
//!   • ARCHITECTURAL commitment — the decision coordinate d lives in an explicit DOUBLE WELL A·(d²−1)² (attractors
//!     at d=±1, barrier at d=0), tilted by a LEARNED evidence coupling g(h)·d. Commitment + hysteresis are built in.
//!   • SIMPLE training — final-label supervision only (which trained fine in v1). The well's gradient is analytic,
//!     so the descent update is written explicitly → first-order backprop, no bilevel instability.
//!   • THINKING-RESOLVABLE — pure gradient descent cannot cross a barrier, so a garden-pathed belief (trapped in the
//!     WRONG well) escapes only by THERMAL crossing (Langevin noise). More descent steps K = more escape chances.
//! Non-circular metric: GP-vs-CTRL final-decision ACCURACY vs thinking budget K, on minimal pairs matched in final
//! label AND total evidence. Falsifiable: GP<CTRL at low K, gap CLOSING as K grows ⇒ real reanalysis compute cost.
//! If GP tracks CTRL at all K ⇒ honest null (no hysteresis effect at this scale).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_garden3 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const DH: usize = 8; const T: usize = 6; const AW: f32 = 1.0; // well depth (barrier height = AW)
const EV: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
fn vidx(v: f32) -> usize { match v as i32 { -2 => 0, -1 => 1, 1 => 2, _ => 3 } }

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn gp_item(seed: u32) -> (Vec<usize>, usize) { // commit −dir early, reverse to +dir late; final sum +3dir
    let dir: f32 = if u(seed, 1) > 0.5 { 1.0 } else { -1.0 };
    let mut e = vec![vidx(-dir); 3]; e.extend(vec![vidx(2.0 * dir); 3]);
    (e, if dir > 0.0 { 1 } else { 0 })
}
fn cl_item(seed: u32) -> (Vec<usize>, usize) { // consistent dir, one late dip that never reverses the sign; final sum +3dir
    let dir: f32 = if u(seed, 1) > 0.5 { 1.0 } else { -1.0 };
    let mut e = vec![vidx(dir); 5]; e.push(vidx(-2.0 * dir));
    (e, if dir > 0.0 { 1 } else { 0 })
}

// incremental descent in the tilted double well. tp = [emb[4,DH], wg[DH,1], wr[1,1], br[1]] (caller owns the leaves).
fn forward(ctx: &Arc<ferric_core::Context>, tp: &[Var], toks: &Vec<Vec<usize>>, bs: usize, k: usize, sig: f32, nseed: u32) -> Var {
    let (emb, wg, wr, br) = (&tp[0], &tp[1], &tp[2], &tp[3]);
    let four = Var::leaf(Tensor::from_vec(ctx, &[4.0 * AW], &[1]));
    let one = Var::leaf(Tensor::from_vec(ctx, &[1.0], &[1]));
    let al = Var::leaf(Tensor::from_vec(ctx, &[0.1], &[1]));
    let mut d = Var::leaf(Tensor::zeros(ctx, &[bs, 1]));
    let mut h = Var::leaf(Tensor::zeros(ctx, &[bs, DH]));
    let mut nc = nseed;
    for t in 0..T {
        let mut oh = vec![0.0f32; bs * 4]; for i in 0..bs { oh[i * 4 + toks[t][i]] = 1.0; }
        h = h.add(&Var::leaf(Tensor::from_vec(ctx, &oh, &[bs, 4])).matmul(emb));
        let tilt = h.matmul(wg); // g(h) = learned evidence coupling → [bs,1]
        for _ in 0..k {
            // ∂E/∂d = 4A·d(d²−1) − tilt  (analytic double-well force); d ← d − α·force + Langevin ξ
            let dsq = d.mul(&d).sub(&one);
            let force = d.mul(&dsq).mul(&four).sub(&tilt);
            let noise = Var::leaf(Tensor::from_vec(ctx, &randn(bs, { nc = nc.wrapping_add(1); nc }, sig), &[bs, 1]));
            d = d.sub(&force.mul(&al)).add(&noise);
        }
    }
    d.mul(wr).add(br) // decision logit = wr·d + br
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — GARDEN-PATH v3 (bistable prior): is reanalysis a real, thinking-resolvable cost?");
    // trainable: emb, wg, wr, br  (the double-well constants A/α are fixed inside forward)
    let mut tp = vec![
        Tensor::from_vec(&ctx, &randn(4 * DH, 13, 0.5), &[4, DH]),
        Tensor::from_vec(&ctx, &randn(DH, 14, 0.4), &[DH, 1]),
        Tensor::from_vec(&ctx, &[3.0], &[1, 1]),
        Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&tp, 0.004);
    // FIX vs v2/v3-first-run: train the descent NOISE-FREE so the belief settles cleanly by evidence
    // (heavy training noise swamped the signal → 58% base). Langevin noise is added ONLY at test, for barrier escape.
    let bs = 64usize; let sig_tr = 0.0f32;

    for step in 0..2000 {
        let mut toks = vec![vec![0usize; bs]; T]; let mut lbl = vec![0.0f32; bs];
        for i in 0..bs { let mut s = 0.0f32; for t in 0..T { let e = (u(step as u32 * 31 + i as u32 * 7 + t as u32, 3) * 4.0) as usize % 4; toks[t][i] = e; s += EV[e]; } lbl[i] = if s > 0.0 { 1.0 } else { 0.0 }; }
        let ktr = 3 + (h32(step as u32 ^ 0x51ec) % 5) as usize; // budget 3..8
        let tpv: Vec<Var> = tp.iter().map(|t| Var::leaf(t.clone())).collect();
        let logit = forward(&ctx, &tpv, &toks, bs, ktr, sig_tr, step as u32 * 101 + 7);
        let one = Var::leaf(Tensor::from_vec(&ctx, &[1.0], &[1]));
        let pr = one.div(&one.add(&logit.neg().exp()));
        let eps = Var::leaf(Tensor::from_vec(&ctx, &vec![1e-6; bs], &[bs, 1]));
        let yv = Var::leaf(Tensor::from_vec(&ctx, &lbl, &[bs, 1]));
        let loss = yv.mul(&pr.add(&eps).log()).add(&one.sub(&yv).mul(&one.sub(&pr).add(&eps).log())).mean_all().neg();
        loss.backward();
        let g: Vec<Tensor> = tpv.iter().zip(&tp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut tp, &g);
    }

    // sanity: accuracy on random sequences
    let rn = 400usize; let mut rtok = vec![vec![0usize; rn]; T]; let mut rlbl = vec![0usize; rn];
    for i in 0..rn { let mut s = 0.0f32; for t in 0..T { let e = (u(77000 + i as u32 * 7 + t as u32, 3) * 4.0) as usize % 4; rtok[t][i] = e; s += EV[e]; } rlbl[i] = if s > 0.0 { 1 } else { 0 }; }
    let rtpv: Vec<Var> = tp.iter().map(|t| Var::leaf(t.clone())).collect();
    let rl = forward(&ctx, &rtpv, &rtok, rn, 12, 0.12, 31).value().to_vec().await;
    let mut racc = 0; for i in 0..rn { if (if rl[i] > 0.0 { 1 } else { 0 }) == rlbl[i] { racc += 1; } }

    // ---- test: GP vs CTRL accuracy vs thinking budget K (Langevin noise makes barrier-crossing possible) ----
    let n = 300usize; let sig_te = 0.12f32;
    let mut gtok = vec![vec![0usize; n]; T]; let mut glbl = vec![0usize; n];
    let mut ctok = vec![vec![0usize; n]; T]; let mut clbl = vec![0usize; n];
    for i in 0..n { let (ge, gl) = gp_item(1000 + i as u32); let (ce, cl) = cl_item(5000 + i as u32); for t in 0..T { gtok[t][i] = ge[t]; ctok[t][i] = ce[t]; } glbl[i] = gl; clbl[i] = cl; }

    println!("\n  sanity: accuracy on random sequences = {:.0}%  (the bistable model must learn the base task first)", racc as f32 / rn as f32 * 100.0);
    println!("\n  matched minimal pairs (same final label & total evidence ±3): GP reverses a commitment, CTRL never does.");
    println!("  final-decision accuracy (%) vs per-token thinking budget K (Langevin σ={}):\n", sig_te);
    print!("      K            "); for k in [1usize, 2, 4, 8, 16, 32] { print!("K={:<4}", k); } println!();
    for (name, tk, lb) in [("garden-path", &gtok, &glbl), ("control    ", &ctok, &clbl)] {
        print!("      {}  ", name);
        for k in [1usize, 2, 4, 8, 16, 32] {
            let tpv: Vec<Var> = tp.iter().map(|t| Var::leaf(t.clone())).collect();
            let lv = forward(&ctx, &tpv, tk, n, k, sig_te, 424242 + k as u32 * 13).value().to_vec().await;
            let mut ok = 0; for i in 0..n { let pred = if lv[i] > 0.0 { 1 } else { 0 }; if pred == lb[i] { ok += 1; } }
            print!("{:>4.0} ", ok as f32 / n as f32 * 100.0);
        }
        println!();
    }
    println!("\n  GP<CTRL at low K with the gap CLOSING as K grows ⇒ reanalysis is a real reasoning-depth compute cost");
    println!("  (a trapped belief needs thermal escape over the barrier). Rows tracking at every K ⇒ honest null.");
}
