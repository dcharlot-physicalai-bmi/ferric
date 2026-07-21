//! EFA energy-first #25 — GARDEN-PATH, done RIGHT: a COMMITTING model + a non-circular accuracy/compute metric.
//!
//! `ebm_garden.rs` was inconclusive by design (effort=displacement was ~circular with the flip; the learned model
//! was a smooth integrator with no commitment → no real barrier). This fixes both flaws:
//!   (1) FORCE COMMITMENT — running supervision toward the incremental best-guess sign(Σevidence-so-far), so the
//!       model commits to the locally-preferred reading and must REVERSE it when misleading early evidence flips
//!       (incremental parsing: "The horse raced past the barn *fell*").
//!   (2) NON-CIRCULAR METRIC — final-decision ACCURACY on garden-path vs control minimal pairs, matched on final
//!       label AND total evidence, as a function of the thinking budget K. Accuracy is not circular with belief
//!       displacement; surprisal is matched; and the K-sweep directly tests the reasoning-depth interpretation.
//! Falsifiable both ways: if GP acc < CTRL at low K and the gap CLOSES as K grows → real reanalysis compute cost.
//! If GP == CTRL at all K → honest null (no garden-path effect at this scale).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_garden2 --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const DZ: usize = 8; const HH: usize = 64; const T: usize = 6;
const EV: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
fn vidx(v: f32) -> usize { match v as i32 { -2 => 0, -1 => 1, 1 => 2, _ => 3 } }

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn energy(z: &Var, h: &Var, p: &[Var], one: &Var) -> Var {
    let a = z.matmul(&p[0]).add(&h.matmul(&p[1])).add(&p[2]);
    a.exp().add(one).log().matmul(&p[3]).add(&p[4])
}

// garden-path item: commit to −dir early, reverse to +dir late; final label = (dir>0). matched final sum = 3·dir.
fn gp_item(seed: u32) -> (Vec<usize>, usize) {
    let dir: f32 = if u(seed, 1) > 0.5 { 1.0 } else { -1.0 };
    let mut e = vec![vidx(-dir); 3]; e.extend(vec![vidx(2.0 * dir); 3]); // [−dir,−dir,−dir, +2dir,+2dir,+2dir] → sum +3dir, reverses
    (e, if dir > 0.0 { 1 } else { 0 })
}
// control item: consistent dir, one late dip that never reverses the sign; matched final sum = 3·dir, no reanalysis.
fn cl_item(seed: u32) -> (Vec<usize>, usize) {
    let dir: f32 = if u(seed, 1) > 0.5 { 1.0 } else { -1.0 };
    let mut e = vec![vidx(dir); 5]; e.push(vidx(-2.0 * dir)); // [+dir×5, −2dir] → cumsum stays sign(dir) throughout, final +3dir
    (e, if dir > 0.0 { 1 } else { 0 })
}

// incremental forward with a FIXED per-position budget k; returns final decision logits [bs,1]
fn forward(ctx: &Arc<ferric_core::Context>, p: &[Tensor], one: &Tensor, alpha: &Tensor, toks: &Vec<Vec<usize>>, bs: usize, k: usize, sup: bool) -> (Var, Vec<Var>) {
    let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let av = Var::leaf(alpha.clone());
    let mut z = Var::leaf(Tensor::zeros(ctx, &[bs, DZ]));
    let mut h = Var::leaf(Tensor::zeros(ctx, &[bs, DZ]));
    let mut perpos: Vec<Var> = Vec::new();
    for t in 0..T {
        let mut oh = vec![0.0f32; bs * 4]; for i in 0..bs { oh[i * 4 + toks[t][i]] = 1.0; }
        let ohv = Var::leaf(Tensor::from_vec(ctx, &oh, &[bs, 4]));
        h = h.add(&ohv.matmul(&pv[5]));
        for _ in 0..k { let e = energy(&z, &h, &pv, &ov).sum_all(); let g = grad(&e, &[z.clone()], None).remove(0); z = z.sub(&g.mul(&av)); }
        if sup { perpos.push(z.matmul(&pv[6]).add(&pv[7])); } // per-position decision logit for running supervision
    }
    let logit = z.matmul(&pv[6]).add(&pv[7]);
    (logit, if sup { perpos } else { pv })
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — GARDEN-PATH (committing model): is reanalysis a real, thinking-resolvable cost?");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let alpha = Tensor::from_vec(&ctx, &[0.25], &[1]);
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(DZ * HH, 10, 1.0 / (DZ as f32).sqrt()), &[DZ, HH]),
        Tensor::from_vec(&ctx, &randn(DZ * HH, 11, 1.0 / (DZ as f32).sqrt()), &[DZ, HH]), Tensor::zeros(&ctx, &[HH]),
        Tensor::from_vec(&ctx, &randn(HH, 12, 1.0 / (HH as f32).sqrt()), &[HH, 1]), Tensor::zeros(&ctx, &[1]),
        Tensor::from_vec(&ctx, &randn(4 * DZ, 13, 0.5), &[4, DZ]),
        Tensor::from_vec(&ctx, &randn(DZ, 14, 1.0 / (DZ as f32).sqrt()), &[DZ, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.003);
    let bs = 64usize;

    for step in 0..2200 {
        // random sequences; RUNNING supervision: target at each position = sign(cumsum so far) → forces commitment + reversal
        let mut toks = vec![vec![0usize; bs]; T]; let mut tgt = vec![vec![0.0f32; bs]; T];
        for i in 0..bs { let mut s = 0.0f32; for t in 0..T { let e = (u(step as u32 * 31 + i as u32 * 7 + t as u32, 3) * 4.0) as usize % 4; toks[t][i] = e; s += EV[e]; tgt[t][i] = if s > 0.0 { 1.0 } else { 0.0 }; } }
        let ktr = 2 + (h32(step as u32 ^ 0x51ec) % 6) as usize; // random budget 2..8
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone()); let av = Var::leaf(alpha.clone());
        let mut z = Var::leaf(Tensor::zeros(&ctx, &[bs, DZ])); let mut h = Var::leaf(Tensor::zeros(&ctx, &[bs, DZ]));
        let mut loss = Var::leaf(Tensor::zeros(&ctx, &[1]));
        for t in 0..T {
            let mut oh = vec![0.0f32; bs * 4]; for i in 0..bs { oh[i * 4 + toks[t][i]] = 1.0; }
            h = h.add(&Var::leaf(Tensor::from_vec(&ctx, &oh, &[bs, 4])).matmul(&pv[5]));
            for _ in 0..ktr { let e = energy(&z, &h, &pv, &ov).sum_all(); let g = grad(&e, &[z.clone()], None).remove(0); z = z.sub(&g.mul(&av)); }
            let lg = z.matmul(&pv[6]).add(&pv[7]);
            let pr = ov.div(&ov.add(&lg.neg().exp()));
            let eps = Var::leaf(Tensor::from_vec(&ctx, &vec![1e-6; bs], &[bs, 1]));
            let yv = Var::leaf(Tensor::from_vec(&ctx, &tgt[t], &[bs, 1]));
            let bce = yv.mul(&pr.add(&eps).log()).add(&ov.sub(&yv).mul(&ov.sub(&pr).add(&eps).log())).mean_all().neg();
            loss = loss.add(&bce);
        }
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }

    // ---- test: GP vs CTRL final-decision accuracy vs thinking budget K ----
    let n = 200usize;
    let acc = |toks: &Vec<Vec<usize>>, lbl: &Vec<usize>, k: usize, ctx: &Arc<ferric_core::Context>, p: &[Tensor]| {
        let (lg, _) = forward(ctx, p, &one, &alpha, toks, toks[0].len(), k, false);
        (lg, lbl.clone())
    };
    // build GP and CTRL batches
    let mut gtok = vec![vec![0usize; n]; T]; let mut glbl = vec![0usize; n];
    let mut ctok = vec![vec![0usize; n]; T]; let mut clbl = vec![0usize; n];
    for i in 0..n { let (ge, gl) = gp_item(1000 + i as u32); let (ce, cl) = cl_item(5000 + i as u32);
        for t in 0..T { gtok[t][i] = ge[t]; ctok[t][i] = ce[t]; } glbl[i] = gl; clbl[i] = cl; }

    println!("\n  matched minimal pairs (same final label & total evidence ±3): GP reverses a commitment, CTRL never does.");
    println!("  final-decision accuracy (%) vs per-token thinking budget K:\n");
    print!("      K            "); for k in [1usize, 2, 4, 8, 16, 32] { print!("K={:<4}", k); } println!();
    for (name, tk, lb) in [("garden-path", &gtok, &glbl), ("control    ", &ctok, &clbl)] {
        print!("      {}  ", name);
        for k in [1usize, 2, 4, 8, 16, 32] {
            let (lg, lbl) = acc(tk, lb, k, &ctx, &p); let lv = lg.value().to_vec().await;
            let mut ok = 0; for i in 0..n { let pred = if lv[i] > 0.0 { 1 } else { 0 }; if pred == lbl[i] { ok += 1; } }
            print!("{:>4.0} ", ok as f32 / n as f32 * 100.0);
        }
        println!();
    }
    println!("\n  If garden-path accuracy is LOWER at small K and the gap CLOSES as K grows, reanalysis is a real");
    println!("  reasoning-depth COMPUTE cost (surprisal is matched; accuracy isn't circular with displacement).");
    println!("  If the two rows track each other at every K, there is no garden-path effect here — an honest null.");
}
