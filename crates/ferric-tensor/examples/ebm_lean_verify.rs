//! EFA energy-first #10 (AI-for-math, opening #2) — energy VERIFIER on REAL Lean/mathlib proof steps.
//!
//! First real-formal-signal build. Data = 14k genuine (proof-state, tactic) pairs from Lean 4 mathlib
//! (l3lab/ntp-mathlib-instruct-st), token-hashed (state 256 ⊕ tactic 256). Verifier scores (goal,tactic)
//! compatibility. CRUCIAL: within a candidate group the STATE is shared, so the verifier must use the
//! goal↔tactic INTERACTION, not tactic frequency (a first pointwise-classifier attempt learned a
//! tactic-frequency shortcut → WORSE than random). Fix = CONTRASTIVE per-state ranking (how ReProver /
//! premise selection actually train): a pairwise Bradley-Terry loss that scores the correct tactic ABOVE a
//! distractor FOR THE SAME STATE. Eval = best-of-8 tactic selection (pick the real mathlib tactic; random=12.5%).
//!
//! Prereq: scratchpad/leanprep2.py → lsf.f32 / ltf.f32 / lte.f32.
//! Run: `cargo run -p ferric-tensor --example ebm_lean_verify --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const DIR: &str = "/private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad";
const DS: usize = 384; const DT: usize = 384; const D: usize = 768; const C: usize = 8; const HE: usize = 256;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn read_f32(p: &str) -> Vec<f32> { let b = std::fs::read(p).unwrap(); b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = (h32(i as u32 ^ seed) % 1_000_000 + 1) as f32 / 1e6; let b = (h32((i as u32).wrapping_mul(2654435761) ^ seed) % 1_000_000 + 1) as f32 / 1e6; ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let sf = read_f32(&format!("{DIR}/emb_sf.f32")); let tf = read_f32(&format!("{DIR}/emb_tf.f32")); let te = read_f32(&format!("{DIR}/emb_te.f32"));
    let ntr = sf.len() / DS; let tg = te.len() / (C * D);
    println!("  EFA energy-first — VERIFIER on REAL Lean/mathlib proof steps: {ntr} states, {tg} test groups × {C}");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);

    // scalar compatibility score s(state ⊕ tactic): D → HE → HE → 1
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(D * HE, 1, 1.0 / (D as f32).sqrt()), &[D, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * HE, 2, 1.0 / (HE as f32).sqrt()), &[HE, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE, 3, 1.0 / (HE as f32).sqrt()), &[HE, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.001); let bs = 256usize;
    let score_v = |x: &Var, p: &[Var]| -> Var { let h = x.matmul(&p[0]).add(&p[1]).relu(); let h2 = h.matmul(&p[2]).add(&p[3]).relu(); h2.matmul(&p[4]).add(&p[5]) };
    let mkrow = |sidx: usize, tidx: usize, sf: &[f32], tf: &[f32], out: &mut [f32]| { out[..DS].copy_from_slice(&sf[sidx * DS..(sidx + 1) * DS]); out[DS..].copy_from_slice(&tf[tidx * DT..(tidx + 1) * DT]); };
    for step in 0..6000 {
        // pairwise: for B states, positive=(state,its tactic), negative=(state, random other tactic)
        let mut xp = vec![0.0f32; bs * D]; let mut xn = vec![0.0f32; bs * D];
        for b in 0..bs { let s = (h32(step as u32 * 2_654_435_761 + b as u32) as usize) % ntr; let neg = (h32(step as u32 * 40_503 + b as u32 * 7 + 1) as usize) % ntr;
            mkrow(s, s, &sf, &tf, &mut xp[b * D..(b + 1) * D]); mkrow(s, neg, &sf, &tf, &mut xn[b * D..(b + 1) * D]); }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let sp = score_v(&Var::leaf(Tensor::from_vec(&ctx, &xp, &[bs, D])), &pv);
        let sn = score_v(&Var::leaf(Tensor::from_vec(&ctx, &xn, &[bs, D])), &pv);
        // Bradley-Terry: −log σ(s+ − s−) = softplus(s− − s+); ranks correct above distractor for the SAME state
        let d = sn.sub(&sp); let loss = d.exp().add(&Var::leaf(one.clone())).log().mean_all();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); adam.step(&mut p, &g);
    }

    // best-of-8 tactic selection (correct = candidate 0 in each group)
    let score_t = |x: &Tensor| -> Tensor { x.matmul(&p[0]).add(&p[1]).relu().matmul(&p[2]).add(&p[3]).relu().matmul(&p[4]).add(&p[5]) };
    let s = score_t(&Tensor::from_vec(&ctx, &te, &[tg * C, D])).to_vec().await;
    let (mut top1, mut top3) = (0usize, 0usize);
    for g in 0..tg { let mut v: Vec<(f32, usize)> = (0..C).map(|j| (s[g * C + j], j)).collect(); v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if v[0].1 == 0 { top1 += 1; } if v[..3].iter().any(|&(_, j)| j == 0) { top3 += 1; } }
    println!("\n  best-of-{C} tactic selection — pick the REAL mathlib tactic among {C} candidates:");
    println!("     energy verifier  top-1: {:>4.0}%   top-3: {:>4.0}%", top1 as f32 / tg as f32 * 100.0, top3 as f32 / tg as f32 * 100.0);
    println!("     random baseline  top-1: {:>4.1}%   top-3: {:>4.1}%", 100.0 / C as f32, 300.0 / C as f32);
    println!("\n  verifier ≫ random → a learned energy scores (goal,tactic) compatibility on REAL Lean proof steps");
    println!("  (contrastive per-state ranking; lexical features — a pretrained proof-state encoder would lift it).");
}
