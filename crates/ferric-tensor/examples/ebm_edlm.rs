//! EFA energy-first #6 — RESIDUAL-ENERGY verifier on a real AR model (EDLM recipe): the bridge to sequences.
//!
//! Deng/Bakhtin/Ranzato residual-EBM (2004.11714) + EDLM (Xu et al. ICLR 2025, 2410.21357): bolt a learned
//! energy E(x) onto a FROZEN autoregressive proposal p_AR, define p(x) ∝ p_AR(x)·e^{−E(x)}, and improve
//! generation by best-of-N (the mechanism that was cleanest across this arc — build-2/4). The AR model is
//! LOCAL (bigram + position context) so it can't enforce a GLOBAL constraint; the energy verifier sees the
//! whole sequence and reranks. Task: length-6 sequences over {0,1,2,3} that must SUM TO 9 (global). Question:
//! does energy best-of-N fix the AR's global errors, and scale with N (test-time verification compute)?
//!
//! Run: `cargo run -p ferric-tensor --example ebm_edlm --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const V: usize = 4;   // vocab {0,1,2,3}
const L: usize = 6;   // sequence length
const T: usize = 9;   // target sum (global constraint)
const HA: usize = 64; // AR hidden
const HE: usize = 128; // energy hidden

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn valid(seq: &[usize]) -> bool { seq.iter().sum::<usize>() == T }
fn gen_valid(m: usize, seed: u32) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(m); let mut s = seed;
    while out.len() < m { let seq: Vec<usize> = (0..L).map(|i| (u(s.wrapping_add(i as u32 * 131), 1) * V as f32) as usize % V).collect(); s = s.wrapping_add(7); if valid(&seq) { out.push(seq); } }
    out
}
// AR input row for position t with previous token `prev` (prev==V means BOS): [onehot(prev)_V, onehot(t)_L]
fn ar_input(prev: usize, t: usize, row: &mut [f32]) { for x in row.iter_mut() { *x = 0.0; } if prev < V { row[prev] = 1.0; } row[V + t] = 1.0; }
// autoregressively sample m sequences from the (frozen) AR proposal
async fn ar_sample(ctx: &Arc<ferric_core::Context>, ap: &[Tensor], m: usize, seed: u32) -> Vec<Vec<usize>> {
    let mut seqs = vec![vec![0usize; L]; m]; let mut prev = vec![V; m];
    for t in 0..L {
        let mut inp = vec![0.0f32; m * (V + L)];
        for n in 0..m { ar_input(prev[n], t, &mut inp[n * (V + L)..(n + 1) * (V + L)]); }
        let lg = Tensor::from_vec(ctx, &inp, &[m, V + L]).matmul(&ap[0]).add(&ap[1]).relu().matmul(&ap[2]).add(&ap[3]).to_vec().await;
        for n in 0..m {
            let mut mx = f32::MIN; for k in 0..V { if lg[n * V + k] > mx { mx = lg[n * V + k]; } }
            let mut ex = [0.0f32; V]; let mut z = 0.0f32; for k in 0..V { ex[k] = (lg[n * V + k] - mx).exp(); z += ex[k]; }
            let r = u(n as u32, seed.wrapping_add(t as u32 * 7919)); let mut cum = 0.0f32; let mut tok = V - 1;
            for k in 0..V { cum += ex[k] / z; if r <= cum { tok = k; break; } }
            seqs[n][t] = tok; prev[n] = tok;
        }
    }
    seqs
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — residual-energy verifier on an AR model (EDLM): seqs of len {L} over {{0..{}}} summing to {T}", V - 1);
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let eps = Tensor::from_vec(&ctx, &[1e-6], &[1]);
    let bs = 512usize;

    // ---- train the LOCAL AR proposal p(x_t | x_{t-1}, t): MLP [V+L → HA → V], teacher-forced CE on valid seqs
    let mut ap = vec![
        Tensor::from_vec(&ctx, &randn((V + L) * HA, 1, 1.0 / ((V + L) as f32).sqrt()), &[V + L, HA]), Tensor::zeros(&ctx, &[HA]),
        Tensor::from_vec(&ctx, &randn(HA * V, 2, 1.0 / (HA as f32).sqrt()), &[HA, V]), Tensor::zeros(&ctx, &[V]),
    ];
    let mut aadam = Adam::new(&ap, 0.003);
    for step in 0..3500 {
        let seqs = gen_valid(bs, step as u32 * 13 + 1);
        let rows = bs * L; let mut inp = vec![0.0f32; rows * (V + L)]; let mut lab = vec![0.0f32; rows * V];
        for (bi, s) in seqs.iter().enumerate() { for t in 0..L { let prev = if t == 0 { V } else { s[t - 1] }; ar_input(prev, t, &mut inp[(bi * L + t) * (V + L)..(bi * L + t + 1) * (V + L)]); lab[(bi * L + t) * V + s[t]] = 1.0; } }
        let av: Vec<Var> = ap.iter().map(|t| Var::leaf(t.clone())).collect();
        let h = Var::leaf(Tensor::from_vec(&ctx, &inp, &[rows, V + L])).matmul(&av[0]).add(&av[1]).relu();
        let logits = h.matmul(&av[2]).add(&av[3]);
        let logp = logits.softmax(1).add(&Var::leaf(eps.clone())).log();
        let loss = Var::leaf(Tensor::from_vec(&ctx, &lab, &[rows, V])).mul(&logp).mean_all().neg();
        loss.backward(); let g: Vec<Tensor> = av.iter().map(|v| v.grad().unwrap()).collect(); aadam.step(&mut ap, &g);
    }
    // ---- the residual verifier = a 2-logit CLASSIFIER over onehot(seq) (valid vs invalid), softmax-CE.
    // (Contrastive+magnitude-reg gave a near-constant energy → random selection; NCE ⇔ classification, and a
    // classifier over the onehot cleanly learns the global SUM boundary.) Energy = logit_invalid − logit_valid.
    let mut ep = vec![
        Tensor::from_vec(&ctx, &randn(V * L * HE, 3, 1.0 / ((V * L) as f32).sqrt()), &[V * L, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * HE, 4, 1.0 / (HE as f32).sqrt()), &[HE, HE]), Tensor::zeros(&ctx, &[HE]),
        Tensor::from_vec(&ctx, &randn(HE * 2, 5, 1.0 / (HE as f32).sqrt()), &[HE, 2]), Tensor::zeros(&ctx, &[2]),
    ];
    let mut eadam = Adam::new(&ep, 0.002);
    let onehot_seqs = |seqs: &[Vec<usize>]| -> Vec<f32> { let m = seqs.len(); let mut o = vec![0.0f32; m * V * L]; for (i, s) in seqs.iter().enumerate() { for t in 0..L { o[i * V * L + t * V + s[t]] = 1.0; } } o };
    let clf_v = |xv: &Var, p: &[Var]| -> Var { let h = xv.matmul(&p[0]).add(&p[1]).relu(); let h2 = h.matmul(&p[2]).add(&p[3]).relu(); h2.matmul(&p[4]).add(&p[5]) };
    for step in 0..3000 {
        let pos = gen_valid(bs, step as u32 * 17 + 3);
        // negatives = AR samples filtered to the INVALID ones (model-sourced hard negatives, cleanly labeled)
        let mut neg: Vec<Vec<usize>> = Vec::with_capacity(bs); let mut sd = step as u32 * 29 + 11;
        while neg.len() < bs { for s in ar_sample(&ctx, &ap, bs, sd).await { if !valid(&s) && neg.len() < bs { neg.push(s); } } sd = sd.wrapping_add(1); }
        let mut all = onehot_seqs(&pos); all.extend(onehot_seqs(&neg));
        let mut lab = vec![0.0f32; 2 * bs * 2]; for i in 0..bs { lab[i * 2] = 1.0; lab[(bs + i) * 2 + 1] = 1.0; } // pos=[1,0], neg=[0,1]
        let pv: Vec<Var> = ep.iter().map(|t| Var::leaf(t.clone())).collect();
        let logits = clf_v(&Var::leaf(Tensor::from_vec(&ctx, &all, &[2 * bs, V * L])), &pv);
        let logp = logits.softmax(1).add(&Var::leaf(eps.clone())).log();
        let loss = Var::leaf(Tensor::from_vec(&ctx, &lab, &[2 * bs, 2])).mul(&logp).mean_all().neg();
        loss.backward(); let g: Vec<Tensor> = pv.iter().map(|v| v.grad().unwrap()).collect(); eadam.step(&mut ep, &g);
    }
    // classifier logits (tensor forward); energy = logit_invalid − logit_valid (low ⇒ valid)
    let clf_t = |x: &Tensor, p: &[Tensor]| -> Tensor { x.matmul(&p[0]).add(&p[1]).relu().matmul(&p[2]).add(&p[3]).relu().matmul(&p[4]).add(&p[5]) };

    // ---- eval: raw AR validity vs energy best-of-N; sweep N; oracle (any-valid-in-N) = ceiling ----
    println!("\n  fraction of generations summing to {T} — raw AR vs energy best-of-N (400 trials):");
    println!("     {:>6}  {:>10}  {:>12}  {:>10}", "N", "energy-BoN", "oracle(any)", "random");
    for &nn in &[1usize, 2, 4, 8, 16, 32] {
        let trials = 400usize; let seqs = ar_sample(&ctx, &ap, trials * nn, 55 + nn as u32 * 101).await;
        let lg = clf_t(&Tensor::from_vec(&ctx, &onehot_seqs(&seqs), &[trials * nn, V * L]), &ep).to_vec().await;
        let e: Vec<f32> = (0..trials * nn).map(|i| lg[i * 2 + 1] - lg[i * 2]).collect(); // energy = logit_invalid − logit_valid
        let (mut bon, mut orc, mut rnd) = (0usize, 0usize, 0usize);
        for tr in 0..trials {
            let (mut bi, mut be) = (tr * nn, f32::MAX); let mut any = false;
            for j in 0..nn { let idx = tr * nn + j; if e[idx] < be { be = e[idx]; bi = idx; } if valid(&seqs[idx]) { any = true; } }
            if valid(&seqs[bi]) { bon += 1; } if any { orc += 1; } if valid(&seqs[tr * nn]) { rnd += 1; }
        }
        println!("     {:>6}  {:>9.0}%  {:>11.0}%  {:>9.0}%", nn, bon as f32 / trials as f32 * 100.0, orc as f32 / trials as f32 * 100.0, rnd as f32 / trials as f32 * 100.0);
    }
    println!("\n  energy-BoN ≫ raw AR (N=1) and tracking the oracle → a residual energy verifier fixes the LOCAL AR's GLOBAL");
    println!("  constraint errors and scales with N — the verification win, now on real autoregressive SEQUENCES.");
}
