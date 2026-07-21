//! EFA — gradient-free MEMORY capacity: additive vs delta-rule WRITE, linear vs Hopfield READ.
//!
//! EFA's Z is a fixed d×d fast-weight memory written at inference by outer products. What we have is the
//! ADDITIVE (pure-Hebbian) write `Z += v·kᵀ` with a LINEAR read `v̂ = Z·k`. The training-free-memory survey
//! says two gradient-free upgrades each push the interference wall out:
//!   (B) DELTA-RULE write `Z += β(v − Z·k)·kᵀ` — write the reconstruction ERROR, not the raw pattern
//!       (= one closed-form gradient step; DeltaNet arXiv:2412.06464). Overwrites instead of superimposing.
//!   (C) modern-HOPFIELD read `v̂ = V·softmax(β·Kᵀk)` — exponential capacity + one-shot error correction
//!       (Hopfield-is-all-you-need arXiv:2008.02217). Keeps every pattern (O(M) store), not compressed.
//! This is an algorithmic study (no GPU needed): store M hetero-associative sparse-positive pairs (EFA's
//! latent statistics: ReLU, top-k, unit-norm) and measure mean recall cosine as M grows past d. It isolates
//! the write/read rule before wiring the winner into the fabric model.
//!
//! Run: `cargo run -p ferric-tensor --example efa_memory --release`

const D: usize = 64;
const TOPK: usize = 6; // ~9% sparse, like EFA's latent

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(i: usize, seed: u32) -> f32 { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); (-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos() }

// sparse-positive unit-normalized pattern (EFA latent statistics)
fn pattern(seed: u32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..D).map(|i| randn(i, seed).max(0.0)).collect();
    // keep top-K, zero the rest
    let mut idx: Vec<usize> = (0..D).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    for &j in idx.iter().skip(TOPK) { v[j] = 0.0; }
    let nrm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-8);
    for x in &mut v { *x /= nrm; }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    d / (na.sqrt() * nb.sqrt() + 1e-9)
}

fn main() {
    println!("  EFA memory — recall cosine vs #patterns stored (d={D}, top-{TOPK} sparse-positive, hetero-associative)");
    println!("  ADDITIVE = pure-Hebbian Z+=vkᵀ (what EFA has) · DELTA = Z+=β(v−Zk)kᵀ · HOPFIELD = softmax readout\n");
    println!("     {:>6}   {:>9}   {:>9}   {:>9}   {:>9}", "M/d", "additive", "delta-1p", "delta-8p", "hopfield");
    for &m in &[16usize, 32, 48, 64, 96, 128, 192, 256] {
        // generate M key→value sparse-positive pairs (keys and values independent)
        let keys: Vec<Vec<f32>> = (0..m).map(|i| pattern(1000 + i as u32)).collect();
        let vals: Vec<Vec<f32>> = (0..m).map(|i| pattern(7000 + i as u32)).collect();

        // ---- ADDITIVE write: Z = Σ v_i k_iᵀ ----
        let mut z_add = vec![0.0f32; D * D];
        for i in 0..m { for a in 0..D { let va = vals[i][a]; if va == 0.0 { continue; } for b in 0..D { z_add[a * D + b] += va * keys[i][b]; } } }

        // ---- DELTA write: Z += β(v − Zk)kᵀ, β=1 (unit-norm keys). 1 online pass, and 8 consolidation
        // passes (re-present the stored pairs — still gradient-free; = iterative least-squares fit ZK≈V). ----
        let delta_write = |passes: usize| -> Vec<f32> {
            let mut z = vec![0.0f32; D * D];
            let beta = 1.0f32;
            for _ in 0..passes { for i in 0..m {
                let mut pred = [0.0f32; D];
                for a in 0..D { let mut s = 0.0f32; for b in 0..D { s += z[a * D + b] * keys[i][b]; } pred[a] = s; }
                for a in 0..D { let err = beta * (vals[i][a] - pred[a]); if err == 0.0 { continue; } for b in 0..D { z[a * D + b] += err * keys[i][b]; } }
            } }
            z
        };
        let z_del1 = delta_write(1);
        let z_del8 = delta_write(8);

        // ---- read each stored key back, measure mean recall cosine ----
        let read_lin = |z: &[f32], k: &[f32]| -> Vec<f32> { (0..D).map(|a| { let mut s = 0.0f32; for b in 0..D { s += z[a * D + b] * k[b]; } s }).collect() };
        let (mut ca, mut cd1, mut cd8, mut ch) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let hop_beta = 12.0f32; // inverse temperature: high β → sharp, separated retrieval
        for i in 0..m {
            ca += cosine(&read_lin(&z_add, &keys[i]), &vals[i]);
            cd1 += cosine(&read_lin(&z_del1, &keys[i]), &vals[i]);
            cd8 += cosine(&read_lin(&z_del8, &keys[i]), &vals[i]);
            // HOPFIELD read: v̂ = Σ_j softmax(β·k_iᵀk_j)_j · v_j  (auto over stored keys, hetero values)
            let mut logits: Vec<f32> = (0..m).map(|j| hop_beta * keys[i].iter().zip(&keys[j]).map(|(x, y)| x * y).sum::<f32>()).collect();
            let lmax = logits.iter().cloned().fold(f32::MIN, f32::max);
            let mut zsum = 0.0f32; for l in &mut logits { *l = (*l - lmax).exp(); zsum += *l; }
            let mut vh = vec![0.0f32; D];
            for j in 0..m { let w = logits[j] / zsum; for a in 0..D { vh[a] += w * vals[j][a]; } }
            ch += cosine(&vh, &vals[i]);
        }
        let mf = m as f32;
        println!("     {:>4} {:.1}   {:>9.3}   {:>9.3}   {:>9.3}   {:>9.3}", m, m as f32 / D as f32, ca / mf, cd1 / mf, cd8 / mf, ch / mf);
    }
    println!("\n  Expect: additive collapses as M→d (interference); delta holds far longer (error-corrected writes);");
    println!("  hopfield stays near-perfect well past d (exponential capacity) — the two gradient-free upgrades for EFA's Z.");
}
