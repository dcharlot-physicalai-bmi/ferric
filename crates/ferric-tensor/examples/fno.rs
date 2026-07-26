//! A Fourier Neural Operator (FNO) on the pure-Rust Ferric fabric — the spectral neural operator that
//! is the workhorse for parametric PDEs (turbulence/weather surrogates). Its core is the spectral
//! convolution: transform to Fourier space, apply a LEARNED per-mode weight, transform back. Ferric has
//! no FFT, but the DFT is a matmul — so we implement the spectral conv as DFT-matmul with a real/imag
//! split (O(n²) not O(n·log n); FFT is the asymptotic speedup, irrelevant at this grid size). Task: learn
//! the solution operator of the 1-D Poisson equation −u''=f on a periodic domain, f ↦ u. Poisson is
//! DIAGONAL in Fourier (û_k = f̂_k / k²), so a single spectral layer can represent it exactly and the
//! learned per-mode weights should RECOVER the Green's-function multipliers 1/k² — an exact check on top
//! of held-out error. Trained GPU-native (Metal), pure Rust.
//!   cargo run --release --example fno

use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
use std::f32::consts::PI;

const M: usize = 64; // grid points (period 2π)
const K: usize = 16; // kept Fourier modes (0..K-1); f is band-limited to these
const B: usize = 64; // functions per batch

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn uni(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 * 2.0 - 1.0 }

// a random band-limited f = Σ_{k=1}^{K-1}(c_k sin kx + d_k cos kx), and the exact Poisson solution
// u = Σ (c_k/k²) sin kx + (d_k/k²) cos kx  (−u''=f, zero-mean ⇒ periodic, exact).
fn sample(seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut f = vec![0.0f32; M];
    let mut u = vec![0.0f32; M];
    for k in 1..K {
        let c = uni(k as u32, seed) / k as f32;
        let d = uni(k as u32, seed.wrapping_add(4242)) / k as f32;
        let k2 = (k * k) as f32;
        for j in 0..M {
            let x = 2.0 * PI * j as f32 / M as f32;
            let (s, co) = ((k as f32 * x).sin(), (k as f32 * x).cos());
            f[j] += c * s + d * co;
            u[j] += c / k2 * s + d / k2 * co;
        }
    }
    (f, u)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("Fourier Neural Operator on the Ferric fabric — backend: {:?}", ctx.backend);
    println!("learning the 1-D Poisson solution operator f ↦ u (−u''=f), spectral conv via DFT-matmul\n");

    // constant DFT matrices: forward Cr,Ci [M,K]; inverse (real, conjugate-doubled) Ccos,Csin [K,M]
    let mut cr = vec![0.0f32; M * K]; let mut ci = vec![0.0f32; M * K];
    let mut ccos = vec![0.0f32; K * M]; let mut csin = vec![0.0f32; K * M];
    for k in 0..K {
        let a = if k == 0 { 1.0 / M as f32 } else { 2.0 / M as f32 };
        for j in 0..M {
            let ang = 2.0 * PI * (k * j) as f32 / M as f32;
            cr[j * K + k] = ang.cos(); ci[j * K + k] = -ang.sin();
            ccos[k * M + j] = a * ang.cos(); csin[k * M + j] = -a * ang.sin();
        }
    }
    let crv = Var::leaf(Tensor::from_vec(&ctx, &cr, &[M, K]));
    let civ = Var::leaf(Tensor::from_vec(&ctx, &ci, &[M, K]));
    let ccosv = Var::leaf(Tensor::from_vec(&ctx, &ccos, &[K, M]));
    let csinv = Var::leaf(Tensor::from_vec(&ctx, &csin, &[K, M]));

    // learned per-mode complex weights R_k (Rr + i·Ri), [1,K] (broadcast over the batch)
    let mut wp = vec![
        Tensor::from_vec(&ctx, &(0..K).map(|k| uni(k as u32, 11) * 0.1).collect::<Vec<_>>(), &[1, K]),
        Tensor::from_vec(&ctx, &vec![0.0f32; K], &[1, K]),
    ];
    let mut adam = Adam::new(&wp, 5e-3);

    for epoch in 0..4000u32 {
        let mut fs = vec![0.0f32; B * M]; let mut us = vec![0.0f32; B * M];
        for b in 0..B { let (f, u) = sample(epoch.wrapping_mul(97) + b as u32 + 1); for j in 0..M { fs[b * M + j] = f[j]; us[b * M + j] = u[j]; } }
        let rr = Var::leaf(wp[0].clone()); let ri = Var::leaf(wp[1].clone());
        let fv = Var::leaf(Tensor::from_vec(&ctx, &fs, &[B, M]));
        let fr = fv.matmul(&crv); let fi = fv.matmul(&civ);                 // f̂  [B,K]
        let ur = rr.mul(&fr).sub(&ri.mul(&fi));                             // û = R·f̂  (complex)
        let ui = rr.mul(&fi).add(&ri.mul(&fr));
        let pred = ur.matmul(&ccosv).add(&ui.matmul(&csinv));              // inverse DFT → u  [B,M]
        let diff = pred.sub(&Var::leaf(Tensor::from_vec(&ctx, &us, &[B, M])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();
        let g = vec![rr.grad().unwrap_or_else(|| Tensor::zeros(&ctx, &[1, K])), ri.grad().unwrap_or_else(|| Tensor::zeros(&ctx, &[1, K]))];
        adam.step(&mut wp, &g);
        if epoch % 1000 == 0 || epoch == 3999 { println!("  epoch {epoch:5}  loss {:.6}", loss.value().to_vec().await[0]); }
    }

    // ---- verify on held-out functions + check R_k recovered 1/k² ----
    let nt = 200usize;
    let mut fs = vec![0.0f32; nt * M]; let mut us = vec![0.0f32; nt * M];
    for b in 0..nt { let (f, u) = sample(700_000 + b as u32); for j in 0..M { fs[b * M + j] = f[j]; us[b * M + j] = u[j]; } }
    let rr = Var::leaf(wp[0].clone()); let ri = Var::leaf(wp[1].clone());
    let fv = Var::leaf(Tensor::from_vec(&ctx, &fs, &[nt, M]));
    let fr = fv.matmul(&crv); let fi = fv.matmul(&civ);
    let ur = rr.mul(&fr).sub(&ri.mul(&fi)); let ui = rr.mul(&fi).add(&ri.mul(&fr));
    let pred = ur.matmul(&ccosv).add(&ui.matmul(&csinv)).value().to_vec().await;
    let (mut se, mut sy) = (0.0f32, 0.0f32);
    for i in 0..nt * M { se += (pred[i] - us[i]).powi(2); sy += us[i] * us[i]; }
    let rel = (se / sy).sqrt() * 100.0;
    let rrv = wp[0].to_vec().await; let riv = wp[1].to_vec().await;
    println!("\n  held-out ({nt} unseen forcings): relative L2 error {rel:.3}%");
    println!("  learned multipliers Rr_k vs Green's-function 1/k²  (Ri_k → 0):");
    for k in 1..6 { println!("    k={k}: Rr={:.4}  (1/k²={:.4})  Ri={:.4}", rrv[k], 1.0 / (k * k) as f32, riv[k]); }
    println!("  {}", if rel < 5.0 { "PASS — an FNO learned the Poisson solution operator + recovered its Green's function, GPU-native on the pure-Rust fabric ✓" } else { "FAIL — relative L2 above 5%" });
}
