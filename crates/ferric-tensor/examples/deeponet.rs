//! A DeepONet — a neural *operator* — trained GPU-native on the pure-Rust Ferric fabric. Where the PINN
//! (pinn_siren) solves ONE instance of an equation, an operator learns the whole SOLUTION MAP: give it a
//! new input function and it returns the solution in a single forward pass, no re-solving. This is the
//! primitive that owns the real-time / parametric / many-query regime (control surrogates, digital twins).
//! Canonical demo (Lu et al.): learn the antiderivative operator G[f](x) = ∫₀ˣ f(τ)dτ — i.e. the solution
//! operator of u'(x)=f(x), u(0)=0 — with EXACT ground truth (cumulative trapezoid). Branch net encodes f
//! sampled at m sensors; trunk net encodes the query x; the operator is their inner product. Trained on
//! random smooth f's, verified on HELD-OUT f's. Pure Rust, resident on the wgpu device (Metal here).
//!   cargo run --release --example deeponet

use ferric_tensor::{Adam, Mlp, Tensor, Var};
use std::sync::Arc;

const M: usize = 40; // sensors = query grid on [0,1]
const P: usize = 32; // latent (branch·trunk) width
const H: usize = 48; // hidden width
const B: usize = 64; // functions per batch

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> {
    (0..n).map(|i| { let a = u01(i as u32, seed); let b = u01(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect()
}

// a random smooth function f = Σ_{k=1..4} a_k sin(kπx), sampled on the sensor grid; + its exact antiderivative
fn sample_fn(seed: u32) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (1..=4).map(|k| (u01(k, seed) * 2.0 - 1.0) / k as f32).collect();
    let f: Vec<f32> = (0..M).map(|j| { let x = j as f32 / (M as f32 - 1.0); (0..4).map(|k| a[k] * ((k as f32 + 1.0) * std::f32::consts::PI * x).sin()).sum() }).collect();
    let dx = 1.0 / (M as f32 - 1.0);
    let mut g = vec![0.0f32; M];
    for j in 1..M { g[j] = g[j - 1] + 0.5 * (f[j] + f[j - 1]) * dx; } // ∫₀ˣ f  (exact-truth cumulative trapezoid)
    (f, g)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("DeepONet (neural operator) on the Ferric fabric — backend: {:?}", ctx.backend);
    println!("learning G[f](x) = ∫₀ˣ f  (the solution operator of u'=f, u(0)=0), verified on held-out f\n");

    // branch (f[M] -> P) + trunk (x[1] -> P) from the sciml library, + a scalar bias
    let branch = Mlp::new(&ctx, &[M, H, P], 1);
    let trunk = Mlp::new(&ctx, &[1, H, P], 3);
    let mut wp: Vec<Tensor> = branch.params.iter().chain(trunk.params.iter()).cloned().collect(); // [b0..3, t4..7]
    wp.push(Tensor::zeros(&ctx, &[1]));                                                             // bias [8]
    let mut adam = Adam::new(&wp, 1e-3);
    let xgrid: Vec<f32> = (0..M).map(|j| j as f32 / (M as f32 - 1.0)).collect();

    for epoch in 0..5000u32 {
        // batch of B random functions + their exact antiderivatives
        let mut fs = vec![0.0f32; B * M];
        let mut gs = vec![0.0f32; B * M];
        for b in 0..B { let (f, g) = sample_fn(epoch.wrapping_mul(131) + b as u32 + 1); for j in 0..M { fs[b * M + j] = f[j]; gs[b * M + j] = g[j]; } }
        let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        let branch = Mlp::forward(&pv[0..4], &Var::leaf(Tensor::from_vec(&ctx, &fs, &[B, M]))); // [B,P]
        let trunk = Mlp::forward(&pv[4..8], &Var::leaf(Tensor::from_vec(&ctx, &xgrid, &[M, 1]))); // [M,P]
        let g_pred = branch.matmul(&trunk.transpose(1, 0)).add(&pv[8]); // [B,P]·[P,M] = [B,M] operator output
        let diff = g_pred.sub(&Var::leaf(Tensor::from_vec(&ctx, &gs, &[B, M])));
        let loss = diff.mul(&diff).mean_all();
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&wp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::zeros(&ctx, &t.shape))).collect();
        adam.step(&mut wp, &g);
        if epoch % 1000 == 0 || epoch == 4999 { println!("  epoch {epoch:5}  loss {:.6}", loss.value().to_vec().await[0]); }
    }

    // ---- verify on HELD-OUT functions (unseen seeds), report relative L2 + max error ----
    let nt = 200usize;
    let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
    let trunk = Mlp::forward(&pv[4..8], &Var::leaf(Tensor::from_vec(&ctx, &xgrid, &[M, 1]))).value().to_vec().await; // [M,P]
    let mut fs = vec![0.0f32; nt * M];
    let mut gs = vec![0.0f32; nt * M];
    for b in 0..nt { let (f, g) = sample_fn(900_000 + b as u32); for j in 0..M { fs[b * M + j] = f[j]; gs[b * M + j] = g[j]; } }
    let branch = Mlp::forward(&pv[0..4], &Var::leaf(Tensor::from_vec(&ctx, &fs, &[nt, M]))).value().to_vec().await; // [nt,P]
    let b0 = wp[8].to_vec().await[0];
    let (mut se, mut sy, mut maxe) = (0.0f32, 0.0f32, 0.0f32);
    for b in 0..nt { for j in 0..M {
        let mut pred = b0; for k in 0..P { pred += branch[b * P + k] * trunk[j * P + k]; }
        let t = gs[b * M + j]; se += (pred - t).powi(2); sy += t * t; maxe = maxe.max((pred - t).abs());
    }}
    let rel = (se / sy).sqrt() * 100.0;
    println!("\n  held-out ({nt} unseen functions): relative L2 error {rel:.3}%  ·  max |G_pred − ∫f| = {maxe:.5}");
    println!("  {}", if rel < 5.0 { "PASS — a neural OPERATOR learned the solution map, GPU-native on the pure-Rust fabric ✓ (one forward pass per new function, no re-solving)" } else { "FAIL — relative L2 above 5%" });
}
