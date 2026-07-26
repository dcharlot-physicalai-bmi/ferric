//! A 2-D PDE solved by a physics-informed neural network, GPU-native on the pure-Rust Ferric fabric —
//! extending the 1-D ODE (pinn_siren) to a real partial differential equation: Poisson's equation
//! ∇²u = u_xx + u_yy = f on the unit square, with Dirichlet boundary u=0. Manufactured exact solution
//! u*(x,y) = sin(πx) sin(πy) ⇒ f = −2π² sin(πx) sin(πy). The loss is the PDE residual at interior
//! collocation points plus the boundary condition — NO solution data. This exercises the hardest autodiff
//! case in MULTIPLE dimensions: the Laplacian needs ∂²u/∂x² and ∂²u/∂y² separately, obtained here by
//! Ferric's differentiable grad() (grad-of-grad, per-axis via a coordinate mask). Verified vs the exact
//! solution on a fine grid. Pure Rust, resident on the wgpu device (Metal here).
//!   cargo run --release --example pinn_poisson2d

use ferric_tensor::{deriv, Adam, Siren, Tensor, Var};
use std::sync::Arc;
use std::f32::consts::PI;

const HW: usize = 40; // hidden width
const GI: usize = 28; // interior grid per axis (GI² collocation points)

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> {
    (0..n).map(|i| { let a = u01(i as u32, seed); let b = u01(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect()
}

// SIREN over (x,y) via the sciml library
fn siren(xy: &Var, p: &[Var]) -> Var { Siren::forward(p, xy) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("2-D PDE PINN on the Ferric fabric — backend: {:?}", ctx.backend);
    println!("solving ∇²u = f on [0,1]², u=0 on ∂  (exact: sin πx · sin πy) from the residual alone\n");

    // interior collocation points (strictly inside), + the forcing f there
    let mut xy = Vec::with_capacity(GI * GI * 2);
    let mut fvec = Vec::with_capacity(GI * GI);
    for i in 0..GI { for j in 0..GI {
        let x = (i as f32 + 1.0) / (GI as f32 + 1.0);
        let y = (j as f32 + 1.0) / (GI as f32 + 1.0);
        xy.push(x); xy.push(y);
        fvec.push(-2.0 * PI * PI * (PI * x).sin() * (PI * y).sin());
    }}
    let ni = GI * GI;
    // boundary points (u=0), 40 per edge
    let mut bxy = Vec::new();
    for k in 0..40 { let t = k as f32 / 39.0;
        bxy.extend_from_slice(&[t, 0.0]); bxy.extend_from_slice(&[t, 1.0]);
        bxy.extend_from_slice(&[0.0, t]); bxy.extend_from_slice(&[1.0, t]); }
    let nb = bxy.len() / 2;
    // per-axis masks to pull u_x / u_y out of the [N,2] gradient
    let mx: Vec<f32> = (0..ni).flat_map(|_| [1.0f32, 0.0]).collect();
    let my: Vec<f32> = (0..ni).flat_map(|_| [0.0f32, 1.0]).collect();

    let net = Siren::new(&ctx, &[2, HW, HW, 1], 1); // 2-D input SIREN from the sciml library
    let mut wp = net.params.clone();
    let mut adam = Adam::new(&wp, 2e-3);
    let lambda = 20.0f32;

    for epoch in 0..10000u32 {
        let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        let xyv = Var::leaf(Tensor::from_vec(&ctx, &xy, &[ni, 2]));
        let mxv = Var::leaf(Tensor::from_vec(&ctx, &mx, &[ni, 2]));
        let myv = Var::leaf(Tensor::from_vec(&ctx, &my, &[ni, 2]));
        let uu = siren(&xyv, &pv);                                          // [ni,1]
        let g1 = deriv(&uu, &xyv);                                          // [∂u/∂x, ∂u/∂y]  [ni,2]
        let ux = g1.mul(&mxv).sum(&[1]);                                    // ∂u/∂x  [ni,1]
        let uy = g1.mul(&myv).sum(&[1]);                                    // ∂u/∂y
        let uxx = deriv(&ux, &xyv).mul(&mxv).sum(&[1]);                     // ∂²u/∂x²
        let uyy = deriv(&uy, &xyv).mul(&myv).sum(&[1]);                     // ∂²u/∂y²
        let lap = uxx.add(&uyy);                                           // ∇²u
        let res = lap.sub(&Var::leaf(Tensor::from_vec(&ctx, &fvec, &[ni, 1])));
        let loss_pde = res.mul(&res).mean_all();
        // boundary: u = 0
        let ub = siren(&Var::leaf(Tensor::from_vec(&ctx, &bxy, &[nb, 2])), &pv);
        let loss_bc = ub.mul(&ub).mean_all();
        let loss = loss_pde.add(&loss_bc.mul(&Var::leaf(Tensor::from_vec(&ctx, &[lambda], &[1]))));

        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&wp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::zeros(&ctx, &t.shape))).collect();
        adam.step(&mut wp, &g);
        if epoch % 2000 == 0 || epoch == 9999 { println!("  epoch {epoch:5}  loss {:.6}", loss.value().to_vec().await[0]); }
    }

    // ---- verify vs exact sin πx sin πy over a fine grid ----
    let ge = 60usize;
    let mut ev = Vec::with_capacity(ge * ge * 2);
    for i in 0..ge { for j in 0..ge { ev.push(i as f32 / (ge as f32 - 1.0)); ev.push(j as f32 / (ge as f32 - 1.0)); } }
    let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
    let ue = siren(&Var::leaf(Tensor::from_vec(&ctx, &ev, &[ge * ge, 2])), &pv).value().to_vec().await;
    let (mut se, mut sy, mut maxe) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..ge { for j in 0..ge {
        let x = i as f32 / (ge as f32 - 1.0); let y = j as f32 / (ge as f32 - 1.0);
        let t = (PI * x).sin() * (PI * y).sin(); let p = ue[i * ge + j];
        se += (p - t).powi(2); sy += t * t; maxe = maxe.max((p - t).abs());
    }}
    let rel = (se / sy).sqrt() * 100.0;
    println!("\n  vs exact sin πx·sin πy:  relative L2 {rel:.3}%  ·  max error {maxe:.5}");
    println!("  {}", if rel < 5.0 { "PASS — a PINN solved a 2-D PDE from physics alone, GPU-native on the pure-Rust fabric ✓" } else { "FAIL — relative L2 above 5%" });
}
