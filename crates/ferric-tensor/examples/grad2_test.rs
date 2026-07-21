//! Second-order autograd correctness test for Ferric's differentiable `grad()`.
//!
//! Checks that grad-of-grad is correct against closed forms:
//!   (1) f = Σ x³      ⇒ df/dxᵢ = 3xᵢ²  ⇒ d/dxᵢ Σⱼ(df/dxⱼ) = 6xᵢ           (a Hessian-diagonal probe)
//!   (2) f = Σ eˣ      ⇒ df/dxᵢ = eˣⁱ    ⇒ d/dxᵢ Σⱼ(df/dxⱼ) = eˣⁱ
//!   (3) an MLP energy H(x): the input-gradient ∂H/∂x is differentiable in the weights (the HNN/EBT need).
//!
//! Run: `cargo run -p ferric-tensor --example grad2_test --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let xv = vec![0.5f32, 1.0, 1.5, 2.0];

    // (1) f = Σ x³ ; g = ∂f/∂x = 3x² ; then backward Σg ⇒ ∂/∂x = 6x
    let x = Var::leaf(Tensor::from_vec(&ctx, &xv, &[4]));
    let f = x.mul(&x).mul(&x).sum_all();
    let g = grad(&f, &[x.clone()], None).remove(0);          // differentiable ∂f/∂x = 3x²
    let g_val = g.value().to_vec().await;
    g.sum_all().backward();                                   // d/dx Σ(3x²) = 6x
    let gg = x.grad().unwrap().to_vec().await;
    println!("  (1) f=Σx³");
    println!("      ∂f/∂x  = {:?}   (want 3x² = {:?})", rnd(&g_val), rnd(&xv.iter().map(|v| 3.0 * v * v).collect::<Vec<_>>()));
    println!("      ∂²     = {:?}   (want 6x  = {:?})", rnd(&gg), rnd(&xv.iter().map(|v| 6.0 * v).collect::<Vec<_>>()));

    // (2) f = Σ eˣ ; g = eˣ ; second deriv (of Σg) = eˣ
    let x2 = Var::leaf(Tensor::from_vec(&ctx, &xv, &[4]));
    let f2 = x2.exp().sum_all();
    let g2 = grad(&f2, &[x2.clone()], None).remove(0);
    g2.sum_all().backward();
    let gg2 = x2.grad().unwrap().to_vec().await;
    let want = xv.iter().map(|v| v.exp()).collect::<Vec<_>>();
    println!("\n  (2) f=Σeˣ:  ∂²={:?}   (want eˣ={:?})", rnd(&gg2), rnd(&want));

    // (3) HNN-style: MLP energy H(x); the input-grad ∂H/∂x must be differentiable in the weights.
    //     Train so ∂H/∂x matches a target field — one Adam step must reduce the loss (impossible w/o 2nd order).
    let (d, h) = (4usize, 16usize);
    let mut w = vec![
        Tensor::from_vec(&ctx, &randn(d * h, 1), &[d, h]), Tensor::zeros(&ctx, &[h]),
        Tensor::from_vec(&ctx, &randn(h, 2), &[h, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&w, 0.01);
    let target = Tensor::from_vec(&ctx, &[1.0, -1.0, 0.5, -0.5], &[1, d]); // desired ∂H/∂x
    let x3 = Tensor::from_vec(&ctx, &[0.2, -0.3, 0.1, 0.4], &[1, d]);
    let mut first = 0.0f32; let mut last = 0.0f32;
    for step in 0..40 {
        let pv: Vec<Var> = w.iter().map(|t| Var::leaf(t.clone())).collect();
        let xin = Var::leaf(x3.clone());
        let hh = xin.matmul(&pv[0]).add(&pv[1]).relu();       // H(x) = w2·relu(w1 x + b1) + b2
        let energy = hh.matmul(&pv[2]).add(&pv[3]).sum_all(); // scalar
        let dhdx = grad(&energy, &[xin.clone()], None).remove(0);   // ∂H/∂x  (differentiable in w)
        let diff = dhdx.sub(&Var::leaf(target.clone()));
        let loss = diff.mul(&diff).mean_all();               // ‖∂H/∂x − target‖²
        let lval = loss.value().to_vec().await[0];
        if step == 0 { first = lval; } last = lval;
        loss.backward();                                     // ∂loss/∂w — SECOND order (loss depends on ∂H/∂x)
        // b1/b2 don't affect the FORCE ∂H/∂x (relu''=0; constant offset) ⇒ no grad ⇒ treat as zero.
        let grads: Vec<Tensor> = pv.iter().zip(&w).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut w, &grads);
    }
    println!("\n  (3) HNN input-grad training: loss ‖∂H/∂x − target‖²  {first:.4} → {last:.4}  ({})",
        if last < first * 0.2 { "✅ 2nd-order training works" } else { "⚠ did not converge" });
}

fn randn(n: usize, s: u32) -> Vec<f32> { (0..n).map(|i| { let a = ((i as u32 * 2654435761 ^ s) % 9973 + 1) as f32 / 9973.0; let b = ((i as u32 * 40503 ^ s) % 9973 + 1) as f32 / 9973.0; ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * 0.4 }).collect() }
fn rnd(v: &[f32]) -> Vec<f32> { v.iter().map(|x| (x * 100.0).round() / 100.0).collect() }
