//! Proves the fabric can TRAIN: (1) a finite-difference gradient check against autograd, and
//! (2) an actual 2-layer MLP trained with SGD on the GPU — loss must fall.
use ferric_core::Context;
use ferric_tensor::{Tensor, Var};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.7 + s).sin()) * 0.5).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());

    // ---- (1) gradient check: f = sum((X·W)²) , compare dW to finite differences ----
    let x = Tensor::from_vec(&ctx, &seq(3 * 4, 1.0), &[3, 4]);
    let w0 = seq(4 * 2, 2.0);
    let loss_of = |w: &[f32]| {
        // forward only, scalar loss (blocking readback)
        pollster::block_on(async {
            let wt = Tensor::from_vec(&ctx, w, &[4, 2]);
            let y = x.matmul(&wt);
            let l = y.mul(&y).sum(&(0..2usize).collect::<Vec<_>>(), false);
            l.to_vec().await[0]
        })
    };
    let wv = Var::leaf(Tensor::from_vec(&ctx, &w0, &[4, 2]));
    let xv = Var::leaf(x.clone());
    let y = xv.matmul(&wv);
    let loss = y.mul(&y).sum_all();
    loss.backward();
    let g = wv.grad().unwrap().to_vec().await;
    let eps = 1e-3;
    let mut max_rel = 0.0f32;
    for i in 0..w0.len() {
        let (mut wp, mut wm) = (w0.clone(), w0.clone());
        wp[i] += eps; wm[i] -= eps;
        let num = (loss_of(&wp) - loss_of(&wm)) / (2.0 * eps);
        let rel = (num - g[i]).abs() / (num.abs().max(1.0));
        max_rel = max_rel.max(rel);
    }
    let gc_ok = max_rel < 1e-2;
    println!("  {} gradient check (autograd vs finite-diff)  max rel err = {:.2e}", if gc_ok { "✅" } else { "❌" }, max_rel);

    // ---- (2) train a 2-layer MLP: fit a random nonlinear target ----
    let (n, di, dh, do_) = (32usize, 4usize, 16usize, 1usize);
    let xd = seq(n * di, 3.0);
    let xt = Tensor::from_vec(&ctx, &xd, &[n, di]);
    // target = relu(X·A)·b  (a fixed teacher), learned by a fresh student
    let a = Tensor::from_vec(&ctx, &seq(di * dh, 4.0), &[di, dh]);
    let b = Tensor::from_vec(&ctx, &seq(dh * do_, 5.0), &[dh, do_]);
    let yt = xt.matmul(&a).relu().matmul(&b); // [n,1]

    let mut w1 = Tensor::from_vec(&ctx, &seq(di * dh, 9.0), &[di, dh]);
    let mut w2 = Tensor::from_vec(&ctx, &seq(dh * do_, 11.0), &[dh, do_]);
    let lr = 0.05f32;
    let mut first = 0.0; let mut last = 0.0;
    for step in 0..120 {
        let x_v = Var::leaf(xt.clone());
        let w1v = Var::leaf(w1.clone());
        let w2v = Var::leaf(w2.clone());
        let pred = x_v.matmul(&w1v).relu().matmul(&w2v);      // [n,1]
        let diff = pred.sub(&Var::leaf(yt.clone()));
        let loss = diff.mul(&diff).mean_all();                 // MSE
        loss.backward();
        let l = loss.value().to_vec().await[0];
        if step == 0 { first = l; }
        last = l;
        // SGD: w -= lr * grad
        let lrt = Tensor::from_vec(&ctx, &[lr], &[1]);
        w1 = w1.sub(&w1v.grad().unwrap().mul(&lrt));
        w2 = w2.sub(&w2v.grad().unwrap().mul(&lrt));
        if step % 30 == 0 { println!("     step {step:>3}  loss = {l:.5}"); }
    }
    let trained = last < first * 0.2;
    println!("  {} MLP trained on GPU: loss {:.5} → {:.5}  ({:.0}% down)", if trained { "✅" } else { "❌" }, first, last, (1.0 - last / first) * 100.0);

    println!("{}", if gc_ok && trained { "✅ The fabric TRAINS — autograd verified + a real model fit end-to-end on the GPU" } else { "❌ training path failed" });
    assert!(gc_ok && trained);
}
