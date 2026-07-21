//! **ferromotion's universal-approximator MLP, migrated onto the fabric.** The same model the
//! Physics-Informed Physical AI course trains on-device (fit a nonlinear curve with a small MLP) — but here
//! it trains on Ferric's GPU-native autograd (`Var` + `Adam`) instead of a scalar CPU tape. Everything stays
//! resident on the device across the whole step (the only host readback is the scalar loss), and the model
//! fits `sin(3x)`. This is the proof that the course's learning engine runs on the heterogeneous fabric —
//! the CPU scalar tape it used before is now just the verify-first oracle.

use ferric_core::Context;
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

// deterministic init in [-0.5,0.5], scaled (He for ReLU)
fn seq(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.7 + s).sin()) * 0.5).collect()
}
fn scaled(n: usize, s: f32, sc: f32) -> Vec<f32> {
    seq(n, s).iter().map(|v| v * sc).collect()
}

async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("training ferromotion's MLP on: {}", ctx.adapter_name);

    // data: fit sin(3x) on [-1,1], resident on-device
    let np = 48usize;
    let xs: Vec<f32> = (0..np).map(|i| -1.0 + 2.0 * i as f32 / (np - 1) as f32).collect();
    let ys: Vec<f32> = xs.iter().map(|x| (3.0 * x).sin()).collect();
    let xt = Tensor::from_vec(&ctx, &xs, &[np, 1]);
    let yt = Tensor::from_vec(&ctx, &ys, &[np, 1]);

    // MLP [1, 32, 32, 1] with ReLU + biases (biases give a 1-D ReLU net kinks away from the origin)
    let h = 32usize;
    let mut params = vec![
        Tensor::from_vec(&ctx, &scaled(h, 1.0, 1.414), &[1, h]),   // W1
        Tensor::zeros(&ctx, &[1, h]),                                   // b1
        Tensor::from_vec(&ctx, &scaled(h * h, 2.0, 0.25), &[h, h]),     // W2
        Tensor::zeros(&ctx, &[1, h]),                                   // b2
        Tensor::from_vec(&ctx, &scaled(h, 3.0, 0.25), &[h, 1]),     // W3
        Tensor::zeros(&ctx, &[1, 1]),                                   // b3
    ];
    let mut opt = Adam::new(&params, 0.01);

    let mut readbacks = 0usize;
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    let steps = 3000;
    for step in 0..steps {
        // wrap current params as leaves; everything below stays resident on the device
        let p: Vec<Var> = params.iter().map(|t| Var::leaf(t.clone())).collect();
        let x = Var::leaf(xt.clone());
        let y = Var::leaf(yt.clone());
        let h1 = x.matmul(&p[0]).add(&p[1]).relu();
        let h2 = h1.matmul(&p[2]).add(&p[3]).relu();
        let out = h2.matmul(&p[4]).add(&p[5]);
        let diff = out.sub(&y);
        let loss = diff.mul(&diff).mean_all(); // MSE
        loss.backward();
        let l = loss.value().to_vec().await[0]; // the ONE readback per step
        readbacks += 1;
        if step == 0 {
            first = l;
        }
        last = l;
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        opt.step(&mut params, &grads);
        if step % 600 == 0 {
            println!("  step {step:>4}  mse = {l:.5}");
        }
    }

    // verify the fit against the same oracle the ferromotion-learn test uses (sin(3x))
    let predict = |x: f32| -> f32 {
        pollster::block_on(async {
            let xin = Tensor::from_vec(&ctx, &[x], &[1, 1]);
            let a = xin.matmul(&params[0]).add(&params[1]).relu();
            let b = a.matmul(&params[2]).add(&params[3]).relu();
            b.matmul(&params[4]).add(&params[5]).to_vec().await[0]
        })
    };
    let mut max_err = 0.0f32;
    for k in 0..40 {
        let x = -1.0 + 2.0 * k as f32 / 39.0;
        max_err = max_err.max((predict(x) - (3.0 * x).sin()).abs());
    }

    println!("\n  MSE {first:.4} → {last:.5}   max|f−sin(3x)| = {max_err:.4}");
    println!(
        "  residency: {} host readbacks over {} steps + {} ops each — i.e. one scalar per step, everything else stayed on-device",
        readbacks, steps, 9
    );
    let ok = last < 5e-3 && max_err < 0.15;
    println!("{}", if ok { "✅ the course's MLP trained GPU-native on the fabric and fit the curve — resident, one framework" } else { "❌ did not fit" });
    assert!(ok, "MLP should fit sin(3x) on the fabric: mse {last}, max_err {max_err}");
}

fn main() {
    pollster::block_on(run());
}
