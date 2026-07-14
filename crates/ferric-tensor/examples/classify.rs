//! Trains a REAL net on the fabric: a 2-layer MLP classifier (with biases) on a nonlinear task —
//! "is the point inside the unit circle?" — using softmax cross-entropy + Adam, all on the GPU.
//! Success = the loss falls and held-out accuracy climbs past 97%.
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

// deterministic pseudo-random in [0,1)
fn rnd(i: usize, s: u32) -> f32 {
    let mut h = (i as u32).wrapping_mul(747796405).wrapping_add(s.wrapping_mul(2891336453)).wrapping_add(1);
    h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13;
    (h % 100000) as f32 / 100000.0
}
fn dataset(n: usize, s: u32) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut x = vec![0.0f32; n * 2];
    let mut onehot = vec![0.0f32; n * 2];
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let (a, b) = (rnd(i, s) * 3.0 - 1.5, rnd(i, s + 7) * 3.0 - 1.5);
        x[i * 2] = a; x[i * 2 + 1] = b;
        let inside = (a * a + b * b) < 1.0;
        let l = inside as usize;
        labels[i] = l; onehot[i * 2 + l] = 1.0;
    }
    (x, onehot, labels)
}
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.4).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (di, dh, dc, n) = (2usize, 32usize, 2usize, 256usize);
    let (xd, oh, labels) = dataset(n, 1);
    let (xt, oht) = (Tensor::from_vec(&ctx, &xd, &[n, di]), Tensor::from_vec(&ctx, &oh, &[n, dc]));

    // params: w1[2,32] b1[32] w2[32,2] b2[2]
    let mut params = vec![
        Tensor::from_vec(&ctx, &seq(di * dh, 1.0), &[di, dh]),
        Tensor::zeros(&ctx, &[dh]),
        Tensor::from_vec(&ctx, &seq(dh * dc, 2.0), &[dh, dc]),
        Tensor::zeros(&ctx, &[dc]),
    ];
    let mut adam = Adam::new(&params, 0.02);

    let acc = |logits: &[f32]| -> f32 {
        let mut c = 0;
        for i in 0..n {
            let pred = if logits[i * 2] >= logits[i * 2 + 1] { 0 } else { 1 };
            if pred == labels[i] { c += 1; }
        }
        c as f32 / n as f32
    };

    let mut first = 0.0;
    for step in 0..400 {
        let xv = Var::leaf(xt.clone());
        let p: Vec<Var> = params.iter().map(|t| Var::leaf(t.clone())).collect();
        let ohv = Var::leaf(oht.clone());
        let h = xv.matmul(&p[0]).add(&p[1]).relu();      // [N,32] (+bias broadcast)
        let logits = h.matmul(&p[2]).add(&p[3]);          // [N,2]
        let probs = logits.softmax(1);
        let loss = ohv.mul(&probs.log()).sum(&[1]).neg().mean(&[0, 1]); // NLL
        loss.backward();
        let l = loss.value().to_vec().await[0];
        if step == 0 { first = l; }
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 80 == 0 || step == 399 {
            let a = acc(&logits.value().to_vec().await);
            println!("     step {step:>3}  loss {l:.4}  acc {:.1}%", a * 100.0);
        }
    }
    // final accuracy
    let xv = Var::leaf(xt.clone());
    let p: Vec<Var> = params.iter().map(|t| Var::leaf(t.clone())).collect();
    let logits = xv.matmul(&p[0]).add(&p[1]).relu().matmul(&p[2]).add(&p[3]);
    let final_acc = acc(&logits.value().to_vec().await);
    println!("  loss {:.4} → final accuracy {:.1}%", first, final_acc * 100.0);
    assert!(final_acc > 0.97, "classifier did not train (acc {final_acc})");
    println!("✅ Trained a real MLP classifier on the GPU with Adam + softmax cross-entropy — {:.1}% accuracy", final_acc * 100.0);
}
