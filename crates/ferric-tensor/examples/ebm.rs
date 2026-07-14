//! Energy-Based Model inference on the fabric: a JEM-style energy E(x) = −logsumexp(f(x)) over a
//! small classifier f, sampled by Langevin dynamics x ← x − ε·∇ₓE + √(2ε)·noise. The crux — the
//! gradient of a scalar energy w.r.t. the INPUT — is exactly what Ferric's autograd provides. Success
//! = the sampler descends the energy (mean energy falls) — i.e. Ferric runs the EBM inference loop.
use ferric_tensor::{Tensor, Var};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.4 - 0.2).collect() }
fn noise(n: usize, step: usize) -> Vec<f32> {
    (0..n).map(|i| { let mut h = (i as u32 ^ (step as u32).wrapping_mul(2654435761)).wrapping_mul(2246822519); h ^= h >> 13; (h % 1000) as f32 / 1000.0 - 0.5 }).collect()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (b, din, hid, k) = (16usize, 8usize, 32usize, 6usize);
    // fixed classifier f(x) = relu(x·W1 + b1)·W2 + b2  → the energy landscape
    let w1 = Var::leaf(Tensor::from_vec(&ctx, &seq(din * hid, 1.0), &[din, hid]));
    let b1 = Var::leaf(Tensor::from_vec(&ctx, &seq(hid, 2.0), &[hid]));
    let w2 = Var::leaf(Tensor::from_vec(&ctx, &seq(hid * k, 3.0), &[hid, k]));
    let b2 = Var::leaf(Tensor::from_vec(&ctx, &seq(k, 4.0), &[k]));

    let mut x = Tensor::from_vec(&ctx, &seq(b * din, 9.0), &[b, din]);
    let (eps, steps) = (0.1f32, 40usize);
    let mut first = 0.0;
    for step in 0..steps {
        let xv = Var::leaf(x.clone());
        let logits = xv.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2); // [B,K]
        // energy per sample = −logsumexp_k(logits);  E = Σ energy
        let m = Var::leaf(logits.value().max(&[1], true)); // detached row-max (stability)
        let lse = logits.sub(&m).exp().sum(&[1]).log().add(&m); // [B,1]
        let energy = lse.neg().sum(&[0, 1]);                    // scalar total energy
        energy.backward();
        let e = energy.value().to_vec().await[0] / b as f32;
        if step == 0 { first = e; }
        // Langevin step in input space: x ← x − ε·∇ₓE + √(2ε)·noise
        let grad = xv.grad().unwrap();
        let nz = Tensor::from_vec(&ctx, &noise(b * din, step), &[b, din]);
        x = x.sub(&grad.mul(&x.scalar(eps))).add(&nz.mul(&x.scalar((2.0 * eps).sqrt() * 0.1)));
        if step % 10 == 0 || step == steps - 1 { println!("     step {step:>2}  mean energy {e:.4}"); }
    }
    // final energy
    let xv = Var::leaf(x.clone());
    let logits = xv.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);
    let m = Var::leaf(logits.value().max(&[1], true));
    let last = logits.sub(&m).exp().sum(&[1]).log().add(&m).neg().sum(&[0, 1]).value().to_vec().await[0] / b as f32;

    println!("  mean energy {:.4} → {:.4}  (Langevin descended the energy)", first, last);
    assert!(last < first - 0.05, "EBM sampler did not descend the energy ({first} → {last})");
    println!("✅ Ferric runs EBM inference — Langevin sampling via autograd-∇ₓE descends the energy");
}
