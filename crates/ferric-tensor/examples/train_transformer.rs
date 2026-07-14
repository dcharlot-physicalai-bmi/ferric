//! Trains an actual TRANSFORMER by autograd — gradients flow through causal self-attention (Q·Kᵀ,
//! softmax, ·V), an FFN, residuals, and learned token+position embeddings. Task: memorize a fixed
//! sequence (next-token prediction, cyclic). Success = cross-entropy → ~0 and next-token accuracy → 100%.
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.2 - 0.1).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (v, t, d, ff) = (6usize, 12usize, 16usize, 32usize);
    let sc = 1.0 / (d as f32).sqrt();

    // fixed target sequence + cyclic next-token one-hot targets + input one-hot
    let s: Vec<usize> = (0..t).map(|i| (i * 7 + 3) % v).collect();
    let mut xoh = vec![0.0f32; t * v];
    let mut toh = vec![0.0f32; t * v];
    for i in 0..t { xoh[i * v + s[i]] = 1.0; toh[i * v + s[(i + 1) % t]] = 1.0; }
    let x_oh = Tensor::from_vec(&ctx, &xoh, &[t, v]);
    let t_oh = Tensor::from_vec(&ctx, &toh, &[t, v]);
    // causal mask [T,T]
    let mut mask = vec![0.0f32; t * t];
    for i in 0..t { for j in (i + 1)..t { mask[i * t + j] = -1e9; } }
    let m_c = Tensor::from_vec(&ctx, &mask, &[t, t]);

    // params
    let names = ["E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wout"];
    let shapes = [[v, d], [t, d], [d, d], [d, d], [d, d], [d, d], [d, ff], [ff, d], [d, v]];
    let mut params: Vec<Tensor> = names.iter().zip(shapes).enumerate()
        .map(|(i, (_, sh))| Tensor::from_vec(&ctx, &seq(sh[0] * sh[1], i as f32 + 1.0), &sh)).collect();
    let mut adam = Adam::new(&params, 0.01);

    let acc = |logits: &[f32]| -> f32 {
        let mut c = 0;
        for i in 0..t {
            let row = &logits[i * v..i * v + v];
            let pred = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            if pred == s[(i + 1) % t] { c += 1; }
        }
        c as f32 / t as f32
    };

    let mut first = 0.0;
    for step in 0..600 {
        let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
        let g = |i: usize| p[i].clone();
        let xv = Var::leaf(x_oh.clone());
        let x0 = xv.matmul(&g(0)).add(&g(1)); // token emb + pos emb
        // single-head causal self-attention
        let q = x0.matmul(&g(2));
        let k = x0.matmul(&g(3));
        let vv = x0.matmul(&g(4));
        let scores = q.matmul(&k.transpose(0, 1)).mul(&Var::leaf(q.value().scalar(sc)));
        let probs = scores.add(&Var::leaf(m_c.clone())).softmax(1);
        let attn = probs.matmul(&vv);
        let x1 = x0.add(&attn.matmul(&g(5)));
        // FFN + residual
        let h = x1.matmul(&g(6)).relu();
        let x2 = x1.add(&h.matmul(&g(7)));
        let logits = x2.matmul(&g(8)); // [T,V]
        // cross-entropy vs cyclic next token, via STABLE log-softmax (no log of an underflowed prob)
        let mx = Var::leaf(logits.value().max(&[1], true)); // detached row-max
        let sh = logits.sub(&mx);
        let logp = sh.sub(&sh.exp().sum(&[1]).log()); // log_softmax = (x−m) − log Σ exp(x−m)
        let loss = Var::leaf(t_oh.clone()).mul(&logp).sum(&[1]).neg().mean(&[0, 1]);
        loss.backward();
        let l = loss.value().to_vec().await[0];
        if step == 0 { first = l; }
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 120 == 0 || step == 599 {
            println!("     step {step:>3}  loss {l:.4}  acc {:.0}%", acc(&logits.value().to_vec().await) * 100.0);
        }
    }

    // final
    let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
    let g = |i: usize| p[i].clone();
    let x0 = Var::leaf(x_oh.clone()).matmul(&g(0)).add(&g(1));
    let scores = x0.matmul(&g(2)).matmul(&x0.matmul(&g(3)).transpose(0, 1)).mul(&Var::leaf(x0.value().scalar(sc)));
    let attn = scores.add(&Var::leaf(m_c.clone())).softmax(1).matmul(&x0.matmul(&g(4)));
    let x1 = x0.add(&attn.matmul(&g(5)));
    let x2 = x1.add(&x1.matmul(&g(6)).relu().matmul(&g(7)));
    let logits = x2.matmul(&g(8)).value().to_vec().await;
    let a = acc(&logits);
    println!("  loss {:.4} → 0 · final next-token accuracy {:.0}%", first, a * 100.0);
    assert!(a > 0.99, "transformer did not memorize the sequence (acc {a})");
    println!("✅ Trained a real TRANSFORMER by autograd (attention→softmax→FFN) — {:.0}% next-token accuracy", a * 100.0);
}
