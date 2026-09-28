//! **What one Bonsai 2 27B decode step spends outside its weight matmuls** — the per-call GPU time of
//! each non-matmul op at the model's decode shapes (t = 1), next to the Q2_0 matmuls for scale. Calls
//! are recorded into ONE batch (as a forward records them) and timed with one sync, so a figure is GPU
//! throughput per call, not submit latency.
//!
//!   cargo run -p ferric-tensor --release --example gdn_decode_bench [calls]
use ferric_core::Context;
use ferric_tensor::{dtype::QMatrix, nn, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(Context::new().await.expect("gpu"));
    let calls: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(48);
    println!("adapter: {} [{:?}]  {calls} calls per op", ctx.adapter_name, ctx.backend);
    let r = |n: usize, s: f32| (0..n).map(|i| ((i as f32 * 0.37 + s).sin()) * 0.5).collect::<Vec<f32>>();
    // Qwen3.5-27B GDN geometry: 16 key heads, 48 value heads, 128 x 128 per head, conv 4.
    let (nk, nv, dk, dv, ck) = (16usize, 48usize, 128usize, 128usize, 4usize);
    let kd = nk * dk; let vd = nv * dv; let qo = 2 * kd + vd; let zo = vd;
    let pw = qo + zo + 2 * nv;
    let proj = Tensor::from_vec(&ctx, &r(pw, 0.1), &[1, pw]);
    let prev_conv = Tensor::from_vec(&ctx, &r((ck - 1) * qo, 0.2), &[ck - 1, qo]);
    let conv_w = Tensor::from_vec(&ctx, &r(qo * ck, 0.3), &[qo, ck]);
    let dt_bias = Tensor::from_vec(&ctx, &r(nv, 0.4), &[nv]);
    let a = Tensor::from_vec(&ctx, &r(nv, 0.5).iter().map(|x| -x.abs() - 0.1).collect::<Vec<_>>(), &[nv]);
    let norm = Tensor::from_vec(&ctx, &r(dv, 0.6), &[dv]);
    let x = Tensor::from_vec(&ctx, &r(5120, 0.7), &[1, 5120]);
    let wn = Tensor::from_vec(&ctx, &r(5120, 0.8), &[5120]);
    let state = Tensor::from_vec(&ctx, &r(nv * dv * dk, 0.9), &[nv, dv, dk]);
    let blocks = |n: usize, seed: u64| {
        let mut s = seed;
        let mut v: Vec<u8> = (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u8 }).collect();
        for blk in v.chunks_exact_mut(34) { blk[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes()); }
        v
    };
    let q2 = |o: usize, i: usize| QMatrix::from_bytes(&ctx, &blocks(o * (i / 128) * 34, (o + i) as u64), 42, o, i).unwrap();
    let (w_in, w_out, w_gu, w_dn) = (q2(qo + zo, 5120), q2(5120, vd), q2(34816, 5120), q2(5120, 17408));

    let (conv, _tail, v) = nn::gdn_conv(&proj, &prev_conv, &conv_w, qo, ck, vd, 2 * kd);
    let gb = nn::gdn_gate(&proj, &dt_bias, &a, nv, qo + zo);
    let (q, k) = nn::gdn_qk(&conv, nk, dk, nv / nk, qo, 1.0 / (dk as f32).sqrt(), 1e-6);
    let v3 = v.reshape(&[1, nv, dv]);
    let (o, _) = q.gated_delta_rule_stateful(&k, &v3, &gb, nv, dk, dv, Some(&state));
    let x17 = Tensor::from_vec(&ctx, &r(17408, 1.1), &[1, 17408]);
    let x6 = Tensor::from_vec(&ctx, &r(6144, 1.2), &[1, 6144]);

    type Op<'a> = Box<dyn Fn() -> Tensor + 'a>;
    let ops: Vec<(&str, Op)> = vec![
        ("gdn (delta rule, t=1)", Box::new(|| q.gated_delta_rule_stateful(&k, &v3, &gb, nv, dk, dv, Some(&state)).0)),
        ("gdn_conv", Box::new(|| nn::gdn_conv(&proj, &prev_conv, &conv_w, qo, ck, vd, 2 * kd).0)),
        ("gdn_gate", Box::new(|| nn::gdn_gate(&proj, &dt_bias, &a, nv, qo + zo))),
        ("gdn_qk", Box::new(|| nn::gdn_qk(&conv, nk, dk, nv / nk, qo, 0.088, 1e-6).0)),
        ("gdn_post", Box::new(|| nn::gdn_post(&o, &proj, &norm, qo, 1e-6))),
        ("rmsnorm 5120", Box::new(|| x.rmsnorm(&wn, 1e-6))),
        ("hadamard 5120 (blk 1024)", Box::new(|| ferric_tensor::fwht::hadamard_rows(&x, 1024, None, None))),
        ("hadamard 17408", Box::new(|| ferric_tensor::fwht::hadamard_rows(&x17, 1024, None, None))),
        ("q2_0 in_proj 5120->16384", Box::new(|| x.matmul_q(&w_in))),
        ("q2_0 gdn out 6144->5120", Box::new(|| x6.matmul_q(&w_out))),
        ("q2_0 gate_up 5120->34816", Box::new(|| x.matmul_q(&w_gu))),
        ("q2_0 down 17408->5120", Box::new(|| x17.matmul_q(&w_dn))),
    ];
    for (name, f) in &ops {
        let _ = f().to_vec().await;
        let mut t = Vec::new();
        for _ in 0..5 {
            ferric_tensor::device_sync(&ctx);
            let t0 = Instant::now();
            let keep: Vec<Tensor> = ferric_tensor::batch(&ctx, || (0..calls).map(|_| f()).collect());
            ferric_tensor::device_sync(&ctx);
            t.push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
            drop(keep);
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name:28} {:8.1} us/call (min {:.1})", t[2], t[0]);
    }
}
