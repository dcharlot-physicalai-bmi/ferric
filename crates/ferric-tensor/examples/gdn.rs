//! Gated Delta Net — the linear-attention recurrence behind Qwen3-Next / Qwen3.5, and therefore
//! PrismML's Bonsai-27B (75% of its 64 layers). Validated against Hugging Face transformers'
//! own `torch_recurrent_gated_delta_rule` (the reference implementation).
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn f32s(p: &str) -> Vec<f32> { std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let d = format!("{}/.cache/ferric/gdn", std::env::var("HOME").unwrap());
    let dims: Vec<usize> = std::fs::read_to_string(format!("{d}/dims.txt")).unwrap().trim().split(',').map(|s| s.parse().unwrap()).collect();
    let (t, h, dk, dv) = (dims[0], dims[1], dims[2], dims[3]);
    let q = Tensor::from_vec(&ctx, &f32s(&format!("{d}/q.bin")), &[t * h, dk]);
    let k = Tensor::from_vec(&ctx, &f32s(&format!("{d}/k.bin")), &[t * h, dk]);
    let v = Tensor::from_vec(&ctx, &f32s(&format!("{d}/v.bin")), &[t * h, dv]);
    let (gv, bv) = (f32s(&format!("{d}/g.bin")), f32s(&format!("{d}/beta.bin")));
    let refo = f32s(&format!("{d}/ref.bin"));

    // l2norm(x) = x · rsqrt(Σx² + 1e-6)  — exactly the reference's definition
    let l2 = |x: &Tensor| x.div(&x.mul(x).sum(&[1], true).add(&x.scalar(1e-6)).sqrt());
    let qn = l2(&q).mul(&q.scalar(1.0 / (dk as f32).sqrt())); // reference scales q by 1/√dk
    let kn = l2(&k);
    // pack (g, beta) interleaved per (t,h)
    let gb: Vec<f32> = (0..t * h).flat_map(|i| [gv[i], bv[i]]).collect();
    let gbt = Tensor::from_vec(&ctx, &gb, &[t * h, 2]);

    let got = qn.gated_delta_rule(&kn, &v, &gbt, h, dk, dv).to_vec().await;
    let e = got.iter().zip(&refo).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let den = refo.iter().map(|x| x.abs()).fold(1e-6, f32::max);
    println!("Ferric · Gated Delta Net (Qwen3.5 / Bonsai linear attention) · {:?}", ctx.backend);
    println!("  T={t} H={h} dk={dk} dv={dv} · max|ferric - transformers| = {e:.2e} (rel {:.2e})", e / den);
    assert!(e / den < 1e-3, "gated delta rule mismatch {e}");
    println!("✅ Ferric implements the gated delta rule — matches HF transformers' reference recurrence");
}
