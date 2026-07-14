//! V-JEPA 2 on the fabric: validates 3D RoPE (the V-JEPA2-specific primitive) against a CPU ref, then
//! runs the full architecture — tubelet patch-embed (unfold+matmul) → ViT encoder (bidirectional
//! attention w/ 3D RoPE + GELU MLP + residuals) → learned mask-token injection (an elementwise blend,
//! no new op) → predictor → latent prediction. Proves Ferric runs JEPA end-to-end.
use ferric_core::Context;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.37 + s).sin()) * 0.3).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (gt, gh, gw) = (2usize, 2, 2);
    let n = gt * gh * gw; // tokens
    let (nh, dh) = (2usize, 12usize); // head_dim divisible by 6
    let d = nh * dh;
    let base = 10000.0f32;

    // ---- 3D RoPE vs CPU reference ----
    let xd = seq(n * d, 1.0);
    let got = Tensor::from_vec(&ctx, &xd, &[n, d]).rope_3d(nh, dh, base, gt, gh, gw).to_vec().await;
    let mut cref = xd.clone();
    let (g, half) = (dh / 3, dh / 6);
    for t in 0..n {
        let coords = [t / (gh * gw), (t / gw) % gh, t % gw];
        for head in 0..nh {
            for gi in 0..3 {
                let off = (t * nh + head) * dh + gi * g;
                for c in 0..half {
                    let inv = (-2.0 * c as f32 / g as f32 * base.ln()).exp();
                    let ang = coords[gi] as f32 * inv;
                    let (cs, sn) = (ang.cos(), ang.sin());
                    let (x1, x2) = (xd[off + c], xd[off + c + half]);
                    cref[off + c] = x1 * cs - x2 * sn;
                    cref[off + c + half] = x2 * cs + x1 * sn;
                }
            }
        }
    }
    let d3 = got.iter().zip(&cref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let ok3 = d3 < 1e-4;
    println!("  {} 3D RoPE (V-JEPA2) vs CPU reference        max|Δ| = {:.2e}", if ok3 { "✅" } else { "❌" }, d3);

    // ---- full V-JEPA2 forward: patch-embed → encoder → mask-token → predictor ----
    let (pf, ff) = (16usize, 48usize); // patch_flat, mlp hidden
    let tv = |v: Vec<f32>, sh: &[usize]| Tensor::from_vec(&ctx, &v, sh);
    let patches = tv(seq(n * pf, 2.0), &[n, pf]);           // n non-overlapping tubelets, flattened
    let w_patch = tv(seq(pf * d, 3.0), &[pf, d]);
    let mut x = patches.matmul(&w_patch);                   // [n, d]  (unfold+matmul patch embed)

    let block = |x: &Tensor, s: f32| {
        let wq = tv(seq(d * d, s + 1.0), &[d, d]);
        let wk = tv(seq(d * d, s + 2.0), &[d, d]);
        let wv = tv(seq(d * d, s + 3.0), &[d, d]);
        let wo = tv(seq(d * d, s + 4.0), &[d, d]);
        let w1 = tv(seq(d * ff, s + 5.0), &[d, ff]);
        let w2 = tv(seq(ff * d, s + 6.0), &[ff, d]);
        let q = nn::linear(x, &wq).rope_3d(nh, dh, base, gt, gh, gw);
        let k = nn::linear(x, &wk).rope_3d(nh, dh, base, gt, gh, gw);
        let v = nn::linear(x, &wv);
        let attn = nn::bidirectional_attention(&q, &k, &v, nh, nh); // ViT = non-causal
        let x1 = x.add(&nn::linear(&attn, &wo));
        x1.add(&nn::linear(&nn::linear(&x1, &w1).gelu(), &w2))       // GELU MLP + residual
    };
    // encoder (2 blocks)
    x = block(&x, 10.0);
    x = block(&x, 20.0);

    // mask-token injection: blend a learned mask token into masked positions (elementwise, no new op)
    let mask_tok = tv(seq(d, 99.0), &[1, d]);               // [1,d] broadcast
    let mrow: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect(); // mask evens
    let m = tv(mrow, &[n, 1]);
    let keep = tv(vec![1.0; n * 1], &[n, 1]).sub(&m);       // 1 − m
    let pred_in = x.mul(&keep).add(&mask_tok.mul(&m));      // where(masked, mask_tok, x)

    // predictor (1 block) → predicted latents
    let pred = block(&pred_in, 30.0).to_vec().await;
    let finite = pred.iter().all(|v| v.is_finite()) && pred.len() == n * d;
    println!("  {} V-JEPA2 forward: patch-embed→encoder→mask-token→predictor ({} latents)", if finite { "✅" } else { "❌" }, pred.len());

    let ok = ok3 && finite;
    println!("{}", if ok { "✅ Ferric runs V-JEPA 2 — 3D RoPE + bidirectional ViT encoder/predictor + mask-token, end to end" } else { "❌ jepa failed" });
    assert!(ok);
}
