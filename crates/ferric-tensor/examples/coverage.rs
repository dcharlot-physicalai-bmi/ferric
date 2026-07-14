//! Model-family coverage: the primitives that let Ferric run Liquid AI (LFM2), BitNet/PrismML
//! (ternary), EBM, and JEPA. Validates causal depthwise conv1d (the LFM2 short-conv mixer) + ReLU²
//! (BitNet FFN) against CPU refs, and composes an LFM2-style gated-conv block from primitives.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.37 + s).sin())).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    let mut check = |name: &str, d: f32| { let p = d < 1e-4; ok &= p; println!("  {} {:<44} max|Δ| = {:.2e}", if p { "✅" } else { "❌" }, name, d); };

    // ---- LFM2: causal depthwise conv1d (kernel L=3) ----
    let (t, c, l) = (7usize, 5usize, 3usize);
    let xd = seq(t * c, 1.0);
    let wd = seq(c * l, 2.0);
    let got = Tensor::from_vec(&ctx, &xd, &[t, c]).depthwise_conv1d_causal(&Tensor::from_vec(&ctx, &wd, &[c, l]), l).to_vec().await;
    let mut cref = vec![0.0f32; t * c];
    for row in 0..t {
        for ch in 0..c {
            let mut acc = 0.0f32;
            for k in 0..l { let off = row as isize - l as isize + 1 + k as isize; if off >= 0 { acc += wd[ch * l + k] * xd[off as usize * c + ch]; } }
            cref[row * c + ch] = acc;
        }
    }
    check("LFM2 causal depthwise conv1d (L=3) vs CPU", maxdiff(&got, &cref));

    // ---- BitNet: ReLU² FFN activation ----
    let z = seq(30, 3.0);
    let r2 = Tensor::from_vec(&ctx, &z, &[30]).relu2().to_vec().await;
    let cr: Vec<f32> = z.iter().map(|&v| { let m = v.max(0.0); m * m }).collect();
    check("BitNet ReLU² activation vs CPU", maxdiff(&r2, &cr));

    // ---- LFM2 gated short-conv block (double-gated), composed from primitives ----
    // in_proj → (B, C, x) → x = B⊙x → causal conv1d → x = C⊙x  (out_proj omitted for the check)
    let x = Tensor::from_vec(&ctx, &xd, &[t, c]);
    let bgate = Tensor::from_vec(&ctx, &seq(t * c, 4.0), &[t, c]);
    let cgate = Tensor::from_vec(&ctx, &seq(t * c, 5.0), &[t, c]);
    let w = Tensor::from_vec(&ctx, &wd, &[c, l]);
    let block = x.mul(&bgate).depthwise_conv1d_causal(&w, l).mul(&cgate).to_vec().await;
    // cpu ref
    let bg = seq(t * c, 4.0); let cg = seq(t * c, 5.0);
    let xb: Vec<f32> = xd.iter().zip(&bg).map(|(a, b)| a * b).collect();
    let mut conv = vec![0.0f32; t * c];
    for row in 0..t { for ch in 0..c { let mut acc = 0.0f32; for k in 0..l { let off = row as isize - l as isize + 1 + k as isize; if off >= 0 { acc += wd[ch * l + k] * xb[off as usize * c + ch]; } } conv[row * c + ch] = acc; } }
    let blockref: Vec<f32> = conv.iter().zip(&cg).map(|(a, b)| a * b).collect();
    check("LFM2 gated short-conv block (B⊙ → conv → C⊙)", maxdiff(&block, &blockref));

    println!("{}", if ok { "✅ Model-family primitives: LFM2 conv1d/gating + BitNet ReLU² run on the fabric (ternary matmul in dtypes)" } else { "❌ coverage failed" });
    assert!(ok);
}
