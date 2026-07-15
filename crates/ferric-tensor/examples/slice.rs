//! narrow (zero-copy strided slice) + softplus — the two primitives the Qwen3.5 / Bonsai GDN layer
//! needs to split its fused interleaved qkvz projection and build the decay gate.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.37 + s).sin())).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    // narrow along the last dim: split [T, 10] into 3 + 3 + 4 (the interleaved-projection pattern)
    let (t, d) = (5usize, 10usize);
    let xd = seq(t * d, 1.0);
    let x = Tensor::from_vec(&ctx, &xd, &[t, d]);
    for (start, len) in [(0usize, 3usize), (3, 3), (6, 4)] {
        let got = x.narrow(1, start, len).to_vec().await;
        let exp: Vec<f32> = (0..t).flat_map(|r| xd[r * d + start..r * d + start + len].to_vec()).collect();
        let e = maxdiff(&got, &exp); let p = e == 0.0; ok &= p;
        println!("  {} narrow(dim=1, {start}..{}) exact", if p { "✅" } else { "❌" }, start + len);
    }
    // narrow along dim 0 (row slice)
    let got = x.narrow(0, 1, 3).to_vec().await;
    let e0 = maxdiff(&got, &xd[d..4 * d]); ok &= e0 == 0.0;
    println!("  {} narrow(dim=0, 1..4) exact", if e0 == 0.0 { "✅" } else { "❌" });
    // a narrowed view feeds ops correctly (materializes through the strided gather)
    let dbl = x.narrow(1, 6, 4).mul(&x.scalar(2.0)).to_vec().await;
    let expd: Vec<f32> = (0..t).flat_map(|r| xd[r * d + 6..r * d + 10].iter().map(|v| v * 2.0).collect::<Vec<_>>()).collect();
    let em = maxdiff(&dbl, &expd); ok &= em < 1e-6;
    println!("  {} narrowed view feeds ops (mul) · max|Δ| = {em:.1e}", if em < 1e-6 { "✅" } else { "❌" });
    // softplus vs CPU
    let z = seq(64, 2.0);
    let sp = Tensor::from_vec(&ctx, &z, &[64]).softplus().to_vec().await;
    let spref: Vec<f32> = z.iter().map(|&v| v.max(0.0) + (1.0 + (-v.abs()).exp()).ln()).collect();
    let es = maxdiff(&sp, &spref); ok &= es < 1e-6;
    println!("  {} softplus vs CPU · max|Δ| = {es:.1e}", if es < 1e-6 { "✅" } else { "❌" });
    // l2norm vs CPU, matching ggml's exact form: x / max(√Σx², eps) — eps clamps the divisor and
    // is NOT added under the root (that's RMSNorm's shape), and there's no mean and no weight.
    let (rows, dd) = (5usize, 16usize);
    let ld = seq(rows * dd, 3.0);
    let ln = Tensor::from_vec(&ctx, &ld, &[rows, dd]).l2norm(1e-6).to_vec().await;
    let mut lref = vec![0.0f32; rows * dd];
    for r in 0..rows {
        let ss: f32 = ld[r * dd..(r + 1) * dd].iter().map(|v| v * v).sum();
        let inv = 1.0 / ss.sqrt().max(1e-6);
        for j in 0..dd { lref[r * dd + j] = ld[r * dd + j] * inv; }
    }
    let el = maxdiff(&ln, &lref); ok &= el < 1e-6;
    println!("  {} l2norm vs CPU (ggml form: x/max(√Σx²,eps)) · max|Δ| = {el:.1e}", if el < 1e-6 { "✅" } else { "❌" });
    // and the property that defines it: each row ends up unit-norm
    let unit = (0..rows).map(|r| (ln[r * dd..(r + 1) * dd].iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs()).fold(0.0f32, f32::max);
    ok &= unit < 1e-5;
    println!("  {} l2norm rows are unit-norm · max|‖row‖−1| = {unit:.1e}", if unit < 1e-5 { "✅" } else { "❌" });

    println!("{}", if ok { "✅ narrow + softplus + l2norm — the Qwen3.5/Bonsai GDN layer's remaining primitives" } else { "❌ failed" });
    assert!(ok);
}
