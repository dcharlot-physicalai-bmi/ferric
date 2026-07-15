//! `cat` (concat) over strided views — the primitive Bonsai's partial RoPE needs (rotate the first
//! 64 of 256 head dims, pass the rest through, rejoin) and that assembles mixed_qkv.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;

    // ---- cat along the last dim ----
    let a = Tensor::from_vec(&ctx, &[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let b = Tensor::from_vec(&ctx, &[7., 8., 9., 10.], &[2, 2]);
    let c = a.cat(&b, 1);
    let e = maxdiff(&c.to_vec().await, &[1., 2., 3., 7., 8., 4., 5., 6., 9., 10.]);
    let p = c.shape == [2, 5] && e < 1e-6; ok &= p;
    println!("{} cat dim=1 [2,3]+[2,2] → {:?} max|Δ| = {e:.1e}", if p { "✅" } else { "❌" }, c.shape);

    // ---- cat along dim 0 ----
    let d = a.cat(&a, 0);
    let e0 = maxdiff(&d.to_vec().await, &[1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.]);
    let p0 = d.shape == [4, 3] && e0 < 1e-6; ok &= p0;
    println!("{} cat dim=0 [2,3]+[2,3] → {:?} max|Δ| = {e0:.1e}", if p0 { "✅" } else { "❌" }, d.shape);

    // ---- narrow → cat round-trip must be the identity: this IS the partial-RoPE path, and it
    // exercises cat reading from *strided views* rather than fresh contiguous buffers ----
    let xd: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let x = Tensor::from_vec(&ctx, &xd, &[2, 12]);
    let rt = x.narrow(1, 0, 4).cat(&x.narrow(1, 4, 8), 1);
    let er = maxdiff(&rt.to_vec().await, &xd);
    let pr = er == 0.0; ok &= pr;
    println!("{} narrow→cat round-trip over strided views max|Δ| = {er:.1e}", if pr { "✅" } else { "❌" });

    // ---- 3D cat on the head dim, the exact Bonsai shape: [T, heads, 64] ++ [T, heads, 192] ----
    let (t, h) = (3usize, 4usize);
    let full: Vec<f32> = (0..t * h * 256).map(|i| (i as f32 * 0.017).sin()).collect();
    let xf = Tensor::from_vec(&ctx, &full, &[t, h, 256]);
    let rejoined = xf.narrow(2, 0, 64).cat(&xf.narrow(2, 64, 192), 2);
    let e3 = maxdiff(&rejoined.to_vec().await, &full);
    let p3 = rejoined.shape == [t, h, 256] && e3 == 0.0; ok &= p3;
    println!("{} [T,{h},64]++[T,{h},192] → {:?} max|Δ| = {e3:.1e}", if p3 { "✅" } else { "❌" }, rejoined.shape);

    println!("{}", if ok { "✅ cat: concat over strided views — partial RoPE + qkv assembly" } else { "❌ cat failed" });
    assert!(ok);
}
