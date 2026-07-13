//! Validates the GENERAL tensor runtime against a CPU reference on the things the old fixed-shape
//! kernels could not do: broadcasting, non-contiguous inputs, arbitrary reduction axes, batched
//! matmul with batch broadcast, and a softmax composed entirely from primitives.
use ferric_core::Context;
use ferric_tensor::{cpu, Tensor};
use std::sync::Arc;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.37 + s).sin())).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    let mut check = |name: &str, g: &[f32], c: &[f32]| {
        let d = maxdiff(g, c);
        let pass = d < 1e-4; ok &= pass;
        println!("  {} {:<34} max|gpu-cpu| = {:.2e}", if pass { "✅" } else { "❌" }, name, d);
    };

    // 1) broadcasting add: [4,1,3] + [1,5,3] -> [4,5,3]
    let a = seq(12, 1.0); let b = seq(15, 2.0);
    let ta = Tensor::from_vec(&ctx, &a, &[4, 1, 3]);
    let tb = Tensor::from_vec(&ctx, &b, &[1, 5, 3]);
    let (cr, _) = cpu::binary(&a, &[4, 1, 3], &b, &[1, 5, 3], "+");
    check("broadcast add [4,1,3]+[1,5,3]", &ta.add(&tb).to_vec().await, &cr);

    // 2) non-contiguous: transpose a [3,4] then multiply by [4] (broadcast) -> [4,3]
    let x = seq(12, 3.0);
    let tx = Tensor::from_vec(&ctx, &x, &[3, 4]).transpose(0, 1); // [4,3], strided
    let v = seq(3, 4.0);
    let tv = Tensor::from_vec(&ctx, &v, &[3]);
    // cpu: transpose x logically then broadcast-mul by v over last dim
    let mut xt = vec![0.0f32; 12];
    for i in 0..4 { for j in 0..3 { xt[i * 3 + j] = x[j * 4 + i]; } }
    let (cr, _) = cpu::binary(&xt, &[4, 3], &v, &[3], "*");
    check("transposed-view mul broadcast", &tx.mul(&tv).to_vec().await, &cr);

    // 3) reductions over arbitrary axes
    let y = seq(24, 5.0);
    let ty = Tensor::from_vec(&ctx, &y, &[2, 3, 4]);
    let (cs, _) = cpu::reduce(&y, &[2, 3, 4], &[1], "sum", false);
    check("sum axis 1 of [2,3,4]", &ty.sum(&[1], false).to_vec().await, &cs);
    let (cm, _) = cpu::reduce(&y, &[2, 3, 4], &[0, 2], "mean", false);
    check("mean axes {0,2}", &ty.mean(&[0, 2], false).to_vec().await, &cm);
    let (cx, _) = cpu::reduce(&y, &[2, 3, 4], &[2], "max", true);
    check("max axis 2 keepdim", &ty.max(&[2], true).to_vec().await, &cx);

    // 4) batched matmul + batch broadcast
    let am = seq(6 * 2 * 3, 6.0); let bm = seq(6 * 3 * 5, 7.0);
    let (cr, _) = cpu::matmul(&am, &[6, 2, 3], &bm, &[6, 3, 5]);
    let tam = Tensor::from_vec(&ctx, &am, &[6, 2, 3]);
    let tbm = Tensor::from_vec(&ctx, &bm, &[6, 3, 5]);
    check("batched matmul [6,2,3]x[6,3,5]", &tam.matmul(&tbm).to_vec().await, &cr);
    let a1 = seq(2 * 3, 8.0);
    let (cr, _) = cpu::matmul(&a1, &[1, 2, 3], &bm, &[6, 3, 5]);
    let ta1 = Tensor::from_vec(&ctx, &a1, &[1, 2, 3]);
    check("matmul batch broadcast [1,..]x[6,..]", &ta1.matmul(&tbm).to_vec().await, &cr);

    // 5) unary
    let u = seq(20, 9.0);
    let tu = Tensor::from_vec(&ctx, &u, &[4, 5]);
    check("exp", &tu.exp().to_vec().await, &cpu::unary(&u, "exp"));
    check("relu", &tu.relu().to_vec().await, &cpu::unary(&u, "relu"));

    // 6) SOFTMAX over the last axis, composed ENTIRELY from primitives (no bespoke kernel)
    let s = seq(2 * 6, 10.0);
    let ts = Tensor::from_vec(&ctx, &s, &[2, 6]);
    let mx = ts.max(&[1], true);          // [2,1]
    let e = ts.sub(&mx).exp();            // broadcast sub, exp
    let sm = e.div(&e.sum(&[1], true));   // divide by row sum (broadcast)
    // cpu softmax
    let mut cref = vec![0.0f32; 12];
    for r in 0..2 {
        let row = &s[r * 6..r * 6 + 6];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = ex.iter().sum();
        for j in 0..6 { cref[r * 6 + j] = ex[j] / sum; }
    }
    check("softmax composed from primitives", &sm.to_vec().await, &cref);

    println!("{}", if ok { "✅ GENERAL tensor runtime matches the CPU reference across all cases" } else { "❌ a case failed" });
    assert!(ok);
}
