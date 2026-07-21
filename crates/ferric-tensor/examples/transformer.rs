//! Unification proof: a full Llama-style decoder layer built ENTIRELY on the general `ferric-tensor`
//! runtime (matmul, broadcasting, fused rmsnorm/softmax/rope, attention composed from primitives),
//! validated against the established `ferric-core` CPU reference. The transformer is now an
//! expression in the general fabric — not a separate kernel library. Also checks that the fused
//! softmax/rmsnorm fast-paths equal the primitive composition.
use ferric_core::cpu as fc; // the reference kernels (rmsnorm, rope, mha_causal, sigmoid, matmul_cpu, add)
use ferric_core::matmul_cpu;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.2 - 0.1).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn emul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x * y).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (t, h, dh, hff, base, eps) = (6usize, 4usize, 16usize, 128usize, 10000.0f32, 1e-5f32);
    let d = h * dh;
    let mut ok = true;
    let mut check = |name: &str, g: &[f32], c: &[f32], tol: f32| {
        let e = maxdiff(g, c); let pass = e < tol; ok &= pass;
        println!("  {} {:<40} max|tensor-ref| = {:.2e}", if pass { "✅" } else { "❌" }, name, e);
    };

    // weights (shared between the ferric-tensor path and the CPU reference)
    let xd = seq(t * d, 1.0);
    let wn1: Vec<f32> = seq(d, 2.0).iter().map(|v| v + 1.0).collect();
    let wn2: Vec<f32> = seq(d, 3.0).iter().map(|v| v + 1.0).collect();
    let (wq, wk, wv, wo) = (seq(d * d, 4.0), seq(d * d, 5.0), seq(d * d, 6.0), seq(d * d, 7.0));
    let (wg, wu, wd) = (seq(d * hff, 8.0), seq(d * hff, 9.0), seq(hff * d, 10.0));
    let tv = |v: &Vec<f32>, s: &[usize]| Tensor::from_vec(&ctx, v, s);

    // ---- fused fast-paths == primitive composition ----
    let sx = tv(&seq(3 * 5, 20.0), &[3, 5]);
    let prim_softmax = { let m = sx.max(&[1], true); let e = sx.sub(&m).exp(); e.div(&e.sum(&[1], true)) };
    check("fused softmax == primitives", &sx.softmax(1).to_vec().await, &prim_softmax.to_vec().await, 1e-5);
    let wn = tv(&wn1, &[d]);
    let rx = tv(&xd, &[t, d]);
    let prim_rms = { let ms = rx.mul(&rx).mean(&[1], true); rx.div(&ms.add(&rx.scalar(eps)).sqrt()).mul(&wn) };
    check("fused rmsnorm == primitives", &rx.rmsnorm(&wn, eps).to_vec().await, &prim_rms.to_vec().await, 1e-5);

    // ---- full decoder layer on the general runtime ----
    let x = tv(&xd, &[t, d]);
    let rms1 = x.rmsnorm(&tv(&wn1, &[d]), eps);
    let q = nn::linear(&rms1, &tv(&wq, &[d, d])).rope(h, dh, base, 0);
    let k = nn::linear(&rms1, &tv(&wk, &[d, d])).rope(h, dh, base, 0);
    let v = nn::linear(&rms1, &tv(&wv, &[d, d]));
    let attn = nn::causal_attention(&q, &k, &v, h, h, 0.0);
    let x2 = x.add(&nn::linear(&attn, &tv(&wo, &[d, d])));
    let rms2 = x2.rmsnorm(&tv(&wn2, &[d]), eps);
    let g = nn::linear(&rms2, &tv(&wg, &[d, hff]));
    let u = nn::linear(&rms2, &tv(&wu, &[d, hff]));
    let down = nn::linear(&g.silu().mul(&u), &tv(&wd, &[hff, d]));
    let y_tensor = x2.add(&down).to_vec().await;

    // ---- CPU reference (ferric-core kernels) ----
    let rms1 = fc::rmsnorm(&xd, &wn1, t, d, eps);
    let q = fc::rope(&matmul_cpu(&rms1, &wq, t, d, d), t, h, dh, base);
    let k = fc::rope(&matmul_cpu(&rms1, &wk, t, d, d), t, h, dh, base);
    let v = matmul_cpu(&rms1, &wv, t, d, d);
    let attn = fc::mha_causal(&q, &k, &v, t, h, h, dh);
    let x2r = fc::add(&xd, &matmul_cpu(&attn, &wo, t, d, d));
    let rms2 = fc::rmsnorm(&x2r, &wn2, t, d, eps);
    let g = matmul_cpu(&rms2, &wg, t, d, hff);
    let u = matmul_cpu(&rms2, &wu, t, d, hff);
    let down = matmul_cpu(&emul(&fc::silu(&g), &u), &wd, t, hff, d);
    let y_ref = fc::add(&x2r, &down);

    check("full decoder layer on general runtime", &y_tensor, &y_ref, 1e-3); // f32 drift over ~8 matmuls

    println!("{}", if ok { "✅ The transformer is now an expression in the general runtime — matches the ferric-core reference" } else { "❌ unification mismatch" });
    assert!(ok);
}
