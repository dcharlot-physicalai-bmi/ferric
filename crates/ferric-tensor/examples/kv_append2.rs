//! **`append2` == two `append`s** — the fused K/V cache write, checked byte-for-byte.
//!
//! K and V are appended at the same point in every layer, into different buffers. `append2` does both in
//! ONE dispatch instead of two, which matters because launches are ~85% of a decode step (see
//! `dispatch_vs_submit.rs`). It is only worth having if it is exactly equivalent — a KV cache that is
//! subtly wrong produces fluent, wrong text with no error.
//!
//! The cases that matter are the awkward ones: strided source views (prefill), growth mid-sequence
//! (the cache doubles and carries its history across), and differing K/V widths (GQA).
//!
//!   cargo run -p ferric-tensor --example kv_append2 --release
use ferric_core::Context;
use ferric_tensor::{append2, KvBuf, Tensor};
use std::sync::Arc;

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f32 / 8388608.0) - 1.0 }).collect()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Fused K/V cache append vs two separate appends\n");
    println!("  {:>6} {:>7} {:>7} {:>9}  {:>12} {:>12}", "kw", "vw", "steps", "strided", "max|Δ|K", "max|Δ|V");
    println!("  {:-<64}", "");

    let mut worst = 0f32;
    // kw != vw exercises GQA; `strided` feeds windows onto a wider buffer, as the real model does.
    for &(kw, vw) in &[(64usize, 64usize), (128, 32), (32, 128)] {
        for &strided in &[false, true] {
            // Enough steps to force the cache to grow past its initial 64-row capacity at least once.
            // 40 steps produced only 52 rows and the anti-vacuity assert below caught it.
            let steps = 90;
            let (mut ka, mut va) = (KvBuf::default(), KvBuf::default());
            let (mut kb, mut vb) = (KvBuf::default(), KvBuf::default());
            let (mut fa, mut fb) = (Vec::new(), Vec::new());
            let (mut wk, mut wv) = (0f32, 0f32);

            for step in 0..steps {
                let t = if step % 7 == 3 { 3 } else { 1 }; // mix prefill-shaped and decode-shaped writes
                let (kt, vt) = if strided {
                    // Windows onto one wide [t, kw+vw] buffer — the fused-QKV shape.
                    let wide = Tensor::from_vec(&ctx, &fill(t * (kw + vw), step as u32), &[t, kw + vw]);
                    (wide.narrow(1, 0, kw), wide.narrow(1, kw, vw))
                } else {
                    (Tensor::from_vec(&ctx, &fill(t * kw, step as u32), &[t, kw]),
                     Tensor::from_vec(&ctx, &fill(t * vw, 900 + step as u32), &[t, vw]))
                };
                // reference: two independent appends
                fa = ka.append(&ctx, &kt).to_vec().await;
                let va_v = va.append(&ctx, &vt).to_vec().await;
                // fused
                let (kfv, vfv) = append2(&ctx, &mut kb, &kt, &mut vb, &vt);
                fb = kfv.to_vec().await;
                let vb_v = vfv.to_vec().await;

                assert_eq!(fa.len(), fb.len(), "step {step}: K view length diverged");
                assert_eq!(va_v.len(), vb_v.len(), "step {step}: V view length diverged");
                let dk = fa.iter().zip(&fb).fold(0f32, |a, (&x, &y)| a.max((x - y).abs()));
                let dv = va_v.iter().zip(&vb_v).fold(0f32, |a, (&x, &y)| a.max((x - y).abs()));
                worst = worst.max(dk).max(dv);
                wk = wk.max(dk); wv = wv.max(dv);
                assert_eq!(dk, 0.0, "step {step}: fused K append differs by {dk:.3e}");
                assert_eq!(dv, 0.0, "step {step}: fused V append differs by {dv:.3e}");
            }
            println!("  {kw:>6} {vw:>7} {steps:>7} {strided:>9}  {wk:>12.3e} {wv:>12.3e}");
            // Anti-vacuity: the caches must actually have grown past their initial capacity, or this
            // never exercised the reallocate-and-carry path it claims to cover.
            assert!(fa.len() > 64 * kw, "the K cache never grew past its initial capacity — growth untested");
            assert!(!fb.is_empty());
        }
    }

    println!("\n  ✅ Fused K/V append is BYTE-IDENTICAL (max|Δ| {worst:.1e}) across GQA widths, strided");
    println!("     source windows, mixed prefill/decode row counts, and cache growth. One dispatch");
    println!("     instead of two, on the structure where being subtly wrong would emit fluent, wrong");
    println!("     text rather than an error.");
}
