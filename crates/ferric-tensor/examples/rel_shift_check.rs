//! **Does the `rel_shift` kernel gather the right elements?** A same-count/wrong-order check.
//!
//! `rel_shift` turns the [t, 2t-1] relative-position score block into the [t, t] one attention
//! wants: `out[i,j] = x[i, (t-1)-i+j]`. Every wrong index inside a row is still a valid read of a
//! real float, so a transposed or sign-flipped version produces finite, plausible scores and a
//! silently wrong model. Nothing downstream can assert on it — only an independent reference can.
//!
//! The CPU reference here is written from the SHAPE CONTRACT, not from the kernel, so it cannot
//! agree by construction. Non-square `t` values are included because a square-only test passes
//! under a row/column transpose.
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let mut bad = 0;
    for t in [1usize, 2, 3, 8, 17, 64] {
        let np = 2 * t - 1;
        // Distinct values: with repeats, a wrong index can read the right number by luck.
        let x: Vec<f32> = (0..t * np).map(|i| i as f32 * 0.5 - 3.0).collect();
        let got = Tensor::from_vec(&ctx, &x, &[t, np]).rel_shift().to_vec().await;
        let mut want = vec![0f32; t * t];
        for i in 0..t { for j in 0..t { want[i * t + j] = x[i * np + (t - 1 - i + j)]; } }
        let n = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("t={t:<3} np={np:<4} {}", if n == 0 { "ok".into() } else { format!("MISMATCH in {n}/{} elements", t * t) });
        if n != 0 { bad += 1; }
    }
    println!("\n{}", if bad == 0 { "rel_shift matches the CPU reference at every size" } else { "REL_SHIFT IS WRONG" });
    assert_eq!(bad, 0);
}
