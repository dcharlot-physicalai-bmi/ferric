//! **`nn::chunked_attention` against the full-attention reference** — synthetic, no checkpoint needed.
//!
//! `causal_attention` requires `q_len == kv_len`: every query sees a history exactly as long as itself.
//! Chunked prefill breaks that — a chunk of 128 queries attends over a 2,048-token history — so
//! `chunked_attention` handles queries that are the *tail* of a longer history.
//!
//! The whole risk lives in one place: the mask offset. With a history of `tkv` and `tq` queries, query
//! `i` is at absolute position `off + i` where `off = tkv - tq`, and it may attend to keys `0 ..= off + i`.
//! Get `off` wrong and the model attends to the wrong span — which produces *fluent, wrong* output rather
//! than an error, so it must be checked against a reference rather than eyeballed.
//!
//! This validates it without a checkpoint, so it runs in CI where the model-based
//! `ferric-llama/examples/chunked_prefill.rs` cannot.
//!
//!   cargo run -p ferric-tensor --example chunked_attn
use ferric_core::Context;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

/// Deterministic pseudo-random fill — a fixed pattern makes a failure reproducible.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 8388608.0) - 1.0
        })
        .collect()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Chunked attention vs the full-attention reference\n");

    // Shapes chosen to exercise grouped-query attention (n_kv_heads < n_heads) as well as MHA, since a
    // wrong head-group mapping and a wrong mask offset look identical from the output shape alone.
    let cases: &[(usize, usize, usize, f32)] = &[
        // (n_heads, n_kv_heads, head_dim, softcap)
        (4, 4, 16, 0.0),
        (8, 2, 16, 0.0),   // GQA: 4 query heads per kv head
        (4, 1, 32, 0.0),   // MQA
        (4, 2, 16, 30.0),  // with logit softcapping
    ];

    println!("  {:>6} {:>7} {:>6} {:>8}  {:>6} {:>6}  {:>13}", "heads", "kv_head", "dim", "softcap", "tkv", "tq", "max|Δ|");
    println!("  {:-<72}", "");

    let mut worst = 0f32;
    for &(nh, nkv, dh, cap) in cases {
        let d = nh * dh;
        let dkv = nkv * dh;

        for &(tkv, tq) in &[(64usize, 64usize), (64, 32), (64, 16), (64, 1), (128, 48), (37, 11)] {
            // Full history of keys/values; queries are the LAST tq rows of a same-length query tensor.
            let kfull = Tensor::from_vec(&ctx, &fill(tkv * dkv, 11), &[tkv, dkv]);
            let vfull = Tensor::from_vec(&ctx, &fill(tkv * dkv, 22), &[tkv, dkv]);
            let qfull = Tensor::from_vec(&ctx, &fill(tkv * d, 33), &[tkv, d]);

            // Reference: full causal attention over the whole sequence, then keep the last tq rows.
            let full = nn::causal_attention(&qfull, &kfull, &vfull, nh, nkv, cap).to_vec().await;
            let want = &full[(tkv - tq) * d..];

            // Chunked: only the tail queries, attending over the whole history.
            let qtail = Tensor::from_vec(&ctx, &fill(tkv * d, 33)[(tkv - tq) * d..], &[tq, d]);
            let got = nn::chunked_attention(&qtail, &kfull, &vfull, nh, nkv, cap).to_vec().await;

            assert_eq!(got.len(), want.len(), "chunked_attention returned {} values, expected {}", got.len(), want.len());
            let dmax = got.iter().zip(want).fold(0f32, |a, (&g, &w)| a.max((g - w).abs()));
            worst = worst.max(dmax);
            println!("  {nh:>6} {nkv:>7} {dh:>6} {cap:>8.0}  {tkv:>6} {tq:>6}  {dmax:>13.3e}");
            assert!(
                dmax < 2e-4,
                "n_heads={nh} n_kv={nkv} tkv={tkv} tq={tq}: chunked attention differs by {dmax:.3e} — \
                 the mask offset is wrong and the queries attended to the wrong span"
            );
        }
    }

    // Anti-vacuity: a mask offset of zero must FAIL this comparison, otherwise the test proves nothing.
    // Without this, a chunked_attention that ignored the offset entirely could pass whenever tq == tkv.
    {
        let (nh, nkv, dh, tkv, tq) = (4usize, 4usize, 16usize, 64usize, 16usize);
        let d = nh * dh;
        let kfull = Tensor::from_vec(&ctx, &fill(tkv * d, 11), &[tkv, d]);
        let vfull = Tensor::from_vec(&ctx, &fill(tkv * d, 22), &[tkv, d]);
        let qtail = Tensor::from_vec(&ctx, &fill(tkv * d, 33)[(tkv - tq) * d..], &[tq, d]);
        // Treating the tail queries as if they were positions 0..tq — i.e. offset 0 — is the bug.
        let wrong = nn::causal_attention(&qtail, &kfull.narrow(0, 0, tq).contiguous(),
                                         &vfull.narrow(0, 0, tq).contiguous(), nh, nkv, 0.0).to_vec().await;
        let right = nn::chunked_attention(&qtail, &kfull, &vfull, nh, nkv, 0.0).to_vec().await;
        let diff = right.iter().zip(&wrong).fold(0f32, |a, (&r, &w)| a.max((r - w).abs()));
        assert!(
            diff > 1e-3,
            "the offset-0 mistake produces the SAME answer (Δ {diff:.3e}) — this comparison cannot \
             detect a wrong mask offset and proves nothing"
        );
        println!("\n  Anti-vacuity: ignoring the offset changes the result by {diff:.3e}, so the check above bites.");
    }

    println!("\n  ✅ chunked_attention matches full causal attention across MHA, GQA, MQA and softcapping,");
    println!("     at every history/query split tried, worst max|Δ| {worst:.3e}. The mask offset is the");
    println!("     entire risk — a wrong one makes the model attend to the wrong span and emit fluent,");
    println!("     wrong text, so it is compared against a reference rather than inspected.");
}
