//! **What angle does the mRoPE kernel actually apply, per sector?**
//!
//! Feeding `x[c] = 1, x[c + half] = 0` makes the rotation return `cos` in the low half and `sin` in
//! the high half, so ONE call reports every sector's angle exactly as the shipped kernel computes it
//! — no reimplementation of the rule, which is the trap this whole area keeps falling into.
//!
//! ⛔ WHY THIS EXISTS. The text path cannot validate the sector→component map at all: with an
//! all-text prompt the three position components are EQUAL, so any assignment gives the same answer.
//! The map is only observable once an image makes them differ.
use ferric_tensor::{MropeMode, Tensor};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let (t, head_dim, base) = (10usize, 128usize, 5_000_000f32);
    let half = head_dim / 2;
    let sections = [24u32, 20, 20, 0];

    // the positions from the committed LM fixture's sequence
    let tt: Vec<u32> = vec![0, 1, 2, 2, 2, 2, 4, 5, 6, 7];
    let hh: Vec<u32> = vec![0, 1, 2, 2, 3, 3, 4, 5, 6, 7];
    let ww: Vec<u32> = vec![0, 1, 2, 3, 2, 3, 4, 5, 6, 7];
    let mut pos = Vec::new();
    for v in [&tt, &hh, &ww] { pos.extend(v.iter().copied()); }
    pos.extend(std::iter::repeat_n(0u32, t));

    let mut probe = vec![0f32; t * head_dim];
    for i in 0..t { for c in 0..half { probe[i * head_dim + c] = 1.0; } }
    let x = Tensor::from_vec(&ctx, &probe, &[t, head_dim]);
    let out = x.rope_mrope(1, head_dim, base, &pos, sections, MropeMode::Interleaved).to_vec().await;

    // ⛔ ONE-HOT POSITIONS, not a difference. A first attempt compared two tokens whose w
    // differed by 1 and called a sector "unmoved" below 1e-6 — but inv_freq[58] is 8.5e-7, so the
    // highest sectors moved by LESS than the threshold and were misclassified. With exactly one
    // component set to 1 and the others 0, a sector's angle is EXACTLY inv_freq[c] if it follows that
    // component and EXACTLY 0 if it does not, at every frequency. No threshold to get wrong.
    let probe_one = |comp: usize| -> Vec<f32> {
        let mut pos = vec![0u32; 4 * t];
        pos[comp * t + 1] = 1;                   // token 1 only, so token 0 stays an all-zero control
        let mut probe = vec![0f32; t * head_dim];
        for i in 0..t { for c in 0..half { probe[i * head_dim + c] = 1.0; } }
        let x = Tensor::from_vec(&ctx, &probe, &[t, head_dim]);
        pollster::block_on(
            x.rope_mrope(1, head_dim, base, &pos, sections, MropeMode::Interleaved).to_vec())
    };
    let runs: Vec<Vec<f32>> = (0..3).map(probe_one).collect();
    let lb = (base as f64).ln();
    let mut map = vec![' '; half];
    for c in 0..half {
        let inv = (-2.0 * c as f64 / head_dim as f64 * lb).exp();
        let mut hit = Vec::new();
        for (k, r) in runs.iter().enumerate() {
            let (cs, sn) = (r[1 * head_dim + c] as f64, r[1 * head_dim + c + half] as f64);
            // angle is exactly inv (this component) or exactly 0 (not) — compare against inv/2
            if sn.atan2(cs).abs() > inv / 2.0 { hit.push(k); }
        }
        map[c] = match hit.as_slice() { [0] => 'T', [1] => 'H', [2] => 'W', [] => 'e', _ => '?' };
    }
    let idx = |ch: char| (0..half).filter(|&c| map[c] == ch).collect::<Vec<_>>();
    println!("KERNEL's actual sector map (head_dim {head_dim}, sections {sections:?}):");
    for ch in ['T', 'H', 'W', 'e', '?'] {
        let v = idx(ch);
        if !v.is_empty() { println!("  {ch} ({:2}): {:?}", v.len(), v); }
    }
    // llama.cpp's rule, written out independently here for comparison.
    let (s0, s1, s2) = (sections[0] as usize, sections[1] as usize, sections[2] as usize);
    let want: Vec<char> = (0..half).map(|c| {
        if c % 3 == 1 && c < 3 * s1 { 'H' }
        else if c % 3 == 2 && c < 3 * s2 { 'W' }
        else if c % 3 == 0 && c < 3 * s0 { 'T' }
        else { 'e' }
    }).collect();
    let diff: Vec<usize> = (0..half).filter(|&c| map[c] != want[c]).collect();
    // Dump the kernel's cos/sin for the REAL positions so it can be diffed against HF's own
    // tensors elementwise, rather than compared through a derived "map".
    {
        let mut probe = vec![0f32; t * head_dim];
        for i in 0..t { for c in 0..half { probe[i * head_dim + c] = 1.0; } }
        let x = Tensor::from_vec(&ctx, &probe, &[t, head_dim]);
        let o = x.rope_mrope(1, head_dim, base, &pos, sections, MropeMode::Interleaved).to_vec().await;
        let path = std::env::var("FERRIC_ROPE_DUMP").unwrap_or_default();
        if !path.is_empty() {
            let mut b: Vec<u8> = ((t * head_dim) as u32).to_le_bytes().to_vec();
            for v in &o { b.extend(v.to_le_bytes()); }
            std::fs::write(&path, &b).unwrap();
            println!("wrote {path} ({} floats: cos in low half, sin in high half)", o.len());
        }
    }
    println!("\nvs llama.cpp's rule: {}",
             if diff.is_empty() { "IDENTICAL".to_string() }
             else { format!("DIFFERS at {diff:?} — kernel {:?} vs rule {:?}",
                            diff.iter().map(|&c| map[c]).collect::<Vec<_>>(),
                            diff.iter().map(|&c| want[c]).collect::<Vec<_>>()) });
    // HF assigns T to every sector llama.cpp leaves unrotated.
    let hf_diff: Vec<usize> = (0..half).filter(|&c| want[c] == 'e').collect();
    println!("vs HF's rule:        differs only where llama.cpp leaves 'e' -> HF uses T: {hf_diff:?}");
}
