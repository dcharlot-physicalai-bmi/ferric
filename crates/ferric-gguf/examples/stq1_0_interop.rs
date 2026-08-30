//! **Does Ferric's STQ1_0 decoder agree with the published Hy4 weights?**
//!
//! A round-trip proves the encoder and decoder agree with each other, which is worth nothing if
//! both misread the layout the same way. This is the interop check: it reads bytes Ferric did not
//! write, from Tencent's own 213 GiB `Hy4-preview-STQ1_0.gguf`.
//!
//! The trick is that AngelSlim published the same 770B model twice. `blk.2.ffn_gate_exps.weight` is
//! STQ1_0 in one file and Q4_K in the other — same weights, two independent encodings, and Ferric
//! already reads Q4_K against real checkpoints. So Q4_K is the oracle, and a layout error in the
//! STQ1_0 decoder destroys the agreement between them while leaving every marginal statistic
//! (element count, value histogram, block scales) exactly as plausible as before.
//!
//! Neither file can be downloaded on a laptop, and neither needs to be: the GGUF header gives each
//! tensor's byte offset, so 43 KB and 147 KB HTTP range reads are enough for 262,144 weights.
//!
//! ```text
//! B=https://huggingface.co/AngelSlim/Hy4-preview-GGUF/resolve/main
//! # blk.2.ffn_gate_exps.weight, first 1024 super-blocks; offsets are data_start(5051520) + tensor offset
//! curl -L -r 8476688576-8476731583   -o hy4_blk2_gate.stq1_0.bin "$B/Hy4-preview-STQ1_0.gguf"
//! curl -L -r 13035700416-13035847871 -o hy4_blk2_gate.q4_k.bin   "$B/Hy4-preview-Q4_K_M.gguf"
//! cargo run --release -p ferric-gguf --example stq1_0_interop -- <dir-holding-those-two-files>
//! ```

use ferric_gguf::{deq_raw, STQ1_0_CODEBOOK};

const N: usize = 262_144;

/// The decoder as it would be written by someone who read the block layout correctly except for the
/// stride. Same 256 values per block, same multiset, 256 wrong positions.
fn deq_contiguous(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(42).take(n / 256).enumerate() {
        let d = half::f16::from_le_bytes([blk[40], blk[41]]).to_f32();
        for g in 0..64 {
            let slot = (blk[g / 2] >> (4 * (g & 1))) & 0x0F;
            let sign = (blk[32 + g / 8] >> (g % 8)) & 1;
            let qpack = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
            for p in 0..4 {
                out[bi * 256 + g * 4 + p] = d * (((qpack >> (2 * p)) & 3) as i32 - 1) as f32;
            }
        }
    }
    out
}

/// ...and by someone who assumed the scale leads the block, as it does in every other ggml type.
fn deq_scale_first(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(42).take(n / 256).enumerate() {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for g in 0..64 {
            let slot = (blk[2 + g / 2] >> (4 * (g & 1))) & 0x0F;
            let sign = (blk[34 + g / 8] >> (g % 8)) & 1;
            let qpack = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
            let base = bi * 256 + (g / 16) * 64 + (g % 16);
            for p in 0..4 { out[base + p * 16] = d * (((qpack >> (2 * p)) & 3) as i32 - 1) as f32 }
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na.sqrt() * nb.sqrt()) }
}

/// Of the weights STQ1_0 did NOT force to zero, how often does its sign match the oracle's?
fn sign_agreement(stq: &[f32], oracle: &[f32]) -> (f64, usize) {
    let (mut hit, mut n) = (0usize, 0usize);
    for (s, o) in stq.iter().zip(oracle) {
        if *s == 0.0 { continue }
        n += 1;
        if (*s > 0.0) == (*o > 0.0) { hit += 1 }
    }
    (hit as f64 / n.max(1) as f64, n)
}

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let (ps, pq) = (format!("{dir}/hy4_blk2_gate.stq1_0.bin"), format!("{dir}/hy4_blk2_gate.q4_k.bin"));
    let (rs, rq) = match (std::fs::read(&ps), std::fs::read(&pq)) {
        (Ok(a), Ok(b)) => (a, b),
        _ => { eprintln!("fixtures not found — see the header of this file for the two curl commands\n  {ps}\n  {pq}"); return Ok(()) }
    };
    assert_eq!(rs.len(), N / 256 * 42, "STQ1_0 slice is the wrong length");
    assert_eq!(rq.len(), N / 256 * 144, "Q4_K slice is the wrong length");

    let oracle = deq_raw(&rq, N, 12)?;               // Q4_K — the same weights, independently encoded
    let ferric = deq_raw(&rs, N, 43)?;               // Ferric's STQ1_0
    let wrong_stride = deq_contiguous(&rs, N);
    let wrong_scale = deq_scale_first(&rs, N);

    println!("blk.2.ffn_gate_exps.weight — {N} weights of Tencent's Hy4-preview, published twice\n");
    println!("{:<26} {:>10} {:>14} {:>12}", "STQ1_0 decoded as", "cos(Q4_K)", "sign agree", "nonzero");
    for (label, v) in [("Ferric (stride-16, d last)", &ferric),
                       ("contiguous groups", &wrong_stride),
                       ("scale-first block", &wrong_scale)] {
        let (sa, nz) = sign_agreement(v, &oracle);
        println!("{:<26} {:>10.4} {:>13.1}% {:>11.1}%", label, cosine(v, &oracle), 100.0 * sa,
                 100.0 * nz as f64 / N as f64);
    }

    let zeros = ferric.iter().filter(|v| **v == 0.0).count();
    println!("\n  forced zeros: {:.1}% (the container guarantees exactly 25.0%)", 100.0 * zeros as f64 / N as f64);
    let scales: Vec<f32> = rs.chunks_exact(42).map(|b| half::f16::from_le_bytes([b[40], b[41]]).to_f32()).collect();
    let bad = scales.iter().filter(|d| !d.is_finite() || **d <= 0.0).count();
    println!("  block scales: {} of {} finite and positive, median {:.5}",
             scales.len() - bad, scales.len(), { let mut s = scales.clone(); s.sort_by(f32::total_cmp); s[s.len() / 2] });

    // The same oracle trick settles IQ2_XXS and IQ3_XXS, which carry far more of this file than
    // STQ1_0 does: of the 213.66 GiB, IQ3_XXS is ~85 GiB and IQ2_XXS ~74 GiB against STQ1_0's ~28.
    for (tag, ty, bpb) in [("iq2_xxs", 16u32, 66usize), ("iq3_xxs", 18, 98)] {
        let (pa, pb) = (format!("{dir}/hy4_{tag}.bin"), format!("{dir}/hy4_{tag}.q4_k.bin"));
        let (ra, rb) = match (std::fs::read(&pa), std::fs::read(&pb)) { (Ok(a), Ok(b)) => (a, b), _ => continue };
        assert_eq!(ra.len(), N / 256 * bpb);
        let (mine, orc) = (deq_raw(&ra, N, ty)?, deq_raw(&rb, N, 12)?);
        let (sa, _) = sign_agreement(&mine, &orc);
        let c = cosine(&mine, &orc);
        println!("{:<26} {:>10.4} {:>13.1}%          {}", tag, c, 100.0 * sa, "(vs Q4_K)");
        assert!(c > 0.8, "{tag} decode disagrees with the Q4_K oracle at cos {c:.4}");
    }

    let c = cosine(&ferric, &oracle);
    println!("\n  {}", if c > 0.7 {
        format!("PASS — Ferric's decoder agrees with the Q4_K oracle at cos = {c:.4}")
    } else {
        format!("FAIL — cos = {c:.4}; a correct 1.3-bit decode of these weights cannot look like this")
    });
    Ok(())
}
