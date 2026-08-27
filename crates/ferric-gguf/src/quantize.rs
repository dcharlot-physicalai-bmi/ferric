//! **Ferric's own quantizer** — f32 in, block-quantized bytes out, pure Rust, no external tool.
//!
//! Ferric could always *read* every block format here and could never *write* one, so producing a
//! test file meant shelling out to another project's binary. That is a dependency in the
//! development pipeline even when it is absent from the dependency tree, and it decides what can be
//! verified: a format Ferric cannot write is a format Ferric can only check by asking someone else
//! what the answer is.
//!
//! ## What a quantizer owes, and what it does not
//!
//! The BYTE LAYOUT is interop and is not negotiable — a Q2_K tensor in a file on disk is laid out
//! one way, and Ferric reproduces it exactly or the file is unreadable by anything else. That much
//! is a container spec, the same kind of obligation as writing a valid PNG.
//!
//! The SEARCH is not. How you pick the scale and min for a sub-block is an optimisation problem
//! with no canonical answer, and copying another project's heuristic would import its tradeoffs
//! along with its ceiling. These use least-squares refinement over the reconstruction that will
//! actually be stored, which is a different — and testable — choice.
//!
//! ## Why the round-trip test is not enough on its own
//!
//! `quantize → dequantize → quantize` returning identical bytes proves the two halves agree with
//! EACH OTHER. If both misread the layout in the same way it still passes, so idempotence is a
//! consistency check and never an interop check. Interop is proven only against a file this crate
//! did not write — see `examples/quant_interop.rs`, which reads a real published checkpoint and
//! compares against the same model's higher-precision weights.

use half::f16;

fn wr_f16(v: f32, out: &mut [u8]) { out.copy_from_slice(&f16::from_f32(v).to_le_bytes()); }

/// Bytes one 256-element super-block occupies, by ggml type id.
pub fn block_bytes(ty: u32) -> Option<usize> {
    Some(match ty { 10 => 84, 11 => 110, _ => return None })
}

/// **Q2_K** — `value = d·sc·q − dmin·m`, `q ∈ 0..3`, per 16-element sub-block.
///
/// ⭐ **`dmin` IS SIGNED AND FERRIC USES THAT.** The stored `m` is an unsigned 4-bit index, so with a
/// non-negative `dmin` every sub-block floor is `≤ 0` and a sub-block whose values are entirely
/// positive must anchor at zero — throwing away most of a 4-level code. But `dmin` is an f16, and
/// the format's reconstruction is a plain multiply, so a NEGATIVE `dmin` puts the floor above zero
/// and any conforming dequantizer handles it with no special case. The usual quantizers never emit
/// one; this one tries both signs per super-block and keeps whichever reconstructs better.
///
/// That matters far more than "offset tensors" suggests. A sub-block is SIXTEEN CONTIGUOUS WEIGHTS,
/// and plenty of those are entirely one-sided even inside a matrix centred on zero — so the
/// one-sided case is not an edge case in a real checkpoint, it is a constant fraction of it.
///
/// ⛔ Found by a test whose prediction was wrong TWICE. First it compared Q2_K against Q3_K and read
/// a bit-width difference as an affine-vs-symmetric one. Re-specified as a within-format control it
/// said the affine format handles a DC offset WORSE — x3.50 against x2.26 — which is the opposite of
/// what an affine format is for, and the reason was already written in this comment: Ferric was
/// clamping the floor at zero because the obvious reading of the layout says it has to be.
pub fn quantize_q2_k(x: &[f32], out: &mut Vec<u8>) { quantize_q2_k_iters(x, out, 3) }

/// `iters` is the number of least-squares refinement passes. Zero is plain min/max fitting — kept
/// reachable so a test can measure what the refinement actually buys instead of asserting a
/// hand-picked error bound that only says the author guessed a number.
pub fn quantize_q2_k_iters(x: &[f32], out: &mut Vec<u8>, iters: usize) {
    assert_eq!(x.len() % 256, 0, "Q2_K quantizes whole 256-element super-blocks");
    for sb in x.chunks_exact(256) {
        // σ = +1 anchors every floor at or below zero (dmin ≥ 0); σ = −1 anchors at or above it.
        // One super-block shares ONE dmin, so the sign is a per-super-block choice — and that is
        // why the negative branch is rare rather than common: a single negative weight anywhere in
        // the 256 forces its sub-block's floor to 0 under σ=−1, which is strictly worse than what
        // σ=+1 gives it. So σ=−1 can only win when the WHOLE super-block is non-negative, and this
        // O(256) scan settles that before paying for a second encode.
        let all_non_negative = sb.iter().all(|&v| v >= 0.0);
        let (blk, sse) = encode_q2_k_super(sb, iters, 1.0);
        let best = if all_non_negative {
            let (b2, s2) = encode_q2_k_super(sb, iters, -1.0);
            if s2 < sse { b2 } else { blk }
        } else { blk };
        out.extend_from_slice(&best);
    }
}

/// Encode one 256-element super-block at a fixed floor polarity, returning the bytes and the squared
/// reconstruction error they cost — so the caller can price the two polarities against each other
/// rather than reasoning about which should win.
pub fn encode_q2_k_super(sb: &[f32], iters: usize, sigma: f32) -> ([u8; 84], f32) {
    let mut scale_f = [0.0f32; 16];
    let mut base_f = [0.0f32; 16];
    for (j, g) in sb.chunks_exact(16).enumerate() {
        let (lo, hi) = g.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        // At σ=+1 the floor may not rise above zero; at σ=−1 it may not fall below it.
        let mut base = if sigma > 0.0 { lo.min(0.0) } else { lo.max(0.0) };
        let mut scale = ((hi - base) / 3.0).max(0.0);
        // Min/max fitting hands the whole sub-block to its two most extreme values. Refit
        // (scale, base) by least squares against the codes they actually produce, which lets a
        // lone outlier lose the argument to the fifteen weights it was distorting.
        for _ in 0..iters {
            if scale <= 0.0 { break }
            let (mut sq, mut sqq, mut sx, mut sqx) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for &v in g {
                let q = ((v - base) / scale).round().clamp(0.0, 3.0);
                sq += q; sqq += q * q; sx += v; sqx += q * v;
            }
            let det = 16.0 * sqq - sq * sq;
            if det.abs() < 1e-12 { break }
            let (ns, nb) = ((16.0 * sqx - sq * sx) / det, (sqq * sx - sq * sqx) / det);
            if !(ns > 0.0) || !nb.is_finite() { break }
            scale = ns;
            base = if sigma > 0.0 { nb.min(0.0) } else { nb.max(0.0) };
        }
        base_f[j] = base;
        scale_f[j] = scale;
    }
    // ml_j = −base_j, and ml_j = dmin·m_j with m_j unsigned — so dmin carries the sign.
    let d = scale_f.iter().cloned().fold(0.0f32, f32::max) / 15.0;
    let ml_max = base_f.iter().map(|b| -b * sigma).fold(0.0f32, f32::max);
    let dmin = sigma * ml_max / 15.0;
    let mut blk = [0u8; 84];
    let (mut sc_i, mut m_i) = ([0u8; 16], [0u8; 16]);
    for j in 0..16 {
        sc_i[j] = if d > 0.0 { (scale_f[j] / d).round().clamp(0.0, 15.0) as u8 } else { 0 };
        m_i[j] = if dmin != 0.0 { (base_f[j] / -dmin).round().clamp(0.0, 15.0) as u8 } else { 0 };
        blk[j] = sc_i[j] | (m_i[j] << 4);
    }
    wr_f16(d, &mut blk[80..82]);
    wr_f16(dmin, &mut blk[82..84]);
    // Re-fit against the scale that will ACTUALLY be reconstructed, not the float one that was
    // used to choose it. Quantizing the scale then quantizing the weights against the
    // pre-rounding scale biases every weight in the sub-block in the same direction.
    let (d, dmin) = (f16::from_f32(d).to_f32(), f16::from_f32(dmin).to_f32());
    let mut sse = 0.0f32;
    for (j, g) in sb.chunks_exact(16).enumerate() {
        let dl = d * sc_i[j] as f32;
        let ml = dmin * m_i[j] as f32;
        for (l, &v) in g.iter().enumerate() {
            let q = if dl > 0.0 { ((v + ml) / dl).round().clamp(0.0, 3.0) } else { 0.0 };
            let e = dl * q - ml - v;
            sse += e * e;
            // Same interleave the dequantizer walks: byte index picks the element, shift picks
            // the sub-block. j>=8 lives in the second 32-byte half.
            let (half, jj) = (j / 8, j % 8);
            let (shift, grp) = (2 * (jj / 2), jj % 2);
            blk[16 + half * 32 + grp * 16 + l] |= (q as u8) << shift;
        }
    }
    (blk, sse)
}

/// **Q3_K** — `value = d·(s−32)·q`, `q ∈ −4..3`, per 16-element sub-block. Symmetric: no min.
///
/// The quant range is ASYMMETRIC — one more step below zero than above — so the sub-block scale has
/// to satisfy both ends, `max(hi/3, −lo/4)`. Fitting to `max|x|/4` alone silently clips every
/// sub-block whose largest magnitude is positive.
pub fn quantize_q3_k(x: &[f32], out: &mut Vec<u8>) { quantize_q3_k_iters(x, out, 3) }

/// See [`quantize_q2_k_iters`] for why `iters` is reachable.
pub fn quantize_q3_k_iters(x: &[f32], out: &mut Vec<u8>, iters: usize) {
    assert_eq!(x.len() % 256, 0, "Q3_K quantizes whole 256-element super-blocks");
    for sb in x.chunks_exact(256) {
        let mut scale_f = [0.0f32; 16];
        for (j, g) in sb.chunks_exact(16).enumerate() {
            let (lo, hi) = g.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let mut scale = (hi / 3.0).max(-lo / 4.0).max(0.0);
            for _ in 0..iters { // symmetric, so one unknown: scale = <x,q>/<q,q>
                if scale <= 0.0 { break }
                let (mut sqq, mut sqx) = (0.0f32, 0.0f32);
                for &v in g {
                    let q = (v / scale).round().clamp(-4.0, 3.0);
                    sqq += q * q; sqx += q * v;
                }
                if sqq <= 0.0 { break }
                let ns = sqx / sqq;
                if !(ns > 0.0) { break }
                scale = ns;
            }
            scale_f[j] = scale;
        }
        let d_all = scale_f.iter().cloned().fold(0.0f32, f32::max) / 31.0;
        let mut s6 = [0u8; 16];
        for j in 0..16 {
            let e = if d_all > 0.0 { (scale_f[j] / d_all).round().clamp(0.0, 31.0) as u8 } else { 0 };
            s6[j] = e + 32; // stored biased; the dequantizer subtracts 32
        }
        let mut blk = [0u8; 110];
        pack_q3_scales(&s6, &mut blk[96..108]);
        wr_f16(d_all, &mut blk[108..110]);
        let d_all = f16::from_f32(d_all).to_f32();
        for (j, g) in sb.chunks_exact(16).enumerate() {
            let dl = d_all * (s6[j] as i32 - 32) as f32;
            for (l, &v) in g.iter().enumerate() {
                let q = if dl != 0.0 { (v / dl).round().clamp(-4.0, 3.0) as i32 } else { 0 };
                let (half, jj) = (j / 8, j % 8);
                let (shift, grp) = (2 * (jj / 2), jj % 2);
                let i = grp * 16 + l;
                blk[32 + half * 32 + i] |= ((q + 4) as u8 & 3) << shift;
                // The high plane is INVERTED: bit SET means "add nothing", bit CLEAR means "−4".
                // So a non-negative quant sets its bit and a negative one leaves it clear.
                // ⚠ `hmask` is 32 bytes for the WHOLE super-block, not 16 per half, and the
                // reference's bit selector is declared OUTSIDE the half loop — so it runs 1..128
                // across both halves rather than restarting. Indexing it per-half instead writes
                // half the block's high bits over the other half's, which reconstructs to finite
                // plausible weights and cost this file its first two test failures.
                if q >= 0 { blk[i] |= 1 << (half * 4 + jj / 2); }
            }
        }
        out.extend_from_slice(&blk);
    }
}

/// Pack sixteen 6-bit scales into Q3_K's 12 bytes — the exact inverse of the `aux`/`kmask` shuffle
/// the dequantizer undoes. Low nibbles go to the first eight bytes, and the high 2 bits of all
/// sixteen scales are interleaved four-to-a-byte across the last four.
fn pack_q3_scales(s: &[u8; 16], out: &mut [u8]) {
    for b in 0..4 {
        out[b] = (s[b] & 0xF) | ((s[8 + b] & 0xF) << 4);
        out[4 + b] = (s[4 + b] & 0xF) | ((s[12 + b] & 0xF) << 4);
        out[8 + b] = ((s[b] >> 4) & 3) | (((s[4 + b] >> 4) & 3) << 2)
                   | (((s[8 + b] >> 4) & 3) << 4) | (((s[12 + b] >> 4) & 3) << 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deq_raw;

    /// Deterministic pseudo-random weights with a realistic heavy-tailed shape — a uniform ramp
    /// would let a quantizer that ignores the distribution score as well as one that does not.
    fn weights(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            u * u * u * 0.4 // cubed: most mass near zero, occasional outlier
        }).collect()
    }

    /// The scale shuffle is 16 six-bit values through a 12-byte keyhole; an off-by-one in the
    /// interleave loses two bits of four scales and shows up downstream as a mildly wrong tensor.
    #[test]
    fn six_bit_scale_packing_round_trips_over_the_full_range() {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        for seed in 0..64u8 {
            let s: [u8; 16] = std::array::from_fn(|i| (i as u8 * 13 + seed * 7) % 64);
            let mut p = [0u8; 12];
            pack_q3_scales(&s, &mut p);
            let mut aux = [0u32; 4];
            for k in 0..3 { aux[k] = u32::from_le_bytes([p[k*4], p[k*4+1], p[k*4+2], p[k*4+3]]); }
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let got: Vec<u8> = aux.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(&got[..], &s[..], "6-bit scale packing is not its own inverse at seed {seed}");
        }
    }

    /// The layout test. Every quant lands in a specific (byte, shift) slot and the interleave is the
    /// easy thing to get wrong in BOTH directions at once — so this fixes one element to a known
    /// value, quantizes, dequantizes, and checks the value comes back where it was put.
    #[test]
    fn every_one_of_the_256_positions_survives_the_interleave() {
        for ty in [10u32, 11u32] {
            for pos in [0usize, 1, 15, 16, 17, 31, 32, 127, 128, 129, 200, 255] {
                let mut x = vec![0.0f32; 256];
                x[pos] = 1.0;
                x[(pos + 7) % 256] = -1.0; // a negative too: Q3_K's high plane is sign-dependent
                let mut raw = Vec::new();
                if ty == 10 { quantize_q2_k(&x, &mut raw) } else { quantize_q3_k(&x, &mut raw) }
                let back = deq_raw(&raw, 256, ty).expect("dequant");
                let peak = back.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                assert_eq!(peak, pos, "type {ty}: put +1.0 at {pos}, largest value came back at {peak}");
                let trough = back.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                assert_eq!(trough, (pos + 7) % 256, "type {ty}: negative landed at {trough}");
            }
        }
    }

    /// ⚠ CONSISTENCY, NOT INTEROP. Both halves could misread the layout identically and still pass.
    /// What it does catch is a quantizer that is not a fixed point — one whose output, fed back in,
    /// moves — which means the search is picking a scale it then fails to reproduce.
    #[test]
    fn quantizing_an_already_quantized_tensor_changes_nothing() {
        for ty in [10u32, 11u32] {
            let x = weights(256 * 8, 0xF00D);
            let mut a = Vec::new();
            if ty == 10 { quantize_q2_k(&x, &mut a) } else { quantize_q3_k(&x, &mut a) }
            let back = deq_raw(&a, x.len(), ty).expect("dequant");
            let mut b = Vec::new();
            if ty == 10 { quantize_q2_k(&back, &mut b) } else { quantize_q3_k(&back, &mut b) }
            assert_eq!(a, b, "type {ty}: re-quantizing its own output produced different bytes");
        }
    }

    /// ⛔ This test first asserted a HAND-PICKED error bound, and Q2_K failed it at 0.1774 against a
    /// guessed 0.145. There is no way to tell from that number alone whether the quantizer is bad or
    /// the guess was — a bound nobody derived grades nothing, and the only available fix is to move
    /// it until it passes, which is the failure mode the assertion was supposed to prevent.
    ///
    /// So the bar is the quantizer's own simpler self. `iters = 0` is plain min/max fitting, the
    /// obvious implementation; `iters = 3` refits each sub-block by least squares against the codes
    /// it actually emits. The refinement has to WIN, on every seed, or it is not earning its cycles.
    #[test]
    fn least_squares_refinement_beats_the_min_max_fit_it_replaces() {
        for ty in [10u32, 11u32] {
            for seed in [1u64, 2, 3, 4] {
                let x = weights(256 * 16, seed);
                let nrmse = |iters: usize| -> f32 {
                    let mut raw = Vec::new();
                    if ty == 10 { quantize_q2_k_iters(&x, &mut raw, iters) }
                    else { quantize_q3_k_iters(&x, &mut raw, iters) }
                    let back = deq_raw(&raw, x.len(), ty).expect("dequant");
                    let se: f32 = x.iter().zip(&back).map(|(a, b)| (a - b) * (a - b)).sum();
                    let sx: f32 = x.iter().map(|a| a * a).sum();
                    (se / sx).sqrt()
                };
                let (plain, refined) = (nrmse(0), nrmse(3));
                assert!(refined < plain,
                        "type {ty} seed {seed}: refinement made it WORSE — {refined:.4} vs min/max {plain:.4}");
                // And a floor, because a quantizer that returned the input unchanged would also
                // "beat" the baseline. 2 bits per weight cannot reconstruct anything this well.
                assert!(refined > 0.001,
                        "type {ty} seed {seed}: NRMSE {refined:.4} is impossible at {} levels per \
                         sub-block — the test is not measuring what it claims",
                        if ty == 10 { 4 } else { 8 });
                println!("  type {ty} seed {seed}: min/max {plain:.4} -> refined {refined:.4}");
            }
        }
    }

    /// ⛔ REFUTED AS FIRST WRITTEN, and the refutation is the useful part. The claim was "the affine
    /// format beats the symmetric one on offset data" — Q2_K 2.5659 against Q3_K 1.3642, so Q3_K won
    /// by nearly 2x. The comparison was confounded: Q2_K and Q3_K differ in the min term AND in bit
    /// width, and a whole extra bit buys more than an offset term costs. The test was measuring
    /// 2-bit-vs-3-bit and reading the answer as affine-vs-symmetric.
    ///
    /// Controlled version: compare each format to ITSELF on the same weights with and without a DC
    /// offset. Bit width is then fixed and only the offset varies. Q2_K stores a per-sub-block min,
    /// so it should absorb the shift almost for free; Q3_K has no min term, so the offset spends its
    /// dynamic range and the error has to climb.
    #[test]
    fn only_the_affine_format_absorbs_a_dc_offset() {
        let centred = weights(256 * 4, 9);
        let shifted: Vec<f32> = centred.iter().map(|v| v + 0.6).collect();
        let rms = |x: &[f32], ty: u32| -> f32 {
            let mut raw = Vec::new();
            if ty == 10 { quantize_q2_k(x, &mut raw) } else { quantize_q3_k(x, &mut raw) }
            let back = deq_raw(&raw, x.len(), ty).unwrap();
            (x.iter().zip(&back).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / x.len() as f32).sqrt()
        };
        let q2 = rms(&shifted, 10) / rms(&centred, 10);
        let q3 = rms(&shifted, 11) / rms(&centred, 11);
        println!("  offset cost — Q2_K x{q2:.2}, Q3_K x{q3:.2}");
        assert!(q2 < q3, "the whole point of the signed dmin is that Q2_K's floor follows the data. \
                          Q2_K grew x{q2:.2} against Q3_K's x{q3:.2}, so it is not following");
    }
}
