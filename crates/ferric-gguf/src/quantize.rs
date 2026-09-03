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
            // Computed in u32: seed reaches 63 and 63 * 7 = 441, so the intermediate
            // overflows u8 and panics in a debug build before the mod ever runs. The
            // values wanted are the mod-64 ones, which u32 gives exactly.
            let s: [u8; 16] =
                std::array::from_fn(|i| ((i as u32 * 13 + seed as u32 * 7) % 64) as u8);
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

// ─────────────────────────────────── STQ1_0 ───────────────────────────────────

use crate::{STQ1_0_BLOCK_BYTES, STQ1_0_CODEBOOK};

/// Pack one group of four ternary lanes into its `(slot, sign)` code pair.
///
/// The reverse map is DERIVED from [`STQ1_0_CODEBOOK`] by scanning it rather than transcribed as a
/// second 256-entry table. Two tables that must agree are two chances to disagree, and a
/// transcription error in the reverse map would show up only as a wrong weight — the encoder would
/// still emit a legal, decodable block. Scanning 32 entries costs nothing here because the encoder
/// is not on any hot path.
///
/// Returns `None` for a group that is not 3:4 — zero, two, three or four zeros have no encoding at
/// all, so a caller that has not enforced the constraint finds out here rather than by silently
/// writing a wrong pattern.
pub(crate) fn stq1_0_pack_group(lanes: [i8; 4]) -> Option<(u8, u8)> {
    let mut qpack = 0u8;
    for (p, &l) in lanes.iter().enumerate() {
        let code: u8 = if l == 0 { 1 } else if l < 0 { 0 } else { 2 };
        qpack |= code << (2 * p);
    }
    let idx = STQ1_0_CODEBOOK.iter().position(|&c| c == qpack)?;
    Some(((idx & 0x0F) as u8, (idx >> 4) as u8))
}

/// The four element indices group `g` covers inside a 256-weight super-block.
///
/// ⚠ Stride 16, not contiguous. See [`crate::STQ1_0_CODEBOOK`] — decoding or encoding these as
/// four consecutive weights permutes the block without changing its length or its multiset, so
/// nothing downstream can detect it.
#[inline]
pub(crate) fn stq1_0_group_idx(g: usize) -> [usize; 4] {
    let base = (g / 16) * 64 + (g % 16);
    [base, base + 16, base + 32, base + 48]
}

fn stq1_0_emit(sel: &[i8; 256], d: f32, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + STQ1_0_BLOCK_BYTES, 0);
    let blk = &mut out[start..];
    for g in 0..64 {
        let ix = stq1_0_group_idx(g);
        let lanes = [sel[ix[0]], sel[ix[1]], sel[ix[2]], sel[ix[3]]];
        let (slot, sign) = stq1_0_pack_group(lanes)
            .expect("STQ1_0 selection must be 3:4 — exactly one zero per stride-16 group");
        blk[g / 2] |= slot << (4 * (g & 1));
        blk[32 + g / 8] |= sign << (g % 8);
    }
    wr_f16(d, &mut blk[40..42]);
}

/// **STQ1_0, reference search** — `d = max|x|` over the super-block, zero the smallest-magnitude
/// lane of each group.
///
/// Kept reachable so a test can measure what the least-squares search in [`quantize_stq1_0`] buys
/// instead of asserting a number nobody checked. It is also the search that pins `d` to the block's
/// largest outlier, which for post-training weights is close to the worst available choice: every
/// surviving lane is reconstructed at ±max, so a block with one heavy tail drags all 192 of its
/// non-zero weights out with it.
pub fn quantize_stq1_0_amax(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len() % 256 == 0, "STQ1_0 needs a multiple of 256 elements, got {}", x.len());
    for xb in x.chunks_exact(256) {
        let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let mut sel = [0i8; 256];
        for g in 0..64 {
            let ix = stq1_0_group_idx(g);
            let zero = ix.iter().copied().min_by(|&a, &b| xb[a].abs().total_cmp(&xb[b].abs())).unwrap();
            for &j in &ix {
                sel[j] = if j == zero { 0 } else if xb[j] < 0.0 { -1 } else { 1 };
            }
        }
        stq1_0_emit(&sel, amax, out);
    }
}

/// **STQ1_0, Ferric's search** — alternating exact minimisation of the weighted reconstruction
/// error, over the scale and over which lane each group forfeits.
///
/// The objective is `Σ_j w_j (x_j − d·s_j)²` with `s ∈ {−1,0,+1}` and exactly one zero per
/// stride-16 group. Both coordinates have closed forms, which is why this converges in three
/// rounds rather than needing a search:
///
/// * **Which lane to zero.** Writing the group cost out, `Σ_j w_j(|x_j| − d)²` is the same
///   whichever lane is dropped, so the choice reduces to minimising `w_p·(x_p² − (|x_p| − d)²)`
///   alone. Note that this is NOT "zero the smallest weight": for `|x_p| > d/2` the term is
///   positive and zeroing is a loss, so the comparison is against what the lane would have cost as
///   `±d`, not against zero. The amax search gets this wrong whenever a group's smallest element is
///   still large relative to `d`.
/// * **The scale.** For a fixed selection, `d = Σ w_j s_j x_j / Σ w_j s_j²` — plain weighted least
///   squares, and `s_j² ∈ {0,1}` so the denominator just counts the surviving lanes' weights.
///
/// `imatrix` is the per-element activation importance; when present it multiplies `w`. STQ1_0's
/// rate is low enough that the importance weighting is doing most of the work, which is why the
/// published Hy4 build ships an imatrix alongside it rather than treating one as optional.
///
/// Unlike a plain fixed-iteration loop this keeps the **best** iterate rather than the last:
/// alternating minimisation on a discrete-continuous objective is monotone only per step, and a
/// selection flip can raise the total. Returning the last one would make more rounds occasionally
/// worse, which is the kind of regression a "more iterations is better" assumption hides.
pub fn quantize_stq1_0(x: &[f32], imatrix: Option<&[f32]>, out: &mut Vec<u8>) {
    quantize_stq1_0_iters(x, imatrix, out, 3)
}

/// `iters = 0` fits `d` once from the amax selection and stops — the ablation point for the doc
/// comment above.
pub fn quantize_stq1_0_iters(x: &[f32], imatrix: Option<&[f32]>, out: &mut Vec<u8>, iters: usize) {
    assert!(x.len() % 256 == 0, "STQ1_0 needs a multiple of 256 elements, got {}", x.len());
    if let Some(im) = imatrix {
        assert_eq!(im.len(), x.len(), "imatrix must have one importance per element");
    }
    for (bi, xb) in x.chunks_exact(256).enumerate() {
        let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        if !(amax > 0.0) {
            // ⚠ STQ1_0 cannot express an all-zero block through its SELECTION: the container
            // forces three non-zero lanes into every group, so `[0; 256]` is not encodable at all
            // and `stq1_0_emit` rightly refuses it. A zero block is expressed through the SCALE —
            // any legal 3:4 pattern reconstructs to zero once `d = 0`. Formats with a free ternary
            // alphabet have no such corner, which is why it is easy to walk into.
            let mut sel = [1i8; 256];
            for g in 0..64 { sel[stq1_0_group_idx(g)[0]] = 0 }
            stq1_0_emit(&sel, 0.0, out);
            continue;
        }
        let sumx2: f32 = xb.iter().map(|v| v * v).sum();
        let sigma2 = 2.0 * sumx2 / 256.0;
        let mut w = [0.0f32; 256];
        for j in 0..256 {
            let base = (sigma2 + xb[j] * xb[j]).sqrt();
            w[j] = match imatrix { Some(im) => im[bi * 256 + j] * base, None => base };
        }

        let mut d = amax;
        let mut sel = [0i8; 256];
        let (mut best_d, mut best_sel, mut best_err) = (amax, [0i8; 256], f32::INFINITY);

        for _ in 0..iters.max(1) {
            for g in 0..64 {
                let ix = stq1_0_group_idx(g);
                let mut zero = ix[0];
                let mut best = f32::INFINITY;
                for &j in &ix {
                    let ax = xb[j].abs();
                    let cost = w[j] * (xb[j] * xb[j] - (ax - d) * (ax - d));
                    if cost < best { best = cost; zero = j; }
                }
                for &j in &ix {
                    sel[j] = if j == zero { 0 } else if xb[j] < 0.0 { -1 } else { 1 };
                }
            }

            let (mut num, mut den) = (0.0f32, 0.0f32);
            for j in 0..256 {
                let q = sel[j] as f32;
                num += w[j] * q * xb[j];
                den += w[j] * q * q;
            }
            // A non-positive optimum would mean flipping every sign; the selection already carries
            // the signs, so keep the previous scale rather than encoding a negative one.
            let dnew = if den > 0.0 && num / den > 0.0 { num / den } else { d };

            let err: f32 = (0..256).map(|j| {
                let r = xb[j] - dnew * sel[j] as f32;
                w[j] * r * r
            }).sum();
            if err < best_err { best_err = err; best_d = dnew; best_sel = sel; }

            let converged = (dnew - d).abs() <= 1e-6 * d;
            d = dnew;
            if converged { break }
        }

        stq1_0_emit(&best_sel, best_d, out);
    }
}

#[cfg(test)]
mod stq1_0_tests {
    use super::*;
    use crate::{deq_raw, type_size, STQ1_0_CODEBOOK};

    const STQ1_0: u32 = 43;

    fn lanes_of(qpack: u8) -> [i32; 4] {
        let mut l = [0i32; 4];
        for p in 0..4 { l[p] = (((qpack >> (2 * p)) & 3) as i32) - 1 }
        l
    }

    /// The codebook is the format. If it is wrong, everything below agrees with it and passes.
    #[test]
    fn codebook_is_exactly_the_3_of_4_ternary_patterns() {
        let mut seen = std::collections::HashSet::new();
        for (i, &c) in STQ1_0_CODEBOOK.iter().enumerate() {
            assert!(seen.insert(c), "codebook entry {i} = {c:#04x} is a duplicate");
            let l = lanes_of(c);
            assert!(l.iter().all(|v| (-1..=1).contains(v)), "entry {i}: lane code 3 is not a value");
            assert_eq!(l.iter().filter(|&&v| v == 0).count(), 1,
                       "entry {i} = {c:#04x} decodes to {l:?} — the format forces exactly one zero");
        }
        // The sign bit is a whole-group negation, so the two halves must mirror.
        for slot in 0..16 {
            let a = lanes_of(STQ1_0_CODEBOOK[slot]);
            let b = lanes_of(STQ1_0_CODEBOOK[16 + slot]);
            assert_eq!(a.map(|v| -v), b, "slot {slot}: sign=1 is not the negation of sign=0");
        }
        // Sign 0 is defined as "first non-zero lane is +1".
        for slot in 0..16 {
            let a = lanes_of(STQ1_0_CODEBOOK[slot]);
            assert_eq!(*a.iter().find(|&&v| v != 0).unwrap(), 1, "slot {slot} breaks the sign convention");
        }
        // 4 zero positions x 8 sign patterns.
        assert_eq!(seen.len(), 32);
    }

    #[test]
    fn block_is_42_bytes_for_256_weights() {
        assert_eq!(type_size(STQ1_0, 256).unwrap(), 42);
        assert_eq!(type_size(STQ1_0, 4096).unwrap(), 42 * 16);
        assert!((42.0_f64 * 8.0 / 256.0 - 1.3125).abs() < 1e-9, "1.3125 bpw is the whole point");
    }

    /// **Golden vector — the stride-16 grouping.** Hand-built from the spec, not from this crate's
    /// encoder, so it is an interop check and not an idempotence check.
    ///
    /// Every group takes slot 0 / sign 0 = `0xA9` = lanes `(0, +1, +1, +1)`. Under the real
    /// stride-16 layout that puts the zeros at the FIRST SIXTEEN elements of each 64-element chunk.
    /// Under the plausible-but-wrong contiguous reading it would put a zero at every fourth
    /// element. Both write 64 zeros and 192 ones into 256 slots, so only their positions separate
    /// them — which is exactly why this needs a golden vector rather than a round-trip.
    #[test]
    fn golden_stride16_layout_not_contiguous() {
        let mut blk = vec![0u8; 42];
        blk[40..42].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let got = deq_raw(&blk, 256, STQ1_0).unwrap();

        let mut want = vec![1.0f32; 256];
        for c in 0..4 { for i in 0..16 { want[c * 64 + i] = 0.0 } }
        assert_eq!(got, want, "STQ1_0 groups are stride-16 within each 64-weight chunk");

        let contiguous: Vec<f32> = (0..256).map(|i| if i % 4 == 0 { 0.0 } else { 1.0 }).collect();
        assert_ne!(got, contiguous, "the test itself would not distinguish the two layouts");
    }

    /// **Golden vector — the scale lives at the END of the block.** Bytes 0..40 are codes; only
    /// 40..42 are `d`. Reading the leading two bytes as the scale yields 0.0 here and a silently
    /// rescaled tensor in general.
    #[test]
    fn golden_scale_is_the_last_field() {
        let mut blk = vec![0u8; 42];
        blk[40..42].copy_from_slice(&half::f16::from_f32(-2.5).to_le_bytes());
        let got = deq_raw(&blk, 256, STQ1_0).unwrap();
        assert_eq!(got[16], -2.5, "d was not read from bytes 40..42");
        assert_eq!(got[0], 0.0);
        let leading = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        assert_eq!(leading, 0.0, "guard: the wrong read must not coincidentally give -2.5");
    }

    /// **Golden vector — the zero position follows the slot, at stride 16.** Slot 12 zeroes lane 3,
    /// which is element `+48`, not element `+3`.
    #[test]
    fn golden_slot_selects_the_stride16_lane() {
        let mut blk = vec![0u8; 42];
        blk[0] = 12; // group 0 -> slot 12 (low nibble); group 1 keeps slot 0 (high nibble)
        blk[40..42].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let got = deq_raw(&blk, 256, STQ1_0).unwrap();
        assert_eq!(lanes_of(STQ1_0_CODEBOOK[12]), [1, 1, 1, 0], "slot 12 should zero lane 3");
        assert_eq!((got[0], got[16], got[32], got[48]), (1.0, 1.0, 1.0, 0.0), "group 0 misplaced");
        assert_eq!((got[1], got[17], got[33], got[49]), (0.0, 1.0, 1.0, 1.0), "group 1 disturbed");
    }

    /// The low nibble is the EVEN group. Getting this backwards swaps every adjacent pair of
    /// groups — 256 correct values, 256 wrong positions.
    #[test]
    fn golden_low_nibble_is_the_even_group() {
        let mut blk = vec![0u8; 42];
        blk[0] = 0xC0; // high nibble = 12 -> group 1; low nibble = 0 -> group 0
        blk[40..42].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let got = deq_raw(&blk, 256, STQ1_0).unwrap();
        assert_eq!(got[0], 0.0, "group 0 should still be slot 0 (zero at lane 0)");
        assert_eq!(got[49], 0.0, "group 1 should be slot 12 (zero at lane 3 -> element 1+48)");
    }

    /// The sign byte is LSB-first: group `g` uses bit `g % 8` of byte `g / 8`.
    #[test]
    fn golden_sign_bit_order() {
        let mut blk = vec![0u8; 42];
        blk[32] = 0b0000_0010; // bit 1 -> group 1 only
        blk[40..42].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let got = deq_raw(&blk, 256, STQ1_0).unwrap();
        assert_eq!(got[16], 1.0, "group 0 must stay sign=0");
        assert_eq!(got[17], -1.0, "group 1 must be negated");
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// Every encoded group is 3:4 whatever the input — the container cannot express anything else,
    /// so this is checking that the ENCODER never tries to.
    #[test]
    fn encoder_always_emits_exactly_one_zero_per_group() {
        let mut seed = 7u64;
        let x: Vec<f32> = (0..256 * 8).map(|_| lcg(&mut seed)).collect();
        for (label, bytes) in [("amax", { let mut o = Vec::new(); quantize_stq1_0_amax(&x, &mut o); o }),
                               ("ls",   { let mut o = Vec::new(); quantize_stq1_0(&x, None, &mut o); o })] {
            let y = deq_raw(&bytes, x.len(), STQ1_0).unwrap();
            for b in 0..8 {
                for g in 0..64 {
                    let ix = stq1_0_group_idx(g);
                    let z = ix.iter().filter(|&&j| y[b * 256 + j] == 0.0).count();
                    assert_eq!(z, 1, "{label}: block {b} group {g} has {z} zeros");
                }
            }
        }
    }

    /// A block already on the `{−d, 0, +d}` grid with a legal 3:4 pattern must survive exactly.
    #[test]
    fn on_grid_values_round_trip_exactly() {
        let d = 0.375f32;
        let mut x = vec![0.0f32; 256];
        let mut seed = 99u64;
        for g in 0..64 {
            let ix = stq1_0_group_idx(g);
            let zero = (seed as usize) % 4;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            for (p, &j) in ix.iter().enumerate() {
                x[j] = if p == zero { 0.0 } else if (seed >> (p + 3)) & 1 == 1 { d } else { -d };
            }
        }
        let mut out = Vec::new();
        quantize_stq1_0(&x, None, &mut out);
        let y = deq_raw(&out, 256, STQ1_0).unwrap();
        assert_eq!(y, x, "an on-grid 3:4 block must be reproduced exactly");
    }

    /// What the least-squares search actually buys over the reference `d = amax`. This asserts a
    /// direction, and prints the size so a regression shows up as a number and not just a pass.
    #[test]
    fn least_squares_beats_amax_on_gaussian_weights() {
        let mut seed = 12345u64;
        // Box-Muller-ish: sum of uniforms is close enough to normal for a weight-like distribution,
        // and a few heavy outliers are what make `d = amax` bad.
        let x: Vec<f32> = (0..256 * 64).map(|i| {
            let v: f32 = (0..6).map(|_| lcg(&mut seed)).sum::<f32>() / 6.0;
            if i % 997 == 0 { v * 12.0 } else { v }
        }).collect();

        let err = |bytes: &[u8]| -> f32 {
            let y = deq_raw(bytes, x.len(), STQ1_0).unwrap();
            (x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / x.len() as f32).sqrt()
        };
        let mut a = Vec::new(); quantize_stq1_0_amax(&x, &mut a);
        let mut l = Vec::new(); quantize_stq1_0(&x, None, &mut l);
        let mut l0 = Vec::new(); quantize_stq1_0_iters(&x, None, &mut l0, 1);

        let (ea, el, e0) = (err(&a), err(&l), err(&l0));
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        eprintln!("STQ1_0 RMSE / signal RMS  amax {:.3}  ls(1) {:.3}  ls(3) {:.3}",
                  ea / rms, e0 / rms, el / rms);
        assert!(el < ea, "least squares ({el:.4}) should beat amax ({ea:.4})");
        assert!(el <= e0 * 1.0001, "three rounds must not be worse than one — best-iterate is kept");
    }

    /// An importance matrix must lower the error *it* measures, or it is not being used.
    ///
    /// ⛔ This test first asserted that a high-importance lane stops being zeroed, and it failed at
    /// 19/64 — because that premise is wrong. The cost of forfeiting lane `p` is
    /// `w_p·(x_p² − (|x_p| − d)²)`, which is NEGATIVE whenever `|x_p| < d/2`: for a lane too small
    /// to be worth rounding up to `±d`, zeroing is the better reconstruction, and importance makes
    /// the encoder *more* eager to take it, not less. Importance protects lanes where zeroing
    /// hurts; it does not protect small lanes, and asserting that it does was a claim about
    /// magnitude dressed up as a claim about importance.
    #[test]
    fn imatrix_lowers_the_error_it_weights() {
        let mut seed = 5u64;
        let x: Vec<f32> = (0..256 * 16).map(|_| lcg(&mut seed)).collect();
        let im: Vec<f32> = (0..x.len()).map(|j| if j % 64 < 16 { 1000.0 } else { 0.001 }).collect();

        let mut flat = Vec::new(); quantize_stq1_0(&x, None, &mut flat);
        let mut weighted = Vec::new(); quantize_stq1_0(&x, Some(&im), &mut weighted);
        assert_ne!(flat, weighted, "the imatrix did not reach the encoder");

        let werr = |bytes: &[u8]| -> f32 {
            let y = deq_raw(bytes, x.len(), STQ1_0).unwrap();
            x.iter().zip(&y).zip(&im).map(|((a, b), w)| w * (a - b) * (a - b)).sum()
        };
        let (ef, ew) = (werr(&flat), werr(&weighted));
        eprintln!("STQ1_0 importance-weighted SSE  flat {ef:.4}  imatrix {ew:.4}  ({:.1}% lower)",
                  100.0 * (ef - ew) / ef);
        assert!(ew < ef, "imatrix-aware encoding ({ew:.4}) must beat flat ({ef:.4}) on weighted error");
    }

    /// An all-zero block has no scale to fit; it must still emit a legal block.
    #[test]
    fn all_zero_block_is_legal() {
        let x = vec![0.0f32; 256];
        let mut out = Vec::new();
        quantize_stq1_0(&x, None, &mut out);
        assert_eq!(out.len(), 42);
        assert_eq!(deq_raw(&out, 256, STQ1_0).unwrap(), x);
    }
}

#[cfg(test)]
mod iq_xxs_tests {
    use crate::{type_size, IQ2XXS_GRID, IQ3XXS_GRID};

    /// A transcribed codebook is where a silent wrong-number bug lives, and neither grid has any
    /// internal redundancy to check against — except its alphabet. Every IQ2_XXS grid byte is one
    /// of three magnitudes and every IQ3_XXS byte one of eight, so a mistyped hex digit almost
    /// certainly lands outside. This is the cheapest available guard on 3 KB of constants.
    #[test]
    fn grid_bytes_stay_inside_their_alphabets() {
        let a2 = [8u8, 25, 43];
        for (i, v) in IQ2XXS_GRID.iter().enumerate() {
            for b in v.to_le_bytes() {
                assert!(a2.contains(&b), "IQ2XXS_GRID[{i}] = {v:#018x} has byte {b}, not in {a2:?}");
            }
        }
        // 62 breaks the step-of-8 progression and is not a typo for 60.
        let a3 = [4u8, 12, 20, 28, 36, 44, 52, 62];
        for (i, v) in IQ3XXS_GRID.iter().enumerate() {
            for b in v.to_le_bytes() {
                assert!(a3.contains(&b), "IQ3XXS_GRID[{i}] = {v:#010x} has byte {b}, not in {a3:?}");
            }
        }
        assert_eq!(IQ2XXS_GRID.len(), 256);
        assert_eq!(IQ3XXS_GRID.len(), 256);
        let d2 = IQ2XXS_GRID.iter().collect::<std::collections::HashSet<_>>().len();
        let d3 = IQ3XXS_GRID.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!((d2, d3), (256, 256), "a duplicated codebook entry wastes an index and hides a typo");
    }

    /// The sign byte is a parity code, which is why Ferric derives it instead of carrying ggml's
    /// 128-byte table. If that is wrong, every eighth weight of every group flips.
    #[test]
    fn derived_signs_are_the_even_parity_code() {
        for i in 0u8..128 {
            let v = super::super::ksigns_for_proof(i);
            assert_eq!(v & 0x7f, i, "low seven bits must be the index itself");
            assert_eq!(v.count_ones() % 2, 0, "ksigns({i}) = {v:#04x} must have even population count");
        }
        let all: std::collections::HashSet<u8> = (0u8..128).map(super::super::ksigns_for_proof).collect();
        assert_eq!(all.len(), 128, "the code must be injective");
    }

    #[test]
    fn block_sizes_are_the_published_rates() {
        assert_eq!(type_size(16, 256).unwrap(), 66);   // 2.0625 bpw
        assert_eq!(type_size(18, 256).unwrap(), 98);   // 3.0625 bpw
        assert_eq!(type_size(16, 4096).unwrap(), 66 * 16);
        assert_eq!(type_size(18, 4096).unwrap(), 98 * 16);
    }
}

// ─────────────────────────────── proofs ───────────────────────────────
//
// What a test cannot do, and why these exist.
//
// Every defect that cost real time in this format was a SAME-COUNT/WRONG-ORDER bug: the stride-16
// grouping, the scale at the end of the block, the low nibble being the even group, the sign bit
// order. None of them changes a length, an element count or a distribution, so no assert fires and
// no summary statistic moves. The golden vectors catch each one — but only at the specific values
// those vectors happen to use, and a golden vector is an example, not a theorem.
//
// Kani closes that gap for the parts that are pure combinatorics on bits. These are not tests over
// chosen inputs; they are bounded model checks over ALL inputs in range, so "no two groups alias"
// means no two, not none of the pairs someone thought to write down.
//
// ⛔ What they deliberately do NOT cover, so that a green proof run is not read as more than it is:
//   * the WGSL kernels. Kani verifies Rust; the shaders are strings handed to a driver. Their
//     INDEX ARITHMETIC is mirrored and proved here, which is where their traps live, but the
//     shader text is tied to the Rust only by the runtime bit-comparison tests.
//   * that Ferric's decode matches Tencent's. That is an empirical claim about someone else's
//     bytes and it is settled by `examples/stq1_0_interop.rs`, not by a proof.
//   * anything floating point. The energy figures and the least-squares search are measurements
//     and heuristics respectively; neither is a theorem and neither is claimed as one.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// **The decoder puts every group exactly where the encoder says it goes.**
    ///
    /// This is the theorem that catches the stride, and it has teeth for a structural reason: the
    /// two sides derive the layout INDEPENDENTLY. The encoder addresses groups through
    /// `stq1_0_group_idx`; `deq_stq1_0` computes `(g/16)*64 + (g%16) + p*16` inline, in another
    /// file, and neither calls the other. Running the real decoder and checking it against the real
    /// encoder map is a cross-check between two derivations, not a function agreeing with itself.
    ///
    /// The neighbouring groups are left at slot 0 / sign 0 rather than zeroed, so they decode to
    /// `(0, +1, +1, +1)` and occupy their slots with known values. If group `g`'s lanes landed
    /// anywhere else, a neighbour's value would be sitting at `g`'s position and the assertion
    /// would see it — a zeroed background would let a misplacement hide in zeros.
    ///
    /// For all 64 groups and all 32 codes, not for the ones a golden vector happened to use.
    ///
    /// WARNING: the f16 conversion is STUBBED here (see `crate::rd_f16_one`) because `half` reaches
    /// runtime CPU-feature detection that Kani cannot encode. This theorem is about WHERE THE BYTES
    /// GO; it says nothing about f16 arithmetic, which the runtime tests cover instead.
    #[kani::proof]
    #[kani::unwind(70)]
    #[kani::stub(crate::rd_f16, crate::rd_f16_one)]
    fn decoder_places_every_group_where_the_encoder_says() {
        let g: usize = kani::any();
        let slot: u8 = kani::any();
        let sign: u8 = kani::any();
        kani::assume(g < 64 && slot < 16 && sign < 2);

        let mut blk = [0u8; 42];
        blk[g / 2] |= slot << (4 * (g & 1));
        blk[32 + g / 8] |= sign << (g % 8);
        blk[40] = 0x00;
        blk[41] = 0x3C; // 1.0, so the decoded value IS the lane

        let out = crate::deq_stq1_0(&blk, 256);
        assert!(out.len() == 256, "the decoder produced the wrong element count");

        let q = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
        let idx = stq1_0_group_idx(g);
        let mut p = 0;
        while p < 4 {
            let want = (((q >> (2 * p)) & 3) as i32 - 1) as f32;
            assert!(out[idx[p]] == want, "group {g} lane {p} is not at index {}", idx[p]);
            p += 1;
        }
    }

    /// **The vectorised shader's traversal addresses every (group, lane) exactly where the decoder
    /// does.** This is the proof for the 1.62x energy win, which until now rested on "packed and
    /// dense agree on random weights".
    ///
    /// `MATMUL_STQ1_0_V4_WGSL` inverts the loop: for chunk `c`, quad `k`, member `t` and lane `p`
    /// it reads code word `c*2 + (k>>1)` at nibble `4*(k&1) + t`, sign word `c>>1` at bit
    /// `(c&1)*16 + 4k + t`, and pairs the result with activation `c*64 + p*16 + 4k + t`. Kani
    /// cannot read WGSL, so `v4_addressing` is that arithmetic transcribed into Rust -- six lines,
    /// reviewable against the shader by eye -- and this harness is the cross-check between it and
    /// the REAL decoder's own inline derivation. The code for group `c*16 + 4k + t` is placed using
    /// the MIRROR's word/nibble/bit; the block is decoded by `deq_stq1_0`; the value must surface at
    /// the MIRROR's element. If any of the five expressions were wrong, the decoder would either
    /// read a different group (slot 0, so lanes (0,+1,+1,+1)) or the value would land elsewhere,
    /// and over all 4x4x4x4 positions and all 32 codes a symbolic `slot` finds the lane that
    /// disagrees.
    ///
    /// What remains unproved is that the shader TEXT matches these six lines. That is a
    /// transcription, and it is tied by `dtype::stq1_0_kernel::every_group_position_reaches_the_kernel`,
    /// which runs the real GPU kernel over every (group, code) pair.
    #[kani::proof]
    #[kani::unwind(70)]
    #[kani::stub(crate::rd_f16, crate::rd_f16_one)]
    fn vec4_traversal_addresses_every_lane_where_the_decoder_does() {
        let c: usize = kani::any();
        let k: usize = kani::any();
        let t: usize = kani::any();
        let p: usize = kani::any();
        let slot: u8 = kani::any();
        let sign: u8 = kani::any();
        kani::assume(c < 4 && k < 4 && t < 4 && p < 4 && slot < 16 && sign < 2);

        // --- the mirror: MATMUL_STQ1_0_V4_WGSL lines `sword`, `w0/w1/word`, `nib0`, `sb0`, `e0` ---
        let code_word = c * 2 + (k >> 1);          // w0 when k < 2, w1 otherwise
        let nibble    = 4 * (k & 1) + t;           // nib0 + t
        let sign_word = c >> 1;
        let sign_bit  = (c & 1) * 16 + 4 * k + t;  // sb0 + t
        let elem      = c * 64 + p * 16 + 4 * k + t; // e0 + t, paired with qp[t] lane p

        // Place the code through the mirror's coordinates. The repacked u32 words are
        // little-endian views of qs[4w..4w+4], so nibble n of word w is byte 4w + n/2, half n&1.
        let mut blk = [0u8; 42];
        blk[code_word * 4 + nibble / 2] |= slot << (4 * (nibble & 1));
        blk[32 + sign_word * 4 + sign_bit / 8] |= sign << (sign_bit % 8);
        blk[40] = 0x00;
        blk[41] = 0x3C;

        let out = crate::deq_stq1_0(&blk, 256);
        let q = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
        let want = (((q >> (2 * p)) & 3) as i32 - 1) as f32;
        assert!(out[elem] == want, "vec4 traversal (c={c},k={k},t={t},p={p}) disagrees with the decoder at {elem}");
    }

    /// **The container refuses everything that is not 3:4.**
    ///
    /// `pack_group` is the only path from lanes to a code, so this is where the structural guarantee
    /// is enforced. A group with zero, two, three or four zeros has no encoding at all, and the
    /// only correct response is to refuse — silently emitting the nearest legal pattern would put
    /// wrong weights in a well-formed file.
    #[kani::proof]
    fn only_three_of_four_groups_are_encodable() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let c: i8 = kani::any();
        let d: i8 = kani::any();
        kani::assume((-1..=1).contains(&a) && (-1..=1).contains(&b)
                  && (-1..=1).contains(&c) && (-1..=1).contains(&d));
        let lanes = [a, b, c, d];
        let zeros = lanes.iter().filter(|v| **v == 0).count();

        match stq1_0_pack_group(lanes) {
            Some((slot, sign)) => {
                assert!(zeros == 1, "a group with {zeros} zeros was given an encoding");
                assert!(slot < 16 && sign < 2, "code out of range");
                let q = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
                let mut p = 0;
                while p < 4 {
                    assert!((((q >> (2 * p)) & 3) as i8) - 1 == lanes[p], "round trip changed lane {p}");
                    p += 1;
                }
            }
            None => assert!(zeros != 1, "a legal 3:4 group was refused an encoding"),
        }
    }

    /// **The codebook is exactly the 32 three-of-four ternary patterns**, and the sign half mirrors
    /// the other by negation.
    ///
    /// ⚠ Unlike the round trip above, this one restates a property rather than exercising a path,
    /// so it cannot catch a mistyped digit that happens to land on another LEGAL pattern — the
    /// distinctness clause is what closes that, and `every_legal_block_round_trips_byte_for_byte`
    /// closes the rest.
    #[kani::proof]
    fn codebook_is_the_thirty_two_patterns() {
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < 32 && j < 32);
        let lanes = |k: usize| -> [i32; 4] {
            let cw = STQ1_0_CODEBOOK[k];
            let mut l = [0i32; 4];
            let mut p = 0;
            while p < 4 { l[p] = (((cw >> (2 * p)) & 3) as i32) - 1; p += 1 }
            l
        };
        let li = lanes(i);
        let mut zeros = 0;
        let mut p = 0;
        while p < 4 {
            assert!(li[p] >= -1 && li[p] <= 1, "lane code 3 is not a value");
            if li[p] == 0 { zeros += 1 }
            p += 1;
        }
        assert!(zeros == 1, "entry {i} does not have exactly one forced zero");
        if i != j { assert!(STQ1_0_CODEBOOK[i] != STQ1_0_CODEBOOK[j], "entries {i} and {j} collide") }
        if i < 16 {
            let lm = lanes(i + 16);
            let mut p = 0;
            while p < 4 { assert!(lm[p] == -li[p], "sign=1 is not the negation of sign=0 at slot {i}"); p += 1 }
        }
    }

    /// **`ksigns` is an injective even-parity code**, which is why it is computed rather than
    /// tabled. Both IQ decoders depend on it; if it is wrong, every eighth weight flips sign.
    #[kani::proof]
    fn ksigns_is_an_injective_even_parity_code() {
        let i: u8 = kani::any();
        let j: u8 = kani::any();
        kani::assume(i < 128 && j < 128);
        let vi = crate::ksigns_for_proof(i);
        assert!(vi & 0x7f == i, "the low seven bits must be the index itself");
        assert!(vi.count_ones() % 2 == 0, "population count must be even");
        if i != j { assert!(vi != crate::ksigns_for_proof(j), "the code is not injective") }
    }
}
