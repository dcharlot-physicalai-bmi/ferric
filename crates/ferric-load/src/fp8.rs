//! **FP8 and the integer dtypes** — what 2026 open-weight checkpoints are actually stored in.
//!
//! A safetensors reader that handles F32/F16/BF16 and nothing else can open a 2023 checkpoint. The
//! releases that matter now ship `F8_E4M3` weights beside `_scale_inv` tensors, or integer-packed
//! GPTQ/AWQ codes beside their zero-points, and a reader that hard-errors on those dtypes cannot
//! open the file at all — which is the difference between reading the ecosystem's native format and
//! reading a converted copy of it.
//!
//! ⚠ **AN FP8 TENSOR IS NOT THE WEIGHT.** The stored byte is a coefficient; the weight is that byte
//! times a scale held in a SEPARATE tensor, usually `<name>_scale_inv` (block-wise, DeepSeek's
//! 128×128 convention) or `<name>_scale` (per-tensor or per-channel, compressed-tensors). Decoding
//! the bytes and handing them back is not "loading an FP8 model" — it silently returns weights off
//! by whatever the scale was, with correct shapes and no error anywhere. [`Dtype::is_scaled`] exists
//! so a loader cannot forget, and `SafeTensors::get` refuses rather than guessing.

/// **E4M3** — 1 sign, 4 exponent (bias 7), 3 mantissa. The `FN` variant every ML framework uses:
/// no infinities, and `0x7F`/`0xFF` are the only NaNs, which buys one extra binade — max normal is
/// 448, not 240.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) as u32;
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as u32;
    if e == 0x0F && m == 0x07 { return f32::NAN; }
    if e == 0 {
        // Subnormal: no implicit leading 1, fixed exponent 2^-6.
        let v = m as f32 / 8.0 * (1.0 / 64.0);
        return if s == 1 { -v } else { v };
    }
    f32::from_bits((s << 31) | (((e - 7 + 127) as u32) << 23) | (m << 20))
}

/// **E5M2** — 1 sign, 5 exponent (bias 15), 2 mantissa. Same exponent field and bias as IEEE f16,
/// so it keeps infinities and the f16 NaN patterns, and trades mantissa for range: max normal 57344.
pub fn e5m2_to_f32(b: u8) -> f32 {
    let s = (b >> 7) as u32;
    let e = ((b >> 2) & 0x1F) as i32;
    let m = (b & 0x03) as u32;
    if e == 0x1F {
        return if m == 0 { if s == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN };
    }
    if e == 0 {
        let v = m as f32 / 4.0 * (1.0 / 16384.0);
        return if s == 1 { -v } else { v };
    }
    f32::from_bits((s << 31) | (((e - 15 + 127) as u32) << 23) | (m << 21))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ E5M2 has f16's exponent width AND f16's bias — only the mantissa is shorter — so every one
    /// of the 256 bytes is EXACTLY an f16 with its low 8 bits cleared. That makes `half`'s f16, an
    /// implementation this crate did not write and does not control, a complete independent oracle
    /// over the entire domain. No sampling, no tolerance, no recollection of a spec.
    #[test]
    fn every_e5m2_byte_equals_the_f16_it_is_the_top_half_of() {
        for b in 0..=255u8 {
            let mine = e5m2_to_f32(b);
            let theirs = half::f16::from_bits((b as u16) << 8).to_f32();
            if theirs.is_nan() { assert!(mine.is_nan(), "0x{b:02x}: expected NaN, got {mine}"); continue }
            assert_eq!(mine.to_bits(), theirs.to_bits(),
                       "0x{b:02x}: {mine} != {theirs} (f16 0x{:04x})", (b as u16) << 8);
        }
    }

    /// E4M3 has no such twin, so the oracle is the FORMAT DEFINITION evaluated in float arithmetic —
    /// an independent implementation of the same rule, sharing no code path with the bit-shifting
    /// one above. Exhaustive over all 256 bytes.
    #[test]
    fn every_e4m3_byte_matches_the_arithmetic_definition() {
        for b in 0..=255u8 {
            let (s, e, m) = ((b >> 7) as i32, ((b >> 3) & 0x0F) as i32, (b & 0x07) as i32);
            let got = e4m3_to_f32(b);
            if e == 0x0F && m == 0x07 { assert!(got.is_nan(), "0x{b:02x} is E4M3's NaN"); continue }
            let mag = if e == 0 { 2f32.powi(-6) * (m as f32 / 8.0) }
                      else { 2f32.powi(e - 7) * (1.0 + m as f32 / 8.0) };
            let want = if s == 1 { -mag } else { mag };
            assert_eq!(got.to_bits(), want.to_bits(), "0x{b:02x}: {got} != {want}");
        }
    }

    /// The published landmarks of both formats. The tests above would ALSO pass if this file and its
    /// oracle agreed on a wrong bias — they check internal agreement over the domain, not that the
    /// domain is the right one. These four numbers are the format's public identity.
    #[test]
    fn the_formats_hit_their_published_extremes() {
        assert_eq!(e4m3_to_f32(0x7E), 448.0, "E4M3FN max normal");
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9), "E4M3 min subnormal");
        assert_eq!(e5m2_to_f32(0x7B), 57344.0, "E5M2 max normal");
        assert_eq!(e5m2_to_f32(0x01), 2f32.powi(-16), "E5M2 min subnormal");
        assert_eq!(e4m3_to_f32(0x80), 0.0);
        assert!(e4m3_to_f32(0x80).is_sign_negative(), "E4M3 keeps signed zero");
    }
}
