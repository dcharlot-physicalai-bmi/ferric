//! **IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_S and NVFP4** — the GGUF types this crate used to refuse.
//!
//! Until these existed, `type_size` answered `unsupported ggml type` for all six, so a file holding any
//! of them failed at [`crate::GgufFile::raw`] before a single weight was read. NVFP4 was the sharpest
//! case: `ferric-tensor` already had a packed NVFP4 matmul, verified against ggml over the full
//! UE4M3 x E2M1 grid, and no NVFP4 GGUF could reach it — the kernel sat behind a stride table that did
//! not list type 40.
//!
//! ## The reference is the authors' code
//!
//! llama.cpp DEFINES these formats — it is not a peer implementation here, it is the authors'. Every
//! function below is a line-by-line transliteration of the matching `dequantize_row_*` in
//! `ggml-quants.c`, kept in the C's evaluation order: each product is formed left to right exactly as
//! the C writes it (`db * grid[j] * sign`, `dl * (grid[j] + delta)`), so where a product could round,
//! it rounds the way ggml's does. The gate is bit-equality with the shipped `libggml-base` — see the
//! tests, whose fixtures were decoded by `ggml_get_type_traits(t)->to_float` itself.
//!
//! Every block is 256 values (`QK_K`) except NVFP4's 64.

use crate::iq_grids2::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID};
use crate::{ksigns, rd_f16};

/// Bytes per 256-value block, from the `static_assert`s in `ggml-common.h`.
pub const IQ2_XS_BYTES: usize = 74; // d(2) + qs u16[32] + scales[8]
pub const IQ2_S_BYTES: usize = 82; // d(2) + qs[64] (32 index + 32 sign bytes) + qh[8] + scales[8]
pub const IQ3_S_BYTES: usize = 110; // d(2) + qs[64] + qh[8] + signs[32] + scales[4]
pub const IQ1_S_BYTES: usize = 50; // d(2) + qs[32] + qh u16[8]
pub const IQ1_M_BYTES: usize = 56; // qs[32] + qh[16] + scales[8] — NO f16 d: it is spread over the scales
/// NVFP4: `d[4]` (UE4M3, one per 16 values) + `qs[32]` (E2M1 pairs) per 64 values = 4.5 bpw.
pub const NVFP4_BYTES: usize = 36;

/// `IQ1S_DELTA` / `IQ1M_DELTA`: the half-step every IQ1 grid value is shifted by, sign chosen per group.
const IQ1_DELTA: f32 = 0.125;

#[inline]
fn sgn(bits: u8, j: usize) -> f32 { if bits & (1u8 << j) != 0 { -1.0 } else { 1.0 } }

/// **IQ2_XS** — 2.3125 bpw. Sixteen-bit `qs` words: the low 9 bits index a 512-entry grid of eight
/// magnitudes, the high 7 a sign pattern (the eighth sign is parity — [`ksigns`]). Each 32-value group
/// has TWO 4-bit sub-scales, one per 16 values: `db[l/2]` — the high nibble governs the second half.
pub(crate) fn deq_iq2_xs(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(IQ2_XS_BYTES).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qs = |i: usize| u16::from_le_bytes([blk[2 + 2 * i], blk[3 + 2 * i]]);
        let scales = &blk[66..74];
        for ib32 in 0..8 {
            let db = [d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
                      d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25];
            for l in 0..4 {
                let q = qs(4 * ib32 + l);
                let g = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                let signs = ksigns((q >> 9) as u8);
                let y = bi * 256 + ib32 * 32 + l * 8;
                for j in 0..8 { out[y + j] = db[l / 2] * g[j] as f32 * sgn(signs, j); }
            }
        }
    }
    out
}

/// **IQ2_S** — 2.5625 bpw. A 1024-entry grid, so each index needs 10 bits: 8 from `qs[l]` and 2 from
/// `qh[ib32]` (`(qh << (8 - 2l)) & 0x300`). The signs are NOT a parity-coded index here: they are 32
/// plain bytes stored AFTER the 32 index bytes, one full byte of signs per 8 values.
///
/// ⚠ `qs` is 64 bytes but only the first 32 are grid indices; `signs = qs + QK_K/8`. Reading the
/// 64 bytes as interleaved (index, sign) pairs keeps every size right and scrambles every group.
pub(crate) fn deq_iq2_s(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(IQ2_S_BYTES).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qs = &blk[2..34];
        let signs = &blk[34..66];
        let qh = &blk[66..74];
        let scales = &blk[74..82];
        for ib32 in 0..8 {
            let db = [d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
                      d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25];
            for l in 0..4 {
                let dl = db[l / 2];
                let idx = qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let g = IQ2S_GRID[idx].to_le_bytes();
                let s = signs[4 * ib32 + l];
                let y = bi * 256 + ib32 * 32 + l * 8;
                for j in 0..8 { out[y + j] = dl * g[j] as f32 * sgn(s, j); }
            }
        }
    }
    out
}

/// **IQ3_S** — 3.4375 bpw. A 512-entry grid of FOUR magnitudes (9-bit index: `qs` byte + one bit of
/// `qh`), two lookups per 8 values sharing one sign byte (bits 0..3 and 4..7). Scales are 4-bit per
/// 32 values, applied as `d * (1 + 2*s)` — odd integers, where the IQ2 types use `(0.5 + s) * 0.25`.
///
/// The loop walks 64 values at a time: `qh[0]` covers the first 32, `qh[1]` the second, and the two
/// halves take `db1`/`db2` from the low/high nibble of one scale byte.
pub(crate) fn deq_iq3_s(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(IQ3_S_BYTES).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qs_all = &blk[2..66];
        let qh_all = &blk[66..74];
        let signs_all = &blk[74..106];
        let scales = &blk[106..110];
        let mut y = bi * 256;
        for ib32 in (0..8).step_by(2) {
            let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf) as i32) as f32;
            let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4) as i32) as f32;
            for (half, db) in [db1, db2].into_iter().enumerate() {
                let qs = &qs_all[(ib32 + half) * 8..];
                let signs = &signs_all[(ib32 + half) * 4..];
                let qh = qh_all[ib32 + half] as usize;
                for l in 0..4 {
                    let g1 = IQ3S_GRID[qs[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)].to_le_bytes();
                    let g2 = IQ3S_GRID[qs[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)].to_le_bytes();
                    for j in 0..4 {
                        out[y + j] = db * g1[j] as f32 * sgn(signs[l], j);
                        out[y + j + 4] = db * g2[j] as f32 * sgn(signs[l], j + 4);
                    }
                    y += 8;
                }
            }
        }
    }
    out
}

/// **IQ1_S** — 1.5625 bpw. An 11-bit index (8 from `qs`, 3 from the `u16 qh`) into a 2048-entry grid of
/// eight values in `{-1, 0, +1}`, SHIFTED by ±`IQ1S_DELTA`: `y = dl * (grid + delta)`. The shift is what
/// makes the effective levels `{-1.125, -0.125, 0.875}` or their mirror — ternary with a per-group bias.
/// `qh` packs, per 32 values: four 3-bit index extensions (bits 0..11), a 3-bit scale (12..14) and the
/// delta's sign (bit 15).
pub(crate) fn deq_iq1_s(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(IQ1_S_BYTES).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qs = &blk[2..34];
        for ib in 0..8 {
            let qh = u16::from_le_bytes([blk[34 + 2 * ib], blk[35 + 2 * ib]]) as usize;
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 { -IQ1_DELTA } else { IQ1_DELTA };
            for l in 0..4 {
                let g = IQ1S_GRID[qs[4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8)].to_le_bytes();
                let y = bi * 256 + ib * 32 + l * 8;
                for j in 0..8 { out[y + j] = dl * (g[j] as i8 as f32 + delta); }
            }
        }
    }
    out
}

/// **IQ1_M** — 1.75 bpw. The same grid as IQ1_S, but with NO block-level `f16` field: the super-block
/// scale is assembled from the TOP NIBBLE of each of the four `u16` scale words,
/// `(sc0 >> 12) | (sc1 >> 8 & 0xf0) | (sc2 >> 4 & 0xf00) | (sc3 & 0xf000)`, and reinterpreted as an
/// f16. The rest of each scale word is 3-bit sub-scales, one per 16 values. And each 8-value group
/// has its OWN delta sign (bit 3 / bit 7 of a `qh` byte), where IQ1_S shares one per 32.
///
/// ⚠ Reading a leading f16 `d` as IQ1_S does decodes the first two grid-index bytes as a scale — a
/// finite, plausible number — and shifts everything after it by two bytes.
pub(crate) fn deq_iq1_m(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(IQ1_M_BYTES).take(n / 256).enumerate() {
        let qs_all = &blk[0..32];
        let qh_all = &blk[32..48];
        let sc: [u16; 4] = std::array::from_fn(|k| u16::from_le_bytes([blk[48 + 2 * k], blk[49 + 2 * k]]));
        let u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = half::f16::from_bits(u).to_f32();
        for ib in 0..8 {
            let s = sc[ib / 2] as usize;
            let dl1 = d * (2 * ((s >> (6 * (ib % 2))) & 0x7) + 1) as f32;
            let dl2 = d * (2 * ((s >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
            let qs = &qs_all[4 * ib..];
            let (h0, h1) = (qh_all[2 * ib] as usize, qh_all[2 * ib + 1] as usize);
            let idx = [qs[0] as usize | ((h0 << 8) & 0x700), qs[1] as usize | ((h0 << 4) & 0x700),
                       qs[2] as usize | ((h1 << 8) & 0x700), qs[3] as usize | ((h1 << 4) & 0x700)];
            let delta = [if h0 & 0x08 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
                         if h0 & 0x80 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
                         if h1 & 0x08 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
                         if h1 & 0x80 != 0 { -IQ1_DELTA } else { IQ1_DELTA }];
            for l in 0..4 {
                let g = IQ1S_GRID[idx[l]].to_le_bytes();
                let dl = if l < 2 { dl1 } else { dl2 };
                let y = bi * 256 + ib * 32 + l * 8;
                for j in 0..8 { out[y + j] = dl * (g[j] as i8 as f32 + delta[l]); }
            }
        }
    }
    out
}

/// `ggml_ue4m3_to_fp32` — an UNSIGNED E4M3 scale (bias 7), halved. The `* 0.5` reconciles a scale that
/// was quantised against E2M1's true maximum (6.0) with a lookup table that stores E2M1 DOUBLED
/// (`kvalues_mxfp4` = `{0,1,2,3,4,6,8,12}`); dropping it doubles every weight. `0x7F` is the NaN slot
/// and decodes to 0, as does 0. Bit 7 is not a sign: ggml masks the exponent with `(x >> 3) & 0xF`.
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F { return 0.0; }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    raw * 0.5
}

/// E2M1 doubled — ggml's `kvalues_mxfp4`, shared by MXFP4 and NVFP4.
const KVALUES_E2M1_2X: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// **NVFP4** (ggml type 40) — 64 values: four 16-value sub-blocks, each with its own UE4M3 scale.
/// Within sub-block `s`, byte `qs[8s + j]` holds element `j` in its LOW nibble and `j + 8` in its HIGH
/// nibble — halves 8 apart, not 32 as in MXFP4's one-scale block. `y = kvalue * d`, as
/// `dequantize_row_nvfp4` forms it.
pub(crate) fn deq_nvfp4(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(NVFP4_BYTES).take(n / 64).enumerate() {
        for s in 0..4 {
            let d = ue4m3_to_f32(blk[s]);
            let y = bi * 64 + s * 16;
            for j in 0..8 {
                let b = blk[4 + s * 8 + j];
                out[y + j] = KVALUES_E2M1_2X[(b & 0x0F) as usize] as f32 * d;
                out[y + j + 8] = KVALUES_E2M1_2X[(b >> 4) as usize] as f32 * d;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deq_raw;

    /// A committed fixture: `<name>.raw` is real block bytes, `<name>.f32` what the shipped
    /// `libggml-base` decoded them to (`ggml_get_type_traits(t)->to_float`, via ctypes — see
    /// `crates/ferric-llama/examples/refgen/ggml_blocks_ref.py`).
    fn fixture(name: &str) -> Option<(Vec<u8>, Vec<f32>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ggml_quants");
        let raw = std::fs::read(dir.join(format!("{name}.raw"))).ok()?;
        let f = std::fs::read(dir.join(format!("{name}.f32"))).ok()?;
        Some((raw, f.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
    }

    /// Every fixture — real published blocks AND the crafted every-grid-entry sweeps — bit-equal to
    /// ggml's own decode. `to_bits` equality, not a tolerance: the functions above are transliterations
    /// in the C's evaluation order, so anything short of identity is a defect.
    fn check(ty: u32, name: &str) {
        let (raw, want) = fixture(name).unwrap_or_else(|| panic!(
            "fixture {name} missing — this test must not pass by finding nothing to compare"));
        let got = deq_raw(&raw, want.len(), ty).expect("decode");
        assert_eq!(got.len(), want.len());
        let bad: Vec<usize> = (0..want.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).collect();
        assert!(bad.is_empty(), "{name}: {} of {} values differ from libggml-base; first at {}: ours {} vs ggml {}",
                bad.len(), want.len(), bad[0], got[bad[0]], want[bad[0]]);
        // A decoder returning zeros matches a fixture of zeros; the fixtures are built not to be one.
        let nz = want.iter().filter(|v| **v != 0.0).count();
        assert!(nz * 2 > want.len(), "{name}: only {nz} of {} reference values are non-zero", want.len());
    }

    #[test] fn iq2_xs_is_bit_identical_to_libggml() { check(17, "iq2_xs_real"); check(17, "iq2_xs_grid"); }
    #[test] fn iq2_s_is_bit_identical_to_libggml() { check(22, "iq2_s_real"); check(22, "iq2_s_grid"); }
    #[test] fn iq3_s_is_bit_identical_to_libggml() { check(21, "iq3_s_real"); check(21, "iq3_s_grid"); }
    #[test] fn iq1_s_is_bit_identical_to_libggml() { check(19, "iq1_s_real"); check(19, "iq1_s_grid"); }
    #[test] fn iq1_m_is_bit_identical_to_libggml() { check(29, "iq1_m_real"); check(29, "iq1_m_grid"); }
    #[test] fn nvfp4_is_bit_identical_to_libggml() { check(40, "nvfp4_real"); check(40, "nvfp4_scales"); }

    /// The structural invariants of the transcribed grids — a mistyped hex digit almost surely lands
    /// outside its table's alphabet. Independent of the fixtures (which catch a wrong-but-in-alphabet
    /// entry), and cheap enough to run everywhere.
    #[test]
    fn grids_stay_inside_their_alphabets() {
        let in_set = |v: u8, s: &[u8]| s.contains(&v);
        for &g in IQ2XS_GRID.iter().chain(IQ2S_GRID.iter()) {
            assert!(g.to_le_bytes().iter().all(|&b| in_set(b, &[8, 25, 43])), "IQ2 grid entry {g:#018x}");
        }
        for &g in IQ3S_GRID.iter() {
            assert!(g.to_le_bytes().iter().all(|&b| b % 2 == 1 && b <= 15), "IQ3_S grid entry {g:#010x}");
        }
        for &g in IQ1S_GRID.iter() {
            assert!(g.to_le_bytes().iter().all(|&b| matches!(b as i8, -1..=1)), "IQ1_S grid entry {g:#018x}");
        }
        // Distinct entries: a duplicated row is the other way a transcription goes wrong.
        let mut v: Vec<u64> = IQ1S_GRID.to_vec(); v.sort(); v.dedup();
        assert_eq!(v.len(), 2048, "IQ1_S grid has duplicate entries");
        let mut v: Vec<u64> = IQ2S_GRID.to_vec(); v.sort(); v.dedup();
        assert_eq!(v.len(), 1024, "IQ2_S grid has duplicate entries");
    }

    #[test]
    fn strides_match_ggml_type_traits() {
        // (type, block values, block bytes) as `ggml_get_type_traits` reports them in libggml-base 0.25.3.
        for (ty, blk, bytes) in [(17, 256, 74), (22, 256, 82), (21, 256, 110), (19, 256, 50), (29, 256, 56), (40, 64, 36)] {
            assert_eq!(crate::block_elems(ty), blk, "type {ty} block");
            assert_eq!(crate::type_size(ty, blk * 3).unwrap(), bytes * 3, "type {ty} stride");
        }
    }
}
