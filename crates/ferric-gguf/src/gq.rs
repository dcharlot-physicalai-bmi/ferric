//! **GQ — Ferric's in-memory form for grouped-quant safetensors checkpoints** (GPTQ, AWQ,
//! compressed-tensors, FP8, ModelOpt NVFP4).
//!
//! Those checkpoints are not GGUF and must not be converted into GGUF types: a GPTQ int4 weight is
//! `s * (q - z)` with an arbitrary integer zero, an AWQ one the same with a different nibble order, an
//! FP8 one `s * e4m3(q)` with an f32 scale per 128x128 block. No ggml type means any of these exactly
//! (Q4_1's `d*q + m` would round `-s*z` into an f16; Q8_0 needs one f16 scale per 32 values and two
//! bytes per weight). So `ferric-load` re-lays each one — losslessly, bits unchanged — into this ONE
//! parametric form, and `ferric-tensor::gq` runs it packed on the GPU.
//!
//! ## The type id carries the whole geometry
//!
//! A GQ tensor presents a synthetic ggml type id `0x47_K_A_B_GGG` — see [`GqSpec::id`]. Encoding the
//! bit width, group size, code kind and act-order flag IN the id is deliberate: loaders fuse
//! same-typed tensors by concatenating their bytes (`qm_cat`), so two tensors of different geometry
//! must never share an id — and with the geometry in the id, they cannot.
//!
//! ## Byte layout (per row = output channel), block-interleaved per group of `G` input values
//!
//! ```text
//! block b (b = 0 .. cols/G):  codes[G*bits/8]   bit-packed little-endian, value c at bit c*bits
//!                             scale  f32        of GROUP b
//!                             zero   f32        of group b            (integer kind only)
//!                             g_idx  u16[G]     group of each value   (act-order only)
//! ```
//!
//! Without act-order, value `c` belongs to group `c / G`, so each block is self-contained and
//! [`deq_gq_blocks`] needs no row geometry. WITH act-order (GPTQ `desc_act`, compressed-tensors
//! `actorder: group`) value `c` belongs to group `g_idx[c]`, which can be ANY group of the row — so
//! decoding needs the row, and [`deq_gq_rows`] takes `cols`.
//!
//! The weight is formed exactly as the defining libraries form it, in f32:
//! integer `s * (q - z)`, FP8 `s * e4m3(q)`, FP4 `s * e2m1(q)` — one rounding at most (none for the
//! integer kinds, whose products are exact in f32).

/// What a code decodes to before the scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GqKind {
    /// Unsigned integer `q`, weight `s * (q - z)`.
    Int = 0,
    /// FP8 E4M3 (the `FN` variant: no infinities), weight `s * e4m3(q)`.
    Fp8E4m3 = 1,
    /// FP4 E2M1, weight `s * e2m1(q)`.
    Fp4E2m1 = 2,
}

/// The geometry of one GQ tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GqSpec {
    pub kind: GqKind,
    /// Bits per code: 2, 3, 4 or 8 (FP8: 8, FP4: 4).
    pub bits: u32,
    /// Values per group (and per stored block). `G * bits` must be a multiple of 32.
    pub group: u32,
    /// Per-value group indices are stored (act-order).
    pub act: bool,
}

const MAGIC: u32 = 0x47;

impl GqSpec {
    /// `0x47 << 24 | kind << 20 | act << 16 | bits << 12 | group`, group < 4096.
    pub fn id(&self) -> u32 {
        (MAGIC << 24) | ((self.kind as u32) << 20) | ((self.act as u32) << 16) | (self.bits << 12) | self.group
    }
    pub fn from_id(id: u32) -> Option<GqSpec> {
        if id >> 24 != MAGIC { return None }
        let kind = match (id >> 20) & 0xf { 0 => GqKind::Int, 1 => GqKind::Fp8E4m3, 2 => GqKind::Fp4E2m1, _ => return None };
        let s = GqSpec { kind, act: (id >> 16) & 1 == 1, bits: (id >> 12) & 0xf, group: id & 0xfff };
        s.valid().ok().map(|_| s)
    }
    pub fn valid(&self) -> Result<(), String> {
        let ok_bits = match self.kind { GqKind::Int => matches!(self.bits, 2 | 3 | 4 | 8), GqKind::Fp8E4m3 => self.bits == 8, GqKind::Fp4E2m1 => self.bits == 4 };
        if !ok_bits { return Err(format!("GQ {:?} cannot have {} bits", self.kind, self.bits)) }
        if self.group == 0 || self.group >= 4096 || (self.group * self.bits) % 32 != 0 {
            return Err(format!("GQ group {} x {} bits is not a whole number of 32-bit words (or >= 4096)", self.group, self.bits));
        }
        Ok(())
    }
    pub fn has_zero(&self) -> bool { self.kind == GqKind::Int }
    /// Bytes of one stored block of `group` values.
    pub fn block_bytes(&self) -> usize {
        let g = self.group as usize;
        g * self.bits as usize / 8 + 4 + if self.has_zero() { 4 } else { 0 } + if self.act { 2 * g } else { 0 }
    }
}

pub fn is_gq(id: u32) -> bool { GqSpec::from_id(id).is_some() }

/// `float8_e4m3fn` — same decode as `ferric_load::fp8::e4m3_to_f32`, repeated here because this crate
/// sits below that one. Exhaustively cross-checked in `ferric-load`'s tests.
pub fn e4m3(b: u8) -> f32 {
    let s = (b >> 7) as u32;
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as u32;
    if e == 0x0F && m == 0x07 { return f32::NAN; }
    if e == 0 { let v = m as f32 / 8.0 * (1.0 / 64.0); return if s == 1 { -v } else { v }; }
    f32::from_bits((s << 31) | (((e - 7 + 127) as u32) << 23) | (m << 20))
}

/// FP4 E2M1: sign, 2 exponent bits (bias 1), 1 mantissa bit — `{0, .5, 1, 1.5, 2, 3, 4, 6}` and negatives.
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// The `i`-th `bits`-wide code of a little-endian bitstream.
#[inline]
pub fn code_at(codes: &[u8], i: usize, bits: u32) -> u32 {
    let bp = i * bits as usize;
    let mut v = 0u32;
    for k in 0..bits as usize {
        let b = bp + k;
        v |= (((codes[b >> 3] >> (b & 7)) & 1) as u32) << k;
    }
    v
}

#[inline]
fn weight(spec: &GqSpec, q: u32, s: f32, z: f32) -> f32 {
    match spec.kind {
        GqKind::Int => s * (q as f32 - z),
        GqKind::Fp8E4m3 => s * e4m3(q as u8),
        GqKind::Fp4E2m1 => s * E2M1[q as usize],
    }
}

/// Decode `n` values of a NON-act-order GQ tensor (each block self-contained).
pub fn deq_gq_blocks(raw: &[u8], n: usize, spec: &GqSpec) -> Result<Vec<f32>, String> {
    if spec.act {
        return Err("act-order GQ needs its row geometry (a value's group can be anywhere in the row): \
                    use gq::deq_gq_rows".into());
    }
    let (g, bb) = (spec.group as usize, spec.block_bytes());
    let cb = g * spec.bits as usize / 8;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(bb).take(n / g) {
        let s = f32::from_le_bytes(blk[cb..cb + 4].try_into().unwrap());
        let z = if spec.has_zero() { f32::from_le_bytes(blk[cb + 4..cb + 8].try_into().unwrap()) } else { 0.0 };
        for i in 0..g { out.push(weight(spec, code_at(&blk[..cb], i, spec.bits), s, z)); }
    }
    Ok(out)
}

/// Decode a whole `[rows, cols]` GQ tensor, act-order or not.
pub fn deq_gq_rows(raw: &[u8], rows: usize, cols: usize, spec: &GqSpec) -> Result<Vec<f32>, String> {
    let (g, bb) = (spec.group as usize, spec.block_bytes());
    if cols % g != 0 { return Err(format!("GQ: {cols} columns is not a whole number of {g}-value groups")) }
    let nb = cols / g;
    if raw.len() != rows * nb * bb { return Err(format!("GQ: {} bytes for [{rows}, {cols}] ({} expected)", raw.len(), rows * nb * bb)) }
    if !spec.act { return deq_gq_blocks(raw, rows * cols, spec) }
    let cb = g * spec.bits as usize / 8;
    let zo = cb + 4 + if spec.has_zero() { 4 } else { 0 };
    let mut out = Vec::with_capacity(rows * cols);
    for row in raw.chunks_exact(nb * bb) {
        let sz: Vec<(f32, f32)> = row.chunks_exact(bb).map(|b| (
            f32::from_le_bytes(b[cb..cb + 4].try_into().unwrap()),
            if spec.has_zero() { f32::from_le_bytes(b[cb + 4..cb + 8].try_into().unwrap()) } else { 0.0 })).collect();
        for blk in row.chunks_exact(bb) {
            for i in 0..g {
                let gi = u16::from_le_bytes([blk[zo + 2 * i], blk[zo + 2 * i + 1]]) as usize;
                let (s, z) = *sz.get(gi).ok_or_else(|| format!("GQ: g_idx {gi} past the row's {nb} groups"))?;
                out.push(weight(spec, code_at(&blk[..cb], i, spec.bits), s, z));
            }
        }
    }
    Ok(out)
}

/// Encode one row. `codes[c]` is value `c`'s code, `scales`/`zeros` are per GROUP (len cols/G),
/// `g_idx` per value (act-order only). The inverse of [`deq_gq_rows`] for one row.
pub fn encode_row(spec: &GqSpec, codes: &[u32], scales: &[f32], zeros: &[f32], g_idx: Option<&[u16]>, out: &mut Vec<u8>) {
    let g = spec.group as usize;
    let cb = g * spec.bits as usize / 8;
    let nb = codes.len() / g;
    for b in 0..nb {
        let start = out.len();
        out.resize(start + cb, 0);
        for i in 0..g {
            let v = codes[b * g + i];
            let bp = i * spec.bits as usize;
            for k in 0..spec.bits as usize {
                if (v >> k) & 1 == 1 { out[start + ((bp + k) >> 3)] |= 1 << ((bp + k) & 7); }
            }
        }
        out.extend_from_slice(&scales[b].to_le_bytes());
        if spec.has_zero() { out.extend_from_slice(&zeros[b].to_le_bytes()); }
        if spec.act {
            let gi = g_idx.expect("act-order row needs g_idx");
            for i in 0..g { out.extend_from_slice(&gi[b * g + i].to_le_bytes()); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_never_collide_with_ggml() {
        for kind in [GqKind::Int, GqKind::Fp8E4m3, GqKind::Fp4E2m1] {
            for bits in [2, 3, 4, 8] { for group in [16, 32, 64, 128, 256] { for act in [false, true] {
                let s = GqSpec { kind, bits, group, act };
                if s.valid().is_err() { assert!(GqSpec::from_id(s.id()).is_none()); continue }
                assert_eq!(GqSpec::from_id(s.id()), Some(s));
                assert!(crate::type_name(s.id()).is_none(), "GQ id {:#x} collides with a ggml type", s.id());
            }}}
        }
        assert!(!is_gq(40) && !is_gq(1042) && !is_gq(0));
    }

    /// 3-bit codes straddle word boundaries (values 10 and 21 of every 32). The encoder and decoder
    /// must agree on EVERY position, including those two, and act-order groups must be looked up.
    #[test]
    fn encode_then_decode_is_exact_for_every_width_and_act_order() {
        for bits in [2u32, 3, 4, 8] {
            for act in [false, true] {
                let spec = GqSpec { kind: GqKind::Int, bits, group: 32, act };
                let (rows, cols) = (3, 128);
                let mut raw = Vec::new();
                let mut want = Vec::new();
                for r in 0..rows {
                    let codes: Vec<u32> = (0..cols).map(|c| ((c * 7 + r * 3) as u32) & ((1 << bits) - 1)).collect();
                    let scales: Vec<f32> = (0..cols / 32).map(|b| 0.01 * (b + 1 + r) as f32).collect();
                    let zeros: Vec<f32> = (0..cols / 32).map(|b| ((b + r) % 3) as f32).collect();
                    // act-order: a permutation-like assignment that crosses blocks
                    let gi: Vec<u16> = (0..cols).map(|c| ((c * 5 + r) % (cols / 32)) as u16).collect();
                    encode_row(&spec, &codes, &scales, &zeros, act.then_some(&gi[..]), &mut raw);
                    for c in 0..cols {
                        let g = if act { gi[c] as usize } else { c / 32 };
                        want.push(scales[g] * (codes[c] as f32 - zeros[g]));
                    }
                }
                assert_eq!(raw.len(), rows * cols / 32 * spec.block_bytes());
                let got = deq_gq_rows(&raw, rows, cols, &spec).unwrap();
                assert_eq!(got, want, "bits {bits} act {act}");
            }
        }
    }
}
