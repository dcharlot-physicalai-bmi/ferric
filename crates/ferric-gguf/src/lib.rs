//! Pure-Rust reader for the llama.cpp **GGUF** container + dequantizers for the common block-quant
//! formats: F32, F16, Q8_0, the legacy Q4_0/Q4_1/Q5_0/Q5_1, the k-quants **Q4_K/Q5_K/Q6_K**, the
//! non-linear codebook quants **IQ4_NL/IQ4_XS**, OCP microscaling **MXFP4** (GPT-OSS's release
//! format), and BitNet-style ternary (TQ2_0 + PrismML Q1_0/Q2_0).
//! GGUF is how the entire llama.cpp / HF
//! quantized-model corpus ships — including Liquid AI's LFM2 and BitNet — so this is the ingest path
//! that lets Ferric run those models. Dequant here is CPU-side (I/O layer); a fused on-GPU dequant
//! matmul is the perf follow-up.

use half::f16;
use std::collections::HashMap;

// ---- ggml tensor type codes we handle ----
pub mod backed;
pub mod imatrix;
mod iq_grids;
pub use iq_grids::{IQ2XXS_GRID, IQ3XXS_GRID};
pub mod quantize;
pub mod quantplan;

const F32: u32 = 0;
const F16T: u32 = 1;
const BF16T: u32 = 30; // brain-float16: f32's top 16 bits (seen on qwen35moe's final-layer routers)
const Q4_0: u32 = 2;
const Q4_1: u32 = 3; // 4-bit affine: value = nibble·d + m (f16 d, f16 min per 32-block)
const Q5_0: u32 = 6; // 5-bit symmetric: value = ((nibble | 5th-bit) − 16)·d (f16 d, u32 qh per 32-block)
const Q5_1: u32 = 7; // 5-bit affine: value = (nibble | 5th-bit)·d + m (f16 d, f16 min, u32 qh per 32-block)
const Q8_0: u32 = 8;
const Q2_K: u32 = 10; // 2-bit K-quant: 16 sub-blocks of 16, 4-bit scale AND 4-bit min per sub-block
const Q3_K: u32 = 11; // 3-bit K-quant: 16 sub-blocks of 16, 6-bit scales packed across 12 bytes
const Q4_K: u32 = 12;
const Q5_K: u32 = 13;
const Q6_K: u32 = 14;
const IQ2_XXS: u32 = 16; // 2.0625 bpw: 8-element grid codebook + 7-bit sign index, 4-bit sub-scale
const IQ3_XXS: u32 = 18; // 3.0625 bpw: two 4-element grid lookups per 8, same sign/scale word
const IQ4_NL: u32 = 20; // 4-bit non-linear codebook, group-32 (kvalues_iq4nl)
const IQ4_XS: u32 = 23; // 4-bit non-linear codebook, 256-super-block w/ 6-bit sub-scales
const TQ2_0: u32 = 35; // llama.cpp ternary (BitNet) quant: 2 bits/weight, {−1,0,+1}·scale
const MXFP4: u32 = 39; // OCP Microscaling FP4: 32×E2M1 elements under one E8M0 shared exponent (GPT-OSS)
const Q1_0: u32 = 41; // PrismML/mainline 1-bit: {−1,+1}·scale, group-128 (1.125 bpw)
const Q2_0: u32 = 42; // PrismML ternary: {−1,0,+1}·scale, group-128 (2.125 bpw on disk)
const STQ1_0: u32 = 43; // Tencent hyv4 ternary: {−1,0,+1}·d with ONE FORCED ZERO per 4 lanes (1.3125 bpw)

#[derive(Debug, Clone)]
pub enum Meta { U(u64), I(i64), F(f64), Bool(bool), Str(String), Arr(Vec<Meta>) }

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct Gguf {
    pub metadata: HashMap<String, Meta>,
    pub tensors: Vec<TensorInfo>,
    data: Vec<u8>,
    data_start: usize,
}

/// Bounds-safe cursor: any read past the end sets `ok = false` and yields a zero value rather than
/// panicking. That makes `parse` total over arbitrary bytes — it rejects malformed files, and lets
/// `GgufFile` probe a truncated *prefix* to discover how large the header is.
struct Cur<'a> { b: &'a [u8], p: usize, ok: bool }
impl<'a> Cur<'a> {
    fn take(&mut self, k: usize) -> Option<&'a [u8]> {
        if !self.ok || self.p + k > self.b.len() { self.ok = false; return None; }
        let s = &self.b[self.p..self.p + k];
        self.p += k;
        Some(s)
    }
    fn u32(&mut self) -> u32 { self.take(4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap())) }
    fn u64(&mut self) -> u64 { self.take(8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap())) }
    fn u16(&mut self) -> u16 { self.take(2).map_or(0, |s| u16::from_le_bytes(s.try_into().unwrap())) }
    fn i64(&mut self) -> i64 { self.u64() as i64 }
    fn f32(&mut self) -> f32 { f32::from_bits(self.u32()) }
    fn f64(&mut self) -> f64 { f64::from_bits(self.u64()) }
    fn u8(&mut self) -> u8 { self.take(1).map_or(0, |s| s[0]) }
    fn str(&mut self) -> String {
        let n = self.u64() as usize;
        // Guard before allocating: a garbage length must not turn into a huge reservation.
        if n > self.b.len().saturating_sub(self.p) { self.ok = false; return String::new(); }
        self.take(n).map_or(String::new(), |s| String::from_utf8_lossy(s).into_owned())
    }
    fn val(&mut self, ty: u32) -> Meta {
        match ty {
            0 => Meta::U(self.u8() as u64),
            1 => Meta::I(self.u8() as i8 as i64),
            2 => Meta::U(self.u16() as u64),
            3 => Meta::I(self.u16() as i16 as i64),
            4 => Meta::U(self.u32() as u64),
            5 => Meta::I(self.u32() as i32 as i64),
            6 => Meta::F(self.f32() as f64),
            7 => Meta::Bool(self.u8() != 0),
            8 => Meta::Str(self.str()),
            9 => {
                let et = self.u32();
                let n = self.u64();
                // Each element costs ≥1 byte on disk, so a count beyond the remaining bytes is garbage.
                if n as usize > self.b.len().saturating_sub(self.p) { self.ok = false; return Meta::Arr(Vec::new()); }
                let mut v = Vec::new();
                for _ in 0..n { if !self.ok { break; } v.push(self.val(et)); }
                Meta::Arr(v)
            }
            10 => Meta::U(self.u64()),
            11 => Meta::I(self.i64()),
            12 => Meta::F(self.f64()),
            _ => { self.ok = false; Meta::U(0) }
        }
    }
}
pub fn parse(bytes: Vec<u8>) -> Result<Gguf, String> {
    let mut c = Cur { b: &bytes, p: 0, ok: true };
    if c.u32() != u32::from_le_bytes(*b"GGUF") { return Err("not a GGUF file".into()); }
    let _ver = c.u32();
    let n_tensors = c.u64();
    let n_meta = c.u64();
    let mut metadata = HashMap::new();
    for _ in 0..n_meta {
        let key = c.str();
        let ty = c.u32();
        metadata.insert(key, c.val(ty));
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = c.str();
        let nd = c.u32();
        let dims = (0..nd).map(|_| c.u64()).collect();
        let ggml_type = c.u32();
        let offset = c.u64();
        tensors.push(TensorInfo { name, dims, ggml_type, offset });
    }
    if !c.ok { return Err("GGUF header truncated or malformed".into()); }
    let align = match metadata.get("general.alignment") { Some(Meta::U(a)) => *a as usize, _ => 32 };
    let data_start = c.p.div_ceil(align) * align;
    check_declared_strides(&tensors, bytes.len().saturating_sub(data_start), align)?;
    resolve_ambiguous_types(&mut tensors, bytes.len().saturating_sub(data_start), align)?;
    Ok(Gguf { metadata, tensors, data: bytes, data_start })
}

/// **FP8 E4M3** (OCP `float8_e4m3fn`) -> f32, as a 256-entry bit-pattern table.
///
/// DeepSeek V4 Flash stores its dense weights as `F8_E4M3_B128`: 128 elements of this format under one
/// shared **E8M0** scale. The scale half already exists here and is bit-exact — it is the same
/// encoding MXFP4 uses, verified over the full 4096-pair grid. This is the element half.
///
/// A table rather than bit arithmetic because E4M3 is small enough to enumerate exhaustively, and
/// because the `fn` variant's edges are where a hand-rolled decoder goes wrong: there are **no
/// infinities**, `0x7F` and `0xFF` are the only NaNs, the maximum finite magnitude is **448.0**, and
/// subnormals run down to 2^-9 (0.001953125). Every entry below is the bit pattern PyTorch's
/// `float8_e4m3fn` produces for that byte, generated rather than typed.
///
/// The `fn` variant is the right reference: the V4 fork's element decoder
/// (`ggml_f8_e4m3fn_to_fp32`, nisparks/llama.cpp @ `9d36408`, `ggml-quants.c:552-566`) implements
/// exactly these semantics — `(x & 0x7F) == 0x7F` is NaN, no infinity branch, max finite `0x7E` =
/// 448 (which its quantizer also uses as the scale target). Element-order/value decode has NOT been
/// diffed against reference dequantized values from a real file, only against the fork's source.
pub const E4M3_TO_F32_BITS: [u32; 256] = [
    0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000,
    0x3c800000, 0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000,
    0x3d000000, 0x3d100000, 0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000,
    0x3d800000, 0x3d900000, 0x3da00000, 0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000,
    0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000, 0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000,
    0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000, 0x3ed00000, 0x3ee00000, 0x3ef00000,
    0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
    0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000, 0x3ff00000,
    0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
    0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000,
    0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
    0x41800000, 0x41900000, 0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000,
    0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
    0x42800000, 0x42900000, 0x42a00000, 0x42b00000, 0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000,
    0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
    0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000, 0x43e00000, 0x7ff00000,
    0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000, 0xbc600000,
    0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
    0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000,
    0xbd800000, 0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000,
    0xbe000000, 0xbe100000, 0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000,
    0xbe800000, 0xbe900000, 0xbea00000, 0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000,
    0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000, 0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000,
    0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000, 0xbfd00000, 0xbfe00000, 0xbff00000,
    0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000, 0xc0600000, 0xc0700000,
    0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000, 0xc0f00000,
    0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
    0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000,
    0xc2000000, 0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000,
    0xc2800000, 0xc2900000, 0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000,
    0xc3000000, 0xc3100000, 0xc3200000, 0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000,
    0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000, 0xc3c00000, 0xc3d00000, 0xc3e00000, 0xfff00000
];

/// One E4M3 byte as f32.
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 { f32::from_bits(E4M3_TO_F32_BITS[b as usize]) }

/// **ggml type 42 is claimed by THREE different formats**, distinguishable only by stride.
///
/// | claimant | values/block | bytes/block | bytes per 128 values |
/// |---|---|---|---|
/// | PrismML `Q2_0` — what this crate has always meant by 42 | 128 | 34 | 34 |
/// | ggml-org mainline `GGML_TYPE_Q2_0` (`block_q2_0`: f16 d + 16 code bytes) — **not decoded here** | 64 | 18 | 36 |
/// | `F8_E4M3_B128` — DeepSeek V4 Flash's dense weights | 128 | 129 | 129 |
///
/// The collision is **confirmed in the wild, both ways** (2026-08-19): a real V4 file
/// (`DeepSeek-V4-Flash-FP4-FP8-native.gguf`, 156,148,189,760 bytes, x-repo-commit `0b34e0b6`) carries
/// 365 tensors of id 42 at exactly 129 bytes per 128 elements, matching the nisparks/llama.cpp WIP
/// branch (`gguf-py constants.py:4115`, `ggml.h:432` @ `9d36408`: `GGML_TYPE_F8_E4M3_B128 = 42`);
/// while ggml-org master (`ggml.h:432` @ `b062ba7`) assigns 42 to its own `Q2_0` at 64 values /
/// 18 bytes. Mainline has NO F8 type at all — it dequantizes FP8 checkpoints at convert time.
/// Note the two `Q2_0`s ALSO differ from each other (34 vs 36 bytes per 128 values).
/// And never key on `general.file_type` either: nisparks ftype `MOSTLY_F8_E4M3_MXFP4` = 41 collides
/// with upstream ftype `MOSTLY_Q2_0` = 41.
///
/// A file carries only the id, so the claimants are distinguishable **only by how many bytes the
/// tensor actually occupies**. That is knowable at parse time: GGUF lays tensors out sequentially, so
/// the gap to the next tensor bounds each one. This resolves it there rather than letting `deq_raw`
/// guess.
///
/// Reading a V4 file as `Q2_0` would land at ~1/4 the correct stride and return plausible garbage
/// with no error — which is exactly the failure [`check_declared_strides`] was added to prevent,
/// written before this collision was known to be real.
pub fn resolve_type_42(n_elements: usize, declared_bytes: usize) -> Result<u32, String> {
    // Mainline ggml-org Q2_0 (64 values / 18 bytes) is the third claimant, and one this crate does
    // NOT decode. Its stride is only 2 bytes per 128 values away from PrismML's, so name it precisely
    // when it appears instead of folding it into the generic refusal below.
    if n_elements % 64 == 0 && declared_bytes == n_elements / 64 * 18 {
        return Err(format!(
            "ggml type 42 with {declared_bytes} bytes for {n_elements} elements matches mainline \
             ggml-org Q2_0 (block_q2_0: 64 values / 18 bytes, ggml.h:432 @ b062ba7), which this crate \
             does not decode. It is NOT PrismML Q2_0 (128 values / 34 bytes) and NOT F8_E4M3_B128 \
             (128 values / 129 bytes). Refusing rather than mis-decoding."));
    }
    if n_elements % 128 != 0 {
        return Err(format!("ggml type 42 needs a multiple of 128 elements, got {n_elements}"));
    }
    let blocks = n_elements / 128;
    match declared_bytes {
        b if b == blocks * 34 => Ok(Q2_0),
        b if b == blocks * F8_E4M3_B128_BYTES => Ok(F8_E4M3_B128),
        b => Err(format!(
            "ggml type 42 is ambiguous and this tensor matches none of the three claimants: \
             {n_elements} elements in {b} bytes is {:.3} bits/value, but PrismML Q2_0 is {} bytes \
             ({blocks} x 34), mainline ggml-org Q2_0 is {} bytes ({n_elements}/64 x 18, undecoded \
             here), and F8_E4M3_B128 is {} bytes ({blocks} x {F8_E4M3_B128_BYTES}). Refusing rather \
             than picking one.",
            b as f64 * 8.0 / n_elements as f64, blocks * 34, n_elements / 64 * 18,
            blocks * F8_E4M3_B128_BYTES)),
    }
}

/// E8M0 under the **OCP bias of 127**: `2^(e - 127)`, with `0xFF` reserved for NaN.
///
/// ⚠ **This is a DIFFERENT bias from [`e8m0_half_to_f32`], and the difference is a factor of two.**
/// That one returns `2^(e - 128)` because ggml pairs it with a *doubled* E2M1 value table — an
/// arithmetic choice ggml makes so that `e = 255` yields a representable `2^127` instead of
/// overflowing. E4M3 values are not doubled, so pairing them with the halved scale would make every
/// V4 weight exactly half its true magnitude: a uniform scaling that produces fluent, confidently
/// wrong output rather than an error.
///
/// **Bias-127 is what the V4 fork implements — verified at source level, 2026-08-19.**
/// `dequantize_row_f8_e4m3_b128` in nisparks/llama.cpp @ `9d36408` (`ggml-quants.c:649`) decodes the
/// scale with `GGML_E8M0_TO_FP32` (`ggml-impl.h:439-473`): `bits = e << 23`, i.e. `2^(e-127)`, with
/// `e = 0` special-cased to bits `0x00400000` = 2^-127 — exactly this function. The quantizer
/// round-trips through the same macro (`ggml-quants.c:634`), and upstream master's checkpoint
/// converter reads the same E8M0 bytes as `torch.exp2(bits - 127.0)` (`conversion/deepseek.py:646`).
/// The halved convention (`ggml_e8m0_to_fp32_half`) is used ONLY by MXFP4's doubled-table path.
/// Byte-level evidence cannot discriminate the two conventions (observed V4 scale bytes 115/116 fit
/// both, one octave apart), so this verdict rests on the fork's source, not on file bytes.
///
/// One deliberate divergence: ggml's macro has its `0xFF -> NaN` branch commented out ("we don't
/// need to handle NaNs"), so `0xFF` yields +Inf there; this function keeps OCP's NaN so a poisoned
/// scale is caught rather than silently multiplied through. `0xFF` = 2^128 is not producible by a
/// sane quantizer either way.
fn e8m0_bias127(e: u8) -> f32 {
    match e {
        // 2^-127 is subnormal in f32 (min normal is 2^-126), so it cannot be written as an exponent
        // field and is built from its mantissa bit instead.
        0 => f32::from_bits(1 << 22),
        // OCP reserves the all-ones exponent for NaN. Left as NaN deliberately: a weight that arrives
        // as NaN should stay NaN and be caught, not be silently clamped to a large finite number.
        0xFF => f32::NAN,
        _ => f32::from_bits((e as u32) << 23),
    }
}

/// `F8_E4M3_B128`: one E8M0 scale byte, then 128 E4M3 payload bytes — **scale FIRST, ggml-style**.
///
/// **The 129-byte container is verified against a real V4 file, 2026-08-19** — no longer a model-card
/// assumption. In `DeepSeek-V4-Flash-FP4-FP8-native.gguf` (156,148,189,760 bytes, x-repo-commit
/// `0b34e0b6`), all 365 type-42 tensors measure exactly `n/128 * 129` bytes by header offset
/// arithmetic (two independent parsers, zero deviants). Scale position was measured from the bytes
/// themselves: byte 0 of every 129-byte block in two independently probed tensors holds one of two
/// values (the shared exponent; entropy ~0 at phase 0 of 129) while bytes 1..=128 show full FP8
/// spread with sign bits near 50% (entropy >= 6.1 at every other phase, method calibrated on the same
/// file's known scale-first MXFP4 tensor). The fork's struct agrees: `{ uint8_t e; uint8_t qs[128]; }`
/// with `static_assert sizeof == 129` (nisparks/llama.cpp @ `9d36408`, `ggml-common.h:222-227`), as
/// does its converter, which writes the scale to byte 0 of each block
/// (`convert_hf_to_gguf.py:9429-9432`).
///
/// ⚠ This crate FIRST SHIPPED the opposite order — 128 payload bytes then the scale LAST — as a
/// labelled assumption. All three reconciliation sources refuted it; corrected 2026-08-19. The wrong
/// order read every element one byte early AND took the exponent from the wrong end of the block.
pub const F8_E4M3_B128_BYTES: usize = 129;
const F8_E4M3_B128: u32 = 1042; // internal id; the FILE always says 42

/// Dequantize `F8_E4M3_B128`: each block is one E8M0 scale byte, then 128 E4M3 elements.
pub fn deq_f8_e4m3_b128(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if n % 128 != 0 { return Err(format!("F8_E4M3_B128 needs a multiple of 128 elements, got {n}")); }
    let blocks = n / 128;
    let need = blocks * F8_E4M3_B128_BYTES;
    if raw.len() < need { return Err(format!("F8_E4M3_B128 needs {need} bytes for {n} elements, got {}", raw.len())); }
    let mut out = Vec::with_capacity(n);
    for b in 0..blocks {
        let base = b * F8_E4M3_B128_BYTES;
        // Scale FIRST — same byte order as MXFP4. Verified, not assumed: see [`F8_E4M3_B128_BYTES`].
        let d = e8m0_bias127(raw[base]);
        for j in 1..=128 { out.push(e4m3_to_f32(raw[base + j]) * d); }
    }
    Ok(out)
}

/// Refuse a file whose tensors do not fit the strides this crate believes their types have.
///
/// Every reader here computes a tensor's byte length from [`type_size`] and reads that many bytes at
/// the declared offset. Nothing else checks it. So if this crate's idea of a type's block layout ever
/// disagrees with the writer's, the read silently lands at the wrong stride and returns plausible
/// garbage: no error, no panic, just wrong weights and fluent wrong output.
///
/// That is not hypothetical. GGUF type ids in the 40s are contested territory — vendor extensions and
/// mainline ggml have both claimed ids there, with different block geometry. This crate maps id 42 to
/// a group-128 / 34-byte ternary layout (2.125 bpw); mainline ggml-org master assigns 42 to its own
/// `Q2_0` (`block_q2_0`: f16 d + 16 code bytes = 18 B / 64 values = 2.25 bpw, `ggml.h:432` @
/// `b062ba7`); and the nisparks V4 fork assigns 42 to `F8_E4M3_B128` (129 B / 128 values) — three
/// layouts, and a file carries only the id. See [`resolve_type_42`].
///
/// The check is cheap and general, and deliberately not a special case for one id: GGUF lays tensors
/// out sequentially, so the gap to the next tensor's offset bounds what a tensor can actually occupy.
/// If our computed size exceeds that gap, our stride is wrong for this file and the only safe answer
/// is to refuse. It cannot prove agreement (a type whose stride is too SMALL still fits the gap and
/// is caught only by the total-size check on the last tensor), so it is a floor, not a proof.
/// Rewrite ambiguous type ids to their internal resolutions, so every consumer downstream sees an
/// UNAMBIGUOUS id and none of them re-derives the disambiguation.
///
/// [`resolve_type_42`] existed and was tested before anything CALLED it — the same
/// written-ahead-of-its-wiring gap this tree has produced twice before (`Model::supports_batching`,
/// the joule router). Until this call, a DeepSeek V4 file was merely REFUSED by the stride guard;
/// with it, the file's type-42 tensors resolve to F8_E4M3_B128 (internal 1042) and load, while a
/// PrismML ternary file keeps meaning what it always meant.
///
/// Runs AFTER [`check_declared_strides`], so a header-probe prefix (where the last tensor's bound is
/// unknowable) resolves what it can: the last tensor of a prefix is left as-is when its size cannot
/// be established, and the full-file parse settles it.
fn resolve_ambiguous_types(tensors: &mut [TensorInfo], data_len: usize, align: usize) -> Result<(), String> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].offset);
    let have_full = order.last().is_none_or(|&i| data_len >= tensors[i].offset as usize);
    for w in 0..order.len() {
        let i = order[w];
        if tensors[i].ggml_type != 42 { continue }
        let n: usize = tensors[i].dims.iter().product::<u64>() as usize;
        let limit = match order.get(w + 1) {
            Some(&j) => tensors[j].offset as usize,
            None if have_full => data_len,
            None => continue, // header probe: the last tensor's bound is unknowable here
        };
        let avail = limit.saturating_sub(tensors[i].offset as usize);
        // `avail` = exact size + alignment padding, so the true size lies in (avail - align, avail].
        // Collect EVERY claimant whose exact stride lands in that window and demand exactly one —
        // first-match with a generous window mis-resolves small tensors, where the claimants' sizes
        // (34 / 36 / 129 bytes per 128 values) sit closer together than a loose slack bound. With the
        // real bound (align, typically 32) the strides separate at >= 16 blocks for 34-vs-36 and from
        // the very first block against 129.
        let matches: Vec<usize> = [34usize, 36, 129].iter()
            .map(|&bpb| n / 128 * bpb)
            .filter(|&sz| sz <= avail && avail < sz + align.max(1))
            .collect();
        match matches.as_slice() {
            [sz] => tensors[i].ggml_type = resolve_type_42(n, *sz)?,
            [] => return Err(format!(
                "tensor '{}' declares ggml type 42 but its {avail} available bytes for {n} elements \
                 match no claimant's stride (PrismML Q2_0 34 B, mainline Q2_0 36 B, F8_E4M3_B128 \
                 129 B per 128 values). Refusing to load rather than guess.", tensors[i].name)),
            _ => return Err(format!(
                "tensor '{}' is too small to disambiguate ggml type 42: {avail} bytes for {n} \
                 elements fits more than one claimant within one {align}-byte alignment. Refusing to \
                 load rather than pick — a mis-resolved tensor decodes to plausible garbage.",
                tensors[i].name)),
        }
    }
    Ok(())
}

fn check_declared_strides(tensors: &[TensorInfo], data_len: usize, align: usize) -> Result<(), String> {
    let mut by_offset: Vec<&TensorInfo> = tensors.iter().collect();
    by_offset.sort_by_key(|t| t.offset);
    // `backed.rs` parses a header out of a PREFIX of the file (`header_probe` starts at 1 MiB and
    // grows), so `data_len` is then the probe's length and not the data section's. Detect that rather
    // than reject every probe: if the buffer ends before the last tensor even begins, it is a prefix.
    // The gap between consecutive tensors is still meaningful in a prefix — only the final tensor's
    // bound needs the true total, so that one check is what gets skipped.
    let have_full_data = by_offset.last().is_none_or(|t| data_len >= t.offset as usize);
    for (i, t) in by_offset.iter().enumerate() {
        let n: usize = t.dims.iter().product::<u64>() as usize;
        // An unknown type is a separate, already-loud failure; nothing to compare against here.
        let Ok(sz) = type_size(t.ggml_type, n) else { continue };
        // The next tensor's start, or the end of the data section for the last one.
        let limit = match by_offset.get(i + 1) {
            Some(next) => next.offset as usize,
            None if have_full_data => data_len,
            None => continue,
        };
        let avail = limit.saturating_sub(t.offset as usize);
        if sz > avail {
            return Err(format!(
                "tensor '{}' (ggml type {}, {n} elements) needs {sz} bytes by this reader's block \
                 layout, but the file leaves only {avail} before the next tensor. The type id's \
                 layout here disagrees with the writer's, so reading it would silently return \
                 garbage at the wrong stride rather than fail. Refusing to load.\n\
                 If this is a mainline ggml file using a type id this crate maps to a vendor \
                 extension, the type table in this crate needs the file's layout, not a wider read.",
                t.name, t.ggml_type,
            ));
        }
        // Padding between tensors is legal and normal; a gap far larger than alignment means the
        // stride is too SMALL, which reads short and also corrupts. Only flag it when unmistakable.
        if i + 1 < by_offset.len() && avail > sz + align.max(64) * 4 && avail > sz * 2 {
            return Err(format!(
                "tensor '{}' (ggml type {}, {n} elements) is {sz} bytes by this reader's block \
                 layout, but the file reserves {avail} for it — more than twice as much, and far \
                 beyond alignment padding. This reader's stride for that type id is too small, which \
                 reads a short prefix of every block. Refusing to load.",
                t.name, t.ggml_type,
            ));
        }
    }
    Ok(())
}

/// Uniform read access over a GGUF, however it's held: the eager in-memory `Gguf` (the browser path
/// — the whole file is fetched into a `Vec<u8>`) and the lazy file-backed `GgufFile` (native, one
/// tensor in RAM at a time) both implement it, so model loaders are written once against the trait.
fn ferric_gguf_type_size(t: &TensorInfo) -> Result<usize, String> {
    type_size(t.ggml_type, t.dims.iter().product::<u64>() as usize)
}

pub trait GgufSource {
    fn metadata(&self) -> &HashMap<String, Meta>;
    fn tensor(&self, name: &str) -> Option<&TensorInfo>;
    fn raw(&self, name: &str) -> Result<Vec<u8>, String>;
    fn dequant(&self, name: &str) -> Result<Vec<f32>, String>;

    /// **Read a sub-range of a tensor without materialising the whole tensor.**
    ///
    /// A MoE's routed-expert tensors are the bulk of a checkpoint — 97.5% of DeepSeek-V4-Flash — so
    /// a caller that wants ONE expert should not have to hold all of them to get it. That is the
    /// difference between streaming experts and merely relocating them.
    ///
    /// The default is correct and eager: it reads the whole tensor and slices. Overriding it is what
    /// makes a source lazy, and [`GgufFile`] does. Callers may therefore use this unconditionally and
    /// get laziness where the source can provide it, rather than branching on the source type.
    /// Where a tensor's bytes live on disk: `(file, absolute offset, length)`.
    ///
    /// `raw_range` is lazy but borrows `self`. A streaming expert cache outlives the loader that
    /// built it, so it needs an OWNED handle — which means a path. Sources that are not file-backed
    /// return `None` and the caller falls back to holding the bytes.
    fn tensor_file_range(&self, _name: &str) -> Option<(std::path::PathBuf, u64, u64)> { None }

    fn raw_range(&self, name: &str, off: u64, dst: &mut [u8]) -> Result<(), String> {
        let all = self.raw(name)?;
        let end = off as usize + dst.len();
        if end > all.len() {
            return Err(format!("{name}: range {off}..{end} exceeds the tensor's {} bytes", all.len()));
        }
        dst.copy_from_slice(&all[off as usize..end]);
        Ok(())
    }
}

impl Gguf {
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.tensors.iter().find(|t| t.name == name) }

    /// A tensor's raw on-disk bytes (packed, as stored) — the in-memory analogue of `GgufFile::raw`.
    pub fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let sz = type_size(t.ggml_type, n)?;
        let start = self.data_start + t.offset as usize;
        Ok(self.data[start..start + sz].to_vec())
    }

    /// Dequantize a tensor to f32 (row-major), whatever its GGUF block-quant type.
    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        deq_raw(&self.data[self.data_start + t.offset as usize..], n, t.ggml_type)
    }
}

impl GgufSource for Gguf {
    fn metadata(&self) -> &HashMap<String, Meta> { &self.metadata }
    fn tensor(&self, name: &str) -> Option<&TensorInfo> { Gguf::tensor(self, name) }
    fn raw(&self, name: &str) -> Result<Vec<u8>, String> { Gguf::raw(self, name) }
    fn dequant(&self, name: &str) -> Result<Vec<f32>, String> { Gguf::dequant(self, name) }
}

/// On-disk byte size of `n` elements stored as ggml type `ty`.
pub fn type_size(ty: u32, n: usize) -> Result<usize, String> {
    Ok(match ty {
        F32 => n * 4,
        F16T => n * 2,
        BF16T => n * 2,
        Q8_0 => n / 32 * 34,
        Q4_0 => n / 32 * 18,
        Q4_1 => n / 32 * 20,
        Q5_0 => n / 32 * 22,
        Q5_1 => n / 32 * 24,
        Q2_K => n / 256 * 84,
        Q3_K => n / 256 * 110,
        Q4_K => n / 256 * 144,
        Q5_K => n / 256 * 176,
        Q6_K => n / 256 * 210,
        IQ2_XXS => n / 256 * 66,
        IQ3_XXS => n / 256 * 98,
        IQ4_NL => n / 32 * 18,
        IQ4_XS => n / 256 * 136,
        TQ2_0 => n / 256 * 66,
        STQ1_0 => n / 256 * STQ1_0_BLOCK_BYTES,
        MXFP4 => n / 32 * 17,
        Q1_0 => n / 128 * 18,
        Q2_0 => n / 128 * 34,
        // Internal id only — the FILE says 42, [`resolve_type_42`] maps it here by stride.
        F8_E4M3_B128 => n / 128 * F8_E4M3_B128_BYTES,
        other => return Err(format!("unsupported ggml type {other}")),
    })
}

/// Dequantize `n` elements of ggml type `ty` out of a raw byte slice.
pub fn deq_raw(raw: &[u8], n: usize, ty: u32) -> Result<Vec<f32>, String> {
    Ok(match ty {
        F32 => raw[..n * 4].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        F16T => raw[..n * 2].chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        BF16T => raw[..n * 2].chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        Q8_0 => deq_q8_0(raw, n),
        Q4_0 => deq_q4_0(raw, n),
        Q4_1 => deq_q4_1(raw, n),
        Q5_0 => deq_q5_0(raw, n),
        Q5_1 => deq_q5_1(raw, n),
        Q2_K => deq_q2_k(raw, n),
        Q3_K => deq_q3_k(raw, n),
        Q4_K => deq_q4_k(raw, n),
        Q5_K => deq_q5_k(raw, n),
        Q6_K => deq_q6_k(raw, n),
        IQ2_XXS => deq_iq2_xxs(raw, n),
        IQ3_XXS => deq_iq3_xxs(raw, n),
        IQ4_NL => deq_iq4_nl(raw, n),
        IQ4_XS => deq_iq4_xs(raw, n),
        TQ2_0 => deq_tq2_0(raw, n),
        STQ1_0 => deq_stq1_0(raw, n),
        MXFP4 => deq_mxfp4(raw, n),
        Q1_0 => deq_q1_0(raw, n),
        Q2_0 => deq_q2_0(raw, n),
        // Internal id only — the FILE says 42, [`resolve_type_42`] maps it here by stride.
        F8_E4M3_B128 => deq_f8_e4m3_b128(raw, n)?,
        other => return Err(format!("unsupported ggml type {other}")),
    })
}

/// **Lazy, file-backed GGUF** — parses the header from a bounded prefix read, then pulls each
/// tensor's bytes on demand. A 27B ternary checkpoint is 7 GB on disk; this keeps exactly one
/// tensor in host RAM at a time so peak memory is the largest tensor, not the whole file.
pub struct GgufFile {
    pub metadata: HashMap<String, Meta>,
    pub tensors: Vec<TensorInfo>,
    /// One entry per FILE. A single-file model has exactly one; a sharded model has `split.count`.
    shards: Vec<Shard>,
    /// `tensors[i]` lives in `shards[owner[i]]`. Parallel to `tensors`, so the merged tensor table
    /// looks exactly like a single file's to every caller while `raw` still reads from the right one.
    owner: Vec<usize>,
}

struct Shard { f: std::cell::RefCell<std::fs::File>, data_start: u64, len: u64,
               /// Kept so a caller can open its OWN handle on this file. `raw_range` is lazy but
               /// borrows `self`; a streaming cache has to outlive the loader that built it, and
               /// an owned path is the only thing that crosses that boundary.
               path: std::path::PathBuf }

/// Parse one file's header, growing the prefix read until it succeeds.
fn open_one(path: &std::path::Path) -> Result<(HashMap<String, Meta>, Vec<TensorInfo>, Shard), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
    // Header = magic + metadata (which includes the tokenizer vocab, often megabytes) + tensor
    // infos. Its size isn't known up front, so read a prefix and grow until it parses.
    let mut cap = (8usize << 20).min(len);
    loop {
        let mut buf = vec![0u8; cap];
        let n = read_prefix(&mut f, &mut buf)?;
        buf.truncate(n);
        match parse(buf) {
            Ok(g) => return Ok((g.metadata, g.tensors,
                                Shard { f: std::cell::RefCell::new(f), data_start: g.data_start as u64,
                                        len: len as u64, path: path.to_path_buf() })),
            Err(e) => {
                if cap >= len { return Err(e); }
                cap = (cap * 4).min(len);
            }
        }
    }
}

/// Rebuild a shard path for `no` (0-based) from any sibling's path.
///
/// `llama-gguf-split` names parts `<prefix>-%05d-of-%05d.gguf`, so the shape is recovered by finding
/// `-of-` and stepping back over the five digits and the dash before it. Derived rather than guessed:
/// the caller may hand us ANY shard, not necessarily the first.
fn shard_path(any: &std::path::Path, no: usize, count: usize) -> Result<std::path::PathBuf, String> {
    let name = any.file_name().and_then(|s| s.to_str()).ok_or("shard path has no file name")?;
    let stem = name.strip_suffix(".gguf").ok_or_else(|| format!("{name}: not a .gguf"))?;
    let at = stem.rfind("-of-").ok_or_else(|| format!(
        "{name} declares split.count {count} but is not named <prefix>-00001-of-{count:05}.gguf, so \
         its siblings cannot be located"))?;
    if at < 6 { return Err(format!("{name}: malformed split suffix")); }
    let prefix = &stem[..at - 6];
    Ok(any.with_file_name(format!("{prefix}-{:05}-of-{:05}.gguf", no + 1, count)))
}

impl GgufFile {
    /// Open a GGUF, following `split.count` to its sibling shards when the file is one part of many.
    ///
    /// Large open-weight checkpoints are distributed as `model-00001-of-00009.gguf` and friends —
    /// llama.cpp, vLLM and Ollama all follow the parts automatically. Before this, Ferric loaded the
    /// first shard alone, saw a tensor table covering a FRACTION of the model, and failed far away
    /// with `no tensor 'blk.11.attn_v.weight'` — a message that names a tensor rather than the fact
    /// that 182 of 310 were in files it never opened. Any shard may be passed; the rest are derived.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<GgufFile, String> {
        let path = path.as_ref();
        let (metadata, tensors, shard) = open_one(path)?;
        let count = match metadata.get("split.count") { Some(Meta::U(v)) => *v as usize, _ => 1 };
        if count <= 1 {
            let owner = vec![0; tensors.len()];
            return Ok(GgufFile { metadata, tensors, shards: vec![shard], owner });
        }

        // Shard 0 carries the model metadata — tokenizer, architecture, everything. Later parts
        // carry only their own tensor tables, so shard 0 is loaded even when a later part was passed.
        let this_no = match metadata.get("split.no") { Some(Meta::U(v)) => *v as usize, _ => 0 };
        let want = match metadata.get("split.tensors.count") { Some(Meta::U(v)) => *v as usize, _ => 0 };

        let mut md: Option<HashMap<String, Meta>> = None;
        let mut all: Vec<TensorInfo> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        let mut shards: Vec<Shard> = Vec::new();
        let mut carried = Some((metadata, tensors, shard));

        for no in 0..count {
            let (m, ts, sh) = if no == this_no {
                carried.take().ok_or("split.no names the same shard twice")?
            } else {
                open_one(&shard_path(path, no, count)?)?
            };
            if no == 0 { md = Some(m); }
            let idx = shards.len();
            shards.push(sh);
            owner.extend(std::iter::repeat(idx).take(ts.len()));
            all.extend(ts);
        }

        let metadata = md.ok_or("shard 0 was never loaded")?;
        // A count checksum is necessary and NOT sufficient: truncating a shard leaves its header and
        // tensor table intact, so the tensor count still adds up while the bytes are gone. Measured —
        // cutting part 3 to 200 KB passed the count check and loaded 310 tensors. `read_exact` would
        // eventually catch it, but as "failed to fill whole buffer" from inside one tensor read rather
        // than as a truncated file. So each shard is checked against the span its own tensors claim.
        for (i, sh) in shards.iter().enumerate() {
            let end = all.iter().zip(&owner).filter(|(_, o)| **o == i)
                .try_fold(0u64, |m, (t, _)| {
                    let n: usize = t.dims.iter().product::<u64>() as usize;
                    Ok::<u64, String>(m.max(sh.data_start + t.offset + type_size(t.ggml_type, n)? as u64))
                })?;
            if end > sh.len {
                return Err(format!(
                    "shard {} of {count} is {} bytes but its tensor table claims data out to {end} — \
                     the file is truncated", i + 1, sh.len));
            }
        }
        // The count is a CHECKSUM on the set of files, and the only thing standing between a missing
        // part and a model that loads with holes in it. A short set must fail here, naming the
        // shortfall, rather than 200 lines later naming one tensor.
        if want != 0 && all.len() != want {
            return Err(format!(
                "split.tensors.count declares {want} tensors but the {count} shards hold {}; a part is \
                 missing or truncated", all.len()));
        }
        Ok(GgufFile { metadata, tensors: all, shards, owner })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.tensors.iter().find(|t| t.name == name) }

    /// Byte offset where the tensor data section begins. `TensorInfo::offset` is relative to this, so a
    /// tensor's absolute position is `data_start() + t.offset` — which is what a streaming reader needs
    /// in order to fetch weights positionally without going through this handle (whose `File` sits behind
    /// a `RefCell` and is therefore not shareable across threads).
    /// ⚠ Shard 0's data offset. For a SHARDED model `TensorInfo::offset` is relative to the data
    /// section of whichever file holds that tensor, so `data_start() + t.offset` is only an absolute
    /// position when [`GgufFile::shard_count`] is 1. Positional readers must check that first.
    pub fn data_start(&self) -> u64 { self.shards[0].data_start }

    /// The tensor's raw on-disk bytes — packed, exactly as stored (feed straight to a native
    /// quantized matmul so the weights never round-trip through f32).
    pub fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        // By INDEX, not by reference: `owner` is parallel to `tensors`, and a sharded model's offsets
        // are relative to the data section of the file that holds them, not to the first file's.
        let i = self.tensors.iter().position(|t| t.name == name)
            .ok_or_else(|| format!("no tensor '{name}'"))?;
        let t = &self.tensors[i];
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let sz = type_size(t.ggml_type, n)?;
        let sh = &self.shards[self.owner[i]];
        let mut buf = vec![0u8; sz];
        read_at(&mut sh.f.borrow_mut(), sh.data_start + t.offset, &mut buf)?;
        Ok(buf)
    }

    /// How many files back this model. 1 for an ordinary GGUF, `split.count` for a sharded one.
    pub fn shard_count(&self) -> usize { self.shards.len() }

    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        deq_raw(&self.raw(name)?, n, t.ggml_type)
    }
}

impl GgufSource for GgufFile {
    fn metadata(&self) -> &HashMap<String, Meta> { &self.metadata }
    fn tensor(&self, name: &str) -> Option<&TensorInfo> { GgufFile::tensor(self, name) }
    fn raw(&self, name: &str) -> Result<Vec<u8>, String> { GgufFile::raw(self, name) }
    fn dequant(&self, name: &str) -> Result<Vec<f32>, String> { GgufFile::dequant(self, name) }
    fn tensor_file_range(&self, name: &str) -> Option<(std::path::PathBuf, u64, u64)> {
        let idx = self.tensors.iter().position(|t| t.name == name)?;
        let t = &self.tensors[idx];
        let sh = &self.shards[self.owner[idx]];
        let n = ferric_gguf_type_size(t).ok()? as u64;
        Some((sh.path.clone(), sh.data_start + t.offset, n))
    }

    /// Seek straight to the range. This is what makes expert streaming possible: the caller reads
    /// one expert's bytes without the other 255 ever entering memory.
    fn raw_range(&self, name: &str, off: u64, dst: &mut [u8]) -> Result<(), String> {
        use std::io::{Read, Seek, SeekFrom};
        let idx = self.tensors.iter().position(|t| t.name == name)
            .ok_or_else(|| format!("no tensor '{name}'"))?;
        let t = &self.tensors[idx];
        let n = ferric_gguf_type_size(t)?;
        let end = off as usize + dst.len();
        if end > n {
            return Err(format!("{name}: range {off}..{end} exceeds the tensor's {n} bytes"));
        }
        let sh = &self.shards[self.owner[idx]];
        let mut f = sh.f.borrow_mut();
        f.seek(SeekFrom::Start(sh.data_start + t.offset + off))
            .map_err(|e| format!("{name}: seek: {e}"))?;
        f.read_exact(dst).map_err(|e| format!("{name}: read: {e}"))
    }
}

fn read_prefix(f: &mut std::fs::File, buf: &mut [u8]) -> Result<usize, String> {
    use std::io::{Read, Seek, SeekFrom};
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut got = 0;
    while got < buf.len() {
        match f.read(&mut buf[got..]).map_err(|e| e.to_string())? { 0 => break, k => got += k }
    }
    Ok(got)
}

fn read_at(f: &mut std::fs::File, off: u64, buf: &mut [u8]) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};
    f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
    f.read_exact(buf).map_err(|e| e.to_string())
}

pub(crate) fn rd_f16(b: &[u8]) -> f32 { f16::from_le_bytes([b[0], b[1]]).to_f32() }

/// Proof-only replacement for [`rd_f16`]: 1.0 for the bit pattern of 1.0, and 0.0 for ANY other
/// two bytes.
///
/// It exists because `half` reaches runtime CPU-feature detection on aarch64, and that path uses
/// C string literals which Kani cannot encode -- so without a stub the model checker fails on the
/// sysctl name of a NEON probe rather than on anything about this format.
///
/// WARNING: the first version of this stub returned 1.0 unconditionally, and a mutation that read
/// the scale from the FRONT of the block instead of the back sailed through every proof -- the stub
/// had erased the very property being tested. Deciding on the bytes fixes that: a decoder that
/// reads the wrong offset is handed payload bytes, gets a zero scale, and every lane collapses to
/// zero where the placement theorems expect +-1.
///
/// It still narrows the claim, and that is the point of saying so: f16 ARITHMETIC is not verified
/// here. Which two bytes are read, and where each decoded value lands, are.
#[cfg(kani)]
pub(crate) fn rd_f16_one(b: &[u8]) -> f32 { if b[0] == 0x00 && b[1] == 0x3C { 1.0 } else { 0.0 } }

/// Q8_0: blocks of 32 → [f16 scale, i8 qs[32]] (34 bytes). x = qs·scale.
fn deq_q8_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(34).take(n / 32) {
        let d = rd_f16(&blk[0..2]);
        for &q in &blk[2..34] { out.push(q as i8 as f32 * d); }
    }
    out
}

/// Q4_0: blocks of 32 → [f16 scale, u8 qs[16]] (18 bytes). x = (nibble-8)·scale, low nibbles then high.
fn deq_q4_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(18).take(n / 32).enumerate() {
        let d = rd_f16(&blk[0..2]);
        for i in 0..16 {
            let byte = blk[2 + i];
            out[bi * 32 + i] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            out[bi * 32 + i + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
    }
    out
}

/// Q4_1: blocks of 32 → [f16 d, f16 min, u8 qs[16]] (20 bytes). x = nibble·d + m (no −8), low
/// nibbles then high. The affine (min-offset) sibling of Q4_0 — better for asymmetric weights.
fn deq_q4_1(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(20).take(n / 32).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let m = rd_f16(&blk[2..4]);
        for i in 0..16 {
            let byte = blk[4 + i];
            out[bi * 32 + i] = (byte & 0x0F) as f32 * d + m;
            out[bi * 32 + i + 16] = (byte >> 4) as f32 * d + m;
        }
    }
    out
}

/// Q5_0: blocks of 32 → [f16 d, u32 qh, u8 qs[16]] (22 bytes). Each value is a 5-bit signed code:
/// the low 4 bits from a `qs` nibble, the 5th (high) bit from `qh` — bit i for value i, bit i+16 for
/// value i+16 — reassembled and offset: x = ((nibble | (bit<<4)) − 16)·d. (llama.cpp ggml layout.)
fn deq_q5_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(22).take(n / 32).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        for i in 0..16 {
            let byte = blk[6 + i];
            let xh0 = ((qh >> i) & 1) << 4;
            let xh1 = ((qh >> (i + 16)) & 1) << 4;
            out[bi * 32 + i] = (((byte & 0x0F) as u32 | xh0) as i32 - 16) as f32 * d;
            out[bi * 32 + i + 16] = (((byte >> 4) as u32 | xh1) as i32 - 16) as f32 * d;
        }
    }
    out
}

/// Q5_1: blocks of 32 → [f16 d, f16 min, u32 qh, u8 qs[16]] (24 bytes). The affine sibling of Q5_0:
/// value = (nibble | (5th-bit<<4))·d + m (no −16). 5th bit i for value i, bit i+16 for value i+16.
fn deq_q5_1(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(24).take(n / 32).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let m = rd_f16(&blk[2..4]);
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        for i in 0..16 {
            let byte = blk[8 + i];
            let xh0 = ((qh >> i) & 1) << 4;
            let xh1 = ((qh >> (i + 16)) & 1) << 4;
            out[bi * 32 + i] = ((byte & 0x0F) as u32 | xh0) as f32 * d + m;
            out[bi * 32 + i + 16] = ((byte >> 4) as u32 | xh1) as f32 * d + m;
        }
    }
    out
}

/// Q4_K super-block (256 values, 144 bytes): [f16 d, f16 dmin, u8 scales[12], u8 qs[128]].
/// 8 sub-blocks of 32; each has a 6-bit scale & 6-bit min packed in `scales`. y = d·sc·q − dmin·m.
/// **Q5_K** — 176-byte super-block, 256 values: `f16 d`, `f16 dmin`, 12 packed scale bytes (same 8
/// six-bit (scale,min) pairs as Q4_K), `qh[32]` (one high bit per value), `qs[128]` (low 4 bits).
/// value = `d·scaleₛ·(nibble + 16·qh_bit) − dmin·minₛ`.
fn deq_q5_k(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let get_sc_min = |scales: &[u8], j: usize| -> (u8, u8) {
        if j < 4 { (scales[j] & 63, scales[j + 4] & 63) }
        else { ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)) }
    };
    for (bi, blk) in raw.chunks_exact(176).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let dmin = rd_f16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let mut y = bi * 256;
        let (mut is, mut q) = (0usize, 0usize);
        for jg in 0..4 {
            let (sc1, m1) = get_sc_min(scales, is);
            let (sc2, m2) = get_sc_min(scales, is + 1);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            let (b1, b2) = (1u8 << (2 * jg), 1u8 << (2 * jg + 1));
            for l in 0..32 { out[y + l] = d1 * ((qs[q + l] & 0x0F) + if qh[l] & b1 != 0 { 16 } else { 0 }) as f32 - mm1; }
            for l in 0..32 { out[y + l + 32] = d2 * ((qs[q + l] >> 4) + if qh[l] & b2 != 0 { 16 } else { 0 }) as f32 - mm2; }
            y += 64; q += 32; is += 2;
        }
    }
    out
}

/// **Q2_K** — 2 bits per weight, 84 bytes per 256 (2.625 bpw). Two f16 super-scales, `d` for the
/// scale and `dmin` for the min, and SIXTEEN sub-blocks of 16 each carrying its own 4-bit scale and
/// 4-bit min packed into one byte: `value = d·(sc & 0xF)·q − dmin·(sc >> 4)`.
///
/// ⚠ The `q` walk is NOT sequential and the byte order is the trap. Each of the 64 `qs` bytes holds
/// four 2-bit quants, but they belong to four DIFFERENT sub-blocks: the shift selects the sub-block
/// and the byte index selects the element within it. Reading `qs` straight through — the obvious
/// translation — produces plausibly-scaled garbage rather than an error, so this mirrors
/// `dequantize_row_q2_K`'s exact loop nesting instead of restructuring it.
fn deq_q2_k(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(84).take(n / 256).enumerate() {
        let sc = &blk[0..16];
        let d = rd_f16(&blk[80..82]);
        let dmin = rd_f16(&blk[82..84]);
        let mut y = bi * 256;
        let mut is = 0usize;
        for half in 0..2 {
            let q = &blk[16 + half * 32..16 + half * 32 + 32];
            for j in 0..4 {
                let shift = 2 * j;
                for grp in 0..2 {
                    let s = sc[is];
                    is += 1;
                    let (dl, ml) = (d * (s & 0xF) as f32, dmin * (s >> 4) as f32);
                    for l in 0..16 {
                        out[y] = dl * ((q[grp * 16 + l] >> shift) & 3) as f32 - ml;
                        y += 1;
                    }
                }
            }
        }
    }
    out
}

/// **Q3_K** — 3 bits per weight, 110 bytes per 256 (3.4375 bpw). One f16 super-scale, 16 sub-blocks
/// of 16, and the third bit of every quant held in a separate 32-byte `hmask` plane.
///
/// Two traps, both silent.
///
/// The high bit is INVERTED: `hmask` set means add nothing, hmask CLEAR subtracts 4. So the quant is
/// `(qs & 3) − 4·(hmask bit == 0)`, giving the signed range −4..3. Reading it the intuitive way —
/// set means add 4 — flips the sign of most weights and still produces finite, plausibly-scaled
/// output. And `m` advances once per sub-block PAIR, not per element.
///
/// The 16 six-bit scales are packed across 12 bytes as four little-endian `u32`s, low 4 bits of each
/// scale in the first eight bytes and the high 2 bits spread across the last four. Reconstructed
/// exactly as the reference does it, then read back as sixteen bytes, biased by −32.
fn deq_q3_k(raw: &[u8], n: usize) -> Vec<f32> {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(110).take(n / 256).enumerate() {
        let hm = &blk[0..32];
        let d_all = rd_f16(&blk[108..110]);

        let mut aux = [0u32; 4];
        for k in 0..3 {
            aux[k] = u32::from_le_bytes([blk[96 + k * 4], blk[97 + k * 4], blk[98 + k * 4], blk[99 + k * 4]]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scales = [0i8; 16];
        for k in 0..4 {
            for (b, v) in aux[k].to_le_bytes().iter().enumerate() { scales[k * 4 + b] = *v as i8; }
        }

        let mut y = bi * 256;
        let mut is = 0usize;
        let mut m = 1u8;
        for half in 0..2 {
            let q = &blk[32 + half * 32..32 + half * 32 + 32];
            for j in 0..4 {
                let shift = 2 * j;
                for grp in 0..2 {
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let i = grp * 16 + l;
                        let q3 = ((q[i] >> shift) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                        out[y] = dl * q3 as f32;
                        y += 1;
                    }
                }
                m <<= 1;
            }
        }
    }
    out
}

/// **Q6_K** — 210-byte super-block, 256 values: `ql[128]` (low 4 bits), `qh[64]` (high 2 bits),
/// `scales[16]` (int8), `d` (f16). Value = `d · scale · (q − 32)`, where q is the reassembled 6-bit
/// quant. Layout follows llama.cpp exactly (two 128-value halves, 4 quant groups per half).
fn deq_q6_k(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(210).take(n / 256).enumerate() {
        let d = rd_f16(&blk[208..210]);
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208]; // int8 scales
        let mut y = bi * 256;
        for half in 0..2 {
            let (qlh, qhh, sch) = (&ql[half * 64..], &qh[half * 32..], &sc[half * 8..]);
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) as i32 - 32;
                out[y + l] = d * sch[is] as i8 as f32 * q1 as f32;
                out[y + l + 32] = d * sch[is + 2] as i8 as f32 * q2 as f32;
                out[y + l + 64] = d * sch[is + 4] as i8 as f32 * q3 as f32;
                out[y + l + 96] = d * sch[is + 6] as i8 as f32 * q4 as f32;
            }
            y += 128;
        }
    }
    out
}

/// The IQ4 non-linear 4-bit codebook (llama.cpp `kvalues_iq4nl`): a 4-bit index maps to one of
/// these 16 signed levels (denser near zero), instead of a uniform grid — this is what lets IQ4
/// beat Q4 at the same bit-width.
const KVALUES_IQ4NL: [i32; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// **IQ4_NL** — 18-byte block, 32 values: `f16 d`, `qs[16]` (two 4-bit codebook indices each).
/// value = `d · kvalues_iq4nl[idx]`.
fn deq_iq4_nl(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(18).take(n / 32).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qs = &blk[2..18];
        let y = bi * 32;
        for j in 0..16 {
            out[y + j] = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
            out[y + j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
    out
}

/// **IQ4_XS** — 136-byte super-block, 256 values: `f16 d`, `u16 scales_h`, `scales_l[4]`, `qs[128]`.
/// 8 sub-blocks of 32; each sub-block `ib` has a 6-bit scale `ls` (low nibble from `scales_l[ib/2]`,
/// high 2 bits from `scales_h` at bit `2·ib`); `dl = d·(ls − 32)`, value = `dl · kvalues_iq4nl[idx]`.
fn deq_iq4_xs(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(136).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        let base = bi * 256;
        for ib in 0..8 {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as u16 | (((scales_h >> (2 * ib)) & 3) << 4)) as i32;
            let dl = d * (ls - 32) as f32;
            let (y, q) = (base + ib * 32, &qs[ib * 16..]);
            for j in 0..16 {
                out[y + j] = dl * KVALUES_IQ4NL[(q[j] & 0x0F) as usize] as f32;
                out[y + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
            }
        }
    }
    out
}

fn deq_q4_k(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let get_sc_min = |scales: &[u8], j: usize| -> (u8, u8) {
        if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            (
                (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        }
    };
    for (bi, blk) in raw.chunks_exact(144).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let dmin = rd_f16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let mut is = 0usize;
        let mut y = bi * 256;
        let mut q = 0usize;
        for _ in 0..4 {
            // 64 values per iteration: low nibbles (sub-block `is`), then high nibbles (`is+1`)
            let (sc1, m1) = get_sc_min(scales, is);
            let (sc2, m2) = get_sc_min(scales, is + 1);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 { out[y + l] = d1 * (qs[q + l] & 0x0F) as f32 - mm1; }
            for l in 0..32 { out[y + l + 32] = d2 * (qs[q + l] >> 4) as f32 - mm2; }
            y += 64; q += 32; is += 2;
        }
    }
    out
}

/// TQ2_0 (llama.cpp ternary / BitNet): 256-value super-block = `qs[64]` (2-bit codes, 4 per byte) then
/// `f16 d`. Value = d·(code−1), code ∈ {0,1,2} → {−1,0,+1}. Output order matches llama.cpp's layout.
/// `ksigns_iq2xs[i]` without the 128-byte table: the low seven bits are `i` itself, and the top bit
/// is set to whatever makes the byte's population count even.
///
/// ggml carries this as a literal table; it is a parity code, so Ferric computes it. A derived
/// constant cannot be mistyped, and the derivation is the documentation — the reader can see that
/// the eighth sign is not free information but a checksum of the other seven.
#[inline]
fn ksigns(i: u8) -> u8 {
    let low = i & 0x7f;
    low | ((low.count_ones() as u8 & 1) << 7)
}

#[cfg(any(test, kani))]
pub(crate) fn ksigns_for_proof(i: u8) -> u8 { ksigns(i) }

/// **IQ2_XXS** — 2.0625 bpw. 66 bytes per 256 weights: an fp16 block scale and 32 `u16` of payload,
/// read as eight 8-byte words, one per 32-weight group.
///
/// Each group's two `u32` carry four grid indices (the low word, one byte each) and one packed
/// control word: seven bits of sign index per sub-block of 8, plus a 4-bit sub-scale in the top
/// nibble. The reconstruction is `d · (0.5 + subscale) · 0.25 · grid[j] · sign`, and the grid byte
/// is a magnitude only — every value in [`IQ2XXS_GRID`] is drawn from `{8, 25, 43}`, so the format
/// spends its bits almost entirely on WHICH of 256 magnitude patterns and WHICH of 128 sign
/// patterns, not on the magnitudes themselves.
///
/// ⚠ The sub-scale is `(aux >> 28)`, a 4-bit field, and it shares the same `u32` as the four 7-bit
/// sign indices — 4·7 + 4 = 32 exactly, with no spare bit. Reading the scale from the other word
/// gives a plausible small float and a quietly rescaled group.
fn deq_iq2_xxs(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(66).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        for ib32 in 0..8 {
            let q = &blk[2 + 8 * ib32..2 + 8 * ib32 + 8];
            let lo = u32::from_le_bytes([q[0], q[1], q[2], q[3]]);
            let hi = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
            let db = d * (0.5 + (hi >> 28) as f32) * 0.25;
            for l in 0..4 {
                let g = IQ2XXS_GRID[((lo >> (8 * l)) & 0xff) as usize].to_le_bytes();
                let signs = ksigns(((hi >> (7 * l)) & 127) as u8);
                for j in 0..8 {
                    let s = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out[bi * 256 + ib32 * 32 + l * 8 + j] = db * g[j] as f32 * s;
                }
            }
        }
    }
    out
}

/// **IQ3_XXS** — 3.0625 bpw. 98 bytes per 256 weights: an fp16 scale, then 64 bytes of grid indices
/// followed by 32 bytes of packed sign-and-scale words.
///
/// ⚠ The two halves of `qs` are not interleaved — all 64 index bytes come first and the eight
/// control words follow at offset `QK_K/4`. Reading them as one interleaved stream keeps the block
/// size, the element count and the value distribution and destroys the pairing.
///
/// Each 32-weight group consumes eight index bytes and one control word. Unlike IQ2_XXS the grid
/// entry is only FOUR bytes, so a sub-block of 8 takes two lookups — and the sign byte splits
/// across them, bits 0..3 for the first and bits 4..7 for the second. The multiplier is `0.5`, not
/// IQ2_XXS's `0.25`.
fn deq_iq3_xxs(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(98).take(n / 256).enumerate() {
        let d = rd_f16(&blk[0..2]);
        let qs = &blk[2..2 + 64];
        let sas = &blk[2 + 64..2 + 96];
        for ib32 in 0..8 {
            let aux = u32::from_le_bytes([sas[4 * ib32], sas[4 * ib32 + 1], sas[4 * ib32 + 2], sas[4 * ib32 + 3]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = ksigns(((aux >> (7 * l)) & 127) as u8);
                let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
                let base = bi * 256 + ib32 * 32 + l * 8;
                for j in 0..4 {
                    let s1 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                    out[base + j] = db * g1[j] as f32 * s1;
                    out[base + j + 4] = db * g2[j] as f32 * s2;
                }
            }
        }
    }
    out
}

/// Bytes one STQ1_0 super-block occupies: `qs[32] + sign[8] + d` = 42 for 256 weights, which is
/// 1.3125 bits per weight — the lowest-rate format Ferric reads.
pub const STQ1_0_BLOCK_BYTES: usize = 42;

/// The **STQ1_0 codebook**. Index is `(sign << 4) | slot`; the entry packs four 2-bit lanes, lane
/// `p` in bits `2p..2p+2`, each decoding as `lane − 1` ∈ {−1, 0, +1}.
///
/// There are exactly 32 legal patterns because the format *forces* one zero into every group of
/// four: 4 choices of which lane is zero × 2³ signs for the other three. `slot` (4 bits) carries
/// the zero position and two of the three signs; `sign` (1 bit) carries the last one, and the
/// second half of the table is the first half with every non-zero lane negated. Sign 0 is the half
/// whose first non-zero lane is `+1`.
///
/// The 3:4 sparsity is therefore a **structural guarantee of the container**, not a property of
/// the weights that happened to hold: no legal byte sequence can encode a group with zero, two or
/// four zeros. That is what buys the rate — a free ternary group would need log₂(3⁴) ≈ 6.34 bits,
/// and this spends 5.
pub const STQ1_0_CODEBOOK: [u8; 32] = [
    // sign = 0 — first non-zero lane is +1
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    // sign = 1 — every non-zero lane negated
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
];

/// **STQ1_0** — the 1.3125-bpw ternary format introduced with Tencent's Hy4 (`hyv4`), where it
/// carries `ffn_gate_exps` and `ffn_up_exps` on 29 of the 77 MoE layers.
///
/// Two details are not free choices, and both are the kind that produce fluent nonsense rather
/// than an error, because neither changes a shape or an element count:
///
/// ⚠ **The scale is at the END of the block, not the start.** `qs[32] | sign[8] | d` — every other
/// ggml block Ferric reads leads with `d`. Reading `blk[0..2]` as the scale gets a plausible small
/// float (it is really eight packed slot codes) and every weight in the block comes out wrong by
/// one shared factor, which an RMS check on a single tensor will not obviously fail.
///
/// ⚠ **A group is stride-16, not contiguous.** Group `g` covers `{c·64 + (g mod 16) + p·16}` for
/// `p ∈ 0..4`, where `c = g / 16`. Decoding groups as four consecutive weights writes exactly the
/// right 256 values into exactly the right 256 slots in the wrong ORDER — same count, same
/// multiset, no assert can fire. It is the same shape of bug as reading a convolution's weight
/// layout under the wrong permutation.
pub(crate) fn deq_stq1_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(STQ1_0_BLOCK_BYTES).take(n / 256).enumerate() {
        let d = rd_f16(&blk[40..42]);
        for g in 0..64 {
            let slot = (blk[g / 2] >> (4 * (g & 1))) & 0x0F;
            let sign = (blk[32 + g / 8] >> (g % 8)) & 1;
            let qpack = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
            let base = bi * 256 + (g / 16) * 64 + (g % 16);
            for p in 0..4 {
                let q = ((qpack >> (2 * p)) & 3) as i32;
                out[base + p * 16] = d * (q - 1) as f32;
            }
        }
    }
    out
}

fn deq_tq2_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(66).take(n / 256).enumerate() {
        let d = rd_f16(&blk[64..66]);
        for jg in 0..2 {               // two 128-value halves (byte groups 0..32, 32..64)
            for l in 0..4 {            // the 4 two-bit lanes in each byte
                for m in 0..32 {
                    let code = ((blk[jg * 32 + m] >> (2 * l)) & 3) as i32;
                    out[bi * 256 + jg * 128 + l * 32 + m] = d * (code - 1) as f32;
                }
            }
        }
    }
    out
}

/// The OCP **E2M1** value table (1 sign bit, 2 exponent bits, 1 mantissa bit), stored **doubled**:
/// the element's true value is `KVALUES_MXFP4_2X[code] / 2`, i.e. `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4,
/// ±6}`. The same magnitudes appear in HF transformers' `integrations/mxfp4.py::FP4_VALUES`, so the
/// two independent references agree on the table.
///
/// Two details are not free choices, and both were read out of ggml rather than reasoned about:
///
/// * **Code 8 is `+0.0`, not `−0.0`.** HF's list writes `-0.0` in slot 8; ggml's table is integral,
///   so the sign never appears, and a bitwise diff sees the difference that `==` cannot.
/// * **The doubling is load-bearing at the top of the exponent range.** Pairing this table with the
///   *half* scale below is ggml's own arithmetic, and it is the reason `e = 255, code = 1` is the
///   finite `2^127` rather than `inf` — see `e8m0_half_to_f32`.
const KVALUES_MXFP4_2X: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

/// The **E8M0** shared scale, with one power of two held back: the block's first byte is a raw f32
/// exponent field, so the scale is `2^(e − 127)`, and this returns `2^(e − 128)` to be paired with the
/// doubled value table above.
///
/// Holding the factor back is not cosmetic. `2^(e − 127)` at `e = 255` is `2^128`, which **is not
/// representable in f32** — computing the scale first and multiplying second turns every element of
/// such a block into `±inf` or `NaN`, where ggml (and `ldexp`, and HF's `torch.ldexp`) return finite
/// values for the small codes: `(e=255, code=1) → 0x7f000000 = 2^127`. This form never forms the
/// unrepresentable intermediate, because `2^(e − 128)` tops out at `2^127`, which is finite.
///
/// The bottom end needs the bits built by hand too: `e = 0` and `e = 1` give `2^−128` and `2^−127`,
/// both **subnormal** (f32's smallest normal is `2^−126`), so their exponent field is 0 and the value
/// lives in the mantissa. Every product with the doubled table stays ≥ `2^−149` and is exact.
///
/// **This function is where the first version of this file was wrong**, and no amount of reading the
/// spec would have said so: a probe of one code (`1.0`) across all 256 exponents agreed with the naive
/// `2^(e − 127)` everywhere, because `1.0 · 2^128` overflows to `inf` and `inf` is what ggml returns
/// for *that* code. Only the full 16-code × edge-exponent grid separates the two.
fn e8m0_half_to_f32(e: u8) -> f32 {
    match e {
        0 => f32::from_bits(1 << 21), // 2^−128 = 2^21 · 2^−149 (subnormal)
        1 => f32::from_bits(1 << 22), // 2^−127 = 2^22 · 2^−149 (subnormal)
        _ => f32::from_bits((e as u32 - 1) << 23),
    }
}

/// **MXFP4** (OCP Microscaling FP4, ggml type 39) — the format GPT-OSS ships in. 17-byte block,
/// 32 values: `u8 e` (the E8M0 shared exponent) then `qs[16]`, two 4-bit E2M1 codes per byte.
/// Element `i` takes the **low** nibble of `qs[i]` and element `i+16` the **high** nibble — the same
/// low-half / high-half split as Q4_0, and *not* the lo/hi *interleave* HF transformers uses in its
/// own packing (the value table is shared between the two; the byte order is not).
///
/// value = `(2·E2M1[code]) · 2^(e − 128)`, which is `E2M1[code] · 2^(e − 127)` wherever that is
/// representable and correct where it is not.
fn deq_mxfp4(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(17).take(n / 32).enumerate() {
        let d = e8m0_half_to_f32(blk[0]);
        for i in 0..16 {
            let byte = blk[1 + i];
            out[bi * 32 + i] = KVALUES_MXFP4_2X[(byte & 0x0F) as usize] * d;
            out[bi * 32 + i + 16] = KVALUES_MXFP4_2X[(byte >> 4) as usize] * d;
        }
    }
    out
}

/// **Q1_0** — PrismML "Bonsai" 1-bit (also mainline llama.cpp type 41). 128-value block = `f16 d`
/// then `qs[16]`; element j → byte j/8, bit j%8 (LSB-first); value = bit ? +d : −d. 1.125 bpw.
fn deq_q1_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(18).take(n / 128).enumerate() {
        let d = rd_f16(&blk[0..2]);
        for j in 0..128 {
            let bit = (blk[2 + j / 8] >> (j % 8)) & 1;
            out[bi * 128 + j] = if bit == 1 { d } else { -d };
        }
    }
    out
}

/// **Q2_0** — PrismML "Ternary Bonsai" (group-128). 128-value block = `f16 d` then `qs[32]`;
/// element j → byte j/4, bits (j%4)*2 (LSB-first, 4/byte); value = (q−1)·d, q ∈ {0..3}
/// (q=3 → +2d is reserved/unused for ternary, but decode the arithmetic form). 2.125 bpw on disk.
fn deq_q2_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, blk) in raw.chunks_exact(34).take(n / 128).enumerate() {
        let d = rd_f16(&blk[0..2]);
        for j in 0..128 {
            let q = ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) as i32;
            out[bi * 128 + j] = (q - 1) as f32 * d;
        }
    }
    out
}

/// Quantize to Q1_0 (PrismML 1-bit): d = mean(|x|) over the 128-group; bit = sign(x) ≥ 0.
pub fn quant_q1_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(128) {
        let d = blk.iter().map(|v| v.abs()).sum::<f32>() / 128.0;
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qs = [0u8; 16];
        for (j, &v) in blk.iter().enumerate() {
            if v >= 0.0 { qs[j / 8] |= 1 << (j % 8); }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantize to Q2_0 (PrismML ternary): d = amax over the 128-group; q = clamp(round(x/d)+1, 0, 3).
pub fn quant_q2_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(128) {
        let d = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qs = [0u8; 32];
        for (j, &v) in blk.iter().enumerate() {
            let q = if d != 0.0 { ((v / d).round() as i32 + 1).clamp(0, 3) } else { 1 };
            qs[j / 4] |= (q as u8) << ((j % 4) * 2);
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Encode ternary values (as codes {−1,0,+1}) into a TQ2_0 block — for test fixtures / writing GGUF.
pub fn quant_tq2_0(codes: &[i8], d: f32) -> Vec<u8> {
    let mut qs = vec![0u8; 64];
    for jg in 0..2 {
        for l in 0..4 {
            for m in 0..32 {
                let code = (codes[jg * 128 + l * 32 + m] + 1) as u8 & 3; // {−1,0,1}→{0,1,2}
                qs[jg * 32 + m] |= code << (2 * l);
            }
        }
    }
    qs.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    qs
}

// ---- quantizers (used to build test fixtures; also handy for writing GGUF) ----
pub fn quant_q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for i in 0..32 {
            let q = if d != 0.0 { (blk.get(i).copied().unwrap_or(0.0) / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
            out.push(q as u8);
        }
    }
    out
}
pub fn quant_q4_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 8.0;
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for i in 0..16 {
            let q = |v: f32| -> u8 { if d != 0.0 { ((v / d).round().clamp(-8.0, 7.0) as i32 + 8) as u8 & 0x0F } else { 8 } };
            out.push(q(blk.get(i).copied().unwrap_or(0.0)) | (q(blk.get(i + 16).copied().unwrap_or(0.0)) << 4));
        }
    }
    out
}
/// Q4_1 encoder (affine, min-offset): `d = (max−min)/15`, `m = min`, `q = round((v−m)/d)∈0..15`.
pub fn quant_q4_1(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(32) {
        let mn = blk.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
        let mx = blk.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(0.0);
        let d = (mx - mn) / 15.0;
        // store f16 d/m, then quantize against the SAME rounded d/m the kernels will read back
        let (df, mf) = (f16::from_f32(d).to_f32(), f16::from_f32(mn).to_f32());
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(mn).to_le_bytes());
        for i in 0..16 {
            let q = |v: f32| -> u8 { if df != 0.0 { ((v - mf) / df).round().clamp(0.0, 15.0) as u8 & 0x0F } else { 0 } };
            out.push(q(blk.get(i).copied().unwrap_or(0.0)) | (q(blk.get(i + 16).copied().unwrap_or(0.0)) << 4));
        }
    }
    out
}
/// Q5_0 encoder (symmetric 5-bit): `d = amax/16`, code = `round(v/d)∈−16..15` offset by +16 → 0..31;
/// low 4 bits go in `qs`, the 5th bit into `qh` (bit i for value i, bit i+16 for value i+16).
pub fn quant_q5_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 16.0;
        let df = f16::from_f32(d).to_f32();
        let enc = |v: f32| -> u32 { if df != 0.0 { (((v / df).round().clamp(-16.0, 15.0) as i32) + 16) as u32 } else { 16 } };
        let mut qh: u32 = 0;
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let c0 = enc(blk.get(i).copied().unwrap_or(0.0));       // 0..31
            let c1 = enc(blk.get(i + 16).copied().unwrap_or(0.0));
            nibbles[i] = ((c0 & 0xF) | ((c1 & 0xF) << 4)) as u8;
            qh |= ((c0 >> 4) & 1) << i;
            qh |= ((c1 >> 4) & 1) << (i + 16);
        }
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&nibbles);
    }
    out
}
/// Q5_1 encoder (affine 5-bit): `d = (max−min)/31`, `m = min`, code = `round((v−m)/d)∈0..31`; low 4
/// bits in `qs`, 5th bit in `qh` (bit i for value i, bit i+16 for value i+16).
pub fn quant_q5_1(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in x.chunks(32) {
        let mn = blk.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
        let mx = blk.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(0.0);
        let d = (mx - mn) / 31.0;
        let (df, mf) = (f16::from_f32(d).to_f32(), f16::from_f32(mn).to_f32());
        let enc = |v: f32| -> u32 { if df != 0.0 { ((v - mf) / df).round().clamp(0.0, 31.0) as u32 } else { 0 } };
        let mut qh: u32 = 0;
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let c0 = enc(blk.get(i).copied().unwrap_or(0.0));
            let c1 = enc(blk.get(i + 16).copied().unwrap_or(0.0));
            nibbles[i] = ((c0 & 0xF) | ((c1 & 0xF) << 4)) as u8;
            qh |= ((c0 >> 4) & 1) << i;
            qh |= ((c1 >> 4) & 1) << (i + 16);
        }
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(mn).to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&nibbles);
    }
    out
}

/// **MXFP4 against llama.cpp's own dequantizer, bit for bit.**
///
/// A dequant of a microscaling format is *exact-representable* — every output is a table value times
/// a power of two — so "close enough" is not the bar here. Every golden below was captured from the
/// shipped `libggml-base`, through the same `ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float` that
/// `llama-eval-callback` prints, and is compared on **f32 bit patterns**, not on `==`. Bits matter:
/// ggml emits `+0.0` for code 8, HF transformers' `FP4_VALUES` writes `-0.0` there, and `==` cannot
/// tell those apart.
///
/// `GG_GRID` pins the value table, the E8M0 bias and both saturating ends of the exponent range.
/// `REAL_*` pins what the table checks cannot: the intra-block nibble order and the 17-byte stride,
/// on bytes lifted out of an actual MXFP4 GGUF (`llama-quantize --tensor-type attn_q=mxfp4` over
/// Qwen2.5-0.5B) rather than a synthetic pattern. The full-tensor version of that same diff —
/// 11.1 M elements over five tensors — is `examples/mxfp4_ref_diff.rs`, which takes every path from
/// argv so it can be re-run against GPT-OSS itself when a checkpoint is on hand.
#[cfg(test)]
mod mxfp4_tests {
    use super::*;

    // ---- GOLDEN A: 10 scale bytes x 16 codes, captured from ggml ----
    const GG_E: [u8; 10] = [0, 1, 2, 64, 126, 127, 128, 200, 254, 255];
    const GG_GRID: [u32; 160] = [
        0x00000000, 0x00200000, 0x00400000, 0x00600000, 0x00800000, 0x00c00000, 0x01000000, 0x01400000,
        0x00000000, 0x80200000, 0x80400000, 0x80600000, 0x80800000, 0x80c00000, 0x81000000, 0x81400000, // e=0
        0x00000000, 0x00400000, 0x00800000, 0x00c00000, 0x01000000, 0x01400000, 0x01800000, 0x01c00000,
        0x00000000, 0x80400000, 0x80800000, 0x80c00000, 0x81000000, 0x81400000, 0x81800000, 0x81c00000, // e=1
        0x00000000, 0x00800000, 0x01000000, 0x01400000, 0x01800000, 0x01c00000, 0x02000000, 0x02400000,
        0x00000000, 0x80800000, 0x81000000, 0x81400000, 0x81800000, 0x81c00000, 0x82000000, 0x82400000, // e=2
        0x00000000, 0x1f800000, 0x20000000, 0x20400000, 0x20800000, 0x20c00000, 0x21000000, 0x21400000,
        0x00000000, 0x9f800000, 0xa0000000, 0xa0400000, 0xa0800000, 0xa0c00000, 0xa1000000, 0xa1400000, // e=64
        0x00000000, 0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000, 0x3fc00000, 0x40000000, 0x40400000,
        0x00000000, 0xbe800000, 0xbf000000, 0xbf400000, 0xbf800000, 0xbfc00000, 0xc0000000, 0xc0400000, // e=126
        0x00000000, 0x3f000000, 0x3f800000, 0x3fc00000, 0x40000000, 0x40400000, 0x40800000, 0x40c00000,
        0x00000000, 0xbf000000, 0xbf800000, 0xbfc00000, 0xc0000000, 0xc0400000, 0xc0800000, 0xc0c00000, // e=127
        0x00000000, 0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40c00000, 0x41000000, 0x41400000,
        0x00000000, 0xbf800000, 0xc0000000, 0xc0400000, 0xc0800000, 0xc0c00000, 0xc1000000, 0xc1400000, // e=128
        0x00000000, 0x63800000, 0x64000000, 0x64400000, 0x64800000, 0x64c00000, 0x65000000, 0x65400000,
        0x00000000, 0xe3800000, 0xe4000000, 0xe4400000, 0xe4800000, 0xe4c00000, 0xe5000000, 0xe5400000, // e=200
        0x00000000, 0x7e800000, 0x7f000000, 0x7f400000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000,
        0x00000000, 0xfe800000, 0xff000000, 0xff400000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, // e=254
        0x00000000, 0x7f000000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000,
        0x00000000, 0xff000000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, // e=255
    ];

    // ---- GOLDEN B: first 8 blocks of blk.0.attn_q.weight in a real MXFP4 gguf ----
    // Scale bytes are 0x77,0x77,0x77,0x77,0x77,0x78,0x76,0x78 — three distinct exponents, so a block
    // read at the wrong stride lands on the wrong scale and cannot pass by luck.
    const REAL_RAW: [u8; 136] = [
        0x77, 0x99, 0x49, 0xd4, 0x54, 0xd1, 0x0a, 0x02, 0xa0, 0x05, 0xa9, 0x7a, 0x96, 0x33, 0xc0, 0x9c, 0xb0,
        0x77, 0xb1, 0xd1, 0xd0, 0xcd, 0x41, 0xc1, 0xba, 0xa9, 0x76, 0x27, 0xcd, 0x6a, 0x19, 0xa5, 0x14, 0xa1,
        0x77, 0x4d, 0x1e, 0x10, 0x05, 0x15, 0x22, 0xa2, 0xb4, 0x24, 0x2f, 0x53, 0x9c, 0xda, 0xa7, 0x9b, 0x09,
        0x77, 0xd3, 0x12, 0xa6, 0xb0, 0xfd, 0x05, 0xca, 0x2a, 0xa0, 0x44, 0xa7, 0x63, 0x7a, 0x6e, 0x32, 0x90,
        0x77, 0x2c, 0x6c, 0x74, 0xa3, 0xa6, 0xa3, 0xcc, 0x36, 0x2d, 0xbe, 0xcf, 0xcc, 0x0a, 0xd1, 0xd5, 0x2b,
        0x78, 0x9c, 0x09, 0x4c, 0x39, 0x23, 0x90, 0xa5, 0x11, 0xc9, 0x01, 0x13, 0xa3, 0x92, 0x99, 0x61, 0x41,
        0x76, 0x67, 0x5c, 0x73, 0x03, 0x4d, 0xf7, 0x44, 0xea, 0xe5, 0x49, 0x96, 0xdf, 0xcd, 0x79, 0xed, 0x6b,
        0x78, 0x0b, 0x00, 0x31, 0x90, 0x0a, 0x01, 0xa2, 0x03, 0x99, 0x1a, 0x03, 0x1c, 0xa0, 0x31, 0x41, 0x26,
    ];
    const REAL_GGML: [u32; 256] = [
        0xbb000000, 0xbb000000, 0x3c000000, 0x3c000000, 0x3b000000, 0xbb800000, 0x3b800000, 0x00000000,
        0x3c400000, 0xbb000000, 0xbb800000, 0x3c800000, 0x3bc00000, 0x00000000, 0xbc000000, 0x00000000,
        0xbb000000, 0x3c000000, 0xbc400000, 0x3c400000, 0xbc400000, 0x00000000, 0x00000000, 0xbb800000,
        0x00000000, 0xbb800000, 0x3cc00000, 0xbb000000, 0x3bc00000, 0xbc000000, 0xbb000000, 0xbbc00000,
        0x3b000000, 0x3b000000, 0x00000000, 0xbc400000, 0x3b000000, 0x3b000000, 0xbb800000, 0xbb000000,
        0x3c800000, 0x3cc00000, 0xbc400000, 0xbb800000, 0xbb000000, 0x3c400000, 0x3c000000, 0x3b000000,
        0xbbc00000, 0xbc400000, 0xbc400000, 0xbc000000, 0x3c000000, 0xbc000000, 0xbbc00000, 0xbb800000,
        0x3cc00000, 0x3b800000, 0xbc000000, 0x3c800000, 0x3b000000, 0xbb800000, 0x3b000000, 0xbb800000,
        0xbc400000, 0xbc800000, 0x00000000, 0x3c400000, 0x3c400000, 0x3b800000, 0x3b800000, 0x3c000000,
        0x3c000000, 0xbcc00000, 0x3bc00000, 0xbc000000, 0xbb800000, 0x3cc00000, 0xbbc00000, 0xbb000000,
        0x3c000000, 0x3b000000, 0x3b000000, 0x00000000, 0x3b000000, 0x3b800000, 0xbb800000, 0xbbc00000,
        0x3b800000, 0x3b800000, 0x3c400000, 0xbb000000, 0xbc400000, 0xbb800000, 0xbb000000, 0x00000000,
        0x3bc00000, 0x3b800000, 0x3c800000, 0x00000000, 0xbc400000, 0x3c400000, 0xbb800000, 0xbb800000,
        0x00000000, 0x3c000000, 0x3cc00000, 0x3bc00000, 0xbb800000, 0xbc800000, 0x3b800000, 0x00000000,
        0xbc400000, 0x3b000000, 0xbb800000, 0xbbc00000, 0xbcc00000, 0x00000000, 0xbc000000, 0x3b800000,
        0xbb800000, 0x3c000000, 0xbb800000, 0x3c800000, 0x3cc00000, 0x3c800000, 0x3bc00000, 0xbb000000,
        0xbc000000, 0xbc000000, 0x3c000000, 0x3bc00000, 0x3c800000, 0x3bc00000, 0xbc000000, 0x3c800000,
        0xbc400000, 0xbc800000, 0xbcc00000, 0xbc000000, 0xbb800000, 0x3b000000, 0x3c400000, 0xbbc00000,
        0x3b800000, 0x3c800000, 0x3cc00000, 0xbb800000, 0xbb800000, 0xbb800000, 0xbc000000, 0x3bc00000,
        0x3b800000, 0xbbc00000, 0xbc000000, 0xbc000000, 0x00000000, 0xbc400000, 0xbc400000, 0x3b800000,
        0xbc800000, 0xbb800000, 0xbc800000, 0xbb800000, 0x3c400000, 0x00000000, 0x3cc00000, 0x3b800000,
        0xbb800000, 0x3b800000, 0x3c400000, 0x3c400000, 0x3c000000, 0xbb800000, 0x3b800000, 0x3b800000,
        0xbb800000, 0x00000000, 0x3c800000, 0x3c400000, 0x3c000000, 0xbb800000, 0xbc000000, 0x3b800000,
        0xbc800000, 0x00000000, 0x3b800000, 0xbc000000, 0xbb800000, 0xbb800000, 0x3d000000, 0x3c800000,
        0x3c400000, 0xbb800000, 0x3b400000, 0x3b400000, 0xbbc00000, 0x3c400000, 0x3b800000, 0xbb000000,
        0x3bc00000, 0xba800000, 0x3c000000, 0xbc400000, 0xbbc00000, 0xba800000, 0xbbc00000, 0xbb400000,
        0x3c000000, 0x3bc00000, 0x3c400000, 0x00000000, 0x3b800000, 0xbc400000, 0x3b800000, 0xbc000000,
        0xbc000000, 0x3b800000, 0xba800000, 0xbbc00000, 0xbb800000, 0x3c400000, 0xbc000000, 0x3c000000,
        0xbc400000, 0x00000000, 0x3b800000, 0x00000000, 0xbc000000, 0x3b800000, 0x3c000000, 0x3c400000,
        0xbb800000, 0xbc000000, 0x3c400000, 0xbc800000, 0x00000000, 0x3b800000, 0x3b800000, 0x3d000000,
        0x00000000, 0x00000000, 0x3c400000, 0xbb800000, 0x00000000, 0x00000000, 0xbc000000, 0x00000000,
        0xbb800000, 0x3b800000, 0x00000000, 0x3b800000, 0xbc000000, 0x3c400000, 0x3c800000, 0x3c000000,
    ];

    /// The value table and the E8M0 bias, at both saturating ends of the exponent range.
    ///
    /// Each code is placed twice — in the low nibble of `qs[0]` (element 0) and in the high nibble of
    /// `qs[0]` (element 16) — so the two nibble halves are checked against the same golden and a
    /// half-specific mistake cannot hide.
    #[test]
    fn mxfp4_grid_is_bit_identical_to_ggml() {
        for (ei, &e) in GG_E.iter().enumerate() {
            for code in 0u8..16 {
                let want = GG_GRID[ei * 16 + code as usize];

                let mut blk = [0u8; 17];
                blk[0] = e;
                blk[1] = code; // low nibble -> element 0
                let lo = deq_raw(&blk, 32, MXFP4).unwrap();
                assert_eq!(lo[0].to_bits(), want,
                    "low nibble: e={e} code={code}: ferric 0x{:08x} ({}) vs ggml 0x{want:08x} ({})",
                    lo[0].to_bits(), lo[0], f32::from_bits(want));

                let mut blk = [0u8; 17];
                blk[0] = e;
                blk[1] = code << 4; // high nibble -> element 16
                let hi = deq_raw(&blk, 32, MXFP4).unwrap();
                assert_eq!(hi[16].to_bits(), want,
                    "high nibble: e={e} code={code}: ferric 0x{:08x} vs ggml 0x{want:08x}",
                    hi[16].to_bits());
                // ...and the other half of that byte is code 0, which is +0.0 in ggml's table.
                assert_eq!(hi[0].to_bits(), 0x0000_0000, "e={e} code={code}: element 0 should be +0.0");
            }
        }
    }

    /// Real bytes out of a real MXFP4 GGUF, against ggml's dequant of the same bytes. This is the
    /// check that pins the *layout*: element `i` takes the low nibble of `qs[i]` and `i+16` the high
    /// nibble (not HF's lo/hi interleave), and blocks stride by 17 bytes.
    #[test]
    fn mxfp4_real_gguf_blocks_are_bit_identical_to_ggml() {
        let got = deq_raw(&REAL_RAW, 256, MXFP4).unwrap();
        assert_eq!(got.len(), 256);
        let mut ndiff = 0;
        for i in 0..256 {
            if got[i].to_bits() != REAL_GGML[i] {
                ndiff += 1;
                if ndiff == 1 {
                    panic!("element {i} (block {}, lane {}): ferric {} (0x{:08x}) vs ggml {} (0x{:08x})",
                        i / 32, i % 32, got[i], got[i].to_bits(),
                        f32::from_bits(REAL_GGML[i]), REAL_GGML[i]);
                }
            }
        }
        assert_eq!(ndiff, 0);
    }

    /// The on-disk stride. 426_496 is the byte length `llama-gguf` reports for the 896x896
    /// `blk.0.attn_q.weight` of the requantized file, so this is a measured number, not arithmetic
    /// restated: 802_816 values / 32 = 25_088 blocks x 17 bytes.
    /// The two E8M0 biases must stay a factor of two apart, and neither may drift into the other.
    ///
    /// `e8m0_half_to_f32` is ggml's `2^(e-128)`, paired with a DOUBLED E2M1 table. `e8m0_bias127` is
    /// OCP's `2^(e-127)`, paired with undoubled E4M3. Using one where the other belongs scales every
    /// weight by 2x or 0.5x — uniform, so the model still produces fluent text, and nothing errors.
    #[test]
    fn the_two_e8m0_biases_differ_by_exactly_two() {
        for e in 1..=0xFEu8 {
            let (ggml, ocp) = (e8m0_half_to_f32(e), e8m0_bias127(e));
            assert!(ggml.is_finite() && ocp.is_finite(), "e={e}: both must be finite in range");
            assert_eq!(ocp, ggml * 2.0, "e={e}: OCP must be exactly twice ggml's");
        }
        // Landmarks that pin the absolute scale, not just the ratio.
        assert_eq!(e8m0_bias127(127), 1.0, "bias 127 means e=127 is unity");
        assert_eq!(e8m0_bias127(128), 2.0);
        assert_eq!(e8m0_bias127(126), 0.5);
        // Deliberate divergence from the V4 fork, documented on `e8m0_bias127`: ggml's macro has its
        // NaN branch commented out so 0xFF decodes to +Inf there; we keep OCP's NaN so a poisoned
        // scale is caught instead of multiplied through.
        assert!(e8m0_bias127(0xFF).is_nan(), "OCP reserves all-ones for NaN");
    }

    /// The block decode: one E8M0 scale byte, THEN 128 E4M3 elements — scale FIRST.
    ///
    /// This layout is verified against a real V4 file (byte-level phase statistics of two tensors of
    /// `DeepSeek-V4-Flash-FP4-FP8-native.gguf`, two independent probes) and the fork's block struct
    /// (`{ uint8_t e; uint8_t qs[128]; }`). The crate originally shipped the OPPOSITE order as a
    /// labelled assumption; the fixture below is deliberately asymmetric so a payload-first decoder
    /// cannot pass it: read back-to-front, block 0's "scale" would be its last payload byte (0x40,
    /// not an exponent of 1) and its first element the byte 127 (= E4M3 NaN).
    #[test]
    fn f8_e4m3_b128_decodes_a_block_under_its_shared_scale() {
        let mut raw = vec![0u8; F8_E4M3_B128_BYTES * 2];
        // Block 0: scale byte FIRST (127 -> 2^0), then values 1.0 (0x38) and 2.0 (0x40).
        raw[0] = 127; raw[1] = 0x38; raw[2] = 0x40;
        // ...and a nonzero LAST payload byte, where the old layout looked for the scale.
        raw[128] = 0x40;
        // Block 1: the same leading payload bytes at scale 2^1, so every value doubles.
        raw[F8_E4M3_B128_BYTES] = 128;
        raw[F8_E4M3_B128_BYTES + 1] = 0x38; raw[F8_E4M3_B128_BYTES + 2] = 0x40;
        let v = deq_f8_e4m3_b128(&raw, 256).expect("two whole blocks");
        assert_eq!(v.len(), 256);
        assert_eq!((e4m3_to_f32(0x38), e4m3_to_f32(0x40)), (1.0, 2.0), "fixture: the chosen bytes");
        assert_eq!((v[0], v[1]), (1.0, 2.0), "block 0 at 2^0, elements from bytes 1..=128");
        assert_eq!(v[127], 2.0, "byte 128 is the LAST ELEMENT of block 0, not its scale");
        assert_eq!((v[128], v[129]), (2.0, 4.0), "block 1 at 2^1 — the scale is PER BLOCK");
        assert_eq!(v[3], 0.0, "0x00 is zero whatever the scale");

        // Refusals, not silent truncation.
        assert!(deq_f8_e4m3_b128(&raw, 100).is_err(), "a partial block must be refused");
        assert!(deq_f8_e4m3_b128(&raw[..10], 256).is_err(), "short input must be refused");
    }

    /// Type 42 is claimed three ways and must be resolved by STRIDE, never by preference.
    /// The parse path itself must resolve type 42 — a tested resolver nothing calls is the
    /// written-ahead-of-its-wiring gap this tree has now produced three times.
    ///
    /// Built as a REAL in-memory GGUF (header, tensor table, data), not a mocked call: the claim is
    /// about what `parse` hands downstream, so the parser is the subject.
    #[test]
    fn parsing_a_file_resolves_type_42_by_stride_and_refuses_tiny_ambiguity() {
        fn gguf_with_type42(n_elems: u64, data_bytes: usize) -> Vec<u8> {
            let mut b: Vec<u8> = Vec::new();
            b.extend(b"GGUF");
            b.extend(3u32.to_le_bytes());
            b.extend(1u64.to_le_bytes());               // one tensor
            b.extend(0u64.to_le_bytes());               // no kv
            let name = b"t";
            b.extend((name.len() as u64).to_le_bytes());
            b.extend(name);
            b.extend(1u32.to_le_bytes());               // n_dims
            b.extend(n_elems.to_le_bytes());
            b.extend(42u32.to_le_bytes());              // the contested id
            b.extend(0u64.to_le_bytes());               // offset
            while b.len() % 32 != 0 { b.push(0); }      // default alignment
            b.extend(std::iter::repeat(0u8).take(data_bytes));
            b
        }
        // 128 blocks at 129 B: unambiguous within one 32-byte alignment -> resolves to F8 (1042).
        let f8 = parse(gguf_with_type42(128 * 128, 128 * 129)).expect("F8 stride must load");
        assert_eq!(f8.tensors[0].ggml_type, 1042,
                   "the PARSER must hand downstream the resolved id, not the contested 42");
        // Same element count at 34 B/block: PrismML ternary keeps meaning what it always meant.
        let q2 = parse(gguf_with_type42(128 * 128, 128 * 34)).expect("Q2_0 stride must load");
        assert_eq!(q2.tensors[0].ggml_type, 42);
        // Mainline ggml-org Q2_0 (18 B / 64 values = 36 B / 128): refused BY NAME, no decoder exists.
        let e = match parse(gguf_with_type42(128 * 128, 128 * 36)) {
            Err(e) => e, Ok(_) => panic!("a mainline-Q2_0 stride must refuse, not load"),
        };
        assert!(e.contains("64 values / 18 bytes") || e.contains("mainline"),
                "mainline must be refused by name, got: {e}");
        // ONE block: 34 vs 36 differ by 2 bytes — inside one alignment, so it must REFUSE as
        // ambiguous rather than first-match to PrismML. This is the case a slack-window resolver
        // silently gets wrong.
        let e = match parse(gguf_with_type42(128, 36)) {
            Err(e) => e, Ok(_) => panic!("a one-block ambiguous tensor must refuse, not first-match"),
        };
        assert!(e.contains("too small to disambiguate"), "got: {e}");
    }
    #[test]
    fn type_42_resolves_by_stride_and_refuses_when_it_matches_neither() {
        let n = 1024usize;                 // 8 blocks of 128
        assert_eq!(resolve_type_42(n, 8 * 34).unwrap(), 42, "34 bytes/block is PrismML Q2_0");
        assert_eq!(resolve_type_42(n, 8 * F8_E4M3_B128_BYTES).unwrap(), 1042, "129 is F8_E4M3_B128");
        // Mainline ggml-org master ALSO assigns 42 (its own Q2_0: 64 values / 18 bytes = 36 bytes
        // per 128). This crate cannot decode that layout — the refusal must NAME it, because 36 is
        // only 2 bytes per 128 values away from PrismML's 34 and a generic message would send the
        // reader hunting the wrong format.
        let e = resolve_type_42(n, n / 64 * 18).unwrap_err();
        assert!(e.contains("mainline") && e.contains("64 values / 18 bytes"), "unexpected: {e}");
        // Anything else is a format this crate does not know, and guessing would mis-decode silently.
        let e = resolve_type_42(n, 8 * 64).unwrap_err();
        assert!(e.contains("none of the three"), "unexpected: {e}");
        assert!(resolve_type_42(100, 3400).is_err(), "not a multiple of 128 elements");
    }

    /// The internal id `resolve_type_42` hands back must be usable end-to-end: `type_size` computes
    /// the 129-byte stride and `deq_raw` routes to the F8 decoder. Fixture chosen so a scale/payload
    /// swap cannot pass: both blocks decode element 0 to 2.0 via DIFFERENT (scale, element) pairs.
    #[test]
    fn f8_internal_id_dispatches_through_type_size_and_deq_raw() {
        assert_eq!(type_size(F8_E4M3_B128, 256).unwrap(), 2 * F8_E4M3_B128_BYTES);
        let mut raw = vec![0u8; 2 * F8_E4M3_B128_BYTES];
        raw[0] = 128; raw[1] = 0x38;                                       // 2^1 x 1.0
        raw[F8_E4M3_B128_BYTES] = 127; raw[F8_E4M3_B128_BYTES + 1] = 0x40; // 2^0 x 2.0
        let v = deq_raw(&raw, 256, F8_E4M3_B128).expect("routed to deq_f8_e4m3_b128");
        assert_eq!((v[0], v[128]), (2.0, 2.0));
        assert_eq!(v.len(), 256);
    }

    #[test]
    fn mxfp4_type_size_matches_the_file() {
        assert_eq!(type_size(MXFP4, 32).unwrap(), 17);
        assert_eq!(type_size(MXFP4, 802_816).unwrap(), 426_496);
        // 0.53125 bytes/elem = 4.25 bits/weight: 4 bits of E2M1 plus 8 shared exponent bits per 32.
        assert_eq!(type_size(MXFP4, 32).unwrap() as f64 * 8.0 / 32.0, 4.25);
    }

    /// `deq_raw` must consume the block stride, not the element count: 8 blocks of input yield 256
    /// values and the last block's distinct scale (0x78) has to reach the last lane.
    /// E4M3 must match PyTorch's `float8_e4m3fn` **exactly, for all 256 bytes**.
    ///
    /// The table is generated, so this test is not checking arithmetic — it is checking that the
    /// generated values are the ones the format actually defines, and pinning the edges a hand-rolled
    /// decoder gets wrong. MXFP4 shipped with an off-by-one in exactly this kind of edge (2^(e-127) at
    /// e=255 overflows where ggml returns finite) and only an exhaustive comparison found it.
    #[test]
    fn e4m3_matches_the_reference_over_every_byte() {
        // Landmarks, taken from torch and asserted here so a regenerated table cannot drift silently.
        assert_eq!(e4m3_to_f32(0x00), 0.0, "zero");
        assert!(e4m3_to_f32(0x80).is_sign_negative() && e4m3_to_f32(0x80) == 0.0, "negative zero");
        assert_eq!(e4m3_to_f32(0x01), 0.001953125f32, "smallest subnormal");
        assert_eq!(e4m3_to_f32(0x7E), 448.0, "max finite magnitude is 448, not 240 or 480");
        assert_eq!(e4m3_to_f32(0xFE), -448.0, "and symmetric");
        assert!(e4m3_to_f32(0x7F).is_nan() && e4m3_to_f32(0xFF).is_nan(), "the two NaN encodings");

        // The `fn` variant has NO infinities. A decoder that reuses IEEE binary8 logic produces them,
        // and an inf weight silently poisons every product it touches.
        let infs = (0..=255u8).filter(|&b| e4m3_to_f32(b).is_infinite()).count();
        assert_eq!(infs, 0, "float8_e4m3fn defines no infinities; found {infs}");

        // Exactly two NaNs, and every other byte finite.
        let nans = (0..=255u8).filter(|&b| e4m3_to_f32(b).is_nan()).count();
        assert_eq!(nans, 2, "0x7F and 0xFF only");
        assert_eq!((0..=255u8).filter(|&b| e4m3_to_f32(b).is_finite()).count(), 254);

        // Monotone over the positive range: byte order is value order for a sign-magnitude float, and
        // a table with a transposed pair would still pass every landmark above.
        // Stops at 0x7E: 0x7F is NaN, and NaN compares false against everything, so including it
        // would fail here for the right reason and the wrong one.
        for b in 1..0x7Eu8 {
            assert!(e4m3_to_f32(b) < e4m3_to_f32(b + 1),
                    "0x{b:02x} -> {} must be < 0x{:02x} -> {}", e4m3_to_f32(b), b + 1, e4m3_to_f32(b + 1));
        }
    }

    #[test]
    fn mxfp4_last_block_uses_its_own_scale() {
        let got = deq_raw(&REAL_RAW, 256, MXFP4).unwrap();
        // block 7's scale byte is 0x78 -> 2^(120-127) = 2^-7; its qs[15] high nibble is 0x2 -> +1.0.
        assert_eq!(REAL_RAW[7 * 17], 0x78);
        assert_eq!(REAL_RAW[7 * 17 + 16] >> 4, 0x2);
        assert_eq!(got[7 * 32 + 31].to_bits(), (1.0f32 * 2f32.powi(-7)).to_bits());
        assert_eq!(got[7 * 32 + 31].to_bits(), REAL_GGML[255]);
    }

    /// A type id whose block layout this crate gets wrong must REFUSE, not read at the wrong stride.
    ///
    /// The concrete risk this guards: GGUF type ids in the 40s are contested. This crate maps id 42 to
    /// a group-128 / 34-byte ternary layout (2.125 bpw); mainline `llama-quantize` advertises its own
    /// `Q2_0` as "2.25 bpw quantization (group 64)". A file carries only the id, and every reader here
    /// computes the byte length from `type_size` and reads that many bytes at the declared offset with
    /// no other check. Wrong stride in, plausible garbage out, no error.
    ///
    /// Both directions matter and they fail differently: too LARGE overruns into the next tensor, too
    /// SMALL reads a short prefix of every block while leaving an implausible hole.
    #[test]
    fn a_stride_that_disagrees_with_the_file_is_refused_in_both_directions() {
        // Two 1024-element Q8_0 tensors laid out back to back: 1024/32*34 = 1088 bytes each.
        let t = |name: &str, off: u64| TensorInfo {
            name: name.into(), dims: vec![1024], ggml_type: Q8_0, offset: off,
        };
        let tensors = vec![t("a", 0), t("b", 1088)];
        let total = 2176;

        check_declared_strides(&tensors, total, 32)
            .expect("the honest layout must load: 1088 bytes each, exactly back to back");

        // Too large: 'a' would need more than the 1088 the file leaves before 'b'.
        let overrun = vec![t("a", 0), t("b", 700)];
        let e = check_declared_strides(&overrun, total, 32).expect_err("must refuse an overrun");
        assert!(e.contains("needs 1088 bytes") && e.contains("only 700"), "unexpected message: {e}");

        // Too small: the file reserves far more for 'a' than this reader thinks it occupies.
        let underrun = vec![t("a", 0), t("b", 40_000)];
        let e = check_declared_strides(&underrun, 41_088, 32).expect_err("must refuse an underrun");
        assert!(e.contains("is 1088 bytes") && e.contains("reserves 40000"), "unexpected message: {e}");

        // A header PROBE holds only a prefix of the file, so the last tensor's bound is unknowable and
        // must not be enforced — `backed.rs` parses exactly this way and every probe would otherwise
        // be rejected. The inter-tensor gaps are still checked, which is what the overrun case above
        // relies on.
        check_declared_strides(&tensors, 16, 32)
            .expect("a header probe must still parse: its buffer ends before the tensors begin");
    }
}

// ─────────────────────────── IQ2_XXS / IQ3_XXS placement proofs ───────────────────────────
//
// The IQ decoders' traps were both about WHICH WORD a field is read from: the IQ2 sub-scale shares
// its word with four 7-bit sign indices (4·7 + 4 = 32, no spare bit), and the IQ3 block's two halves
// are NOT interleaved — 64 index bytes, then 8 control words. Both are index arithmetic, so both
// are provable for every position rather than for the ones a golden vector used.
//
// The f16 scale is STUBBED to 1.0 here for the same reason as the STQ1_0 proofs (see
// `rd_f16_one`): these theorems are about where bytes go, not about f16 arithmetic.
#[cfg(kani)]
mod iq_proofs {
    use super::*;

    /// **IQ3_XXS reads its control words from the SECOND region, not interleaved with the first.**
    ///
    /// Put a symbolic sub-scale nibble into control word `ib` (at byte offset `2 + 64 + 4·ib`) and
    /// nothing else; every weight of group `ib` must scale by `(0.5 + nib)·0.5` and no other group
    /// may move. Reading the halves as one interleaved stream keeps the block size and the value
    /// distribution and breaks exactly this pairing.
    #[kani::proof]
    // 258: the assertion below walks all 256 positions. A bound that is too small is reported as
    // an unwinding-assertion FAILURE, not as a truncated success -- 10 here failed both harnesses
    // on their first run, which is the behaviour that makes a bounded check trustworthy.
    #[kani::unwind(258)]
    #[kani::stub(crate::rd_f16, crate::rd_f16_one)]
    fn iq3_control_words_live_after_the_index_bytes() {
        let ib: usize = kani::any();
        let nib: u32 = kani::any();
        kani::assume(ib < 8 && nib < 16);

        let mut blk = [0u8; 98];
        blk[0] = 0x00; blk[1] = 0x3C; // d = 1.0 (stubbed anyway)
        // Every index byte is 0 -> grid point 0. Control words all 0 except group ib's sub-scale.
        let off = 2 + 64 + 4 * ib;
        blk[off + 3] = (nib << 4) as u8; // top nibble of the little-endian u32 = bits 28..31

        let out = deq_iq3_xxs(&blk, 256);
        let g0 = IQ3XXS_GRID[0].to_le_bytes();
        let db_on = (0.5 + nib as f32) * 0.5;
        let db_off = 0.5 * 0.5;

        let mut j = 0;
        while j < 256 {
            let grp = j / 32;
            let want_mag = g0[j % 4] as f32; // both grid points in a sub-block are index 0
            let scale = if grp == ib { db_on } else { db_off };
            // sign index 0 -> ksigns(0) = 0 -> every sign positive
            assert!(out[j] == want_mag * scale, "position {j} scaled by the wrong group's word");
            j += 1;
        }
    }

    /// **IQ2_XXS reads its sub-scale from the SIGN word, not the index word.**
    ///
    /// The two words of a group are `lo` (four grid indices) and `hi` (four sign indices plus the
    /// sub-scale in bits 28..31). A symbolic sub-scale placed in `hi` must scale group `ib`; the
    /// same bits placed in `lo` are a grid index and must NOT act as a scale.
    #[kani::proof]
    // 258: the assertion below walks all 256 positions. A bound that is too small is reported as
    // an unwinding-assertion FAILURE, not as a truncated success -- 10 here failed both harnesses
    // on their first run, which is the behaviour that makes a bounded check trustworthy.
    #[kani::unwind(258)]
    #[kani::stub(crate::rd_f16, crate::rd_f16_one)]
    fn iq2_subscale_is_in_the_sign_word() {
        let ib: usize = kani::any();
        let nib: u32 = kani::any();
        kani::assume(ib < 8 && nib < 16);

        let mut blk = [0u8; 66];
        blk[0] = 0x00; blk[1] = 0x3C;
        // hi word of group ib is at byte 2 + 8*ib + 4; its top nibble is byte +3, bits 4..7.
        blk[2 + 8 * ib + 4 + 3] = (nib << 4) as u8;

        let out = deq_iq2_xxs(&blk, 256);
        let g0 = IQ2XXS_GRID[0].to_le_bytes();
        let db_on = (0.5 + nib as f32) * 0.25;
        let db_off = 0.5 * 0.25;

        let mut j = 0;
        while j < 256 {
            let grp = j / 32;
            let want_mag = g0[j % 8] as f32;
            let scale = if grp == ib { db_on } else { db_off };
            assert!(out[j] == want_mag * scale, "position {j} scaled by the wrong word");
            j += 1;
        }
    }
}
