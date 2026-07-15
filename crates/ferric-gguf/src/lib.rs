//! Pure-Rust reader for the llama.cpp **GGUF** container + dequantizers for the common block-quant
//! formats (F32, F16, Q8_0, Q4_0, and the k-quant **Q4_K**). GGUF is how the entire llama.cpp / HF
//! quantized-model corpus ships — including Liquid AI's LFM2 and BitNet — so this is the ingest path
//! that lets Ferric run those models. Dequant here is CPU-side (I/O layer); a fused on-GPU dequant
//! matmul is the perf follow-up.

use half::f16;
use std::collections::HashMap;

// ---- ggml tensor type codes we handle ----
const F32: u32 = 0;
const F16T: u32 = 1;
const Q4_0: u32 = 2;
const Q8_0: u32 = 8;
const Q4_K: u32 = 12;
const TQ2_0: u32 = 35; // llama.cpp ternary (BitNet) quant: 2 bits/weight, {−1,0,+1}·scale
const Q1_0: u32 = 41; // PrismML/mainline 1-bit: {−1,+1}·scale, group-128 (1.125 bpw)
const Q2_0: u32 = 42; // PrismML ternary: {−1,0,+1}·scale, group-128 (2.125 bpw on disk)

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

struct Cur<'a> { b: &'a [u8], p: usize }
impl<'a> Cur<'a> {
    fn u32(&mut self) -> u32 { let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap()); self.p += 4; v }
    fn u64(&mut self) -> u64 { let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap()); self.p += 8; v }
    fn i64(&mut self) -> i64 { self.u64() as i64 }
    fn f32(&mut self) -> f32 { f32::from_bits(self.u32()) }
    fn f64(&mut self) -> f64 { f64::from_bits(self.u64()) }
    fn u8(&mut self) -> u8 { let v = self.b[self.p]; self.p += 1; v }
    fn str(&mut self) -> String { let n = self.u64() as usize; let s = String::from_utf8_lossy(&self.b[self.p..self.p + n]).into_owned(); self.p += n; s }
    fn val(&mut self, ty: u32) -> Meta {
        match ty {
            0 => Meta::U(self.u8() as u64),
            1 => Meta::I(self.u8() as i8 as i64),
            2 => { let v = u16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]) as u64; self.p += 2; Meta::U(v) }
            3 => { let v = i16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]) as i64; self.p += 2; Meta::I(v) }
            4 => Meta::U(self.u32() as u64),
            5 => Meta::I(self.u32() as i32 as i64),
            6 => Meta::F(self.f32() as f64),
            7 => Meta::Bool(self.u8() != 0),
            8 => Meta::Str(self.str()),
            9 => { let et = self.u32(); let n = self.u64(); Meta::Arr((0..n).map(|_| self.val(et)).collect()) }
            10 => Meta::U(self.u64()),
            11 => Meta::I(self.i64()),
            12 => Meta::F(self.f64()),
            _ => Meta::U(0),
        }
    }
}
pub fn parse(bytes: Vec<u8>) -> Result<Gguf, String> {
    let mut c = Cur { b: &bytes, p: 0 };
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
    let align = match metadata.get("general.alignment") { Some(Meta::U(a)) => *a as usize, _ => 32 };
    let data_start = c.p.div_ceil(align) * align;
    Ok(Gguf { metadata, tensors, data: bytes, data_start })
}

impl Gguf {
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.tensors.iter().find(|t| t.name == name) }

    /// Dequantize a tensor to f32 (row-major), whatever its GGUF block-quant type.
    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let raw = &self.data[self.data_start + t.offset as usize..];
        Ok(match t.ggml_type {
            F32 => raw[..n * 4].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            F16T => raw[..n * 2].chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            Q8_0 => deq_q8_0(raw, n),
            Q4_0 => deq_q4_0(raw, n),
            Q4_K => deq_q4_k(raw, n),
            TQ2_0 => deq_tq2_0(raw, n),
            Q1_0 => deq_q1_0(raw, n),
            Q2_0 => deq_q2_0(raw, n),
            other => return Err(format!("unsupported ggml type {other}")),
        })
    }
}

fn rd_f16(b: &[u8]) -> f32 { f16::from_le_bytes([b[0], b[1]]).to_f32() }

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

/// Q4_K super-block (256 values, 144 bytes): [f16 d, f16 dmin, u8 scales[12], u8 qs[128]].
/// 8 sub-blocks of 32; each has a 6-bit scale & 6-bit min packed in `scales`. y = d·sc·q − dmin·m.
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
