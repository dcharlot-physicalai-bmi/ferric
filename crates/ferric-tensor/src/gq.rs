//! **GQ packed matmul** — GPTQ, AWQ, compressed-tensors, FP8 and ModelOpt NVFP4 weights, run packed.
//!
//! `ferric_load` re-lays every grouped-quant safetensors checkpoint, losslessly, into one parametric
//! form (`ferric_gguf::gq`: integer / FP8 / FP4 codes, a scale and optional zero per group, optional
//! per-value group indices for act-order). The geometry travels in the synthetic ggml type id, so a
//! tensor can never be fused with one of different geometry. This is the kernel that consumes it.
//!
//! ## Resident layout
//!
//! The presented bytes are block-interleaved (codes, scale, zero, g_idx per group). On the device they
//! are split so the inner loop reads codes as whole words: `codes` holds each row's bitstream
//! (`cols * bits / 32` words per row — 32 values are always exactly `bits` words), `sz` each row's
//! scales then zeros as f32, and for act-order `gtab` the DISTINCT g_idx arrays (a fused q|k|v carries
//! three; a lone weight one) with `rowg` giving each row's offset into it. Resident size is the code
//! bits plus 32 or 64 bits per group — 4.25-4.5 bpw for a group-128 int4 weight, against 32 dense.
//!
//! ## Arithmetic
//!
//! The weight is formed exactly as the defining libraries form it: `s * (q - z)` (GPTQ/AWQ/
//! compressed-tensors int), `s * e4m3(q)` (FP8), `s * e2m1(q)` (FP4), in f32, BEFORE it meets the
//! activation — the same value `ferric_gguf::gq::deq_gq_rows` produces on the CPU, so the kernel and
//! the dense fallback differ only in the dot product's reduction order.
//!
//! ⚠ What is slow: every value re-reads its group's scale (a cached load, but a load), the act-order
//! variant adds a gather per value, and one output is 16 lanes of a split-K over 32-value chunks.
//! Correct at the format's size; a vec4 inner loop and one scale read per group are the next steps.

use crate::{empty, run, unibuf, Tensor};
use ferric_core::Context;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Decoded GQ type id. Mirrors `ferric_gguf::gq::GqSpec` (this crate sits below that one); the test
/// below pins the two decoders to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GqId { pub kind: u32, pub act: bool, pub bits: u32, pub group: u32 }

pub fn decode_id(id: u32) -> Option<GqId> {
    if id >> 24 != 0x47 { return None }
    let s = GqId { kind: (id >> 20) & 0xf, act: (id >> 16) & 1 == 1, bits: (id >> 12) & 0xf, group: id & 0xfff };
    let ok_bits = match s.kind { 0 => matches!(s.bits, 2 | 3 | 4 | 8), 1 => s.bits == 8, 2 => s.bits == 4, _ => false };
    (ok_bits && s.group > 0 && (s.group * s.bits) % 32 == 0).then_some(s)
}

pub fn is_gq(id: u32) -> bool { decode_id(id).is_some() }

fn has_zero(s: &GqId) -> bool { s.kind == 0 }

fn block_len(s: &GqId) -> usize {
    let g = s.group as usize;
    g * s.bits as usize / 8 + 4 + if has_zero(s) { 4 } else { 0 } + if s.act { 2 * g } else { 0 }
}

/// `(values per block, bytes per block)` for `QMatrix::block_bytes`.
pub fn block_bytes(id: u32) -> Option<(usize, usize)> {
    decode_id(id).map(|s| (s.group as usize, block_len(&s)))
}

pub struct GqWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,
    sz: Arc<wgpu::Buffer>,
    gtab: Arc<wgpu::Buffer>,
    rowg: Arc<wgpu::Buffer>,
    spec: GqId,
    pub rows: usize,
    pub cols: usize,
    n_gidx: usize,
}

impl GqWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], id: u32, rows: usize, cols: usize) -> Result<GqWeights, String> {
        let spec = decode_id(id).ok_or_else(|| format!("{id:#x} is not a GQ type id"))?;
        let (g, bits) = (spec.group as usize, spec.bits as usize);
        if cols % g != 0 || cols % 32 != 0 {
            return Err(format!("GQ: {cols} columns is not a whole number of {g}-value groups and 32-value chunks"));
        }
        let (ng, bb, cb) = (cols / g, block_len(&spec), g * bits / 8);
        if bytes.len() != rows * ng * bb {
            return Err(format!("GQ: {} bytes for [{rows}, {cols}], expected {}", bytes.len(), rows * ng * bb));
        }
        let cw = cols * bits / 32;
        let per_row_sz = ng * if has_zero(&spec) { 2 } else { 1 };
        let mut codes = vec![0u32; rows * cw];
        let mut sz = vec![0f32; rows * per_row_sz];
        let mut gtab: Vec<u32> = Vec::new();
        let mut rowg = vec![0u32; rows];
        let mut seen: HashMap<Vec<u16>, u32> = HashMap::new();
        let f = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        for r in 0..rows {
            let row = &bytes[r * ng * bb..(r + 1) * ng * bb];
            let mut gi_row: Vec<u16> = if spec.act { Vec::with_capacity(cols) } else { Vec::new() };
            for b in 0..ng {
                let blk = &row[b * bb..(b + 1) * bb];
                for w in 0..cb / 4 {
                    codes[r * cw + b * cb / 4 + w] = u32::from_le_bytes([blk[4 * w], blk[4 * w + 1], blk[4 * w + 2], blk[4 * w + 3]]);
                }
                sz[r * per_row_sz + b] = f(&blk[cb..]);
                if has_zero(&spec) { sz[r * per_row_sz + ng + b] = f(&blk[cb + 4..]); }
                if spec.act {
                    let o = cb + if has_zero(&spec) { 8 } else { 4 };
                    for i in 0..g { gi_row.push(u16::from_le_bytes([blk[o + 2 * i], blk[o + 2 * i + 1]])); }
                }
            }
            if spec.act {
                if let Some(&bad) = gi_row.iter().find(|&&x| x as usize >= ng) {
                    return Err(format!("GQ: row {r} has g_idx {bad} but only {ng} groups"));
                }
                let next = gtab.len() as u32;
                let off = *seen.entry(gi_row.clone()).or_insert_with(|| {
                    gtab.extend(gi_row.iter().map(|&x| x as u32));
                    next
                });
                rowg[r] = off;
            }
        }
        if gtab.is_empty() { gtab.push(0); }
        let n_gidx = seen.len();
        let mk = |label: &str, data: &[u8]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Ok(GqWeights {
            ctx: ctx.clone(),
            codes: mk("gq.codes", bytemuck::cast_slice(&codes)),
            sz: mk("gq.scales", bytemuck::cast_slice(&sz)),
            gtab: mk("gq.gidx", bytemuck::cast_slice(&gtab)),
            rowg: mk("gq.rowg", bytemuck::cast_slice(&rowg)),
            spec, rows, cols, n_gidx,
        })
    }
    /// Resident bytes, read back from the buffers (codes + scales/zeros + distinct g_idx arrays).
    pub fn nbytes(&self) -> usize {
        (self.codes.size() + self.sz.size() + if self.spec.act { self.gtab.size() + self.rowg.size() } else { 0 }) as usize
    }
    /// Distinct act-order index arrays held (0 without act-order).
    pub fn distinct_gidx(&self) -> usize { self.n_gidx }
}

const WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:     array<f32>;
@group(0) @binding(1) var<storage,read>        codes: array<u32>;
@group(0) @binding(2) var<storage,read>        sz:    array<f32>;
@group(0) @binding(3) var<storage,read>        gtab:  array<u32>;
@group(0) @binding(4) var<storage,read>        rowg:  array<u32>;
@group(0) @binding(5) var<storage,read_write>  out:   array<f32>;
@group(0) @binding(6) var<uniform>             info:  vec4<u32>;   // rows, out, in, grid width
var<workgroup> partial: array<f32, 64>;
const BITS: u32 = __BITS__u;
const G: u32 = __G__u;
const MASK: u32 = __MASK__u;
const L: u32 = 16u;
const OPW: u32 = 4u;
// float8_e4m3fn: no infinities, 0x7f/0xff NaN, subnormals m * 2^-9.
fn e4m3(v: u32) -> f32 {
    let s = select(1.0, -1.0, (v & 0x80u) != 0u);
    let e = (v >> 3u) & 15u; let m = v & 7u;
    if (e == 15u && m == 7u) { return bitcast<f32>(0x7fc00000u); }
    if (e == 0u) { return s * f32(m) * 0.001953125; }
    return bitcast<f32>(((v & 0x80u) << 24u) | ((e + 120u) << 23u) | (m << 20u));
}
// E2M1: {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit.
fn e2m1(v: u32) -> f32 {
    let s = select(1.0, -1.0, (v & 8u) != 0u);
    let e = (v >> 1u) & 3u; let m = f32(v & 1u);
    let mag = select((1.0 + 0.5 * m) * exp2(f32(e) - 1.0), 0.5 * m, e == 0u);
    return s * mag;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let t = lid.x; let bl = t % L; let sub = t / L;
    let idx = (wg.x + wg.y * info.w) * OPW + sub;
    let n_all = rows * o_dim;
    var acc = 0.0;
    if (idx < n_all) {
        let o = idx % o_dim; let r = idx / o_dim;
        let ng = in_dim / G;
        let szs = ng * __SZMUL__u;
        let cw = in_dim * BITS / 32u;
        // `gtab` is referenced even when unused so every variant has the same seven bindings (the
        // bind group layout is derived from the shader, and an unreferenced binding vanishes from it).
        let gbase = rowg[o] + (gtab[0] & 0u);
        for (var c: u32 = bl; c < in_dim / 32u; c = c + L) {
            let wb = o * cw + c * BITS;
            let xb = r * in_dim + c * 32u;
            for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                let bp = k * BITS; let wi = bp >> 5u; let sh = bp & 31u;
                var v = codes[wb + wi] >> sh;
                if (sh + BITS > 32u) { v = v | (codes[wb + wi + 1u] << (32u - sh)); }
                v = v & MASK;
                let col = c * 32u + k;
                let g = __GROUP__;
                let s = sz[o * szs + g];
                acc = acc + x[xb + k] * (__WEIGHT__);
            }
        }
    }
    partial[t] = acc;
    workgroupBarrier();
    for (var s2: u32 = L / 2u; s2 > 0u; s2 = s2 >> 1u) { if (bl < s2) { partial[t] = partial[t] + partial[t + s2]; } workgroupBarrier(); }
    if (bl == 0u && idx < n_all) { out[idx] = partial[t]; }
}
"#;

fn wgsl_for(s: &GqId) -> String {
    let weight = match s.kind {
        0 => "s * (f32(v) - sz[o * szs + ng + g])",
        1 => "s * e4m3(v)",
        _ => "s * e2m1(v)",
    };
    WGSL.replace("__BITS__", &s.bits.to_string())
        .replace("__G__", &s.group.to_string())
        .replace("__MASK__", &((1u32 << s.bits) - 1).to_string())
        .replace("__SZMUL__", if has_zero(s) { "2" } else { "1" })
        .replace("__GROUP__", if s.act { "gtab[gbase + col]" } else { "col / G" })
        .replace("__WEIGHT__", weight)
}

impl Tensor {
    /// `y = x·Wᵀ` for a GQ weight, decoded in-kernel.
    pub fn matmul_gq(&self, w: &GqWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let nwg = (rows * w.rows).div_ceil(4);
        let gw = nwg.min(32768);
        run(&self.ctx, &wgsl_for(&w.spec), "matmul_gq",
            &[x.buf.as_ref(), w.codes.as_ref(), w.sz.as_ref(), w.gtab.as_ref(), w.rowg.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, gw as u32])],
            (gw as u32, nwg.div_ceil(gw) as u32, 1));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_gguf::gq::{encode_row, deq_gq_rows, GqKind, GqSpec};

    #[test]
    fn id_decoder_agrees_with_ferric_gguf() {
        for kind in [GqKind::Int, GqKind::Fp8E4m3, GqKind::Fp4E2m1] {
            for bits in [2, 3, 4, 8] { for group in [16, 32, 64, 128] { for act in [false, true] {
                let s = GqSpec { kind, bits, group, act };
                let mine = decode_id(s.id());
                assert_eq!(mine.is_some(), s.valid().is_ok(), "{s:?}");
                if let Some(m) = mine {
                    assert_eq!((m.kind, m.act, m.bits, m.group), (kind as u32, act, bits, group));
                    assert_eq!(block_len(&m), s.block_bytes());
                }
            }}}
        }
    }

    /// Every width, every code kind, with and without act-order — including 3-bit, whose values 10 and
    /// 21 of every 32 straddle a word — against `ferric_gguf::gq::deq_gq_rows` (the CPU decoder the
    /// loader's bit-exactness gates are run on), for several activation rows. A fused-style weight
    /// whose rows carry TWO different g_idx arrays checks the dedup table.
    #[test]
    fn gq_kernel_matches_the_cpu_decoder_for_every_variant() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED gq_kernel_matches_the_cpu_decoder_for_every_variant: no GPU — NOTHING was checked");
            return;
        };
        let ctx = Arc::new(ctx);
        let mut checked = 0;
        for (kind, bits) in [(GqKind::Int, 2), (GqKind::Int, 3), (GqKind::Int, 4), (GqKind::Int, 8), (GqKind::Fp8E4m3, 8), (GqKind::Fp4E2m1, 4)] {
            for (group, act) in [(32u32, false), (64, true), (16, false)] {
                let spec = GqSpec { kind, bits, group, act };
                if spec.valid().is_err() { continue }
                let (rows, cols) = (24usize, 256usize);
                let ng = cols / group as usize;
                let mut raw = Vec::new();
                for r in 0..rows {
                    let codes: Vec<u32> = (0..cols).map(|c| {
                        let q = ((c * 37 + r * 11 + c / 7) as u32) % (1 << bits);
                        // every code but FP8's two NaN slots (0x7f, 0xff), which would poison the sum
                        if kind == GqKind::Fp8E4m3 && q & 0x7f == 0x7f { q - 1 } else { q }
                    }).collect();
                    let scales: Vec<f32> = (0..ng).map(|b| 0.003 * (1 + (b * 5 + r) % 9) as f32).collect();
                    let zeros: Vec<f32> = (0..ng).map(|b| ((b + 2 * r) % (1 << bits.min(4))) as f32).collect();
                    // two distinct act-order maps, alternating by half of the rows
                    let gi: Vec<u16> = (0..cols).map(|c| ((c * 13 + (r / 12) * 5) % ng) as u16).collect();
                    encode_row(&spec, &codes, &scales, &zeros, act.then_some(&gi[..]), &mut raw);
                }
                let w = deq_gq_rows(&raw, rows, cols, &spec).unwrap();
                let qm = GqWeights::from_bytes(&ctx, &raw, spec.id(), rows, cols).unwrap();
                if act { assert_eq!(qm.distinct_gidx(), 2, "the g_idx table should hold exactly the two distinct maps"); }
                let t = 3;
                let xv: Vec<f32> = (0..t * cols).map(|i| ((i * 7919 % 97) as f32 - 48.0) / 31.0).collect();
                let got = pollster::block_on(Tensor::from_vec(&ctx, &xv, &[t, cols]).matmul_gq(&qm).to_vec());
                let mut worst = 0f64;
                for r in 0..t { for o in 0..rows {
                    let (mut want, mut mag) = (0f64, 0f64);
                    for c in 0..cols { let p = xv[r * cols + c] as f64 * w[o * cols + c] as f64; want += p; mag += p.abs(); }
                    worst = worst.max((got[r * rows + o] as f64 - want).abs() / mag.max(1e-30));
                }}
                assert!(worst < 2e-6, "{spec:?}: GPU diverges from the CPU decoder by {worst:.2e}");
                checked += 1;
            }
        }
        assert!(checked >= 12, "only {checked} variants ran");
    }
}
