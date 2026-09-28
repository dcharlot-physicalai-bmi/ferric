//! **Packed GPU matmul for IQ1_S, IQ1_M, IQ2_XS, IQ2_S and IQ3_S** — the on-disk blocks, resident as-is.
//!
//! Before this, a GGUF holding any of these five types could not be read at all (`ferric-gguf` had no
//! stride for them); with only a CPU decoder it would have loaded through the f32 dense fallback, which
//! for IQ1_S is **20.5x** the format's own footprint (32 bits against 1.5625) — giving up exactly the
//! property the format exists for.
//!
//! ## One kernel shape, five bodies, and the bytes are never rearranged
//!
//! The existing IQ2_XXS/IQ3_XXS kernels repack each block into `u32` lanes. These five have block
//! sizes (74, 82, 110, 50, 56 bytes) that are not multiples of four, and three of them mix `u8`,
//! `u16` and `f16` fields at odd offsets. Rather than a repack per type, the raw block bytes are
//! uploaded unchanged (padded to a whole word at the END only) and the shader reads bytes by address
//! (`b8`, `b16`, `hf`). Each body is then a line-by-line transliteration of the matching
//! `dequantize_row_*` in `ggml-quants.c`, reading the same fields at the same offsets as the CPU
//! decoder in `ferric_gguf::more_quants` — the weight each element gets is formed in the C's order
//! (`dl * grid * sign`, `dl * (grid + delta)`), and every such product is exact in f32 (an f16 scale
//! times a small odd integer times a grid byte), so the kernel's weights ARE ggml's weights; only the
//! reduction order of the dot product differs from a CPU sum.
//!
//! ⚠ What is slow: byte-addressed loads cost several ALU ops per weight byte, and the split-K walks one
//! 32-value group per lane step. It is a correct first kernel at the on-disk size, not a tuned one — a
//! repacked-lane layout like `Iq2XxsWeights` is the obvious next step for whichever of these a real
//! model makes hot.

use crate::{empty, run, unibuf, Tensor};
use ferric_core::Context;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// ggml type ids handled here, with their block bytes (per 256 values).
pub const IQ_RAW_TYPES: [(u32, usize); 5] = [(17, 74), (22, 82), (21, 110), (19, 50), (29, 56)];

/// Block bytes for a type this module handles, or `None`.
pub fn iq_raw_block_bytes(ty: u32) -> Option<usize> {
    IQ_RAW_TYPES.iter().find(|(t, _)| *t == ty).map(|(_, b)| *b)
}

/// One IQ-family weight `[rows (out), cols (in)]` with its block bytes resident unchanged.
pub struct IqRawWeights {
    ctx: Arc<Context>,
    raw: Arc<wgpu::Buffer>,
    grid: Arc<wgpu::Buffer>,
    pub ty: u32,
    pub rows: usize,
    pub cols: usize,
}

impl IqRawWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], ty: u32, rows: usize, cols: usize) -> IqRawWeights {
        let bpb = iq_raw_block_bytes(ty).unwrap_or_else(|| panic!("ggml type {ty} is not an iq_raw type"));
        assert_eq!(cols % 256, 0, "IQ cols must be a multiple of 256 (got {cols})");
        assert_eq!(bytes.len(), rows * (cols / 256) * bpb, "unexpected byte length for ggml type {ty}");
        let mut words = vec![0u32; bytes.len().div_ceil(4).max(1)];
        for (i, &b) in bytes.iter().enumerate() { words[i >> 2] |= (b as u32) << (8 * (i & 3)); }
        let grid: &[u32] = match ty {
            17 => &crate::iq_grids2::IQ2XS_GRID_U32,
            22 => &crate::iq_grids2::IQ2S_GRID_U32,
            21 => &crate::iq_grids2::IQ3S_GRID_U32,
            _ => &crate::iq_grids2::IQ1S_GRID_U32, // IQ1_S and IQ1_M share one grid
        };
        let mk = |label: &str, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        IqRawWeights { ctx: ctx.clone(), raw: mk("iq_raw.blocks", &words), grid: mk("iq_raw.grid", grid), ty, rows, cols }
    }
    /// Resident bytes: the on-disk size (plus at most 3 bytes of end padding), excluding the shared grid.
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * iq_raw_block_bytes(self.ty).unwrap() }
}

const COMMON: &str = r#"
@group(0) @binding(0) var<storage,read>        x:     array<f32>;
@group(0) @binding(1) var<storage,read>        raw:   array<u32>;
@group(0) @binding(2) var<storage,read>        grid:  array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:   array<f32>;
@group(0) @binding(4) var<uniform>             info:  vec4<u32>;   // rows, out, in, grid width
var<workgroup> partial: array<f32, 64>;
fn b8(off: u32) -> u32 { return (raw[off >> 2u] >> ((off & 3u) << 3u)) & 0xffu; }
fn b16(off: u32) -> u32 { return b8(off) | (b8(off + 1u) << 8u); }
fn hf(off: u32) -> f32 { return unpack2x16float(b16(off)).x; }
// byte j (0..7) of a 64-bit grid entry e, stored (low, high)
fn g8(e: u32, j: u32) -> u32 { return (grid[2u * e + (j >> 2u)] >> ((j & 3u) << 3u)) & 0xffu; }
// byte j (0..3) of a 32-bit grid entry e
fn g4(e: u32, j: u32) -> u32 { return (grid[e] >> (j << 3u)) & 0xffu; }
// ksigns_iq2xs: seven sign bits plus an even-parity eighth
fn ks(i: u32) -> u32 { let lo = i & 127u; return lo | ((countOneBits(lo) & 1u) << 7u); }
fn sg(bits: u32, j: u32) -> f32 { return select(1.0, -1.0, ((bits >> j) & 1u) != 0u); }
fn s8(v: u32) -> f32 { return f32(i32(v << 24u) >> 24u); }
const L: u32 = 16u;     // lanes per output
const OPW: u32 = 4u;    // outputs per workgroup (L * OPW = 64)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let t = lid.x; let bl = t % L; let sub = t / L;
    let idx = (wg.x + wg.y * info.w) * OPW + sub;
    let n_all = rows * o_dim;
    var acc = 0.0;
    // Guard the work, barrier unconditionally, guard the store (barriers must be in uniform flow).
    if (idx < n_all) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 256u;
        for (var gi: u32 = bl; gi < nblk * 8u; gi = gi + L) {
            let blk = gi >> 3u; let ib = gi & 7u;
            let bo = (o * nblk + blk) * __BPB__u;
            let xb = r * in_dim + blk * 256u + ib * 32u;
__BODY__
        }
    }
    partial[t] = acc;
    workgroupBarrier();
    for (var s: u32 = L / 2u; s > 0u; s = s >> 1u) { if (bl < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
    if (bl == 0u && idx < n_all) { out[idx] = partial[t]; }
}
"#;

/// IQ2_XS: d, qs u16[32] at 2, scales[8] at 66. Index = q & 511, signs = ksigns(q >> 9).
const BODY_IQ2_XS: &str = r#"
            let d = hf(bo);
            let sc = b8(bo + 66u + ib);
            let db0 = d * (0.5 + f32(sc & 15u)) * 0.25;
            let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let q = b16(bo + 2u + 2u * (4u * ib + l));
                let e = q & 511u; let sn = ks(q >> 9u);
                let dl = select(db0, db1, l >= 2u);
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    acc = acc + x[xb + 8u * l + j] * (dl * f32(g8(e, j)) * sg(sn, j));
                }
            }
"#;

/// IQ2_S: d, qs[32] at 2, signs[32] at 34, qh[8] at 66, scales[8] at 74. 10-bit index.
const BODY_IQ2_S: &str = r#"
            let d = hf(bo);
            let sc = b8(bo + 74u + ib);
            let db0 = d * (0.5 + f32(sc & 15u)) * 0.25;
            let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
            let qh = b8(bo + 66u + ib);
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let e = b8(bo + 2u + 4u * ib + l) | ((qh << (8u - 2u * l)) & 0x300u);
                let sn = b8(bo + 34u + 4u * ib + l);
                let dl = select(db0, db1, l >= 2u);
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    acc = acc + x[xb + 8u * l + j] * (dl * f32(g8(e, j)) * sg(sn, j));
                }
            }
"#;

/// IQ3_S: d, qs[64] at 2, qh[8] at 66, signs[32] at 74, scales[4] at 106. Two 4-value lookups per 8.
const BODY_IQ3_S: &str = r#"
            let d = hf(bo);
            let sc = b8(bo + 106u + (ib >> 1u));
            let db = d * f32(1u + 2u * ((sc >> (4u * (ib & 1u))) & 15u));
            let qh = b8(bo + 66u + ib);
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let e1 = b8(bo + 2u + 8u * ib + 2u * l) | ((qh << (8u - 2u * l)) & 256u);
                let e2 = b8(bo + 3u + 8u * ib + 2u * l) | ((qh << (7u - 2u * l)) & 256u);
                let sn = b8(bo + 74u + 4u * ib + l);
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc = acc + x[xb + 8u * l + j]      * (db * f32(g4(e1, j)) * sg(sn, j));
                    acc = acc + x[xb + 8u * l + j + 4u] * (db * f32(g4(e2, j)) * sg(sn, j + 4u));
                }
            }
"#;

/// IQ1_S: d, qs[32] at 2, qh u16[8] at 34. y = dl * (grid + delta), grid bytes signed.
const BODY_IQ1_S: &str = r#"
            let d = hf(bo);
            let qh = b16(bo + 34u + 2u * ib);
            let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
            let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let e = b8(bo + 2u + 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8u);
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    acc = acc + x[xb + 8u * l + j] * (dl * (s8(g8(e, j)) + delta));
                }
            }
"#;

/// IQ1_M: qs[32] at 0, qh[16] at 32, scales u16[4] at 48; the f16 super-scale is the four top nibbles.
const BODY_IQ1_M: &str = r#"
            let s0 = b16(bo + 48u); let s1 = b16(bo + 50u); let s2 = b16(bo + 52u); let s3 = b16(bo + 54u);
            let u = (s0 >> 12u) | ((s1 >> 8u) & 0xf0u) | ((s2 >> 4u) & 0xf00u) | (s3 & 0xf000u);
            let d = unpack2x16float(u).x;
            let s = b16(bo + 48u + 2u * (ib >> 1u));
            let sh = 6u * (ib & 1u);
            let dl1 = d * f32(2u * ((s >> sh) & 7u) + 1u);
            let dl2 = d * f32(2u * ((s >> (sh + 3u)) & 7u) + 1u);
            let h0 = b8(bo + 32u + 2u * ib); let h1 = b8(bo + 33u + 2u * ib);
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let h = select(h0, h1, l >= 2u);
                let odd = (l & 1u) == 1u;
                let e = b8(bo + 4u * ib + l) | (select(h << 8u, h << 4u, odd) & 0x700u);
                let delta = select(0.125, -0.125, (h & select(0x08u, 0x80u, odd)) != 0u);
                let dl = select(dl1, dl2, l >= 2u);
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    acc = acc + x[xb + 8u * l + j] * (dl * (s8(g8(e, j)) + delta));
                }
            }
"#;

fn wgsl_for(ty: u32) -> String {
    let body = match ty {
        17 => BODY_IQ2_XS, 22 => BODY_IQ2_S, 21 => BODY_IQ3_S, 19 => BODY_IQ1_S, 29 => BODY_IQ1_M,
        other => panic!("iq_raw: no body for ggml type {other}"),
    };
    let src = COMMON.replace("__BODY__", body).replace("__BPB__", &iq_raw_block_bytes(ty).unwrap().to_string());
    // Negative controls for `scripts/quant_formats_conformance.sh` — one plausible misreading each, which
    // must move the logits >= 20x the clean distance. `iq_signs`: every sign bit ignored (IQ2/IQ3);
    // `iq_delta`: the IQ1 grid shift taken with the wrong sign.
    match std::env::var("FERRIC_IQ_CONTROL").as_deref() {
        Ok("iq_signs") => src.replace("fn sg(bits: u32, j: u32) -> f32 { return select(1.0, -1.0, ((bits >> j) & 1u) != 0u); }",
                                      "fn sg(bits: u32, j: u32) -> f32 { return 1.0; }"),
        Ok("iq_delta") => src.replace("select(0.125, -0.125,", "select(-0.125, 0.125,"),
        _ => src,
    }
}

impl Tensor {
    /// `y = x·Wᵀ` where `W` is IQ1_S / IQ1_M / IQ2_XS / IQ2_S / IQ3_S, decoded in-kernel from its block bytes.
    pub fn matmul_iq_raw(&self, w: &IqRawWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let nwg = (rows * w.rows).div_ceil(4);
        let gw = nwg.min(32768);
        let label = match w.ty { 17 => "matmul_iq2_xs", 22 => "matmul_iq2_s", 21 => "matmul_iq3_s", 19 => "matmul_iq1_s", _ => "matmul_iq1_m" };
        run(&self.ctx, &wgsl_for(w.ty), label,
            &[x.buf.as_ref(), w.raw.as_ref(), w.grid.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, gw as u32])],
            (gw as u32, nwg.div_ceil(gw) as u32, 1));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU grids are a cache of `ferric-gguf`'s — word for word, or this fails.
    #[test]
    fn gpu_grids_equal_the_cpu_grids() {
        let split = |v: &[u64]| -> Vec<u32> { v.iter().flat_map(|&g| [g as u32, (g >> 32) as u32]).collect() };
        assert_eq!(split(&ferric_gguf::IQ2XS_GRID), crate::iq_grids2::IQ2XS_GRID_U32.to_vec());
        assert_eq!(split(&ferric_gguf::IQ2S_GRID), crate::iq_grids2::IQ2S_GRID_U32.to_vec());
        assert_eq!(split(&ferric_gguf::IQ1S_GRID), crate::iq_grids2::IQ1S_GRID_U32.to_vec());
        assert_eq!(ferric_gguf::IQ3S_GRID.to_vec(), crate::iq_grids2::IQ3S_GRID_U32.to_vec());
    }

    /// The kernel against the CPU decoder (itself bit-identical to libggml-base, see
    /// `ferric_gguf::more_quants`), on REAL blocks from the committed fixtures, several activation rows.
    /// The weights are exact on both sides, so the only legitimate difference is the dot product's
    /// reduction order: gated at 2e-6 relative to the row's magnitude sum.
    #[test]
    fn iq_raw_kernels_match_the_cpu_decoder_on_real_blocks() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED iq_raw_kernels_match_the_cpu_decoder_on_real_blocks: no GPU — NOTHING was checked");
            return;
        };
        let ctx = Arc::new(ctx);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ferric-gguf/tests/fixtures/ggml_quants");
        for (ty, name) in [(17u32, "iq2_xs"), (22, "iq2_s"), (21, "iq3_s"), (19, "iq1_s"), (29, "iq1_m")] {
            let raw = std::fs::read(dir.join(format!("{name}_real.raw"))).unwrap_or_else(|e| panic!("{name}: {e}"));
            let bpb = iq_raw_block_bytes(ty).unwrap();
            let nblk = raw.len() / bpb;
            // Lay the real blocks out as a [rows, 512] weight (two blocks per row).
            let cols = 512usize; let rows = nblk / 2;
            let bytes = &raw[..rows * 2 * bpb];
            let w = ferric_gguf::deq_raw(bytes, rows * cols, ty).unwrap();
            let t = 3usize;
            let xv: Vec<f32> = (0..t * cols).map(|i| ((i * 7919 % 101) as f32 - 50.0) / 37.0).collect();
            let qm = IqRawWeights::from_bytes(&ctx, bytes, ty, rows, cols);
            let x = Tensor::from_vec(&ctx, &xv, &[t, cols]);
            let got = pollster::block_on(x.matmul_iq_raw(&qm).to_vec());
            let mut worst = 0f64; let mut nz = 0;
            for r in 0..t { for o in 0..rows {
                let (mut want, mut mag) = (0f64, 0f64);
                for c in 0..cols { let p = xv[r * cols + c] as f64 * w[o * cols + c] as f64; want += p; mag += p.abs(); }
                if want != 0.0 { nz += 1; }
                worst = worst.max((got[r * rows + o] as f64 - want).abs() / mag.max(1e-30));
            }}
            assert!(nz > t * rows / 2, "{name}: reference mostly zero — the check would pass on anything");
            eprintln!("{name}: {rows}x{cols} x {t} rows, worst |gpu - cpu| / sum|x*w| = {worst:.2e}");
            assert!(worst < 2e-6, "{name}: GPU kernel diverges from the CPU decoder ({worst:.2e})");
        }
    }
}
