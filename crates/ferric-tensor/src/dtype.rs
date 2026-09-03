//! Precision / storage dtypes for the fabric. Compute stays f32 (WebGPU-baseline has no shader-f16),
//! but weights can LIVE on the GPU in half precision and be dequantized on-device — half the memory,
//! and the path real fp16/bf16 checkpoints take. `Half` is a packed storage tensor (2 values per u32
//! word); `dequant()` expands to a compute `Tensor`, `Tensor::to_half()` packs one down.

use crate::{empty, groups, groups2d, run, u32buf, unibuf, Tensor};
use ferric_core::Context;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DType {
    F16,
    BF16,
}
impl DType {
    fn code(self) -> u32 { match self { DType::F16 => 0, DType::BF16 => 1 } }
}

/// A half-precision tensor stored packed (2×16-bit per 32-bit word) in GPU memory.
pub struct Half {
    ctx: Arc<Context>,
    buf: Arc<wgpu::Buffer>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Half {
    pub fn numel(&self) -> usize { self.shape.iter().product() }
    /// Bytes actually stored on device (half of the f32 equivalent).
    pub fn nbytes(&self) -> usize { self.numel().div_ceil(2) * 4 }
    /// The packed f16 GPU buffer — its bytes are a contiguous `array<f16>`, which the 16×16 coop
    /// kernel binds directly as f16 operands.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer { &self.buf }

    /// Build from raw 16-bit values (e.g. an fp16/bf16 slice straight out of a safetensors file).
    pub fn from_bits(ctx: &Arc<Context>, bits: &[u16], shape: &[usize], dtype: DType) -> Half {
        assert_eq!(bits.len(), shape.iter().product::<usize>(), "bits len != shape");
        let words: Vec<u32> = bits.chunks(2).map(|c| c[0] as u32 | ((*c.get(1).unwrap_or(&0) as u32) << 16)).collect();
        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("half"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Half { ctx: ctx.clone(), buf: Arc::new(buf), shape: shape.to_vec(), dtype }
    }

    /// Dequantize to an f32 compute tensor, on-device.
    pub fn dequant(&self) -> Tensor {
        let n = self.numel();
        let out = empty(&self.ctx, n);
        run(&self.ctx, DEQUANT_WGSL, "dequant", &[self.buf.as_ref(), &out, &u32buf(&self.ctx, &[n as u32, self.dtype.code()])], groups(n));
        Tensor::from_parts(&self.ctx, out, self.shape.clone())
    }
}

impl Tensor {
    /// Pack this f32 tensor down to half precision (round-to-nearest-even), on-device.
    pub fn to_half(&self, dtype: DType) -> Half {
        let c = self.contiguous();
        let n = c.numel();
        let words = n.div_ceil(2);
        let out = empty(&self.ctx, words);
        let (grid, rs) = groups2d(words);
        run(&self.ctx, QUANTIZE_WGSL, "quantize", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[n as u32, dtype.code(), rs])], grid);
        Half { ctx: self.ctx.clone(), buf: Arc::new(out), shape: c.shape.clone(), dtype }
    }
}

/// A per-tensor symmetric int8-quantized tensor (4 values packed per u32) plus its scale.
pub struct QTensor {
    ctx: Arc<Context>,
    buf: Arc<wgpu::Buffer>,
    pub scale: f32, // = max|x|/127
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Symmetric per-tensor int8 quantization (scale = max|x|/127). Async: the scalar scale is read
    /// back so the quantized matmul can fold both scales into one small buffer (WebGPU allows only 4
    /// storage buffers per shader — scalars ride in the info buffer instead of their own bindings).
    pub async fn quantize_i8(&self) -> QTensor {
        let c = self.contiguous();
        let n = c.numel();
        let axes: Vec<usize> = (0..c.rank()).collect();
        let s = c.abs().max(&axes, false).to_vec().await[0] / 127.0;
        let s = if s == 0.0 { 1.0 } else { s };
        let words = n.div_ceil(4);
        let out = empty(&self.ctx, words);
        run(&self.ctx, QUANT_I8_WGSL, "quant_i8", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[n as u32, s.to_bits()])], groups(words));
        QTensor { ctx: self.ctx.clone(), buf: Arc::new(out), scale: s, shape: c.shape.clone() }
    }
}

impl QTensor {
    /// Quantized matmul [m,k]·[k,n] → f32 (int accumulation, rescaled by both scales).
    pub fn matmul(&self, o: &QTensor) -> Tensor {
        let (ra, rb) = (self.shape.len(), o.shape.len());
        assert!(ra == 2 && rb == 2, "quantized matmul is 2D for now");
        let (m, k, n) = (self.shape[0], self.shape[1], o.shape[1]);
        assert_eq!(k, o.shape[0], "inner dims mismatch");
        let out = empty(&self.ctx, m * n);
        let info = [m as u32, k as u32, n as u32, (self.scale * o.scale).to_bits()];
        run(&self.ctx, MATMUL_I8_WGSL, "matmul_i8", &[self.buf.as_ref(), o.buf.as_ref(), &out, &u32buf(&self.ctx, &info)], groups(m * n));
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }
}

/// Per-row (per-output-channel) quantized 2D matrix at `bits` ∈ {4,8}, packed 32/bits per word,
/// with one scale per row — more accurate than a single per-tensor scale, and int4 is 1/8 the memory.
pub struct QRow {
    ctx: Arc<Context>,
    buf: Arc<wgpu::Buffer>,
    scale: Arc<wgpu::Buffer>, // [rows] f32
    pub rows: usize,
    pub cols: usize,
    pub bits: u32,
}

impl Tensor {
    /// Per-row symmetric quantization of a 2D matrix at 4 or 8 bits (scale = max|row|/(2^(bits-1)−1)).
    pub fn quantize_rowwise(&self, bits: u32) -> QRow {
        let c = self.contiguous();
        assert_eq!(c.rank(), 2, "rowwise quant is 2D");
        let (rows, cols) = (c.shape[0], c.shape[1]);
        let qmax = ((1u32 << (bits - 1)) - 1) as f32;
        let scale = c.abs().max(&[1], false).mul(&c.scalar(1.0 / qmax)); // [rows]
        let per_word = (32 / bits) as usize;
        let words = (rows * cols).div_ceil(per_word);
        let out = empty(&self.ctx, words);
        let (grid, rs) = crate::groups2d(words);
        run(&self.ctx, QUANT_ROW_WGSL, "quant_row", &[c.buf.as_ref(), scale.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, cols as u32, bits, qmax.to_bits(), rs])], grid);
        QRow { ctx: self.ctx.clone(), buf: Arc::new(out), scale: scale.buf.clone(), rows, cols, bits }
    }
}

impl Tensor {
    /// Weight-only quantized matmul (the efficient-inference path): x [rows, in] · Wᵀ where W is a
    /// per-row-quantized [out, in] matrix that stays packed in memory — dequantized on the fly in the
    /// kernel. Returns [rows, out]. This is W4A16/W8A16-style: activations f32, weights int4/int8.
    pub fn matmul_qweight(&self, w: &QRow) -> Tensor {
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "matmul_qweight is 2D");
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dims mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        run(&self.ctx, MATMUL_QW_WGSL, "matmul_qw", &[x.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out, &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, w.bits])], groups(rows * w.rows));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

impl QRow {
    pub fn nbytes(&self) -> usize { (self.rows * self.cols * self.bits as usize).div_ceil(8) }
    /// Dequantize back to an f32 [rows, cols] tensor, on-device.
    pub fn dequant(&self) -> Tensor {
        let n = self.rows * self.cols;
        let out = empty(&self.ctx, n);
        run(&self.ctx, DEQUANT_ROW_WGSL, "dequant_row", &[self.buf.as_ref(), self.scale.as_ref(), &out, &u32buf(&self.ctx, &[self.rows as u32, self.cols as u32, self.bits])], groups(n));
        Tensor::from_parts(&self.ctx, out, vec![self.rows, self.cols])
    }
}

/// A ternary-weight matrix (BitNet b1.58 family): weights ∈ {−1,0,+1} packed 16 per u32 (2 bits
/// each), with a per-output-channel scale (absmean). The matmul is effectively multiply-free — each
/// weight just adds, subtracts, or skips an activation. 1.58 bits/weight ≈ 1/16 the memory of f32.
pub struct Ternary {
    ctx: Arc<Context>,
    buf: Arc<wgpu::Buffer>,
    scale: Arc<wgpu::Buffer>, // [out] = absmean per row
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    /// Quantize a 2D [out,in] weight to ternary {−1,0,+1} with per-row absmean scale (BitNet-style).
    pub fn quantize_ternary(&self) -> Ternary {
        let c = self.contiguous();
        assert_eq!(c.rank(), 2, "ternary quant is 2D");
        let (rows, cols) = (c.shape[0], c.shape[1]);
        let scale = c.abs().mean(&[1], false); // [rows] absmean
        let words = (rows * cols).div_ceil(16);
        let out = empty(&self.ctx, words);
        run(&self.ctx, QUANT_TERNARY_WGSL, "quant_ternary", &[c.buf.as_ref(), scale.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, cols as u32])], groups(words));
        Ternary { ctx: self.ctx.clone(), buf: Arc::new(out), scale: scale.buf.clone(), rows, cols }
    }
    /// Multiply-free ternary matmul: x [rows,in] · Wᵀ where W is ternary [out,in]. Returns [rows,out].
    pub fn matmul_ternary(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dims mismatch");
        let out = empty(&self.ctx, rows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_WGSL, "matmul_ternary", &[x.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out, &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, 0])], groups(rows * w.rows));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
    /// Multiply-free ternary matmul (add/sub/skip via branchless select) — see MATMUL_TERNARY_MF_WGSL.
    pub fn matmul_ternary_mf(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dims mismatch");
        let out = empty(&self.ctx, rows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_MF_WGSL, "matmul_ternary_mf", &[x.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out, &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, 0])], groups(rows * w.rows));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
    /// Cooperative-matrix (tensor-core) ternary matmul — same 8x8 tile structure as `matmul_q8_0_coop`.
    /// Unpacks a ternary weight tile into shared memory, then uses the matrix units. NOT multiply-free by
    /// design: the test is whether tensor cores beat multiply-free scalar arithmetic on GPU.
    pub fn matmul_ternary_coop(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0 && inn % 8 == 0, "matmul_ternary_coop needs N,K multiples of 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_COOP_WGSL, "matmul_ternary_coop",
            &[xp.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, 0])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
    /// Wider N-tile (8x16 per workgroup) — reuses the activation tile across two matmuls.
    pub fn matmul_ternary_coop_n16(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 16 == 0 && inn % 8 == 0, "matmul_ternary_coop_n16 needs N mult 16, K mult 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_COOP_N16_WGSL, "matmul_ternary_coop_n16",
            &[xp.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, 0])],
            ((w.rows / 16) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
    /// N=32 tile — activation reused 4x.
    pub fn matmul_ternary_coop_n32(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 32 == 0 && inn % 8 == 0, "matmul_ternary_coop_n32 needs N mult 16, K mult 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_COOP_N32_WGSL, "matmul_ternary_coop_n32",
            &[xp.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, 0])],
            ((w.rows / 32) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
    /// N=64 tile — activation reused 8x; tests the register-pressure limit.
    pub fn matmul_ternary_coop_n64(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 64 == 0 && inn % 8 == 0, "matmul_ternary_coop_n64 needs N mult 16, K mult 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_COOP_N64_WGSL, "matmul_ternary_coop_n64",
            &[xp.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, 0])],
            ((w.rows / 64) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
    /// Deeper K-blocking (32) — four coopMultiplyAdds per barrier round.
    pub fn matmul_ternary_coop4(&self, w: &Ternary) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0 && inn % 32 == 0, "matmul_ternary_coop4 needs N mult of 8, K mult of 32");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        run(&self.ctx, MATMUL_TERNARY_COOP4_WGSL, "matmul_ternary_coop4",
            &[xp.buf.as_ref(), w.buf.as_ref(), w.scale.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, 0])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
}
impl Ternary {
    pub fn nbytes(&self) -> usize { (self.rows * self.cols * 2).div_ceil(8) }
}

const QUANT_TERNARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        inp: array<f32>;
@group(0) @binding(1) var<storage,read>        scale: array<f32>; // [rows] absmean
@group(0) @binding(2) var<storage,read_write>  out: array<u32>;   // 16 ternary codes per word
@group(0) @binding(3) var<storage,read>        info: array<u32>;  // rows, cols
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x; let rows = info[0]; let cols = info[1]; let n = rows * cols; let words = (n + 15u) / 16u;
    if (w >= words) { return; }
    var word: u32 = 0u;
    for (var lane: u32 = 0u; lane < 16u; lane = lane + 1u) {
        let idx = 16u * w + lane;
        if (idx < n) {
            var s = scale[idx / cols]; if (s == 0.0) { s = 1.0; }
            let t = clamp(round(inp[idx] / s), -1.0, 1.0);      // {−1,0,+1}
            let code = u32(i32(t) + 1);                          // {0,1,2}
            word = word | (code << (2u * lane));
        }
    }
    out[w] = word;
}
"#;

const MATMUL_TERNARY_COOP_N64_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:     array<f32>;
@group(0) @binding(1) var<storage,read>       tw:    array<u32>;
@group(0) @binding(2) var<storage,read>       scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> c:     array<f32>;
@group(0) @binding(4) var<uniform>            dims:  vec4<u32>;
var<workgroup> bs: array<f32, 512>;   // EIGHT 8x8 weight tiles (N-tile = 64)
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // Push the intensity lever to its limit: activation tile loaded ONCE, reused EIGHT times. Eight live
    // accumulators start pressuring registers, so this should find where the direction stops paying.
    let kk = dims.y; let nn = dims.z;
    let m0 = wid.y * 8u; let n0 = wid.x * 64u; let t = lid.x;
    var a0 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 0u], nn);
    var a1 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 8u], nn);
    var a2 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 16u], nn);
    var a3 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 24u], nn);
    var a4 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 32u], nn);
    var a5 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 40u], nn);
    var a6 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 48u], nn);
    var a7 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 56u], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 16u; e = e + 1u) {      // 512 values / 32 threads
            let i = t + e * 32u;
            let q = i / 64u; let rem = i % 64u; let kl = rem / 8u; let nl = rem % 8u;
            let n = n0 + q * 8u + nl; let k = k0 + kl;
            let widx = n * kk + k;
            let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u;
            bs[i] = f32(i32(code) - 1) * scale[n];
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        a0 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u), a0);
        a1 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[64], 8u), a1);
        a2 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[128], 8u), a2);
        a3 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[192], 8u), a3);
        a4 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[256], 8u), a4);
        a5 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[320], 8u), a5);
        a6 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[384], 8u), a6);
        a7 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[448], 8u), a7);
        workgroupBarrier();
    }
    coopStoreT(a0, &c[m0 * nn + n0 + 0u], nn);
    coopStoreT(a1, &c[m0 * nn + n0 + 8u], nn);
    coopStoreT(a2, &c[m0 * nn + n0 + 16u], nn);
    coopStoreT(a3, &c[m0 * nn + n0 + 24u], nn);
    coopStoreT(a4, &c[m0 * nn + n0 + 32u], nn);
    coopStoreT(a5, &c[m0 * nn + n0 + 40u], nn);
    coopStoreT(a6, &c[m0 * nn + n0 + 48u], nn);
    coopStoreT(a7, &c[m0 * nn + n0 + 56u], nn);
}
"#;

const MATMUL_TERNARY_COOP_N32_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:     array<f32>;
@group(0) @binding(1) var<storage,read>       tw:    array<u32>;
@group(0) @binding(2) var<storage,read>       scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> c:     array<f32>;
@group(0) @binding(4) var<uniform>            dims:  vec4<u32>;
var<workgroup> bs: array<f32, 256>;   // FOUR 8x8 weight tiles (N-tile = 32)
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // Push the winning lever: 8x32 output tile ⇒ the activation tile is loaded ONCE and reused across FOUR
    // matmuls (vs 2 at N=16). Note this uses the SAME 256 floats of shared memory as the K=32 experiment that
    // REGRESSED — the difference is that this spend BUYS arithmetic intensity, which K=32 did not.
    let kk = dims.y; let nn = dims.z;
    let m0 = wid.y * 8u; let n0 = wid.x * 32u; let t = lid.x;
    var a0 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0], nn);
    var a1 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 8u], nn);
    var a2 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 16u], nn);
    var a3 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 24u], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 8u; e = e + 1u) {
            let i = t + e * 32u;
            let q = i / 64u; let rem = i % 64u; let kl = rem / 8u; let nl = rem % 8u;
            let n = n0 + q * 8u + nl; let k = k0 + kl;
            let widx = n * kk + k;
            let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u;
            bs[i] = f32(i32(code) - 1) * scale[n];
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);   // ONE load, FOUR uses
        a0 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u), a0);
        a1 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[64], 8u), a1);
        a2 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[128], 8u), a2);
        a3 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[192], 8u), a3);
        workgroupBarrier();
    }
    coopStoreT(a0, &c[m0 * nn + n0], nn);
    coopStoreT(a1, &c[m0 * nn + n0 + 8u], nn);
    coopStoreT(a2, &c[m0 * nn + n0 + 16u], nn);
    coopStoreT(a3, &c[m0 * nn + n0 + 24u], nn);
}
"#;

const MATMUL_TERNARY_COOP_N16_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:     array<f32>;
@group(0) @binding(1) var<storage,read>       tw:    array<u32>;
@group(0) @binding(2) var<storage,read>       scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> c:     array<f32>;
@group(0) @binding(4) var<uniform>            dims:  vec4<u32>;   // M, K, N, _
var<workgroup> bs: array<f32, 128>;   // TWO 8x8 weight tiles (N-tile = 16)
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // WIDER N-TILE (8x16 output per workgroup, two accumulators). The K=32 experiment showed the bottleneck is
    // NOT barrier count, so target ARITHMETIC INTENSITY instead: the activation tile `ma` is loaded ONCE and
    // reused across TWO coopMultiplyAdds, halving global x-traffic per output element.
    let kk = dims.y; let nn = dims.z;
    let m0 = wid.y * 8u; let n0 = wid.x * 16u; let t = lid.x;
    var acc0 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0], nn);
    var acc1 = coopLoadT<coop_mat8x8<f32, C>>(&c[m0 * nn + n0 + 8u], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 4u; e = e + 1u) {          // 128 values / 32 threads = 4 each
            let i = t + e * 32u;
            let half = i / 64u; let rem = i % 64u; let kl = rem / 8u; let nl = rem % 8u;
            let n = n0 + half * 8u + nl; let k = k0 + kl;
            let widx = n * kk + k;
            let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u;
            bs[i] = f32(i32(code) - 1) * scale[n];
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);   // loaded ONCE, used twice
        acc0 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u), acc0);
        acc1 = coopMultiplyAdd(ma, coopLoadT<coop_mat8x8<f32, B>>(&bs[64], 8u), acc1);
        workgroupBarrier();
    }
    coopStoreT(acc0, &c[m0 * nn + n0], nn);
    coopStoreT(acc1, &c[m0 * nn + n0 + 8u], nn);
}
"#;

const MATMUL_TERNARY_COOP4_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:     array<f32>;
@group(0) @binding(1) var<storage,read>       tw:    array<u32>;
@group(0) @binding(2) var<storage,read>       scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> c:     array<f32>;
@group(0) @binding(4) var<uniform>            dims:  vec4<u32>;   // M, K, N, _
var<workgroup> bs: array<f32, 256>;   // FOUR 8x8 weight tiles (K-block = 32)
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // DEEPER K-BLOCKING: the 8-wide version pays a workgroupBarrier PAIR for every single coopMultiplyAdd.
    // Staging four 8x8 weight tiles per barrier round amortizes that 4x — the same total matmul work with a
    // quarter of the synchronisation.
    let kk = dims.y; let nn = dims.z;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 32u) {
        for (var e: u32 = 0u; e < 8u; e = e + 1u) {          // 256 values / 32 threads = 8 each
            let i = t + e * 32u;
            let j = i / 64u; let rem = i % 64u; let kl = rem / 8u; let nl = rem % 8u;
            let n = n0 + nl; let k = k0 + j * 8u + kl;
            let widx = n * kk + k;
            let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u;
            bs[i] = f32(i32(code) - 1) * scale[n];
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0 + j * 8u], kk);
            let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[j * 64u], 8u);
            acc = coopMultiplyAdd(ma, mb, acc);
        }
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

const MATMUL_TERNARY_COOP_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:     array<f32>;   // [M,K]
@group(0) @binding(1) var<storage,read>       tw:    array<u32>;   // packed ternary [N,K], 16 codes/word
@group(0) @binding(2) var<storage,read>       scale: array<f32>;   // [N] absmean
@group(0) @binding(3) var<storage,read_write> c:     array<f32>;   // [M,N]
@group(0) @binding(4) var<uniform>            dims:  vec4<u32>;    // M, K, N, _
var<workgroup> bs: array<f32, 64>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // Same cooperative-matrix (tensor-core) structure as MATMUL_Q8_0_COOP: unpack a weight TILE into shared
    // memory, then let the matrix units do the multiply-accumulate. This is deliberately NOT multiply-free —
    // the point is to test whether tensor cores beat multiply-free scalar arithmetic on a GPU. Ternary's real
    // GPU advantage is 16× fewer weight BYTES; this kernel is what actually cashes it.
    let kk = dims.y; let nn = dims.z;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = t + e * 32u; let nl = i / 8u; let kl = i % 8u;
            let n = n0 + nl; let k = k0 + kl;
            let widx = n * kk + k;
            let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u;  // 0=-1, 1=0, 2=+1
            bs[kl * 8u + nl] = f32(i32(code) - 1) * scale[n];
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u);
        acc = coopMultiplyAdd(ma, mb, acc);
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

const MATMUL_TERNARY_MF_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read>        tw: array<u32>;
@group(0) @binding(2) var<storage,read>        scale: array<f32>;
@group(0) @binding(3) var<storage,read_write>  out: array<f32>;
@group(0) @binding(4) var<uniform>             info: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    // GENUINELY MULTIPLY-FREE: accumulate +x / −x / skip. `select` is branchless (no warp divergence),
    // so this is the honest test of whether removing the multiply helps on a GPU — where fused
    // multiply-add is already a single instruction, unlike CPUs where BitNet.cpp/T-MAC win big.
    var pos = 0.0; var neg = 0.0;
    for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
        let widx = o * in_dim + i;
        let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u;   // 0=−1, 1=0, 2=+1
        let xv = x[r * in_dim + i];
        pos = pos + select(0.0, xv, code == 2u);
        neg = neg + select(0.0, xv, code == 0u);
    }
    out[idx] = (pos - neg) * scale[o];
}
"#;

const MATMUL_TERNARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;     // [rows, in]
@group(0) @binding(1) var<storage,read>        tw: array<u32>;    // packed ternary [out, in]
@group(0) @binding(2) var<storage,read>        scale: array<f32>; // [out]
@group(0) @binding(3) var<storage,read_write>  out: array<f32>;   // [rows, out]
@group(0) @binding(4) var<uniform>             info: vec4<u32>;   // rows, out, in
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    var acc = 0.0;
    for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
        let widx = o * in_dim + i;
        let code = (tw[widx / 16u] >> (2u * (widx % 16u))) & 3u; // {0,1,2}
        let t = f32(i32(code) - 1);                              // {−1,0,+1}  (multiply-free in spirit)
        acc = acc + x[r * in_dim + i] * t;
    }
    out[idx] = acc * scale[o];
}
"#;

/// PrismML **Q2_0** ternary weights held on the GPU in their native packed form (group-128 blocks:
/// `f16 d` + 32 bytes of 2-bit codes = 34 B / 128 weights ≈ 2.125 bpw). A 27B model stays ~7 GB
/// instead of the 108 GB it would need dequantized to f32 — so the matmul must read the packed
/// blocks directly, which is what `Tensor::matmul_q2_0` does.
pub struct Q2_0Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 8 u32 per block — 16 two-bit codes per word, u32-aligned
    scales: Arc<wgpu::Buffer>, // f16 per block, two packed per u32
    pub rows: usize, // out features
    pub cols: usize, // in features (multiple of 128)
}

impl Q2_0Weights {
    /// Upload raw Q2_0 block bytes (as they appear in the GGUF) for an [out, in] weight.
    ///
    /// The on-disk block is `f16 d` + 32 code bytes = **34 bytes**, which is not a multiple of 4 —
    /// so a shader can't address the codes as `u32` and is forced into a byte-extract that re-reads
    /// the same word once per weight (16× the necessary traffic). Since the GPU-side layout is ours
    /// to choose, split the blocks on upload into an aligned codes array and a separate scales
    /// array. Identical bytes and identical math, but the inner loop reads 8 words per block instead
    /// of 128.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q2_0Weights {
        assert_eq!(cols % 128, 0, "Q2_0 rows must be a multiple of 128");
        assert_eq!(bytes.len(), rows * (cols / 128) * 34, "unexpected Q2_0 byte length");
        let bpr = cols / 128; // blocks per output row
        let nblk = rows * bpr;
        let mut codes: Vec<u32> = vec![0; nblk * 8];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        // **Output-major (transposed) layout.** In a GEMV every weight byte is read exactly once, so
        // the only way to coalesce is for adjacent *threads* to read adjacent bytes — and adjacent
        // threads own adjacent outputs. Indexing by [word][output] rather than [output][word] lets a
        // 32-wide SIMD group sweep one contiguous run while each thread still owns a whole output:
        // no reduction, no barriers, full work per thread. Row-major forces a choice between the
        // two — threads-per-output land 1280 B apart, and split-K coalesces but leaves ~5 words per
        // thread against a 6-barrier tree. Both measured ~70 GB/s against a 325 GB/s ceiling.
        let transposed = q2_0_transposed();
        for b in 0..nblk {
            let src = &bytes[b * 34..b * 34 + 34];
            let (o, blk) = (b / bpr, b % bpr); // this block belongs to output o
            let d = u16::from_le_bytes([src[0], src[1]]) as u32;
            let si = if transposed { blk * rows + o } else { b };
            scales[si / 2] |= d << (16 * (si % 2));
            for w in 0..8 {
                let c = &src[2 + w * 4..2 + w * 4 + 4];
                let ci = if transposed { (blk * 8 + w) * rows + o } else { b * 8 + w };
                codes[ci] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Q2_0Weights { ctx: ctx.clone(), codes: mk("q2_0.codes", &codes), scales: mk("q2_0.scales", &scales), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 128) * 34 }
}

/// **STQ1_0** weights held packed on the GPU — Tencent's 1.3125-bpw ternary, the lowest-rate format
/// Ferric runs.
///
/// This is the format's whole point. Dequantising a 1.3125 bpw tensor to f32 on load costs **24.4×
/// its on-disk footprint**, which is more than the entire saving the format exists to deliver: a
/// model stored at 1.3 bits and resident at 32 is not a low-bit model. Reading the packed bytes in
/// the kernel is not an optimisation here, it is the reason to support the format at all.
///
/// The on-disk block is `qs[32] | sign[8] | f16 d` = 42 bytes, which is not a multiple of 4, so the
/// same repack-on-upload trick as [`Q2_0Weights`] applies — three aligned arrays instead of one
/// byte-addressed blob. And the same **output-major** layout, for the same measured reason: in a
/// GEMV every weight word is read exactly once, so coalescing requires adjacent *threads* to read
/// adjacent addresses, and adjacent threads own adjacent outputs.
pub struct Stq1_0Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 8 u32 per block — 8 four-bit slot codes per word
    signs: Arc<wgpu::Buffer>,  // 2 u32 per block — one bit per group of 4
    scales: Arc<wgpu::Buffer>, // f16 per block, two packed per u32
    /// The 32 codebook patterns pre-expanded to their four lane values, `[-1, 0, +1]` as f32.
    ///
    /// 512 bytes in a storage buffer rather than 32 words in `var<private>`. WGSL scopes `private`
    /// PER INVOCATION, so a dynamically-indexed private array is per-thread state the compiler has
    /// to keep somewhere — and pre-expanding the lanes also deletes the shift-mask-subtract the
    /// shader would otherwise redo for every weight.
    codebook: Arc<wgpu::Buffer>,
    pub rows: usize, // out features
    pub cols: usize, // in features (multiple of 256)
}

impl Stq1_0Weights {
    /// Upload raw STQ1_0 block bytes exactly as they appear in the GGUF, for an `[out, in]` weight.
    ///
    /// ⚠ The scale is the LAST field of the block, not the first. Every other packed type here reads
    /// `d` from `src[0..2]`; this one reads `src[40..42]`, and the obvious copy-paste produces a
    /// plausible small float taken from eight packed slot codes.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Stq1_0Weights {
        assert_eq!(cols % 256, 0, "STQ1_0 cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 42, "unexpected STQ1_0 byte length");
        let bpr = cols / 256; // blocks per output row
        let nblk = rows * bpr;
        let mut codes: Vec<u32> = vec![0; nblk * 8];
        let mut signs: Vec<u32> = vec![0; nblk * 2];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        for b in 0..nblk {
            let src = &bytes[b * 42..b * 42 + 42];
            let (o, blk) = (b / bpr, b % bpr);
            let si = blk * rows + o;
            scales[si / 2] |= (u16::from_le_bytes([src[40], src[41]]) as u32) << (16 * (si % 2));
            for w in 0..8 {
                let c = &src[w * 4..w * 4 + 4];
                codes[(blk * 8 + w) * rows + o] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            for w in 0..2 {
                let c = &src[32 + w * 4..32 + w * 4 + 4];
                signs[(blk * 2 + w) * rows + o] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        // Four f32 lanes per codebook entry, in the same (sign << 4) | slot order the shader uses.
        let cb: Vec<f32> = (0..32).flat_map(|i| {
            let q = crate::dtype::STQ1_0_SHADER_CODEBOOK[i];
            (0..4).map(move |p| ((q >> (2 * p)) & 3) as f32 - 1.0)
        }).collect();
        Stq1_0Weights {
            ctx: ctx.clone(),
            codes: mk("stq1_0.codes", &codes),
            signs: mk("stq1_0.signs", &signs),
            scales: mk("stq1_0.scales", &scales),
            codebook: Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stq1_0.codebook"), contents: bytemuck::cast_slice(&cb),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })),
            rows, cols,
        }
    }
    /// Resident bytes. Equal to the on-disk size — the repack rearranges, it does not expand.
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 42 }
}

/// Build the shader-side grid buffer for a codebook quant. Two kilobytes, uploaded once per weight
/// — negligible beside the weight itself, and it keeps the table out of `var<private>`, which WGSL
/// scopes PER INVOCATION: a 512-word private array is 2 KiB of register/stack pressure on every
/// thread, not one shared copy.
fn grid_buffer(ctx: &Arc<Context>, label: &str, data: &[u32]) -> Arc<wgpu::Buffer> {
    Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    }))
}

/// **IQ2_XXS** weights held packed on the GPU — 2.0625 bpw, and the format that carries more of a
/// low-bit MoE checkpoint than anything else in it.
///
/// The block is `f16 d` + 32 `u16` = 66 bytes for 256 weights, read as eight 32-weight groups of two
/// `u32`. The low word holds four grid indices, one byte each; the high word packs four 7-bit sign
/// indices and a 4-bit sub-scale — `4·7 + 4 = 32` exactly, with no spare bit.
///
/// Almost all of the information is *which* of 256 magnitude patterns and *which* of 128 sign
/// patterns, not the magnitudes: every byte of the grid is one of `{8, 25, 43}`.
pub struct Iq2XxsWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 16 u32 per block, output-major
    scales: Arc<wgpu::Buffer>, // f16 per block, two per u32
    grid: Arc<wgpu::Buffer>,
    pub rows: usize,
    pub cols: usize,
}

impl Iq2XxsWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Iq2XxsWeights {
        assert_eq!(cols % 256, 0, "IQ2_XXS cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 66, "unexpected IQ2_XXS byte length");
        let bpr = cols / 256;
        let nblk = rows * bpr;
        let mut codes: Vec<u32> = vec![0; nblk * 16];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        for b in 0..nblk {
            let src = &bytes[b * 66..b * 66 + 66];
            let (o, blk) = (b / bpr, b % bpr);
            let si = blk * rows + o;
            scales[si / 2] |= (u16::from_le_bytes([src[0], src[1]]) as u32) << (16 * (si % 2));
            for w in 0..16 {
                let c = &src[2 + w * 4..2 + w * 4 + 4];
                codes[(blk * 16 + w) * rows + o] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Iq2XxsWeights {
            ctx: ctx.clone(), codes: mk("iq2xxs.codes", &codes), scales: mk("iq2xxs.scales", &scales),
            grid: grid_buffer(ctx, "iq2xxs.grid", &crate::iq_grids::IQ2XXS_GRID_U32),
            rows, cols,
        }
    }
    /// Resident bytes, excluding the shared 2 KiB grid.
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 66 }
}

/// **IQ3_XXS** weights held packed on the GPU — 3.0625 bpw.
///
/// The block is `f16 d` + 96 bytes for 256 weights, and ⚠ the two halves are **not interleaved**:
/// all 64 grid-index bytes come first, then the eight sign-and-scale words. Reading them as one
/// interleaved stream preserves the block size, the element count and the value distribution, and
/// destroys the pairing.
///
/// A grid point is only four bytes here, so a sub-block of eight takes two lookups and the sign
/// byte splits across them — bits 0..3 for the first, 4..7 for the second. The sub-scale multiplier
/// is `0.5`, not IQ2_XXS's `0.25`.
pub struct Iq3XxsWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 24 u32 per block (16 index words then 8 sign/scale words)
    scales: Arc<wgpu::Buffer>,
    grid: Arc<wgpu::Buffer>,
    pub rows: usize,
    pub cols: usize,
}

impl Iq3XxsWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Iq3XxsWeights {
        assert_eq!(cols % 256, 0, "IQ3_XXS cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 98, "unexpected IQ3_XXS byte length");
        let bpr = cols / 256;
        let nblk = rows * bpr;
        let mut codes: Vec<u32> = vec![0; nblk * 24];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        for b in 0..nblk {
            let src = &bytes[b * 98..b * 98 + 98];
            let (o, blk) = (b / bpr, b % bpr);
            let si = blk * rows + o;
            scales[si / 2] |= (u16::from_le_bytes([src[0], src[1]]) as u32) << (16 * (si % 2));
            for w in 0..24 {
                let c = &src[2 + w * 4..2 + w * 4 + 4];
                codes[(blk * 24 + w) * rows + o] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Iq3XxsWeights {
            ctx: ctx.clone(), codes: mk("iq3xxs.codes", &codes), scales: mk("iq3xxs.scales", &scales),
            grid: grid_buffer(ctx, "iq3xxs.grid", &crate::iq_grids::IQ3XXS_GRID_U32),
            rows, cols,
        }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 98 }
}

/// **Q4_0** weights held packed on the GPU — the canonical llama.cpp 4-bit format (blocks of 32:
/// `f16 scale` + 16 nibble-bytes; value = (nibble − 8)·scale). Most quantized GGUF models on Hugging
/// Face ship in Q4-family formats, so a *native* packed matmul (dequant in-kernel, weights never
/// expanded to f32) is what makes Ferric fast — and 8× lighter — on the standard model ecosystem, the
/// way `Q2_0Weights` does for ternary. Same repack-on-upload trick: the 18-byte block isn't u32-
/// aligned, so split it into an aligned `codes` array (4 u32/block) and a separate `scales` array.
pub struct Q4_0Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 4 u32 per block (16 nibble-bytes)
    scales: Arc<wgpu::Buffer>, // f16 per block, two packed per u32
    pub rows: usize,           // out features
    pub cols: usize,           // in features (multiple of 32)
}

impl Q4_0Weights {
    /// Upload raw Q4_0 block bytes (exactly as they appear in the GGUF) for an [out, in] weight.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q4_0Weights {
        assert_eq!(cols % 32, 0, "Q4_0 cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 18, "unexpected Q4_0 byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        for b in 0..nblk {
            let src = &bytes[b * 18..b * 18 + 18];
            let d = u16::from_le_bytes([src[0], src[1]]) as u32;
            scales[b / 2] |= d << (16 * (b % 2));
            for w in 0..4 {
                let c = &src[2 + w * 4..2 + w * 4 + 4];
                codes[b * 4 + w] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Q4_0Weights { ctx: ctx.clone(), codes: mk("q4_0.codes", &codes), scales: mk("q4_0.scales", &scales), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 18 }
}

/// **Q4_1** weights held packed on the GPU — llama.cpp's affine 4-bit (`value = nibble·d + m`, no
/// −8). Same repack trick as Q4_0; the 20-byte block adds an `f16 min`, so `d` and `m` are packed
/// together — one u32 per block — and read back with `unpack2x16float` as `(d, m)`.
pub struct Q4_1Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 4 u32 per block (16 nibble-bytes)
    scales: Arc<wgpu::Buffer>, // (d, m) f16 pair packed one u32 per block
    pub rows: usize,           // out features
    pub cols: usize,           // in features (multiple of 32)
}

impl Q4_1Weights {
    /// Upload raw Q4_1 block bytes (exactly as they appear in the GGUF) for an [out, in] weight.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q4_1Weights {
        assert_eq!(cols % 32, 0, "Q4_1 cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 20, "unexpected Q4_1 byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut scales: Vec<u32> = vec![0; nblk];
        for b in 0..nblk {
            let src = &bytes[b * 20..b * 20 + 20];
            let d = u16::from_le_bytes([src[0], src[1]]) as u32;
            let m = u16::from_le_bytes([src[2], src[3]]) as u32;
            scales[b] = d | (m << 16); // unpack2x16float → (d, m)
            for w in 0..4 {
                let c = &src[4 + w * 4..4 + w * 4 + 4];
                codes[b * 4 + w] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Q4_1Weights { ctx: ctx.clone(), codes: mk("q4_1.codes", &codes), scales: mk("q4_1.scales", &scales), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 20 }
}

/// **Q5_0** weights held packed on the GPU — llama.cpp's symmetric 5-bit (`value = (nibble|5th-bit)
/// − 16, ·d`). The 5th bits live in a per-block `u32 qh`; `scales` holds two words per block —
/// `[qh, d]` — so the kernel stays within the 4-storage-buffer budget (x, codes, scales, out).
pub struct Q5_0Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 4 u32 per block (16 nibble-bytes)
    scales: Arc<wgpu::Buffer>, // [qh (u32), d (f16 in low 16 bits)] per block
    pub rows: usize,           // out features
    pub cols: usize,           // in features (multiple of 32)
}

impl Q5_0Weights {
    /// Upload raw Q5_0 block bytes (exactly as they appear in the GGUF) for an [out, in] weight.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q5_0Weights {
        assert_eq!(cols % 32, 0, "Q5_0 cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 22, "unexpected Q5_0 byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut scales: Vec<u32> = vec![0; nblk * 2];
        for b in 0..nblk {
            let src = &bytes[b * 22..b * 22 + 22];
            let d = u16::from_le_bytes([src[0], src[1]]) as u32;
            let qh = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);
            scales[b * 2] = qh;
            scales[b * 2 + 1] = d; // unpack2x16float(.).x = d
            for w in 0..4 {
                let c = &src[6 + w * 4..6 + w * 4 + 4];
                codes[b * 4 + w] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Q5_0Weights { ctx: ctx.clone(), codes: mk("q5_0.codes", &codes), scales: mk("q5_0.scales", &scales), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 22 }
    /// Overwrite `n` consecutive rows in place. See [`Q4_KWeights::write_rows`].
    ///
    /// Unlike Q8_0 nothing is shared between blocks here — each block owns `scales[2b]` (qh) and
    /// `scales[2b+1]` (d) — so any row range is a clean byte range and no alignment rule applies.
    pub fn write_rows(&self, row0: usize, bytes: &[u8], n_rows: usize) -> Result<(), String> {
        let bpr = self.cols / 32;
        if row0 + n_rows > self.rows {
            return Err(format!("rows {row0}..{} exceed the weight's {} rows", row0 + n_rows, self.rows));
        }
        if bytes.len() != n_rows * bpr * 22 {
            return Err(format!("{} bytes for {n_rows} rows x {bpr} blocks (need {})",
                               bytes.len(), n_rows * bpr * 22));
        }
        let (blk0, nblk) = (row0 * bpr, n_rows * bpr);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut scales: Vec<u32> = vec![0; nblk * 2];
        for b in 0..nblk {
            let src = &bytes[b * 22..b * 22 + 22];
            scales[b * 2] = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);        // qh
            scales[b * 2 + 1] = u16::from_le_bytes([src[0], src[1]]) as u32;             // d
            for w in 0..4 {
                let c = &src[6 + w * 4..6 + w * 4 + 4];
                codes[b * 4 + w] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        self.ctx.queue.write_buffer(&self.codes, (blk0 * 4 * 4) as u64, bytemuck::cast_slice(&codes));
        self.ctx.queue.write_buffer(&self.scales, (blk0 * 2 * 4) as u64, bytemuck::cast_slice(&scales));
        Ok(())
    }

}

/// **Q5_1** weights held packed on the GPU — the affine 5-bit (`value = (nibble|5th-bit)·d + m`).
/// Combines Q5_0's `qh` with Q4_1's `min`; `scales` holds two words per block — `[pack(d,m), qh]`.
pub struct Q5_1Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 4 u32 per block (16 nibble-bytes)
    scales: Arc<wgpu::Buffer>, // [(d,m) f16 pair, qh (u32)] per block
    pub rows: usize,           // out features
    pub cols: usize,           // in features (multiple of 32)
}

impl Q5_1Weights {
    /// Upload raw Q5_1 block bytes (exactly as they appear in the GGUF) for an [out, in] weight.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q5_1Weights {
        assert_eq!(cols % 32, 0, "Q5_1 cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 24, "unexpected Q5_1 byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut scales: Vec<u32> = vec![0; nblk * 2];
        for b in 0..nblk {
            let src = &bytes[b * 24..b * 24 + 24];
            let d = u16::from_le_bytes([src[0], src[1]]) as u32;
            let m = u16::from_le_bytes([src[2], src[3]]) as u32;
            let qh = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
            scales[b * 2] = d | (m << 16); // unpack2x16float → (d, m)
            scales[b * 2 + 1] = qh;
            for w in 0..4 {
                let c = &src[8 + w * 4..8 + w * 4 + 4];
                codes[b * 4 + w] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let mk = |label, data: &[u32]| {
            Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }))
        };
        Q5_1Weights { ctx: ctx.clone(), codes: mk("q5_1.codes", &codes), scales: mk("q5_1.scales", &scales), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 24 }
}

/// A packed-quant weight matrix of *any* supported GGUF format, behind one `matmul_q`. This is what
/// makes a model loader format-agnostic: build a `QMatrix` per weight from its ggml type, and the
/// same forward code runs a Q2_0 ternary model, a Q4_K_M model, a Q8_0 model, … — each with its
/// weights dequantized inside the matmul, never expanded to f32.
/// One packed-quant weight shard that fits in a single GPU storage buffer.
pub enum QShard {
    Q2_0(Q2_0Weights),
    Stq1_0(Stq1_0Weights),
    Iq2Xxs(Iq2XxsWeights),
    Iq3Xxs(Iq3XxsWeights),
    Q4_0(Q4_0Weights),
    Q4_1(Q4_1Weights),
    Q5_0(Q5_0Weights),
    Q5_1(Q5_1Weights),
    Q2_K(Q2_KWeights),
    Q3_K(Q3_KWeights),
    Q4_K(Q4_KWeights),
    Q5_K(Q5_KWeights),
    Q6_K(Q6_KWeights),
    Q8_0(Q8_0Weights),
    Iq4Xs(Iq4XsWeights),
    Iq4Nl(Iq4NlWeights),
    Mxfp4(Mxfp4Weights),
    /// Fallback for any GGUF quant with no native packed kernel yet (e.g. IQ4_NL): the weight
    /// is dequantized to f32 on load and run through a plain matmul. Correct and format-complete, at
    /// the cost of f32 weight memory — a native kernel can replace it later purely as a speed/size win.
    Dense(DenseWeight),
}

/// A dequantized weight held as `Wᵀ` (`[cols, rows]`, f32) on the GPU, so `x[T,cols]·Wᵀ = y[T,rows]`
/// is one ordinary matmul. The dequant-on-load fallback that makes IQ-class GGUFs runnable.
pub struct DenseWeight {
    wt: Tensor,
    rows: usize,
    cols: usize,
}

impl DenseWeight {
    fn nbytes(&self) -> usize { self.rows * self.cols * 4 }
    /// `w` is a row-major `[rows, cols]` (already-dequantized) weight; store it transposed as
    /// `Wᵀ = [cols, rows]` so `x·Wᵀ` is a plain matmul.
    fn from_f32(ctx: &Arc<Context>, w: &[f32], rows: usize, cols: usize) -> DenseWeight {
        let mut wt = vec![0f32; rows * cols];
        for r in 0..rows { for c in 0..cols { wt[c * rows + r] = w[r * cols + c]; } }
        DenseWeight { wt: Tensor::from_vec(ctx, &wt, &[cols, rows]), rows, cols }
    }
}

impl QShard {
    fn rows(&self) -> usize { match self { QShard::Iq2Xxs(w) => w.rows, QShard::Iq3Xxs(w) => w.rows, QShard::Stq1_0(w) => w.rows, QShard::Q2_0(w) => w.rows, QShard::Q4_0(w) => w.rows, QShard::Q4_1(w) => w.rows, QShard::Q5_0(w) => w.rows, QShard::Q5_1(w) => w.rows, QShard::Q2_K(w) => w.rows, QShard::Q3_K(w) => w.rows, QShard::Q4_K(w) => w.rows, QShard::Q5_K(w) => w.rows, QShard::Q6_K(w) => w.rows, QShard::Q8_0(w) => w.rows, QShard::Iq4Xs(w) => w.rows, QShard::Iq4Nl(w) => w.rows, QShard::Mxfp4(w) => w.rows, QShard::Dense(w) => w.rows } }
    fn nbytes(&self) -> usize { match self { QShard::Iq2Xxs(w) => w.nbytes(), QShard::Iq3Xxs(w) => w.nbytes(), QShard::Stq1_0(w) => w.nbytes(), QShard::Q2_0(w) => w.nbytes(), QShard::Q4_0(w) => w.nbytes(), QShard::Q4_1(w) => w.nbytes(), QShard::Q5_0(w) => w.nbytes(), QShard::Q5_1(w) => w.nbytes(), QShard::Q2_K(w) => w.nbytes(), QShard::Q3_K(w) => w.nbytes(), QShard::Q4_K(w) => w.nbytes(), QShard::Q5_K(w) => w.nbytes(), QShard::Q6_K(w) => w.nbytes(), QShard::Q8_0(w) => w.nbytes(), QShard::Iq4Xs(w) => w.nbytes(), QShard::Iq4Nl(w) => w.nbytes(), QShard::Mxfp4(w) => w.nbytes(), QShard::Dense(w) => w.nbytes() } }
    fn build(ctx: &Arc<Context>, bytes: &[u8], ggml_type: u32, rows: usize, cols: usize) -> Result<QShard, String> {
        Ok(match ggml_type {
            2 => QShard::Q4_0(Q4_0Weights::from_bytes(ctx, bytes, rows, cols)),
            3 => QShard::Q4_1(Q4_1Weights::from_bytes(ctx, bytes, rows, cols)),
            6 => QShard::Q5_0(Q5_0Weights::from_bytes(ctx, bytes, rows, cols)),
            7 => QShard::Q5_1(Q5_1Weights::from_bytes(ctx, bytes, rows, cols)),
            8 => QShard::Q8_0(Q8_0Weights::from_bytes(ctx, bytes, rows, cols)),
            12 => QShard::Q4_K(Q4_KWeights::from_bytes(ctx, bytes, rows, cols)),
            13 => QShard::Q5_K(Q5_KWeights::from_bytes(ctx, bytes, rows, cols)),
            10 => QShard::Q2_K(Q2_KWeights::from_bytes(ctx, bytes, rows, cols)),
            11 => QShard::Q3_K(Q3_KWeights::from_bytes(ctx, bytes, rows, cols)),
            14 => QShard::Q6_K(Q6_KWeights::from_bytes(ctx, bytes, rows, cols)),
            20 => QShard::Iq4Nl(Iq4NlWeights::from_bytes(ctx, bytes, rows, cols)),
            23 => QShard::Iq4Xs(Iq4XsWeights::from_bytes(ctx, bytes, rows, cols)),
            39 => QShard::Mxfp4(Mxfp4Weights::from_bytes(ctx, bytes, rows, cols)),
            42 => QShard::Q2_0(Q2_0Weights::from_bytes(ctx, bytes, rows, cols)),
            43 => QShard::Stq1_0(Stq1_0Weights::from_bytes(ctx, bytes, rows, cols)),
            16 => QShard::Iq2Xxs(Iq2XxsWeights::from_bytes(ctx, bytes, rows, cols)),
            18 => QShard::Iq3Xxs(Iq3XxsWeights::from_bytes(ctx, bytes, rows, cols)),
            // Types with no native packed kernel take the dense fallback via `QMatrix::from_dense`
            // (the loader dequantizes them), so they never reach this packed-build path.
            other => return Err(format!("QMatrix: no native matmul for ggml type {other}")),
        })
    }
}

/// A packed-quant weight matrix of any supported GGUF format, **sharded across GPU buffers** so a
/// tensor larger than `maxStorageBufferBindingSize` (WebGPU baseline 128 MB) still loads — the split
/// is along output rows, and `matmul_q` runs each shard and concatenates, which is exact (`cat`). This
/// is what lets big-vocab LM heads / embeddings and larger models run in a browser tab. One shard is
/// the common case (no overhead); sharding kicks in only for oversized weights.
pub struct QMatrix {
    shards: Vec<QShard>,
    rows: usize,
    cols: usize,
}

impl QMatrix {
    /// ggml block-size in bytes for a supported type, or None if we have no native matmul for it.
    pub fn block_bytes(ggml_type: u32) -> Option<(usize, usize)> {
        match ggml_type {          // (values per block, bytes per block)
            2 => Some((32, 18)),   // Q4_0
            3 => Some((32, 20)),   // Q4_1
            6 => Some((32, 22)),   // Q5_0
            7 => Some((32, 24)),   // Q5_1
            8 => Some((32, 34)),   // Q8_0
            10 => Some((256, 84)), // Q2_K
            11 => Some((256, 110)),// Q3_K
            12 => Some((256, 144)),// Q4_K
            13 => Some((256, 176)),// Q5_K
            14 => Some((256, 210)),// Q6_K
            20 => Some((32, 18)),  // IQ4_NL
            23 => Some((256, 136)),// IQ4_XS
            39 => Some((32, 17)),  // MXFP4
            42 => Some((128, 34)), // Q2_0
            43 => Some((256, 42)), // STQ1_0
            16 => Some((256, 66)), // IQ2_XXS
            18 => Some((256, 98)), // IQ3_XXS
            _ => None,
        }
    }
    /// Build from raw GGUF block bytes for an [out(rows), in(cols)] weight, sharding along rows so no
    /// shard's buffers exceed the device's binding limit. The derived (codes/scales/aux) buffers are
    /// each ≤ the raw block bytes, so bounding raw shard bytes bounds every buffer.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], ggml_type: u32, rows: usize, cols: usize) -> Result<QMatrix, String> {
        let row_bytes = if rows == 0 { 0 } else { bytes.len() / rows };
        // Effective per-shard byte budget: the device limit, or a smaller test override, with headroom.
        let limit = std::env::var("FERRIC_MAX_BINDING").ok().and_then(|s| s.parse().ok())
            .unwrap_or_else(|| (ctx.max_binding as usize).saturating_sub(1 << 20).max(1 << 20));
        let max_rows = if row_bytes == 0 { rows } else { (limit / row_bytes).max(1) };
        let mut shards = Vec::new();
        let mut r0 = 0;
        while r0 < rows {
            let n = (rows - r0).min(max_rows);
            shards.push(QShard::build(ctx, &bytes[r0 * row_bytes..(r0 + n) * row_bytes], ggml_type, n, cols)?);
            r0 += n;
        }
        if shards.is_empty() { shards.push(QShard::build(ctx, bytes, ggml_type, rows, cols)?); }
        Ok(QMatrix { shards, rows, cols })
    }
    /// Build from an already-dequantized, row-major `[rows, cols]` weight — the fallback for GGUF
    /// quants with no native packed kernel (IQ4_XS/IQ4_NL/…): the loader dequantizes to f32 and hands
    /// it here. Sharded along rows to respect the device binding limit, exactly like `from_bytes`.
    pub fn from_dense(ctx: &Arc<Context>, w: &[f32], rows: usize, cols: usize) -> QMatrix {
        let row_bytes = cols * 4;
        let limit = std::env::var("FERRIC_MAX_BINDING").ok().and_then(|s| s.parse().ok())
            .unwrap_or_else(|| (ctx.max_binding as usize).saturating_sub(1 << 20).max(1 << 20));
        let max_rows = if row_bytes == 0 { rows.max(1) } else { (limit / row_bytes).max(1) };
        let mut shards = Vec::new();
        let mut r0 = 0;
        while r0 < rows {
            let n = (rows - r0).min(max_rows);
            shards.push(QShard::Dense(DenseWeight::from_f32(ctx, &w[r0 * cols..(r0 + n) * cols], n, cols)));
            r0 += n;
        }
        if shards.is_empty() { shards.push(QShard::Dense(DenseWeight::from_f32(ctx, w, rows, cols))); }
        QMatrix { shards, rows, cols }
    }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn nbytes(&self) -> usize { self.shards.iter().map(|s| s.nbytes()).sum() }
    pub fn n_shards(&self) -> usize { self.shards.len() }
}

impl Tensor {
    /// y = x·Wᵀ for a packed weight of any supported format (dispatches to the format's kernel).
    pub fn matmul_q(&self, w: &QMatrix) -> Tensor {
        if w.shards.len() == 1 { return self.matmul_qshard(&w.shards[0]); }
        // Sharded weight: each shard produces [rows, shard_out]; concatenate along the output dim.
        let mut acc: Option<Tensor> = None;
        for sh in &w.shards {
            let o = self.matmul_qshard(sh);
            acc = Some(match acc { None => o, Some(prev) => prev.cat(&o, 1) });
        }
        acc.unwrap()
    }
    /// Fused FFN gate/up + SwiGLU when the gate_up weight is a single Q4_K shard (the common case
    /// for Q4_K_M models). Returns `Some(silu(gate)·up)` — `[t, cols/2]` — computed in one kernel with
    /// no `[t, 2·n_ff]` intermediate; `None` when the weight isn't a lone Q4_K/Q5_K/Q6_K shard
    /// (caller falls back to `matmul_q(w).swiglu(n_ff)`).
    pub fn try_matmul_swiglu(&self, w: &QMatrix) -> Option<Tensor> {
        if w.shards.len() == 1 {
            match &w.shards[0] {
                QShard::Q4_K(sh) => return Some(self.matmul_q4_k_swiglu(sh)),
                QShard::Q5_K(sh) => return Some(self.matmul_q5_k_swiglu(sh)),
                QShard::Q6_K(sh) => return Some(self.matmul_q6_k_swiglu(sh)),
                _ => {}
            }
        }
        None
    }
    /// Full FFN in one dispatch when gate_up is a lone Q4_K shard and down a lone Q6_K shard (the
    /// Qwen3 Q4_K_M layout) and n_ff/n_embd are 256-block-aligned.
    ///
    /// MEASURED NEGATIVE, so OPT-IN (set `FERRIC_MEGA`): correct (token-for-token identical), but the
    /// one-workgroup-per-token design underfills the GPU at decode and runs **~2× slower** than the
    /// staged fused-SwiGLU + down path (46 vs ~25 ms/tok, Qwen3-0.6B-Q4_K_M, interleaved). The
    /// dispatch/intermediate-traffic saved is dwarfed by the occupancy lost. Kept, off by default, as
    /// a documented experiment — a batched/multi-workgroup redesign would be needed to make it pay.
    pub fn try_ffn_mega(&self, gate_up: &QMatrix, down: &QMatrix, n_ff: usize) -> Option<Tensor> {
        if std::env::var("FERRIC_MEGA").is_err() { return None; }
        if gate_up.shards.len() != 1 || down.shards.len() != 1 { return None; }
        let gu = match &gate_up.shards[0] { QShard::Q4_K(w) => w, _ => return None };
        let dn = match &down.shards[0] { QShard::Q6_K(w) => w, _ => return None };
        if gu.rows != 2 * n_ff || dn.cols != n_ff || n_ff % 256 != 0 || dn.rows % 256 != 0 { return None; }
        Some(self.ffn_mega_q4k_q6k(gu, dn, n_ff))
    }
    pub fn ffn_mega_q4k_q6k(&self, gu: &Q4_KWeights, dn: &Q6_KWeights, n_ff: usize) -> Tensor {
        let x = self.contiguous();
        let (rows, n_embd) = (x.shape[0], x.shape[1]);
        assert_eq!(n_embd, gu.cols, "gate_up in dim mismatch");
        assert_eq!(dn.rows, n_embd, "down out dim must equal n_embd");
        assert_eq!(dn.cols, n_ff, "down in dim must equal n_ff");
        let out = empty(&self.ctx, rows * n_embd);
        let src = FFN_MEGA_Q4K_Q6K_WGSL.replace("__NFF__", &n_ff.to_string());
        run(&self.ctx, &src, "ffn_mega_q4k_q6k",
            &[x.buf.as_ref(), gu.codes.as_ref(), gu.aux.as_ref(), dn.codes.as_ref(), dn.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, n_ff as u32, n_embd as u32, 0])],
            (rows as u32, 1, 1));
        Tensor::from_parts(&self.ctx, out, vec![rows, n_embd])
    }
    fn matmul_qshard(&self, w: &QShard) -> Tensor {
        match w {
            // Opt-in NVIDIA tensor-core prefill (`FERRIC_COOP16`): a multi-row (prefill) Q2_0 matmul on
            // (matmul_q2_0 itself carries the opt-in coop16 prefill fast-path — see there.)
            QShard::Iq2Xxs(w) => self.matmul_iq2_xxs(w),
            QShard::Iq3Xxs(w) => self.matmul_iq3_xxs(w),
            QShard::Stq1_0(w) => self.matmul_stq1_0(w),
            QShard::Q2_0(w) => self.matmul_q2_0(w),
            QShard::Q4_0(w) => self.matmul_q4_0(w),
            QShard::Q4_1(w) => self.matmul_q4_1(w),
            QShard::Q5_0(w) => self.matmul_q5_0(w),
            QShard::Q5_1(w) => self.matmul_q5_1(w),
            QShard::Q4_K(w) => self.matmul_q4_k(w),
            QShard::Q5_K(w) => self.matmul_q5_k(w),
            QShard::Q2_K(w) => self.matmul_q2_k(w),
            QShard::Q3_K(w) => self.matmul_q3_k(w),
            QShard::Q6_K(w) => self.matmul_q6_k(w),
            QShard::Q8_0(w) => self.matmul_q8_0(w),
            QShard::Iq4Xs(w) => self.matmul_iq4_xs(w),
            QShard::Iq4Nl(w) => self.matmul_iq4_nl(w),
            QShard::Mxfp4(w) => self.matmul_mxfp4(w),
            QShard::Dense(w) => self.matmul(&w.wt),
        }
    }
}

/// **Q5_K** weights held packed on the GPU — llama.cpp's 5-bit K-quant (`Q5_K_M` is a common
/// higher-quality choice). Same super-block as Q4_K plus a 32-byte `qh` array giving each quant a 5th
/// (high) bit: value = `d·scaleₛ·(nibble + 16·qh_bit) − dmin·minₛ`. codes = qs|qh (40 u32/block);
/// aux = d/dmin + 12 scale bytes (4 u32/block), identical to Q4_K.
pub struct Q5_KWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 40 u32/block: 32 words qs, then 8 words qh
    aux: Arc<wgpu::Buffer>,   // 4 u32/block: d|dmin, 12 scale bytes
    pub rows: usize,
    pub cols: usize,
}

impl Q5_KWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q5_KWeights {
        assert_eq!(cols % 256, 0, "Q5_K cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 176, "unexpected Q5_K byte length");
        let nblk = rows * (cols / 256);
        let mut codes: Vec<u32> = vec![0; nblk * 40];
        let mut aux: Vec<u32> = vec![0; nblk * 4];
        let word = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        for b in 0..nblk {
            let src = &bytes[b * 176..b * 176 + 176]; // d,dmin,scales[12],qh[32],qs[128]
            aux[b * 4] = u16::from_le_bytes([src[0], src[1]]) as u32 | ((u16::from_le_bytes([src[2], src[3]]) as u32) << 16);
            for w in 0..3 { aux[b * 4 + 1 + w] = word(src, 4 + w * 4); }        // 12 scale bytes
            for w in 0..32 { codes[b * 40 + w] = word(src, 48 + w * 4); }        // qs (128 bytes)
            for w in 0..8 { codes[b * 40 + 32 + w] = word(src, 16 + w * 4); }    // qh (32 bytes)
        }
        let mk = |label, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Q5_KWeights { ctx: ctx.clone(), codes: mk("q5k.codes", &codes), aux: mk("q5k.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 176 }
}

impl Tensor {
    /// y = x·Wᵀ where W is a packed **Q5_K** [out, in] weight, dequantized per-super-block in-kernel.
    pub fn matmul_q5_k(&self, w: &Q5_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q5_K_SPLITK_WGSL, "matmul_q5_k_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q5_K_FLAT_WGSL, "matmul_q5_k_flat")
        };
        if rows >= 8 && w.rows % 8 == 0 && self.ctx.coop_shared_ok() && std::env::var("FERRIC_COOP").is_ok() {
            return self.matmul_q5_k_coop(w);
        }
        let src = wgsl.replace("__HELPERS__", Q4_K_HELPERS).replace("__INNER__", Q5_K_INNER);
        let src = if use_subgroup(&self.ctx) { sg_reduce(&src) } else { src };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// Cooperative-matrix Q5_K prefill matmul — Q4_K plus the 5th (qh) bit. Completes Q5_K_M models.
    pub fn matmul_q5_k_coop(&self, w: &Q5_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0, "matmul_q5_k_coop needs N a multiple of 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        let src = MATMUL_Q5_K_COOP_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
        run(&self.ctx, &src, "matmul_q5_k_coop",
            &[xp.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, (inn / 256) as u32])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
}

const MATMUL_Q5_K_COOP_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:      array<f32>;
@group(0) @binding(1) var<storage,read>       codes:  array<u32>;
@group(0) @binding(2) var<storage,read>       aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write> c:      array<f32>;
@group(0) @binding(4) var<uniform>            dims:   vec4<u32>;   // M, K, N, nblk(=K/256)
var<workgroup> bs: array<f32, 64>;
__HELPERS__
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z; let nblk = dims.w;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = t + e * 32u; let nl = i / 8u; let kl = i % 8u;
            let n = n0 + nl; let k = k0 + kl;
            let gblk = n * nblk + (k / 256u); let v = k % 256u;
            let s = v / 32u; let l = v % 32u; let hi = s & 1u;
            let ab = gblk * 4u; let cb = gblk * 40u;
            let dd = unpack2x16float(aux[ab]); let sm = scmin(ab, s);
            let ds = dd.x * f32(sm.x); let mm = dd.y * f32(sm.y);
            let comp = l & 3u; let wl = l >> 2u;
            let qsw = codes[cb + 8u * (s >> 1u) + wl];
            let nib = (qsw >> (8u * comp + select(0u, 4u, hi == 1u))) & 0xFu;
            let qhw = codes[cb + 32u + wl];
            let bit = (qhw >> (8u * comp + s)) & 1u;
            bs[kl * 8u + nl] = ds * f32(nib + bit * 16u) - mm;
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u);
        acc = coopMultiplyAdd(ma, mb, acc);
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

/// **Q2_K packed weight** — 2 bits/weight, 84 bytes per 256 (2.625 bpw), the smallest K-quant tier.
///
/// `codes` is the 64 `qs` bytes as 16 words; `aux` is `[d|dmin, 16 scale bytes]` as 5 words. Both
/// super-block scales ride in ONE word so a single `unpack2x16float` yields the pair, and the 16
/// sub-block bytes need no rearrangement — each already packs its 4-bit scale and 4-bit min.
pub struct Q2_KWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 16 u32/block: qs
    aux: Arc<wgpu::Buffer>,   // 5 u32/block: [d|dmin, 16 scale/min bytes]
    pub rows: usize,
    pub cols: usize,
}

impl Q2_KWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q2_KWeights {
        assert_eq!(cols % 256, 0, "Q2_K cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 84, "unexpected Q2_K byte length");
        let nblk = rows * (cols / 256);
        let mut codes: Vec<u32> = vec![0; nblk * 16];
        let mut aux: Vec<u32> = vec![0; nblk * 5];
        let word = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        for b in 0..nblk {
            let src = &bytes[b * 84..b * 84 + 84];
            for w in 0..16 { codes[b * 16 + w] = word(src, 16 + w * 4); }
            // d in the low half, dmin in the high half — one unpack2x16float in-kernel.
            // ⚠ dmin is SIGNED and Ferric's quantizer emits negative ones; f16 bit-patterns pass
            // through unchanged here, and `unpack2x16float` decodes the sign for free.
            let d = u16::from_le_bytes([src[80], src[81]]) as u32;
            let dmin = u16::from_le_bytes([src[82], src[83]]) as u32;
            aux[b * 5] = d | (dmin << 16);
            for w in 0..4 { aux[b * 5 + 1 + w] = word(src, w * 4); }
        }
        let mk = |label, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Q2_KWeights { ctx: ctx.clone(), codes: mk("q2k.codes", &codes), aux: mk("q2k.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 84 }
}

/// **Q3_K packed weight** — 3 bits/weight, 110 bytes per 256 (3.4375 bpw).
///
/// ⭐ The sixteen 6-bit scales are UNSHUFFLED ON LOAD, not in the kernel. On disk they are woven
/// through 12 bytes as low-nibbles plus a plane of high-2-bit pairs; every matmul would otherwise
/// redo that `aux`/`kmask` dance per super-block per output row. Done once here, `aux` carries them
/// as plain bytes and the kernel does one subtract.
pub struct Q3_KWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 24 u32/block: 8 words hmask, then 16 words qs
    aux: Arc<wgpu::Buffer>,   // 5 u32/block: [d, 16 already-unshuffled 6-bit scale bytes]
    pub rows: usize,
    pub cols: usize,
}

impl Q3_KWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q3_KWeights {
        assert_eq!(cols % 256, 0, "Q3_K cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 110, "unexpected Q3_K byte length");
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let nblk = rows * (cols / 256);
        let mut codes: Vec<u32> = vec![0; nblk * 24];
        let mut aux: Vec<u32> = vec![0; nblk * 5];
        let word = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        for b in 0..nblk {
            let src = &bytes[b * 110..b * 110 + 110];
            for w in 0..8 { codes[b * 24 + w] = word(src, w * 4); }            // hmask (32 B)
            for w in 0..16 { codes[b * 24 + 8 + w] = word(src, 32 + w * 4); }  // qs (64 B)
            aux[b * 5] = u16::from_le_bytes([src[108], src[109]]) as u32;      // d
            let mut a = [0u32; 4];
            for k in 0..3 { a[k] = word(src, 96 + k * 4); }
            let tmp = a[2];
            a[2] = ((a[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            a[3] = ((a[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            a[0] = (a[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
            a[1] = (a[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            aux[b * 5 + 1..b * 5 + 5].copy_from_slice(&a);
        }
        let mk = |label, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Q3_KWeights { ctx: ctx.clone(), codes: mk("q3k.codes", &codes), aux: mk("q3k.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 110 }
}

impl Tensor {
    /// y = x·Wᵀ for a packed **Q2_K** [out, in] weight, dequantized per-super-block in-kernel.
    pub fn matmul_q2_k(&self, w: &Q2_KWeights) -> Tensor {
        self.matmul_k_packed(&w.codes, &w.aux, w.rows, w.cols, Q2_K_HELPERS, Q2_K_BODY, "q2_k")
    }
    /// y = x·Wᵀ for a packed **Q3_K** [out, in] weight, dequantized per-super-block in-kernel.
    pub fn matmul_q3_k(&self, w: &Q3_KWeights) -> Tensor {
        self.matmul_k_packed(&w.codes, &w.aux, w.rows, w.cols, Q3_K_HELPERS, Q3_K_BODY, "q3_k")
    }

    /// The shared driver for a 256-super-block packed matmul: pick flat or split-k by shape, splice
    /// the format's helpers and body into the shell, dispatch.
    ///
    /// The two shells bind exactly `(x, codes, aux, out, info)` and know nothing about the format —
    /// only the spliced body does — so a new K-quant needs a body, not a kernel. They carry Q6_K's
    /// name because it was the first to use them.
    fn matmul_k_packed(&self, codes: &Arc<wgpu::Buffer>, aux: &Arc<wgpu::Buffer>, o_dim: usize,
                       cols: usize, helpers: &str, body: &str, tag: &str) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, cols, "inner dim mismatch: x[..,{inn}] vs W[..,{cols}]");
        let out = empty(&self.ctx, rows * o_dim);
        let n = rows * o_dim;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, o_dim) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_K_SPLITK_WGSL, format!("matmul_{tag}_splitk"))
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_K_FLAT_WGSL, format!("matmul_{tag}_flat"))
        };
        let src = wgsl.replace("__HELPERS__", helpers).replace("__BODY__", body);
        let src = if use_subgroup(&self.ctx) { sg_reduce(&src) } else { src };
        run(&self.ctx, &src, &label,
            &[x.buf.as_ref(), codes.as_ref(), aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, o_dim as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, o_dim])
    }
}

/// **Q6_K** weights held packed on the GPU — llama.cpp's 6-bit K-quant. `Q4_K_M`, the default, stores
/// its embedding/output and some `ffn_down` tensors as Q6_K, so a real Q4_K_M model can't run without
/// it. 210-byte super-block / 256 values: `ql[128]` (low 4 bits), `qh[64]` (high 2 bits),
/// `scales[16]` (int8), `d` (f16); value = `d·scale·(q − 32)`. codes = ql|qh (48 u32/block); aux =
/// d + 16 scale bytes (5 u32/block), keeping within the 4-storage-buffer baseline.
pub struct Q6_KWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 48 u32/block: 32 words ql, then 16 words qh
    aux: Arc<wgpu::Buffer>,   // 5 u32/block: [d|_, 16 scale bytes]
    pub rows: usize,
    pub cols: usize,          // multiple of 256
}

impl Q6_KWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q6_KWeights {
        assert_eq!(cols % 256, 0, "Q6_K cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 210, "unexpected Q6_K byte length");
        let nblk = rows * (cols / 256);
        let mut codes: Vec<u32> = vec![0; nblk * 48];
        let mut aux: Vec<u32> = vec![0; nblk * 5];
        let word = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        for b in 0..nblk {
            let src = &bytes[b * 210..b * 210 + 210];
            for w in 0..32 { codes[b * 48 + w] = word(src, w * 4); }          // ql (128 bytes)
            for w in 0..16 { codes[b * 48 + 32 + w] = word(src, 128 + w * 4); } // qh (64 bytes)
            aux[b * 5] = u16::from_le_bytes([src[208], src[209]]) as u32;       // d
            for w in 0..4 { aux[b * 5 + 1 + w] = word(src, 192 + w * 4); }      // 16 scale bytes
        }
        let mk = |label, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Q6_KWeights { ctx: ctx.clone(), codes: mk("q6k.codes", &codes), aux: mk("q6k.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 210 }

    /// Overwrite `n` consecutive rows in place — the Q6_K twin of [`Q4_KWeights::write_rows`].
    ///
    /// A Q4_K_M MoE stores gate|up as Q4_K and **down as Q6_K**, so streaming an expert needs both.
    /// Adding only the Q4_K half would have made the gate|up slab swappable and left the down slab
    /// pinned to whatever it was built with — the two halves of one expert disagreeing, which
    /// produces finite, plausible, wrong output rather than an error.
    ///
    /// Same layout argument: blocks are row-major, so a row range is contiguous in both buffers.
    /// 210 bytes per block — `ql` 128, `qh` 64, 16 int8 scales, then the f16 `d`.
    pub fn write_rows(&self, row0: usize, bytes: &[u8], n_rows: usize) -> Result<(), String> {
        let bpr = self.cols / 256;
        if row0 + n_rows > self.rows {
            return Err(format!("rows {row0}..{} exceed the weight's {} rows", row0 + n_rows, self.rows));
        }
        if bytes.len() != n_rows * bpr * 210 {
            return Err(format!("{} bytes for {n_rows} rows x {bpr} blocks (need {})",
                               bytes.len(), n_rows * bpr * 210));
        }
        let nblk = n_rows * bpr;
        let mut codes: Vec<u32> = vec![0; nblk * 48];
        let mut aux: Vec<u32> = vec![0; nblk * 5];
        let word = |src: &[u8], o: usize| u32::from_le_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
        for b in 0..nblk {
            let src = &bytes[b * 210..b * 210 + 210];
            for w in 0..32 { codes[b * 48 + w] = word(src, w * 4); }            // ql
            for w in 0..16 { codes[b * 48 + 32 + w] = word(src, 128 + w * 4); } // qh
            aux[b * 5] = u16::from_le_bytes([src[208], src[209]]) as u32;       // d
            for w in 0..4 { aux[b * 5 + 1 + w] = word(src, 192 + w * 4); }      // 16 scale bytes
        }
        let blk0 = row0 * bpr;
        self.ctx.queue.write_buffer(&self.codes, (blk0 * 48 * 4) as u64, bytemuck::cast_slice(&codes));
        self.ctx.queue.write_buffer(&self.aux, (blk0 * 5 * 4) as u64, bytemuck::cast_slice(&aux));
        Ok(())
    }

}

impl Tensor {
    /// y = x·Wᵀ where W is a packed **Q6_K** [out, in] weight, dequantized per-super-block in-kernel.
    pub fn matmul_q6_k(&self, w: &Q6_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q6_K_SPLITK_WGSL, "matmul_q6_k_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q6_K_FLAT_WGSL, "matmul_q6_k_flat")
        };
        if rows >= 8 && w.rows % 8 == 0 && self.ctx.coop_shared_ok() && std::env::var("FERRIC_COOP").is_ok() {
            return self.matmul_q6_k_coop(w);
        }
        let src = wgsl.replace("__HELPERS__", Q6_K_HELPERS).replace("__BODY__", Q6_K_BODY);
        let src = if use_subgroup(&self.ctx) { sg_reduce(&src) } else { src };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// Cooperative-matrix Q6_K prefill matmul — used by every Q4_K_M / Q5_K_M model's embed/output
    /// tensors, so it lifts those models' prefill further. Reassembles the 6-bit quant (4 low bits
    /// from ql, 2 high from qh) with the int8 super-block scale, dequant tile → shared → matrix unit.
    pub fn matmul_q6_k_coop(&self, w: &Q6_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0, "matmul_q6_k_coop needs N (out) a multiple of 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        let src = MATMUL_Q6_K_COOP_WGSL.replace("__HELPERS__", Q6_K_HELPERS);
        run(&self.ctx, &src, "matmul_q6_k_coop",
            &[xp.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, (inn / 256) as u32])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
}

const MATMUL_Q6_K_COOP_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:      array<f32>;
@group(0) @binding(1) var<storage,read>       codes:  array<u32>;
@group(0) @binding(2) var<storage,read>       aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write> c:      array<f32>;
@group(0) @binding(4) var<uniform>            dims:   vec4<u32>;   // M, K, N, nblk(=K/256)
var<workgroup> bs: array<f32, 64>;
__HELPERS__
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z; let nblk = dims.w;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = t + e * 32u; let nl = i / 8u; let kl = i % 8u;
            let n = n0 + nl; let k = k0 + kl;
            let gblk = n * nblk + (k / 256u); let v = k % 256u;
            let cb = gblk * 48u; let ab = gblk * 5u;
            let d = unpack2x16float(aux[ab]).x;
            let hf = v / 128u; let within = v % 128u; let g = within / 32u; let l = within % 32u;
            let is = l >> 4u; let qlo = 64u * hf; let qho = 32u * hf; let sco = 8u * hf;
            let sc = scb(ab, sco + is + 2u * g);
            let h = qhb(cb, qho + l);
            let qlbyte = qlb(cb, qlo + l + (g & 1u) * 32u);
            let nib = select(qlbyte & 0xFu, qlbyte >> 4u, g >= 2u);
            let q = i32(nib | (((h >> (2u * g)) & 3u) << 4u)) - 32;
            bs[kl * 8u + nl] = d * sc * f32(q);
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u);
        acc = coopMultiplyAdd(ma, mb, acc);
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

/// **Q8_0** weights held packed on the GPU — llama.cpp's 8-bit format (blocks of 32: `f16 scale` +
/// 32 int8; value = int8·scale). Common for high-quality quants and for the embedding/output tensors
/// even inside mixed-precision models. Native packed matmul, dequant in-kernel.
pub struct Q8_0Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>,  // 8 u32 per block (32 int8)
    scales: Arc<wgpu::Buffer>, // f16 per block, two packed per u32
    pub rows: usize,
    pub cols: usize,           // multiple of 32
}

impl Q8_0Weights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q8_0Weights {
        assert_eq!(cols % 32, 0, "Q8_0 cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 34, "unexpected Q8_0 byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 8];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        for b in 0..nblk {
            let src = &bytes[b * 34..b * 34 + 34];
            scales[b / 2] |= (u16::from_le_bytes([src[0], src[1]]) as u32) << (16 * (b % 2));
            for w in 0..8 { codes[b * 8 + w] = u32::from_le_bytes([src[2 + w * 4], src[3 + w * 4], src[4 + w * 4], src[5 + w * 4]]); }
        }
        let mk = |label, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Q8_0Weights { ctx: ctx.clone(), codes: mk("q8_0.codes", &codes), scales: mk("q8_0.scales", &scales), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 34 }
    /// Overwrite `n` consecutive rows in place. See [`Q4_KWeights::write_rows`].
    ///
    /// ⚠ **Q8_0 PACKS TWO BLOCKS' SCALES INTO ONE `u32`** (`scales[b/2] |= d << (16 * (b%2))`), and
    /// `write_buffer` overwrites rather than read-modify-writes. So a block range that starts or
    /// ends mid-word would clobber the neighbouring block's scale — a block belonging to a DIFFERENT
    /// expert, which then dequantises against someone else's `d`. Refused rather than risked.
    ///
    /// In practice expert-major slabs are even-aligned (`blocks_per_row = cols/32` is even for any
    /// `cols >= 64`), so this refuses nothing real — but "in practice" is not a guarantee, and the
    /// failure it guards is silent.
    pub fn write_rows(&self, row0: usize, bytes: &[u8], n_rows: usize) -> Result<(), String> {
        let bpr = self.cols / 32;
        if row0 + n_rows > self.rows {
            return Err(format!("rows {row0}..{} exceed the weight's {} rows", row0 + n_rows, self.rows));
        }
        if bytes.len() != n_rows * bpr * 34 {
            return Err(format!("{} bytes for {n_rows} rows x {bpr} blocks (need {})",
                               bytes.len(), n_rows * bpr * 34));
        }
        let (blk0, nblk) = (row0 * bpr, n_rows * bpr);
        if blk0 % 2 != 0 || nblk % 2 != 0 {
            return Err(format!("Q8_0 shares one scale word between blocks {blk0} and {}; writing an \
                                odd-aligned range ({blk0}, {nblk} blocks) would clobber a neighbouring \
                                expert's scale", blk0 + 1));
        }
        let mut codes: Vec<u32> = vec![0; nblk * 8];
        let mut scales: Vec<u32> = vec![0; nblk / 2];
        for b in 0..nblk {
            let src = &bytes[b * 34..b * 34 + 34];
            scales[b / 2] |= (u16::from_le_bytes([src[0], src[1]]) as u32) << (16 * (b % 2));
            for w in 0..8 { codes[b * 8 + w] = u32::from_le_bytes([src[2 + w * 4], src[3 + w * 4], src[4 + w * 4], src[5 + w * 4]]); }
        }
        self.ctx.queue.write_buffer(&self.codes, (blk0 * 8 * 4) as u64, bytemuck::cast_slice(&codes));
        self.ctx.queue.write_buffer(&self.scales, (blk0 / 2 * 4) as u64, bytemuck::cast_slice(&scales));
        Ok(())
    }

}

impl Tensor {
    /// y = x·Wᵀ where W is a packed **Q8_0** [out, in] weight, dequantized per-block in-kernel.
    pub fn matmul_q8_0(&self, w: &Q8_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q8_0_SPLITK_WGSL, "matmul_q8_0_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q8_0_FLAT_WGSL, "matmul_q8_0_flat")
        };
        if rows >= 8 && w.rows % 8 == 0 && self.ctx.coop_shared_ok() && std::env::var("FERRIC_COOP").is_ok() {
            return self.matmul_q8_0_coop(w);
        }
        let src = if use_subgroup(&self.ctx) { sg_reduce(wgsl) } else { wgsl.to_string() };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// Cooperative-matrix Q8_0 prefill matmul — 8-bit (int8·scale), the simplest dequant.
    pub fn matmul_q8_0_coop(&self, w: &Q8_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0, "matmul_q8_0_coop needs N a multiple of 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        run(&self.ctx, MATMUL_Q8_0_COOP_WGSL, "matmul_q8_0_coop",
            &[xp.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, (inn / 32) as u32])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }
}

const MATMUL_Q8_0_COOP_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:      array<f32>;
@group(0) @binding(1) var<storage,read>       codes:  array<u32>;  // Q8_0 int8, W [N,K]
@group(0) @binding(2) var<storage,read>       scales: array<u32>;  // f16/block
@group(0) @binding(3) var<storage,read_write> c:      array<f32>;
@group(0) @binding(4) var<uniform>            dims:   vec4<u32>;   // M, K, N, nblk(=K/32)
var<workgroup> bs: array<f32, 64>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z; let nblk = dims.w;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = t + e * 32u; let nl = i / 8u; let kl = i % 8u;
            let n = n0 + nl; let k = k0 + kl;
            let gblk = n * nblk + (k / 32u); let j = k % 32u;
            let sw = unpack2x16float(scales[gblk >> 1u]);
            let d = select(sw.y, sw.x, (gblk & 1u) == 0u);
            let word = codes[gblk * 8u + (j >> 2u)];
            let byte = (word >> (8u * (j & 3u))) & 0xffu;
            bs[kl * 8u + nl] = f32(i32(byte << 24u) >> 24u) * d;
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u);
        acc = coopMultiplyAdd(ma, mb, acc);
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

/// **Q4_K** weights held packed on the GPU — the *default* llama.cpp quant (`Q4_K_M`), so the single
/// most common format on Hugging Face. A 144-byte super-block holds 256 values: `f16 d`, `f16 dmin`,
/// 12 bytes of 8 six-bit (scale, min) pairs, and 128 bytes of 4-bit quants; value =
/// `d·scaleₛ·q − dmin·minₛ` for its sub-block s. Native packed matmul (dequant in-kernel) instead of
/// dequant-to-f32. To stay within WebGPU's 4-storage-buffer baseline, d/dmin + the 12 scale bytes are
/// packed together into one `aux` buffer (4 u32/block); the 128 quant bytes are `codes` (32 u32/block).
pub struct Q4_KWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 32 u32 per block (128 quant bytes)
    aux: Arc<wgpu::Buffer>,   // 4 u32 per block: [d|dmin<<16, scale bytes 0..4, 4..8, 8..12]
    // Transposed (output-minor) copies for the coalesced GEMV experiment (FERRIC_Q4K_TRANS): the same
    // words/aux reordered so consecutive output-threads read consecutive memory. Built only when the
    // env flag is set (else empty), since it doubles weight memory. Same math → bit-identical.
    codes_t: Option<Arc<wgpu::Buffer>>, // [block][word][output]
    aux_t: Option<Arc<wgpu::Buffer>>,   // [block][k][output]
    pub rows: usize,
    pub cols: usize,          // multiple of 256
}

impl Q4_KWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Q4_KWeights {
        assert_eq!(cols % 256, 0, "Q4_K cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 144, "unexpected Q4_K byte length");
        let nblk = rows * (cols / 256);
        let mut codes: Vec<u32> = vec![0; nblk * 32];
        let mut aux: Vec<u32> = vec![0; nblk * 4];
        for b in 0..nblk {
            let src = &bytes[b * 144..b * 144 + 144];
            // aux[0] = d | dmin<<16 (both already f16 bit patterns); aux[1..4] = the 12 scale bytes.
            aux[b * 4] = u16::from_le_bytes([src[0], src[1]]) as u32 | ((u16::from_le_bytes([src[2], src[3]]) as u32) << 16);
            for w in 0..3 { aux[b * 4 + 1 + w] = u32::from_le_bytes([src[4 + w * 4], src[5 + w * 4], src[6 + w * 4], src[7 + w * 4]]); }
            for w in 0..32 { codes[b * 32 + w] = u32::from_le_bytes([src[16 + w * 4], src[17 + w * 4], src[18 + w * 4], src[19 + w * 4]]); }
        }
        let mk = |label, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        // Optional transposed (coalesced-GEMV) copies: codes_t[(j*32+w)*rows + o], aux_t[(j*4+k)*rows + o].
        let (codes_t, aux_t) = if std::env::var("FERRIC_Q4K_TRANS").is_ok() {
            let nbpr = cols / 256;
            let mut ct = vec![0u32; nblk * 32];
            let mut at = vec![0u32; nblk * 4];
            for o in 0..rows {
                for j in 0..nbpr {
                    let bi = o * nbpr + j;
                    for k in 0..4 { at[(j * 4 + k) * rows + o] = aux[bi * 4 + k]; }
                    for w in 0..32 { ct[(j * 32 + w) * rows + o] = codes[bi * 32 + w]; }
                }
            }
            (Some(mk("q4k.codes_t", &ct)), Some(mk("q4k.aux_t", &at)))
        } else { (None, None) };
        Q4_KWeights { ctx: ctx.clone(), codes: mk("q4k.codes", &codes), aux: mk("q4k.aux", &aux), codes_t, aux_t, rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 144 }

    /// **Overwrite `n` consecutive rows in place** — the primitive expert residency needs and the
    /// tensor layer did not have. Nothing in this file wrote into a built weight before now
    /// (`write_buffer` count was zero), which is why no runtime can stream an expert: there was no
    /// way to put a fetched one anywhere.
    ///
    /// Blocks are row-major (`b = row * cols/256 + j`), so a row RANGE is a contiguous byte range in
    /// both `codes` and `aux`, and the write is two `write_buffer` calls with no repacking of
    /// neighbours. The buffers already carry `COPY_DST`.
    ///
    /// ⚠ **Refuses when the transposed copies exist.** `FERRIC_Q4K_TRANS` builds `codes_t`/`aux_t`
    /// as a reordered duplicate; writing only the row-major pair would leave them stale and every
    /// kernel that reads them would return the PREVIOUS expert — finite, plausible, and wrong.
    /// Refusing beats silently updating one of two representations.
    ///
    /// ⚠ Cost is not assumed. `lib.rs:1488` measured `write_buffer` LOSING to fresh allocation for
    /// a few dozen bytes, because wgpu's staging belt has a fixed overhead. An expert here is ~2 MB,
    /// where that overhead amortises — but this is stated as a reason not to extrapolate the old
    /// measurement, not as a claim about the new one. Measure it on a working streaming path.
    pub fn write_rows(&self, row0: usize, bytes: &[u8], n_rows: usize) -> Result<(), String> {
        if self.codes_t.is_some() || self.aux_t.is_some() {
            return Err("write_rows on a weight with FERRIC_Q4K_TRANS transposed copies: they would \
                        go stale and kernels reading them would return the previous expert".into());
        }
        let bpr = self.cols / 256;
        if row0 + n_rows > self.rows {
            return Err(format!("rows {row0}..{} exceed the weight's {} rows", row0 + n_rows, self.rows));
        }
        if bytes.len() != n_rows * bpr * 144 {
            return Err(format!("{} bytes for {n_rows} rows x {bpr} blocks (need {})",
                               bytes.len(), n_rows * bpr * 144));
        }
        let nblk = n_rows * bpr;
        let mut codes: Vec<u32> = vec![0; nblk * 32];
        let mut aux: Vec<u32> = vec![0; nblk * 4];
        for b in 0..nblk {
            let src = &bytes[b * 144..b * 144 + 144];
            aux[b * 4] = u16::from_le_bytes([src[0], src[1]]) as u32
                       | ((u16::from_le_bytes([src[2], src[3]]) as u32) << 16);
            for w in 0..3 { aux[b * 4 + 1 + w] = u32::from_le_bytes([src[4 + w * 4], src[5 + w * 4], src[6 + w * 4], src[7 + w * 4]]); }
            for w in 0..32 { codes[b * 32 + w] = u32::from_le_bytes([src[16 + w * 4], src[17 + w * 4], src[18 + w * 4], src[19 + w * 4]]); }
        }
        let blk0 = row0 * bpr;
        self.ctx.queue.write_buffer(&self.codes, (blk0 * 32 * 4) as u64, bytemuck::cast_slice(&codes));
        self.ctx.queue.write_buffer(&self.aux, (blk0 * 4 * 4) as u64, bytemuck::cast_slice(&aux));
        Ok(())
    }

}

impl Tensor {
    /// y = x·Wᵀ where W is a packed **Q4_K** [out, in] weight, dequantized per-super-block in-kernel.
    pub fn matmul_q4_k(&self, w: &Q4_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q4_K_SPLITK_WGSL, "matmul_q4_k_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q4_K_FLAT_WGSL, "matmul_q4_k_flat")
        };
        // Coalesced-GEMV experiment: transposed weight layout so output-threads read contiguous memory.
        // Gated on a SEPARATE flag (measured to HURT the small qkv/o matmuls) so FERRIC_Q4K_TRANS alone
        // isolates the fused-swiglu transpose on the big gate_up.
        if let (Some(ct), Some(at)) = (&w.codes_t, &w.aux_t) {
          if std::env::var("FERRIC_Q4K_TRANS_M").is_ok() {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            let grid = ((gw as u32), wg.div_ceil(gw) as u32, 1u32);
            run(&self.ctx, MATMUL_Q4_K_TRANS_WGSL, "matmul_q4_k_trans",
                &[x.buf.as_ref(), ct.as_ref(), at.as_ref(), &out,
                  &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, (gw * 64) as u32])], grid);
            return Tensor::from_parts(&self.ctx, out, vec![rows, w.rows]);
          }
        }
        // K-split subgroup GEMV (opt-in, needs subgroups): one subgroup per output, lanes split the
        // blocks then subgroupAdd. Non-bit-identical → opt-in fast path.
        if std::env::var("FERRIC_SGGEMV").is_ok() && self.ctx.subgroups {
            let gw = n.min(65535);
            let grid = ((gw as u32), n.div_ceil(gw) as u32, 1u32);
            let src = MATMUL_Q4_K_SGGEMV_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
            run(&self.ctx, &src, "matmul_q4_k_sggemv",
                &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
                  &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, gw as u32])], grid);
            return Tensor::from_parts(&self.ctx, out, vec![rows, w.rows]);
        }
        // Prefill tensor-core fast-path (opt-in, Metal), same discipline as Q2_0.
        if rows >= 8 && w.rows % 8 == 0 && self.ctx.coop_shared_ok() && std::env::var("FERRIC_COOP").is_ok() {
            return self.matmul_q4_k_coop(w);
        }
        let src = wgsl.replace("__HELPERS__", Q4_K_HELPERS).replace("__INNER__", Q4_K_INNER);
        let src = if use_subgroup(&self.ctx) { sg_reduce(&src) } else { src };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// **Two-pass** cooperative-matrix Q2_0 matmul for NON-Metal (NVIDIA): dispatch 1 dequantizes the
    /// whole weight to a global f32 buffer, dispatch 2 runs the f32 coop GEMM `x·Wᵀ` on it. The coop
    /// load then reads a *pre-written* buffer (never written-then-read in one kernel), which is the
    /// pattern NVIDIA handles correctly. Costs 8× transient f32 for the weight — fine at prefill.
    pub fn matmul_q2_0_coop2pass(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0, "coop2pass needs N a multiple of 8");
        // dispatch 1: packed Q2_0 [N,K] → f32 **[K,N]** (transposed), so the plain row-major coop GEMM
        // computes x·[K,N] = x·Wᵀ. Column-major coop-load (the direct-Wᵀ approach) is mis-generated on
        // NVIDIA's SPIR-V; only row-major loads work there, so we transpose in the dequant instead.
        let (n, k) = (w.rows, inn);
        let wf = empty(&self.ctx, k * n);
        let (grid, rs) = groups2d(n * k);
        run(&self.ctx, DEQ_Q2_0_T_WGSL, "deq_q2_0_t", &[w.codes.as_ref(), w.scales.as_ref(), &wf,
            &u32buf(&self.ctx, &[(n * k) as u32, k as u32, (k / 128) as u32, n as u32, rs])], grid);
        let wf_t = Tensor::from_parts(&self.ctx, wf, vec![k, n]);
        // dispatch 2: the exact-f32 8×8 coop GEMM. **Metal-only** — on NVIDIA the 8×8-f32 coop shape is
        // not an enumerated config so it runs as zeros (see coop_gemm_ok / matmul_coop16). For NVIDIA
        // tensor-core Q2_0 prefill use `matmul_q2_0_coop16` instead (f16 inputs, the supported config).
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let full = xp.matmul_coop(&wf_t);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }

    /// **Metal-4 tensor-unit Q2_0 prefill** (via `FERRIC_METAL4`): dequantize the packed weight to
    /// f32 `[K,N]` with the existing transposed-dequant kernel, then plain `matmul` — which routes
    /// through the resident tensor units. The dequant target comes from the resident out-pool, so
    /// ONE transient buffer is recycled across a forward's layers instead of accumulating — the OOM
    /// that stopped the coop16 model-facing hook (~140 live buffers) cannot happen here, and the
    /// resident path's per-call poll lets wgpu actually reclaim what drops.
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    pub fn matmul_q2_0_metal4(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        let (n, k) = (w.rows, inn);
        let (wf, _fresh) = crate::metal4::pooled_out(&self.ctx, k * n);
        let (grid, rs) = groups2d(n * k);
        run(&self.ctx, DEQ_Q2_0_NT_WGSL, "deq_q2_0_nt", &[w.codes.as_ref(), w.scales.as_ref(), wf.as_ref(),
            &u32buf(&self.ctx, &[(n * k) as u32, k as u32, (k / 128) as u32, n as u32, rs])], grid);
        let wf_nk = Tensor::from_arc(&self.ctx, wf, &[n, k]);
        let _ = rows;
        x.matmul_bt(&wf_nk)
    }

    /// **NVIDIA tensor-core Q2_0 prefill**: dequant the packed weight to f32 `[K,N]` (transposed, so
    /// the row-major coop load computes x·Wᵀ), then `matmul_coop16` (f16 inputs, f32 accumulate) on the
    /// tensor cores. The dequant is O(weight) and the matmul O(M·weight), so it amortizes with M — the
    /// prefill win the 8×8-f32 path couldn't deliver on NVIDIA. Mixed precision ⇒ opt-in fast path.
    pub fn matmul_q2_0_coop16(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 16 == 0 && inn % 16 == 0, "coop16 needs N,K multiples of 16");
        let (n, k) = (w.rows, inn);
        let wf = empty(&self.ctx, k * n);
        let (grid, rs) = groups2d(n * k);
        run(&self.ctx, DEQ_Q2_0_T_WGSL, "deq_q2_0_t", &[w.codes.as_ref(), w.scales.as_ref(), &wf,
            &u32buf(&self.ctx, &[(n * k) as u32, k as u32, (k / 128) as u32, n as u32, rs])], grid);
        let wf_t = Tensor::from_parts(&self.ctx, wf, vec![k, n]);
        let mrows = rows.div_ceil(32) * 32; // pad to 32 so the 2×2 register-blocked coop16 path applies
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let full = xp.matmul_coop16(&wf_t);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }

    /// Cooperative-matrix Q4_K prefill matmul — dequant an 8×8 Q4_K tile (super-block scale/min +
    /// nibble) into shared memory, then feed the matrix unit. Brings tensor-core prompt processing to
    /// the *default* llama.cpp format. Same coop tiling as Q2_0; only the dequant differs.
    /// **Fused FFN gate/up + SwiGLU** for a Q4_K gate_up weight `[2·n_ff, in]`: each thread computes
    /// both the gate row `o` and the up row `o+n_ff` (dequant inline) and writes `silu(gate)·up`
    /// directly — the `[t, 2·n_ff]` intermediate is never materialized and the separate SwiGLU
    /// dispatch is gone. Output `[t, n_ff]`, bit-identical to `matmul_q4_k(w).swiglu(n_ff)`.
    pub fn matmul_q4_k_swiglu(&self, w: &Q4_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        assert_eq!(w.rows % 2, 0, "gate_up weight must have an even row count (gate|up)");
        let n_ff = w.rows / 2;
        let out = empty(&self.ctx, rows * n_ff);
        let n = rows * n_ff;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        // Fused ∧ coalesced (FERRIC_Q4K_TRANS): the transposed weight layout inside the fused swiglu —
        // the only path to a Vulkan win (fusion already beats a plain transposed matmul). Bit-identical.
        if let (Some(ct), Some(at)) = (&w.codes_t, &w.aux_t) {
            run(&self.ctx, MATMUL_Q4_K_SWIGLU_TRANS_WGSL, "matmul_q4_k_swiglu_trans",
                &[x.buf.as_ref(), ct.as_ref(), at.as_ref(), &out,
                  &unibuf(&self.ctx, &[rows as u32, n_ff as u32, inn as u32, gw as u32])], grid);
            return Tensor::from_parts(&self.ctx, out, vec![rows, n_ff]);
        }
        let src = MATMUL_Q4_K_SWIGLU_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
        run(&self.ctx, &src, "matmul_q4_k_swiglu",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, n_ff as u32, inn as u32, gw as u32])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, n_ff])
    }
    /// MoE router top-k, entirely on the GPU — kills the per-layer CPU readback sync. `self` is the
    /// router logits [T, n_expert] (one row per token — `h.matmul_bt(router)` order); per token:
    /// computes scores (softmax or sigmoid), selects the top-k by score(+bias) with the bias never
    /// entering the weights, renormalizes the selected scores to sum 1 (× `scale`), and writes a
    /// `[w_0..w_{k-1} | idx_0..idx_{k-1}]` row (indices stored as f32 — exact for n_expert ≤ 2^24)
    /// → [T, 2k] for the `*_id` expert kernels to consume without any CPU round-trip. One thread
    /// per token, scanning its own contiguous logits row.
    pub fn moe_topk(&self, bias: Option<&Tensor>, k: usize, sigmoid: bool, scale: f32) -> Tensor {
        self.moe_topk_ex(bias, k, sigmoid, scale, true)
    }

    /// [`Tensor::moe_topk`] with explicit control over renormalisation.
    ///
    /// `norm = true` divides the selected scores by their sum, so the routed weights sum to `scale`.
    /// `norm = false` uses the raw scores — the selected top-k probabilities out of a softmax over ALL
    /// experts, which sum to LESS than 1.
    ///
    /// This is not a tuning knob. llama.cpp reads it as `expert_weights_norm`, and it is **absent from
    /// DeepSeek-V2 checkpoints**, where it defaults to false. Renormalising anyway rescales every
    /// routed contribution by `1/Σp` — a token whose top-6 experts captured 40% of the mass gets its
    /// MoE output multiplied by 2.5. Nothing errors; the model is simply a different model.
    pub fn moe_topk_ex(&self, bias: Option<&Tensor>, k: usize, sigmoid: bool, scale: f32, norm: bool) -> Tensor {
        let x = self.contiguous();
        let (t, ne) = if x.shape.len() == 2 { (x.shape[0], x.shape[1]) } else { (1, x.numel()) };
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert!(ne <= 1024, "moe_topk: n_expert ≤ 1024 (single-thread scan)");
        let out = empty(&self.ctx, t * 2 * k);
        // bias is optional: bind a 1-element dummy when absent (has_bias=0 ignores it)
        let dummy;
        let bias_buf = match bias {
            Some(b) => b.contiguous().buf.clone(),
            None => { dummy = Tensor::from_vec(&self.ctx, &[0.0], &[1]); dummy.buf.clone() }
        };
        run(&self.ctx, MOE_TOPK_WGSL, "moe_topk",
            &[x.buf.as_ref(), bias_buf.as_ref(), &out,
              &unibuf(&self.ctx, &[ne as u32, k as u32, sigmoid as u32, bias.is_some() as u32, scale.to_bits(), t as u32, norm as u32, 0])],
            (t as u32, 1, 1));
        Tensor::from_parts(&self.ctx, out, vec![t, 2 * k])
    }

    /// Batched selected-expert gate|up + SwiGLU (mixture-of-experts decode). `self` is [T, in] hidden
    /// rows (each token routes independently); `w` packs ALL experts' fused gate|up weights,
    /// expert-major: rows = n_expert · 2·eff; `selw` is the [T, 2k] `moe_topk` output. One dispatch
    /// computes silu(gate)·up for every (token, selected expert) → [T, k, eff]. Replaces T·k separate
    /// matmul+swiglu dispatches — the MoE dispatch-count fix.
    pub fn matmul_q4_k_swiglu_id(&self, w: &Q4_KWeights, selw: &Tensor, k: usize, eff: usize) -> Tensor {
        let x = self.contiguous();
        let (t, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        assert!((1..=8).contains(&k), "k must be 1..=8");
        let sw = selw.contiguous();
        let out = empty(&self.ctx, t * k * eff);
        let n = t * k * eff;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        let src = MATMUL_Q4_K_SWIGLU_ID_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
        run(&self.ctx, &src, "matmul_q4_k_swiglu_id",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out, sw.buf.as_ref(),
              &unibuf(&self.ctx, &[k as u32, eff as u32, inn as u32, gw as u32, n as u32, 0, 0, 0])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, k, eff])
    }

    /// Batched selected-expert down projection (mixture-of-experts decode). `self` is [k, in] — row s
    /// is expert slot s's SwiGLU output; `w` packs ALL experts' down weights, expert-major
    /// (rows = n_expert · out_pe). One dispatch → [k, out_pe]; caller weight-sums the k rows.
    pub fn matmul_q6_k_id(&self, w: &Q6_KWeights, selw: &Tensor, out_pe: usize) -> Tensor {
        let x = self.contiguous();
        let (k, inn) = (x.shape[0], x.shape[1]);
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let sw = selw.contiguous();
        let out = empty(&self.ctx, k * out_pe);
        let n = k * out_pe;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        let src = MATMUL_Q6_K_ID_WGSL.replace("__HELPERS__", Q6_K_HELPERS).replace("__BODY__", Q6_K_BODY);
        run(&self.ctx, &src, "matmul_q6_k_id",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out, sw.buf.as_ref(),
              &unibuf(&self.ctx, &[k as u32, out_pe as u32, inn as u32, gw as u32])], grid);
        Tensor::from_parts(&self.ctx, out, vec![k, out_pe])
    }


    /// Batched selected-expert down projection WITH the weighted sum fused: `self` is [T, k, in]
    /// (row (t,s) = token t's expert-slot-s SwiGLU output; a 2D [k, in] means T=1); returns
    /// [T, out_pe] with row t = Σ_s w_{t,s} · down_{e_{t,s}}(x_{t,s}), w/idx read from the [T, 2k]
    /// `moe_topk` buffer. One dispatch for the whole routed-expert combine, all tokens.
    pub fn matmul_q6_k_id_wsum(&self, w: &Q6_KWeights, selw: &Tensor, out_pe: usize) -> Tensor {
        let x = self.contiguous();
        let (t, k, inn) = if x.shape.len() == 3 { (x.shape[0], x.shape[1], x.shape[2]) } else { (1, x.shape[0], x.shape[1]) };
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let sw = selw.contiguous();
        let out = empty(&self.ctx, t * out_pe);
        let n = t * out_pe;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        let src = MATMUL_Q6_K_ID_WSUM_WGSL.replace("__HELPERS__", Q6_K_HELPERS).replace("__BODY__", Q6_K_BODY);
        run(&self.ctx, &src, "matmul_q6_k_id_wsum",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out, sw.buf.as_ref(),
              &unibuf(&self.ctx, &[k as u32, out_pe as u32, inn as u32, gw as u32, n as u32, 0, 0, 0])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, out_pe])
    }
    /// `matmul_q8_0_id_wsum` for a **Q5_0** expert down slab.
    ///
    /// Needed because the expert quant type VARIES BY LAYER in a real mixed-precision checkpoint:
    /// Nemotron 3.5 Lightning Q4_K_M stores `ffn_down_exps` as Q8_0 in 11 blocks and Q5_0 in the other
    /// 13. Assuming one type per role loads the first few blocks and then fails on a byte-length
    /// assert, which is the good outcome; the bad one is a runtime that silently reinterprets bytes.
    ///
    /// `self` is [T, k, in]; returns [T, out_pe] with row t = Σ_s w_{t,s} · down_{e_{t,s}}(x_{t,s}).
    pub fn matmul_q5_0_id_wsum(&self, w: &Q5_0Weights, selw: &Tensor, out_pe: usize) -> Tensor {
        let x = self.contiguous();
        let (t, k, inn) = if x.shape.len() == 3 { (x.shape[0], x.shape[1], x.shape[2]) } else { (1, x.shape[0], x.shape[1]) };
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let sw = selw.contiguous();
        let n = t * out_pe;
        let out = empty(&self.ctx, n);
        let wg = n.div_ceil(64);
        let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        run(&self.ctx, MATMUL_Q5_0_ID_WSUM_WGSL, "matmul_q5_0_id_wsum",
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out, sw.buf.as_ref(),
              &u32buf(&self.ctx, &[k as u32, out_pe as u32, inn as u32, gw as u32, n as u32])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, out_pe])
    }

    /// **Batched selected-expert UP projection with ReLU² fused**, for a Q5_0 expert slab.
    ///
    /// Nemotron-H's MoE is `LLM_FFN_RELU_SQR`, not SwiGLU — which is *why* its checkpoints carry no
    /// `ffn_gate_exps` tensor at all. The existing `*_swiglu_id` kernels cannot serve it (they consume
    /// a fused gate|up slab and apply silu), and dequantising 128 experts x 1856 x 2688 to f32 would
    /// cost ~2.5 GB per layer across 23 layers, so the format needs its own indexed kernel.
    ///
    /// `self` is [T, in]; `selw` is the [T, 2k] buffer from `moe_topk` (weights, then expert ids);
    /// returns [T, k, eff] with `max(x,0)^2` already applied.
    pub fn matmul_q5_0_relu2_id(&self, w: &Q5_0Weights, selw: &Tensor, k: usize, eff: usize) -> Tensor {
        let x = self.contiguous();
        let (t, inn) = (x.shape[0], x.shape[1]);
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        assert_eq!(inn % 32, 0, "Q5_0 inner dim must be a multiple of 32");
        let sw = selw.contiguous();
        let n = t * k * eff;
        let out = empty(&self.ctx, n);
        let wg = n.div_ceil(64);
        let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        run(&self.ctx, MATMUL_Q5_0_RELU2_ID_WGSL, "matmul_q5_0_relu2_id",
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out, sw.buf.as_ref(),
              &u32buf(&self.ctx, &[k as u32, eff as u32, inn as u32, gw as u32, n as u32])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, k, eff])
    }

    /// `matmul_q4_k_swiglu_id` for a Q8_0 gate|up slab (the MTP draft block's expert format).
    pub fn matmul_q8_0_swiglu_id(&self, w: &Q8_0Weights, selw: &Tensor, k: usize, eff: usize) -> Tensor {
        let x = self.contiguous();
        let (t, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        assert!((1..=8).contains(&k), "k must be 1..=8");
        let sw = selw.contiguous();
        let out = empty(&self.ctx, t * k * eff);
        let n = t * k * eff;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        run(&self.ctx, MATMUL_Q8_0_SWIGLU_ID_WGSL, "matmul_q8_0_swiglu_id",
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out, sw.buf.as_ref(),
              &unibuf(&self.ctx, &[k as u32, eff as u32, inn as u32, gw as u32, n as u32, 0, 0, 0])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, k, eff])
    }

    /// `matmul_q6_k_id_wsum` for a Q8_0 down slab — companion to `matmul_q8_0_swiglu_id`.
    pub fn matmul_q8_0_id_wsum(&self, w: &Q8_0Weights, selw: &Tensor, out_pe: usize) -> Tensor {
        let x = self.contiguous();
        let (t, k, inn) = if x.shape.len() == 3 { (x.shape[0], x.shape[1], x.shape[2]) } else { (1, x.shape[0], x.shape[1]) };
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let sw = selw.contiguous();
        let out = empty(&self.ctx, t * out_pe);
        let n = t * out_pe;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        run(&self.ctx, MATMUL_Q8_0_ID_WSUM_WGSL, "matmul_q8_0_id_wsum",
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out, sw.buf.as_ref(),
              &unibuf(&self.ctx, &[k as u32, out_pe as u32, inn as u32, gw as u32, n as u32, 0, 0, 0])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, out_pe])
    }

    /// `matmul_q6_k_id_wsum` for a Q4_K down slab — same [T,k,in]→[T,out] weighted combine, Q4_K math.
    pub fn matmul_q4_k_id_wsum(&self, w: &Q4_KWeights, selw: &Tensor, out_pe: usize) -> Tensor {
        let x = self.contiguous();
        let (t, k, inn) = if x.shape.len() == 3 { (x.shape[0], x.shape[1], x.shape[2]) } else { (1, x.shape[0], x.shape[1]) };
        assert!((1..=8).contains(&k), "k must be 1..=8");
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let sw = selw.contiguous();
        let out = empty(&self.ctx, t * out_pe);
        let n = t * out_pe;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        let src = MATMUL_Q4_K_ID_WSUM_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
        run(&self.ctx, &src, "matmul_q4_k_id_wsum",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out, sw.buf.as_ref(),
              &unibuf(&self.ctx, &[k as u32, out_pe as u32, inn as u32, gw as u32, n as u32, 0, 0, 0])], grid);
        Tensor::from_parts(&self.ctx, out, vec![t, out_pe])
    }

    /// Fused gate/up + SwiGLU for a Q5_K gate_up weight (Q5_K_M FFNs). Same whole-block fusion as
    /// `matmul_q4_k_swiglu`, plus the 5th (qh) bit per quant.
    pub fn matmul_q5_k_swiglu(&self, w: &Q5_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        assert_eq!(w.rows % 2, 0, "gate_up weight must have an even row count (gate|up)");
        let n_ff = w.rows / 2;
        let out = empty(&self.ctx, rows * n_ff);
        let n = rows * n_ff;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        let src = MATMUL_Q5_K_SWIGLU_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
        run(&self.ctx, &src, "matmul_q5_k_swiglu",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, n_ff as u32, inn as u32, gw as u32])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, n_ff])
    }
    /// Fused gate/up + SwiGLU for a Q6_K gate_up weight (Q6_K FFNs). Same whole-block fusion; Q6_K
    /// x is a plain f32 array (not vec4-packed).
    pub fn matmul_q6_k_swiglu(&self, w: &Q6_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        assert_eq!(w.rows % 2, 0, "gate_up weight must have an even row count (gate|up)");
        let n_ff = w.rows / 2;
        let out = empty(&self.ctx, rows * n_ff);
        let n = rows * n_ff;
        let wg = n.div_ceil(64); let gw = wg.min(32768);
        let grid = (gw as u32, wg.div_ceil(gw) as u32, 1u32);
        let src = MATMUL_Q6_K_SWIGLU_WGSL.replace("__HELPERS__", Q6_K_HELPERS);
        run(&self.ctx, &src, "matmul_q6_k_swiglu",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, n_ff as u32, inn as u32, gw as u32])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, n_ff])
    }
    pub fn matmul_q4_k_coop(&self, w: &Q4_KWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0, "matmul_q4_k_coop needs N (out) a multiple of 8");
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        let nblk = (inn / 256) as u32;
        let src = MATMUL_Q4_K_COOP_WGSL.replace("__HELPERS__", Q4_K_HELPERS);
        run(&self.ctx, &src, "matmul_q4_k_coop",
            &[xp.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, nblk])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }

    /// y = x·Wᵀ where W is a packed **Q4_0** [out, in] weight, dequantized per-block inside the kernel.
    /// Same rows-aware flat/split-K selection as Q2_0. x [rows, in] → [rows, out].
    pub fn matmul_q4_0(&self, w: &Q4_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q4_0_SPLITK_WGSL, "matmul_q4_0_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q4_0_FLAT_WGSL, "matmul_q4_0_flat")
        };
        let src = if use_subgroup(&self.ctx) { sg_reduce(wgsl) } else { wgsl.to_string() };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// y = x·Wᵀ where W is a packed **Q4_1** [out, in] weight (affine `nibble·d + m`), dequantized
    /// per-block inside the kernel. Same flat/split-K selection and bindings as Q4_0.
    pub fn matmul_q4_1(&self, w: &Q4_1Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q4_1_SPLITK_WGSL, "matmul_q4_1_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q4_1_FLAT_WGSL, "matmul_q4_1_flat")
        };
        let src = if use_subgroup(&self.ctx) { sg_reduce(wgsl) } else { wgsl.to_string() };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// y = x·Wᵀ where W is a packed **Q5_0** [out, in] weight (5-bit symmetric, the 5th bit from a
    /// per-block `qh`), dequantized per-block inside the kernel. Same selection/bindings as Q4_0.
    pub fn matmul_q5_0(&self, w: &Q5_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q5_0_SPLITK_WGSL, "matmul_q5_0_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q5_0_FLAT_WGSL, "matmul_q5_0_flat")
        };
        let src = if use_subgroup(&self.ctx) { sg_reduce(wgsl) } else { wgsl.to_string() };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// y = x·Wᵀ where W is a packed **Q5_1** [out, in] weight (affine 5-bit, the 5th bit from a
    /// per-block `qh`), dequantized per-block inside the kernel. Same selection/bindings as Q4_0.
    pub fn matmul_q5_1(&self, w: &Q5_1Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl, label) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q5_1_SPLITK_WGSL, "matmul_q5_1_splitk")
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q5_1_FLAT_WGSL, "matmul_q5_1_flat")
        };
        let src = if use_subgroup(&self.ctx) { sg_reduce(wgsl) } else { wgsl.to_string() };
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// y = x·Wᵀ where W is PrismML Q2_0 ternary held PACKED on the GPU (dequantized per-block on the
    /// fly inside the kernel). x [rows, in] → [rows, out]. This is what makes a 27B ternary model fit.
    /// **STQ1_0 GEMV, weights never expanded.** One thread per output element.
    ///
    /// Each 42-byte block covers 256 weights as 64 groups of four, and a group's four lanes are
    /// **stride 16 apart inside a 64-weight chunk**, not adjacent — `x[c*64 + g%16 + p*16]`. That is
    /// the format, not an implementation choice, and reading them contiguously would load exactly
    /// the right 256 activations against exactly the right 256 weights in the wrong pairing.
    ///
    /// The codebook lives in `var<private>` rather than `const` because it is indexed by a runtime
    /// value, which naga will not accept on a module-scope `const`.
    /// **IQ2_XXS GEMV**, weights never expanded. 2.0625 bpw against 32 dense — 15.5x.
    pub fn matmul_iq2_xxs(&self, w: &Iq2XxsWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let (gw, gh) = grid2d(rows * w.rows);
        run(&self.ctx, &matmul_iq2_xxs_wgsl(), "matmul_iq2_xxs",
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), w.grid.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, (gw * 64) as u32])],
            (gw as u32, gh as u32, 1));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// **IQ3_XXS GEMV**, weights never expanded. 3.0625 bpw against 32 dense — 10.4x.
    pub fn matmul_iq3_xxs(&self, w: &Iq3XxsWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let (gw, gh) = grid2d(rows * w.rows);
        run(&self.ctx, &matmul_iq3_xxs_wgsl(), "matmul_iq3_xxs",
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), w.grid.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, (gw * 64) as u32])],
            (gw as u32, gh as u32, 1));
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    pub fn matmul_stq1_0(&self, w: &Stq1_0Weights) -> Tensor {
        let f = match std::env::var("FERRIC_STQ1_FORM").as_deref() {
            Ok("scalar") => Stq1Form::Scalar,
            Ok("vec4") => Stq1Form::Vec4,
            _ => Stq1Form::Vec4Table,
        };
        self.matmul_stq1_0_form(w, f)
    }

    /// The same matmul with the traversal order chosen explicitly, so the two forms can be compared
    /// **inside one process**. A cross-launch comparison on a contended laptop has inverted a
    /// conclusion in this tree before; an env var alone would have forced exactly that shape.
    pub fn matmul_stq1_0_form(&self, w: &Stq1_0Weights, form: Stq1Form) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let wg = n.div_ceil(64);
        let gw = wg.min(32768);
        let gh = wg.div_ceil(gw);
        // `FERRIC_STQ1_SCALAR` selects the original scalar-load form. It is kept reachable so the
        // vec4 rewrite can be A/B'd in one process rather than across two builds — a cross-launch
        // comparison on a contended laptop has inverted a conclusion in this tree before.
        let info = unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, (gw * 64) as u32]);
        match form {
            Stq1Form::Vec4Table => run(&self.ctx, MATMUL_STQ1_0_V4T_WGSL, "matmul_stq1_0_v4t",
                &[x.buf.as_ref(), w.codes.as_ref(), w.signs.as_ref(), w.scales.as_ref(),
                  w.codebook.as_ref(), &out, &info],
                (gw as u32, gh as u32, 1)),
            f => {
                let (src, label) = if matches!(f, Stq1Form::Scalar) { (MATMUL_STQ1_0_WGSL, "matmul_stq1_0_scalar") }
                                   else { (MATMUL_STQ1_0_V4_WGSL, "matmul_stq1_0_v4") };
                run(&self.ctx, src, label,
                    &[x.buf.as_ref(), w.codes.as_ref(), w.signs.as_ref(), w.scales.as_ref(), &out, &info],
                    (gw as u32, gh as u32, 1))
            }
        }
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    pub fn matmul_q2_0(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        // Metal-4 tensor-unit prefill (opt-in FERRIC_METAL4): dequant once, then a real GEMM on the
        // tensor units (~10 TFLOP/s resident) — dequant is O(weight), the GEMM O(M·weight), so it
        // amortizes with M. Decode (small rows) stays on the fused scalar kernel, which reads only
        // the packed bytes. Checked before coop so the faster unit wins when both are opted in.
        #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
        if rows >= 32 && crate::metal4::resident_ready(&self.ctx, 2 * rows * inn * w.rows) {
            return self.matmul_q2_0_metal4(w);
        }
        // Prefill tensor-core fast-path (opt-in, Metal): many tokens make this a real GEMM where the
        // matrix unit's 3-4× beats the scalar dequant kernel. Decode (rows < 8) stays on the scalar
        // path. fp-order/precision dependent, so gated behind FERRIC_COOP, never the default.
        if rows >= 8 && w.rows % 8 == 0 && self.ctx.coop_shared_ok() && std::env::var("FERRIC_COOP").is_ok() {
            return self.matmul_q2_0_coop(w);
        }
        // NOTE: a model-facing coop16 hook (route prefill Q2_0 through matmul_q2_0_coop16 on Vulkan)
        // was prototyped here but NOT shipped: it dequants each weight to f32 [K,N] per call, and one
        // batched forward keeps ~140 such buffers live (~4.6 GB) → OOMs a 6 GB card before completing,
        // so it can't be validated end-to-end. Needs f16 weight caching (dequant once at load, reuse)
        // — the kernel + microbenchmark (matmul_q2_0_coop16, up to 6.2×) are proven; wiring waits on that.
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        if q2_0_split_k(rows, w.rows) {
            // One workgroup per output element, laid out 2D because rows·out overruns the 65535
            // per-dimension cap (e.g. 5 tokens × 17408 outputs).
            let grid_w = n.min(32768);
            let grid_h = n.div_ceil(grid_w);
            let src = if use_subgroup(&self.ctx) { sg_reduce(MATMUL_Q2_0_SPLITK_WGSL) } else { MATMUL_Q2_0_SPLITK_WGSL.to_string() };
            run(&self.ctx, &src, "matmul_q2_0_splitk",
                &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
                  &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, grid_w as u32])],
                (grid_w as u32, grid_h as u32, 1));
        } else {
            // 2D for the same reason as split-K: one row of the grid tops out at 65535 workgroups.
            let wg = n.div_ceil(64);
            let gw = wg.min(32768);
            let gh = wg.div_ceil(gw);
            let (wgsl, label) = if q2_0_transposed() {
                (MATMUL_Q2_0_TRANS_WGSL, "matmul_q2_0_trans")
            } else {
                (MATMUL_Q2_0_FLAT_WGSL, "matmul_q2_0_flat")
            };
            run(&self.ctx, wgsl, label,
                &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
                  &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, (gw * 64) as u32])],
                (gw as u32, gh as u32, 1));
        }
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// **Cooperative-matrix (tensor-core) Q2_0 matmul for PREFILL** — where the weight read is
    /// amortized over many tokens and the multiply is a real GEMM. Each subgroup owns an 8×8 output
    /// tile; per K-step it dequantizes the packed 8×8 W tile into shared memory (transposed to
    /// [K,N]), loads it + the f32 activation tile as coop matrices, and `coopMultiplyAdd`s. This is
    /// where the 6–32× matrix-unit speedup meets a real quantized model. Requires rows(M)%8==0 and
    /// out(N)%8==0 (cols already %128), plus `ctx.coop_gemm_ok()`; fp-order/precision dependent
    /// (NVIDIA TF32), so a prefill fast-path, not the deterministic default.
    pub fn matmul_q2_0_coop(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch");
        assert!(w.rows % 8 == 0, "matmul_q2_0_coop needs N (out) a multiple of 8");
        // Pad the token dimension up to a multiple of 8 (the coop tile), compute, then slice back —
        // so any prompt length works. The pad rows are wasted tiles, cheap at prefill.
        // `FERRIC_COOP_2PASS` selects the two-pass (dequant→f32, then row-major coop GEMM) alternative;
        // correct + fast on Metal, but it does NOT fix NVIDIA (see coop_shared_ok — the NVIDIA coop
        // load reads a GPU-written buffer as stale, a wgpu/naga barrier gap no kernel shape works around).
        if std::env::var("FERRIC_COOP_2PASS").is_ok() {
            return self.matmul_q2_0_coop2pass(w);
        }
        let mrows = rows.div_ceil(8) * 8;
        let xp = if mrows == rows { x } else { x.pad_rows(mrows) };
        let out = empty(&self.ctx, mrows * w.rows);
        let nblk = (inn / 128) as u32;
        run(&self.ctx, MATMUL_Q2_0_COOP_WGSL, "matmul_q2_0_coop",
            &[xp.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[mrows as u32, inn as u32, w.rows as u32, nblk])],
            ((w.rows / 8) as u32, (mrows / 8) as u32, 1));
        let full = Tensor::from_parts(&self.ctx, out, vec![mrows, w.rows]);
        if mrows == rows { full } else { full.narrow(0, 0, rows).contiguous() }
    }

    /// Zero-pad row count up to `mrows` (rows ≥ current). For coop tile alignment at prefill.
    fn pad_rows(&self, mrows: usize) -> Tensor {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let out = empty(&self.ctx, mrows * cols);
        let c = self.contiguous();
        // copy the real rows into the (zeroed) padded buffer
        run(&self.ctx, PAD_ROWS_WGSL, "pad_rows", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[(rows * cols) as u32, 0])], groups(rows * cols));
        Tensor::from_parts(&self.ctx, out, vec![mrows, cols])
    }
}

const MATMUL_Q2_0_COOP_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:      array<f32>;  // [M,K] activations
@group(0) @binding(1) var<storage,read>       codes:  array<u32>;  // Q2_0 codes, W [N,K]
@group(0) @binding(2) var<storage,read>       scales: array<u32>;  // Q2_0 scales
@group(0) @binding(3) var<storage,read_write> c:      array<f32>;  // [M,N]
@group(0) @binding(4) var<uniform>            dims:   vec4<u32>;   // M, K, N, nblk
var<workgroup> bs: array<f32, 64>;                                 // dequantized W tile, [K,N] layout
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z; let nblk = dims.w;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        // dequant the 8×8 W tile [n0..+8, k0..+8] into bs, TRANSPOSED to [k,n] row-major for role B
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = t + e * 32u;                       // 0..64 over (n_local, k_local)
            let nl = i / 8u; let kl = i % 8u;
            let n = n0 + nl; let k = k0 + kl;
            let gblk = n * nblk + (k / 128u); let j = k % 128u;
            let sw = unpack2x16float(scales[gblk >> 1u]);
            let d = select(sw.y, sw.x, (gblk & 1u) == 0u);
            let word = codes[gblk * 8u + (j >> 4u)];
            let code = (word >> ((j & 15u) * 2u)) & 3u;
            bs[kl * 8u + nl] = f32(i32(code) - 1) * d;  // bs[k*8+n] = W[n][k]  → B[k][n]
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u);
        acc = coopMultiplyAdd(ma, mb, acc);
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

const MATMUL_Q4_K_COOP_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       x:      array<f32>;  // [M,K]
@group(0) @binding(1) var<storage,read>       codes:  array<u32>;  // Q4_K codes, W [N,K]
@group(0) @binding(2) var<storage,read>       aux:    array<u32>;  // Q4_K aux (d|dmin, scales)
@group(0) @binding(3) var<storage,read_write> c:      array<f32>;  // [M,N]
@group(0) @binding(4) var<uniform>            dims:   vec4<u32>;   // M, K, N, nblk(=K/256)
var<workgroup> bs: array<f32, 64>;
__HELPERS__
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z; let nblk = dims.w;
    let m0 = wid.y * 8u; let n0 = wid.x * 8u; let t = lid.x;
    let ci = m0 * nn + n0;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k0: u32 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let i = t + e * 32u; let nl = i / 8u; let kl = i % 8u;
            let n = n0 + nl; let k = k0 + kl;
            let gblk = n * nblk + (k / 256u); let v = k % 256u;   // super-block index, value in block
            let s = v / 32u; let l = v % 32u;                     // sub-block, position
            let ab = gblk * 4u;
            let dd = unpack2x16float(aux[ab]); let sm = scmin(ab, s);
            let ds = dd.x * f32(sm.x); let mm = dd.y * f32(sm.y);
            let word = codes[gblk * 32u + 8u * (s >> 1u) + (l >> 2u)];
            let sh = 8u * (l & 3u) + select(0u, 4u, (s & 1u) == 1u);
            let nib = (word >> sh) & 0xFu;
            bs[kl * 8u + nl] = ds * f32(nib) - mm;
        }
        workgroupBarrier();
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&x[m0 * kk + k0], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&bs[0], 8u);
        acc = coopMultiplyAdd(ma, mb, acc);
        workgroupBarrier();
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

// Dequant packed Q2_0 [N,K] → f32 **[K,N]** (transposed). One thread per source element; 2D grid.
const DEQ_Q2_0_T_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;   // [K,N]
@group(0) @binding(3) var<storage,read>       info:   array<u32>;   // n_elem, K, nblk, N, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Every storage index is let-bound before use: the forked naga SPIR-V backend panics
    // ("Expression is not cached") when an Access index is an inline compound expression under
    // ReadZeroSkipWrite bounds checks — the same workaround the coop GEMM uses for its coopLoad ptrs.
    let rs = info[4]; let ne = info[0]; let kk = info[1]; let nblk = info[2]; let nn = info[3];
    let e = gid.x + gid.y * rs;
    if (e >= ne) { return; }
    let n = e / kk; let k = e % kk;
    let blk = k / 128u; let j = k % 128u; let gblk = n * nblk + blk;
    let si = gblk >> 1u; let sw = unpack2x16float(scales[si]);
    let d = select(sw.y, sw.x, (gblk & 1u) == 0u);
    let ci = gblk * 8u + (j >> 4u); let word = codes[ci];
    let code = (word >> ((j & 15u) * 2u)) & 3u;
    let oi = k * nn + n; out[oi] = f32(i32(code) - 1) * d;   // transposed write [K,N]
}
"#;

// Non-transposed variant for the Metal-4 NT route: the resident matmul_bt consumes W as [N,K]
// directly (the packed layout's own row order), so the dequant is a straight linear write —
// coalesced on both sides, where the transposed variant's 68 KB-strided writes ran ~7x below
// bandwidth and dominated small prefills.
const DEQ_Q2_0_NT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;   // [N,K] — same order as packed
@group(0) @binding(3) var<storage,read>       info:   array<u32>;   // n_elem, K, nblk, N, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rs = info[4]; let ne = info[0]; let kk = info[1]; let nblk = info[2];
    let e = gid.x + gid.y * rs;
    if (e >= ne) { return; }
    let n = e / kk; let k = e % kk;
    let blk = k / 128u; let j = k % 128u; let gblk = n * nblk + blk;
    let si = gblk >> 1u; let sw = unpack2x16float(scales[si]);
    let d = select(sw.y, sw.x, (gblk & 1u) == 0u);
    let ci = gblk * 8u + (j >> 4u); let word = codes[ci];
    let code = (word >> ((j & 15u) * 2u)) & 3u;
    out[e] = f32(i32(code) - 1) * d;
}
"#;

const PAD_ROWS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       inp: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;
@group(0) @binding(2) var<storage,read>       info: array<u32>; // n_real
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= info[0]) { return; }
    out[i] = inp[i];   // extra rows stay zero (empty() buffer is zeroed)
}
"#;

/// Which `matmul_q2_0` kernel to use. The deciding factor is measured, and it is the number of
/// *output elements* (`rows·out`) — not the K depth, as one might assume:
///
///   ffn_down 17408→5120, 1 token   flat 1.04 ms → split-K 0.40 ms   (2.6× faster)
///   gdn qkv  5120→10240,  1 token   flat 0.64 ms → split-K 0.34 ms   (1.9× faster)
///   gdn qkv  5120→10240,  5 tokens  flat 0.90 ms → split-K 1.46 ms   (1.6× slower)
///
/// Flat gives one thread per output; split-K gives a whole workgroup (64× the threads) per output,
/// paid for with a barrier reduction. The deciding factor is **rows** (tokens in flight), which the
/// per-shape microbenchmarks obscured — those 0.2 ms decode matmuls swing 3× run-to-run (clock ramp,
/// contention), so the selector was tuned on whole-model ms/token instead:
///   decode, rows=1 → split-K wins on every shape but the LM head (168 vs 179 ms/tok all-split-K)
///   prefill, rows≥4 → flat wins for large matmuls; the rows already fill the machine, barriers cost
/// So: at decode (few rows) use split-K broadly; at prefill fall back to the output-count threshold.
/// `FERRIC_Q2_0_KERNEL=flat|splitk|trans` forces one; `FERRIC_Q2_0_SPLITK_MAX` overrides the
/// prefill threshold for sweeps.
fn q2_0_split_k(rows: usize, n_out: usize) -> bool {
    match std::env::var("FERRIC_Q2_0_KERNEL").as_deref() {
        Ok("flat") | Ok("trans") => false,
        Ok("splitk") => true,
        _ => {
            let thresh = std::env::var("FERRIC_Q2_0_SPLITK_MAX").ok().and_then(|s| s.parse().ok());
            if rows <= 2 {
                // decode: enough K-parallelism to matter, and even the 248320-wide LM head prefers it
                n_out < thresh.unwrap_or(1 << 20)
            } else {
                n_out < thresh.unwrap_or(16384)
            }
        }
    }
}

/// Whether weights are uploaded output-major. This is a *layout* choice made at upload, so the
/// kernel must agree with it.
///
/// **Not the default: measured slower.** Output-major makes adjacent threads read adjacent words,
/// which is the textbook GEMV fix — but it *lost* (cold LM head 70.5 → 49.1 GB/s). Row-major is
/// already fine here because each thread streams 1280 contiguous bytes and consumes whole cache
/// lines on its own; coalescing across threads buys nothing, while output-major scatters each
/// thread's own stream ~1 MB per step. Kept behind `FERRIC_Q2_0_KERNEL=trans` as evidence.
fn q2_0_transposed() -> bool { matches!(std::env::var("FERRIC_Q2_0_KERNEL").as_deref(), Ok("trans")) }

/// Rewrite a split-K quant-matmul kernel's final reduction from a shared-memory **barrier tree**
/// (6 `workgroupBarrier`s over 64 lanes) into a single hardware **`subgroupAdd`** per subgroup, then
/// a tiny combine of the (≤ a handful of) subgroup partials. All six split-K kernels share the exact
/// signature + tail this matches, so one transform serves them all. Applied only when the device has
/// the `subgroups` feature; `FERRIC_NO_SUBGROUP=1` forces the barrier path for A/B comparison.
fn sg_reduce(wgsl: &str) -> String {
    wgsl
        .replace(
            "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {",
            "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(subgroup_invocation_id) sglid: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {",
        )
        .replace(
            "        partial[t] = acc;\n        workgroupBarrier();\n        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }\n        if (t == 0u) { out[idx] = partial[0]; }",
            "        let sgsum = subgroupAdd(acc);\n        if (sglid == 0u) { partial[sgid] = sgsum; }\n        workgroupBarrier();\n        if (t == 0u) { var tot = 0.0; for (var i: u32 = 0u; i < nsg; i = i + 1u) { tot = tot + partial[i]; } out[idx] = tot; }",
        )
}

/// Whether to use the subgroup reduction. **Opt-in** (`FERRIC_SUBGROUP=1`), NOT the default —
/// deliberately. `subgroupAdd` reduces in hardware-cooperative order, which differs from the barrier
/// tree's pairwise order, so a fabric with subgroups and one without produce fp-different (though
/// argmax-identical, llama.cpp-matching) results. Ferric's distinctive moat is **bit-identical
/// cross-fabric** output, and not every fabric exposes subgroups (e.g. Chrome/ANGLE-Metal here did
/// not), so the deterministic barrier path is the default and subgroups are a speed opt-in for
/// single-fabric use. Measured ~5-10% on M5; re-evaluate if a cooperative-matrix path makes it larger.
fn use_subgroup(ctx: &Context) -> bool { ctx.subgroups && std::env::var("FERRIC_SUBGROUP").is_ok() }

/// Output-major GEMV: one thread per output, walking all of K. Adjacent threads read adjacent
/// words, so a SIMD group's loads coalesce into one contiguous run — the property split-K bought
/// with barriers, obtained here for free from the layout.
const MATMUL_Q2_0_TRANS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;   // [word][output]
@group(0) @binding(2) var<storage,read>        scales: array<u32>;   // [block][output], f16 x2 per u32
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;    // rows, out, in, threads_per_grid_row
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;   // adjacent idx → adjacent o → adjacent addresses
    let nblk = in_dim / 128u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let si = blk * o_dim + o;
        let sw = unpack2x16float(scales[si >> 1u]);
        let d = select(sw.y, sw.x, (si & 1u) == 0u);
        let xbase = r * in_dim + blk * 128u;
        var bacc = 0.0;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word = codes[(blk * 8u + w) * o_dim + o];   // coalesced across threads
            let xb = xbase + w * 16u;
            for (var b: u32 = 0u; b < 16u; b = b + 1u) {
                bacc = bacc + x[xb + b] * f32(i32((word >> (2u * b)) & 3u) - 1);
            }
        }
        acc = acc + bacc * d;   // block scale is constant over the 128-group
    }
    out[idx] = acc;
}
"#;

/// One thread per output element, walking all of K itself. No barriers, but a long dependent
/// accumulate chain and only `rows·out` threads in flight.
///
/// Dispatched 2D: a 1D grid caps at 65535 workgroups = 4.19M threads, which a real LM head blows
/// straight through (17 tokens × 248320 vocab = 4.22M outputs → 65960 workgroups).
/// `x` is read as `vec4<f32>`, four activations per load, and each group of four weights is reduced
/// with `dot()`. The scalar form issues **16 x-loads per code word** — 5120 per output against only
/// 320 code loads — so the activation loads, not the weights, dominate the instruction stream.
/// Every thread in a wave reads the same `x` (same token), so these all hit cache; the cost is
/// issue slots, not bandwidth, which is exactly what a latency-bound kernel cannot afford.
// Q4_K super-block = 256 values / 8 sub-blocks. Shared preamble: extract a sub-block's 6-bit
// (scale, min) from the 12 packed scale bytes, and dequant value = d·scaleₛ·q − dmin·minₛ.
const Q4_K_HELPERS: &str = r#"
fn scbyte(base: u32, i: u32) -> u32 { return (aux[base + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn scmin(base: u32, j: u32) -> vec2<u32> {
    if (j < 4u) { return vec2<u32>(scbyte(base, j) & 63u, scbyte(base, j + 4u) & 63u); }
    let a = scbyte(base, j + 4u); let lo = scbyte(base, j - 4u); let hi = scbyte(base, j);
    return vec2<u32>((a & 0x0Fu) | ((lo >> 6u) << 4u), (a >> 4u) | ((hi >> 6u) << 4u));
}
"#;

// Inner sub-block accumulate, vectorized: one u32 code-word feeds 4 quants, read against a vec4 of
// activations. Per sub-block s (32 values = 8 words): contribution = d·scaleₛ·Σ(x·q) − dmin·minₛ·Σx.
const Q4_K_INNER: &str = r#"
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb8 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
"#;

// Q5_K inner: like Q4_K but each 4-bit quant gains a 5th bit from qh (word codes[qh_base+w], bit s).
const Q5_K_INNER: &str = r#"
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb40 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let qhw = codes[cb40 + 32u + w];
                let bit = vec4<f32>(f32((qhw >> s) & 1u), f32((qhw >> (8u + s)) & 1u), f32((qhw >> (16u + s)) & 1u), f32((qhw >> (24u + s)) & 1u)) * 16.0;
                let q = nib + bit;
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, q) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
"#;

const MATMUL_Q5_K_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim; let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o * nblk + blk; let ab = bi * 4u; let cb40 = bi * 40u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
__INNER__
        }
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q5_K_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;
var<workgroup> partial: array<f32, 64>;
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim; let nblk = in_dim / 256u;
        var acc = 0.0;
        for (var blk: u32 = t; blk < nblk; blk = blk + 64u) {
            let bi = o * nblk + blk; let ab = bi * 4u; let cb40 = bi * 40u;
            let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
            let xbb = r * in_dim + blk * 256u;
            for (var s: u32 = 0u; s < 8u; s = s + 1u) {
__INNER__
            }
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Fused FFN: gate/up projection + SwiGLU in one kernel. Each thread computes the gate row (o) and
// the up row (o+n_ff) via the same Q4_K dequant-dot, then writes silu(gate)·up — no 2·n_ff
// intermediate, no separate SwiGLU dispatch. info.y = n_ff (the output width); the weight has 2·n_ff rows.
const MATMUL_Q4_K_SWIGLU_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, n_ff(out), in, row_stride
__HELPERS__
fn qk_dot(o_row: u32, r: u32, nblk: u32, in_dim: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb8 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let n_ff = info.y; let in_dim = info.z;
    if (idx >= rows * n_ff) { return; }
    let o = idx % n_ff; let r = idx / n_ff;
    let nblk = in_dim / 256u;
    let g = qk_dot(o, r, nblk, in_dim);         // gate row
    let u = qk_dot(o + n_ff, r, nblk, in_dim);  // up row
    out[idx] = (g / (1.0 + exp(-g))) * u;       // silu(g)·u
}
"#;

// MoE router top-k on the GPU: scores (softmax/sigmoid), biased SELECTION (bias never in weights),
// renormalized scaled weights → out = [w_0..w_{k-1} | idx_0..idx_{k-1}]. One thread scans ≤1024
// experts serially — nanoseconds of ALU against a saved CPU round-trip per layer per token.
const MOE_TOPK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       logits: array<f32>;
@group(0) @binding(1) var<storage,read>       bias:   array<f32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>; // ne,k,sigmoid,has_bias | scale_bits
fn score(lb: u32, i: u32, sig: u32, maxl: f32, ssum: f32) -> f32 {
    if (sig == 1u) { return 1.0 / (1.0 + exp(-logits[lb + i])); }
    return exp(logits[lb + i] - maxl) / ssum;
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ne = info[0].x; let k = info[0].y; let sig = info[0].z; let hb = info[0].w;
    let scale = bitcast<f32>(info[1].x); let t = info[1].y;
    let ti = gid.x;
    if (ti >= t) { return; }
    let lb = ti * ne;   // this token's contiguous logits row
    var maxl = -3.4e38; var ssum = 0.0;
    if (sig == 0u) {
        for (var i = 0u; i < ne; i = i + 1u) { maxl = max(maxl, logits[lb + i]); }
        for (var i = 0u; i < ne; i = i + 1u) { ssum = ssum + exp(logits[lb + i] - maxl); }
    }
    var picked: array<u32, 32>;
    for (var w = 0u; w < 32u; w = w + 1u) { picked[w] = 0u; }
    let ob = ti * 2u * k;
    var wsum = 0.0;
    for (var s = 0u; s < k; s = s + 1u) {
        var bi = 0u; var bv = -3.4e38;
        for (var i = 0u; i < ne; i = i + 1u) {
            if ((picked[i >> 5u] & (1u << (i & 31u))) != 0u) { continue; }
            var m = score(lb, i, sig, maxl, ssum);
            if (hb == 1u) { m = m + bias[i]; }
            if (m > bv) { bv = m; bi = i; }
        }
        picked[bi >> 5u] = picked[bi >> 5u] | (1u << (bi & 31u));
        let sc = score(lb, bi, sig, maxl, ssum);
        out[ob + s] = sc; out[ob + k + s] = f32(bi);
        wsum = wsum + sc;
    }
    // info[1].z = renormalise. DeepSeek-V2 omits `expert_weights_norm`, which llama.cpp defaults to
    // false: the routed weights stay the raw top-k probabilities and do NOT sum to 1.
    let den = select(1.0, wsum, info[1].z != 0u);
    for (var s = 0u; s < k; s = s + 1u) { out[ob + s] = out[ob + s] / den * scale; }
}
"#;

// Batched selected-expert gate|up + SwiGLU (MoE): weight rows are expert-major in one slab; the k
// selected expert ids ride in the uniform (info[1..3]); x is a single hidden row. Same qk_dot math as
// MATMUL_Q4_K_SWIGLU_WGSL — identical bytes per expert row, just indirect row addressing.
const MATMUL_Q4_K_SWIGLU_ID_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>; // [T, w_0..w_{k-1} | idx_0..idx_{k-1}] from moe_topk
@group(0) @binding(5) var<uniform>             info:   array<vec4<u32>, 2>;  // k, eff, in, gw | tot
__HELPERS__
fn qk_dot(o_row: u32, r: u32, nblk: u32, in_dim: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb8 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[0].w; let k = info[0].x; let eff = info[0].y; let in_dim = info[0].z;
    if (idx >= info[1].x) { return; }
    let ti = idx / (k * eff); let rem = idx % (k * eff);
    let s = rem / eff; let o = rem % eff;
    let e = u32(selw[ti * 2u * k + k + s]);
    let base = e * (2u * eff);
    let nblk = in_dim / 256u;
    let g = qk_dot(base + o, ti, nblk, in_dim);         // this expert's gate row · token ti's hidden
    let u = qk_dot(base + eff + o, ti, nblk, in_dim);   // this expert's up row
    out[idx] = (g / (1.0 + exp(-g))) * u;               // silu(g)·u
}
"#;

// Batched selected-expert down projection (MoE): x row s is expert slot s's swiglu output; weight rows
// are expert-major in one slab. Same per-block math as MATMUL_Q6_K_FLAT_WGSL (__BODY__ reads `bi`,`r`).
const MATMUL_Q6_K_ID_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>; // [w | idx] from moe_topk
@group(0) @binding(5) var<uniform>             info:   vec4<u32>;  // k, out_pe, in, gw
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let k = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= k * o_dim) { return; }
    let s = idx / o_dim; let o = idx % o_dim;
    let e = u32(selw[k + s]);
    let row = e * o_dim + o;      // absolute weight row in the expert slab
    let r = s;                    // x row for this slot (each expert has its own hidden)
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = row * nblk + blk;
__BODY__
    }
    out[idx] = acc;
}
"#;


// Batched selected-expert down + WEIGHTED SUM fused (MoE): out[o] = Σ_s w_s · dot(x_s, W[e_s][o]) —
// one dispatch replaces the down matmul plus the narrow/reshape/broadcast/mul/sum combine chain.
const MATMUL_Q6_K_ID_WSUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>; // [T, w | idx] from moe_topk
@group(0) @binding(5) var<uniform>             info:   array<vec4<u32>, 2>;  // k, out_pe, in, gw | tot
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[0].w; let k = info[0].x; let o_dim = info[0].y; let in_dim = info[0].z;
    if (idx >= info[1].x) { return; }
    let ti = idx / o_dim; let o = idx % o_dim;
    let nblk = in_dim / 256u;
    var total = 0.0;
    for (var s = 0u; s < k; s = s + 1u) {
        let e = u32(selw[ti * 2u * k + k + s]);
        let row = e * o_dim + o;
        let r = ti * k + s;
        var acc = 0.0;
        for (var blk = 0u; blk < nblk; blk = blk + 1u) {
            let bi = row * nblk + blk;
__BODY__
        }
        total = total + selw[ti * 2u * k + s] * acc;
    }
    out[idx] = total;
}
"#;

// Batched selected-expert down + WEIGHTED SUM for a Q4_K down slab (Q4_K_M quantizes half the
// layers' down_exps as Q4_K, the other half Q6_K) — same structure as MATMUL_Q6_K_ID_WSUM_WGSL,
// Q4_K block math (identical to the swiglu_id kernel's qk_dot).
const MATMUL_Q4_K_ID_WSUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;  // [T·k, in] swiglu outputs
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>; // [T, w | idx] from moe_topk
@group(0) @binding(5) var<uniform>             info:   array<vec4<u32>, 2>;  // k, out_pe, in, gw | tot
__HELPERS__
fn qk_dot(o_row: u32, r: u32, nblk: u32, in_dim: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb8 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[0].w; let k = info[0].x; let o_dim = info[0].y; let in_dim = info[0].z;
    if (idx >= info[1].x) { return; }
    let ti = idx / o_dim; let o = idx % o_dim;
    let nblk = in_dim / 256u;
    var total = 0.0;
    for (var s = 0u; s < k; s = s + 1u) {
        let e = u32(selw[ti * 2u * k + k + s]);
        total = total + selw[ti * 2u * k + s] * qk_dot(e * o_dim + o, ti * k + s, nblk, in_dim);
    }
    out[idx] = total;
}
"#;

// Q8_0 selected-expert gate|up + SwiGLU — the MTP draft block's experts are Q8_0; without a slab
// path they fall back to per-token CPU routing whose mid-batch readbacks are both slow and racy.
// Same block/word order as MATMUL_Q8_0_FLAT_WGSL.
const MATMUL_Q8_0_SWIGLU_ID_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>; // [T, w | idx] from moe_topk
@group(0) @binding(5) var<uniform>             info:   array<vec4<u32>, 2>;  // k, eff, in, gw | tot
fn q8_dot(o_row: u32, r: u32, in_dim: u32) -> f32 {
    let nblk = in_dim / 32u; let nwords = nblk * 8u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 3u; let bi = o_row * nblk + blk;
        let sw = unpack2x16float(scales[bi >> 1u]);
        let d = select(sw.y, sw.x, (bi & 1u) == 0u);
        let word = codes[o_row * nwords + w];
        let xi = (r * in_dim + blk * 32u + (w & 7u) * 4u) >> 2u;
        let v = vec4<f32>(f32(i32(word << 24u) >> 24u), f32(i32(word << 16u) >> 24u), f32(i32(word << 8u) >> 24u), f32(i32(word) >> 24u));
        acc = acc + d * dot(x[xi], v);
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[0].w; let k = info[0].x; let eff = info[0].y; let in_dim = info[0].z;
    if (idx >= info[1].x) { return; }
    let ti = idx / (k * eff); let rem = idx % (k * eff);
    let s = rem / eff; let o = rem % eff;
    let e = u32(selw[ti * 2u * k + k + s]);
    let base = e * (2u * eff);
    let g = q8_dot(base + o, ti, in_dim);
    let u = q8_dot(base + eff + o, ti, in_dim);
    out[idx] = (g / (1.0 + exp(-g))) * u;
}
"#;

// Q8_0 selected-expert down + weighted sum — companion to MATMUL_Q8_0_SWIGLU_ID_WGSL.
const MATMUL_Q8_0_ID_WSUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;  // [T·k, in] swiglu outputs
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>; // [T, w | idx] from moe_topk
@group(0) @binding(5) var<uniform>             info:   array<vec4<u32>, 2>;  // k, out_pe, in, gw | tot
fn q8_dot(o_row: u32, r: u32, in_dim: u32) -> f32 {
    let nblk = in_dim / 32u; let nwords = nblk * 8u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 3u; let bi = o_row * nblk + blk;
        let sw = unpack2x16float(scales[bi >> 1u]);
        let d = select(sw.y, sw.x, (bi & 1u) == 0u);
        let word = codes[o_row * nwords + w];
        let xi = (r * in_dim + blk * 32u + (w & 7u) * 4u) >> 2u;
        let v = vec4<f32>(f32(i32(word << 24u) >> 24u), f32(i32(word << 16u) >> 24u), f32(i32(word << 8u) >> 24u), f32(i32(word) >> 24u));
        acc = acc + d * dot(x[xi], v);
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[0].w; let k = info[0].x; let o_dim = info[0].y; let in_dim = info[0].z;
    if (idx >= info[1].x) { return; }
    let ti = idx / o_dim; let o = idx % o_dim;
    var total = 0.0;
    for (var s = 0u; s < k; s = s + 1u) {
        let e = u32(selw[ti * 2u * k + k + s]);
        total = total + selw[ti * 2u * k + s] * q8_dot(e * o_dim + o, ti * k + s, in_dim);
    }
    out[idx] = total;
}
"#;

// Fused FFN gate/up + SwiGLU for Q5_K — Q4_K plus the 5th (qh) bit per quant.
const MATMUL_Q5_K_SWIGLU_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, n_ff(out), in, row_stride
__HELPERS__
fn qk_dot(o_row: u32, r: u32, nblk: u32, in_dim: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk; let ab = bi * 4u; let cb40 = bi * 40u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb40 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let qhw = codes[cb40 + 32u + w];
                let bit = vec4<f32>(f32((qhw >> s) & 1u), f32((qhw >> (8u + s)) & 1u), f32((qhw >> (16u + s)) & 1u), f32((qhw >> (24u + s)) & 1u)) * 16.0;
                let q = nib + bit;
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, q) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let n_ff = info.y; let in_dim = info.z;
    if (idx >= rows * n_ff) { return; }
    let o = idx % n_ff; let r = idx / n_ff;
    let nblk = in_dim / 256u;
    let g = qk_dot(o, r, nblk, in_dim);
    let u = qk_dot(o + n_ff, r, nblk, in_dim);
    out[idx] = (g / (1.0 + exp(-g))) * u;
}
"#;

// Fused FFN gate/up + SwiGLU for Q6_K — x is a plain f32 array; 6-bit reassembly per Q6_K_BODY.
const MATMUL_Q6_K_SWIGLU_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, n_ff(out), in, row_stride
__HELPERS__
fn qk_dot(o_row: u32, r: u32, nblk: u32, in_dim: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk;
        let cb = bi * 48u; let ab = bi * 5u;
        let d = unpack2x16float(aux[ab]).x;
        let xbb = r * in_dim + blk * 256u;
        for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
            let qlo = 64u * hf; let qho = 32u * hf; let sco = 8u * hf; let xh = xbb + 128u * hf;
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let is = l >> 4u; let h = qhb(cb, qho + l);
                let q1 = i32((qlb(cb, qlo + l) & 0xFu) | ((h & 3u) << 4u)) - 32;
                let q2 = i32((qlb(cb, qlo + l + 32u) & 0xFu) | (((h >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32((qlb(cb, qlo + l) >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32((qlb(cb, qlo + l + 32u) >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32;
                acc = acc + x[xh + l]        * d * scb(ab, sco + is)      * f32(q1);
                acc = acc + x[xh + 32u + l]  * d * scb(ab, sco + is + 2u) * f32(q2);
                acc = acc + x[xh + 64u + l]  * d * scb(ab, sco + is + 4u) * f32(q3);
                acc = acc + x[xh + 96u + l]  * d * scb(ab, sco + is + 6u) * f32(q4);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let n_ff = info.y; let in_dim = info.z;
    if (idx >= rows * n_ff) { return; }
    let o = idx % n_ff; let r = idx / n_ff;
    let nblk = in_dim / 256u;
    let g = qk_dot(o, r, nblk, in_dim);
    let u = qk_dot(o + n_ff, r, nblk, in_dim);
    out[idx] = (g / (1.0 + exp(-g))) * u;
}
"#;

// FULL FFN megakernel (spike): gate/up (Q4_K) + SwiGLU + down (Q6_K) in ONE dispatch. One workgroup
// per token computes the whole FFN — the n_ff-wide SwiGLU activation lives in workgroup shared memory
// (`hff`) and is never written to global, so the down projection reads it from fast on-chip memory and
// the [t, n_ff] intermediate + a whole dispatch vanish. The trade: one workgroup/token underfills the
// GPU at decode. Whether locality+fewer-dispatches beats the occupancy loss is a measured question.
// __NFF__ = n_ff (constant workgroup-array size, stamped at dispatch).
const FFN_MEGA_Q4K_Q6K_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:        array<vec4<f32>>;  // [rows, n_embd]
@group(0) @binding(1) var<storage,read>        gu_codes: array<u32>;         // Q4_K gate_up [2n_ff, n_embd]
@group(0) @binding(2) var<storage,read>        gu_aux:   array<u32>;
@group(0) @binding(3) var<storage,read>        dn_codes: array<u32>;         // Q6_K down [n_embd, n_ff]
@group(0) @binding(4) var<storage,read>        dn_aux:   array<u32>;
@group(0) @binding(5) var<storage,read_write>  out:      array<f32>;          // [rows, n_embd]
@group(0) @binding(6) var<uniform>             info:     vec4<u32>;           // rows, n_ff, n_embd, _
var<workgroup> hff: array<f32, __NFF__>;
// Q4_K gate_up dequant-dot (row o_row of gate_up against token r), vec4 over x.
fn gu_scbyte(base: u32, i: u32) -> u32 { return (gu_aux[base + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn gu_scmin(base: u32, j: u32) -> vec2<u32> {
    if (j < 4u) { return vec2<u32>(gu_scbyte(base, j) & 63u, gu_scbyte(base, j + 4u) & 63u); }
    let a = gu_scbyte(base, j + 4u); let lo = gu_scbyte(base, j - 4u); let hi = gu_scbyte(base, j);
    return vec2<u32>((a & 0x0Fu) | ((lo >> 6u) << 4u), (a >> 4u) | ((hi >> 6u) << 4u));
}
fn gu_dot(o_row: u32, r: u32, nblk: u32, in_dim: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
        let dd = unpack2x16float(gu_aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = gu_scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb8 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = gu_codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    return acc;
}
// Q6_K down dequant-dot (row o_row of down against the shared hff vector), scalar over hff.
fn dn_qlb(cb: u32, i: u32) -> u32 { return (dn_codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn dn_qhb(cb: u32, i: u32) -> u32 { return (dn_codes[cb + 32u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn dn_scb(ab: u32, i: u32) -> f32 { let b = (dn_aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; return f32(i32(b << 24u) >> 24u); }
fn dn_dot(o_row: u32, nblk: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o_row * nblk + blk; let cb = bi * 48u; let ab = bi * 5u;
        let d = unpack2x16float(dn_aux[ab]).x;
        let xbb = blk * 256u;
        for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
            let qlo = 64u * hf; let qho = 32u * hf; let sco = 8u * hf; let xh = xbb + 128u * hf;
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let is = l >> 4u; let h = dn_qhb(cb, qho + l);
                let q1 = i32((dn_qlb(cb, qlo + l) & 0xFu) | ((h & 3u) << 4u)) - 32;
                let q2 = i32((dn_qlb(cb, qlo + l + 32u) & 0xFu) | (((h >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32((dn_qlb(cb, qlo + l) >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32((dn_qlb(cb, qlo + l + 32u) >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32;
                acc = acc + hff[xh + l]        * d * dn_scb(ab, sco + is)      * f32(q1);
                acc = acc + hff[xh + 32u + l]  * d * dn_scb(ab, sco + is + 2u) * f32(q2);
                acc = acc + hff[xh + 64u + l]  * d * dn_scb(ab, sco + is + 4u) * f32(q3);
                acc = acc + hff[xh + 96u + l]  * d * dn_scb(ab, sco + is + 6u) * f32(q4);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x; let tid = lid.x;
    let n_ff = info.y; let n_embd = info.z;
    let nblk_e = n_embd / 256u; let nblk_f = n_ff / 256u;
    for (var j: u32 = tid; j < n_ff; j = j + 256u) {
        let g = gu_dot(j, row, nblk_e, n_embd);
        let u = gu_dot(j + n_ff, row, nblk_e, n_embd);
        hff[j] = (g / (1.0 + exp(-g))) * u;
    }
    workgroupBarrier();
    for (var k: u32 = tid; k < n_embd; k = k + 256u) {
        out[row * n_embd + k] = dn_dot(k, nblk_f);
    }
}
"#;

/// Fused SwiGLU with the transposed (coalesced) gate_up layout — combines fusion (no [t,2·n_ff]
/// intermediate) with coalesced weight reads. Same math as MATMUL_Q4_K_SWIGLU_WGSL, bit-identical.
const MATMUL_Q4_K_SWIGLU_TRANS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:       array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes_t: array<u32>;   // [block][word][row] over the full 2·n_ff rows
@group(0) @binding(2) var<storage,read>        aux_t:   array<u32>;   // [block][k][row]
@group(0) @binding(3) var<storage,read_write>  out:     array<f32>;
@group(0) @binding(4) var<uniform>             info:    vec4<u32>;    // rows, n_ff, in, grid_width
fn scbyte_t(base: u32, od: u32, i: u32) -> u32 { return (aux_t[base + od * (1u + (i >> 2u))] >> (8u * (i & 3u))) & 0xffu; }
fn scmin_t(base: u32, od: u32, j: u32) -> vec2<u32> {
    if (j < 4u) { return vec2<u32>(scbyte_t(base, od, j) & 63u, scbyte_t(base, od, j + 4u) & 63u); }
    let a = scbyte_t(base, od, j + 4u); let lo = scbyte_t(base, od, j - 4u); let hi = scbyte_t(base, od, j);
    return vec2<u32>((a & 0x0Fu) | ((lo >> 6u) << 4u), (a >> 4u) | ((hi >> 6u) << 4u));
}
fn qk_dot_t(o_row: u32, r: u32, nblk: u32, in_dim: u32, nrow: u32) -> f32 {
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let ab = blk * 4u * nrow + o_row;
        let dd = unpack2x16float(aux_t[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        let cbase = blk * 32u * nrow + o_row;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin_t(ab, nrow, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cbase + (8u * (s >> 1u)) * nrow; let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes_t[cw + w * nrow];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let n_ff = info.y; let in_dim = info.z;
    if (idx >= rows * n_ff) { return; }
    let o = idx % n_ff; let r = idx / n_ff;
    let nblk = in_dim / 256u; let nrow = 2u * n_ff;
    let g = qk_dot_t(o, r, nblk, in_dim, nrow);
    let u = qk_dot_t(o + n_ff, r, nblk, in_dim, nrow);
    out[idx] = (g / (1.0 + exp(-g))) * u;
}
"#;

/// **K-split subgroup GEMV** (opt-in `FERRIC_SGGEMV`): one subgroup per output, its lanes split the
/// blocks (lane L does blocks L, L+sgsz, …) into partial dots, then `subgroupAdd` reduces. The classic
/// warp-cooperative GEMV — more memory parallelism per output than one-thread-per-output. Reuses the
/// FLAT weight buffers. NOT bit-identical (reduction order differs) → opt-in fast path, not the default.
const MATMUL_Q4_K_SGGEMV_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_width
__HELPERS__
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) sgsz: u32) {
    let idx = wg.x + wg.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = lane; blk < nblk; blk = blk + sgsz) {
        let bi = o * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin(ab, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cb8 + 8u * (s >> 1u); let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cw + w];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    let total = subgroupAdd(acc);
    if (lane == 0u) { out[idx] = total; }
}
"#;

/// Transposed (output-minor) Q4_K GEMV — same math as the flat kernel, weights reordered so the 64
/// output-threads of a workgroup read contiguous memory (coalesced). A/B experiment vs the flat kernel.
const MATMUL_Q4_K_TRANS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:       array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes_t: array<u32>;   // [block][word][output]
@group(0) @binding(2) var<storage,read>        aux_t:   array<u32>;   // [block][k][output]
@group(0) @binding(3) var<storage,read_write>  out:     array<f32>;
@group(0) @binding(4) var<uniform>             info:    vec4<u32>;    // rows, out(=o_dim), in, row_stride
fn scbyte_t(base: u32, od: u32, i: u32) -> u32 { return (aux_t[base + od * (1u + (i >> 2u))] >> (8u * (i & 3u))) & 0xffu; }
fn scmin_t(base: u32, od: u32, j: u32) -> vec2<u32> {
    if (j < 4u) { return vec2<u32>(scbyte_t(base, od, j) & 63u, scbyte_t(base, od, j + 4u) & 63u); }
    let a = scbyte_t(base, od, j + 4u); let lo = scbyte_t(base, od, j - 4u); let hi = scbyte_t(base, od, j);
    return vec2<u32>((a & 0x0Fu) | ((lo >> 6u) << 4u), (a >> 4u) | ((hi >> 6u) << 4u));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var j: u32 = 0u; j < nblk; j = j + 1u) {
        let ab = j * 4u * o_dim + o;
        let dd = unpack2x16float(aux_t[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + j * 256u;
        let cbase = j * 32u * o_dim + o;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sm = scmin_t(ab, o_dim, s); let ds = d * f32(sm.x); let mm = dmin * f32(sm.y);
            let cw = cbase + (8u * (s >> 1u)) * o_dim; let hi = s & 1u; let xv = (xbb + 32u * s) >> 2u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes_t[cw + w * o_dim];
                var nib: vec4<f32>;
                if (hi == 0u) { nib = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)); }
                else          { nib = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)); }
                let xw = x[xv + w];
                acc = acc + ds * dot(xw, nib) - mm * (xw.x + xw.y + xw.z + xw.w);
            }
        }
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q4_K_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;  // 32 u32/block (128 quant bytes)
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;  // 4 u32/block: d|dmin, 12 scale bytes
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, row_stride
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
        let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
        let xbb = r * in_dim + blk * 256u;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
__INNER__
        }
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q4_K_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w
var<workgroup> partial: array<f32, 64>;
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 256u;
        var acc = 0.0;
        for (var blk: u32 = t; blk < nblk; blk = blk + 64u) {
            let bi = o * nblk + blk; let ab = bi * 4u; let cb8 = bi * 32u;
            let dd = unpack2x16float(aux[ab]); let d = dd.x; let dmin = dd.y;
            let xbb = r * in_dim + blk * 256u;
            for (var s: u32 = 0u; s < 8u; s = s + 1u) {
__INNER__
            }
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Q6_K: byte accessors into the packed ql|qh codes and int8 scales, plus the per-super-block body
// that reassembles each 6-bit quant (4 low bits from ql, 2 high from qh) and accumulates
// x · d · scale · (q−32). Two 128-value halves, 4 quant groups per half — the llama.cpp layout.
/// The flat and split-k shells bind exactly `(x, codes, aux, out, info)` and contain no format
/// knowledge — only the spliced `__BODY__` does. They are named for Q6_K because it was the first
/// format to use them; these aliases say so at the point of reuse rather than leaving a Q2_K matmul
/// looking like it borrowed someone else's kernel.
const MATMUL_K_FLAT_WGSL: &str = MATMUL_Q6_K_FLAT_WGSL;
const MATMUL_K_SPLITK_WGSL: &str = MATMUL_Q6_K_SPLITK_WGSL;

// ---- Q2_K: 4 levels per 16-element sub-block, affine (scale AND min per sub-block) ----
//
// ⚠ The `qs` walk is the trap. A byte holds four 2-bit quants that belong to four DIFFERENT
// sub-blocks: the SHIFT selects the sub-block and the byte INDEX selects the element within it.
// Reading `qs` sequentially — the obvious loop — yields plausibly-scaled garbage, not an error.
// Element `e` of the super-block is at `qs[hf*32 + g*16 + l] >> (2*j)` for
// `e = hf*128 + j*32 + g*16 + l`, which is why the loop nests half → j → group → l.
const Q2_K_HELPERS: &str = r#"
fn q2b(cb: u32, i: u32) -> u32 { return (codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn q2s(ab: u32, i: u32) -> u32 { return (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
"#;
const Q2_K_BODY: &str = r#"
            let cb = bi * 16u; let ab = bi * 5u;
            let dd = unpack2x16float(aux[ab]);
            let xbb = r * in_dim + blk * 256u;
            for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
                        let sc = q2s(ab, hf * 8u + j * 2u + g);
                        let dl = dd.x * f32(sc & 0xFu);
                        let ml = dd.y * f32(sc >> 4u);
                        let qo = hf * 32u + g * 16u;
                        let xo = xbb + hf * 128u + j * 32u + g * 16u;
                        for (var l: u32 = 0u; l < 16u; l = l + 1u) {
                            let q = (q2b(cb, qo + l) >> (2u * j)) & 3u;
                            acc = acc + x[xo + l] * (dl * f32(q) - ml);
                        }
                    }
                }
            }
"#;

// ---- Q3_K: 8 levels per 16-element sub-block, symmetric, third bit on its own plane ----
//
// ⚠ Two traps. The high plane is INVERTED — bit SET means add nothing, bit CLEAR means subtract 4 —
// so the quant range is −4..3 and reading it the intuitive way flips the sign of most weights while
// staying finite. And the bit selector runs 1..128 across BOTH halves rather than restarting, so the
// mask bit is `hf*4 + j` over a 32-byte plane shared by the whole super-block, not a per-half one.
//
// The sixteen 6-bit scales are unshuffled at LOAD time (see `Q3_KWeights::from_bytes`), so `aux`
// holds plain bytes here and the kernel only subtracts the bias.
const Q3_K_HELPERS: &str = r#"
fn q3h(cb: u32, i: u32) -> u32 { return (codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn q3q(cb: u32, i: u32) -> u32 { return (codes[cb + 8u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn q3s(ab: u32, i: u32) -> f32 { let b = (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; return f32(i32(b) - 32); }
"#;
const Q3_K_BODY: &str = r#"
            let cb = bi * 24u; let ab = bi * 5u;
            let d = unpack2x16float(aux[ab]).x;
            let xbb = r * in_dim + blk * 256u;
            for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let m = 1u << (hf * 4u + j);
                    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
                        let dl = d * q3s(ab, hf * 8u + j * 2u + g);
                        let qo = hf * 32u + g * 16u;
                        let xo = xbb + hf * 128u + j * 32u + g * 16u;
                        for (var l: u32 = 0u; l < 16u; l = l + 1u) {
                            var q = i32((q3q(cb, qo + l) >> (2u * j)) & 3u);
                            if ((q3h(cb, g * 16u + l) & m) == 0u) { q = q - 4; }
                            acc = acc + x[xo + l] * dl * f32(q);
                        }
                    }
                }
            }
"#;

const Q6_K_HELPERS: &str = r#"
fn qlb(cb: u32, i: u32) -> u32 { return (codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn qhb(cb: u32, i: u32) -> u32 { return (codes[cb + 32u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn scb(ab: u32, i: u32) -> f32 { let b = (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; return f32(i32(b << 24u) >> 24u); }
"#;
const Q6_K_BODY: &str = r#"
            let cb = bi * 48u; let ab = bi * 5u;
            let d = unpack2x16float(aux[ab]).x;
            let xbb = r * in_dim + blk * 256u;
            for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
                let qlo = 64u * hf; let qho = 32u * hf; let sco = 8u * hf; let xh = xbb + 128u * hf;
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let is = l >> 4u; let h = qhb(cb, qho + l);
                    let q1 = i32((qlb(cb, qlo + l) & 0xFu) | ((h & 3u) << 4u)) - 32;
                    let q2 = i32((qlb(cb, qlo + l + 32u) & 0xFu) | (((h >> 2u) & 3u) << 4u)) - 32;
                    let q3 = i32((qlb(cb, qlo + l) >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32;
                    let q4 = i32((qlb(cb, qlo + l + 32u) >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32;
                    acc = acc + x[xh + l]        * d * scb(ab, sco + is)      * f32(q1);
                    acc = acc + x[xh + 32u + l]  * d * scb(ab, sco + is + 2u) * f32(q2);
                    acc = acc + x[xh + 64u + l]  * d * scb(ab, sco + is + 4u) * f32(q3);
                    acc = acc + x[xh + 96u + l]  * d * scb(ab, sco + is + 6u) * f32(q4);
                }
            }
"#;


// ---- IQ4_XS: 4-bit non-linear codebook, 256-value super-block ----
//
// Layout (136 bytes): f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]. Eight sub-blocks of 32 values;
// sub-block `ib` carries a 6-bit scale assembled from a nibble of `scales_l[ib/2]` and two bits of
// `scales_h` at bit 2*ib, giving `dl = d * (ls - 32)`. Each byte of `qs` holds two 4-bit indices into a
// FIXED 16-entry codebook, so the dequantised value is `dl * kvalues[idx]` and the levels are
// deliberately non-uniform. That codebook is why this is "IQ": the quantiser fits indices to a curve
// rather than a linear grid.
//
// Ported directly from this workspace's CPU reference (`ferric_gguf::deq_iq4_xs`), which is the
// authority the GPU path is asserted exact against.
const IQ4_XS_HELPERS: &str = r#"
const KV: array<i32, 16> = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
fn qsb(cb: u32, i: u32) -> u32 { return (codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
"#;
/// Two changes from the obvious transcription of the format. **Kept for accuracy, NOT for speed — the
/// speed hypothesis they were written to test was refuted.**
///
/// 1. **One `codes` load per four bytes.** `qsb` indexes `codes[cb + (i >> 2u)]`, so a loop over
///    `j = 0..16` touches four distinct words while issuing sixteen loads. This reads each word once
///    and shifts four bytes out of it.
/// 2. **Scale the sub-block once, not every element.** `dl` is constant across a 32-value sub-block, so
///    it leaves the inner loop: 1 multiply instead of 32.
///
/// The theory was that these explain IQ4_XS's headroom — it streams the fewest bytes of any format
/// (204.8 MB/token vs Q8_0's 531.1) and is still the slowest, at 2.97x Q8_0's effective bandwidth.
/// **Measured: no change.** IQ4_XS/Q8_0 went 1.143 → 1.147, a 0.4% move inside a 1.03x within-format
/// spread. Sixteen loads of one address apparently cost what four do (they hit cache), and 32 f32
/// multiplies are not what binds. **So the headroom is real but is NOT these two things**, and the next
/// attempt should profile rather than assume — the remaining suspects are the serial 6-bit sub-scale
/// unpack and the `KV` codebook lookup, neither of which Q8_0 pays at all.
///
/// What it did buy, measured on real checkpoint weights against `ferric_gguf`'s CPU dequant: **max
/// relative error 3.586e-7 → 1.550e-7**, because (2) reassociates to `dl*(Σ x·kv)` and removes a
/// rounding step per element. Not bit-identical to the previous kernel, and more accurate than it.
/// `qsb` is retained because IQ4_NL still uses it.
const IQ4_XS_BODY: &str = r#"
            let cb = bi * 32u; let ab = bi * 3u;
            let d = unpack2x16float(aux[ab]).x;
            let scales_h = aux[ab + 1u];
            let scales_l = aux[ab + 2u];
            let xbb = r * in_dim + blk * 256u;
            for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
                // 6-bit sub-scale: low nibble from scales_l[ib/2], high 2 bits from scales_h at 2*ib.
                let sl = (scales_l >> (8u * (ib >> 1u))) & 0xffu;
                let lo = (sl >> (4u * (ib & 1u))) & 0x0Fu;
                let hi = (scales_h >> (2u * ib)) & 3u;
                let dl = d * f32(i32(lo | (hi << 4u)) - 32);
                let xh = xbb + ib * 32u;
                let w0 = cb + ib * 4u;   // the 4 u32 words holding this sub-block's 16 packed bytes
                var sub = 0.0;
                for (var w: u32 = 0u; w < 4u; w = w + 1u) {
                    let word = codes[w0 + w];   // one load, four bytes used
                    for (var t: u32 = 0u; t < 4u; t = t + 1u) {
                        let b = (word >> (8u * t)) & 0xffu;
                        let j = w * 4u + t;
                        sub = sub + x[xh + j]       * f32(KV[b & 0x0Fu]);
                        sub = sub + x[xh + j + 16u] * f32(KV[b >> 4u]);
                    }
                }
                acc = acc + dl * sub;
            }
"#;

/// Packed **IQ4_XS** weights: `[out, in]`, `in` a multiple of 256.
///
/// Replaces the `Dense` dequant-on-load fallback, which held the whole weight as f32 on the GPU. That
/// fallback is correct and was an 8x memory blow-up over the 4.25 bits/weight this format actually
/// costs, which is the entire reason it was worth writing a kernel for.
pub struct Iq4XsWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 32 u32/block: qs[128]
    aux: Arc<wgpu::Buffer>,   // 3 u32/block: [d|_, scales_h, scales_l packed]
    pub rows: usize,
    pub cols: usize,
}

impl Iq4XsWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Iq4XsWeights {
        assert_eq!(cols % 256, 0, "IQ4_XS cols must be a multiple of 256");
        assert_eq!(bytes.len(), rows * (cols / 256) * 136, "unexpected IQ4_XS byte length");
        let nblk = rows * (cols / 256);
        let mut codes: Vec<u32> = vec![0; nblk * 32];
        let mut aux: Vec<u32> = vec![0; nblk * 3];
        let word = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        for b in 0..nblk {
            let src = &bytes[b * 136..b * 136 + 136];
            aux[b * 3] = u16::from_le_bytes([src[0], src[1]]) as u32;      // f16 d
            aux[b * 3 + 1] = u16::from_le_bytes([src[2], src[3]]) as u32;  // scales_h
            aux[b * 3 + 2] = word(src, 4);                                 // scales_l[4]
            for w in 0..32 { codes[b * 32 + w] = word(src, 8 + w * 4); }   // qs[128]
        }
        let mk = |label: &str, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Iq4XsWeights { ctx: ctx.clone(), codes: mk("iq4xs.codes", &codes), aux: mk("iq4xs.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 136 }
}

impl Tensor {
    /// `y = x·Wᵀ` where `W` is packed **IQ4_XS**, dequantised per sub-block in-kernel.
    pub fn matmul_iq4_xs(&self, w: &Iq4XsWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q6_K_SPLITK_WGSL)
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q6_K_FLAT_WGSL)
        };
        let src = wgsl.replace("__HELPERS__", IQ4_XS_HELPERS).replace("__BODY__", IQ4_XS_BODY);
        run(&self.ctx, &src, "matmul_iq4_xs",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

/// The 32-value block body. Same codebook and same nibble order as IQ4_XS, minus the 6-bit sub-scales:
/// IQ4_NL has one `d` per 32 values, so there is nothing between `d` and the codebook lookup.
const IQ4_NL_BODY: &str = r#"
            let cb = bi * 4u;
            let d = unpack2x16float(aux[bi]).x;
            let xbb = r * in_dim + blk * 32u;
            for (var j: u32 = 0u; j < 16u; j = j + 1u) {
                let b = qsb(cb, j);
                acc = acc + x[xbb + j]       * d * f32(KV[b & 0x0Fu]);
                acc = acc + x[xbb + j + 16u] * d * f32(KV[b >> 4u]);
            }
"#;

/// Packed **IQ4_NL** weights: `[out, in]`, `in` a multiple of 32.
///
/// Worth writing for a reason that only shows up on a real checkpoint. A file distributed as "IQ4_XS"
/// is mostly *not* IQ4_XS: measured on `bartowski/Qwen2.5-0.5B-Instruct-IQ4_XS.gguf`, IQ4_XS covers 24
/// tensors / 104.6 M params (only the rows whose length divides 256), while **IQ4_NL covers 120
/// tensors / 250.5 M params** — the majority of the model. Wiring IQ4_XS alone leaves 51% of the
/// weights on the f32 dense fallback, so the format's byte cost is never actually paid at runtime.
pub struct Iq4NlWeights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 4 u32/block: qs[16]
    aux: Arc<wgpu::Buffer>,   // 1 u32/block: f16 d in the low half
    pub rows: usize,
    pub cols: usize,
}

impl Iq4NlWeights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Iq4NlWeights {
        assert_eq!(cols % 32, 0, "IQ4_NL cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 18, "unexpected IQ4_NL byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut aux: Vec<u32> = vec![0; nblk];
        for b in 0..nblk {
            let src = &bytes[b * 18..b * 18 + 18];
            aux[b] = u16::from_le_bytes([src[0], src[1]]) as u32; // f16 d
            for w in 0..4 {
                codes[b * 4 + w] = u32::from_le_bytes([src[2 + w * 4], src[3 + w * 4], src[4 + w * 4], src[5 + w * 4]]);
            }
        }
        let mk = |label: &str, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Iq4NlWeights { ctx: ctx.clone(), codes: mk("iq4nl.codes", &codes), aux: mk("iq4nl.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 18 }
}

impl Tensor {
    /// `y = x·Wᵀ` where `W` is packed **IQ4_NL**, dequantised per 32-value block in-kernel.
    pub fn matmul_iq4_nl(&self, w: &Iq4NlWeights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q6_K_SPLITK_WGSL)
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q6_K_FLAT_WGSL)
        };
        // The shared template walks 256-value super-blocks; IQ4_NL's are 32. Anchored on the whole
        // expression, not the bare literal `256u`: the file has 17 of those and this must hit exactly
        // the one that sets the block stride. Asserted rather than trusted, because a future edit to
        // the template would otherwise silently produce a kernel that reads 1/8th of each weight.
        debug_assert_eq!(wgsl.matches("in_dim / 256u").count(), 1,
            "IQ4_NL rewrites the template's block stride and expects exactly one site");
        let src = wgsl.replace("in_dim / 256u", "in_dim / 32u")
            .replace("__HELPERS__", IQ4_XS_HELPERS).replace("__BODY__", IQ4_NL_BODY);
        run(&self.ctx, &src, "matmul_iq4_nl",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

/// **MXFP4** in-kernel dequant helpers — the E2M1 value table and the E8M0 scale, in the *ggml*
/// arithmetic rather than the obvious one.
///
/// `KM` is the OCP E2M1 table stored **doubled** (`{0,1,2,3,4,6,8,12}` = 2x `{0,.5,1,1.5,2,3,4,6}`),
/// so the shared scale it pairs with is `2^(e−128)`, not `2^(e−127)`. That halving is ggml's own
/// arithmetic and it is the one thing a from-the-spec implementation gets wrong: `2^(e−127)` at
/// `e = 255` is `2^128`, **not representable in f32**, so forming the scale first sends every element
/// of such a block to `±inf` where ggml returns finite values for the small codes
/// (`e=255, code=1 → 0x7f000000 = 2^127`). `2^(e−128)` tops out at the representable `2^127`.
///
/// **`e8m0h` returns the scale SPLIT IN TWO, and that is a GPU-specific requirement the scalar path
/// does not have.** `ferric_gguf::e8m0_half_to_f32` hands back a single f32 and builds the two bottom
/// exponents (`2^−128`, `2^−127`) as subnormal bit patterns, which is exact on the CPU. Transliterated
/// literally it loses data on the fabric: **this device flushes f32 subnormals to zero** (probed
/// independently with an elementwise multiply, see `device_keeps_subnormals`), so a subnormal *scale*
/// is zeroed the instant it is read and takes the whole block's weights with it — **56 of the 4096
/// (scale, code) pairs came back zero, of which only 16 had answers that were subnormal at all**. The
/// other 40 were ordinary normal-range f32 values destroyed by a subnormal intermediate.
///
/// So the scale is returned as `2^(a−64) · 2^(b−64)` with `a = e>>1`, `b = e−a`. Both exponents land
/// in `[−64, 64]`, so **neither factor is ever subnormal and neither ever overflows**, for every one
/// of the 256 scale bytes. `(KM · d.x) · d.y` then reproduces ggml bit for bit: the first product
/// peaks at `12·2^63 ≈ 2^66.6` (finite), and only the second can saturate — which is exactly where
/// ggml saturates too. What remains after this is irreducible: 16 pairs whose *answer* is genuinely
/// below `f32::MIN_POSITIVE`, which no arrangement of multiplies can make a flush-to-zero device
/// represent. Those are values under `1.2e−38`; the tests pin them as flushed rather than ignore them.
///
/// The table is integral, so `i32` holds it exactly and `f32(...)` is lossless — same shape as
/// `IQ4_XS_HELPERS`. `mxsc` unpacks the scale byte of block `bi` from the 4-blocks-per-word `aux`,
/// which is what keeps the resident footprint at the format's own 17 bytes per block instead of 20.
const MXFP4_HELPERS: &str = r#"
const KM: array<i32, 16> = array<i32, 16>(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
fn qsb(cb: u32, i: u32) -> u32 { return (codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu; }
fn mxsc(bi: u32) -> u32 { return (aux[bi >> 2u] >> (8u * (bi & 3u))) & 0xffu; }
fn e8m0h(e: u32) -> vec2<f32> {
    let a = e >> 1u;
    let b = e - a;
    return vec2<f32>(bitcast<f32>((a + 63u) << 23u), bitcast<f32>((b + 63u) << 23u));
}
"#;

/// The 32-value MXFP4 block body. Same low-half/high-half nibble split as Q4_0 and IQ4_NL: element
/// `j` is the LOW nibble of `qs[j]` and element `j+16` the HIGH nibble.
///
/// The weight is formed **before** it meets the activation — `x · (KM[c] · d)`, not `x · d · KM[c]`.
/// That is deliberate and it is the order the f32 dense fallback uses: the dense path materialises
/// `w = KM[c]·d` and then multiplies, so any other association would disagree with it exactly where
/// the product saturates (a block at `e = 255` whose weight is `inf` must contribute `inf`, whatever
/// the activation is, rather than a finite `x·d·KM`).
const MXFP4_BODY: &str = r#"
            let cb = bi * 4u;
            let d = e8m0h(mxsc(bi));
            let xbb = r * in_dim + blk * 32u;
            for (var j: u32 = 0u; j < 16u; j = j + 1u) {
                let b = qsb(cb, j);
                acc = acc + x[xbb + j]       * ((f32(KM[b & 0x0Fu]) * d.x) * d.y);
                acc = acc + x[xbb + j + 16u] * ((f32(KM[b >> 4u]) * d.x) * d.y);
            }
"#;

/// Packed **MXFP4** (OCP Microscaling FP4, ggml type 39) weights: `[out, in]`, `in` a multiple of 32.
///
/// This is the format GPT-OSS ships in, and the *only* reason it exists is that the weights fit:
/// 17 bytes per 32 values = 0.53125 bytes/element = 4.25 bpw. Running it through the f32 dense
/// fallback — which is what happened until this type had a kernel — costs 4 bytes/element, **7.53x**
/// the format's own footprint, which gives up precisely the property the format was chosen for.
///
/// The GPU layout keeps that 7.53x rather than most of it. `codes` is the 16 `qs` bytes as 4 u32
/// (16 B/block); `aux` packs the single E8M0 scale byte **four blocks to a word** (1 B/block), so the
/// resident total is 17 bytes per block — bit-for-bit the on-disk size, with no padding. The obvious
/// one-word-per-scale layout that IQ4_NL/Q4_0 use would cost 20 B/block (0.625 B/elem); it is only
/// their `f16` scale that makes a whole word the natural unit, and MXFP4's scale is one byte.
pub struct Mxfp4Weights {
    ctx: Arc<Context>,
    codes: Arc<wgpu::Buffer>, // 4 u32/block: qs[16]
    aux: Arc<wgpu::Buffer>,   // 1 BYTE/block: the E8M0 scale, 4 blocks packed per u32
    pub rows: usize,
    pub cols: usize,
}

impl Mxfp4Weights {
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], rows: usize, cols: usize) -> Mxfp4Weights {
        assert_eq!(cols % 32, 0, "MXFP4 cols must be a multiple of 32");
        assert_eq!(bytes.len(), rows * (cols / 32) * 17, "unexpected MXFP4 byte length");
        let nblk = rows * (cols / 32);
        let mut codes: Vec<u32> = vec![0; nblk * 4];
        let mut aux: Vec<u32> = vec![0; nblk.div_ceil(4)];
        for b in 0..nblk {
            let src = &bytes[b * 17..b * 17 + 17]; // e, qs[16]
            aux[b >> 2] |= (src[0] as u32) << (8 * (b & 3));
            for w in 0..4 {
                codes[b * 4 + w] = u32::from_le_bytes([src[1 + w * 4], src[2 + w * 4], src[3 + w * 4], src[4 + w * 4]]);
            }
        }
        let mk = |label: &str, data: &[u32]| Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        Mxfp4Weights { ctx: ctx.clone(), codes: mk("mxfp4.codes", &codes), aux: mk("mxfp4.aux", &aux), rows, cols }
    }
    /// On-disk block size, which here is also the exact resident size (see the struct docs).
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 32) * 17 }
    /// The bytes actually allocated on the GPU, read back from the buffers rather than computed from
    /// the format — so a layout change cannot leave the memory claim describing the old one.
    pub fn gpu_bytes(&self) -> usize { (self.codes.size() + self.aux.size()) as usize }
}

impl Tensor {
    /// `y = x·Wᵀ` where `W` is packed **MXFP4**, dequantised per 32-value block in-kernel.
    pub fn matmul_mxfp4(&self, w: &Mxfp4Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        let (grid, rs, wgsl) = if q2_0_split_k(rows, w.rows) {
            let gw = n.min(32768);
            (((gw as u32), n.div_ceil(gw) as u32, 1u32), gw as u32, MATMUL_Q6_K_SPLITK_WGSL)
        } else {
            let wg = n.div_ceil(64); let gw = wg.min(32768);
            (((gw as u32), wg.div_ceil(gw) as u32, 1u32), (gw * 64) as u32, MATMUL_Q6_K_FLAT_WGSL)
        };
        // Same template-stride rewrite as IQ4_NL: the shared body walks 256-value super-blocks and
        // MXFP4's are 32. Anchored on the whole expression, and asserted rather than trusted — a
        // template edit that moved this would otherwise silently read 1/8th of every weight.
        debug_assert_eq!(wgsl.matches("in_dim / 256u").count(), 1,
            "MXFP4 rewrites the template's block stride and expects exactly one site");
        let src = wgsl.replace("in_dim / 256u", "in_dim / 32u")
            .replace("__HELPERS__", MXFP4_HELPERS).replace("__BODY__", MXFP4_BODY);
        run(&self.ctx, &src, "matmul_mxfp4",
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

const MATMUL_Q6_K_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim; let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o * nblk + blk;
__BODY__
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q6_K_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        aux:    array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;
var<workgroup> partial: array<f32, 64>;
__HELPERS__
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim; let nblk = in_dim / 256u;
        var acc = 0.0;
        for (var blk: u32 = t; blk < nblk; blk = blk + 64u) {
            let bi = o * nblk + blk;
__BODY__
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Q8_0 block = 32 int8 (8 u32 words) + f16 scale; value = int8·d. Per word, sign-extend the 4 bytes
// (shift a byte to the top and arithmetic-shift back) into a vec4 and dot with 4 activations.
const MATMUL_Q8_0_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 32u; let nwords = nblk * 8u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 3u; let bi = o * nblk + blk;
        let sw = unpack2x16float(scales[bi >> 1u]);
        let d = select(sw.y, sw.x, (bi & 1u) == 0u);
        let word = codes[o * nwords + w];
        let xi = (r * in_dim + blk * 32u + (w & 7u) * 4u) >> 2u;
        let v = vec4<f32>(f32(i32(word << 24u) >> 24u), f32(i32(word << 16u) >> 24u), f32(i32(word << 8u) >> 24u), f32(i32(word) >> 24u));
        acc = acc + d * dot(x[xi], v);
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q8_0_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 32u; let nwords = nblk * 8u;
        var acc = 0.0;
        for (var w: u32 = t; w < nwords; w = w + 64u) {
            let blk = w >> 3u; let bi = o * nblk + blk;
            let sw = unpack2x16float(scales[bi >> 1u]);
            let d = select(sw.y, sw.x, (bi & 1u) == 0u);
            let word = codes[o * nwords + w];
            let xi = (r * in_dim + blk * 32u + (w & 7u) * 4u) >> 2u;
            let v = vec4<f32>(f32(i32(word << 24u) >> 24u), f32(i32(word << 16u) >> 24u), f32(i32(word << 8u) >> 24u), f32(i32(word) >> 24u));
            acc = acc + d * dot(x[xi], v);
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Q4_0 block = 32 values, 4 u32 code-words + f16 scale. Byte j's low nibble is value j, high nibble
// is value j+16 (llama.cpp layout); value = (nibble − 8)·d. Per word (4 bytes) that's 4 low + 4 high
// activations, two vec4 dots. x is bound as vec4<f32> for coalesced 4-at-a-time activation loads.
const MATMUL_Q4_0_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 2u;
        let bi = o * nblk + blk;
        let sw = unpack2x16float(scales[bi >> 1u]);
        let d = select(sw.y, sw.x, (bi & 1u) == 0u);
        let word = codes[o * nwords + w];
        let xlo = (r * in_dim + blk * 32u + (w & 3u) * 4u) >> 2u;
        let lo = vec4<f32>(f32(i32(word & 0xfu) - 8), f32(i32((word >> 8u) & 0xfu) - 8), f32(i32((word >> 16u) & 0xfu) - 8), f32(i32((word >> 24u) & 0xfu) - 8));
        let hi = vec4<f32>(f32(i32((word >> 4u) & 0xfu) - 8), f32(i32((word >> 12u) & 0xfu) - 8), f32(i32((word >> 20u) & 0xfu) - 8), f32(i32((word >> 28u) & 0xfu) - 8));
        acc = acc + (dot(x[xlo], lo) + dot(x[xlo + 4u], hi)) * d;
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q4_0_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 32u; let nwords = nblk * 4u;
        var acc = 0.0;
        for (var w: u32 = t; w < nwords; w = w + 64u) {
            let blk = w >> 2u;
            let bi = o * nblk + blk;
            let sw = unpack2x16float(scales[bi >> 1u]);
            let d = select(sw.y, sw.x, (bi & 1u) == 0u);
            let word = codes[o * nwords + w];
            let xlo = (r * in_dim + blk * 32u + (w & 3u) * 4u) >> 2u;
            let lo = vec4<f32>(f32(i32(word & 0xfu) - 8), f32(i32((word >> 8u) & 0xfu) - 8), f32(i32((word >> 16u) & 0xfu) - 8), f32(i32((word >> 24u) & 0xfu) - 8));
            let hi = vec4<f32>(f32(i32((word >> 4u) & 0xfu) - 8), f32(i32((word >> 12u) & 0xfu) - 8), f32(i32((word >> 20u) & 0xfu) - 8), f32(i32((word >> 28u) & 0xfu) - 8));
            acc = acc + (dot(x[xlo], lo) + dot(x[xlo + 4u], hi)) * d;
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Q4_1 block = 32 values, 4 u32 code-words + an (f16 d, f16 m) pair packed one u32/block. Same
// nibble layout as Q4_0 (byte j low→value j, high→value j+16) but the affine dequant value =
// nibble·d + m (no −8), so we build the actual weight vec4 (nibbles·d + m) and dot with x.
const MATMUL_Q4_1_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;   // (d, m) f16 pair per block
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 2u;
        let bi = o * nblk + blk;
        let dm = unpack2x16float(scales[bi]);
        let d = dm.x; let m = dm.y;
        let word = codes[o * nwords + w];
        let xlo = (r * in_dim + blk * 32u + (w & 3u) * 4u) >> 2u;
        let lo = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) * d + m;
        let hi = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) * d + m;
        acc = acc + dot(x[xlo], lo) + dot(x[xlo + 4u], hi);
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q4_1_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 32u; let nwords = nblk * 4u;
        var acc = 0.0;
        for (var w: u32 = t; w < nwords; w = w + 64u) {
            let blk = w >> 2u;
            let bi = o * nblk + blk;
            let dm = unpack2x16float(scales[bi]);
            let d = dm.x; let m = dm.y;
            let word = codes[o * nwords + w];
            let xlo = (r * in_dim + blk * 32u + (w & 3u) * 4u) >> 2u;
            let lo = vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) * d + m;
            let hi = vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) * d + m;
            acc = acc + dot(x[xlo], lo) + dot(x[xlo + 4u], hi);
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Q5_0 block = 32 values, 4 u32 code-words + [qh (u32), d (f16)] per block. Same nibble layout as
// Q4_0, but each value gets a 5th (high) bit from qh — bit i for value i, bit i+16 for value i+16 —
// and the offset is −16: value = ((nibble | (bit<<4)) − 16)·d.


const MATMUL_Q5_0_ID_WSUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;  // [T, k, in]
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;        // [T, out_pe]
@group(0) @binding(4) var<storage,read>        selw:   array<f32>;
@group(0) @binding(5) var<storage,read>        info:   array<u32>;        // k, out_pe, in, gw, tot
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = info[0]; let out_pe = info[1]; let in_dim = info[2]; let gw = info[3]; let tot = info[4];
    let idx = gid.x + gid.y * gw;
    if (idx >= tot) { return; }
    let o = idx % out_pe;
    let t = idx / out_pe;
    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    var acc = 0.0;
    for (var sl: u32 = 0u; sl < k; sl = sl + 1u) {
        let wgt = selw[t * 2u * k + sl];
        let e   = u32(selw[t * 2u * k + k + sl]);
        let o_row = e * out_pe + o;
        let xrow = (t * k + sl) * in_dim;
        var dp = 0.0;
        for (var w: u32 = 0u; w < nwords; w = w + 1u) {
            let blk = w >> 2u;
            let bi = o_row * nblk + blk;
            let qh = scales[bi * 2u];
            let d = unpack2x16float(scales[bi * 2u + 1u]).x;
            let word = codes[o_row * nwords + w];
            let base = (w & 3u) * 4u;
            let xlo = (xrow + blk * 32u + base) >> 2u;
            let lo = vec4<f32>(
                f32(i32(( word         & 0xfu) | (((qh >> (base + 0u))  & 1u) << 4u)) - 16),
                f32(i32(((word >> 8u)  & 0xfu) | (((qh >> (base + 1u))  & 1u) << 4u)) - 16),
                f32(i32(((word >> 16u) & 0xfu) | (((qh >> (base + 2u))  & 1u) << 4u)) - 16),
                f32(i32(((word >> 24u) & 0xfu) | (((qh >> (base + 3u))  & 1u) << 4u)) - 16));
            let hi = vec4<f32>(
                f32(i32(((word >> 4u)  & 0xfu) | (((qh >> (base + 16u)) & 1u) << 4u)) - 16),
                f32(i32(((word >> 12u) & 0xfu) | (((qh >> (base + 17u)) & 1u) << 4u)) - 16),
                f32(i32(((word >> 20u) & 0xfu) | (((qh >> (base + 18u)) & 1u) << 4u)) - 16),
                f32(i32(((word >> 28u) & 0xfu) | (((qh >> (base + 19u)) & 1u) << 4u)) - 16));
            dp = dp + (dot(x[xlo], lo) + dot(x[xlo + 4u], hi)) * d;
        }
        acc = acc + wgt * dp;
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q5_0_RELU2_ID_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;   // [qh, d] per block
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<storage,read>        selw:   array<f32>;   // [T, w_0..w_{k-1} | id_0..id_{k-1}]
@group(0) @binding(5) var<storage,read>        info:   array<u32>;   // k, eff, in, gw, tot
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = info[0]; let eff = info[1]; let in_dim = info[2]; let gw = info[3]; let tot = info[4];
    let idx = gid.x + gid.y * gw;
    if (idx >= tot) { return; }
    let o  = idx % eff;
    let ts = idx / eff;          // t*k + s
    let s  = ts % k;
    let t  = ts / k;
    // Expert ids live in the SECOND half of each moe_topk row.
    let e = u32(selw[t * 2u * k + k + s]);
    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    let o_row = e * eff + o;     // the slab is [n_expert, eff, in] flattened over rows
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 2u;
        let bi = o_row * nblk + blk;
        let qh = scales[bi * 2u];
        let d = unpack2x16float(scales[bi * 2u + 1u]).x;
        let word = codes[o_row * nwords + w];
        let base = (w & 3u) * 4u;
        let xlo = (t * in_dim + blk * 32u + base) >> 2u;
        let lo = vec4<f32>(
            f32(i32(( word         & 0xfu) | (((qh >> (base + 0u))  & 1u) << 4u)) - 16),
            f32(i32(((word >> 8u)  & 0xfu) | (((qh >> (base + 1u))  & 1u) << 4u)) - 16),
            f32(i32(((word >> 16u) & 0xfu) | (((qh >> (base + 2u))  & 1u) << 4u)) - 16),
            f32(i32(((word >> 24u) & 0xfu) | (((qh >> (base + 3u))  & 1u) << 4u)) - 16));
        let hi = vec4<f32>(
            f32(i32(((word >> 4u)  & 0xfu) | (((qh >> (base + 16u)) & 1u) << 4u)) - 16),
            f32(i32(((word >> 12u) & 0xfu) | (((qh >> (base + 17u)) & 1u) << 4u)) - 16),
            f32(i32(((word >> 20u) & 0xfu) | (((qh >> (base + 18u)) & 1u) << 4u)) - 16),
            f32(i32(((word >> 28u) & 0xfu) | (((qh >> (base + 19u)) & 1u) << 4u)) - 16));
        acc = acc + (dot(x[xlo], lo) + dot(x[xlo + 4u], hi)) * d;
    }
    let a = max(acc, 0.0);
    out[idx] = a * a;            // ReLU squared
}
"#;

const MATMUL_Q5_0_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;   // [qh, d] per block (2 u32)
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 2u;
        let bi = o * nblk + blk;
        let qh = scales[bi * 2u];
        let d = unpack2x16float(scales[bi * 2u + 1u]).x;
        let word = codes[o * nwords + w];
        let base = (w & 3u) * 4u;
        let xlo = (r * in_dim + blk * 32u + base) >> 2u;
        let lo = vec4<f32>(
            f32(i32(( word         & 0xfu) | (((qh >> (base + 0u))  & 1u) << 4u)) - 16),
            f32(i32(((word >> 8u)  & 0xfu) | (((qh >> (base + 1u))  & 1u) << 4u)) - 16),
            f32(i32(((word >> 16u) & 0xfu) | (((qh >> (base + 2u))  & 1u) << 4u)) - 16),
            f32(i32(((word >> 24u) & 0xfu) | (((qh >> (base + 3u))  & 1u) << 4u)) - 16));
        let hi = vec4<f32>(
            f32(i32(((word >> 4u)  & 0xfu) | (((qh >> (base + 16u)) & 1u) << 4u)) - 16),
            f32(i32(((word >> 12u) & 0xfu) | (((qh >> (base + 17u)) & 1u) << 4u)) - 16),
            f32(i32(((word >> 20u) & 0xfu) | (((qh >> (base + 18u)) & 1u) << 4u)) - 16),
            f32(i32(((word >> 28u) & 0xfu) | (((qh >> (base + 19u)) & 1u) << 4u)) - 16));
        acc = acc + (dot(x[xlo], lo) + dot(x[xlo + 4u], hi)) * d;
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q5_0_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 32u; let nwords = nblk * 4u;
        var acc = 0.0;
        for (var w: u32 = t; w < nwords; w = w + 64u) {
            let blk = w >> 2u;
            let bi = o * nblk + blk;
            let qh = scales[bi * 2u];
            let d = unpack2x16float(scales[bi * 2u + 1u]).x;
            let word = codes[o * nwords + w];
            let base = (w & 3u) * 4u;
            let xlo = (r * in_dim + blk * 32u + base) >> 2u;
            let lo = vec4<f32>(
                f32(i32(( word         & 0xfu) | (((qh >> (base + 0u))  & 1u) << 4u)) - 16),
                f32(i32(((word >> 8u)  & 0xfu) | (((qh >> (base + 1u))  & 1u) << 4u)) - 16),
                f32(i32(((word >> 16u) & 0xfu) | (((qh >> (base + 2u))  & 1u) << 4u)) - 16),
                f32(i32(((word >> 24u) & 0xfu) | (((qh >> (base + 3u))  & 1u) << 4u)) - 16));
            let hi = vec4<f32>(
                f32(i32(((word >> 4u)  & 0xfu) | (((qh >> (base + 16u)) & 1u) << 4u)) - 16),
                f32(i32(((word >> 12u) & 0xfu) | (((qh >> (base + 17u)) & 1u) << 4u)) - 16),
                f32(i32(((word >> 20u) & 0xfu) | (((qh >> (base + 18u)) & 1u) << 4u)) - 16),
                f32(i32(((word >> 28u) & 0xfu) | (((qh >> (base + 19u)) & 1u) << 4u)) - 16));
            acc = acc + (dot(x[xlo], lo) + dot(x[xlo + 4u], hi)) * d;
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

// Q5_1 block = 32 values, 4 u32 code-words + [pack(d,m), qh] per block (2 u32). Affine 5-bit: the
// 5th bit comes from qh (bit i / i+16), and value = (nibble | (bit<<4))·d + m (no −16 offset).
const MATMUL_Q5_1_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;   // [pack(d,m), qh] per block (2 u32)
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    var acc = 0.0;
    for (var w: u32 = 0u; w < nwords; w = w + 1u) {
        let blk = w >> 2u;
        let bi = o * nblk + blk;
        let dm = unpack2x16float(scales[bi * 2u]);
        let d = dm.x; let m = dm.y;
        let qh = scales[bi * 2u + 1u];
        let word = codes[o * nwords + w];
        let base = (w & 3u) * 4u;
        let xlo = (r * in_dim + blk * 32u + base) >> 2u;
        let lo = vec4<f32>(
            f32(( word         & 0xfu) | (((qh >> (base + 0u))  & 1u) << 4u)),
            f32(((word >> 8u)  & 0xfu) | (((qh >> (base + 1u))  & 1u) << 4u)),
            f32(((word >> 16u) & 0xfu) | (((qh >> (base + 2u))  & 1u) << 4u)),
            f32(((word >> 24u) & 0xfu) | (((qh >> (base + 3u))  & 1u) << 4u))) * d + m;
        let hi = vec4<f32>(
            f32(((word >> 4u)  & 0xfu) | (((qh >> (base + 16u)) & 1u) << 4u)),
            f32(((word >> 12u) & 0xfu) | (((qh >> (base + 17u)) & 1u) << 4u)),
            f32(((word >> 20u) & 0xfu) | (((qh >> (base + 18u)) & 1u) << 4u)),
            f32(((word >> 28u) & 0xfu) | (((qh >> (base + 19u)) & 1u) << 4u))) * d + m;
        acc = acc + dot(x[xlo], lo) + dot(x[xlo + 4u], hi);
    }
    out[idx] = acc;
}
"#;

const MATMUL_Q5_1_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let idx = wg.x + wg.y * info.w; let t = lid.x;
    if (idx < rows * o_dim) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 32u; let nwords = nblk * 4u;
        var acc = 0.0;
        for (var w: u32 = t; w < nwords; w = w + 64u) {
            let blk = w >> 2u;
            let bi = o * nblk + blk;
            let dm = unpack2x16float(scales[bi * 2u]);
            let d = dm.x; let m = dm.y;
            let qh = scales[bi * 2u + 1u];
            let word = codes[o * nwords + w];
            let base = (w & 3u) * 4u;
            let xlo = (r * in_dim + blk * 32u + base) >> 2u;
            let lo = vec4<f32>(
                f32(( word         & 0xfu) | (((qh >> (base + 0u))  & 1u) << 4u)),
                f32(((word >> 8u)  & 0xfu) | (((qh >> (base + 1u))  & 1u) << 4u)),
                f32(((word >> 16u) & 0xfu) | (((qh >> (base + 2u))  & 1u) << 4u)),
                f32(((word >> 24u) & 0xfu) | (((qh >> (base + 3u))  & 1u) << 4u))) * d + m;
            let hi = vec4<f32>(
                f32(((word >> 4u)  & 0xfu) | (((qh >> (base + 16u)) & 1u) << 4u)),
                f32(((word >> 12u) & 0xfu) | (((qh >> (base + 17u)) & 1u) << 4u)),
                f32(((word >> 20u) & 0xfu) | (((qh >> (base + 18u)) & 1u) << 4u)),
                f32(((word >> 28u) & 0xfu) | (((qh >> (base + 19u)) & 1u) << 4u))) * d + m;
            acc = acc + dot(x[xlo], lo) + dot(x[xlo + 4u], hi);
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

const MATMUL_Q2_0_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;  // rows, out, in, threads_per_grid_row
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 128u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let bi = o * nblk + blk;
        let sw = unpack2x16float(scales[bi >> 1u]);
        let d = select(sw.y, sw.x, (bi & 1u) == 0u);
        let cbase = bi * 8u;
        let xq = (r * in_dim + blk * 128u) >> 2u;   // vec4 index of this 128-group
        var bacc = 0.0;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word = codes[cbase + w];            // 16 codes
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let s = 8u * q;                     // codes 4q..4q+3 sit at bit offsets 8q..8q+6
                let cv = vec4<f32>(
                    f32(i32((word >> s) & 3u) - 1),
                    f32(i32((word >> (s + 2u)) & 3u) - 1),
                    f32(i32((word >> (s + 4u)) & 3u) - 1),
                    f32(i32((word >> (s + 6u)) & 3u) - 1));
                bacc = bacc + dot(x[xq + w * 4u + q], cv);   // w = (q−1)·d
            }
        }
        acc = acc + bacc * d;
    }
    out[idx] = acc;
}
"#;

// **Split-K**: one workgroup per output element, its 64 threads splitting the K reduction and
// combining through shared memory. The obvious one-thread-per-output shape leaves each thread
// walking a 5120-long dependent accumulate chain, and with only `rows·out` threads there isn't
// enough work in flight to hide memory latency — measurably so: 1-token and 5-token matmuls took
// the *same* wall time, which is the signature of a latency-bound kernel, not a bandwidth-bound
// one. Splitting K gives 64× the parallelism and shortens each chain by 64×, and it makes adjacent
// threads read adjacent code words instead of rows 1360 B apart.
const MATMUL_Q2_0_SPLITK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<vec4<f32>>; // [rows, in], 4/load
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;  // 8 u32/block, 16 codes per word
@group(0) @binding(2) var<storage,read>        scales: array<u32>;  // f16/block, 2 packed per u32
@group(0) @binding(3) var<storage,read_write>  out:    array<f32>;  // [rows, out]
@group(0) @binding(4) var<uniform>             info:   vec4<u32>;   // rows, out, in, grid_w

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    // 2D grid: rows·out exceeds the 65535 per-dimension workgroup cap at real shapes.
    let idx = wg.x + wg.y * info.w;
    let t = lid.x;
    let n = rows * o_dim;
    // Uniform across the workgroup (depends only on workgroup_id), so the barriers below stay in
    // uniform control flow.
    if (idx < n) {
        let o = idx % o_dim; let r = idx / o_dim;
        let nblk = in_dim / 128u;
        let nwords = nblk * 8u;
        let wbase = o * nwords;
        var acc = 0.0;
        // Stride over *words*, not blocks: thread t takes word t, t+64, … so adjacent threads read
        // adjacent u32s and a SIMD group sweeps one contiguous run. Striding by block instead puts
        // adjacent threads 32 B apart, scattering a 32-wide group across 32 cache lines and using
        // 4 bytes of each. Measured: gdn qkv @1 token 0.34 ms → 0.24 ms, attn q 0.41 → 0.28.
        // (A vec4<u32> variant — 64 codes per load — was tried and is *worse*: it cuts the work
        // units 4×, wrecking load balance, and Apple already coalesces consecutive u32 loads.)
        for (var w: u32 = t; w < nwords; w = w + 64u) {
            let blk = w >> 3u;
            let bi = o * nblk + blk;
            let sw = unpack2x16float(scales[bi >> 1u]);
            let d = select(sw.y, sw.x, (bi & 1u) == 0u);
            let word = codes[wbase + w];        // coalesced; one load feeds 16 weights
            // Read x four at a time and reduce with dot(): the scalar form issues 16 activation
            // loads per code word, which dominates the instruction stream and starves a
            // latency-bound kernel of issue slots.
            let xq = (r * in_dim + blk * 128u + (w & 7u) * 16u) >> 2u;
            var bacc = 0.0;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let s = 8u * q;                 // codes 4q..4q+3 sit at bit offsets 8q..8q+6
                let cv = vec4<f32>(
                    f32(i32((word >> s) & 3u) - 1),
                    f32(i32((word >> (s + 2u)) & 3u) - 1),
                    f32(i32((word >> (s + 4u)) & 3u) - 1),
                    f32(i32((word >> (s + 6u)) & 3u) - 1));
                bacc = bacc + dot(x[xq + q], cv);   // w = (q−1)·d
            }
            acc = acc + bacc * d;   // the scale is constant across the 128-group
        }
        partial[t] = acc;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) { if (t < s) { partial[t] = partial[t] + partial[t + s]; } workgroupBarrier(); }
        if (t == 0u) { out[idx] = partial[0]; }
    }
}
"#;

const MATMUL_QW_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;      // [rows, in]
@group(0) @binding(1) var<storage,read>        qw: array<u32>;     // packed per-row int, [out, in]
@group(0) @binding(2) var<storage,read>        scale: array<f32>;  // [out]
@group(0) @binding(3) var<storage,read_write>  out: array<f32>;    // [rows, out]
@group(0) @binding(4) var<uniform>             info: vec4<u32>;    // rows, out, in, bits
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let rows = info.x; let o_dim = info.y; let in_dim = info.z; let bits = info.w;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let per = 32u / bits; let mask = (1u << bits) - 1u; let signbit = 1u << (bits - 1u);
    var acc = 0.0;
    for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
        let widx = o * in_dim + i;                       // element in W's flat [out,in]
        var q = i32((qw[widx / per] >> (bits * (widx % per))) & mask);
        if (q >= i32(signbit)) { q = q - i32(1u << bits); }
        acc = acc + x[r * in_dim + i] * f32(q);          // weight dequantized on the fly
    }
    out[idx] = acc * scale[o];
}
"#;

const QUANT_ROW_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        inp: array<f32>;
@group(0) @binding(1) var<storage,read>        scale: array<f32>; // [rows]
@group(0) @binding(2) var<storage,read_write>  out: array<u32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>;  // rows, cols, bits, bitcast(qmax), row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = info[0]; let cols = info[1]; let bits = info[2]; let qmax = bitcast<f32>(info[3]);
    let w = gid.x + gid.y * info[4]; // 2D grid: words can exceed the 65535-workgroup 1D cap
    let per = 32u / bits; let n = rows * cols; let words = (n + per - 1u) / per;
    if (w >= words) { return; }
    let mask = (1u << bits) - 1u;
    var word: u32 = 0u;
    for (var lane: u32 = 0u; lane < per; lane = lane + 1u) {
        let idx = w * per + lane;
        if (idx < n) {
            var s = scale[idx / cols]; if (s == 0.0) { s = 1.0; }
            let q = i32(clamp(round(inp[idx] / s), -qmax, qmax));
            word = word | ((u32(q) & mask) << (bits * lane));
        }
    }
    out[w] = word;
}
"#;

const DEQUANT_ROW_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        inp: array<u32>;
@group(0) @binding(1) var<storage,read>        scale: array<f32>; // [rows]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>;  // rows, cols, bits
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let rows = info[0]; let cols = info[1]; let bits = info[2];
    let n = rows * cols; if (idx >= n) { return; }
    let per = 32u / bits; let mask = (1u << bits) - 1u; let signbit = 1u << (bits - 1u);
    let word = inp[idx / per]; let lane = idx % per;
    var v = i32((word >> (bits * lane)) & mask);
    if (v >= i32(signbit)) { v = v - i32(1u << bits); }
    out[idx] = f32(v) * scale[idx / cols];
}
"#;

const QUANT_I8_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        inp: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<u32>;   // 4x int8 per word
@group(0) @binding(2) var<storage,read>        info: array<u32>;  // n, bitcast(scale)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x; let n = info[0]; let words = (n + 3u) / 4u;
    if (w >= words) { return; }
    let s = bitcast<f32>(info[1]);
    var word: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let idx = 4u * w + lane;
        if (idx < n) {
            let q = i32(clamp(round(inp[idx] / s), -127.0, 127.0));
            word = word | ((u32(q) & 0xffu) << (8u * lane));
        }
    }
    out[w] = word;
}
"#;

const MATMUL_I8_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<u32>;  // packed [m,k]
@group(0) @binding(1) var<storage,read>        b: array<u32>;  // packed [k,n]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>; // m,k,n, bitcast(scaleA*scaleB)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let m = info[0]; let k = info[1]; let n = info[2];
    let sc = bitcast<f32>(info[3]);
    if (idx >= m * n) { return; }
    let j = idx % n; let i = idx / n;
    var acc: i32 = 0;
    for (var l: u32 = 0u; l < k; l = l + 1u) {
        let ai = i * k + l; let wa = a[ai >> 2u]; var av = i32((wa >> (8u * (ai & 3u))) & 0xffu); if (av > 127) { av = av - 256; }
        let bi = l * n + j; let wb = b[bi >> 2u]; var bv = i32((wb >> (8u * (bi & 3u))) & 0xffu); if (bv > 127) { bv = bv - 256; }
        acc = acc + av * bv;
    }
    out[idx] = f32(acc) * sc;
}
"#;

const DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        inp: array<u32>; // packed 2x16
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // n, kind(0=f16,1=bf16)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let n = info[0]; let kind = info[1];
    if (i >= n) { return; }
    let word = inp[i >> 1u]; let sel = i & 1u;
    if (kind == 0u) {
        let pair = unpack2x16float(word);      // two f16 → f32
        out[i] = select(pair.x, pair.y, sel == 1u);
    } else {
        let h = (word >> (16u * sel)) & 0xffffu;
        out[i] = bitcast<f32>(h << 16u);        // bf16 → f32
    }
}
"#;

const QUANTIZE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        inp: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<u32>; // packed 2x16
@group(0) @binding(2) var<storage,read>        info: array<u32>; // n, kind, row_stride
fn bf16_rne(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let r = b + 0x7fffu + ((b >> 16u) & 1u); // round-to-nearest-even bias
    return r >> 16u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D dispatch: a big weight (e.g. 5120² for coop16's f16 convert) exceeds the 65535 1D cap.
    let w = gid.x + gid.y * info[2]; let n = info[0]; let kind = info[1];
    let words = (n + 1u) / 2u;
    if (w >= words) { return; }
    let i0 = 2u * w; let i1 = i0 + 1u;
    let x0 = inp[i0];
    var x1 = 0.0;
    if (i1 < n) { x1 = inp[i1]; }
    if (kind == 0u) {
        out[w] = pack2x16float(vec2<f32>(x0, x1));
    } else {
        out[w] = bf16_rne(x0) | (bf16_rne(x1) << 16u);
    }
}
"#;

/// **Every format the loader can READ must be explicitly classified as packed or deliberately dense.**
///
/// This exists because of a bug that produced no error anywhere. `Iq4XsWeights` and `matmul_iq4_xs`
/// were written, tested and correct, and the model never called them: [`QMatrix::block_bytes`] did not
/// list ggml type 23, so `qm()` took the `from_dense` branch and every IQ4 weight was dequantised to
/// f32 on load. The model ran, 8x fatter, exactly as if the kernel had not been written. The kernel's
/// own test passed throughout, because it tested the kernel and not the path to it.
///
/// Three tables have to agree and nothing was asserting it: `ferric_gguf::type_size` (what we can
/// decode), [`QMatrix::block_bytes`] (what the loader's gate believes we can run packed), and
/// [`QShard::build`]'s match arms (what actually constructs a packed weight). IQ4_XS had the third
/// without the second. `packed_formats_do_not_land_on_the_dense_fallback` goes through
/// `QMatrix::from_bytes`, so it exercises the real path and covers all three at once.
///
/// **What it catches automatically:** a new format added to `ferric-gguf` that nobody wires up, and a
/// packed format silently dropped out of `block_bytes`.
///
/// **What it cannot catch, stated plainly:** a kernel written for a type already sitting in
/// `DENSE_BY_DESIGN` without moving its entry. Nothing in Rust can see that a `matmul_*` appeared. The
/// mitigation is that each entry below carries the reason it is dense, so writing the kernel means
/// coming here to falsify a comment.
#[cfg(test)]
mod format_reachability {
    use super::*;

    /// Types with a native packed kernel. Must be in `block_bytes`.
    const PACKED: &[(u32, &str)] = &[
        (2, "Q4_0"), (3, "Q4_1"), (6, "Q5_0"), (7, "Q5_1"), (8, "Q8_0"),
        (10, "Q2_K"), (11, "Q3_K"), (12, "Q4_K"), (13, "Q5_K"), (14, "Q6_K"),
        (20, "IQ4_NL"), (23, "IQ4_XS"),
        (39, "MXFP4"),
        (42, "Q2_0"),
        (43, "STQ1_0"),
        (16, "IQ2_XXS"),
        (18, "IQ3_XXS"),
    ];

    /// Types the loader decodes to f32 and runs dense, each with the reason. Must NOT be in
    /// `block_bytes` — an entry here that gains a kernel has to move to `PACKED`.
    const DENSE_BY_DESIGN: &[(u32, &str, &str)] = &[
        (0,  "F32",   "not a block quant; f32 already is the dense representation"),
        (1,  "F16",   "not a block quant; widening to f32 is the whole conversion"),
        (30, "BF16",  "not a block quant; widening to f32 is the whole conversion"),
        (35, "TQ2_0", "llama.cpp ternary: no packed kernel written (Q2_0/42 is the PrismML one that has one)"),
        (41, "Q1_0",  "PrismML 1-bit: no packed kernel written"),
    ];

    /// Probe `type_size` rather than importing a list, so a format added to ferric-gguf shows up here
    /// without anyone remembering to mirror it.
    fn readable_types() -> Vec<u32> {
        (0u32..=64).filter(|&t| ferric_gguf::type_size(t, 256).is_ok()).collect()
    }

    #[test]
    fn every_readable_format_is_classified() {
        let mut unclassified = Vec::new();
        for ty in readable_types() {
            let packed = PACKED.iter().any(|&(t, _)| t == ty);
            let dense = DENSE_BY_DESIGN.iter().any(|&(t, _, _)| t == ty);
            assert!(!(packed && dense), "ggml type {ty} is in BOTH tables");
            if !packed && !dense { unclassified.push(ty); }
        }
        assert!(unclassified.is_empty(),
            "ggml types {unclassified:?} can be decoded by ferric-gguf but are classified neither packed \
             nor deliberately dense. Add each to PACKED (and to QMatrix::block_bytes) if it has a kernel, \
             or to DENSE_BY_DESIGN with the reason. Leaving it unclassified is how IQ4_XS shipped with a \
             correct, tested, unreachable kernel.");
    }

    #[test]
    fn packed_formats_are_reachable_from_block_bytes() {
        for &(ty, name) in PACKED {
            assert!(QMatrix::block_bytes(ty).is_some(),
                "{name} (ggml type {ty}) has a packed kernel but block_bytes does not list it, so the \
                 loader will take the from_dense branch and the kernel will never run. This is the exact \
                 IQ4_XS bug.");
        }
        for &(ty, name, why) in DENSE_BY_DESIGN {
            assert!(QMatrix::block_bytes(ty).is_none(),
                "{name} (ggml type {ty}) is listed as dense-by-design ({why}) but block_bytes claims a \
                 packed kernel. One of the two is wrong.");
        }
    }

    /// The table check above compares two lists. This one walks the actual path a weight takes.
    #[test]
    fn packed_formats_do_not_land_on_the_dense_fallback() {
        // A silent skip here would be a green test that checked nothing, which is the whole failure
        // mode this module exists to prevent. Say so on the way past.
        let ctx = match pollster::block_on(ferric_core::Context::new()) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                eprintln!("SKIPPED packed_formats_do_not_land_on_the_dense_fallback: no GPU context \
                           ({e:?}). The table checks still ran; the real-path check did NOT.");
                return;
            }
        };
        let mut checked = 0usize;
        // 256 columns satisfies every block size in play (32, 128 and 256 all divide it).
        let (rows, cols) = (4usize, 256usize);
        for &(ty, name) in PACKED {
            let nbytes = ferric_gguf::type_size(ty, rows * cols)
                .unwrap_or_else(|e| panic!("{name}: type_size failed: {e}"));
            let bytes = vec![0u8; nbytes];
            // Reproduce the LOADER'S branch, not a direct call to the packed constructor. `qm()` in
            // ferric-llama gates on `block_bytes(ty).is_some()`, so calling `from_bytes` unconditionally
            // here would test `QShard::build`'s table while stepping over the gate that was actually
            // wrong. Verified by negative control: with type 23 removed from `block_bytes`, a direct
            // `from_bytes` still passed and only this form fails.
            let qm = if QMatrix::block_bytes(ty).is_some() {
                QMatrix::from_bytes(&ctx, &bytes, ty, rows, cols)
                    .unwrap_or_else(|e| panic!("{name}: from_bytes failed: {e}"))
            } else {
                let deq = ferric_gguf::deq_raw(&bytes, rows * cols, ty)
                    .unwrap_or_else(|e| panic!("{name}: deq_raw failed: {e}"));
                QMatrix::from_dense(&ctx, &deq, rows, cols)
            };
            for sh in &qm.shards {
                assert!(!matches!(sh, QShard::Dense(_)),
                    "{name} (ggml type {ty}) fell through to the f32 dense fallback despite being listed \
                     as packed. The weight would load at 4x-8x its format's size and run the wrong kernel.");
            }
            checked += 1;
        }
        assert_eq!(checked, PACKED.len(), "not every packed format was exercised");
        eprintln!("real-path check: {checked} packed formats built a non-Dense shard on a live GPU");
    }
}

/// The indexed ReLU² expert kernel against the trusted flat Q5_0 matmul.
///
/// The oracle is `matmul_q5_0` restricted to the selected expert's rows: if routing and the slab
/// offset are right, picking expert `e` out of a 4-expert slab must equal running the dense kernel on
/// that expert's rows alone. That isolates the two things this kernel adds — the expert offset and the
/// activation — from the dequant math, which is already trusted.
#[cfg(test)]
mod moe_relu2_id_tests {
    use super::*;

    fn synth_q5_0(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
        let nblk = rows * (cols / 32);
        let mut raw = Vec::with_capacity(nblk * 22);
        for _ in 0..nblk {
            raw.extend_from_slice(&0x2c00u16.to_le_bytes());        // f16 0.0625
            raw.extend_from_slice(&(rnd() as u32).to_le_bytes());   // qh
            for _ in 0..16 { raw.push((rnd() & 0xFF) as u8); }      // qs
        }
        raw
    }

    /// The weighted-sum down kernel, same oracle. Written and NOT validated on the first pass, which
    /// is exactly the kernel to suspect when a model runs and produces garbage.
    #[test]
    fn id_wsum_matches_dense_on_the_selected_expert() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED id_wsum: no GPU");
            return;
        };
        let ctx = Arc::new(ctx);
        let (n_exp, out_pe, inn) = (4usize, 8usize, 64usize);
        let raw = synth_q5_0(n_exp * out_pe, inn, 11);
        let slab = Q5_0Weights::from_bytes(&ctx, &raw, n_exp * out_pe, inn);
        let t = 3usize;
        // x is [T, k, in]; with k = 1 that is [T, 1, in], i.e. one expert slot per token.
        let xv: Vec<f32> = (0..t * inn).map(|i| ((i * 29 % 97) as f32 - 48.0) / 55.0).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[t, 1, inn]);
        let x2 = Tensor::from_vec(&ctx, &xv, &[t, inn]);

        for e in 0..n_exp {
            let w = 0.5 + e as f32 * 0.25;   // a non-unit weight, so the wsum factor is exercised
            let sel: Vec<f32> = (0..t).flat_map(|_| [w, e as f32]).collect();
            let selw = Tensor::from_vec(&ctx, &sel, &[t, 2]);
            let got = pollster::block_on(x.matmul_q5_0_id_wsum(&slab, &selw, out_pe).to_vec());

            let row_bytes = (inn / 32) * 22;
            let lo = e * out_pe * row_bytes;
            let one = Q5_0Weights::from_bytes(&ctx, &raw[lo..lo + out_pe * row_bytes], out_pe, inn);
            let want: Vec<f32> = pollster::block_on(x2.matmul_q5_0(&one).to_vec())
                .into_iter().map(|v| v * w).collect();

            let scale = want.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
            let err = got.iter().zip(&want).fold(0f32, |a, (&g, &q)| a.max((g - q).abs())) / scale;
            assert!(err < 1e-5, "expert {e}: id_wsum differs from dense by {err:.3e} relative");
        }
        eprintln!("id_wsum: all {n_exp} experts match dense x weight");
    }

    #[test]
    fn relu2_id_matches_dense_on_the_selected_expert() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED relu2_id_matches_dense_on_the_selected_expert: no GPU");
            return;
        };
        let ctx = Arc::new(ctx);
        let (n_exp, eff, inn) = (4usize, 8usize, 64usize);
        let raw = synth_q5_0(n_exp * eff, inn, 3);
        let slab = Q5_0Weights::from_bytes(&ctx, &raw, n_exp * eff, inn);

        let t = 3usize;
        let xv: Vec<f32> = (0..t * inn).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[t, inn]);

        for e in 0..n_exp {
            // k = 1, every token routed to expert e. moe_topk layout: [w_0.., id_0..].
            let sel: Vec<f32> = (0..t).flat_map(|_| [1.0f32, e as f32]).collect();
            let selw = Tensor::from_vec(&ctx, &sel, &[t, 2]);
            let got = pollster::block_on(x.matmul_q5_0_relu2_id(&slab, &selw, 1, eff).to_vec());

            // Oracle: the dense kernel on just this expert's rows, then ReLU² by hand.
            let row_bytes = (inn / 32) * 22;
            let lo = e * eff * row_bytes;
            let one = Q5_0Weights::from_bytes(&ctx, &raw[lo..lo + eff * row_bytes], eff, inn);
            let want: Vec<f32> = pollster::block_on(x.matmul_q5_0(&one).to_vec())
                .into_iter().map(|v| { let a = v.max(0.0); a * a }).collect();

            let scale = want.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
            let err = got.iter().zip(&want).fold(0f32, |a, (&g, &w)| a.max((g - w).abs())) / scale;
            assert!(err < 1e-5, "expert {e}: indexed kernel differs from dense by {err:.3e} relative");
        }
        eprintln!("relu2_id: all {n_exp} experts match the dense kernel exactly");
    }
}

/// **The MXFP4 packed kernel against the scalar dequant that is itself bit-diffed against ggml.**
///
/// The oracle is `ferric_gguf::deq_raw(.., 39)`, which is pinned to
/// `ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float` over the full 4096-pair (E8M0 scale x E2M1 code)
/// grid and over 11.1 M real checkpoint elements. So the question here is only whether the WGSL says
/// the same thing as the Rust — and "close" is the wrong bar, because a microscaling dequant is
/// exact-representable: every output is a table entry times a power of two, so the kernel is either
/// bit-identical or wrong.
///
/// **Both tests recover weights exactly rather than comparing sums under a tolerance.** A matmul of
/// random activations against a 4-bit weight hides a wrong nibble inside an accumulation, and the
/// tolerance needed to pass legitimately is wide enough to pass illegitimately too. Instead each test
/// arranges the activations so that every output element is one product plus exact zeros, which makes
/// the GPU's accumulator reproduce a single dequantized weight with no rounding at all — and then
/// compares `f32::to_bits`, which sees the `+0.0`/`−0.0` distinction that `==` cannot.
///
/// The two tests split the work: the grid covers **every** (scale, code) pair including the
/// saturating ends nobody's file contains, and the identity probe covers **layout** — nibble halves,
/// the 17-byte stride, the 4-blocks-per-word scale packing and multi-block rows — on pseudo-random
/// blocks. They also cover the two dispatch templates: shapes are chosen so `q2_0_split_k` routes the
/// grid's second half through the flat kernel and everything else through split-K, because the two
/// have different reduction structure and a bug can live in one alone.
#[cfg(test)]
mod mxfp4_kernel_tests {
    use super::*;

    const MXFP4: u32 = 39;

    /// **Does this fabric keep f32 subnormals, or flush them to zero?**
    ///
    /// Probed with an ordinary elementwise multiply — a kernel with nothing whatsoever to do with
    /// MXFP4 — so the answer is a measured property of the device rather than an excuse manufactured
    /// by the code under test. `1e-30 · 1e-9 = 1e-39` is below f32's smallest **normal** (1.18e-38)
    /// and well above its smallest subnormal (1.4e-45), so an IEEE-complete device returns it and a
    /// flush-to-zero device returns exactly 0.
    fn device_keeps_subnormals(ctx: &Arc<Context>) -> bool {
        let a = Tensor::from_vec(ctx, &[1e-30f32], &[1, 1]);
        let b = Tensor::from_vec(ctx, &[1e-9f32], &[1, 1]);
        pollster::block_on(a.mul(&b).to_vec())[0] != 0.0
    }

    /// One 17-byte block: scale byte `e`, and `code` placed in the low nibble of `qs[0]` (element 0)
    /// or the high nibble (element 16). Every other nibble is 0, whose table entry is `+0.0`.
    fn probe_block(e: u8, code: u8, high: bool) -> [u8; 17] {
        let mut b = [0u8; 17];
        b[0] = e;
        b[1] = if high { code << 4 } else { code };
        b
    }

    /// Every one of the 256 E8M0 scale bytes x 16 E2M1 codes x both nibble halves — 8192 blocks, one
    /// per weight row — read back through the kernel and compared to the scalar oracle on bits.
    ///
    /// The activation is a constant `m` per row, so `acc = Σⱼ m·wⱼ` collapses to `m·w` exactly (31 of
    /// the 32 terms are `m·0.0`). `m` differs per activation row so a kernel that ignored `r` in
    /// `x[r·in_dim + …]` would produce equal rows and fail; the multipliers are powers of two so the
    /// product stays exact even where `w` is subnormal.
    ///
    /// This is the test the task description warned about: `2^(e−127)` overflows at `e = 255` where
    /// ggml is finite, and a kernel that computes the scale before the table lookup passes every
    /// realistic exponent and fails only here.
    #[test]
    fn mxfp4_kernel_matches_ggml_over_the_full_e8m0_x_e2m1_grid() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED mxfp4_kernel_matches_ggml_over_the_full_e8m0_x_e2m1_grid: no GPU. \
                       NOTHING about the kernel was checked.");
            return;
        };
        let ctx = Arc::new(ctx);

        // 8192 rows = 256 scales x 16 codes x {low, high}. Row order is (e, code, half).
        let mut raw: Vec<u8> = Vec::with_capacity(8192 * 17);
        let mut want: Vec<f32> = Vec::with_capacity(8192);
        for e in 0..=255u8 {
            for code in 0..16u8 {
                for &high in &[false, true] {
                    let blk = probe_block(e, code, high);
                    raw.extend_from_slice(&blk);
                    let deq = ferric_gguf::deq_raw(&blk, 32, MXFP4).expect("scalar oracle");
                    want.push(deq[if high { 16 } else { 0 }]);
                }
            }
        }
        assert_eq!(want.len(), 4096 * 2, "the grid must be every (scale, code) pair, both halves");

        // Two shapes so both dispatch templates run. q2_0_split_k(rows, n_out): rows<=2 → split-K;
        // rows>2 with n_out >= 16384 → flat. Asserted, not assumed — the routing is a plain function
        // and this pins which branch each half of the test is actually taking.
        assert!(q2_0_split_k(1, 8192), "grid part 1 was meant to exercise the split-K kernel");
        assert!(!q2_0_split_k(3, 16384), "grid part 2 was meant to exercise the flat kernel");

        // ---- split-K: 1 activation row, 8192 weight rows ----
        let w = Mxfp4Weights::from_bytes(&ctx, &raw, 8192, 32);
        let x = Tensor::from_vec(&ctx, &vec![1.0f32; 32], &[1, 32]);
        let got = pollster::block_on(x.matmul_mxfp4(&w).to_vec());
        // A divergence is EXCUSED only if the exact answer is a subnormal f32 AND this device was
        // independently measured not to keep subnormals AND the kernel returned a clean zero. Every
        // other difference is a kernel bug and fails. The excused population is derived from the
        // ORACLE (|want| below f32::MIN_POSITIVE), never from what happened to differ, so a kernel
        // that lost a normal-range value cannot be absorbed into it.
        let keeps_sub = device_keeps_subnormals(&ctx);
        let want_subnormal = |v: f32| v != 0.0 && v.abs() < f32::MIN_POSITIVE;
        let n_subnormal = want.iter().filter(|&&v| want_subnormal(v)).count();
        let mut flushed = 0usize;
        let mut ndiff = 0usize;
        let mut first = None;
        for i in 0..8192 {
            if got[i].to_bits() == want[i].to_bits() { continue; }
            if !keeps_sub && want_subnormal(want[i]) && got[i] == 0.0 { flushed += 1; continue; }
            ndiff += 1;
            if first.is_none() { first = Some(i); }
        }
        if let Some(i) = first {
            let (e, code, high) = (i / 32, (i / 2) % 16, i % 2 == 1);
            panic!("split-K MXFP4 kernel differs from ggml on {ndiff}/8192 (scale,code) probes at values \
                    the device CAN represent; first at e={e} code={code} high={high}: gpu {} (0x{:08x}) \
                    vs ggml {} (0x{:08x})",
                   got[i], got[i].to_bits(), want[i], want[i].to_bits());
        }
        // Flush-to-zero must be TOTAL, not selective: if the device drops subnormals it must drop all
        // 16 of them. A kernel that formed a subnormal INTERMEDIATE would take normal-range results
        // down with it and land here with flushed > n_subnormal — which is exactly how the first
        // version of this kernel behaved (56 lost, of which only 16 were subnormal answers).
        assert_eq!(flushed, if keeps_sub { 0 } else { n_subnormal },
            "device keeps_subnormals={keeps_sub}: expected {} flushed-to-zero results, saw {flushed}. \
             Anything else means the kernel is destroying values the device could have represented.",
            if keeps_sub { 0 } else { n_subnormal });

        // ---- flat: 3 activation rows with different constants, 16384 weight rows (grid twice) ----
        let mut raw2 = raw.clone();
        raw2.extend_from_slice(&raw);
        let w2 = Mxfp4Weights::from_bytes(&ctx, &raw2, 16384, 32);
        let mults = [1.0f32, 2.0, 4.0];
        let xv: Vec<f32> = mults.iter().flat_map(|&m| std::iter::repeat(m).take(32)).collect();
        let x2 = Tensor::from_vec(&ctx, &xv, &[3, 32]);
        let got2 = pollster::block_on(x2.matmul_mxfp4(&w2).to_vec());
        let mut ndiff2 = 0usize;
        let mut flushed2 = 0usize;
        let mut first2 = None;
        for (r, &m) in mults.iter().enumerate() {
            for o in 0..16384usize {
                let w0 = want[o % 8192];
                let exp = m * w0;
                let g = got2[r * 16384 + o];
                if g.to_bits() == exp.to_bits() { continue; }
                // Classify on the WEIGHT, not on m*w. The device drops the subnormal the moment the
                // kernel forms it, so a subnormal weight is lost even when m lifts the exact product
                // back into the normal range — measured: with the predicate on m*w instead, 56 of
                // these read as kernel bugs across m in {1,2,4}, all of them this one effect.
                if !keeps_sub && want_subnormal(w0) && g == 0.0 { flushed2 += 1; continue; }
                ndiff2 += 1;
                if first2.is_none() { first2 = Some((r, o, g, exp)); }
            }
        }
        if let Some((r, o, g, exp)) = first2 {
            let i = o % 8192;
            panic!("flat MXFP4 kernel differs on {ndiff2}/49152; first at act-row {r} (x={}) weight-row {o} \
                    (e={} code={} high={}): gpu {g} (0x{:08x}) vs expected {exp} (0x{:08x})",
                   mults[r], i / 32, (i / 2) % 16, i % 2 == 1, g.to_bits(), exp.to_bits());
        }
        assert_eq!(flushed2, if keeps_sub { 0 } else { mults.len() * 2 * n_subnormal },
            "flat kernel: expected {} flushed subnormal weights over {} activation rows x 2 grid \
             copies, saw {flushed2}", if keeps_sub { 0 } else { mults.len() * 2 * n_subnormal }, mults.len());
        eprintln!("MXFP4: bit-identical to ggml on all 4096 (E8M0, E2M1) pairs x 2 nibble halves, \
                   split-K and flat kernels, 3 activation rows ({} probes). device keeps subnormals: \
                   {keeps_sub}; results below f32::MIN_POSITIVE flushed to zero: {flushed}/{n_subnormal}",
                  8192 + 49152);
    }

    /// Pseudo-random blocks over the exponent range a real checkpoint uses, read back **weight by
    /// weight** with an identity activation: `x = I[cols,cols]` makes `out[j][o] = w[o][j]` exactly,
    /// so every one of the `rows·cols` weights is compared to the scalar oracle on bits.
    ///
    /// This is what the grid cannot see: multi-block rows (so the 17-byte stride and the
    /// 4-blocks-per-word scale unpack must both be right), both nibble halves in every byte, and the
    /// `o · nblk + blk` block indexing. A stride or scale-packing error moves every weight after the
    /// first block and shows up as thousands of differences, not a rounding wobble.
    #[test]
    fn mxfp4_kernel_recovers_every_weight_of_a_multi_block_row() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED mxfp4_kernel_recovers_every_weight_of_a_multi_block_row: no GPU. \
                       NOTHING about the kernel was checked.");
            return;
        };
        let ctx = Arc::new(ctx);
        let (rows, cols) = (48usize, 128usize);      // 4 blocks per row, 192 blocks total
        let nblk = rows * (cols / 32);
        let mut s = 0x9E37_79B9u32;
        let mut rnd = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        let mut raw = Vec::with_capacity(nblk * 17);
        for _ in 0..nblk {
            // Scale bytes 0x6C..0x8B: 2^-19 .. 2^12, the band a requantized checkpoint actually lands
            // in. Deliberately several distinct exponents per row, so a block read at the wrong
            // stride picks up a neighbour's scale and cannot pass by luck.
            raw.push(0x6C + (rnd() % 32) as u8);
            for _ in 0..16 { raw.push((rnd() % 256) as u8) }
        }
        let want = ferric_gguf::deq_raw(&raw, rows * cols, MXFP4).expect("scalar oracle");
        assert!(want.iter().any(|v| *v != 0.0), "degenerate fixture: every oracle weight is zero");

        let w = Mxfp4Weights::from_bytes(&ctx, &raw, rows, cols);
        let mut eye = vec![0.0f32; cols * cols];
        for j in 0..cols { eye[j * cols + j] = 1.0; }
        let x = Tensor::from_vec(&ctx, &eye, &[cols, cols]);
        let got = pollster::block_on(x.matmul_mxfp4(&w).to_vec());   // [cols, rows], got[j*rows+o] = w[o][j]

        let mut ndiff = 0usize;
        let mut first = None;
        for o in 0..rows {
            for j in 0..cols {
                let g = got[j * rows + o];
                let e = want[o * cols + j];
                if g.to_bits() != e.to_bits() {
                    ndiff += 1;
                    if first.is_none() { first = Some((o, j, g, e)); }
                }
            }
        }
        if let Some((o, j, g, e)) = first {
            panic!("MXFP4 kernel recovered {ndiff}/{} weights wrongly; first at row {o} col {j} \
                    (block {} of the row, {} nibble): gpu {g} (0x{:08x}) vs ggml-equivalent {e} (0x{:08x})",
                   rows * cols, j / 32, if j % 32 < 16 { "low" } else { "high" }, g.to_bits(), e.to_bits());
        }
        eprintln!("MXFP4: all {} weights of a {rows}x{cols} ({} blocks/row) tensor recovered bit-exactly",
                  rows * cols, cols / 32);
    }

    /// The point of the format. Resident bytes are read back from the **live GPU buffers**
    /// (`wgpu::Buffer::size()`), not computed from the block formula, so a layout that padded
    /// differently than the doc claims would be caught here rather than described correctly and
    /// implemented wrongly.
    #[test]
    fn mxfp4_resident_bytes_are_the_formats_own_and_not_f32() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED mxfp4_resident_bytes_are_the_formats_own_and_not_f32: no GPU.");
            return;
        };
        let ctx = Arc::new(ctx);
        let (rows, cols) = (256usize, 512usize);
        let nblk = rows * (cols / 32);
        let raw = vec![0x7Fu8; nblk * 17];
        let dense = rows * cols * 4;

        // ⚠ GO THROUGH `QMatrix::from_bytes`, NOT `Mxfp4Weights::from_bytes`. This test previously
        // constructed the packed weight directly and asked its size — which is compact whether or not
        // a loader ever ROUTES an MXFP4 tensor to it. Mutating `block_bytes(39)` to `None`, which sends
        // every real MXFP4 weight down the f32 dense fallback, left that version green. The claim in
        // this test's own name is about what a loaded weight costs, so the loader path is the subject.
        // Half one of the residency claim, and the half that lives in ANOTHER CRATE. Every loader
        // picks its path with `if QMatrix::block_bytes(ty).is_some() { from_bytes } else
        // { from_dense }` — see `ferric_llama::qwen35::qm`, which the dense runtime also uses. So
        // `block_bytes(39)` returning `None` silently routes every real MXFP4 tensor to the f32 dense
        // fallback. Nothing in ferric-llama's suite covers that today, and a ferric-tensor test cannot
        // reach it, so the published predicate is asserted here where it is defined.
        assert_eq!(QMatrix::block_bytes(39), Some((32, 17)),
                   "block_bytes is the predicate every loader uses to choose the packed path over the \
                    f32 fallback; `None` here costs ~7.5x resident memory and still generates correct \
                    text, so no output check would catch it");
        // Half two: given the loader does choose `from_bytes`, the type actually reaches a packed shard.
        let m = QMatrix::from_bytes(&ctx, &raw, 39, rows, cols).expect("MXFP4 must load as a QMatrix");
        assert_eq!(m.n_shards(), 1, "one shard for a lone MXFP4 tensor");
        assert!(matches!(m.shards[0], QShard::Mxfp4(_)),
                "MXFP4 must reach the packed shard. A `QShard::Dense` here means `block_bytes(39)` or \
                 `QShard::build`'s type arm stopped routing it, and the weight is silently f32-resident \
                 at ~7.5x the format's footprint — which still generates correct text, so nothing else \
                 would notice.");
        let packed = m.nbytes();
        assert_eq!(packed, raw.len(),
            "MXFP4 should be resident at exactly its on-disk size; buffers hold {packed} B for a \
             {} B tensor", raw.len());
        assert_eq!(packed as f64 / (rows * cols) as f64, 0.53125, "4.25 bits/weight is the format");
        eprintln!("MXFP4 resident: {packed} B ({:.5} B/elem) vs {dense} B f32 ({:.2}x) — through \
                   QMatrix::from_bytes, from live buffer sizes",
                  packed as f64 / (rows * cols) as f64, dense as f64 / packed as f64);
    }
}

/// STQ1_0 GEMV. `codes`/`signs` are output-major so adjacent threads read adjacent words; the
/// codebook maps `(sign << 4) | slot` to four 2-bit lanes, each decoding as `lane − 1`.
/// **STQ1_0 GEMV, vec4 activation form.** Same arithmetic as [`MATMUL_STQ1_0_WGSL`], different
/// traversal order.
///
/// The scalar form walks groups and gathers each group's four lanes at stride 16, which costs four
/// separate `x` loads per group — 256 scalar loads per 256-weight block, against 8 weight-word
/// loads. Ferric's Q2_0 kernel already records the consequence: the ACTIVATION loads, not the
/// weights, dominate the instruction stream, and a kernel that is issue-bound cannot spend the
/// bandwidth its format saves.
///
/// The stride-16 layout looks like it forbids a vector load, and it does — in that direction. But
/// invert the loop: inside a 64-weight chunk, holding the lane `p` fixed and stepping four
/// consecutive groups walks four CONSECUTIVE activations. So `x` is read as `vec4<f32>` and the
/// four weights are reduced with one `dot()`, at 64 vector loads per block instead of 256 scalar
/// ones — and the four groups' slot codes are four adjacent nibbles of the same word, so the weight
/// side gets cheaper too.
const MATMUL_STQ1_0_V4_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        signs:  array<u32>;
@group(0) @binding(3) var<storage,read>        scales: array<u32>;
@group(0) @binding(4) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(5) var<uniform>             info:   vec4<u32>;

var<private> CODEBOOK: array<u32, 32> = array<u32, 32>(
    0xA9u, 0x89u, 0x29u, 0x09u, 0xA6u, 0x86u, 0x26u, 0x06u,
    0x9Au, 0x92u, 0x1Au, 0x12u, 0x6Au, 0x62u, 0x4Au, 0x42u,
    0x01u, 0x21u, 0x81u, 0xA1u, 0x04u, 0x24u, 0x84u, 0xA4u,
    0x10u, 0x18u, 0x90u, 0x98u, 0x40u, 0x48u, 0x60u, 0x68u
);

fn lane(qpack: u32, p: u32) -> f32 { return f32(i32((qpack >> (2u * p)) & 3u) - 1); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let si = blk * o_dim + o;
        let sw = unpack2x16float(scales[si >> 1u]);
        let d = select(sw.y, sw.x, (si & 1u) == 0u);
        let xbase = r * in_dim + blk * 256u;
        var bacc = 0.0;
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            // One sign word covers two chunks; one code word covers eight groups, so a chunk's
            // sixteen groups are exactly two words. Both are hoisted out of the k loop.
            let sword = signs[(blk * 2u + (c >> 1u)) * o_dim + o];
            let w0 = codes[(blk * 8u + c * 2u) * o_dim + o];
            let w1 = codes[(blk * 8u + c * 2u + 1u) * o_dim + o];
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                let word = select(w1, w0, k < 2u);
                let nib0 = 4u * (k & 1u);            // first of four adjacent nibbles
                let sb0  = (c & 1u) * 16u + 4u * k;  // first of four adjacent sign bits
                var qp: array<u32, 4>;
                for (var t: u32 = 0u; t < 4u; t = t + 1u) {
                    let slot = (word >> (4u * (nib0 + t))) & 0x0Fu;
                    let sbit = (sword >> (sb0 + t)) & 1u;
                    qp[t] = CODEBOOK[(sbit << 4u) | slot];
                }
                for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                    // Lane p of four consecutive groups IS four consecutive activations.
                    let e0 = xbase + c * 64u + p * 16u + 4u * k;
                    let xv = vec4<f32>(x[e0], x[e0 + 1u], x[e0 + 2u], x[e0 + 3u]);
                    let wv = vec4<f32>(lane(qp[0], p), lane(qp[1], p), lane(qp[2], p), lane(qp[3], p));
                    bacc = bacc + dot(xv, wv);
                }
            }
        }
        acc = acc + bacc * d;
    }
    out[idx] = acc;
}
"#;

const MATMUL_STQ1_0_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;   // [word][output], 8 words/block
@group(0) @binding(2) var<storage,read>        signs:  array<u32>;   // [word][output], 2 words/block
@group(0) @binding(3) var<storage,read>        scales: array<u32>;   // [block][output], f16 x2 per u32
@group(0) @binding(4) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(5) var<uniform>             info:   vec4<u32>;    // rows, out, in, threads_per_grid_row

var<private> CODEBOOK: array<u32, 32> = array<u32, 32>(
    0xA9u, 0x89u, 0x29u, 0x09u, 0xA6u, 0x86u, 0x26u, 0x06u,
    0x9Au, 0x92u, 0x1Au, 0x12u, 0x6Au, 0x62u, 0x4Au, 0x42u,
    0x01u, 0x21u, 0x81u, 0xA1u, 0x04u, 0x24u, 0x84u, 0xA4u,
    0x10u, 0x18u, 0x90u, 0x98u, 0x40u, 0x48u, 0x60u, 0x68u
);

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let si = blk * o_dim + o;
        let sw = unpack2x16float(scales[si >> 1u]);
        let d = select(sw.y, sw.x, (si & 1u) == 0u);
        let xbase = r * in_dim + blk * 256u;
        var bacc = 0.0;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word  = codes[(blk * 8u + w) * o_dim + o];
            let sword = signs[(blk * 2u + (w >> 2u)) * o_dim + o];
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let g     = w * 8u + i;
                let slot  = (word >> (4u * i)) & 0x0Fu;
                // ⚠ `w & 3u` is load-bearing, not cosmetic: one sign word covers four code words,
                // so words 4..7 index bits 0..31 of the SECOND word. Writing `w * 8u + i` shifts by
                // 32..63, which WGSL leaves undefined — Metal happens to mask it mod 32 and give the
                // identical answer, so a mutation to that form passes every test on this machine and
                // is a portability bug waiting for a backend that does not mask.
                let sbit  = (sword >> ((w & 3u) * 8u + i)) & 1u;
                let qpack = CODEBOOK[(sbit << 4u) | slot];
                // ⚠ stride 16 inside the 64-weight chunk, not four adjacent weights.
                let base  = xbase + (g >> 4u) * 64u + (g & 15u);
                for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                    bacc = bacc + x[base + p * 16u] * f32(i32((qpack >> (2u * p)) & 3u) - 1);
                }
            }
        }
        acc = acc + bacc * d;   // one scale for the whole 256-block
    }
    out[idx] = acc;
}
"#;

#[cfg(test)]
mod stq1_0_kernel {
    use super::*;
    use ferric_gguf::quantize::quantize_stq1_0;

    macro_rules! ctx_or_skip {
        () => { match pollster::block_on(Context::new()) { Ok(c) => Arc::new(c), Err(_) => { eprintln!("no GPU context — skipping"); return } } };
    }

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// The packed kernel must agree with dequantise-then-matmul to f32 round-off.
    ///
    /// Not a round-trip: `deq_raw` is the reader verified against Tencent's published weights at
    /// 99.3% sign agreement, so this pins the kernel against an independently checked decoder. The
    /// stride-16 grouping is the thing most likely to be wrong here, and it cannot hide — reading
    /// the groups contiguously pairs the right 256 activations with the right 256 weights in the
    /// wrong order, which changes every output.
    #[test]
    fn packed_matmul_matches_dequant_then_matmul() {
        let ctx = ctx_or_skip!();
        let (rows, cols, toks) = (17usize, 512usize, 3usize);

        let mut seed = 4242u64;
        let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed)).collect();
        let mut bytes = Vec::new();
        for r in 0..rows { quantize_stq1_0(&wf[r * cols..(r + 1) * cols], None, &mut bytes) }
        assert_eq!(bytes.len(), rows * (cols / 256) * 42);

        let xv: Vec<f32> = (0..toks * cols).map(|_| lcg(&mut seed)).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[toks, cols]);

        let deq = ferric_gguf::deq_raw(&bytes, rows * cols, 43).unwrap();
        let wdense = Tensor::from_vec(&ctx, &deq, &[rows, cols]);
        let want = pollster::block_on(x.matmul_bt(&wdense).to_vec());

        let packed = Stq1_0Weights::from_bytes(&ctx, &bytes, rows, cols);
        let got = pollster::block_on(x.matmul_stq1_0(&packed).to_vec());

        assert_eq!(got.len(), toks * rows);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-2, "reference is ~zero; this would pass on anything");
        let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("STQ1_0 packed vs dense: max |Δ| = {worst:.3e} on outputs of magnitude {scale:.3e}");
        assert!(worst < 1e-4 * scale, "packed kernel diverges by {worst}");
    }

    /// The reason the kernel exists. The packed weight must be resident at its on-disk size — the
    /// repack rearranges bytes, it does not expand them — while the dense fallback holds f32.
    #[test]
    fn packed_residency_is_the_on_disk_size() {
        let ctx = ctx_or_skip!();
        let (rows, cols) = (8usize, 1024usize);
        let mut seed = 7u64;
        let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed)).collect();
        let mut bytes = Vec::new();
        for r in 0..rows { quantize_stq1_0(&wf[r * cols..(r + 1) * cols], None, &mut bytes) }

        let m = QMatrix::from_bytes(&ctx, &bytes, 43, rows, cols).expect("STQ1_0 must load as a QMatrix");
        assert!(matches!(m.shards[0], QShard::Stq1_0(_)),
                "a QShard::Dense here means block_bytes(43) or QShard::build's arm stopped routing \
                 it, and the weight is silently f32-resident at 24.4x the format's footprint — which \
                 still produces correct numbers, so nothing else would notice");
        let packed = m.nbytes();
        assert_eq!(packed, bytes.len(), "resident {packed} B for a {} B tensor", bytes.len());
        let bpw = packed as f64 * 8.0 / (rows * cols) as f64;
        assert!((bpw - 1.3125).abs() < 1e-9, "1.3125 bits/weight is the whole point, got {bpw}");
        eprintln!("STQ1_0 resident: {packed} B ({bpw} bits/weight) vs {} B f32 ({:.1}x)",
                  rows * cols * 4, (rows * cols * 4) as f64 / packed as f64);
    }

    /// **Every (group position, code) pair, through the real kernel, in every traversal form.**
    ///
    /// `every_codebook_pattern_reaches_the_kernel` gives every group of a row the SAME slot, so a
    /// kernel that decoded group g but wrote its lanes to group g' position would still match the
    /// reference: every group looks identical. This fixture breaks that symmetry -- row o carries
    /// code o/64 in group o%64 alone, everything else at slot 0 -- against a position-distinct
    /// activation ramp, so a misplaced group changes the dot. 64 x 32 = 2048 rows.
    ///
    /// This is also what ties the WGSL text to the Kani proof
    /// `vec4_traversal_addresses_every_lane_where_the_decoder_does`: that proves a Rust mirror of
    /// the shader's addressing; this runs the shader itself over the same exhaustive set.
    #[test]
    fn every_group_position_reaches_the_kernel() {
        let ctx = ctx_or_skip!();
        let (rows, cols) = (64 * 32, 256usize);
        let mut bytes = vec![0u8; rows * 42];
        for o in 0..rows {
            let (g, code) = (o % 64, o / 64);
            let (slot, sign) = ((code % 16) as u8, (code / 16) as u8);
            let blk = &mut bytes[o * 42..(o + 1) * 42];
            blk[g / 2] |= slot << (4 * (g & 1));
            blk[32 + g / 8] |= sign << (g % 8);
            blk[40..42].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
        // Position-distinct and mixed-sign, so no two positions contribute the same product.
        let xv: Vec<f32> = (0..cols).map(|i| ((i * 37) % 97) as f32 - 48.0).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[1, cols]);
        let deq = ferric_gguf::deq_raw(&bytes, rows * cols, 43).unwrap();
        let want = pollster::block_on(x.matmul_bt(&Tensor::from_vec(&ctx, &deq, &[rows, cols])).to_vec());
        let packed = Stq1_0Weights::from_bytes(&ctx, &bytes, rows, cols);
        // EXACT, not within a tolerance. Activations are integers in [-48, 48], weights are
        // {-1, 0, +1} and d is exactly 1.0, so every product and every partial sum is an integer
        // below 2^24 -- representable exactly in f32 whatever order the GPU accumulates in. A
        // tolerance here would be doing no work while looking like it was; and a non-exact result
        // would mean a non-integer path somewhere, which is worth failing on.
        for form in [Stq1Form::Scalar, Stq1Form::Vec4, Stq1Form::Vec4Table] {
            let got = pollster::block_on(x.matmul_stq1_0_form(&packed, form).to_vec());
            for o in 0..rows {
                assert!(want[o] == got[o],
                        "{form:?}: group {} code {} (slot {}, sign {}): {} vs {}",
                        o % 64, o / 64, (o / 64) % 16, (o / 64) / 16, got[o], want[o]);
            }
        }
        // The fixture must let a misplaced group SHOW. Rows 0..64 carry code 0 in their group,
        // which is the background pattern, so they are the all-slot-0 block; every later row
        // differs from its code-0 twin in exactly one group. If that change does not move the dot,
        // a kernel writing the group's lanes to the wrong positions could match the reference.
        // (A first version of this guard demanded 1024 globally distinct sums out of ~289 possible
        // values -- sums of three +-1 terms over [-48,48] -- and failed by pigeonhole while every
        // row had in fact passed. The property is per-row, not global.)
        let unmoved = (64..rows).filter(|&o| want[o] == want[o % 64]).count();
        assert!(unmoved * 20 < rows, "{unmoved} of {rows} rows have a dot unchanged by their group's \
                 code; the activation ramp does not separate positions");
    }

    /// Every legal codebook entry must survive the shader's decode. A single mistyped hex digit in
    /// the WGSL table would corrupt one pattern in thirty-two — rare enough that a random matmul
    /// could miss it, so this drives all 32 through deliberately.
    #[test]
    fn every_codebook_pattern_reaches_the_kernel() {
        let ctx = ctx_or_skip!();
        let (rows, cols) = (32usize, 256usize);
        // Row o uses slot (o % 16) and sign (o / 16) in every one of its 64 groups.
        let mut bytes = vec![0u8; rows * 42];
        for o in 0..rows {
            let (slot, sign) = ((o % 16) as u8, (o / 16) as u8);
            let blk = &mut bytes[o * 42..(o + 1) * 42];
            for g in 0..64 {
                blk[g / 2] |= slot << (4 * (g & 1));
                blk[32 + g / 8] |= sign << (g % 8);
            }
            blk[40..42].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
        let xv: Vec<f32> = (0..cols).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[1, cols]);

        let deq = ferric_gguf::deq_raw(&bytes, rows * cols, 43).unwrap();
        let want = pollster::block_on(x.matmul_bt(&Tensor::from_vec(&ctx, &deq, &[rows, cols])).to_vec());
        let packed = Stq1_0Weights::from_bytes(&ctx, &bytes, rows, cols);
        let got = pollster::block_on(x.matmul_stq1_0(&packed).to_vec());
        for o in 0..rows {
            assert!((want[o] - got[o]).abs() < 1e-3,
                    "codebook entry {} (slot {}, sign {}): {} vs {}", o, o % 16, o / 16, got[o], want[o]);
        }
        // The 32 patterns must not all give the same answer, or this proves nothing.
        let spread = want.iter().fold(f32::MIN, |a, b| a.max(*b)) - want.iter().fold(f32::MAX, |a, b| a.min(*b));
        assert!(spread > 1.0, "the 32 codebook patterns produced near-identical outputs ({spread})");
    }
}

/// One workgroup row tops out at 65535 workgroups, so a real LM head needs a 2D grid.
fn grid2d(n: usize) -> (usize, usize) {
    let wg = n.div_ceil(64);
    let gw = wg.min(32768);
    (gw, wg.div_ceil(gw))
}

/// Shared by both IQ kernels: `ksigns_iq2xs[i]` is a parity code — the low seven bits are the index
/// and the top bit makes the population count even — so it is computed, never tabled.
const IQ_SIGNS_WGSL: &str = r#"
fn ksigns(i: u32) -> u32 {
    let low = i & 0x7fu;
    return low | ((countOneBits(low) & 1u) << 7u);
}
/// The four bytes of a grid word as floats. One shift-and-mask vector instead of four scalars.
fn bytes4(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(w & 0xffu, (w >> 8u) & 0xffu, (w >> 16u) & 0xffu, (w >> 24u) & 0xffu));
}
/// Four sign bits as ±1. `1 − 2b` is branchless where `select` per lane is not.
fn signs4(sg: u32, base: u32) -> vec4<f32> {
    let b = vec4<u32>((sg >> base) & 1u, (sg >> (base + 1u)) & 1u,
                      (sg >> (base + 2u)) & 1u, (sg >> (base + 3u)) & 1u);
    return vec4<f32>(1.0, 1.0, 1.0, 1.0) - 2.0 * vec4<f32>(b);
}
"#;

const MATMUL_IQ2_XXS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;   // [word][output], 16 words/block
@group(0) @binding(2) var<storage,read>        scales: array<u32>;   // [block][output], f16 x2 per u32
@group(0) @binding(3) var<storage,read>        grid:   array<u32>;   // 256 points, low/high pairs
@group(0) @binding(4) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(5) var<uniform>             info:   vec4<u32>;
__SIGNS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let si = blk * o_dim + o;
        let sw = unpack2x16float(scales[si >> 1u]);
        let d = select(sw.y, sw.x, (si & 1u) == 0u);
        let xbase = r * in_dim + blk * 256u;
        for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
            let lo = codes[(blk * 16u + ib * 2u) * o_dim + o];
            let hi = codes[(blk * 16u + ib * 2u + 1u) * o_dim + o];
            // ⚠ the 4-bit sub-scale shares this word with four 7-bit sign indices: 4*7 + 4 = 32,
            // no spare bit. Reading it from the other word gives a plausible small float.
            let db = d * (0.5 + f32(hi >> 28u)) * 0.25;
            var sacc = 0.0;
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let gi = (lo >> (8u * l)) & 0xffu;
                let g0 = grid[gi * 2u];
                let g1 = grid[gi * 2u + 1u];
                let sg = ksigns((hi >> (7u * l)) & 127u);
                let xb = xbase + ib * 32u + l * 8u;
                // The eight activations behind one grid point are CONTIGUOUS, so this is a plain
                // pair of vector loads. The scalar form issued eight separate ones, and on this
                // kernel the activation loads — not the weight bytes — are what fills the
                // instruction stream.
                let x0 = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
                let x1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
                sacc = sacc + dot(x0, bytes4(g0) * signs4(sg, 0u))
                            + dot(x1, bytes4(g1) * signs4(sg, 4u));
            }
            acc = acc + sacc * db;
        }
    }
    out[idx] = acc;
}
"#;

const MATMUL_IQ3_XXS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;   // [word][output], 24 words/block
@group(0) @binding(2) var<storage,read>        scales: array<u32>;
@group(0) @binding(3) var<storage,read>        grid:   array<u32>;   // 256 four-byte points
@group(0) @binding(4) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(5) var<uniform>             info:   vec4<u32>;
__SIGNS__
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let si = blk * o_dim + o;
        let sw = unpack2x16float(scales[si >> 1u]);
        let d = select(sw.y, sw.x, (si & 1u) == 0u);
        let xbase = r * in_dim + blk * 256u;
        for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
            // ⚠ NOT interleaved: all 16 index words come first, then the 8 sign/scale words.
            let aux = codes[(blk * 24u + 16u + ib) * o_dim + o];
            let db = d * (0.5 + f32(aux >> 28u)) * 0.5;   // 0.5 here, not IQ2_XXS's 0.25
            var sacc = 0.0;
            for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                let bi = ib * 8u + l * 2u;               // byte offset into the 64 index bytes
                let w0 = codes[(blk * 24u + (bi >> 2u)) * o_dim + o];
                let i1 = (w0 >> (8u * (bi & 3u))) & 0xffu;
                let b2 = bi + 1u;
                let w1 = codes[(blk * 24u + (b2 >> 2u)) * o_dim + o];
                let i2 = (w1 >> (8u * (b2 & 3u))) & 0xffu;
                let g1 = grid[i1];
                let g2 = grid[i2];
                let sg = ksigns((aux >> (7u * l)) & 127u);
                let xb = xbase + ib * 32u + l * 8u;
                let x0 = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
                let x1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
                sacc = sacc + dot(x0, bytes4(g1) * signs4(sg, 0u))
                            + dot(x1, bytes4(g2) * signs4(sg, 4u));
            }
            acc = acc + sacc * db;
        }
    }
    out[idx] = acc;
}
"#;

fn matmul_iq2_xxs_wgsl() -> String { MATMUL_IQ2_XXS_WGSL.replace("__SIGNS__", IQ_SIGNS_WGSL) }
fn matmul_iq3_xxs_wgsl() -> String { MATMUL_IQ3_XXS_WGSL.replace("__SIGNS__", IQ_SIGNS_WGSL) }

#[cfg(test)]
mod iq_kernel {
    use super::*;

    macro_rules! ctx_or_skip {
        () => { match pollster::block_on(Context::new()) { Ok(c) => Arc::new(c), Err(_) => { eprintln!("no GPU context — skipping"); return } } };
    }

    /// ⛔ The duplicated grid tables are only safe because this fails when they diverge.
    /// `ferric-tensor` cannot depend on `ferric-gguf` — the layering runs the other way — so the
    /// kernel keeps its own copy and this binds it to the decoder's.
    #[test]
    fn the_two_copies_of_the_grid_agree() {
        for (i, &v) in ferric_gguf::IQ2XXS_GRID.iter().enumerate() {
            let lo = crate::iq_grids::IQ2XXS_GRID_U32[i * 2] as u64;
            let hi = crate::iq_grids::IQ2XXS_GRID_U32[i * 2 + 1] as u64;
            assert_eq!(lo | (hi << 32), v, "IQ2XXS grid entry {i} differs between the two copies");
        }
        for (i, &v) in ferric_gguf::IQ3XXS_GRID.iter().enumerate() {
            assert_eq!(crate::iq_grids::IQ3XXS_GRID_U32[i], v, "IQ3XXS grid entry {i} differs");
        }
        assert_eq!(crate::iq_grids::IQ2XXS_GRID_U32.len(), 512);
        assert_eq!(crate::iq_grids::IQ3XXS_GRID_U32.len(), 256);
    }

    /// Every byte pattern is a legal IQ2/IQ3 block — grid indices span the full 0..255, sign indices
    /// 0..127 and sub-scales 0..15 — so pseudo-random bytes exercise the whole decode surface
    /// without needing an encoder.
    ///
    /// ⚠ The two scale bytes are NOT left random. `f16::from_le_bytes` of an arbitrary pair is
    /// happily NaN or infinity, both arms would then produce NaN, and `(a − b).abs() < tol` is
    /// FALSE for NaN — so the test would fail on an unlucky seed and pass on a lucky one. A
    /// randomised fixture needs its degenerate values excluded on purpose, not by luck.
    fn blocks(n: usize, seed: u64, bpb: usize) -> Vec<u8> {
        let mut s = seed;
        let mut v: Vec<u8> = (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u8
        }).collect();
        for (i, blk) in v.chunks_exact_mut(bpb).enumerate() {
            let d = half::f16::from_f32(0.01 + 0.003 * (i % 7) as f32);
            blk[0..2].copy_from_slice(&d.to_le_bytes());
        }
        v
    }
    fn acts(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0 }).collect()
    }

    /// Both packed kernels against dequantise-then-matmul. `deq_raw` is the decoder verified on
    /// Tencent's published weights, so this pins each kernel to an independently checked reference
    /// rather than to itself.
    #[test]
    fn packed_iq_matmuls_match_dequant_then_matmul() {
        let ctx = ctx_or_skip!();
        let (rows, cols, toks) = (13usize, 512usize, 3usize);
        let x = Tensor::from_vec(&ctx, &acts(toks * cols, 99), &[toks, cols]);

        for (ty, bpb, label) in [(16u32, 66usize, "iq2_xxs"), (18, 98, "iq3_xxs")] {
            let bytes = blocks(rows * (cols / 256) * bpb, 1234 + ty as u64, bpb);
            let deq = ferric_gguf::deq_raw(&bytes, rows * cols, ty).unwrap();
            let want = pollster::block_on(x.matmul_bt(&Tensor::from_vec(&ctx, &deq, &[rows, cols])).to_vec());

            let m = QMatrix::from_bytes(&ctx, &bytes, ty, rows, cols).expect("must load as a QMatrix");
            let got = pollster::block_on(x.matmul_q(&m).to_vec());

            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            assert!(scale > 1e-2, "{label}: reference is ~zero; this would pass on anything");
            assert!(want.iter().all(|v| v.is_finite()), "{label}: the reference has non-finite values");
            assert!(got.iter().all(|v| v.is_finite()), "{label}: the kernel produced non-finite values");
            let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("{label} packed vs dense: max |Δ| = {worst:.3e} on magnitude {scale:.3e}");
            assert!(worst < 1e-4 * scale, "{label} packed kernel diverges by {worst}");
        }
    }

    /// The reason these kernels exist: 160 of Hy4-preview's 213 GiB is IQ2_XXS and IQ3_XXS, and
    /// both were f32-resident until now.
    #[test]
    fn packed_iq_residency_is_the_on_disk_size() {
        let ctx = ctx_or_skip!();
        let (rows, cols) = (8usize, 512usize);
        for (ty, bpb, bpw, label) in [(16u32, 66usize, 2.0625f64, "IQ2_XXS"), (18, 98, 3.0625, "IQ3_XXS")] {
            let bytes = blocks(rows * (cols / 256) * bpb, 7 + ty as u64, bpb);
            let m = QMatrix::from_bytes(&ctx, &bytes, ty, rows, cols).unwrap();
            assert!(!matches!(m.shards[0], QShard::Dense(_)),
                    "{label} fell back to Dense — block_bytes({ty}) or QShard::build stopped routing it, \
                     and the weight is silently f32-resident while still producing correct numbers");
            assert_eq!(m.nbytes(), bytes.len(), "{label} resident {} B for {} B on disk", m.nbytes(), bytes.len());
            let got = m.nbytes() as f64 * 8.0 / (rows * cols) as f64;
            assert!((got - bpw).abs() < 1e-9, "{label} should be {bpw} bits/weight, got {got}");
            eprintln!("{label} resident: {} B ({got} bits/weight) vs {} B f32 ({:.1}x)",
                      m.nbytes(), rows * cols * 4, (rows * cols * 4) as f64 / m.nbytes() as f64);
        }
    }
}

/// Which STQ1_0 traversal the kernel uses. Three forms compute the same matmul; they differ only in
/// how many instructions they spend per weight, which is the whole question for a format whose
/// bandwidth saving is already banked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stq1Form {
    /// 256 scalar activation loads per block. The first working version.
    Scalar,
    /// 64 `vec4` loads and one `dot()` per four weights, with the codebook in `var<private>`.
    Vec4,
    /// As `Vec4`, but the codebook is a storage buffer of pre-expanded lane vectors and the four
    /// groups' lanes are transposed with one `mat4x4` instead of sixteen shift-mask-subtracts.
    Vec4Table,
}

/// The codebook as the shaders index it: `(sign << 4) | slot`.
pub(crate) const STQ1_0_SHADER_CODEBOOK: [u32; 32] = [
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
];

const MATMUL_STQ1_0_V4T_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
@group(0) @binding(1) var<storage,read>        codes:  array<u32>;
@group(0) @binding(2) var<storage,read>        signs:  array<u32>;
@group(0) @binding(3) var<storage,read>        scales: array<u32>;
@group(0) @binding(4) var<storage,read>        cb:     array<vec4<f32>>;  // 32 entries, lanes pre-expanded
@group(0) @binding(5) var<storage,read_write>  out:    array<f32>;
@group(0) @binding(6) var<uniform>             info:   vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info.w; let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    let nblk = in_dim / 256u;
    var acc = 0.0;
    for (var blk: u32 = 0u; blk < nblk; blk = blk + 1u) {
        let si = blk * o_dim + o;
        let sw = unpack2x16float(scales[si >> 1u]);
        let d = select(sw.y, sw.x, (si & 1u) == 0u);
        let xbase = r * in_dim + blk * 256u;
        var bacc = 0.0;
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let sword = signs[(blk * 2u + (c >> 1u)) * o_dim + o];
            let w0 = codes[(blk * 8u + c * 2u) * o_dim + o];
            let w1 = codes[(blk * 8u + c * 2u + 1u) * o_dim + o];
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                let word = select(w1, w0, k < 2u);
                let nib0 = 4u * (k & 1u);
                let sb0  = (c & 1u) * 16u + 4u * k;
                // Four groups' lane vectors, straight out of the table — no shifting per lane.
                let c0 = cb[(((sword >> sb0) & 1u) << 4u) | ((word >> (4u * nib0)) & 0x0Fu)];
                let c1 = cb[(((sword >> (sb0 + 1u)) & 1u) << 4u) | ((word >> (4u * (nib0 + 1u))) & 0x0Fu)];
                let c2 = cb[(((sword >> (sb0 + 2u)) & 1u) << 4u) | ((word >> (4u * (nib0 + 2u))) & 0x0Fu)];
                let c3 = cb[(((sword >> (sb0 + 3u)) & 1u) << 4u) | ((word >> (4u * (nib0 + 3u))) & 0x0Fu)];
                // Columns are the four groups; after the transpose row p is lane p of each group,
                // which is exactly the vector the four consecutive activations pair with.
                let mt = transpose(mat4x4<f32>(c0, c1, c2, c3));
                let base = xbase + c * 64u + 4u * k;
                for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                    let e0 = base + p * 16u;
                    bacc = bacc + dot(vec4<f32>(x[e0], x[e0 + 1u], x[e0 + 2u], x[e0 + 3u]), mt[p]);
                }
            }
        }
        acc = acc + bacc * d;
    }
    out[idx] = acc;
}
"#;

/// Every quant shader must parse and validate under naga with no GPU present.
///
/// Pipeline creation validates shaders too -- but only on the machine that has a device, on the
/// backend it has. A shader that Metal accepts and Vulkan or the browser's WGSL front-end rejects
/// is invisible here until someone runs it there. naga's validator is the shared front-end, so
/// running it in an ordinary test catches the class of "compiles on my laptop" breakage in CI on a
/// runner with no GPU at all.
///
/// This validates STRUCTURE (types, bindings, bounds, undefined behaviour the validator can see)
/// and says nothing about what the shader computes; the kernel tests above carry that.
#[cfg(test)]
mod shader_valid {
    use super::*;
    use wgpu::naga;

    fn check(label: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{label}: WGSL does not parse: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: WGSL does not validate: {e:?}"));
    }

    #[test]
    fn every_quant_shader_validates_without_a_gpu() {
        check("stq1_0 scalar", MATMUL_STQ1_0_WGSL);
        check("stq1_0 vec4", MATMUL_STQ1_0_V4_WGSL);
        check("stq1_0 vec4+table", MATMUL_STQ1_0_V4T_WGSL);
        check("iq2_xxs", &matmul_iq2_xxs_wgsl());
        check("iq3_xxs", &matmul_iq3_xxs_wgsl());
    }
}
