//! Precision / storage dtypes for the fabric. Compute stays f32 (WebGPU-baseline has no shader-f16),
//! but weights can LIVE on the GPU in half precision and be dequantized on-device — half the memory,
//! and the path real fp16/bf16 checkpoints take. `Half` is a packed storage tensor (2 values per u32
//! word); `dequant()` expands to a compute `Tensor`, `Tensor::to_half()` packs one down.

use crate::{empty, groups, run, u32buf, unibuf, Tensor};
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
        run(&self.ctx, QUANTIZE_WGSL, "quantize", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[n as u32, dtype.code()])], groups(words));
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
        run(&self.ctx, QUANT_ROW_WGSL, "quant_row", &[c.buf.as_ref(), scale.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, cols as u32, bits, qmax.to_bits()])], groups(words));
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

/// A packed-quant weight matrix of *any* supported GGUF format, behind one `matmul_q`. This is what
/// makes a model loader format-agnostic: build a `QMatrix` per weight from its ggml type, and the
/// same forward code runs a Q2_0 ternary model, a Q4_K_M model, a Q8_0 model, … — each with its
/// weights dequantized inside the matmul, never expanded to f32.
pub enum QMatrix {
    Q2_0(Q2_0Weights),
    Q4_0(Q4_0Weights),
    Q4_K(Q4_KWeights),
    Q5_K(Q5_KWeights),
    Q6_K(Q6_KWeights),
    Q8_0(Q8_0Weights),
}

impl QMatrix {
    /// ggml block-size in bytes for a supported type, or None if we have no native matmul for it.
    pub fn block_bytes(ggml_type: u32) -> Option<(usize, usize)> {
        match ggml_type {          // (values per block, bytes per block)
            2 => Some((32, 18)),   // Q4_0
            8 => Some((32, 34)),   // Q8_0
            12 => Some((256, 144)),// Q4_K
            13 => Some((256, 176)),// Q5_K
            14 => Some((256, 210)),// Q6_K
            42 => Some((128, 34)), // Q2_0
            _ => None,
        }
    }
    /// Build from raw GGUF block bytes for an [out(rows), in(cols)] weight.
    pub fn from_bytes(ctx: &Arc<Context>, bytes: &[u8], ggml_type: u32, rows: usize, cols: usize) -> Result<QMatrix, String> {
        Ok(match ggml_type {
            2 => QMatrix::Q4_0(Q4_0Weights::from_bytes(ctx, bytes, rows, cols)),
            8 => QMatrix::Q8_0(Q8_0Weights::from_bytes(ctx, bytes, rows, cols)),
            12 => QMatrix::Q4_K(Q4_KWeights::from_bytes(ctx, bytes, rows, cols)),
            13 => QMatrix::Q5_K(Q5_KWeights::from_bytes(ctx, bytes, rows, cols)),
            14 => QMatrix::Q6_K(Q6_KWeights::from_bytes(ctx, bytes, rows, cols)),
            42 => QMatrix::Q2_0(Q2_0Weights::from_bytes(ctx, bytes, rows, cols)),
            other => return Err(format!("QMatrix: no native matmul for ggml type {other}")),
        })
    }
    pub fn rows(&self) -> usize { match self { QMatrix::Q2_0(w) => w.rows, QMatrix::Q4_0(w) => w.rows, QMatrix::Q4_K(w) => w.rows, QMatrix::Q5_K(w) => w.rows, QMatrix::Q6_K(w) => w.rows, QMatrix::Q8_0(w) => w.rows } }
    pub fn cols(&self) -> usize { match self { QMatrix::Q2_0(w) => w.cols, QMatrix::Q4_0(w) => w.cols, QMatrix::Q4_K(w) => w.cols, QMatrix::Q5_K(w) => w.cols, QMatrix::Q6_K(w) => w.cols, QMatrix::Q8_0(w) => w.cols } }
    pub fn nbytes(&self) -> usize { match self { QMatrix::Q2_0(w) => w.nbytes(), QMatrix::Q4_0(w) => w.nbytes(), QMatrix::Q4_K(w) => w.nbytes(), QMatrix::Q5_K(w) => w.nbytes(), QMatrix::Q6_K(w) => w.nbytes(), QMatrix::Q8_0(w) => w.nbytes() } }
}

impl Tensor {
    /// y = x·Wᵀ for a packed weight of any supported format (dispatches to the format's kernel).
    pub fn matmul_q(&self, w: &QMatrix) -> Tensor {
        match w {
            QMatrix::Q2_0(w) => self.matmul_q2_0(w),
            QMatrix::Q4_0(w) => self.matmul_q4_0(w),
            QMatrix::Q4_K(w) => self.matmul_q4_k(w),
            QMatrix::Q5_K(w) => self.matmul_q5_k(w),
            QMatrix::Q6_K(w) => self.matmul_q6_k(w),
            QMatrix::Q8_0(w) => self.matmul_q8_0(w),
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
        let src = wgsl.replace("__HELPERS__", Q4_K_HELPERS).replace("__INNER__", Q5_K_INNER);
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
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
        let src = wgsl.replace("__HELPERS__", Q6_K_HELPERS).replace("__BODY__", Q6_K_BODY);
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

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
        run(&self.ctx, wgsl, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }
}

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
        Q4_KWeights { ctx: ctx.clone(), codes: mk("q4k.codes", &codes), aux: mk("q4k.aux", &aux), rows, cols }
    }
    pub fn nbytes(&self) -> usize { self.rows * (self.cols / 256) * 144 }
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
        let src = wgsl.replace("__HELPERS__", Q4_K_HELPERS).replace("__INNER__", Q4_K_INNER);
        run(&self.ctx, &src, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.aux.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
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
        run(&self.ctx, wgsl, label,
            &[x.buf.as_ref(), w.codes.as_ref(), w.scales.as_ref(), &out,
              &unibuf(&self.ctx, &[rows as u32, w.rows as u32, inn as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![rows, w.rows])
    }

    /// y = x·Wᵀ where W is PrismML Q2_0 ternary held PACKED on the GPU (dequantized per-block on the
    /// fly inside the kernel). x [rows, in] → [rows, out]. This is what makes a 27B ternary model fit.
    pub fn matmul_q2_0(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        if q2_0_split_k(rows, w.rows) {
            // One workgroup per output element, laid out 2D because rows·out overruns the 65535
            // per-dimension cap (e.g. 5 tokens × 17408 outputs).
            let grid_w = n.min(32768);
            let grid_h = n.div_ceil(grid_w);
            run(&self.ctx, MATMUL_Q2_0_SPLITK_WGSL, "matmul_q2_0_splitk",
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
}

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
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
            if (t < s) { partial[t] = partial[t] + partial[t + s]; }
            workgroupBarrier();
        }
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
@group(0) @binding(3) var<storage,read>        info: array<u32>;  // rows, cols, bits, bitcast(qmax)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x; let rows = info[0]; let cols = info[1]; let bits = info[2]; let qmax = bitcast<f32>(info[3]);
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
@group(0) @binding(2) var<storage,read>        info: array<u32>; // n, kind
fn bf16_rne(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let r = b + 0x7fffu + ((b >> 16u) & 1u); // round-to-nearest-even bias
    return r >> 16u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x; let n = info[0]; let kind = info[1];
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
