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
        let nblk = rows * (cols / 128);
        let mut codes: Vec<u32> = vec![0; nblk * 8];
        let mut scales: Vec<u32> = vec![0; nblk.div_ceil(2)];
        for b in 0..nblk {
            let src = &bytes[b * 34..b * 34 + 34];
            let d = u16::from_le_bytes([src[0], src[1]]) as u32;
            scales[b / 2] |= d << (16 * (b % 2));
            for w in 0..8 {
                let c = &src[2 + w * 4..2 + w * 4 + 4];
                codes[b * 8 + w] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
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

impl Tensor {
    /// y = x·Wᵀ where W is PrismML Q2_0 ternary held PACKED on the GPU (dequantized per-block on the
    /// fly inside the kernel). x [rows, in] → [rows, out]. This is what makes a 27B ternary model fit.
    pub fn matmul_q2_0(&self, w: &Q2_0Weights) -> Tensor {
        let x = self.contiguous();
        let (rows, inn) = (x.shape[0], x.shape[1]);
        assert_eq!(inn, w.cols, "inner dim mismatch: x[..,{inn}] vs W[..,{}]", w.cols);
        let out = empty(&self.ctx, rows * w.rows);
        let n = rows * w.rows;
        if q2_0_split_k(n) {
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
            run(&self.ctx, MATMUL_Q2_0_FLAT_WGSL, "matmul_q2_0_flat",
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
/// Flat gives one thread per output, so a small output count (decode, one token) leaves the GPU
/// starved and split-K's 64×-more-threads wins. Once the output count is large (prefill), flat
/// already saturates and split-K's barriers are pure overhead. Crossover sits near 16K outputs.
/// `FERRIC_Q2_0_KERNEL=flat|splitk` forces one, for benchmarking.
fn q2_0_split_k(n_out: usize) -> bool {
    match std::env::var("FERRIC_Q2_0_KERNEL").as_deref() {
        Ok("flat") => false,
        Ok("splitk") => true,
        _ => n_out < 16384,
    }
}

/// One thread per output element, walking all of K itself. No barriers, but a long dependent
/// accumulate chain and only `rows·out` threads in flight.
///
/// Dispatched 2D: a 1D grid caps at 65535 workgroups = 4.19M threads, which a real LM head blows
/// straight through (17 tokens × 248320 vocab = 4.22M outputs → 65960 workgroups).
const MATMUL_Q2_0_FLAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x:      array<f32>;
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
        let xbase = r * in_dim + blk * 128u;
        var bacc = 0.0;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word = codes[cbase + w];
            let xb = xbase + w * 16u;
            for (var b: u32 = 0u; b < 16u; b = b + 1u) {
                bacc = bacc + x[xb + b] * f32(i32((word >> (2u * b)) & 3u) - 1);
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
@group(0) @binding(0) var<storage,read>        x:      array<f32>;  // [rows, in]
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
        var acc = 0.0;
        for (var blk: u32 = t; blk < nblk; blk = blk + 64u) {
            let bi = o * nblk + blk;
            let sw = unpack2x16float(scales[bi >> 1u]);
            let d = select(sw.y, sw.x, (bi & 1u) == 0u);
            let cbase = bi * 8u;
            let xbase = r * in_dim + blk * 128u;
            // Sum the ternary-weighted x first, then apply the block scale once (it's constant over
            // the 128-group) — 1 multiply per block instead of 128.
            var bacc = 0.0;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = codes[cbase + w];   // one load feeds 16 weights
                let xb = xbase + w * 16u;
                for (var b: u32 = 0u; b < 16u; b = b + 1u) {
                    let code = (word >> (2u * b)) & 3u;
                    bacc = bacc + x[xb + b] * f32(i32(code) - 1);   // w = (q−1)·d
                }
            }
            acc = acc + bacc * d;
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
