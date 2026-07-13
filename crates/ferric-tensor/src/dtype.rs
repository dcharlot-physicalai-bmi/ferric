//! Precision / storage dtypes for the fabric. Compute stays f32 (WebGPU-baseline has no shader-f16),
//! but weights can LIVE on the GPU in half precision and be dequantized on-device — half the memory,
//! and the path real fp16/bf16 checkpoints take. `Half` is a packed storage tensor (2 values per u32
//! word); `dequant()` expands to a compute `Tensor`, `Tensor::to_half()` packs one down.

use crate::{empty, groups, run, u32buf, Tensor};
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
