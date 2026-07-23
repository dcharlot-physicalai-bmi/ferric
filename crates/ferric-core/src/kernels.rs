//! Ferric L1 kernel set toward a transformer forward pass: elementwise add, SiLU, LayerNorm, Softmax.
//! Each is one WGSL kernel written once, run on any fabric (native GPU + browser WebGPU), and
//! validated against the plain-Rust CPU reference in `cpu`. Composed with matmul (in lib.rs), these
//! cover the bulk of a transformer block; attention is matmul + softmax + matmul.

use crate::{Context, Result, Tensor};

/// On-GPU tensor ops — the L3 graph-executor primitives. Each dispatches and returns a Tensor whose
/// buffer stays on the device; nothing reads back until `to_vec`, so a whole model runs on-GPU.
impl Context {
    /// Upload host data to a GPU tensor.
    pub fn tensor(&self, data: &[f32], shape: &[usize]) -> Tensor {
        Tensor { buf: self.storage("t", data), shape: shape.to_vec() }
    }
    /// Allocate an empty (device-only) tensor.
    pub fn empty(&self, shape: &[usize]) -> Tensor {
        let len: usize = shape.iter().product();
        Tensor { buf: self.out_buffer(len), shape: shape.to_vec() }
    }
    /// Read a tensor back to host (the only sync point).
    pub async fn to_vec(&self, t: &Tensor) -> Result<Vec<f32>> { self.readback(&t.buf, t.len()).await }

    /// C = A(m×k)·B(k×n), on-GPU.
    pub fn mm(&self, a: &Tensor, b: &Tensor, m: u32, k: u32, n: u32) -> Tensor {
        let out = self.empty(&[m as usize, n as usize]);
        let dims = self.uniform_u32("d", &[m, k, n, 0]);
        let pipe = self.pipeline("mm", crate::MATMUL_WGSL);
        self.dispatch(&pipe, &[&a.buf, &b.buf, &out.buf, &dims], ((m + 15) / 16, (n + 15) / 16, 1));
        out
    }
    /// C = scale·(A(m×k)·B(n×k)ᵀ), on-GPU.
    pub fn mm_bt(&self, a: &Tensor, b: &Tensor, m: u32, n: u32, k: u32, scale: f32) -> Tensor {
        let out = self.empty(&[m as usize, n as usize]);
        let dims = self.uniform_u32("d", &[m, n, k, scale.to_bits()]);
        let pipe = self.pipeline("mm_bt", MATMUL_BT_WGSL);
        self.dispatch(&pipe, &[&a.buf, &b.buf, &out.buf, &dims], ((m + 15) / 16, (n + 15) / 16, 1));
        out
    }
    pub fn silu_t(&self, x: &Tensor) -> Tensor {
        let n = x.len();
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("silu", SILU_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    /// Reinterpret a tensor's data under a new shape (contiguous row-major) — Reshape/Cast/Squeeze.
    pub fn dup(&self, x: &Tensor, shape: Vec<usize>) -> Tensor {
        Tensor { buf: self.copy_buf(&x.buf, x.len()), shape }
    }
    /// Row-gather along axis 0: out[i, :] = data[idx[i], :]. Embedding lookup / index_select.
    pub fn gather0(&self, data: &Tensor, idx: &[u32], d: usize) -> Tensor {
        let n = idx.len();
        let out = self.empty(&[n, d]);
        let idx_buf = self.storage_u32("idx", idx);
        let dims = self.uniform_u32("d", &[n as u32, d as u32, 0, 0]);
        let pipe = self.pipeline("gather", GATHER_WGSL);
        self.dispatch(&pipe, &[&data.buf, &idx_buf, &out.buf, &dims], ((n * d) as u32 / 64 + 1, 1, 1));
        out
    }
    pub fn sigmoid_t(&self, x: &Tensor) -> Tensor { self.unary(x, SIGMOID_WGSL, "sigmoid") }
    pub fn sqrt_t(&self, x: &Tensor) -> Tensor { self.unary(x, SQRT_WGSL, "sqrt") }
    pub fn gelu_t(&self, x: &Tensor) -> Tensor { self.unary(x, GELU_WGSL, "gelu") }
    pub fn sub_t(&self, a: &Tensor, b: &Tensor) -> Tensor { self.binary(a, b, SUB_WGSL, "sub") }
    pub fn div_t(&self, a: &Tensor, b: &Tensor) -> Tensor { self.binary(a, b, DIV_WGSL, "div") }
    fn unary(&self, x: &Tensor, wgsl: &str, name: &str) -> Tensor {
        let n = x.len();
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline(name, wgsl);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    fn binary(&self, a: &Tensor, b: &Tensor, wgsl: &str, name: &str) -> Tensor {
        let n = a.len();
        let out = self.empty(&a.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline(name, wgsl);
        self.dispatch(&pipe, &[&a.buf, &b.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    /// Elementwise C = A ⊙ B.
    pub fn mul_t(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let n = a.len();
        let out = self.empty(&a.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("mul", MUL_WGSL);
        self.dispatch(&pipe, &[&a.buf, &b.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    /// C = A · scalar, where `s` is a 1-element tensor (broadcast).
    pub fn mul_scalar_t(&self, a: &Tensor, s: &Tensor) -> Tensor {
        let n = a.len();
        let out = self.empty(&a.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("mul_scalar", MUL_SCALAR_WGSL);
        self.dispatch(&pipe, &[&a.buf, &s.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    /// 2D transpose [rows×cols] → [cols×rows].
    pub fn transpose2d_t(&self, x: &Tensor, rows: u32, cols: u32) -> Tensor {
        let out = self.empty(&[cols as usize, rows as usize]);
        let dims = self.uniform_u32("d", &[rows, cols, 0, 0]);
        let pipe = self.pipeline("transpose", TRANSPOSE_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims], ((rows + 15) / 16, (cols + 15) / 16, 1));
        out
    }
    /// C = A + bias broadcast per row: out[i] = a[i] + bias[i % d]. (a is [.,d], bias is [d].)
    pub fn add_bias_t(&self, a: &Tensor, bias: &Tensor) -> Tensor {
        let (n, d) = (a.len(), bias.len());
        let out = self.empty(&a.shape);
        let dims = self.uniform_u32("d", &[n as u32, d as u32, 0, 0]);
        let pipe = self.pipeline("add_bias", ADD_BIAS_WGSL);
        self.dispatch(&pipe, &[&a.buf, &bias.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    pub fn relu_t(&self, x: &Tensor) -> Tensor {
        let n = x.len();
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("relu", RELU_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    pub fn add_t(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let n = a.len();
        let out = self.empty(&a.shape);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("add", ADD_WGSL);
        self.dispatch(&pipe, &[&a.buf, &b.buf, &out.buf, &dims], ((n as u32 + 63) / 64, 1, 1));
        out
    }
    pub fn softmax_t(&self, x: &Tensor, rows: u32, d: u32) -> Tensor {
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, 0, 0]);
        // det-scratch: storage slot for the exp-sum
        let scr = self.out_buffer(rows as usize * 2);
        let pipe = self.pipeline("softmax", SOFTMAX_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims, &scr], ((rows + 63) / 64, 1, 1));
        out
    }
    /// Rotary position embedding (rotate-half, NeoX/Llama style). x is [T, H·dh]; rotates each head's
    /// dh-vector by position-dependent angles. Applied to Q and K before attention.
    pub fn rope_t(&self, x: &Tensor, t: u32, h: u32, dh: u32, base: f32) -> Tensor {
        self.rope_off_t(x, t, h, dh, base, 0)
    }
    /// RoPE with an absolute-position offset: row i is rotated for position (i + offset). Prefill uses
    /// offset 0; incremental decode of the token at position `pos` uses a 1-row input with offset=pos.
    pub fn rope_off_t(&self, x: &Tensor, t: u32, h: u32, dh: u32, base: f32, offset: u32) -> Tensor {
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[t, h, dh, base.to_bits()]);
        let meta = self.uniform_u32("m", &[offset, 0, 0, 0]);
        let pipe = self.pipeline("rope", ROPE_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims, &meta], (t * h / 64 + 1, 1, 1));
        out
    }
    /// Attention for one query token against a KV cache of `s` past keys/values (incremental decode).
    /// q is [1, H·dh]; k/v are [s, H·dh]; no mask (all cached keys precede the query). Returns [1, H·dh].
    pub fn mha_decode_t(&self, q: &Tensor, k: &Tensor, v: &Tensor, hq: u32, hkv: u32, dh: u32, s: u32) -> Tensor {
        let out = self.empty(&q.shape);
        let scale = 1.0f32 / (dh as f32).sqrt();
        let dims = self.uniform_u32("d", &[s, hq, dh, scale.to_bits()]);
        let gqa = self.uniform_u32("g", &[hkv, 0, 0, 0]);
        let pipe = self.pipeline("mha_decode", MHA_DECODE_WGSL);
        self.dispatch(&pipe, &[&q.buf, &k.buf, &v.buf, &out.buf, &dims, &gqa], (hq / 64 + 1, 1, 1));
        out
    }
    /// Causal multi-head attention, single flash-style pass, with grouped-query attention (GQA): Q has
    /// `hq` heads, K/V have `hkv` heads (hq % hkv == 0); query head h reads kv head h/(hq/hkv). q is
    /// [T, hq·dh]; k/v are [T, hkv·dh]; scale = 1/√dh; query i attends to keys 0..=i. Returns [T, hq·dh].
    pub fn mha_causal_t(&self, q: &Tensor, k: &Tensor, v: &Tensor, t: u32, hq: u32, hkv: u32, dh: u32) -> Tensor {
        let out = self.empty(&q.shape);
        let scale = 1.0f32 / (dh as f32).sqrt();
        let dims = self.uniform_u32("d", &[t, hq, dh, scale.to_bits()]);
        let gqa = self.uniform_u32("g", &[hkv, 0, 0, 0]);
        let pipe = self.pipeline("mha", MHA_CAUSAL_WGSL);
        self.dispatch(&pipe, &[&q.buf, &k.buf, &v.buf, &out.buf, &dims, &gqa], (t * hq / 64 + 1, 1, 1));
        out
    }
    /// RMSNorm (Llama/SmolVLA norm): out = x / sqrt(mean(x²)+eps) · weight. No mean-subtraction, no bias.
    pub fn rmsnorm_t(&self, x: &Tensor, w: &Tensor, rows: u32, d: u32, eps: f32) -> Tensor {
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, eps.to_bits(), 0]);
        // det-scratch: storage-space slots for the rsqrt Newton chain
        let scr = self.out_buffer(rows as usize * 16);
        let pipe = self.pipeline("rmsnorm", RMSNORM_WGSL);
        self.dispatch(&pipe, &[&x.buf, &w.buf, &out.buf, &dims, &scr], (rows, 1, 1));
        out
    }
    pub fn layernorm_t(&self, x: &Tensor, w: &Tensor, b: &Tensor, rows: u32, d: u32, eps: f32) -> Tensor {
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, eps.to_bits(), 0]);
        // det-scratch: storage-space slots for the per-row scalar chain
        let scr = self.out_buffer(rows as usize * 16);
        let pipe = self.pipeline("layernorm", LAYERNORM_WGSL);
        self.dispatch(&pipe, &[&x.buf, &w.buf, &b.buf, &out.buf, &dims, &scr], ((rows + 63) / 64, 1, 1));
        out
    }
    /// Single-head attention on-GPU: softmax(scale·Q·Kᵀ)·V, all buffers stay on device.
    pub fn attention_t(&self, q: &Tensor, k: &Tensor, v: &Tensor, rq: u32, rk: u32, d: u32, dv: u32, scale: f32) -> Tensor {
        let scores = self.mm_bt(q, k, rq, rk, d, scale);
        let probs = self.softmax_t(&scores, rq, rk);
        self.mm(&probs, v, rq, rk, dv)
    }
}

impl Context {
    /// Elementwise C = A + B (f32).
    pub async fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let (ab, bb) = (self.storage("a", a), self.storage("b", b));
        let out = self.out_buffer(n);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("add", ADD_WGSL);
        self.dispatch(&pipe, &[&ab, &bb, &out, &dims], ((n as u32 + 63) / 64, 1, 1));
        self.readback(&out, n).await
    }

    /// SiLU / swish: x · sigmoid(x).
    pub async fn silu(&self, x: &[f32]) -> Result<Vec<f32>> {
        let n = x.len();
        let xb = self.storage("x", x);
        let out = self.out_buffer(n);
        let dims = self.uniform_u32("n", &[n as u32, 0, 0, 0]);
        let pipe = self.pipeline("silu", SILU_WGSL);
        self.dispatch(&pipe, &[&xb, &out, &dims], ((n as u32 + 63) / 64, 1, 1));
        self.readback(&out, n).await
    }

    /// LayerNorm over the last dim `d` for `rows` rows: (x−μ)/√(σ²+eps) · weight + bias.
    pub async fn layernorm(&self, x: &[f32], weight: &[f32], bias: &[f32], rows: u32, d: u32, eps: f32) -> Result<Vec<f32>> {
        assert_eq!(x.len(), (rows * d) as usize);
        assert_eq!(weight.len(), d as usize);
        assert_eq!(bias.len(), d as usize);
        let (xb, wb, bb) = (self.storage("x", x), self.storage("w", weight), self.storage("b", bias));
        let out = self.out_buffer((rows * d) as usize);
        let dims = self.uniform_u32("dims", &[rows, d, eps.to_bits(), 0]);
        let scr = self.out_buffer(rows as usize * 16);
        let pipe = self.pipeline("layernorm", LAYERNORM_WGSL);
        self.dispatch(&pipe, &[&xb, &wb, &bb, &out, &dims, &scr], ((rows + 63) / 64, 1, 1));
        self.readback(&out, (rows * d) as usize).await
    }

    /// Row-wise softmax over the last dim `d` for `rows` rows (numerically stable).
    pub async fn softmax(&self, x: &[f32], rows: u32, d: u32) -> Result<Vec<f32>> {
        assert_eq!(x.len(), (rows * d) as usize);
        let xb = self.storage("x", x);
        let out = self.out_buffer((rows * d) as usize);
        let dims = self.uniform_u32("dims", &[rows, d, 0, 0]);
        let scr = self.out_buffer(rows as usize * 2);
        let pipe = self.pipeline("softmax", SOFTMAX_WGSL);
        self.dispatch(&pipe, &[&xb, &out, &dims, &scr], ((rows + 63) / 64, 1, 1));
        self.readback(&out, (rows * d) as usize).await
    }

    /// C = scale · (A(m×k) · B(n×k)ᵀ). Row-major; B is [n×k] (not transposed in memory). This is the
    /// Q·Kᵀ shape for attention, with the 1/√d scale folded in.
    pub async fn matmul_bt(&self, a: &[f32], b: &[f32], m: u32, n: u32, k: u32, scale: f32) -> Result<Vec<f32>> {
        assert_eq!(a.len(), (m * k) as usize);
        assert_eq!(b.len(), (n * k) as usize);
        let (ab, bb) = (self.storage("a", a), self.storage("b", b));
        let out = self.out_buffer((m * n) as usize);
        let dims = self.uniform_u32("dims", &[m, n, k, scale.to_bits()]);
        let pipe = self.pipeline("matmul_bt", MATMUL_BT_WGSL);
        self.dispatch(&pipe, &[&ab, &bb, &out, &dims], ((m + 15) / 16, (n + 15) / 16, 1));
        self.readback(&out, (m * n) as usize).await
    }

    /// Single-head scaled dot-product attention: softmax(scale · Q·Kᵀ) · V.
    /// Q[rows_q×d], K[rows_k×d], V[rows_k×dv] → [rows_q×dv]. Composed from matmul_bt + softmax + matmul.
    pub async fn attention(&self, q: &[f32], k: &[f32], v: &[f32], rows_q: u32, rows_k: u32, d: u32, dv: u32, scale: f32) -> Result<Vec<f32>> {
        let scores = self.matmul_bt(q, k, rows_q, rows_k, d, scale).await?; // [rows_q × rows_k], scaled
        let probs = self.softmax(&scores, rows_q, rows_k).await?;
        self.matmul(&probs, v, rows_q, rows_k, dv).await                    // [rows_q × dv]
    }
}

const MATMUL_BT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // m, n, k, bitcast(scale)
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = dims.x; let n = dims.y; let k = dims.z; let scale = bitcast<f32>(dims.w);
    let row = gid.x; let col = gid.y;
    if (row >= m || col >= n) { return; }
    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < k; l = l + 1u) {
        acc = acc + a[row * k + l] * b[col * k + l];   // A · Bᵀ
    }
    out[row * n + col] = acc * scale;
}
"#;

const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }
    out[i] = a[i] + b[i];
}
"#;

const ADD_BIAS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // n, d
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = a[i] + bias[i % dims.y];
}
"#;

const MUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = a[i] * b[i];
}
"#;

const MUL_SCALAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       s: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = a[i] * s[0];
}
"#;

const TRANSPOSE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // rows, cols
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = dims.x; let cols = dims.y;
    let r = gid.x; let c = gid.y;
    if (r >= rows || c >= cols) { return; }
    out[c * rows + r] = x[r * cols + c];   // [rows,cols] → [cols,rows]
}
"#;

const RELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }
    out[i] = max(x[i], 0.0);
}
"#;

// ─── det-math: deterministic transcendentals for cross-fabric bit-identity ───
// The fabric_probe measurement showed builtin exp/sin/cos/sqrt AND division
// diverge between Metal and Vulkan, while the matmul kernel (dependent mul-add
// chains) is bit-identical. These helpers therefore use ONLY +,−,× dependent
// chains, integer bit ops, floor, and comparisons — no builtin transcendental,
// no division — so every kernel built on them inherits matmul's cross-fabric
// bit-identity, at ~1 ULP accuracy cost vs libm. Prepended automatically by
// `Context::pipeline` when a kernel source references `det_`.
pub(crate) const DET_MATH_WGSL: &str = r#"
// Optimization barrier: round-trip through the integer domain XORed with a
// RUNTIME zero (an unused uniform slot). Metal's compiler folds `^ 0u`
// (compile-time constant) and keeps fma-fusing across it — an unprovable
// runtime value it cannot fold, so both fabrics are forced to the plain
// rounded mul/add sequence. Free at runtime; priceless for bit-identity.
fn det_bar(v: f32, z: u32) -> f32 { return bitcast<f32>(bitcast<u32>(v) ^ z); }
// PORTABLE-DET: every det function is barriered at each step, so even
// compilers we cannot configure (the browser's Dawn/Tint) are forced to the
// plain rounded sequence. Under strict compilers the barriers are value
// no-ops — native digests do not move; contracted compilers converge onto
// them. Operation ORDER matches the original expressions exactly.
fn det_recip(y: f32, z: u32) -> f32 {
    var x = bitcast<f32>(0x7EF311C3u - bitcast<u32>(y));
    var t = det_bar(y * x, z);
    var w = det_bar(2.0 - t, z);
    x = det_bar(x * w, z);
    t = det_bar(y * x, z);
    w = det_bar(2.0 - t, z);
    x = det_bar(x * w, z);
    t = det_bar(y * x, z);
    w = det_bar(2.0 - t, z);
    x = det_bar(x * w, z);
    return x;
}
fn det_rsqrt(y: f32, z: u32) -> f32 {
    // The Newton chain is pinned through a private array indexed at
    // runtime-opaque offsets (z is 0 at runtime, unprovable at compile time,
    // and every index expression differs) — stores can't be forwarded or
    // elided, so every intermediate is a real rounded value. The stage-by-
    // stage browser forensic proved this exact structure computes plain-IEEE
    // on all inputs under Tint/ANGLE, where XOR-only barriers on a register
    // chain do not survive fast-math. `z` = a spare uniform slot.
    var m: array<f32, 16>;
    let hy = 0.5 * y;
    m[z] = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(y) >> 1u));
    m[z + 1u] = m[z] * m[z];
    m[z + 2u] = hy * m[z + 1u];
    m[z + 3u] = 1.5 - m[z + 2u];
    m[z + 4u] = m[z] * m[z + 3u];
    m[z + 5u] = m[z + 4u] * m[z + 4u];
    m[z + 6u] = hy * m[z + 5u];
    m[z + 7u] = 1.5 - m[z + 6u];
    m[z + 8u] = m[z + 4u] * m[z + 7u];
    m[z + 9u] = m[z + 8u] * m[z + 8u];
    m[z + 10u] = hy * m[z + 9u];
    m[z + 11u] = 1.5 - m[z + 10u];
    m[z + 12u] = m[z + 8u] * m[z + 11u];
    return m[z + 12u];
}
fn det_sqrt(y: f32, z: u32) -> f32 {
    if (y <= 0.0) { return 0.0; }
    let rb = det_bar(det_rsqrt(y, z), z);
    return y * rb;
}
// e^v via Cody-Waite ln2 split (exact for |k|<256) + Taylor-6 on [-0.347,0.347].
fn det_exp(v: f32, z: u32) -> f32 {
    let x = clamp(v, -87.0, 88.0);
    let kf = floor(det_bar(x * 1.4426950216293335, z) + 0.5);
    let a = det_bar(x - det_bar(kf * 0.693115234375, z), z);
    let r = det_bar(a - det_bar(kf * 3.194618329871446e-05, z), z);
    var p = 0.0013888889;
    p = det_bar(det_bar(p * r, z) + 0.008333334, z);
    p = det_bar(det_bar(p * r, z) + 0.041666668, z);
    p = det_bar(det_bar(p * r, z) + 0.16666667, z);
    p = det_bar(det_bar(p * r, z) + 0.5, z);
    p = det_bar(det_bar(p * r, z) + 1.0, z);
    p = det_bar(det_bar(p * r, z) + 1.0, z);
    let scale = bitcast<f32>(u32(clamp(i32(kf), -126, 127) + 127) << 23u);
    return p * scale;
}
fn det_sin_poly(r: f32, z: u32) -> f32 {
    let r2 = det_bar(r * r, z);
    var p = 2.7557319e-6;
    p = det_bar(det_bar(p * r2, z) + -1.9841270e-4, z);
    p = det_bar(det_bar(p * r2, z) + 0.008333334, z);
    p = det_bar(det_bar(p * r2, z) + -0.16666667, z);
    p = det_bar(det_bar(p * r2, z) + 1.0, z);
    return r * p;
}
fn det_cos_poly(r: f32, z: u32) -> f32 {
    let r2 = det_bar(r * r, z);
    var p = 2.4801587e-5;
    p = det_bar(det_bar(p * r2, z) + -0.0013888889, z);
    p = det_bar(det_bar(p * r2, z) + 0.041666668, z);
    p = det_bar(det_bar(p * r2, z) + -0.5, z);
    p = det_bar(det_bar(p * r2, z) + 1.0, z);
    return p;
}
// (sin, cos) via 3-term π/2 Cody-Waite reduction — exact for |q| < 4096.
fn det_sincos(ang: f32, z: u32) -> vec2<f32> {
    let qf = floor(det_bar(ang * 0.6366197466850281, z) + 0.5);
    var r = det_bar(ang - det_bar(qf * 1.5703125, z), z);
    r = det_bar(r - det_bar(qf * 4.8382679e-4, z), z);
    r = det_bar(r - det_bar(qf * 2.5632829e-12, z), z);
    let q = i32(qf) & 3;
    let s = det_sin_poly(r, z);
    let c = det_cos_poly(r, z);
    if (q == 0) { return vec2<f32>(s, c); }
    if (q == 1) { return vec2<f32>(c, -s); }
    if (q == 2) { return vec2<f32>(-s, -c); }
    return vec2<f32>(-c, s);
}
// ln(y), y > 0: exponent split + atanh series on the mantissa (s ≤ 1/3).
fn det_ln(y: f32, z: u32) -> f32 {
    let bits = bitcast<u32>(y);
    let e = i32(bits >> 23u) - 127;
    let m = bitcast<f32>((bits & 0x007FFFFFu) | 0x3F800000u);
    let s = det_bar((m - 1.0) * det_recip(m + 1.0, z), z);
    let s2 = det_bar(s * s, z);
    var p = 0.11111111;
    p = det_bar(det_bar(p * s2, z) + 0.14285715, z);
    p = det_bar(det_bar(p * s2, z) + 0.2, z);
    p = det_bar(det_bar(p * s2, z) + 0.33333334, z);
    p = det_bar(det_bar(p * s2, z) + 1.0, z);
    let ts = det_bar(2.0 * s, z);
    let lnm = det_bar(ts * p, z);
    return det_bar(det_bar(f32(e) * 0.6931471824645996, z) + lnm, z);
}
"#;

const SILU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }
    let v = x[i];
    out[i] = v * det_recip(1.0 + det_exp(-v, dims.y), dims.y);
}
"#;

const GATHER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       data: array<f32>;
@group(0) @binding(1) var<storage, read>       idx:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // n, d
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dims.x; let d = dims.y;
    let t = gid.x; if (t >= n * d) { return; }
    let i = t / d; let j = t % d;
    out[i * d + j] = data[idx[i] * d + j];
}
"#;

const SIGMOID_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = det_recip(1.0 + det_exp(-x[i], dims.y), dims.y);
}
"#;

const SQRT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = det_sqrt(x[i], dims.y);
}
"#;

// Exact erf-based GELU: 0.5·x·(1+erf(x/√2)). WGSL has no erf, so use a high-accuracy
// Abramowitz-Stegun 7.1.26 rational approximation (|err| < 1.5e-7) — matches ONNX Gelu (erf).
const GELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
fn erf(zz: f32, z: u32) -> f32 {
    let s = sign(zz); let a = abs(zz);
    let t = det_recip(1.0 + det_bar(0.3275911 * a, z), z);
    var p = det_bar(det_bar(1.061405429 * t, z) - 1.453152027, z);
    p = det_bar(det_bar(p * t, z) + 1.421413741, z);
    p = det_bar(det_bar(p * t, z) - 0.284496736, z);
    p = det_bar(det_bar(p * t, z) + 0.254829592, z);
    var q = det_bar(p * t, z);
    q = det_bar(q * det_exp(-det_bar(a * a, z), z), z);
    let y = 1.0 - q;
    return s * y;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    let v = x[i];
    let h = det_bar(0.5 * v, dims.y);
    out[i] = h * (1.0 + erf(v * 0.7071067811865476, dims.y));
}
"#;

const SUB_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = a[i] - b[i];
}
"#;

const DIV_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = a[i] / b[i];
}
"#;

const ROPE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // T, H, dh, bitcast(base)
@group(0) @binding(3) var<uniform>             rmeta: vec4<u32>; // pos_offset, _, _, _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = dims.x; let h = dims.y; let dh = dims.z; let base = bitcast<f32>(dims.w);
    let id = gid.x; if (id >= t * h) { return; }
    let i = id / h; let head = id % h;
    let half = dh / 2u;
    let o = (i * h + head) * dh;
    let lb = det_ln(base, rmeta.y);
    let inv_dh = det_recip(f32(dh), rmeta.y);
    for (var c: u32 = 0u; c < half; c = c + 1u) {
        var e1 = det_bar(-2.0 * f32(c), rmeta.y);
        e1 = det_bar(e1 * inv_dh, rmeta.y);
        e1 = det_bar(e1 * lb, rmeta.y);
        let inv = det_exp(e1, rmeta.y);
        let ang = f32(i + rmeta.x) * inv;
        let sc = det_sincos(ang, rmeta.y);
        let cs = sc.y; let sn = sc.x;
        let x1 = x[o + c]; let x2 = x[o + c + half];
        out[o + c] = det_bar(x1 * cs, rmeta.y) - det_bar(x2 * sn, rmeta.y);
        out[o + c + half] = det_bar(x2 * cs, rmeta.y) + det_bar(x1 * sn, rmeta.y);
    }
}
"#;

const MHA_CAUSAL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             dims: vec4<u32>; // T, Hq, dh, bitcast(scale)
@group(0) @binding(5) var<uniform>             gqa:  vec4<u32>; // Hkv, _, _, _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = dims.x; let hq = dims.y; let dh = dims.z; let scale = bitcast<f32>(dims.w);
    let hkv = gqa.x;
    let id = gid.x; if (id >= t * hq) { return; }
    let i = id / hq; let head = id % hq;
    let kvhead = head / (hq / hkv);                       // GQA: query head → shared kv head
    let qo = (i * hq + head) * dh;
    var acc: array<f32, 128>;
    for (var c: u32 = 0u; c < dh; c = c + 1u) { acc[c] = 0.0; }
    // memory-pinned scratch (see det_rsqrt note): every product is stored and
    // re-loaded at a runtime-opaque index before its add — adds see two
    // LOADS, so no compiler can contract them into fma.
    var sc_: array<f32, 16>;
    let zz = gqa.y;
    var m: f32 = -3.0e38; var l: f32 = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {           // causal: attend to keys 0..=i
        let ko = (j * hkv + kvhead) * dh;
        sc_[zz] = 0.0;
        for (var c: u32 = 0u; c < dh; c = c + 1u) {
            sc_[zz + 1u] = q[qo + c] * k[ko + c];
            sc_[zz] = sc_[zz] + sc_[zz + 1u];
        }
        let s = sc_[zz] * scale;
        let mnew = max(m, s);
        let corr = det_exp(m - mnew, zz);
        let p = det_exp(s - mnew, zz);
        sc_[zz + 2u] = l * corr;
        l = sc_[zz + 2u] + p;
        for (var c: u32 = 0u; c < dh; c = c + 1u) {
            sc_[zz + 3u] = p * v[ko + c];
            sc_[zz + 4u] = acc[c] * corr;
            acc[c] = sc_[zz + 4u] + sc_[zz + 3u];
        }
        m = mnew;
    }
    let invl = det_recip(l, zz);
    for (var c: u32 = 0u; c < dh; c = c + 1u) { out[qo + c] = acc[c] * invl; }
}
"#;

const MHA_DECODE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       q: array<f32>;   // [1, H*dh]
@group(0) @binding(1) var<storage, read>       k: array<f32>;   // [S, H*dh]
@group(0) @binding(2) var<storage, read>       v: array<f32>;   // [S, H*dh]
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [1, H*dh]
@group(0) @binding(4) var<uniform>             dims: vec4<u32>; // S, Hq, dh, bitcast(scale)
@group(0) @binding(5) var<uniform>             gqa:  vec4<u32>; // Hkv, _, _, _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = dims.x; let hq = dims.y; let dh = dims.z; let scale = bitcast<f32>(dims.w);
    let hkv = gqa.x;
    let head = gid.x; if (head >= hq) { return; }
    let kvhead = head / (hq / hkv);           // GQA: query head → shared kv head
    let qo = head * dh;
    var acc: array<f32, 128>;
    for (var c: u32 = 0u; c < dh; c = c + 1u) { acc[c] = 0.0; }
    // memory-pinned scratch (see MHA_CAUSAL note)
    var scm: array<f32, 16>;
    let zz = gqa.y;
    var m: f32 = -3.0e38; var l: f32 = 0.0;
    for (var j: u32 = 0u; j < s; j = j + 1u) {
        let ko = (j * hkv + kvhead) * dh;
        scm[zz] = 0.0;
        for (var c: u32 = 0u; c < dh; c = c + 1u) {
            scm[zz + 1u] = q[qo + c] * k[ko + c];
            scm[zz] = scm[zz] + scm[zz + 1u];
        }
        let sc = scm[zz] * scale;
        let mnew = max(m, sc);
        let corr = det_exp(m - mnew, zz);
        let p = det_exp(sc - mnew, zz);
        scm[zz + 2u] = l * corr;
        l = scm[zz + 2u] + p;
        for (var c: u32 = 0u; c < dh; c = c + 1u) {
            scm[zz + 3u] = p * v[ko + c];
            scm[zz + 4u] = acc[c] * corr;
            acc[c] = scm[zz + 4u] + scm[zz + 3u];
        }
        m = mnew;
    }
    let invl = det_recip(l, zz);
    for (var c: u32 = 0u; c < dh; c = c + 1u) { out[qo + c] = acc[c] * invl; }
}
"#;

const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // rows, d, bitcast(eps), _
@group(0) @binding(4) var<storage, read_write> scr: array<f32>; // rows*16 det-scratch
var<workgroup> wacc: array<f32, 64>;
var<workgroup> wprod: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y; let eps = bitcast<f32>(dims.z);
    if (row >= rows) { return; }
    let base = row * d;
    // Sum and product go through WORKGROUP memory — the dynamic-trip loop
    // blocks register promotion there (probe-verified clean). The straight-
    // line rsqrt chain below goes through STORAGE scratch instead: barrier-
    // free workgroup slots in straight-line code DO get promoted (row-10
    // forensic, 2 ULP), but device memory is host-visible — no conforming
    // compiler may promote or fuse across it.
    let zz = dims.w;
    let sb = row * 16u;
    // full-STORAGE per-row chain: workgroup slots proved promotable in this
    // kernel's context too (7/12 rows fused, direct-diff) — device memory is
    // the only space every compiler must treat as opaque.
    scr[sb + 13u] = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let v = x[base + j];
        scr[sb + 14u] = v * v;
        scr[sb + 13u] = scr[sb + 13u] + scr[sb + 14u];
    }
    scr[sb + 15u] = scr[sb + 13u] * det_recip(f32(d), zz);
    let yy = scr[sb + 15u] + eps;
    let hy = 0.5 * yy;
    scr[sb] = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(yy) >> 1u));
    scr[sb + 1u] = scr[sb] * scr[sb];
    scr[sb + 2u] = hy * scr[sb + 1u];
    scr[sb + 3u] = 1.5 - scr[sb + 2u];
    scr[sb + 4u] = scr[sb] * scr[sb + 3u];
    scr[sb + 5u] = scr[sb + 4u] * scr[sb + 4u];
    scr[sb + 6u] = hy * scr[sb + 5u];
    scr[sb + 7u] = 1.5 - scr[sb + 6u];
    scr[sb + 8u] = scr[sb + 4u] * scr[sb + 7u];
    scr[sb + 9u] = scr[sb + 8u] * scr[sb + 8u];
    scr[sb + 10u] = hy * scr[sb + 9u];
    scr[sb + 11u] = 1.5 - scr[sb + 10u];
    scr[sb + 12u] = scr[sb + 8u] * scr[sb + 11u];
    let inv = scr[sb + 12u];
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        scr[sb + 14u] = x[base + j] * inv;
        out[base + j] = scr[sb + 14u] * weight[j];
    }
}
"#;

const LAYERNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             dims: vec4<u32>; // rows, d, bitcast(eps), _
@group(0) @binding(5) var<storage, read_write> scr: array<f32>; // rows*16 det-scratch
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y; let eps = bitcast<f32>(dims.z);
    if (row >= rows) { return; }
    let base = row * d;
    let zz = dims.w;
    let sb = row * 16u;
    // full-STORAGE per-row chain (see RMSNORM): device memory is the only
    // address space no conforming compiler may promote or fuse across.
    let invd = det_recip(f32(d), zz);
    scr[sb + 13u] = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { scr[sb + 13u] = scr[sb + 13u] + x[base + j]; }
    scr[sb + 15u] = scr[sb + 13u] * invd;
    let mean = scr[sb + 15u];
    scr[sb + 13u] = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let c = x[base + j] - mean;
        scr[sb + 14u] = c * c;
        scr[sb + 13u] = scr[sb + 13u] + scr[sb + 14u];
    }
    scr[sb + 15u] = scr[sb + 13u] * invd;
    let yy = scr[sb + 15u] + eps;
    let hy = 0.5 * yy;
    scr[sb] = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(yy) >> 1u));
    scr[sb + 1u] = scr[sb] * scr[sb];
    scr[sb + 2u] = hy * scr[sb + 1u];
    scr[sb + 3u] = 1.5 - scr[sb + 2u];
    scr[sb + 4u] = scr[sb] * scr[sb + 3u];
    scr[sb + 5u] = scr[sb + 4u] * scr[sb + 4u];
    scr[sb + 6u] = hy * scr[sb + 5u];
    scr[sb + 7u] = 1.5 - scr[sb + 6u];
    scr[sb + 8u] = scr[sb + 4u] * scr[sb + 7u];
    scr[sb + 9u] = scr[sb + 8u] * scr[sb + 8u];
    scr[sb + 10u] = hy * scr[sb + 9u];
    scr[sb + 11u] = 1.5 - scr[sb + 10u];
    scr[sb + 12u] = scr[sb + 8u] * scr[sb + 11u];
    let inv = scr[sb + 12u];
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        scr[sb + 14u] = (x[base + j] - mean) * inv;
        scr[sb + 15u] = scr[sb + 14u] * weight[j];
        out[base + j] = scr[sb + 15u] + bias[j];
    }
}
"#;

const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // rows, d
@group(0) @binding(3) var<storage, read_write> scr: array<f32>; // rows*2 det-scratch
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y;
    if (row >= rows) { return; }
    let base = row * d;
    var mx: f32 = x[base];
    for (var j: u32 = 1u; j < d; j = j + 1u) { mx = max(mx, x[base + j]); }
    // sum through STORAGE: e is pinned by its store to out[]; the running sum
    // lives in the det-scratch slot — the add sees two device-memory loads.
    let sb = row * 2u;
    scr[sb] = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        out[base + j] = det_exp(x[base + j] - mx, dims.z);
        scr[sb] = scr[sb] + out[base + j];
    }
    let inv = det_recip(scr[sb], dims.z);
    for (var j: u32 = 0u; j < d; j = j + 1u) { out[base + j] = out[base + j] * inv; }
}
"#;

/// Plain-Rust CPU references — the source of truth every kernel is validated against.
pub mod cpu {
    pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() }
    pub fn silu(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| v / (1.0 + (-v).exp())).collect() }
    pub fn relu(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| v.max(0.0)).collect() }
    pub fn sigmoid(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect() }
    pub fn sqrt(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| v.sqrt()).collect() }
    pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x - y).collect() }
    pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x / y).collect() }
    pub fn gelu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 0.5 * v * (1.0 + libm_erf(v * std::f32::consts::FRAC_1_SQRT_2))).collect()
    }
    fn libm_erf(z: f32) -> f32 {
        let s = z.signum(); let a = z.abs();
        let t = 1.0 / (1.0 + 0.3275911 * a);
        let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-a * a).exp();
        s * y
    }
    pub fn layernorm(x: &[f32], w: &[f32], b: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
        let mut o = vec![0.0f32; rows * d];
        for r in 0..rows {
            let base = r * d;
            let mean = x[base..base + d].iter().sum::<f32>() / d as f32;
            let var = x[base..base + d].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..d { o[base + j] = (x[base + j] - mean) * inv * w[j] + b[j]; }
        }
        o
    }
    pub fn rmsnorm(x: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
        let mut o = vec![0.0f32; rows * d];
        for r in 0..rows {
            let base = r * d;
            let ms = x[base..base + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for j in 0..d { o[base + j] = x[base + j] * inv * w[j]; }
        }
        o
    }
    pub fn softmax(x: &[f32], rows: usize, d: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; rows * d];
        for r in 0..rows {
            let base = r * d;
            let mx = x[base..base + d].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..d { let e = (x[base + j] - mx).exp(); o[base + j] = e; sum += e; }
            for j in 0..d { o[base + j] /= sum; }
        }
        o
    }
    pub fn matmul_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, scale: f32) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k { acc += a[i * k + l] * b[j * k + l]; }
                c[i * n + j] = acc * scale;
            }
        }
        c
    }
    pub fn rope(x: &[f32], t: usize, h: usize, dh: usize, base: f32) -> Vec<f32> {
        let mut o = x.to_vec();
        let half = dh / 2;
        for i in 0..t {
            for head in 0..h {
                let off = (i * h + head) * dh;
                for c in 0..half {
                    let inv = (-2.0 * c as f32 / dh as f32 * base.ln()).exp(); // matches the WGSL f32 path
                    let ang = i as f32 * inv;
                    let (cs, sn) = (ang.cos(), ang.sin());
                    let (x1, x2) = (x[off + c], x[off + c + half]);
                    o[off + c] = x1 * cs - x2 * sn;
                    o[off + c + half] = x2 * cs + x1 * sn;
                }
            }
        }
        o
    }
    /// Causal attention with grouped-query attention (hq query heads, hkv kv heads). hkv==hq = plain MHA.
    pub fn mha_causal(q: &[f32], k: &[f32], v: &[f32], t: usize, hq: usize, hkv: usize, dh: usize) -> Vec<f32> {
        let scale = 1.0 / (dh as f32).sqrt();
        let mut o = vec![0.0f32; t * hq * dh];
        for i in 0..t {
            for head in 0..hq {
                let kvhead = head / (hq / hkv);
                let qo = (i * hq + head) * dh;
                let mut scores = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let ko = (j * hkv + kvhead) * dh;
                    scores[j] = (0..dh).map(|c| q[qo + c] * k[ko + c]).sum::<f32>() * scale;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for c in 0..dh {
                    let mut acc = 0.0;
                    for j in 0..=i { acc += scores[j] / sum * v[(j * hkv + kvhead) * dh + c]; }
                    o[qo + c] = acc;
                }
            }
        }
        o
    }
    pub fn attention(q: &[f32], k: &[f32], v: &[f32], rows_q: usize, rows_k: usize, d: usize, dv: usize, scale: f32) -> Vec<f32> {
        let scores = matmul_bt(q, k, rows_q, rows_k, d, scale);
        let probs = softmax(&scores, rows_q, rows_k);
        crate::matmul_cpu(&probs, v, rows_q, rows_k, dv)
    }
}

impl Context {
    /// Forensic: run the REAL det_rsqrt/det_sqrt register-only chain but
    /// export only stage `k` (0 = initial guess; then per Newton iteration
    /// t,u,w,x; 13 = y·rsqrt). One dispatch per stage keeps the chain
    /// register-resident up to the exported value — dumping every stage in
    /// one kernel would itself act as a barrier and could hide the very
    /// compiler transform being hunted.
    pub async fn det_rsqrt_stage(&self, y: &[f32], stage: u32) -> crate::Result<Vec<f32>> {
        // NOTE: pipeline() auto-prepends DET_MATH_WGSL when it sees det_ —
        // do not embed it here or det_bar gets redeclared.
        let src = format!(
            r#"@group(0) @binding(0) var<storage, read>       y: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n, z, stage, _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x; if (i >= dims.x) {{ return; }}
    let z = dims.y;
    let yy = y[i];
    let hy = 0.5 * yy;
    var x = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(yy) >> 1u));
    var stages: array<f32, 14>;
    stages[0] = x;
    var t = det_bar(x * x, z);      stages[1] = t;
    var u = det_bar(hy * t, z);     stages[2] = u;
    var w = det_bar(1.5 - u, z);    stages[3] = w;
    x = det_bar(x * w, z);          stages[4] = x;
    t = det_bar(x * x, z);          stages[5] = t;
    u = det_bar(hy * t, z);         stages[6] = u;
    w = det_bar(1.5 - u, z);        stages[7] = w;
    x = det_bar(x * w, z);          stages[8] = x;
    t = det_bar(x * x, z);          stages[9] = t;
    u = det_bar(hy * t, z);         stages[10] = u;
    w = det_bar(1.5 - u, z);        stages[11] = w;
    x = det_bar(x * w, z);          stages[12] = x;
    stages[13] = yy * det_bar(x, z);
    out[i] = stages[dims.z];
}}
"#
        );
        let n = y.len();
        let yb = self.storage("fy", y);
        let ob = self.out_buffer(n);
        let dims = self.uniform_u32("fd", &[n as u32, 0, stage, 0]);
        let pipe = self.pipeline("rsqrt_stage", &src);
        self.dispatch(&pipe, &[&yb, &ob, &dims], ((n as u32 + 63) / 64, 1, 1));
        self.readback(&ob, n).await
    }
}

impl Context {
    /// Forensic twin of the shipped RMSNORM kernel: identical code, but
    /// exports per-row stage `k` (0=ms accum, 1=recip(d), 2=ms·recip,
    /// 3=+eps, 4=rsqrt, 5=x[0]·inv, 6=out[0]). One dispatch per stage.
    pub async fn det_rmsnorm_stage(
        &self,
        x: &[f32],
        w: &[f32],
        rows: u32,
        d: u32,
        eps: f32,
        stage: u32,
    ) -> crate::Result<Vec<f32>> {
        let src = r#"@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // rows, d, eps, stage
var<workgroup> wacc: array<f32, 64>;
var<workgroup> wprod: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y; let eps = bitcast<f32>(dims.z);
    if (row >= rows) { return; }
    let base = row * d;
    let zz = dims.x - rows; // runtime zero the compiler cannot fold
    var stg: array<f32, 8>;
    var s: array<f32, 16>;
    wacc[li] = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let v = x[base + j];
        wprod[li] = v * v;
        wacc[li] = wacc[li] + wprod[li];
    }
    stg[0] = wacc[li];
    let rc = det_recip(f32(d), zz);
    stg[1] = rc;
    let ms = wacc[li] * rc;
    stg[2] = ms;
    stg[3] = ms + eps;
    let yy = ms + eps;
    let hy = 0.5 * yy;
    wacc[li] = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(yy) >> 1u));
    wprod[li] = wacc[li] * wacc[li];
    wprod[li] = hy * wprod[li];
    wprod[li] = 1.5 - wprod[li];
    wacc[li] = wacc[li] * wprod[li];
    wprod[li] = wacc[li] * wacc[li];
    wprod[li] = hy * wprod[li];
    wprod[li] = 1.5 - wprod[li];
    wacc[li] = wacc[li] * wprod[li];
    wprod[li] = wacc[li] * wacc[li];
    wprod[li] = hy * wprod[li];
    wprod[li] = 1.5 - wprod[li];
    wacc[li] = wacc[li] * wprod[li];
    let inv = wacc[li];
    stg[4] = inv;
    wprod[li] = x[base] * inv;
    stg[5] = wprod[li];
    stg[6] = wprod[li] * weight[0];
    out[row] = stg[dims.w];
}
"#;
        let xb = self.storage("fx", x);
        let wb = self.storage("fw", w);
        let ob = self.out_buffer(rows as usize);
        let dims = self.uniform_u32("fd", &[rows, d, eps.to_bits(), stage]);
        let pipe = self.pipeline("rmsnorm_stage", src);
        self.dispatch(&pipe, &[&xb, &wb, &ob, &dims], ((rows + 63) / 64, 1, 1));
        self.readback(&ob, rows as usize).await
    }
}

/// Deterministic PARALLEL rmsnorm: one workgroup (64 threads) per row.
/// Squares are pinned by workgroup stores; the reduction is a fixed-shape
/// tree of pure adds (strided per-thread partials, then a 6-level tree over
/// 64 partials) separated by workgroupBarrier() — barriers make the shared
/// memory real, and an adds-only tree has no mul to contract, so the
/// summation order is the algorithm's on every compiler. The per-row rsqrt
/// runs once (thread 0) through the storage scratch and is shared back.
/// Digest differs from the sequential kernel (different summation order) —
/// this is a NEW op; migration of default rmsnorm is a deliberate,
/// versioned change. Supports d ≤ 4096.
const RMSNORM_TREE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // rows, d, bitcast(eps), _
@group(0) @binding(4) var<storage, read_write> scr: array<f32>; // rows*16 det-scratch
var<workgroup> ws: array<f32, 4096>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wid.x;
    let d = dims.y;
    let eps = bitcast<f32>(dims.z);
    let base = row * d;
    // squares, strided; each product pinned by its workgroup store
    for (var j = li; j < d; j = j + 64u) { ws[j] = x[base + j] * x[base + j]; }
    workgroupBarrier();
    // per-thread partial over a DIFFERENT thread's stride class: reader ≠
    // writer across the barrier, so no compiler can privatize ws into
    // registers (Tint proved same-thread stride arrays private and re-fused
    // the squares — probe-caught). Values are identical; only the label of
    // which partial holds which class shifts, fixed on every substrate.
    let cls = (li + 1u) & 63u;
    red[li] = 0.0;
    for (var j = cls; j < d; j = j + 64u) { red[li] = red[li] + ws[j]; }
    workgroupBarrier();
    // fixed 6-level tree over the 64 partials — pure adds
    for (var s = 32u; s >= 1u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    // per-row scalars once, on thread 0 — the ENTIRE tail through storage
    // (recip included): the XOR-chain recip and the ms·recip product were
    // the last register ops, and Tint fused them in this kernel's context.
    if (li == 0u) {
        let sb = row * 32u;
        let df = f32(d);
        scr[sb + 16u] = bitcast<f32>(0x7EF311C3u - bitcast<u32>(df));
        scr[sb + 17u] = df * scr[sb + 16u];
        scr[sb + 18u] = 2.0 - scr[sb + 17u];
        scr[sb + 19u] = scr[sb + 16u] * scr[sb + 18u];
        scr[sb + 20u] = df * scr[sb + 19u];
        scr[sb + 21u] = 2.0 - scr[sb + 20u];
        scr[sb + 22u] = scr[sb + 19u] * scr[sb + 21u];
        scr[sb + 23u] = df * scr[sb + 22u];
        scr[sb + 24u] = 2.0 - scr[sb + 23u];
        scr[sb + 25u] = scr[sb + 22u] * scr[sb + 24u];
        scr[sb + 26u] = red[0] * scr[sb + 25u];
        let yy = scr[sb + 26u] + eps;
        let hy = 0.5 * yy;
        scr[sb] = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(yy) >> 1u));
        scr[sb + 1u] = scr[sb] * scr[sb];
        scr[sb + 2u] = hy * scr[sb + 1u];
        scr[sb + 3u] = 1.5 - scr[sb + 2u];
        scr[sb + 4u] = scr[sb] * scr[sb + 3u];
        scr[sb + 5u] = scr[sb + 4u] * scr[sb + 4u];
        scr[sb + 6u] = hy * scr[sb + 5u];
        scr[sb + 7u] = 1.5 - scr[sb + 6u];
        scr[sb + 8u] = scr[sb + 4u] * scr[sb + 7u];
        scr[sb + 9u] = scr[sb + 8u] * scr[sb + 8u];
        scr[sb + 10u] = hy * scr[sb + 9u];
        scr[sb + 11u] = 1.5 - scr[sb + 10u];
        scr[sb + 12u] = scr[sb + 8u] * scr[sb + 11u];
        red[1] = scr[sb + 12u];
    }
    workgroupBarrier();
    let inv = red[1];
    // output in two phases with a barrier: writes on the own stride, reads on
    // the offset stride — reader ≠ writer keeps ws un-privatizable here too.
    for (var j = li; j < d; j = j + 64u) { ws[j] = x[base + j] * inv; }
    workgroupBarrier();
    for (var j = cls; j < d; j = j + 64u) { out[base + j] = ws[j] * weight[j]; }
}
"#;

impl Context {
    /// Deterministic tree-reduction RMSNorm (see RMSNORM_TREE_WGSL). One
    /// workgroup per row; d ≤ 4096. Digest differs from `rmsnorm_t` by
    /// summation order — cross-fabric identity is what the probe certifies.
    pub fn rmsnorm_tree_t(&self, x: &Tensor, w: &Tensor, rows: u32, d: u32, eps: f32) -> Tensor {
        assert!(d <= 4096, "rmsnorm_tree_t supports d ≤ 4096");
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, eps.to_bits(), 0]);
        let scr = self.out_buffer(rows as usize * 32);
        let pipe = self.pipeline("rmsnorm_tree", RMSNORM_TREE_WGSL);
        self.dispatch(&pipe, &[&x.buf, &w.buf, &out.buf, &dims, &scr], (rows, 1, 1));
        out
    }
}

/// CPU replica of the tree rmsnorm — the SAME fixed-shape algorithm (64
/// strided partials → 6-level add tree → Newton scalars) in plain IEEE Rust.
/// Rows are independent, so row-level threading never affects the digest:
/// this is the heterogeneous-fabric claim made concrete — one algorithm,
/// one digest, any substrate (GPU shader or CPU cores, native or wasm).
pub fn rmsnorm_tree_cpu(x: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    fn recip(y: f32) -> f32 {
        let mut r = f32::from_bits(0x7EF3_11C3u32.wrapping_sub(y.to_bits()));
        for _ in 0..3 {
            let t = y * r;
            let ww = 2.0 - t;
            r = r * ww;
        }
        r
    }
    fn rsqrt(y: f32) -> f32 {
        let hy = 0.5 * y;
        let mut r = f32::from_bits(0x5F37_59DFu32.wrapping_sub(y.to_bits() >> 1));
        for _ in 0..3 {
            let t = r * r;
            let u = hy * t;
            let ww = 1.5 - u;
            r = r * ww;
        }
        r
    }
    let mut out = vec![0f32; rows * d];
    // row-parallel across available cores (sequential on wasm — no threads);
    // per-row math is fixed-shape, so scheduling never touches the digest
    #[cfg(target_arch = "wasm32")]
    let n_threads = 1usize;
    #[cfg(not(target_arch = "wasm32"))]
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    std::thread::scope(|scope| {
        let chunk = rows.div_ceil(n_threads);
        for (ci, out_chunk) in out.chunks_mut(chunk * d).enumerate() {
            let x = &x;
            let w = &w;
            #[allow(unused_mut)]
            let mut work = move || {
                for (ri, orow) in out_chunk.chunks_mut(d).enumerate() {
                    let row = ci * chunk + ri;
                    let base = row * d;
                    // squares (rounded muls), then 64 strided partials
                    let mut red = [0f32; 64];
                    for li in 0..64usize {
                        let cls = (li + 1) & 63;
                        let mut p = 0f32;
                        let mut j = cls;
                        while j < d {
                            let sq = x[base + j] * x[base + j];
                            p += sq;
                            j += 64;
                        }
                        red[li] = p;
                    }
                    // fixed 6-level tree
                    let mut s = 32usize;
                    while s >= 1 {
                        for li in 0..s {
                            red[li] += red[li + s];
                        }
                        if s == 1 { break; }
                        s >>= 1;
                    }
                    let ms = red[0] * recip(d as f32);
                    let inv = rsqrt(ms + eps);
                    for j in 0..d {
                        let t = x[base + j] * inv;
                        orow[j] = t * w[j];
                    }
                }
            };
            #[cfg(target_arch = "wasm32")]
            work();
            #[cfg(not(target_arch = "wasm32"))]
            scope.spawn(work);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = scope;
    });
    out
}

/// Deterministic PARALLEL layernorm: rmsnorm_tree's structure with two
/// chained reductions (mean, then variance) and a three-phase output whose
/// writer/reader stride classes differ at every barrier (no compiler can
/// privatize ws between phases). d ≤ 4096.
const LAYERNORM_TREE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             dims: vec4<u32>; // rows, d, bitcast(eps), _
@group(0) @binding(5) var<storage, read_write> scr: array<f32>; // rows*32 det-scratch
var<workgroup> ws: array<f32, 4096>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wid.x;
    let d = dims.y;
    let eps = bitcast<f32>(dims.z);
    let base = row * d;
    let cls = (li + 1u) & 63u;
    let cls2 = (li + 2u) & 63u;
    let sb = row * 32u;
    // ── mean: partials of storage loads (unprivatizable), fixed tree ──
    red[li] = 0.0;
    for (var j = cls; j < d; j = j + 64u) { red[li] = red[li] + x[base + j]; }
    workgroupBarrier();
    for (var s = 32u; s >= 1u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        let df = f32(d);
        scr[sb + 16u] = bitcast<f32>(0x7EF311C3u - bitcast<u32>(df));
        scr[sb + 17u] = df * scr[sb + 16u];
        scr[sb + 18u] = 2.0 - scr[sb + 17u];
        scr[sb + 19u] = scr[sb + 16u] * scr[sb + 18u];
        scr[sb + 20u] = df * scr[sb + 19u];
        scr[sb + 21u] = 2.0 - scr[sb + 20u];
        scr[sb + 22u] = scr[sb + 19u] * scr[sb + 21u];
        scr[sb + 23u] = df * scr[sb + 22u];
        scr[sb + 24u] = 2.0 - scr[sb + 23u];
        scr[sb + 25u] = scr[sb + 22u] * scr[sb + 24u];   // recip(d)
        scr[sb + 26u] = red[0] * scr[sb + 25u];          // mean
    }
    workgroupBarrier();
    let mean = scr[sb + 26u];
    // ── variance: centered squares pinned by ws stores, entangled partials ──
    for (var j = li; j < d; j = j + 64u) {
        let c = x[base + j] - mean;
        ws[j] = c * c;
    }
    workgroupBarrier();
    red[li] = 0.0;
    for (var j = cls; j < d; j = j + 64u) { red[li] = red[li] + ws[j]; }
    workgroupBarrier();
    for (var s = 32u; s >= 1u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        scr[sb + 27u] = red[0] * scr[sb + 25u];          // var
        let yy = scr[sb + 27u] + eps;
        let hy = 0.5 * yy;
        scr[sb] = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(yy) >> 1u));
        scr[sb + 1u] = scr[sb] * scr[sb];
        scr[sb + 2u] = hy * scr[sb + 1u];
        scr[sb + 3u] = 1.5 - scr[sb + 2u];
        scr[sb + 4u] = scr[sb] * scr[sb + 3u];
        scr[sb + 5u] = scr[sb + 4u] * scr[sb + 4u];
        scr[sb + 6u] = hy * scr[sb + 5u];
        scr[sb + 7u] = 1.5 - scr[sb + 6u];
        scr[sb + 8u] = scr[sb + 4u] * scr[sb + 7u];
        scr[sb + 9u] = scr[sb + 8u] * scr[sb + 8u];
        scr[sb + 10u] = hy * scr[sb + 9u];
        scr[sb + 11u] = 1.5 - scr[sb + 10u];
        scr[sb + 12u] = scr[sb + 8u] * scr[sb + 11u];    // inv
    }
    workgroupBarrier();
    let inv = scr[sb + 12u];
    // ── output: three phases, writer ≠ reader at every barrier ──
    for (var j = li; j < d; j = j + 64u) { ws[j] = (x[base + j] - mean) * inv; }
    workgroupBarrier();
    for (var j = cls; j < d; j = j + 64u) { ws[j] = ws[j] * weight[j]; }
    workgroupBarrier();
    for (var j = cls2; j < d; j = j + 64u) { out[base + j] = ws[j] + bias[j]; }
}
"#;

impl Context {
    /// Deterministic tree-reduction LayerNorm (see LAYERNORM_TREE_WGSL). d ≤ 4096.
    pub fn layernorm_tree_t(&self, x: &Tensor, w: &Tensor, b: &Tensor, rows: u32, d: u32, eps: f32) -> Tensor {
        assert!(d <= 4096, "layernorm_tree_t supports d ≤ 4096");
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, eps.to_bits(), 0]);
        let scr = self.out_buffer(rows as usize * 32);
        let pipe = self.pipeline("layernorm_tree", LAYERNORM_TREE_WGSL);
        self.dispatch(&pipe, &[&x.buf, &w.buf, &b.buf, &out.buf, &dims, &scr], (rows, 1, 1));
        out
    }
}

/// CPU twin of the tree layernorm — identical fixed shape, plain IEEE Rust.
pub fn layernorm_tree_cpu(x: &[f32], w: &[f32], b: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    fn recip(y: f32) -> f32 {
        let mut r = f32::from_bits(0x7EF3_11C3u32.wrapping_sub(y.to_bits()));
        for _ in 0..3 {
            let t = y * r;
            let ww = 2.0 - t;
            r = r * ww;
        }
        r
    }
    fn rsqrt(y: f32) -> f32 {
        let hy = 0.5 * y;
        let mut r = f32::from_bits(0x5F37_59DFu32.wrapping_sub(y.to_bits() >> 1));
        for _ in 0..3 {
            let t = r * r;
            let u = hy * t;
            let ww = 1.5 - u;
            r = r * ww;
        }
        r
    }
    fn tree64(red: &mut [f32; 64]) -> f32 {
        let mut s = 32usize;
        loop {
            for li in 0..s {
                red[li] += red[li + s];
            }
            if s == 1 { break; }
            s >>= 1;
        }
        red[0]
    }
    let mut out = vec![0f32; rows * d];
    for row in 0..rows {
        let base = row * d;
        let mut red = [0f32; 64];
        for li in 0..64usize {
            let cls = (li + 1) & 63;
            let mut p = 0f32;
            let mut j = cls;
            while j < d {
                p += x[base + j];
                j += 64;
            }
            red[li] = p;
        }
        let rc = recip(d as f32);
        let mean = tree64(&mut red) * rc;
        let mut red = [0f32; 64];
        for li in 0..64usize {
            let cls = (li + 1) & 63;
            let mut p = 0f32;
            let mut j = cls;
            while j < d {
                let c = x[base + j] - mean;
                let sq = c * c;
                p += sq;
                j += 64;
            }
            red[li] = p;
        }
        let var = tree64(&mut red) * rc;
        let inv = rsqrt(var + eps);
        for j in 0..d {
            let t = (x[base + j] - mean) * inv;
            let t2 = t * w[j];
            out[base + j] = t2 + b[j];
        }
    }
    out
}

/// Deterministic PARALLEL softmax: max-tree (exact, order-free — fixed shape
/// kept anyway), exp wave with each result pinned by its ws store, sum-tree
/// over entangled classes, storage-tail reciprocal. The exp INTERNALS are
/// XOR-chain det_exp — the known fragile spot under Tint inlining; the
/// six-substrate probe is the arbiter, and the sequential softmax remains
/// the fallback if this context fuses. d ≤ 4096.
const SOFTMAX_TREE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // rows, d, _, z
@group(0) @binding(3) var<storage, read_write> scr: array<f32>; // rows*32 det-scratch
var<workgroup> ws: array<f32, 4096>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wid.x;
    let d = dims.y;
    let base = row * d;
    let cls = (li + 1u) & 63u;
    let cls2 = (li + 2u) & 63u;
    let sb = row * 32u;
    // ── max (exact any order; fixed shape for uniformity) ──
    red[li] = -3.0e38;
    for (var j = cls; j < d; j = j + 64u) { red[li] = max(red[li], x[base + j]); }
    workgroupBarrier();
    for (var s = 32u; s >= 1u; s = s >> 1u) {
        if (li < s) { red[li] = max(red[li], red[li + s]); }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    // ── exps, each pinned by its ws store ──
    for (var j = li; j < d; j = j + 64u) { ws[j] = det_exp(x[base + j] - mx, dims.w); }
    workgroupBarrier();
    // ── sum over entangled classes, fixed tree ──
    red[li] = 0.0;
    for (var j = cls; j < d; j = j + 64u) { red[li] = red[li] + ws[j]; }
    workgroupBarrier();
    for (var s = 32u; s >= 1u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        let y = red[0];
        scr[sb + 16u] = bitcast<f32>(0x7EF311C3u - bitcast<u32>(y));
        scr[sb + 17u] = y * scr[sb + 16u];
        scr[sb + 18u] = 2.0 - scr[sb + 17u];
        scr[sb + 19u] = scr[sb + 16u] * scr[sb + 18u];
        scr[sb + 20u] = y * scr[sb + 19u];
        scr[sb + 21u] = 2.0 - scr[sb + 20u];
        scr[sb + 22u] = scr[sb + 19u] * scr[sb + 21u];
        scr[sb + 23u] = y * scr[sb + 22u];
        scr[sb + 24u] = 2.0 - scr[sb + 23u];
        scr[sb + 25u] = scr[sb + 22u] * scr[sb + 24u];   // 1/sum
    }
    workgroupBarrier();
    let inv = scr[sb + 25u];
    for (var j = cls2; j < d; j = j + 64u) { out[base + j] = ws[j] * inv; }
}
"#;

impl Context {
    /// Deterministic tree-reduction softmax (see SOFTMAX_TREE_WGSL). d ≤ 4096.
    pub fn softmax_tree_t(&self, x: &Tensor, rows: u32, d: u32) -> Tensor {
        assert!(d <= 4096, "softmax_tree_t supports d ≤ 4096");
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, 0, 0]);
        let scr = self.out_buffer(rows as usize * 32);
        let pipe = self.pipeline("softmax_tree", SOFTMAX_TREE_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims, &scr], (rows, 1, 1));
        out
    }
}

/// CPU twin of the tree softmax — including a bit-exact replica of det_exp's
/// operation sequence (same f32 constants, same order, plain IEEE).
pub fn softmax_tree_cpu(x: &[f32], rows: usize, d: usize) -> Vec<f32> {
    fn exp_det(v: f32) -> f32 {
        let xx = v.clamp(-87.0, 88.0);
        let log2e = f32::from_bits(0x3FB8_AA3B);
        let ln2hi = 0.693115234375f32;
        let ln2lo = f32::from_bits(0x3805_FDF4);
        let kf = (xx * log2e + 0.5).floor();
        let a = xx - kf * ln2hi;
        let r = a - kf * ln2lo;
        let mut p = 0.0013888889f32;
        p = p * r + 0.008333334;
        p = p * r + 0.041666668;
        p = p * r + 0.16666667;
        p = p * r + 0.5;
        p = p * r + 1.0;
        p = p * r + 1.0;
        let k = (kf as i32).clamp(-126, 127);
        let scale = f32::from_bits(((k + 127) as u32) << 23);
        p * scale
    }
    fn recip(y: f32) -> f32 {
        let mut r = f32::from_bits(0x7EF3_11C3u32.wrapping_sub(y.to_bits()));
        for _ in 0..3 {
            let t = y * r;
            let ww = 2.0 - t;
            r = r * ww;
        }
        r
    }
    fn tree64(red: &mut [f32; 64]) -> f32 {
        let mut s = 32usize;
        loop {
            for li in 0..s {
                red[li] += red[li + s];
            }
            if s == 1 { break; }
            s >>= 1;
        }
        red[0]
    }
    let mut out = vec![0f32; rows * d];
    let mut ws = vec![0f32; d];
    for row in 0..rows {
        let base = row * d;
        let mut mx = -3.0e38f32;
        for j in 0..d {
            mx = mx.max(x[base + j]);
        }
        for j in 0..d {
            ws[j] = exp_det(x[base + j] - mx);
        }
        let mut red = [0f32; 64];
        for li in 0..64usize {
            let cls = (li + 1) & 63;
            let mut p = 0f32;
            let mut j = cls;
            while j < d {
                p += ws[j];
                j += 64;
            }
            red[li] = p;
        }
        let inv = recip(tree64(&mut red));
        for j in 0..d {
            out[base + j] = ws[j] * inv;
        }
    }
    out
}
