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
        let pipe = self.pipeline("softmax", SOFTMAX_WGSL);
        self.dispatch(&pipe, &[&x.buf, &out.buf, &dims], ((rows + 63) / 64, 1, 1));
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
        let pipe = self.pipeline("rmsnorm", RMSNORM_WGSL);
        self.dispatch(&pipe, &[&x.buf, &w.buf, &out.buf, &dims], (rows, 1, 1));
        out
    }
    pub fn layernorm_t(&self, x: &Tensor, w: &Tensor, b: &Tensor, rows: u32, d: u32, eps: f32) -> Tensor {
        let out = self.empty(&x.shape);
        let dims = self.uniform_u32("d", &[rows, d, eps.to_bits(), 0]);
        let pipe = self.pipeline("layernorm", LAYERNORM_WGSL);
        self.dispatch(&pipe, &[&x.buf, &w.buf, &b.buf, &out.buf, &dims], ((rows + 63) / 64, 1, 1));
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
        let pipe = self.pipeline("layernorm", LAYERNORM_WGSL);
        self.dispatch(&pipe, &[&xb, &wb, &bb, &out, &dims], ((rows + 63) / 64, 1, 1));
        self.readback(&out, (rows * d) as usize).await
    }

    /// Row-wise softmax over the last dim `d` for `rows` rows (numerically stable).
    pub async fn softmax(&self, x: &[f32], rows: u32, d: u32) -> Result<Vec<f32>> {
        assert_eq!(x.len(), (rows * d) as usize);
        let xb = self.storage("x", x);
        let out = self.out_buffer((rows * d) as usize);
        let dims = self.uniform_u32("dims", &[rows, d, 0, 0]);
        let pipe = self.pipeline("softmax", SOFTMAX_WGSL);
        self.dispatch(&pipe, &[&xb, &out, &dims], ((rows + 63) / 64, 1, 1));
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

const SILU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }
    let v = x[i];
    out[i] = v / (1.0 + exp(-v));
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
    out[i] = 1.0 / (1.0 + exp(-x[i]));
}
"#;

const SQRT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    out[i] = sqrt(x[i]);
}
"#;

// Exact erf-based GELU: 0.5·x·(1+erf(x/√2)). WGSL has no erf, so use a high-accuracy
// Abramowitz-Stegun 7.1.26 rational approximation (|err| < 1.5e-7) — matches ONNX Gelu (erf).
const GELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // n
fn erf(z: f32) -> f32 {
    let s = sign(z); let a = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= dims.x) { return; }
    let v = x[i];
    out[i] = 0.5 * v * (1.0 + erf(v * 0.7071067811865476));
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
    let lb = log(base);
    for (var c: u32 = 0u; c < half; c = c + 1u) {
        let inv = exp(-2.0 * f32(c) / f32(dh) * lb);
        let ang = f32(i + rmeta.x) * inv;
        let cs = cos(ang); let sn = sin(ang);
        let x1 = x[o + c]; let x2 = x[o + c + half];
        out[o + c] = x1 * cs - x2 * sn;
        out[o + c + half] = x2 * cs + x1 * sn;
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
    var m: f32 = -3.0e38; var l: f32 = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {           // causal: attend to keys 0..=i
        let ko = (j * hkv + kvhead) * dh;
        var s: f32 = 0.0;
        for (var c: u32 = 0u; c < dh; c = c + 1u) { s = s + q[qo + c] * k[ko + c]; }
        s = s * scale;
        let mnew = max(m, s);
        let corr = exp(m - mnew);
        let p = exp(s - mnew);
        l = l * corr + p;
        for (var c: u32 = 0u; c < dh; c = c + 1u) { acc[c] = acc[c] * corr + p * v[ko + c]; }
        m = mnew;
    }
    for (var c: u32 = 0u; c < dh; c = c + 1u) { out[qo + c] = acc[c] / l; }
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
    var m: f32 = -3.0e38; var l: f32 = 0.0;
    for (var j: u32 = 0u; j < s; j = j + 1u) {
        let ko = (j * hkv + kvhead) * dh;
        var sc: f32 = 0.0;
        for (var c: u32 = 0u; c < dh; c = c + 1u) { sc = sc + q[qo + c] * k[ko + c]; }
        sc = sc * scale;
        let mnew = max(m, sc);
        let corr = exp(m - mnew);
        let p = exp(sc - mnew);
        l = l * corr + p;
        for (var c: u32 = 0u; c < dh; c = c + 1u) { acc[c] = acc[c] * corr + p * v[ko + c]; }
        m = mnew;
    }
    for (var c: u32 = 0u; c < dh; c = c + 1u) { out[qo + c] = acc[c] / l; }
}
"#;

const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // rows, d, bitcast(eps), _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y; let eps = bitcast<f32>(dims.z);
    if (row >= rows) { return; }
    let base = row * d;
    var ms: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let v = x[base + j]; ms = ms + v * v; }
    ms = ms / f32(d);
    let inv = 1.0 / sqrt(ms + eps);
    for (var j: u32 = 0u; j < d; j = j + 1u) { out[base + j] = x[base + j] * inv * weight[j]; }
}
"#;

const LAYERNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             dims: vec4<u32>; // rows, d, bitcast(eps), _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y; let eps = bitcast<f32>(dims.z);
    if (row >= rows) { return; }
    let base = row * d;
    var mean: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { mean = mean + x[base + j]; }
    mean = mean / f32(d);
    var vari: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let c = x[base + j] - mean; vari = vari + c * c; }
    vari = vari / f32(d);
    let inv = 1.0 / sqrt(vari + eps);
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        out[base + j] = (x[base + j] - mean) * inv * weight[j] + bias[j];
    }
}
"#;

const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             dims: vec4<u32>; // rows, d
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = dims.x; let d = dims.y;
    if (row >= rows) { return; }
    let base = row * d;
    var mx: f32 = x[base];
    for (var j: u32 = 1u; j < d; j = j + 1u) { mx = max(mx, x[base + j]); }
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let e = exp(x[base + j] - mx); out[base + j] = e; sum = sum + e; }
    let inv = 1.0 / sum;
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
