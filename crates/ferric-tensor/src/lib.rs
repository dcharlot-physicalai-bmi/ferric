//! Ferric L2 — a general N-dimensional tensor runtime on the GPU fabric.
//!
//! This is the substrate the whole ecosystem is meant to stand on: not fixed-shape, hand-fused
//! kernels for one architecture, but a real tensor with **arbitrary rank, strided views, and
//! broadcasting**, plus **general** elementwise ops, **general** reductions over any axes, and
//! **batched** matmul. The transformer kernels in `ferric-core` become fused fast-paths of this.
//!
//! Design: eager execution, tensors are `Arc`-shared f32 buffers described by (shape, strides,
//! offset). Views (`reshape`/`permute`/`transpose`/`broadcast_to`) are zero-copy stride tricks;
//! `contiguous()` materializes. One general strided kernel powers elementwise + broadcasting; a
//! segmented kernel powers reductions; a batched kernel powers matmul. Validated against a strided
//! CPU reference on general shapes (broadcasting, non-contiguous inputs, arbitrary reduction axes).
//!
//! Next fabric layers (in progress): dtypes (f16/bf16/int), autograd tape for training, op fusion,
//! and the heterogeneous scheduler.

use ferric_core::Context;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub mod autograd; // reverse-mode autodiff (training)
pub mod cpu; // strided CPU reference (validation source of truth)
pub mod dtype; // f16/bf16 half-precision storage + on-device dequant
pub mod fuse; // kernel fusion via runtime WGSL codegen (the optimizing-compiler seed)
pub mod nn; // transformer blocks expressed on the general runtime
pub mod optim; // optimizers (Adam)
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
pub mod metal4; // Metal 4 tensor-unit GEMM backend (~280× the WGSL path on Apple silicon)
#[cfg(not(target_arch = "wasm32"))]
pub mod sched; // L7 heterogeneous scheduler (GPU + CPU as one fabric)
#[cfg(not(target_arch = "wasm32"))]
pub mod ws; // WebSocket bridge so a browser tab is a scheduler device
pub use autograd::{grad, Var};
pub use dtype::{DType, Half, QMatrix, QShard, Q2_0Weights, Q4_0Weights, Q4_1Weights, Q5_0Weights, Q5_1Weights, Q4_KWeights, Q5_KWeights, Q6_KWeights, Q8_0Weights, QRow, QTensor, Ternary};
pub use optim::Adam;

/// A general N-D f32 tensor: an Arc-shared device buffer viewed through (shape, strides, offset).
#[derive(Clone)]
pub struct Tensor {
    ctx: Arc<Context>,
    buf: Arc<wgpu::Buffer>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>, // element strides (row-major by default)
    offset: usize,
}

fn contig_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}
fn numel(shape: &[usize]) -> usize { shape.iter().product() }

/// Broadcast two shapes NumPy-style (right-aligned). Returns the result shape.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for i in 0..r {
        let da = if i + a.len() >= r { a[i + a.len() - r] } else { 1 };
        let db = if i + b.len() >= r { b[i + b.len() - r] } else { 1 };
        assert!(da == db || da == 1 || db == 1, "shapes {a:?} and {b:?} not broadcastable at dim {i}");
        out[i] = da.max(db);
    }
    out
}

impl Tensor {
    pub fn numel(&self) -> usize { numel(&self.shape) }
    /// The underlying wgpu buffer (external interop — e.g. handing tensors to the Metal-4 resident
    /// path or other raw-backend consumers). Offsets/strides still apply; see `offset`/`strides`.
    pub fn buffer(&self) -> &wgpu::Buffer { &self.buf }
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn is_contiguous(&self) -> bool { self.strides == contig_strides(&self.shape) && self.offset == 0 }

    // ---- construction / io ----
    pub fn from_vec(ctx: &Arc<Context>, data: &[f32], shape: &[usize]) -> Tensor {
        assert_eq!(data.len(), numel(shape), "data len != shape product");
        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tensor"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Tensor { ctx: ctx.clone(), buf: Arc::new(buf), shape: shape.to_vec(), strides: contig_strides(shape), offset: 0 }
    }

    /// A contiguous [shape] view over a shared GPU buffer (offset 0). The buffer may be **larger**
    /// than `numel(shape)` — e.g. a preallocated KV cache; the kernel reads only the first
    /// `numel(shape)` elements. This is what lets the cache grow-in-place without re-copying.
    pub fn from_arc(ctx: &Arc<Context>, buf: Arc<wgpu::Buffer>, shape: &[usize]) -> Tensor {
        Tensor { ctx: ctx.clone(), buf, shape: shape.to_vec(), strides: contig_strides(shape), offset: 0 }
    }
    pub fn zeros(ctx: &Arc<Context>, shape: &[usize]) -> Tensor { Self::from_vec(ctx, &vec![0.0; numel(shape)], shape) }
    /// Wrap a freshly-computed contiguous device buffer as a tensor (crate-internal).
    pub(crate) fn from_parts(ctx: &Arc<Context>, buf: wgpu::Buffer, shape: Vec<usize>) -> Tensor {
        let strides = contig_strides(&shape);
        Tensor { ctx: ctx.clone(), buf: Arc::new(buf), shape, strides, offset: 0 }
    }

    /// Materialize (contiguous) and read back to host in logical row-major order.
    pub async fn to_vec(&self) -> Vec<f32> {
        let c = self.contiguous();
        readback(&c.ctx, &c.buf, c.numel()).await
    }

    // ---- zero-copy views ----
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(numel(shape), self.numel(), "reshape changes numel");
        let c = self.contiguous();
        Tensor { ctx: c.ctx, buf: c.buf, strides: contig_strides(shape), shape: shape.to_vec(), offset: 0 }
    }
    pub fn permute(&self, perm: &[usize]) -> Tensor {
        assert_eq!(perm.len(), self.rank(), "permute rank mismatch");
        Tensor {
            ctx: self.ctx.clone(), buf: self.buf.clone(), offset: self.offset,
            shape: perm.iter().map(|&p| self.shape[p]).collect(),
            strides: perm.iter().map(|&p| self.strides[p]).collect(),
        }
    }
    pub fn transpose(&self, a: usize, b: usize) -> Tensor {
        let mut p: Vec<usize> = (0..self.rank()).collect();
        p.swap(a, b);
        self.permute(&p)
    }
    /// Broadcast to a larger shape (right-aligned); broadcast dims get stride 0.
    pub fn broadcast_to(&self, shape: &[usize]) -> Tensor {
        let r = shape.len();
        assert!(r >= self.rank(), "cannot broadcast to fewer dims");
        let mut strides = vec![0usize; r];
        for i in 0..self.rank() {
            let (si, di) = (self.rank() - 1 - i, r - 1 - i);
            if self.shape[si] == shape[di] {
                strides[di] = self.strides[si];
            } else {
                assert_eq!(self.shape[si], 1, "cannot broadcast dim {} of {:?} to {:?}", si, self.shape, shape);
                strides[di] = 0;
            }
        }
        Tensor { ctx: self.ctx.clone(), buf: self.buf.clone(), shape: shape.to_vec(), strides, offset: self.offset }
    }

    /// Zero-copy slice along `dim`: rows/cols [start, start+len). A strided view — no copy until an
    /// op materializes it. This is how interleaved projections (e.g. Qwen3.5's fused qkvz) get split.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        assert!(start + len <= self.shape[dim], "narrow {start}+{len} out of range for dim {dim} ({})", self.shape[dim]);
        let mut shape = self.shape.clone();
        shape[dim] = len;
        Tensor {
            ctx: self.ctx.clone(), buf: self.buf.clone(), shape,
            strides: self.strides.clone(), offset: self.offset + start * self.strides[dim],
        }
    }

    /// Concatenate two tensors along `dim`. Both may be strided views (no pre-materialization).
    /// Output-indexed: each thread decides which side it reads from, so this stays within the
    /// WebGPU baseline 4-storage-buffer limit. `cat_all` folds a slice of tensors through this.
    pub fn cat(&self, other: &Tensor, dim: usize) -> Tensor {
        assert_eq!(self.rank(), other.rank(), "cat: rank mismatch");
        for d in 0..self.rank() {
            if d != dim { assert_eq!(self.shape[d], other.shape[d], "cat: dim {d} mismatch {:?} vs {:?}", self.shape, other.shape); }
        }
        let mut shape = self.shape.clone();
        shape[dim] += other.shape[dim];
        let n = numel(&shape);
        let out = empty(&self.ctx, n);
        // info: [rank, dim, n, offA, offB, aDim, shape[r], aStr[r], bStr[r]]
        let mut info = vec![self.rank() as u32, dim as u32, n as u32, self.offset as u32, other.offset as u32, self.shape[dim] as u32];
        info.extend(shape.iter().map(|&x| x as u32));
        info.extend(self.strides.iter().map(|&x| x as u32));
        info.extend(other.strides.iter().map(|&x| x as u32));
        run(&self.ctx, CAT_WGSL, "cat", &[&self.buf, &other.buf, &out, &u32buf(&self.ctx, &info)], groups(n));
        Tensor::from_parts(&self.ctx, out, shape)
    }

    /// Materialize a (possibly strided/broadcast) view into a fresh contiguous buffer.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        let n = self.numel();
        let out = empty(&self.ctx, n);
        // info: [rank, n, offset, row_stride, shape..., strides...]; 2D grid so n can exceed 4.19M.
        let (grid, rs) = groups2d(n);
        let mut info = vec![self.rank() as u32, n as u32, self.offset as u32, rs];
        info.extend(self.shape.iter().map(|&x| x as u32));
        info.extend(self.strides.iter().map(|&x| x as u32));
        run(&self.ctx, GATHER_WGSL, "gather", &[&self.buf, &out, &u32buf(&self.ctx, &info)], grid);
        Tensor { ctx: self.ctx.clone(), buf: Arc::new(out), shape: self.shape.clone(), strides: contig_strides(&self.shape), offset: 0 }
    }

    // ---- general broadcasting elementwise ----
    fn binary(&self, other: &Tensor, op: u32) -> Tensor {
        let shape = broadcast_shapes(&self.shape, &other.shape);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        let n = numel(&shape);
        let out = empty(&self.ctx, n);
        // info: [rank, op, n, offA, offB, shape..., aStr..., bStr...]
        let mut info = vec![shape.len() as u32, op, n as u32, a.offset as u32, b.offset as u32];
        info.extend(shape.iter().map(|&x| x as u32));
        info.extend(a.strides.iter().map(|&x| x as u32));
        info.extend(b.strides.iter().map(|&x| x as u32));
        let (bgrid, brs) = groups2d(n); info.push(brs);
        run(&self.ctx, BINARY_WGSL, "binary", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &info)], bgrid);
        Tensor { ctx: self.ctx.clone(), buf: Arc::new(out), shape: shape.clone(), strides: contig_strides(&shape), offset: 0 }
    }
    pub fn add(&self, o: &Tensor) -> Tensor { self.binary(o, 0) }
    pub fn sub(&self, o: &Tensor) -> Tensor { self.binary(o, 1) }
    pub fn mul(&self, o: &Tensor) -> Tensor { self.binary(o, 2) }
    pub fn div(&self, o: &Tensor) -> Tensor { self.binary(o, 3) }
    pub fn maximum(&self, o: &Tensor) -> Tensor { self.binary(o, 4) }

    fn unary(&self, op: u32) -> Tensor {
        let c = self.contiguous();
        let n = c.numel();
        let out = empty(&self.ctx, n);
        let (ugrid, urs) = groups2d(n);
        run(&self.ctx, UNARY_WGSL, "unary", &[&c.buf, &out, &u32buf(&self.ctx, &[op, n as u32, urs])], ugrid);
        Tensor { ctx: self.ctx.clone(), buf: Arc::new(out), shape: c.shape, strides: c.strides, offset: 0 }
    }
    pub fn exp(&self) -> Tensor { self.unary(0) }
    pub fn neg(&self) -> Tensor { self.unary(1) }
    pub fn relu(&self) -> Tensor { self.unary(2) }
    pub fn sqrt(&self) -> Tensor { self.unary(3) }
    pub fn relu_mask(&self) -> Tensor { self.unary(4) } // 1 where x>0 else 0 (relu' )
    pub fn abs(&self) -> Tensor { self.unary(5) }
    pub fn sigmoid(&self) -> Tensor { self.unary(6) }
    pub fn silu(&self) -> Tensor { self.unary(7) }
    pub fn gelu(&self) -> Tensor { self.unary(8) }
    /// GELU via the tanh approximation (`gelu_pytorch_tanh`) — what Gemma and GPT-2 actually use, vs
    /// the exact erf `gelu()`. Matters for matching those models' reference numerics.
    pub fn gelu_tanh(&self) -> Tensor { self.unary(12) }
    /// `tanh` (clamped to ±15 for f32 stability). Building block for Gemma-2 logit softcapping.
    pub fn tanh(&self) -> Tensor { self.unary(13) }
    /// Gemma-2 logit softcap: `cap · tanh(x / cap)` — squashes logits into (−cap, cap).
    pub fn softcap(&self, cap: f32) -> Tensor { self.mul(&self.scalar(1.0 / cap)).tanh().mul(&self.scalar(cap)) }
    pub fn log(&self) -> Tensor { self.unary(9) }
    pub fn sin(&self) -> Tensor { self.unary(14) }
    pub fn cos(&self) -> Tensor { self.unary(15) }
    pub fn relu2(&self) -> Tensor { self.unary(10) } // ReLU² (BitNet FFN)
    pub fn softplus(&self) -> Tensor { self.unary(11) } // log(1+eˣ) — Qwen3.5 gate
    pub fn scalar(&self, s: f32) -> Tensor { Tensor::from_vec(&self.ctx, &[s], &[1]) }

    // ---- fused transformer fast-paths (same result as composing primitives, fewer dispatches) ----
    /// Softmax over `axis` (fused: per-row max/exp/sum/div in one kernel).
    pub fn softmax(&self, axis: usize) -> Tensor {
        let r = self.rank();
        let mut perm: Vec<usize> = (0..r).collect();
        perm.remove(axis);
        perm.push(axis);
        let p = self.permute(&perm).contiguous();
        let d = p.shape[r - 1];
        let rows = p.numel() / d;
        let out = empty(&self.ctx, p.numel());
        run(&self.ctx, SOFTMAX_WGSL, "softmax", &[p.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, d as u32])], groups(rows));
        let sm = Tensor::from_parts(&self.ctx, out, p.shape.clone());
        let mut inv = vec![0usize; r];
        for (i, &pp) in perm.iter().enumerate() { inv[pp] = i; }
        sm.permute(&inv).contiguous()
    }
    /// RMSNorm over the last dim: x/sqrt(mean(x²)+eps)·weight (fused).
    pub fn rmsnorm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let c = self.contiguous();
        let d = *c.shape.last().unwrap();
        let rows = c.numel() / d;
        let out = empty(&self.ctx, c.numel());
        run(&self.ctx, RMSNORM_WGSL, "rmsnorm", &[c.buf.as_ref(), weight.contiguous().buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, d as u32, eps.to_bits()])], groups(rows));
        Tensor::from_parts(&self.ctx, out, c.shape.clone())
    }

    /// Fused residual-add + RMSNorm: returns `(sum, rmsnorm(sum)·weight)` where `sum = self + other`.
    /// Every transformer layer boundary does exactly `xy = x + y; xy.rmsnorm(w)` and needs *both* the
    /// sum (as the next residual) and its norm (as the next block's input) — so folding the add into
    /// the norm kernel removes one dispatch per layer from the latency-bound decode chain, with output
    /// **bit-identical** to `self.add(other).rmsnorm(w)` (same f32 ops, same order).
    pub fn add_rmsnorm(&self, other: &Tensor, weight: &Tensor, eps: f32) -> (Tensor, Tensor) {
        let a = self.contiguous();
        let b = other.contiguous();
        assert_eq!(a.shape, b.shape, "add_rmsnorm: shape mismatch {:?} vs {:?}", a.shape, b.shape);
        let d = *a.shape.last().unwrap();
        let rows = a.numel() / d;
        let sumo = empty(&self.ctx, a.numel());
        let normo = empty(&self.ctx, a.numel());
        run(&self.ctx, ADD_RMSNORM_WGSL, "add_rmsnorm",
            &[a.buf.as_ref(), b.buf.as_ref(), weight.contiguous().buf.as_ref(), &sumo, &normo,
              &u32buf(&self.ctx, &[rows as u32, d as u32, eps.to_bits()])], groups(rows));
        (Tensor::from_parts(&self.ctx, sumo, a.shape.clone()), Tensor::from_parts(&self.ctx, normo, a.shape.clone()))
    }
    /// L2 normalize over the last dim: `x / max(√Σx², eps)`. Distinct from RMSNorm — no mean, no
    /// learned weight, and eps clamps the divisor rather than being added under the root. This is
    /// what Qwen3.5 / Bonsai applies to the gated-delta-net q and k.
    pub fn l2norm(&self, eps: f32) -> Tensor {
        let c = self.contiguous();
        let d = *c.shape.last().unwrap();
        let rows = c.numel() / d;
        let out = empty(&self.ctx, c.numel());
        run(&self.ctx, L2NORM_WGSL, "l2norm", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, d as u32, eps.to_bits()])], groups(rows));
        Tensor::from_parts(&self.ctx, out, c.shape.clone())
    }

    /// Rotary position embedding (NeoX rotate-half) on a [T, n_heads·head_dim] tensor.
    pub fn rope(&self, n_heads: usize, head_dim: usize, base: f32, offset: usize) -> Tensor {
        let c = self.contiguous();
        let t = c.numel() / (n_heads * head_dim);
        let out = empty(&self.ctx, c.numel());
        run(&self.ctx, ROPE_WGSL, "rope", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[t as u32, n_heads as u32, head_dim as u32, base.to_bits(), offset as u32])], groups(t * n_heads));
        Tensor::from_parts(&self.ctx, out, c.shape.clone())
    }

    /// RoPE with a per-frequency scale (`freq_scale` is `[head_dim/2]`) — Llama-3's rope-scaling, where
    /// `rope_freqs.weight` multiplies each inverse frequency to stretch the effective context window.
    pub fn rope_scaled(&self, freq_scale: &Tensor, n_heads: usize, head_dim: usize, base: f32, offset: usize) -> Tensor {
        let c = self.contiguous();
        let t = c.numel() / (n_heads * head_dim);
        let out = empty(&self.ctx, c.numel());
        run(&self.ctx, ROPE_SCALED_WGSL, "rope_scaled",
            &[c.buf.as_ref(), &out, freq_scale.contiguous().buf.as_ref(),
              &u32buf(&self.ctx, &[t as u32, n_heads as u32, head_dim as u32, base.to_bits(), offset as u32])],
            groups(t * n_heads));
        Tensor::from_parts(&self.ctx, out, c.shape.clone())
    }
    /// **Gated delta rule** — the linear-attention recurrence behind Qwen3-Next / Qwen3.5 (and so
    /// PrismML Bonsai-27B, whose 64 layers are 75% linear attention). Per head, a recurrent state
    /// `S [dk, dv]` evolves over the sequence:
    ///   `S = S·gₜ` ; `Δ = (vₜ − kₜᵀS)·βₜ` ; `S += kₜ ⊗ Δ` ; `outₜ = qₜᵀS`
    /// self = q [T,H,dk] (expected L2-normed and scaled), k [T,H,dk], v [T,H,dv], gb [T,H,2] = (g, β)
    /// where the decay is `exp(g)`. Returns [T,H,dv]. One thread owns a column S[:,j] and walks T.
    pub fn gated_delta_rule(&self, k: &Tensor, v: &Tensor, gb: &Tensor, h: usize, dk: usize, dv: usize) -> Tensor {
        self.gated_delta_rule_stateful(k, v, gb, h, dk, dv, None).0
    }

    /// Gated delta rule that can **resume from a carried state** and returns the evolved one —
    /// what turns generation from re-running the whole prefix per token into a single step.
    /// `state` is `[H, dv, dk]`; `None` starts from zero. Returns `(out [T,H,dv], state)`.
    pub fn gated_delta_rule_stateful(&self, k: &Tensor, v: &Tensor, gb: &Tensor, h: usize, dk: usize, dv: usize, state: Option<&Tensor>) -> (Tensor, Tensor) {
        assert!(dk <= 128, "gated_delta_rule: head_k_dim ≤ 128");
        let (q, k, v, gb) = (self.contiguous(), k.contiguous(), v.contiguous(), gb.contiguous());
        let t = q.numel() / (h * dk);
        let out = empty(&self.ctx, t * h * dv);
        // The kernel reads and writes state in place, so hand it a buffer it owns either way.
        let st = match state {
            Some(s) => s.contiguous(),
            None => Tensor::zeros(&self.ctx, &[h, dv, dk]),
        };
        run(&self.ctx, GATED_DELTA_WGSL, "gdn",
            &[q.buf.as_ref(), k.buf.as_ref(), v.buf.as_ref(), gb.buf.as_ref(), &out, st.buf.as_ref(),
              &u32buf(&self.ctx, &[t as u32, h as u32, dk as u32, dv as u32, state.is_some() as u32])],
            groups(h * dv));
        (Tensor::from_parts(&self.ctx, out, vec![t, h, dv]), st)
    }

    /// 3D rotary position embedding (V-JEPA 2): head_dim split into 3 groups (temporal/height/width),
    /// each rotated by the token's coordinate along that axis. self is [T, n_heads·head_dim], T=gt·gh·gw.
    pub fn rope_3d(&self, n_heads: usize, head_dim: usize, base: f32, gt: usize, gh: usize, gw: usize) -> Tensor {
        let c = self.contiguous();
        let t = c.numel() / (n_heads * head_dim);
        assert_eq!(t, gt * gh * gw, "T must equal gt·gh·gw");
        assert_eq!(head_dim % 6, 0, "head_dim must be divisible by 6 for 3D RoPE");
        let out = empty(&self.ctx, c.numel());
        run(&self.ctx, ROPE_3D_WGSL, "rope3d", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[t as u32, n_heads as u32, head_dim as u32, gt as u32, gh as u32, gw as u32, base.to_bits()])], groups(t * n_heads));
        Tensor::from_parts(&self.ctx, out, c.shape.clone())
    }

    /// Causal depthwise conv1d — the LFM2 / Liquid AI short-conv mixer. self is [T, C] (sequence ×
    /// channels), weight is [C, L] (per-channel kernel of length L). Causal: out[t] sees only t-L+1..t.
    pub fn depthwise_conv1d_causal(&self, weight: &Tensor, l: usize) -> Tensor {
        let c = self.contiguous();
        let (t, ch) = (c.shape[0], c.shape[1]);
        let out = empty(&self.ctx, t * ch);
        run(&self.ctx, CONV1D_WGSL, "conv1d", &[c.buf.as_ref(), weight.contiguous().buf.as_ref(), &out, &u32buf(&self.ctx, &[t as u32, ch as u32, l as u32, 0])], groups(t * ch));
        Tensor::from_parts(&self.ctx, out, vec![t, ch])
    }

    /// Fused **SwiGLU**: `self` is a [t, 2d] tensor holding [gate | up] side by side; returns
    /// `silu(gate) ⊙ up` as [t, d] in one kernel. Replaces the silu + mul pair in every FFN.
    pub fn swiglu(&self, d: usize) -> Tensor {
        let c = self.contiguous();
        let t = c.numel() / (2 * d);
        let out = empty(&self.ctx, t * d);
        run(&self.ctx, SWIGLU_WGSL, "swiglu", &[c.buf.as_ref(), &out, &u32buf(&self.ctx, &[(t * d) as u32, d as u32])], groups(t * d));
        Tensor::from_parts(&self.ctx, out, vec![t, d])
    }

    /// **Flash-attention prefill** (causal, GQA) — one workgroup per (query, head) runs three shared-
    /// memory phases (scores over keys ≤ query → softmax → weighted-V), so it **never materializes the
    /// [nh, T, T] scores matrix** the composed `causal_attention` does (268 MB at T=2048). O(T) memory
    /// instead of O(T²). `self`=q [T, nh·dh], k/v [T, nkv·dh]. Any T (keys are streamed in 2048-key
    /// chunks with online softmax); T ≤ 65535 for the dispatch grid. Same math as `causal_attention`.
    pub fn flash_attention_prefill(&self, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize) -> Tensor {
        let (q, k, v) = (self.contiguous(), k.contiguous(), v.contiguous());
        let t = q.shape[0];
        assert!(dh <= 128 && t <= 65535, "flash prefill: head_dim ≤ 128, T ≤ 65535");
        let out = empty(&self.ctx, t * nh * dh);
        let scale = 1.0 / (dh as f32).sqrt();
        run(&self.ctx, FLASH_ATTN_PREFILL_WGSL, "flash_prefill",
            &[q.buf.as_ref(), k.buf.as_ref(), v.buf.as_ref(), &out,
              &unibuf(&self.ctx, &[nh as u32, nkv as u32, dh as u32, t as u32, scale.to_bits(), 0, 0, 0])],
            (nh as u32, t as u32, 1));
        Tensor::from_parts(&self.ctx, out, vec![t, nh * dh])
    }

    /// **Fused single-query attention** (the decode step). `self`=q [1, nh·dh], k/v [S, nkv·dh].
    /// One workgroup per query head runs the whole head in one pass — scores, softmax, and the
    /// weighted V-sum — with no intermediate tensors. Replaces the ~12-dispatch composed path
    /// (reshape/permute/contiguous/broadcast/matmul/softmax/matmul), which dominates decode when a
    /// model has attention in every layer. Any cache length S (keys stream in 2048-key chunks with
    /// online softmax — O(dh) state, no S cap). Returns [1, nh·dh].
    pub fn fused_decode_attention(&self, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize) -> Tensor {
        let (q, k, v) = (self.contiguous(), k.contiguous(), v.contiguous());
        let s = k.numel() / (nkv * dh);
        assert!(dh <= 128, "fused_decode_attention: head_dim ≤ 128");
        let out = empty(&self.ctx, nh * dh);
        let scale = 1.0 / (dh as f32).sqrt();
        run(&self.ctx, FUSED_ATTN_WGSL, "fattn",
            &[q.buf.as_ref(), k.buf.as_ref(), v.buf.as_ref(), &out,
              &unibuf(&self.ctx, &[nh as u32, nkv as u32, dh as u32, s as u32, scale.to_bits(), 0, 0, 0])],
            (nh as u32, 1, 1));
        Tensor::from_parts(&self.ctx, out, vec![1, nh * dh])
    }

    /// y = x·Wᵀ where x is [rows,in] and W is stored [out,in] (HF linear convention) — computed
    /// directly, without materializing Wᵀ. Essential for big tied LM heads (avoids a huge transpose).
    pub fn matmul_bt(&self, w: &Tensor) -> Tensor {
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "matmul_bt is 2D");
        let (rows, inn) = (x.shape[0], x.shape[1]);
        let wc = w.contiguous();
        let out_f = wc.shape[0];
        assert_eq!(inn, wc.shape[1], "inner dims mismatch");
        #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
        if let Some(t) = metal4_linear(&self.ctx, &x, &wc, rows, inn, out_f, 0) {
            return t;
        }
        let out = empty(&self.ctx, rows * out_f);
        run(&self.ctx, MATMUL_BT_WGSL, "matmul_bt", &[x.buf.as_ref(), wc.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, out_f as u32, inn as u32])], groups(rows * out_f));
        Tensor::from_parts(&self.ctx, out, vec![rows, out_f])
    }

    /// y = act(x·Wᵀ) — a linear projection with the activation fused into the matmul epilogue (one
    /// kernel, no intermediate). act: 0 identity, 1 relu, 2 silu, 3 gelu, 4 sigmoid. Every gated FFN
    /// (silu(x·Wgateᵀ)) and every relu/gelu MLP hidden layer collapses to a single dispatch.
    pub fn matmul_bt_act(&self, w: &Tensor, act: u32) -> Tensor {
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "matmul_bt_act is 2D");
        let (rows, inn) = (x.shape[0], x.shape[1]);
        let wc = w.contiguous();
        let out_f = wc.shape[0];
        #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
        if let Some(t) = metal4_linear(&self.ctx, &x, &wc, rows, inn, out_f, act) {
            return t;
        }
        let out = empty(&self.ctx, rows * out_f);
        run(&self.ctx, MATMUL_BT_ACT_WGSL, "matmul_bt_act", &[x.buf.as_ref(), wc.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, out_f as u32, inn as u32, act])], groups(rows * out_f));
        Tensor::from_parts(&self.ctx, out, vec![rows, out_f])
    }

    /// 2D convolution: `self` is NHWC activations `[n, h, w, c]`, `w` is HWIO weights
    /// `[kh, kw, c, o]` (the MPP/TF layout family), output NHWO `[n, ho, wo, o]` with
    /// `ho = (h + 2·pad.0 − kh)/stride.0 + 1` (and likewise for width). Direct portable kernel —
    /// the fabric's conv baseline on every backend.
    pub fn conv2d(&self, w: &Tensor, stride: (usize, usize), pad: (usize, usize)) -> Tensor {
        let x = self.contiguous();
        let wc = w.contiguous();
        assert_eq!(x.rank(), 4, "conv2d activations must be [n,h,w,c]");
        assert_eq!(wc.rank(), 4, "conv2d weights must be [kh,kw,c,o]");
        let (n, h, wd, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let (kh, kw, wc_c, o) = (wc.shape[0], wc.shape[1], wc.shape[2], wc.shape[3]);
        assert_eq!(c, wc_c, "conv2d channel mismatch: activations {c} vs weights {wc_c}");
        assert!(h + 2 * pad.0 >= kh && wd + 2 * pad.1 >= kw, "kernel larger than padded input");
        let ho = (h + 2 * pad.0 - kh) / stride.0 + 1;
        let wo = (wd + 2 * pad.1 - kw) / stride.1 + 1;
        let out = empty(&self.ctx, n * ho * wo * o);
        let (grid, rs) = groups2d(n * ho * wo * o);
        run(&self.ctx, CONV2D_WGSL, "conv2d",
            &[&x.buf, &wc.buf, &out,
              &u32buf(&self.ctx, &[n as u32, h as u32, wd as u32, c as u32, kh as u32, kw as u32,
                  o as u32, ho as u32, wo as u32, stride.0 as u32, stride.1 as u32,
                  pad.0 as u32, pad.1 as u32, rs])],
            grid);
        Tensor::from_parts(&self.ctx, out, vec![n, ho, wo, o])
    }

    /// Row gather (embedding lookup): self is a [vocab, d] table; returns [idx.len(), d].
    pub fn gather_rows(&self, idx: &[u32]) -> Tensor {
        let d = *self.shape.last().unwrap();
        let c = self.contiguous();
        let out = empty(&self.ctx, idx.len() * d);
        let idxbuf = u32buf(&self.ctx, idx);
        run(&self.ctx, GATHER_ROWS_WGSL, "gather_rows", &[c.buf.as_ref(), &idxbuf, &out, &u32buf(&self.ctx, &[idx.len() as u32, d as u32])], groups(idx.len() * d));
        Tensor::from_parts(&self.ctx, out, vec![idx.len(), d])
    }
    pub(crate) fn ctx_arc(&self) -> Arc<Context> { self.ctx.clone() }

    // ---- general reduction over arbitrary axes ----
    fn reduce(&self, axes: &[usize], op: u32, keepdim: bool) -> Tensor {
        let mut ax: Vec<usize> = axes.to_vec();
        ax.sort_unstable();
        ax.dedup();
        let keep: Vec<usize> = (0..self.rank()).filter(|d| !ax.contains(d)).collect();
        // permute reduced axes to the end, materialize → [outer, red]
        let perm: Vec<usize> = keep.iter().chain(ax.iter()).copied().collect();
        let moved = self.permute(&perm).contiguous();
        let red: usize = ax.iter().map(|&d| self.shape[d]).product();
        let outer: usize = moved.numel() / red.max(1);
        // Large reduced axes go through staged grid-stride passes so the work parallelizes —
        // the plain kernel gives each output ONE thread, catastrophic when outer is small
        // (a scalar sum over 8M elements = one thread, ~100 ms).
        let mut src = moved.buf.clone();
        let mut red = red;
        while red > 4096 {
            let nchunk = red.div_ceil(4096);
            let stage = Arc::new(empty(&self.ctx, outer * nchunk));
            run(&self.ctx, REDUCE_STAGE_WGSL, "reduce_stage",
                &[&src, &stage, &u32buf(&self.ctx, &[outer as u32, red as u32, nchunk as u32, op])],
                groups(outer * nchunk));
            src = stage;
            red = nchunk;
        }
        let out = empty(&self.ctx, outer);
        run(&self.ctx, REDUCE_WGSL, "reduce", &[&src, &out, &u32buf(&self.ctx, &[outer as u32, red as u32, op])], groups(outer));
        let mut oshape: Vec<usize> = keep.iter().map(|&d| self.shape[d]).collect();
        if keepdim {
            oshape = (0..self.rank()).map(|d| if ax.contains(&d) { 1 } else { self.shape[d] }).collect();
        }
        if oshape.is_empty() { oshape.push(1); }
        Tensor { ctx: self.ctx.clone(), buf: Arc::new(out), strides: contig_strides(&oshape), shape: oshape, offset: 0 }
    }
    pub fn sum(&self, axes: &[usize], keepdim: bool) -> Tensor { self.reduce(axes, 0, keepdim) }
    pub fn max(&self, axes: &[usize], keepdim: bool) -> Tensor { self.reduce(axes, 1, keepdim) }
    pub fn mean(&self, axes: &[usize], keepdim: bool) -> Tensor {
        let n: usize = axes.iter().map(|&d| self.shape[d]).product();
        let s = self.sum(axes, keepdim);
        let inv = Tensor::from_vec(&self.ctx, &[1.0 / n as f32], &[1]);
        s.mul(&inv)
    }

    // ---- batched matmul: [..., m, k] x [..., k, n] -> [..., m, n], batch dims broadcast ----
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (ra, rb) = (self.rank(), other.rank());
        assert!(ra >= 2 && rb >= 2, "matmul needs rank >= 2");
        let (m, ka) = (self.shape[ra - 2], self.shape[ra - 1]);
        let (kb, n) = (other.shape[rb - 2], other.shape[rb - 1]);
        assert_eq!(ka, kb, "matmul inner dims {ka} != {kb}");
        let batch_a = &self.shape[..ra - 2];
        let batch_b = &other.shape[..rb - 2];
        let batch = broadcast_shapes(batch_a, batch_b);
        let bn: usize = numel(&batch);
        let a_full: Vec<usize> = batch.iter().chain([m, ka].iter()).copied().collect();
        let b_full: Vec<usize> = batch.iter().chain([kb, n].iter()).copied().collect();
        // Cooperative-matrix (tensor-core) fast path — opt-in (`FERRIC_COOP=1`), for the f32-GEMM
        // heavy paths (training, prefill) where the 6–32× beats naive. Not the default: coop is
        // fp-order/precision dependent (NVIDIA TF32), so it must not silently change inference logits
        // or the bit-identical cross-fabric guarantee. Only plain 2D GEMMs with 8-aligned dims.
        if bn == 1 && ra == 2 && rb == 2 && m % 8 == 0 && ka % 8 == 0 && n % 8 == 0
            && self.ctx.coop_gemm_ok() && std::env::var("FERRIC_COOP").is_ok()
        {
            return self.matmul_coop(other);
        }
        let a = self.broadcast_to(&a_full).contiguous();
        let b = other.broadcast_to(&b_full).contiguous();
        // Metal-4 tensor-unit resident path — opt-in (`FERRIC_METAL4=1`): like FERRIC_COOP, a
        // precision-changing fast path (fp16 inputs by contract) must never silently alter default
        // results. Everything stays on-GPU — pad+convert, `matmul2d` on the tensor units, and unpad
        // run as one MTL4 command buffer on wgpu's own MTLDevice; only control crosses the host.
        // Below ~1e8 flops the ~0.2 ms tensor-unit dispatch loses to the WGSL kernels, so small ops
        // stay on the portable path.
        #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
        if crate::metal4::resident_ready(&self.ctx, 2 * bn * m * ka * n) {
            if let Some(g) = crate::metal4::resident_for(&self.ctx) {
                // Fresh out buffers need one clear_buffer pass: it marks the buffer INITIALIZED in
                // wgpu's init tracker — without it, wgpu lazily zero-fills the buffer on first wgpu
                // use, clobbering the external queue's writes. Pooled buffers already had it, so
                // reuse (the training pattern) skips the ~170 µs clear-submit round trip. Either
                // submit also flushes staged uploads (poll alone never runs them); the poll then
                // drains the queue so a/b are fully produced — and any in-flight readers of a
                // recycled out buffer are finished — before the tensor units touch them.
                let (out, fresh) = crate::metal4::pooled_out(&self.ctx, bn * m * n);
                if fresh {
                    let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    enc.clear_buffer(&out, 0, None);
                    self.ctx.queue.submit([enc.finish()]);
                } else {
                    self.ctx.queue.submit([]);
                }
                device_sync(&self.ctx);
                if g.bmm_resident(&a.buf, a.offset * 4, &b.buf, b.offset * 4, &out, bn, m, ka, n).is_some() {
                    let oshape: Vec<usize> = batch.iter().chain([m, n].iter()).copied().collect();
                    return Tensor::from_arc(&self.ctx, out, &oshape);
                }
            }
        }
        let out = empty(&self.ctx, bn * m * n);
        // Pick the GEMM kernel by what the autotuner measured fastest for this shape+device (naive vs
        // register-blocked tiled). No single kernel wins on every GPU, so we select by measurement:
        // on M5 Max/Metal that's naive (~587 GFLOP/s); on GPUs where tiling wins, it's tiled. Untuned
        // shapes default to naive (never a regression). See `autotune_matmul` + docs/SOTA.md #1/#6.
        let use_tiled = bn == 1 && gemm_choice(m, ka, n) == Gemm::Tiled;
        if use_tiled {
            let (gx, gy) = ((n as u32).div_ceil(64), (m as u32).div_ceil(64));
            run(&self.ctx, TILED_MATMUL_WGSL, "mm_tiled", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[m as u32, ka as u32, n as u32, 0])], (gx, gy, 1));
        } else {
            let (grid, rs) = groups2d(bn * m * n);
            run(&self.ctx, MATMUL_WGSL, "bmm", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[bn as u32, m as u32, ka as u32, n as u32, rs])], grid);
        }
        let oshape: Vec<usize> = batch.iter().chain([m, n].iter()).copied().collect();
        Tensor { ctx: self.ctx.clone(), buf: Arc::new(out), strides: contig_strides(&oshape), shape: oshape, offset: 0 }
    }

    /// Autotune the GEMM kernel for this shape on this device: time naive vs tiled and cache the
    /// winner (keyed by shape bucket). Subsequent `matmul`s of the same bucket use it. Returns the
    /// choice. This is how GEMM stays fast *portably* — the winner differs across GPUs.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn autotune_matmul(&self, other: &Tensor) -> &'static str {
        let (m, ka) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let n = other.shape[other.rank() - 1];
        let time = |f: &dyn Fn() -> Tensor| {
            let t0 = std::time::Instant::now();
            for _ in 0..8 { let _ = pollster::block_on(f().to_vec()); }
            t0.elapsed()
        };
        let naive = time(&|| self.matmul_naive(other));
        let tiled = time(&|| self.matmul_tiled(other));
        let win = if tiled < naive { Gemm::Tiled } else { Gemm::Naive };
        GEMM_CACHE.with(|c| c.borrow_mut().insert(gemm_bucket(m, ka, n), win));
        if win == Gemm::Tiled { "tiled" } else { "naive" }
    }

    /// Force the register-blocked tiled 2D GEMM (for benchmarking / large matmuls).
    pub fn matmul_tiled(&self, other: &Tensor) -> Tensor {
        let (m, ka) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let n = other.shape[other.rank() - 1];
        let (a, b) = (self.contiguous(), other.contiguous());
        let out = empty(&self.ctx, m * n);
        let (gx, gy) = ((n as u32).div_ceil(64), (m as u32).div_ceil(64));
        run(&self.ctx, TILED_MATMUL_WGSL, "mm_tiled", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[m as u32, ka as u32, n as u32, 0])], (gx, gy, 1));
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }

    /// Register-tiled 2D matmul (1×8 per thread). Requires n % 8 == 0. The measured-fastest f32 GEMM
    /// on Apple Silicon — it beats both naive and shared-memory tiling by reusing each A load across
    /// 8 columns while letting the hardware cache serve reuse instead of staging through shared memory.
    pub fn matmul_rt(&self, other: &Tensor) -> Tensor {
        let (m, ka) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let n = other.shape[other.rank() - 1];
        assert_eq!(n % 8, 0, "matmul_rt needs n % 8 == 0");
        let (a, b) = (self.contiguous(), other.contiguous());
        let out = empty(&self.ctx, m * n);
        let (grid, rs) = groups2d(m * (n / 8));
        run(&self.ctx, MATMUL_RT_WGSL, "mm_rt", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[m as u32, ka as u32, n as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }

    /// The naive (non-tiled) matmul, kept for benchmarking the tiled fast-path against.
    pub fn matmul_naive(&self, other: &Tensor) -> Tensor {
        let (ra, rb) = (self.rank(), other.rank());
        let (m, ka) = (self.shape[ra - 2], self.shape[ra - 1]);
        let n = other.shape[rb - 1];
        let a = self.contiguous();
        let b = other.contiguous();
        let out = empty(&self.ctx, m * n);
        let (grid, rs) = groups2d(m * n);
        run(&self.ctx, MATMUL_WGSL, "bmm", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[1, m as u32, ka as u32, n as u32, rs])], grid);
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }
}

// A tiny copy kernel: dst[dst_off + i] = src[i]. Used to append K/V rows into a preallocated cache
// buffer in place. Dispatched through `run()`, so it inherits batch ordering + buffer retention.
const KV_WRITE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       src: array<f32>;
@group(0) @binding(1) var<storage,read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>            info: vec4<u32>;   // dst_off, n, grid_w, _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info.z;
    if (i < info.y) { dst[info.x + i] = src[i]; }
}
"#;
fn write_rows(ctx: &Context, dst: &wgpu::Buffer, dst_off: usize, src: &Tensor) {
    let src = src.contiguous();
    let n = src.numel();
    if n == 0 { return; }
    let (grid, rs) = groups2d(n);
    run(ctx, KV_WRITE_WGSL, "kv_write", &[&src.buf, dst, &unibuf(ctx, &[dst_off as u32, n as u32, rs, 0])], grid);
}

/// A **grow-in-place K/V cache buffer**. Instead of re-concatenating the whole history every decode
/// step (`pk.cat(&k, 0)` — O(len) copy per step, O(T²) total), this preallocates and appends only the
/// new rows into a capacity buffer that doubles when full — amortized O(1) rows copied per step, so a
/// long generation is O(T) not O(T²). Attention reads a zero-copy [len, width] view of the buffer.
/// The stored data is byte-identical to the concatenated tensor, so logits are unchanged.
#[derive(Default)]
pub struct KvBuf {
    buf: Option<Arc<wgpu::Buffer>>,
    len: usize,   // filled rows
    cap: usize,   // capacity rows
    width: usize, // row width in elements
}
impl KvBuf {
    pub fn len(&self) -> usize { self.len }
    /// Append `src` ([t, width]) rows in place, growing (doubling) if needed, and return a contiguous
    /// [len, width] view over the cache buffer covering all rows so far.
    pub fn append(&mut self, ctx: &Arc<Context>, src: &Tensor) -> Tensor {
        assert_eq!(src.rank(), 2, "KvBuf::append expects a 2D [t, width] tensor");
        let (t, width) = (src.shape[0], src.shape[1]);
        if self.width == 0 { self.width = width; }
        assert_eq!(width, self.width, "KvBuf width changed");
        let need = self.len + t;
        if self.cap < need {
            let new_cap = need.max(self.cap * 2).max(64);
            let nb = Arc::new(empty(ctx, new_cap * width));
            if self.len > 0 {
                // carry the existing rows into the bigger buffer (amortized — happens log(T) times)
                let old = Tensor::from_arc(ctx, self.buf.clone().unwrap(), &[self.len, width]);
                write_rows(ctx, &nb, 0, &old);
            }
            self.buf = Some(nb);
            self.cap = new_cap;
        }
        write_rows(ctx, self.buf.as_ref().unwrap(), self.len * width, src);
        self.len = need;
        Tensor::from_arc(ctx, self.buf.clone().unwrap(), &[self.len, width])
    }
}

// ---------- device plumbing (uses ferric-core Context's public device/queue) ----------
/// The Metal-4 resident linear fast path (`y = act(x·Wᵀ)`, W in HF [out,in] layout) — opt-in via
/// `FERRIC_METAL4=1` under the same precision doctrine as the matmul fast path (fp16 inputs by
/// contract, never silently on), with the same ~1e8-flop floor and the same sync contract: the
/// clear-or-empty submit flushes staged uploads and marks a fresh out buffer initialized, the poll
/// drains wgpu, then the tensor units consume the wgpu buffers directly (NT — no Wᵀ materialized,
/// activation fused into the unpad epilogue).
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
fn metal4_linear(ctx: &Arc<Context>, x: &Tensor, w: &Tensor, rows: usize, inn: usize, out_f: usize, act: u32) -> Option<Tensor> {
    if !crate::metal4::resident_ready(ctx, 2 * rows * inn * out_f) {
        return None;
    }
    let g = crate::metal4::resident_for(ctx)?;
    let (out, fresh) = crate::metal4::pooled_out(ctx, rows * out_f);
    if fresh {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.clear_buffer(&out, 0, None);
        ctx.queue.submit([enc.finish()]);
    } else {
        ctx.queue.submit([]);
    }
    device_sync(ctx);
    g.linear_resident(&x.buf, x.offset * 4, &w.buf, w.offset * 4, &out, rows, inn, out_f, act)?;
    Some(Tensor::from_arc(ctx, out, &[rows, out_f]))
}

fn empty(ctx: &Context, n: usize) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("t"), size: (n.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn u32buf(ctx: &Context, data: &[u32]) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("info"), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}
pub(crate) fn unibuf(ctx: &Context, data: &[u32]) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uinfo"), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
fn groups(n: usize) -> (u32, u32, u32) { (((n as u32) + 63) / 64, 1, 1) }
/// 2D workgroup grid for large launches: a 1D grid caps at 65535 workgroups. Returns the grid plus
/// `row_stride` (threads per grid row = gx·64) so the kernel can reconstruct a flat index.
pub(crate) fn groups2d(n: usize) -> ((u32, u32, u32), u32) {
    let wg = (n as u32).div_ceil(64);
    let gx = wg.min(32768);
    let gy = wg.div_ceil(gx);
    ((gx, gy, 1), gx * 64)
}

// Compile each WGSL kernel's pipeline ONCE and reuse it — recompiling every dispatch (as before)
// dominated runtime for real workloads. Keyed by the kernel's &'static str address (stable per
// kernel); assumes one device per thread, which holds for all Ferric usage. Every SOTA runtime
// caches compiled kernels; this is the single biggest per-op overhead removed.
thread_local! {
    static PIPELINES: std::cell::RefCell<std::collections::HashMap<(usize, u64), wgpu::ComputePipeline>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    // Autotuner: shape-bucket → measured-fastest GEMM kernel for this device.
    static GEMM_CACHE: std::cell::RefCell<std::collections::HashMap<(u32, u32, u32), Gemm>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}
#[derive(Clone, Copy, PartialEq)]
enum Gemm { Naive, Tiled }
fn gemm_bucket(m: usize, k: usize, n: usize) -> (u32, u32, u32) {
    let b = |x: usize| -> u32 { if x <= 128 { 128 } else if x <= 256 { 256 } else if x <= 512 { 512 } else { 1024 } };
    (b(m), b(k), b(n))
}
fn gemm_choice(m: usize, k: usize, n: usize) -> Gemm {
    GEMM_CACHE.with(|c| c.borrow().get(&gemm_bucket(m, k, n)).copied()).unwrap_or(Gemm::Naive)
}
fn pipeline_for(ctx: &Context, wgsl: &str, label: &str) -> wgpu::ComputePipeline {
    // key by (device, content-hash): caches dynamically-generated fusion shaders too, and stays
    // correct across multiple GPUs (a device-A pipeline is never reused on device B).
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    wgsl.hash(&mut h);
    let key = ((&ctx.device as *const wgpu::Device) as usize, h.finish());
    PIPELINES.with(|c| {
        c.borrow_mut().entry(key).or_insert_with(|| {
            let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
            });
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: None, module: &module, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None,
            })
        }).clone()
    })
}
thread_local! {
    // Diagnostics: how many dispatches issued, and how many actual queue submits they cost.
    // With batching those diverge — one submit can carry hundreds of dispatches.
    static DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SUBMITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    // When Some, run() records dispatches into this encoder instead of submitting each one. The
    // Vec retains every bind group (and through it, the buffers) referenced by a recorded-but-
    // unsubmitted pass — otherwise an intermediate tensor dropped mid-batch would free a buffer the
    // encoder still points at, which wgpu rejects at submit as an invalid resource.
    static BATCH: std::cell::RefCell<Option<(wgpu::CommandEncoder, Vec<wgpu::BindGroup>)>> = const { std::cell::RefCell::new(None) };
}

/// (dispatches, submits) issued so far on this thread.
pub fn op_counters() -> (u64, u64) { (DISPATCHES.with(|c| c.get()), SUBMITS.with(|c| c.get())) }
pub fn reset_op_counters() { DISPATCHES.with(|c| c.set(0)); SUBMITS.with(|c| c.set(0)); }

fn run(ctx: &Context, wgsl: &str, label: &str, binds: &[&wgpu::Buffer], g: (u32, u32, u32)) {
    let pipe = pipeline_for(ctx, wgsl, label);
    let entries: Vec<wgpu::BindGroupEntry> = binds.iter().enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout: &pipe.get_bind_group_layout(0), entries: &entries,
    });
    DISPATCHES.with(|c| c.set(c.get() + 1));
    let record = |enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(g.0, g.1, g.2);
    };
    // If a batch is open, append to it (retaining the bind group) and defer the submit; otherwise
    // submit this op alone. `bg` comes back out when unbatched so it can be recorded standalone.
    let bg = BATCH.with(|b| {
        if let Some((enc, keep)) = b.borrow_mut().as_mut() { record(enc, &bg); keep.push(bg); None } else { Some(bg) }
    });
    if let Some(bg) = bg {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        record(&mut enc, &bg);
        ctx.queue.submit([enc.finish()]);
        SUBMITS.with(|c| c.set(c.get() + 1));
    }
}

/// Batch every dispatch issued inside `f` into a single command submission. Ops still run in issue
/// order (compute passes on one queue execute serially), so results are identical — this only
/// removes the per-op encoder+submit overhead, which dominates when a forward pass issues hundreds
/// of small kernels. Buffers read inside `f` (`to_vec`) flush the batch first, so reads stay correct.
pub fn batch<R>(ctx: &Arc<Context>, f: impl FnOnce() -> R) -> R {
    // Re-entrant safe: an inner batch() just joins the outer one.
    let outermost = BATCH.with(|b| {
        if b.borrow().is_some() { false } else {
            *b.borrow_mut() = Some((ctx.device.create_command_encoder(&Default::default()), Vec::new())); true
        }
    });
    let r = f();
    if outermost {
        if let Some((enc, _keep)) = BATCH.with(|b| b.borrow_mut().take()) {
            ctx.queue.submit([enc.finish()]);
            SUBMITS.with(|c| c.set(c.get() + 1));
        }
    }
    r
}
async fn readback(ctx: &Context, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    // A read must see all prior compute, so flush any open batch first — otherwise the copy below
    // would be submitted ahead of the deferred dispatches and read stale data.
    if let Some((enc, _keep)) = BATCH.with(|b| b.borrow_mut().take()) {
        ctx.queue.submit([enc.finish()]);
        SUBMITS.with(|c| c.set(c.get() + 1));
    }
    let bytes = (n * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    ctx.queue.submit([enc.finish()]);
    let (tx, rx) = flume::bounded(1);
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv_async().await.unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

// ---------- general kernels ----------
// row-major decode of a linear output index into per-input strided offsets.
const CAT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;
@group(0) @binding(1) var<storage,read>        b: array<f32>;
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>; // rank,dim,n,offA,offB,aDim,shape[r],aStr[r],bStr[r]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let rank = info[0]; let dim = info[1]; let n = info[2]; let aDim = info[5];
    if (i >= n) { return; }
    // decompose i over the OUTPUT shape, then route each thread to whichever side owns its slot
    var rem = i; var from_a = true; var idx_at_dim: u32 = 0u;
    var coord = array<u32, 8>();
    for (var dd: u32 = 0u; dd < rank; dd = dd + 1u) {
        let d = rank - 1u - dd;
        let sz = info[6u + d];
        coord[d] = rem % sz; rem = rem / sz;
    }
    idx_at_dim = coord[dim];
    from_a = idx_at_dim < aDim;
    if (!from_a) { coord[dim] = idx_at_dim - aDim; }
    let stride_base = select(6u + 2u * rank, 6u + rank, from_a);
    var src = select(info[4], info[3], from_a);
    for (var d: u32 = 0u; d < rank; d = d + 1u) { src = src + coord[d] * info[stride_base + d]; }
    out[i] = select(b[src], a[src], from_a);
}
"#;

const BINARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;
@group(0) @binding(1) var<storage,read>        b: array<f32>;
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>; // rank,op,n,offA,offB,shape[r],aStr[r],bStr[r]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rank = info[0]; let op = info[1]; let n = info[2];
    let i = gid.x + gid.y * info[5u + 3u * rank];   // 2D grid: n can exceed 4.19M
    if (i >= n) { return; }
    var ia = info[3]; var ib = info[4]; var rem = i;
    for (var dd: u32 = 0u; dd < rank; dd = dd + 1u) {
        let d = rank - 1u - dd;
        let sz = info[5u + d];
        let idx = rem % sz; rem = rem / sz;
        ia = ia + idx * info[5u + rank + d];
        ib = ib + idx * info[5u + 2u * rank + d];
    }
    let x = a[ia]; let y = b[ib];
    var r: f32 = 0.0;
    switch (op) {
        case 0u: { r = x + y; }
        case 1u: { r = x - y; }
        case 2u: { r = x * y; }
        case 3u: { r = x / y; }
        case 4u: { r = max(x, y); }
        default: { r = x + y; }
    }
    out[i] = r;
}
"#;

const UNARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // op, n, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[2]; if (i >= info[1]) { return; }
    let v = x[i]; var r: f32 = v;
    switch (info[0]) {
        case 0u: { r = exp(v); }
        case 1u: { r = -v; }
        case 2u: { r = max(v, 0.0); }
        case 3u: { r = sqrt(v); }
        case 4u: { if (v > 0.0) { r = 1.0; } else { r = 0.0; } }
        case 5u: { r = abs(v); }
        case 6u: { r = 1.0 / (1.0 + exp(-v)); }
        case 7u: { r = v / (1.0 + exp(-v)); }
        case 8u: {
            let t = 1.0 / (1.0 + 0.3275911 * abs(v * 0.7071067811865476));
            let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-(v * 0.7071067811865476) * (v * 0.7071067811865476));
            let erf = select(-e, e, v >= 0.0);
            r = 0.5 * v * (1.0 + erf);
        }
        case 9u: { r = log(v); }
        case 10u: { let z = max(v, 0.0); r = z * z; } // ReLU² (BitNet FFN)
        case 11u: { r = max(v, 0.0) + log(1.0 + exp(-abs(v))); } // softplus (stable)
        case 12u: { let a = 0.7978845608028654 * (v + 0.044715 * v * v * v); r = 0.5 * v * (1.0 + tanh(clamp(a, -15.0, 15.0))); } // gelu (tanh approx — Gemma/GPT-2); clamp: tanh saturates by ±15 and its exp form overflows f32 past ~44
        case 13u: { r = tanh(clamp(v, -15.0, 15.0)); } // tanh (clamped — WGSL tanh's exp form overflows f32 past ~44); used for Gemma-2 logit softcapping c·tanh(x/c)
        case 14u: { r = sin(v); }
        case 15u: { r = cos(v); }
        default: { r = v; }
    }
    out[i] = r;
}
"#;

const GATHER_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // rank,n,offset,row_stride,shape[r],strides[r]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[3]; let rank = info[0]; let n = info[1]; // 2D grid: n can exceed 4.19M
    if (i >= n) { return; }
    var src = info[2]; var rem = i;
    for (var dd: u32 = 0u; dd < rank; dd = dd + 1u) {
        let d = rank - 1u - dd;
        let sz = info[4u + d];
        let idx = rem % sz; rem = rem / sz;
        src = src + idx * info[4u + rank + d];
    }
    out[i] = x[src];
}
"#;

// One stage of a large-axis reduction: [outer, red] → [outer, nchunk], thread c of each outer row
// accumulating the grid-stride slice {c, c+nchunk, c+2·nchunk, …} (coalesced across threads). The
// driver loops stages until red is small, then the plain kernel finishes. Without this, a scalar
// sum over N elements ran on ONE thread — 40–170 ms per training step at MLP sizes.
const REDUCE_STAGE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;   // [outer, red] contiguous
@group(0) @binding(1) var<storage,read_write>  out: array<f32>; // [outer, nchunk]
@group(0) @binding(2) var<storage,read>        info: array<u32>; // outer, red, nchunk, op
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let outer = info[0]; let red = info[1]; let nchunk = info[2]; let op = info[3];
    if (i >= outer * nchunk) { return; }
    let o = i / nchunk; let c = i % nchunk;
    let base = o * red;
    var acc = x[base + c]; // c < red always (nchunk <= red)
    if (op == 1u) {
        for (var j = c + nchunk; j < red; j = j + nchunk) { acc = max(acc, x[base + j]); }
    } else {
        for (var j = c + nchunk; j < red; j = j + nchunk) { acc = acc + x[base + j]; }
    }
    out[i] = acc;
}
"#;

const REDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;   // [outer, red] contiguous
@group(0) @binding(1) var<storage,read_write>  out: array<f32>; // [outer]
@group(0) @binding(2) var<storage,read>        info: array<u32>; // outer, red, op(0=sum,1=max)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let outer = info[0]; let red = info[1]; let op = info[2];
    if (i >= outer) { return; }
    let base = i * red;
    if (op == 1u) {
        var acc = x[base];
        for (var j: u32 = 1u; j < red; j = j + 1u) { acc = max(acc, x[base + j]); }
        out[i] = acc;
    } else {
        var acc = 0.0;
        for (var j: u32 = 0u; j < red; j = j + 1u) { acc = acc + x[base + j]; }
        out[i] = acc;
    }
}
"#;

const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // rows, d
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; let rows = info[0]; let d = info[1];
    if (row >= rows) { return; }
    let base = row * d;
    var mx = x[base];
    for (var j: u32 = 1u; j < d; j = j + 1u) { mx = max(mx, x[base + j]); }
    var sum = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let e = exp(x[base + j] - mx); out[base + j] = e; sum = sum + e; }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < d; j = j + 1u) { out[base + j] = out[base + j] * inv; }
}
"#;

const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read>        weight: array<f32>;
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>; // rows, d, bitcast(eps)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; let rows = info[0]; let d = info[1]; let eps = bitcast<f32>(info[2]);
    if (row >= rows) { return; }
    let base = row * d;
    var ms = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let v = x[base + j]; ms = ms + v * v; }
    let inv = 1.0 / sqrt(ms / f32(d) + eps);
    for (var j: u32 = 0u; j < d; j = j + 1u) { out[base + j] = x[base + j] * inv * weight[j]; }
}
"#;

// Fused (a+b) then RMSNorm — one workgroup-thread per row, two outputs (the sum and its norm).
const ADD_RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;
@group(0) @binding(1) var<storage,read>        b: array<f32>;
@group(0) @binding(2) var<storage,read>        weight: array<f32>;
@group(0) @binding(3) var<storage,read_write>  sumo:  array<f32>;   // a + b (the next residual)
@group(0) @binding(4) var<storage,read_write>  normo: array<f32>;   // rmsnorm(a+b)·weight
@group(0) @binding(5) var<storage,read>        info: array<u32>;    // rows, d, bitcast(eps)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; let rows = info[0]; let d = info[1]; let eps = bitcast<f32>(info[2]);
    if (row >= rows) { return; }
    let base = row * d;
    var ms = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let s = a[base + j] + b[base + j]; sumo[base + j] = s; ms = ms + s * s; }
    let inv = 1.0 / sqrt(ms / f32(d) + eps);
    for (var j: u32 = 0u; j < d; j = j + 1u) { normo[base + j] = (a[base + j] + b[base + j]) * inv * weight[j]; }
}
"#;

const L2NORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // rows, d, bitcast(eps)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; let rows = info[0]; let d = info[1]; let eps = bitcast<f32>(info[2]);
    if (row >= rows) { return; }
    let base = row * d;
    var ss = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { let v = x[base + j]; ss = ss + v * v; }
    let inv = 1.0 / max(sqrt(ss), eps);   // eps clamps the divisor (not added under the root)
    for (var j: u32 = 0u; j < d; j = j + 1u) { out[base + j] = x[base + j] * inv; }
}
"#;

// Like ROPE_WGSL but each inverse frequency is multiplied by scale[c] (Llama-3 rope_freqs.weight).
const ROPE_SCALED_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        scale: array<f32>; // [dh/2] per-freq multiplier
@group(0) @binding(3) var<storage,read>        info: array<u32>;  // t, h, dh, bitcast(base), offset
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = info[0]; let h = info[1]; let dh = info[2]; let base = bitcast<f32>(info[3]); let off = info[4];
    let id = gid.x; if (id >= t * h) { return; }
    let i = id / h; let head = id % h; let half = dh / 2u;
    let o = (i * h + head) * dh; let lb = log(base);
    for (var c: u32 = 0u; c < half; c = c + 1u) {
        let inv = exp(-2.0 * f32(c) / f32(dh) * lb) * scale[c];
        let ang = f32(i + off) * inv; let cs = cos(ang); let sn = sin(ang);
        let x1 = x[o + c]; let x2 = x[o + c + half];
        out[o + c] = x1 * cs - x2 * sn;
        out[o + c + half] = x2 * cs + x1 * sn;
    }
}
"#;

const ROPE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // t, h, dh, bitcast(base), offset
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = info[0]; let h = info[1]; let dh = info[2]; let base = bitcast<f32>(info[3]); let off = info[4];
    let id = gid.x; if (id >= t * h) { return; }
    let i = id / h; let head = id % h; let half = dh / 2u;
    let o = (i * h + head) * dh; let lb = log(base);
    for (var c: u32 = 0u; c < half; c = c + 1u) {
        let inv = exp(-2.0 * f32(c) / f32(dh) * lb);
        let ang = f32(i + off) * inv; let cs = cos(ang); let sn = sin(ang);
        let x1 = x[o + c]; let x2 = x[o + c + half];
        out[o + c] = x1 * cs - x2 * sn;
        out[o + c + half] = x2 * cs + x1 * sn;
    }
}
"#;

const GATED_DELTA_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        q:  array<f32>;   // [T,H,dk] (L2-normed, scaled)
@group(0) @binding(1) var<storage,read>        k:  array<f32>;   // [T,H,dk] (L2-normed)
@group(0) @binding(2) var<storage,read>        v:  array<f32>;   // [T,H,dv]
@group(0) @binding(3) var<storage,read>        gb: array<f32>;   // [T,H,2] = (g, beta)
@group(0) @binding(4) var<storage,read_write>  out: array<f32>;  // [T,H,dv]
@group(0) @binding(5) var<storage,read_write>  state: array<f32>; // [H,dv,dk] — carried across calls
@group(0) @binding(6) var<storage,read>        info: array<u32>; // T,H,dk,dv,load_state
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t_len = info[0]; let h = info[1]; let dk = info[2]; let dv = info[3];
    let load_state = info[4];
    if (idx >= h * dv) { return; }
    let head = idx / dv; let j = idx % dv;   // this thread owns state column S[:, j]
    let sbase = (head * dv + j) * dk;
    var s: array<f32, 128>;
    // Resume from the carried state (incremental decode) or start fresh (prefill).
    if (load_state == 1u) {
        for (var i: u32 = 0u; i < dk; i = i + 1u) { s[i] = state[sbase + i]; }
    } else {
        for (var i: u32 = 0u; i < dk; i = i + 1u) { s[i] = 0.0; }
    }
    for (var t: u32 = 0u; t < t_len; t = t + 1u) {
        let gbase = (t * h + head) * 2u;
        let decay = exp(gb[gbase]);
        let beta = gb[gbase + 1u];
        let kb = (t * h + head) * dk;
        // decay the state, and accumulate kv_mem = kᵀ·S[:,j] against the DECAYED state
        var kv = 0.0;
        for (var i: u32 = 0u; i < dk; i = i + 1u) { s[i] = s[i] * decay; kv = kv + s[i] * k[kb + i]; }
        let delta = (v[(t * h + head) * dv + j] - kv) * beta;
        // rank-1 update S[:,j] += k·delta, then read out = qᵀ·S[:,j] from the UPDATED state
        var o = 0.0;
        for (var i: u32 = 0u; i < dk; i = i + 1u) { s[i] = s[i] + k[kb + i] * delta; o = o + s[i] * q[kb + i]; }
        out[(t * h + head) * dv + j] = o;
    }
    // Hand the evolved state back so the next call can continue the sequence.
    for (var i: u32 = 0u; i < dk; i = i + 1u) { state[sbase + i] = s[i]; }
}
"#;

const ROPE_3D_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // T, H, dh, gt, gh, gw, bitcast(base)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tt = info[0]; let h = info[1]; let dh = info[2];
    let gt = info[3]; let gh = info[4]; let gw = info[5]; let base = bitcast<f32>(info[6]);
    let id = gid.x; if (id >= tt * h) { return; }
    let t = id / h; let head = id % h;
    var co = array<u32, 3>(t / (gh * gw), (t / gw) % gh, t % gw); // (it, ih, iw)
    let g = dh / 3u; let half = g / 2u; let lb = log(base);
    for (var gi: u32 = 0u; gi < 3u; gi = gi + 1u) {
        let coord = f32(co[gi]);
        let off = (t * h + head) * dh + gi * g;
        for (var c: u32 = 0u; c < half; c = c + 1u) {
            let inv = exp(-2.0 * f32(c) / f32(g) * lb);
            let ang = coord * inv; let cs = cos(ang); let sn = sin(ang);
            let x1 = x[off + c]; let x2 = x[off + c + half];
            out[off + c] = x1 * cs - x2 * sn;
            out[off + c + half] = x2 * cs + x1 * sn;
        }
    }
}
"#;

const CONV1D_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;      // [T, C]
@group(0) @binding(1) var<storage,read>        w: array<f32>;      // [C, L]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;    // [T, C]
@group(0) @binding(3) var<storage,read>        info: array<u32>;   // T, C, L
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let t = info[0]; let ch = info[1]; let l = info[2];
    if (idx >= t * ch) { return; }
    let row = idx / ch; let c = idx % ch;
    var acc = 0.0;
    for (var k: u32 = 0u; k < l; k = k + 1u) {
        // causal: source position = row - (L-1) + k
        let off = i32(row) - i32(l) + 1 + i32(k);
        if (off >= 0) { acc = acc + w[c * l + k] * x[u32(off) * ch + c]; }
    }
    out[idx] = acc;
}
"#;

const MATMUL_BT_ACT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;    // [rows, in]
@group(0) @binding(1) var<storage,read>        w: array<f32>;    // [out, in]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;  // [rows, out]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // rows, out, in, act
fn act(v: f32, a: u32) -> f32 {
    switch (a) {
        case 1u: { return max(v, 0.0); }
        case 2u: { return v / (1.0 + exp(-v)); }
        case 3u: {
            let t = 1.0 / (1.0 + 0.3275911 * abs(v * 0.7071067811865476));
            let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-(v * 0.7071067811865476) * (v * 0.7071067811865476));
            let erf = select(-e, e, v >= 0.0);
            return 0.5 * v * (1.0 + erf);
        }
        case 4u: { return 1.0 / (1.0 + exp(-v)); }
        default: { return v; }
    }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let rows = info[0]; let o_dim = info[1]; let in_dim = info[2];
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    var acc = 0.0;
    for (var c: u32 = 0u; c < in_dim; c = c + 1u) { acc = acc + x[r * in_dim + c] * w[o * in_dim + c]; }
    out[idx] = act(acc, info[3]);
}
"#;

const MATMUL_BT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;    // [rows, in]
@group(0) @binding(1) var<storage,read>        w: array<f32>;    // [out, in]  (HF layout)
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;  // [rows, out]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // rows, out, in
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let rows = info[0]; let o_dim = info[1]; let in_dim = info[2];
    if (idx >= rows * o_dim) { return; }
    let o = idx % o_dim; let r = idx / o_dim;
    var acc = 0.0;
    for (var c: u32 = 0u; c < in_dim; c = c + 1u) { acc = acc + x[r * in_dim + c] * w[o * in_dim + c]; }
    out[idx] = acc;
}
"#;

// Direct 2D convolution, NHWC activations x HWIO weights -> NHWO (the MPP/TF layout family).
// One thread per output element; the portable baseline every backend must match.
const CONV2D_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;   // [n, h, w, c]
@group(0) @binding(1) var<storage,read>        wt: array<f32>;  // [kh, kw, c, o]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>; // [n, ho, wo, o]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // n,h,w,c,kh,kw,o,ho,wo,sh,sw,ph,pw,rs
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rs = info[13];
    let i = gid.x + gid.y * rs;
    let n = info[0]; let h = info[1]; let w = info[2]; let c = info[3];
    let kh = info[4]; let kw = info[5]; let o = info[6]; let ho = info[7]; let wo = info[8];
    if (i >= n * ho * wo * o) { return; }
    let oc = i % o; let r1 = i / o;
    let xo = r1 % wo; let r2 = r1 / wo;
    let yo = r2 % ho; let b = r2 / ho;
    var acc = 0.0;
    for (var ky: u32 = 0u; ky < kh; ky = ky + 1u) {
        let yi = i32(yo * info[9]) + i32(ky) - i32(info[11]);
        if (yi < 0 || yi >= i32(h)) { continue; }
        for (var kx: u32 = 0u; kx < kw; kx = kx + 1u) {
            let xi = i32(xo * info[10]) + i32(kx) - i32(info[12]);
            if (xi < 0 || xi >= i32(w)) { continue; }
            let xb = ((b * h + u32(yi)) * w + u32(xi)) * c;
            let wb = (ky * kw + kx) * c * o + oc;
            for (var ci: u32 = 0u; ci < c; ci = ci + 1u) {
                acc = acc + x[xb + ci] * wt[wb + ci * o];
            }
        }
    }
    out[i] = acc;
}
"#;

const GATHER_ROWS_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        table: array<f32>;
@group(0) @binding(1) var<storage,read>        idx: array<u32>;
@group(0) @binding(2) var<storage,read_write>  out: array<f32>;
@group(0) @binding(3) var<storage,read>        info: array<u32>; // n, d
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = info[0]; let d = info[1]; let t = gid.x;
    if (t >= n * d) { return; }
    let i = t / d; let j = t % d;
    out[i * d + j] = table[idx[i] * d + j];
}
"#;

// High-intensity register-blocked GEMM: an 8×8 workgroup (64 threads) computes a 64×64 output tile,
// EACH THREAD an 8×8 micro-tile held in 64 registers. Per K-step every thread loads 8 A-values and
// 8 B-values from shared memory and does 64 FMAs — arithmetic intensity 4× the 4×4 version, which is
// what lifts a WebGPU GEMM toward the >1 TFLOP tier (vs. a cache-friendly naive kernel). BM=BN=64,
// BK=8, TM=TN=8, 64 threads.
// Register-blocked tiled GEMM. 16×16 = 256 threads per workgroup, each owning a **4×4** output
// micro-tile → a 64×64 output tile per workgroup, with a K-tile depth of 16.
//
// The 4×4 (16-accumulator) micro-tile is the crux. The previous version used an 8×8 tile = 64
// accumulators per thread, and WGSL/Metal spills a 64-element register array to thread-local memory,
// which made the "fast" path 2-13× *slower* than the naive kernel it was meant to beat. 16 f32
// accumulators stay in registers (the loops are constant-bound, so they unroll), so each FMA hits a
// register rather than a memory round-trip — the difference between spilling and not.
// Fused single-query attention, one workgroup (128 threads) per query head. GQA: query head h reads
// key/value head h/(nh/nkv). Three phases through shared memory — no per-key barrier, no intermediate
// tensors: (1) each thread computes scores for its slice of the S keys (full dh-dot), (2) a parallel
// max+exp+sum softmax over the shared scores, (3) each thread accumulates the weighted-V for its
// slice of the dh output dims. S ≤ 2048 so scores fit in shared memory.
// gate_up is [t, 2d] laid out row-major as [gate(d) | up(d)] per row; out[i] = silu(gate)·up.
const SWIGLU_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        gu:  array<f32>;   // [t, 2d]
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;   // [t, d]
@group(0) @binding(2) var<storage,read>        info: array<u32>;  // n(=t·d), d
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let n = info[0]; let d = info[1];
    if (i >= n) { return; }
    let row = i / d; let col = i % d;
    let g = gu[row * 2u * d + col];
    let u = gu[row * 2u * d + d + col];
    out[i] = (g / (1.0 + exp(-g))) * u;   // silu(g)·u
}
"#;

// Flash prefill: grid (nh, T). Workgroup (head=wg.x, query=wg.y) attends over causal keys 0..=query,
// processed in chunks of 2048 with **online softmax** (running max m_run, sum l_run, per-thread output
// accumulator accd), so any context length works with O(dh) state — never a [T,T] scores matrix, and
// no 2048 cap. m_run/l_run stay identical across threads (both derive from shared reductions).
const FLASH_ATTN_PREFILL_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        q:   array<f32>;   // [T, nh·dh]
@group(0) @binding(1) var<storage,read>        k:   array<f32>;   // [T, nkv·dh]
@group(0) @binding(2) var<storage,read>        v:   array<f32>;   // [T, nkv·dh]
@group(0) @binding(3) var<storage,read_write>  out: array<f32>;   // [T, nh·dh]
struct Info { a: vec4<u32>, b: vec4<u32> }         // a = (nh,nkv,dh,T); b.x = scale bits
@group(0) @binding(4) var<uniform>             info: Info;
var<workgroup> qs: array<f32, 128>;
var<workgroup> sc: array<f32, 2048>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let nh = info.a.x; let nkv = info.a.y; let dh = info.a.z;
    let scale = bitcast<f32>(info.b.x);
    let head = wg.x; let qi = wg.y; let t = lid.x;
    let g = nh / nkv; let kvh = head / g;
    let s = qi + 1u;                          // causal: keys 0..=qi
    let qbase = qi * nh * dh + head * dh; let kvbase = kvh * dh;
    if (t < dh) { qs[t] = q[qbase + t]; }
    workgroupBarrier();
    var m_run = -3.0e38; var l_run = 0.0; var accd = 0.0;   // running online-softmax state
    for (var c0 = 0u; c0 < s; c0 = c0 + 2048u) {
        let clen = min(2048u, s - c0);
        for (var i = t; i < clen; i = i + 128u) {
            var dot = 0.0; let kb = (c0 + i) * nkv * dh + kvbase;
            for (var d = 0u; d < dh; d = d + 1u) { dot = dot + qs[d] * k[kb + d]; }
            sc[i] = dot * scale;
        }
        workgroupBarrier();
        var cm = -3.0e38;
        for (var i = t; i < clen; i = i + 128u) { cm = max(cm, sc[i]); }
        red[t] = cm; workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) { if (t < stride) { red[t] = max(red[t], red[t + stride]); } workgroupBarrier(); }
        let m_new = max(m_run, red[0]); let corr = exp(m_run - m_new); workgroupBarrier();
        var cs = 0.0;
        for (var i = t; i < clen; i = i + 128u) { let e = exp(sc[i] - m_new); sc[i] = e; cs = cs + e; }
        red[t] = cs; workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) { if (t < stride) { red[t] = red[t] + red[t + stride]; } workgroupBarrier(); }
        l_run = l_run * corr + red[0];
        if (t < dh) {
            var a = 0.0;
            for (var i = 0u; i < clen; i = i + 1u) { a = a + sc[i] * v[(c0 + i) * nkv * dh + kvbase + t]; }
            accd = accd * corr + a;
        }
        m_run = m_new;
        workgroupBarrier();
    }
    if (t < dh) { out[qbase + t] = accd / l_run; }
}
"#;

const FUSED_ATTN_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        q:   array<f32>;   // [nh·dh]
@group(0) @binding(1) var<storage,read>        k:   array<f32>;   // [S, nkv·dh]
@group(0) @binding(2) var<storage,read>        v:   array<f32>;   // [S, nkv·dh]
@group(0) @binding(3) var<storage,read_write>  out: array<f32>;   // [nh·dh]
struct Info { a: vec4<u32>, b: vec4<u32> }         // a = (nh,nkv,dh,S); b.x = scale bits
@group(0) @binding(4) var<uniform>             info: Info;
var<workgroup> qs: array<f32, 128>;        // this head's query
var<workgroup> sc: array<f32, 2048>;       // scores over the S keys
var<workgroup> red: array<f32, 128>;       // reduction scratch
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let nh = info.a.x; let nkv = info.a.y; let dh = info.a.z; let s = info.a.w;
    let scale = bitcast<f32>(info.b.x);
    let head = wg.x; let t = lid.x;
    let g = nh / nkv; let kvh = head / g;
    let qbase = head * dh; let kvbase = kvh * dh;
    // load this head's query into shared
    if (t < dh) { qs[t] = q[qbase + t]; }
    workgroupBarrier();
    // Stream the S keys in 2048-key chunks with online softmax (running max m_run, sum l_run,
    // per-thread output accumulator accd) — O(dh) state, so any cache length works with no S cap.
    var m_run = -3.0e38; var l_run = 0.0; var accd = 0.0;
    for (var c0 = 0u; c0 < s; c0 = c0 + 2048u) {
        let clen = min(2048u, s - c0);
        for (var i = t; i < clen; i = i + 128u) {
            var dot = 0.0; let kb = (c0 + i) * nkv * dh + kvbase;
            for (var d = 0u; d < dh; d = d + 1u) { dot = dot + qs[d] * k[kb + d]; }
            sc[i] = dot * scale;
        }
        workgroupBarrier();
        var cm = -3.0e38;
        for (var i = t; i < clen; i = i + 128u) { cm = max(cm, sc[i]); }
        red[t] = cm; workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) { if (t < stride) { red[t] = max(red[t], red[t + stride]); } workgroupBarrier(); }
        let m_new = max(m_run, red[0]); let corr = exp(m_run - m_new); workgroupBarrier();
        var cs = 0.0;
        for (var i = t; i < clen; i = i + 128u) { let e = exp(sc[i] - m_new); sc[i] = e; cs = cs + e; }
        red[t] = cs; workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) { if (t < stride) { red[t] = red[t] + red[t + stride]; } workgroupBarrier(); }
        l_run = l_run * corr + red[0];
        if (t < dh) {
            var a = 0.0;
            for (var i = 0u; i < clen; i = i + 1u) { a = a + sc[i] * v[(c0 + i) * nkv * dh + kvbase + t]; }
            accd = accd * corr + a;
        }
        m_run = m_new;
        workgroupBarrier();
    }
    if (t < dh) { out[head * dh + t] = accd / l_run; }
}
"#;

const TILED_MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;   // [M,K]
@group(0) @binding(1) var<storage,read>        b: array<f32>;   // [K,N]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>; // [M,N]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // M,K,N
const TILE = 64u; const KT = 16u; const TW = 16u; // 16×16 threads, 4×4 each
var<workgroup> As: array<f32, 1024>; // 64×16
var<workgroup> Bs: array<f32, 1024>; // 16×64
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = info[0]; let k = info[1]; let n = info[2];
    let row0 = wid.y * TILE; let col0 = wid.x * TILE;
    let tr = (li / TW) * 4u; let tc = (li % TW) * 4u; // this thread's 4×4 micro-tile origin in the tile
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    let ntiles = (k + KT - 1u) / KT;
    for (var t = 0u; t < ntiles; t++) {
        // 256 threads stage A[64×16] and B[16×64] = 1024 elems each → 4 loads per thread.
        for (var e = 0u; e < 4u; e++) {
            let ia = li + e * 256u;
            let ar = ia / KT; let ak = ia % KT;        // As is 64 rows × 16 cols
            let gr = row0 + ar; let gk = t * KT + ak;
            As[ia] = select(0.0, a[gr * k + gk], gr < m && gk < k);
            let br = ia / TILE; let bc = ia % TILE;     // Bs is 16 rows × 64 cols
            let gk2 = t * KT + br; let gc = col0 + bc;
            Bs[ia] = select(0.0, b[gk2 * n + gc], gk2 < k && gc < n);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < KT; kk++) {
            var ra: array<f32, 4>; var rb: array<f32, 4>;
            for (var i = 0u; i < 4u; i++) { ra[i] = As[(tr + i) * KT + kk]; rb[i] = Bs[kk * TILE + tc + i]; }
            for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] = acc[i * 4u + j] + ra[i] * rb[j]; } }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let r = row0 + tr + i; let c = col0 + tc + j;
            if (r < m && c < n) { out[r * n + c] = acc[i * 4u + j]; }
        }
    }
}
"#;

// info[4] = threads per grid row: a 1D dispatch caps at 65535 workgroups = 4.19M threads, which a
// 2048³ matmul (4.19M outputs) reaches exactly, so the grid is 2D and idx is reconstructed here.
// Register-tiled GEMM: one thread computes a **1×8** row segment. It loads each A value once per k
// and reuses it across 8 output columns (8 FMAs), so the A-load instruction count drops 8× and the
// 8 independent accumulators give the scheduler ILP to hide latency. No shared memory, no barriers —
// it leans on the Apple GPU's caches (which already serve the reuse) rather than fighting them, which
// is why hand-tiling with barriers lost. 2D grid: outputs = m·(n/8), which can exceed the 1D cap.
const MATMUL_RT_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;   // [M,K]
@group(0) @binding(1) var<storage,read>        b: array<f32>;   // [K,N]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>; // [M,N]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // M,K,N, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = info[0]; let k = info[1]; let n = info[2];
    let blocks = n / 8u;                       // columns are handled 8 at a time (n multiple of 8)
    let idx = gid.x + gid.y * info[3];
    if (idx >= m * blocks) { return; }
    let i = idx / blocks; let j0 = (idx % blocks) * 8u;
    let ao = i * k;
    var acc = array<f32, 8>();
    for (var l = 0u; l < k; l++) {
        let av = a[ao + l];                    // one load, reused 8×
        let bo = l * n + j0;
        for (var jj = 0u; jj < 8u; jj++) { acc[jj] = acc[jj] + av * b[bo + jj]; }
    }
    for (var jj = 0u; jj < 8u; jj++) { out[i * n + j0 + jj] = acc[jj]; }
}
"#;

const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;   // [batch, m, k]
@group(0) @binding(1) var<storage,read>        b: array<f32>;   // [batch, k, n]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>; // [batch, m, n]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // batch, m, k, n, row_stride
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[4]; let batch = info[0]; let m = info[1]; let k = info[2]; let n = info[3];
    if (idx >= batch * m * n) { return; }
    let j = idx % n; let i = (idx / n) % m; let bt = idx / (m * n);
    let ao = bt * m * k + i * k; let bo = bt * k * n;
    var acc = 0.0;
    for (var l: u32 = 0u; l < k; l = l + 1u) { acc = acc + a[ao + l] * b[bo + l * n + j]; }
    out[idx] = acc;
}
"#;

/// Read-bandwidth probe: stream a buffer that is far larger than any cache and do only enough
/// arithmetic to keep the loads live. Isolates *memory* throughput from the ALU work a real kernel
/// layers on top, which is the only way to tell a bandwidth wall from an ALU wall.
/// `per_thread` = 1 reads scalar u32s, 4 reads vec4<u32>. Returns (seconds, bytes).
pub async fn probe_read_bandwidth(ctx: &Arc<Context>, bytes: usize, per_thread: u32) -> (f64, usize) {
    let words = bytes / 4;
    let src = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bw.src"), size: (words * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let nwg = 16384usize; // enough workgroups to fill the machine; each strides over the buffer
    let out = empty(ctx, nwg);
    let wgsl = if per_thread == 4 { BW_VEC4_WGSL } else { BW_SCALAR_WGSL };
    let unit = if per_thread == 4 { words / 4 } else { words };
    let info = unibuf(ctx, &[unit as u32, nwg as u32, 0, 0]);
    // warm (shader compile + first touch)
    run(ctx, wgsl, "bw", &[&src, &out, &info], (nwg as u32, 1, 1));
    let _ = readback(ctx, &out, 1).await;
    let reps = 3;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { run(ctx, wgsl, "bw", &[&src, &out, &info], (nwg as u32, 1, 1)); }
    let _ = readback(ctx, &out, 1).await;
    ((t0.elapsed().as_secs_f64()) / reps as f64, words * 4)
}

const BW_SCALAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       src:  array<u32>;
@group(0) @binding(1) var<storage,read_write> out:  array<f32>;
@group(0) @binding(2) var<uniform>            info: vec4<u32>; // n_units, n_workgroups
var<workgroup> partial: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = info.x; let nwg = info.y;
    // Each workgroup takes a contiguous slab; within it threads read consecutive words.
    let per = (n + nwg - 1u) / nwg;
    let start = wg.x * per;
    var acc = 0u;
    for (var i: u32 = start + lid.x; i < min(start + per, n); i = i + 64u) { acc = acc + src[i]; }
    partial[lid.x] = acc;
    workgroupBarrier();
    if (lid.x == 0u) {
        var s = 0u;
        for (var j: u32 = 0u; j < 64u; j = j + 1u) { s = s + partial[j]; }
        out[wg.x] = f32(s);   // consumed, so the reads can't be optimized away
    }
}
"#;

const BW_VEC4_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       src:  array<vec4<u32>>;
@group(0) @binding(1) var<storage,read_write> out:  array<f32>;
@group(0) @binding(2) var<uniform>            info: vec4<u32>; // n_units (vec4s), n_workgroups
var<workgroup> partial: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = info.x; let nwg = info.y;
    let per = (n + nwg - 1u) / nwg;
    let start = wg.x * per;
    var acc = 0u;
    for (var i: u32 = start + lid.x; i < min(start + per, n); i = i + 64u) {
        let v = src[i];
        acc = acc + v.x + v.y + v.z + v.w;
    }
    partial[lid.x] = acc;
    workgroupBarrier();
    if (lid.x == 0u) {
        var s = 0u;
        for (var j: u32 = 0u; j < 64u; j = j + 1u) { s = s + partial[j]; }
        out[wg.x] = f32(s);
    }
}
"#;

// ---- lightweight GPU profiler: force a device sync, then attribute the elapsed wall time to a
// named bucket. Off unless FERRIC_PROFILE is set. Perturbs timing (it serializes the pipeline), so
// it measures *where* work is, not the fastest achievable — exactly what's needed to pick a target.
thread_local! {
    static PROF: std::cell::RefCell<(std::collections::BTreeMap<String, f64>, Option<std::time::Instant>)>
        = std::cell::RefCell::new((std::collections::BTreeMap::new(), None));
}

/// Block until the GPU has finished all submitted work.
pub fn device_sync(ctx: &Context) { let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely()); }

/// Sync, then charge the time since the previous mark to `label`. First call in a region just
/// starts the clock. No-op unless FERRIC_PROFILE is set.
pub fn prof(ctx: &Context, label: &str) {
    if std::env::var("FERRIC_PROFILE").is_err() { return; }
    device_sync(ctx);
    PROF.with(|c| {
        let mut b = c.borrow_mut();
        let now = std::time::Instant::now();
        if let Some(prev) = b.1 {
            let dt = now.duration_since(prev).as_secs_f64() * 1e3;
            *b.0.entry(label.to_string()).or_insert(0.0) += dt;
        }
        b.1 = Some(now);
    });
}

/// Print the accumulated buckets (descending) and reset. No-op unless FERRIC_PROFILE is set.
pub fn prof_report() {
    if std::env::var("FERRIC_PROFILE").is_err() { return; }
    PROF.with(|c| {
        let mut b = c.borrow_mut();
        let mut v: Vec<_> = b.0.iter().map(|(k, &t)| (k.clone(), t)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = v.iter().map(|(_, t)| t).sum();
        eprintln!("  ── profile (ms, {total:.1} total) ──");
        for (k, t) in v { eprintln!("    {k:<16} {t:8.1}  {:4.1}%", 100.0 * t / total); }
        b.0.clear(); b.1 = None;
    });
}

/// Spike/validation for the `subgroups` feature: 64 threads contribute their index; each subgroup
/// sums via `subgroupAdd` (a hardware warp reduction), lane 0 of each subgroup atomic-adds its
/// partial, thread 0 writes the total. Correct total = sum(0..64) = 2016. Confirms the primitive
/// that will replace the split-K barrier-tree reduction. Returns out[0].
pub async fn run_subgroup_sum_test(ctx: &Arc<Context>) -> f32 {
    let out = empty(ctx, 1);
    run(ctx, SUBGROUP_SUM_WGSL, "sg_sum", &[&out], (1, 1, 1));
    readback(ctx, &out, 1).await[0]
}

const SUBGROUP_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read_write> out: array<f32>;
var<workgroup> total: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_invocation_id) sglid: u32) {
    if (lid.x == 0u) { atomicStore(&total, 0u); }
    workgroupBarrier();
    let s = subgroupAdd(lid.x);            // sum of lid.x within this subgroup
    if (sglid == 0u) { atomicAdd(&total, s); }
    workgroupBarrier();
    if (lid.x == 0u) { out[0] = f32(atomicLoad(&total)); }
}
"#;

/// Spike: an 8×8 f32 matmul C = A·B on the **hardware matrix unit** via WGSL `coop_mat` (lowers to
/// Metal simdgroup_matrix / Vulkan cooperative-matrix → tensor cores). Proves the tensor-core path
/// compiles + runs + is correct through our fork before building a full tiled GEMM on it. `a`/`b` are
/// row-major 8×8; returns C (64 f32). One subgroup cooperates on the tile.
pub async fn run_coop_matmul_test(ctx: &Arc<Context>, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), 64); assert_eq!(b.len(), 64);
    let ab = Tensor::from_vec(ctx, a, &[8, 8]);
    let bb = Tensor::from_vec(ctx, b, &[8, 8]);
    let cb = empty(ctx, 64); // zero-initialized accumulator
    run(ctx, COOP_MATMUL_WGSL, "coop_mm", &[ab.buf.as_ref(), bb.buf.as_ref(), &cb], (1, 1, 1));
    readback(ctx, &cb, 64).await
}

const COOP_MATMUL_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       a: array<f32>;
@group(0) @binding(1) var<storage,read>       b: array<f32>;
@group(0) @binding(2) var<storage,read_write> c: array<f32>;
@compute @workgroup_size(32)
fn main() {
    let ma = coopLoadT<coop_mat8x8<f32, A>>(&a[0], 8u);
    let mb = coopLoadT<coop_mat8x8<f32, B>>(&b[0], 8u);
    var mc = coopLoadT<coop_mat8x8<f32, C>>(&c[0], 8u);
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(mc, &c[0], 8u);
}
"#;

impl Tensor {
    /// f32 GEMM C=A·B on the **hardware matrix unit** via cooperative-matrix 8×8 tiles: one subgroup
    /// accumulates an 8×8 output tile across the K dimension with `coopMultiplyAdd`. Requires M,K,N
    /// multiples of 8 and `ctx.coop_matrix`; the caller falls back to the naive kernel otherwise.
    /// fp-order differs from the scalar kernels (hardware reduction), so this is a fast-path, not the
    /// bit-identical default. **8×8 is the portable tile size**: Metal's `simdgroup_matrix<f32>` is
    /// 8×8 only (no `simdgroup_float16x16`), so `coop_mat16x16` crashes MSL — don't use 16×16 for f32.
    pub fn matmul_coop(&self, other: &Tensor) -> Tensor {
        let (m, k) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let n = other.shape[other.rank() - 1];
        assert!(m % 8 == 0 && k % 8 == 0 && n % 8 == 0, "matmul_coop needs M,K,N multiples of 8");
        let (a, b) = (self.contiguous(), other.contiguous());
        let out = empty(&self.ctx, m * n); // zeroed accumulator
        // Register-blocked (2×2 of 8×8 tiles per workgroup) when M,N are 16-aligned: each subgroup
        // reuses 2 A-tiles + 2 B-tiles across 4 MMAs (half the loads per MMA), so it's compute-denser.
        if m % 16 == 0 && n % 16 == 0 {
            run(&self.ctx, COOP_GEMM_RB_WGSL, "coop_gemm_rb",
                &[a.buf.as_ref(), b.buf.as_ref(), &out, &unibuf(&self.ctx, &[m as u32, k as u32, n as u32, 0])],
                ((n / 16) as u32, (m / 16) as u32, 1));
        } else {
            run(&self.ctx, COOP_GEMM_WGSL, "coop_gemm",
                &[a.buf.as_ref(), b.buf.as_ref(), &out, &unibuf(&self.ctx, &[m as u32, k as u32, n as u32, 0])],
                ((n / 8) as u32, (m / 8) as u32, 1));
        }
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }

    /// **NVIDIA / Intel tensor-core GEMM** (`coop16_ok()` fabrics). Those vendors enumerate only
    /// f16-input cooperative-matrix configs (A=B=f16, C=f32) at 16×16 — never the 8×8 f32 the Metal
    /// path uses — so this converts A,B to f16 and accumulates in f32 on the matrix unit. Mixed
    /// precision → NOT bit-identical (f16 rounding on the inputs), an opt-in fast path. C=A·B.
    pub fn matmul_coop16(&self, other: &Tensor) -> Tensor {
        let (m, k) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let n = other.shape[other.rank() - 1];
        assert!(m % 16 == 0 && k % 16 == 0 && n % 16 == 0, "matmul_coop16 needs M,K,N multiples of 16");
        let ah = self.contiguous().to_half(dtype::DType::F16);
        let bh = other.contiguous().to_half(dtype::DType::F16);
        let out = empty(&self.ctx, m * n); // zeroed f32 accumulator
        // Register-blocked (2×2 of 16×16 tiles / workgroup) when M,N are 32-aligned — each subgroup
        // reuses 2 A-tiles + 2 B-tiles across 4 MMAs, ~halving loads per MMA. Else the single-tile path.
        if m % 32 == 0 && n % 32 == 0 {
            run(&self.ctx, COOP_GEMM16_RB_WGSL, "coop_gemm16_rb",
                &[ah.buffer(), bh.buffer(), &out, &unibuf(&self.ctx, &[m as u32, k as u32, n as u32, 0])],
                ((n / 32) as u32, (m / 32) as u32, 1));
        } else {
            run(&self.ctx, COOP_GEMM16_WGSL, "coop_gemm16",
                &[ah.buffer(), bh.buffer(), &out, &unibuf(&self.ctx, &[m as u32, k as u32, n as u32, 0])],
                ((n / 16) as u32, (m / 16) as u32, 1));
        }
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }
}

// Register-blocked 16×16 f16 coop GEMM: one workgroup (subgroup) owns a 32×32 output block = 2×2 grid
// of 16×16 tiles in 4 f32 accumulators. Per K-step loads 2 A-tiles (top/bottom 16 rows) + 2 B-tiles
// (left/right 16 cols) and issues 4 coopMultiplyAdd, reusing each loaded tile twice. Same shape lesson
// as the Metal f32 RB (2×2 sweet spot); f16 A/B, f32 accumulate — the NVIDIA tensor-core config.
const COOP_GEMM16_RB_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage,read>       a: array<f16>;
@group(0) @binding(1) var<storage,read>       b: array<f16>;
@group(0) @binding(2) var<storage,read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>            dims: vec4<u32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z;
    let r0 = wid.y * 32u; let r1 = r0 + 16u;
    let c0 = wid.x * 32u; let c1 = c0 + 16u;
    let i00 = r0 * nn + c0; let i01 = r0 * nn + c1; let i10 = r1 * nn + c0; let i11 = r1 * nn + c1;
    var a00 = coopLoadT<coop_mat16x16<f32, C>>(&c[i00], nn);
    var a01 = coopLoadT<coop_mat16x16<f32, C>>(&c[i01], nn);
    var a10 = coopLoadT<coop_mat16x16<f32, C>>(&c[i10], nn);
    var a11 = coopLoadT<coop_mat16x16<f32, C>>(&c[i11], nn);
    for (var k: u32 = 0u; k < kk; k = k + 16u) {
        let ja0 = r0 * kk + k; let ja1 = r1 * kk + k; let jb0 = k * nn + c0; let jb1 = k * nn + c1;
        let ma0 = coopLoadT<coop_mat16x16<f16, A>>(&a[ja0], kk);
        let ma1 = coopLoadT<coop_mat16x16<f16, A>>(&a[ja1], kk);
        let mb0 = coopLoadT<coop_mat16x16<f16, B>>(&b[jb0], nn);
        let mb1 = coopLoadT<coop_mat16x16<f16, B>>(&b[jb1], nn);
        a00 = coopMultiplyAdd(ma0, mb0, a00);
        a01 = coopMultiplyAdd(ma0, mb1, a01);
        a10 = coopMultiplyAdd(ma1, mb0, a10);
        a11 = coopMultiplyAdd(ma1, mb1, a11);
    }
    coopStoreT(a00, &c[i00], nn);
    coopStoreT(a01, &c[i01], nn);
    coopStoreT(a10, &c[i10], nn);
    coopStoreT(a11, &c[i11], nn);
}
"#;

// 16×16 f16-input coop GEMM for NVIDIA tensor cores / Intel XMX: one workgroup (one subgroup, 32
// lanes) owns a 16×16 output tile, streams K in steps of 16 loading f16 A/B tiles, accumulates in an
// f32 coop matrix. This is the ONLY coop shape/type those vendors support (verified by querying
// VkCooperativeMatrixPropertiesKHR: A=B=f16, C=f32, M=N=K=16); the 8×8-f32 kernel matches nothing
// there and executes as zeros. Metal keeps the exact-f32 8×8 path.
const COOP_GEMM16_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage,read>       a: array<f16>;   // [M,K] row-major f16
@group(0) @binding(1) var<storage,read>       b: array<f16>;   // [K,N] row-major f16
@group(0) @binding(2) var<storage,read_write> c: array<f32>;   // [M,N] f32 (zeroed)
@group(0) @binding(3) var<uniform>            dims: vec4<u32>; // M, K, N, _
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z;
    let trow = wid.y * 16u; let tcol = wid.x * 16u;
    let ci = trow * nn + tcol;                    // let-bind coop pointer indices (naga SPIR-V caching)
    var acc = coopLoadT<coop_mat16x16<f32, C>>(&c[ci], nn);
    for (var k: u32 = 0u; k < kk; k = k + 16u) {
        let ai = trow * kk + k; let bi = k * nn + tcol;
        let ma = coopLoadT<coop_mat16x16<f16, A>>(&a[ai], kk);
        let mb = coopLoadT<coop_mat16x16<f16, B>>(&b[bi], nn);
        acc = coopMultiplyAdd(ma, mb, acc);
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

// Register-blocked coop GEMM: one workgroup owns a 16×16 output block = a 2×2 grid of 8×8 tiles, held
// in 4 accumulators. **2×2 is the measured sweet spot on the M5**: a 4×4 grid (16 accumulators + 4+4
// operand tiles = 24 coop matrices/subgroup) spills registers and runs ~50× slower (9500 → 155 GFLOP/s). Per K-step it loads 2 A-tiles (top/bottom 8 rows) and 2 B-tiles (left/right 8
// cols) and issues 4 coopMultiplyAdd — reusing each loaded tile twice, halving loads per MMA.
const COOP_GEMM_RB_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       a: array<f32>;
@group(0) @binding(1) var<storage,read>       b: array<f32>;
@group(0) @binding(2) var<storage,read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>            dims: vec4<u32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let kk = dims.y; let nn = dims.z;
    let r0 = wid.y * 16u; let r1 = r0 + 8u;
    let c0 = wid.x * 16u; let c1 = c0 + 8u;
    // Every coop pointer index is let-bound: the forked naga SPIR-V backend panics
    // ("Expression is not cached", write_bounds_check under the Unchecked policy) when a coopLoad/
    // coopStore pointer arg is indexed by an inline compound expression. This is what kept the RB
    // kernel Metal-only — MSL is unaffected; Vulkan/SPIR-V is not. Same fix as COOP_GEMM_WGSL.
    let i00 = r0 * nn + c0; let i01 = r0 * nn + c1; let i10 = r1 * nn + c0; let i11 = r1 * nn + c1;
    var a00 = coopLoadT<coop_mat8x8<f32, C>>(&c[i00], nn);
    var a01 = coopLoadT<coop_mat8x8<f32, C>>(&c[i01], nn);
    var a10 = coopLoadT<coop_mat8x8<f32, C>>(&c[i10], nn);
    var a11 = coopLoadT<coop_mat8x8<f32, C>>(&c[i11], nn);
    for (var k: u32 = 0u; k < kk; k = k + 8u) {
        let ja0 = r0 * kk + k; let ja1 = r1 * kk + k; let jb0 = k * nn + c0; let jb1 = k * nn + c1;
        let ma0 = coopLoadT<coop_mat8x8<f32, A>>(&a[ja0], kk);
        let ma1 = coopLoadT<coop_mat8x8<f32, A>>(&a[ja1], kk);
        let mb0 = coopLoadT<coop_mat8x8<f32, B>>(&b[jb0], nn);
        let mb1 = coopLoadT<coop_mat8x8<f32, B>>(&b[jb1], nn);
        a00 = coopMultiplyAdd(ma0, mb0, a00);
        a01 = coopMultiplyAdd(ma0, mb1, a01);
        a10 = coopMultiplyAdd(ma1, mb0, a10);
        a11 = coopMultiplyAdd(ma1, mb1, a11);
    }
    coopStoreT(a00, &c[i00], nn);
    coopStoreT(a01, &c[i01], nn);
    coopStoreT(a10, &c[i10], nn);
    coopStoreT(a11, &c[i11], nn);
}
"#;

const COOP_GEMM_WGSL: &str = r#"
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage,read>       a: array<f32>;   // [M,K] row-major
@group(0) @binding(1) var<storage,read>       b: array<f32>;   // [K,N] row-major
@group(0) @binding(2) var<storage,read_write> c: array<f32>;   // [M,N] (zeroed)
@group(0) @binding(3) var<uniform>            dims: vec4<u32>; // M, K, N, _
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let mm = dims.x; let kk = dims.y; let nn = dims.z;
    let trow = wid.y * 8u; let tcol = wid.x * 8u;
    let ci = trow * nn + tcol;                    // let-bind indices so the SPIR-V backend caches them
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[ci], nn);
    for (var k: u32 = 0u; k < kk; k = k + 8u) {
        let ai = trow * kk + k;
        let bi = k * nn + tcol;
        let ma = coopLoadT<coop_mat8x8<f32, A>>(&a[ai], kk);
        let mb = coopLoadT<coop_mat8x8<f32, B>>(&b[bi], nn);
        acc = coopMultiplyAdd(ma, mb, acc);
    }
    coopStoreT(acc, &c[ci], nn);
}
"#;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod reduce_tests {
    use super::*;

    /// Staged large-axis reduction vs exact expectations — the sizes that used to run on one thread.
    #[test]
    fn large_axis_reductions_are_exact_and_parallel() {
        let Ok(ctx) = pollster::block_on(Context::new()) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let ctx = std::sync::Arc::new(ctx);
        // scalar sum over 8M elements (multi-stage: 8M → 2048 → final)
        let n = 8 * 1024 * 1024;
        let t = Tensor::from_vec(&ctx, &vec![0.25f32; n], &[n]);
        let s = pollster::block_on(t.sum(&[0], false).to_vec())[0];
        assert_eq!(s, 0.25 * n as f32, "8M scalar sum");
        // max with a planted needle (staged max path)
        let mut v = vec![-1.0f32; n];
        v[5_000_017] = 42.5;
        let t = Tensor::from_vec(&ctx, &v, &[n]);
        let m = pollster::block_on(t.max(&[0], false).to_vec())[0];
        assert_eq!(m, 42.5, "8M scalar max");
        // 2D: reduce a large axis with outer kept — [8, 1<<20] sum over axis 1
        let (o, r) = (8usize, 1 << 20);
        let v: Vec<f32> = (0..o * r).map(|i| ((i / r) + 1) as f32 * 1e-3).collect();
        let t = Tensor::from_vec(&ctx, &v, &[o, r]);
        let s = pollster::block_on(t.sum(&[1], false).to_vec());
        for (i, &x) in s.iter().enumerate() {
            let want = (i + 1) as f32 * 1e-3 * r as f32;
            assert!((x - want).abs() < want * 1e-4, "row {i}: {x} vs {want}");
        }
        // small axes must still hit the single-pass path and stay exact
        let t = Tensor::from_vec(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(pollster::block_on(t.sum(&[1], false).to_vec()), vec![6.0, 15.0]);
        assert_eq!(pollster::block_on(t.max(&[0], false).to_vec()), vec![4.0, 5.0, 6.0]);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod conv_tests {
    use super::*;

    fn cpu_conv2d(
        x: &[f32], w: &[f32], n: usize, h: usize, wd: usize, c: usize,
        kh: usize, kw: usize, o: usize, stride: (usize, usize), pad: (usize, usize),
    ) -> Vec<f32> {
        let ho = (h + 2 * pad.0 - kh) / stride.0 + 1;
        let wo = (wd + 2 * pad.1 - kw) / stride.1 + 1;
        let mut out = vec![0.0f32; n * ho * wo * o];
        for b in 0..n {
            for yo in 0..ho {
                for xo in 0..wo {
                    for oc in 0..o {
                        let mut acc = 0.0f32;
                        for ky in 0..kh {
                            let yi = (yo * stride.0 + ky) as isize - pad.0 as isize;
                            if yi < 0 || yi >= h as isize { continue; }
                            for kx in 0..kw {
                                let xi = (xo * stride.1 + kx) as isize - pad.1 as isize;
                                if xi < 0 || xi >= wd as isize { continue; }
                                for ci in 0..c {
                                    acc += x[((b * h + yi as usize) * wd + xi as usize) * c + ci]
                                        * w[((ky * kw + kx) * c + ci) * o + oc];
                                }
                            }
                        }
                        out[((b * ho + yo) * wo + xo) * o + oc] = acc;
                    }
                }
            }
        }
        out
    }

    /// The portable conv2d against a CPU oracle: 1x1 identity, 3x3 same-pad, strided, batched,
    /// asymmetric kernel + rectangular input.
    #[test]
    fn conv2d_matches_the_cpu_oracle() {
        let Ok(ctx) = pollster::block_on(Context::new()) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let ctx = std::sync::Arc::new(ctx);
        let check = |n: usize, h: usize, wd: usize, c: usize, kh: usize, kw: usize, o: usize,
                     stride: (usize, usize), pad: (usize, usize)| {
            let x: Vec<f32> = (0..n * h * wd * c).map(|i| 0.1 * (((i + 3) % 11) as f32 - 5.0)).collect();
            let w: Vec<f32> = (0..kh * kw * c * o).map(|i| 0.1 * (((i + 5) % 7) as f32 - 3.0)).collect();
            let xt = Tensor::from_vec(&ctx, &x, &[n, h, wd, c]);
            let wt = Tensor::from_vec(&ctx, &w, &[kh, kw, c, o]);
            let got = pollster::block_on(xt.conv2d(&wt, stride, pad).to_vec());
            let want = cpu_conv2d(&x, &w, n, h, wd, c, kh, kw, o, stride, pad);
            let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-4, "n={n} h={h} w={wd} c={c} k={kh}x{kw} o={o} s={stride:?} p={pad:?}: err {err}");
        };
        // 1x1 identity kernel passes channels through
        {
            let (h, wd, c) = (5usize, 4usize, 3usize);
            let x: Vec<f32> = (0..h * wd * c).map(|i| i as f32 * 0.1).collect();
            let mut eye = vec![0.0f32; c * c];
            for i in 0..c { eye[i * c + i] = 1.0; }
            let xt = Tensor::from_vec(&ctx, &x, &[1, h, wd, c]);
            let wt = Tensor::from_vec(&ctx, &eye, &[1, 1, c, c]);
            let got = pollster::block_on(xt.conv2d(&wt, (1, 1), (0, 0)).to_vec());
            assert_eq!(got, x, "1x1 identity conv must pass through");
        }
        check(1, 8, 8, 4, 3, 3, 8, (1, 1), (1, 1)); // 3x3 same-pad
        check(2, 9, 7, 3, 3, 3, 5, (2, 2), (1, 1)); // strided + batch + rectangular
        check(1, 6, 6, 2, 5, 3, 4, (1, 1), (2, 1)); // asymmetric kernel + pads
        check(2, 16, 16, 8, 3, 3, 16, (1, 1), (1, 1)); // a real CNN-ish layer
    }
}
