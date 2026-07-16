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
#[cfg(not(target_arch = "wasm32"))]
pub mod sched; // L7 heterogeneous scheduler (GPU + CPU as one fabric)
#[cfg(not(target_arch = "wasm32"))]
pub mod ws; // WebSocket bridge so a browser tab is a scheduler device
pub use autograd::Var;
pub use dtype::{DType, Half, Q2_0Weights, QRow, QTensor, Ternary};
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
        // info: [rank, n, offset, shape..., strides...]
        let mut info = vec![self.rank() as u32, n as u32, self.offset as u32];
        info.extend(self.shape.iter().map(|&x| x as u32));
        info.extend(self.strides.iter().map(|&x| x as u32));
        run(&self.ctx, GATHER_WGSL, "gather", &[&self.buf, &out, &u32buf(&self.ctx, &info)], groups(n));
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
        run(&self.ctx, BINARY_WGSL, "binary", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &info)], groups(n));
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
        run(&self.ctx, UNARY_WGSL, "unary", &[&c.buf, &out, &u32buf(&self.ctx, &[op, n as u32])], groups(n));
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
    pub fn log(&self) -> Tensor { self.unary(9) }
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

    /// y = x·Wᵀ where x is [rows,in] and W is stored [out,in] (HF linear convention) — computed
    /// directly, without materializing Wᵀ. Essential for big tied LM heads (avoids a huge transpose).
    pub fn matmul_bt(&self, w: &Tensor) -> Tensor {
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "matmul_bt is 2D");
        let (rows, inn) = (x.shape[0], x.shape[1]);
        let wc = w.contiguous();
        let out_f = wc.shape[0];
        assert_eq!(inn, wc.shape[1], "inner dims mismatch");
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
        let out = empty(&self.ctx, rows * out_f);
        run(&self.ctx, MATMUL_BT_ACT_WGSL, "matmul_bt_act", &[x.buf.as_ref(), wc.buf.as_ref(), &out, &u32buf(&self.ctx, &[rows as u32, out_f as u32, inn as u32, act])], groups(rows * out_f));
        Tensor::from_parts(&self.ctx, out, vec![rows, out_f])
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
        let out = empty(&self.ctx, outer);
        run(&self.ctx, REDUCE_WGSL, "reduce", &[&moved.buf, &out, &u32buf(&self.ctx, &[outer as u32, red as u32, op])], groups(outer));
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
        let a = self.broadcast_to(&a_full).contiguous();
        let b = other.broadcast_to(&b_full).contiguous();
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
            run(&self.ctx, MATMUL_WGSL, "bmm", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[bn as u32, m as u32, ka as u32, n as u32])], groups(bn * m * n));
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

    /// The naive (non-tiled) matmul, kept for benchmarking the tiled fast-path against.
    pub fn matmul_naive(&self, other: &Tensor) -> Tensor {
        let (ra, rb) = (self.rank(), other.rank());
        let (m, ka) = (self.shape[ra - 2], self.shape[ra - 1]);
        let n = other.shape[rb - 1];
        let a = self.contiguous();
        let b = other.contiguous();
        let out = empty(&self.ctx, m * n);
        run(&self.ctx, MATMUL_WGSL, "bmm", &[&a.buf, &b.buf, &out, &u32buf(&self.ctx, &[1, m as u32, ka as u32, n as u32])], groups(m * n));
        Tensor::from_parts(&self.ctx, out, vec![m, n])
    }
}

// ---------- device plumbing (uses ferric-core Context's public device/queue) ----------
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
    let i = gid.x; let rank = info[0]; let op = info[1]; let n = info[2];
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
@group(0) @binding(2) var<storage,read>        info: array<u32>; // op, n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= info[1]) { return; }
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
        default: { r = v; }
    }
    out[i] = r;
}
"#;

const GATHER_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        x: array<f32>;
@group(0) @binding(1) var<storage,read_write>  out: array<f32>;
@group(0) @binding(2) var<storage,read>        info: array<u32>; // rank,n,offset,shape[r],strides[r]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let rank = info[0]; let n = info[1];
    if (i >= n) { return; }
    var src = info[2]; var rem = i;
    for (var dd: u32 = 0u; dd < rank; dd = dd + 1u) {
        let d = rank - 1u - dd;
        let sz = info[3u + d];
        let idx = rem % sz; rem = rem / sz;
        src = src + idx * info[3u + rank + d];
    }
    out[i] = x[src];
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
const TILED_MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;   // [M,K]
@group(0) @binding(1) var<storage,read>        b: array<f32>;   // [K,N]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>; // [M,N]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // M,K,N
var<workgroup> As: array<f32, 512>; // 64×8
var<workgroup> Bs: array<f32, 512>; // 8×64
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = info[0]; let k = info[1]; let n = info[2];
    let row0 = wid.y * 64u; let col0 = wid.x * 64u;
    let li = lid.y * 8u + lid.x;             // 0..63
    let tr = lid.y * 8u; let tc = lid.x * 8u; // this thread's 8×8 micro-tile origin within the 64×64 tile
    var acc: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) { acc[i] = 0.0; }
    let ntiles = (k + 7u) / 8u;
    for (var t = 0u; t < ntiles; t++) {
        // stage A[64×8] and B[8×64] into shared memory (64 threads × 8 elems each)
        for (var e = 0u; e < 8u; e++) {
            let ia = li + e * 64u; let ar = ia / 8u; let ak = ia % 8u;
            let gr = row0 + ar; let gk = t * 8u + ak;
            As[ia] = select(0.0, a[gr * k + gk], gr < m && gk < k);
            let bk = ia / 64u; let bc = ia % 64u;
            let gk2 = t * 8u + bk; let gc = col0 + bc;
            Bs[ia] = select(0.0, b[gk2 * n + gc], gk2 < k && gc < n);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 8u; kk++) {
            var ra: array<f32, 8>; var rb: array<f32, 8>;
            for (var i = 0u; i < 8u; i++) { ra[i] = As[(tr + i) * 8u + kk]; rb[i] = Bs[kk * 64u + tc + i]; }
            for (var i = 0u; i < 8u; i++) { for (var j = 0u; j < 8u; j++) { acc[i * 8u + j] = acc[i * 8u + j] + ra[i] * rb[j]; } }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 8u; i++) {
        for (var j = 0u; j < 8u; j++) {
            let r = row0 + tr + i; let c = col0 + tc + j;
            if (r < m && c < n) { out[r * n + c] = acc[i * 8u + j]; }
        }
    }
}
"#;

const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        a: array<f32>;   // [batch, m, k]
@group(0) @binding(1) var<storage,read>        b: array<f32>;   // [batch, k, n]
@group(0) @binding(2) var<storage,read_write>  out: array<f32>; // [batch, m, n]
@group(0) @binding(3) var<storage,read>        info: array<u32>; // batch, m, k, n
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let batch = info[0]; let m = info[1]; let k = info[2]; let n = info[3];
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
