//! **KV-cache quantization** — block-quantized K/V storage that grows one row at a time.
//!
//! Today a Ferric KV cache ([`crate::KvBuf`]) is f32: `cap * width * 4` bytes per layer per tensor.
//! That is 32 bits per cached value, and on the small devices Ferric targets the cache — not the
//! weights — is what caps context length. This module stores the same cache as llama.cpp-layout
//! quantization blocks: **8.5 bits/value (Q8_0), 5.0 (Q4_1), 4.5 (Q4_0)** — 3.8x/6.4x/7.1x less
//! cache memory for the same number of tokens.
//!
//! # The append constraint is what picks the scheme
//!
//! A KV cache is appended to **once per token**. So the scaling granularity is not a free choice:
//!
//! | granularity | scale shared over | append cost |
//! |---|---|---|
//! | per-tensor | everything | requantize the whole cache when a new token moves the max |
//! | per-token | one row | in place |
//! | **per-block(32) along the row** | 32 channels of one row | **in place** ← what ships here |
//! | per-channel | one channel, all tokens | requantize the whole cache when a new token moves a channel max |
//! | per-channel, grouped by G tokens | one channel, G tokens | in place after buffering G rows in f32 |
//!
//! [`append_cost`] states this mechanically. Per-channel scaling is the granularity the KV-quant
//! literature recommends for **K** (K has outlier channels; a per-token scale is dragged up by them
//! and the other ~120 channels lose resolution). It is also the granularity that cannot be appended
//! to: token *t+1* can exceed a channel's running max, which invalidates every code already stored
//! for that channel. **Per-block-of-32 along the row is the compromise this module ships**, and the
//! reason it works is measurable rather than assumed: an outlier channel pollutes only the 32-channel
//! block it sits in, not the whole 128-wide row. `examples/kv_quant_error.rs` measures all five
//! granularities on **real captured K and V**, which is the only way to settle it — see that example's
//! output, not this comment.
//!
//! # Layout
//!
//! For a `[t, width]` cache (`width % 32 == 0`), `nblk = width / 32` blocks per row, indexed **flat**
//! as `blk = row * nblk + col/32`. That flat order is exactly `Q8_0Weights`/`Q4_0Weights`/
//! `Q4_1Weights`' `[rows, cols]` order, and the codes/scales split is the same one those types use
//! (`codes: array<u32>`, `scales:` f16 pairs). So a quantized K cache **is** a packed weight matrix in
//! memory, and a future attention kernel can read it with the existing `matmul_q*` machinery rather
//! than dequantizing first. Nothing here depends on that; it is why the layout was chosen.
//!
//! # What is a reference for what
//!
//! [`reference`] is a CPU implementation of the exact same arithmetic, in the role `crate::cpu` plays
//! for the general kernels: the GPU kernels are validated against it, not against reasoning about
//! them. The number formats themselves (`d = amax/127`, `roundf`, the q4_0 `-8` bias, the q4_1
//! `(max-min)/15` affine) are llama.cpp's, so that the blocks are the blocks the rest of the crate
//! already knows how to read.

use crate::{empty, groups2d, run, unibuf, Tensor};
use ferric_core::Context;
use std::sync::Arc;

/// Values per quantization block. Matches llama.cpp's `QK8_0`/`QK4_0`/`QK4_1`.
pub const QK: usize = 32;

/// Block format for a quantized KV cache. All three are llama.cpp block layouts.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvqFmt {
    /// 32 int8 + f16 scale; `v = q·d`, `d = amax/127`. 34 B/block = **8.5 bits/value**.
    Q8_0,
    /// 32 nibbles + f16 scale; `v = (q−8)·d`, `d = max/−8`. 18 B/block = **4.5 bits/value**.
    Q4_0,
    /// 32 nibbles + f16 (scale, min); `v = q·d + m`, `d = (max−min)/15`. 20 B/block = **5.0 bits/value**.
    Q4_1,
}

impl KvqFmt {
    /// u32 words of packed codes per 32-value block.
    pub const fn code_words(self) -> usize {
        match self {
            KvqFmt::Q8_0 => 8,
            KvqFmt::Q4_0 | KvqFmt::Q4_1 => 4,
        }
    }
    /// u32 words of scale data needed for `nblk` blocks. Q8_0/Q4_0 pack two f16 `d` per word; Q4_1
    /// packs `(d, m)` into one word per block.
    pub const fn scale_words(self, nblk: usize) -> usize {
        match self {
            KvqFmt::Q8_0 | KvqFmt::Q4_0 => nblk.div_ceil(2),
            KvqFmt::Q4_1 => nblk,
        }
    }
    /// Bytes per block in the equivalent llama.cpp on-disk block (34 / 18 / 20).
    pub const fn block_bytes(self) -> usize {
        match self {
            KvqFmt::Q8_0 => 34,
            KvqFmt::Q4_0 => 18,
            KvqFmt::Q4_1 => 20,
        }
    }
    /// Storage cost per cached value, in bits. f32 (what `KvBuf` costs today) is 32.
    pub fn bits_per_value(self) -> f32 {
        (self.block_bytes() * 8) as f32 / QK as f32
    }
    pub const fn name(self) -> &'static str {
        match self {
            KvqFmt::Q8_0 => "q8_0",
            KvqFmt::Q4_0 => "q4_0",
            KvqFmt::Q4_1 => "q4_1",
        }
    }
    /// All formats, so a caller sweeping them cannot silently miss one that was added later.
    pub const ALL: [KvqFmt; 3] = [KvqFmt::Q8_0, KvqFmt::Q4_0, KvqFmt::Q4_1];
}

// ---------------------------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------------------------

/// A **grow-in-place quantized K or V cache**.
///
/// Same contract as [`crate::KvBuf`] — preallocate, append rows, double when full — except the stored
/// rows are quantization blocks rather than f32. Appending `t` rows quantizes exactly those rows'
/// blocks and writes them at their flat block offset; nothing already in the cache is touched or
/// re-read. That is the property that makes the scheme usable during decode.
///
/// Not wired into any runtime's `Cache` yet: see the module docs for the (small) shape that change
/// takes.
pub struct QKvCache {
    fmt: KvqFmt,
    codes: Option<Arc<wgpu::Buffer>>,
    scales: Option<Arc<wgpu::Buffer>>,
    len: usize,   // filled rows
    cap: usize,   // capacity in rows
    width: usize, // row width in elements (multiple of QK)
    /// Learned on the first [`Self::append`], exactly as `width` is, and for the same reason: the
    /// constructors that build these live in runtime `Cache::with_kvq`, which has a `Cfg` and no
    /// device. Holding it is what lets [`Clone`] be a real copy — `Clone::clone` takes no arguments,
    /// so without a `Context` in the struct a correct clone is not expressible and the derive would
    /// hand out an aliasing handle. `Tensor` holds its context for the same reason.
    ctx: Option<Arc<Context>>,
}

/// Fused transposing dequantize for the GROUPED layout — the permute folded into index math.
///
/// [`GroupedKvCache::dequantize`] paid two passes: the block dequantize (rows in `[ngroups*width,
/// GROUP]` order) and then a permute+contiguous over the whole cache to turn stored-transposed back
/// into `[rows, width]`. Attention reads the cache EVERY step, so that second pass was per-token
/// overhead proportional to the entire history. Here one invocation per output element derives its
/// source block directly: row `r`, channel `c` lives in block `(r/GROUP)*width + c`, lane `r%GROUP` —
/// same arithmetic per element as the two-pass path, so the results are BIT-identical, which is
/// exactly what the test demands (the old path stays as the oracle).
///
///   info[0] = (n, grid_w, width, GROUP)   info[1] = (rows, 0, 0, 0)
const DEQUANT_GROUPED_Q8_0_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let width = info[0].z;
    let g = (i / width) / info[0].w;
    let j = (i / width) % info[0].w;
    let b = g * width + (i % width);
    let sw = unpack2x16float(scales[b >> 1u]);
    let d = select(sw.y, sw.x, (b & 1u) == 0u);
    let word = codes[b * 8u + (j >> 2u)];
    let q = i32(word << (24u - 8u * (j & 3u))) >> 24u;
    out[i] = f32(q) * d;
}
"#;

const DEQUANT_GROUPED_Q4_0_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let width = info[0].z;
    let g = (i / width) / info[0].w;
    let j = (i / width) % info[0].w;
    let b = g * width + (i % width);
    let sw = unpack2x16float(scales[b >> 1u]);
    let d = select(sw.y, sw.x, (b & 1u) == 0u);
    let byte_i = j & 15u;
    let word = codes[b * 4u + (byte_i >> 2u)];
    let byte = (word >> (8u * (byte_i & 3u))) & 255u;
    let nib = select(byte >> 4u, byte & 15u, j < 16u);
    out[i] = (f32(nib) - 8.0) * d;
}
"#;

const DEQUANT_GROUPED_Q4_1_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let width = info[0].z;
    let g = (i / width) / info[0].w;
    let j = (i / width) % info[0].w;
    let b = g * width + (i % width);
    let dm = unpack2x16float(scales[b]);
    let byte_i = j & 15u;
    let word = codes[b * 4u + (byte_i >> 2u)];
    let byte = (word >> (8u * (byte_i & 3u))) & 255u;
    let nib = select(byte >> 4u, byte & 15u, j < 16u);
    out[i] = f32(nib) * dm.x + dm.y;
}
"#;

/// Tokens per group in [`GroupedKvCache`]. Fixed at the block size so a group's transpose is exactly
/// one block per channel, which is what lets this reuse [`QKvCache`] instead of new kernels.
pub const GROUP: usize = QK;

/// **K/V quantized per CHANNEL across `GROUP` tokens**, rather than per 32 consecutive channels.
///
/// The shipped [`QKvCache`] puts 32 adjacent CHANNELS in one block, which is where K's outlier
/// channels live: one outlier sets the scale and crushes the other 31. Grouping along TOKENS instead
/// gives each channel its own scale. Measured on captured K/V, mean over 24 layers, relative RMSE at
/// 4 bits: K **0.09591 → 0.03495**, near 3x better for the same bit budget. V barely moves (0.08660 →
/// 0.07010), so the asymmetric pairing — grouped K, per-block V — is what the data actually asks for.
///
/// ## Why this needed no new kernel
///
/// A group is `[GROUP, width]`. Transposed it is `[width, GROUP]`, which the existing block quantizer
/// reads as `width` rows of exactly one 32-value block each — i.e. one block per channel, spanning
/// GROUP tokens. So the whole thing is a permute plus [`QKvCache`], and the pack/unpack, the scale
/// formats and the two-blocks-share-a-word rounding are all inherited rather than rewritten.
///
/// Block `g*width + c` holds channel `c` of group `g`, so a flush appends `width` blocks at the end
/// and never rewrites an earlier one — the property that makes an append-in-place cache possible.
///
/// ## The cost, stated rather than hidden
///
/// A group cannot be quantized until `GROUP` tokens exist, so the tail stays f32 in `staged` —
/// bounded at `(GROUP-1) * width * 4` bytes, independent of context length. And `dequantize` pays one
/// permute over the whole cache, on top of the dequantize it already pays. A fused transposing
/// dequantize kernel would remove that; it is not written.
pub struct GroupedKvCache {
    inner: QKvCache,
    /// `[k, width]` with `k < GROUP` — the tail that has not filled a group yet.
    staged: Option<Tensor>,
    width: usize,
    rows: usize,
}

impl Clone for GroupedKvCache {
    /// Deep by construction: [`QKvCache`]'s own `Clone` copies device memory, and `staged` is an
    /// immutable `Tensor` that is replaced rather than written in place.
    fn clone(&self) -> Self {
        GroupedKvCache { inner: self.inner.clone(), staged: self.staged.clone(), width: self.width, rows: self.rows }
    }
}

impl GroupedKvCache {
    pub fn new(fmt: KvqFmt) -> Self {
        GroupedKvCache { inner: QKvCache::new(fmt), staged: None, width: 0, rows: 0 }
    }
    pub fn fmt(&self) -> KvqFmt { self.inner.fmt() }
    pub fn len(&self) -> usize { self.rows }
    pub fn is_empty(&self) -> bool { self.rows == 0 }
    pub fn width(&self) -> usize { self.width }
    /// Device bytes: the quantized groups plus the f32 tail.
    pub fn bytes(&self) -> usize {
        self.inner.bytes() + self.staged.as_ref().map_or(0, |t| t.shape[0] * self.width * 4)
    }
    /// Live bytes, slack excluded — see [`QKvCache::live_bytes`].
    pub fn live_bytes(&self) -> usize {
        self.inner.live_bytes() + self.staged.as_ref().map_or(0, |t| t.shape[0] * self.width * 4)
    }
    pub fn f32_bytes(&self) -> usize { self.rows * self.width * 4 }

    /// Append `[t, width]` rows, flushing every complete group.
    pub fn append(&mut self, ctx: &Arc<Context>, src: &Tensor) {
        assert_eq!(src.rank(), 2, "GroupedKvCache::append expects [t, width]");
        let (t, width) = (src.shape[0], src.shape[1]);
        if t == 0 { return }
        if self.width == 0 { self.width = width }
        assert_eq!(width, self.width, "GroupedKvCache width changed");

        let full = match self.staged.take() {
            Some(st) => st.cat(src, 0),
            None => src.contiguous(),
        };
        let mut off = 0;
        while full.shape[0] - off >= GROUP {
            // `[GROUP, width]` -> `[width, GROUP]`: one block per channel, spanning GROUP tokens.
            let group = full.narrow(0, off, GROUP).transpose(0, 1).contiguous();
            self.inner.append(ctx, &group);
            off += GROUP;
        }
        let keep = full.shape[0] - off;
        self.staged = if keep == 0 { None } else { Some(full.narrow(0, off, keep).contiguous()) };
        self.rows += t;
    }

    /// The whole history as `[rows, width]` f32.
    pub fn dequantize(&self, ctx: &Arc<Context>) -> Tensor {
        let done = self.rows - self.staged.as_ref().map_or(0, |t| t.shape[0]);
        let complete = if done == 0 {
            None
        } else {
            // ONE pass: each output element derives its source block directly (row r, channel c ->
            // block (r/GROUP)*width + c, lane r%GROUP), so the transpose costs no second kernel over
            // the history. Attention reads the cache every step; the permute+contiguous this replaces
            // was per-token overhead proportional to the whole cache. Same per-element arithmetic as
            // the two-pass path — `grouped_roundtrip_paths_agree` holds them bit-identical.
            let n = done * self.width;
            let out = empty(ctx, n);
            let (grid, gw) = groups2d(n);
            let wgsl = match self.inner.fmt() {
                KvqFmt::Q8_0 => DEQUANT_GROUPED_Q8_0_WGSL,
                KvqFmt::Q4_0 => DEQUANT_GROUPED_Q4_0_WGSL,
                KvqFmt::Q4_1 => DEQUANT_GROUPED_Q4_1_WGSL,
            };
            run(ctx, wgsl, "kvq_dequant_grouped",
                &[self.inner.codes().expect("done>0 implies buffers").as_ref(),
                  self.inner.scales().expect("done>0 implies buffers").as_ref(),
                  &out,
                  &unibuf(ctx, &[n as u32, gw, self.width as u32, GROUP as u32, 0, 0, 0, 0])],
                grid);
            Some(Tensor::from_parts(ctx, out, vec![done, self.width]))
        };
        match (complete, &self.staged) {
            (Some(c), Some(st)) => c.cat(st, 0),
            (Some(c), None) => c,
            (None, Some(st)) => st.clone(),
            (None, None) => Tensor::from_vec(ctx, &[], &[0, self.width.max(1)]),
        }
    }
}

/// One layer's K or V history, in whichever layout that side was configured for.
///
/// K and V want DIFFERENT answers and always have — the measurement is unambiguous. At 4 bits on real
/// captured K/V, grouping along tokens takes K from 0.09591 to 0.03495 relative RMSE, and V from
/// 0.08660 to only 0.07010. K has outlier channels and V does not, so the axis that rescues K buys
/// V almost nothing while costing it a staging tail and a permute per read.
///
/// So the two sides are configured separately rather than as one cache "format". They were already
/// separate objects; this is the type that lets them be separate *kinds*.
#[derive(Clone)]
pub enum KvStore {
    /// 32 consecutive CHANNELS per scale — appends with no staging, no permute on read.
    Block(QKvCache),
    /// One channel across [`GROUP`] TOKENS per scale — better on outlier channels, at the cost of a
    /// sub-group f32 tail and a permute per dequantize.
    Grouped(GroupedKvCache),
}

impl KvStore {
    pub fn block(fmt: KvqFmt) -> KvStore { KvStore::Block(QKvCache::new(fmt)) }
    pub fn grouped(fmt: KvqFmt) -> KvStore { KvStore::Grouped(GroupedKvCache::new(fmt)) }
    pub fn fmt(&self) -> KvqFmt {
        match self { KvStore::Block(c) => c.fmt(), KvStore::Grouped(c) => c.fmt() }
    }
    /// True when this side groups along tokens — the property that differs, named rather than
    /// inferred from the variant by every caller.
    pub fn is_grouped(&self) -> bool { matches!(self, KvStore::Grouped(_)) }
    pub fn len(&self) -> usize {
        match self { KvStore::Block(c) => c.len(), KvStore::Grouped(c) => c.len() }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn width(&self) -> usize {
        match self { KvStore::Block(c) => c.width(), KvStore::Grouped(c) => c.width() }
    }
    pub fn bytes(&self) -> usize {
        match self { KvStore::Block(c) => c.bytes(), KvStore::Grouped(c) => c.bytes() }
    }
    pub fn live_bytes(&self) -> usize {
        match self { KvStore::Block(c) => c.live_bytes(), KvStore::Grouped(c) => c.live_bytes() }
    }
    pub fn f32_bytes(&self) -> usize {
        match self { KvStore::Block(c) => c.f32_bytes(), KvStore::Grouped(c) => c.f32_bytes() }
    }
    pub fn append(&mut self, ctx: &Arc<Context>, src: &Tensor) {
        match self { KvStore::Block(c) => c.append(ctx, src), KvStore::Grouped(c) => c.append(ctx, src) }
    }
    pub fn dequantize(&self, ctx: &Arc<Context>) -> Tensor {
        match self { KvStore::Block(c) => c.dequantize(ctx), KvStore::Grouped(c) => c.dequantize(ctx) }
    }
    /// The first `rows`, in a buffer of its own — for prefix reuse.
    ///
    /// Refused on the grouped variant for now: a prefix that ends mid-group would have to split a
    /// quantized block, and silently rounding the length down to a group boundary would hand back
    /// FEWER tokens than asked for while reporting success — a prefix cache that quietly caches less
    /// than it claims. Returning `None` makes the caller treat it as a miss, which is a prefill.
    pub fn clone_prefix(&self, ctx: &Arc<Context>, rows: usize) -> Option<KvStore> {
        match self {
            KvStore::Block(c) => Some(KvStore::Block(c.clone_prefix(ctx, rows))),
            KvStore::Grouped(_) => None,
        }
    }
}

/// A **real copy**, not a handle share.
///
/// [`QKvCache::append`] writes into `codes`/`scales` in place at flat block offsets, so the derived
/// `Clone` would produce a cache that observes its source's later writes — and, worse, one that
/// silently clobbers the source when both are appended to, since both write at the same offset. That
/// is the bug shape already found once in this tree in the gated-delta-net state.
impl Clone for QKvCache {
    fn clone(&self) -> QKvCache {
        match &self.ctx {
            Some(ctx) => self.deep_clone(ctx),
            // Never appended to, so there is nothing to alias: no buffers, no width, no device.
            None => QKvCache { fmt: self.fmt, codes: None, scales: None, len: 0, cap: 0, width: 0, ctx: None },
        }
    }
}

impl QKvCache {
    pub fn new(fmt: KvqFmt) -> QKvCache {
        QKvCache { fmt, codes: None, scales: None, len: 0, cap: 0, width: 0, ctx: None }
    }

    /// One-shot: quantize a whole `[t, width]` tensor into a fresh cache.
    pub fn from_tensor(ctx: &Arc<Context>, src: &Tensor, fmt: KvqFmt) -> QKvCache {
        let mut c = QKvCache::new(fmt);
        c.append(ctx, src);
        c
    }

    pub fn fmt(&self) -> KvqFmt { self.fmt }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Row width in elements, or 0 before the first append.
    pub fn width(&self) -> usize { self.width }
    /// Device bytes currently allocated (codes + scales), i.e. what the cache actually costs.
    pub fn bytes(&self) -> usize {
        let nblk = self.cap * (self.width / QK);
        (nblk * self.fmt.code_words() + self.fmt.scale_words(nblk)) * 4
    }
    /// What the FILLED rows cost, ignoring allocated slack — `bytes()` with `len` in place of `cap`.
    ///
    /// Exists because `bytes() / f32_bytes()` is not the format's ratio and must not be read as one.
    /// `bytes()` counts **allocated** capacity and `f32_bytes()` counts **live** rows, so the quotient
    /// is the format ratio divided by the current slack. Growth here doubles (`new_cap =
    /// need.max(cap*2).max(64)`), and [`crate::KvBuf`] doubles too but exposes no capacity — so an
    /// apples-to-apples allocated-vs-allocated comparison is not expressible, and the shipped quotient
    /// understates by up to 2x. Measured on LFM2.5-1.2B: 1.93x at 262 rows where the format ratio is
    /// 3.76x, the whole gap being one doubling.
    ///
    /// Report both. `bytes()` is what the device is actually holding; `live_bytes()` over
    /// `f32_bytes()` is what the FORMAT buys.
    pub fn live_bytes(&self) -> usize {
        let nblk = self.len * (self.width / QK);
        (nblk * self.fmt.code_words() + self.fmt.scale_words(nblk)) * 4
    }
    /// What the same `len × width` cache would cost as f32 — the number `live_bytes()` is measured
    /// against. See [`QKvCache::live_bytes`] for why `bytes()` is not the right numerator here.
    pub fn f32_bytes(&self) -> usize { self.len * self.width * 4 }
    /// Packed code words. Public because the point of matching the `Q*Weights` layout is that an
    /// attention kernel can consume the cache directly.
    pub fn codes(&self) -> Option<&Arc<wgpu::Buffer>> { self.codes.as_ref() }
    /// Packed scale words (f16 `d` pairs, or `(d, m)` per block for Q4_1).
    pub fn scales(&self) -> Option<&Arc<wgpu::Buffer>> { self.scales.as_ref() }

    fn nblk_per_row(&self) -> usize { self.width / QK }

    /// Append `src` (`[t, width]`, possibly a strided view) by quantizing exactly those rows.
    ///
    /// Grows (doubling) if needed; growth copies the existing code/scale words into the larger
    /// buffers, which is amortized O(1) per row exactly as in `KvBuf`.
    pub fn append(&mut self, ctx: &Arc<Context>, src: &Tensor) {
        assert_eq!(src.rank(), 2, "QKvCache::append expects a 2D [t, width] tensor");
        let (t, width) = (src.shape[0], src.shape[1]);
        if t == 0 { return; }
        assert_eq!(width % QK, 0, "QKvCache row width {width} must be a multiple of {QK}");
        if self.width == 0 { self.width = width; }
        if self.ctx.is_none() { self.ctx = Some(ctx.clone()); }
        assert_eq!(width, self.width, "QKvCache width changed");

        let need = self.len + t;
        if self.cap < need {
            let new_cap = need.max(self.cap * 2).max(64);
            self.grow(ctx, new_cap);
        }

        // Read the source in place when it is row-major (the decode case is a window onto a larger
        // fused QKV buffer); pack it first otherwise. Same rule as `write_rows`.
        let owned;
        let v = if src.strides[1] == 1 {
            src
        } else {
            owned = src.contiguous();
            &owned
        };
        let row_stride = if t > 1 { v.strides[0] } else { width };

        let nblk = self.nblk_per_row();
        let blk_start = self.len * nblk;
        let blk_end = need * nblk;
        // One thread per PAIR of blocks: Q8_0/Q4_0 pack two f16 scales into one u32, so the thread
        // that owns a scale word must own both of its halves or two threads would race on it. The
        // pair grid is anchored at block 0 (not at `blk_start`), so an append that begins mid-word
        // is handled by the same thread that owns the other half — which reads that half back and
        // rewrites it unchanged rather than clobbering it.
        let first_word = blk_start / 2;
        let last_word = (blk_end - 1) / 2;
        let nthreads = last_word - first_word + 1;
        let (grid, rs) = groups2d(nthreads);
        let (wgsl, label) = match self.fmt {
            KvqFmt::Q8_0 => (QUANT_Q8_0_WGSL, "kvq_quant_q8_0"),
            KvqFmt::Q4_0 => (QUANT_Q4_0_WGSL, "kvq_quant_q4_0"),
            KvqFmt::Q4_1 => (QUANT_Q4_1_WGSL, "kvq_quant_q4_1"),
        };
        run(
            ctx,
            wgsl,
            label,
            &[
                &v.buf,
                self.codes.as_ref().unwrap(),
                self.scales.as_ref().unwrap(),
                &unibuf(ctx, &[
                    nthreads as u32, rs, blk_start as u32, blk_end as u32,
                    v.offset as u32, row_stride as u32, nblk as u32, self.len as u32,
                ]),
            ],
            grid,
        );
        self.len = need;
    }

    fn grow(&mut self, ctx: &Arc<Context>, new_cap: usize) {
        let nblk = self.nblk_per_row();
        let new_code_words = new_cap * nblk * self.fmt.code_words();
        let new_scale_words = self.fmt.scale_words(new_cap * nblk);
        let nc = Arc::new(empty(ctx, new_code_words));
        let ns = Arc::new(empty(ctx, new_scale_words));
        if self.len > 0 {
            let live = self.len * nblk;
            copy_u32(ctx, self.codes.as_ref().unwrap(), &nc, live * self.fmt.code_words());
            copy_u32(ctx, self.scales.as_ref().unwrap(), &ns, self.fmt.scale_words(live));
        }
        self.codes = Some(nc);
        self.scales = Some(ns);
        self.cap = new_cap;
    }

    /// A copy that shares **no device memory** with `self`.
    ///
    /// [`QKvCache`] deliberately does not implement `Clone`, and this is the reason: [`Self::append`]
    /// writes into the existing `codes`/`scales` buffers in place, at flat block offsets. Those
    /// buffers are `Arc<wgpu::Buffer>`, so a derived `Clone` would hand back a handle that *observes
    /// every later append* — a snapshot that silently moves with the thing it was taken from. That
    /// exact bug shipped in this tree once already, in the gated-delta-net state, where
    /// `Tensor::contiguous()` returned `self.clone()` for an already-contiguous input and a
    /// speculative-decode rollback restored a state the draft had advanced. See
    /// `examples/gdn_state_aliasing.rs`.
    ///
    /// Three features need this primitive and each is blocked without it:
    ///   - **prefix caching** — copy a prefix's KV into a new sequence (`clone_prefix`);
    ///   - **speculative decoding** — `qwen35::Cache::snapshot` derives `Clone`, so a quantized cache
    ///     cannot live in it at all until the copy is real;
    ///   - **batched decode** — N caches indexed per layer, forked from a shared prefill.
    ///
    /// Copies only the LIVE region. Capacity beyond `len` is uninitialised on both sides, and copying
    /// it would make a cache's cost depend on its growth history rather than its contents.
    pub fn deep_clone(&self, ctx: &Arc<Context>) -> QKvCache {
        let mut out = QKvCache {
            fmt: self.fmt, codes: None, scales: None, len: self.len, cap: self.cap, width: self.width,
            ctx: Some(ctx.clone()),
        };
        let (Some(c), Some(s)) = (&self.codes, &self.scales) else { return out };
        let nblk = self.nblk_per_row();
        let nc = Arc::new(empty(ctx, self.cap * nblk * self.fmt.code_words()));
        let ns = Arc::new(empty(ctx, self.fmt.scale_words(self.cap * nblk)));
        let live = self.len * nblk;
        if live > 0 {
            // Same word accounting as `grow`: for the formats that pack two block scales per word, an
            // odd `len` shares its last word with a block that is not live, and `scale_words` rounds up
            // to include it. Copying that word is correct — it is the live block's own scale too.
            copy_u32(ctx, c, &nc, live * self.fmt.code_words());
            copy_u32(ctx, s, &ns, self.fmt.scale_words(live));
        }
        out.codes = Some(nc);
        out.scales = Some(ns);
        out
    }

    /// The first `rows` of this cache, in a buffer of its own — the quantized twin of
    /// [`crate::KvBuf::clone_prefix`], and the primitive prefix caching was missing.
    ///
    /// `crate::prefix` refuses a quantized cache today for exactly this reason: it copies a cached
    /// prefix's K/V into a new sequence, and without a row-bounded copy the only options were sharing
    /// a buffer that [`Self::append`] writes in place, or refusing. It refused, which was right.
    ///
    /// Row counts need no alignment care: blocks run along the row (`width / 32` of them per row), so
    /// the flat block index of row `r` is exactly `r * nblk_per_row` and any `rows` lands on a block
    /// boundary. The SCALE words are the subtle part — Q8_0 and Q4_0 pack two block scales per word,
    /// so an odd block count shares its last word with a block that is not being copied. Copying that
    /// whole word is correct: the half that matters is the live one, and the other half is overwritten
    /// by the first append. This is the same rounding `grow` does.
    pub fn clone_prefix(&self, ctx: &Arc<Context>, rows: usize) -> QKvCache {
        let rows = rows.min(self.len);
        if rows == 0 || self.width == 0 {
            return QKvCache { fmt: self.fmt, codes: None, scales: None, len: 0, cap: 0, width: 0, ctx: None };
        }
        let nblk = self.nblk_per_row();
        // `max(64)` mirrors KvBuf::clone_prefix and `append`'s own floor: a forked sequence is about to
        // be appended to, so handing it a cache sized exactly to the prefix guarantees a grow on the
        // very next token.
        let cap = rows.max(64);
        let nc = Arc::new(empty(ctx, cap * nblk * self.fmt.code_words()));
        let ns = Arc::new(empty(ctx, self.fmt.scale_words(cap * nblk)));
        let live = rows * nblk;
        copy_u32(ctx, self.codes.as_ref().expect("len>0 implies buffers"), &nc, live * self.fmt.code_words());
        copy_u32(ctx, self.scales.as_ref().expect("len>0 implies buffers"), &ns, self.fmt.scale_words(live));
        QKvCache {
            fmt: self.fmt, codes: Some(nc), scales: Some(ns),
            len: rows, cap, width: self.width, ctx: Some(ctx.clone()),
        }
    }

    /// Dequantize the whole cache back to a `[len, width]` f32 tensor.
    pub fn dequantize(&self, ctx: &Arc<Context>) -> Tensor {
        self.dequantize_rows(ctx, 0, self.len)
    }

    /// Dequantize `rows` rows starting at `row0` into a fresh `[rows, width]` f32 tensor.
    pub fn dequantize_rows(&self, ctx: &Arc<Context>, row0: usize, rows: usize) -> Tensor {
        assert!(row0 + rows <= self.len, "dequantize_rows({row0}, {rows}) past len {}", self.len);
        let n = rows * self.width;
        let out = empty(ctx, n.max(1));
        if n > 0 {
            let (grid, rs) = groups2d(n);
            let (wgsl, label) = match self.fmt {
                KvqFmt::Q8_0 => (DEQUANT_Q8_0_WGSL, "kvq_dequant_q8_0"),
                KvqFmt::Q4_0 => (DEQUANT_Q4_0_WGSL, "kvq_dequant_q4_0"),
                KvqFmt::Q4_1 => (DEQUANT_Q4_1_WGSL, "kvq_dequant_q4_1"),
            };
            run(
                ctx,
                wgsl,
                label,
                &[
                    self.codes.as_ref().unwrap(),
                    self.scales.as_ref().unwrap(),
                    &out,
                    &unibuf(ctx, &[
                        n as u32, rs, self.nblk_per_row() as u32, self.width as u32,
                        row0 as u32, 0, 0, 0,
                    ]),
                ],
                grid,
            );
        }
        Tensor::from_parts(ctx, out, vec![rows, self.width])
    }

    /// Read the packed representation back to the host — `(codes, scales)` — for comparison against
    /// [`reference`].
    pub async fn to_host(&self, ctx: &Arc<Context>) -> (Vec<u32>, Vec<u32>) {
        let live = self.len * self.nblk_per_row();
        let cw = live * self.fmt.code_words();
        let sw = self.fmt.scale_words(live);
        let c = match &self.codes {
            Some(b) => readback_u32(ctx, b, cw).await,
            None => Vec::new(),
        };
        let s = match &self.scales {
            Some(b) => readback_u32(ctx, b, sw).await,
            None => Vec::new(),
        };
        (c, s)
    }
}

/// Straight `dst[i] = src[i]` over `n` u32 words.
///
/// A u32 kernel rather than reusing an f32 copy: code words are bit patterns, and moving them through
/// an f32 load/store would let a word that happens to be a signalling NaN come out canonicalized —
/// a corruption that changes a handful of codes and nothing else, i.e. one that reads as a slightly
/// worse quantizer instead of as a bug.
fn copy_u32(ctx: &Context, src: &wgpu::Buffer, dst: &wgpu::Buffer, n: usize) {
    if n == 0 { return; }
    let (grid, rs) = groups2d(n);
    run(ctx, COPY_U32_WGSL, "kvq_copy_u32", &[src, dst, &unibuf(ctx, &[n as u32, rs, 0, 0])], grid);
}

const COPY_U32_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       src: array<u32>;
@group(0) @binding(1) var<storage,read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform>            info: vec4<u32>; // n, grid_w, _, _
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info.y;
    if (i >= info.x) { return; }
    dst[i] = src[i];
}
"#;

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------
//
// Shared uniform layout for the three quantize kernels:
//   info[0] = (nthreads, grid_w, blk_start, blk_end)
//   info[1] = (src_off, src_row_stride, nblk_per_row, row0)
// One invocation owns blocks `2w` and `2w+1` where `w = blk_start/2 + i`; it skips whichever of the
// two falls outside [blk_start, blk_end).
//
// `rnd` is round-half-AWAY-from-zero, which is what C's `roundf` does and therefore what llama.cpp's
// reference quantizers do. WGSL's own `round()` is round-half-to-EVEN, so using it would put half of
// the exact ties one step away from the reference — invisible on random data and systematic on the
// data that has ties, which for a KV cache (post-rope, post-norm) is not nothing.
macro_rules! quant_wgsl {
    ($body:literal) => {
        concat!(
            r#"
@group(0) @binding(0) var<storage,read>       src:    array<f32>;
@group(0) @binding(1) var<storage,read_write> codes:  array<u32>;
@group(0) @binding(2) var<storage,read_write> scales: array<u32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
fn rnd(x: f32) -> f32 { return sign(x) * floor(abs(x) + 0.5); }
"#,
            $body
        )
    };
}

const QUANT_Q8_0_WGSL: &str = quant_wgsl!(r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let w = (info[0].z >> 1u) + i;
    let nblk = info[1].z;
    var sw = unpack2x16float(scales[w]);
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
        let b = 2u * w + h;
        if (b < info[0].z || b >= info[0].w) { continue; }
        let r = b / nblk;
        let cb = b - r * nblk;
        let base = info[1].x + (r - info[1].w) * info[1].y + cb * 32u;
        var amax = 0.0;
        for (var j: u32 = 0u; j < 32u; j = j + 1u) { amax = max(amax, abs(src[base + j])); }
        let d = amax / 127.0;
        var id = 0.0;
        if (d > 0.0) { id = 1.0 / d; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            var word: u32 = 0u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let qi = i32(clamp(rnd(src[base + k * 4u + q] * id), -127.0, 127.0));
                word = word | ((bitcast<u32>(qi) & 255u) << (8u * q));
            }
            codes[b * 8u + k] = word;
        }
        if (h == 0u) { sw.x = d; } else { sw.y = d; }
    }
    scales[w] = pack2x16float(sw);
}
"#);

// q4_0: d = max/-8 where `max` is the SIGNED value with the largest magnitude, code = min(15,
// floor(x/d + 8.5)). Nibble j of the block lives in byte j%16: low nibble is value j, high nibble is
// value j+16 (llama.cpp's split-half interleave, not sequential pairs).
const QUANT_Q4_0_WGSL: &str = quant_wgsl!(r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let w = (info[0].z >> 1u) + i;
    let nblk = info[1].z;
    var sw = unpack2x16float(scales[w]);
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
        let b = 2u * w + h;
        if (b < info[0].z || b >= info[0].w) { continue; }
        let r = b / nblk;
        let cb = b - r * nblk;
        let base = info[1].x + (r - info[1].w) * info[1].y + cb * 32u;
        var amax = 0.0;
        var mx = 0.0;
        for (var j: u32 = 0u; j < 32u; j = j + 1u) {
            let v = src[base + j];
            if (amax < abs(v)) { amax = abs(v); mx = v; }
        }
        let d = mx / -8.0;
        var id = 0.0;
        if (d != 0.0) { id = 1.0 / d; }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            var word: u32 = 0u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let j = k * 4u + q;
                let lo = u32(clamp(floor(src[base + j] * id + 8.5), 0.0, 15.0));
                let hi = u32(clamp(floor(src[base + j + 16u] * id + 8.5), 0.0, 15.0));
                word = word | ((lo | (hi << 4u)) << (8u * q));
            }
            codes[b * 4u + k] = word;
        }
        if (h == 0u) { sw.x = d; } else { sw.y = d; }
    }
    scales[w] = pack2x16float(sw);
}
"#);

// q4_1: affine. d = (max-min)/15, m = min, code = min(15, floor((x-min)/d + 0.5)). Scales are one
// word per block — (d, m) — so no half-word is shared and the pair loop just writes both.
const QUANT_Q4_1_WGSL: &str = quant_wgsl!(r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let w = (info[0].z >> 1u) + i;
    let nblk = info[1].z;
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
        let b = 2u * w + h;
        if (b < info[0].z || b >= info[0].w) { continue; }
        let r = b / nblk;
        let cb = b - r * nblk;
        let base = info[1].x + (r - info[1].w) * info[1].y + cb * 32u;
        var mn = src[base];
        var mx = src[base];
        for (var j: u32 = 1u; j < 32u; j = j + 1u) {
            let v = src[base + j];
            mn = min(mn, v);
            mx = max(mx, v);
        }
        let d = (mx - mn) / 15.0;
        var id = 0.0;
        if (d != 0.0) { id = 1.0 / d; }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            var word: u32 = 0u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let j = k * 4u + q;
                let lo = u32(clamp(floor((src[base + j] - mn) * id + 0.5), 0.0, 15.0));
                let hi = u32(clamp(floor((src[base + j + 16u] - mn) * id + 0.5), 0.0, 15.0));
                word = word | ((lo | (hi << 4u)) << (8u * q));
            }
            codes[b * 4u + k] = word;
        }
        scales[b] = pack2x16float(vec2<f32>(d, mn));
    }
}
"#);

// Dequantize: one invocation per output element.
//   info[0] = (n, grid_w, nblk_per_row, width)
//   info[1] = (row0, _, _, _)
const DEQUANT_Q8_0_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let width = info[0].w;
    let r = i / width;
    let c = i - r * width;
    let b = (info[1].x + r) * info[0].z + (c >> 5u);
    let j = c & 31u;
    let sw = unpack2x16float(scales[b >> 1u]);
    let d = select(sw.y, sw.x, (b & 1u) == 0u);
    let word = codes[b * 8u + (j >> 2u)];
    let q = i32(word << (24u - 8u * (j & 3u))) >> 24u;
    out[i] = f32(q) * d;
}
"#;

const DEQUANT_Q4_0_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let width = info[0].w;
    let r = i / width;
    let c = i - r * width;
    let b = (info[1].x + r) * info[0].z + (c >> 5u);
    let j = c & 31u;
    let sw = unpack2x16float(scales[b >> 1u]);
    let d = select(sw.y, sw.x, (b & 1u) == 0u);
    let byte_i = j & 15u;
    let word = codes[b * 4u + (byte_i >> 2u)];
    let byte = (word >> (8u * (byte_i & 3u))) & 255u;
    let nib = select(byte >> 4u, byte & 15u, j < 16u);
    out[i] = (f32(nib) - 8.0) * d;
}
"#;

const DEQUANT_Q4_1_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>       codes:  array<u32>;
@group(0) @binding(1) var<storage,read>       scales: array<u32>;
@group(0) @binding(2) var<storage,read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>            info:   array<vec4<u32>, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * info[0].y;
    if (i >= info[0].x) { return; }
    let width = info[0].w;
    let r = i / width;
    let c = i - r * width;
    let b = (info[1].x + r) * info[0].z + (c >> 5u);
    let j = c & 31u;
    let dm = unpack2x16float(scales[b]);
    let byte_i = j & 15u;
    let word = codes[b * 4u + (byte_i >> 2u)];
    let byte = (word >> (8u * (byte_i & 3u))) & 255u;
    let nib = select(byte >> 4u, byte & 15u, j < 16u);
    out[i] = f32(nib) * dm.x + dm.y;
}
"#;

/// Read `n` u32 words back to the host.
///
/// The crate's `readback` returns f32; codes are bit patterns and must not travel through an f32
/// value, so this is its u32 twin. Same batch-flush contract: a read must see every dispatch already
/// issued, including ones still sitting in an open batch.
async fn readback_u32(ctx: &Context, buf: &wgpu::Buffer, n: usize) -> Vec<u32> {
    crate::flush_batch(ctx);
    if n == 0 { return Vec::new(); }
    let bytes = (n * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kvq.staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    ctx.queue.submit([enc.finish()]);
    let (tx, rx) = flume::bounded(1);
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv_async().await.unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

// ---------------------------------------------------------------------------------------------
// CPU reference — the validation source of truth for the kernels above
// ---------------------------------------------------------------------------------------------

pub mod reference {
    //! Bit-for-bit CPU twin of the kernels, in the role [`crate::cpu`] plays for the general ops.
    //!
    //! Every constant, rounding rule and index expression here mirrors the WGSL. A GPU/CPU
    //! disagreement is therefore a real defect in one of them, not a difference of convention —
    //! which is the only way this comparison is worth running.

    use super::{KvqFmt, QK};

    /// f32 → IEEE binary16 bits, round-to-nearest-even. Matches WGSL `pack2x16float`.
    ///
    /// Written out rather than pulled from `half`, which this crate only depends on for macOS
    /// targets; the tests below check this against `half` exhaustively over the f16 grid.
    pub fn f32_to_f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x007f_ffff;
        if exp == 0xff {
            // inf / NaN — keep NaN a NaN (quiet), never let it become inf
            return sign | 0x7c00 | if mant != 0 { 0x0200 | (mant >> 13) as u16 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign; // underflows even the f16 subnormal grid
            }
            let m = mant | 0x0080_0000;
            let shift = (14 - e) as u32;
            let r = m >> shift;
            let rem = m & ((1u32 << shift) - 1);
            let half = 1u32 << (shift - 1);
            let r = if rem > half || (rem == half && (r & 1) == 1) { r + 1 } else { r };
            return sign | r as u16;
        }
        let mut m = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            m += 1;
        }
        let mut ee = e as u32;
        if m == 0x400 {
            m = 0;
            ee += 1;
            if ee >= 0x1f {
                return sign | 0x7c00;
            }
        }
        sign | ((ee as u16) << 10) | m as u16
    }

    /// IEEE binary16 bits → f32. Matches WGSL `unpack2x16float`.
    pub fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h as u32) & 0x8000) << 16;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h as u32) & 0x3ff;
        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign);
            }
            let mut m = mant;
            let mut s = 0u32;
            while m & 0x400 == 0 {
                m <<= 1;
                s += 1;
            }
            return f32::from_bits(sign | ((113 - s) << 23) | ((m & 0x3ff) << 13));
        }
        if exp == 0x1f {
            return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
        }
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }

    /// Round half away from zero — C `roundf`, and what the kernels' `rnd()` does.
    #[inline]
    fn rnd(x: f32) -> f32 {
        x.signum() * (x.abs() + 0.5).floor()
    }

    /// Words needed to hold `rows × width` quantized as `fmt`.
    pub fn sizes(rows: usize, width: usize, fmt: KvqFmt) -> (usize, usize) {
        let nblk = rows * (width / QK);
        (nblk * fmt.code_words(), fmt.scale_words(nblk))
    }

    /// Quantize `rows × width` f32 into `codes`/`scales` at cache rows `row0 .. row0+rows`.
    ///
    /// `codes`/`scales` must already be large enough for `row0+rows`; halves of a shared scale word
    /// that belong to blocks outside this range are left alone, exactly as the kernel leaves them.
    pub fn quantize_into(
        x: &[f32],
        rows: usize,
        width: usize,
        fmt: KvqFmt,
        row0: usize,
        codes: &mut [u32],
        scales: &mut [u32],
    ) {
        assert_eq!(width % QK, 0, "width {width} must be a multiple of {QK}");
        assert_eq!(x.len(), rows * width, "x is not rows*width");
        let nblk = width / QK;
        for r in 0..rows {
            for cb in 0..nblk {
                let b = (row0 + r) * nblk + cb;
                let blk = &x[r * width + cb * QK..r * width + cb * QK + QK];
                match fmt {
                    KvqFmt::Q8_0 => {
                        let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
                        let d = amax / 127.0;
                        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                        for k in 0..8 {
                            let mut word = 0u32;
                            for q in 0..4 {
                                let qi = rnd(blk[k * 4 + q] * id).clamp(-127.0, 127.0) as i32;
                                word |= ((qi as u32) & 255) << (8 * q);
                            }
                            codes[b * 8 + k] = word;
                        }
                        put_half(scales, b, d);
                    }
                    KvqFmt::Q4_0 => {
                        let (mut amax, mut mx) = (0f32, 0f32);
                        for &v in blk {
                            if amax < v.abs() {
                                amax = v.abs();
                                mx = v;
                            }
                        }
                        let d = mx / -8.0;
                        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                        for k in 0..4 {
                            let mut word = 0u32;
                            for q in 0..4 {
                                let j = k * 4 + q;
                                let lo = (blk[j] * id + 8.5).floor().clamp(0.0, 15.0) as u32;
                                let hi = (blk[j + 16] * id + 8.5).floor().clamp(0.0, 15.0) as u32;
                                word |= (lo | (hi << 4)) << (8 * q);
                            }
                            codes[b * 4 + k] = word;
                        }
                        put_half(scales, b, d);
                    }
                    KvqFmt::Q4_1 => {
                        let mn = blk.iter().fold(f32::INFINITY, |a, &v| a.min(v));
                        let mx = blk.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                        let d = (mx - mn) / 15.0;
                        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                        for k in 0..4 {
                            let mut word = 0u32;
                            for q in 0..4 {
                                let j = k * 4 + q;
                                let lo = ((blk[j] - mn) * id + 0.5).floor().clamp(0.0, 15.0) as u32;
                                let hi = ((blk[j + 16] - mn) * id + 0.5).floor().clamp(0.0, 15.0) as u32;
                                word |= (lo | (hi << 4)) << (8 * q);
                            }
                            codes[b * 4 + k] = word;
                        }
                        scales[b] = (f32_to_f16_bits(d) as u32) | ((f32_to_f16_bits(mn) as u32) << 16);
                    }
                }
            }
        }
    }

    fn put_half(scales: &mut [u32], b: usize, d: f32) {
        let w = b / 2;
        let bits = f32_to_f16_bits(d) as u32;
        if b % 2 == 0 {
            scales[w] = (scales[w] & 0xffff_0000) | bits;
        } else {
            scales[w] = (scales[w] & 0x0000_ffff) | (bits << 16);
        }
    }

    /// Quantize a standalone `rows × width` slab.
    pub fn quantize(x: &[f32], rows: usize, width: usize, fmt: KvqFmt) -> (Vec<u32>, Vec<u32>) {
        let (cw, sw) = sizes(rows, width, fmt);
        let mut codes = vec![0u32; cw];
        let mut scales = vec![0u32; sw];
        quantize_into(x, rows, width, fmt, 0, &mut codes, &mut scales);
        (codes, scales)
    }

    /// Dequantize `rows` rows starting at cache row `row0`.
    pub fn dequantize_rows(
        codes: &[u32],
        scales: &[u32],
        row0: usize,
        rows: usize,
        width: usize,
        fmt: KvqFmt,
    ) -> Vec<f32> {
        let nblk = width / QK;
        let mut out = vec![0f32; rows * width];
        for r in 0..rows {
            for cb in 0..nblk {
                let b = (row0 + r) * nblk + cb;
                match fmt {
                    KvqFmt::Q8_0 => {
                        let d = get_half(scales, b);
                        for j in 0..QK {
                            let word = codes[b * 8 + j / 4];
                            let q = ((word >> (8 * (j % 4))) & 255) as u8 as i8;
                            out[r * width + cb * QK + j] = q as f32 * d;
                        }
                    }
                    KvqFmt::Q4_0 => {
                        let d = get_half(scales, b);
                        for j in 0..QK {
                            let bi = j % 16;
                            let byte = (codes[b * 4 + bi / 4] >> (8 * (bi % 4))) & 255;
                            let nib = if j < 16 { byte & 15 } else { byte >> 4 };
                            out[r * width + cb * QK + j] = (nib as f32 - 8.0) * d;
                        }
                    }
                    KvqFmt::Q4_1 => {
                        let d = f16_bits_to_f32((scales[b] & 0xffff) as u16);
                        let m = f16_bits_to_f32((scales[b] >> 16) as u16);
                        for j in 0..QK {
                            let bi = j % 16;
                            let byte = (codes[b * 4 + bi / 4] >> (8 * (bi % 4))) & 255;
                            let nib = if j < 16 { byte & 15 } else { byte >> 4 };
                            out[r * width + cb * QK + j] = nib as f32 * d + m;
                        }
                    }
                }
            }
        }
        out
    }

    fn get_half(scales: &[u32], b: usize) -> f32 {
        let w = scales[b / 2];
        f16_bits_to_f32(if b % 2 == 0 { (w & 0xffff) as u16 } else { (w >> 16) as u16 })
    }

    /// Quantize then dequantize — the reconstruction a runtime would actually attend over.
    pub fn roundtrip(x: &[f32], rows: usize, width: usize, fmt: KvqFmt) -> Vec<f32> {
        let (c, s) = quantize(x, rows, width, fmt);
        dequantize_rows(&c, &s, 0, rows, width, fmt)
    }
}

// ---------------------------------------------------------------------------------------------
// Scaling-granularity study
// ---------------------------------------------------------------------------------------------

/// How a scale is shared across a `[rows, width]` K or V tensor.
///
/// This exists to answer one question with a measurement instead of a citation: **does K need
/// per-channel scaling, or is per-block-of-32-along-the-row enough?** The two differ in exactly the
/// way that matters for a cache — see [`append_cost`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GranKind {
    /// One scale for the entire tensor.
    Tensor,
    /// One scale per token (row) — a scale shared by all `width` channels.
    PerToken,
    /// One scale per contiguous run of `n` channels within a row. `PerBlock(32)` is what this module
    /// ships.
    PerBlock(usize),
    /// One scale per channel (column), shared by every token.
    PerChannel,
    /// One scale per channel per group of `g` consecutive tokens.
    PerChannelGroup(usize),
}

/// What it costs to append one token's row under a granularity.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AppendCost {
    /// Only the new row's own codes are written. O(width) per token — usable during decode.
    InPlace,
    /// The most recent `g` rows must be held in f32; every `g`-th token quantizes one group.
    /// O(width) amortized, plus a `g × width` f32 residual buffer.
    GroupFlush(usize),
    /// A new token can move a scale that older codes were computed against, so every code in the
    /// cache must be recomputed. **O(len × width) per token** — quadratic over a generation, which
    /// is the same cost the whole `KvBuf` design exists to avoid.
    FullRequant,
}

/// The append cost of a granularity. Stated as code so that "this scheme cannot be appended to" is a
/// property of the design rather than a remark in a comment.
pub fn append_cost(kind: GranKind) -> AppendCost {
    match kind {
        GranKind::Tensor => AppendCost::FullRequant,
        GranKind::PerToken => AppendCost::InPlace,
        GranKind::PerBlock(_) => AppendCost::InPlace,
        GranKind::PerChannel => AppendCost::FullRequant,
        GranKind::PerChannelGroup(g) => AppendCost::GroupFlush(g),
    }
}

/// Reconstruction error of a quantize→dequantize round trip.
#[derive(Copy, Clone, Debug)]
pub struct ErrStats {
    pub rmse: f32,
    /// `rmse / rms(x)` — the scale-free number to compare across tensors, layers and models.
    pub rel_rmse: f32,
    pub max_abs: f32,
    /// Cosine similarity between the original and the reconstruction, flattened. Attention is a dot
    /// product, so the angle is closer to what actually degrades than the norm is.
    pub cos: f32,
    /// Storage cost per value including scales, in bits.
    pub bits_per_value: f32,
}

fn group_of(kind: GranKind, r: usize, c: usize, width: usize) -> usize {
    match kind {
        GranKind::Tensor => 0,
        GranKind::PerToken => r,
        GranKind::PerBlock(n) => r * width.div_ceil(n) + c / n,
        GranKind::PerChannel => c,
        GranKind::PerChannelGroup(g) => (r / g) * width + c,
    }
}

fn n_groups(kind: GranKind, rows: usize, width: usize) -> usize {
    match kind {
        GranKind::Tensor => 1,
        GranKind::PerToken => rows,
        GranKind::PerBlock(n) => rows * width.div_ceil(n),
        GranKind::PerChannel => width,
        GranKind::PerChannelGroup(g) => rows.div_ceil(g) * width,
    }
}

/// Values sharing one scale, under a granularity.
fn group_size(kind: GranKind, rows: usize, width: usize) -> f32 {
    (rows * width) as f32 / n_groups(kind, rows, width) as f32
}

/// Round-trip error of a `[rows, width]` tensor under a scaling granularity, at `bits` bits.
///
/// `asym` picks the affine codebook (`q·d + m`, `d = (max−min)/(2^bits − 1)`, the q4_1 family) over
/// the symmetric one (`q·d`, `d = amax/(2^(bits−1) − 1)`, the q8_0/q4_0 family).
///
/// Scales here are **f32**, deliberately: this function compares granularities, and folding the f16
/// scale in would mix two effects. [`shipped_err`] measures the formats as they actually store.
pub fn roundtrip_err(x: &[f32], rows: usize, width: usize, kind: GranKind, bits: u32, asym: bool) -> ErrStats {
    assert_eq!(x.len(), rows * width);
    let ng = n_groups(kind, rows, width);
    let mut lo = vec![f32::INFINITY; ng];
    let mut hi = vec![f32::NEG_INFINITY; ng];
    for r in 0..rows {
        for c in 0..width {
            let g = group_of(kind, r, c, width);
            let v = x[r * width + c];
            if v < lo[g] { lo[g] = v; }
            if v > hi[g] { hi[g] = v; }
        }
    }
    let levels_sym = ((1u32 << (bits - 1)) - 1) as f32;
    let levels_asym = ((1u32 << bits) - 1) as f32;
    let mut recon = vec![0f32; rows * width];
    for r in 0..rows {
        for c in 0..width {
            let g = group_of(kind, r, c, width);
            let v = x[r * width + c];
            recon[r * width + c] = if asym {
                let d = (hi[g] - lo[g]) / levels_asym;
                if d == 0.0 {
                    lo[g]
                } else {
                    let q = (((v - lo[g]) / d) + 0.5).floor().clamp(0.0, levels_asym);
                    q * d + lo[g]
                }
            } else {
                let amax = lo[g].abs().max(hi[g].abs());
                let d = amax / levels_sym;
                if d == 0.0 {
                    0.0
                } else {
                    let q = (v / d).signum() * ((v / d).abs() + 0.5).floor();
                    q.clamp(-levels_sym, levels_sym) * d
                }
            };
        }
    }
    // Scale storage: one f32 (asym: two) per group, amortized over the group.
    let per_group_bits = if asym { 64.0 } else { 32.0 };
    let bpv = bits as f32 + per_group_bits / group_size(kind, rows, width);
    stats(x, &recon, bpv)
}

/// Round-trip error of a **shipped block format**, exactly as this module stores it (f16 scales,
/// llama.cpp codebooks, per-block-of-32 along the row).
pub fn shipped_err(x: &[f32], rows: usize, width: usize, fmt: KvqFmt) -> ErrStats {
    let recon = reference::roundtrip(x, rows, width, fmt);
    stats(x, &recon, fmt.bits_per_value())
}

fn stats(x: &[f32], y: &[f32], bits_per_value: f32) -> ErrStats {
    let n = x.len() as f64;
    let mut se = 0f64;
    let mut sx = 0f64;
    let mut mx = 0f32;
    let (mut dxy, mut nx, mut ny) = (0f64, 0f64, 0f64);
    for i in 0..x.len() {
        let e = (y[i] - x[i]) as f64;
        se += e * e;
        sx += (x[i] as f64) * (x[i] as f64);
        mx = mx.max((y[i] - x[i]).abs());
        dxy += (x[i] as f64) * (y[i] as f64);
        nx += (x[i] as f64) * (x[i] as f64);
        ny += (y[i] as f64) * (y[i] as f64);
    }
    let rmse = (se / n).sqrt();
    let rms = (sx / n).sqrt();
    ErrStats {
        rmse: rmse as f32,
        rel_rmse: if rms > 0.0 { (rmse / rms) as f32 } else { 0.0 },
        max_abs: mx,
        cos: if nx > 0.0 && ny > 0.0 { (dxy / (nx.sqrt() * ny.sqrt())) as f32 } else { 0.0 },
        bits_per_value,
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;

    /// Deterministic LCG — tests must not depend on an rng crate or on run-to-run luck.
    struct Lcg(u64);
    impl Lcg {
        fn new(s: u64) -> Lcg { Lcg(s.wrapping_mul(6364136223846793005).wrapping_add(1)) }
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
        fn next_u32(&mut self, n: u32) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32) % n
        }
    }

    /// First differing word, so a failure reports a location instead of two thousand numbers.
    fn first_diff(a: &[u32], b: &[u32]) -> Option<(usize, u32, u32)> {
        assert_eq!(a.len(), b.len(), "word-count mismatch: {} vs {}", a.len(), b.len());
        (0..a.len()).find(|&i| a[i] != b[i]).map(|i| (i, a[i], b[i]))
    }

    fn ctx() -> Option<Arc<Context>> {
        match pollster::block_on(ferric_core::Context::new()) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                eprintln!("no GPU context ({e:?}); GPU half of this test did NOT run");
                None
            }
        }
    }

    // ---- f16 conversion, checked against the `half` crate -----------------------------------

    #[test]
    fn f16_decode_matches_half_over_the_whole_grid() {
        for bits in 0u32..=0xffff {
            let b = bits as u16;
            let mine = f16_bits_to_f32(b);
            let theirs = half::f16::from_bits(b).to_f32();
            if theirs.is_nan() {
                assert!(mine.is_nan(), "bits {b:#06x}: expected NaN, got {mine}");
            } else {
                assert_eq!(
                    mine.to_bits(),
                    theirs.to_bits(),
                    "f16 bits {b:#06x} decoded to {mine} (half says {theirs})"
                );
            }
        }
    }

    #[test]
    fn f16_encode_matches_half() {
        // Every f16 value round-trips (encode∘decode is the identity on the grid) ...
        for bits in 0u32..=0xffff {
            let b = bits as u16;
            let v = half::f16::from_bits(b).to_f32();
            if v.is_nan() { continue; }
            assert_eq!(f32_to_f16_bits(v), b, "round trip broke at f16 bits {b:#06x} (= {v})");
        }
        // ... and arbitrary f32s, where rounding actually happens, agree with `half`.
        let mut r = Lcg::new(0xC0FFEE);
        for i in 0..200_000 {
            // Sweep magnitudes across the whole f16 range: normals, subnormals, overflow.
            let mag = 10f32.powf((r.next_f32() * 9.0) as f32);
            let v = r.next_f32() * mag;
            assert_eq!(
                f32_to_f16_bits(v),
                half::f16::from_f32(v).to_bits(),
                "iteration {i}: f32 {v:e} encoded to {:#06x}, half says {:#06x}",
                f32_to_f16_bits(v),
                half::f16::from_f32(v).to_bits()
            );
        }
    }

    // ---- exact known-answer cases ------------------------------------------------------------

    /// Build `[rows, width]` whose quantization is *exactly* determined: every value is an integer
    /// code times a power-of-two scale, and each block pins its own scale by containing the extreme
    /// code. Nothing here sits near a rounding tie, so CPU and GPU must agree bit-for-bit no matter
    /// how each one's f32 divide rounds.
    ///
    /// Returns `(x, expected_codes_per_block)`.
    fn exact_case(rows: usize, width: usize, fmt: KvqFmt, seed: u64) -> (Vec<f32>, Vec<[i32; 32]>) {
        let mut r = Lcg::new(seed);
        let nblk = rows * (width / QK);
        let mut x = vec![0f32; rows * width];
        let mut want = vec![[0i32; 32]; nblk];
        for b in 0..nblk {
            let row = b / (width / QK);
            let cb = b % (width / QK);
            let s = 2f32.powi(-3 + (b % 5) as i32);
            // one block in eight is all-zero: the d == 0 branch is a real path, not a curiosity
            let zero = b % 8 == 3;
            let pin = r.next_u32(32) as usize;
            for j in 0..QK {
                let c: i32 = match fmt {
                    KvqFmt::Q8_0 => {
                        if j == pin { -127 } else { r.next_u32(255) as i32 - 127 }
                    }
                    KvqFmt::Q4_0 => {
                        // pin the -8 end so `max` (the signed value at amax) is negative and d = s
                        if j == pin { 0 } else { r.next_u32(16) as i32 }
                    }
                    KvqFmt::Q4_1 => {
                        // pin both ends so min/max are exactly -4s and 11s
                        if j == pin { 0 } else if j == (pin + 1) % 32 { 15 } else { r.next_u32(16) as i32 }
                    }
                };
                let v = if zero {
                    0.0
                } else {
                    match fmt {
                        KvqFmt::Q8_0 => c as f32 * s,
                        KvqFmt::Q4_0 => (c - 8) as f32 * s,
                        KvqFmt::Q4_1 => (c - 4) as f32 * s,
                    }
                };
                x[row * width + cb * QK + j] = v;
                want[b][j] = if zero {
                    match fmt {
                        KvqFmt::Q8_0 => 0,
                        // an all-zero block has d == 0; q4_0 stores floor(0 + 8.5) = 8, q4_1 floor(0.5) = 0
                        KvqFmt::Q4_0 => 8,
                        KvqFmt::Q4_1 => 0,
                    }
                } else {
                    c
                };
            }
        }
        (x, want)
    }

    fn codes_of(codes: &[u32], b: usize, fmt: KvqFmt) -> [i32; 32] {
        let mut out = [0i32; 32];
        for j in 0..QK {
            out[j] = match fmt {
                KvqFmt::Q8_0 => {
                    let w = codes[b * 8 + j / 4];
                    (((w >> (8 * (j % 4))) & 255) as u8 as i8) as i32
                }
                KvqFmt::Q4_0 | KvqFmt::Q4_1 => {
                    let bi = j % 16;
                    let byte = (codes[b * 4 + bi / 4] >> (8 * (bi % 4))) & 255;
                    (if j < 16 { byte & 15 } else { byte >> 4 }) as i32
                }
            };
        }
        out
    }

    #[test]
    fn cpu_reference_produces_the_exact_codes() {
        for fmt in KvqFmt::ALL {
            let (rows, width) = (7usize, 96usize); // 3 blocks/row: an ODD per-row block count, so the
                                                   // shared scale word straddles rows
            let (x, want) = exact_case(rows, width, fmt, 7);
            let (codes, _scales) = reference::quantize(&x, rows, width, fmt);
            for b in 0..rows * (width / QK) {
                assert_eq!(
                    codes_of(&codes, b, fmt),
                    want[b],
                    "{}: block {b} codes wrong",
                    fmt.name()
                );
            }
        }
    }

    #[test]
    fn cpu_reference_roundtrips_exactly_representable_data() {
        // Values that are exactly on the codebook grid must come back bit-identical: the only error a
        // block quantizer is allowed is the rounding it is asked to do.
        for fmt in KvqFmt::ALL {
            let (rows, width) = (5usize, 64usize);
            let (x, _) = exact_case(rows, width, fmt, 11);
            let y = reference::roundtrip(&x, rows, width, fmt);
            for i in 0..x.len() {
                assert_eq!(x[i], y[i], "{}: element {i} was {} came back {}", fmt.name(), x[i], y[i]);
            }
        }
    }

    #[test]
    fn round_half_away_from_zero_not_to_even() {
        // A block whose values land exactly on .5 boundaries. WGSL's `round()` (half-to-even) would
        // send 0.5 -> 0 and 2.5 -> 2; C's roundf sends them to 1 and 3. The kernels and this
        // reference must both do the latter, or blocks with tied values drift apart.
        let width = 32;
        let s = 1.0f32; // d = amax/127 = 1 requires amax = 127
        let mut x = vec![0f32; width];
        x[0] = 127.0 * s;
        for (j, v) in [0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5].iter().enumerate() {
            x[j + 1] = *v * s;
        }
        let (codes, _) = reference::quantize(&x, 1, width, KvqFmt::Q8_0);
        let got = codes_of(&codes, 0, KvqFmt::Q8_0);
        assert_eq!(&got[1..7], &[1, 2, 3, -1, -2, -3], "ties must round AWAY from zero");
    }

    // ---- GPU kernels against the CPU reference ------------------------------------------------

    /// The fused transposing dequantize must be BIT-identical to the two-pass oracle it replaced.
    ///
    /// The oracle — block dequantize then reshape/permute/contiguous — still exists as `inner`'s own
    /// path, so it is reconstructed here rather than kept as dead shipping code. Identical per-element
    /// arithmetic means identical bits, not merely close: any deviation is an index-mapping bug in the
    /// fused kernel (group/lane swapped, block row mis-derived), which is precisely the class of
    /// silent wrongness a tolerance would absorb.
    #[test]
    fn grouped_fused_dequant_matches_the_two_pass_oracle_bit_for_bit() {
        let Some(ctx) = ctx() else { return };
        let width = 96usize; // 3 blocks per row on the inner store: not a power of two, on purpose
        let rows = 2 * GROUP; // two full groups, no staged tail (the tail is f32 both ways)
        let v: Vec<f32> = (0..rows * width).map(|i| {
            let (r, c) = (i / width, i % width);
            let base = (r as f32 * 0.13 + c as f32 * 1.9).sin();
            if c % 16 == 5 { base * 25.0 } else { base }
        }).collect();
        let src = Tensor::from_vec(&ctx, &v, &[rows, width]);

        for fmt in KvqFmt::ALL {
            let mut g = GroupedKvCache::new(fmt);
            g.append(&ctx, &src);
            let fused = pollster::block_on(g.dequantize(&ctx).to_vec());

            // The oracle, reconstructed: [ngroups*width, GROUP] -> permute -> [rows, width].
            let ngroups = rows / GROUP;
            let oracle = pollster::block_on(
                g.inner.dequantize(&ctx)
                    .reshape(&[ngroups, width, GROUP])
                    .permute(&[0, 2, 1])
                    .contiguous()
                    .reshape(&[rows, width])
                    .to_vec());
            assert_eq!(fused.len(), oracle.len(), "{}: shape", fmt.name());
            let diff = fused.iter().zip(&oracle)
                .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(diff, 0,
                       "{}: fused dequant differs from the two-pass oracle in {diff} of {} elements — \
                        the transpose-in-index-math is wrong somewhere", fmt.name(), fused.len());
        }
    }

    #[test]
    fn gpu_quantize_matches_the_exact_codes() {
        let Some(ctx) = ctx() else { return };
        for fmt in KvqFmt::ALL {
            let (rows, width) = (7usize, 96usize);
            let (x, want) = exact_case(rows, width, fmt, 7);
            let t = Tensor::from_vec(&ctx, &x, &[rows, width]);
            let q = QKvCache::from_tensor(&ctx, &t, fmt);
            let (codes, scales) = pollster::block_on(q.to_host(&ctx));
            let (rc, rs) = reference::quantize(&x, rows, width, fmt);
            for b in 0..rows * (width / QK) {
                assert_eq!(codes_of(&codes, b, fmt), want[b], "{}: GPU block {b} codes wrong", fmt.name());
            }
            first_diff(&codes, &rc).map(|(i, a, b)| panic!(
                "{}: GPU code word {i} is {a:#010x}, CPU reference says {b:#010x}", fmt.name(), ));
            first_diff(&scales, &rs).map(|(i, a, b)| panic!(
                "{}: GPU scale word {i} is {a:#010x}, CPU reference says {b:#010x}", fmt.name(), ));
        }
    }

    /// K and V may be different KINDS, and each must behave as its own kind.
    ///
    /// This is the pairing the measurement asks for — grouped K, per-block V — and the thing worth
    /// pinning is that configuring one side does not change the other. A single shared "format" field
    /// would make that impossible to express, and the failure would be invisible: both sides would
    /// still quantize, still reconstruct, and just leave K's outlier channels crushed.
    #[test]
    fn k_and_v_can_use_different_layouts_independently() {
        let Some(ctx) = ctx() else { return };
        let width = 64usize;
        let rows = 40usize; // one full group plus a staged tail, so the two kinds differ observably
        let mk = |scale_outliers: bool| {
            let v: Vec<f32> = (0..rows * width).map(|i| {
                let (r, c) = (i / width, i % width);
                let base = (r as f32 * 0.31 + c as f32 * 1.7).sin();
                if scale_outliers && c % 16 == 3 { base * 30.0 } else { base }
            }).collect();
            Tensor::from_vec(&ctx, &v, &[rows, width])
        };

        let (mut k, mut v) = (KvStore::grouped(KvqFmt::Q4_0), KvStore::block(KvqFmt::Q4_0));
        assert!(k.is_grouped() && !v.is_grouped(), "the pair must be able to differ in kind");
        k.append(&ctx, &mk(true));
        v.append(&ctx, &mk(false));
        assert_eq!((k.len(), v.len()), (rows, rows), "both sides hold every row");

        // The grouped side carries an f32 tail; the block side does not. That is the observable
        // difference between the kinds, and it must survive being put behind one enum.
        assert!(k.bytes() > 0 && v.bytes() > 0);
        let k_tail = rows % GROUP;
        assert!(k_tail > 0, "fixture must leave a staged tail or this asserts nothing");

        // And each reconstructs its own data.
        for (name, store, src) in [("K grouped", &k, mk(true)), ("V block", &v, mk(false))] {
            let want = pollster::block_on(src.to_vec());
            let got = pollster::block_on(store.dequantize(&ctx).to_vec());
            assert_eq!(got.len(), want.len(), "{name}: shape");
            let se: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
            let rf: f32 = want.iter().map(|x| x * x).sum();
            assert!((se / rf).sqrt() < 0.25, "{name}: relative RMSE {:.4}", (se / rf).sqrt());
        }

        // Prefix reuse is available on the block kind and REFUSED on the grouped one, rather than
        // silently rounding down to a group boundary and returning fewer tokens than asked for.
        assert!(v.clone_prefix(&ctx, 8).is_some(), "the block kind supports prefix reuse");
        assert!(k.clone_prefix(&ctx, 8).is_none(),
                "the grouped kind must REFUSE a mid-group prefix, not silently truncate it");
    }

    #[test]
    fn gpu_dequantize_matches_the_cpu_reference_bit_for_bit() {
        let Some(ctx) = ctx() else { return };
        let mut r = Lcg::new(99);
        for fmt in KvqFmt::ALL {
            let (rows, width) = (11usize, 128usize);
            // Real-ish spread: a heavy-tailed row scale so blocks differ by orders of magnitude.
            let x: Vec<f32> = (0..rows * width)
                .map(|i| r.next_f32() * 10f32.powi((i / width) as i32 % 3 - 1))
                .collect();
            let t = Tensor::from_vec(&ctx, &x, &[rows, width]);
            let q = QKvCache::from_tensor(&ctx, &t, fmt);
            let (codes, scales) = pollster::block_on(q.to_host(&ctx));
            // Dequantize on the GPU from the GPU's own codes, and on the CPU from the same codes:
            // this isolates the dequant kernel from the quantize kernel.
            let gpu = pollster::block_on(q.dequantize(&ctx).to_vec());
            let cpu = reference::dequantize_rows(&codes, &scales, 0, rows, width, fmt);
            for i in 0..gpu.len() {
                assert_eq!(
                    gpu[i].to_bits(),
                    cpu[i].to_bits(),
                    "{}: element {i} GPU {} vs CPU {}",
                    fmt.name(),
                    gpu[i],
                    cpu[i]
                );
            }
        }
    }

    /// GPU quantize must produce the CPU reference's exact words on data with the shape real K/V
    /// has — heavy-tailed, outlier channels, per-row scale spread.
    ///
    /// Exact equality is a stronger claim than it looks: `id = 1/d` is an f32 divide, and WGSL only
    /// promises 2.5 ULP there against Rust's correctly-rounded one, so a value sitting within ~1e-5
    /// of a `.5` code boundary could legitimately land on either side. `examples/kv_quant_error.rs`
    /// checks the same thing on **captured** K/V — 0 differing words out of 504,832 on Llama-3.2-1B
    /// and 187,680 on Qwen2.5-0.5B, all three formats — so the strict form is what this asserts. A
    /// failure here on some other fabric is a real finding about that fabric's divide, not a flake to
    /// be relaxed away: report it, do not widen it.
    /// The grouped cache must reconstruct the SAME rows in the SAME order, however they arrive.
    ///
    /// The permute is where this can go wrong quietly: a group is stored transposed, so an incorrect
    /// `[ngroups, width, GROUP] -> [ngroups, GROUP, width]` mapping scrambles rows against channels
    /// and still returns a tensor of the right shape and plausible magnitude. The input below is
    /// deliberately NOT smooth across either axis — `sin(r*7.1 + c*0.9)` differs sharply between
    /// neighbouring rows and channels — so a transposed or shifted reconstruction cannot look right.
    #[test]
    fn the_grouped_cache_round_trips_rows_in_order_however_they_are_appended() {
        let Some(ctx) = ctx() else { return };
        let width = 64usize;
        let mk = |r0: usize, n: usize| {
            let v: Vec<f32> = (0..n * width)
                .map(|i| { let (r, c) = (r0 + i / width, i % width); (r as f32 * 7.1 + c as f32 * 0.9).sin() })
                .collect();
            Tensor::from_vec(&ctx, &v, &[n, width])
        };
        let rows = 70usize; // 2 full groups + 6 staged
        let want = pollster::block_on(mk(0, rows).to_vec());

        // Three arrival patterns that must agree: one shot, one row at a time, and a ragged mix that
        // straddles the group boundary.
        let mut one = GroupedKvCache::new(KvqFmt::Q8_0);
        one.append(&ctx, &mk(0, rows));

        let mut drip = GroupedKvCache::new(KvqFmt::Q8_0);
        for r in 0..rows { drip.append(&ctx, &mk(r, 1)); }

        let mut ragged = GroupedKvCache::new(KvqFmt::Q8_0);
        let mut at = 0usize;
        for chunk in [5usize, 30, 1, 20, 14] { ragged.append(&ctx, &mk(at, chunk)); at += chunk; }
        assert_eq!(at, rows, "the ragged schedule must cover every row");

        for (name, c) in [("one-shot", &one), ("one-per-step", &drip), ("ragged", &ragged)] {
            assert_eq!(c.len(), rows, "{name}: row count");
            let got = pollster::block_on(c.dequantize(&ctx).to_vec());
            assert_eq!(got.len(), rows * width, "{name}: shape");
            let rms: f32 = (got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                            / got.len() as f32).sqrt();
            let ref_rms: f32 = (want.iter().map(|x| x * x).sum::<f32>() / want.len() as f32).sqrt();
            assert!(rms / ref_rms < 0.02,
                    "{name}: relative RMSE {:.4} — q8_0 should be ~0.005; a scrambled permute lands \
                     near 1.0 and a shifted one near 1.4", rms / ref_rms);
            // Row-level check: a permute that transposes rows against channels would leave row 0
            // resembling channel 0. Compare each row to its OWN source row specifically.
            for r in [0usize, 31, 32, 63, 64, 69] {
                let d: f32 = (0..width).map(|c2| (got[r * width + c2] - want[r * width + c2]).abs())
                    .fold(0.0, f32::max);
                assert!(d < 0.05, "{name}: row {r} differs from its source by {d:e} — rows and \
                                   channels are crossed");
            }
        }
    }

    /// The payoff: grouping along TOKENS beats grouping along CHANNELS on data with outlier channels.
    ///
    /// Synthesises the structure that makes real K hard — most channels small, a few 30x larger — and
    /// checks the grouped cache actually reconstructs it better than the shipped per-block one. If
    /// this ever fails, the axis change is not buying what §M claims and the claim should go, not the
    /// threshold.
    #[test]
    fn grouping_along_tokens_beats_grouping_along_channels_on_outlier_data() {
        let Some(ctx) = ctx() else { return };
        let (width, rows) = (64usize, 64usize);
        // 4 of 64 channels are ~30x the rest — the outlier-channel structure K actually has.
        let v: Vec<f32> = (0..rows * width)
            .map(|i| {
                let (r, c) = (i / width, i % width);
                let base = (r as f32 * 0.31 + c as f32 * 1.7).sin();
                if c % 16 == 3 { base * 30.0 } else { base }
            })
            .collect();
        let src = Tensor::from_vec(&ctx, &v, &[rows, width]);

        let mut per_block = QKvCache::new(KvqFmt::Q4_0);
        per_block.append(&ctx, &src);
        let mut grouped = GroupedKvCache::new(KvqFmt::Q4_0);
        grouped.append(&ctx, &src);

        let rel = |got: Vec<f32>| -> f32 {
            let se: f32 = got.iter().zip(&v).map(|(a, b)| (a - b) * (a - b)).sum();
            let rf: f32 = v.iter().map(|x| x * x).sum();
            (se / rf).sqrt()
        };
        let e_block = rel(pollster::block_on(per_block.dequantize(&ctx).to_vec()));
        let e_group = rel(pollster::block_on(grouped.dequantize(&ctx).to_vec()));
        eprintln!("4-bit relative RMSE — per-block(32) along row: {e_block:.5}, per-channel x {GROUP} tokens: {e_group:.5}");
        assert!(e_group < e_block * 0.6,
                "grouping along tokens gave {e_group:.5} against per-block's {e_block:.5} — the axis \
                 change is supposed to be a large win on outlier-channel data, not a wash");
    }

    #[test]
    fn gpu_quantize_matches_the_cpu_reference_on_heavy_tailed_data() {
        let Some(ctx) = ctx() else { return };
        let mut r = Lcg::new(0xBEEF);
        let (rows, width) = (64usize, 128usize);
        // one dominant channel, per-row scale spread over three decades, heavy tails via a cube
        let x: Vec<f32> = (0..rows * width)
            .map(|i| {
                let c = i % width;
                let u = r.next_f32();
                let heavy = u * u * u;
                heavy * 10f32.powi((i / width) as i32 % 3 - 1) * if c == 7 { 40.0 } else { 1.0 }
            })
            .collect();
        let t = Tensor::from_vec(&ctx, &x, &[rows, width]);
        for fmt in KvqFmt::ALL {
            let q = QKvCache::from_tensor(&ctx, &t, fmt);
            let (gc, gs) = pollster::block_on(q.to_host(&ctx));
            let (rc, rs) = reference::quantize(&x, rows, width, fmt);
            first_diff(&gc, &rc).map(|(i, a, b)| panic!(
                "{}: GPU code word {i} is {a:#010x}, reference says {b:#010x} ({} of {} differ)",
                fmt.name(), gc.iter().zip(&rc).filter(|(x, y)| x != y).count(), gc.len()));
            first_diff(&gs, &rs).map(|(i, a, b)| panic!(
                "{}: GPU scale word {i} is {a:#010x}, reference says {b:#010x} ({} of {} differ)",
                fmt.name(), gs.iter().zip(&rs).filter(|(x, y)| x != y).count(), gs.len()));
        }
    }

    /// The decode case: append one row at a time and require the result to be identical, word for
    /// word, to quantizing the whole slab at once.
    ///
    /// This is the test that a wrong block offset, a wrong `row0`, or a scale word clobbered across
    /// an append boundary cannot pass. `width = 96` makes `nblk` odd on purpose, so consecutive rows
    /// SHARE a scale word and the boundary case is exercised on every single append.
    /// `clone_prefix` must copy exactly `rows`, own its memory, and leave the source alone.
    ///
    /// The interesting row counts are the ODD ones. Q8_0 and Q4_0 pack two block scales per u32, so an
    /// odd block count shares its final word with a block that is not being copied — get that rounding
    /// wrong and the last row of the prefix reconstructs with the wrong scale, which is a plausible
    /// value rather than an error.
    #[test]
    fn clone_prefix_copies_exactly_that_many_rows_and_shares_nothing() {
        let Some(ctx) = ctx() else { return };
        // ⚠ width MUST give an ODD number of blocks per row, or the scale-word rounding this test
        // exists to check is never exercised. At width 64 there are 2 blocks per row, so every row
        // count yields an EVEN block count, the round-up is always a no-op, and rounding DOWN passes —
        // which is exactly what happened to the first version of this test. width 32 is 1 block per
        // row, so an odd `take` gives an odd block count and the shared final scale word is real.
        let width = QK; // 1 block per row
        assert_eq!(width / QK, 1, "this test needs an odd blocks-per-row to be meaningful");
        let rows = |n: usize, off: f32| {
            let v: Vec<f32> = (0..n * width).map(|i| ((i as f32) * 0.023 + off).sin()).collect();
            Tensor::from_vec(&ctx, &v, &[n, width])
        };
        for fmt in KvqFmt::ALL {
            let mut src = QKvCache::new(fmt);
            src.append(&ctx, &rows(10, 1.0));
            let full = pollster::block_on(src.dequantize(&ctx).to_vec());

            for take in [1usize, 3, 7, 10] {
                let mut p = src.clone_prefix(&ctx, take);
                assert_eq!(p.len(), take, "{}: clone_prefix({take}) kept {} rows", fmt.name(), p.len());
                let got = pollster::block_on(p.dequantize(&ctx).to_vec());
                assert_eq!(got.len(), take * width, "{}: shape", fmt.name());
                let d = got.iter().zip(&full[..take * width])
                    .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert_eq!(d, 0.0,
                           "{}: prefix of {take} rows differs from the source's first {take} rows by \
                            {d:e} — an odd block count's shared scale word is the usual cause",
                           fmt.name());

                // Independence, in the direction that matters: the fork continues, the source must not.
                p.append(&ctx, &rows(4, 90.0));
                let after = pollster::block_on(src.dequantize(&ctx).to_vec());
                assert_eq!(src.len(), 10, "{}: the source's length moved", fmt.name());
                let d2 = after.iter().zip(&full).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert_eq!(d2, 0.0,
                           "{}: appending to the prefix copy changed the SOURCE by {d2:e} — they are \
                            sharing a buffer", fmt.name());
            }

            // Asking for more than exists clamps rather than reading past the end.
            assert_eq!(src.clone_prefix(&ctx, 999).len(), 10, "{}: over-long request clamps", fmt.name());
            assert_eq!(src.clone_prefix(&ctx, 0).len(), 0, "{}: empty prefix is empty", fmt.name());
        }
    }

    #[test]
    fn one_row_at_a_time_equals_one_shot() {
        let Some(ctx) = ctx() else { return };
        let mut r = Lcg::new(4242);
        for fmt in KvqFmt::ALL {
            let (rows, width) = (37usize, 96usize);
            let x: Vec<f32> = (0..rows * width).map(|_| r.next_f32()).collect();

            let whole = Tensor::from_vec(&ctx, &x, &[rows, width]);
            let one_shot = QKvCache::from_tensor(&ctx, &whole, fmt);
            let (oc, os) = pollster::block_on(one_shot.to_host(&ctx));

            let mut inc = QKvCache::new(fmt);
            for t in 0..rows {
                let row = Tensor::from_vec(&ctx, &x[t * width..(t + 1) * width], &[1, width]);
                inc.append(&ctx, &row);
                assert_eq!(inc.len(), t + 1);
            }
            let (ic, is) = pollster::block_on(inc.to_host(&ctx));

            first_diff(&ic, &oc).map(|(i, a, b)| {
                panic!(
                    "{}: row-at-a-time codes differ from one-shot at word {i} ({a:#010x} vs {b:#010x}); \
                     {} of {} words differ",
                    fmt.name(),
                    ic.iter().zip(&oc).filter(|(x, y)| x != y).count(),
                    ic.len()
                )
            });
            first_diff(&is, &os).map(|(i, a, b)| {
                panic!(
                    "{}: row-at-a-time scales differ from one-shot at word {i} ({a:#010x} vs {b:#010x}); \
                     {} of {} words differ",
                    fmt.name(),
                    is.iter().zip(&os).filter(|(x, y)| x != y).count(),
                    is.len()
                )
            });

            let a = pollster::block_on(inc.dequantize(&ctx).to_vec());
            let b = pollster::block_on(one_shot.dequantize(&ctx).to_vec());
            assert_eq!(a, b, "{}: dequantized incremental cache differs", fmt.name());
        }
    }

    /// Growth must be invisible: a cache appended past several doublings holds exactly what a cache
    /// that never grew holds.
    ///
    /// Run at width 64 (`nblk = 2`, every row starts on an even block) **and** width 96 (`nblk = 3`,
    /// so rows share scale words) — the second is the case where an append landing in a freshly
    /// grown buffer has to read back the half-word the growth copy just carried over. All three
    /// formats, because only Q4_1 escapes the shared-word problem.
        /// A `deep_clone` must survive **both** halves continuing independently.
    ///
    /// The obvious test — clone, append to the source, check the copy is unchanged — CANNOT FAIL, and
    /// I shipped it before checking: an append writes at the flat block offset *after* `len`, so it
    /// lands beyond everything the copy reads, and a plain `Arc` handle share passes just as happily
    /// as a real buffer copy. Mutating `deep_clone` to share its source's buffers left that version
    /// green.
    ///
    /// The case that actually distinguishes them is a FORK: clone, then append DIFFERENT rows to each
    /// half. Both write at the same block offset, so shared buffers make the second writer overwrite
    /// the first and both reads return the same wrong thing. That is exactly what prefix caching and
    /// batched decode do — continue two sequences from one prefill — which is why this is the shape
    /// the primitive has to be correct for.
    /// `Clone` must be the deep copy, not a handle share — the same fork test, through the trait.
    ///
    /// Worth its own test because `#[derive(Clone)]` is one keystroke away and would compile, pass
    /// every other test in this file, and reintroduce the aliasing. This is what stops it.
    #[test]
    fn the_clone_impl_is_the_deep_copy_and_not_a_handle_share() {
        let Some(ctx) = ctx() else { return };
        let width = 64usize;
        let rows = |n: usize, off: f32| {
            let v: Vec<f32> = (0..n * width).map(|i| ((i as f32) * 0.017 + off).sin()).collect();
            Tensor::from_vec(&ctx, &v, &[n, width])
        };
        let mut a = QKvCache::new(KvqFmt::Q8_0);
        a.append(&ctx, &rows(8, 1.0));
        let mut b = a.clone();
        a.append(&ctx, &rows(4, 5.0));
        b.append(&ctx, &rows(4, 90.0));
        let va = pollster::block_on(a.dequantize(&ctx).to_vec());
        let vb = pollster::block_on(b.dequantize(&ctx).to_vec());
        let tail = va[8 * width..].iter().zip(&vb[8 * width..])
            .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(tail > 1e-3,
                "Clone produced an aliasing handle: two caches appended DIFFERENT rows read back \
                 identical tails (max delta {tail:e}). Clone must route through deep_clone.");
    }

    /// A cache that was never appended to has no device to copy with, and must still clone.
    #[test]
    fn cloning_an_untouched_cache_needs_no_context() {
        let a = QKvCache::new(KvqFmt::Q4_1);
        let b = a.clone();
        assert_eq!((b.len(), b.width(), b.bytes()), (0, 0, 0));
        assert_eq!(b.fmt(), KvqFmt::Q4_1, "the format survives a contextless clone");
    }

    #[test]
    fn a_deep_clone_and_its_source_can_be_appended_to_independently() {
        let Some(ctx) = ctx() else { return };
        let width = 64usize;
        let rows = |n: usize, off: f32| {
            let v: Vec<f32> = (0..n * width).map(|i| ((i as f32) * 0.031 + off).sin()).collect();
            Tensor::from_vec(&ctx, &v, &[n, width])
        };

        for fmt in KvqFmt::ALL {
            let mut a = QKvCache::new(fmt);
            a.append(&ctx, &rows(8, 1.0));
            let shared_prefix = pollster::block_on(a.dequantize(&ctx).to_vec());

            let mut b = a.deep_clone(&ctx);

            // Diverge: each half continues with its own rows, at the SAME block offset.
            a.append(&ctx, &rows(4, 5.0));
            b.append(&ctx, &rows(4, 90.0));
            assert_eq!((a.len(), b.len()), (12, 12), "{}: both halves grew", fmt.name());

            let va = pollster::block_on(a.dequantize(&ctx).to_vec());
            let vb = pollster::block_on(b.dequantize(&ctx).to_vec());

            // The shared prefix is intact in both.
            for (name, v) in [("source", &va), ("clone", &vb)] {
                let d = v[..8 * width].iter().zip(&shared_prefix)
                    .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                assert_eq!(d, 0.0, "{}: the fork disturbed the shared prefix in the {name} by {d:e}",
                           fmt.name());
            }

            // And the tails are DIFFERENT, which is the assertion a shared buffer fails: with one
            // allocation the second writer wins and both tails read identical.
            let tail_delta = va[8 * width..].iter().zip(&vb[8 * width..])
                .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(tail_delta > 1e-3,
                    "{}: the two halves produced IDENTICAL tails (max delta {tail_delta:e}) after \
                     being appended different rows — they are sharing one device buffer, so the \
                     second append overwrote the first. deep_clone must own its memory.",
                    fmt.name());
        }
    }

    #[test]
    fn growth_preserves_every_earlier_row() {
        let Some(ctx) = ctx() else { return };
        for width in [64usize, 96] {
            for fmt in KvqFmt::ALL {
                let mut r = Lcg::new(31337 + width as u64);
                let rows = 200usize; // 64 initial capacity -> 64, 128, 256: two doublings
                let x: Vec<f32> = (0..rows * width).map(|_| r.next_f32() * 3.0).collect();

                let mut inc = QKvCache::new(fmt);
                for t in 0..rows {
                    let row = Tensor::from_vec(&ctx, &x[t * width..(t + 1) * width], &[1, width]);
                    inc.append(&ctx, &row);
                }
                let got = pollster::block_on(inc.dequantize(&ctx).to_vec());
                let want = reference::roundtrip(&x, rows, width, fmt);
                assert_eq!(got.len(), want.len());
                let bad = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits());
                if let Some(i) = bad {
                    panic!(
                        "{} width {width}: element {i} (row {}, col {}) lost across a growth: {} vs {}",
                        fmt.name(), i / width, i % width, got[i], want[i]
                    );
                }
            }
        }
    }

    /// The append path must read a **strided view** in place — during decode the K row is a window
    /// onto a larger fused QKV buffer, never a freshly packed tensor. A quantizer that silently read
    /// the wrong stride would still produce well-formed blocks.
    #[test]
    fn append_reads_a_strided_view_correctly() {
        let Some(ctx) = ctx() else { return };
        let mut r = Lcg::new(5150);
        let (rows, width, pad) = (9usize, 64usize, 32usize);
        // A [rows, width+pad] buffer; the cache row is the LAST `width` columns of each row.
        let big: Vec<f32> = (0..rows * (width + pad)).map(|_| r.next_f32()).collect();
        let bt = Tensor::from_vec(&ctx, &big, &[rows, width + pad]);
        let view = bt.narrow(1, pad, width);
        let mut want = vec![0f32; rows * width];
        for t in 0..rows {
            want[t * width..(t + 1) * width]
                .copy_from_slice(&big[t * (width + pad) + pad..t * (width + pad) + pad + width]);
        }
        for fmt in KvqFmt::ALL {
            let q = QKvCache::from_tensor(&ctx, &view, fmt);
            let (codes, scales) = pollster::block_on(q.to_host(&ctx));
            let (rc, rs) = reference::quantize(&want, rows, width, fmt);
            first_diff(&codes, &rc).map(|(i, a, b)| panic!(
                "{}: strided-view append read the wrong elements — code word {i} is {a:#010x}, \
                 packing the same rows by hand gives {b:#010x}", fmt.name()));
            first_diff(&scales, &rs).map(|(i, a, b)| panic!(
                "{}: strided-view append wrote the wrong scales — word {i} is {a:#010x} vs {b:#010x}",
                fmt.name()));
        }
    }

    #[test]
    fn cache_bytes_match_the_advertised_bits_per_value() {
        let Some(ctx) = ctx() else { return };
        let (rows, width) = (64usize, 128usize); // exactly the initial capacity: no slack
        let x = vec![0.25f32; rows * width];
        let t = Tensor::from_vec(&ctx, &x, &[rows, width]);
        for fmt in KvqFmt::ALL {
            let q = QKvCache::from_tensor(&ctx, &t, fmt);
            let bits = q.bytes() as f32 * 8.0 / (rows * width) as f32;
            assert!(
                (bits - fmt.bits_per_value()).abs() < 1e-6,
                "{}: cache is {bits} bits/value, format claims {}",
                fmt.name(),
                fmt.bits_per_value()
            );
            assert!(q.bytes() < q.f32_bytes(), "{}: quantized cache is not smaller", fmt.name());
        }
    }

    // ---- granularity study machinery ----------------------------------------------------------

    #[test]
    fn append_cost_names_the_schemes_that_cannot_decode() {
        assert_eq!(append_cost(GranKind::PerBlock(32)), AppendCost::InPlace);
        assert_eq!(append_cost(GranKind::PerToken), AppendCost::InPlace);
        assert_eq!(append_cost(GranKind::PerChannel), AppendCost::FullRequant);
        assert_eq!(append_cost(GranKind::Tensor), AppendCost::FullRequant);
        assert_eq!(append_cost(GranKind::PerChannelGroup(64)), AppendCost::GroupFlush(64));
    }

    /// The partition each granularity induces, pinned directly.
    ///
    /// Added because an injected fault that coarsened `PerChannel` to one scale per EIGHT channels
    /// went undetected by the ordering test below — a coarser per-channel still beat per-block(32),
    /// so the ordering held while the study silently reported the wrong scheme. The ordering test
    /// checks a consequence; this checks the thing itself.
    #[test]
    fn each_granularity_partitions_exactly_as_named() {
        let (rows, width) = (8usize, 64usize);
        let cases: [(GranKind, usize); 6] = [
            (GranKind::Tensor, 1),
            (GranKind::PerToken, rows),
            (GranKind::PerBlock(32), rows * 2),
            (GranKind::PerBlock(16), rows * 4),
            (GranKind::PerChannel, width),
            (GranKind::PerChannelGroup(4), (rows / 4) * width),
        ];
        for (k, want_groups) in cases {
            let mut seen = std::collections::HashSet::new();
            for r in 0..rows {
                for c in 0..width {
                    let g = group_of(k, r, c, width);
                    assert!(g < n_groups(k, rows, width), "{k:?}: group id {g} >= n_groups");
                    seen.insert(g);
                }
            }
            assert_eq!(seen.len(), want_groups, "{k:?} produced {} distinct groups", seen.len());
            assert_eq!(n_groups(k, rows, width), want_groups, "{k:?}: n_groups disagrees with the partition");
        }
        // Memberships that define each scheme, stated one at a time.
        // per-channel: same column, ANY two rows -> one scale; different columns -> never.
        assert_eq!(group_of(GranKind::PerChannel, 0, 5, width), group_of(GranKind::PerChannel, 7, 5, width));
        assert_ne!(group_of(GranKind::PerChannel, 0, 5, width), group_of(GranKind::PerChannel, 0, 6, width));
        // per-block(32): same row, adjacent columns inside one block -> one scale; across the
        // block boundary -> not; same column different row -> not.
        assert_eq!(group_of(GranKind::PerBlock(32), 3, 0, width), group_of(GranKind::PerBlock(32), 3, 31, width));
        assert_ne!(group_of(GranKind::PerBlock(32), 3, 31, width), group_of(GranKind::PerBlock(32), 3, 32, width));
        assert_ne!(group_of(GranKind::PerBlock(32), 3, 0, width), group_of(GranKind::PerBlock(32), 4, 0, width));
        // per-token: whole row shares one scale.
        assert_eq!(group_of(GranKind::PerToken, 2, 0, width), group_of(GranKind::PerToken, 2, width - 1, width));
        // per-channel x 4 tokens: same column within a 4-row window -> one; across the window -> not.
        let g = GranKind::PerChannelGroup(4);
        assert_eq!(group_of(g, 0, 9, width), group_of(g, 3, 9, width));
        assert_ne!(group_of(g, 3, 9, width), group_of(g, 4, 9, width));
        assert_ne!(group_of(g, 0, 9, width), group_of(g, 0, 10, width));
    }

    /// The granularity ladder must be monotone on data that has the structure it claims to model:
    /// with one outlier channel, a finer scale can only help. Built so the ordering is forced.
    #[test]
    fn finer_granularity_reduces_error_on_outlier_channel_data() {
        let (rows, width) = (64usize, 128usize);
        let mut r = Lcg::new(2718);
        let mut x = vec![0f32; rows * width];
        for t in 0..rows {
            for c in 0..width {
                // channel 5 is 50x the others — the "K has outlier channels" structure
                let s = if c == 5 { 50.0 } else { 1.0 };
                x[t * width + c] = r.next_f32() * s;
            }
        }
        let e = |k: GranKind| roundtrip_err(&x, rows, width, k, 4, false).rel_rmse;
        let (tensor, token, block, chan) =
            (e(GranKind::Tensor), e(GranKind::PerToken), e(GranKind::PerBlock(32)), e(GranKind::PerChannel));
        assert!(token <= tensor * 1.001, "per-token {token} should not be worse than per-tensor {tensor}");
        assert!(block < token, "per-block(32) {block} should beat per-token {token} when one channel is an outlier");
        assert!(chan < block, "per-channel {chan} should beat per-block(32) {block} on this data");
    }

    #[test]
    fn shipped_formats_order_by_bit_width_on_real_shaped_data() {
        let (rows, width) = (48usize, 128usize);
        let mut r = Lcg::new(1618);
        let x: Vec<f32> = (0..rows * width).map(|_| r.next_f32() * 2.0).collect();
        let q8 = shipped_err(&x, rows, width, KvqFmt::Q8_0);
        let q40 = shipped_err(&x, rows, width, KvqFmt::Q4_0);
        let q41 = shipped_err(&x, rows, width, KvqFmt::Q4_1);
        assert!(q8.rel_rmse < q41.rel_rmse, "q8_0 {} should beat q4_1 {}", q8.rel_rmse, q41.rel_rmse);
        assert!(q41.rel_rmse < q40.rel_rmse, "q4_1 {} should beat q4_0 {}", q41.rel_rmse, q40.rel_rmse);
        // and the error must actually be small enough to be worth shipping at all
        assert!(q8.rel_rmse < 0.01, "q8_0 rel_rmse {} is too large to be a q8_0", q8.rel_rmse);
    }
}
