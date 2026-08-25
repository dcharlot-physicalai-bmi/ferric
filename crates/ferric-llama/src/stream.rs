//! **Dense-path weight streaming**: run a transformer whose layer weights are not all resident.
//!
//! The MoE half of this capability is demonstrated in `examples/moe_streaming.rs`; this is the dense
//! half, and it is the one that lets a model exceed memory outright rather than only its expert set.
//!
//! ## How it works, and why the model owns it
//!
//! Every weight in [`crate::qwen3::build_layer`] is fetched through `GgufSource`. So streaming needs no
//! change to the model's arithmetic — only a source whose bytes come from a [`LayerCache`] instead of
//! from the whole file. [`LayerBytes`] is that source: it serves a layer's tensors by slicing the run the
//! tier just delivered, and delegates everything else to the file.
//!
//! Residency lives inside the model rather than beside it because the model is what knows the access
//! order. That order — layers 0..N, cyclically, once per token — is exactly what makes a *pinned prefix*
//! the right policy and an LRU worthless (see `ferric-tier`), and it is what makes the one-ahead prefetch
//! exact rather than speculative.
//!
//! ## What it costs
//!
//! Measured on Qwen2.5-0.5B, greedy decode, against a fully-resident run. **The `built/reused` counts
//! are exact and reproduce identically every run; the milliseconds do not** — wall clock on this machine
//! carries roughly 20% run-to-run spread, so the ratios are given as observed ranges rather than points.
//!
//! ```text
//!     budget   pinned   built/reused   vs resident (5 runs)
//!   resident    24/24        24/-              1.0x
//!    15.9 MB     0/24       288/0          8.4 - 11.3x
//!    95.1 MB     4/24       244/44         7.0 - 10.1x
//!   190.2 MB    10/24       178/110         5.3 - 7.7x
//! ```
//!
//! **The dominant cost is rebuilding a layer's GPU tensors, not the disk read.** That was not the
//! expected answer. The first version rebuilt every layer on every visit, and its wall clock barely moved
//! as the byte-cache hit rate went 0% → 41.7% — which is what showed I/O was not the bottleneck. Building
//! pinned layers **once** removes 110 of 288 rebuilds at the 190 MB rung, and the timing improved with
//! it; the deterministic evidence is the rebuild count, not the millisecond figure.
//!
//! A corollary worth knowing when reading the numbers: the tier's byte-level hit rate *fell* (41.7% →
//! 5.6%) as a direct result of that speedup, because a pinned layer stops calling the tier once built.
//! The tier now sees only the streamed remainder — a smaller and harder workload. `built/reused` is the
//! honest reuse figure.
//!
//! ## Overlap
//!
//! Reads are issued one layer ahead on a worker thread ([`ferric_tier::PrefetchCache`]). On a warm page
//! cache that is worth little — which is consistent with the rebuild dominating above — but it is not the
//! regime streaming exists for. Measured against a backing with an injected per-read delay
//! (`examples/stream_overlap.rs`), token ids identical throughout:
//!
//! ```text
//!   read delay   0 us -> 1.1-1.2x (noise floor)   2000 us -> 1.53x   8000 us -> 1.50x (saturated)
//! ```
//!
//! The saturation point is where the read exceeds the compute available to hide it behind. Pass
//! `overlap: false` to [`open_with`] to compare.
//!
//! This path remains **slower than resident by design**; the point is that it runs at all when resident
//! is not an option. The saving is memory; the cost is bandwidth.

use crate::qwen3::{build_layer, Cfg, Layer};
use ferric_core::Context;
use ferric_gguf::backed::GgufBacked;
use ferric_gguf::deq_raw;
use ferric_gguf::{GgufFile, GgufSource, Meta, TensorInfo};
use ferric_tier::{plan_layers, Backing, LayerCache, LayerDesc, LayerPlan};
#[cfg(not(target_arch = "wasm32"))]
use ferric_tier::{FileBacking, PrefetchCache};
use std::collections::HashMap;
use std::sync::Arc;

/// A `GgufSource` that serves one layer's tensors from bytes the tier delivered, and everything else
/// from the file.
///
/// The slicing is what makes a layer exactly one read: a layer's tensors form a contiguous run, so their
/// absolute offsets map to `abs - run_start` inside the fetched buffer.
pub struct LayerBytes<'a> {
    inner: &'a GgufBacked,
    run_start: u64,
    bytes: &'a [u8],
}

impl LayerBytes<'_> {
    /// Byte range of `name` inside the fetched run, if it belongs to this layer.
    fn local(&self, name: &str) -> Option<(usize, usize)> {
        let t = self.inner.tensor(name)?;
        let abs = self.inner.data_start() + t.offset;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let sz = ferric_gguf::type_size(t.ggml_type, n).ok()?;
        let off = abs.checked_sub(self.run_start)? as usize;
        (off + sz <= self.bytes.len()).then_some((off, sz))
    }
}

impl GgufSource for LayerBytes<'_> {
    fn metadata(&self) -> &HashMap<String, Meta> { self.inner.metadata() }
    fn tensor(&self, n: &str) -> Option<&TensorInfo> { self.inner.tensor(n) }
    fn raw(&self, n: &str) -> Result<Vec<u8>, String> {
        match self.local(n) {
            Some((o, sz)) => Ok(self.bytes[o..o + sz].to_vec()),
            None => self.inner.raw(n), // not part of this run (embeddings, head, norms outside the layer)
        }
    }
    fn dequant(&self, n: &str) -> Result<Vec<f32>, String> {
        match self.local(n) {
            Some((o, sz)) => {
                let t = self.tensor(n).ok_or("missing tensor")?;
                let count: usize = t.dims.iter().product::<u64>() as usize;
                deq_raw(&self.bytes[o..o + sz], count, t.ggml_type)
            }
            None => self.inner.dequant(n),
        }
    }
}

/// A layer for one forward step: borrowed when pinned, owned when streamed.
///
/// The distinction is the whole optimisation. A *pinned* layer's bytes never leave memory, so rebuilding
/// its GPU tensors on every visit is pure waste — measured, it was the dominant cost, not the I/O: at a
/// budget with a 41.7% byte-cache hit rate the wall clock barely moved, because every layer was rebuilt
/// regardless. Pinned layers are now built **once**.
pub enum LayerRef<'a> {
    Pinned(&'a Layer),
    /// A resident (non-streamed) layer, borrowed from the model.
    Borrowed(&'a Layer),
    Built(Layer),
}

impl std::ops::Deref for LayerRef<'_> {
    type Target = Layer;
    fn deref(&self) -> &Layer {
        match self { LayerRef::Pinned(l) | LayerRef::Borrowed(l) => l, LayerRef::Built(l) => l }
    }
}

/// Owns everything a streamed model needs to rebuild a layer on demand.
/// Which tier the stream binds through.
///
/// Both implement the same pinned-prefix policy; `Overlapped` additionally reads layer *L+1* on a worker
/// thread while the caller is still using *L*. Whether that pays depends entirely on how slow the backing
/// is — see the module docs.
enum Tier2 {
    Sync(LayerCache),
    /// Reads issued one layer ahead on a worker thread. Native only — wasm has no threads, and there the
    /// equivalent is `StagedBacking`: the caller fetches ahead from async code and the synchronous read
    /// finds the bytes already there. Same one-ahead idea, different mechanism for the same reason.
    #[cfg(not(target_arch = "wasm32"))]
    Overlapped(PrefetchCache),
}

pub struct LayerStream {
    src: GgufBacked,
    cache: std::cell::RefCell<Tier2>,
    backing: Arc<dyn Backing + Send + Sync>,
    plan: LayerPlan,
    /// Precomputed layer runs. Recomputing these per bind meant re-walking the whole tensor table on
    /// every layer of every token — an O(tensors) scan on the hot path.
    runs: Vec<LayerDesc>,
    ctx: Arc<Context>,
    cfg: Cfg,
    /// Built layers for the pinned prefix. `OnceCell` rather than `RefCell` because a pinned layer is
    /// written exactly once and read forever — which is also what lets `layer()` hand back a borrow
    /// instead of a clone.
    pinned: Vec<std::cell::OnceCell<Layer>>,
    /// Layers rebuilt so far — a counter, not a cache. Reported so a caller can see the cost it is
    /// paying rather than infer it.
    pub rebuilds: std::cell::Cell<u64>,
    /// Runs where the layer was already built and resident.
    pub reuses: std::cell::Cell<u64>,
}

impl LayerStream {
    pub fn plan(&self) -> &LayerPlan { &self.plan }
    /// Byte-cache hit rate. Note this counts only binds that reach the tier — a pinned layer stops
    /// calling it once built, so this measures the streamed remainder, not overall reuse.
    pub fn hit_rate(&self) -> f64 {
        match &*self.cache.borrow() {
            Tier2::Sync(c) => c.stats().hit_rate(),
            #[cfg(not(target_arch = "wasm32"))]
            Tier2::Overlapped(c) => c.stats().overlap_rate(),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn overlapped(&self) -> bool { matches!(&*self.cache.borrow(), Tier2::Overlapped(_)) }

    /// Materialise layer `il`.
    ///
    /// A pinned layer is built on first touch and then simply borrowed. A streamed one is fetched through
    /// the tier and rebuilt, and dropped by the caller at the end of the step — that drop is the eviction.
    pub fn layer(&self, il: usize) -> Result<LayerRef<'_>, String> {
        if il < self.plan.npin {
            if let Some(l) = self.pinned[il].get() {
                self.reuses.set(self.reuses.get() + 1);
                return Ok(LayerRef::Pinned(l));
            }
            let built = self.build(il)?;
            let _ = self.pinned[il].set(built);
            return Ok(LayerRef::Pinned(self.pinned[il].get().expect("just set")));
        }
        Ok(LayerRef::Built(self.build(il)?))
    }

    fn build(&self, il: usize) -> Result<Layer, String> {
        let mut cache = self.cache.borrow_mut();
        let bytes = match &mut *cache {
            Tier2::Sync(c) => c.bind(il as u32, &*self.backing).map(|(b, _)| b),
            #[cfg(not(target_arch = "wasm32"))]
            Tier2::Overlapped(c) => c.bind(il as u32).map(|(b, _)| b),
        }
        .map_err(|e| format!("tier bind for layer {il}: {e}"))?;
        let src = LayerBytes { inner: &self.src, run_start: self.runs[il].offset, bytes };
        self.rebuilds.set(self.rebuilds.get() + 1);
        build_layer(&self.ctx, &src, &self.cfg, il)
    }
}

/// Build a stream from an already-open reader — the constructor a browser uses.
///
/// Takes the pieces rather than a path so wasm never touches the filesystem, and so both embodiments
/// derive the identical plan from the identical inputs.
pub fn open_from_source(
    ctx: &Arc<Context>,
    src: GgufBacked,
    backing: Arc<dyn Backing + Send + Sync>,
    budget_bytes: u64,
    cfg: Cfg,
) -> Result<LayerStream, String> {
    let runs = layer_runs_of(&src.tensors, src.data_start())?;
    let plan = plan_layers(&runs, budget_bytes, 0, 4096);
    if !plan.fits(budget_bytes) {
        return Err(format!(
            "budget {budget_bytes} B cannot hold one streaming slot (needs {} B)", plan.spent));
    }
    let mut c = LayerCache::new(plan.clone(), runs.clone());
    c.prefill(&*backing).map_err(|e| e.to_string())?;
    let n_layer = runs.len();
    Ok(LayerStream {
        src,
        cache: std::cell::RefCell::new(Tier2::Sync(c)),
        backing,
        plan,
        runs,
        ctx: ctx.clone(),
        cfg,
        pinned: (0..n_layer).map(|_| std::cell::OnceCell::new()).collect(),
        rebuilds: std::cell::Cell::new(0),
        reuses: std::cell::Cell::new(0),
    })
}

/// Group a checkpoint's tensors into per-layer runs, **verifying each is contiguous**.
///
/// Contiguity is what makes one layer one read. It is a property of how the converter emitted the file,
/// not a guarantee — so it is checked, and a gap is reported with its size rather than silently included
/// in the read (which would inflate every byte figure and quietly pull in other tensors' data).
#[cfg(not(target_arch = "wasm32"))]
pub fn layer_runs(g: &GgufFile) -> Result<Vec<LayerDesc>, String> {
    // Streaming reads ONE backing object positionally, so `data_start() + offset` must be an absolute
    // file position. In a sharded checkpoint each tensor's offset is relative to the data section of
    // whichever part holds it, so that arithmetic silently addresses the wrong bytes — the right
    // COUNT of bytes, from the wrong place, which produces fluent garbage rather than an error.
    // Streaming across parts is a different feature; until it exists this refuses.
    if g.shard_count() > 1 {
        return Err(format!(
            "this checkpoint is {} shards and the streaming reader addresses one file positionally;              merge it (llama-gguf-split --merge) or load it whole", g.shard_count()));
    }
    layer_runs_of(&g.tensors, g.data_start())
}

/// Layer runs from a tensor table — the embodiment-independent form.
///
/// Takes the pieces rather than a reader, so the browser (`GgufBacked` over staged fetches) and the
/// native path (`GgufFile`) compute the identical plan from the identical inputs. A second
/// implementation for wasm would be a second place for this to drift.
pub fn layer_runs_of(tensors: &[TensorInfo], data_start: u64) -> Result<Vec<LayerDesc>, String> {
    let base = data_start;
    let mut per: Vec<Vec<(u64, u64)>> = Vec::new();
    for t in tensors {
        let Some(rest) = t.name.strip_prefix("blk.") else { continue };
        let Some((idx, _)) = rest.split_once('.') else { continue };
        let Ok(il) = idx.parse::<usize>() else { continue };
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let sz = ferric_gguf::type_size(t.ggml_type, n)? as u64;
        if per.len() <= il { per.resize(il + 1, Vec::new()); }
        per[il].push((base + t.offset, sz));
    }
    if per.is_empty() { return Err("no blk.N.* tensors — not a transformer GGUF".into()); }
    let mut out = Vec::with_capacity(per.len());
    for (il, mut v) in per.into_iter().enumerate() {
        if v.is_empty() { return Err(format!("layer {il} has no tensors")); }
        v.sort_unstable();
        let lo = v[0].0;
        let hi = v.iter().map(|(o, s)| o + s).max().unwrap();
        let own: u64 = v.iter().map(|(_, s)| *s).sum();
        if hi - lo != own {
            return Err(format!(
                "layer {il} is not one contiguous run: spans {} bytes but owns {own} \
                 ({} bytes belong to other tensors)", hi - lo, hi - lo - own));
        }
        out.push(LayerDesc { offset: lo, bytes: own });
    }
    Ok(out)
}

/// Build a [`LayerStream`] for `path` under a byte budget for layer weights.
#[cfg(not(target_arch = "wasm32"))]
pub fn open(ctx: &Arc<Context>, path: &str, budget_bytes: u64, cfg: Cfg) -> Result<LayerStream, String> {
    let backing = Arc::new(FileBacking::open(path).map_err(|e| e.to_string())?);
    open_with(ctx, path, backing, budget_bytes, cfg, true)
}

/// Build a stream over a caller-supplied backing.
///
/// The injection point exists so the tier can be measured against a *slow* device without one — on a
/// local SSD with a warm page cache the read is not the bottleneck, which makes it impossible to tell
/// from timings alone whether the overlap is doing anything.
pub fn open_with(
    ctx: &Arc<Context>,
    path: &str,
    backing: Arc<dyn Backing + Send + Sync>,
    budget_bytes: u64,
    cfg: Cfg,
    overlap: bool,
) -> Result<LayerStream, String> {
    let file = GgufFile::open(path)?;
    let (header, _) = ferric_gguf::backed::header_probe(
        &*backing, u64::MAX, 1 << 20, 64 << 20)
        .or_else(|_| std::fs::read(path).map(|b| { let n = b.len(); (b, n) }).map_err(|e| e.to_string()))?;
    let src = GgufBacked::new(header, Arc::clone(&backing))?;
    // Same reason as `layer_runs`: one `Backing` cannot address a multi-part checkpoint.
    if file.shard_count() > 1 {
        return Err(format!(
            "this checkpoint is {} shards and the streaming reader addresses one file positionally;              merge it (llama-gguf-split --merge) or load it whole", file.shard_count()));
    }
    let runs = layer_runs_of(&file.tensors, file.data_start())?;
    let plan = plan_layers(&runs, budget_bytes, 0, 4096);
    if !plan.fits(budget_bytes) {
        return Err(format!(
            "budget {budget_bytes} B cannot hold even one streaming slot (needs {} B)", plan.spent));
    }
    #[cfg(not(target_arch = "wasm32"))]
    let cache = if overlap {
        let mut c = PrefetchCache::new(plan.clone(), runs.clone(), Arc::clone(&backing))
            .map_err(|e| e.to_string())?;
        c.prefill().map_err(|e| e.to_string())?;
        Tier2::Overlapped(c)
    } else {
        let mut c = LayerCache::new(plan.clone(), runs.clone());
        c.prefill(&*backing).map_err(|e| e.to_string())?;
        Tier2::Sync(c)
    };
    // wasm has no threads: the overlap there is StagedBacking, driven by the caller from async code.
    #[cfg(target_arch = "wasm32")]
    let cache = {
        let _ = overlap;
        let mut c = LayerCache::new(plan.clone(), runs.clone());
        c.prefill(&*backing).map_err(|e| e.to_string())?;
        Tier2::Sync(c)
    };
    let n_layer = runs.len();
    Ok(LayerStream {
        src,
        cache: std::cell::RefCell::new(cache),
        backing,
        plan,
        runs,
        ctx: ctx.clone(),
        cfg,
        pinned: (0..n_layer).map(|_| std::cell::OnceCell::new()).collect(),
        rebuilds: std::cell::Cell::new(0),
        reuses: std::cell::Cell::new(0),
    })
}
