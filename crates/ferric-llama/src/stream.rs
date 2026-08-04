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
//! A streamed layer is rebuilt per visit, which means re-uploading its weights to the GPU every token.
//! That is real work and this path is **slower than resident** — the point is that it runs at all when
//! resident is not an option. The saving is memory; the cost is bandwidth.

use crate::qwen3::{build_layer, Cfg, Layer};
use ferric_core::Context;
use ferric_gguf::{deq_raw, GgufFile, GgufSource, Meta, TensorInfo};
use ferric_tier::{plan_layers, Backing, FileBacking, LayerCache, LayerDesc, LayerPlan};
use std::collections::HashMap;
use std::sync::Arc;

/// A `GgufSource` that serves one layer's tensors from bytes the tier delivered, and everything else
/// from the file.
///
/// The slicing is what makes a layer exactly one read: a layer's tensors form a contiguous run, so their
/// absolute offsets map to `abs - run_start` inside the fetched buffer.
pub struct LayerBytes<'a> {
    inner: &'a GgufFile,
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

/// Owns everything a streamed model needs to rebuild a layer on demand.
pub struct LayerStream {
    file: GgufFile,
    cache: std::cell::RefCell<LayerCache>,
    backing: Arc<FileBacking>,
    plan: LayerPlan,
    ctx: Arc<Context>,
    cfg: Cfg,
    /// Layers rebuilt so far — a counter, not a cache. Reported so a caller can see the cost it is
    /// paying rather than infer it.
    pub rebuilds: std::cell::Cell<u64>,
}

impl LayerStream {
    pub fn plan(&self) -> &LayerPlan { &self.plan }
    pub fn stats(&self) -> ferric_tier::LayerStats { self.cache.borrow().stats() }

    /// Materialise layer `il`: fetch its run through the tier, then build from those bytes.
    pub fn layer(&self, il: usize) -> Result<Layer, String> {
        let mut cache = self.cache.borrow_mut();
        let (bytes, _tier) = cache
            .bind(il as u32, &*self.backing)
            .map_err(|e| format!("tier bind for layer {il}: {e}"))?;
        let run_start = self.run_start(il);
        let src = LayerBytes { inner: &self.file, run_start, bytes };
        self.rebuilds.set(self.rebuilds.get() + 1);
        build_layer(&self.ctx, &src, &self.cfg, il)
    }

    fn run_start(&self, il: usize) -> u64 {
        layer_runs(&self.file).map(|v| v[il].offset).unwrap_or(0)
    }
}

/// Group a checkpoint's tensors into per-layer runs, **verifying each is contiguous**.
///
/// Contiguity is what makes one layer one read. It is a property of how the converter emitted the file,
/// not a guarantee — so it is checked, and a gap is reported with its size rather than silently included
/// in the read (which would inflate every byte figure and quietly pull in other tensors' data).
pub fn layer_runs(g: &GgufFile) -> Result<Vec<LayerDesc>, String> {
    let base = g.data_start();
    let mut per: Vec<Vec<(u64, u64)>> = Vec::new();
    for t in &g.tensors {
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
pub fn open(
    ctx: &Arc<Context>,
    path: &str,
    budget_bytes: u64,
    cfg: Cfg,
) -> Result<LayerStream, String> {
    let file = GgufFile::open(path)?;
    let runs = layer_runs(&file)?;
    let plan = plan_layers(&runs, budget_bytes, 0, 4096);
    if !plan.fits(budget_bytes) {
        return Err(format!(
            "budget {budget_bytes} B cannot hold even one streaming slot (needs {} B)", plan.spent));
    }
    let backing = Arc::new(FileBacking::open(path).map_err(|e| e.to_string())?);
    let mut cache = LayerCache::new(plan.clone(), runs);
    cache.prefill(&*backing).map_err(|e| e.to_string())?;
    Ok(LayerStream {
        file,
        cache: std::cell::RefCell::new(cache),
        backing,
        plan,
        ctx: ctx.clone(),
        cfg,
        rebuilds: std::cell::Cell::new(0),
    })
}
