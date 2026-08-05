//! **Streaming inference in the browser** — the same tier, the same reader, the same policy as native.
//!
//! ## The one adaptation a browser forces
//!
//! `Backing::read_at` is synchronous; `fetch` is not; a wasm main thread cannot block on a promise. So
//! bytes are **staged** before the forward pass reads them, which the fixed layer walk (0, 1, … N−1,
//! repeat) makes exact rather than speculative. A read of un-staged bytes is a named error
//! (`TierError::NotStaged`), never zeros — a model that runs and lies is the failure this avoids.
//!
//! `stream_embodiments` verifies this produces **byte-identical logits** to a file handle.
//!
//! ## The app supplies the bytes
//!
//! [`FerricStream`] takes a JS function `(offset, length) => Promise<Uint8Array>`. That is deliberately
//! more general than baking in `fetch`, and it means no `web-sys` dependency:
//!
//! - HTTP **Range** against a static host or CDN — the common case;
//! - the **Cache API** or a service worker, so a reload costs nothing;
//! - **OPFS** / IndexedDB for a model already on the device;
//! - anything needing auth, a signed URL, or a proxy.
//!
//! ```js
//! const get = (off, len) =>
//!   fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` } })
//!     .then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
//!
//! const s = await FerricStream.open(get, totalBytes, 64 * 1024 * 1024);
//! console.log(s.plan());                       // what will be pinned vs streamed
//! console.log(await s.generate("Hello", 16));
//! ```

use ferric_gguf::backed::{header_probe, GgufBacked};
use ferric_gguf::{GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::stream::layer_runs_of;
use ferric_tier::{plan_layers, Backing, LayerDesc, LayerPlan, StagedBacking};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Ask JS for `[offset, offset+len)`.
async fn fetch_range(f: &js_sys::Function, offset: u64, len: usize) -> Result<Vec<u8>, JsValue> {
    let p = f.call2(&JsValue::NULL, &JsValue::from_f64(offset as f64), &JsValue::from_f64(len as f64))?;
    let v = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(p)).await?;
    let arr = js_sys::Uint8Array::new(&v);
    let got = arr.length() as usize;
    if got != len {
        // A host that ignores `Range` answers 200 with the WHOLE body instead of 206 with the slice.
        // Staging that at this offset would corrupt every weight read from it, so it is refused by name.
        return Err(JsValue::from_str(&format!(
            "range fetch returned {got} bytes, expected {len}. Does the host honour Range requests \
             (206 Partial Content)? A 200 with the full body produces exactly this."
        )));
    }
    Ok(arr.to_vec())
}

#[wasm_bindgen]
pub struct FerricStream {
    model: Qwen3,
    staged: Arc<StagedBacking>,
    fetcher: js_sys::Function,
    runs: Vec<LayerDesc>,
    plan: LayerPlan,
    bpe: Bpe,
    total: u64,
    header_len: usize,
    /// Embeddings, norms and the LM head: resident regardless of the budget, and on a small model with a
    /// large vocabulary they DOMINATE. Reported so a peak figure is explicable rather than surprising.
    nonlayer_bytes: u64,
}

#[wasm_bindgen]
impl FerricStream {
    /// Open a checkpoint over a JS byte-fetcher.
    ///
    /// `budget_bytes` bounds **layer weights only**. Embeddings, norms and the LM head stay resident, as
    /// on every production streaming engine and on Ferric's native path: a small share of the parameters,
    /// touched on every token regardless.
    pub async fn open(
        fetcher: js_sys::Function,
        total_bytes: f64,
        budget_bytes: f64,
    ) -> Result<FerricStream, JsValue> {
        console_error_panic_hook::set_once();
        let total = total_bytes as u64;
        let staged = Arc::new(StagedBacking::new());

        // The header carries the tensor table and its length is not recorded in the format, so grow a
        // prefix until it parses. Done here against JS fetches directly, because `header_probe` needs a
        // synchronous backing and this is the one place we are still async.
        let mut n = (1usize << 20).min(total as usize);
        let header = loop {
            let bytes = fetch_range(&fetcher, 0, n).await?;
            if ferric_gguf::parse(bytes.clone()).is_ok() { break bytes; }
            if n >= total as usize { return Err(JsValue::from_str("not a GGUF file")); }
            if n >= (64 << 20) { return Err(JsValue::from_str("no GGUF header in the first 64 MB")); }
            n = (n * 4).min(64 << 20).min(total as usize);
        };
        staged.stage(0, header.clone());
        let header_len = header.len();

        let backing: Arc<dyn Backing + Send + Sync> = staged.clone();
        let src = GgufBacked::new(header.clone(), Arc::clone(&backing))
            .map_err(|e| JsValue::from_str(&e))?;
        let runs = layer_runs_of(&src.tensors, src.data_start()).map_err(|e| JsValue::from_str(&e))?;
        let plan = plan_layers(&runs, budget_bytes as u64, 0, 4096);
        if !plan.fits(budget_bytes as u64) {
            return Err(JsValue::from_str(&format!(
                "budget {:.1} MB cannot hold one streaming slot (needs {:.1} MB)",
                budget_bytes / 1e6, plan.spent as f64 / 1e6)));
        }

        // Stage what stays resident: the non-layer tensors (embeddings, norms, head) plus the pinned
        // prefix. Each is one range request, which is why a contiguous layer run matters.
        //
        // Streamed layers are deliberately NOT staged here — they are fetched per step. An earlier
        // version called Qwen3::load (the RESIDENT loader, which builds every layer) after staging only
        // the prefix, and the browser caught it exactly as designed: `NotStaged` naming the missing
        // range rather than returning zeros.
        let mut nonlayer_bytes = 0u64;
        for t in src.tensors.clone() {
            if t.name.starts_with("blk.") { continue; }
            let (off, sz) = src.extent(&t.name).ok_or_else(|| JsValue::from_str("bad extent"))?;
            nonlayer_bytes += sz as u64;
            if !staged.is_staged(off, sz) {
                let b = fetch_range(&fetcher, off, sz).await?;
                staged.stage(off, b);
            }
        }
        for il in 0..plan.npin {
            let d = runs[il];
            let b = fetch_range(&fetcher, d.offset, d.bytes as usize).await?;
            staged.stage(d.offset, b);
        }

        // Tokenizer out of the metadata already in hand.
        let toks: Vec<String> = match src.metadata().get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(a)) => a.iter()
                .map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
            _ => return Err(JsValue::from_str("checkpoint has no tokenizer.ggml.tokens")),
        };
        let vocab: HashMap<String, u32> =
            toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let merges: Vec<(String, String)> = match src.metadata().get("tokenizer.ggml.merges") {
            Some(Meta::Arr(a)) => a.iter().filter_map(|m| {
                if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }
            }).collect(),
            _ => Vec::new(),
        };
        let bpe = Bpe::new(vocab, &merges);

        let ctx = Arc::new(ferric_core::Context::new().await
            .map_err(|e| JsValue::from_str(&format!("no WebGPU adapter: {e:?}")))?);
        let cfg = ferric_llama::qwen3::Cfg::from_gguf(&src).map_err(|e| JsValue::from_str(&e))?;
        // The streaming path, not the resident one: layers are materialised per visit from staged bytes.
        let src2 = GgufBacked::new(header.clone(), Arc::clone(&backing))
            .map_err(|e| JsValue::from_str(&e))?;
        let ls = ferric_llama::stream::open_from_source(
            &ctx, src2, Arc::clone(&backing), budget_bytes as u64, cfg)
            .map_err(|e| JsValue::from_str(&e))?;
        let model = Qwen3::from_stream(&ctx, &src, ls).map_err(|e| JsValue::from_str(&e))?;

        Ok(FerricStream { model, staged, fetcher, runs, plan, bpe, total, header_len, nonlayer_bytes })
    }

    /// What the budget bought, as JSON — for a caller that wants to show it before generating.
    pub fn plan(&self) -> String {
        format!(
            "{{\"layers\":{},\"pinned\":{},\"hit_rate\":{:.4},\"layer_bytes\":{},\"total_bytes\":{},\
              \"header_bytes\":{},\"nonlayer_bytes\":{},\"resident_bytes\":{},\"peak_bytes\":{},\
               \"fetched_bytes\":{}}}",
            self.runs.len(), self.plan.npin, self.plan.hit_rate(),
            self.runs.iter().map(|r| r.bytes).sum::<u64>(), self.total,
            self.header_len, self.nonlayer_bytes, self.staged.resident_bytes(),
            self.staged.peak_bytes(), self.staged.staged_total()
        )
    }

    pub fn resident_bytes(&self) -> f64 { self.staged.resident_bytes() as f64 }
    /// Peak residency observed — the number a budget claim must be judged against, not the trough.
    pub fn peak_bytes(&self) -> f64 { self.staged.peak_bytes() as f64 }
    /// Total bytes pulled over the wire — the figure that matters on a metered link.
    pub fn fetched_bytes(&self) -> f64 { self.staged.staged_total() as f64 }
    pub fn n_layers(&self) -> usize { self.runs.len() }
    pub fn pinned_layers(&self) -> usize { self.plan.npin }

    /// Stage every streamed layer for the next step, releasing the previous one to stay in budget.
    ///
    /// Pinned layers are staged once at `open` and never released — that is the pinned prefix, and it is
    /// why a cyclic walk gets `npin/n` reuse instead of the zero an LRU would give.
    async fn stage_streamed(&self) -> Result<(), JsValue> {
        for il in self.plan.npin..self.runs.len() {
            let d = self.runs[il];
            if self.staged.is_staged(d.offset, d.bytes as usize) { continue; }
            let b = fetch_range(&self.fetcher, d.offset, d.bytes as usize).await?;
            self.staged.stage(d.offset, b);
        }
        Ok(())
    }

    /// Release every streamed layer, keeping only the pinned prefix resident.
    pub fn release_streamed(&self) {
        for il in self.plan.npin..self.runs.len() {
            self.staged.release(self.runs[il].offset);
        }
    }

    /// Greedy generation, streaming one layer at a time.
    ///
    /// The loop stages layer *il*, applies it, then releases *il−1* — so peak residency is the pinned set
    /// plus a couple of layers rather than the whole model. That is only possible because
    /// [`Qwen3::step_layer`] can be awaited between layers; a monolithic forward cannot pause, and on
    /// wasm there is no thread to block on a read, which is what previously pinned the peak at the entire
    /// weight set regardless of the budget.
    pub async fn generate(&mut self, prompt: &str, n: usize) -> Result<String, JsValue> {
        let mut all = self.bpe.encode(prompt);
        let mut cache = Cache::new(&self.model.cfg);
        let mut fed = 0usize;
        let mut out = String::new();
        let npin = self.plan.npin;

        for _ in 0..n {
            let mut st = self.model.step_begin(&all[fed..], &cache);
            while let Some(il) = st.next_layer() {
                if il >= npin {
                    let d = self.runs[il];
                    if !self.staged.is_staged(d.offset, d.bytes as usize) {
                        let b = fetch_range(&self.fetcher, d.offset, d.bytes as usize).await?;
                        self.staged.stage(d.offset, b);
                    }
                }
                let done = self.model.step_layer(&mut st, &mut cache);
                // The tier copied these bytes into its own slot during the bind, so the staged range can
                // go immediately — this release is what actually bounds the peak.
                if il > npin { self.staged.release(self.runs[il - 1].offset); }
                if done { break; }
            }
            if self.runs.len() > npin { self.staged.release(self.runs[self.runs.len() - 1].offset); }

            let logits = self.model.step_finish(st, &mut cache).to_vec().await;
            fed = all.len();
            let v = self.model.cfg.n_vocab;
            let last = &logits[logits.len() - v..];
            let (mut best, mut bv) = (0u32, f32::MIN);
            for (i, &x) in last.iter().enumerate() { if x > bv { bv = x; best = i as u32; } }
            all.push(best);
            out.push_str(&self.bpe.decode(&[best]));
        }
        Ok(out)
    }
}
