//! **Dense Qwen-family transformer** — PrismML's Ternary Bonsai (1.7B/4B, arch `qwen3`) *and* standard
//! `qwen3` **and `qwen2`** GGUFs off Hugging Face. GQA every layer, SwiGLU, RoPE, RMSNorm; the arch
//! differences are handled by feature-detection: QK-norm (Qwen3 only) and QKV bias (Qwen2 only) are
//! read from tensor presence, and all metadata keys are architecture-prefixed. **Format-agnostic**: each weight
//! loads in whatever quant the GGUF stored it (`QMatrix` over Q2_0/Q4_0/Q4_K/Q6_K/Q8_0 natively, plus a
//! dequant-to-f32 dense fallback for IQ4_XS/IQ4_NL and other kernel-less types), so this runs
//! a PrismML ternary model *and* a genuine `Q4_K_M` model off Hugging Face — which mixes Q4_K and
//! Q6_K, even within one qkv (see `Proj`). The ternary 1.7B is ~450 MB packed, so it fits WebGPU's
//! memory limits and this same code compiles to wasm32 to drive a browser tab.
//!
//! Projection-fusion + KV-cache tricks proven on the 27B: q/k/v fuse into one matmul (when they share
//! a format), gate/up into another, attention resumes from cached K/V so decode is one step per token.
use crate::qwen35::{f32t, qm, qm_cat};
use ferric_core::Context;
use ferric_gguf::{deq_raw, GgufSource, Meta};
use ferric_tensor::{nn, KvBuf, QMatrix, Tensor};
use std::sync::Arc;


/// Env-gated dump matching `llama-eval-callback`'s tensor names, for bisecting a divergence against
/// the reference one tensor at a time. `FERRIC_DUMP=<block index>`.
pub(crate) fn dump(tag: &str, il: usize, t: &Tensor) {
    let Ok(want) = std::env::var("FERRIC_DUMP") else { return };
    if want.parse::<usize>().ok() != Some(il) { return }
    let v = pollster::block_on(t.to_vec());
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    println!("  [{il}] {tag:<12} {:?} n={} sum {sum:+.6}", t.shape, v.len());
}


/// Whether `arch` uses NORM (interleaved, partners `2c`/`2c+1`) rather than NEOX (split-half) rope.
///
/// This loader defaults to NEOX because it was written around the Qwen family, so every NORM
/// architecture it picks up rotates the wrong partners until it is named here. That is how `llama`
/// held a `verified` badge while answering "The capital of France is located in the United States" —
/// beating " Paris" by 0.04 logits, near enough to read as working.
///
/// Audited against llama.cpp's `llama_model_rope_type` switch, 2026-08-15.
pub(crate) fn rope_is_interleaved(arch: &str) -> bool {
    matches!(arch, "muse-glimmer" | "llama")
}

pub struct Cfg {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub has_qk_norm: bool, // Qwen3 has it, Qwen2/Llama don't
    pub qkv_bias: bool,    // Qwen2 has q/k/v biases; Qwen3/Llama don't
    pub is_gemma: bool,    // Gemma: (1+w) norms, √d embed scale, post-attn/post-ffn norms, gelu, per-layer rope
    pub embd_scale: f32,   // Gemma scales token embeddings by √n_embd; 1.0 otherwise
    pub sliding_window: usize, // Gemma local layers attend to the last `sliding_window` tokens (0 = full)
    pub sliding_pattern: usize, // 1 global layer every `pattern` (Gemma-3=6, Gemma-2=2); 0 = non-Gemma
    pub gemma2: bool,      // Gemma-2 only: logit softcapping + uniform rope (Gemma-3 dropped both)
    pub attn_softcap: f32, // Gemma-2 attention-score softcap (50); 0 = none
    pub final_softcap: f32, // Gemma-2 final-logit softcap (30); 0 = none
    /// **Per-layer** sliding-window flag, length `n_layer`. `true` = local (windowed) attention.
    ///
    /// A scalar `sliding_pattern` cannot express every schedule, and reading one where the file stores
    /// an array is not a graceful degradation — `u()` returns `Err`, the fallback yields 0, and the
    /// model silently runs every layer global. Muse Glimmer ships the schedule as an
    /// `attention.sliding_window_pattern` array of 52 flags (1,1,1,0 repeating), so it is read directly
    /// and the modular Gemma rule is used only to synthesise this vector when a scalar is what's there.
    pub swa: Vec<bool>,
    /// Multiplies the final logits, after the LM head and **before** `final_softcap`.
    ///
    /// Verified against llama.cpp's `muse-glimmer.cpp` rather than the model card: Meta's write-up
    /// describes "extra query scaling to set the target logit scale", which reads as an *attention*
    /// scale, but the graph applies `ggml_scale(cur, f_logit_scale)` after `lm_head` and keeps
    /// `kq_scale = 1/sqrt(head_dim)`. Putting it on the queries would have produced fluent, wrong text.
    pub logit_scale: f32,
    /// Post-attention / post-FFN norms exist (Gemma, Muse Glimmer). Detected from the tensors present,
    /// not from the architecture name, because two unrelated families use them.
    pub post_norms: bool,
    /// **NoPE on global layers**: RoPE is applied only to sliding-window layers, and the full-attention
    /// layers carry no positional encoding at all. Muse Glimmer's design — local layers keep relative
    /// order via RoPE while the global layers stay position-free. `hparams.is_swa(il)` gates the rope
    /// call in llama.cpp's graph; this mirrors that.
    pub nope_global: bool,
    /// RoPE pairs consecutive dims (`2c`, `2c+1`) rather than split-half (`c`, `c+half`).
    ///
    /// llama.cpp calls these NORM and NEOX and resolves one per architecture. Muse Glimmer is NORM,
    /// and its GGUF converter runs `_unpermute_for_rope` so the stored Q/K rows are already in that
    /// order. Rotating the wrong pairs is invisible: it loads, the logits stay finite, and the text is
    /// fluent nonsense.
    pub rope_interleaved: bool,
    /// Epsilon for the POST-attention / POST-FFN norms, which on Muse Glimmer is 1e-8 — not the
    /// `attention.layer_norm_rms_epsilon` every other norm uses. llama.cpp hardcodes it with the
    /// comment "Different to f_norm_rms_eps for post-attn / post-FFN norms".
    pub post_norm_eps: f32,
    /// Normalise the token embeddings to unit RMS with NO learned weight before layer 0.
    ///
    /// `muse-glimmer.cpp` does `build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)`. Measured on this
    /// checkpoint the table is *already* row-normalised (per-row RMS 0.062409..0.062552, ratio 1.0023),
    /// so the op is within 0.3% of a uniform x16 — but it lands on the RESIDUAL, which the scale-
    /// invariant RMSNorms downstream cannot recover.
    pub embd_rmsnorm: bool,
    /// YaRN context-extension factor from `rope.scaling.type == "yarn"`; 1.0 means none.
    ///
    /// This is a SEPARATE mechanism from the Llama-3 `rope_freqs.weight` tensor, and this loader used
    /// to implement only that one — so any qwen3-arch checkpoint declaring YaRN (PrismML's Ternary
    /// Bonsai declares factor 4) silently got no rope scaling at all.
    pub yarn_factor: f32,
    pub yarn_orig_ctx: usize,
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        // The metadata keys are prefixed by the architecture (qwen2.*, qwen3.*, llama.*, …). Read it
        // once so one loader serves the whole dense Qwen/Llama family.
        let arch = match g.metadata().get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "qwen3".into() };
        let u = |k: &str| match g.metadata().get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing {arch}.{k}")) };
        let f = |k: &str| match g.metadata().get(&format!("{arch}.{k}")) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing {arch}.{k}")) };
        let n_vocab = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.len(), _ => return Err("no tokens".into()) };
        let n_head = u("attention.head_count")?;
        // Some arches (qwen2) omit key_length; then head_dim = embedding_length / head_count.
        let head_dim = u("attention.key_length").unwrap_or_else(|_| u("embedding_length").unwrap_or(0) / n_head.max(1));
        let is_gemma = arch.starts_with("gemma");
        let gemma2 = arch == "gemma2";
        let n_embd = u("embedding_length")?;
        Ok(Cfg {
            n_embd,
            n_layer: u("block_count")?,
            n_head,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim,
            n_ff: u("feed_forward_length")?,
            n_vocab,
            eps: f("attention.layer_norm_rms_epsilon")?,
            // Gemma-2 omits rope.freq_base (uniform 10000 default); other arches require it.
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            has_qk_norm: g.tensor("blk.0.attn_q_norm.weight").is_some(),
            qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
            is_gemma,
            embd_scale: if is_gemma { (n_embd as f32).sqrt() } else { 1.0 },
            sliding_window: u("attention.sliding_window").unwrap_or(0),
            // Global-attention layer every `pattern` layers: Gemma-2 alternates (2), Gemma-3 is 1-in-6.
            sliding_pattern: u("attention.sliding_window_pattern").unwrap_or(if gemma2 { 2 } else if is_gemma { 6 } else { 0 }),
            gemma2,
            attn_softcap: f("attn_logit_softcapping").unwrap_or(0.0),
            final_softcap: f("final_logit_softcapping").unwrap_or(0.0),
            swa: {
                // Prefer the explicit per-layer array; fall back to the modular rule only when the
                // file stores a scalar. Order matters: `u()` on an Arr returns Err, so checking the
                // array FIRST is what stops a hybrid schedule from collapsing to all-global.
                let n = u("block_count").unwrap_or(0);
                match g.metadata().get(&format!("{arch}.attention.sliding_window_pattern")) {
                    Some(Meta::Arr(a)) => a.iter().map(|m| match m {
                        Meta::U(v) => *v != 0,
                        Meta::I(v) => *v != 0,
                        Meta::Bool(v) => *v,
                        _ => false,
                    }).collect(),
                    _ => {
                        let p = u("attention.sliding_window_pattern")
                            .unwrap_or(if gemma2 { 2 } else if is_gemma { 6 } else { 0 });
                        (0..n).map(|il| is_gemma && p > 0 && il % p != p - 1).collect()
                    }
                }
            },
            logit_scale: f("logit_scale").unwrap_or(1.0),
            post_norms: g.tensor("blk.0.post_attention_norm.weight").is_some(),
            nope_global: arch == "muse-glimmer",
            // llama.cpp resolves NORM vs NEOX per architecture. `llama` is listed under "normal RoPE,
            // operating on pairs of consecutive head values" — NORM — while the Qwen family this
            // loader was written around is NEOX. One loader serves both, so this must be per-arch.
            //
            // Getting it wrong cost a `verified` badge: Llama-3.2-1B answered "The capital of France
            // is located in the United States" with NEOX and "Paris. The Eiffel Tower is a famous"
            // with NORM. Ġlocated beat ĠParis by 0.04 logits — near-miss, not noise, which is exactly
            // why it read as working.
            rope_interleaved: rope_is_interleaved(&arch) || std::env::var("FERRIC_ROPE_NORM").is_ok(),
            post_norm_eps: if arch == "muse-glimmer" { 1e-8 } else { f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) },
            embd_rmsnorm: arch == "muse-glimmer",
            yarn_factor: if matches!(g.metadata().get(&format!("{arch}.rope.scaling.type")), Some(Meta::Str(t)) if t == "yarn")
                { f("rope.scaling.factor").unwrap_or(1.0) } else { 1.0 },
            yarn_orig_ctx: u("rope.scaling.original_context_length").unwrap_or(0),
        })
    }
}

/// A projection that is *logically* one matmul emitting several stacked outputs (q|k|v, gate|up).
/// If every part shares a quant format it's byte-fused into one QMatrix (the fast path); real Q4_K_M
/// models mix formats even within qkv (V is often Q6_K while Q/K are Q4_K), so it falls back to one
/// matmul per part, concatenated — same result, one extra dispatch.
pub(crate) enum Proj {
    Fused(QMatrix),
    Split(Vec<QMatrix>),
}
impl Proj {
    pub(crate) fn load(ctx: &Arc<Context>, g: &impl GgufSource, names: &[&str]) -> Result<Proj, String> {
        let types: Vec<u32> = names.iter().map(|n| g.tensor(n).map(|t| t.ggml_type).unwrap_or(0)).collect();
        if names.len() > 1 && types.windows(2).all(|w| w[0] == w[1]) {
            Ok(Proj::Fused(qm_cat(ctx, g, names)?))
        } else if names.len() == 1 {
            Ok(Proj::Fused(qm(ctx, g, names[0])?))
        } else {
            Ok(Proj::Split(names.iter().map(|n| qm(ctx, g, n)).collect::<Result<_, _>>()?))
        }
    }
    pub(crate) fn matmul(&self, x: &Tensor) -> Tensor {
        match self {
            Proj::Fused(w) => x.matmul_q(w),
            Proj::Split(ws) => {
                let mut out = x.matmul_q(&ws[0]);
                for w in &ws[1..] { out = out.cat(&x.matmul_q(w), 1); }
                out
            }
        }
    }
    /// gate_up projection + SwiGLU. When gate|up is one fused Q4_K/Q5_K/Q6_K weight, one fused kernel
    /// does both (no [t, 2·n_ff] intermediate); otherwise the plain matmul + SwiGLU. Same result either way.
    pub(crate) fn gate_up_swiglu(&self, x: &Tensor, n_ff: usize) -> Tensor {
        // FERRIC_NOFUSE forces the un-fused path — for controlled A/B of the fusion, same binary.
        if std::env::var("FERRIC_NOFUSE").is_err() {
            if let Proj::Fused(w) = self {
                if let Some(o) = x.try_matmul_swiglu(w) { return o; }
            }
        }
        self.matmul(x).swiglu(n_ff)
    }
}

pub struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    q_norm: Option<Tensor>, // QK-norm: Qwen3 only
    k_norm: Option<Tensor>,
    wqkv: Proj, // q | k | v stacked (fused if same format, else separate matmuls concatenated)
    qkv_bias: Option<Tensor>, // Qwen2: concatenated q|k|v bias, added after the projection
    q_out: usize,
    kv_out: usize,
    wo: QMatrix,
    ffn_gate_up: Proj,
    ffn_gate_out: usize,
    ffn_down: QMatrix,
    /// **Gated GQA** (Muse Glimmer): a sigmoid gate computed from the layer's *normed input* — not
    /// from the attention output — and multiplied into the attention result BEFORE `wo`. Matches
    /// llama.cpp: `gate = sigmoid(wqkv_gate · attn_inp); cur = cur * gate`.
    attn_gate: Option<QMatrix>,
    /// Whether this layer applies RoPE at all. False on Muse Glimmer's global layers (NoPE).
    rope: bool,
    post_attn_norm: Option<Tensor>, // Gemma: normalizes the attn output before the residual add
    post_ffn_norm: Option<Tensor>,  // Gemma: normalizes the ffn output before the residual add
    rope_base: f32,                 // per-layer RoPE θ (Gemma alternates local 1e4 / global 1e6)
    window: usize,                  // sliding-window size for this layer (0 = full attention)
}

/// Per-layer attention K/V history. One step per token: append the new K/V into a grow-in-place
/// `KvBuf` (no O(len) re-concatenate), then attend over the [len, width] view of all of it.
#[derive(Default)]
pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
}
impl Cache {
    pub fn new(cfg: &Cfg) -> Cache { Cache { pos: 0, kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect() } }
    /// Per-layer (K, V) buffers — for a prefix cache that copies them.
    pub fn layers(&self) -> &[(KvBuf, KvBuf)] { &self.kv }
    /// Mutable access to layer `il`'s (K, V) — used by batched decode, which walks N caches per layer.
    pub fn kv_mut(&mut self, il: usize) -> &mut (KvBuf, KvBuf) { &mut self.kv[il] }
    /// Install pre-computed KV, e.g. a prefix copied from an earlier request.
    ///
    /// The caller must set [`Cache::pos`] to match; `crate::prefix::PrefixCache::seed` does both, and
    /// getting them out of step means the model ropes the next token at the wrong position.
    pub fn set_layers(&mut self, kv: Vec<(KvBuf, KvBuf)>) { self.kv = kv; }
}

/// Where the token-embedding rows come from.
///
/// `embed` already gathers only the prompt's rows — it never needs the whole table — so on a large
/// vocabulary the table is pure resident weight for no benefit. On Qwen2.5-0.5B `token_embd` is 144.6 MB
/// against 380 MB of layers, and a prompt touches a few kilobytes of it.
enum EmbdTable {
    Resident(Vec<u8>),
    /// Rows fetched per lookup. The caller must have them available — in a browser that means staged, and
    /// the tokens are known before the forward, so it is exact rather than speculative.
    Streamed { backing: Arc<dyn ferric_tier::Backing + Send + Sync>, base: u64 },
}

pub struct Qwen3 {
    pub cfg: Cfg,
    /// Set when layer weights are streamed rather than resident. `None` is the ordinary path and costs
    /// nothing — the branch in `run_layers` is one `Option` check per layer.
    pub stream: Option<crate::stream::LayerStream>,
    ctx: Arc<Context>,
    tok_embd: EmbdTable,
    layers: Vec<Layer>,
    out_norm: Tensor,
    lm_head: QMatrix,
    embd_type: u32,
    rope_freqs: Option<Tensor>, // Llama-3 rope-scaling factors [head_dim/2]; None for Qwen
    /// GPTQ calibration hook: when Some, each linear's input activation is captured (name → tensor) during
    /// the forward, for building per-layer input Hessians. None (default) = zero overhead.
    pub cap: std::cell::RefCell<Option<Vec<(String, Tensor)>>>,
}
/// Build one transformer layer from a weight source.
///
/// Extracted from `Qwen3::load` unchanged, so a layer can also be built **on demand** from bytes a tier
/// has just fetched — which is what [`Qwen3::load_streaming`] needs. Every weight here goes through
/// `GgufSource`, and that is the property which makes streaming a matter of swapping the source rather
/// than rewriting the model.
pub fn build_layer(
    ctx: &Arc<Context>,
    g: &impl GgufSource,
    cfg: &Cfg,
    il: usize,
) -> Result<Layer, String> {
    let nrm = |name: &str, n: usize| -> Result<Tensor, String> {
        Ok(Tensor::from_vec(ctx, &g.dequant(name)?, &[n]))
    };
            let b = |s: &str| format!("blk.{il}.{s}");
            let qkv_bias = if cfg.qkv_bias {
                let mut bias = g.dequant(&b("attn_q.bias"))?;
                bias.extend(g.dequant(&b("attn_k.bias"))?);
                bias.extend(g.dequant(&b("attn_v.bias"))?);
                Some(Tensor::from_vec(ctx, &bias, &[1, bias.len()]))
            } else { None };
            // Q/K/V: Qwen/Llama store three separate weights (we fuse them); Phi-3 stores ONE pre-fused
            // `attn_qkv` (q|k|v stacked) — load it directly and take the split widths from the config.
            let (wqkv, q_out, kv_out) = if g.tensor(&b("attn_qkv.weight")).is_some() {
                (Proj::load(ctx, g, &[&b("attn_qkv.weight")])?, cfg.n_head * cfg.head_dim, cfg.n_head_kv * cfg.head_dim)
            } else {
                (Proj::load(ctx, g, &[&b("attn_q.weight"), &b("attn_k.weight"), &b("attn_v.weight")])?,
                 g.tensor(&b("attn_q.weight")).ok_or("no attn_q")?.dims[1] as usize,
                 g.tensor(&b("attn_k.weight")).ok_or("no attn_k")?.dims[1] as usize)
            };
            // FFN gate|up: Qwen/Llama store separate `ffn_gate`+`ffn_up`; Phi-3 pre-fuses them into
            // `ffn_up` ([2·n_ff, n_embd], gate first) — same layout our SwiGLU fast-path already expects.
            let (ffn_gate_up, ffn_gate_out) = if g.tensor(&b("ffn_gate.weight")).is_some() {
                (Proj::load(ctx, g, &[&b("ffn_gate.weight"), &b("ffn_up.weight")])?,
                 g.tensor(&b("ffn_gate.weight")).unwrap().dims[1] as usize)
            } else {
                (Proj::load(ctx, g, &[&b("ffn_up.weight")])?, cfg.n_ff)
            };
            // Gemma alternates attention: 1 global layer every 6 (full attn, θ=rope_base=1e6), the rest
            // local (sliding-window, θ=1e4). Non-Gemma layers are always full causal (window 0).
            // Local (sliding-window) layer unless it's the global one every `sliding_pattern` layers.
            let is_local = cfg.swa.get(il).copied().unwrap_or(false);
            // Gemma-3 alternates rope θ (local 1e4 / global rope_base=1e6); Gemma-2 is uniform (rope_base=1e4).
            // Gemma-3 alone uses a DUAL theta: local layers rotate at 1e4 while global layers use
            // rope_base (1e6). Gemma-2 is uniform. This rule must stay keyed on `is_gemma` — it was
            // keyed on `is_local` alone, and the moment `is_local` stopped meaning "Gemma local layer"
            // every other windowed architecture silently had its theta replaced by 10000. Muse Glimmer
            // rotates its local layers at rope_base = 500000; at 1e4 the model loads, produces finite
            // logits, and emits newlines forever.
            let rope_base = if cfg.is_gemma && !cfg.gemma2 && is_local { 10000.0 } else { cfg.rope_base };
            let window = if is_local { cfg.sliding_window } else { 0 };
            Ok(Layer {
                attn_norm: nrm(&b("attn_norm.weight"), cfg.n_embd)?,
                ffn_norm: nrm(&b("ffn_norm.weight"), cfg.n_embd)?,
                q_norm: if cfg.has_qk_norm { Some(nrm(&b("attn_q_norm.weight"), cfg.head_dim)?) } else { None },
                k_norm: if cfg.has_qk_norm { Some(nrm(&b("attn_k_norm.weight"), cfg.head_dim)?) } else { None },
                wqkv,
                qkv_bias,
                q_out,
                kv_out,
                wo: qm(ctx, g, &b("attn_output.weight"))?,
                ffn_gate_up,
                ffn_gate_out,
                ffn_down: qm(ctx, g, &b("ffn_down.weight"))?,
                attn_gate: match g.tensor(&b("attn_gate.weight")) {
                    Some(_) => Some(qm(ctx, g, &b("attn_gate.weight"))?),
                    None => None,
                },
                // NoPE on the global layers; every other architecture ropes every layer.
                rope: !cfg.nope_global || is_local,
                post_attn_norm: if cfg.post_norms { Some(nrm(&b("post_attention_norm.weight"), cfg.n_embd)?) } else { None },
                post_ffn_norm: if cfg.post_norms { Some(nrm(&b("post_ffw_norm.weight"), cfg.n_embd)?) } else { None },
                rope_base,
                window,
            })
}

/// A forward pass paused between layers.
///
/// Exists for callers that must **await** something mid-pass — a browser fetching the next layer's
/// weights. A loop that runs to completion cannot do that, and on wasm there is no thread to block on a
/// read, so without this the entire layer set must be resident before a step begins. That is precisely
/// what capped the browser path's peak memory at the whole model regardless of its budget.
pub struct Step {
    x: Tensor,
    pos: usize,
    il: usize,
    n_layer: usize,
    n_tokens: usize,
}

impl Step {
    /// The layer [`Qwen3::step_layer`] will apply next, or `None` when the layers are done.
    ///
    /// A caller stages *this* layer's weights before calling `step_layer`, and may release the previous
    /// one afterwards — which is what bounds peak residency to the pinned set plus one layer.
    pub fn next_layer(&self) -> Option<usize> { (self.il < self.n_layer).then_some(self.il) }
    pub fn layers_done(&self) -> usize { self.il }
}

impl Qwen3 {
    /// Start/stop capturing linear-input activations for GPTQ calibration.
    pub fn set_capture(&self, on: bool) { *self.cap.borrow_mut() = if on { Some(Vec::new()) } else { None }; }
    /// Take the captured (name, activation) pairs, leaving capture off.
    pub fn take_capture(&self) -> Vec<(String, Tensor)> { self.cap.borrow_mut().take().unwrap_or_default() }
    fn grab(&self, name: String, t: &Tensor) { if let Some(v) = self.cap.borrow_mut().as_mut() { v.push((name, t.clone())); } }
}

impl Qwen3 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Qwen3, String> {
        Self::load_inner(ctx, g, true)
    }

    /// Load with layer weights **streamed** from `path` under a byte budget.
    ///
    /// Only the layer weights stream; embeddings, norms and the LM head stay resident, which mirrors
    /// every production streaming engine — they are a small share of the parameters and are touched on
    /// every token regardless. Slower than resident by design: the saving is memory, the cost is
    /// re-uploading each layer per visit.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_streaming(ctx: &Arc<Context>, path: &str, budget_bytes: u64) -> Result<Qwen3, String> {
        Self::load_streaming_with(ctx, path, budget_bytes, None, true)
    }

    /// Streamed load with an explicit backing and overlap setting — the seam a benchmark needs in order
    /// to measure the tier against a device slower than a warm page cache.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_streaming_with(
        ctx: &Arc<Context>,
        path: &str,
        budget_bytes: u64,
        backing: Option<Arc<dyn ferric_tier::Backing + Send + Sync>>,
        overlap: bool,
    ) -> Result<Qwen3, String> {
        let file = ferric_gguf::GgufFile::open(path)?;
        let cfg = Cfg::from_gguf(&file)?;
        // Build everything EXCEPT the layers, so peak memory never includes the full weight set — a
        // load path that materialised them first and then dropped them would defeat the purpose.
        let mut m = Self::load_inner(ctx, &file, false)?;
        let b = match backing {
            Some(b) => b,
            None => Arc::new(ferric_tier::FileBacking::open(path).map_err(|e| e.to_string())?),
        };
        m.stream = Some(crate::stream::open_with(ctx, path, b, budget_bytes, cfg, overlap)?);
        Ok(m)
    }

    /// Load everything EXCEPT the layers, then attach a stream that materialises them on demand.
    ///
    /// The constructor a browser uses: no path, no filesystem, and peak memory never includes the full
    /// weight set — a load path that built the layers and then dropped them would defeat the purpose.
    pub fn from_stream(
        ctx: &Arc<Context>,
        g: &impl GgufSource,
        stream: crate::stream::LayerStream,
        embd: Option<(Arc<dyn ferric_tier::Backing + Send + Sync>, u64)>,
    ) -> Result<Qwen3, String> {
        let mut m = Self::load_inner_embd(ctx, g, false, embd)?;
        m.stream = Some(stream);
        Ok(m)
    }

    fn load_inner(ctx: &Arc<Context>, g: &impl GgufSource, resident: bool) -> Result<Qwen3, String> {
        Self::load_inner_embd(ctx, g, resident, None)
    }

    /// `embd = Some((backing, base))` streams the token-embedding table instead of loading it.
    ///
    /// Skipping the load matters rather than loading-then-freeing: on this model the table is 144.6 MB,
    /// and a path that materialised it first would put that in the peak — which is the number a memory
    /// budget is judged on.
    fn load_inner_embd(
        ctx: &Arc<Context>,
        g: &impl GgufSource,
        resident: bool,
        embd: Option<(Arc<dyn ferric_tier::Backing + Send + Sync>, u64)>,
    ) -> Result<Qwen3, String> {
        let cfg = Cfg::from_gguf(g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        // Gemma's `(1+w)` RMSNorm is folded into the weight at GGUF-conversion time (llama.cpp adds 1 to
        // every `*_norm` weight), so at runtime it's a plain rmsnorm·weight — no offset here. `nrm` just
        // loads a norm tensor (kept as one helper so the Gemma post-norms load the same way).
        let nrm = |name: &str, n: usize| -> Result<Tensor, String> {
            Ok(Tensor::from_vec(ctx, &g.dequant(name)?, &[n]))
        };
        if resident {
            for il in 0..cfg.n_layer {
                layers.push(build_layer(ctx, g, &cfg, il)?);
            }
        }
        let head = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        Ok(Qwen3 {
            cap: std::cell::RefCell::new(None),
            tok_embd: match &embd {
                Some((b, base)) => EmbdTable::Streamed { backing: Arc::clone(b), base: *base },
                None => EmbdTable::Resident(g.raw("token_embd.weight")?),
            },
            out_norm: nrm("output_norm.weight", cfg.n_embd)?,
            lm_head: qm(ctx, g, head)?,
            embd_type: g.tensor("token_embd.weight").ok_or("no token_embd")?.ggml_type,
            // ⚠ RECIPROCAL. `rope_freqs.weight` holds ggml's `freq_factors`, which ggml applies as
            // `theta / ff` (ops.cpp: `rope_yarn(theta/ff, …)`). Ferric's `rope_scaled` kernel instead
            // MULTIPLIES each inverse frequency by its scale, so the factors must be inverted on the
            // way in. Llama-3.2-1B ships values running 1.0 → 32.0 (1.0 on the high-frequency dims),
            // so passing them through unchanged shortened the low-frequency wavelengths by 32x
            // instead of stretching them — the exact inverse of the context extension they encode.
            // Nothing errors and short prompts look fine; only long context degrades.
            //
            // The qwen35 path is unaffected: `yarn_freq_scale` there builds
            // `(1/factor)·(1−ramp) + ramp`, already in multiplier form.
            // YaRN and Llama-3 rope scaling are different mechanisms that both end up as a per-dim
            // multiplier on the inverse frequency. Llama-3 ships an explicit tensor of DIVISORS;
            // YaRN is computed from metadata. Only one is ever present.
            rope_freqs: if std::env::var("FERRIC_NO_ROPE_FREQS").is_ok() { None }
                else if g.tensor("rope_freqs.weight").is_none() && cfg.yarn_factor > 1.0 {
                    let v = crate::qwen35::yarn_freq_scale(cfg.head_dim, cfg.rope_base, cfg.yarn_factor,
                                                           cfg.yarn_orig_ctx, 32.0, 1.0);
                    Some(Tensor::from_vec(ctx, &v, &[cfg.head_dim / 2]))
                } else { g.tensor("rope_freqs.weight").map(|t| {
                let n = t.dims[0] as usize;
                let f = g.dequant("rope_freqs.weight")?;
                let inv: Vec<f32> = f[..n].iter().map(|&x| if x != 0.0 { 1.0 / x } else { 1.0 }).collect();
                Ok::<_, String>(Tensor::from_vec(ctx, &inv, &[n]))
            }).transpose()? },
            cfg, ctx: ctx.clone(), layers, stream: None,
        })
    }

    /// `embed` without the weightless embedding norm — the row gather only.
    ///
    /// The token path normalises immediately; the multimodal path must not, because the norm has to
    /// see the image rows too. Both call this.
    pub fn embed_raw(&self, tokens: &[u32]) -> Tensor { self.embed_inner(tokens, false) }

    pub fn embed(&self, tokens: &[u32]) -> Tensor { self.embed_inner(tokens, true) }

    fn embed_inner(&self, tokens: &[u32], norm: bool) -> Tensor {
        let d = self.cfg.n_embd;
        // Gather + dequantize just the prompt's rows on the CPU, in whatever format the embedding
        // table is stored (Q2_0/Q4_K/…) — beats parking the whole table on the GPU for a gather.
        let row_bytes = ferric_gguf::type_size(self.embd_type, d).expect("embd type");
        let mut v = Vec::with_capacity(tokens.len() * d);
        let mut scratch = vec![0u8; row_bytes];
        for &t in tokens {
            let off = t as usize * row_bytes;
            let row: &[u8] = match &self.tok_embd {
                EmbdTable::Resident(tbl) => &tbl[off..off + row_bytes],
                EmbdTable::Streamed { backing, base } => {
                    backing.read_at(base + off as u64, &mut scratch).expect("embedding row not available");
                    &scratch
                }
            };
            v.extend(deq_raw(row, d, self.embd_type).expect("embed row"));
        }
        // Gemma scales the token embeddings by √n_embd (identity elsewhere, embd_scale == 1.0).
        if self.cfg.embd_scale != 1.0 { for x in &mut v { *x *= self.cfg.embd_scale; } }
        // Muse Glimmer normalises the embedding rows to unit RMS with NO learned weight before the
        // first layer. Done on the CPU here because the rows were just dequantised here anyway.
        if norm && self.cfg.embd_rmsnorm && std::env::var("FERRIC_NO_EMBDNORM").is_err() {
            let eps = self.cfg.eps;
            for row in v.chunks_mut(d) {
                let ms = row.iter().map(|x| x * x).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                for x in row.iter_mut() { *x *= inv; }
            }
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// Full RoPE over head_dim (Qwen rotates the whole head). Llama-3 applies its per-frequency
    /// `rope_freqs` scaling; Qwen has none, so it's plain RoPE.
    fn rope(&self, x: &Tensor, n_heads: usize, offset: usize, base: f32) -> Tensor {
        let r = self.rope_inner(x, n_heads, offset, base);
        // YaRN scales cos/sin by (1 + 0.1·ln factor). Derivation from llama-context.cpp: the two
        // adjustments there cancel to yarn_attn_factor = 1.0 for a non-DeepSeek model, and ggml's
        // `rope_yarn` then re-multiplies by exactly this term inside the kernel. (The same arithmetic
        // yields 1.0 for deepseek2, which is why that model wants no extra scale.)
        //
        // Llama-3 `rope_freqs` scaling carries NO such factor, so this is gated on YaRN specifically.
        if self.cfg.yarn_factor > 1.0 {
            let m = 1.0 + 0.1 * self.cfg.yarn_factor.ln();
            return r.mul(&r.scalar(m));
        }
        r
    }

    fn rope_inner(&self, x: &Tensor, n_heads: usize, offset: usize, base: f32) -> Tensor {
        match &self.rope_freqs {
            // The scaled path must honour the pairing too. It did not: `rope_interleaved` was
            // consulted only in the `None` arm, so any model with rope_freqs (every Llama-3.1+)
            // silently got NEOX regardless — and an A/B on the flag appeared to "rule out" pairing
            // while never actually changing anything.
            Some(fs) if self.cfg.rope_interleaved && std::env::var("FERRIC_NEOX").is_err() =>
                x.rope_scaled_interleaved(fs, n_heads, self.cfg.head_dim, base, offset),
            Some(fs) => x.rope_scaled(fs, n_heads, self.cfg.head_dim, base, offset),
            None if self.cfg.rope_interleaved && std::env::var("FERRIC_NEOX").is_err() => x.rope_interleaved(n_heads, self.cfg.head_dim, base, offset),
            None => x.rope(n_heads, self.cfg.head_dim, base, offset),
        }
    }

    /// LM head → `logit_scale` → `final_softcap`, in that order.
    ///
    /// One helper because three call sites (decode, stepped decode, profiled decode) applied the
    /// softcap independently, and a fourth thing now has to happen between the matmul and the cap.
    /// llama.cpp's order is `ggml_scale(cur, f_logit_scale)` and then the tanh cap, so scaling after
    /// the cap — the easy mistake when bolting it on — would saturate against the wrong bound.
    fn head(&self, x: &Tensor) -> Tensor {
        let lg = x.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q(&self.lm_head);
        let lg = if self.cfg.logit_scale != 1.0 && std::env::var("FERRIC_NOLOGITSCALE").is_err() {
            lg.mul(&lg.scalar(self.cfg.logit_scale))
        } else { lg };
        if self.cfg.final_softcap > 0.0 { lg.softcap(self.cfg.final_softcap) } else { lg }
    }

    fn attn(&self, h: &Tensor, l: &Layer, cache: &mut (KvBuf, KvBuf), offset: usize, il: usize) -> Tensor {
        let (t, hd, nh, nkv) = (h.shape[0], self.cfg.head_dim, self.cfg.n_head, self.cfg.n_head_kv);
        self.grab(format!("l{il}.qkv"), h); // GPTQ calibration: capture wqkv input
        // One fused matmul emits [q | k | v]; (+ bias for Qwen2); split, optional QK-norm, RoPE.
        let qkv = l.wqkv.matmul(h);
        let qkv = match &l.qkv_bias { Some(bias) => qkv.add(bias), None => qkv };
        // QK-norm (Qwen3) normalizes each head; without it (Qwen2/Llama) q/k pass through unchanged.
        let qn = |x: Tensor, n: usize, norm: &Option<Tensor>| match norm {
            Some(w) => x.reshape(&[t, n, hd]).rmsnorm(w, self.cfg.eps).reshape(&[t, n * hd]),
            None => x,
        };
        // No `.contiguous()` on these windows either, and it is correct on BOTH branches of `qn`:
        //   - QK-norm present (Qwen3): `reshape` materialises internally, exactly as before.
        //   - QK-norm absent (Qwen2/Llama): the view flows into `rope`, which reads a row-major window in
        //     place — so the packing dispatch disappears rather than moving.
        // During prefill these narrows are genuinely strided, so this removes a real copy there too, not
        // just the decode-time one the size-1 stride rule already handled.
        // No `.contiguous()`: `KvBuf::append` reads a strided view in place, so the window onto the fused
        // QKV output goes straight into the cache. That removes one `gather` dispatch per layer per token
        // whose only job was moving bytes — in decode AND in prefill, where the view is genuinely strided.
        let v = qkv.narrow(1, l.q_out + l.kv_out, l.kv_out);

        // RoPE q and k in ONE dispatch instead of two.
        //
        // The rotation angle is `f32(i + pos) * exp(-2c/dh * ln base)` — a function of the token position
        // and the dimension WITHIN a head. It does not depend on which head. So roping the contiguous
        // `[q|k]` span as a single tensor with `nh + nkv` heads is not an approximation of roping them
        // separately, it is the identical computation: head `h < nh` lands at `(i*(nh+nkv) + h)*dh`, which
        // is exactly where q's head `h` already lives in the fused QKV row. Verified bit-identical
        // (max|Δ| 0.000e0) across t ∈ {1,5,37} × pos ∈ {0,7,129}.
        //
        // Only valid when q and k are still ADJACENT at rope time, which rules out two cases:
        //   - QK-norm (Qwen3) normalises each separately first, so they are no longer one span;
        //   - rope-scaling (Llama-3) goes through a different kernel that has not been shown equivalent.
        // Both fall back to the two-dispatch path, which is unchanged.
        let fuse_rope = (l.rope || std::env::var("FERRIC_NONOPE").is_ok()) && l.q_norm.is_none() && l.k_norm.is_none() && self.rope_freqs.is_none();
        let nope = !l.rope && std::env::var("FERRIC_NONOPE").is_err();
        let (q, k) = if nope {
            // NoPE layer (Muse Glimmer's global attention): QK-norm still applies, RoPE does not.
            // Skipping the rotation is the whole difference — position enters this layer only through
            // which keys exist in the cache, which is what "preserve information globally" means here.
            (qn(qkv.narrow(1, 0, l.q_out), nh, &l.q_norm),
             qn(qkv.narrow(1, l.q_out, l.kv_out), nkv, &l.k_norm))
        } else if fuse_rope {
            let qk = qkv.narrow(1, 0, l.q_out + l.kv_out).rope(nh + nkv, hd, l.rope_base, offset);
            // q's window has offset 0 (free during decode via the size-1 stride rule); k's carries an
            // offset and flows into `KvBuf::append`, which reads views in place.
            (qk.narrow(1, 0, l.q_out), qk.narrow(1, l.q_out, l.kv_out))
        } else {
            let q = qn(qkv.narrow(1, 0, l.q_out), nh, &l.q_norm);
            let k = qn(qkv.narrow(1, l.q_out, l.kv_out), nkv, &l.k_norm);
            {
                let (qr, kr) = (self.rope(&q, nh, offset, l.rope_base), self.rope(&k, nkv, offset, l.rope_base));
                dump("Qcur_rope", il, &qr);
                dump("Kcur_rope", il, &kr);
                (qr, kr)
            }
        };

        // Append the new K/V rows into the grow-in-place cache and read a view over all rows so far.
        // Byte-identical to the old `pk.cat(&k, 0)`, but without re-copying the history each step —
        // and in ONE dispatch rather than two, since K and V land at the same point in every layer.
        // Verified byte-identical to two separate appends across GQA widths, strided source windows and
        // cache growth in `ferric-tensor/examples/kv_append2.rs`.
        let (kc, vc) = ferric_tensor::append2(&self.ctx, &mut cache.0, &k, &mut cache.1, &v);
        // decode: fused single-query; prefill: flash (O(T) memory, no [nh,T,T] matrix) up to its
        // shared-memory limit, else the composed causal path. All three are the same math.
        let s = kc.shape[0];
        // FERRIC_NOWINDOW disables the sliding window (attends to all keys) — for A/B-ing its effect.
        let win = if std::env::var("FERRIC_NOWINDOW").is_ok() { 0 } else { l.window };
        let sc = self.cfg.attn_softcap; // Gemma-2 attention-score softcap (0 elsewhere)
        let o = if win > 0 {
            // Sliding-window (Gemma local layer): the query attends only to the last `window` keys.
            if t == 1 { nn::decode_attention_win(&q, &kc, &vc, nh, nkv, win, sc) }
            else { nn::causal_attention_win(&q, &kc, &vc, nh, nkv, win, sc) }
        } else if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv, sc)
        } else if t == s && s <= 65535 && hd <= 128 && sc == 0.0 {
            q.flash_attention_prefill(&kc, &vc, nh, nkv, hd)
        } else {
            // chunked_attention delegates to causal_attention when q covers the whole history, so
            // this one call serves full prefill, prefix-cached suffixes and chunked prefill alike.
            nn::chunked_attention(&q, &kc, &vc, nh, nkv, sc)
        };
        // Gated GQA: the gate is a projection of the layer's NORMED INPUT `h`, not of the attention
        // output, so it cannot be folded into `wo`. Sigmoid, then elementwise into the attention result
        // before the output projection.
        // FERRIC_NOGATE / FERRIC_NONOPE / FERRIC_NOLOGITSCALE exist so a new architecture's pieces can
        // be ablated one at a time against the same binary and the same weights. A 30B model gives no
        // useful signal from reading the code once it is structurally plausible; it gives a lot from
        // turning one thing off.
        let o = match &l.attn_gate {
            Some(wg) if std::env::var("FERRIC_NOGATE").is_err() => o.mul(&h.matmul_q(wg).sigmoid()),
            _ => o,
        };
        self.grab(format!("l{il}.wo"), &o); // GPTQ calibration: capture wo input
        o.matmul_q(&l.wo)
    }

    fn ffn(&self, h: &Tensor, l: &Layer, il: usize) -> Tensor {
        self.grab(format!("l{il}.ffn_gu"), h); // GPTQ calibration: capture ffn_gate_up input
        // Gemma uses GEGLU (gelu gate) not SwiGLU (silu), so it can't use the silu-fused fast paths:
        // project gate|up, gelu the gate half, multiply by the up half, then the down projection.
        if self.cfg.is_gemma {
            let gu = l.ffn_gate_up.matmul(h);
            let n = l.ffn_gate_out;
            let gate = gu.narrow(1, 0, n).contiguous().gelu_tanh();
            let up = gu.narrow(1, n, n).contiguous();
            return gate.mul(&up).matmul_q(&l.ffn_down);
        }
        // Whole-FFN megakernel (gate_up Q4_K + SwiGLU + down Q6_K in one dispatch), OPT-IN via
        // FERRIC_MEGA — correct but ~2× slower at decode (occupancy-bound); off by default.
        if let Proj::Fused(gu) = &l.ffn_gate_up {
            if let Some(o) = h.try_ffn_mega(gu, &l.ffn_down, l.ffn_gate_out) { return o; }
        }
        // staged: gate_up + SwiGLU (one fused kernel when gate|up is a k-quant) → down projection.
        let sw = l.ffn_gate_up.gate_up_swiglu(h, l.ffn_gate_out);
        self.grab(format!("l{il}.ffn_down"), &sw); // GPTQ calibration: capture ffn_down input
        sw.matmul_q(&l.ffn_down)
    }

    /// Prefill (stateless): logits [T, n_vocab].
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        let mut cache = Cache::new(&self.cfg);
        self.forward_cached(tokens, &mut cache)
    }

    /// Run embed + all transformer layers, carrying K/V in `cache`. Returns the last layer's hidden
    /// state `x` (BEFORE the final norm / lm_head) — shared by decode (→ logits) and embedding (→ pooled).
    /// One transformer layer, applied to `x`. Extracted from `run_layers` unchanged.
    ///
    /// Pulling it out is what makes a **stepping** forward possible: a caller that must await something
    /// between layers — a browser fetching the next layer's weights — cannot use a loop that runs to
    /// completion. See [`Qwen3::step_layer`].
    fn apply_layer(&self, x: &Tensor, l: &Layer, lc: &mut (KvBuf, KvBuf), pos: usize, il: usize) -> Tensor {
        use ferric_tensor::{batch, prof};
        dump("inpL", il, x);
        dump("attn_norm", il, &x.rmsnorm(&l.attn_norm, self.cfg.eps));
        let profiling = std::env::var("FERRIC_PROFILE").is_ok();
        let mut out;
        let xin = x;
        if profiling {
            // Eager per-category so the sync'd timer attributes attn vs ffn (see qwen35).
            let y = batch(&self.ctx, || self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos, il));
            prof(&self.ctx, "attn");
            out = batch(&self.ctx, || { let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps); self.ffn(&xy_n, l, il).add(&xy) });
            prof(&self.ctx, "ffn");
        } else if self.cfg.post_norms {
            // Gemma and Muse Glimmer normalize the attn AND ffn *outputs* (post-norms) before each add:
            //   x = x + post_attn_norm(attn(input_norm(x))); x = x + post_ffn_norm(ffn(pre_ffn_norm(x)))
            let eps = self.cfg.eps;
            // FERRIC_POSTNORM=off|sum ablates this. `off` drops the post-norms entirely (plain
            // pre-norm residual); `sum` normalises the RESIDUAL SUM rather than the branch output,
            // which is the other reading of "post-norm ... then residual added" and produces a
            // different model from the same weights.
            let mode = std::env::var("FERRIC_POSTNORM").unwrap_or_default();
            out = batch(&self.ctx, || {
                let a = self.attn(&xin.rmsnorm(&l.attn_norm, eps), l, lc, pos, il);
                match mode.as_str() {
                    "off" => {
                        let x1 = xin.add(&a);
                        let f = self.ffn(&x1.rmsnorm(&l.ffn_norm, eps), l, il);
                        x1.add(&f)
                    }
                    "sum" => {
                        let x1 = xin.add(&a).rmsnorm(l.post_attn_norm.as_ref().unwrap(), eps);
                        let f = self.ffn(&x1.rmsnorm(&l.ffn_norm, eps), l, il);
                        x1.add(&f).rmsnorm(l.post_ffn_norm.as_ref().unwrap(), eps)
                    }
                    _ => {
                        // POST norms use their own epsilon (1e-8 on Muse Glimmer); the PRE norms use
                        // the model's rms eps. Same tensor op, different constant.
                        let pe = if std::env::var("FERRIC_POSTEPS").is_ok() { eps } else { self.cfg.post_norm_eps };
                        let x1 = xin.add(&a.rmsnorm(l.post_attn_norm.as_ref().unwrap(), pe));
                        let f = self.ffn(&x1.rmsnorm(&l.ffn_norm, eps), l, il);
                        x1.add(&f.rmsnorm(l.post_ffn_norm.as_ref().unwrap(), pe))
                    }
                }
            });
        } else {
            out = batch(&self.ctx, || {
                let y = self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos, il);
                // fused: xy = xin + y (next residual), xy_n = rmsnorm(xy) — one kernel, not two.
                let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps);
                self.ffn(&xy_n, l, il).add(&xy)
            });
        }

        out
    }

    /// Attention for **N independent sequences**, one token each.
    ///
    /// The whole point is the first line: `wqkv.matmul` runs once over `[N, d]` instead of N times over
    /// `[1, d]`, so the 525 MB of weights are read **once for N tokens** rather than N times. That is the
    /// entire source of the win — decode is a weight-streaming problem, and batching is what amortises it.
    ///
    /// What cannot be batched is attention itself: sequence `i` attends *its own* KV history, at *its own*
    /// position, and those histories have different lengths. So the per-sequence work stays a loop, and
    /// only the projections are shared. (Paged attention is what would collapse that loop too.)
    fn attn_batch(&self, h: &Tensor, l: &Layer, caches: &mut [&mut Cache], il: usize) -> Tensor {
        let (n, hd, nh, nkv) = (h.shape[0], self.cfg.head_dim, self.cfg.n_head, self.cfg.n_head_kv);
        debug_assert_eq!(n, caches.len(), "one row per sequence");

        let qkv = l.wqkv.matmul(h);                                  // <-- batched: the win
        let qkv = match &l.qkv_bias { Some(bias) => qkv.add(bias), None => qkv };
        let qn = |x: Tensor, hn: usize, norm: &Option<Tensor>| match norm {
            Some(w) => x.reshape(&[n, hn, hd]).rmsnorm(w, self.cfg.eps).reshape(&[n, hn * hd]),
            None => x,
        };

        // Each row sits at a different absolute position, which is exactly what `rope_at` exists for.
        let positions: Vec<u32> = caches.iter().map(|c| c.pos as u32).collect();
        let fuse = l.q_norm.is_none() && l.k_norm.is_none() && self.rope_freqs.is_none();
        let (q, k) = if fuse {
            let qk = qkv.narrow(1, 0, l.q_out + l.kv_out).rope_at(nh + nkv, hd, l.rope_base, &positions);
            (qk.narrow(1, 0, l.q_out), qk.narrow(1, l.q_out, l.kv_out))
        } else {
            let q = qn(qkv.narrow(1, 0, l.q_out), nh, &l.q_norm);
            let k = qn(qkv.narrow(1, l.q_out, l.kv_out), nkv, &l.k_norm);
            (q.rope_at(nh, hd, l.rope_base, &positions), k.rope_at(nkv, hd, l.rope_base, &positions))
        };
        let v = qkv.narrow(1, l.q_out + l.kv_out, l.kv_out);

        let sc = self.cfg.attn_softcap;
        let win = if std::env::var("FERRIC_NOWINDOW").is_ok() { 0 } else { l.window };
        let mut outs: Vec<Tensor> = Vec::with_capacity(n);
        for (i, c) in caches.iter_mut().enumerate() {
            let (ki, vi) = (k.narrow(0, i, 1), v.narrow(0, i, 1));
            let lc = c.kv_mut(il);
            let (kc, vc) = ferric_tensor::append2(&self.ctx, &mut lc.0, &ki, &mut lc.1, &vi);
            let qi = q.narrow(0, i, 1).contiguous();
            outs.push(if win > 0 { nn::decode_attention_win(&qi, &kc, &vc, nh, nkv, win, sc) }
                      else { nn::decode_attention(&qi, &kc, &vc, nh, nkv, sc) });
        }
        let o = outs.iter().skip(1).fold(outs[0].clone(), |acc, t| acc.cat(t, 0));
        o.matmul_q(&l.wo)                                            // <-- batched again
    }

    /// One layer for N sequences. Identical structure to `apply_layer`; only attention differs.
    fn apply_layer_batch(&self, x: &Tensor, l: &Layer, caches: &mut [&mut Cache], il: usize) -> Tensor {
        use ferric_tensor::batch;
        let xin = x;
        if self.cfg.is_gemma {
            let eps = self.cfg.eps;
            batch(&self.ctx, || {
                let a = self.attn_batch(&xin.rmsnorm(&l.attn_norm, eps), l, caches, il);
                let x1 = xin.add(&a.rmsnorm(l.post_attn_norm.as_ref().unwrap(), eps));
                let f = self.ffn(&x1.rmsnorm(&l.ffn_norm, eps), l, il);
                x1.add(&f.rmsnorm(l.post_ffn_norm.as_ref().unwrap(), eps))
            })
        } else {
            batch(&self.ctx, || {
                let y = self.attn_batch(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, caches, il);
                let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps);
                self.ffn(&xy_n, l, il).add(&xy)
            })
        }
    }

    /// **Batched decode**: advance N independent sequences by one token each, in one forward pass.
    ///
    /// `tokens[i]` is the next token for `caches[i]`. Returns `[N, n_vocab]` logits — row `i` belongs to
    /// sequence `i`.
    ///
    /// Every sequence's logits are **identical** to calling `forward_cached` on it alone; batching changes
    /// only how the work is scheduled. That is asserted in `examples/batched_decode.rs`, because a batched
    /// path that crossed sequences would still emit fluent text.
    pub fn forward_batch(&self, tokens: &[u32], caches: &mut [&mut Cache]) -> Tensor {
        use ferric_tensor::batch;
        assert_eq!(tokens.len(), caches.len(), "one token per sequence");
        assert!(!tokens.is_empty(), "forward_batch needs at least one sequence");
        let mut x = self.embed(tokens);
        for il in 0..self.cfg.n_layer {
            let l = self.layer_ref(il);
            x = self.apply_layer_batch(&x, &l, caches, il);
        }
        for c in caches.iter_mut() { c.pos += 1; }
        batch(&self.ctx, || self.head(&x))
    }

    /// The layer at `il`: resident, or materialised from the tier.
    fn layer_ref(&self, il: usize) -> crate::stream::LayerRef<'_> {
        match &self.stream {
            Some(s) => s.layer(il).expect("streamed layer"),
            None => crate::stream::LayerRef::Borrowed(&self.layers[il]),
        }
    }

    /// Begin a forward pass that can be advanced one layer at a time.
    pub fn step_begin(&self, tokens: &[u32], cache: &Cache) -> Step {
        Step {
            x: self.embed(tokens),
            pos: cache.pos,
            il: 0,
            n_layer: self.cfg.n_layer,
            n_tokens: tokens.len(),
        }
    }

    /// Apply the next layer. Returns `true` when all layers are done.
    ///
    /// The caller owns what happens between calls, which is the entire point: stage the weights for
    /// `step.next_layer()`, call this, release the layer before it.
    pub fn step_layer(&self, step: &mut Step, cache: &mut Cache) -> bool {
        let Some(il) = step.next_layer() else { return true };
        let l = self.layer_ref(il);
        step.x = self.apply_layer(&step.x, &l, &mut cache.kv[il], step.pos, il);
        step.il += 1;
        step.il >= step.n_layer
    }

    /// Final norm + LM head, and advance the cache. Consumes the step.
    pub fn step_finish(&self, step: Step, cache: &mut Cache) -> Tensor {
        use ferric_tensor::batch;
        debug_assert!(step.next_layer().is_none(), "step_finish called before every layer ran");
        cache.pos += step.n_tokens;
        batch(&self.ctx, || self.head(&step.x))
    }

    fn run_layers(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        use ferric_tensor::prof;
        let mut x = self.embed(tokens);
        prof(&self.ctx, "embed");
        let pos = cache.pos;
        for il in 0..self.cfg.n_layer {
            // A streamed layer is dropped at the end of the iteration — that drop IS the eviction.
            let l = self.layer_ref(il);
            x = self.apply_layer(&x, &l, &mut cache.kv[il], pos, il);
        }
        cache.pos += tokens.len();
        x
    }

    /// Feed **precomputed embeddings** instead of token ids — the multimodal entry point.
    ///
    /// A vision tower produces embedding rows, not tokens, so the image cannot go in through
    /// `forward_cached`. `x` is `[T, n_embd]` and takes the place of the embedding gather; everything
    /// downstream (position, cache, layers, head) is identical, which is the point — an image is a run
    /// of rows in the same sequence, not a separate code path.
    pub fn forward_embeds(&self, x: &Tensor, cache: &mut Cache) -> Tensor {
        use ferric_tensor::batch;
        assert_eq!(x.shape[1], self.cfg.n_embd, "embeddings must be [T, n_embd]");
        let t = x.shape[0];
        let pos = cache.pos;
        // The weightless embedding RMS norm applies to the WHOLE input row block, image rows
        // included: llama.cpp normalises the OUTPUT of build_inp_embd, and build_inp_embd is exactly
        // where multimodal embeddings are substituted for token ids. Skipping it here mixes unit-RMS
        // text rows with a vision adapter's raw output (measured -44..86 on this model) and yields
        // fluent text about the wrong subject — which is why `embed_tokens` returns RAW rows and the
        // normalisation happens once, here, after the splice.
        let mut h = if self.cfg.embd_rmsnorm { x.rmsnorm_weightless(self.cfg.eps) } else { x.clone() };
        for il in 0..self.cfg.n_layer {
            let l = self.layer_ref(il);
            h = self.apply_layer(&h, &l, &mut cache.kv[il], pos, il);
        }
        cache.pos += t;
        batch(&self.ctx, || self.head(&h))
    }

    /// **Raw** embedding rows for `tokens` — deliberately WITHOUT the weightless embedding norm.
    ///
    /// The norm belongs after the splice, applied once to text and image rows together, because that
    /// is where the reference applies it. Normalising here would normalise the text twice and the
    /// image never.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Tensor { self.embed_raw(tokens) }

    /// Feed `tokens`, carrying K/V in `cache`. Prompt once, then one token per step. Returns logits.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        use ferric_tensor::{batch, prof};
        let x = self.run_layers(tokens, cache);
        let out = batch(&self.ctx, || self.head(&x));
        prof(&self.ctx, "lm_head");
        out
    }

    /// Fetch embedding rows on demand instead of holding the whole table.
    ///
    /// `base` is the table's absolute byte offset. Frees the resident copy, so peak drops by the table's
    /// full size — 144.6 MB on Qwen2.5-0.5B, where it is 21% of the checkpoint and is touched a few
    /// kilobytes at a time.
    ///
    /// The caller owns availability: with a `StagedBacking` the rows for the current tokens must be
    /// staged first, and a miss is `NotStaged` naming the range rather than a wrong embedding.
    pub fn stream_embeddings(&mut self, backing: Arc<dyn ferric_tier::Backing + Send + Sync>, base: u64) {
        self.tok_embd = EmbdTable::Streamed { backing, base };
    }

    /// Byte range of one embedding row, for a caller that must stage it.
    pub fn embd_row_extent(&self, token: u32, base: u64) -> (u64, usize) {
        let rb = ferric_gguf::type_size(self.embd_type, self.cfg.n_embd).expect("embd type");
        (base + token as u64 * rb as u64, rb)
    }

    /// The frozen quantized LM head [n_vocab, n_embd] — for `Var::matmul_qf` (LoRA around it without
    /// dequantizing to fp).
    pub fn lm_head(&self) -> &ferric_tensor::QMatrix { &self.lm_head }

    /// The hidden state ENTERING block `first` (output of block `first−1`, before its attn_norm) — the
    /// frozen input a multi-block fine-tuner reconstructs the last `n_layer−first` blocks on top of.
    pub fn hidden_before_block(&self, tokens: &[u32], first: usize) -> Tensor {
        use ferric_tensor::batch;
        let mut cache = Cache::new(&self.cfg);
        let mut x = self.embed(tokens);
        let pos = cache.pos;
        for (il, l) in self.layers.iter().enumerate().take(first) {
            let lc = &mut cache.kv[il];
            let xin = &x;
            x = batch(&self.ctx, || {
                let y = self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos, il);
                let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps);
                self.ffn(&xy_n, l, il).add(&xy)
            });
        }
        x
    }

    /// The hidden state ENTERING the last transformer block — `hidden_before_block(tokens, n_layer−1)`.
    pub fn block_input_last(&self, tokens: &[u32]) -> Tensor {
        self.hidden_before_block(tokens, self.layers.len() - 1)
    }

    /// The post-attention residual entering the LAST block's FFN (`x + attn(rmsnorm(x))` of the final
    /// layer) — the frozen intermediate a fine-tuner reconstructs the last FFN on top of. Every earlier
    /// block runs normally; the last block computes only attention + residual and stops before its FFN.
    /// (Non-Gemma path; Qwen3 is non-Gemma.)
    pub fn ffn_input_last(&self, tokens: &[u32]) -> Tensor {
        use ferric_tensor::batch;
        let mut cache = Cache::new(&self.cfg);
        let mut x = self.embed(tokens);
        let pos = cache.pos;
        let last = self.layers.len() - 1;
        for (il, l) in self.layers.iter().enumerate() {
            let lc = &mut cache.kv[il];
            let xin = &x;
            if il == last {
                return batch(&self.ctx, || {
                    let y = self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos, il);
                    xin.add(&y) // post-attention residual, BEFORE the FFN
                });
            }
            x = batch(&self.ctx, || {
                let y = self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos, il);
                let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps);
                self.ffn(&xy_n, l, il).add(&xy)
            });
        }
        unreachable!()
    }

    /// The final hidden state `out_norm(x)` — shape [T, n_embd] — for embedding models. No lm_head; the
    /// caller pools (last-token / mean) and L2-normalizes. A full stateless forward over `tokens`.
    pub fn forward_hidden(&self, tokens: &[u32]) -> Tensor {
        use ferric_tensor::batch;
        let mut cache = Cache::new(&self.cfg);
        let x = self.run_layers(tokens, &mut cache);
        batch(&self.ctx, || x.rmsnorm(&self.out_norm, self.cfg.eps))
    }
}

#[cfg(test)]
mod rope_type_tests {
    use super::rope_is_interleaved;

    /// Architectures served by this loader, with the rope type llama.cpp resolves for each.
    /// Audited against its `llama_model_rope_type` switch on 2026-08-15.
    const NORM_ARCHES: &[&str] = &["llama", "muse-glimmer"];
    const NEOX_ARCHES: &[&str] = &["qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "gemma4", "lfm2"];

    #[test]
    fn every_norm_arch_is_interleaved_and_every_neox_arch_is_not() {
        // Calls the SHIPPED predicate, not a copy of it. An earlier version of this test kept its own
        // copy and passed happily when the real one was mutated — a check that cannot fail is worse
        // than no check, because it also stops anyone else from writing a real one.
        for a in NORM_ARCHES {
            assert!(rope_is_interleaved(a), "{a} is NORM in llama.cpp; this loader would rotate NEOX pairs");
        }
        for a in NEOX_ARCHES {
            assert!(!rope_is_interleaved(a), "{a} is NEOX in llama.cpp; this loader would rotate NORM pairs");
        }
    }

    #[test]
    fn every_dense_arch_in_the_registry_has_an_audited_rope_type() {
        for a in NORM_ARCHES { assert!(!NEOX_ARCHES.contains(a), "{a} is in both lists"); }
        for e in crate::arch::REGISTRY {
            if e.runtime == crate::arch::Runtime::Dense {
                assert!(NORM_ARCHES.contains(&e.name) || NEOX_ARCHES.contains(&e.name),
                        "{} is served by the dense loader but its rope type was never audited", e.name);
            }
        }
    }
}
