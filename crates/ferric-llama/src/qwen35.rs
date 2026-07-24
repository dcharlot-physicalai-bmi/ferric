//! **Qwen3.5 hybrid** — the architecture behind PrismML's *Ternary Bonsai* (27B).
//!
//! 64 blocks, 3 of every 4 using **gated delta net** linear attention and the 4th using full
//! gated GQA attention. Every projection is PrismML **Q2_0 ternary** ({−1,0,+1}·scale, group-128,
//! 2.125 bpw) and stays packed on the GPU — dequantized per-block inside the matmul — which is why
//! 27B parameters occupy ~7 GB instead of the ~108 GB f32 would need.
//!
//! Conventions are taken from PrismML's llama.cpp fork (`src/models/qwen35.cpp`) and its converter
//! (`conversion/qwen.py`), because the GGUF is written to match them and they differ from the HF
//! reference in two ways that silently produce wrong numbers if assumed:
//!
//!  1. **`ssm_a` is stored already negated-and-exponentiated** (`-exp(A_log)`), so the decay gate is
//!     a plain multiply: `g = ssm_a · softplus(alpha + dt_bias)`, not `-exp(A_log)·softplus(…)`.
//!  2. **V/Z/beta/alpha/conv-V/out_proj heads are pre-permuted to *tiled* order** by the converter,
//!     because ggml broadcasts q/k across v heads with `head % n_k_heads`. HF instead keeps grouped
//!     order and uses `repeat_interleave`. Loading GGUF weights therefore requires the *tiled*
//!     broadcast — interleaving here would mismatch the on-disk permutation.
//!
//! Text-only inference is exact under standard partial RoPE: the checkpoint uses interleaved MRoPE
//! with sections [11,11,10,0], but for text every position component is equal, so all sections
//! rotate by the same angle and MRoPE collapses to ordinary RoPE over the first `n_rot` dims.

use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
#[allow(unused_imports)] use ferric_gguf::GgufFile;
use ferric_tensor::dtype::{Q2_0Weights, Q4_KWeights, Q6_KWeights, Q8_0Weights};
use ferric_tensor::QMatrix;
use crate::qwen3::Proj; // Fused-or-Split projection: real Q4_K_M models mix quants within a fused qkv/gate
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

#[derive(Debug, Clone)]
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
    pub n_rot: usize,
    pub full_attention_interval: usize,
    // gated delta net
    pub conv_kernel: usize,
    pub n_k_heads: usize,  // ssm.group_count
    pub n_v_heads: usize,  // ssm.time_step_rank
    pub head_k_dim: usize, // ssm.state_size
    pub d_inner: usize,    // ssm.inner_size
    // mixture-of-experts (qwen35moe / laguna) — all 0 for the dense `qwen35`
    pub n_expert: usize,      // routed expert count (0 = dense FFN)
    pub n_expert_used: usize, // top-k experts per token
    pub expert_ff: usize,     // each routed expert's intermediate width
    pub shared_ff: usize,     // always-on shared expert's intermediate width
    // laguna (Poolside): sliding-window/full interleave with per-layer head counts, YaRN long-rope on
    // the full-attention layers, sigmoid router with a selection bias, softplus per-head attn gating.
    pub arch: String,
    pub n_head_arr: Vec<usize>,   // per-layer head count (constant for qwen35*)
    pub sliding_window: usize,    // 0 = no SWA layers
    pub freq_base_swa: f32,       // rope θ for sliding layers (laguna: 1e4 vs 5e5 full)
    pub n_rot_swa: usize,         // rotary dims on sliding layers (laguna: 128 vs 64 full)
    pub yarn_factor: f32,         // 0 = no YaRN; else the context-extension factor
    pub yarn_orig: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_attn_factor: f32,
    pub gating_sigmoid: bool,     // expert_gating_func == 2 (sigmoid router scores)
    pub expert_scale: f32,        // multiplier on the renormalized expert weights (laguna: 2.5)
    pub n_dense_lead: usize,      // leading_dense_block_count — first N layers use a dense FFN
    pub n_nextn: usize,           // MTP draft blocks stored after the main layers (speculative decoding)
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        // Both `qwen35` (dense) and `qwen35moe` (mixture-of-experts) share the hybrid attention/GDN; the
        // only difference is the FFN. Key prefix follows general.architecture.
        let p = match g.metadata().get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "qwen35".into() };
        let key = |k: &str| format!("{p}.{k}");
        let u = |k: &str| -> Result<usize, String> {
            match g.metadata().get(&key(k)) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing metadata {}", key(k))) }
        };
        let uo = |k: &str| -> usize { match g.metadata().get(&key(k)) { Some(Meta::U(v)) => *v as usize, _ => 0 } };
        let f = |k: &str| -> Result<f32, String> {
            match g.metadata().get(&key(k)) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing metadata {}", key(k))) }
        };
        let fo = |k: &str, d: f32| -> f32 { match g.metadata().get(&key(k)) { Some(Meta::F(v)) => *v as f32, _ => d } };
        let n_vocab = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.len(), _ => return Err("missing tokenizer.ggml.tokens".into()) };
        // Multi-token-prediction (MTP/"nextn") draft blocks are stored as the LAST `nextn_predict_layers`
        // blocks and are NOT part of the main decode graph (llama.cpp: l_out-(N-1) → output head). Running
        // them as main layers silently corrupts the final hidden state — exclude them. (Speculative
        // decoding with the draft block is a possible future use.)
        let n_nextn = uo("nextn_predict_layers");
        let n_layer = u("block_count")? - n_nextn;
        // head_count is a SCALAR for qwen35*, a PER-LAYER ARRAY for laguna (64 heads on the cheap
        // sliding layers, 48 on the full-attention ones).
        let n_head_arr: Vec<usize> = match g.metadata().get(&key("attention.head_count")) {
            Some(Meta::U(v)) => vec![*v as usize; n_layer],
            Some(Meta::Arr(a)) => a.iter().map(|m| match m { Meta::U(v) => *v as usize, Meta::I(v) => *v as usize, _ => 0 }).collect(),
            _ => return Err(format!("missing metadata {}", key("attention.head_count"))),
        };
        Ok(Cfg {
            n_embd: u("embedding_length")?,
            n_layer,
            n_head: n_head_arr[0],
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_ff: uo("feed_forward_length"), // MoE models have no single dense FFN width
            n_vocab,
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope_base: f("rope.freq_base")?,
            n_rot: u("rope.dimension_count")?,
            full_attention_interval: g.metadata().get(&key("full_attention_interval")).and_then(|m| if let Meta::U(v) = m { Some(*v as usize) } else { None }).unwrap_or(4),
            conv_kernel: uo("ssm.conv_kernel"),
            n_k_heads: uo("ssm.group_count"),
            n_v_heads: uo("ssm.time_step_rank"),
            head_k_dim: uo("ssm.state_size"),
            d_inner: uo("ssm.inner_size"),
            n_expert: uo("expert_count"),
            n_expert_used: uo("expert_used_count"),
            expert_ff: uo("expert_feed_forward_length"),
            shared_ff: uo("expert_shared_feed_forward_length"),
            arch: p.clone(),
            n_head_arr,
            sliding_window: uo("attention.sliding_window"),
            freq_base_swa: fo("rope.freq_base_swa", 0.0),
            n_rot_swa: uo("rope.dimension_count_swa"),
            yarn_factor: if matches!(g.metadata().get(&key("rope.scaling.type")), Some(Meta::Str(s)) if s == "yarn") { fo("rope.scaling.factor", 0.0) } else { 0.0 },
            yarn_orig: uo("rope.scaling.original_context_length"),
            yarn_beta_fast: fo("rope.scaling.yarn_beta_fast", 32.0),
            yarn_beta_slow: fo("rope.scaling.yarn_beta_slow", 1.0),
            yarn_attn_factor: fo("rope.scaling.yarn_attn_factor", 1.0),
            gating_sigmoid: uo("expert_gating_func") == 2,
            expert_scale: fo("expert_weights_scale", 1.0),
            n_dense_lead: uo("leading_dense_block_count"),
            n_nextn,
        })
    }
    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
    pub fn head_v_dim(&self) -> usize { if self.n_v_heads == 0 { 0 } else { self.d_inner / self.n_v_heads } }
    pub fn key_dim(&self) -> usize { self.head_k_dim * self.n_k_heads }
    /// llama.cpp: layer is recurrent (linear attention) unless it's every `interval`-th one.
    pub fn is_recurrent(&self, il: usize) -> bool { (il + 1) % self.full_attention_interval != 0 }
}

pub struct AttnW {
    pub wqkv: Proj, // wq | wk | wv stacked: one matmul, then split by q_out/k_out
    pub q_out: usize,      // n_head·head_dim·2 (query and gate, interleaved per head)
    pub kv_out: usize,     // n_head_kv·head_dim (each of k and v)
    pub wo: QMatrix,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct GdnW {
    // in_proj = qkv | z | alpha | beta stacked: the four projections all read the same h, so one
    // fused matmul replaces four (48 GDN layers × 4 → 1).
    pub in_proj: Proj,
    pub qkv_out: usize, // 2·key_dim + d_inner
    pub z_out: usize,   // d_inner
    pub ba_out: usize,  // n_v_heads (each of alpha, beta)
    pub conv1d: Tensor,  // [conv_dim, L]
    pub dt_bias: Tensor, // [n_v_heads]
    pub a: Tensor,       // [n_v_heads] — already -exp(A_log)
    pub norm: Tensor,    // [head_v_dim]
    pub out: QMatrix,
}

/// Laguna (Poolside) attention: separate q/k/v (+ per-head softplus output gate `g_proj`), QK-norm,
/// per-layer head count, and per-layer rope — plain θ_swa on sliding-window layers, YaRN-scaled θ on
/// the full-attention ones.
pub struct LagAttnW {
    pub q: QMatrix, pub k: QMatrix, pub v: QMatrix, pub o: QMatrix,
    pub gate: QMatrix,     // [n_head, n_embd] → softplus per-head gate on the attention output
    pub q_norm: Tensor, pub k_norm: Tensor,
    pub n_head: usize,     // per-layer (laguna: 48 full / 64 sliding)
    pub window: usize,     // 0 = full attention, else sliding window size
    pub base: f32,         // rope θ for this layer
    pub n_rot: usize,      // rotary dims for this layer
    /// YaRN long-rope (full-attention layers): per-dim inverse-frequency scale + the cos/sin
    /// magnitude factor `mscale = attn_factor·(1 + 0.1·ln(factor))`.
    pub yarn: Option<(Tensor, f32)>,
}

pub enum Mixer { Attn(AttnW), Gdn(GdnW), Lag(LagAttnW) }

/// Per-layer carried state. Attention layers keep the usual K/V history; gated-delta-net layers keep
/// the recurrent state plus the last `conv_kernel-1` conv inputs (the short conv's receptive field).
/// With both, generating token N costs one step instead of re-running the whole prefix.
#[derive(Clone)]
pub(crate) enum LayerCache {
    Attn { k: Tensor, v: Tensor },   // [S, n_kv·head_dim]
    Gdn { state: Tensor, conv: Tensor }, // state [n_v_heads, dv, dk]; conv [conv_kernel-1, conv_dim]
}

#[derive(Default, Clone)]
pub struct Cache {
    pub pos: usize,
    layers: Vec<Option<LayerCache>>,
}

impl Cache {
    pub fn new(cfg: &Cfg) -> Cache { Cache { pos: 0, layers: (0..cfg.n_layer).map(|_| None).collect() } }
    /// O(1) snapshot for speculative-decoding rollback: tensors are immutable Arc-shared buffers, so
    /// cloning the cache clones handles, never GPU data. Restoring = assigning the snapshot back.
    pub fn snapshot(&self) -> Cache { self.clone() }
}

/// The MTP draft block's own running state: one layer's cache plus its stream position.
/// Clone is O(1) handle copies (immutable Arc-shared tensors) — used for prompt-prefix reuse.
#[derive(Default, Clone)]
pub struct MtpCache {
    pub pos: usize,
    layer: Option<LayerCache>,
}

/// One routed/shared expert (a SwiGLU FFN): `down(swiglu(gate_up(x)))`. gate|up are byte-fused into ONE
/// QMatrix — halves the buffer count (55k+ separate expert buffers silently broke the Metal device) and
/// the per-expert dispatches.
pub struct Expert { pub gate_up: QMatrix, pub down: QMatrix }

/// The down-projection slab's quant format: Q4_K_M GGUFs alternate Q6_K and Q4_K on `down_exps`
/// per layer, so BOTH need slab kernels — a Q4_K-down layer falling back to per-expert routing
/// costs ~50 eager dispatches + a readback sync per token per layer.
pub enum DownSlab { Q6(Q6_KWeights), Q4(Q4_KWeights) }

impl DownSlab {
    fn wsum(&self, mid: &Tensor, selw: &Tensor, d: usize) -> Tensor {
        match self {
            DownSlab::Q6(w) => mid.matmul_q6_k_id_wsum(w, selw, d),
            DownSlab::Q4(w) => mid.matmul_q4_k_id_wsum(w, selw, d),
        }
    }
}

/// The routed experts' weights, in one of two layouts:
pub enum MoeExperts {
    /// ALL experts packed expert-major into ONE slab per projection — the batched-kernel fast path
    /// (one `matmul_q4_k_swiglu_id` + one down+weighted-sum dispatch for ALL tokens instead of ~2k
    /// per token). Also collapses the buffer count from ~500/layer to 4/layer. Q4_K gate|up with a
    /// Q6_K or Q4_K down (the two formats real Q4_K_M GGUFs use).
    Slab { gate_up: Q4_KWeights, down: DownSlab },
    /// All-Q8_0 slab — the MTP draft block's expert format.
    Slab8 { gate_up: Q8_0Weights, down: Q8_0Weights },
    /// One QMatrix pair per expert — the general fallback for other quant combinations.
    /// ⚠️ Routes on the CPU via a mid-forward readback: correct but slow, and the readback lands
    /// inside the caller's `batch()` — avoid for anything on the hot path.
    PerExpert(Vec<Expert>),
}

/// Mixture-of-experts FFN (qwen35moe): a softmax router picks the top-k of `experts`, each a SwiGLU FFN,
/// summed by router weight; plus an always-on `shared` expert scaled by a sigmoid gate (`sh_gate`).
pub struct MoeFfn {
    pub router: Tensor,        // [n_expert, n_embd] f32 — routed-expert gate
    pub experts: MoeExperts,   // n_expert routed experts (each [n_embd→expert_ff→n_embd])
    pub shared: Expert,        // always-on shared expert
    pub sh_gate: Option<Tensor>, // [n_embd] f32 — sigmoid(x·sh_gate) scales the shared expert (qwen35moe); laguna's shared expert is ungated
    pub n_used: usize,         // top-k
    pub sigmoid: bool,         // router scores via sigmoid (laguna/DeepSeek-style) instead of softmax
    pub sel_bias: Option<Vec<f32>>, // exp_probs_b — added to scores for SELECTION only, never weights
    pub gpu_bias: Option<Tensor>,   // the same bias resident on-GPU for the sync-free slab path
    pub scale: f32,            // multiplier on the renormalized weights (laguna: 2.5)
}

pub enum Ffn {
    Dense { gate_up: Proj, gate_out: usize, down: QMatrix }, // gate|up fused, then down
    Moe(MoeFfn),
}

pub struct Layer {
    pub attn_norm: Tensor,
    pub post_norm: Tensor,
    pub ffn: Ffn,
    pub mixer: Mixer,
}

/// The MTP ("nextn") draft block: `eh_proj(cat[enorm(embed(tokenᵢ₊₁)), hnorm(hiddenᵢ)])` feeds one
/// standard transformer block, whose output (through `head_norm`) reuses the main LM head. Drafts
/// tokenᵢ₊₂ — the self-drafter for speculative decoding.
pub struct Mtp {
    pub eh_proj: QMatrix,   // [2d → d]
    pub enorm: Tensor,      // RMSNorm weight on the token embedding
    pub hnorm: Tensor,      // RMSNorm weight on the main model's last-layer hidden
    pub head_norm: Tensor,  // pre-LM-head norm (replaces the main output_norm for drafts)
    pub layer: Layer,
}

pub struct Qwen35 {
    pub cfg: Cfg,
    ctx: Arc<Context>,
    /// Token embeddings stay packed in *host* RAM: only the prompt's rows are ever needed, so
    /// dequantizing those few rows on the CPU beats parking 338 MB on the GPU for a gather.
    tok_embd: Vec<u8>,
    emb_type: u32,       // ggml type id of token_embd (for per-row dequant — any quant, not just Q2_0)
    emb_row_bytes: usize, // packed bytes per embedding row
    pub layers: Vec<Layer>,
    pub out_norm: Tensor,
    pub lm_head: QMatrix,
    pub mtp: Option<Mtp>,
}

/// GGUF stores dims fastest-varying first, so a listed `[in, out]` is a row-major `[out, in]`
/// weight — the HF linear convention, which is what `matmul_q2_0` wants.
pub(crate) fn q2(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Q2_0Weights, String> {
    let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
    if t.ggml_type != 42 { return Err(format!("{name}: expected Q2_0 (42), got {}", t.ggml_type)); }
    let (inn, out) = (t.dims[0] as usize, t.dims[1] as usize);
    Ok(Q2_0Weights::from_bytes(ctx, &g.raw(name)?, out, inn))
}

/// Stack several same-input `[out, in]` weights along the output dim into one `[Σout, in]`.
/// Q2_0 is row-major (each output row is a contiguous run of 34-byte blocks), so stacking outputs
/// is literally concatenating the raw byte streams — no repacking. One fused matmul over the group
/// beats the separate ones at decode width (1.79× measured on gate+up), because a lone-token GEMV
/// is occupancy-starved and merging the output counts fills the machine.
pub(crate) fn q2_cat(ctx: &Arc<Context>, g: &impl GgufSource, names: &[&str]) -> Result<Q2_0Weights, String> {
    let mut inn = None;
    let mut out = 0usize;
    let mut raw = Vec::new();
    for &name in names {
        let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        if t.ggml_type != 42 { return Err(format!("{name}: expected Q2_0 (42), got {}", t.ggml_type)); }
        let i = t.dims[0] as usize;
        if *inn.get_or_insert(i) != i { return Err(format!("{name}: input dim differs")); }
        out += t.dims[1] as usize;
        raw.extend(g.raw(name)?);
    }
    Ok(Q2_0Weights::from_bytes(ctx, &raw, out, inn.unwrap()))
}

pub(crate) fn f32t(ctx: &Arc<Context>, g: &impl GgufSource, name: &str, shape: &[usize]) -> Result<Tensor, String> {
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, shape))
}

/// Load a weight in **whatever packed format** the GGUF stored it (Q2_0/Q4_0/Q4_K/Q8_0) as a QMatrix.
pub(crate) fn qm(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<ferric_tensor::QMatrix, String> {
    let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
    let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
    // Native packed kernel if we have one; otherwise dequantize to f32 and run the dense fallback
    // (e.g. IQ4_XS/IQ4_NL — the quant is decoded, just not matmul'd in packed form).
    if ferric_tensor::QMatrix::block_bytes(ty).is_some() {
        ferric_tensor::QMatrix::from_bytes(ctx, &g.raw(name)?, ty, rows, cols)
    } else {
        Ok(ferric_tensor::QMatrix::from_dense(ctx, &g.dequant(name)?, rows, cols))
    }
}

/// Concatenate several same-format weights along the output dim into one QMatrix (fused qkv, gate_up).
/// In a real GGUF every projection in a layer shares one quant format, so this just stacks their bytes.
pub(crate) fn qm_cat(ctx: &Arc<Context>, g: &impl GgufSource, names: &[&str]) -> Result<ferric_tensor::QMatrix, String> {
    let (mut inn, mut out, mut ty) = (None, 0usize, None);
    for &name in names {
        let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let this = *ty.get_or_insert(t.ggml_type);
        if t.ggml_type != this { return Err(format!("{name}: mixed quant formats in one fused matmul")); }
        let i = t.dims[0] as usize;
        if *inn.get_or_insert(i) != i { return Err(format!("{name}: input dim differs")); }
        out += t.dims[1] as usize;
    }
    let (ty, inn) = (ty.unwrap(), inn.unwrap());
    // Concatenate along the output dim: for the packed path stack raw block bytes; for the dense
    // fallback stack the dequantized row-major [out_i, inn] blocks — both yield the fused [out, inn].
    if ferric_tensor::QMatrix::block_bytes(ty).is_some() {
        let mut raw = Vec::new();
        for &name in names { raw.extend(g.raw(name)?); }
        ferric_tensor::QMatrix::from_bytes(ctx, &raw, ty, out, inn)
    } else {
        let mut f = Vec::new();
        for &name in names { f.extend(g.dequant(name)?); }
        Ok(ferric_tensor::QMatrix::from_dense(ctx, &f, out, inn))
    }
}

/// YaRN long-rope per-dim inverse-frequency scale (llama.cpp/ggml formulation): for pair-dim i, the
/// final θᵢ interpolates between the extrapolated (original) frequency and the /factor-compressed one,
/// ramped between the correction dims derived from β_fast/β_slow. Returned as the multiplier the
/// `rope_scaled` kernel applies to each standard inverse frequency.
fn yarn_freq_scale(n_rot: usize, base: f32, factor: f32, orig_ctx: usize, beta_fast: f32, beta_slow: f32) -> Vec<f32> {
    let corr_dim = |beta: f32| -> f32 {
        (n_rot as f32) * ((orig_ctx as f32) / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
    };
    let low = corr_dim(beta_fast).floor().max(0.0);
    let high = corr_dim(beta_slow).ceil().min(n_rot as f32 - 1.0);
    (0..n_rot / 2).map(|i| {
        // ggml rope_yarn_ramp on the pair index: 1 at low (high-frequency) dims, 0 at high dims.
        let y = (i as f32 - low) / (high - low).max(0.001);
        let ramp_mix = 1.0 - y.clamp(0.0, 1.0);
        // θ = θ_interp·(1−ramp) + θ_extrap·ramp  ⇒  scale = (1/factor)·(1−ramp) + ramp
        (1.0 / factor) * (1.0 - ramp_mix) + ramp_mix
    }).collect()
}

/// Load a layer's FFN — a dense SwiGLU (qwen35), or a mixture-of-experts (qwen35moe): softmax router +
/// `n_expert` routed SwiGLU experts + an always-on sigmoid-gated shared expert. The stacked 3D `*_exps`
/// tensors slice cleanly per expert (each expert is whole rows, so quant-block boundaries are respected).
fn load_ffn(ctx: &Arc<Context>, g: &impl GgufSource, il: usize, cfg: &Cfg) -> Result<Ffn, String> {
    let b = |s: &str| format!("blk.{il}.{s}");
    // Dense FFN: non-MoE models, and a MoE model's leading dense blocks (laguna: layer 0).
    if !cfg.is_moe() || il < cfg.n_dense_lead {
        return Ok(Ffn::Dense {
            gate_up: Proj::load(ctx, g, &[&b("ffn_gate.weight"), &b("ffn_up.weight")])?,
            gate_out: g.tensor(&b("ffn_gate.weight")).ok_or("no ffn_gate")?.dims[1] as usize,
            down: qm(ctx, g, &b("ffn_down.weight"))?,
        });
    }
    let (ne, eff, d) = (cfg.n_expert, cfg.expert_ff, cfg.n_embd);
    // Slice a stacked [inn, out, n_expert] tensor into one QMatrix per expert (expert = slowest dim →
    // each expert's [out, inn] plane is a contiguous byte range of len total/n_expert).
    let experts_of = |name: &str, out: usize, inn: usize| -> Result<Vec<QMatrix>, String> {
        let ty = g.tensor(name).ok_or_else(|| format!("no {name}"))?.ggml_type;
        let full = g.raw(name)?;
        let per = full.len() / ne;
        (0..ne).map(|e| {
            let s = &full[e * per..(e + 1) * per];
            if QMatrix::block_bytes(ty).is_some() { QMatrix::from_bytes(ctx, s, ty, out, inn) }
            else { Ok(QMatrix::from_dense(ctx, &ferric_gguf::deq_raw(s, out * inn, ty)?, out, inn)) }
        }).collect()
    };
    // Fuse each expert's gate|up slabs into one QMatrix [2·eff, d] (they share the input and, in every
    // real GGUF, the quant format); `swiglu(eff)` splits the fused output. Down stays per-expert.
    let (gt, ut) = (g.tensor(&b("ffn_gate_exps.weight")).ok_or("no gate_exps")?.ggml_type,
                    g.tensor(&b("ffn_up_exps.weight")).ok_or("no up_exps")?.ggml_type);
    if gt != ut { return Err(format!("blk.{il}: gate/up expert quants differ ({gt} vs {ut})")); }
    let g_full = g.raw(&b("ffn_gate_exps.weight"))?;
    let u_full = g.raw(&b("ffn_up_exps.weight"))?;
    let (g_per, u_per) = (g_full.len() / ne, u_full.len() / ne);
    let dt = g.tensor(&b("ffn_down_exps.weight")).ok_or("no down_exps")?.ggml_type;
    // Fast path: pack ALL experts into one slab per projection for the batched selected-expert
    // kernels. gate|up interleaved per expert (matching swiglu_id's `base + o` / `base + eff + o`
    // addressing); the down tensor is already expert-major on disk — its raw bytes ARE the slab.
    let experts = if gt == 12 && (dt == 14 || dt == 12) {
        // Q4_K gate|up + Q6_K or Q4_K down (Q4_K_M alternates the down format per layer)
        let mut gu = Vec::with_capacity(g_full.len() + u_full.len());
        for e in 0..ne {
            gu.extend_from_slice(&g_full[e * g_per..(e + 1) * g_per]);
            gu.extend_from_slice(&u_full[e * u_per..(e + 1) * u_per]);
        }
        let d_full = g.raw(&b("ffn_down_exps.weight"))?;
        MoeExperts::Slab {
            gate_up: Q4_KWeights::from_bytes(ctx, &gu, ne * 2 * eff, d),
            down: if dt == 14 { DownSlab::Q6(Q6_KWeights::from_bytes(ctx, &d_full, ne * d, eff)) }
                  else { DownSlab::Q4(Q4_KWeights::from_bytes(ctx, &d_full, ne * d, eff)) },
        }
    } else if gt == 8 && dt == 8 {
        // All-Q8_0 experts (the MTP draft block): same slab layout, Q8_0 kernels.
        let mut gu = Vec::with_capacity(g_full.len() + u_full.len());
        for e in 0..ne {
            gu.extend_from_slice(&g_full[e * g_per..(e + 1) * g_per]);
            gu.extend_from_slice(&u_full[e * u_per..(e + 1) * u_per]);
        }
        let d_full = g.raw(&b("ffn_down_exps.weight"))?;
        MoeExperts::Slab8 {
            gate_up: Q8_0Weights::from_bytes(ctx, &gu, ne * 2 * eff, d),
            down: Q8_0Weights::from_bytes(ctx, &d_full, ne * d, eff),
        }
    } else {
        let down = experts_of(&b("ffn_down_exps.weight"), d, eff)?;
        MoeExperts::PerExpert(down.into_iter().enumerate().map(|(e, down)| {
            let mut bytes = Vec::with_capacity(g_per + u_per);
            bytes.extend_from_slice(&g_full[e * g_per..(e + 1) * g_per]);
            bytes.extend_from_slice(&u_full[e * u_per..(e + 1) * u_per]);
            let gate_up = if QMatrix::block_bytes(gt).is_some() { QMatrix::from_bytes(ctx, &bytes, gt, 2 * eff, d)? }
                else { QMatrix::from_dense(ctx, &ferric_gguf::deq_raw(&bytes, 2 * eff * d, gt)?, 2 * eff, d) };
            Ok(Expert { gate_up, down })
        }).collect::<Result<Vec<_>, String>>()?)
    };
    // Shared-expert gate (qwen35moe) and expert-selection bias (laguna) are arch-dependent extras —
    // presence-detected, like everything else.
    let sh_gate = if g.tensor(&b("ffn_gate_inp_shexp.weight")).is_some() {
        Some(f32t(ctx, g, &b("ffn_gate_inp_shexp.weight"), &[d])?)
    } else { None };
    let sel_bias = if g.tensor(&b("exp_probs_b.bias")).is_some() {
        Some(g.dequant(&b("exp_probs_b.bias"))?)
    } else { None };
    let gpu_bias = sel_bias.as_ref().map(|v| Tensor::from_vec(ctx, v, &[v.len()]));
    Ok(Ffn::Moe(MoeFfn {
        router: f32t(ctx, g, &b("ffn_gate_inp.weight"), &[ne, d])?,
        experts,
        shared: Expert {
            gate_up: qm_cat(ctx, g, &[&b("ffn_gate_shexp.weight"), &b("ffn_up_shexp.weight")])?,
            down: qm(ctx, g, &b("ffn_down_shexp.weight"))?,
        },
        sh_gate,
        n_used: cfg.n_expert_used,
        sigmoid: cfg.gating_sigmoid,
        sel_bias,
        gpu_bias,
        scale: cfg.expert_scale,
    }))
}

impl Qwen35 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Qwen35, String> {
        let mut cfg = Cfg::from_gguf(g)?;
        // Debug aid: FERRIC_MAX_LAYERS=N truncates the model (e.g. to isolate GPU-resource-limit issues).
        if let Ok(n) = std::env::var("FERRIC_MAX_LAYERS") { if let Ok(n) = n.parse::<usize>() { cfg.n_layer = cfg.n_layer.min(n); } }
        let conv_dim = cfg.key_dim() * 2 + cfg.d_inner;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            layers.push(Self::load_layer(ctx, g, il, &cfg, conv_dim)?);
            // MoE layers allocate ~500 buffers (~300 MB) each; flush per layer so pending buffer
            // initializations commit — past ~10 GB un-flushed, Metal silently zeroes later buffers.
            if cfg.is_moe() { ctx.flush(); }
        }

        // The MTP ("nextn") draft block: a standard attn+FFN layer stored after the main layers, plus
        // the eh_proj/enorm/hnorm glue and its own pre-head norm. Loaded for speculative decoding;
        // it shares the main embedding and LM head.
        let mtp = if cfg.n_nextn > 0 {
            let il = cfg.n_layer + cfg.n_nextn - 1; // draft block index (40 for qwen35moe)
            let b = |s: &str| format!("blk.{il}.{s}");
            if g.tensor(&b("nextn.eh_proj.weight")).is_some() {
                let m = Mtp {
                    eh_proj: qm(ctx, g, &b("nextn.eh_proj.weight"))?,
                    enorm: f32t(ctx, g, &b("nextn.enorm.weight"), &[cfg.n_embd])?,
                    hnorm: f32t(ctx, g, &b("nextn.hnorm.weight"), &[cfg.n_embd])?,
                    head_norm: f32t(ctx, g, &b("nextn.shared_head_norm.weight"), &[cfg.n_embd])?,
                    layer: Self::load_layer(ctx, g, il, &cfg, conv_dim)?,
                };
                if cfg.is_moe() { ctx.flush(); }
                Some(m)
            } else { None }
        } else { None };

        // Tied head: if `output.weight` is absent the embeddings double as the LM head.
        let head = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let emb = g.tensor("token_embd.weight").ok_or("no token_embd")?;
        let emb_type = emb.ggml_type;
        let tok_embd = g.raw("token_embd.weight")?;
        let emb_row_bytes = tok_embd.len() / cfg.n_vocab;
        Ok(Qwen35 {
            tok_embd, emb_type, emb_row_bytes,
            out_norm: f32t(ctx, g, "output_norm.weight", &[cfg.n_embd])?,
            lm_head: qm(ctx, g, head)?,
            cfg, ctx: ctx.clone(), layers, mtp,
        })
    }

    /// One transformer block's weights (mixer feature-detected from tensor presence) — shared by the
    /// main-layer loop and the MTP draft block, which is a standard block plus glue tensors.
    fn load_layer(ctx: &Arc<Context>, g: &impl GgufSource, il: usize, cfg: &Cfg, conv_dim: usize) -> Result<Layer, String> {
        {
            let b = |s: &str| format!("blk.{il}.{s}");
            // Feature-detect the mixer from tensor presence, not the interval formula: qwen35moe makes
            // the FINAL layer full-attention regardless of `full_attention_interval` (41 layers ⇒ blk.40
            // is ATTN even though (40+1)%4≠0), and presence is always ground truth.
            let mixer = if cfg.arch == "laguna" {
                // laguna: full attention (48 heads, YaRN θ) at il%4==0, sliding-512 (64 heads, plain
                // θ_swa) elsewhere — the per-layer head_count array encodes the same pattern.
                let full = il % cfg.full_attention_interval == 0;
                let (base, n_rot) = if full { (cfg.rope_base, cfg.n_rot) } else { (cfg.freq_base_swa, cfg.n_rot_swa) };
                let yarn = if full && cfg.yarn_factor > 0.0 {
                    let scale = yarn_freq_scale(n_rot, base, cfg.yarn_factor, cfg.yarn_orig, cfg.yarn_beta_fast, cfg.yarn_beta_slow);
                    let mscale = cfg.yarn_attn_factor * (1.0 + 0.1 * cfg.yarn_factor.ln());
                    Some((Tensor::from_vec(ctx, &scale, &[n_rot / 2]), mscale))
                } else { None };
                Mixer::Lag(LagAttnW {
                    q: qm(ctx, g, &b("attn_q.weight"))?,
                    k: qm(ctx, g, &b("attn_k.weight"))?,
                    v: qm(ctx, g, &b("attn_v.weight"))?,
                    o: qm(ctx, g, &b("attn_output.weight"))?,
                    gate: qm(ctx, g, &b("attn_gate.weight"))?,
                    q_norm: f32t(ctx, g, &b("attn_q_norm.weight"), &[cfg.head_dim])?,
                    k_norm: f32t(ctx, g, &b("attn_k_norm.weight"), &[cfg.head_dim])?,
                    n_head: cfg.n_head_arr[il],
                    window: if full { 0 } else { cfg.sliding_window },
                    base, n_rot, yarn,
                })
            } else if g.tensor(&b("ssm_out.weight")).is_some() {
                Mixer::Gdn(GdnW {
                    in_proj: Proj::load(ctx, g, &[&b("attn_qkv.weight"), &b("attn_gate.weight"), &b("ssm_alpha.weight"), &b("ssm_beta.weight")])?,
                    qkv_out: g.tensor(&b("attn_qkv.weight")).ok_or("no attn_qkv")?.dims[1] as usize,
                    z_out: g.tensor(&b("attn_gate.weight")).ok_or("no attn_gate")?.dims[1] as usize,
                    ba_out: g.tensor(&b("ssm_alpha.weight")).ok_or("no ssm_alpha")?.dims[1] as usize,
                    conv1d: f32t(ctx, g, &b("ssm_conv1d.weight"), &[conv_dim, cfg.conv_kernel])?,
                    dt_bias: f32t(ctx, g, &b("ssm_dt.bias"), &[cfg.n_v_heads])?,
                    a: f32t(ctx, g, &b("ssm_a"), &[cfg.n_v_heads])?,
                    norm: f32t(ctx, g, &b("ssm_norm.weight"), &[cfg.head_v_dim()])?,
                    out: qm(ctx, g, &b("ssm_out.weight"))?,
                })
            } else {
                Mixer::Attn(AttnW {
                    wqkv: Proj::load(ctx, g, &[&b("attn_q.weight"), &b("attn_k.weight"), &b("attn_v.weight")])?,
                    q_out: g.tensor(&b("attn_q.weight")).ok_or("no attn_q")?.dims[1] as usize,
                    kv_out: g.tensor(&b("attn_k.weight")).ok_or("no attn_k")?.dims[1] as usize,
                    wo: qm(ctx, g, &b("attn_output.weight"))?,
                    q_norm: f32t(ctx, g, &b("attn_q_norm.weight"), &[cfg.head_dim])?,
                    k_norm: f32t(ctx, g, &b("attn_k_norm.weight"), &[cfg.head_dim])?,
                })
            };
            // The pre-FFN norm on the residual: qwen35 names it `post_attention_norm`, laguna `ffn_norm`
            // — same role, same placement.
            let pn = if g.tensor(&b("post_attention_norm.weight")).is_some() { b("post_attention_norm.weight") } else { b("ffn_norm.weight") };
            Ok(Layer {
                attn_norm: f32t(ctx, g, &b("attn_norm.weight"), &[cfg.n_embd])?,
                post_norm: f32t(ctx, g, &pn, &[cfg.n_embd])?,
                ffn: load_ffn(ctx, g, il, cfg)?,
                mixer,
            })
        }
    }

    /// Dequantize just the rows the prompt touches, straight out of the packed blocks — for whatever
    /// quant the token-embedding tensor is stored in (Q2_0 for Bonsai, Q4_K for Qwen3.6-27B, …).
    pub fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.n_embd;
        let rb = self.emb_row_bytes;
        if std::env::var("FERRIC_EMB_DEBUG").is_ok() {
            eprintln!("embed dbg: d={d} rb={rb} emb_type={} tok_embd_len={} n_vocab={} len/nv={}",
                self.emb_type, self.tok_embd.len(), self.cfg.n_vocab, self.tok_embd.len() / self.cfg.n_vocab);
        }
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let off = t as usize * rb;
            v.extend(ferric_gguf::deq_raw(&self.tok_embd[off..off + rb], d, self.emb_type).expect("embed row"));
        }
        if std::env::var("FERRIC_EMB_DEBUG").is_ok() {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("embed dbg: cpu-side v norm={norm:.4} first3={:?}", &v[..3.min(v.len())]);
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// Rotate only the first `n_rot` of each head's dims, passing the rest through untouched.
    fn rope_partial(&self, x: &Tensor, n_heads: usize, offset: usize) -> Tensor {
        let (t, hd, n_rot) = (x.shape[0], self.cfg.head_dim, self.cfg.n_rot);
        let x3 = x.reshape(&[t, n_heads, hd]);
        let rot = x3.narrow(2, 0, n_rot).contiguous().reshape(&[t, n_heads * n_rot])
            .rope(n_heads, n_rot, self.cfg.rope_base, offset)
            .reshape(&[t, n_heads, n_rot]);
        rot.cat(&x3.narrow(2, n_rot, hd - n_rot), 2).reshape(&[t, n_heads * hd])
    }

    fn attn(&self, h: &Tensor, w: &AttnW, cache: &mut Option<LayerCache>, offset: usize) -> Tensor {
        let (t, hd, nh) = (h.shape[0], self.cfg.head_dim, self.cfg.n_head);
        let nkv = self.cfg.n_head_kv;
        // One fused matmul emits [q_and_gate | k | v]; split it back out.
        let qkv = w.wqkv.matmul(h);
        let qf = qkv.narrow(1, 0, w.q_out).reshape(&[t, nh, hd * 2]);
        let q = qf.narrow(2, 0, hd).rmsnorm(&w.q_norm, self.cfg.eps).reshape(&[t, nh * hd]);
        let gate = qf.narrow(2, hd, hd).contiguous().reshape(&[t, nh * hd]);

        let k = qkv.narrow(1, w.q_out, w.kv_out).reshape(&[t, nkv, hd]).rmsnorm(&w.k_norm, self.cfg.eps).reshape(&[t, nkv * hd]);
        let v = qkv.narrow(1, w.q_out + w.kv_out, w.kv_out).contiguous();

        let q = self.rope_partial(&q, nh, offset);
        let k = self.rope_partial(&k, nkv, offset);

        // Append this step's K/V to the history, then attend over all of it.
        let (kc, vc) = match cache.take() {
            Some(LayerCache::Attn { k: pk, v: pv }) => (pk.cat(&k, 0), pv.cat(&v, 0)),
            _ => (k, v),
        };
        let s = kc.shape[0];
        let noflash = std::env::var("FERRIC_NOFLASH").is_ok();
        let o = if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv, 0.0)
        } else if t == s && s <= 65535 && hd <= 128 && !noflash {
            // Prefill: q and the history are the same span. Flash streams keys in chunks with
            // online softmax — O(hd) memory, any T, same math as the composed causal path.
            q.flash_attention_prefill(&kc, &vc, nh, nkv, hd)
        } else if t == s {
            nn::causal_attention(&q, &kc, &vc, nh, nkv, 0.0)
        } else {
            // t new queries against a longer cache — the speculative-verify shape.
            nn::causal_attention_kv(&q, &kc, &vc, nh, nkv, 0.0)
        };
        *cache = Some(LayerCache::Attn { k: kc, v: vc });
        o.mul(&gate.sigmoid()).matmul_q(&w.wo)
    }

    fn gdn(&self, h: &Tensor, w: &GdnW, cache: &mut Option<LayerCache>) -> Tensor {
        let c = &self.cfg;
        let (t, nk, nv) = (h.shape[0], c.n_k_heads, c.n_v_heads);
        let (dk, dv, kd) = (c.head_k_dim, c.head_v_dim(), c.key_dim());

        // One fused matmul emits [qkv | z | alpha | beta]; the three fused prep kernels read their
        // slices straight out of it — no narrows/copies. The conv is causal and depthwise over the
        // fused q|k|v block, looking back conv_kernel-1 steps: the carried tail (zeros at the start
        // of a sequence — exactly the standalone conv's causal zero-padding) is prepended virtually
        // inside the kernel, which also emits the next tail and the (conv'd, silu'd) V block.
        let proj = w.in_proj.matmul(h);
        let (qo, zo) = (w.qkv_out, w.z_out);
        let pad = c.conv_kernel - 1;
        let (prev_conv, prev_state) = match cache.take() {
            Some(LayerCache::Gdn { state, conv }) => (conv, Some(state)),
            _ => (Tensor::zeros(&self.ctx, &[pad, qo]), None),
        };
        // 3 dispatches replace the ~15 small ops (cat/conv/narrow/silu/l2norm/tile/softplus/
        // sigmoid/cat) that dominated hybrid decode:
        //   gdn_conv — silu(depthwise conv) + carried tail + V slice
        //   gdn_gate — g = ssm_a·softplus(alpha+dt_bias), β = sigmoid(beta), packed [T, nv, 2]
        //   gdn_qk   — per-head l2norm, 1/√dv folded into q, tiled across v heads (head % nk)
        let (conv, conv_tail, v) = nn::gdn_conv(&proj, &prev_conv, &w.conv1d, qo, c.conv_kernel, c.d_inner, 2 * kd);
        let gb = nn::gdn_gate(&proj, &w.dt_bias, &w.a, nv, qo + zo);
        let (q, k) = nn::gdn_qk(&conv, nk, dk, nv / nk, qo, 1.0 / (dv as f32).sqrt(), c.eps);
        let v = v.reshape(&[t, nv, dv]);

        let (o, state) = q.gated_delta_rule_stateful(&k, &v, &gb, nv, dk, dv, prev_state.as_ref());
        *cache = Some(LayerCache::Gdn { state, conv: conv_tail });

        // gated RMSNorm over head_v_dim gated by silu(z) — fused, z read in place from the in_proj
        nn::gdn_post(&o, &proj, &w.norm, qo, c.eps).matmul_q(&w.out)
    }

    /// Laguna attention: GQA with QK-norm, per-layer rope (plain or YaRN-scaled, partial rotary), a
    /// sliding-window mask on 3 of every 4 layers, and a per-head softplus gate on the output.
    fn lag_attn(&self, h: &Tensor, w: &LagAttnW, cache: &mut Option<LayerCache>, offset: usize) -> Tensor {
        let (t, hd) = (h.shape[0], self.cfg.head_dim);
        let (nh, nkv) = (w.n_head, self.cfg.n_head_kv);
        let q = h.matmul_q(&w.q).reshape(&[t, nh, hd]).rmsnorm(&w.q_norm, self.cfg.eps).reshape(&[t, nh * hd]);
        let k = h.matmul_q(&w.k).reshape(&[t, nkv, hd]).rmsnorm(&w.k_norm, self.cfg.eps).reshape(&[t, nkv * hd]);
        let v = h.matmul_q(&w.v).contiguous();
        let q = self.lag_rope(&q, nh, offset, w);
        let k = self.lag_rope(&k, nkv, offset, w);
        let (kc, vc) = match cache.take() {
            Some(LayerCache::Attn { k: pk, v: pv }) => (pk.cat(&k, 0), pv.cat(&v, 0)),
            _ => (k, v),
        };
        let s = kc.shape[0];
        let o = if w.window > 0 {
            if t == 1 { nn::decode_attention_win(&q, &kc, &vc, nh, nkv, w.window, 0.0) }
            else { nn::causal_attention_win(&q, &kc, &vc, nh, nkv, w.window, 0.0) }
        } else if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv, 0.0)
        } else if t == s && s <= 65535 && hd <= 128 {
            q.flash_attention_prefill(&kc, &vc, nh, nkv, hd)
        } else {
            nn::causal_attention(&q, &kc, &vc, nh, nkv, 0.0)
        };
        *cache = Some(LayerCache::Attn { k: kc, v: vc });
        // per-head softplus output gate, broadcast over head_dim
        let gate = h.matmul_q(&w.gate).softplus();  // [t, nh]
        let o = o.reshape(&[t, nh, hd]).mul(&gate.reshape(&[t, nh, 1]).broadcast_to(&[t, nh, hd])).reshape(&[t, nh * hd]);
        o.matmul_q(&w.o)
    }

    /// Per-layer partial rope with optional YaRN: rotate the first `n_rot` dims of each head with this
    /// layer's θ (frequency-scaled + mscale'd when YaRN is on), pass the rest through.
    fn lag_rope(&self, x: &Tensor, n_heads: usize, offset: usize, w: &LagAttnW) -> Tensor {
        let (t, hd, n_rot) = (x.shape[0], self.cfg.head_dim, w.n_rot);
        let rope1 = |r: &Tensor| -> Tensor {
            match &w.yarn {
                Some((fs, ms)) => {
                    let rot = r.rope_scaled(fs, n_heads, n_rot, w.base, offset);
                    rot.mul(&Tensor::from_vec(&self.ctx, &[*ms], &[1, 1]).broadcast_to(&[t, n_heads * n_rot]))
                }
                None => r.rope(n_heads, n_rot, w.base, offset),
            }
        };
        if n_rot == hd { return rope1(x); }
        let x3 = x.reshape(&[t, n_heads, hd]);
        let rot = rope1(&x3.narrow(2, 0, n_rot).contiguous().reshape(&[t, n_heads * n_rot])).reshape(&[t, n_heads, n_rot]);
        rot.cat(&x3.narrow(2, n_rot, hd - n_rot), 2).reshape(&[t, n_heads * hd])
    }

    fn ffn(&self, h: &Tensor, l: &Layer) -> Tensor {
        match &l.ffn {
            // gate_up matmul → fused SwiGLU (silu(gate)·up in one kernel) → down projection.
            Ffn::Dense { gate_up, gate_out, down } => gate_up.gate_up_swiglu(h, *gate_out).matmul_q(down),
            Ffn::Moe(m) => self.moe_ffn(h, m),
        }
    }

    /// One SwiGLU expert: `down(swiglu(gate_up(x)))` — `ff` is the split point of the fused gate|up output.
    fn expert(&self, h: &Tensor, e: &Expert, ff: usize) -> Tensor {
        h.matmul_q(&e.gate_up).swiglu(ff).matmul_q(&e.down)
    }

    /// Mixture-of-experts FFN (each token routes independently): softmax the router logits over all
    /// experts, take the top-k, renormalize their weights (Qwen's norm_topk_prob), run only those k
    /// experts, and add the always-on shared expert scaled by `sigmoid(x·sh_gate)`.
    fn moe_ffn(&self, h: &Tensor, m: &MoeFfn) -> Tensor {
        let (t, d) = (h.shape[0], self.cfg.n_embd);
        // Slab fast path: ALL tokens in 4 dispatches, fully on-GPU. router [ne,d]·hᵀ → per-token
        // top-k rows [T,2k] (scores+selection+renormed weights, no CPU readback) → ONE batched
        // gate|up+swiglu dispatch [T,k,eff] → ONE down+weighted-sum dispatch [T,d]. The shared
        // expert and its sigmoid gate are plain matmuls that batch over rows already. Zero syncs,
        // and the dispatch count is independent of T — a multi-token speculative-verify forward
        // costs the same FFN dispatches as single-token decode.
        if !matches!(&m.experts, MoeExperts::PerExpert(_)) {
            let k = m.n_used;
            // h·routerᵀ directly (no transpose materialized — a permute+contiguous here measured a
            // fixed ~17 ms/layer batch-splitting stall): logits [T, ne], token-major, which is also
            // the layout moe_topk's per-token thread scans contiguously.
            let selw = h.matmul_bt(&m.router).moe_topk(m.gpu_bias.as_ref(), k, m.sigmoid, m.scale); // [T, 2k]
            let routed = match &m.experts {
                MoeExperts::Slab { gate_up, down } => {
                    let mid = h.matmul_q4_k_swiglu_id(gate_up, &selw, k, self.cfg.expert_ff); // [T, k, eff]
                    down.wsum(&mid, &selw, d)                                                 // [T, d]
                }
                MoeExperts::Slab8 { gate_up, down } => {
                    let mid = h.matmul_q8_0_swiglu_id(gate_up, &selw, k, self.cfg.expert_ff);
                    mid.matmul_q8_0_id_wsum(down, &selw, d)
                }
                MoeExperts::PerExpert(_) => unreachable!(),
            };
            let sh = self.expert(h, &m.shared, self.cfg.shared_ff);
            let sh = match &m.sh_gate {
                Some(sg) => {
                    let gate = h.matmul_bt(&sg.reshape(&[1, d])).sigmoid().broadcast_to(&[t, d]); // [T,1]→[T,d]
                    sh.mul(&gate)
                }
                None => sh,
            };
            return routed.add(&sh);
        }
        let mut rows: Vec<Tensor> = Vec::with_capacity(t);
        for ti in 0..t {
            let h_t = h.narrow(0, ti, 1).contiguous(); // [1, d]
            let routed = match &m.experts {
                MoeExperts::Slab { .. } | MoeExperts::Slab8 { .. } => unreachable!("slab path handled above"),
                // General fallback (other quant combos): CPU routing via one small readback.
                MoeExperts::PerExpert(experts) => {
                    let lg = pollster::block_on(m.router.matmul(&h_t.reshape(&[d, 1])).to_vec());
                    let probs: Vec<f32> = if m.sigmoid {
                        lg.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
                    } else {
                        let maxl = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = lg.iter().map(|x| (x - maxl).exp()).collect();
                        let sum: f32 = e.iter().sum();
                        e.into_iter().map(|x| x / sum).collect()
                    };
                    let selp: Vec<f32> = match &m.sel_bias {
                        Some(bias) => probs.iter().zip(bias).map(|(p, b)| p + b).collect(),
                        None => probs.clone(),
                    };
                    let mut idx: Vec<usize> = (0..selp.len()).collect();
                    idx.sort_by(|&a, &b| selp[b].partial_cmp(&selp[a]).unwrap());
                    let sel = &idx[..m.n_used.min(idx.len())];
                    let wsum: f32 = sel.iter().map(|&e| probs[e]).sum();
                    let mut acc: Option<Tensor> = None;
                    for &e in sel {
                        let w = probs[e] / wsum * m.scale;
                        // scale via a broadcast [1,1] scalar — a 4-byte upload instead of an n_embd-sized one
                        let ws = Tensor::from_vec(&self.ctx, &[w], &[1, 1]).broadcast_to(&[1, d]);
                        let y = self.expert(&h_t, &experts[e], self.cfg.expert_ff).mul(&ws);
                        acc = Some(match acc { Some(a) => a.add(&y), None => y });
                    }
                    acc.expect("top-k is at least 1")
                }
            };
            // shared expert — sigmoid-gated for qwen35moe (GPU-side, no readback), plain for laguna
            let sh = self.expert(&h_t, &m.shared, self.cfg.shared_ff);
            let sh = match &m.sh_gate {
                Some(sg) => {
                    let gate = sg.reshape(&[1, d]).matmul(&h_t.reshape(&[d, 1])).sigmoid().broadcast_to(&[1, d]);
                    sh.mul(&gate)
                }
                None => sh,
            };
            rows.push(routed.add(&sh));
        }
        let mut it = rows.into_iter();
        let mut o = it.next().expect("moe_ffn: empty input");
        for r in it { o = o.cat(&r, 0); }
        o
    }

    /// Prefill forward over `tokens` → logits [T, n_vocab]. Stateless (allocates a throwaway cache).
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        self.forward_upto(tokens, self.cfg.n_layer)
    }

    /// Forward through the first `n` layers (then final norm + head) — `n < n_layer` is how the
    /// per-layer comparison against llama.cpp is done.
    pub fn forward_upto(&self, tokens: &[u32], n: usize) -> Tensor {
        let mut cache = Cache::new(&self.cfg);
        self.forward_cached(tokens, &mut cache, n)
    }

    /// Feed `tokens` through the model, carrying every layer's state in `cache`. Call it once with
    /// the prompt, then once per generated token — the incremental result is identical to
    /// re-running the whole prefix, because both the attention K/V and the gated-delta-net
    /// recurrence resume exactly where they left off.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut Cache, n: usize) -> Tensor {
        // Batch each layer's ~10 dispatches into one submission — cutting ~640 submits/token to
        // ~70 and removing most of the per-op encoder+submit overhead (measured ~38 ms/token).
        // Batching is *per layer*, not whole-forward: one command buffer must retain every bind
        // group it records until submit, and holding all 640 across 64 layers exhausts the driver's
        // per-submission resource budget. Ops still run in issue order, so the result is identical.
        use ferric_tensor::{batch, prof};
        let mut x = self.embed(tokens);
        prof(&self.ctx, "embed");
        let pos = cache.pos;
        // FERRIC_PROFILE splits each layer into per-category submissions so the sync'd timer can
        // attribute time (mixer vs ffn); otherwise the whole layer is one batch (fewer submits).
        let profiling = std::env::var("FERRIC_PROFILE").is_ok();
        for (il, l) in self.layers.iter().enumerate().take(n) {
            let lc = &mut cache.layers[il];
            let xin = &x;
            if profiling {
                // One submit per category (mixer, then ffn) so the sync'd timer attributes GPU work,
                // not op count. Eager-per-op would over-charge whichever region has the most small
                // dispatches; per-category batching keeps the split honest.
                let is_attn = matches!(l.mixer, Mixer::Attn(_));
                let y = batch(&self.ctx, || {
                    let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                    match &l.mixer { Mixer::Attn(w) => self.attn(&h, w, lc, pos), Mixer::Gdn(w) => self.gdn(&h, w, lc), Mixer::Lag(w) => self.lag_attn(&h, w, lc, pos) }
                });
                prof(&self.ctx, if is_attn { "attn" } else { "gdn" });
                let xy = xin.add(&y);
                x = batch(&self.ctx, || self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy));
                prof(&self.ctx, "ffn");
            } else {
                x = batch(&self.ctx, || {
                    let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                    let y = match &l.mixer {
                        Mixer::Attn(w) => self.attn(&h, w, lc, pos),
                        Mixer::Gdn(w) => self.gdn(&h, w, lc),
                        Mixer::Lag(w) => self.lag_attn(&h, w, lc, pos),
                    };
                    let xy = xin.add(&y);
                    self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy)
                });
            }
            if std::env::var("FERRIC_LAYER_SUMS").is_ok() {
                let v = pollster::block_on(x.to_vec());
                let d = self.cfg.n_embd;
                let lr = &v[(v.len() / d - 1) * d..];
                eprintln!("l_out-{il} sum={:.6} lastrow3=[{:.4}, {:.4}, {:.4}]", v.iter().sum::<f32>(), lr[0], lr[1], lr[2]);
            }
        }
        cache.pos += tokens.len();
        let out = batch(&self.ctx, || x.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q(&self.lm_head));
        prof(&self.ctx, "lm_head");
        if std::env::var("FERRIC_LAYER_SUMS").is_ok() {
            let rn: f32 = pollster::block_on(x.rmsnorm(&self.out_norm, self.cfg.eps).to_vec()).iter().sum();
            let ro: f32 = pollster::block_on(out.to_vec()).iter().sum();
            eprintln!("result_norm {rn:.6}\nresult_output {ro:.3}");
        }
        out
    }

    /// The pre-lm-head hidden state `x` [T, n_embd] (before the final norm) — for head-stage debugging
    /// and, later, embedding-style pooling. Same layer loop as `forward_cached`, no head.
    pub fn forward_hidden_cached(&self, tokens: &[u32], cache: &mut Cache, n: usize) -> Tensor {
        use ferric_tensor::batch;
        let mut x = self.embed(tokens);
        let pos = cache.pos;
        for (il, l) in self.layers.iter().enumerate().take(n) {
            let lc = &mut cache.layers[il];
            let xin = &x;
            x = batch(&self.ctx, || {
                let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                let y = match &l.mixer {
                    Mixer::Attn(w) => self.attn(&h, w, lc, pos),
                    Mixer::Gdn(w) => self.gdn(&h, w, lc),
                    Mixer::Lag(w) => self.lag_attn(&h, w, lc, pos),
                };
                let xy = xin.add(&y);
                self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy)
            });
        }
        cache.pos += tokens.len();
        x
    }

    /// Forward returning BOTH the logits of the last min(T, 2) positions [≤2, n_vocab] and the
    /// pre-final-norm hidden [T, n_embd] (all rows — they feed the MTP draft block during
    /// speculative decoding; the logits rows are the only ones a speculative loop reads).
    pub fn forward_spec(&self, tokens: &[u32], cache: &mut Cache, n: usize) -> (Tensor, Tensor) {
        self.forward_spec_k(tokens, cache, n, 2)
    }

    /// As `forward_spec` but heads the last `keep` positions — a k-token speculative verify reads the
    /// last k+1 rows (one truth-check per draft + the trailing continuation). Rows are the LAST `keep`,
    /// so row 0 of the returned logits is absolute position `T - min(T, keep)`.
    pub fn forward_spec_k(&self, tokens: &[u32], cache: &mut Cache, n: usize, keep: usize) -> (Tensor, Tensor) {
        use ferric_tensor::{batch, prof};
        let mut x = self.embed(tokens);
        prof(&self.ctx, "embed");
        let pos = cache.pos;
        let profiling = std::env::var("FERRIC_PROFILE").is_ok();
        for (il, l) in self.layers.iter().enumerate().take(n) {
            let lc = &mut cache.layers[il];
            let xin = &x;
            if profiling {
                let is_attn = matches!(l.mixer, Mixer::Attn(_));
                let y = batch(&self.ctx, || {
                    let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                    match &l.mixer { Mixer::Attn(w) => self.attn(&h, w, lc, pos), Mixer::Gdn(w) => self.gdn(&h, w, lc), Mixer::Lag(w) => self.lag_attn(&h, w, lc, pos) }
                });
                prof(&self.ctx, if is_attn { "attn" } else { "gdn" });
                let xy = xin.add(&y);
                x = batch(&self.ctx, || self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy));
                prof(&self.ctx, "ffn");
            } else {
                x = batch(&self.ctx, || {
                    let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                    let y = match &l.mixer {
                        Mixer::Attn(w) => self.attn(&h, w, lc, pos),
                        Mixer::Gdn(w) => self.gdn(&h, w, lc),
                        Mixer::Lag(w) => self.lag_attn(&h, w, lc, pos),
                    };
                    let xy = xin.add(&y);
                    self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy)
                });
            }
        }
        cache.pos += tokens.len();
        // Head only the LAST `keep` rows — all a speculative loop ever reads (per-draft truth-checks +
        // the trailing continuation at verify, the last row at prefill). Heading a whole long prompt
        // would build a [T, n_vocab] logits tensor (1.3 GB at T≈1300) for rows nobody looks at.
        let t = tokens.len();
        let keep = t.min(keep);
        let xl = if t > keep { x.narrow(0, t - keep, keep).contiguous() } else { x.clone() };
        let logits = batch(&self.ctx, || xl.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q(&self.lm_head));
        prof(&self.ctx, "lm_head");
        (logits, x)
    }

    /// Debug: time one layer's FFN at batch size `t` (median of `iters`, synced) — isolates the
    /// slab MoE path's t-scaling from everything else in a forward.
    pub fn bench_ffn(&self, il: usize, t: usize, iters: usize) -> f64 {
        let l = &self.layers[il];
        let h = Tensor::from_vec(&self.ctx, &vec![0.01f32; t * self.cfg.n_embd], &[t, self.cfg.n_embd]);
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            let o = ferric_tensor::batch(&self.ctx, || self.ffn(&h, l));
            pollster::block_on(o.to_vec());
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[iters / 2]
    }

    /// MTP draft forward: each input pair is (tokenᵢ₊₁, main-model hiddenᵢ) at tokenᵢ₊₁'s position;
    /// output logits draft tokenᵢ₊₂ at the LAST pair. `hiddens` is [T, n_embd] pre-final-norm rows
    /// aligned with `tokens`. The draft layer keeps its own cache — feed it every committed token's
    /// pair, in order, exactly once.
    pub fn mtp_forward(&self, tokens: &[u32], hiddens: &Tensor, mc: &mut MtpCache) -> Tensor {
        self.mtp_forward_h(tokens, hiddens, mc).0
    }

    /// As `mtp_forward`, but also returns the draft layer's LAST pre-head hidden [1, n_embd]. Feeding
    /// that hidden back in (paired with the just-drafted token) drafts the NEXT token — recursive MTP
    /// for multi-token speculation. The chained hidden is the block's own output (an approximation of
    /// the main model's hidden the pair would normally use), so deeper drafts accept progressively less.
    pub fn mtp_forward_h(&self, tokens: &[u32], hiddens: &Tensor, mc: &mut MtpCache) -> (Tensor, Tensor) {
        use ferric_tensor::batch;
        let m = self.mtp.as_ref().expect("model has no MTP draft block");
        let l = &m.layer;
        let eps = self.cfg.eps;
        let pos = mc.pos;
        let lc = &mut mc.layer;
        let emb = self.embed(tokens);
        let (logits, hid) = batch(&self.ctx, || {
            let x = emb.rmsnorm(&m.enorm, eps).cat(&hiddens.rmsnorm(&m.hnorm, eps), 1).matmul_q(&m.eh_proj);
            let h = x.rmsnorm(&l.attn_norm, eps);
            let y = match &l.mixer {
                Mixer::Attn(w) => self.attn(&h, w, lc, pos),
                Mixer::Gdn(w) => self.gdn(&h, w, lc),
                Mixer::Lag(w) => self.lag_attn(&h, w, lc, pos),
            };
            let xy = x.add(&y);
            let x = self.ffn(&xy.rmsnorm(&l.post_norm, eps), l).add(&xy);
            // Only the last position drafts — don't head (or read back) the whole pair stream.
            let t = x.shape[0];
            let xl = if t > 1 { x.narrow(0, t - 1, 1).contiguous() } else { x };
            (xl.rmsnorm(&m.head_norm, eps).matmul_q(&self.lm_head), xl)
        });
        mc.pos += tokens.len();
        (logits, hid)
    }
}
