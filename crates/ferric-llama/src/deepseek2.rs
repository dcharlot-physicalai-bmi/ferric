//! **DeepSeek-V2 / V3 / Coder-V2** (`deepseek2`) — Multi-head Latent Attention plus DeepSeekMoE.
//!
//! This is the architecture family behind the models that actually get used: DeepSeek V4 Flash and
//! V4 Pro sat at #2 and #5 on OpenRouter's July 2026 routed-traffic ranking, both open-weight.
//!
//! Two ideas, both of which have a convention that silently changes the model if assumed:
//!
//! ## 1. MLA — the KV cache is a low-rank latent
//!
//! Instead of caching per-head K and V, one `kv_lora_rank`-wide vector per position is cached and
//! decompressed on use. The RoPE dimensions live *outside* that compression and are **shared across
//! all heads** — one 64-wide vector per position, not one per head. A query head is therefore
//! `[nope | rope]` where only the second part is positional.
//!
//! Two tensor layouts exist in the wild and they are not interchangeable:
//!
//! ```text
//!   legacy   attn_kv_b  [kv_lora_rank, n_head*(qk_nope + v_head)]   decompress, then plain MHA
//!   absorbed attn_k_b + attn_v_b                                    fold wk_b into the query and
//!                                                                   attend against the latent (MQA)
//! ```
//!
//! This module implements the **legacy** path, which is what `attn_kv_b` in a checkpoint means. The
//! absorbed path is the one that makes the cache small; it is a separate implementation, not a flag.
//!
//! ## 2. DeepSeekMoE — routed experts plus an always-on shared expert
//!
//! `n_expert_used` of `n_expert` routed experts, **plus** a shared expert evaluated for every token
//! whose output is added. Leading `leading_dense_block_count` blocks are plain dense SwiGLU.
//!
//! ⚠ **`expert_weights_norm` is absent from DeepSeek-V2 checkpoints and defaults to false.** The
//! routed weights are then the raw top-k probabilities of a softmax over ALL experts, which sum to
//! *less* than 1. Renormalising anyway multiplies every routed contribution by `1/Σp` — often 2x or
//! more. See [`ferric_tensor::Tensor::moe_topk_ex`].
//!
//! ## 3. YaRN, which is where the constants hide
//!
//! DeepSeek folds YaRN into two places at once, and both are easy to miss:
//!
//! ```text
//!   attn_factor  scales the RoPE output — and only the rope lanes, so it is NOT a global rescale
//!   mscale²      is folded into the attention scale: kq = mscale² / sqrt(qk_head_dim)
//! ```
//!
//! On Coder-V2-Lite (factor 40, `yarn_log_multiplier` 0.0707) that makes the effective attention
//! scale **1.59x** what `1/sqrt(192)` would give. Deriving the scale instead of computing it from the
//! checkpoint is a silent, uniform distortion of every attention distribution.
//!
//! Note llama.cpp divides the stored `yarn_log_multiplier` by 0.1 on load, cancelling a factor its
//! own conversion script applied — so the value in the file is not the value in the formula.
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::kvquant::{KvqFmt, QKvCache};
use ferric_tensor::{append2, nn, KvBuf, QMatrix, Tensor};
use std::sync::Arc;


/// Env-gated tensor dump for bisecting against llama.cpp's `llama-eval-callback`, whose `ggml_debug`
/// lines carry the same names (`attn_norm`, `q`, `kv_cmpr`, `k_pe`, `kv`, `attn_out`, `ffn_out`,
/// `l_out`). Set `FERRIC_DUMP=<block index>`.
///
/// Prints shape, first values and summary stats. Stats alone are not enough — two tensors can share a
/// mean and a range and still be permuted relative to each other, which is exactly the failure mode
/// that a summary statistic hides.
fn dump(tag: &str, il: usize, t: &Tensor) {
    let Ok(want) = std::env::var("FERRIC_DUMP") else { return };
    if want.parse::<usize>().ok() != Some(il) { return }
    let v = pollster::block_on(t.to_vec());
    let (mut mn, mut mx, mut sum) = (f32::MAX, f32::MIN, 0f64);
    for &x in &v { mn = mn.min(x); mx = mx.max(x); sum += x as f64; }
    let head: Vec<String> = v.iter().take(6).map(|x| format!("{x:+.5}")).collect();
    println!("  [{il}] {tag:<12} {:?} n={} min {mn:+.5} max {mx:+.5} mean {:+.6}\n               {}",
             t.shape, v.len(), sum / v.len() as f64, head.join(" "));
}

pub struct Cfg {
    pub n_layer: usize,
    pub d: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// Non-positional part of a query/key head.
    pub qk_nope: usize,
    /// RoPE part of a query/key head. Shared across heads on the key side.
    pub qk_rope: usize,
    /// Value head width. Differs from the query/key head width.
    pub v_head: usize,
    pub kv_lora_rank: usize,
    /// Blocks before the MoE blocks start, which are plain dense SwiGLU.
    pub dense_lead: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    /// Per-expert FFN width. The shared expert is `n_expert_shared` of these wide.
    pub expert_ff: usize,
    /// Dense-block FFN width.
    pub n_ff: usize,
    /// Multiplier on the routed combining weights (`expert_weights_scale`).
    pub routed_scale: f32,
    /// Whether the routed weights are renormalised. **False on DeepSeek-V2.**
    pub expert_norm: bool,
    /// Sigmoid gating (V3-style `noaux_tc`) rather than softmax (V2).
    pub sigmoid_gate: bool,
    pub rope_base: f32,
    /// YaRN context-extension factor; 1.0 means no scaling.
    pub yarn_factor: f32,
    pub yarn_orig_ctx: usize,
    /// `rope.scaling.yarn_log_multiplier`, already divided by 0.1 as llama.cpp does on load.
    pub yarn_log_mul: f32,
}

impl Cfg {
    /// Full query/key head width.
    pub fn qk_head(&self) -> usize { self.qk_nope + self.qk_rope }

    /// The YaRN scale applied to the RoPE output. Applies to the rope lanes only, so it cannot be
    /// folded into the attention scale.
    pub fn attn_factor(&self) -> f32 {
        // ⚠ 1.0, NOT 1/(1 + 0.1·ln(factor)).
        //
        // ggml's `rope_yarn` does `mscale *= 1 + 0.1*logf(1/freq_scale)` INSIDE the kernel whenever
        // ext_factor != 0. llama.cpp's deepseek2 graph pre-divides the attn_factor it passes in
        // precisely so that multiply restores it — the comment there literally reads "first cancel the
        // adjustment ... to get the original attn_factor". The NET scale on cos/sin is therefore
        // attn_factor_org, which is 1.0 at the default.
        //
        // Applying the pre-divided value in a runtime whose rope kernel does NOT re-multiply it scales
        // every rope lane by 0.73 and nothing says so. Caught by diffing k_pe against
        // llama-eval-callback: 100.07 against the reference's 123.955, with every earlier tensor in
        // the block matching to 4 significant figures.
        1.0
    }

    /// `mscale`, whose square is folded into the attention scale.
    pub fn mscale(&self) -> f32 {
        if self.yarn_factor <= 1.0 { return 1.0; }
        1.0 + 0.1 * self.yarn_log_mul * (self.yarn_factor).ln()
    }

    /// The factor Q must be pre-multiplied by so that a kernel baking in `1/sqrt(qk_head)` ends up
    /// applying `mscale² / sqrt(qk_head)`.
    pub fn q_prescale(&self) -> f32 { let m = self.mscale(); m * m }

    pub fn is_moe(&self, il: usize) -> bool { il >= self.dense_lead && self.n_expert > 0 }

    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let u = |k: &str| match md.get(&format!("deepseek2.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing deepseek2.{k}")) };
        let f = |k: &str| match md.get(&format!("deepseek2.{k}")) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing deepseek2.{k}")) };
        let s = |k: &str| match md.get(&format!("deepseek2.{k}")) { Some(Meta::Str(v)) => Some(v.clone()), _ => None };

        let key_len = u("attention.key_length")?;
        let qk_rope = u("rope.dimension_count")?;
        if qk_rope >= key_len { return Err(format!("rope dim {qk_rope} must be below key length {key_len}")); }

        let yarn = s("rope.scaling.type").as_deref() == Some("yarn");
        let n_vocab = match md.get("deepseek2.vocab_size") {
            Some(Meta::U(v)) => *v as usize,
            _ => g.tensor("token_embd.weight").ok_or("no token_embd")?.dims[1] as usize,
        };

        Ok(Cfg {
            n_layer: u("block_count")?,
            d: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_vocab,
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_nope: key_len - qk_rope,
            qk_rope,
            v_head: u("attention.value_length")?,
            kv_lora_rank: u("attention.kv_lora_rank")?,
            dense_lead: u("leading_dense_block_count").unwrap_or(0),
            n_expert: u("expert_count").unwrap_or(0),
            n_expert_used: u("expert_used_count").unwrap_or(0),
            n_expert_shared: u("expert_shared_count").unwrap_or(0),
            expert_ff: u("expert_feed_forward_length").unwrap_or(0),
            n_ff: u("feed_forward_length")?,
            routed_scale: f("expert_weights_scale").unwrap_or(1.0),
            // ABSENT MEANS FALSE. Defaulting this to true rescales every routed contribution.
            expert_norm: matches!(md.get("deepseek2.expert_weights_norm"), Some(Meta::Bool(true))),
            // V2 has no gating-func key and uses softmax; V3 declares sigmoid (`noaux_tc`).
            sigmoid_gate: matches!(md.get("deepseek2.expert_gating_func"), Some(Meta::U(2))),
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            yarn_factor: if yarn { f("rope.scaling.factor").unwrap_or(1.0) } else { 1.0 },
            yarn_orig_ctx: u("rope.scaling.original_context_length").unwrap_or(4096),
            // llama.cpp cancels a 0.1 the conversion script applied, so the file value is not the
            // formula's value.
            yarn_log_mul: f("rope.scaling.yarn_log_multiplier").unwrap_or(0.0) / 0.1,
        })
    }
}

enum Ffn {
    Dense { gate: QMatrix, up: QMatrix, down: QMatrix },
    Moe {
        router: QMatrix,
        bias: Option<Tensor>,
        /// All experts' gate|up fused expert-major, so one dispatch serves every routed expert.
        gate_up: ferric_tensor::Q4_KWeights,
        down: crate::qwen35::DownSlab,
        sh_gate: QMatrix, sh_up: QMatrix, sh_down: QMatrix,
    },
}

/// Fuse per-expert `gate` and `up` into ONE expert-major slab: `[e0 gate | e0 up | e1 gate | ...]`.
///
/// The GGUF stores them as two separate `[n_expert, ff, d]` stacks, and the batched expert kernel
/// wants them adjacent per expert. Interleaving whole ROW BLOCKS is exact on quantised bytes because
/// a row is a whole number of quant blocks; slicing anywhere else would cut a block in half.
fn fuse_gate_up(ctx: &Arc<Context>, g: &impl GgufSource, il: usize, n_expert: usize)
    -> Result<ferric_tensor::Q4_KWeights, String> {
    let (gn, un) = (format!("blk.{il}.ffn_gate_exps.weight"), format!("blk.{il}.ffn_up_exps.weight"));
    let (gt, ut) = (g.tensor(&gn).ok_or("no gate_exps")?, g.tensor(&un).ok_or("no up_exps")?);
    if gt.ggml_type != ut.ggml_type {
        return Err(format!("gate_exps is type {} and up_exps is {}; fusing needs one type", gt.ggml_type, ut.ggml_type));
    }
    if gt.ggml_type != 12 {
        return Err(format!("expert slab expects Q4_K (12), got {}", gt.ggml_type));
    }
    let (d, ff) = (gt.dims[0] as usize, gt.dims[1] as usize);
    let row = ferric_gguf::type_size(gt.ggml_type, d)?;
    let (gr, ur) = (g.raw(&gn)?, g.raw(&un)?);
    let per = ff * row;                       // bytes of one expert's rows in one projection
    let mut out = Vec::with_capacity(gr.len() + ur.len());
    for e in 0..n_expert {
        out.extend_from_slice(&gr[e * per..(e + 1) * per]);
        out.extend_from_slice(&ur[e * per..(e + 1) * per]);
    }
    Ok(ferric_tensor::Q4_KWeights::from_bytes(ctx, &out, n_expert * 2 * ff, d))
}

struct Block {
    attn_norm: Tensor,
    q: QMatrix,
    kv_a_mqa: QMatrix,
    kv_a_norm: Tensor,
    kv_b: QMatrix,
    o: QMatrix,
    ffn_norm: Tensor,
    ffn: Ffn,
}

/// Where each row's RoPE angle takes its absolute position from.
///
/// One sequence's rows are consecutive (`offset + i`); N *independent* sequences batched one token
/// each are not — row `i` sits wherever sequence `i`'s own history reached. Making that an enum
/// threaded through one shared projection helper, rather than writing the projections twice, is
/// deliberate: the batched and solo paths must differ in EXACTLY this one place. A batched copy that
/// drifted anywhere else — a dropped YaRN table, NEOX pairing, a shared position — still emits fluent
/// text with finite logits and raises nothing at all.
enum Pos<'a> {
    /// One sequence: `t` consecutive rows starting at this absolute position.
    Run(usize),
    /// One row per sequence, each at its own absolute position.
    Rows(&'a [u32]),
}

pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
    /// Block-quantized twin of `kv`. **Empty unless KV quantization is on**, and when it is on `kv` is
    /// the empty one — holding both would spend the memory the quantization exists to save.
    q: Vec<(QKvCache, QKvCache)>,
    fmt: Option<KvqFmt>,
}

impl Cache {
    /// Default (f32) cache, unless `FERRIC_KVQ` asks otherwise. See [`Cache::with_kvq`].
    pub fn new(cfg: &Cfg) -> Cache { Cache::with_kvq(cfg, crate::qwen3::kvq_from_env()) }

    /// A cache whose K/V rows are stored as `fmt` quantization blocks — `None` for today's f32.
    ///
    /// **Opt-in, and it must stay opt-in**: this trades accuracy for memory, so it is the caller's
    /// choice and never a silent change to an existing run.
    ///
    /// ⚠ **This runtime caches DECOMPRESSED K/V, not the latent.** MLA's selling point is that one
    /// `kv_lora_rank`-wide vector per position is enough, but that is the *absorbed* path, which
    /// [`DeepSeek2::load`] refuses. The legacy `attn_kv_b` path decompresses to `n_head * qk_dim` keys
    /// and `n_head * v_head` values and caches those — far larger than the latent. So KV quantization
    /// buys MORE here than on a model whose cache is already small, not less, and the two are
    /// complementary rather than redundant: the absorbed path would shrink the row count's width, this
    /// shrinks each element.
    ///
    /// K and V have DIFFERENT widths here (qk_dim vs v_head per head), which each `QKvCache` learns on
    /// its own first append — they are separate caches, so nothing forces them to agree.
    pub fn with_kvq(cfg: &Cfg, fmt: Option<KvqFmt>) -> Cache {
        match fmt {
            None => Cache {
                pos: 0,
                kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect(),
                q: Vec::new(),
                fmt: None,
            },
            Some(f) => Cache {
                pos: 0,
                kv: Vec::new(),
                q: (0..cfg.n_layer).map(|_| (QKvCache::new(f), QKvCache::new(f))).collect(),
                fmt: Some(f),
            },
        }
    }

    /// The KV-cache quantization format in force, or `None` for f32.
    pub fn kvq_fmt(&self) -> Option<KvqFmt> { self.fmt }

    /// **Device bytes the K/V caches actually occupy right now.**
    pub fn kv_bytes(&self) -> usize {
        match self.fmt {
            None => self.kv.iter().map(|(k, v)| (k.len() * k.width() + v.len() * v.width()) * 4).sum(),
            Some(_) => self.q.iter().map(|(k, v)| k.bytes() + v.bytes()).sum(),
        }
    }

    /// Live K/V bytes, ignoring allocated slack — what the FORMAT buys. See
    /// [`ferric_tensor::kvquant::QKvCache::live_bytes`] for why this differs from `kv_bytes()`.
    pub fn kv_live_bytes(&self) -> usize {
        match self.fmt {
            None => self.kv_bytes(),
            Some(_) => self.q.iter().map(|(k, v)| k.live_bytes() + v.live_bytes()).sum(),
        }
    }

    /// What the same live rows would cost as f32 — `kv_live_bytes()`'s denominator.
    pub fn kv_f32_bytes(&self) -> usize {
        match self.fmt {
            None => self.kv_bytes(),
            Some(_) => self.q.iter().map(|(k, v)| k.f32_bytes() + v.f32_bytes()).sum(),
        }
    }
}

pub struct DeepSeek2 {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    blocks: Vec<Block>,
    out_norm: Tensor,
    head: QMatrix,
    embd_ty: u32,
    embd_raw: Vec<u8>,
    /// Per-dim YaRN multiplier for the rope lanes.
    yarn: Option<Tensor>,
}

impl DeepSeek2 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<DeepSeek2, String> {
        let cfg = Cfg::from_gguf(g)?;

        // The absorbed layout is a different attention, not a variant of this one. Refuse rather than
        // half-run it.
        if g.tensor("blk.0.attn_k_b.weight").is_some() {
            return Err("this checkpoint uses the ABSORBED MLA layout (attn_k_b/attn_v_b); this module \
                        implements the legacy attn_kv_b path".into());
        }
        // Non-lite variants factor Q through a LoRA. Detect rather than assume, because the tensor
        // simply would not be found and the error would point at the wrong thing.
        if g.tensor("blk.0.attn_q_a.weight").is_some() {
            return Err("this checkpoint factors Q through a LoRA (attn_q_a/attn_q_b); only the lite \
                        direct-Q path is implemented".into());
        }

        let qm = |name: &str| -> Result<QMatrix, String> {
            let t = g.tensor(name).ok_or_else(|| format!("missing {name}"))?;
            let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
            if QMatrix::block_bytes(ty).is_some() {
                QMatrix::from_bytes(ctx, &g.raw(name)?, ty, rows, cols)
            } else {
                Ok(QMatrix::from_dense(ctx, &g.dequant(name)?, rows, cols))
            }
        };
        let ft = |name: &str, shape: &[usize]| -> Result<Tensor, String> {
            Ok(Tensor::from_vec(ctx, &g.dequant(name)?, shape))
        };

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let ffn = if cfg.is_moe(il) {
                let dn = b("ffn_down_exps.weight");
                let dt = g.tensor(&dn).ok_or("no down_exps")?;
                let (dd, drows) = (dt.dims[0] as usize, dt.dims[1] as usize * cfg.n_expert);
                Ffn::Moe {
                    router: qm(&b("ffn_gate_inp.weight"))?,
                    bias: g.tensor(&b("ffn_exp_probs_b.bias")).map(|_| ft(&b("ffn_exp_probs_b.bias"), &[cfg.n_expert])).transpose()?,
                    gate_up: fuse_gate_up(ctx, g, il, cfg.n_expert)?,
                    // Q4_K_M alternates Q6_K and Q4_K on down_exps per layer, so both need a slab.
                    down: match dt.ggml_type {
                        14 => crate::qwen35::DownSlab::Q6(ferric_tensor::Q6_KWeights::from_bytes(ctx, &g.raw(&dn)?, drows, dd)),
                        12 => crate::qwen35::DownSlab::Q4(ferric_tensor::Q4_KWeights::from_bytes(ctx, &g.raw(&dn)?, drows, dd)),
                        8 => crate::qwen35::DownSlab::Q8(ferric_tensor::Q8_0Weights::from_bytes(ctx, &g.raw(&dn)?, drows, dd)),
                        6 => crate::qwen35::DownSlab::Q5(ferric_tensor::Q5_0Weights::from_bytes(ctx, &g.raw(&dn)?, drows, dd)),
                        o => return Err(format!("blk.{il}.ffn_down_exps is quant type {o}, which has no \
                            indexed-expert kernel (have Q4_K/12, Q5_0/6, Q6_K/14, Q8_0/8)")),
                    },
                    sh_gate: qm(&b("ffn_gate_shexp.weight"))?,
                    sh_up: qm(&b("ffn_up_shexp.weight"))?,
                    sh_down: qm(&b("ffn_down_shexp.weight"))?,
                }
            } else {
                Ffn::Dense { gate: qm(&b("ffn_gate.weight"))?, up: qm(&b("ffn_up.weight"))?, down: qm(&b("ffn_down.weight"))? }
            };
            blocks.push(Block {
                attn_norm: ft(&b("attn_norm.weight"), &[cfg.d])?,
                q: qm(&b("attn_q.weight"))?,
                kv_a_mqa: qm(&b("attn_kv_a_mqa.weight"))?,
                kv_a_norm: ft(&b("attn_kv_a_norm.weight"), &[cfg.kv_lora_rank])?,
                kv_b: qm(&b("attn_kv_b.weight"))?,
                o: qm(&b("attn_output.weight"))?,
                ffn_norm: ft(&b("ffn_norm.weight"), &[cfg.d])?,
                ffn,
            });
        }

        let yarn = (cfg.yarn_factor > 1.0).then(|| {
            let v = crate::qwen35::yarn_freq_scale(cfg.qk_rope, cfg.rope_base, cfg.yarn_factor, cfg.yarn_orig_ctx, 32.0, 1.0);
            Tensor::from_vec(ctx, &v, &[cfg.qk_rope / 2])
        });

        Ok(DeepSeek2 {
            ctx: ctx.clone(),
            blocks,
            out_norm: ft("output_norm.weight", &[cfg.d])?,
            head: if g.tensor("output.weight").is_some() { qm("output.weight")? } else { qm("token_embd.weight")? },
            embd_ty: g.tensor("token_embd.weight").ok_or("no token_embd")?.ggml_type,
            embd_raw: g.raw("token_embd.weight")?,
            yarn,
            cfg,
        })
    }

    fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.d;
        let row = ferric_gguf::type_size(self.embd_ty, d).expect("embd row");
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let o = t as usize * row;
            v.extend(ferric_gguf::deq_raw(&self.embd_raw[o..o + row], d, self.embd_ty).expect("embed row"));
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// RoPE the positional lanes, then apply the YaRN `attn_factor`.
    ///
    /// The scale lands on the rope lanes only, never the nope lanes, so it cannot be folded into the
    /// attention scale the way `mscale²` can.
    fn rope_pe(&self, x: &Tensor, heads: usize, pos: &Pos) -> Tensor {
        let cfg = &self.cfg;
        // ⚠ INTERLEAVED. llama.cpp lists deepseek2 under "normal RoPE, operating on pairs of
        // consecutive head values" (ROPE_TYPE_NORM), not the split-half NEOX pairing every Qwen-family
        // model here uses. Rotating the wrong partners produces finite logits and wrong text.
        // A/B via FERRIC_ROPE_NEOX to settle the pairing empirically rather than by reading alone.
        let neox = std::env::var("FERRIC_ROPE_NEOX").is_ok();
        let r = match pos {
            Pos::Run(off) => match (&self.yarn, neox) {
                (Some(fs), false) => x.rope_scaled_interleaved(fs, heads, cfg.qk_rope, cfg.rope_base, *off),
                (Some(fs), true) => x.rope_scaled(fs, heads, cfg.qk_rope, cfg.rope_base, *off),
                (None, false) => x.rope_interleaved(heads, cfg.qk_rope, cfg.rope_base, *off),
                (None, true) => x.rope(heads, cfg.qk_rope, cfg.rope_base, *off),
            },
            // The SAME two kernels, with each row's position read from a list instead of derived from
            // one offset: `Some(fs)` selects ROPE_SCALED_WGSL exactly as `rope_scaled*` does, and
            // `interleaved` is the same __PAIRLO__/__PAIRHI__ substitution. Both arguments must be
            // threaded through, not defaulted — `rope_at` (the unscaled NEOX-only version) is what
            // silently turned the dense runtime's batched decode into a different model: no YaRN
            // stretch and the wrong rotation partners, fluent output, no error. Here that would also
            // drop `mscale`'s sibling on the rope lanes and rotate DeepSeek's NORM pairs as NEOX.
            Pos::Rows(p) => x.rope_at_ex(heads, cfg.qk_rope, cfg.rope_base, p, self.yarn.as_ref(), !neox),
        };
        let af = cfg.attn_factor();
        if (af - 1.0).abs() < 1e-9 { r } else { r.mul(&r.scalar(af)) }
    }

    /// Everything in MLA up to the attention itself: Q split into `[nope | rope]`, the compressed KV
    /// decompressed to per-head K-nope and V, the SHARED rope vector broadcast across heads, and the
    /// `mscale²` prescale on Q. Returns `(q_full [t, nh*qk], k [t, nh*qk], v [t, nh*vh])`.
    ///
    /// Solo decode and batched decode share this verbatim so they CANNOT drift. Every op here is
    /// row-wise, so a row's projections do not depend on which rows travel with it; the one thing that
    /// does depend on the row is the RoPE position, which is why `pos` is the only parameter that
    /// differs between the two callers. Keeping that difference down to one enum is the whole defence
    /// against the failure this path has no symptom for.
    fn attn_proj(&self, h: &Tensor, blk: &Block, il: usize, pos: &Pos) -> (Tensor, Tensor, Tensor) {
        let cfg = &self.cfg;
        let (eps, t) = (cfg.eps, h.shape[0]);
        let (nh, nope, rope, vh, r) = (cfg.n_head, cfg.qk_nope, cfg.qk_rope, cfg.v_head, cfg.kv_lora_rank);
        let qk = cfg.qk_head();

        // Q: [t, nh*qk] -> split each head into its nope and rope lanes.
        let q = h.matmul_q(&blk.q).reshape(&[t, nh, qk]);
        let q_nope = q.narrow(2, 0, nope).contiguous();
        let q_pe = self.rope_pe(&q.narrow(2, nope, rope).contiguous().reshape(&[t, nh * rope]), nh, pos)
            .reshape(&[t, nh, rope]);

        // The compressed KV plus the SHARED rope vector: [t, r + rope].
        dump("q", il, &q.reshape(&[t, nh * qk]));
        let kvp = h.matmul_q(&blk.kv_a_mqa);
        let kv_cmpr = kvp.narrow(1, 0, r).contiguous().rmsnorm(&blk.kv_a_norm, eps);
        let k_pe = self.rope_pe(&kvp.narrow(1, r, rope).contiguous(), 1, pos);

        // Decompress to per-head K-nope and V: [t, nh*(nope+vh)].
        dump("kv_cmpr", il, &kv_cmpr);
        dump("k_pe", il, &k_pe);
        let kv = kv_cmpr.matmul_q(&blk.kv_b).reshape(&[t, nh, nope + vh]);
        dump("kv", il, &kv.reshape(&[t, nh * (nope + vh)]));
        let k_nope = kv.narrow(2, 0, nope).contiguous();
        let v = kv.narrow(2, nope, vh).contiguous().reshape(&[t, nh * vh]);

        // K is [nope | shared rope], so the one rope vector is broadcast to every head.
        let k_pe_all = k_pe.reshape(&[t, 1, rope]).broadcast_to(&[t, nh, rope]).contiguous();
        let k = k_nope.cat(&k_pe_all, 2).reshape(&[t, nh * qk]);
        let q_full = q_nope.cat(&q_pe, 2).reshape(&[t, nh * qk]);
        // The kernels bake in 1/sqrt(qk); DeepSeek wants mscale²/sqrt(qk).
        let q_full = q_full.mul(&q_full.scalar(cfg.q_prescale()));
        (q_full, k, v)
    }

    /// DeepSeekMoE (or the dense SwiGLU on the leading blocks), on `t` already-normed rows.
    ///
    /// Row-wise throughout — the router picks each row's own top-k and the indexed expert kernels take
    /// a `[t, k, ff]` mid — so this is the same function for one token, a prefill block, or N batched
    /// sequences. Extracted rather than duplicated for the batched path precisely because a second copy
    /// could pick up a different `expert_norm`/`routed_scale` and nothing would ever say so.
    fn ffn(&self, f: &Tensor, blk: &Block) -> Tensor {
        let cfg = &self.cfg;
        match &blk.ffn {
            Ffn::Dense { gate, up, down } => f.matmul_q(gate).silu().mul(&f.matmul_q(up)).matmul_q(down),
            Ffn::Moe { router, bias, gate_up, down, sh_gate, sh_up, sh_down } => {
                let selw = f.matmul_q(router).moe_topk_ex(
                    bias.as_ref(), cfg.n_expert_used, cfg.sigmoid_gate, cfg.routed_scale, cfg.expert_norm);
                let mid = f.matmul_q4_k_swiglu_id(gate_up, &selw, cfg.n_expert_used, cfg.expert_ff);
                let routed = down.wsum(&mid, &selw, cfg.d);
                let shared = f.matmul_q(sh_gate).silu().mul(&f.matmul_q(sh_up)).matmul_q(sh_down);
                routed.add(&shared)
            }
        }
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        self.forward_hidden_cached(tokens, cache).matmul_q(&self.head)
    }

    pub fn forward_hidden_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        let cfg = &self.cfg;
        let (eps, t, pos) = (cfg.eps, tokens.len(), cache.pos);
        let (nh, vh) = (cfg.n_head, cfg.v_head);
        let qk = cfg.qk_head();
        let mut x = self.embed(tokens);

        for (il, blk) in self.blocks.iter().enumerate() {
            let h = x.rmsnorm(&blk.attn_norm, eps);
            dump("attn_norm", il, &h);

            // One sequence: its `t` rows are consecutive from the cache's position.
            let (q_full, k, v) = self.attn_proj(&h, blk, il, &Pos::Run(pos));

            let (kc, vc) = match cache.fmt {
                None => { let e = &mut cache.kv[il]; append2(&self.ctx, &mut e.0, &k, &mut e.1, &v) }
                Some(_) => {
                    let e = &mut cache.q[il];
                    e.0.append(&self.ctx, &k);
                    e.1.append(&self.ctx, &v);
                    (e.0.dequantize(&self.ctx), e.1.dequantize(&self.ctx))
                }
            };

            // V is narrower than K here, so the attention helper cannot infer the value width from
            // the key width. Heads are 1:1 (no GQA) after decompression.
            let att = nn::causal_attention_split(&q_full, &kc, &vc, nh, qk, vh, 0.0);
            dump("attn", il, &att);
            let ao = att.matmul_q(&blk.o);
            dump("attn_out", il, &ao);
            x = x.add(&ao);

            let f = x.rmsnorm(&blk.ffn_norm, eps);
            let out = self.ffn(&f, blk);
            dump("ffn_out", il, &out);
            x = x.add(&out);
            dump("l_out", il, &x);
        }

        cache.pos += t;
        x.rmsnorm(&self.out_norm, eps)
    }

    /// MLA for **N independent sequences**, one token each.
    ///
    /// The win is `attn_proj`: `attn_q` (2048×3072), `attn_kv_a_mqa`, `attn_kv_b` and `attn_output` are
    /// read ONCE for N rows instead of once per row. Decode is weight-bound, so that amortisation is
    /// the entire point of batching — and on this architecture the FFN amortises too, since the shared
    /// expert and the router are dense over all rows.
    ///
    /// Attention itself stays a loop, and has to: sequence `i` attends *its own* latent KV
    /// history, which is a different length from its neighbours'. `causal_attention_split` builds one
    /// `[t, s]` mask per call, and `s` differs per sequence. Folding that loop away is what paged
    /// attention buys; it is not something batching can do on its own.
    fn attn_batch(&self, h: &Tensor, blk: &Block, caches: &mut [&mut Cache], il: usize) -> Tensor {
        let cfg = &self.cfg;
        let (nh, vh) = (cfg.n_head, cfg.v_head);
        let qk = cfg.qk_head();
        let n = h.shape[0];
        debug_assert_eq!(n, caches.len(), "one row per sequence");

        // Row `i` sits wherever sequence `i`'s own history reached — NOT at a shared offset. The rope
        // lanes are the only place the position enters the projections, so this vector is the entire
        // difference between the batched and solo paths. Read from `c.pos` BEFORE any cache is
        // advanced, which is why `forward_hidden_batch` bumps `pos` only after the last block.
        let positions: Vec<u32> = caches.iter().map(|c| c.pos as u32).collect();
        let (q_full, k, v) = self.attn_proj(h, blk, il, &Pos::Rows(&positions));

        let mut outs: Vec<Tensor> = Vec::with_capacity(n);
        for (i, c) in caches.iter_mut().enumerate() {
            // Row `i` of the batch goes into cache `i` and is attended against cache `i` only. A
            // narrow onto the wrong row here is the archetypal silent failure: shapes still match,
            // logits stay finite, and the model writes fluent text from another sequence's history.
            let (ki, vi) = (k.narrow(0, i, 1), v.narrow(0, i, 1));
            // Both stores index by SEQUENCE then layer, so batching needs no fork: row `i` appends
            // to cache `i` exactly as the solo path does.
            let (kc, vc) = match c.fmt {
                None => { let e = &mut c.kv[il]; append2(&self.ctx, &mut e.0, &ki, &mut e.1, &vi) }
                Some(_) => {
                    let e = &mut c.q[il];
                    e.0.append(&self.ctx, &ki);
                    e.1.append(&self.ctx, &vi);
                    (e.0.dequantize(&self.ctx), e.1.dequantize(&self.ctx))
                }
            };
            let qi = q_full.narrow(0, i, 1).contiguous();
            outs.push(nn::causal_attention_split(&qi, &kc, &vc, nh, qk, vh, 0.0));
        }
        let att = outs.iter().skip(1).fold(outs[0].clone(), |acc, t| acc.cat(t, 0));
        dump("attn", il, &att);
        att.matmul_q(&blk.o)
    }

    /// **Batched decode**: advance N independent sequences by one token each, in one forward pass.
    ///
    /// `tokens[i]` is the next token for `caches[i]`. Returns `[N, n_vocab]` logits — row `i` belongs
    /// to sequence `i`.
    ///
    /// Every sequence's logits are **identical** to calling [`Self::forward`] on it alone; batching
    /// changes only how the work is scheduled. `examples/batched_decode_deepseek2.rs` asserts exactly
    /// that on token ids, because there is no other way to see it: a batched path that crossed
    /// sequences produces fluent text and finite logits with nothing to catch it.
    pub fn forward_batch(&self, tokens: &[u32], caches: &mut [&mut Cache]) -> Tensor {
        self.forward_hidden_batch(tokens, caches).matmul_q(&self.head)
    }

    /// [`Self::forward_batch`] without the LM head: `[N, d]` final hidden states.
    pub fn forward_hidden_batch(&self, tokens: &[u32], caches: &mut [&mut Cache]) -> Tensor {
        assert_eq!(tokens.len(), caches.len(), "one token per sequence");
        assert!(!tokens.is_empty(), "forward_batch needs at least one sequence");
        let eps = self.cfg.eps;
        let mut x = self.embed(tokens);

        for (il, blk) in self.blocks.iter().enumerate() {
            let h = x.rmsnorm(&blk.attn_norm, eps);
            dump("attn_norm", il, &h);
            let ao = self.attn_batch(&h, blk, caches, il);
            dump("attn_out", il, &ao);
            x = x.add(&ao);

            let f = x.rmsnorm(&blk.ffn_norm, eps);
            let out = self.ffn(&f, blk);
            dump("ffn_out", il, &out);
            x = x.add(&out);
            dump("l_out", il, &x);
        }

        // AFTER every block, not inside the loop: each block reads `c.pos` to build its rope positions,
        // so bumping it early would rope block 1 one step ahead of block 0 — a per-layer position skew
        // that is invisible in the output shape and produces perfectly fluent, wrong text.
        for c in caches.iter_mut() { c.pos += 1; }
        x.rmsnorm(&self.out_norm, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-Coder-V2-Lite's real numbers.
    fn lite() -> Cfg {
        Cfg {
            n_layer: 27, d: 2048, n_head: 16, n_vocab: 102400, eps: 1e-6,
            qk_nope: 128, qk_rope: 64, v_head: 128, kv_lora_rank: 512,
            dense_lead: 1, n_expert: 64, n_expert_used: 6, n_expert_shared: 2,
            expert_ff: 1408, n_ff: 10944,
            routed_scale: 1.0, expert_norm: false, sigmoid_gate: false,
            rope_base: 10000.0, yarn_factor: 40.0, yarn_orig_ctx: 4096, yarn_log_mul: 0.707,
        }
    }

    #[test]
    fn the_query_key_head_is_wider_than_the_value_head() {
        // 192 vs 128. Every attention helper that infers the value width from the key width is wrong
        // for this architecture, which is why the forward passes both.
        let c = lite();
        assert_eq!(c.qk_head(), 192);
        assert_ne!(c.qk_head(), c.v_head, "MLA's asymmetry is the whole point");
    }

    #[test]
    fn yarn_moves_the_attention_scale_by_more_than_half_again() {
        // mscale = 1 + 0.1 * 0.707 * ln(40) = 1.2608; the scale carries mscale².
        let c = lite();
        let m = c.mscale();
        assert!((m - 1.26078).abs() < 1e-4, "mscale {m}");
        let q = c.q_prescale();
        assert!((q - 1.58957).abs() < 1e-4, "prescale {q}");
        // Deriving 1/sqrt(192) and stopping there understates the scale by ~37%.
        assert!(q > 1.5, "a derived scale would be {q}x too small");
    }

    #[test]
    fn no_yarn_means_no_adjustment_at_all() {
        // A checkpoint without rope scaling must not pick up a stray factor.
        let mut c = lite();
        c.yarn_factor = 1.0;
        assert_eq!(c.mscale(), 1.0);
        assert_eq!(c.attn_factor(), 1.0);
        assert_eq!(c.q_prescale(), 1.0);
    }

    #[test]
    fn the_rope_output_is_not_rescaled_but_the_attention_score_is() {
        // I originally had attn_factor = 1/(1 + 0.1·ln(factor)) = 0.7305, reasoning from the
        // pre-division in llama.cpp's deepseek2 graph. That was wrong: ggml's `rope_yarn` RE-MULTIPLIES
        // by the same term inside the kernel, so the pre-division exists only to cancel it and the NET
        // scale on cos/sin is attn_factor_org = 1.0.
        //
        // Established by diffing k_pe against llama-eval-callback, not by re-reading: 123.88 against
        // the reference's 123.955, where the 0.7305 version gave 100.07.
        let c = lite();
        assert_eq!(c.attn_factor(), 1.0, "the rope output carries no extra YaRN scale");
        // mscale² still very much applies, to the attention score. The two are NOT the same constant.
        assert!((c.mscale() - 1.26078).abs() < 1e-4);
        assert_ne!(c.attn_factor(), c.mscale(), "one is 1.0 and the other is not");
    }

    #[test]
    fn expert_weight_normalisation_defaults_off_for_v2() {
        // The key is ABSENT from DeepSeek-V2 checkpoints. Defaulting it on rescales every routed
        // contribution by 1/Σp, which for top-6 of 64 is routinely 2x or more.
        let c = lite();
        assert!(!c.expert_norm, "V2 must not renormalise routed weights");
        assert!(!c.sigmoid_gate, "V2 gates with softmax; sigmoid is the V3 noaux_tc router");
        assert_eq!(c.routed_scale, 1.0);
    }

    #[test]
    fn the_leading_block_is_dense_and_the_rest_are_not() {
        let c = lite();
        assert!(!c.is_moe(0), "block 0 is dense on Coder-V2-Lite");
        assert!(c.is_moe(1));
        assert!(c.is_moe(26));
    }
}

#[cfg(test)]
mod kvq_tests {
    use super::*;

    fn cfg(n_layer: usize) -> Cfg {
        // Coder-V2-Lite's shape. Only the widths matter here; the MoE fields are along for the ride.
        Cfg {
            n_layer, d: 2048, n_head: 16, n_vocab: 102400, eps: 1e-6,
            qk_nope: 128, qk_rope: 64, v_head: 128, kv_lora_rank: 512,
            dense_lead: 1, n_expert: 64, n_expert_used: 6, n_expert_shared: 2,
            expert_ff: 1408, n_ff: 10944, routed_scale: 1.0, expert_norm: false,
            sigmoid_gate: false, rope_base: 10000.0, yarn_factor: 40.0,
            yarn_orig_ctx: 4096, yarn_log_mul: 0.0,
        }
    }

    /// The two stores are never both populated, in either direction.
    ///
    /// Worth pinning on this runtime specifically because its K and V have DIFFERENT widths — 
    /// `n_head * (qk_nope + qk_rope)` against `n_head * v_head` — so the two `QKvCache`s in a pair
    /// learn different widths on their first append. Nothing forces them to agree, and nothing should.
    #[test]
    fn a_quantized_cache_holds_no_f32_rows_and_pairs_may_differ_in_width() {
        let c = cfg(4);
        let f = Cache::with_kvq(&c, None);
        assert!(f.kvq_fmt().is_none());
        assert_eq!(f.kv.len(), c.n_layer);
        assert!(f.q.is_empty(), "an f32 cache must not allocate the quantized twin");

        for fmt in KvqFmt::ALL {
            let q = Cache::with_kvq(&c, Some(fmt));
            assert_eq!(q.kvq_fmt(), Some(fmt));
            assert!(q.kv.is_empty(), "{}: the f32 vector must be EMPTY, not merely unused", fmt.name());
            assert_eq!(q.q.len(), c.n_layer, "{}: one slot per layer", fmt.name());
            assert_eq!(q.kv_bytes(), 0, "{}: an untouched cache costs nothing", fmt.name());
            // Both start width-less; each learns its own on first append.
            assert!(q.q.iter().all(|(k, v)| k.width() == 0 && v.width() == 0),
                    "{}: widths are learned per cache, not fixed at construction", fmt.name());
        }

        // The widths this runtime will actually use are both block-aligned, which QKvCache requires.
        let (kw, vw) = (c.n_head * (c.qk_nope + c.qk_rope), c.n_head * c.v_head);
        assert_eq!((kw % 32, vw % 32), (0, 0),
                   "MLA K width {kw} and V width {vw} must be multiples of the 32-value block");
        assert_ne!(kw, vw, "this runtime's K and V widths differ — that is the case being pinned");
    }
}
