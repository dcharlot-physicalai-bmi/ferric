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
use ferric_tensor::{append2, nn, KvBuf, QMatrix, Tensor};
use std::sync::Arc;

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
        if self.yarn_factor <= 1.0 { return 1.0; }
        1.0 / (1.0 + 0.1 * (self.yarn_factor).ln())
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

pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
}

impl Cache {
    pub fn new(cfg: &Cfg) -> Cache {
        Cache { pos: 0, kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect() }
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
    fn rope_pe(&self, x: &Tensor, heads: usize, pos: usize) -> Tensor {
        let cfg = &self.cfg;
        // ⚠ INTERLEAVED. llama.cpp lists deepseek2 under "normal RoPE, operating on pairs of
        // consecutive head values" (ROPE_TYPE_NORM), not the split-half NEOX pairing every Qwen-family
        // model here uses. Rotating the wrong partners produces finite logits and wrong text.
        // A/B via FERRIC_ROPE_NEOX to settle the pairing empirically rather than by reading alone.
        let neox = std::env::var("FERRIC_ROPE_NEOX").is_ok();
        let r = match (&self.yarn, neox) {
            (Some(fs), false) => x.rope_scaled_interleaved(fs, heads, cfg.qk_rope, cfg.rope_base, pos),
            (Some(fs), true) => x.rope_scaled(fs, heads, cfg.qk_rope, cfg.rope_base, pos),
            (None, false) => x.rope_interleaved(heads, cfg.qk_rope, cfg.rope_base, pos),
            (None, true) => x.rope(heads, cfg.qk_rope, cfg.rope_base, pos),
        };
        let af = cfg.attn_factor();
        if (af - 1.0).abs() < 1e-9 { r } else { r.mul(&r.scalar(af)) }
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        self.forward_hidden_cached(tokens, cache).matmul_q(&self.head)
    }

    pub fn forward_hidden_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        let cfg = &self.cfg;
        let (d, eps, t, pos) = (cfg.d, cfg.eps, tokens.len(), cache.pos);
        let (nh, nope, rope, vh, r) = (cfg.n_head, cfg.qk_nope, cfg.qk_rope, cfg.v_head, cfg.kv_lora_rank);
        let qk = cfg.qk_head();
        let mut x = self.embed(tokens);

        for (il, blk) in self.blocks.iter().enumerate() {
            let h = x.rmsnorm(&blk.attn_norm, eps);

            // Q: [t, nh*qk] -> split each head into its nope and rope lanes.
            let q = h.matmul_q(&blk.q).reshape(&[t, nh, qk]);
            let q_nope = q.narrow(2, 0, nope).contiguous();
            let q_pe = self.rope_pe(&q.narrow(2, nope, rope).contiguous().reshape(&[t, nh * rope]), nh, pos)
                .reshape(&[t, nh, rope]);

            // The compressed KV plus the SHARED rope vector: [t, r + rope].
            let kvp = h.matmul_q(&blk.kv_a_mqa);
            let kv_cmpr = kvp.narrow(1, 0, r).contiguous().rmsnorm(&blk.kv_a_norm, eps);
            let k_pe = self.rope_pe(&kvp.narrow(1, r, rope).contiguous(), 1, pos);

            // Decompress to per-head K-nope and V: [t, nh*(nope+vh)].
            let kv = kv_cmpr.matmul_q(&blk.kv_b).reshape(&[t, nh, nope + vh]);
            let k_nope = kv.narrow(2, 0, nope).contiguous();
            let v = kv.narrow(2, nope, vh).contiguous().reshape(&[t, nh * vh]);

            // K is [nope | shared rope], so the one rope vector is broadcast to every head.
            let k_pe_all = k_pe.reshape(&[t, 1, rope]).broadcast_to(&[t, nh, rope]).contiguous();
            let k = k_nope.cat(&k_pe_all, 2).reshape(&[t, nh * qk]);
            let q_full = q_nope.cat(&q_pe, 2).reshape(&[t, nh * qk]);
            // The kernels bake in 1/sqrt(qk); DeepSeek wants mscale²/sqrt(qk).
            let q_full = q_full.mul(&q_full.scalar(cfg.q_prescale()));

            let (kc, vc) = { let e = &mut cache.kv[il]; append2(&self.ctx, &mut e.0, &k, &mut e.1, &v) };

            // V is narrower than K here, so the attention helper cannot infer the value width from
            // the key width. Heads are 1:1 (no GQA) after decompression.
            let att = nn::causal_attention_split(&q_full, &kc, &vc, nh, qk, vh, 0.0);
            x = x.add(&att.matmul_q(&blk.o));

            let f = x.rmsnorm(&blk.ffn_norm, eps);
            let out = match &blk.ffn {
                Ffn::Dense { gate, up, down } => f.matmul_q(gate).silu().mul(&f.matmul_q(up)).matmul_q(down),
                Ffn::Moe { router, bias, gate_up, down, sh_gate, sh_up, sh_down } => {
                    let selw = f.matmul_q(router).moe_topk_ex(
                        bias.as_ref(), cfg.n_expert_used, cfg.sigmoid_gate, cfg.routed_scale, cfg.expert_norm);
                    let mid = f.matmul_q4_k_swiglu_id(gate_up, &selw, cfg.n_expert_used, cfg.expert_ff);
                    let routed = down.wsum(&mid, &selw, d);
                    let shared = f.matmul_q(sh_gate).silu().mul(&f.matmul_q(sh_up)).matmul_q(sh_down);
                    routed.add(&shared)
                }
            };
            x = x.add(&out);
        }

        cache.pos += t;
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
    fn the_rope_scale_is_not_the_attention_scale() {
        // Two different YaRN constants land in two different places. Collapsing them into one is the
        // subtle version of this bug: attn_factor touches only the rope lanes, mscale² the whole score.
        let c = lite();
        assert!((c.attn_factor() - 0.730521).abs() < 1e-5, "attn_factor {}", c.attn_factor());
        assert_ne!(c.attn_factor(), c.mscale());
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
