//! NVIDIA **Cosmos 3 Edge** (`cosmos3_edge`, July 2026) — the autoregressive TEXT/reasoning tower of
//! the 4B omnimodal world-foundation model, running on Ferric. Cosmos 3 Edge is a Mixture-of-
//! Transformers (an AR tower for text/vision understanding + a diffusion tower for video/audio/action
//! generation); this module is Stage 1 — the AR language tower, which is a dense GQA transformer with
//! three twists vs a Qwen3: a NON-gated **ReLU²** FFN (`down(relu²(up(x)))`), per-head **k-norm only**
//! (no q-norm), and RoPE θ=1e8 (M-RoPE, which for text degenerates to standard 1D RoPE).
//!
//! Weights load from the model's BF16 **safetensors** (via `ferric_load`, selectively — the shards
//! also hold the diffusion tower + vision encoder we skip here). fp forward on the general runtime.
use ferric_core::Context;
use ferric_load::safetensors_filtered;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

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
}

impl Cfg {
    /// Cosmos 3 Edge text_config (config.json, verified 2026-07-23).
    pub fn cosmos3_edge() -> Cfg {
        Cfg { n_embd: 2048, n_layer: 28, n_head: 16, n_head_kv: 8, head_dim: 128, n_ff: 9216, n_vocab: 131072, eps: 1e-5, rope_base: 1e8 }
    }
}

struct Layer {
    in_ln: Tensor,   // input_layernorm [d]
    to_q: Tensor,    // [nh·hd, d]
    to_k: Tensor,    // [nkv·hd, d]
    to_v: Tensor,    // [nkv·hd, d]
    to_out: Tensor,  // [d, nh·hd]
    k_norm: Tensor,  // k_norm_und_for_gen [hd] — applied per k-head (no q-norm in Cosmos)
    post_ln: Tensor, // post_attention_layernorm [d]
    up: Tensor,      // mlp.up_proj [ff, d]
    down: Tensor,    // mlp.down_proj [d, ff]
}

pub struct Cosmos {
    pub cfg: Cfg,
    ctx: Arc<Context>,
    embed: Tensor,   // embed_tokens [vocab, d]
    layers: Vec<Layer>,
    norm: Tensor,    // final norm [d]
    lm_head: Tensor, // [vocab, d]
}

impl Cosmos {
    /// Load the AR text tower from a Cosmos 3 Edge checkpoint directory containing the transformer
    /// safetensors shards (`transformer/*.safetensors`). Only the language tensors are materialized.
    pub fn load(ctx: &Arc<Context>, dir: &str) -> Result<Cosmos, String> {
        let cfg = Cfg::cosmos3_edge();
        // The AR language tower: token embedding, the 28 `layers.N.*`, the final norm, the LM head.
        let keep = |n: &str| n == "embed_tokens.weight" || n == "norm.weight" || n == "lm_head.weight" || n.starts_with("layers.");
        let mut w: HashMap<String, ferric_load::STensor> = HashMap::new();
        for entry in std::fs::read_dir(format!("{dir}/transformer")).map_err(|e| format!("read_dir: {e}"))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") { continue; }
            let bytes = std::fs::read(&path).map_err(|e| format!("read {path:?}: {e}"))?;
            w.extend(safetensors_filtered(&bytes, keep)?);
        }
        let take = |w: &mut HashMap<String, ferric_load::STensor>, name: &str| -> Result<Tensor, String> {
            let s = w.remove(name).ok_or_else(|| format!("missing tensor {name}"))?;
            Ok(Tensor::from_vec(ctx, &s.data, &s.shape))
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("layers.{il}.{s}");
            layers.push(Layer {
                in_ln: take(&mut w, &b("input_layernorm.weight"))?,
                to_q: take(&mut w, &b("self_attn.to_q.weight"))?,
                to_k: take(&mut w, &b("self_attn.to_k.weight"))?,
                to_v: take(&mut w, &b("self_attn.to_v.weight"))?,
                to_out: take(&mut w, &b("self_attn.to_out.weight"))?,
                k_norm: take(&mut w, &b("self_attn.k_norm_und_for_gen.weight"))?,
                post_ln: take(&mut w, &b("post_attention_layernorm.weight"))?,
                up: take(&mut w, &b("mlp.up_proj.weight"))?,
                down: take(&mut w, &b("mlp.down_proj.weight"))?,
            });
        }
        Ok(Cosmos {
            embed: take(&mut w, "embed_tokens.weight")?,
            norm: take(&mut w, "norm.weight")?,
            lm_head: take(&mut w, "lm_head.weight")?,
            cfg, ctx: ctx.clone(), layers,
        })
    }

    /// Stateless prefill: logits [T, vocab] for the whole sequence (no KV cache — re-run per token for
    /// a simple greedy loop; a cache is the obvious next step).
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        use ferric_tensor::batch;
        let c = &self.cfg;
        let t = tokens.len();
        let (nh, nkv, hd) = (c.n_head, c.n_head_kv, c.head_dim);
        let mut x = self.embed.gather_rows(tokens); // [t, d]
        for l in &self.layers {
            x = batch(&self.ctx, || {
                let h = x.rmsnorm(&l.in_ln, c.eps);
                // Reasoner attention (per the real modeling code): q_proj/k_proj/v_proj → RoPE → GQA,
                // NO q/k-norm — `k_norm_und_for_gen` is a GENERATION-path weight, unused in the pure
                // text reasoner. RoPE θ=1e8; M-RoPE → 1D for text.
                let q = h.matmul_bt(&l.to_q).rope(nh, hd, c.rope_base, 0);
                let k = h.matmul_bt(&l.to_k).rope(nkv, hd, c.rope_base, 0);
                let v = h.matmul_bt(&l.to_v);
                let o = if t == 1 { nn::decode_attention(&q, &k, &v, nh, nkv, 0.0) } else { nn::causal_attention(&q, &k, &v, nh, nkv, 0.0) };
                let x1 = x.add(&o.matmul_bt(&l.to_out));
                // non-gated ReLU² FFN: down(relu²(up(h)))
                let ffn = x1.rmsnorm(&l.post_ln, c.eps).matmul_bt(&l.up).relu2().matmul_bt(&l.down);
                x1.add(&ffn)
            });
        }
        batch(&self.ctx, || x.rmsnorm(&self.norm, c.eps).matmul_bt(&self.lm_head))
    }
}

// ── Stage 2: the SigLIP vision encoder + projector ──────────────────────────────────────────────
struct VLayer { ln1_w: Tensor, ln1_b: Tensor, q: Tensor, qb: Tensor, k: Tensor, kb: Tensor, v: Tensor, vb: Tensor,
    o: Tensor, ob: Tensor, ln2_w: Tensor, ln2_b: Tensor, fc1: Tensor, fc1b: Tensor, fc2: Tensor, fc2b: Tensor }

pub struct CosmosVision {
    ctx: Arc<Context>,
    patch_w: Tensor, patch_b: Tensor,   // linear patch embed [1152, 768] (768 = 3·16·16, conv order)
    pos: Tensor,                        // position_embedding [256, 1152]
    layers: Vec<VLayer>,
    post_w: Tensor, post_b: Tensor,     // post_layernorm
    proj_nw: Tensor, proj_nb: Tensor,   // projector pre-shuffle norm [1152]
    fc1: Tensor, fc1b: Tensor, fc2: Tensor, fc2b: Tensor, // projector MLP 4608→11520→2048
}

impl CosmosVision {
    pub const D: usize = 1152;
    pub const HEADS: usize = 16;
    pub const PATCHES: usize = 256; // 16×16 grid for a 256px image
    const EPS: f32 = 1e-6;

    pub fn load(ctx: &Arc<Context>, dir: &str) -> Result<CosmosVision, String> {
        let bytes = std::fs::read(format!("{dir}/vision_encoder/model.safetensors")).map_err(|e| format!("read vision: {e}"))?;
        let mut w = ferric_load::safetensors(&bytes)?;
        let mut t = |name: &str| -> Result<Tensor, String> {
            let s = w.remove(name).ok_or_else(|| format!("missing {name}"))?;
            Ok(Tensor::from_vec(ctx, &s.data, &s.shape))
        };
        let mut layers = Vec::with_capacity(27);
        for il in 0..27 {
            let b = |s: &str| format!("model.visual.encoder.layers.{il}.{s}");
            layers.push(VLayer {
                ln1_w: t(&b("layer_norm1.weight"))?, ln1_b: t(&b("layer_norm1.bias"))?,
                q: t(&b("self_attn.q_proj.weight"))?, qb: t(&b("self_attn.q_proj.bias"))?,
                k: t(&b("self_attn.k_proj.weight"))?, kb: t(&b("self_attn.k_proj.bias"))?,
                v: t(&b("self_attn.v_proj.weight"))?, vb: t(&b("self_attn.v_proj.bias"))?,
                o: t(&b("self_attn.out_proj.weight"))?, ob: t(&b("self_attn.out_proj.bias"))?,
                ln2_w: t(&b("layer_norm2.weight"))?, ln2_b: t(&b("layer_norm2.bias"))?,
                fc1: t(&b("mlp.fc1.weight"))?, fc1b: t(&b("mlp.fc1.bias"))?,
                fc2: t(&b("mlp.fc2.weight"))?, fc2b: t(&b("mlp.fc2.bias"))?,
            });
        }
        Ok(CosmosVision {
            patch_w: t("model.visual.embeddings.patch_embedding.weight")?, patch_b: t("model.visual.embeddings.patch_embedding.bias")?,
            pos: t("model.visual.embeddings.position_embedding.weight")?,
            post_w: t("model.visual.post_layernorm.weight")?, post_b: t("model.visual.post_layernorm.bias")?,
            proj_nw: t("model.projector.norm.weight")?, proj_nb: t("model.projector.norm.bias")?,
            fc1: t("model.projector.linear_fc1.weight")?, fc1b: t("model.projector.linear_fc1.bias")?,
            fc2: t("model.projector.linear_fc2.weight")?, fc2b: t("model.projector.linear_fc2.bias")?,
            ctx: ctx.clone(), layers,
        })
    }

    /// `patches`: [256, 768] flattened patches in Conv2d (channel,h,w) order. Returns the [64, 2048]
    /// vision tokens ready to splice into the LM's token stream (256 patches → 64 after 2×2 merge).
    pub fn forward(&self, patches: &[f32]) -> Tensor {
        use ferric_tensor::{batch, nn};
        let n = Self::PATCHES;
        let bias = |x: Tensor, b: &Tensor| x.add(&b.reshape(&[1, b.shape[0]]));
        let lin = |x: &Tensor, w: &Tensor, b: &Tensor| bias(x.matmul_bt(w), b);
        let p = Tensor::from_vec(&self.ctx, patches, &[n, 768]);
        // Reorder the learned position embeddings to BLOCK-MAJOR to match `resize_positional_embeddings`
        // (real code: reshape [8,2,8,2,d] → transpose(1,2) → flatten): output i=((by·8+bx)·2+dy)·2+dx
        // takes native raster pos (by·2+dy)·16+(bx·2+dx). Patches are assumed block-major (image_processor).
        let (gg, gb) = (16usize, 8usize);
        let mut perm = vec![0u32; n];
        for by in 0..gb { for bx in 0..gb { for dy in 0..2 { for dx in 0..2 {
            perm[((by * gb + bx) * 2 + dy) * 2 + dx] = ((by * 2 + dy) * gg + (bx * 2 + dx)) as u32;
        }}}}
        let pos_bm = self.pos.gather_rows(&perm);
        let mut x = bias(p.matmul_bt(&self.patch_w), &self.patch_b).add(&pos_bm); // [256, 1152]
        for l in &self.layers {
            x = batch(&self.ctx, || {
                let h = x.layernorm(&l.ln1_w, &l.ln1_b, Self::EPS);
                let (q, k, v) = (lin(&h, &l.q, &l.qb), lin(&h, &l.k, &l.kb), lin(&h, &l.v, &l.vb));
                let o = nn::bidirectional_attention(&q, &k, &v, Self::HEADS, Self::HEADS); // full attention
                let x1 = x.add(&lin(&o, &l.o, &l.ob));
                let h2 = x1.layernorm(&l.ln2_w, &l.ln2_b, Self::EPS);
                let mlp = lin(&lin(&h2, &l.fc1, &l.fc1b).gelu_tanh(), &l.fc2, &l.fc2b);
                x1.add(&mlp)
            });
        }
        x = x.layernorm(&self.post_w, &self.post_b, Self::EPS);
        // Projector (real `Cosmos3EdgePatchMerger`): norm(1152) per patch → group each 2×2 block →
        // fc1 → erf-GELU → fc2. The image_processor packs patches in BLOCK-MAJOR order (2×2 blocks
        // consecutive), so the merge is a plain reshape [256,1152]→[64,4608] (group consecutive 4),
        // NOT a raster gather. ⚠️ Correct ONLY if `patches` arrive block-major (match the real
        // image_processor for real images — the synthetic-patch path is order-agnostic).
        x = x.layernorm(&self.proj_nw, &self.proj_nb, Self::EPS);
        let merged = x.reshape(&[Self::PATCHES / 4, 4 * Self::D]); // [64, 4608]
        batch(&self.ctx, || bias(bias(merged.matmul_bt(&self.fc1), &self.fc1b).gelu().matmul_bt(&self.fc2), &self.fc2b))
    }
}

/// Cosmos 3 Edge **interleaved 3-axis mRoPE** cos/sin table (`Cosmos3VLTextRotaryEmbedding`).
/// The `head_dim/2` frequency slots are partitioned `[axes.0 T, axes.1 H, axes.2 W]` and interleaved
/// so slot `i` (for `i < 3·(axes.1) == 3·(axes.2)`) takes axis `i % 3` (0=T,1=H,2=W) and the tail
/// slots stay on the T axis. For text tokens (T==H==W position) this collapses to standard 1D RoPE.
/// Returns `(cos, sin)`, each row-major `[n_tokens * head_dim]` (the `[freqs, freqs]` doubled layout,
/// pairing with the split-half `rotate_half` convention). Verified vs the real module to 3.9e-8.
pub fn interleaved_mrope(
    pos_t: &[i64],
    pos_h: &[i64],
    pos_w: &[i64],
    head_dim: usize,
    theta: f64,
    axes: (usize, usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let n = pos_t.len();
    let half = head_dim / 2;
    // inv_freq[j] = theta^(-(2j)/head_dim)
    let inv_freq: Vec<f64> = (0..half).map(|j| theta.powf(-((2 * j) as f64) / head_dim as f64)).collect();
    // pick source axis position per (token, slot): interleave T/H/W then T-tail
    let len_h = axes.1 * 3;
    let len_w = axes.2 * 3;
    let mut cos = vec![0f32; n * head_dim];
    let mut sin = vec![0f32; n * head_dim];
    for tok in 0..n {
        for j in 0..half {
            // default T; H overwrites slots {1,4,..<len_h}; W overwrites {2,5,..<len_w}
            let p = if j % 3 == 1 && j < len_h { pos_h[tok] }
                    else if j % 3 == 2 && j < len_w { pos_w[tok] }
                    else { pos_t[tok] };
            let f = inv_freq[j] * p as f64;
            let (c, s) = (f.cos() as f32, f.sin() as f32);
            cos[tok * head_dim + j] = c;             // first half
            cos[tok * head_dim + half + j] = c;      // doubled
            sin[tok * head_dim + j] = s;
            sin[tok * head_dim + half + j] = s;
        }
    }
    (cos, sin)
}
