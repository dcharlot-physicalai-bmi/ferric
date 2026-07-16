//! Ferric Llama bridge — maps a Llama/SmolLM-family checkpoint (HF safetensors tensor layout) onto
//! Ferric's kernels and runs the forward pass. Handles the HF conventions: linear weights stored
//! `[out, in]` (applied via `mm_bt` = x·Wᵀ), grouped-query attention (`n_kv_heads` < `n_heads`),
//! RMSNorm, rotate-half RoPE, SwiGLU MLP, and optionally tied embeddings (`lm_head` == `embed_tokens`).
//!
//! This is what turns "a transformer forward pass" into "runs Llama": point it at a real checkpoint's
//! bytes + config and it generates. No Python, no C++ — pure Rust, cross-fabric.

use ferric_core::{Context, Tensor};
use ferric_load::{safetensors, STensor};
use std::collections::HashMap;

pub struct Config {
    pub n_layers: usize,
    pub d: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

pub struct Llama {
    pub cfg: Config,
    w: HashMap<String, STensor>,
}

impl Llama {
    pub fn from_safetensors(bytes: &[u8], cfg: Config) -> Result<Self, String> {
        Ok(Llama { cfg, w: safetensors(bytes)? })
    }

    fn get(&self, name: &str) -> Result<&STensor, String> {
        self.w.get(name).ok_or_else(|| format!("missing weight '{name}'"))
    }
    /// Linear layer y = x·Wᵀ, W stored HF-style as [out, in]. Returns [m, out].
    fn linear(&self, ctx: &Context, x: &Tensor, name: &str, m: u32) -> Result<Tensor, String> {
        let w = self.get(name)?;
        let (out, inp) = (w.shape[0] as u32, w.shape[1] as u32);
        let wt = ctx.tensor(&w.data, &w.shape);
        Ok(ctx.mm_bt(x, &wt, m, out, inp, 1.0))
    }
    fn rmsnorm(&self, ctx: &Context, x: &Tensor, name: &str, rows: u32) -> Result<Tensor, String> {
        let w = self.get(name)?;
        let wt = ctx.tensor(&w.data, &w.shape);
        Ok(ctx.rmsnorm_t(x, &wt, rows, self.cfg.d as u32, self.cfg.eps))
    }

    /// Prefill forward over `ids`. Returns logits [T, vocab].
    pub async fn forward(&self, ctx: &Context, ids: &[u32]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let t = ids.len() as u32;
        let (nh, nkv, dh, base) = (c.n_heads as u32, c.n_kv_heads as u32, c.head_dim as u32, c.rope_theta);

        let emb = self.get("model.embed_tokens.weight")?;
        let embt = ctx.tensor(&emb.data, &emb.shape);
        let mut x = ctx.gather0(&embt, ids, c.d);

        for i in 0..c.n_layers {
            let p = format!("model.layers.{i}");
            let h = self.rmsnorm(ctx, &x, &format!("{p}.input_layernorm.weight"), t)?;
            let q = ctx.rope_t(&self.linear(ctx, &h, &format!("{p}.self_attn.q_proj.weight"), t)?, t, nh, dh, base);
            let k = ctx.rope_t(&self.linear(ctx, &h, &format!("{p}.self_attn.k_proj.weight"), t)?, t, nkv, dh, base);
            let v = self.linear(ctx, &h, &format!("{p}.self_attn.v_proj.weight"), t)?;
            let attn = ctx.mha_causal_t(&q, &k, &v, t, nh, nkv, dh);
            x = ctx.add_t(&x, &self.linear(ctx, &attn, &format!("{p}.self_attn.o_proj.weight"), t)?);

            let h2 = self.rmsnorm(ctx, &x, &format!("{p}.post_attention_layernorm.weight"), t)?;
            let g = self.linear(ctx, &h2, &format!("{p}.mlp.gate_proj.weight"), t)?;
            let up = self.linear(ctx, &h2, &format!("{p}.mlp.up_proj.weight"), t)?;
            let act = ctx.mul_t(&g, &ctx.sigmoid_t(&g)); // SiLU
            x = ctx.add_t(&x, &self.linear(ctx, &ctx.mul_t(&act, &up), &format!("{p}.mlp.down_proj.weight"), t)?);
        }

        let x = self.rmsnorm(ctx, &x, "model.norm.weight", t)?;
        // tied embeddings: fall back to embed_tokens if there's no separate lm_head
        let head = if self.w.contains_key("lm_head.weight") { "lm_head.weight" } else { "model.embed_tokens.weight" };
        let logits = self.linear(ctx, &x, head, t)?;
        ctx.to_vec(&logits).await
    }
}

pub mod qwen35;
pub mod qwen3;
