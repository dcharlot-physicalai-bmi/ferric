//! **BERT encoders** — the architecture the field's small embedding and reranker checkpoints use.
//!
//! bge, gte, MiniLM, e5 and the cross-encoder rerankers are all BERT, and none of them could run
//! here before this. The size argument is the whole point: `bge-small-en-v1.5` is **67 MB** against
//! the 396 MB decoder-based retriever this project had been using for the same job.
//!
//! It is also the SIMPLEST runtime in this crate, which is worth stating because the instinct is to
//! expect the opposite from an unfamiliar architecture:
//!
//! | | decoder (qwen3 &c.) | BERT encoder |
//! |---|---|---|
//! | attention mask | causal | **none — bidirectional** |
//! | positions | RoPE, applied per step | **learned lookup, added once** |
//! | KV cache | required | **none; one forward over the whole sequence** |
//! | norm | pre-RMSNorm | **post-LayerNorm, with bias** |
//! | FFN | gated SwiGLU | **plain GELU** |
//!
//! `bert.attention.causal = false` is read rather than assumed, because an encoder run with a causal
//! mask does not fail — it returns embeddings where every token saw only its left context, which is
//! a different and worse vector that no test comparing Ferric to itself could catch.
use ferric_gguf::{GgufSource, Meta};
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

pub struct Cfg {
    pub n_layer: usize,
    pub d: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub eps: f32,
    /// GGUF `bert.pooling_type`: 1 MEAN, 2 CLS. Read, never assumed — see the module note.
    pub pooling: u32,
    pub causal: bool,
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let u = |k: &str| match md.get(&format!("bert.{k}")) {
            Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing bert.{k}")),
        };
        let f = |k: &str| match md.get(&format!("bert.{k}")) {
            Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing bert.{k}")),
        };
        let causal = matches!(md.get("bert.attention.causal"), Some(Meta::Bool(true)));
        if causal {
            return Err("bert.attention.causal is true; this runtime is bidirectional only".into());
        }
        let n_vocab = g.tensor("token_embd.weight").ok_or("no token_embd.weight")?.dims[1] as usize;
        Ok(Cfg {
            n_layer: u("block_count")?,
            d: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_ff: u("feed_forward_length")?,
            n_ctx: u("context_length").unwrap_or(512),
            n_vocab,
            eps: f("attention.layer_norm_epsilon").unwrap_or(1e-12),
            pooling: match md.get("bert.pooling_type") { Some(Meta::U(v)) => *v as u32, _ => 2 },
            causal,
        })
    }
}

struct Block {
    q: Tensor, qb: Tensor,
    k: Tensor, kb: Tensor,
    v: Tensor, vb: Tensor,
    o: Tensor, ob: Tensor,
    attn_norm_w: Tensor, attn_norm_b: Tensor,
    up: Tensor, upb: Tensor,
    down: Tensor, downb: Tensor,
    out_norm_w: Tensor, out_norm_b: Tensor,
}

pub struct Bert {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    tok_embd: Vec<f32>,      // [n_vocab, d], host-side: one row per lookup, no GPU gather needed
    pos_embd: Vec<f32>,      // [n_ctx, d]
    typ_embd: Vec<f32>,      // [n_type, d]
    embd_norm_w: Tensor, embd_norm_b: Tensor,
    blocks: Vec<Block>,
}

/// Load a `[rows, cols]` tensor as f32. GGUF dims are reversed relative to row-major, so a weight
/// listed `[in, out]` is `[out, in]` in memory — which is exactly `matmul_bt`'s expected layout.
fn t2(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Tensor, String> {
    let i = g.tensor(name).ok_or_else(|| format!("no {name}"))?;
    let (a, b) = (i.dims[0] as usize, *i.dims.get(1).unwrap_or(&1) as usize);
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, &[b, a]))
}
fn t1(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Tensor, String> {
    let i = g.tensor(name).ok_or_else(|| format!("no {name}"))?;
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, &[1, i.dims[0] as usize]))
}

impl Bert {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Bert, String> {
        let cfg = Cfg::from_gguf(g)?;
        let blocks = (0..cfg.n_layer).map(|i| {
            let n = |s: &str| format!("blk.{i}.{s}");
            Ok(Block {
                q: t2(ctx, g, &n("attn_q.weight"))?,  qb: t1(ctx, g, &n("attn_q.bias"))?,
                k: t2(ctx, g, &n("attn_k.weight"))?,  kb: t1(ctx, g, &n("attn_k.bias"))?,
                v: t2(ctx, g, &n("attn_v.weight"))?,  vb: t1(ctx, g, &n("attn_v.bias"))?,
                o: t2(ctx, g, &n("attn_output.weight"))?, ob: t1(ctx, g, &n("attn_output.bias"))?,
                attn_norm_w: t1(ctx, g, &n("attn_output_norm.weight"))?,
                attn_norm_b: t1(ctx, g, &n("attn_output_norm.bias"))?,
                up: t2(ctx, g, &n("ffn_up.weight"))?, upb: t1(ctx, g, &n("ffn_up.bias"))?,
                down: t2(ctx, g, &n("ffn_down.weight"))?, downb: t1(ctx, g, &n("ffn_down.bias"))?,
                out_norm_w: t1(ctx, g, &n("layer_output_norm.weight"))?,
                out_norm_b: t1(ctx, g, &n("layer_output_norm.bias"))?,
            })
        }).collect::<Result<Vec<_>, String>>()?;
        Ok(Bert {
            ctx: ctx.clone(),
            tok_embd: g.dequant("token_embd.weight")?,
            pos_embd: g.dequant("position_embd.weight")?,
            typ_embd: g.dequant("token_types.weight")?,
            embd_norm_w: t1(ctx, g, "token_embd_norm.weight")?,
            embd_norm_b: t1(ctx, g, "token_embd_norm.bias")?,
            blocks, cfg,
        })
    }

    /// One bidirectional forward over the whole sequence. Returns `[t, d]` hidden states.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor, String> {
        let (d, t) = (self.cfg.d, ids.len());
        if t > self.cfg.n_ctx {
            return Err(format!("{t} tokens exceeds this encoder's {} position embeddings; BERT has \
                                no RoPE to extrapolate with, so a longer input must be truncated by \
                                the caller rather than silently wrapped", self.cfg.n_ctx));
        }
        // token + position + segment, summed on the host — three gathers over a 30k-row table are
        // cheaper to index here than to dispatch.
        let mut e = vec![0f32; t * d];
        for (p, &id) in ids.iter().enumerate() {
            let (tk, ps) = ((id as usize) * d, p * d);
            for j in 0..d {
                e[ps + j] = self.tok_embd[tk + j] + self.pos_embd[ps + j] + self.typ_embd[j];
            }
        }
        let mut h = Tensor::from_vec(&self.ctx, &e, &[t, d])
            .layernorm(&self.embd_norm_w, &self.embd_norm_b, self.cfg.eps);

        let (nh, dh) = (self.cfg.n_head, d / self.cfg.n_head);
        let scale = 1.0 / (dh as f32).sqrt();
        for b in &self.blocks {
            let q = h.matmul_bt(&b.q).add(&b.qb);
            let k = h.matmul_bt(&b.k).add(&b.kb);
            let v = h.matmul_bt(&b.v).add(&b.vb);
            // [t, nh, dh] → [nh, t, dh] so each head is a contiguous [t, dh] slab.
            let sh = |x: &Tensor| x.reshape(&[t, nh, dh]).permute(&[1, 0, 2]).contiguous();
            let (q, k, v) = (sh(&q), sh(&k), sh(&v));
            let mut heads: Vec<Tensor> = Vec::with_capacity(nh);
            for i in 0..nh {
                let qi = q.narrow(0, i, 1).reshape(&[t, dh]);
                let ki = k.narrow(0, i, 1).reshape(&[t, dh]);
                let vi = v.narrow(0, i, 1).reshape(&[t, dh]);
                // NO causal mask: every token attends to every token, which is the defining
                // difference from every other runtime in this crate.
                // 1/sqrt(dh) as a broadcast multiply — the scale must land BEFORE the softmax, or
                // the distribution sharpens with head width and every embedding shifts.
                let sc = Tensor::from_vec(&self.ctx, &[scale], &[1, 1]).broadcast_to(&[t, t]);
                let a = qi.matmul_bt(&ki).mul(&sc).softmax(1);
                heads.push(a.matmul(&vi));
            }
            // Heads back to [t, d] in head order, which is the layout attn_output.weight expects.
            let cat = heads.iter().skip(1).fold(heads[0].clone(), |acc, x| acc.cat(x, 1));
            let attn = cat.reshape(&[t, d]).matmul_bt(&b.o).add(&b.ob);
            // POST-norm: normalise the residual sum, not the input to the sublayer.
            h = h.add(&attn).layernorm(&b.attn_norm_w, &b.attn_norm_b, self.cfg.eps);
            let ff = h.matmul_bt(&b.up).add(&b.upb).gelu().matmul_bt(&b.down).add(&b.downb);
            h = h.add(&ff).layernorm(&b.out_norm_w, &b.out_norm_b, self.cfg.eps);
        }
        Ok(h)
    }
}
