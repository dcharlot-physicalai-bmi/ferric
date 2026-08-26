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
//! ## ⚠ VERIFIED FOR BERT, **NOT** FOR XLM-ROBERTA
//!
//! Reference-diffed against `llama-embedding`:
//!
//! | checkpoint | arch | quant | cosine |
//! |---|---|---|---|
//! | bge-small-en-v1.5 | BERT, 12L d=384 | F16 | **0.999999–1.000000** |
//! | bge-small-en-v1.5 | BERT, 12L d=384 | Q4_K_M | **0.999996** |
//! | bge-reranker-v2-m3 | XLM-R, 24L d=1024 | Q4_K_M | **0.9615** ❌ |
//!
//! The XLM-R gap is an OPEN BUG. What it is not, each ruled out by measurement rather than argument:
//!
//! * **not quantisation** — Q4_K_M bge-small is 0.999996, same code path, so the Q4_K matmul agrees
//!   with llama.cpp. (This also clears every Q4_K result elsewhere in the project.)
//! * **not tokenisation** — Ferric's ids are identical to `llama-tokenize` on both halves of a pair.
//! * **not the classification head** — the plain EMBEDDING path diverges by the same amount.
//! * **not accumulation over depth or length** — it is 0.926 at FOUR tokens and 0.961 at fourteen,
//!   so it is present from the first block rather than compounding.
//! * **not the LayerNorm epsilon** — declared 1e-5 here against bge-small's 1e-12, and read correctly.
//! * **not primarily the position offset** — sweeping 0..3 moves it 0.9615 → 0.9762, never near 0.999.
//! * **not the GELU variant** — ggml uses the tanh approximation and this now does too, worth 0.0004
//!   (0.972166 erf → 0.972585 tanh). A real correctness fix, not the bug.
//! * **not the pooler** — `--pooling cls` returns the RAW CLS state, confirmed by comparing against
//!   `tanh(dense(CLS))` instead: cosine 0.032, near-orthogonal. The two are very different vectors,
//!   so this rules the question out rather than leaving it ambiguous.
//! * **not the token-type embedding** — non-zero here (max |v| 0.205) and added by both.
//!
//! Best achieved: **0.9726** (offset 2, tanh GELU). The remaining gap is roughly 0.027 of cosine and
//! no single-parameter hypothesis accounts for it.
//!
//! **The right next instrument is `llama-eval-callback`**, which dumps every tensor in llama.cpp's
//! graph. Comparing per-LAYER hidden states finds the first block where the two diverge, which is a
//! bounded search; comparing whole-model outputs and guessing at parameters is not, and four wrong
//! turns here are the evidence for that.
//!
//! So it is structural and specific to this checkpoint family, in the embedding construction or an
//! early block. The reranker head itself is implemented (`score`) and DISCRIMINATES correctly —
//! +6 relevant, −9 irrelevant, same ordering as llama.cpp — but the logits differ, so `/v1/rerank`
//! is deliberately NOT built on it yet. Ordering being right is exactly what would make a rank
//! benchmark pass while the model is wrong.
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
    /// **Where position 0 lives in `position_embd`.** RoBERTa-family checkpoints (XLM-R, and so
    /// bge-reranker-v2-m3) reserve the first slots for padding and start real positions at
    /// `padding_idx + 1` = 2; original BERT starts at 0. The table is sized for the offset —
    /// bge-reranker declares 8192 rows for a 512-token context — so reading from 0 silently uses the
    /// wrong row for EVERY token and produces a plausible, wrong score.
    pub pos_offset: usize,
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
            // Inferred from the padding id, which is what the convention is actually keyed on:
            // RoBERTa sets pad=1 and starts positions at 2; BERT sets pad=0 and starts at 0.
            pos_offset: std::env::var("FERRIC_BERT_POS_OFFSET").ok().and_then(|v| v.parse().ok())
                .unwrap_or(match md.get("tokenizer.ggml.padding_token_id") {
                    Some(Meta::U(v)) if *v > 0 => *v as usize + 1,
                    _ => 0,
                }),
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
    /// The **classification head**, present only on cross-encoder rerankers. Its existence is what
    /// separates a reranker from an embedder in an otherwise identical file: same `general.architecture
    /// = bert`, same blocks, but four extra tensors that turn a pooled vector into ONE relevance
    /// score. Absent on bge-small; present on bge-reranker-v2-m3.
    cls: Option<ClsHead>,
}

struct ClsHead { w: Tensor, b: Tensor, ow: Tensor, ob: Tensor }

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
            cls: match g.tensor("cls.weight") {
                Some(_) => Some(ClsHead {
                    w: t2(ctx, g, "cls.weight")?, b: t1(ctx, g, "cls.bias")?,
                    ow: t2(ctx, g, "cls.output.weight")?, ob: t1(ctx, g, "cls.output.bias")?,
                }),
                None => None,
            },
            blocks, cfg,
        })
    }

    /// One bidirectional forward over the whole sequence. Returns `[t, d]` hidden states.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor, String> {
        let (d, t) = (self.cfg.d, ids.len());
        if t + self.cfg.pos_offset > self.pos_embd.len() / d {
            return Err(format!("{t} tokens exceeds this encoder's {} position embeddings; BERT has \
                                no RoPE to extrapolate with, so a longer input must be truncated by \
                                the caller rather than silently wrapped", self.cfg.n_ctx));
        }
        // token + position + segment, summed on the host — three gathers over a 30k-row table are
        // cheaper to index here than to dispatch.
        let mut e = vec![0f32; t * d];
        for (p, &id) in ids.iter().enumerate() {
            let (tk, ps) = ((id as usize) * d, p * d);
            let pe = (p + self.cfg.pos_offset) * d;
            for j in 0..d {
                e[ps + j] = self.tok_embd[tk + j] + self.pos_embd[pe + j] + self.typ_embd[j];
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
            // ggml's `ggml_gelu` is the TANH approximation, not the exact erf form, and llama.cpp's
            // BERT graph uses LLM_FFN_GELU which maps to it. Selectable while this is being pinned
            // down; the default follows ggml.
            let up = h.matmul_bt(&b.up).add(&b.upb);
            let act = if std::env::var("FERRIC_BERT_GELU_ERF").ok().as_deref() == Some("1") {
                up.gelu()
            } else {
                up.gelu_tanh()
            };
            let ff = act.matmul_bt(&b.down).add(&b.downb);
            h = h.add(&ff).layernorm(&b.out_norm_w, &b.out_norm_b, self.cfg.eps);
        }
        Ok(h)
    }

    /// Whether this checkpoint carries a reranker head.
    pub fn is_reranker(&self) -> bool { self.cls.is_some() }

    /// **BERT's pooler**: `tanh(dense(CLS))`, the first half of the classification head.
    ///
    /// This is a distinct output from the raw CLS hidden state, and which one a tool means by "CLS
    /// pooling" is not obvious: a checkpoint with no `cls.*` tensors can only mean the raw state,
    /// while one that has them may mean either. Exposed separately so a reference diff can say which,
    /// instead of a mismatch being blamed on the encoder.
    pub async fn pooler(&self, h: &Tensor) -> Result<Vec<f32>, String> {
        let c = self.cls.as_ref().ok_or("no cls.* pooler on this checkpoint")?;
        let pooled = h.narrow(0, 0, 1).reshape(&[1, self.cfg.d]);
        Ok(pooled.matmul_bt(&c.w).add(&c.b).tanh().to_vec().await)
    }

    /// **Cross-encoder relevance score** for one (query, passage) pair, already joined into `ids`.
    ///
    /// This is what makes a reranker worth its cost and a bi-encoder cheap: the query and the passage
    /// go through the network TOGETHER, so every query token can attend to every passage token. A
    /// bi-encoder embeds them apart and compares two summaries. Scoring N passages therefore costs N
    /// forwards, which is why it runs over a retrieved shortlist rather than the corpus.
    ///
    /// Head: pooled CLS → dense → tanh → linear → one logit. Returned RAW, not squashed: llama.cpp
    /// reports the logit, ordering is invariant to any monotone squash, and a sigmoid here would make
    /// scores from two implementations incomparable for no gain.
    pub async fn score(&self, ids: &[u32]) -> Result<f32, String> {
        let c = self.cls.as_ref().ok_or(
            "this checkpoint has no cls.* head, so it embeds but cannot score a pair;              reranking needs a cross-encoder such as bge-reranker")?;
        let h = self.forward(ids)?;
        // CLS is position 0 for every cross-encoder in this family.
        let pooled = h.narrow(0, 0, 1).reshape(&[1, self.cfg.d]);
        let z = pooled.matmul_bt(&c.w).add(&c.b).tanh();
        let out = z.matmul_bt(&c.ow).add(&c.ob).to_vec().await;
        out.first().copied().ok_or_else(|| "empty score output".into())
    }
}
