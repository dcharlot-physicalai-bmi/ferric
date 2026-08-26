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
//! ## Verified against llama.cpp
//!
//! | checkpoint | arch | quant | cosine |
//! |---|---|---|---|
//! | bge-small-en-v1.5 | BERT, 12L d=384 | F16 | **0.999999–1.000000** |
//! | bge-small-en-v1.5 | BERT, 12L d=384 | Q4_K_M | **0.999996** |
//! | bge-reranker-v2-m3 | XLM-R, 24L d=1024 | Q4_K_M | **0.999995–1.000000** |
//!
//! Cross-encoder scoring matches too: 6.585 against the reference's 6.570 on a relevant pair, −8.366
//! against −8.361 on an irrelevant one.
//!
//! ## ⛔ The "XLM-R divergence" was a bug in the TEST, and it cost nine hypotheses
//!
//! For several hours this file carried a note describing a 0.9615 encoder divergence on XLM-R as an
//! open bug, with quantisation, the pooler, GELU, token types, the LayerNorm epsilon and the position
//! offset each eliminated by measurement. Every one of those eliminations was correct and none of
//! them was the cause, because the cause was not in the model at all: `bert_reference` built a
//! **WordPiece** tokenizer unconditionally, which is right for `tokenizer.ggml.model == "bert"` and
//! wrong for XLM-R's `"t5"`. It fed the encoder four tokens for "Paris" where llama.cpp's graph uses
//! three. A harness that tokenises differently from the reference is not comparing the model.
//!
//! Two things follow, and they are worth more than the fix:
//!
//! 1. **Verification does not transfer between files.** `rerank_reference` used SPM and its ids were
//!    confirmed identical to `llama-tokenize`; that confirmation was silently carried over to a
//!    sibling example written an hour earlier with a different tokenizer hardcoded.
//! 2. **Per-op tracing found it on the first run, and end-to-end comparison never could.** The final
//!    cosine is one scalar every stage feeds, so it can be nudged by tuning any of them — an offset
//!    of 2 scored a BETTER cosine (0.9726) than the correct 0 (0.9615) while being wrong. The trace
//!    prints token COUNT and per-tensor sums, and "4 tokens vs 3" is unambiguous.
//!
//! `FERRIC_BERT_TRACE=1` emits a sum per checkpoint tensor for diffing against `llama-eval-callback`,
//! whose own dump gives 512 reference sums for a three-token input. Start there for the next port.
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
            // ZERO. llama.cpp indexes `position_embd` with raw positions for every BERT-family
            // checkpoint, RoBERTa included — settled against the reference rather than reasoned from
            // the RoBERTa papers: `llama-eval-callback` reports the position-embedding sum for three
            // tokens as -37.445366, and rows 0..2 of this file's table sum to -37.445358, while rows
            // 2..4 give -20.41. The `padding_idx + 1` convention lives in the HF modelling code, not
            // in the converted table.
            //
            // ⚠ An offset of 2 scored a BETTER cosine (0.9726 vs 0.9615) while being WRONG. Tuning a
            // parameter toward a better whole-model number, with another defect still present, moves
            // it away from the reference. Only a per-op comparison can say which value is correct.
            pos_offset: std::env::var("FERRIC_BERT_POS_OFFSET").ok().and_then(|v| v.parse().ok())
                .unwrap_or(0),
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
/// Summary statistics for a trace line. Mean and max|v| localise a divergence without dumping
/// megabytes: a wrong op moves them immediately, a correct one keeps them equal to several digits.
fn stats(v: &[f32]) -> String {
    let n = v.len().max(1) as f32;
    let sum: f32 = v.iter().sum();
    let mean = sum / n;
    let absmean = v.iter().map(|x| x.abs()).sum::<f32>() / n;
    let mx = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    format!("sum {sum:+.6}  mean {mean:+.6}  mean|v| {absmean:.6}  max|v| {mx:.6}")
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
        self.forward_traced(ids).map(|(h, _)| h)
    }

    /// `forward`, plus a checkpoint tensor after each named op, for diffing against
    /// `llama-eval-callback`. Tensors are Arc-backed so collecting them is a handle copy, and the
    /// caller reads them asynchronously — which is why this returns them instead of printing.
    pub fn forward_traced(&self, ids: &[u32]) -> Result<(Tensor, Vec<(String, Tensor)>), String> {
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
        let mut tr: Vec<(String, Tensor)> = Vec::new();
        if std::env::var("FERRIC_BERT_TRACE").ok().as_deref() == Some("1") {
            // Host-side already, so no GPU read is needed and it prints directly. This is the tensor
            // llama.cpp calls `inp_embd`: token + type + position, before the embedding LayerNorm.
            eprintln!("TRACE inp_embd          {}", stats(&e));
        }
        let mut h = Tensor::from_vec(&self.ctx, &e, &[t, d])
            .layernorm(&self.embd_norm_w, &self.embd_norm_b, self.cfg.eps);

        // Per-tensor trace, to be diffed against `llama-eval-callback`. Comparing whole-model
        // outputs and guessing at parameters is an unbounded search — four wrong turns' worth of
        // evidence for that. The first layer whose stats disagree localises the bug to one op.
        let trace = std::env::var("FERRIC_BERT_TRACE").ok().as_deref() == Some("1");
        if trace { tr.push(("inp_norm".into(), h.clone())); }
        let (nh, dh) = (self.cfg.n_head, d / self.cfg.n_head);
        let scale = 1.0 / (dh as f32).sqrt();
        for (_il, b) in self.blocks.iter().enumerate() {
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
            if trace { tr.push((format!("l{_il}.attn_out_norm"), h.clone())); }
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
            if trace { tr.push((format!("l{_il}.layer_out_norm"), h.clone())); }
        }
        Ok((h, tr))
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
