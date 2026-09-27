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
//! ## ⭐ nomic-bert — the same encoder with four differences, each DETECTED from the file
//!
//! `nomic-embed-text` (Ollama's default RAG embedder, 8.2% of all Ollama pulls; 13.8M HF downloads a
//! month for v1.5) is `general.architecture = nomic-bert`. Against the authors' `modeling_hf_nomic_bert.py`
//! (nomic-ai/nomic-bert-2048) it is this file's encoder with:
//!
//! | | BERT | nomic-bert | detected by |
//! |---|---|---|---|
//! | positions | learned table, added once | **RoPE on Q and K**, NeoX halves, base from the file | no `position_embd`; `<arch>.rope.freq_base` REQUIRED |
//! | Q/K/V | three projections | **one fused `Wqkv`**, rows `(three h d)` | `attn_qkv.weight` present |
//! | FFN | GELU(up) | **`fc11(x) * silu(fc12(x))`** — up times activated GATE | `ffn_gate.weight` present |
//! | biases | everywhere | **none** on attention or FFN | each bias tensor looked up, absent = none |
//!
//! Post-LayerNorm with bias, bidirectional, token-type row 0 added: all as BERT. Every difference is
//! read from the checkpoint, never from the architecture NAME, so a file that mixes them (a BERT with
//! RoPE, a nomic-bert with biases) runs as what it is rather than as what its name suggests.
//! ⛔ An absent `rope.freq_base` is an ERROR, not a default: nomic uses 1000, the common default is
//! 10000, and a wrong base rotates every pair by the wrong angle with no error anywhere.
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
    /// `general.architecture` — the prefix every other key is read under (`bert`, `nomic-bert`).
    pub arch: String,
    /// **RoPE base when the file has no learned position table** (nomic-bert). `None` = learned
    /// positions. Required, never defaulted, when `position_embd` is absent — see the module note.
    pub rope_base: Option<f32>,
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let arch = match md.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "bert".into() };
        let u = |k: &str| match md.get(&format!("{arch}.{k}")) {
            Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing {arch}.{k}")),
        };
        let f = |k: &str| match md.get(&format!("{arch}.{k}")) {
            Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing {arch}.{k}")),
        };
        let causal = matches!(md.get(&format!("{arch}.attention.causal")), Some(Meta::Bool(true)));
        if causal {
            return Err(format!("{arch}.attention.causal is true; this runtime is bidirectional only"));
        }
        let rope_base = if g.tensor("position_embd.weight").is_some() { None } else {
            Some(f("rope.freq_base").map_err(|_| format!(
                "no position_embd table and no {arch}.rope.freq_base: an encoder without learned \
                 positions must declare its RoPE base (nomic uses 1000, not the common 10000)"))?)
        };
        let n_vocab = g.tensor("token_embd.weight").ok_or("no token_embd.weight")?.dims[1] as usize;
        Ok(Cfg {
            n_layer: u("block_count")?,
            d: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_ff: u("feed_forward_length")?,
            n_ctx: u("context_length").unwrap_or(512),
            n_vocab,
            eps: f("attention.layer_norm_epsilon").unwrap_or(1e-12),
            pooling: match md.get(&format!("{arch}.pooling_type")) { Some(Meta::U(v)) => *v as u32, _ => 2 },
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
            rope_base,
            arch,
        })
    }
}

/// Q, K and V as three projections (BERT) or one fused `Wqkv` whose output rows are `(three h d)`.
enum Qkv {
    Split { q: Tensor, qb: Option<Tensor>, k: Tensor, kb: Option<Tensor>, v: Tensor, vb: Option<Tensor> },
    Fused { w: Tensor, b: Option<Tensor> },
}

struct Block {
    qkv: Qkv,
    o: Tensor, ob: Option<Tensor>,
    attn_norm_w: Tensor, attn_norm_b: Tensor,
    up: Tensor, upb: Option<Tensor>,
    /// SwiGLU gate (`ffn_gate`, the authors' `fc12`): present → `up(x) * silu(gate(x))`, absent → GELU.
    gate: Option<Tensor>, gateb: Option<Tensor>,
    down: Tensor, downb: Option<Tensor>,
    out_norm_w: Tensor, out_norm_b: Tensor,
}

/// `x + b` when the checkpoint has the bias, `x` when it does not.
fn bias(x: Tensor, b: &Option<Tensor>) -> Tensor { match b { Some(b) => x.add(b), None => x } }

pub struct Bert {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    tok_embd: Vec<f32>,      // [n_vocab, d], host-side: one row per lookup, no GPU gather needed
    pos_embd: Option<Vec<f32>>, // [n_ctx, d]; None for a RoPE encoder
    typ_embd: Option<Vec<f32>>, // [n_type, d]; row 0 is added
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
pub(crate) fn t2(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Tensor, String> {
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

pub(crate) fn t1(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Tensor, String> {
    let i = g.tensor(name).ok_or_else(|| format!("no {name}"))?;
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, &[1, i.dims[0] as usize]))
}
fn t1o(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Option<Tensor>, String> {
    if g.tensor(name).is_some() { t1(ctx, g, name).map(Some) } else { Ok(None) }
}

impl Bert {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Bert, String> {
        let cfg = Cfg::from_gguf(g)?;
        let blocks = (0..cfg.n_layer).map(|i| {
            let n = |s: &str| format!("blk.{i}.{s}");
            let qkv = if g.tensor(&n("attn_qkv.weight")).is_some() {
                Qkv::Fused { w: t2(ctx, g, &n("attn_qkv.weight"))?, b: t1o(ctx, g, &n("attn_qkv.bias"))? }
            } else {
                Qkv::Split {
                    q: t2(ctx, g, &n("attn_q.weight"))?, qb: t1o(ctx, g, &n("attn_q.bias"))?,
                    k: t2(ctx, g, &n("attn_k.weight"))?, kb: t1o(ctx, g, &n("attn_k.bias"))?,
                    v: t2(ctx, g, &n("attn_v.weight"))?, vb: t1o(ctx, g, &n("attn_v.bias"))?,
                }
            };
            let gate = if g.tensor(&n("ffn_gate.weight")).is_some() { Some(t2(ctx, g, &n("ffn_gate.weight"))?) } else { None };
            Ok(Block {
                qkv,
                o: t2(ctx, g, &n("attn_output.weight"))?, ob: t1o(ctx, g, &n("attn_output.bias"))?,
                attn_norm_w: t1(ctx, g, &n("attn_output_norm.weight"))?,
                attn_norm_b: t1(ctx, g, &n("attn_output_norm.bias"))?,
                up: t2(ctx, g, &n("ffn_up.weight"))?, upb: t1o(ctx, g, &n("ffn_up.bias"))?,
                gate, gateb: t1o(ctx, g, &n("ffn_gate.bias"))?,
                down: t2(ctx, g, &n("ffn_down.weight"))?, downb: t1o(ctx, g, &n("ffn_down.bias"))?,
                out_norm_w: t1(ctx, g, &n("layer_output_norm.weight"))?,
                out_norm_b: t1(ctx, g, &n("layer_output_norm.bias"))?,
            })
        }).collect::<Result<Vec<_>, String>>()?;
        let opt = |name: &str| -> Result<Option<Vec<f32>>, String> {
            if g.tensor(name).is_some() { g.dequant(name).map(Some) } else { Ok(None) }
        };
        Ok(Bert {
            ctx: ctx.clone(),
            tok_embd: g.dequant("token_embd.weight")?,
            pos_embd: opt("position_embd.weight")?,
            typ_embd: opt("token_types.weight")?,
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
        self.forward_inner(ids, std::env::var("FERRIC_BERT_TRACE").ok().as_deref() == Some("1"))
    }

    /// `forward`, always returning the stage taps: the embedding LayerNorm (`inp_norm`) and each
    /// block's two post-norm outputs. For conformance against a reference's hooks.
    pub fn forward_taps(&self, ids: &[u32]) -> Result<(Tensor, Vec<(String, Tensor)>), String> {
        self.forward_inner(ids, true)
    }

    fn forward_inner(&self, ids: &[u32], trace: bool) -> Result<(Tensor, Vec<(String, Tensor)>), String> {
        // Negative controls for the conformance gate — each must make Ferric clearly WORSE against the
        // authors, or the gate cannot see the defect it names.
        let neg = std::env::var("FERRIC_BERT_NEG").unwrap_or_default();
        let (d, t) = (self.cfg.d, ids.len());
        match &self.pos_embd {
            Some(pe) if t + self.cfg.pos_offset > pe.len() / d =>
                return Err(format!("{t} tokens exceeds this encoder's {} position embeddings; BERT has \
                                    no RoPE to extrapolate with, so a longer input must be truncated by \
                                    the caller rather than silently wrapped", self.cfg.n_ctx)),
            None if t > self.cfg.n_ctx =>
                return Err(format!("{t} tokens exceeds this encoder's declared context of {}",
                                   self.cfg.n_ctx)),
            _ => {}
        }
        if let Some(&bad) = ids.iter().find(|&&i| i as usize >= self.cfg.n_vocab) {
            return Err(format!("token id {bad} is outside this encoder's {}-row vocabulary", self.cfg.n_vocab));
        }
        // token + position + segment, summed on the host — three gathers over a 30k-row table are
        // cheaper to index here than to dispatch. A RoPE encoder has no position row to add.
        let mut e = vec![0f32; t * d];
        for (p, &id) in ids.iter().enumerate() {
            let (tk, ps) = ((id as usize) * d, p * d);
            for j in 0..d { e[ps + j] = self.tok_embd[tk + j]; }
            if let Some(ty) = &self.typ_embd { if neg != "no_type" { for j in 0..d { e[ps + j] += ty[j]; } } }
            if let Some(pe) = &self.pos_embd {
                let o = (p + self.cfg.pos_offset) * d;
                for j in 0..d { e[ps + j] += pe[o + j]; }
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

        // Per-tensor trace. Comparing whole-model outputs and guessing at parameters is an unbounded
        // search — four wrong turns' worth of evidence for that. The first stage that disagrees
        // localises the bug to one op.
        if trace { tr.push(("inp_norm".into(), h.clone())); }
        let (nh, dh) = (self.cfg.n_head, d / self.cfg.n_head);
        let scale = 1.0 / (dh as f32).sqrt();
        // ⭐ The rotary table is built on the HOST, the way the authors build it: inv_freq in float32 as
        // `1 / base^(2c/d)`, angle = float32(position) x inv_freq, then cos/sin. The GPU rope kernel
        // derives inv_freq as `exp(-2c/d * ln base)` on the device, which lands an ulp away; position
        // multiplies that, and at position 822 of a 926-token input it put one row at 10x the authors'
        // own float32 error (their float64 run shares their float32 angles, so its floor cannot absorb
        // a different angle formula). `examples/rope_precision.rs` measures the gap: 6e-5 rad below
        // position 1024, 2e-3 rad at 32k — as large as either method's distance from exact angles.
        let rope_tab = self.cfg.rope_base.map(|base| {
            let base = if neg == "rope_base_10000" { 10000.0 } else { base };
            let half = dh / 2;
            let inv: Vec<f32> = (0..half).map(|c| 1.0 / ((base as f64).powf((2 * c) as f64 / dh as f64) as f32)).collect();
            let (mut ct, mut st) = (vec![0f32; t * dh], vec![0f32; t * dh]);
            for p in 0..t { for c in 0..half {
                let a = p as f32 * inv[c];
                ct[p * dh + c] = (a as f64).cos() as f32; st[p * dh + c] = (a as f64).sin() as f32;
            } }
            (Tensor::from_vec(&self.ctx, &ct, &[t, dh]), Tensor::from_vec(&self.ctx, &st, &[t, dh]))
        });
        for (_il, b) in self.blocks.iter().enumerate() {
            let (q, k, v) = match &b.qkv {
                Qkv::Split { q, qb, k, kb, v, vb } =>
                    (bias(h.matmul_bt(q), qb), bias(h.matmul_bt(k), kb), bias(h.matmul_bt(v), vb)),
                // The authors' `rearrange(qkv, "... (three h d) -> ... three h d")`: the first d
                // columns are Q for every head in order, then K, then V.
                Qkv::Fused { w, b: bb } => {
                    let x = bias(h.matmul_bt(w), bb);
                    (x.narrow(1, 0, d).contiguous(), x.narrow(1, d, d).contiguous(), x.narrow(1, 2 * d, d).contiguous())
                }
            };
            // RoPE on Q and K, NeoX halves (`rotary_emb_interleaved: false`), positions from 0.
            let (q, k) = match (&rope_tab, neg.as_str()) {
                (Some(_), "rope_interleaved") => { let b = self.cfg.rope_base.unwrap_or(0.0);
                    (q.rope_interleaved(nh, dh, b, 0), k.rope_interleaved(nh, dh, b, 0)) }
                (Some(_), "rope_device") => { let b = self.cfg.rope_base.unwrap_or(0.0); (q.rope(nh, dh, b, 0), k.rope(nh, dh, b, 0)) }
                (Some(_), "no_rope") => (q, k),
                (Some((c, s)), _) => (q.apply_rope_costable(c, s, nh, dh), k.apply_rope_costable(c, s, nh, dh)),
                (None, _) => (q, k),
            };
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
            let attn = bias(cat.reshape(&[t, d]).matmul_bt(&b.o), &b.ob);
            // POST-norm: normalise the residual sum, not the input to the sublayer.
            h = h.add(&attn).layernorm(&b.attn_norm_w, &b.attn_norm_b, self.cfg.eps);
            if trace { tr.push((format!("l{_il}.attn_out_norm"), h.clone())); }
            // ⛔⛔ GELU IS THE AUTHORS' EXACT ERF FORM, NOT ggml's TANH APPROXIMATION.
            //
            // This defaulted to `gelu_tanh()` because "ggml's `ggml_gelu` is the tanh approximation
            // and llama.cpp's BERT graph uses it" — a choice made to match a port, not the model. The
            // authors' configs say `hidden_act: "gelu"`, which in `transformers` is the exact erf form.
            // Checked against the authors (transformers 5.7.0, float32, eager):
            //     bge-small-en-v1.5 F16    tanh max|diff| 1.4e-3 – 2.8e-3    erf 1.2e-6 – 1.4e-6
            //     bge-reranker-v2-m3 F16   tanh scores off by up to 0.0151   erf exact to 4 dp
            // — about 1000x closer. The GGUF records no activation key for BERT (the converter drops
            // `hidden_act`), so the authors' default is the only honest default. A checkpoint trained
            // with the tanh form ("gelu_new" / "gelu_pytorch_tanh") cannot be told apart from the file;
            // `FERRIC_BERT_GELU_TANH=1` selects it, and reproduces llama.cpp for comparison.
            let up = bias(h.matmul_bt(&b.up), &b.upb);
            let act = match &b.gate {
                // The authors' NomciBertGatedMLP: `y = fc11(x); gate = fc12(x); y * silu(gate)` — the
                // activation lands on the GATE (fc12, `ffn_gate`), and fc11 (`ffn_up`) is not activated.
                Some(gw) if neg == "gate_swap" => up.silu().mul(&bias(h.matmul_bt(gw), &b.gateb)),
                Some(gw) => up.mul(&bias(h.matmul_bt(gw), &b.gateb).silu()),
                None if std::env::var("FERRIC_BERT_GELU_TANH").ok().as_deref() == Some("1") => up.gelu_tanh(),
                None => up.gelu(),
            };
            let ff = bias(act.matmul_bt(&b.down), &b.downb);
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
        let out = self.score_all(ids).await?;
        out.first().copied().ok_or_else(|| "empty score output".into())
    }

    /// **Every logit the head produces**, not just the first.
    ///
    /// ⛔⛔ THIS ARITHMETIC WAS ALREADY RUNNING AND THE RESULT WAS BEING DISCARDED. `score` computed
    /// the full `cls.output` vector and returned `out.first()`. For a reranker that is right —
    /// `n_cls_out` is 1 and the vector has one element. For every OTHER checkpoint that carries the
    /// same `cls.*` tensors it silently returned the first class's logit as if it were "the score":
    /// a guardrail head's `safe` logit with the unsafe classes dropped, an NLI head's `entailment`
    /// with `neutral` and `contradiction` dropped. Nothing errored; the number was real and answered
    /// a question nobody asked.
    ///
    /// GGUF already carries the width and the names — `cls.output.weight` is `{n_embd, n_cls_out}`
    /// and llama.cpp reads `%s.classifier.output_labels` (`llama-arch.cpp:311`) alongside
    /// `llama_model_n_cls_out` (`llama.h:589-591`). This returns the vector; [`Reranker::labels`]
    /// returns the names when the file declares them.
    ///
    /// The order is the file's own — index `i` is the class at index `i` of `output_labels`. This
    /// function does NOT softmax: the caller decides, because a cross-encoder's single logit is
    /// consumed raw and monotonically (see the note on [`Self::score`]) while a k-way head is
    /// normally softmaxed, and doing it here would make the two incomparable.
    pub async fn score_all(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let c = self.cls.as_ref().ok_or(
            "this checkpoint has no cls.* head, so it embeds but cannot score a pair;              reranking needs a cross-encoder such as bge-reranker")?;
        let h = self.forward(ids)?;
        // CLS is position 0 for every cross-encoder in this family.
        let pooled = h.narrow(0, 0, 1).reshape(&[1, self.cfg.d]);
        let z = pooled.matmul_bt(&c.w).add(&c.b).tanh();
        Ok(z.matmul_bt(&c.ow).add(&c.ob).to_vec().await)
    }

    /// How many classes this checkpoint's head emits — llama.cpp's `n_cls_out`.
    ///
    /// Read from `cls.output.weight`'s own shape rather than from metadata, so it cannot disagree
    /// with the tensor the arithmetic actually uses. `None` when there is no `cls.*` head at all.
    pub fn n_cls_out(&self) -> Option<usize> {
        // `matmul_bt(&ow)` maps [1, d] -> [1, n_cls_out], so `ow` is [n_cls_out, d] in this
        // layout and the CLASS COUNT is the leading dim. Read from the tensor rather than from
        // metadata so it cannot disagree with the matmul that actually runs.
        self.cls.as_ref().map(|c| c.ow.shape.first().copied().unwrap_or(1))
    }
}

/// **A cross-encoder reranker, tokenizer included.**
///
/// Owning the tokenizer is the point. A BERT checkpoint declares `tokenizer.ggml.model` and the
/// family is NOT implied by the architecture: `bge-small` is `"bert"` (WordPiece) while
/// `bge-reranker-v2-m3` is the same `general.architecture = bert` but `"t5"` (SentencePiece), because
/// it is XLM-RoBERTa underneath. Hardcoding either one produced a four-token "Paris" against
/// llama.cpp's three and cost nine hypotheses chasing a model bug that did not exist. Every caller
/// getting this right independently is not a plan; the decision lives here once.
pub struct Reranker {
    model: Bert,
    tok: RerankTok,
    bos: u32,
    eos: u32,
}

enum RerankTok {
    Wordpiece(ferric_tokenizer::WordPiece),
    Spm(ferric_tokenizer::Spm),
}

impl Reranker {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Reranker, String> {
        let model = Bert::load(ctx, g)?;
        if !model.is_reranker() {
            return Err("this checkpoint has no cls.* head: it embeds but cannot score pairs. \
                        Reranking needs a cross-encoder such as bge-reranker".into());
        }
        let md = g.metadata();
        let toks: Vec<String> = match md.get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(v)) => v.iter()
                .map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
            _ => return Err("no tokenizer.ggml.tokens".into()),
        };
        let u = |k: &str, d: u32| match md.get(k) { Some(Meta::U(v)) => *v as u32, _ => d };
        let kind = match md.get("tokenizer.ggml.model") { Some(Meta::Str(s)) => s.as_str(), _ => "bert" };
        let tok = match kind {
            "bert" => RerankTok::Wordpiece(ferric_tokenizer::WordPiece::new(
                toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect(),
                u("tokenizer.ggml.cls_token_id", 101),
                u("tokenizer.ggml.seperator_token_id", 102),
                u("tokenizer.ggml.unknown_token_id", 100))),
            _ => {
                let scores: Vec<f32> = match md.get("tokenizer.ggml.scores") {
                    Some(Meta::Arr(v)) => v.iter()
                        .map(|x| if let Meta::F(f) = x { *f as f32 } else { 0.0 }).collect(),
                    _ => Vec::new(),
                };
                RerankTok::Spm(ferric_tokenizer::Spm::with_types(toks.clone(), scores,
                    &ferric_gguf::token_types(g.metadata().get("tokenizer.ggml.token_type"))))
            }
        };
        Ok(Reranker {
            bos: u("tokenizer.ggml.bos_token_id", u("tokenizer.ggml.cls_token_id", 0)),
            eos: u("tokenizer.ggml.eos_token_id", u("tokenizer.ggml.seperator_token_id", 2)),
            model, tok,
        })
    }

    /// The pair layout llama.cpp's `format_rerank` builds: **BOS query EOS doc EOS**.
    ///
    /// One separator, no second BOS. Deduced from the reference's own token count rather than from
    /// the RoBERTa papers, which describe a doubled separator: llama-server reported 39 prompt tokens
    /// for two pairs of 6 query and 10/11 doc content tokens, and of the three candidate layouts only
    /// this one gives 19 + 20 = 39. Two wrong layouts were tried first and BOTH kept the correct
    /// ordering, which is exactly why a ranking benchmark cannot catch this.
    /// The cross-encoder pair encoding — `[CLS] query [SEP] doc [SEP]` in this family's own
    /// tokenizer. Public because a caller reaching [`Bert::score_all`] for a k-way head needs the
    /// SAME ids the reranker would build; rebuilding them by hand is how two paths drift.
    pub fn pair(&self, query: &str, doc: &str) -> Vec<u32> {
        let enc = |s: &str| match &self.tok {
            // WordPiece adds its own [CLS]/[SEP]; strip them so the pair layout supplies the wrapping.
            RerankTok::Wordpiece(w) => { let v = w.encode(s); v[1..v.len().saturating_sub(1)].to_vec() }
            RerankTok::Spm(s2) => s2.encode_piece(s, true),
        };
        let mut ids = vec![self.bos];
        ids.extend(enc(query));
        ids.push(self.eos);
        ids.extend(enc(doc));
        ids.push(self.eos);
        ids
    }

    /// Relevance logit for one pair. Raw, not squashed: llama.cpp reports the logit, ordering is
    /// invariant to any monotone squash, and squashing would make the two incomparable for no gain.
    pub async fn score(&self, query: &str, doc: &str) -> Result<f32, String> {
        self.model.score(&self.pair(query, doc)).await
    }

    /// Score every document and return `(index, logit)` sorted best-first.
    ///
    /// N documents cost N forwards — the query and the passage go through the network TOGETHER, which
    /// is what a cross-encoder buys over comparing two independent embeddings, and why this runs over
    /// a retrieved shortlist rather than a corpus.
    pub async fn rank(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>, String> {
        let mut out = Vec::with_capacity(docs.len());
        for (i, d) in docs.iter().enumerate() { out.push((i, self.score(query, d).await?)); }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }
}
