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
use ferric_gguf::{GgufFile, Meta};
use ferric_tensor::dtype::Q2_0Weights;
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
}

impl Cfg {
    pub fn from_gguf(g: &GgufFile) -> Result<Cfg, String> {
        let u = |k: &str| -> Result<usize, String> {
            match g.metadata.get(k) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing metadata {k}")) }
        };
        let f = |k: &str| -> Result<f32, String> {
            match g.metadata.get(k) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing metadata {k}")) }
        };
        let n_vocab = match g.metadata.get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.len(), _ => return Err("missing tokenizer.ggml.tokens".into()) };
        Ok(Cfg {
            n_embd: u("qwen35.embedding_length")?,
            n_layer: u("qwen35.block_count")?,
            n_head: u("qwen35.attention.head_count")?,
            n_head_kv: u("qwen35.attention.head_count_kv")?,
            head_dim: u("qwen35.attention.key_length")?,
            n_ff: u("qwen35.feed_forward_length")?,
            n_vocab,
            eps: f("qwen35.attention.layer_norm_rms_epsilon")?,
            rope_base: f("qwen35.rope.freq_base")?,
            n_rot: u("qwen35.rope.dimension_count")?,
            full_attention_interval: u("qwen35.full_attention_interval").unwrap_or(4),
            conv_kernel: u("qwen35.ssm.conv_kernel")?,
            n_k_heads: u("qwen35.ssm.group_count")?,
            n_v_heads: u("qwen35.ssm.time_step_rank")?,
            head_k_dim: u("qwen35.ssm.state_size")?,
            d_inner: u("qwen35.ssm.inner_size")?,
        })
    }
    pub fn head_v_dim(&self) -> usize { self.d_inner / self.n_v_heads }
    pub fn key_dim(&self) -> usize { self.head_k_dim * self.n_k_heads }
    /// llama.cpp: layer is recurrent (linear attention) unless it's every `interval`-th one.
    pub fn is_recurrent(&self, il: usize) -> bool { (il + 1) % self.full_attention_interval != 0 }
}

pub struct AttnW {
    pub wq: Q2_0Weights, // [n_head·head_dim·2, n_embd] — query and gate, interleaved per head
    pub wk: Q2_0Weights,
    pub wv: Q2_0Weights,
    pub wo: Q2_0Weights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct GdnW {
    pub wqkv: Q2_0Weights, // [2·key_dim + d_inner, n_embd]
    pub wz: Q2_0Weights,   // [d_inner, n_embd]
    pub beta: Q2_0Weights, // [n_v_heads, n_embd]
    pub alpha: Q2_0Weights,
    pub conv1d: Tensor,  // [conv_dim, L]
    pub dt_bias: Tensor, // [n_v_heads]
    pub a: Tensor,       // [n_v_heads] — already -exp(A_log)
    pub norm: Tensor,    // [head_v_dim]
    pub out: Q2_0Weights,
}

pub enum Mixer { Attn(AttnW), Gdn(GdnW) }

/// Per-layer carried state. Attention layers keep the usual K/V history; gated-delta-net layers keep
/// the recurrent state plus the last `conv_kernel-1` conv inputs (the short conv's receptive field).
/// With both, generating token N costs one step instead of re-running the whole prefix.
enum LayerCache {
    Attn { k: Tensor, v: Tensor },   // [S, n_kv·head_dim]
    Gdn { state: Tensor, conv: Tensor }, // state [n_v_heads, dv, dk]; conv [conv_kernel-1, conv_dim]
}

#[derive(Default)]
pub struct Cache {
    pub pos: usize,
    layers: Vec<Option<LayerCache>>,
}

impl Cache {
    pub fn new(cfg: &Cfg) -> Cache { Cache { pos: 0, layers: (0..cfg.n_layer).map(|_| None).collect() } }
}

pub struct Layer {
    pub attn_norm: Tensor,
    pub post_norm: Tensor,
    pub ffn_gate: Q2_0Weights,
    pub ffn_up: Q2_0Weights,
    pub ffn_down: Q2_0Weights,
    pub mixer: Mixer,
}

pub struct Qwen35 {
    pub cfg: Cfg,
    ctx: Arc<Context>,
    /// Token embeddings stay packed in *host* RAM: only the prompt's rows are ever needed, so
    /// dequantizing those few rows on the CPU beats parking 338 MB on the GPU for a gather.
    tok_embd: Vec<u8>,
    pub layers: Vec<Layer>,
    pub out_norm: Tensor,
    pub lm_head: Q2_0Weights,
}

/// GGUF stores dims fastest-varying first, so a listed `[in, out]` is a row-major `[out, in]`
/// weight — the HF linear convention, which is what `matmul_q2_0` wants.
fn q2(ctx: &Arc<Context>, g: &GgufFile, name: &str) -> Result<Q2_0Weights, String> {
    let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
    if t.ggml_type != 42 { return Err(format!("{name}: expected Q2_0 (42), got {}", t.ggml_type)); }
    let (inn, out) = (t.dims[0] as usize, t.dims[1] as usize);
    Ok(Q2_0Weights::from_bytes(ctx, &g.raw(name)?, out, inn))
}

fn f32t(ctx: &Arc<Context>, g: &GgufFile, name: &str, shape: &[usize]) -> Result<Tensor, String> {
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, shape))
}

impl Qwen35 {
    pub fn load(ctx: &Arc<Context>, g: &GgufFile) -> Result<Qwen35, String> {
        let cfg = Cfg::from_gguf(g)?;
        let conv_dim = cfg.key_dim() * 2 + cfg.d_inner;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let mixer = if cfg.is_recurrent(il) {
                Mixer::Gdn(GdnW {
                    wqkv: q2(ctx, g, &b("attn_qkv.weight"))?,
                    wz: q2(ctx, g, &b("attn_gate.weight"))?,
                    beta: q2(ctx, g, &b("ssm_beta.weight"))?,
                    alpha: q2(ctx, g, &b("ssm_alpha.weight"))?,
                    conv1d: f32t(ctx, g, &b("ssm_conv1d.weight"), &[conv_dim, cfg.conv_kernel])?,
                    dt_bias: f32t(ctx, g, &b("ssm_dt.bias"), &[cfg.n_v_heads])?,
                    a: f32t(ctx, g, &b("ssm_a"), &[cfg.n_v_heads])?,
                    norm: f32t(ctx, g, &b("ssm_norm.weight"), &[cfg.head_v_dim()])?,
                    out: q2(ctx, g, &b("ssm_out.weight"))?,
                })
            } else {
                Mixer::Attn(AttnW {
                    wq: q2(ctx, g, &b("attn_q.weight"))?,
                    wk: q2(ctx, g, &b("attn_k.weight"))?,
                    wv: q2(ctx, g, &b("attn_v.weight"))?,
                    wo: q2(ctx, g, &b("attn_output.weight"))?,
                    q_norm: f32t(ctx, g, &b("attn_q_norm.weight"), &[cfg.head_dim])?,
                    k_norm: f32t(ctx, g, &b("attn_k_norm.weight"), &[cfg.head_dim])?,
                })
            };
            layers.push(Layer {
                attn_norm: f32t(ctx, g, &b("attn_norm.weight"), &[cfg.n_embd])?,
                post_norm: f32t(ctx, g, &b("post_attention_norm.weight"), &[cfg.n_embd])?,
                ffn_gate: q2(ctx, g, &b("ffn_gate.weight"))?,
                ffn_up: q2(ctx, g, &b("ffn_up.weight"))?,
                ffn_down: q2(ctx, g, &b("ffn_down.weight"))?,
                mixer,
            });
        }

        // Tied head: if `output.weight` is absent the embeddings double as the LM head.
        let head = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        Ok(Qwen35 {
            tok_embd: g.raw("token_embd.weight")?,
            out_norm: f32t(ctx, g, "output_norm.weight", &[cfg.n_embd])?,
            lm_head: q2(ctx, g, head)?,
            cfg, ctx: ctx.clone(), layers,
        })
    }

    /// Dequantize just the rows the prompt touches, straight out of the packed Q2_0 blocks.
    pub fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.n_embd;
        let row_bytes = d / 128 * 34;
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let off = t as usize * row_bytes;
            v.extend(ferric_gguf::deq_raw(&self.tok_embd[off..off + row_bytes], d, 42).expect("embed row"));
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
        // One projection emits query and gate interleaved per head: [t, nh, 2·hd].
        let qf = h.matmul_q2_0(&w.wq).reshape(&[t, nh, hd * 2]);
        let q = qf.narrow(2, 0, hd).rmsnorm(&w.q_norm, self.cfg.eps).reshape(&[t, nh * hd]);
        let gate = qf.narrow(2, hd, hd).contiguous().reshape(&[t, nh * hd]);

        let nkv = self.cfg.n_head_kv;
        let k = h.matmul_q2_0(&w.wk).reshape(&[t, nkv, hd]).rmsnorm(&w.k_norm, self.cfg.eps).reshape(&[t, nkv * hd]);
        let v = h.matmul_q2_0(&w.wv);

        let q = self.rope_partial(&q, nh, offset);
        let k = self.rope_partial(&k, nkv, offset);

        // Append this step's K/V to the history, then attend over all of it.
        let (kc, vc) = match cache.take() {
            Some(LayerCache::Attn { k: pk, v: pv }) => (pk.cat(&k, 0), pv.cat(&v, 0)),
            _ => (k, v),
        };
        let o = if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv)
        } else {
            // Prefill: q and the history are the same span, so the causal mask applies as usual.
            nn::causal_attention(&q, &kc, &vc, nh, nkv)
        };
        *cache = Some(LayerCache::Attn { k: kc, v: vc });
        o.mul(&gate.sigmoid()).matmul_q2_0(&w.wo)
    }

    fn gdn(&self, h: &Tensor, w: &GdnW, cache: &mut Option<LayerCache>) -> Tensor {
        let c = &self.cfg;
        let (t, nk, nv) = (h.shape[0], c.n_k_heads, c.n_v_heads);
        let (dk, dv, kd) = (c.head_k_dim, c.head_v_dim(), c.key_dim());

        // conv over the whole fused q|k|v, then split — the conv is causal and depthwise, so
        // splitting after it is identical to convolving each part separately.
        let qkv = h.matmul_q2_0(&w.wqkv);
        // The short conv looks back conv_kernel-1 steps, so a single new token can't be convolved
        // alone: prepend the carried tail (zeros at the start of a sequence, which is exactly the
        // causal zero-padding the standalone conv applies) and keep only the new rows.
        let pad = c.conv_kernel - 1;
        let (prev_conv, prev_state) = match cache.take() {
            Some(LayerCache::Gdn { state, conv }) => (conv, Some(state)),
            _ => (Tensor::zeros(&self.ctx, &[pad, qkv.shape[1]]), None),
        };
        let cin = prev_conv.cat(&qkv, 0);
        let conv = cin.depthwise_conv1d_causal(&w.conv1d, c.conv_kernel).narrow(0, pad, t).contiguous().silu();
        let conv_tail = cin.narrow(0, cin.shape[0] - pad, pad).contiguous();

        // l2norm (not RMSNorm) over head_k_dim, then fold the recurrence's 1/√dv into q.
        let q = conv.narrow(1, 0, kd).reshape(&[t, nk, dk]).l2norm(c.eps);
        let k = conv.narrow(1, kd, kd).reshape(&[t, nk, dk]).l2norm(c.eps);
        let v = conv.narrow(1, 2 * kd, c.d_inner).reshape(&[t, nv, dv]);
        let q = q.mul(&Tensor::from_vec(&self.ctx, &[1.0 / (dv as f32).sqrt()], &[1]));

        // Tiled broadcast of q/k across v heads (head % nk) — matches the converter's permutation.
        let rep = nv / nk;
        let tile = |x: &Tensor| x.reshape(&[t, 1, nk, dk]).broadcast_to(&[t, rep, nk, dk]).reshape(&[t, nv, dk]);
        let (q, k) = (tile(&q), tile(&k));

        // g = ssm_a · softplus(alpha + dt_bias) ; β = sigmoid(beta). Packed as [T, nv, 2] for the kernel.
        let alpha = h.matmul_q2_0(&w.alpha).add(&w.dt_bias.reshape(&[1, nv])).softplus().mul(&w.a.reshape(&[1, nv]));
        let beta = h.matmul_q2_0(&w.beta).sigmoid();
        let gb = alpha.reshape(&[t, nv, 1]).cat(&beta.reshape(&[t, nv, 1]), 2);

        let (o, state) = q.gated_delta_rule_stateful(&k, &v, &gb, nv, dk, dv, prev_state.as_ref());
        *cache = Some(LayerCache::Gdn { state, conv: conv_tail });

        // gated RMSNorm over head_v_dim, gated by silu(z)
        let z = h.matmul_q2_0(&w.wz).reshape(&[t, nv, dv]);
        o.rmsnorm(&w.norm, c.eps).mul(&z.silu()).reshape(&[t, c.d_inner]).matmul_q2_0(&w.out)
    }

    fn ffn(&self, h: &Tensor, l: &Layer) -> Tensor {
        h.matmul_q2_0(&l.ffn_gate).silu().mul(&h.matmul_q2_0(&l.ffn_up)).matmul_q2_0(&l.ffn_down)
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
        let mut x = self.embed(tokens);
        let pos = cache.pos;
        for (il, l) in self.layers.iter().enumerate().take(n) {
            let h = x.rmsnorm(&l.attn_norm, self.cfg.eps);
            let lc = &mut cache.layers[il];
            let y = match &l.mixer {
                Mixer::Attn(w) => self.attn(&h, w, lc, pos),
                Mixer::Gdn(w) => self.gdn(&h, w, lc),
            };
            x = x.add(&y);
            let r = x.clone();
            x = self.ffn(&x.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&r);
        }
        cache.pos += tokens.len();
        x.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q2_0(&self.lm_head)
    }
}
