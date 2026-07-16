//! **Qwen3** dense transformer — the architecture PrismML's smaller Ternary Bonsai models use
//! (1.7B / 4B), as opposed to the 27B's qwen3.5 hybrid. Full GQA every layer, QK-norm, SwiGLU,
//! RoPE; every projection ternary in PrismML's `Q2_0` (2.125 bpw). Because it's ~450 MB packed it
//! fits WebGPU's memory limits, so this is the model the browser path runs — the same code that
//! runs it here compiles to wasm32 and drives WebGPU in a tab.
//!
//! Reuses the crate's Q2_0 loaders and the same projection-fusion + KV-cache tricks proven on the
//! 27B: q/k/v fuse into one matmul, gate/up into another, and attention resumes from cached K/V so
//! decode is one step per token.
use crate::qwen35::{f32t, q2, q2_cat};
use ferric_core::Context;
use ferric_gguf::{deq_raw, GgufSource, Meta};
use ferric_tensor::{nn, Q2_0Weights, Tensor};
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
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let u = |k: &str| match g.metadata().get(k) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing {k}")) };
        let f = |k: &str| match g.metadata().get(k) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing {k}")) };
        let n_vocab = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.len(), _ => return Err("no tokens".into()) };
        Ok(Cfg {
            n_embd: u("qwen3.embedding_length")?,
            n_layer: u("qwen3.block_count")?,
            n_head: u("qwen3.attention.head_count")?,
            n_head_kv: u("qwen3.attention.head_count_kv")?,
            head_dim: u("qwen3.attention.key_length")?,
            n_ff: u("qwen3.feed_forward_length")?,
            n_vocab,
            eps: f("qwen3.attention.layer_norm_rms_epsilon")?,
            rope_base: f("qwen3.rope.freq_base")?,
        })
    }
}

pub struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    wqkv: Q2_0Weights, // q | k | v stacked
    q_out: usize,
    kv_out: usize,
    wo: Q2_0Weights,
    ffn_gate_up: Q2_0Weights,
    ffn_gate_out: usize,
    ffn_down: Q2_0Weights,
}

/// Per-layer attention K/V history. One step per token: append, then attend over all of it.
#[derive(Default)]
pub struct Cache {
    pub pos: usize,
    kv: Vec<Option<(Tensor, Tensor)>>,
}
impl Cache {
    pub fn new(cfg: &Cfg) -> Cache { Cache { pos: 0, kv: (0..cfg.n_layer).map(|_| None).collect() } }
}

pub struct Qwen3 {
    pub cfg: Cfg,
    ctx: Arc<Context>,
    tok_embd: Vec<u8>, // Q2_0 rows, gathered + dequantized on the CPU (avoids parking the table on GPU)
    layers: Vec<Layer>,
    out_norm: Tensor,
    lm_head: Q2_0Weights,
}

impl Qwen3 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Qwen3, String> {
        let cfg = Cfg::from_gguf(g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            layers.push(Layer {
                attn_norm: f32t(ctx, g, &b("attn_norm.weight"), &[cfg.n_embd])?,
                ffn_norm: f32t(ctx, g, &b("ffn_norm.weight"), &[cfg.n_embd])?,
                q_norm: f32t(ctx, g, &b("attn_q_norm.weight"), &[cfg.head_dim])?,
                k_norm: f32t(ctx, g, &b("attn_k_norm.weight"), &[cfg.head_dim])?,
                wqkv: q2_cat(ctx, g, &[&b("attn_q.weight"), &b("attn_k.weight"), &b("attn_v.weight")])?,
                q_out: g.tensor(&b("attn_q.weight")).ok_or("no attn_q")?.dims[1] as usize,
                kv_out: g.tensor(&b("attn_k.weight")).ok_or("no attn_k")?.dims[1] as usize,
                wo: q2(ctx, g, &b("attn_output.weight"))?,
                ffn_gate_up: q2_cat(ctx, g, &[&b("ffn_gate.weight"), &b("ffn_up.weight")])?,
                ffn_gate_out: g.tensor(&b("ffn_gate.weight")).ok_or("no ffn_gate")?.dims[1] as usize,
                ffn_down: q2(ctx, g, &b("ffn_down.weight"))?,
            });
        }
        let head = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        Ok(Qwen3 {
            tok_embd: g.raw("token_embd.weight")?,
            out_norm: f32t(ctx, g, "output_norm.weight", &[cfg.n_embd])?,
            lm_head: q2(ctx, g, head)?,
            cfg, ctx: ctx.clone(), layers,
        })
    }

    pub fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.n_embd;
        let row_bytes = d / 128 * 34;
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let off = t as usize * row_bytes;
            v.extend(deq_raw(&self.tok_embd[off..off + row_bytes], d, 42).expect("embed row"));
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// Full RoPE over head_dim (Qwen3 rotates the whole head, unlike the 27B's partial 64/256).
    fn rope(&self, x: &Tensor, n_heads: usize, offset: usize) -> Tensor {
        x.rope(n_heads, self.cfg.head_dim, self.cfg.rope_base, offset)
    }

    fn attn(&self, h: &Tensor, l: &Layer, cache: &mut Option<(Tensor, Tensor)>, offset: usize) -> Tensor {
        let (t, hd, nh, nkv) = (h.shape[0], self.cfg.head_dim, self.cfg.n_head, self.cfg.n_head_kv);
        // One fused matmul emits [q | k | v]; split, QK-norm per head, RoPE.
        let qkv = h.matmul_q2_0(&l.wqkv);
        let q = qkv.narrow(1, 0, l.q_out).reshape(&[t, nh, hd]).rmsnorm(&l.q_norm, self.cfg.eps).reshape(&[t, nh * hd]);
        let k = qkv.narrow(1, l.q_out, l.kv_out).reshape(&[t, nkv, hd]).rmsnorm(&l.k_norm, self.cfg.eps).reshape(&[t, nkv * hd]);
        let v = qkv.narrow(1, l.q_out + l.kv_out, l.kv_out).contiguous();

        let q = self.rope(&q, nh, offset);
        let k = self.rope(&k, nkv, offset);

        let (kc, vc) = match cache.take() {
            Some((pk, pv)) => (pk.cat(&k, 0), pv.cat(&v, 0)),
            None => (k, v),
        };
        let o = if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv)
        } else {
            nn::causal_attention(&q, &kc, &vc, nh, nkv)
        };
        *cache = Some((kc, vc));
        o.matmul_q2_0(&l.wo)
    }

    fn ffn(&self, h: &Tensor, l: &Layer) -> Tensor {
        let gu = h.matmul_q2_0(&l.ffn_gate_up);
        let d = l.ffn_gate_out;
        gu.narrow(1, 0, d).silu().mul(&gu.narrow(1, d, d)).matmul_q2_0(&l.ffn_down)
    }

    /// Prefill (stateless): logits [T, n_vocab].
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        let mut cache = Cache::new(&self.cfg);
        self.forward_cached(tokens, &mut cache)
    }

    /// Feed `tokens`, carrying K/V in `cache`. Prompt once, then one token per step.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        use ferric_tensor::batch;
        let mut x = self.embed(tokens);
        let pos = cache.pos;
        for (il, l) in self.layers.iter().enumerate() {
            let lc = &mut cache.kv[il];
            let xin = &x;
            x = batch(&self.ctx, || {
                let y = self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos);
                let xy = xin.add(&y);
                self.ffn(&xy.rmsnorm(&l.ffn_norm, self.cfg.eps), l).add(&xy)
            });
        }
        cache.pos += tokens.len();
        batch(&self.ctx, || x.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q2_0(&self.lm_head))
    }
}
