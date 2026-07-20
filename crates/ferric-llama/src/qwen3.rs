//! **Dense Qwen-family transformer** — PrismML's Ternary Bonsai (1.7B/4B, arch `qwen3`) *and* standard
//! `qwen3` **and `qwen2`** GGUFs off Hugging Face. GQA every layer, SwiGLU, RoPE, RMSNorm; the arch
//! differences are handled by feature-detection: QK-norm (Qwen3 only) and QKV bias (Qwen2 only) are
//! read from tensor presence, and all metadata keys are architecture-prefixed. **Format-agnostic**: each weight
//! loads in whatever quant the GGUF stored it (`QMatrix` over Q2_0/Q4_0/Q4_K/Q6_K/Q8_0 natively, plus a
//! dequant-to-f32 dense fallback for IQ4_XS/IQ4_NL and other kernel-less types), so this runs
//! a PrismML ternary model *and* a genuine `Q4_K_M` model off Hugging Face — which mixes Q4_K and
//! Q6_K, even within one qkv (see `Proj`). The ternary 1.7B is ~450 MB packed, so it fits WebGPU's
//! memory limits and this same code compiles to wasm32 to drive a browser tab.
//!
//! Projection-fusion + KV-cache tricks proven on the 27B: q/k/v fuse into one matmul (when they share
//! a format), gate/up into another, attention resumes from cached K/V so decode is one step per token.
use crate::qwen35::{f32t, qm, qm_cat};
use ferric_core::Context;
use ferric_gguf::{deq_raw, GgufSource, Meta};
use ferric_tensor::{nn, KvBuf, QMatrix, Tensor};
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
    pub has_qk_norm: bool, // Qwen3 has it, Qwen2/Llama don't
    pub qkv_bias: bool,    // Qwen2 has q/k/v biases; Qwen3/Llama don't
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        // The metadata keys are prefixed by the architecture (qwen2.*, qwen3.*, llama.*, …). Read it
        // once so one loader serves the whole dense Qwen/Llama family.
        let arch = match g.metadata().get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "qwen3".into() };
        let u = |k: &str| match g.metadata().get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing {arch}.{k}")) };
        let f = |k: &str| match g.metadata().get(&format!("{arch}.{k}")) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing {arch}.{k}")) };
        let n_vocab = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.len(), _ => return Err("no tokens".into()) };
        let n_head = u("attention.head_count")?;
        // Some arches (qwen2) omit key_length; then head_dim = embedding_length / head_count.
        let head_dim = u("attention.key_length").unwrap_or_else(|_| u("embedding_length").unwrap_or(0) / n_head.max(1));
        Ok(Cfg {
            n_embd: u("embedding_length")?,
            n_layer: u("block_count")?,
            n_head,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim,
            n_ff: u("feed_forward_length")?,
            n_vocab,
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope_base: f("rope.freq_base")?,
            has_qk_norm: g.tensor("blk.0.attn_q_norm.weight").is_some(),
            qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
        })
    }
}

/// A projection that is *logically* one matmul emitting several stacked outputs (q|k|v, gate|up).
/// If every part shares a quant format it's byte-fused into one QMatrix (the fast path); real Q4_K_M
/// models mix formats even within qkv (V is often Q6_K while Q/K are Q4_K), so it falls back to one
/// matmul per part, concatenated — same result, one extra dispatch.
enum Proj {
    Fused(QMatrix),
    Split(Vec<QMatrix>),
}
impl Proj {
    fn load(ctx: &Arc<Context>, g: &impl GgufSource, names: &[&str]) -> Result<Proj, String> {
        let types: Vec<u32> = names.iter().map(|n| g.tensor(n).map(|t| t.ggml_type).unwrap_or(0)).collect();
        if names.len() > 1 && types.windows(2).all(|w| w[0] == w[1]) {
            Ok(Proj::Fused(qm_cat(ctx, g, names)?))
        } else if names.len() == 1 {
            Ok(Proj::Fused(qm(ctx, g, names[0])?))
        } else {
            Ok(Proj::Split(names.iter().map(|n| qm(ctx, g, n)).collect::<Result<_, _>>()?))
        }
    }
    fn matmul(&self, x: &Tensor) -> Tensor {
        match self {
            Proj::Fused(w) => x.matmul_q(w),
            Proj::Split(ws) => {
                let mut out = x.matmul_q(&ws[0]);
                for w in &ws[1..] { out = out.cat(&x.matmul_q(w), 1); }
                out
            }
        }
    }
    /// gate_up projection + SwiGLU. When gate|up is one fused Q4_K/Q5_K/Q6_K weight, one fused kernel
    /// does both (no [t, 2·n_ff] intermediate); otherwise the plain matmul + SwiGLU. Same result either way.
    fn gate_up_swiglu(&self, x: &Tensor, n_ff: usize) -> Tensor {
        // FERRIC_NOFUSE forces the un-fused path — for controlled A/B of the fusion, same binary.
        if std::env::var("FERRIC_NOFUSE").is_err() {
            if let Proj::Fused(w) = self {
                if let Some(o) = x.try_matmul_swiglu(w) { return o; }
            }
        }
        self.matmul(x).swiglu(n_ff)
    }
}

pub struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    q_norm: Option<Tensor>, // QK-norm: Qwen3 only
    k_norm: Option<Tensor>,
    wqkv: Proj, // q | k | v stacked (fused if same format, else separate matmuls concatenated)
    qkv_bias: Option<Tensor>, // Qwen2: concatenated q|k|v bias, added after the projection
    q_out: usize,
    kv_out: usize,
    wo: QMatrix,
    ffn_gate_up: Proj,
    ffn_gate_out: usize,
    ffn_down: QMatrix,
}

/// Per-layer attention K/V history. One step per token: append the new K/V into a grow-in-place
/// `KvBuf` (no O(len) re-concatenate), then attend over the [len, width] view of all of it.
#[derive(Default)]
pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
}
impl Cache {
    pub fn new(cfg: &Cfg) -> Cache { Cache { pos: 0, kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect() } }
}

pub struct Qwen3 {
    pub cfg: Cfg,
    ctx: Arc<Context>,
    tok_embd: Vec<u8>, // Q2_0 rows, gathered + dequantized on the CPU (avoids parking the table on GPU)
    layers: Vec<Layer>,
    out_norm: Tensor,
    lm_head: QMatrix,
    embd_type: u32,
    rope_freqs: Option<Tensor>, // Llama-3 rope-scaling factors [head_dim/2]; None for Qwen
}

impl Qwen3 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Qwen3, String> {
        let cfg = Cfg::from_gguf(g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let qkv_bias = if cfg.qkv_bias {
                let mut bias = g.dequant(&b("attn_q.bias"))?;
                bias.extend(g.dequant(&b("attn_k.bias"))?);
                bias.extend(g.dequant(&b("attn_v.bias"))?);
                Some(Tensor::from_vec(ctx, &bias, &[1, bias.len()]))
            } else { None };
            // Q/K/V: Qwen/Llama store three separate weights (we fuse them); Phi-3 stores ONE pre-fused
            // `attn_qkv` (q|k|v stacked) — load it directly and take the split widths from the config.
            let (wqkv, q_out, kv_out) = if g.tensor(&b("attn_qkv.weight")).is_some() {
                (Proj::load(ctx, g, &[&b("attn_qkv.weight")])?, cfg.n_head * cfg.head_dim, cfg.n_head_kv * cfg.head_dim)
            } else {
                (Proj::load(ctx, g, &[&b("attn_q.weight"), &b("attn_k.weight"), &b("attn_v.weight")])?,
                 g.tensor(&b("attn_q.weight")).ok_or("no attn_q")?.dims[1] as usize,
                 g.tensor(&b("attn_k.weight")).ok_or("no attn_k")?.dims[1] as usize)
            };
            // FFN gate|up: Qwen/Llama store separate `ffn_gate`+`ffn_up`; Phi-3 pre-fuses them into
            // `ffn_up` ([2·n_ff, n_embd], gate first) — same layout our SwiGLU fast-path already expects.
            let (ffn_gate_up, ffn_gate_out) = if g.tensor(&b("ffn_gate.weight")).is_some() {
                (Proj::load(ctx, g, &[&b("ffn_gate.weight"), &b("ffn_up.weight")])?,
                 g.tensor(&b("ffn_gate.weight")).unwrap().dims[1] as usize)
            } else {
                (Proj::load(ctx, g, &[&b("ffn_up.weight")])?, cfg.n_ff)
            };
            layers.push(Layer {
                attn_norm: f32t(ctx, g, &b("attn_norm.weight"), &[cfg.n_embd])?,
                ffn_norm: f32t(ctx, g, &b("ffn_norm.weight"), &[cfg.n_embd])?,
                q_norm: if cfg.has_qk_norm { Some(f32t(ctx, g, &b("attn_q_norm.weight"), &[cfg.head_dim])?) } else { None },
                k_norm: if cfg.has_qk_norm { Some(f32t(ctx, g, &b("attn_k_norm.weight"), &[cfg.head_dim])?) } else { None },
                wqkv,
                qkv_bias,
                q_out,
                kv_out,
                wo: qm(ctx, g, &b("attn_output.weight"))?,
                ffn_gate_up,
                ffn_gate_out,
                ffn_down: qm(ctx, g, &b("ffn_down.weight"))?,
            });
        }
        let head = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        Ok(Qwen3 {
            tok_embd: g.raw("token_embd.weight")?,
            out_norm: f32t(ctx, g, "output_norm.weight", &[cfg.n_embd])?,
            lm_head: qm(ctx, g, head)?,
            embd_type: g.tensor("token_embd.weight").ok_or("no token_embd")?.ggml_type,
            rope_freqs: g.tensor("rope_freqs.weight").map(|t| {
                let n = t.dims[0] as usize;
                f32t(ctx, g, "rope_freqs.weight", &[n])
            }).transpose()?,
            cfg, ctx: ctx.clone(), layers,
        })
    }

    pub fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.n_embd;
        // Gather + dequantize just the prompt's rows on the CPU, in whatever format the embedding
        // table is stored (Q2_0/Q4_K/…) — beats parking the whole table on the GPU for a gather.
        let row_bytes = ferric_gguf::type_size(self.embd_type, d).expect("embd type");
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let off = t as usize * row_bytes;
            v.extend(deq_raw(&self.tok_embd[off..off + row_bytes], d, self.embd_type).expect("embed row"));
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// Full RoPE over head_dim (Qwen rotates the whole head). Llama-3 applies its per-frequency
    /// `rope_freqs` scaling; Qwen has none, so it's plain RoPE.
    fn rope(&self, x: &Tensor, n_heads: usize, offset: usize) -> Tensor {
        match &self.rope_freqs {
            Some(fs) => x.rope_scaled(fs, n_heads, self.cfg.head_dim, self.cfg.rope_base, offset),
            None => x.rope(n_heads, self.cfg.head_dim, self.cfg.rope_base, offset),
        }
    }

    fn attn(&self, h: &Tensor, l: &Layer, cache: &mut (KvBuf, KvBuf), offset: usize) -> Tensor {
        let (t, hd, nh, nkv) = (h.shape[0], self.cfg.head_dim, self.cfg.n_head, self.cfg.n_head_kv);
        // One fused matmul emits [q | k | v]; (+ bias for Qwen2); split, optional QK-norm, RoPE.
        let qkv = l.wqkv.matmul(h);
        let qkv = match &l.qkv_bias { Some(bias) => qkv.add(bias), None => qkv };
        // QK-norm (Qwen3) normalizes each head; without it (Qwen2/Llama) q/k pass through unchanged.
        let qn = |x: Tensor, n: usize, norm: &Option<Tensor>| match norm {
            Some(w) => x.reshape(&[t, n, hd]).rmsnorm(w, self.cfg.eps).reshape(&[t, n * hd]),
            None => x,
        };
        let q = qn(qkv.narrow(1, 0, l.q_out).contiguous(), nh, &l.q_norm);
        let k = qn(qkv.narrow(1, l.q_out, l.kv_out).contiguous(), nkv, &l.k_norm);
        let v = qkv.narrow(1, l.q_out + l.kv_out, l.kv_out).contiguous();

        let q = self.rope(&q, nh, offset);
        let k = self.rope(&k, nkv, offset);

        // Append the new K/V rows into the grow-in-place cache and read a view over all rows so far.
        // Byte-identical to the old `pk.cat(&k, 0)`, but without re-copying the history each step.
        let kc = cache.0.append(&self.ctx, &k);
        let vc = cache.1.append(&self.ctx, &v);
        // decode: fused single-query; prefill: flash (O(T) memory, no [nh,T,T] matrix) up to its
        // shared-memory limit, else the composed causal path. All three are the same math.
        let s = kc.shape[0];
        let o = if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv)
        } else if t == s && s <= 65535 && hd <= 128 {
            q.flash_attention_prefill(&kc, &vc, nh, nkv, hd)
        } else {
            nn::causal_attention(&q, &kc, &vc, nh, nkv)
        };
        o.matmul_q(&l.wo)
    }

    fn ffn(&self, h: &Tensor, l: &Layer) -> Tensor {
        // Whole-FFN megakernel (gate_up Q4_K + SwiGLU + down Q6_K in one dispatch), OPT-IN via
        // FERRIC_MEGA — correct but ~2× slower at decode (occupancy-bound); off by default.
        if let Proj::Fused(gu) = &l.ffn_gate_up {
            if let Some(o) = h.try_ffn_mega(gu, &l.ffn_down, l.ffn_gate_out) { return o; }
        }
        // staged: gate_up + SwiGLU (one fused kernel when gate|up is a k-quant) → down projection.
        l.ffn_gate_up.gate_up_swiglu(h, l.ffn_gate_out).matmul_q(&l.ffn_down)
    }

    /// Prefill (stateless): logits [T, n_vocab].
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        let mut cache = Cache::new(&self.cfg);
        self.forward_cached(tokens, &mut cache)
    }

    /// Feed `tokens`, carrying K/V in `cache`. Prompt once, then one token per step.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        use ferric_tensor::{batch, prof};
        let profiling = std::env::var("FERRIC_PROFILE").is_ok();
        let mut x = self.embed(tokens);
        prof(&self.ctx, "embed");
        let pos = cache.pos;
        for (il, l) in self.layers.iter().enumerate() {
            let lc = &mut cache.kv[il];
            let xin = &x;
            if profiling {
                // Eager per-category so the sync'd timer attributes attn vs ffn (see qwen35).
                let y = batch(&self.ctx, || self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos));
                prof(&self.ctx, "attn");
                x = batch(&self.ctx, || { let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps); self.ffn(&xy_n, l).add(&xy) });
                prof(&self.ctx, "ffn");
            } else {
                x = batch(&self.ctx, || {
                    let y = self.attn(&xin.rmsnorm(&l.attn_norm, self.cfg.eps), l, lc, pos);
                    // fused: xy = xin + y (next residual), xy_n = rmsnorm(xy) — one kernel, not two.
                    let (xy, xy_n) = xin.add_rmsnorm(&y, &l.ffn_norm, self.cfg.eps);
                    self.ffn(&xy_n, l).add(&xy)
                });
            }
        }
        cache.pos += tokens.len();
        let out = batch(&self.ctx, || x.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q(&self.lm_head));
        prof(&self.ctx, "lm_head");
        out
    }
}
