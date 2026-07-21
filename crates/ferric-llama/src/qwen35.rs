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
use ferric_gguf::{GgufSource, Meta};
#[allow(unused_imports)] use ferric_gguf::GgufFile;
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
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let u = |k: &str| -> Result<usize, String> {
            match g.metadata().get(k) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing metadata {k}")) }
        };
        let f = |k: &str| -> Result<f32, String> {
            match g.metadata().get(k) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing metadata {k}")) }
        };
        let n_vocab = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.len(), _ => return Err("missing tokenizer.ggml.tokens".into()) };
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
    pub wqkv: Q2_0Weights, // wq | wk | wv stacked: one matmul, then split by q_out/k_out
    pub q_out: usize,      // n_head·head_dim·2 (query and gate, interleaved per head)
    pub kv_out: usize,     // n_head_kv·head_dim (each of k and v)
    pub wo: Q2_0Weights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct GdnW {
    // in_proj = qkv | z | alpha | beta stacked: the four projections all read the same h, so one
    // fused matmul replaces four (48 GDN layers × 4 → 1).
    pub in_proj: Q2_0Weights,
    pub qkv_out: usize, // 2·key_dim + d_inner
    pub z_out: usize,   // d_inner
    pub ba_out: usize,  // n_v_heads (each of alpha, beta)
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
pub(crate) enum LayerCache {
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
    pub ffn_gate_up: Q2_0Weights, // gate stacked over up: one matmul, then split
    pub ffn_gate_out: usize,      // where gate ends / up begins in the fused output
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
pub(crate) fn q2(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Q2_0Weights, String> {
    let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
    if t.ggml_type != 42 { return Err(format!("{name}: expected Q2_0 (42), got {}", t.ggml_type)); }
    let (inn, out) = (t.dims[0] as usize, t.dims[1] as usize);
    Ok(Q2_0Weights::from_bytes(ctx, &g.raw(name)?, out, inn))
}

/// Stack several same-input `[out, in]` weights along the output dim into one `[Σout, in]`.
/// Q2_0 is row-major (each output row is a contiguous run of 34-byte blocks), so stacking outputs
/// is literally concatenating the raw byte streams — no repacking. One fused matmul over the group
/// beats the separate ones at decode width (1.79× measured on gate+up), because a lone-token GEMV
/// is occupancy-starved and merging the output counts fills the machine.
pub(crate) fn q2_cat(ctx: &Arc<Context>, g: &impl GgufSource, names: &[&str]) -> Result<Q2_0Weights, String> {
    let mut inn = None;
    let mut out = 0usize;
    let mut raw = Vec::new();
    for &name in names {
        let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        if t.ggml_type != 42 { return Err(format!("{name}: expected Q2_0 (42), got {}", t.ggml_type)); }
        let i = t.dims[0] as usize;
        if *inn.get_or_insert(i) != i { return Err(format!("{name}: input dim differs")); }
        out += t.dims[1] as usize;
        raw.extend(g.raw(name)?);
    }
    Ok(Q2_0Weights::from_bytes(ctx, &raw, out, inn.unwrap()))
}

pub(crate) fn f32t(ctx: &Arc<Context>, g: &impl GgufSource, name: &str, shape: &[usize]) -> Result<Tensor, String> {
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, shape))
}

/// Load a weight in **whatever packed format** the GGUF stored it (Q2_0/Q4_0/Q4_K/Q8_0) as a QMatrix.
pub(crate) fn qm(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<ferric_tensor::QMatrix, String> {
    let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
    let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
    // Native packed kernel if we have one; otherwise dequantize to f32 and run the dense fallback
    // (e.g. IQ4_XS/IQ4_NL — the quant is decoded, just not matmul'd in packed form).
    if ferric_tensor::QMatrix::block_bytes(ty).is_some() {
        ferric_tensor::QMatrix::from_bytes(ctx, &g.raw(name)?, ty, rows, cols)
    } else {
        Ok(ferric_tensor::QMatrix::from_dense(ctx, &g.dequant(name)?, rows, cols))
    }
}

/// Concatenate several same-format weights along the output dim into one QMatrix (fused qkv, gate_up).
/// In a real GGUF every projection in a layer shares one quant format, so this just stacks their bytes.
pub(crate) fn qm_cat(ctx: &Arc<Context>, g: &impl GgufSource, names: &[&str]) -> Result<ferric_tensor::QMatrix, String> {
    let (mut inn, mut out, mut ty) = (None, 0usize, None);
    for &name in names {
        let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let this = *ty.get_or_insert(t.ggml_type);
        if t.ggml_type != this { return Err(format!("{name}: mixed quant formats in one fused matmul")); }
        let i = t.dims[0] as usize;
        if *inn.get_or_insert(i) != i { return Err(format!("{name}: input dim differs")); }
        out += t.dims[1] as usize;
    }
    let (ty, inn) = (ty.unwrap(), inn.unwrap());
    // Concatenate along the output dim: for the packed path stack raw block bytes; for the dense
    // fallback stack the dequantized row-major [out_i, inn] blocks — both yield the fused [out, inn].
    if ferric_tensor::QMatrix::block_bytes(ty).is_some() {
        let mut raw = Vec::new();
        for &name in names { raw.extend(g.raw(name)?); }
        ferric_tensor::QMatrix::from_bytes(ctx, &raw, ty, out, inn)
    } else {
        let mut f = Vec::new();
        for &name in names { f.extend(g.dequant(name)?); }
        Ok(ferric_tensor::QMatrix::from_dense(ctx, &f, out, inn))
    }
}

impl Qwen35 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Qwen35, String> {
        let cfg = Cfg::from_gguf(g)?;
        let conv_dim = cfg.key_dim() * 2 + cfg.d_inner;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let mixer = if cfg.is_recurrent(il) {
                Mixer::Gdn(GdnW {
                    in_proj: q2_cat(ctx, g, &[&b("attn_qkv.weight"), &b("attn_gate.weight"), &b("ssm_alpha.weight"), &b("ssm_beta.weight")])?,
                    qkv_out: g.tensor(&b("attn_qkv.weight")).ok_or("no attn_qkv")?.dims[1] as usize,
                    z_out: g.tensor(&b("attn_gate.weight")).ok_or("no attn_gate")?.dims[1] as usize,
                    ba_out: g.tensor(&b("ssm_alpha.weight")).ok_or("no ssm_alpha")?.dims[1] as usize,
                    conv1d: f32t(ctx, g, &b("ssm_conv1d.weight"), &[conv_dim, cfg.conv_kernel])?,
                    dt_bias: f32t(ctx, g, &b("ssm_dt.bias"), &[cfg.n_v_heads])?,
                    a: f32t(ctx, g, &b("ssm_a"), &[cfg.n_v_heads])?,
                    norm: f32t(ctx, g, &b("ssm_norm.weight"), &[cfg.head_v_dim()])?,
                    out: q2(ctx, g, &b("ssm_out.weight"))?,
                })
            } else {
                Mixer::Attn(AttnW {
                    wqkv: q2_cat(ctx, g, &[&b("attn_q.weight"), &b("attn_k.weight"), &b("attn_v.weight")])?,
                    q_out: g.tensor(&b("attn_q.weight")).ok_or("no attn_q")?.dims[1] as usize,
                    kv_out: g.tensor(&b("attn_k.weight")).ok_or("no attn_k")?.dims[1] as usize,
                    wo: q2(ctx, g, &b("attn_output.weight"))?,
                    q_norm: f32t(ctx, g, &b("attn_q_norm.weight"), &[cfg.head_dim])?,
                    k_norm: f32t(ctx, g, &b("attn_k_norm.weight"), &[cfg.head_dim])?,
                })
            };
            layers.push(Layer {
                attn_norm: f32t(ctx, g, &b("attn_norm.weight"), &[cfg.n_embd])?,
                post_norm: f32t(ctx, g, &b("post_attention_norm.weight"), &[cfg.n_embd])?,
                ffn_gate_up: q2_cat(ctx, g, &[&b("ffn_gate.weight"), &b("ffn_up.weight")])?,
                ffn_gate_out: g.tensor(&b("ffn_gate.weight")).ok_or("no ffn_gate")?.dims[1] as usize,
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
        let nkv = self.cfg.n_head_kv;
        // One fused matmul emits [q_and_gate | k | v]; split it back out.
        let qkv = h.matmul_q2_0(&w.wqkv);
        let qf = qkv.narrow(1, 0, w.q_out).reshape(&[t, nh, hd * 2]);
        let q = qf.narrow(2, 0, hd).rmsnorm(&w.q_norm, self.cfg.eps).reshape(&[t, nh * hd]);
        let gate = qf.narrow(2, hd, hd).contiguous().reshape(&[t, nh * hd]);

        let k = qkv.narrow(1, w.q_out, w.kv_out).reshape(&[t, nkv, hd]).rmsnorm(&w.k_norm, self.cfg.eps).reshape(&[t, nkv * hd]);
        let v = qkv.narrow(1, w.q_out + w.kv_out, w.kv_out).contiguous();

        let q = self.rope_partial(&q, nh, offset);
        let k = self.rope_partial(&k, nkv, offset);

        // Append this step's K/V to the history, then attend over all of it.
        let (kc, vc) = match cache.take() {
            Some(LayerCache::Attn { k: pk, v: pv }) => (pk.cat(&k, 0), pv.cat(&v, 0)),
            _ => (k, v),
        };
        let s = kc.shape[0];
        let o = if t == 1 {
            nn::decode_attention(&q, &kc, &vc, nh, nkv, 0.0)
        } else if t == s && s <= 65535 && hd <= 128 {
            // Prefill: q and the history are the same span. Flash streams keys in chunks with
            // online softmax — O(hd) memory, any T, same math as the composed causal path.
            q.flash_attention_prefill(&kc, &vc, nh, nkv, hd)
        } else {
            nn::causal_attention(&q, &kc, &vc, nh, nkv, 0.0)
        };
        *cache = Some(LayerCache::Attn { k: kc, v: vc });
        o.mul(&gate.sigmoid()).matmul_q2_0(&w.wo)
    }

    fn gdn(&self, h: &Tensor, w: &GdnW, cache: &mut Option<LayerCache>) -> Tensor {
        let c = &self.cfg;
        let (t, nk, nv) = (h.shape[0], c.n_k_heads, c.n_v_heads);
        let (dk, dv, kd) = (c.head_k_dim, c.head_v_dim(), c.key_dim());

        // One fused matmul emits [qkv | z | alpha | beta]; split it back out. qkv feeds the conv,
        // the rest are used as-is.
        let proj = h.matmul_q2_0(&w.in_proj);
        let (qo, zo, bo) = (w.qkv_out, w.z_out, w.ba_out);
        let qkv = proj.narrow(1, 0, qo).contiguous();
        let z = proj.narrow(1, qo, zo);
        let alpha_raw = proj.narrow(1, qo + zo, bo);
        let beta_raw = proj.narrow(1, qo + zo + bo, bo);

        // conv over the whole fused q|k|v, then split — the conv is causal and depthwise, so
        // splitting after it is identical to convolving each part separately.
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
        let alpha = alpha_raw.add(&w.dt_bias.reshape(&[1, nv])).softplus().mul(&w.a.reshape(&[1, nv]));
        let beta = beta_raw.sigmoid();
        let gb = alpha.reshape(&[t, nv, 1]).cat(&beta.reshape(&[t, nv, 1]), 2);

        let (o, state) = q.gated_delta_rule_stateful(&k, &v, &gb, nv, dk, dv, prev_state.as_ref());
        *cache = Some(LayerCache::Gdn { state, conv: conv_tail });

        // gated RMSNorm over head_v_dim, gated by silu(z)
        let z = z.reshape(&[t, nv, dv]);
        o.rmsnorm(&w.norm, c.eps).mul(&z.silu()).reshape(&[t, c.d_inner]).matmul_q2_0(&w.out)
    }

    fn ffn(&self, h: &Tensor, l: &Layer) -> Tensor {
        // gate_up matmul → fused SwiGLU (silu(gate)·up in one kernel) → down projection.
        h.matmul_q2_0(&l.ffn_gate_up).swiglu(l.ffn_gate_out).matmul_q2_0(&l.ffn_down)
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
        // Batch each layer's ~10 dispatches into one submission — cutting ~640 submits/token to
        // ~70 and removing most of the per-op encoder+submit overhead (measured ~38 ms/token).
        // Batching is *per layer*, not whole-forward: one command buffer must retain every bind
        // group it records until submit, and holding all 640 across 64 layers exhausts the driver's
        // per-submission resource budget. Ops still run in issue order, so the result is identical.
        use ferric_tensor::{batch, prof};
        let mut x = self.embed(tokens);
        prof(&self.ctx, "embed");
        let pos = cache.pos;
        // FERRIC_PROFILE splits each layer into per-category submissions so the sync'd timer can
        // attribute time (mixer vs ffn); otherwise the whole layer is one batch (fewer submits).
        let profiling = std::env::var("FERRIC_PROFILE").is_ok();
        for (il, l) in self.layers.iter().enumerate().take(n) {
            let lc = &mut cache.layers[il];
            let xin = &x;
            if profiling {
                // One submit per category (mixer, then ffn) so the sync'd timer attributes GPU work,
                // not op count. Eager-per-op would over-charge whichever region has the most small
                // dispatches; per-category batching keeps the split honest.
                let is_attn = matches!(l.mixer, Mixer::Attn(_));
                let y = batch(&self.ctx, || {
                    let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                    match &l.mixer { Mixer::Attn(w) => self.attn(&h, w, lc, pos), Mixer::Gdn(w) => self.gdn(&h, w, lc) }
                });
                prof(&self.ctx, if is_attn { "attn" } else { "gdn" });
                let xy = xin.add(&y);
                x = batch(&self.ctx, || self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy));
                prof(&self.ctx, "ffn");
            } else {
                x = batch(&self.ctx, || {
                    let h = xin.rmsnorm(&l.attn_norm, self.cfg.eps);
                    let y = match &l.mixer {
                        Mixer::Attn(w) => self.attn(&h, w, lc, pos),
                        Mixer::Gdn(w) => self.gdn(&h, w, lc),
                    };
                    let xy = xin.add(&y);
                    self.ffn(&xy.rmsnorm(&l.post_norm, self.cfg.eps), l).add(&xy)
                });
            }
        }
        cache.pos += tokens.len();
        let out = batch(&self.ctx, || x.rmsnorm(&self.out_norm, self.cfg.eps).matmul_q2_0(&self.lm_head));
        prof(&self.ctx, "lm_head");
        out
    }
}
