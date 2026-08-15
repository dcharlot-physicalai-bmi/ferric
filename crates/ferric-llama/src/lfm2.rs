//! **Liquid AI LFM2 / LFM2.5** — the conv/attention hybrid, with real incremental state.
//!
//! Both mixers are recurrent in the sense that matters for decode: attention carries a KV cache, and
//! the short-conv carries `l_cache - 1` timesteps of the *pre-convolution* gated signal. Carrying
//! both is what makes a 128k-context model actually usable — re-running the prefix per token is
//! O(n²) and turns a long context into a number on a spec sheet.
//!
//! Per-token cost here is O(1) in the conv layers (a 2-row state, independent of sequence length)
//! and O(s) in the 8 attention layers, against O(n) layers × O(n) tokens for a prefix re-run.
//!
//! ```text
//! conv block:  BCx = in_proj(norm(h)); B,C,x = chunks 0,1,2
//!              bx  = B ⊙ x                      (no activation anywhere in this block)
//!              y   = C ⊙ conv(concat(state, bx))[-t:] ; state = last L-1 rows of concat(state, bx)
//!              out = out_proj(y)
//! attn block:  q,k,v → per-head QK RMSNorm → RoPE at the ABSOLUTE position → KV cache → GQA
//! every block: + SwiGLU FFN, both residual
//! final:       token_embd_norm (the FINAL norm despite its name) → tied head
//! ```
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::{append2, nn, KvBuf, QMatrix, Tensor};
use std::sync::Arc;

pub struct Cfg {
    pub n_layer: usize, pub d: usize, pub n_head: usize, pub head_dim: usize,
    pub n_vocab: usize, pub eps: f32, pub rope_base: f32, pub conv_l: usize,
    /// Per-block KV head count; `0` marks a short-conv block. Read from the ARRAY, never a scalar.
    pub kv: Vec<usize>,
}

enum Mixer {
    Conv { in_proj: QMatrix, conv: Tensor, out_proj: QMatrix },
    Attn { q: QMatrix, k: QMatrix, v: QMatrix, o: QMatrix, q_norm: Tensor, k_norm: Tensor, n_kv: usize },
}

struct Block {
    norm: Tensor, mixer: Mixer, ffn_norm: Tensor,
    gate: QMatrix, up: QMatrix, down: QMatrix,
}

/// Per-sequence decode state: KV for attention blocks, a rolling window for conv blocks.
pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
    /// `[l_cache-1, d]` of the PRE-convolution gated signal `B⊙x` — the reference stores the signal,
    /// not the conv output, and storing the wrong one produces plausible drift rather than an error.
    conv: Vec<Option<Tensor>>,
}

impl Cache {
    pub fn new(cfg: &Cfg) -> Cache {
        Cache {
            pos: 0,
            kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect(),
            conv: (0..cfg.n_layer).map(|_| None).collect(),
        }
    }
}

pub struct Lfm2 {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    blocks: Vec<Block>,
    out_norm: Tensor,
    head: QMatrix,
    embd_ty: u32,
    embd_raw: Vec<u8>,
}

impl Lfm2 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Lfm2, String> {
        let md = g.metadata();
        let u = |k: &str| match md.get(&format!("lfm2.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing lfm2.{k}")) };
        let f = |k: &str| match md.get(&format!("lfm2.{k}")) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing lfm2.{k}")) };
        let n_layer = u("block_count")?;
        let d = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let rope_base = f("rope.freq_base")?;
        let conv_l = u("shortconv.l_cache")?;
        let kv: Vec<usize> = match md.get("lfm2.attention.head_count_kv") {
            Some(Meta::Arr(v)) => v.iter().map(|m| match m { Meta::U(x) => *x as usize, Meta::I(x) => *x as usize, _ => 0 }).collect(),
            Some(Meta::U(v)) => vec![*v as usize; n_layer],
            _ => return Err("no lfm2.attention.head_count_kv".into()),
        };
        if kv.len() != n_layer { return Err(format!("schedule covers {} of {n_layer} blocks", kv.len())); }
        let head_dim = d / n_head;
        let n_vocab = g.tensor("token_embd.weight").ok_or("no token_embd")?.dims[1] as usize;

        let qm = |name: &str| -> QMatrix {
            let t = g.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
            let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
            if QMatrix::block_bytes(ty).is_some() {
                QMatrix::from_bytes(ctx, &g.raw(name).unwrap(), ty, rows, cols).unwrap()
            } else {
                QMatrix::from_dense(ctx, &g.dequant(name).unwrap(), rows, cols)
            }
        };
        let ft = |name: &str, shape: &[usize]| Tensor::from_vec(ctx, &g.dequant(name).unwrap(), shape);

        let blocks = (0..n_layer).map(|il| {
            let b = |s: &str| format!("blk.{il}.{s}");
            let n_kv = kv[il];
            let mixer = if n_kv > 0 {
                Mixer::Attn {
                    q: qm(&b("attn_q.weight")), k: qm(&b("attn_k.weight")),
                    v: qm(&b("attn_v.weight")), o: qm(&b("attn_output.weight")),
                    q_norm: ft(&b("attn_q_norm.weight"), &[head_dim]),
                    k_norm: ft(&b("attn_k_norm.weight"), &[head_dim]),
                    n_kv,
                }
            } else {
                // GGUF [L, d] dequantises row-major to [d, L] = the [C, L] the conv kernel wants.
                Mixer::Conv {
                    in_proj: qm(&b("shortconv.in_proj.weight")),
                    conv: ft(&b("shortconv.conv.weight"), &[d, conv_l]),
                    out_proj: qm(&b("shortconv.out_proj.weight")),
                }
            };
            Block {
                norm: ft(&b("attn_norm.weight"), &[d]), mixer,
                ffn_norm: ft(&b("ffn_norm.weight"), &[d]),
                gate: qm(&b("ffn_gate.weight")), up: qm(&b("ffn_up.weight")), down: qm(&b("ffn_down.weight")),
            }
        }).collect();

        Ok(Lfm2 {
            ctx: ctx.clone(),
            cfg: Cfg { n_layer, d, n_head, head_dim, n_vocab, eps, rope_base, conv_l, kv },
            blocks,
            // `token_embd_norm` is the FINAL norm, not a norm on the embeddings — llama.cpp maps it
            // through a dedicated enum whose comment reads "fix for wrong tensor name".
            out_norm: ft("token_embd_norm.weight", &[d]),
            head: if g.tensor("output.weight").is_some() { qm("output.weight") } else { qm("token_embd.weight") },
            embd_ty: g.tensor("token_embd.weight").unwrap().ggml_type,
            embd_raw: g.raw("token_embd.weight")?,
        })
    }

    fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.d;
        let row = ferric_gguf::type_size(self.embd_ty, d).expect("embd type");
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let o = t as usize * row;
            v.extend(ferric_gguf::deq_raw(&self.embd_raw[o..o + row], d, self.embd_ty).expect("embed row"));
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// Feed `tokens`, carrying attention KV and conv state in `cache`. Prompt once, then one token
    /// per step — the per-token cost does not grow with how much has already been fed.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        let (d, n_head, head_dim, eps) = (self.cfg.d, self.cfg.n_head, self.cfg.head_dim, self.cfg.eps);
        let t = tokens.len();
        let pos = cache.pos;
        let mut x = self.embed(tokens);

        for (il, blk) in self.blocks.iter().enumerate() {
            let h = x.rmsnorm(&blk.norm, eps);
            let op = match &blk.mixer {
                Mixer::Attn { q, k, v, o, q_norm, k_norm, n_kv } => {
                    // RoPE at the ABSOLUTE position, so a cached decode step rotates by `pos`, not 0.
                    let qh = h.matmul_q(q).reshape(&[t * n_head, head_dim]).rmsnorm(q_norm, eps)
                        .reshape(&[t, n_head * head_dim]).rope(n_head, head_dim, self.cfg.rope_base, pos);
                    let kh = h.matmul_q(k).reshape(&[t * n_kv, head_dim]).rmsnorm(k_norm, eps)
                        .reshape(&[t, n_kv * head_dim]).rope(*n_kv, head_dim, self.cfg.rope_base, pos);
                    let vh = h.matmul_q(v);
                    // One `&mut` to the tuple, then split into its two disjoint fields: the borrow
                    // checker cannot see that .0 and .1 do not alias when reached through two separate
                    // index expressions.
                    let (kc, vc) = { let e = &mut cache.kv[il]; append2(&self.ctx, &mut e.0, &kh, &mut e.1, &vh) };
                    let att = if t == 1 {
                        nn::decode_attention(&qh, &kc, &vc, n_head, *n_kv, 0.0)
                    } else {
                        // Serves full prefill AND a chunk against a longer cache.
                        nn::chunked_attention(&qh, &kc, &vc, n_head, *n_kv, 0.0)
                    };
                    att.matmul_q(o)
                }
                Mixer::Conv { in_proj, conv, out_proj } => {
                    let bcx = h.matmul_q(in_proj);
                    let b = bcx.narrow(1, 0, d).contiguous();
                    let c = bcx.narrow(1, d, d).contiguous();
                    let xx = bcx.narrow(1, 2 * d, d).contiguous();
                    let bx = b.mul(&xx);                       // no activation in this block
                    let keep = self.cfg.conv_l - 1;
                    // Prepend the carried window. The conv is causal with zero padding, so rows
                    // [0, keep) of the concatenated signal see a wrong (zero) history — they are
                    // discarded; only the `t` new rows are kept, and those see the real history.
                    let full = match &cache.conv[il] {
                        Some(st) => st.cat(&bx, 0),
                        None => Tensor::from_vec(&self.ctx, &vec![0f32; keep * d], &[keep, d]).cat(&bx, 0),
                    };
                    let n = full.shape[0];
                    let conv_out = full.depthwise_conv1d_causal(conv, self.cfg.conv_l).narrow(0, n - t, t).contiguous();
                    // Carry the last `keep` rows of the PRE-conv signal, matching the reference.
                    cache.conv[il] = Some(full.narrow(0, n - keep, keep).contiguous());
                    c.mul(&conv_out).matmul_q(out_proj)
                }
            };
            x = x.add(&op);
            let hn = x.rmsnorm(&blk.ffn_norm, eps);
            x = x.add(&hn.matmul_q(&blk.gate).silu().mul(&hn.matmul_q(&blk.up)).matmul_q(&blk.down));
        }
        cache.pos += t;
        x.rmsnorm(&self.out_norm, eps).matmul_q(&self.head)
    }
}
