//! **Google Gemma 4** (`gemma4`, 2026-04-02, Apache-2.0) — E2B / E4B / 26B-A4B / 31B.
//!
//! Not a Gemma 3 with different numbers. Six things differ, and every one of them produces fluent,
//! confident, wrong text rather than an error if it is assumed away — which is why
//! [`crate::arch`] refuses `gemma4` outright rather than letting `arch.starts_with("gemma")` route it
//! down the Gemma-3 path.
//!
//! ```text
//! 1. PER-LAYER EMBEDDINGS   a second embedding table, 256 wide PER LAYER, gated into the residual
//!                           at the end of every block. This is what "E2B" means: 4.6B stored
//!                           parameters presented as ~2B effective.
//! 2. SHARED KV              only the first `n_layer - shared_kv_layers` blocks own K/V at all.
//!                           On E2B that is 15 of 35. Blocks 15..34 project Q only and attend
//!                           against an EARLIER block's cache — 13 for sliding, 14 for global.
//!                           The file still ships K/V weights for those blocks; they are dead.
//! 3. TWO HEAD WIDTHS        global blocks use head_dim 512, sliding blocks 256, in one model.
//! 4. V IS NORMED            a WEIGHTLESS RMS norm on V. There is no `attn_v_norm` tensor to hint
//!                           at it, so it is invisible unless you read the reference.
//! 5. GELU, NOT SWIGLU       `down(gelu(gate(x)) * up(x))`.
//! 6. NO ATTENTION SCALE     f_attention_scale = 1.0 — no 1/sqrt(head_dim) anywhere. The learned
//!                           per-head Q norm absorbs it.
//! ```
//!
//! Plus: separate RoPE base for sliding (1e4) and global (1e6) blocks, proportional RoPE factors on
//! the global blocks only, a per-block output scalar, per-block FFN widths from an array, and final
//! logit softcapping at 30.
//!
//! ## What is not here
//!
//! The 26B-A4B and 31B variants add MoE blocks (router reads the attention output, softmax gating,
//! plus a shared expert whose output is *added* to the routed one). E2B and E4B are dense throughout,
//! so [`Gemma4::load`] refuses a checkpoint carrying `ffn_gate_inp` rather than silently ignoring the
//! experts and running a much smaller model than the file describes.
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::{append2, nn, KvBuf, QMatrix, Tensor};
use std::sync::Arc;

pub struct Cfg {
    pub n_layer: usize,
    pub d: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// Head width on global blocks (`attention.key_length`).
    pub head_dim: usize,
    /// Head width on sliding blocks (`attention.key_length_swa`). Differs from `head_dim`.
    pub head_dim_swa: usize,
    /// Per-block: true = sliding window, false = global. Read from the ARRAY.
    pub swa: Vec<bool>,
    /// Per-block FFN width. An array on Gemma 4 (6144 for the first 15 blocks of E2B, 12288 after).
    pub n_ff: Vec<usize>,
    pub window: usize,
    /// Blocks `< kv_from_start` own K/V. The rest reuse an earlier block's cache.
    pub kv_from_start: usize,
    /// Width of the per-layer embedding (`embedding_length_per_layer_input`).
    pub ple: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub final_softcap: f32,
}

impl Cfg {
    /// Head width of block `il` — 512 or 256 on the same model.
    pub fn head_dim_at(&self, il: usize) -> usize {
        if self.swa[il] { self.head_dim_swa } else { self.head_dim }
    }

    /// Whether block `il` computes its own K/V.
    pub fn has_kv(&self, il: usize) -> bool { il < self.kv_from_start }

    /// Which block's KV cache block `il` reads.
    ///
    /// The reference rule is `kv_from_start - (is_swa ? 2 : 1)`, which lands on the most recent block
    /// of the SAME attention type — necessary, because the two types do not even share a head width.
    /// On E2B: sliding blocks read 13, global blocks read 14.
    pub fn kv_src(&self, il: usize) -> usize {
        if self.has_kv(il) { il } else { self.kv_from_start - if self.swa[il] { 2 } else { 1 } }
    }

    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let u = |k: &str| match md.get(&format!("gemma4.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing gemma4.{k}")) };
        let f = |k: &str| match md.get(&format!("gemma4.{k}")) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing gemma4.{k}")) };

        let n_layer = u("block_count")?;
        let d = u("embedding_length")?;

        // Both of these are ARRAYS. A scalar accessor returns Err on an array and would fall back to a
        // default, which is how a per-block schedule silently collapses into a uniform one.
        let swa: Vec<bool> = match md.get("gemma4.attention.sliding_window_pattern") {
            Some(Meta::Arr(a)) => a.iter().map(|m| matches!(m, Meta::U(v) if *v != 0) || matches!(m, Meta::I(v) if *v != 0) || matches!(m, Meta::Bool(true))).collect(),
            _ => return Err("gemma4.attention.sliding_window_pattern must be a per-block array".into()),
        };
        let n_ff: Vec<usize> = match md.get("gemma4.feed_forward_length") {
            Some(Meta::Arr(a)) => a.iter().map(|m| match m { Meta::U(v) => *v as usize, Meta::I(v) => *v as usize, _ => 0 }).collect(),
            Some(Meta::U(v)) => vec![*v as usize; n_layer],
            _ => return Err("no gemma4.feed_forward_length".into()),
        };
        if swa.len() != n_layer { return Err(format!("swa pattern covers {} of {n_layer} blocks", swa.len())); }
        if n_ff.len() != n_layer { return Err(format!("ffn widths cover {} of {n_layer} blocks", n_ff.len())); }

        let shared = u("attention.shared_kv_layers").unwrap_or(0);
        if shared >= n_layer { return Err(format!("shared_kv_layers {shared} >= block_count {n_layer}")); }
        let kv_from_start = n_layer - shared;
        // The reuse rule indexes `kv_from_start - 2`, so anything below 2 would underflow into a block
        // that does not exist.
        if shared > 0 && kv_from_start < 2 { return Err("need at least 2 KV-owning blocks".into()); }

        let n_vocab = g.tensor("token_embd.weight").ok_or("no token_embd")?.dims[1] as usize;

        Ok(Cfg {
            n_layer, d,
            n_head: u("attention.head_count")?,
            n_kv: u("attention.head_count_kv")?,
            n_vocab,
            eps: f("attention.layer_norm_rms_epsilon")?,
            head_dim: u("attention.key_length")?,
            head_dim_swa: u("attention.key_length_swa").unwrap_or_else(|_| u("attention.key_length").unwrap_or(0)),
            swa, n_ff,
            window: u("attention.sliding_window")?,
            kv_from_start,
            ple: u("embedding_length_per_layer_input").unwrap_or(0),
            rope_base: f("rope.freq_base")?,
            rope_base_swa: f("rope.freq_base_swa").unwrap_or_else(|_| f("rope.freq_base").unwrap_or(10000.0)),
            final_softcap: f("final_logit_softcapping").unwrap_or(0.0),
        })
    }
}

struct Block {
    attn_norm: Tensor,
    q: QMatrix,
    /// `None` on the shared-KV blocks. The file may still carry the weights; they are not used.
    k: Option<QMatrix>,
    v: Option<QMatrix>,
    o: QMatrix,
    q_norm: Tensor,
    k_norm: Option<Tensor>,
    attn_post_norm: Tensor,
    ffn_norm: Tensor,
    gate: QMatrix,
    up: QMatrix,
    down: QMatrix,
    ffn_post_norm: Tensor,
    /// Per-layer embedding path: gate into `ple` width, multiply by this block's slice, project back.
    inp_gate: QMatrix,
    proj: QMatrix,
    ple_post_norm: Tensor,
    /// A single learned scalar applied to the block output.
    out_scale: Option<Tensor>,
}

/// Decode state. Only the KV-owning blocks hold buffers; the rest read those.
pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
}

impl Cache {
    pub fn new(cfg: &Cfg) -> Cache {
        Cache { pos: 0, kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect() }
    }
}

pub struct Gemma4 {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    blocks: Vec<Block>,
    out_norm: Tensor,
    head: QMatrix,
    embd_ty: u32,
    embd_raw: Vec<u8>,
    /// Per-layer embedding table, `[n_vocab, ple * n_layer]`.
    ple_embd_ty: u32,
    ple_embd_raw: Vec<u8>,
    ple_model_proj: QMatrix,
    ple_proj_norm: Tensor,
    /// Proportional-RoPE factors for the GLOBAL blocks, stored as the multiplier Ferric's kernel
    /// wants (i.e. the reciprocal of ggml's divisors).
    rope_freqs: Option<Tensor>,
}

impl Gemma4 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Gemma4, String> {
        let cfg = Cfg::from_gguf(g)?;

        // A 26B-A4B / 31B checkpoint carries MoE blocks. Running it without them would quietly serve a
        // fraction of the model the file describes, which is worse than refusing.
        if g.tensor("blk.0.ffn_gate_inp.weight").is_some() || g.tensor("blk.15.ffn_gate_inp.weight").is_some() {
            return Err("this checkpoint has MoE blocks (ffn_gate_inp); only the dense E2B/E4B path is implemented".into());
        }

        let qm = |name: &str| -> Result<QMatrix, String> {
            let t = g.tensor(name).ok_or_else(|| format!("missing {name}"))?;
            let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
            if QMatrix::block_bytes(ty).is_some() {
                QMatrix::from_bytes(ctx, &g.raw(name)?, ty, rows, cols)
            } else {
                Ok(QMatrix::from_dense(ctx, &g.dequant(name)?, rows, cols))
            }
        };
        let ft = |name: &str, shape: &[usize]| -> Result<Tensor, String> {
            Ok(Tensor::from_vec(ctx, &g.dequant(name)?, shape))
        };

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let has_kv = cfg.has_kv(il);
            let hd = cfg.head_dim_at(il);
            blocks.push(Block {
                attn_norm: ft(&b("attn_norm.weight"), &[cfg.d])?,
                q: qm(&b("attn_q.weight"))?,
                // Load K/V only where the graph uses them. The dead copies in blocks 15.. would load
                // fine and cost memory for nothing.
                k: if has_kv { Some(qm(&b("attn_k.weight"))?) } else { None },
                v: if has_kv { Some(qm(&b("attn_v.weight"))?) } else { None },
                o: qm(&b("attn_output.weight"))?,
                q_norm: ft(&b("attn_q_norm.weight"), &[hd])?,
                k_norm: if has_kv { Some(ft(&b("attn_k_norm.weight"), &[hd])?) } else { None },
                attn_post_norm: ft(&b("post_attention_norm.weight"), &[cfg.d])?,
                ffn_norm: ft(&b("ffn_norm.weight"), &[cfg.d])?,
                gate: qm(&b("ffn_gate.weight"))?,
                up: qm(&b("ffn_up.weight"))?,
                down: qm(&b("ffn_down.weight"))?,
                ffn_post_norm: ft(&b("post_ffw_norm.weight"), &[cfg.d])?,
                inp_gate: qm(&b("inp_gate.weight"))?,
                proj: qm(&b("proj.weight"))?,
                ple_post_norm: ft(&b("post_norm.weight"), &[cfg.d])?,
                out_scale: ft(&b("layer_output_scale.weight"), &[1]).ok(),
            });
        }

        let ple_tok = g.tensor("per_layer_token_embd.weight").ok_or("no per_layer_token_embd")?;

        Ok(Gemma4 {
            ctx: ctx.clone(),
            blocks,
            out_norm: ft("output_norm.weight", &[cfg.d])?,
            // Gemma ties the head to the embedding table unless an explicit `output.weight` exists.
            head: if g.tensor("output.weight").is_some() { qm("output.weight")? } else { qm("token_embd.weight")? },
            embd_ty: g.tensor("token_embd.weight").ok_or("no token_embd")?.ggml_type,
            embd_raw: g.raw("token_embd.weight")?,
            ple_embd_ty: ple_tok.ggml_type,
            ple_embd_raw: g.raw("per_layer_token_embd.weight")?,
            ple_model_proj: qm("per_layer_model_proj.weight")?,
            ple_proj_norm: ft("per_layer_proj_norm.weight", &[cfg.ple])?,
            // See qwen3.rs: ggml applies these as `theta / ff`, Ferric's kernel multiplies the inverse
            // frequency, so they invert on the way in.
            rope_freqs: match g.tensor("rope_freqs.weight") {
                Some(t) => {
                    let n = t.dims[0] as usize;
                    let f = g.dequant("rope_freqs.weight")?;
                    let inv: Vec<f32> = f[..n].iter().map(|&x| if x != 0.0 { 1.0 / x } else { 1.0 }).collect();
                    Some(Tensor::from_vec(ctx, &inv, &[n]))
                }
                None => None,
            },
            cfg,
        })
    }

    /// Gather rows from a raw quantised table.
    fn gather(&self, raw: &[u8], ty: u32, width: usize, tokens: &[u32]) -> Tensor {
        let row = ferric_gguf::type_size(ty, width).expect("row size");
        let mut v = Vec::with_capacity(tokens.len() * width);
        for &t in tokens {
            let o = t as usize * row;
            v.extend(ferric_gguf::deq_raw(&raw[o..o + row], width, ty).expect("row"));
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), width])
    }

    /// Build the per-layer inputs: `[t, n_layer, ple]`.
    ///
    /// Mirrors `build_inp_per_layer` + `project_per_layer_inputs`. The token side is scaled by
    /// `sqrt(ple)`, the projected side by `1/sqrt(d)` and normed, and their sum by `1/sqrt(2)`.
    fn per_layer_inputs(&self, tokens: &[u32], scaled_embd: &Tensor) -> Tensor {
        let (t, nl, ple) = (tokens.len(), self.cfg.n_layer, self.cfg.ple);
        let wide = ple * nl;

        let from_tok = self
            .gather(&self.ple_embd_raw, self.ple_embd_ty, wide, tokens);
        let from_tok = from_tok.mul(&from_tok.scalar((ple as f32).sqrt()));

        // [t, d] · [d, ple*n_layer] -> [t, ple*n_layer], then normed over the ple axis.
        let proj = scaled_embd
            .matmul_q(&self.ple_model_proj);
        let proj = proj.mul(&proj.scalar(1.0 / (self.cfg.d as f32).sqrt()))
            .reshape(&[t * nl, ple])
            .rmsnorm(&self.ple_proj_norm, self.cfg.eps)
            .reshape(&[t, wide]);

        let sum = proj.add(&from_tok);
        sum.mul(&sum.scalar(1.0 / 2f32.sqrt())).reshape(&[t, nl, ple])
    }

    /// Logits for `tokens`, carrying KV in `cache`.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        let h = self.forward_hidden_cached(tokens, cache);
        let logits = h.matmul_q(&self.head);
        if self.cfg.final_softcap > 0.0 { logits.softcap(self.cfg.final_softcap) } else { logits }
    }

    /// Final normed hidden state, before the head.
    pub fn forward_hidden_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        let cfg = &self.cfg;
        let (d, eps, t, pos) = (cfg.d, cfg.eps, tokens.len(), cache.pos);

        // Token embeddings are scaled by sqrt(d) BEFORE anything else, including the per-layer
        // projection, which reads this scaled value.
        let mut x = self
            .gather(&self.embd_raw, self.embd_ty, d, tokens);
        let mut x = x.mul(&x.scalar((d as f32).sqrt()));

        let per_layer = (cfg.ple > 0).then(|| self.per_layer_inputs(tokens, &x));

        for (il, blk) in self.blocks.iter().enumerate() {
            let hd = cfg.head_dim_at(il);
            let is_swa = cfg.swa[il];
            let base = if is_swa { cfg.rope_base_swa } else { cfg.rope_base };
            let h = x.rmsnorm(&blk.attn_norm, eps);

            let rope = |v: Tensor, heads: usize| -> Tensor {
                match (&self.rope_freqs, is_swa) {
                    // Proportional RoPE applies to the GLOBAL blocks only.
                    (Some(f), false) => v.rope_scaled(f, heads, hd, base, pos),
                    _ => v.rope(heads, hd, base, pos),
                }
            };

            // Q is projected on every block, KV-owning or not.
            let q = rope(
                h.matmul_q(&blk.q)
                    .reshape(&[t * cfg.n_head, hd])
                    .rmsnorm(&blk.q_norm, eps)
                    .reshape(&[t, cfg.n_head * hd]),
                cfg.n_head,
            );
            // f_attention_scale = 1.0: Gemma 4 applies NO 1/sqrt(head_dim). The shared attention
            // kernels bake that factor in, so pre-multiplying Q by sqrt(hd) cancels it exactly.
            let q = q.mul(&q.scalar((hd as f32).sqrt()));

            if cfg.has_kv(il) {
                let k = rope(
                    h.matmul_q(blk.k.as_ref().expect("kv block has k"))
                        .reshape(&[t * cfg.n_kv, hd])
                        .rmsnorm(blk.k_norm.as_ref().expect("kv block has k_norm"), eps)
                        .reshape(&[t, cfg.n_kv * hd]),
                    cfg.n_kv,
                );
                // V gets a WEIGHTLESS RMS norm. No tensor exists to hint at this; it is only in the
                // reference graph, and omitting it changes every value without erroring.
                let v = h
                    .matmul_q(blk.v.as_ref().expect("kv block has v"))
                    .reshape(&[t * cfg.n_kv, hd])
                    .rmsnorm_weightless(eps)
                    .reshape(&[t, cfg.n_kv * hd]);
                let e = &mut cache.kv[il];
                append2(&self.ctx, &mut e.0, &k, &mut e.1, &v);
            }

            // Shared blocks read an earlier block's cache, which this same pass has already filled
            // (kv_src(il) < kv_from_start <= il).
            let src = cfg.kv_src(il);
            let (kc, vc) = (cache.kv[src].0.view(&self.ctx), cache.kv[src].1.view(&self.ctx));

            let window = if is_swa { cfg.window } else { 0 };
            let att = if t == 1 {
                nn::decode_attention_win(&q, &kc, &vc, cfg.n_head, cfg.n_kv, window, 0.0)
            } else {
                nn::causal_attention_win(&q, &kc, &vc, cfg.n_head, cfg.n_kv, window, 0.0)
            };

            let attn_out = x.add(&att.matmul_q(&blk.o).rmsnorm(&blk.attn_post_norm, eps));

            // GELU-gated FFN, and a post-norm INSIDE the residual.
            let f = attn_out.rmsnorm(&blk.ffn_norm, eps);
            let ffn = f.matmul_q(&blk.gate).gelu().mul(&f.matmul_q(&blk.up)).matmul_q(&blk.down);
            x = attn_out.add(&ffn.rmsnorm(&blk.ffn_post_norm, eps));

            // Per-layer embedding, gated by this block's slice and added back.
            if let Some(pl) = &per_layer {
                let slice = pl.narrow(1, il, 1).contiguous().reshape(&[t, cfg.ple]);
                let gated = x.matmul_q(&blk.inp_gate).gelu().mul(&slice);
                x = x.add(&gated.matmul_q(&blk.proj).rmsnorm(&blk.ple_post_norm, eps));
            }

            if let Some(s) = &blk.out_scale {
                x = x.mul(&s.broadcast_to(&[t, d]));
            }
        }

        cache.pos += t;
        x.rmsnorm(&self.out_norm, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_layer: usize, shared: usize, pattern: &[bool]) -> Cfg {
        Cfg {
            n_layer, d: 1536, n_head: 8, n_kv: 1, n_vocab: 262144, eps: 1e-6,
            head_dim: 512, head_dim_swa: 256,
            swa: pattern.to_vec(),
            n_ff: vec![6144; n_layer],
            window: 512,
            kv_from_start: n_layer - shared,
            ple: 256,
            rope_base: 1e6, rope_base_swa: 1e4, final_softcap: 30.0,
        }
    }

    /// E2B's real schedule: 35 blocks, 4 sliding then 1 global, 20 shared.
    fn e2b() -> Cfg {
        let pat: Vec<bool> = (0..35).map(|i| i % 5 != 4).collect();
        cfg(35, 20, &pat)
    }

    #[test]
    fn shared_blocks_read_the_last_kv_block_of_their_own_attention_type() {
        // The rule is `kv_from_start - (swa ? 2 : 1)`. Getting it backwards points a sliding block at
        // a global block's cache — which does not even have the same head width, so it would be wrong
        // in a way that only shows up as bad text.
        let c = e2b();
        assert_eq!(c.kv_from_start, 15);
        for il in 0..15 { assert!(c.has_kv(il), "block {il} should own KV"); assert_eq!(c.kv_src(il), il); }
        for il in 15..35 {
            assert!(!c.has_kv(il));
            let src = c.kv_src(il);
            assert!(src < c.kv_from_start, "block {il} reads {src}, which is not a KV-owning block");
            assert_eq!(c.swa[src], c.swa[il], "block {il} reads a cache of the other attention type");
            assert_eq!(c.head_dim_at(src), c.head_dim_at(il), "head width mismatch at block {il}");
        }
        assert_eq!(c.kv_src(15), 13, "sliding blocks read 13");
        assert_eq!(c.kv_src(19), 14, "block 19 is global and reads 14");
    }

    #[test]
    fn the_two_head_widths_follow_the_swa_pattern() {
        let c = e2b();
        assert_eq!(c.head_dim_at(0), 256, "block 0 is sliding");
        assert_eq!(c.head_dim_at(4), 512, "block 4 is global");
        assert!(c.swa[13] && !c.swa[14], "the reuse rule depends on 13 sliding and 14 global");
    }

    #[test]
    fn a_shared_source_block_is_always_filled_before_it_is_read() {
        // The forward pass fills caches in block order, so every reader must point strictly backwards
        // into the KV-owning prefix. A rule that pointed forward would read an empty buffer on the
        // first token and a stale one afterwards, with no error either time.
        let c = e2b();
        for il in 0..c.n_layer {
            assert!(c.kv_src(il) <= il, "block {il} reads block {} — forwards", c.kv_src(il));
        }
    }

    #[test]
    fn too_few_kv_blocks_is_refused_rather_than_underflowing() {
        // `kv_from_start - 2` underflows when the model claims to share nearly everything. That is a
        // panic in release and an index into nothing in the worst case.
        let g = Cfg::from_gguf_parts(4, 4);
        assert!(g.is_err(), "shared == n_layer must be refused");
        let g = Cfg::from_gguf_parts(4, 3);
        assert!(g.is_err(), "leaving 1 KV block must be refused: the rule indexes kv_from_start-2");
        assert!(Cfg::from_gguf_parts(4, 2).is_ok());
    }

    impl Cfg {
        /// Just the two checks that guard the reuse arithmetic, factored out so they can be tested
        /// without a GGUF.
        fn from_gguf_parts(n_layer: usize, shared: usize) -> Result<usize, String> {
            if shared >= n_layer { return Err(format!("shared_kv_layers {shared} >= block_count {n_layer}")); }
            let kv_from_start = n_layer - shared;
            if shared > 0 && kv_from_start < 2 { return Err("need at least 2 KV-owning blocks".into()); }
            Ok(kv_from_start)
        }
    }

    #[test]
    fn per_block_ffn_widths_are_not_assumed_uniform() {
        // E2B's real array is 6144 for the first 15 blocks and 12288 for the rest. A scalar read would
        // Err on the array and fall back to a default, building every block at one width.
        let mut c = e2b();
        c.n_ff = (0..35).map(|i| if i < 15 { 6144 } else { 12288 }).collect();
        assert_eq!(c.n_ff[14], 6144);
        assert_eq!(c.n_ff[15], 12288);
        assert_ne!(c.n_ff[0], c.n_ff[34], "the widths differ; a uniform read would hide it");
    }
}
