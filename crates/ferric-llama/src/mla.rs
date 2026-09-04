//! **Multi-head Latent Attention** (MLA) — DeepSeek-V3 style, with the AMD Instella gate as an option.
//!
//! MLA replaces the per-head K/V projections with a low-rank *latent*: one `kv_lora_rank`-wide vector per
//! position, up-projected to per-head keys and values on use. The decoupled RoPE dimensions live outside
//! that compression and are **shared across all heads** (one vector per position, not per head), which is
//! what makes the cache small.
//!
//! Why this belongs in `src/` rather than in an example: MLA's whole reason for existing is that it
//! shrinks the KV cache, which makes it the attention a memory-tiered engine wants — so `ferric-tier`
//! needs it as a library API, not as example code. It was verified layer-exact against AMD's real
//! `MLAGatedAttention` module before promotion, and `examples/instella_gmla.rs` still runs that same
//! comparison **through this code**, so the check is on the shipped path rather than on a copy of it.
//!
//! ## What is here, and what is not
//!
//! Here: the prefill forward, exactly as verified — **maxΔ 5.96e-7** against AMD's real module, measured
//! through this code by `examples/instella_gmla.rs` — plus the cache-footprint arithmetic a budget
//! planner needs.
//!
//! **Not here: cached incremental decode.** The reference comparison covers a full-sequence forward, so a
//! decode path would be unverified code wearing a verified module's name. It is a deliberate omission,
//! not an oversight — see [`CachePolicy`] for the design decision it will have to make first.

use ferric_tensor::{nn, Tensor};

/// Shapes and constants for one MLA block.
#[derive(Debug, Clone, Copy)]
pub struct MlaConfig {
    pub n_heads: usize,
    /// Non-positional part of each query/key head.
    pub qk_nope_dim: usize,
    /// RoPE part of each query/key head. **Shared across heads on the key side.**
    pub qk_rope_dim: usize,
    /// Value head width. May differ from the query/key head width.
    pub v_head_dim: usize,
    /// Width of the compressed KV latent.
    pub kv_lora_rank: usize,
    /// Attention scale. Not `1/sqrt(head_dim)` in general — DeepSeek folds a YaRN `mscale` into it, so it
    /// is taken from the checkpoint rather than derived. Deriving it silently changes the model.
    pub scaling: f32,
    pub eps: f32,
    /// `true` for the HuggingFace `apply_rotary_pos_emb_interleave` convention, where a head's RoPE lanes
    /// are stored interleaved `(a0,b0,a1,b1,...)` and must be de-interleaved to `(a0,a1,...,b0,b1,...)`
    /// before split-half RoPE. Getting this wrong produces right values in wrong places: every norm and
    /// every summary statistic still looks correct, and the model is simply worse.
    pub rope_interleaved: bool,
}

impl MlaConfig {
    /// Full query/key head width.
    pub fn qk_head_dim(&self) -> usize { self.qk_nope_dim + self.qk_rope_dim }

    /// Floats an **uncompressed** attention would cache per position: full per-head K and V.
    pub fn dense_cache_floats(&self) -> usize {
        self.n_heads * (self.qk_head_dim() + self.v_head_dim)
    }

    /// Floats MLA caches per position under [`CachePolicy::Latent`]: the latent plus the single shared
    /// RoPE vector.
    pub fn latent_cache_floats(&self) -> usize { self.kv_lora_rank + self.qk_rope_dim }

    /// Floats MLA caches per position under [`CachePolicy::Expanded`]: per-head nope-key and value.
    pub fn expanded_cache_floats(&self) -> usize {
        self.n_heads * (self.qk_nope_dim + self.v_head_dim) + self.qk_rope_dim
    }

    /// How much smaller the latent cache is than a dense one. This is the number MLA exists for — on
    /// GLM-5.2 shapes (64 heads, 192+64 qk, 256 v, 512 latent) it is 56.9x.
    pub fn latent_compression(&self) -> f64 {
        self.dense_cache_floats() as f64 / self.latent_cache_floats() as f64
    }
}

/// Where an incremental-decode implementation should keep its KV, and the tradeoff it is choosing.
///
/// Frontier engines split on this, and both choices are defensible:
///
/// - [`CachePolicy::Latent`] stores `kv_lora_rank + qk_rope_dim` floats per position and re-derives
///   per-head K/V via *weight absorption* — folding `kv_b_proj` into the query and output projections so
///   attention runs against the latent directly, at `O(T · kv_lora_rank)` instead of `O(T · H · nope)`.
///   Smallest cache by a large factor.
/// - [`CachePolicy::Expanded`] stores the already-expanded per-head keys and values. Larger by roughly
///   `n_heads · (nope + v) / (latent + rope)`, but no up-projection per cached position per step.
///
/// The choice is a genuine memory/compute trade and depends on context length and on whether the engine
/// is streaming weights. It is recorded here as an enum rather than baked in, because a decode path that
/// silently picks one has picked a context-length ceiling for its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    Latent,
    Expanded,
}

impl CachePolicy {
    /// Bytes per position per layer at f32.
    pub fn bytes_per_position(&self, cfg: &MlaConfig) -> usize {
        4 * match self {
            CachePolicy::Latent => cfg.latent_cache_floats(),
            CachePolicy::Expanded => cfg.expanded_cache_floats(),
        }
    }
}

/// How the query is produced. Checkpoints differ, and the difference is not a flag on one code path:
/// the low-rank form has a norm in the middle, so it cannot be folded into a single matrix.
pub enum QProj {
    /// Instella keeps it whole. `[n_heads * qk_head_dim, hidden]`.
    Whole(Tensor),
    /// DeepSeek-V3 and hyv4 factor it through a LoRA with a norm between the halves:
    /// `hidden → q_lora_rank → RMSNorm(·)·gain → n_heads * qk_head_dim`.
    ///
    /// `a` is `[q_lora_rank, hidden]`, `a_norm` is `[q_lora_rank]`, `b` is
    /// `[n_heads * qk_head_dim, q_lora_rank]`.
    LowRank { a: Tensor, a_norm: Tensor, b: Tensor },
}

/// How the latent becomes keys and values — and this one is not a storage detail, it is **a
/// different attention**.
///
/// [`KvUp::Fused`] decompresses the latent to per-head keys and values and attends over
/// `qk_head_dim`. [`KvUp::Absorbed`] never materialises them: it folds `W_K` into the query, attends
/// directly against the `kv_lora_rank + qk_rope_dim` latent row, and decompresses only the
/// *output*. Both compute the same function — `q_nope · (W_K c) = (W_Kᵀ q_nope) · c`, and
/// `Σⱼ Pⱼ (W_V cⱼ) = W_V (Σⱼ Pⱼ cⱼ)` — which is why `absorbed_and_fused_agree` can use one as the
/// other's oracle without any reference implementation.
///
/// What differs is what has to be cached. Fused wants per-head K and V; absorbed wants only the
/// latent row, which is [`MlaConfig::latent_compression`] times smaller.
pub enum KvUp {
    /// `[n_heads * (qk_nope_dim + v_head_dim), kv_lora_rank]` — one matrix producing both.
    Fused(Tensor),
    /// Per-head decompressors, stored exactly as GGUF lays them out.
    ///
    /// `k_b` is `attn_k_b`, GGUF `ne = [qk_nope_dim, kv_lora_rank, n_heads]`, i.e.
    /// `[n_heads, kv_lora_rank, qk_nope_dim]` here. `v_b` is `attn_v_b`, GGUF
    /// `ne = [kv_lora_rank, v_head_dim, n_heads]`, i.e. `[n_heads, v_head_dim, kv_lora_rank]`.
    ///
    /// ⚠ They are NOT the same orientation as each other. `k_b` contracts its last axis against the
    /// query's nope part; `v_b` contracts its last axis against the latent attention output. Storing
    /// both "the obvious way" transposes one of them, and a transposed square-ish factor produces
    /// fluent output.
    Absorbed { k_b: Tensor, v_b: Tensor },
}

impl KvUp {
    /// Rewrite a [`KvUp::Fused`] decompressor into the [`KvUp::Absorbed`] pair it is equivalent to.
    ///
    /// The two forms hold the same numbers in a different arrangement, so this is a repacking and
    /// not an approximation — and it is worth doing at load time, because absorbing is what lets the
    /// KV cache hold `kv_lora_rank + qk_rope_dim` floats per position instead of
    /// `n_heads · (qk_nope_dim + v_head_dim)`. On DeepSeek-V3 shapes that is the difference between
    /// a context that fits and one that does not.
    ///
    /// A checkpoint that already ships `attn_k_b` / `attn_v_b` needs none of this; it is for the
    /// ones that only ship the fused matrix.
    pub fn absorb(ctx: &std::sync::Arc<ferric_core::Context>, kv_b: &Tensor, cfg: &MlaConfig) -> KvUp {
        let (h, nope, vh, kvl) = (cfg.n_heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        assert_eq!(kv_b.shape, vec![h * (nope + vh), kvl],
                   "fused kv_b must be [n_heads*(nope+v), kv_lora_rank]");
        let w = pollster::block_on(kv_b.to_vec());
        let stride = nope + vh;
        let mut kv = vec![0.0f32; h * kvl * nope];
        let mut vv = vec![0.0f32; h * vh * kvl];
        for hh in 0..h {
            for e in 0..nope {
                for r in 0..kvl { kv[(hh * kvl + r) * nope + e] = w[(hh * stride + e) * kvl + r] }
            }
            for v in 0..vh {
                for r in 0..kvl { vv[(hh * vh + v) * kvl + r] = w[(hh * stride + nope + v) * kvl + r] }
            }
        }
        KvUp::Absorbed {
            k_b: Tensor::from_vec(ctx, &kv, &[h, kvl, nope]),
            v_b: Tensor::from_vec(ctx, &vv, &[h, vh, kvl]),
        }
    }
}

/// Projection weights for one MLA block. Row-major `[out, in]`, consumed with `matmul_bt`.
pub struct MlaWeights {
    /// Query projection — whole, or factored through a LoRA. See [`QProj`].
    pub q: QProj,
    /// `[kv_lora_rank + qk_rope_dim, hidden]` — emits the latent and the shared RoPE key together.
    pub kv_a_proj_with_mqa: Tensor,
    /// RMSNorm gain over the latent **only**; it never touches the RoPE lanes.
    pub kv_a_layernorm: Tensor,
    /// How the latent becomes K and V. See [`KvUp`].
    pub kv_up: KvUp,
    /// `[hidden, n_heads * v_head_dim]`.
    pub o_proj: Tensor,
    /// One learnable softmax sink per head, `[n_heads]`, raw. `None` is ordinary attention, and that
    /// is the path verified against AMD's module — a `Some` here takes a different softmax.
    pub sinks: Option<Tensor>,
    /// AMD Instella's addition: `attn_out * sigmoid(gate_proj(x))` applied **before** `o_proj`.
    ///
    /// The order matters and differs between architectures — Kimi's KDA norms first and then gates, MLA
    /// gates without a norm. Sharing one code path between them is wrong.
    pub gate_proj: Option<Tensor>,
}

/// One MLA block.
pub struct Mla {
    pub cfg: MlaConfig,
    pub w: MlaWeights,
}

impl Mla {
    pub fn new(cfg: MlaConfig, w: MlaWeights) -> Self { Self { cfg, w } }

    /// De-interleave `(a0,b0,a1,b1,...) -> (a0,a1,...,b0,b1,...)` so split-half RoPE with a doubled
    /// cos/sin table reproduces HuggingFace's interleaved convention.
    fn deinterleave(&self, x: &Tensor, rows: usize) -> Tensor {
        let r = self.cfg.qk_rope_dim;
        if !self.cfg.rope_interleaved {
            return x.clone();
        }
        x.reshape(&[rows, r / 2, 2]).transpose(1, 2).contiguous().reshape(&[rows, r])
    }

    /// Full-sequence forward. `hs` is `[seq, hidden]`; `cos`/`sin` are doubled RoPE tables `[seq, rope]`.
    ///
    /// Returns `[seq, hidden]`.
    pub fn forward(&self, hs: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let (q_nope, q_rot, latent, k_rot) = self.project(hs, cos, sin);
        let ao = match &self.w.kv_up {
            KvUp::Fused(kv_b) => self.attend_fused(&q_nope, &q_rot, &latent, &k_rot, kv_b),
            KvUp::Absorbed { k_b, v_b } => self.attend_absorbed(&q_nope, &q_rot, &latent, &k_rot, k_b, v_b),
        };

        // --- optional gate, then output projection ---
        let ao = match &self.w.gate_proj {
            Some(g) => ao.mul(&hs.matmul_bt(g).sigmoid()),
            None => ao,
        };
        ao.matmul_bt(&self.w.o_proj)
    }

    /// Everything both paths share: the query, the compressed latent, and the one shared RoPE key.
    ///
    /// Returns `(q_nope [s,h,nope], q_rot [s,h,rope], latent [s,kv_lora] (normed), k_rot [s,rope])`.
    fn project(&self, hs: &Tensor, cos: &Tensor, sin: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        let c = &self.cfg;
        let (h, nope, rope, kvl) = (c.n_heads, c.qk_nope_dim, c.qk_rope_dim, c.kv_lora_rank);
        let qk = c.qk_head_dim();
        let s = hs.shape[0];

        // --- Q: whole, or down → norm → up. The norm is why these cannot be one matrix. ---
        let qf = match &self.w.q {
            QProj::Whole(w) => hs.matmul_bt(w),
            QProj::LowRank { a, a_norm, b } => hs.matmul_bt(a).rmsnorm(a_norm, c.eps).matmul_bt(b),
        };
        let q = qf.reshape(&[s, h, qk]);
        let q_pass = q.narrow(2, 0, nope).contiguous();
        let q_rot = self
            .deinterleave(&q.narrow(2, nope, rope).contiguous(), s * h)
            .reshape(&[s, h * rope])
            .apply_rope_costable(cos, sin, h, rope)
            .reshape(&[s, h, rope]);

        // --- KV: one projection emits the latent and the shared RoPE key side by side ---
        let ckv = hs.matmul_bt(&self.w.kv_a_proj_with_mqa);
        // ⚠ The norm covers the latent ONLY. The RoPE lanes sit in the same tensor and are not
        // normed — including them keeps every shape and quietly rescales the positional signal.
        let latent = ckv.narrow(1, 0, kvl).contiguous().rmsnorm(&self.w.kv_a_layernorm, c.eps);
        let k_rot = self
            .deinterleave(&ckv.narrow(1, kvl, rope).contiguous(), s)
            .apply_rope_costable(cos, sin, 1, rope);
        (q_pass, q_rot, latent, k_rot)
    }

    /// **Fused (expanded) path.** Decompress the latent into per-head keys and values, then attend
    /// over `qk_head_dim`. This is the path verified layer-exact against AMD's module.
    fn attend_fused(&self, q_pass: &Tensor, q_rot: &Tensor, latent: &Tensor, k_rot: &Tensor, kv_b: &Tensor) -> Tensor {
        let c = &self.cfg;
        let (h, nope, rope, vh) = (c.n_heads, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim);
        let qk = c.qk_head_dim();
        let s = q_pass.shape[0];

        let kb = latent.matmul_bt(kv_b).reshape(&[s, h, nope + vh]);
        let k_nope = kb.narrow(2, 0, nope).contiguous();
        let value = kb.narrow(2, nope, vh).contiguous();

        // The RoPE key is ONE vector per position shared by every head — this is the decoupling that
        // keeps the cache small, and broadcasting it here is what makes that shape explicit.
        let k_rot = k_rot.reshape(&[s, 1, rope]).broadcast_to(&[s, h, rope]).contiguous();

        let qh = q_pass.cat(q_rot, 2).reshape(&[s, h * qk]);
        let kh = k_nope.cat(&k_rot, 2).reshape(&[s, h * qk]);
        let vv = value.reshape(&[s, h * vh]);
        // `causal_attention` divides by sqrt(head_dim) internally; pre-multiplying by that factor makes
        // the effective scale exactly `cfg.scaling`, which carries YaRN's mscale and is not derivable.
        let qh = qh.mul(&qh.scalar(c.scaling * (qk as f32).sqrt()));
        // ⚠ `causal_attention` derives ONE head width for q, k and v, so it silently requires
        // `v_head_dim == qk_head_dim`. Instella satisfies that (128 = 96+32) and this module was
        // written against Instella, so the constraint was invisible — but DeepSeek-V3 has qk 256 and
        // v 128, and `MlaConfig` has always documented that the two may differ. `_split` takes both
        // widths, and reduces to exactly the same arithmetic when they are equal.
        match &self.w.sinks {
            None => nn::causal_attention_split(&qh, &kh, &vv, h, qk, vh, 0.0),
            Some(sk) => nn::causal_attention_split_sinks(&qh, &kh, &vv, h, qk, vh, sk, 0.0),
        }
    }

    /// **Absorbed path.** Never materialise the per-head keys and values at all.
    ///
    /// `W_K` folds into the query — `q_nope · (W_K c) = (W_Kᵀ q_nope) · c` — so the score is a dot
    /// over `kv_lora_rank + qk_rope_dim` against the cached latent row itself. `W_V` folds out of the
    /// output — `Σⱼ Pⱼ (W_V cⱼ) = W_V (Σⱼ Pⱼ cⱼ)` — so the value side decompresses once per query
    /// instead of once per cached position.
    ///
    /// Both identities are exact, not approximations, which is the whole reason a frontier engine
    /// takes this path: it is the same function against a cache
    /// [`MlaConfig::latent_compression`]× smaller.
    ///
    /// ⚠ The score is a dot over `kv_lora_rank + qk_rope_dim` but the scale is still the one the
    /// checkpoint gives, which is set by `qk_head_dim`. Deriving `1/sqrt(width of the dot)` here is
    /// the natural thing to write and is a different model.
    fn attend_absorbed(&self, q_pass: &Tensor, q_rot: &Tensor, latent: &Tensor, k_rot: &Tensor,
                       k_b: &Tensor, v_b: &Tensor) -> Tensor {
        let c = &self.cfg;
        let (h, rope, vh, kvl) = (c.n_heads, c.qk_rope_dim, c.v_head_dim, c.kv_lora_rank);
        let s = q_pass.shape[0];
        let cw = kvl + rope; // the width the score is actually taken over

        // q_abs[h,t,r] = Σ_e k_b[h,r,e] · q_nope[h,t,e]
        let qh = q_pass.permute(&[1, 0, 2]).contiguous();                 // [h, s, nope]
        let q_abs = qh.matmul(&k_b.transpose(2, 1).contiguous());         // [h, s, kvl]
        let q_all = q_abs.cat(&q_rot.permute(&[1, 0, 2]).contiguous(), 2); // [h, s, cw]

        // One KV "head": the latent row plus the shared RoPE key, broadcast across query heads.
        let k_all = latent.cat(k_rot, 1).reshape(&[1, s, cw]).broadcast_to(&[h, s, cw]).contiguous();

        let scores = q_all.matmul(&k_all.transpose(2, 1)).mul(&q_all.scalar(c.scaling));
        let masked = scores.add(&nn::causal_mask_hw(&scores, h, s));
        let probs = match &self.w.sinks {
            None => masked.softmax(2),
            Some(sk) => nn::softmax_with_sinks(&masked, sk),
        };

        // Attend in the latent, THEN decompress once: o[h,t,v] = Σ_r v_b[h,v,r] · o_lat[h,t,r]
        let lat = latent.reshape(&[1, s, kvl]).broadcast_to(&[h, s, kvl]).contiguous();
        let o_lat = probs.matmul(&lat);                                   // [h, s, kvl]
        let o = o_lat.matmul(&v_b.transpose(2, 1).contiguous());          // [h, s, vh]
        o.permute(&[1, 0, 2]).contiguous().reshape(&[s, h * vh])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AMD Instella-MoE-16B-A3B.
    fn instella() -> MlaConfig {
        MlaConfig {
            n_heads: 16,
            qk_nope_dim: 96,
            qk_rope_dim: 32,
            v_head_dim: 128,
            kv_lora_rank: 512,
            scaling: 0.165_626_88,
            eps: 1e-6,
            rope_interleaved: true,
        }
    }

    /// GLM-5.2 / DeepSeek-V3 class shapes, where the compression actually pays.
    fn glm() -> MlaConfig {
        MlaConfig {
            n_heads: 64,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            kv_lora_rank: 512,
            scaling: 1.0 / (256f32).sqrt(),
            eps: 1e-5,
            rope_interleaved: false,
        }
    }

    #[test]
    fn latent_compression_matches_the_published_figure() {
        // colibri documents 56.9x on GLM-5.2, derived as
        //   dense  = 64*(192+64) + 64*256 = 32768 floats
        //   latent = 512 + 64             =   576 floats
        // If this drifts, a shape constant is wrong somewhere.
        let c = glm();
        assert_eq!(c.dense_cache_floats(), 32768);
        assert_eq!(c.latent_cache_floats(), 576);
        assert!((c.latent_compression() - 56.888).abs() < 0.01, "got {}", c.latent_compression());
    }

    /// Kimi K3: 96 heads, 128 nope, 128 value, 512 latent, 64 rope. Included because the widely-quoted
    /// "expanded KV is ~42x the latent" figure is K3's, and it is easy to mis-attribute to GLM.
    fn kimi_k3() -> MlaConfig {
        MlaConfig {
            n_heads: 96,
            qk_nope_dim: 128,
            qk_rope_dim: 64,
            v_head_dim: 128,
            kv_lora_rank: 512,
            scaling: 1.0 / (192f32).sqrt(),
            eps: 1e-5,
            rope_interleaved: false,
        }
    }

    #[test]
    fn expanded_caching_costs_far_more_than_latent() {
        // The trade CachePolicy exists to make explicit: caching expanded K/V avoids an up-projection per
        // cached position per step, and costs tens of times the memory. An engine that picks one silently
        // has picked a context-length ceiling for its caller.
        //
        // The ratio is SHAPE-SPECIFIC, and this test exists partly to stop it being quoted as a constant:
        // the familiar "~42x" belongs to Kimi K3, not to GLM, which is 49.9x. Both are pinned here.
        let g = glm();
        assert_eq!(CachePolicy::Latent.bytes_per_position(&g), 576 * 4);
        assert_eq!(CachePolicy::Expanded.bytes_per_position(&g), (64 * (192 + 256) + 64) * 4);
        let g_ratio = g.expanded_cache_floats() as f64 / g.latent_cache_floats() as f64;
        assert!((g_ratio - 49.89).abs() < 0.02, "GLM expanded/latent {g_ratio:.2}");

        let k = kimi_k3();
        let k_ratio = k.expanded_cache_floats() as f64 / k.latent_cache_floats() as f64;
        assert!((k_ratio - 42.78).abs() < 0.02, "K3 expanded/latent {k_ratio:.2}");
    }

    #[test]
    fn kimi_k3_kv_bytes_per_position_match_the_published_figure() {
        // K3 documents 2.37 MB/position across its 24 MLA layers under the EXPANDED policy — the check
        // that the shape constants and the policy arithmetic agree with a real shipped engine.
        let k = kimi_k3();
        let per_layer = CachePolicy::Expanded.bytes_per_position(&k) as f64;
        let across_mla_layers = per_layer * 24.0;
        assert!(
            (across_mla_layers - 2.37e6).abs() < 2.0e4,
            "expected ~2.37 MB/position across 24 MLA layers, got {:.3} MB",
            across_mla_layers / 1e6
        );
    }

    #[test]
    fn head_dim_is_nope_plus_rope_even_though_rope_is_unrotated_in_some_models() {
        // Kimi K3 lists this among the invariants that silently produce a different model: MLA there uses
        // NoPE, yet the rope dimensions still exist, are still concatenated, still scored, still cached.
        // Dropping them changes the head width and the scale.
        let c = instella();
        assert_eq!(c.qk_head_dim(), 128);
        let k = glm();
        assert_eq!(k.qk_head_dim(), 256);
    }

    #[test]
    fn instella_cache_is_smaller_than_dense_but_by_less_than_glm() {
        // Compression scales with head count, so a 16-head model gains far less than a 64-head one. Worth
        // pinning: it is the reason MLA is a frontier-model technique rather than a universal win.
        let c = instella();
        assert!(c.latent_compression() > 7.0 && c.latent_compression() < 8.0,
                "instella compression {:.2}", c.latent_compression());
        assert!(glm().latent_compression() > 7.0 * c.latent_compression());
    }
}

#[cfg(test)]
mod absorbed_tests {
    use super::*;
    use ferric_core::Context;
    use std::sync::Arc;

    const H: usize = 3;
    const NOPE: usize = 6;
    const ROPE: usize = 4;
    const VH: usize = 8;
    const KVL: usize = 10;
    const HID: usize = 12;
    const T: usize = 5;

    fn cfg() -> MlaConfig {
        MlaConfig {
            n_heads: H, qk_nope_dim: NOPE, qk_rope_dim: ROPE, v_head_dim: VH,
            kv_lora_rank: KVL, scaling: 0.1234, eps: 1e-5, rope_interleaved: false,
        }
    }

    macro_rules! ctx_or_skip {
        () => { match pollster::block_on(Context::new()) { Ok(c) => Arc::new(c), Err(_) => { eprintln!("no GPU context — skipping"); return } } };
    }

    fn rnd(ctx: &Arc<Context>, shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed;
        let v: Vec<f32> = (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // ⛔ `>> 33` here gave [-0.5, 0) -- every input negative -- until 2026-09-04. See dsa.rs::rnd.
            (((s >> 32) as f32 / (1u64 << 31) as f32) - 1.0) * 0.5
        }).collect();
        Tensor::from_vec(ctx, &v, shape)
    }

    fn weights(ctx: &Arc<Context>, kv_up: KvUp, sinks: Option<Tensor>) -> MlaWeights {
        MlaWeights {
            q: QProj::Whole(rnd(ctx, &[H * (NOPE + ROPE), HID], 1)),
            kv_a_proj_with_mqa: rnd(ctx, &[KVL + ROPE, HID], 2),
            kv_a_layernorm: rnd(ctx, &[KVL], 3),
            kv_up,
            o_proj: rnd(ctx, &[HID, H * VH], 4),
            gate_proj: None,
            sinks,
        }
    }

    /// ⛔ The generator is two-signed and spans its range. Until 2026-09-04 it was uniform in
    /// [-1, 0) -- a `>> 33` where `>> 32` was meant -- and every test in this module ran on
    /// negative-only inputs without anything noticing. A fixture needs a guard like any other claim.
    #[test]
    fn the_fixture_generator_is_two_signed() {
        let ctx = ctx_or_skip!();
        let v = pollster::block_on(rnd(&ctx, &[64, 16], 777).to_vec());
        let (mx, mn) = v.iter().fold((f32::MIN, f32::MAX), |(a, b), x| (a.max(*x), b.min(*x)));
        assert!(mx > 0.25 && mn < -0.25, "generator does not span both signs: max {mx}, min {mn}");
        assert!(v.iter().filter(|x| **x > 0.0).count() * 4 > v.len(), "fewer than a quarter of the draws are positive");
    }

    /// **Absorption is exact, so each path is the other's oracle.**
    ///
    /// `q_nope · (W_K c) = (W_Kᵀ q_nope) · c` and `Σⱼ Pⱼ (W_V cⱼ) = W_V (Σⱼ Pⱼ cⱼ)`. Both identities
    /// hold for any weights, so the fused path — which is verified layer-exact against AMD's real
    /// module at 5.96e-7 — pins the absorbed one without needing a second reference. A transposed
    /// `k_b`, a `v_b` folded on the wrong axis, a score scaled by `1/sqrt(kv_lora + rope)` instead of
    /// the checkpoint's own factor, or a norm applied to the RoPE lanes all break this while leaving
    /// every shape intact.
    ///
    /// ⛔ **What it cannot see, by construction:** anything in the shared prologue. `project()`
    /// produces the query, the latent and the RoPE key for both arms, so a wrong epsilon, a wrong
    /// RoPE base or a norm over the wrong slice moves both sides identically and the difference
    /// stays zero. Verified by mutation — changing `c.eps` by 100x leaves this test green. The
    /// prologue is pinned instead by `examples/instella_gmla.rs`, which is a comparison against a
    /// real reference module rather than against another arm of this file.
    #[test]
    fn absorbed_and_fused_agree() {
        let ctx = ctx_or_skip!();
        let c = cfg();
        let kv_b = rnd(&ctx, &[H * (NOPE + VH), KVL], 5);
        let (hs, cos, sin) = (rnd(&ctx, &[T, HID], 6), rnd(&ctx, &[T, ROPE], 7), rnd(&ctx, &[T, ROPE], 8));

        let fused = Mla::new(c, weights(&ctx, KvUp::Fused(kv_b.clone()), None));
        let absorbed = Mla::new(c, weights(&ctx, KvUp::absorb(&ctx, &kv_b, &c), None));

        let a = pollster::block_on(fused.forward(&hs, &cos, &sin).to_vec());
        let b = pollster::block_on(absorbed.forward(&hs, &cos, &sin).to_vec());
        assert_eq!(a.len(), T * HID);

        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-3, "the reference output is ~zero; this comparison would pass on anything");
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        eprintln!("absorbed vs fused: max |Δ| = {worst:.3e} on outputs of magnitude {scale:.3e}");
        assert!(worst < 2e-5 * scale.max(1.0), "absorbed path diverges from fused by {worst}");
    }

    /// The same equivalence must survive a sink, since the sink changes the softmax that sits between
    /// the two folds. If absorption were only valid for a normalised softmax this would catch it.
    #[test]
    fn absorbed_and_fused_agree_with_a_sink() {
        let ctx = ctx_or_skip!();
        let c = cfg();
        let kv_b = rnd(&ctx, &[H * (NOPE + VH), KVL], 15);
        let sk = Tensor::from_vec(&ctx, &[0.0, 1.5, -2.0], &[H]);
        let (hs, cos, sin) = (rnd(&ctx, &[T, HID], 16), rnd(&ctx, &[T, ROPE], 17), rnd(&ctx, &[T, ROPE], 18));

        let fused = Mla::new(c, weights(&ctx, KvUp::Fused(kv_b.clone()), Some(sk.clone())));
        let absorbed = Mla::new(c, weights(&ctx, KvUp::absorb(&ctx, &kv_b, &c), Some(sk)));
        let a = pollster::block_on(fused.forward(&hs, &cos, &sin).to_vec());
        let b = pollster::block_on(absorbed.forward(&hs, &cos, &sin).to_vec());
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-5 * scale.max(1.0), "sinked absorbed path diverges by {worst}");

        // And the sink must actually be doing something, or the test above is about nothing.
        let plain = Mla::new(c, weights(&ctx, KvUp::Fused(kv_b), None));
        let p = pollster::block_on(plain.forward(&hs, &cos, &sin).to_vec());
        let moved = a.iter().zip(&p).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-4, "the sink changed nothing ({moved}); this test is vacuous without it");
    }

    /// A Q-LoRA is a down-projection, a NORM, and an up-projection. The norm is the whole point: it
    /// is why the two matrices cannot be collapsed into one, and a port that drops it still produces
    /// a `[T, hidden]` tensor of plausible magnitude.
    #[test]
    fn the_q_lora_norm_is_not_optional() {
        let ctx = ctx_or_skip!();
        let c = cfg();
        let rq = 7usize;
        let kv_b = rnd(&ctx, &[H * (NOPE + VH), KVL], 25);
        let (hs, cos, sin) = (rnd(&ctx, &[T, HID], 26), rnd(&ctx, &[T, ROPE], 27), rnd(&ctx, &[T, ROPE], 28));
        let (a, b) = (rnd(&ctx, &[rq, HID], 29), rnd(&ctx, &[H * (NOPE + ROPE), rq], 30));

        let mk = |gain: Vec<f32>| {
            let mut w = weights(&ctx, KvUp::Fused(kv_b.clone()), None);
            w.q = QProj::LowRank { a: a.clone(), a_norm: Tensor::from_vec(&ctx, &gain, &[rq]), b: b.clone() };
            pollster::block_on(Mla::new(c, w).forward(&hs, &cos, &sin).to_vec())
        };
        let ones = mk(vec![1.0; rq]);
        let twos = mk(vec![2.0; rq]);
        let far = ones.iter().zip(&twos).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(far > 1e-4, "the q_a_norm gain changed nothing ({far}) — it is being skipped");
        assert_eq!(ones.len(), T * HID);
    }

    /// The number the absorbed path exists for.
    #[test]
    fn absorbing_is_what_makes_the_cache_small() {
        let ds = MlaConfig { n_heads: 64, qk_nope_dim: 192, qk_rope_dim: 64, v_head_dim: 256,
                             kv_lora_rank: 512, scaling: 0.0625, eps: 1e-5, rope_interleaved: false };
        assert_eq!(ds.latent_cache_floats(), 576);
        assert_eq!(ds.dense_cache_floats(), 64 * (256 + 256));
        assert!((ds.latent_compression() - 56.888).abs() < 0.01,
                "hyv4/GLM shapes give 56.9x, got {}", ds.latent_compression());
        // The fused path cannot reach that number: it has to hold per-head nope keys and values.
        assert_eq!(ds.expanded_cache_floats(), 64 * (192 + 256) + 64);
        assert!(ds.expanded_cache_floats() > 40 * ds.latent_cache_floats());
    }
}
