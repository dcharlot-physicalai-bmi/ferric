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

/// Projection weights for one MLA block. Row-major `[out, in]`, consumed with `matmul_bt`.
pub struct MlaWeights {
    /// Query projection. `[n_heads * qk_head_dim, hidden]`.
    ///
    /// DeepSeek factors this into `q_a_proj` / `q_a_layernorm` / `q_b_proj`; Instella keeps it whole.
    /// Only the whole form is verified here, so only the whole form is offered — a low-rank variant
    /// should be added when there is a reference to check it against.
    pub q_proj: Tensor,
    /// `[kv_lora_rank + qk_rope_dim, hidden]` — emits the latent and the shared RoPE key together.
    pub kv_a_proj_with_mqa: Tensor,
    /// RMSNorm gain over the latent **only**; it never touches the RoPE lanes.
    pub kv_a_layernorm: Tensor,
    /// `[n_heads * (qk_nope_dim + v_head_dim), kv_lora_rank]`.
    pub kv_b_proj: Tensor,
    /// `[hidden, n_heads * v_head_dim]`.
    pub o_proj: Tensor,
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
        let c = &self.cfg;
        let (h, nope, rope, vh, kvl) =
            (c.n_heads, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim, c.kv_lora_rank);
        let qk = c.qk_head_dim();
        let s = hs.shape[0];

        // --- Q: project, split nope/rope, rotate only the rope part ---
        let q = hs.matmul_bt(&self.w.q_proj).reshape(&[s, h, qk]);
        let q_pass = q.narrow(2, 0, nope).contiguous();
        let q_rot = self
            .deinterleave(&q.narrow(2, nope, rope).contiguous(), s * h)
            .reshape(&[s, h * rope])
            .apply_rope_costable(cos, sin, h, rope)
            .reshape(&[s, h, rope]);

        // --- KV: compress -> norm -> up-project -> split nope/value ---
        let ckv = hs.matmul_bt(&self.w.kv_a_proj_with_mqa);
        let latent = ckv.narrow(1, 0, kvl).contiguous();
        let k_rot = ckv.narrow(1, kvl, rope).contiguous();
        let kb = latent
            .rmsnorm(&self.w.kv_a_layernorm, c.eps)
            .matmul_bt(&self.w.kv_b_proj)
            .reshape(&[s, h, nope + vh]);
        let k_nope = kb.narrow(2, 0, nope).contiguous();
        let value = kb.narrow(2, nope, vh).contiguous();

        // The RoPE key is ONE vector per position shared by every head — this is the decoupling that
        // keeps the cache small, and broadcasting it here is what makes that shape explicit.
        let k_rot = self
            .deinterleave(&k_rot, s)
            .apply_rope_costable(cos, sin, 1, rope)
            .reshape(&[s, 1, rope])
            .broadcast_to(&[s, h, rope])
            .contiguous();

        // --- assemble and attend ---
        let qh = q_pass.cat(&q_rot, 2).reshape(&[s, h * qk]);
        let kh = k_nope.cat(&k_rot, 2).reshape(&[s, h * qk]);
        let vv = value.reshape(&[s, h * vh]);
        // `causal_attention` divides by sqrt(head_dim) internally; pre-multiplying by that factor makes
        // the effective scale exactly `cfg.scaling`, which carries YaRN's mscale and is not derivable.
        let qh = qh.mul(&qh.scalar(c.scaling * (qk as f32).sqrt()));
        let ao = nn::causal_attention(&qh, &kh, &vv, h, h, 0.0);

        // --- optional gate, then output projection ---
        let ao = match &self.w.gate_proj {
            Some(g) => ao.mul(&hs.matmul_bt(g).sigmoid()),
            None => ao,
        };
        ao.matmul_bt(&self.w.o_proj)
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
