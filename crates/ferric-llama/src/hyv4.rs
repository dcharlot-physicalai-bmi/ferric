//! **Tencent Hy4 (`hyv4`)** — 770B total, 49B active, 1M context, and supported by no upstream
//! runtime.
//!
//! `llama.cpp` does not have this architecture; the published GGUFs ship with two out-of-tree
//! patches. So this is not a port: it is an independent implementation from the format, and it is
//! what lets Ferric load a checkpoint nothing else can.
//!
//! ## What the architecture actually is
//!
//! DeepSeek-V3's MLA with three grafts, and one of them is genuinely new:
//!
//! * **Hyper-connections** ([`crate::hc`]) — the single residual stream becomes four, read down to
//!   one vector before each sublayer by a learned gate and written back into all four by another.
//!   Algebraically a rank-4 factorised DenseNet over sublayer outputs.
//! * **Gated MLA with a learnable sink** — the attention output is gated elementwise by a
//!   projection of the *layer input*, and each head carries one raw sink logit so its attention sums
//!   to strictly less than one.
//! * **DeepSeek Sparse Attention** ([`crate::dsa`]) — a cheap 32-head indexer scores every position
//!   and only the best `top_k` enter the real attention. Only 21 of 78 layers own indexer weights;
//!   the rest reuse the most recent preceding full layer's selection.
//!
//! plus DeepSeekMoE (256 routed + 1 shared, sigmoid gating, an expert bias used for selection) with
//! a **clamped** SwiGLU, and Q-LoRA + absorbed MLA, both of which `deepseek2` refuses.
//!
//! ## What is verified, and what a green test here does not mean
//!
//! Every component is verified on its own: the hyper-connection closed form and both absorption
//! folds exactly over GF(2⁶¹−1), the sink's softmax against its definition, the indexer's schedule
//! by bounded model checking for every `is_full` pattern, the quant formats by Kani plus an
//! interop check against Tencent's own published weights. See `VERIFICATION.md`.
//!
//! ⛔ **What is NOT verified is this file against the real model.** The smallest published
//! checkpoint is 213.66 GiB and this machine has ~47 GB free, so nothing here has ever seen the
//! trained weights. The test in `examples/hyv4_synthetic.rs` writes a tiny checkpoint and runs a
//! forward pass through it, which proves the WIRING — that the pieces compose, the tensor names
//! resolve, the shapes agree end to end. It cannot prove fidelity to Tencent's model, because the
//! same convention used to write the file is used to read it back. Those are different claims and
//! this file does not blur them.

use crate::dsa::{Indexer, IndexSchedule, IndexerCfg, IndexerWeights};
use crate::hc::{Hc, HcConfig, HcGate, HcHead};
use crate::mla::{KvUp, Mla, MlaConfig, MlaWeights, QProj};
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::{dtype::QMatrix, Tensor};
use std::sync::Arc;

/// Everything `hyv4.*` puts in the GGUF KV block, read once at load.
#[derive(Debug, Clone)]
pub struct Cfg {
    pub n_layer: usize,
    pub d: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// `attention.key_length_mla` = qk_nope + qk_rope. The 576-wide `key_length` is the CACHE row
    /// (kv_lora + rope), a different quantity that happens to live under a similar key.
    pub qk_head: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub rope_base: f32,
    pub dense_lead: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub expert_ff: usize,
    pub routed_scale: f32,
    pub expert_norm: bool,
    pub sigmoid_gate: bool,
    /// Per-layer SwiGLU clamp. The GGUF carries 78 of these and they are all 10.0, but it is an
    /// array and reading it as a scalar would silently take layer 0's value for every layer.
    pub swiglu_clamp: Vec<f32>,
    pub hc: usize,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub idx_heads: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    pub idx_is_full: Vec<bool>,
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let arch = match md.get("general.architecture") {
            Some(Meta::Str(s)) => s.clone(),
            _ => return Err("no general.architecture".into()),
        };
        if arch != "hyv4" { return Err(format!("not a hyv4 checkpoint: architecture is '{arch}'")) }
        // ⛔ ACCEPT BOTH SIGNAGES. GGUF's numeric KV tags distinguish signed from unsigned, and a
        // writer picks whichever it likes for a count that is never negative. Tencent's file stores
        // `indexer.is_full` as **I32**; reading only `Meta::U` silently produced an all-false
        // schedule, which `IndexSchedule::new` then refused — this loader rejected the real
        // checkpoint outright. The synthetic test could not catch it because that file was written
        // with `kv_arr_u32`: the same convention on both sides, cancelling.
        let u = |k: &str| -> Result<usize, String> {
            match md.get(&format!("hyv4.{k}")) {
                Some(Meta::U(v)) => Ok(*v as usize),
                Some(Meta::I(v)) if *v >= 0 => Ok(*v as usize),
                _ => Err(format!("hyv4.{k} missing or not a non-negative integer")),
            }
        };
        let f = |k: &str| -> Result<f32, String> {
            match md.get(&format!("hyv4.{k}")) {
                Some(Meta::F(v)) => Ok(*v as f32),
                Some(Meta::U(v)) => Ok(*v as f32),
                Some(Meta::I(v)) => Ok(*v as f32),
                _ => Err(format!("hyv4.{k} missing or not a number")),
            }
        };
        let arr_f = |k: &str| -> Result<Vec<f32>, String> {
            match md.get(&format!("hyv4.{k}")) {
                Some(Meta::Arr(v)) => v.iter().map(|m| match m {
                    Meta::F(x) => Ok(*x as f32), Meta::U(x) => Ok(*x as f32), Meta::I(x) => Ok(*x as f32),
                    _ => Err(format!("hyv4.{k} has a non-numeric element")),
                }).collect(),
                _ => Err(format!("hyv4.{k} missing or not an array")),
            }
        };
        let arr_b = |k: &str| -> Result<Vec<bool>, String> {
            match md.get(&format!("hyv4.{k}")) {
                Some(Meta::Arr(v)) => v.iter().map(|m| match m {
                    Meta::U(x) => Ok(*x != 0),
                    Meta::I(x) => Ok(*x != 0),
                    Meta::Bool(b) => Ok(*b),
                    // Never default to false here: an unreadable flag that reads as "not full"
                    // produces a schedule that is wrong rather than a load that fails.
                    _ => Err(format!("hyv4.{k} has an element that is not a flag")),
                }).collect(),
                _ => Err(format!("hyv4.{k} missing or not an array")),
            }
        };

        let n_layer = u("block_count")?;
        let swiglu_clamp = arr_f("swiglu_clamp_exp").unwrap_or_else(|_| vec![f32::INFINITY; n_layer]);
        let idx_is_full = arr_b("attention.indexer.is_full")?;
        if idx_is_full.len() != n_layer {
            return Err(format!("indexer.is_full has {} entries for {n_layer} blocks", idx_is_full.len()));
        }
        if swiglu_clamp.len() != n_layer {
            return Err(format!("swiglu_clamp_exp has {} entries for {n_layer} blocks", swiglu_clamp.len()));
        }
        Ok(Cfg {
            n_layer,
            d: u("embedding_length")?,
            n_ff: u("feed_forward_length")?,
            n_head: u("attention.head_count")?,
            n_vocab: u("vocab_size").or_else(|_| u("context_length").map(|_| 0)).unwrap_or(0),
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_head: u("attention.key_length_mla")?,
            qk_rope: u("rope.dimension_count")?,
            v_head: u("attention.value_length_mla")?,
            q_lora_rank: u("attention.q_lora_rank")?,
            kv_lora_rank: u("attention.kv_lora_rank")?,
            rope_base: f("rope.freq_base").unwrap_or(10_000_000.0),
            dense_lead: u("leading_dense_block_count").unwrap_or(1),
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_expert_shared: u("expert_shared_count").unwrap_or(1),
            expert_ff: u("expert_feed_forward_length")?,
            routed_scale: f("expert_weights_scale").unwrap_or(1.0),
            expert_norm: match md.get("hyv4.expert_weights_norm") {
                Some(Meta::Bool(b)) => *b, Some(Meta::U(v)) => *v != 0, Some(Meta::I(v)) => *v != 0, _ => false },
            // Gating function 2 is sigmoid; anything else here is softmax, and the two differ by
            // more than a nonlinearity — sigmoid gating does not normalise across experts.
            sigmoid_gate: matches!(md.get("hyv4.expert_gating_func"), Some(Meta::U(2)) | Some(Meta::I(2))),
            swiglu_clamp,
            hc: u("hyper_connection.count")?,
            hc_eps: f("hyper_connection.epsilon").unwrap_or(1e-6),
            hc_magnitude: f("hyper_connection.magnitude").unwrap_or(2.0),
            idx_heads: u("attention.indexer.head_count")?,
            idx_head_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            idx_is_full,
        })
    }

    /// The non-positional half of a query/key head.
    pub fn qk_nope(&self) -> usize { self.qk_head - self.qk_rope }
    /// Floats the KV cache holds per position per layer: the latent plus the one shared RoPE key.
    pub fn cache_floats(&self) -> usize { self.kv_lora_rank + self.qk_rope }
}

enum Ffn {
    Dense { gate: QMatrix, up: QMatrix, down: QMatrix },
    Moe {
        router: QMatrix, bias: Option<Vec<f32>>,
        /// One `QMatrix` per expert per projection, sliced out of the stacked slab at load.
        ///
        /// ⛔ NOT the fused expert-major slab `deepseek2` uses. That path runs one batched kernel
        /// over every routed expert and requires Q4_K; hyv4's published checkpoint stores its
        /// experts as STQ1_0, IQ2_XXS and IQ3_XXS, none of which has an indexed-expert kernel. A
        /// fused path would therefore refuse the real model outright. Slicing per expert keeps the
        /// weights at their on-disk size, works for every quant type Ferric reads, and costs one
        /// dispatch per routed expert instead of one per layer. That is the capability/throughput
        /// trade taken deliberately: a slow forward that runs beats a fast one that cannot load.
        gate: Vec<QMatrix>, up: Vec<QMatrix>, down: Vec<QMatrix>,
        sh_gate: QMatrix, sh_up: QMatrix, sh_down: QMatrix,
    },
}

struct Block {
    hc_attn: HcGate,
    hc_ffn: HcGate,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    mla: Mla,
    /// Present only on the 21 `is_full` layers; the others reuse a preceding selection.
    indexer: Option<Indexer>,
    ffn: Ffn,
}

pub struct Hyv4 {
    pub cfg: Cfg,
    ctx: Arc<Context>,
    tok_embd: Tensor,
    blocks: Vec<Block>,
    hc: Hc,
    head: HcHead,
    output_norm: Tensor,
    output: QMatrix,
    schedule: IndexSchedule,
}

impl Hyv4 {
    /// Every tensor this loader will ask for, with the dims it expects, in **GGUF `ne` order**
    /// (`ne0` fastest — a `[out, in]` matrix appears as `[in, out]`).
    ///
    /// This is the loader's own expectations, not a second copy of them: `load` resolves the same
    /// names, and `examples/hyv4_validate.rs` checks this table against a real published header. A
    /// shape stated here and read differently there is the transposition class of bug, and it is
    /// the one thing about a format that a synthetic checkpoint can never catch — writing and
    /// reading with the same wrong convention cancels.
    pub fn expected_tensors(cfg: &Cfg, schedule: &IndexSchedule) -> Vec<(String, Vec<u64>)> {
        let (d, h) = (cfg.d as u64, cfg.n_head as u64);
        let (qk, vh) = (cfg.qk_head as u64, cfg.v_head as u64);
        let (ql, kvl, rope) = (cfg.q_lora_rank as u64, cfg.kv_lora_rank as u64, cfg.qk_rope as u64);
        let (hc, nope) = (cfg.hc as u64, cfg.qk_nope() as u64);
        let (ne, eff, ff) = (cfg.n_expert as u64, cfg.expert_ff as u64, cfg.n_ff as u64);
        let mut v: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![d, cfg.n_vocab as u64]),
            ("output.weight".into(), vec![d, cfg.n_vocab as u64]),
            ("output_norm.weight".into(), vec![d]),
            ("output_hc_fn.weight".into(), vec![hc * d, hc]),
            ("output_hc_base.weight".into(), vec![hc]),
            ("output_hc_scale.weight".into(), vec![1]),
        ];
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            v.extend([
                (b("attn_norm.weight"), vec![d]),
                (b("ffn_norm.weight"), vec![d]),
                (b("attn_q_a.weight"), vec![d, ql]),
                (b("attn_q_a_norm.weight"), vec![ql]),
                (b("attn_q_b.weight"), vec![ql, h * qk]),
                (b("attn_kv_a_mqa.weight"), vec![d, kvl + rope]),
                (b("attn_kv_a_norm.weight"), vec![kvl]),
                (b("attn_k_b.weight"), vec![nope, kvl, h]),
                (b("attn_v_b.weight"), vec![kvl, vh, h]),
                (b("attn_gate.weight"), vec![d, h * vh]),
                (b("attn_output.weight"), vec![h * vh, d]),
                (b("attn_sinks.weight"), vec![h]),
                (b("hc_attn_fn.weight"), vec![hc * d, 2 * hc]),
                (b("hc_attn_base.weight"), vec![2 * hc]),
                (b("hc_attn_scale.weight"), vec![2]),
                (b("hc_ffn_fn.weight"), vec![hc * d, 2 * hc]),
                (b("hc_ffn_base.weight"), vec![2 * hc]),
                (b("hc_ffn_scale.weight"), vec![2]),
            ]);
            if schedule.is_full(il) {
                let (ih, idk) = (cfg.idx_heads as u64, cfg.idx_head_dim as u64);
                v.extend([
                    (b("indexer.attn_q_b.weight"), vec![ql, ih * idk]),
                    (b("indexer.attn_k.weight"), vec![d, idk]),
                    (b("indexer.k_norm.weight"), vec![idk]),
                    (b("indexer.k_norm.bias"), vec![idk]),
                    (b("indexer.proj.weight"), vec![d, ih]),
                ]);
            }
            if il < cfg.dense_lead {
                v.extend([
                    (b("ffn_gate.weight"), vec![d, ff]),
                    (b("ffn_up.weight"), vec![d, ff]),
                    (b("ffn_down.weight"), vec![ff, d]),
                ]);
            } else {
                v.extend([
                    (b("ffn_gate_inp.weight"), vec![d, ne]),
                    // Optional in `load` (a checkpoint may route without a selection bias) but
                    // present on every MoE block of the published file, so it belongs here: a
                    // tensor the loader reads and the table omits shows up as "never read".
                    (b("exp_probs_b.bias"), vec![ne]),
                    (b("ffn_gate_exps.weight"), vec![d, eff, ne]),
                    (b("ffn_up_exps.weight"), vec![d, eff, ne]),
                    (b("ffn_down_exps.weight"), vec![eff, d, ne]),
                    (b("ffn_gate_shexp.weight"), vec![d, eff]),
                    (b("ffn_up_shexp.weight"), vec![d, eff]),
                    (b("ffn_down_shexp.weight"), vec![eff, d]),
                ]);
            }
        }
        v
    }

    /// The tensor names this architecture requires, as they appear in the published checkpoints.
    ///
    /// Listed because a missing tensor should fail at load with a name, not at the first forward
    /// pass with a shape mismatch four layers deep.
    pub fn required_block_tensors(is_full: bool, dense: bool) -> Vec<&'static str> {
        let mut v = vec![
            "attn_norm.weight", "ffn_norm.weight",
            "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
            "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
            "attn_k_b.weight", "attn_v_b.weight",
            "attn_gate.weight", "attn_output.weight", "attn_sinks.weight",
            "hc_attn_base.weight", "hc_attn_fn.weight", "hc_attn_scale.weight",
            "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight",
        ];
        if is_full {
            v.extend(["indexer.attn_q_b.weight", "indexer.attn_k.weight",
                      "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight"]);
        }
        if dense {
            v.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
        } else {
            v.extend(["ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                      "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                      "ffn_down_shexp.weight"]);
        }
        v
    }
}

impl Hyv4 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Hyv4, String> {
        let cfg = Cfg::from_gguf(g)?;
        let schedule = IndexSchedule::new(cfg.idx_is_full.clone())?;

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
            let v = g.dequant(name)?;
            let want: usize = shape.iter().product();
            if v.len() != want {
                return Err(format!("{name}: {} elements for shape {shape:?} ({want})", v.len()));
            }
            Ok(Tensor::from_vec(ctx, &v, shape))
        };
        // ⚠ The MLA projections load DENSE, not as QMatrix: `MlaWeights` consumes `Tensor` and the
        // absorbed path needs `matmul_bt`. On the 770B that is a real memory cost and the right fix
        // is a packed MLA path; it is not a correctness issue and it is not what this file is for.
        //
        // ⚠ A 2-D GGUF weight is ne = [in, out]; Ferric wants [out, in]. A 3-D one is
        // ne = [a, b, heads] and reverses whole. Getting this backwards is the transposition class
        // of bug: right element count, wrong arrangement, fluent output.
        let ft3 = |name: &str, a: usize, b: usize, h: usize| -> Result<Tensor, String> { ft(name, &[h, b, a]) };

        let (hc, d) = (cfg.hc, cfg.d);
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let dense = il < cfg.dense_lead;
            for t in Hyv4::required_block_tensors(schedule.is_full(il), dense) {
                if g.tensor(&b(t)).is_none() {
                    return Err(format!("block {il} is missing {t}; hyv4 requires it \
                                        ({} layer, {} FFN)",
                                       if schedule.is_full(il) { "indexer" } else { "shared-index" },
                                       if dense { "dense" } else { "MoE" }));
                }
            }

            let hcg = |half: &str| -> Result<HcGate, String> {
                Ok(HcGate {
                    fn_w: ft(&b(&format!("hc_{half}_fn.weight")), &[2 * hc, hc * d])?,
                    base: ft(&b(&format!("hc_{half}_base.weight")), &[2 * hc])?,
                    scale: {
                        let s = g.dequant(&b(&format!("hc_{half}_scale.weight")))?;
                        if s.len() != 2 { return Err(format!("hc_{half}_scale has {} entries, want 2", s.len())) }
                        [s[0], s[1]]
                    },
                })
            };

            let mla = Mla::new(
                MlaConfig {
                    n_heads: cfg.n_head,
                    qk_nope_dim: cfg.qk_nope(),
                    qk_rope_dim: cfg.qk_rope,
                    v_head_dim: cfg.v_head,
                    kv_lora_rank: cfg.kv_lora_rank,
                    // hyv4 writes no YaRN keys, so the scale is the plain 1/sqrt(qk_head) --
                    // over the FULL head width, not the 576-wide dot the absorbed path takes.
                    scaling: 1.0 / (cfg.qk_head as f32).sqrt(),
                    eps: cfg.eps,
                    rope_interleaved: true,
                },
                MlaWeights {
                    q: QProj::LowRank {
                        a: ft(&b("attn_q_a.weight"), &[cfg.q_lora_rank, d])?,
                        a_norm: ft(&b("attn_q_a_norm.weight"), &[cfg.q_lora_rank])?,
                        b: ft(&b("attn_q_b.weight"), &[cfg.n_head * cfg.qk_head, cfg.q_lora_rank])?,
                    },
                    kv_a_proj_with_mqa: ft(&b("attn_kv_a_mqa.weight"), &[cfg.kv_lora_rank + cfg.qk_rope, d])?,
                    kv_a_layernorm: ft(&b("attn_kv_a_norm.weight"), &[cfg.kv_lora_rank])?,
                    kv_up: KvUp::Absorbed {
                        k_b: ft3(&b("attn_k_b.weight"), cfg.qk_nope(), cfg.kv_lora_rank, cfg.n_head)?,
                        v_b: ft3(&b("attn_v_b.weight"), cfg.kv_lora_rank, cfg.v_head, cfg.n_head)?,
                    },
                    o_proj: ft(&b("attn_output.weight"), &[d, cfg.n_head * cfg.v_head])?,
                    gate_proj: Some(ft(&b("attn_gate.weight"), &[cfg.n_head * cfg.v_head, d])?),
                    sinks: Some(ft(&b("attn_sinks.weight"), &[cfg.n_head])?),
                },
            );

            let indexer = if schedule.is_full(il) {
                Some(Indexer::new(
                    IndexerCfg {
                        n_heads: cfg.idx_heads,
                        head_dim: cfg.idx_head_dim,
                        rope_dim: cfg.qk_rope,
                        top_k: cfg.idx_top_k,
                        eps: cfg.eps,
                        rope_interleaved: true,
                    },
                    IndexerWeights {
                        q_b: ft(&b("indexer.attn_q_b.weight"), &[cfg.idx_heads * cfg.idx_head_dim, cfg.q_lora_rank])?,
                        k: ft(&b("indexer.attn_k.weight"), &[cfg.idx_head_dim, d])?,
                        k_norm_w: ft(&b("indexer.k_norm.weight"), &[cfg.idx_head_dim])?,
                        k_norm_b: ft(&b("indexer.k_norm.bias"), &[cfg.idx_head_dim])?,
                        proj: ft(&b("indexer.proj.weight"), &[cfg.idx_heads, d])?,
                    },
                ))
            } else { None };

            let ffn = if dense {
                Ffn::Dense { gate: qm(&b("ffn_gate.weight"))?, up: qm(&b("ffn_up.weight"))?, down: qm(&b("ffn_down.weight"))? }
            } else {
                // Slice a stacked [n_expert, rows, cols] slab into one QMatrix per expert.
                let slab = |name: &str| -> Result<Vec<QMatrix>, String> {
                    let t = g.tensor(name).ok_or_else(|| format!("missing {name}"))?;
                    let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
                    let raw = g.raw(name)?;
                    let per = rows * ferric_gguf::type_size(t.ggml_type, cols)?;
                    if raw.len() != per * cfg.n_expert {
                        return Err(format!("{name}: {} bytes for {} experts of {per}", raw.len(), cfg.n_expert));
                    }
                    (0..cfg.n_expert).map(|e| {
                        let bytes = &raw[e * per..(e + 1) * per];
                        if QMatrix::block_bytes(t.ggml_type).is_some() {
                            QMatrix::from_bytes(ctx, bytes, t.ggml_type, rows, cols)
                        } else {
                            let v = ferric_gguf::deq_raw(bytes, rows * cols, t.ggml_type)?;
                            Ok(QMatrix::from_dense(ctx, &v, rows, cols))
                        }
                    }).collect()
                };
                Ffn::Moe {
                    router: qm(&b("ffn_gate_inp.weight"))?,
                    bias: match g.tensor(&b("exp_probs_b.bias")) {
                        Some(_) => Some(g.dequant(&b("exp_probs_b.bias"))?),
                        None => None,
                    },
                    gate: slab(&b("ffn_gate_exps.weight"))?,
                    up: slab(&b("ffn_up_exps.weight"))?,
                    down: slab(&b("ffn_down_exps.weight"))?,
                    sh_gate: qm(&b("ffn_gate_shexp.weight"))?,
                    sh_up: qm(&b("ffn_up_shexp.weight"))?,
                    sh_down: qm(&b("ffn_down_shexp.weight"))?,
                }
            };

            blocks.push(Block {
                hc_attn: hcg("attn")?, hc_ffn: hcg("ffn")?,
                attn_norm: ft(&b("attn_norm.weight"), &[d])?,
                ffn_norm: ft(&b("ffn_norm.weight"), &[d])?,
                mla, indexer, ffn,
            });
        }

        Ok(Hyv4 {
            hc: Hc::new(HcConfig { hc, d, eps_hc: cfg.hc_eps, eps_rms: cfg.eps, magnitude: cfg.hc_magnitude }),
            head: HcHead {
                fn_w: ft("output_hc_fn.weight", &[hc, hc * d])?,
                base: ft("output_hc_base.weight", &[hc])?,
                scale: { let s = g.dequant("output_hc_scale.weight")?; s[0] },
            },
            tok_embd: {
                let v = g.dequant("token_embd.weight")?;
                let n = v.len() / d;
                Tensor::from_vec(ctx, &v, &[n, d])
            },
            output_norm: ft("output_norm.weight", &[d])?,
            output: qm("output.weight")?,
            ctx: ctx.clone(), cfg, blocks, schedule,
        })
    }
}

impl Hyv4 {
    /// SwiGLU with hyv4's per-layer clamp: `silu(clamp(gate)) * clamp(up)`.
    ///
    /// ⚠ The clamp is symmetric and applies to BOTH branches. Clamping only the gate, or only the
    /// positive side, leaves every shape intact and changes the model. `f32::INFINITY` (the default
    /// when the key is absent) makes `maximum`/`minimum` the identity, so an unclamped checkpoint
    /// takes the same code path rather than a second one.
    fn swiglu_clamped(&self, gate: &Tensor, up: &Tensor, limit: f32) -> Tensor {
        if !limit.is_finite() { return gate.silu().mul(up) }
        // No `minimum` op: `min(x, L) = −max(−x, −L)`, so the upper clamp is the lower one twice
        // through a negation. Same arithmetic, and it avoids a second kernel for one bound.
        let clamp = |t: &Tensor| {
            let lo = t.maximum(&t.scalar(-limit));
            lo.neg().maximum(&t.scalar(-limit)).neg()
        };
        clamp(gate).silu().mul(&clamp(up))
    }

    /// Pick this token's experts and their combining weights.
    ///
    /// The order is the whole content of the function, and every step of it is a same-shape trap:
    ///
    /// 1. **Gate first.** Sigmoid (`expert_gating_func == 2`) or softmax. Sigmoid does NOT normalise
    ///    across experts, which is why the renormalisation below is a separate, configurable step
    ///    rather than something the gate already did.
    /// 2. **The bias is added for SELECTION ONLY.** `exp_probs_b` shifts which experts win and is
    ///    then dropped; the combining weight is the UNBIASED gate value. Carrying the bias into the
    ///    weight keeps every shape and silently reweights the mixture.
    /// 3. **Renormalise over the chosen k** if `expert_weights_norm`, not over all experts.
    /// 4. **Scale last** by `expert_weights_scale` (2.827). Scaling before the renormalisation
    ///    cancels it out entirely, which is the failure that looks like nothing happening.
    fn route(&self, logits: &[f32], bias: Option<&[f32]>) -> Vec<(usize, f32)> {
        let cfg = &self.cfg;
        let gated: Vec<f32> = if cfg.sigmoid_gate {
            logits.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect()
        } else {
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = logits.iter().map(|z| (z - m).exp()).collect();
            let s: f32 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        };
        let mut order: Vec<usize> = (0..cfg.n_expert).collect();
        let key = |i: usize| gated[i] + bias.map_or(0.0, |b| b[i]);
        order.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(cfg.n_expert_used);

        // The WEIGHT is the unbiased gate value, whatever the bias did to the ordering.
        let mut w: Vec<f32> = order.iter().map(|&i| gated[i]).collect();
        if cfg.expert_norm {
            let s: f32 = w.iter().sum();
            if s > 0.0 { for v in &mut w { *v /= s } }
        }
        for v in &mut w { *v *= cfg.routed_scale }
        order.into_iter().zip(w).collect()
    }

    fn ffn(&self, f: &Tensor, blk: &Block, il: usize) -> Tensor {
        let cfg = &self.cfg;
        let limit = cfg.swiglu_clamp[il];
        match &blk.ffn {
            Ffn::Dense { gate, up, down } =>
                self.swiglu_clamped(&f.matmul_q(gate), &f.matmul_q(up), limit).matmul_q(down),
            Ffn::Moe { router, bias, gate, up, down, sh_gate, sh_up, sh_down } => {
                let t = f.shape[0];
                let logits = pollster::block_on(f.matmul_q(router).to_vec());
                let mut routed = Tensor::from_vec(&self.ctx, &vec![0.0f32; t * cfg.d], &[t, cfg.d]);
                for tok in 0..t {
                    let sel = self.route(&logits[tok * cfg.n_expert..(tok + 1) * cfg.n_expert], bias.as_deref());
                    let row = f.narrow(0, tok, 1).contiguous();
                    let mut acc = Tensor::from_vec(&self.ctx, &vec![0.0f32; cfg.d], &[1, cfg.d]);
                    for (e, w) in sel {
                        let mid = self.swiglu_clamped(&row.matmul_q(&gate[e]), &row.matmul_q(&up[e]), limit);
                        acc = acc.add(&mid.matmul_q(&down[e]).mul(&row.scalar(w)));
                    }
                    // Scatter this token's row back. `narrow` is a view, so the write is an explicit
                    // concatenation rather than an in-place store.
                    let before = if tok == 0 { None } else { Some(routed.narrow(0, 0, tok).contiguous()) };
                    let after = if tok + 1 == t { None } else { Some(routed.narrow(0, tok + 1, t - tok - 1).contiguous()) };
                    routed = match (before, after) {
                        (None, None) => acc,
                        (None, Some(a)) => acc.cat(&a, 0),
                        (Some(b), None) => b.cat(&acc, 0),
                        (Some(b), Some(a)) => b.cat(&acc, 0).cat(&a, 0),
                    };
                }
                let shared = self.swiglu_clamped(&f.matmul_q(sh_gate), &f.matmul_q(sh_up), limit).matmul_q(sh_down);
                routed.add(&shared)
            }
        }
    }

    /// Full-sequence forward. Returns logits `[seq, n_vocab]`.
    ///
    /// The block schedule, which is the part that has to be right in order rather than in isolation:
    ///
    /// ```text
    /// H = replicate(emb)
    /// for il:
    ///     res = H;  (x, q) = hc_pre(H, hc_attn[il])
    ///     H = hc_post( attention( rmsnorm(x, attn_norm[il]) ), res, q )
    ///     res = H;  (x, q) = hc_pre(H, hc_ffn[il])      // RE-READ after attention
    ///     H = hc_post( ffn( rmsnorm(x, ffn_norm[il]) ), res, q )
    /// y = rmsnorm( collapse(H), output_norm )
    /// ```
    ///
    /// ⚠ `res` is re-read after the attention sublayer. Hoisting it out of the loop, or reusing the
    /// pre-attention value for the FFN write-back, keeps every shape and silently drops the
    /// attention's contribution from the FFN's residual.
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        let cfg = &self.cfg;
        let t = tokens.len();
        let rows: Vec<u32> = tokens.to_vec();
        let emb = self.tok_embd.gather_rows(&rows);
        let (cos, sin) = self.rope_tables(t);

        let mut h = self.hc.replicate(&emb);
        // The selection is graph-local: recomputed at every full layer, reused by the layers after
        // it. It is NOT cached across forward passes -- both the query and the key set move.
        let mut last_mask: Option<Tensor> = None;

        for il in 0..cfg.n_layer {
            let blk = &self.blocks[il];

            let res = h.clone();
            let (x, q) = self.hc.pre(&h, &blk.hc_attn);
            let cur = x.rmsnorm(&blk.attn_norm, cfg.eps);

            debug_assert_eq!(blk.indexer.is_some(), self.schedule.is_full(il),
                             "block {il}: indexer presence disagrees with the schedule");
            if let Some(ix) = &blk.indexer {
                // `cur` feeds the indexer's key and per-head weights; its QUERY comes off the q_a
                // latent, which `Indexer::scores` recomputes from `qr`. Both are the same `cur` here
                // because this file does not yet share the latent between them -- a compute saving,
                // not a semantic difference.
                let qr = cur.matmul_bt(match &blk.mla.w.q { QProj::LowRank { a, .. } => a, QProj::Whole(w) => w })
                    .rmsnorm(match &blk.mla.w.q { QProj::LowRank { a_norm, .. } => a_norm, QProj::Whole(_) => &blk.attn_norm }, cfg.eps);
                let keys = ix.keys(&cur, &cos, &sin);
                let scores = ix.scores(&qr, &cur, &keys, &cos, &sin);
                let m = crate::dsa::top_k_mask(&scores, cfg.idx_top_k, 0);
                last_mask = Some(m.reshape(&[1, t, t]).broadcast_to(&[cfg.n_head, t, t]).contiguous());
            }
            assert!(last_mask.is_some(), "layer {il} has no selection; is_full[0] must be true");

            let a = blk.mla.forward_masked(&cur, &cos, &sin, last_mask.as_ref());
            h = self.hc.post(&a, &res, &q);

            let res = h.clone();
            let (x, q) = self.hc.pre(&h, &blk.hc_ffn);
            let cur = x.rmsnorm(&blk.ffn_norm, cfg.eps);
            let f = self.ffn(&cur, blk, il);
            h = self.hc.post(&f, &res, &q);
        }

        let y = self.hc.collapse(&h, &self.head).rmsnorm(&self.output_norm, cfg.eps);
        y.matmul_q(&self.output)
    }

    /// Which layers own an indexer, and therefore how many index-cache slots to reserve.
    ///
    /// A naive allocator reserves one per block. Only the `is_full` layers ever write a key — 21 of
    /// 78 on the published checkpoint — so the difference is 5.63 GiB against 20.9 GiB per sequence
    /// at the full 1M context.
    pub fn index_schedule(&self) -> &IndexSchedule { &self.schedule }

    /// Doubled cos/sin tables for the `qk_rope`-wide rotation. hyv4 writes no `rope.scaling.*`, so
    /// there is no YaRN term and no `mscale` folded into the attention scale.
    fn rope_tables(&self, seq: usize) -> (Tensor, Tensor) {
        let r = self.cfg.qk_rope;
        let (mut c, mut s) = (vec![0.0f32; seq * r], vec![0.0f32; seq * r]);
        for p in 0..seq {
            for i in 0..r / 2 {
                let th = p as f32 * (self.cfg.rope_base as f64).powf(-2.0 * i as f64 / r as f64) as f32;
                let (ct, st) = (th.cos(), th.sin());
                c[p * r + i] = ct; c[p * r + i + r / 2] = ct;
                s[p * r + i] = st; s[p * r + i + r / 2] = st;
            }
        }
        (Tensor::from_vec(&self.ctx, &c, &[seq, r]), Tensor::from_vec(&self.ctx, &s, &[seq, r]))
    }
}
