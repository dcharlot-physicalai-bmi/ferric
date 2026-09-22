//! **ModernBERT** (`modern-bert`) — the encoder behind gte-reranker-modernbert, mmBERT, and the
//! 2026 typed-decision models that answer in one forward pass instead of generating tokens.
//!
//! It is NOT the BERT in [`crate::bert`]. Every structural choice differs:
//!
//! | | BERT ([`crate::bert`]) | ModernBERT |
//! |---|---|---|
//! | positions | learned lookup, added once | **RoPE (NeoX), per layer** |
//! | rope base | — | **two of them** — see the warning below |
//! | attention | full bidirectional | **alternating symmetric-band / global** |
//! | norm | POST-LayerNorm, with bias | **PRE-LayerNorm, no bias** |
//! | FFN | plain GELU, up → down | **GeGLU**, `ffn_up` is `{d, 2·n_ff}` |
//! | qkv | three tensors | **one fused `{d, 3·d}`** |
//! | layer 0 | — | **`attn_norm` absent = identity** |
//! | tokenizer | WordPiece | **byte-level BPE** (`tokenizer.ggml.model = gpt2`) |
//!
//! ## ⛔⛔ THE TRAP: TWO ROPE BASES, AND ONE CHECKPOINT HIDES IT
//!
//! Global layers use `rope.freq_base`; sliding layers use `rope.freq_base_swa`. From the real files:
//!
//! | | `freq_base` | `freq_base_swa` |
//! |---|---|---|
//! | ModernBERT-large / gte-reranker-modernbert-base | 160000 | **10000** |
//! | mmBERT-base | 160000 | **160000** |
//!
//! **16× apart on one, identical on the other.** An implementation that reads one base is bit-exact
//! on mmBERT and silently wrong on ModernBERT-large's two-in-three local layers — so a conformance
//! run against the multilingual checkpoint comes back clean while the English one quietly degrades.
//! Laya ships both. `llama.cpp` models this as `get_rope_freq_base(cparams, il)`, per layer
//! (`models/modern-bert.cpp:82`).
//!
//! ## The schedule, and why it is not written here
//!
//! `set_swa_pattern(swa_period, /*dense_first=*/true)` (`modern-bert.cpp:10`) — the **first** layer
//! of each group is global, not the last. That rule lives once, in
//! [`ferric_tensor::nn::swa_schedule`], because getting its phase wrong produces the right *number*
//! of global layers in the wrong places and nothing errors.
//!
//! ## Verification
//!
//! `.reference/llama.cpp` carries the converter (`conversion/bert.py:590`, which registers
//! `ModernBertForSequenceClassification`) and the graph (`src/models/modern-bert.cpp`, 174 lines).
//! `llama-embedding -m <file> --pooling cls` runs it, so there is a numeric oracle, and
//! `llama-eval-callback` gives per-tensor traces for seam-by-seam diffing. Use
//! `examples/modern_bert_ref.rs`.
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

use crate::bert::{t1, t2};

#[derive(Debug, Clone)]
pub struct Cfg {
    pub d: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub eps: f32,
    /// `attention.sliding_window`. The symmetric band's FULL width parameter — the visible span is
    /// `2*(n_swa/2) + 1`, which for 128 is 129, not 128. See [`nn::symmetric_band_masked`].
    pub n_swa: usize,
    /// `rope.freq_base` — the GLOBAL layers' base.
    pub rope_base: f32,
    /// `rope.freq_base_swa` — the SLIDING layers' base. ⛔ Falls back to `rope_base` when absent,
    /// which is right for mmBERT (they are equal) and would be WRONG for ModernBERT-large if that
    /// file omitted the key. It does not omit it; the fallback exists so a file that genuinely has
    /// one base still loads.
    pub rope_base_swa: f32,
    /// Per-layer: `true` = sliding (symmetric band), `false` = global.
    pub swa: Vec<bool>,
    /// `classifier.output_labels`, when the file declares them.
    pub labels: Vec<String>,
}

/// The sliding layers' rope base, given what the file stated and the global base.
///
/// ⛔ NOT `unwrap_or(10000.0)`. A file that omits `rope.freq_base_swa` is saying "I have one base",
/// and substituting llama.cpp's library default would rotate the local layers at 1/16 the right
/// frequency on a file whose single base is 160000 — a silent 16x error on two layers in three.
///
/// ⭐ This is a free function because the version of it that lived inline was UNTESTABLE: the test
/// for it built a `Cfg` directly and so exercised the struct literal, not this rule. Mutating the
/// fallback to 10000.0 left that test green. Pulling the rule out is what let the mutation be caught.
pub(crate) fn resolve_swa_base(stated: Option<f32>, global_base: f32) -> f32 {
    stated.unwrap_or(global_base)
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let arch = match md.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "modern-bert".into() };
        if arch != "modern-bert" {
            return Err(format!("this loader serves `modern-bert`; the file declares `{arch}`"));
        }
        let u = |k: &str| match md.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => Some(*v as usize), _ => None };
        let f = |k: &str| match md.get(&format!("{arch}.{k}")) { Some(Meta::F(v)) => Some(*v as f32), _ => None };
        let d = u("embedding_length").ok_or("missing embedding_length")?;
        let n_layer = u("block_count").ok_or("missing block_count")?;
        let n_head = u("attention.head_count").ok_or("missing attention.head_count")?;
        let n_ff = u("feed_forward_length").ok_or("missing feed_forward_length")?;
        if d % n_head != 0 {
            return Err(format!("embedding_length {d} is not divisible by head_count {n_head}"));
        }
        let eps = f("attention.layer_norm_epsilon").or_else(|| f("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let n_swa = u("attention.sliding_window").unwrap_or(0);
        let rope_base = f("rope.freq_base").unwrap_or(10000.0);
        // ⛔ NOT `unwrap_or(10000)`: a missing SWA base means "this file has one base", and guessing
        // llama.cpp's default here would silently rotate the local layers at 10000 on a file whose
        // single base is 160000. Fall back to the base the file DID state.
        let rope_base_swa = resolve_swa_base(f("rope.freq_base_swa"), rope_base);
        // The scalar arm; `dense_first = true` for this architecture (modern-bert.cpp:10).
        let pattern = u("attention.sliding_window_pattern");
        let swa = match md.get(&format!("{arch}.attention.sliding_window_pattern")) {
            Some(Meta::Arr(a)) => a.iter().map(|m| matches!(m, Meta::U(v) if *v != 0) || matches!(m, Meta::Bool(true))).collect(),
            _ if n_swa == 0 => vec![false; n_layer],
            _ => nn::swa_schedule(n_layer, pattern.or(Some(3)), true),
        };
        let labels = match md.get(&format!("{arch}.classifier.output_labels")) {
            Some(Meta::Arr(a)) => a.iter().filter_map(|m| match m { Meta::Str(s) => Some(s.clone()), _ => None }).collect(),
            _ => Vec::new(),
        };
        Ok(Cfg { d, n_layer, n_head, n_ff, eps, n_swa, rope_base, rope_base_swa, swa, labels })
    }

    /// The rope base layer `il` uses — the per-layer lookup `get_rope_freq_base(cparams, il)`.
    pub fn rope_base_for(&self, il: usize) -> f32 {
        if self.swa.get(il).copied().unwrap_or(false) { self.rope_base_swa } else { self.rope_base }
    }
}

struct Block {
    /// ⛔ Layer 0's `attn_norm` is genuinely absent in the reference (`TENSOR_NOT_REQUIRED`,
    /// `modern-bert.cpp:50`) and the graph then skips the norm entirely — identity, NOT a norm with
    /// unit weights, which would still re-center and re-scale.
    attn_norm: Option<Tensor>,
    qkv: Tensor,
    o: Tensor,
    ffn_norm: Tensor,
    up: Tensor,
    down: Tensor,
}

pub struct ModernBert {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    tok_embd: Vec<f32>,
    tok_norm: Tensor,
    out_norm: Tensor,
    zeros_d: Tensor,
    blocks: Vec<Block>,
}

impl ModernBert {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<ModernBert, String> {
        let cfg = Cfg::from_gguf(g)?;
        let tok_embd = g.dequant("token_embd.weight")?;
        let tok_norm = t1(ctx, g, "token_embd_norm.weight")?;
        let out_norm = t1(ctx, g, "output_norm.weight")?;
        let zeros_d = Tensor::from_vec(ctx, &vec![0.0f32; cfg.d], &[cfg.d]);
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            blocks.push(Block {
                attn_norm: t1(ctx, g, &format!("blk.{il}.attn_norm.weight")).ok(),
                qkv: t2(ctx, g, &format!("blk.{il}.attn_qkv.weight"))?,
                o: t2(ctx, g, &format!("blk.{il}.attn_output.weight"))?,
                ffn_norm: t1(ctx, g, &format!("blk.{il}.ffn_norm.weight"))?,
                up: t2(ctx, g, &format!("blk.{il}.ffn_up.weight"))?,
                down: t2(ctx, g, &format!("blk.{il}.ffn_down.weight"))?,
            });
        }
        Ok(ModernBert { ctx: ctx.clone(), cfg, tok_embd, tok_norm, out_norm, zeros_d, blocks })
    }

    /// LayerNorm with weight and NO bias — what `build_norm(x, w, nullptr, LLM_NORM, il)` does.
    fn ln(&self, x: &Tensor, w: &Tensor) -> Tensor { x.layernorm(w, &self.zeros_d, self.cfg.eps) }

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor, String> {
        self.forward_traced(ids).map(|(h, _)| h)
    }

    /// `forward`, plus a checkpoint after each named op for diffing against `llama-eval-callback`.
    /// Names match the reference's `cb(...)` labels so the two traces line up by row.
    pub fn forward_traced(&self, ids: &[u32]) -> Result<(Tensor, Vec<(String, Tensor)>), String> {
        let (d, t) = (self.cfg.d, ids.len());
        if t == 0 { return Err("empty token sequence".into()); }
        let (nh, dh) = (self.cfg.n_head, d / self.cfg.n_head);
        let scale = 1.0 / (dh as f32).sqrt();
        let positions: Vec<u32> = (0..t as u32).collect();

        let mut e = vec![0f32; t * d];
        for (p, &id) in ids.iter().enumerate() {
            let src = (id as usize) * d;
            if src + d > self.tok_embd.len() {
                return Err(format!("token id {id} is outside this vocabulary"));
            }
            e[p * d..(p + 1) * d].copy_from_slice(&self.tok_embd[src..src + d]);
        }
        let mut tr: Vec<(String, Tensor)> = Vec::new();
        let trace = std::env::var("FERRIC_MB_TRACE").ok().as_deref() == Some("1");

        // ⭐ No position embedding and no type embedding — RoPE carries order, inside every layer.
        let mut inp = Tensor::from_vec(&self.ctx, &e, &[t, d]);
        if trace { tr.push(("inp_embd".into(), inp.clone())); }
        inp = self.ln(&inp, &self.tok_norm);
        if trace { tr.push(("inp_norm".into(), inp.clone())); }

        for (il, b) in self.blocks.iter().enumerate() {
            // PRE-norm, and layer 0 has no norm tensor at all: identity, not a unit-weight norm.
            let cur = match &b.attn_norm { Some(w) => self.ln(&inp, w), None => inp.clone() };

            // One fused projection, then split [q | k | v] along the feature axis.
            let qkv = cur.matmul_bt(&b.qkv);
            let q = qkv.narrow(1, 0, d).contiguous();
            let k = qkv.narrow(1, d, d).contiguous();
            let v = qkv.narrow(1, 2 * d, d).contiguous();

            // RoPE with THIS layer's base. NeoX pairing (LLAMA_ROPE_TYPE_NEOX, llama-model.cpp:2606),
            // which is what `rope_at` implements.
            let base = if std::env::var("FERRIC_MB_ONE_ROPE").is_ok() {
                self.cfg.rope_base   // the "read one base" mistake, on purpose
            } else { self.cfg.rope_base_for(il) };
            let q = q.rope_at(nh, dh, base, &positions);
            let k = k.rope_at(nh, dh, base, &positions);

            let sh = |x: &Tensor| x.reshape(&[t, nh, dh]).permute(&[1, 0, 2]).contiguous();
            let (qh, kh, vh) = (sh(&q), sh(&k), sh(&v));

            // Local layers get a SYMMETRIC band centred on the query — not a causal one. Global
            // layers get no mask at all.
            // ⭐ TWO NEGATIVE CONTROLS, so "it matches the reference" is a claim about the HARD
            // parts and not a coincidence. Each disables one mechanism; if the diff against
            // llama.cpp does not get dramatically worse, that mechanism was never load-bearing and
            // the conformance number was proving nothing. See scripts/modern_bert_conformance.sh.
            let is_local = self.cfg.swa.get(il).copied().unwrap_or(false)
                && std::env::var("FERRIC_MB_NO_SWA").is_err();
            let mask = if is_local && self.cfg.n_swa > 0 {
                Some(nn::symmetric_band_mask(&inp, t, t, 0, self.cfg.n_swa))
            } else { None };

            let sc = Tensor::from_vec(&self.ctx, &[scale], &[1, 1]).broadcast_to(&[t, t]);
            let mut heads: Vec<Tensor> = Vec::with_capacity(nh);
            for i in 0..nh {
                let qi = qh.narrow(0, i, 1).reshape(&[t, dh]);
                let ki = kh.narrow(0, i, 1).reshape(&[t, dh]);
                let vi = vh.narrow(0, i, 1).reshape(&[t, dh]);
                let mut a = qi.matmul_bt(&ki).mul(&sc);
                if let Some(m) = &mask { a = a.add(m); }
                heads.push(a.softmax(1).matmul(&vi));
            }
            let cat = heads.iter().skip(1).fold(heads[0].clone(), |acc, x| acc.cat(x, 1));
            let attn = cat.reshape(&[t, d]).matmul_bt(&b.o);

            // Residual from the UN-normalised input — pre-norm, so `inp` is what comes back.
            let ffn_inp = attn.add(&inp);
            if trace { tr.push((format!("l{il}.ffn_inp"), ffn_inp.clone())); }

            let h = self.ln(&ffn_inp, &b.ffn_norm);
            // GeGLU. `ffn_up` emits 2·n_ff and ggml splits it in half with `swapped = false`:
            // `ggml_vec_geglu_f32(n, y, x, g)` is `y = gelu(x) * g` with x the FIRST half and g the
            // SECOND (ggml.c `ggml_glu_impl`, ops.cpp `src0_p += swapped ? nc : 0`). Getting the two
            // halves the wrong way round runs, and produces confident nonsense.
            let ff = h.matmul_bt(&b.up);
            let nf = self.cfg.n_ff;
            let gate = ff.narrow(1, 0, nf).contiguous();
            let lin = ff.narrow(1, nf, nf).contiguous();
            // ggml's `gelu` is the TANH approximation (via an fp16 table), not the exact erf form.
            let act = gate.gelu_tanh().mul(&lin);
            inp = act.matmul_bt(&b.down).add(&ffn_inp);
            if trace { tr.push((format!("l{il}.layer_out"), inp.clone())); }
        }
        let out = self.ln(&inp, &self.out_norm);
        if trace { tr.push(("final_norm_out".into(), out.clone())); }
        Ok((out, tr))
    }
}

#[cfg(test)]
mod cfg_tests {
    use super::*;

    fn cfg(n_layer: usize, swa: Vec<bool>, base: f32, base_swa: f32) -> Cfg {
        Cfg { d: 768, n_layer, n_head: 12, n_ff: 1152, eps: 1e-5, n_swa: 128,
              rope_base: base, rope_base_swa: base_swa, swa, labels: vec![] }
    }

    /// ⛔⛔ THE TRAP THIS ARCHITECTURE HIDES, AND THE ONLY GUARD ON IT THAT RUNS IN CI.
    ///
    /// `scripts/modern_bert_conformance.sh` catches a one-base port at 257x the error — but it needs
    /// a checkpoint and `llama-embedding`, and CI has neither. Without this test the 16x split is
    /// unguarded exactly where regressions land.
    ///
    /// The fixture is the REAL geometry of both files: gte-reranker-modernbert-base and
    /// ModernBERT-large split 160000/10000; mmBERT-base has 160000 for both. A port that reads one
    /// base is bit-exact on the second and wrong on the first, which is why both are pinned here.
    #[test]
    fn sliding_layers_use_the_swa_rope_base_and_global_layers_do_not() {
        // dense_first at p=3 over 6 layers: global at 0 and 3.
        let swa = ferric_tensor::nn::swa_schedule(6, Some(3), true);
        assert_eq!(swa, vec![false, true, true, false, true, true], "dense_first phase");

        let split = cfg(6, swa.clone(), 160000.0, 10000.0);
        let bases: Vec<f32> = (0..6).map(|il| split.rope_base_for(il)).collect();
        assert_eq!(bases, vec![160000.0, 10000.0, 10000.0, 160000.0, 10000.0, 10000.0],
                   "global layers take rope_base, sliding layers take rope_base_swa");

        // ⭐ THE ASYMMETRY THAT MAKES THIS SILENT. mmBERT declares the SAME base twice, so a
        // one-base implementation is bit-exact on it — conformance against the multilingual
        // checkpoint comes back clean while the English one degrades.
        let same = cfg(6, swa, 160000.0, 160000.0);
        assert!((0..6).all(|il| same.rope_base_for(il) == 160000.0),
                "mmBERT's two bases are equal, so this checkpoint CANNOT detect a one-base port");
        assert_ne!(bases, vec![160000.0; 6],
                   "if the split fixture ever stops differing, this test has lost its subject");
    }

    /// A file that states only one base must fall back to the base it DID state — not to
    /// llama.cpp's 10000 default, which would rotate the local layers at 1/16 of the right rate.
    ///
    /// ⛔ THE FIRST VERSION OF THIS TEST COULD NOT FAIL. It built a `Cfg` with both bases already
    /// set and asserted on `rope_base_for`, so it exercised the struct literal in the test itself —
    /// mutating `from_gguf`'s fallback to `10000.0` left it green. It now calls the rule.
    #[test]
    fn a_missing_swa_base_falls_back_to_the_stated_base_not_to_ten_thousand() {
        assert_eq!(super::resolve_swa_base(None, 160000.0), 160000.0,
                   "absent means 'one base', so the STATED base is the answer");
        assert_ne!(super::resolve_swa_base(None, 160000.0), 10000.0,
                   "falling back to the library default would silently divide the frequency by 16");
        assert_eq!(super::resolve_swa_base(Some(10000.0), 160000.0), 10000.0,
                   "a stated base always wins");
        assert_eq!(super::resolve_swa_base(Some(160000.0), 160000.0), 160000.0, "mmBERT's shape");
    }

    /// With no sliding window declared, every layer is global and one base is correct.
    #[test]
    fn no_window_means_every_layer_is_global() {
        let c = cfg(4, vec![false; 4], 160000.0, 10000.0);
        assert!((0..4).all(|il| c.rope_base_for(il) == 160000.0));
    }

    /// An out-of-range layer index must not panic and must not silently pick the sliding base.
    #[test]
    fn a_layer_index_past_the_schedule_reads_as_global_rather_than_panicking() {
        let c = cfg(2, vec![true, true], 160000.0, 10000.0);
        assert_eq!(c.rope_base_for(99), 160000.0);
    }
}
