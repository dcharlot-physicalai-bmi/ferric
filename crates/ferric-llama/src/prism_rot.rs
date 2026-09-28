//! **PrismML's rotated-basis weights, carried by the weights themselves.**
//!
//! Bonsai 2 folds a normalized blockwise Hadamard rotation (and ±1 sign flips) into its ternary
//! weights: every folded `W'` must see `H(s ⊙ x)` instead of `x` (`ferric_gguf::prism`). The fork
//! applies that inside `build_lora_mm`, the one helper its graphs route matmuls through, and then
//! *verifies the built graph* (`llama_verify_hadamard_graph`) because a path that bypasses the helper
//! runs untransformed and says nothing.
//!
//! Here the same guarantee comes from the TYPES: a folded weight is an [`RQ`] / [`RProj`] that owns
//! its input transform, and the only way to multiply by it is a method that applies that transform.
//! A call site written as `x.matmul_q(&w.wo)` against an `RQ` does not compile. And every folded name
//! in the contract must be CLAIMED by a load site that built one of these wrappers ([`Rotations::
//! verify_all_claimed`]) — the analogue of the fork's graph check, so a tensor loaded some other way
//! refuses the model instead of running raw.

use crate::qwen3::Proj;
use ferric_core::Context;
use ferric_gguf::prism::{self, HadamardContract};
use ferric_gguf::GgufSource;
use ferric_tensor::{QMatrix, Tensor};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// One activation transform: optional tiled→grouped permutation, sign flip, blockwise `H`.
pub struct RotIn {
    pub block: usize,
    pub width: usize,
    signs: Option<Tensor>,
    pub perm: Option<(usize, usize, usize)>,
}

impl RotIn {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        debug_assert_eq!(*x.shape.last().unwrap(), self.width, "rotated-basis activation width");
        // `FERRIC_PRISM_OFF=<what>`: NEGATIVE CONTROLS for scripts/bonsai2_conformance.sh — each drops
        // one piece of the transform so the gate can show it sees that piece. Never for real use.
        match std::env::var("FERRIC_PRISM_OFF").as_deref() {
            Ok("hadamard") => return x.clone(),
            Ok("signs") => return ferric_tensor::fwht::hadamard_rows(x, self.block, None, self.perm),
            Ok("perm") => return ferric_tensor::fwht::hadamard_rows(x, self.block, self.signs.as_ref(), None),
            Ok("block64") => return ferric_tensor::fwht::hadamard_rows(x, 64, self.signs.as_ref(), self.perm),
            _ => {}
        }
        ferric_tensor::fwht::hadamard_rows(x, self.block, self.signs.as_ref(), self.perm)
    }
}

/// A packed weight whose input may live in the rotated basis. `mm` is the only way to use it.
pub struct RQ { w: QMatrix, rot: Option<Arc<RotIn>> }

impl RQ {
    pub fn plain(w: QMatrix) -> RQ { RQ { w, rot: None } }
    pub fn mm(&self, x: &Tensor) -> Tensor {
        match &self.rot { Some(r) => r.apply(x).matmul_q(&self.w), None => x.matmul_q(&self.w) }
    }
    pub fn is_rotated(&self) -> bool { self.rot.is_some() }
}

/// A fused (or format-split) projection whose input may live in the rotated basis. One transform
/// feeds every part — the fork's `hadamard_memo` does the same for weights sharing an activation.
pub struct RProj { p: Proj, rot: Option<Arc<RotIn>> }

impl RProj {
    pub(crate) fn plain(p: Proj) -> RProj { RProj { p, rot: None } }
    fn input(&self, x: &Tensor) -> Tensor { match &self.rot { Some(r) => r.apply(x), None => x.clone() } }
    pub fn matmul(&self, x: &Tensor) -> Tensor { self.p.matmul(&self.input(x)) }
    pub fn gate_up_swiglu(&self, x: &Tensor, n_ff: usize) -> Tensor { self.p.gate_up_swiglu(&self.input(x), n_ff) }
    pub fn is_rotated(&self) -> bool { self.rot.is_some() }
}

/// The model's Hadamard contract bound to the device: sign vectors uploaded once per width, and the
/// set of folded names every load site has claimed.
pub struct Rotations {
    pub contract: HadamardContract,
    ctx: Arc<Context>,
    signs: Mutex<HashMap<usize, Tensor>>,
    claimed: Mutex<HashSet<String>>,
}

impl Rotations {
    /// Parse the file's contract. `Ok(None)` = an ordinary file.
    pub fn from_gguf(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Option<Rotations>, String> {
        let c = HadamardContract::from_meta(g.metadata(), |n| g.tensor(n).is_some())?;
        Ok(c.map(|contract| Rotations { contract, ctx: ctx.clone(), signs: Mutex::default(), claimed: Mutex::default() }))
    }

    fn rot_in(&self, width: usize, perm: Option<(usize, usize, usize)>) -> Result<Arc<RotIn>, String> {
        let signs = match self.contract.signs_for(width)? {
            None => None,
            Some(v) => {
                let mut m = self.signs.lock().map_err(|_| "sign cache poisoned")?;
                Some(m.entry(width).or_insert_with(|| Tensor::from_vec(&self.ctx, v, &[width])).clone())
            }
        };
        Ok(Arc::new(RotIn { block: self.contract.block_size, width, signs, perm }))
    }

    /// The input transform for the weights `names`, which are consumed together on ONE activation of
    /// `width` features. `None` if none is folded; an error if the contract folds only some of them —
    /// one activation cannot be both rotated and not for a single fused matmul.
    pub fn for_weights(&self, names: &[&str], width: usize, perm: Option<(usize, usize, usize)>) -> Result<Option<Arc<RotIn>>, String> {
        let folded: Vec<bool> = names.iter().map(|n| self.contract.is_folded(n)).collect();
        if folded.iter().all(|f| !f) { return Ok(None); }
        if !folded.iter().all(|f| *f) {
            return Err(format!("prism.hadamard folds only some of {names:?}, which this runtime consumes as one \
                                fused projection on one activation; refusing rather than rotating the rest"));
        }
        let r = self.rot_in(width, perm)?;
        let mut c = self.claimed.lock().map_err(|_| "claim set poisoned")?;
        for n in names { c.insert((*n).to_string()); }
        Ok(Some(r))
    }

    /// Host-side inverse after an embedding-row lookup, `h = s ⊙ (H z)`, when the table is latent.
    pub fn inverse_rows(&self, table: &str, rows: &mut [f32], width: usize) -> Result<(), String> {
        if !self.contract.is_inverse(table) { return Ok(()); }
        let s = self.contract.signs_for(width)?;
        if std::env::var("FERRIC_PRISM_OFF").as_deref() == Ok("hadamard") { return Ok(()); }
        let s = if std::env::var("FERRIC_PRISM_OFF").as_deref() == Ok("signs") { None } else { s };
        prism::inverse_rows(rows, width, self.contract.block_size, s);
        Ok(())
    }

    /// The fork's graph check, at load: every folded weight must have been claimed by a site that
    /// applies its transform.
    pub fn verify_all_claimed(&self) -> Result<(), String> {
        let c = self.claimed.lock().map_err(|_| "claim set poisoned")?;
        let mut missing: Vec<&String> = self.contract.folded.iter().filter(|n| !c.contains(*n)).collect();
        missing.sort();
        if missing.is_empty() { return Ok(()); }
        Err(format!("prism.hadamard folds {} weight(s) this runtime did not load through an activation \
                     transform (first: {:?}); refusing to run them untransformed", missing.len(), &missing[..missing.len().min(4)]))
    }
}

/// Load one packed weight, applying its input transform if the contract folds it.
pub fn rq(ctx: &Arc<Context>, g: &impl GgufSource, rot: Option<&Rotations>, name: &str,
          perm: Option<(usize, usize, usize)>) -> Result<RQ, String> {
    let w = crate::qwen35::qm(ctx, g, name)?;
    let width = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?.dims[0] as usize;
    Ok(RQ { w, rot: match rot { Some(r) => r.for_weights(&[name], width, perm)?, None => None } })
}

/// Load a fused projection over `names` (all consuming one activation), with its input transform.
pub fn rproj(ctx: &Arc<Context>, g: &impl GgufSource, rot: Option<&Rotations>, names: &[&str]) -> Result<RProj, String> {
    let p = Proj::load(ctx, g, names)?;
    let width = g.tensor(names[0]).ok_or_else(|| format!("no tensor '{}'", names[0]))?.dims[0] as usize;
    Ok(RProj { p, rot: match rot { Some(r) => r.for_weights(names, width, None)?, None => None } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_gguf::{parse, write::GgufWriter};

    fn file() -> ferric_gguf::Gguf {
        let mut w = GgufWriter::new("qwen35");
        let signs: Vec<i32> = (0..8).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        w.kv_u32("prism.hadamard.version", 1)
            .kv_u32("prism.hadamard.block_size", 4)
            .kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
            .kv_str("prism.hadamard.axis", "input-last-dimension")
            .kv_str("prism.hadamard.sign_mode", "explicit")
            .kv_arr_i32("prism.hadamard.sign_widths", &[8])
            .kv_arr_i32("prism.hadamard.sign_values", &signs)
            .kv_arr_str("prism.hadamard.weight_names", &["blk.0.attn_qkv.weight".into(), "blk.0.ffn_up.weight".into()])
            .tensor_f32("blk.0.attn_qkv.weight", &[8, 2], &[0.0; 16])
            .tensor_f32("blk.0.ssm_alpha.weight", &[8, 2], &[0.0; 16])
            .tensor_f32("blk.0.ffn_up.weight", &[8, 2], &[0.0; 16]);
        parse(w.finish().unwrap()).unwrap()
    }

    #[test]
    fn a_fused_projection_must_be_all_folded_or_none_and_every_fold_must_be_claimed() {
        let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU; skipped"); return };
        let ctx = Arc::new(ctx);
        let g = file();
        let r = Rotations::from_gguf(&ctx, &g).unwrap().expect("contract present");
        assert!(r.for_weights(&["blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight"], 8, None).is_err(),
                "one activation cannot be both rotated and not");
        assert!(r.for_weights(&["blk.0.ssm_alpha.weight"], 8, None).unwrap().is_none());
        assert!(r.for_weights(&["blk.0.attn_qkv.weight"], 8, None).unwrap().is_some());
        assert!(r.verify_all_claimed().is_err(), "ffn_up is folded and nothing claimed it");
        r.for_weights(&["blk.0.ffn_up.weight"], 8, None).unwrap();
        assert!(r.verify_all_claimed().is_ok());
    }

    #[test]
    fn a_rotated_weight_multiplies_the_transformed_activation() {
        let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU; skipped"); return };
        let ctx = Arc::new(ctx);
        let g = file();
        let r = Rotations::from_gguf(&ctx, &g).unwrap().unwrap();
        let w: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) / 4.0).collect(); // [out 2, in 8]
        let x: Vec<f32> = (0..8).map(|i| (i * i) as f32 / 10.0 - 1.0).collect();
        let rot = r.for_weights(&["blk.0.ffn_up.weight"], 8, None).unwrap();
        let q = RQ { w: QMatrix::from_dense(&ctx, &w, 2, 8), rot };
        let got = pollster::block_on(q.mm(&Tensor::from_vec(&ctx, &x, &[1, 8])).to_vec());
        let s = r.contract.signs_for(8).unwrap().unwrap().to_vec();
        let mut u = x.clone();
        prism::forward_rows(&mut u, 8, 4, Some(&s), None);
        for o in 0..2 {
            let want: f32 = (0..8).map(|i| w[o * 8 + i] * u[i]).sum();
            assert!((got[o] - want).abs() < 1e-5, "output {o}: {} vs W·H(s⊙x) {want}", got[o]);
            let raw: f32 = (0..8).map(|i| w[o * 8 + i] * x[i]).sum();
            assert!((got[o] - raw).abs() > 1e-3, "output {o} equals the UNtransformed product — no transform ran");
        }
    }
}
