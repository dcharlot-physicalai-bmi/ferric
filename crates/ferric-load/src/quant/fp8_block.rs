//! **Block-scaled FP8** (`quant_method: fp8`) — DeepSeek-V3 / Qwen3 `-FP8` checkpoints, as transformers'
//! fine-grained FP8 integration loads them.
//!
//! Per linear: `weight` F8_E4M3 `[out, in]` and `weight_scale_inv` (f32 or bf16) `[out/bm, in/bn]`,
//! `weight_block_size: [bm, bn]` (128x128 everywhere seen). Despite the name the scale is a MULTIPLIER:
//! `w = e4m3(q) * scale_inv[o / bm][i / bn]`. transformers (`Fp8Dequantize._dequantize_one`, the path it
//! takes on any machine without an FP8 GPU) derives the block from the two SHAPES — `rows // scale_rows`
//! — refusing a grid that does not divide, forms `q.float() * s.float()` in f32 and casts to the model
//! dtype. At float32 that cast is the identity, so the f32 product here is its value bit for bit.
//!
//! Activation quantization (`activation_scheme: dynamic`) belongs to the FP8 GEMM kernel on GPUs that
//! have one; transformers' dequantized path runs the activations unquantized, and so does this.

use super::{floats, ignored, prefixes, storage_group, str_list, u8s, QModule, Rows, Src};
use crate::SafeTensors;
use ferric_gguf::gq::{GqKind, GqSpec};

#[derive(Debug, Clone)]
pub struct Mod { bm: usize, bn: usize }

pub(crate) fn discover(qc: &serde_json::Value, st: &SafeTensors) -> Result<Vec<QModule>, String> {
    if let Some(f) = qc["fmt"].as_str() { if f != "e4m3" { return Err(format!("fp8: fmt '{f}' is not read here")) } }
    let skip = str_list(&qc["modules_to_not_convert"]);
    let mut mods = Vec::new();
    for prefix in prefixes(st, ".weight_scale_inv") {
        if ignored(&skip, &prefix)? { continue }
        let w = st.info(&format!("{prefix}.weight")).ok_or_else(|| format!("{prefix}: scale without weight"))?;
        if !matches!(w.dtype.as_str(), "F8_E4M3" | "F8_E4M3FN") { return Err(format!("{prefix}.weight is {}, not FP8", w.dtype)) }
        let s = &st.info(&format!("{prefix}.weight_scale_inv")).unwrap().shape;
        let (out, inp) = (w.shape[0], w.shape[1]);
        let (sr, sc) = match s.as_slice() { [a, b] => (*a, *b), [] | [1] => (1, 1), _ => return Err(format!("{prefix}: scale shape {s:?}")) };
        if out % sr != 0 || inp % sc != 0 {
            return Err(format!("{prefix}: weight [{out}, {inp}] not divisible by scale grid [{sr}, {sc}] (transformers refuses this too)"));
        }
        let (bm, bn) = (out / sr, inp / sc);
        if let Some(b) = qc["weight_block_size"].as_array() {
            let want = (b[0].as_u64().unwrap_or(0) as usize, b[1].as_u64().unwrap_or(0) as usize);
            if (bm, bn) != want && (sr, sc) != (1, 1) {
                return Err(format!("{prefix}: scale grid implies blocks [{bm}, {bn}], config says {want:?}"));
            }
        }
        let spec = GqSpec { kind: GqKind::Fp8E4m3, bits: 8, group: storage_group(bn, inp, 8)?, act: false };
        spec.valid()?;
        mods.push(QModule { prefix, out, inp, spec, src: Src::Fp8(Mod { bm, bn }) });
    }
    Ok(mods)
}

pub(crate) fn rows(st: &SafeTensors, m: &QModule, f: &Mod) -> Result<Rows, String> {
    let p = &m.prefix;
    let (out, inp) = (m.out, m.inp);
    let (w, _) = u8s(st, &format!("{p}.weight"))?;
    let (s, _) = floats(st, &format!("{p}.weight_scale_inv"))?;
    let ng = inp / f.bn;
    let mut scales = vec![0f32; out * ng];
    // `scale_rows` (negative control): every row reads the FIRST block-row's scales.
    let brow = |o: usize| if super::control("scale_rows") { 0 } else { o / f.bm };
    for o in 0..out { for g in 0..ng { scales[o * ng + g] = s[brow(o) * ng + g]; } }
    Ok(Rows { out, inp, codes: w.iter().map(|&b| b as u32).collect(), scales, zeros: Vec::new(), ng, group: f.bn, g_idx: None })
}
