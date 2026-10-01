//! **NVIDIA ModelOpt** exports — NVFP4 (and per-tensor FP8) — decoded as `modelopt`'s own
//! `NVFP4QTensor.dequantize` decodes them.
//!
//! NVFP4 per linear: `weight` u8 `[out, in/2]` (two E2M1 codes per byte, LOW nibble first),
//! `weight_scale` F8_E4M3 `[out, in/16]`, `weight_scale_2` f32 scalar. ModelOpt (0.47, `fast=False`):
//! `per_block_scale = weight_scale.float() * weight_scale_2` (one f32 rounding), then
//! `e2m1 * per_block_scale` (a second) — that order is kept: the stored GQ scale IS the f32 product.
//!
//! This is the SAME value grid as GGUF's NVFP4 (type 40): ggml stores the per-16 scale as UE4M3 and
//! halves it against a doubled E2M1 table, with no global scale — i.e. `weight_scale_2 == 1`.
//!
//! `input_scale` (static activation quantization for the FP4 GEMM) is not applied: Ferric runs these
//! weights with f32 activations, as ModelOpt's own fake-quant-free dequantize does.

use super::{floats, ignored, prefixes, storage_group, str_list, u8s, QModule, Rows, Src};
use crate::SafeTensors;
use ferric_gguf::gq::{GqKind, GqSpec};

#[derive(Debug, Clone)]
pub struct Mod { nvfp4: bool, group: usize }

pub(crate) fn discover(qc: Option<&serde_json::Value>, file: Option<&serde_json::Value>, st: &SafeTensors) -> Result<Vec<QModule>, String> {
    let q = file.map(|f| &f["quantization"]);
    let algo = q.and_then(|q| q["quant_algo"].as_str()).or(qc.and_then(|c| c["quant_algo"].as_str()))
        .ok_or("modelopt: no quant_algo")?.to_string();
    let group = q.and_then(|q| q["group_size"].as_u64()).unwrap_or(16) as usize;
    let mut skip = q.map(|q| str_list(&q["exclude_modules"])).unwrap_or_default();
    if let Some(c) = qc { skip.extend(str_list(&c["ignore"])); }
    let nvfp4 = match algo.as_str() { "NVFP4" => true, "FP8" => false, a => return Err(format!("modelopt: quant_algo '{a}' is not read here")) };
    let mut mods = Vec::new();
    let sfx = if nvfp4 { ".weight_scale_2" } else { ".weight_scale" };
    for prefix in prefixes(st, sfx) {
        if ignored(&skip, &prefix)? { continue }
        let w = st.info(&format!("{prefix}.weight")).ok_or_else(|| format!("{prefix}: scale without weight"))?.clone();
        let spec = if nvfp4 {
            if w.dtype != "U8" { return Err(format!("{prefix}.weight is {}, expected packed U8 FP4", w.dtype)) }
            GqSpec { kind: GqKind::Fp4E2m1, bits: 4, group: storage_group(group, w.shape[1] * 2, 4)?, act: false }
        } else {
            if !matches!(w.dtype.as_str(), "F8_E4M3" | "F8_E4M3FN") { continue }
            GqSpec { kind: GqKind::Fp8E4m3, bits: 8, group: storage_group(w.shape[1], w.shape[1], 8)?, act: false }
        };
        let (out, inp) = (w.shape[0], if nvfp4 { w.shape[1] * 2 } else { w.shape[1] });
        spec.valid()?;
        mods.push(QModule { prefix, out, inp, spec, src: Src::ModelOpt(Mod { nvfp4, group: if nvfp4 { group } else { inp } }) });
    }
    Ok(mods)
}

pub(crate) fn rows(st: &SafeTensors, m: &QModule, o: &Mod) -> Result<Rows, String> {
    let p = &m.prefix;
    let (out, inp) = (m.out, m.inp);
    let (w, _) = u8s(st, &format!("{p}.weight"))?;
    if o.nvfp4 {
        let (bs, _) = floats(st, &format!("{p}.weight_scale"))?;
        let (s2, _) = floats(st, &format!("{p}.weight_scale_2"))?;
        if s2.len() != 1 { return Err(format!("{p}.weight_scale_2 has {} values", s2.len())) }
        let ng = inp / o.group;
        if bs.len() != out * ng { return Err(format!("{p}.weight_scale has {} values for [{out}, {ng}]", bs.len())) }
        let scales: Vec<f32> = bs.iter().map(|&s| s * s2[0]).collect();
        let mut codes = vec![0u32; out * inp];
        // LOW nibble first (`unpacked[..., 0::2] = input & 0x0F`); `fp4_nibbles` is the swapped reading.
        let hi_first = super::control("fp4_nibbles");
        for r in 0..out { for i in 0..inp {
            let sh = 4 * ((i % 2) ^ hi_first as usize);
            let c = ((w[r * (inp / 2) + i / 2] >> sh) & 0xF) as u32;
            // ModelOpt's `e2m1_values` table holds +0 for code 8 (E2M1's -0), so its dequant yields +0.0
            // where compressed-tensors (and the GQ table) yield -0.0. Same number; the sign bit is
            // matched so the fixture comparison can stay bit-for-bit (15.9M of 440M values differed
            // by exactly this on NVFP4/Qwen3-0.6B-FP4 before).
            codes[r * inp + i] = if c == 8 { 0 } else { c };
        }}
        Ok(Rows { out, inp, codes, scales, zeros: Vec::new(), ng, group: o.group, g_idx: None })
    } else {
        let (s, _) = floats(st, &format!("{p}.weight_scale"))?;
        let scales = if s.len() == 1 { vec![s[0]; out] } else if s.len() == out { s } else { return Err(format!("{p}.weight_scale shape")) };
        Ok(Rows { out, inp, codes: w.iter().map(|&b| b as u32).collect(), scales, zeros: Vec::new(), ng: 1, group: inp, g_idx: None })
    }
}
