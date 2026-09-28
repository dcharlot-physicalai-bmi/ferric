//! **AWQ** (`quant_method: awq`, `version: gemm`) — AutoAWQ checkpoints, as `dequantize_gemm` decodes them.
//!
//! Per linear: `qweight` int32 `[in, out/8]`, `qzeros` int32 `[in/group, out/8]`, `scales` `[in/group, out]`.
//! Unlike GPTQ the packing runs along the OUTPUT dimension, and not in order: AutoAWQ packs output
//! columns `8c + [0, 2, 4, 6, 1, 3, 5, 7]` into nibbles 0..7 (`order_map` in its packer), so output
//! column `8c + j` sits in nibble `[0, 4, 1, 5, 2, 6, 3, 7][j]` — `AWQ_REVERSE_ORDER`. The weight is
//! `(q - z) * s`; zeros are stored as-is (no v1 offset).
//!
//! ⚠ A second reading exists and it is wrong: `gptqmodel.utils.model_dequant.convert_awq_file` (GPTQModel
//! 7.5's offline converter) unpacks nibbles IN ORDER, without `reverse_awq_order`. Its runtime kernels
//! (`AwqTorchLinear`, `TorchAtenAwqLinear`) go through the vendored AutoAWQ `dequantize_gemm`, which
//! does reorder — that is what this reader matches, and the in-order reading is this format's negative
//! control (`FERRIC_QUANT_CONTROL=awq_order`).

use super::{bit_code, control, floats, i32s, ignored, int_spec, prefixes, storage_group, str_list, QModule, Rows, Src};
use crate::SafeTensors;

#[derive(Debug, Clone)]
pub struct Mod { group: usize }

/// `AWQ_REVERSE_ORDER`: output column `8c + j` lives in nibble `REV[j]` of word `c`.
pub const AWQ_REVERSE_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

pub(crate) fn discover(qc: &serde_json::Value, st: &SafeTensors) -> Result<Vec<QModule>, String> {
    let bits = qc["bits"].as_u64().or(qc["w_bit"].as_u64()).unwrap_or(4);
    if bits != 4 { return Err(format!("awq: {bits}-bit is not read here")) }
    let version = qc["version"].as_str().unwrap_or("gemm").to_ascii_lowercase();
    if version != "gemm" {
        return Err(format!("awq: version '{version}' packs differently from GEMM and is not read here"));
    }
    if qc["zero_point"].as_bool() == Some(false) { return Err("awq: zero_point: false is not read here".into()) }
    let gs = qc["group_size"].as_i64().or(qc["q_group_size"].as_i64()).unwrap_or(128);
    let skip = str_list(&qc["modules_to_not_convert"]);
    let mut mods = Vec::new();
    for prefix in prefixes(st, ".qweight") {
        if ignored(&skip, &prefix)? { continue }
        let qw = st.info(&format!("{prefix}.qweight")).unwrap().shape.clone();
        let sc = st.info(&format!("{prefix}.scales")).ok_or_else(|| format!("{prefix}: qweight without scales"))?.shape.clone();
        let qz = st.info(&format!("{prefix}.qzeros")).ok_or_else(|| format!("{prefix}: qweight without qzeros"))?.shape.clone();
        let (inp, out) = (qw[0], qw[1] * 8);
        let group = if gs <= 0 { inp } else { gs as usize };
        let ng = inp.div_ceil(group);
        if sc != [ng, out] || qz != [ng, out / 8] || inp % 32 != 0 {
            return Err(format!("{prefix}: qweight {qw:?} / scales {sc:?} / qzeros {qz:?} disagree for AWQ group {group}"));
        }
        let spec = int_spec(4, storage_group(group, inp, 4)?, false);
        mods.push(QModule { prefix, out, inp, spec, src: Src::Awq(Mod { group }) });
    }
    Ok(mods)
}

pub(crate) fn rows(st: &SafeTensors, m: &QModule, a: &Mod) -> Result<Rows, String> {
    let p = &m.prefix;
    let (qw, _) = i32s(st, &format!("{p}.qweight"))?;
    let (qz, _) = i32s(st, &format!("{p}.qzeros"))?;
    let (sc, ss) = floats(st, &format!("{p}.scales"))?;
    let (out, inp, ng) = (m.out, m.inp, ss[0]);
    let wpr = out / 8;
    let in_order = control("awq_order");
    let nib = |j: usize| if in_order { j } else { AWQ_REVERSE_ORDER[j] };
    let mut codes = vec![0u32; out * inp];
    for i in 0..inp {
        for o in 0..out {
            codes[o * inp + i] = bit_code(&qw[i * wpr + o / 8..i * wpr + o / 8 + 1], nib(o % 8), 4);
        }
    }
    let (mut scales, mut zeros) = (vec![0f32; out * ng], vec![0f32; out * ng]);
    for g in 0..ng {
        for o in 0..out {
            zeros[o * ng + g] = bit_code(&qz[g * wpr + o / 8..g * wpr + o / 8 + 1], nib(o % 8), 4) as f32;
            scales[o * ng + g] = sc[g * out + o];
        }
    }
    Ok(Rows { out, inp, codes, scales, zeros, ng, group: a.group, g_idx: None })
}
