//! **GPTQ** (`quant_method: gptq`) — AutoGPTQ / GPTQModel checkpoints, as transformers + GPTQModel load them.
//!
//! Per linear: `qweight` int32 `[in*bits/32, out]` (each OUTPUT column is a little-endian bitstream of
//! its `in` codes — for 3 bits, 32 codes span three words and codes 10 and 21 straddle a word
//! boundary), `qzeros` int32 `[groups, out*bits/32]` (each GROUP row a bitstream of `out` zero codes),
//! `scales` `[groups, out]`, `g_idx` int32 `[in]`. The weight is `scales[g] * (q - zeros[g])` with
//! `g = g_idx[i]` — act-order (`desc_act`) is just a `g_idx` that is not `i / group_size`.
//!
//! ## The v1 zero-point, as the runtime applies it
//!
//! `checkpoint_format: gptq` ("v1", the default and what AutoGPTQ wrote) stores every zero MINUS ONE.
//! GPTQModel's runtime converts v1 to v2 by adding a constant to each packed WORD
//! (`convert_gptq_v1_to_v2_format_module`: `+0x11111111` at 4 bits, `0x55555555` at 2, `0x01010101`
//! at 8) — so a field holding the v1 encoding of zero (all ones) CARRIES into its neighbour — and at 3
//! bits it shifts field by field with a mask. That is what this reader does.
//!
//! ⚠ A second reading exists in the same library: `gptqmodel.utils.model_dequant.convert_gptq_file`
//! (the offline converter) shifts every width field by field with a mask, so the two disagree exactly
//! where a v1 field is all ones. For symmetric checkpoints (every published one checked here: zero 8 at
//! 4 bits, stored 7) no field is all ones and both readings agree bit for bit.
//!
//! ## Negative controls (`FERRIC_QUANT_CONTROL`)
//!
//! `gptq_zero` skips the v1 +1; `gidx` ignores `g_idx` (reads group `i / group_size`).

use super::{act_gidx, bit_code, control, floats, i32s, int_spec, ints, prefixes, storage_group, trivial_gidx, QModule, Rows, Src};
use crate::SafeTensors;

#[derive(Debug, Clone)]
pub struct Mod { bits: u32, group: usize, v1: bool, act: bool }

pub(crate) fn discover(qc: &serde_json::Value, st: &SafeTensors) -> Result<Vec<QModule>, String> {
    let bits = qc["bits"].as_u64().ok_or("gptq: quantization_config has no bits")? as u32;
    if !matches!(bits, 2 | 3 | 4 | 8) { return Err(format!("gptq: {bits}-bit is not read here (2, 3, 4, 8)")) }
    let fmt = qc["checkpoint_format"].as_str().or(qc["format"].as_str()).unwrap_or("gptq").to_ascii_lowercase();
    let v1 = match fmt.as_str() {
        "gptq" => true,
        "gptq_v2" => false,
        other => return Err(format!("gptq: checkpoint_format '{other}' (planar / other layouts) is not read here")),
    };
    if let Some(p) = qc["pack_dtype"].as_str() { if p != "int32" { return Err(format!("gptq: pack_dtype {p} is not read here")) } }
    if qc["dynamic"].as_object().is_some_and(|d| !d.is_empty()) {
        return Err("gptq: per-module `dynamic` overrides are not read here — refusing rather than apply one config to all".into());
    }
    let gs = qc["group_size"].as_i64().unwrap_or(128);
    let mut mods = Vec::new();
    for prefix in prefixes(st, ".qweight") {
        let qw = st.info(&format!("{prefix}.qweight")).unwrap().shape.clone();
        let sc = st.info(&format!("{prefix}.scales")).ok_or_else(|| format!("{prefix}: qweight without scales"))?.shape.clone();
        let qz = st.info(&format!("{prefix}.qzeros")).ok_or_else(|| format!("{prefix}: qweight without qzeros"))?.shape.clone();
        if qw.len() != 2 || sc.len() != 2 || qz.len() != 2 { return Err(format!("{prefix}: GPTQ tensors must be 2-D")) }
        let (out, inp) = (qw[1], qw[0] * 32 / bits as usize);
        if (qw[0] * 32) % bits as usize != 0 || inp % 32 != 0 { return Err(format!("{prefix}: qweight rows {} do not pack {bits}-bit codes", qw[0])) }
        let group = if gs <= 0 { inp } else { gs as usize };
        let ng = sc[0];
        if sc[1] != out || ng != inp.div_ceil(group) || qz[0] != ng || qz[1] * 32 != out * bits as usize {
            return Err(format!("{prefix}: qweight {qw:?} / scales {sc:?} / qzeros {qz:?} disagree for {bits}-bit, group {group}"));
        }
        let g_name = format!("{prefix}.g_idx");
        let act = match st.info(&g_name) {
            Some(_) => {
                let g = ints(st, &g_name)?;
                if g.len() != inp { return Err(format!("{g_name}: {} entries for {inp} inputs", g.len())) }
                let a = !trivial_gidx(&g, group);
                if a { act_gidx(&g, ng, group)?; }
                a && !control("gidx")
            }
            None => false,
        };
        let spec = if act { int_spec(bits, group as u32, true) } else { int_spec(bits, storage_group(group, inp, bits)?, false) };
        spec.valid()?;
        mods.push(QModule { prefix, out, inp, spec, src: Src::Gptq(Mod { bits, group, v1, act }) });
    }
    Ok(mods)
}

pub(crate) fn rows(st: &SafeTensors, m: &QModule, g: &Mod) -> Result<Rows, String> {
    let p = &m.prefix;
    let bits = g.bits as usize;
    let (qw, _) = i32s(st, &format!("{p}.qweight"))?;
    let (mut qz, zs) = i32s(st, &format!("{p}.qzeros"))?;
    let (sc, ss) = floats(st, &format!("{p}.scales"))?;
    let (out, inp, ng, zw) = (m.out, m.inp, ss[0], zs[1]);
    let kq = inp * bits / 32;
    // v1 -> v2 exactly as GPTQModel's runtime does it: a packed-WORD add at 2/4/8 bits.
    let shift_fields = g.v1 && !control("gptq_zero");
    if shift_fields && bits != 3 {
        let c: u32 = match bits { 2 => 0x5555_5555, 4 => 0x1111_1111, _ => 0x0101_0101 };
        for w in qz.iter_mut() { *w = w.wrapping_add(c); }
    }
    let mut codes = vec![0u32; out * inp];
    let mut col = vec![0u32; kq + 1];
    for o in 0..out {
        for k in 0..kq { col[k] = qw[k * out + o]; }
        for i in 0..inp { codes[o * inp + i] = bit_code(&col, i, bits); }
    }
    let (mut scales, mut zeros) = (vec![0f32; out * ng], vec![0f32; out * ng]);
    for gi in 0..ng {
        let row = &qz[gi * zw..(gi + 1) * zw];
        for o in 0..out {
            let mut z = bit_code(row, o, bits);
            // 3-bit: GPTQModel shifts field by field (a word add would carry across straddled fields).
            if shift_fields && bits == 3 { z = (z + 1) & 7; }
            zeros[o * ng + gi] = z as f32;
            scales[o * ng + gi] = sc[gi * out + o];
        }
    }
    let g_idx = if g.act { Some(act_gidx(&ints(st, &format!("{p}.g_idx"))?, ng, g.group)?) } else { None };
    Ok(Rows { out, inp, codes, scales, zeros, ng, group: g.group, g_idx })
}
