//! **compressed-tensors** (`quant_method: compressed-tensors`) — llm-compressor / neuralmagic / RedHatAI
//! checkpoints, decoded as the `compressed-tensors` library decompresses them.
//!
//! Four storage formats, chosen per config group (`format`, falling back to the global one):
//!
//! | format                  | weight tensors                                              | value            |
//! |-------------------------|-------------------------------------------------------------|------------------|
//! | `pack-quantized`        | `weight_packed` i32 `[out, ⌈in·b/32⌉]`, `weight_shape`       | `(q − zp) · s`   |
//! | `int-quantized`/`naive` | `weight` i8 `[out, in]`                                      | `(q − zp) · s`   |
//! | `float-quantized`       | `weight` F8_E4M3 `[out, in]`                                  | `e4m3(q) · s`    |
//! | `nvfp4-pack-quantized`  | `weight_packed` u8 `[out, in/2]`, `weight_global_scale` f32   | `e2m1(q) · (s / gs)` |
//!
//! with `weight_scale` shaped by the strategy: `tensor` `[1]`, `channel` `[out, 1]`, `group`
//! `[out, in/group]`, `block` `[⌈out/bh⌉, ⌈in/bw⌉]`, `tensor_group` (NVFP4) `[out, in/16]` in FP8.
//!
//! ## Packing (`pack_to_int32`)
//!
//! Each ROW is one little-endian bitstream: value `i` at bit `i·b` (dense, crossing words — the 0.19
//! layout; at 4 and 8 bits every version's layout is the same one). Values are SIGNED and stored with
//! `+2^(b−1)`, so the stored code is unsigned. Asymmetric zero points are packed the same way along
//! dim 0 (`[⌈out·b/32⌉, groups]`, each column a bitstream over the output rows), also offset. In the
//! unsigned domain the weight is `s · (u − (zp + 2^(b−1)))` — the same integer difference, exactly.
//!
//! ## act-order: `actorder: group` and `weight_g_idx`
//!
//! Checkpoints written by compressed-tensors <= 0.11 with `actorder: group` carry `weight_g_idx`; those
//! versions decompress by sorting the columns by `g_idx` and dequantizing consecutive groups. Under the
//! uniform maps GPTQ always produces that is the same as "value `i` uses group `g_idx[i]`", which is
//! what is stored here (non-uniform maps are refused — see `act_gidx`).
//!
//! ⚠ compressed-tensors 0.12+ (0.19 checked) REFUSES these checkpoints: `actorder='group' has been
//! removed` raises in its config validator, so the current library cannot load e.g.
//! `RedHatAI/Qwen2.5-0.5B-quantized.w4a16` at all. The reference for them is 0.11's decompress.
//!
//! ## NVFP4 arithmetic
//!
//! `_dequantize` divides first — `scale / global_scale` in f32 — then multiplies the E2M1 value: two
//! roundings, and that order is kept (the stored GQ scale IS the f32 quotient).
//!
//! ## Negative control (`FERRIC_QUANT_CONTROL`)
//!
//! `ct_offset` drops the `2^(b−1)` from the zero, i.e. reads the stored unsigned code as the signed value.

use super::{act_gidx, bit_code, control, floats, i32s, ignored, ints, prefixes, storage_group, str_list, trivial_gidx, u8s, QModule, Rows, Src};
use crate::SafeTensors;
use ferric_gguf::gq::{GqKind, GqSpec};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fmt { PackInt, NaiveInt, Float8, Nvfp4 }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strategy { Tensor, Channel, Group, Block }

#[derive(Debug, Clone)]
pub struct Mod { fmt: Fmt, bits: u32, strategy: Strategy, group: usize, block_rows: usize, symmetric: bool, act: bool }

struct Scheme { targets: Vec<String>, w: serde_json::Value, fmt: String }

pub(crate) fn discover(qc: &serde_json::Value, st: &SafeTensors) -> Result<Vec<QModule>, String> {
    let status = qc["quantization_status"].as_str().unwrap_or("compressed");
    if status != "compressed" {
        return Err(format!("compressed-tensors: quantization_status '{status}' — only 'compressed' checkpoints hold quantized codes"));
    }
    if qc["sparsity_config"].as_object().is_some_and(|s| !s.is_empty() && s.get("format").and_then(|f| f.as_str()) != Some("dense")) {
        return Err("compressed-tensors: a sparsity_config is set; sparse compression is not read here".into());
    }
    if qc["transform_config"].as_object().is_some_and(|t| !t.is_empty()) {
        return Err("compressed-tensors: transform_config (rotations) is set and would change the forward — not read here".into());
    }
    let global_fmt = qc["format"].as_str().unwrap_or("dense").to_string();
    let groups = qc["config_groups"].as_object().ok_or("compressed-tensors: no config_groups")?;
    let schemes: Vec<Scheme> = groups.values().map(|g| Scheme {
        targets: str_list(&g["targets"]),
        w: g["weights"].clone(),
        fmt: g["format"].as_str().map(str::to_string).unwrap_or_else(|| global_fmt.clone()),
    }).collect();
    let ignore = str_list(&qc["ignore"]);
    let mut cands = prefixes(st, ".weight_packed");
    for p in prefixes(st, ".weight_scale") {
        if st.info(&format!("{p}.weight")).is_some() && !cands.contains(&p) { cands.push(p); }
    }
    cands.sort();
    let mut mods = Vec::new();
    for prefix in cands {
        if ignored(&ignore, &prefix)? { continue }
        let sch = pick(&schemes, &prefix)?;
        let w = &sch.w;
        if w.is_null() { continue }
        let bits = w["num_bits"].as_u64().ok_or_else(|| format!("{prefix}: weights.num_bits missing"))? as u32;
        let ty = w["type"].as_str().unwrap_or("int");
        let symmetric = w["symmetric"].as_bool().unwrap_or(true);
        if w["dynamic"].as_bool() == Some(true) { return Err(format!("{prefix}: dynamic WEIGHT quantization has no stored codes")) }
        let fmt = match (sch.fmt.as_str(), ty) {
            ("pack-quantized", "int") => Fmt::PackInt,
            ("int-quantized" | "naive-quantized", "int") => Fmt::NaiveInt,
            ("float-quantized" | "naive-quantized", "float") if bits == 8 => Fmt::Float8,
            ("nvfp4-pack-quantized", "float") if bits == 4 => Fmt::Nvfp4,
            (f, t) => return Err(format!("{prefix}: compressed-tensors format '{f}' with {t}{bits} weights is not read here")),
        };
        let (out, inp) = match fmt {
            Fmt::PackInt => { let s = ints(st, &format!("{prefix}.weight_shape"))?; (s[0] as usize, s[1] as usize) }
            Fmt::Nvfp4 => { let s = &st.info(&format!("{prefix}.weight_packed")).unwrap().shape; (s[0], s[1] * 2) }
            _ => { let s = &st.info(&format!("{prefix}.weight")).ok_or_else(|| format!("{prefix}.weight missing"))?.shape; (s[0], s[1]) }
        };
        let strategy = match w["strategy"].as_str().unwrap_or("tensor") {
            "tensor" => Strategy::Tensor, "channel" => Strategy::Channel,
            "group" | "tensor_group" => Strategy::Group, "block" => Strategy::Block,
            s => return Err(format!("{prefix}: weight strategy '{s}' is not read here")),
        };
        let (group, block_rows) = match strategy {
            Strategy::Tensor | Strategy::Channel => (inp, 1),
            Strategy::Group => (w["group_size"].as_u64().ok_or_else(|| format!("{prefix}: group strategy without group_size"))? as usize, 1),
            Strategy::Block => {
                let b = w["block_structure"].as_array().ok_or_else(|| format!("{prefix}: block strategy without block_structure"))?;
                (b[1].as_u64().unwrap_or(0) as usize, b[0].as_u64().unwrap_or(0) as usize)
            }
        };
        if group == 0 || block_rows == 0 || inp % group != 0 { return Err(format!("{prefix}: group {group} does not tile {inp} inputs")) }
        if !symmetric && matches!(fmt, Fmt::Float8 | Fmt::Nvfp4) { return Err(format!("{prefix}: asymmetric float weights are not read here")) }
        let ng = inp / group;
        let act = match (fmt, st.info(&format!("{prefix}.weight_g_idx"))) {
            (Fmt::PackInt, Some(_)) => {
                let g = ints(st, &format!("{prefix}.weight_g_idx"))?;
                let a = g.len() == inp && !trivial_gidx(&g, group);
                if a { act_gidx(&g, ng, group)?; }
                a && !control("gidx")
            }
            (_, Some(_)) => return Err(format!("{prefix}: weight_g_idx on a {fmt:?} weight is not read here")),
            _ => false,
        };
        let spec = match fmt {
            Fmt::PackInt | Fmt::NaiveInt => {
                let b = if fmt == Fmt::NaiveInt { 8 } else { bits };
                // ⚠ 2/4/8 only. compressed-tensors <= 0.11 packed `32 // bits` values per word (3-bit:
                // ten per word, two bits wasted); 0.12+ packs a dense bitstream. At 2, 4 and 8 bits the
                // two layouts are the same bytes; at any other width the checkpoint does not say which
                // library wrote it, so it is refused rather than read with a guessed layout.
                if !matches!(b, 2 | 4 | 8) { return Err(format!("{prefix}: {b}-bit pack-quantized is ambiguous across compressed-tensors versions — not read here")) }
                let g = if act { group as u32 } else { storage_group(group, inp, b)? };
                GqSpec { kind: GqKind::Int, bits: b, group: g, act }
            }
            Fmt::Float8 => GqSpec { kind: GqKind::Fp8E4m3, bits: 8, group: storage_group(group, inp, 8)?, act: false },
            Fmt::Nvfp4 => GqSpec { kind: GqKind::Fp4E2m1, bits: 4, group: storage_group(group, inp, 4)?, act: false },
        };
        spec.valid()?;
        mods.push(QModule { prefix, out, inp, spec, src: Src::Ct(Mod { fmt, bits, strategy, group, block_rows, symmetric, act }) });
    }
    Ok(mods)
}

/// The config group whose targets cover this module. `Linear` covers every candidate (only linears
/// carry these tensors); `re:.*<literal>` is honoured; anything else is refused.
fn pick<'a>(schemes: &'a [Scheme], prefix: &str) -> Result<&'a Scheme, String> {
    if schemes.len() == 1 { return Ok(&schemes[0]) }
    let mut hit = None;
    for s in schemes {
        for t in &s.targets {
            let m = if t == "Linear" { true } else { ignored(std::slice::from_ref(t), prefix)? };
            if m {
                if hit.is_some() { return Err(format!("{prefix}: matched by more than one config group")) }
                hit = Some(s);
            }
        }
    }
    hit.ok_or_else(|| format!("{prefix}: no config group targets this module"))
}

/// Per-row scales `[out, ng]` (ng = in / group) from the strategy's scale shape.
fn row_scales(v: &[f32], shape: &[usize], m: &Mod, out: usize, inp: usize) -> Result<(Vec<f32>, usize), String> {
    let ng = inp / m.group;
    let mut s = vec![0f32; out * ng];
    match m.strategy {
        Strategy::Tensor => { if v.len() != 1 { return Err(format!("tensor scale has {} values", v.len())) } s.fill(v[0]); }
        Strategy::Channel => {
            if v.len() != out { return Err(format!("channel scale {shape:?} for {out} rows")) }
            // `scale_rows`: the per-channel scale read as per-tensor (row 0's for every row)
            for o in 0..out { s[o] = v[if control("scale_rows") { 0 } else { o }]; }
        }
        Strategy::Group => {
            if v.len() != out * ng { return Err(format!("group scale {shape:?} for [{out}, {ng}]")) }
            s.copy_from_slice(v);
        }
        Strategy::Block => {
            let (br, bc) = (out.div_ceil(m.block_rows), ng);
            if v.len() != br * bc { return Err(format!("block scale {shape:?} for [{br}, {bc}]")) }
            let rows_of = |o: usize| if control("scale_rows") { 0 } else { o / m.block_rows };
            for o in 0..out { for g in 0..ng { s[o * ng + g] = v[rows_of(o) * bc + g]; } }
        }
    }
    Ok((s, ng))
}

pub(crate) fn rows(st: &SafeTensors, m: &QModule, c: &Mod) -> Result<Rows, String> {
    let p = &m.prefix;
    let (out, inp) = (m.out, m.inp);
    let (sv, ss) = floats(st, &format!("{p}.weight_scale"))?;
    let (mut scales, ng) = row_scales(&sv, &ss, c, out, inp)?;
    let mut codes = vec![0u32; out * inp];
    let mut zeros = Vec::new();
    match c.fmt {
        Fmt::PackInt => {
            let b = c.bits as usize;
            let (w, ws) = i32s(st, &format!("{p}.weight_packed"))?;
            let wpr = ws[1];
            if wpr * 32 < inp * b { return Err(format!("{p}.weight_packed {ws:?} holds fewer than {inp} {b}-bit codes")) }
            for o in 0..out { for i in 0..inp { codes[o * inp + i] = bit_code(&w[o * wpr..(o + 1) * wpr], i, b); } }
            let offset = if control("ct_offset") { 0.0 } else { (1u32 << (b - 1)) as f32 };
            zeros = vec![offset; out * ng];
            if !c.symmetric {
                // packed along dim 0: column g is a bitstream over the output rows
                let (z, zs) = i32s(st, &format!("{p}.weight_zero_point"))?;
                let (zr, zc) = (zs[0], zs.get(1).copied().unwrap_or(1));
                if zc != ng || zr * 32 < out * b { return Err(format!("{p}.weight_zero_point {zs:?} for [{out}, {ng}]")) }
                let mut col = vec![0u32; zr + 1];
                for g in 0..ng {
                    for k in 0..zr { col[k] = z[k * zc + g]; }
                    // stored = zp + 2^(b-1): the unsigned code IS the zero in the unsigned domain
                    for o in 0..out { zeros[o * ng + g] = bit_code(&col, o, b) as f32 + offset - (1u32 << (b - 1)) as f32; }
                }
            }
        }
        Fmt::NaiveInt => {
            let (w, _) = u8s(st, &format!("{p}.weight"))?;
            for (d, &q) in codes.iter_mut().zip(w.iter()) { *d = (q as i8 as i32 + 128) as u32; }
            zeros = vec![if control("ct_offset") { 0.0 } else { 128.0 }; out * ng];
            if !c.symmetric {
                let zp = ints(st, &format!("{p}.weight_zero_point"))?;
                let (zv, _) = row_scales(&zp.iter().map(|&x| x as f32).collect::<Vec<_>>(), &[zp.len()], c, out, inp)?;
                for (z, v) in zeros.iter_mut().zip(zv) { *z += v; }
            }
        }
        Fmt::Float8 => {
            let (w, _) = u8s(st, &format!("{p}.weight"))?;
            for (d, &q) in codes.iter_mut().zip(w.iter()) { *d = q as u32; }
        }
        Fmt::Nvfp4 => {
            let (w, _) = u8s(st, &format!("{p}.weight_packed"))?;
            let hi_first = control("fp4_nibbles");
            for o in 0..out { for i in 0..inp {
                let sh = 4 * ((i % 2) ^ hi_first as usize);
                codes[o * inp + i] = ((w[o * (inp / 2) + i / 2] >> sh) & 0xF) as u32;
            }}
            let (gs, _) = floats(st, &format!("{p}.weight_global_scale"))?;
            if gs.len() != 1 { return Err(format!("{p}.weight_global_scale has {} values", gs.len())) }
            for s in scales.iter_mut() { *s /= gs[0]; }
        }
    }
    let g_idx = if c.act { Some(act_gidx(&ints(st, &format!("{p}.weight_g_idx"))?, ng, c.group)?) } else { None };
    Ok(Rows { out, inp, codes, scales, zeros, ng, group: c.group, g_idx })
}
