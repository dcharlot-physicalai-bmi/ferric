//! **Quantized safetensors checkpoints** — GPTQ, AWQ, compressed-tensors, FP8, ModelOpt NVFP4 — read as
//! published, each decoded the way ITS defining library decodes it.
//!
//! These are the formats people actually publish quantized weights in on the Hub: `quant_method`
//! `gptq` (AutoGPTQ / GPTQModel), `awq` (AutoAWQ), `compressed-tensors` (neuralmagic / llm-compressor),
//! `fp8` (DeepSeek/Qwen block-scaled FP8 as transformers loads it) and NVIDIA ModelOpt's NVFP4. Before
//! this module, `HfCheckpoint` refused every one of them: no reader mapped a `qweight`, and FP8 was
//! decoded but refused for want of a kernel.
//!
//! ## One target form
//!
//! Every format is re-laid, LOSSLESSLY, into [`ferric_gguf::gq`]'s parametric form — integer / FP8 /
//! FP4 codes, a scale and optional zero per group, optional per-value group indices — and presented to
//! the runtime under a synthetic ggml type id that carries the geometry. `ferric-tensor::gq` runs it
//! packed. The codes are never requantized and never widened: a GPTQ int4 weight stays 4 bits.
//!
//! ## What "decoded the way the library decodes it" means here
//!
//! The weight each format DEFINES is `s * (q - z)` / `s * e4m3(q)` / `s * e2m1(q)` with the format's
//! own scale — and the libraries compute exactly that, in the dtype they are run at. At float32 the
//! product is exact for the integer formats (an f16 scale times a small integer) and one IEEE rounding
//! for the float ones, and Ferric forms the identical f32 value. At float16/bfloat16 (their default
//! compute dtype) the libraries round that same product once more to the compute dtype; that rounding
//! is theirs, and the gates check it as `round(ferric) == library`. See `scripts/quant_formats_conformance.sh`.
//!
//! Each format module states where it and its library disagree with a SECOND reading of the same
//! format (there are several: gptqmodel's offline AWQ converter skips the AWQ nibble order; its offline
//! GPTQ v1 zero-point shift masks where its runtime carries; compressed-tensors >= 0.12 refuses the
//! `actorder: group` checkpoints its older versions wrote).

use crate::{fp8, SafeTensors};
use ferric_gguf::gq::{deq_gq_rows, encode_row, GqKind, GqSpec};
use std::collections::BTreeMap;
use std::path::Path;

pub mod awq;
pub mod ctensors;
pub mod fp8_block;
pub mod gptq;
pub mod modelopt;

/// One quantized linear, as the runtime sees it: a `[out, in]` weight under a GQ type id.
#[derive(Debug, Clone)]
pub struct QModule {
    /// The module path, e.g. `model.layers.0.self_attn.q_proj`. Its weight is presented as `<prefix>.weight`.
    pub prefix: String,
    pub out: usize,
    pub inp: usize,
    pub spec: GqSpec,
    /// Format-specific facts the encoder needs (group size in the SOURCE, strategy, ...).
    pub(crate) src: Src,
}

#[derive(Debug, Clone)]
pub(crate) enum Src {
    Gptq(gptq::Mod),
    Awq(awq::Mod),
    Ct(ctensors::Mod),
    Fp8(fp8_block::Mod),
    ModelOpt(modelopt::Mod),
}

/// A quantized checkpoint's quantization, detected from its config.
pub struct Quant {
    /// `gptq`, `awq`, `compressed-tensors`, `fp8`, `modelopt`.
    pub method: String,
    /// One line on what was detected, for logs and the gate.
    pub summary: String,
    modules: BTreeMap<String, QModule>,
}

/// `FERRIC_QUANT_CONTROL=<name>` switches on ONE deliberately wrong reading, for the negative controls
/// in `scripts/quant_formats_conformance.sh`: each must move the logits >= 20x the clean distance, or
/// the gate has not shown it can see the mechanism. Never set outside that gate.
pub(crate) fn control(name: &str) -> bool {
    std::env::var("FERRIC_QUANT_CONTROL").is_ok_and(|v| v == name)
}

impl Quant {
    /// Read `quantization_config` (or ModelOpt's `hf_quant_config.json`). `Ok(None)` for an unquantized
    /// checkpoint; `Err` for a quantized one this reader will not decode — never a silent fallthrough to
    /// "load the tensors as they are".
    pub fn detect(dir: &Path, cfg: &serde_json::Value, st: &SafeTensors) -> Result<Option<Quant>, String> {
        let qc = cfg.get("quantization_config").or_else(|| cfg.get("compression_config"));
        let modelopt_file = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
        let method = qc.and_then(|q| q["quant_method"].as_str()).map(str::to_string);
        let is_modelopt = modelopt_file.is_some()
            || qc.is_some_and(|q| q["quant_library"].as_str() == Some("modelopt") || q["producer"]["name"].as_str() == Some("modelopt"));
        let (method, modules) = match (method.as_deref(), is_modelopt) {
            (_, true) => ("modelopt".to_string(), modelopt::discover(qc, modelopt_file.as_ref(), st)?),
            (Some("gptq"), _) => ("gptq".into(), gptq::discover(qc.unwrap(), st)?),
            (Some("awq"), _) => ("awq".into(), awq::discover(qc.unwrap(), st)?),
            (Some("compressed-tensors"), _) => ("compressed-tensors".into(), ctensors::discover(qc.unwrap(), st)?),
            (Some("fp8"), _) => ("fp8".into(), fp8_block::discover(qc.unwrap(), st)?),
            (Some(other), _) => return Err(format!("quantization_config.quant_method '{other}' has no reader here")),
            (None, false) => {
                if qc.is_some() { return Err("quantization_config without a quant_method".into()) }
                // Unquantized — but a checkpoint with GPTQ-style tensors and no config is refused, not guessed.
                if st.names().any(|n| n.ends_with(".qweight") || n.ends_with(".weight_packed")) {
                    return Err("the checkpoint holds packed quantized tensors but config.json has no \
                                quantization_config — refusing to guess their format".into());
                }
                return Ok(None);
            }
        };
        if modules.is_empty() { return Err(format!("{method}: no quantized modules found")) }
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for m in &modules { *kinds.entry(format!("{:?}{}b g{}{}", m.spec.kind, m.spec.bits, m.spec.group,
                                                  if m.spec.act { " act-order" } else { "" })).or_default() += 1; }
        let summary = format!("{method}: {} quantized linears ({})", modules.len(),
                              kinds.iter().map(|(k, n)| format!("{n}x {k}")).collect::<Vec<_>>().join(", "));
        Ok(Some(Quant { method, summary, modules: modules.into_iter().map(|m| (format!("{}.weight", m.prefix), m)).collect() }))
    }

    pub fn modules(&self) -> impl Iterator<Item = &QModule> { self.modules.values() }
    /// The module whose presented weight is `weight_name` (`<prefix>.weight`).
    pub fn module(&self, weight_name: &str) -> Option<&QModule> { self.modules.get(weight_name) }

    /// Make each presented `<prefix>.weight` visible to name lookups, so architecture maps that check a
    /// weight exists find the quantized one. The entry's dtype is `GQ` — [`SafeTensors::get`] refuses
    /// it (it has no bytes of its own); only [`Quant::gq_bytes`] / [`Quant::dequant`] can read it.
    pub(crate) fn register_virtual(&self, st: &mut SafeTensors) {
        for (name, m) in &self.modules {
            // ⚠ Never over an existing entry: for FP8 / int8 / ModelOpt the presented name IS a stored
            // tensor (the codes), which the format reader must still be able to read.
            if st.tensors.contains_key(name) { continue }
            let e = crate::Entry { dtype: "GQ".into(), shape: vec![m.out, m.inp], shard: 0, start: 0, end: 0 };
            st.tensors.insert(name.clone(), e);
        }
    }

    /// The presented bytes: GQ rows (`ferric_gguf::gq`).
    pub fn gq_bytes(&self, st: &SafeTensors, weight_name: &str) -> Result<Vec<u8>, String> {
        let m = self.module(weight_name).ok_or_else(|| format!("{weight_name} is not a quantized weight"))?;
        let rows = match &m.src {
            Src::Gptq(g) => gptq::rows(st, m, g)?,
            Src::Awq(a) => awq::rows(st, m, a)?,
            Src::Ct(c) => ctensors::rows(st, m, c)?,
            Src::Fp8(f) => fp8_block::rows(st, m, f)?,
            Src::ModelOpt(o) => modelopt::rows(st, m, o)?,
        };
        rows.encode(&m.spec)
    }

    /// The weight, `[out, in]` row-major f32, exactly as the defining library forms it at float32.
    pub fn dequant(&self, st: &SafeTensors, weight_name: &str) -> Result<Vec<f32>, String> {
        let m = self.module(weight_name).ok_or_else(|| format!("{weight_name} is not a quantized weight"))?;
        deq_gq_rows(&self.gq_bytes(st, weight_name)?, m.out, m.inp, &m.spec)
    }
}

/// A decoded module in the format-neutral shape the encoder takes: per output row, the codes of every
/// input value, and per GROUP of the row a scale and zero; plus per-value group indices (act-order).
pub(crate) struct Rows {
    pub out: usize,
    pub inp: usize,
    /// `codes[o * inp + i]`
    pub codes: Vec<u32>,
    /// `scales[o * ng + g]`, `zeros[o * ng + g]` (zeros empty for float kinds)
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
    pub ng: usize,
    /// Source group size: value `i` belongs to group `i / group` unless `g_idx` says otherwise.
    pub group: usize,
    /// act-order: `g_idx[i]`, shared by every row of the module.
    pub g_idx: Option<Vec<u16>>,
}

impl Rows {
    /// Lay out as GQ, duplicating a group's scale/zero across storage blocks when the SOURCE group is
    /// wider than the storage one (channel-wise and per-tensor scales, groups >= 4096) — exact, since a
    /// duplicated parameter is the same number.
    fn encode(&self, spec: &GqSpec) -> Result<Vec<u8>, String> {
        let gs = spec.group as usize;
        if self.inp % gs != 0 { return Err(format!("{} inputs is not a whole number of {gs}-value blocks", self.inp)) }
        let nb = self.inp / gs;
        let has_zero = spec.has_zero();
        if spec.act && (gs != self.group || nb != self.ng) {
            return Err(format!("act-order needs one storage block per group ({} groups of {}, stored {nb} of {gs})", self.ng, self.group));
        }
        let mut out = Vec::with_capacity(self.out * nb * spec.block_bytes());
        let (mut sc, mut ze) = (vec![0f32; nb], vec![0f32; nb]);
        for o in 0..self.out {
            for b in 0..nb {
                let g = if spec.act { b } else { (b * gs) / self.group };
                sc[b] = self.scales[o * self.ng + g];
                if has_zero { ze[b] = self.zeros[o * self.ng + g]; }
            }
            let gi = if spec.act { self.g_idx.as_deref() } else { None };
            encode_row(spec, &self.codes[o * self.inp..(o + 1) * self.inp], &sc, &ze, gi, &mut out);
        }
        Ok(out)
    }
}

/// The storage group for a source group of `group` values over `inp` inputs at `bits` bits: the source
/// group itself when the GQ form can hold it, else the widest of 128/64/32/16 that divides `inp` and
/// packs into whole words (the source parameter is then duplicated per block — exact).
pub(crate) fn storage_group(group: usize, inp: usize, bits: u32) -> Result<u32, String> {
    let fits = |g: usize| g > 0 && g < 4096 && inp % g == 0 && (g * bits as usize) % 32 == 0;
    if fits(group) { return Ok(group as u32) }
    [128usize, 64, 32, 16].into_iter().find(|&g| fits(g) && group % g == 0)
        .map(|g| g as u32).ok_or_else(|| format!("no storage group divides {inp} inputs for source group {group} at {bits} bits"))
}

// ---- raw tensor readers (the codes are bit patterns: never go through f32) ----------------------

pub(crate) fn i32s(st: &SafeTensors, name: &str) -> Result<(Vec<u32>, Vec<usize>), String> {
    let e = st.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if e.dtype != "I32" && e.dtype != "U32" { return Err(format!("{name} is {}, expected packed int32", e.dtype)) }
    let raw = st.raw(name)?;
    Ok((raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), e.shape.clone()))
}

pub(crate) fn u8s(st: &SafeTensors, name: &str) -> Result<(Vec<u8>, Vec<usize>), String> {
    let e = st.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
    Ok((st.raw(name)?, e.shape.clone()))
}

/// An integer tensor (g_idx, weight_shape) as i64 — through its own dtype, never through f32.
pub(crate) fn ints(st: &SafeTensors, name: &str) -> Result<Vec<i64>, String> {
    let e = st.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let raw = st.raw(name)?;
    Ok(match e.dtype.as_str() {
        "I32" => raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
        "I64" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        "I16" => raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i64).collect(),
        "I8" => raw.iter().map(|&b| b as i8 as i64).collect(),
        "U8" => raw.iter().map(|&b| b as i64).collect(),
        other => return Err(format!("{name}: {other} is not an integer dtype")),
    })
}

/// A float tensor (F32/F16/BF16, or an FP8 SCALE tensor decoded as its value) as f32 — all exact widenings.
pub(crate) fn floats(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let e = st.info(name).ok_or_else(|| format!("missing tensor {name}"))?.clone();
    let v = match e.dtype.as_str() {
        "F32" | "F16" | "BF16" | "F64" => st.get(name)?.data,
        "F8_E4M3" | "F8_E4M3FN" => st.raw(name)?.iter().map(|&b| fp8::e4m3_to_f32(b)).collect(),
        other => return Err(format!("{name}: {other} is not a scale dtype")),
    };
    Ok((v, e.shape))
}

/// Code `i` of a little-endian bitstream held in `u32` words.
#[inline]
pub(crate) fn bit_code(words: &[u32], i: usize, bits: usize) -> u32 {
    let bp = i * bits;
    let (w, sh) = (bp >> 5, bp & 31);
    let mut v = words[w] >> sh;
    if sh + bits > 32 { v |= words[w + 1] << (32 - sh); }
    v & ((1u32 << bits) - 1)
}

/// Resolve a config `ignore` / `modules_to_not_convert` / `exclude_modules` entry against a module path:
/// `re:.*<literal>` is honoured (the only regex shape seen in published configs), any other regex is
/// refused rather than guessed; a plain entry matches the whole path or a trailing component run.
pub(crate) fn ignored(list: &[String], prefix: &str) -> Result<bool, String> {
    for pat in list {
        if let Some(re) = pat.strip_prefix("re:") {
            match re.strip_prefix(".*") {
                Some(lit) if !lit.contains(['*', '[', '(', '|', '+', '?', '\\', '^', '$']) => {
                    if prefix.ends_with(lit) { return Ok(true) }
                }
                _ => return Err(format!("ignore pattern '{pat}' is a regex this reader does not evaluate")),
            }
        } else if prefix == pat || prefix.ends_with(&format!(".{pat}")) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn int_spec(bits: u32, group: u32, act: bool) -> GqSpec { GqSpec { kind: GqKind::Int, bits, group, act } }

/// Is `g_idx` just `i / group` — i.e. NOT act-order, whatever the config says?
pub(crate) fn trivial_gidx(g: &[i64], group: usize) -> bool { g.iter().enumerate().all(|(i, &x)| x == (i / group) as i64) }

/// Validate an act-order map: every entry a group of the row and every group used exactly `group`
/// times — the shape GPTQ's column permutation always produces, and the one under which "value `i`
/// uses group `g_idx[i]`" (vLLM, GPTQModel) and "sort by g_idx, then consecutive groups" (compressed-
/// tensors <= 0.11) are the same rule. Any other shape would make those two readings disagree, so it
/// is refused rather than resolved by picking one.
pub(crate) fn act_gidx(g: &[i64], ng: usize, group: usize) -> Result<Vec<u16>, String> {
    let mut count = vec![0usize; ng];
    for &x in g {
        if x < 0 || x as usize >= ng { return Err(format!("g_idx {x} outside 0..{ng}")) }
        count[x as usize] += 1;
    }
    if let Some((k, c)) = count.iter().copied().enumerate().find(|&(_, c)| c != group) {
        return Err(format!("g_idx uses group {k} {c} times, not {group}: a non-uniform act-order map is ambiguous between readers"));
    }
    if ng > u16::MAX as usize { return Err(format!("{ng} groups exceeds the u16 g_idx storage")) }
    Ok(g.iter().map(|&x| x as u16).collect())
}

/// Every module prefix that has a tensor ending in `suffix`, in name order.
pub(crate) fn prefixes(st: &SafeTensors, suffix: &str) -> Vec<String> {
    st.names().filter_map(|n| n.strip_suffix(suffix).map(str::to_string)).collect()
}

pub(crate) fn str_list(v: &serde_json::Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests;
