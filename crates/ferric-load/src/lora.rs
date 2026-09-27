//! **LoRA adapters** — read the format fine-tuning emits, write it back, and merge it into a base.
//!
//! Every fine-tuning library in use (PEFT itself, Unsloth, Axolotl, TRL, torchtune, LLaMA-Factory, MLX-LM)
//! emits a Hugging Face PEFT adapter: `adapter_config.json` + `adapter_model.safetensors`. The reference
//! for what those files MEAN is PEFT (`peft/tuners/lora/layer.py`), and its forward is the definition:
//!
//! ```text
//! y = base(x) + lora_B(lora_A(x)) * scaling,   scaling = lora_alpha / r        (use_rslora = false)
//!                                               scaling = lora_alpha / sqrt(r)  (use_rslora = true)
//! ```
//!
//! with `lora_A.weight` stored `[r, in]` and `lora_B.weight` stored `[out, r]` (nn.Linear layout), so the
//! weight delta is `scaling · B·A`, `[out, in]` — the base weight's own shape.
//!
//! llama.cpp's GGUF adapter (`general.type = adapter`, `adapter.type = lora`, `adapter.lora.alpha`,
//! tensors `blk.N.attn_q.weight.lora_a` / `.lora_b`) is read too, as a PEER format: its `lora_a` is
//! `ne = [in, r]` and `lora_b` is `ne = [r, out]`, which are the SAME row-major bytes as PEFT's
//! `[r, in]` / `[out, r]`, and its scale is `alpha / rank` — `alpha = 0` meaning 1 (`llama-adapter.h`,
//! `get_scale`).
//!
//! ⛔ **llama.cpp's format has no rslora.** `convert_lora_to_gguf.py` copies `lora_alpha` into
//! `adapter.lora.alpha` and ignores `use_rslora` (llama.cpp `0cea36222`), and the loader divides by
//! `rank`. So an rslora adapter converted by that script runs at `alpha/r` where PEFT runs it at
//! `alpha/sqrt(r)` — a √r error in the delta, silently. This reader follows the file (a GGUF adapter
//! is `alpha/rank`, because that is all it says), and [`LoraAdapter::save_gguf`] writes
//! `alpha·√r` for an rslora adapter so that the one number the format carries comes out right.
//!
//! ## What is refused, by name
//!
//! Anything this crate would load and not apply: DoRA (`use_dora`), bias training (`bias != none`,
//! `lora_bias`), `modules_to_save` (whole replaced modules), `fan_in_fan_out` (GPT-2 Conv1D layout),
//! `alpha_pattern`, aLoRA (`alora_invocation_tokens` — applies only after an invocation sequence),
//! `layer_replication`, `trainable_token_indices`, QA-LoRA, `target_parameters`, the variant configs
//! (BD-LoRA, Arrow, KaSA, VeLoRA, MonteCLoRA), inits that REWRITE THE BASE (PiSSA, OLoRA, CorDA,
//! LoftQ, LoRA-GA — such an adapter is only right on the modified base), embedding / LM-head
//! adapters, and any module this runtime has no weight for. `rank_pattern` is honoured: the scale is
//! taken from each module's own rank, read from its tensor, which is what PEFT does.
//!
//! ## Row order of q/k
//!
//! ⛔ A GGUF of a NORM-rope architecture (`llama`) stores `attn_q` / `attn_k` with their rows PERMUTED
//! relative to the Hugging Face weights (`convert_hf_to_gguf.py`, `LlamaModel.permute`). A PEFT adapter
//! was trained against the HF rows, so its `lora_B` for q_proj/k_proj must be permuted the same way
//! before it touches that base — or the delta lands on the wrong rows of every head, loads, and runs.
//! [`LoraAdapter::bind`] does it; llama.cpp's converter does it on the way into GGUF.

use crate::SafeTensors;
use ferric_gguf::{GgufSource, Meta, TensorInfo};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// One adapted linear: the pair and its scaling.
#[derive(Debug, Clone)]
pub struct LoraTarget {
    pub r: usize,
    pub n_in: usize,
    pub n_out: usize,
    /// `lora_A.weight`, `[r, n_in]` row-major (PEFT's layout; GGUF `lora_a` has the same bytes).
    pub a: Vec<f32>,
    /// `lora_B.weight`, `[n_out, r]` row-major.
    pub b: Vec<f32>,
    /// PEFT's `scaling` for this module — `alpha/r` or `alpha/√r`, `r` being THIS module's rank.
    pub scale: f32,
}

impl LoraTarget {
    /// `scale · B·A`, `[n_out, n_in]` row-major — the weight delta, computed as PEFT's
    /// `get_delta_weight` does it: `(B @ A) * scaling` in float32.
    pub fn delta(&self) -> Vec<f32> {
        let (r, ni, no) = (self.r, self.n_in, self.n_out);
        let mut d = vec![0f32; no * ni];
        for o in 0..no {
            let row = &mut d[o * ni..(o + 1) * ni];
            for k in 0..r {
                let bk = self.b[o * r + k];
                if bk == 0.0 { continue; }
                let a = &self.a[k * ni..(k + 1) * ni];
                for (x, &ai) in row.iter_mut().zip(a) { *x += bk * ai; }
            }
            for x in row.iter_mut() { *x *= self.scale; }
        }
        d
    }
}

/// Where the adapter came from — it decides how its q/k rows are ordered and how its scale was set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraFormat { Peft, Gguf }

/// Row order of the `lora_B` of attn_q / attn_k. See the module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOrder {
    /// As the authors' (Hugging Face) weights order them — what a PEFT adapter holds.
    Hf,
    /// As the base GGUF orders them — what the runtime multiplies against.
    Gguf,
}

/// A LoRA adapter, keyed by the GGUF name of the base weight each pair adapts (`blk.3.attn_q.weight`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub targets: BTreeMap<String, LoraTarget>,
    /// `lora_alpha` as declared. The per-module scaling is on each [`LoraTarget`].
    pub alpha: f32,
    /// The declared rank (`r`). Modules may differ under `rank_pattern`; each target carries its own.
    pub r: usize,
    pub use_rslora: bool,
    pub format: LoraFormat,
    pub qk_rows: RowOrder,
    /// `base_model_name_or_path` from a PEFT config — informational; shapes are what is checked.
    pub base_model: Option<String>,
    /// `general.architecture` of a GGUF adapter — must equal the base's, as llama.cpp requires.
    pub arch: Option<String>,
    /// A label for messages and for a server's adapter list (the directory or file name).
    pub name: String,
}

/// `FERRIC_LORA_NEG`: a deliberately WRONG reading, for the negative controls of
/// `scripts/lora_conformance.sh`. Each is a mistake a port could make and still load:
///   `rslora`    the other scaling formula (alpha/√r where alpha/r is right, and vice versa)
///   `alpha`     scaling = alpha, not divided by the rank
///   `transpose` lora_A / lora_B bytes read in the transposed layout (same element count, no error)
///   `off`       the adapter contributes nothing
///   `rows`      (read by the runtime, not here) every row of a batch takes row 0's adapters
///   `noperm`    a PEFT adapter's q/k rows left in HF order on a base whose GGUF permuted them
/// Unset (or empty) is the correct reading. An unknown value PANICS: a typo would silently run the
/// correct path and the control would "pass" by measuring the adapter against itself.
fn neg_control() -> Option<&'static str> {
    let v = std::env::var("FERRIC_LORA_NEG").unwrap_or_default();
    match v.trim() {
        "" => None,
        "rslora" => Some("rslora"),
        "alpha" => Some("alpha"),
        "transpose" => Some("transpose"),
        "off" => Some("off"),
        "rows" => Some("rows"),
        "noperm" => Some("noperm"),
        other => panic!("FERRIC_LORA_NEG={other:?} is not a control: rslora | alpha | transpose | off | rows | noperm"),
    }
}

/// PEFT's scaling, or a control's wrong version of it.
fn scaling(alpha: f32, r: usize, rslora: bool) -> f32 {
    match neg_control() {
        Some("rslora") => if rslora { alpha / r as f32 } else { alpha / (r as f32).sqrt() },
        Some("alpha") => alpha,
        Some("off") => 0.0,
        _ => if rslora { alpha / (r as f32).sqrt() } else { alpha / r as f32 },
    }
}

/// `m` is `[rows, cols]` row-major; return the same bytes read as `[cols, rows]` and transposed back —
/// the `transpose` control's mistake. Same shape out, scrambled contents.
fn misread(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0f32; m.len()];
    for i in 0..rows { for j in 0..cols { o[i * cols + j] = m[j * rows + i]; } }
    o
}

/// The HF sub-module path (after `layers.N.`) → the GGUF tensor stem. The same names
/// `hf.rs::qwen2_map` maps for the base weights, plus Phi-3's pre-fused `qkv_proj` / `gate_up_proj`
/// (the GGUF keeps those fused as `attn_qkv` and `ffn_up`).
const MODULES: &[(&str, &str)] = &[
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("self_attn.qkv_proj", "attn_qkv"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
    ("mlp.gate_up_proj", "ffn_up"),
];

/// The text-model roots a PEFT key may carry before `layers.N` — a text LM (`model.`), and the
/// transformers-5 nesting of a vision-language model's text stack (`model.language_model.`).
const ROOTS: &[&str] = &["base_model.model.model.layers.", "base_model.model.model.language_model.layers."];

/// `base_model.model.model.layers.3.self_attn.q_proj` → (`blk.3.attn_q.weight`, `self_attn.q_proj`).
fn gguf_name_of(module: &str) -> Result<(String, &'static str), String> {
    let rest = ROOTS.iter().find_map(|r| module.strip_prefix(r)).ok_or_else(|| format!(
        "PEFT module '{module}' is not under a text model's decoder layers — embedding, LM-head and \
         vision-tower adapters are not applied by this runtime"))?;
    let (n, sub) = rest.split_once('.').ok_or_else(|| format!("PEFT module '{module}': no sub-module"))?;
    let il: usize = n.parse().map_err(|_| format!("PEFT module '{module}': '{n}' is not a layer index"))?;
    let (hf, stem) = MODULES.iter().find(|(hf, _)| *hf == sub).ok_or_else(|| format!(
        "PEFT module '{module}': '{sub}' has no weight in this runtime's layers (it adapts only \
         {:?})", MODULES.iter().map(|m| m.0).collect::<Vec<_>>()))?;
    Ok((format!("blk.{il}.{stem}.weight"), hf))
}

/// Refuse a PEFT config field that changes the math this crate applies.
fn check_peft_config(c: &serde_json::Value) -> Result<(), String> {
    let truthy = |k: &str| -> bool {
        match &c[k] {
            serde_json::Value::Null => false,
            serde_json::Value::Bool(b) => *b,
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Object(o) => !o.is_empty(),
            serde_json::Value::String(s) => !s.is_empty(),
            serde_json::Value::Number(_) => true,
        }
    };
    match c["peft_type"].as_str() {
        Some("LORA") => {}
        other => return Err(format!("peft_type {other:?}: only LORA adapters are read")),
    }
    let refuse: &[(&str, &str)] = &[
        ("use_dora", "DoRA rescales every column of W+BA by a learned magnitude vector"),
        ("lora_bias", "lora_B carries a bias"),
        ("modules_to_save", "the adapter REPLACES whole modules (a trained copy of e.g. lm_head)"),
        ("fan_in_fan_out", "the pair is stored transposed (GPT-2 Conv1D layout)"),
        ("alpha_pattern", "per-module alpha overrides are not implemented"),
        ("alora_invocation_tokens", "an activated LoRA applies only after its invocation tokens"),
        ("layer_replication", "the adapter duplicates base layers"),
        ("trainable_token_indices", "the adapter trains embedding rows"),
        ("use_qalora", "QA-LoRA pools the input before lora_A"),
        ("target_parameters", "the adapter targets raw parameters (MoE experts), not linears"),
        ("use_bdlora", "BD-LoRA is a different factorisation"),
        ("arrow_config", "Arrow routes between several adapters per token"),
        ("kasa_config", "KaSA is an SVD-based variant"),
        ("velora_config", "VeLoRA changes the forward"),
        ("monteclora_config", "MonteCLoRA samples its delta"),
    ];
    for (k, why) in refuse {
        if truthy(k) { return Err(format!("adapter_config.json: {k} = {} — refused: {why}", c[*k])); }
    }
    match c["bias"].as_str() {
        None | Some("none") => {}
        Some(b) => return Err(format!("adapter_config.json: bias = {b:?} — refused: the adapter trains \
                                       base biases, which a LoRA pair does not carry")),
    }
    if let Some(init) = c["init_lora_weights"].as_str() {
        let lower = init.to_ascii_lowercase();
        if ["pissa", "olora", "corda", "loftq", "lora_ga"].iter().any(|p| lower.starts_with(p)) {
            return Err(format!("adapter_config.json: init_lora_weights = {init:?} — refused: that init \
                                REWRITES THE BASE WEIGHTS, so the pair is only right on the modified \
                                base. PEFT converts such an adapter to plain LoRA with \
                                `save_pretrained(path_initial_model_for_weight_conversion=...)`"));
        }
    }
    Ok(())
}

impl LoraAdapter {
    /// Open a PEFT adapter directory, or a llama.cpp `.gguf` adapter file.
    pub fn open(path: impl AsRef<Path>) -> Result<LoraAdapter, String> {
        let p = path.as_ref();
        if p.is_dir() { Self::open_peft(p) } else { Self::open_gguf(p) }
    }

    /// Read `adapter_config.json` + `adapter_model.safetensors`.
    pub fn open_peft(dir: impl AsRef<Path>) -> Result<LoraAdapter, String> {
        let dir = dir.as_ref();
        let txt = std::fs::read_to_string(dir.join("adapter_config.json"))
            .map_err(|e| format!("{}/adapter_config.json: {e}", dir.display()))?;
        let c: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("adapter_config.json: {e}"))?;
        check_peft_config(&c)?;
        let r = c["r"].as_u64().ok_or("adapter_config.json: no integer r")? as usize;
        let alpha = c["lora_alpha"].as_f64().ok_or("adapter_config.json: no lora_alpha")? as f32;
        let rslora = c["use_rslora"].as_bool().unwrap_or(false);
        let rank_pattern = c["rank_pattern"].as_object().is_some_and(|o| !o.is_empty());
        let st_path = dir.join("adapter_model.safetensors");
        if !st_path.exists() {
            return Err(format!("{}: no adapter_model.safetensors{}", dir.display(),
                if dir.join("adapter_model.bin").exists() {
                    " (there is an adapter_model.bin — a pickle, which this reader does not execute; \
                     re-save with safe_serialization=True)" } else { "" }));
        }
        let st = SafeTensors::open(&st_path)?;
        // Pair lora_A / lora_B by module. Every key must be one of the two; anything else is a part of
        // the adapter this reader would silently drop.
        let mut pairs: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
        for k in st.names() {
            let (module, is_a) = if let Some(m) = k.strip_suffix(".lora_A.weight") { (m, true) }
                else if let Some(m) = k.strip_suffix(".lora_B.weight") { (m, false) }
                else {
                    let why = if k.contains("lora_magnitude_vector") { "a DoRA magnitude vector" }
                        else if k.contains("lora_embedding") { "an embedding adapter" }
                        else if k.ends_with(".lora_B.bias") { "a lora_B bias" }
                        else if k.contains("modules_to_save") || k.contains("original_module") { "a replaced module" }
                        else { "not a lora_A/lora_B weight" };
                    return Err(format!("adapter tensor '{k}' is {why} — refused rather than dropped"));
                };
            let e = pairs.entry(module.to_string()).or_default();
            if is_a { e.0 = Some(k.clone()) } else { e.1 = Some(k.clone()) }
        }
        let ctl = neg_control();
        let mut targets = BTreeMap::new();
        for (module, (ka, kb)) in pairs {
            let (ka, kb) = match (ka, kb) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(format!("{module}: lora_A and lora_B must both be present")),
            };
            let (name, _) = gguf_name_of(&module)?;
            let (ta, tb) = (st.get(&ka)?, st.get(&kb)?);
            if ta.shape.len() != 2 || tb.shape.len() != 2 || ta.shape[0] != tb.shape[1] {
                return Err(format!("{module}: lora_A {:?} and lora_B {:?} do not form a rank-r pair",
                                   ta.shape, tb.shape));
            }
            let (tr, n_in, n_out) = (ta.shape[0], ta.shape[1], tb.shape[0]);
            if !rank_pattern && tr != r {
                return Err(format!("{module}: rank {tr} but the config says r = {r} and has no rank_pattern"));
            }
            let (a, b) = if ctl == Some("transpose") {
                (misread(&ta.data, tr, n_in), misread(&tb.data, n_out, tr))
            } else { (ta.data, tb.data) };
            if targets.insert(name.clone(), LoraTarget { r: tr, n_in, n_out, a, b, scale: scaling(alpha, tr, rslora) }).is_some() {
                return Err(format!("{name} is adapted twice (two PEFT modules map to one weight)"));
            }
        }
        if targets.is_empty() { return Err(format!("{}: the adapter holds no lora_A/lora_B pairs", dir.display())); }
        Ok(LoraAdapter {
            targets, alpha, r, use_rslora: rslora, format: LoraFormat::Peft, qk_rows: RowOrder::Hf,
            base_model: c["base_model_name_or_path"].as_str().map(String::from), arch: None,
            name: dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        })
    }

    /// Read a llama.cpp GGUF LoRA adapter (`convert_lora_to_gguf.py`'s output).
    pub fn open_gguf(path: impl AsRef<Path>) -> Result<LoraAdapter, String> {
        let p = path.as_ref();
        let g = ferric_gguf::GgufFile::open(p)?;
        let s = |k: &str| match g.metadata().get(k) { Some(Meta::Str(v)) => Some(v.clone()), _ => None };
        if s("general.type").as_deref() != Some("adapter") {
            return Err(format!("{}: general.type is {:?}, not \"adapter\"", p.display(), s("general.type")));
        }
        if s("adapter.type").as_deref() != Some("lora") {
            return Err(format!("{}: adapter.type is {:?}, not \"lora\"", p.display(), s("adapter.type")));
        }
        if g.metadata().contains_key("adapter.alora.invocation_tokens") {
            return Err("adapter.alora.invocation_tokens: an activated LoRA applies only after its \
                        invocation tokens — refused".into());
        }
        let alpha = match g.metadata().get("adapter.lora.alpha") { Some(Meta::F(v)) => *v as f32, _ => 0.0 };
        let mut pairs: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
        for t in g.tensors.iter() {
            let (base, is_a) = if let Some(b) = t.name.strip_suffix(".lora_a") { (b, true) }
                else if let Some(b) = t.name.strip_suffix(".lora_b") { (b, false) }
                else {
                    return Err(format!("GGUF adapter tensor '{}' is not a lora_a/lora_b (mergekit adds \
                                        norm weights to some adapters) — refused rather than dropped", t.name));
                };
            let e = pairs.entry(base.to_string()).or_default();
            if is_a { e.0 = Some(t.name.clone()) } else { e.1 = Some(t.name.clone()) }
        }
        let ctl = neg_control();
        let mut targets = BTreeMap::new();
        let mut rank = 0;
        for (name, (ka, kb)) in pairs {
            let (ka, kb) = match (ka, kb) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(format!("{name}: lora_a and lora_b must both be present")),
            };
            let stem = name.strip_prefix("blk.").and_then(|r| r.split_once('.')).map(|(_, s)| s)
                .and_then(|s| s.strip_suffix(".weight"));
            if !stem.is_some_and(|s| MODULES.iter().any(|(_, g)| *g == s)) {
                return Err(format!("GGUF adapter targets '{name}', which this runtime does not adapt \
                                    (token_embd is transposed by llama.cpp's converter and is refused too)"));
            }
            let (ia, ib) = (g.tensor(&ka).unwrap().clone(), g.tensor(&kb).unwrap().clone());
            // lora_a ne = [in, r], lora_b ne = [r, out]; llama.cpp refuses a[1] != b[0] as "not transposed".
            if ia.dims.len() != 2 || ib.dims.len() != 2 || ia.dims[1] != ib.dims[0] {
                return Err(format!("{name}: lora_a {:?} / lora_b {:?} (ne order) do not form a pair", ia.dims, ib.dims));
            }
            let (n_in, r, n_out) = (ia.dims[0] as usize, ia.dims[1] as usize, ib.dims[1] as usize);
            rank = rank.max(r);
            let (a, b) = (g.dequant(&ka)?, g.dequant(&kb)?);
            let (a, b) = if ctl == Some("transpose") { (misread(&a, r, n_in), misread(&b, n_out, r)) } else { (a, b) };
            // llama.cpp: `alpha ? adapter_scale * alpha / rank : adapter_scale`. No rslora in the format.
            let scale = if alpha == 0.0 { if ctl == Some("off") { 0.0 } else { 1.0 } } else { scaling(alpha, r, false) };
            targets.insert(name, LoraTarget { r, n_in, n_out, a, b, scale });
        }
        if targets.is_empty() { return Err(format!("{}: the adapter holds no lora_a/lora_b pairs", p.display())); }
        Ok(LoraAdapter {
            targets, alpha, r: rank, use_rslora: false, format: LoraFormat::Gguf, qk_rows: RowOrder::Gguf,
            base_model: s("general.base_model.0.repo_url").or_else(|| s("general.base_model.0.name")),
            arch: s("general.architecture"),
            name: p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        })
    }

    /// An adapter built in memory — a Ferric fine-tune — from `(gguf name, lora_A [r,in], lora_B
    /// [out,r])` triples, with rows in the base GGUF's order (which is what the runtime trained against).
    pub fn from_pairs(name: &str, alpha: f32, use_rslora: bool, arch: &str,
                      pairs: Vec<(String, Vec<f32>, Vec<f32>, usize, usize, usize)>) -> Result<LoraAdapter, String> {
        let mut targets = BTreeMap::new();
        let mut rank = 0;
        for (n, a, b, r, n_in, n_out) in pairs {
            if a.len() != r * n_in || b.len() != n_out * r {
                return Err(format!("{n}: lora_A has {} values for [{r}, {n_in}], lora_B {} for [{n_out}, {r}]",
                                   a.len(), b.len()));
            }
            rank = rank.max(r);
            let scale = if use_rslora { alpha / (r as f32).sqrt() } else { alpha / r as f32 };
            targets.insert(n, LoraTarget { r, n_in, n_out, a, b, scale });
        }
        Ok(LoraAdapter { targets, alpha, r: rank, use_rslora, format: LoraFormat::Peft, qk_rows: RowOrder::Gguf,
                         base_model: None, arch: Some(arch.to_string()), name: name.to_string() })
    }

    /// The same adapter with its q/k rows in `to` order, for a base of architecture `arch`.
    ///
    /// Also checks a GGUF adapter's architecture against the base's (llama.cpp refuses a mismatch).
    /// ⛔ Refuses architectures whose GGUF q/k row order has not been audited against their converter:
    /// guessing "unpermuted" for one that is permuted produces a model that runs and is wrong.
    pub fn bind(&self, arch: &str, n_head: usize, n_head_kv: usize, to: RowOrder) -> Result<LoraAdapter, String> {
        if let Some(a) = &self.arch {
            if self.format == LoraFormat::Gguf && a != arch {
                return Err(format!("adapter '{}' is for architecture '{a}', the base is '{arch}'", self.name));
            }
        }
        // convert_hf_to_gguf.py: `LlamaModel.permute` on q_proj/k_proj for `llama`; none of the Qwen,
        // Phi-3 or Gemma converters permute.
        let permuted = match arch {
            "llama" => true,
            "qwen2" | "qwen3" | "qwen2vl" | "qwen3vl" | "qwen3vlmoe" | "phi3" | "gemma" | "gemma2" | "gemma3" => false,
            other => return Err(format!("the q/k row order a '{other}' GGUF stores has not been audited \
                                         against its converter — refusing rather than guessing")),
        };
        let mut out = self.clone();
        if !permuted || self.qk_rows == to || neg_control() == Some("noperm") {
            out.qk_rows = to;
            return Ok(out);
        }
        for (name, t) in out.targets.iter_mut() {
            let heads = if name.ends_with(".attn_q.weight") { n_head }
                else if name.ends_with(".attn_k.weight") { n_head_kv } else { continue };
            if t.n_out % (2 * heads) != 0 {
                return Err(format!("{name}: {} rows do not split into {heads} heads of even size", t.n_out));
            }
            t.b = permute_rows(&t.b, t.n_out, t.r, heads, to == RowOrder::Gguf);
        }
        out.qk_rows = to;
        Ok(out)
    }

    /// Check every target against the base's weight of the same name: present, and `[n_out, n_in]`
    /// equal to the base's (GGUF `ne = [in, out]`). A mismatch is a different base model.
    pub fn check_base(&self, base: &impl GgufSource) -> Result<(), String> {
        for (name, t) in &self.targets {
            let Some(w) = base.tensor(name) else {
                return Err(format!("adapter '{}' adapts {name}, which the base does not have (a \
                                    different architecture, or more layers than the base)", self.name));
            };
            if w.dims.len() != 2 || w.dims[0] as usize != t.n_in || w.dims[1] as usize != t.n_out {
                return Err(format!("adapter '{}': {name} is [{} out, {} in] in the adapter but {:?} (ne) \
                                    in the base — not the base this adapter was trained on",
                                   self.name, t.n_out, t.n_in, w.dims));
            }
        }
        Ok(())
    }

    /// Distinct layer indices the adapter touches, ascending.
    pub fn layers(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.targets.keys().filter_map(|n| n.strip_prefix("blk.")?.split('.').next()?.parse().ok()).collect();
        v.sort(); v.dedup(); v
    }

    /// Parameter count (both factors of every pair).
    pub fn params(&self) -> usize { self.targets.values().map(|t| t.a.len() + t.b.len()).sum() }

    /// Write a PEFT adapter directory that `PeftModel.from_pretrained(base, dir)` loads.
    ///
    /// Requires HF row order — call [`LoraAdapter::bind`] with [`RowOrder::Hf`] first. Tensors go out as
    /// F32 under `base_model.model.model.layers.N.<module>.lora_{A,B}.weight`; the config names each
    /// adapted module by its FULL path when the targets are not a plain modules × layers grid, since a
    /// short name plus `layers_to_transform` can only describe a grid.
    pub fn save_peft(&self, dir: impl AsRef<Path>, base_model_name_or_path: &str) -> Result<(), String> {
        if self.qk_rows != RowOrder::Hf {
            return Err("save_peft: the adapter is in the base GGUF's q/k row order; bind(.., RowOrder::Hf) first".into());
        }
        // ⚠ Phi-3's pre-fused qkv_proj / gate_up_proj share GGUF stems with the split layout (ffn_up is
        // `up_proj` on Qwen and `gate_up_proj` on Phi-3), so the name alone cannot say which module to
        // write. Refused until an exporter that knows the base's layout exists and is checked.
        if self.arch.as_deref() == Some("phi3") || self.targets.keys().any(|n| n.ends_with(".attn_qkv.weight")) {
            return Err("save_peft: Phi-3's fused qkv_proj / gate_up_proj are not exported".into());
        }
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut hf_of: HashMap<&str, &str> = HashMap::new();
        for (hf, g) in MODULES { hf_of.entry(g).or_insert(hf); }
        let mut entries: Vec<(String, Vec<usize>, &[f32])> = Vec::new();
        let (mut subs, mut layers, mut cells) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), 0usize);
        for (name, t) in &self.targets {
            let rest = name.strip_prefix("blk.").ok_or_else(|| format!("{name}: not a blk.N name"))?;
            let (il, stem) = rest.split_once('.').ok_or_else(|| format!("{name}: malformed"))?;
            let stem = stem.strip_suffix(".weight").ok_or_else(|| format!("{name}: malformed"))?;
            let hf = *hf_of.get(stem).ok_or_else(|| format!("{name}: no PEFT module for '{stem}'"))?;
            let module = format!("model.layers.{il}.{hf}");
            subs.insert(hf.rsplit('.').next().unwrap().to_string());
            layers.insert(il.parse::<usize>().map_err(|_| format!("{name}: bad layer"))?);
            cells += 1;
            entries.push((format!("base_model.model.{module}.lora_A.weight"), vec![t.r, t.n_in], &t.a));
            entries.push((format!("base_model.model.{module}.lora_B.weight"), vec![t.n_out, t.r], &t.b));
        }
        let grid = cells == subs.len() * layers.len();
        let ranks: std::collections::BTreeSet<usize> = self.targets.values().map(|t| t.r).collect();
        let mut rank_pattern = serde_json::Map::new();
        if ranks.len() > 1 {
            // PEFT reads rank_pattern keys as module-name patterns; a full path matches exactly one.
            for (name, t) in &self.targets {
                if t.r != self.r {
                    let (il, stem) = name.strip_prefix("blk.").unwrap().split_once('.').unwrap();
                    let hf = hf_of[stem.strip_suffix(".weight").unwrap()];
                    rank_pattern.insert(format!("model.layers.{il}.{hf}"), t.r.into());
                }
            }
        }
        let target_modules: Vec<String> = if grid { subs.iter().cloned().collect() } else {
            self.targets.keys().map(|name| {
                let (il, stem) = name.strip_prefix("blk.").unwrap().split_once('.').unwrap();
                format!("model.layers.{il}.{}", hf_of[stem.strip_suffix(".weight").unwrap()])
            }).collect()
        };
        let cfg = serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_model_name_or_path,
            "r": self.r,
            "lora_alpha": self.alpha,
            "use_rslora": self.use_rslora,
            "target_modules": target_modules,
            "layers_to_transform": if grid { serde_json::json!(layers.iter().collect::<Vec<_>>()) } else { serde_json::Value::Null },
            "rank_pattern": rank_pattern,
            "alpha_pattern": {},
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": false,
            "use_dora": false,
            "modules_to_save": null,
            "init_lora_weights": true,
            "inference_mode": true,
            "revision": null,
        });
        std::fs::write(dir.join("adapter_config.json"), serde_json::to_string_pretty(&cfg).unwrap())
            .map_err(|e| format!("adapter_config.json: {e}"))?;
        write_safetensors(&dir.join("adapter_model.safetensors"), &entries)
    }

    /// Write a llama.cpp GGUF adapter (`general.type = adapter`) for a base of architecture `arch`.
    ///
    /// Requires GGUF row order. The format carries ONE alpha and llama.cpp scales by `alpha / rank`, so
    /// `alpha` goes out as `scale · r` — for an rslora adapter that is `alpha·√r`, which is what makes
    /// llama.cpp's arithmetic land on PEFT's `alpha/√r`. Refuses an adapter whose modules need
    /// different values (mixed ranks under rslora), which the format cannot express.
    pub fn save_gguf(&self, path: impl AsRef<Path>, arch: &str) -> Result<(), String> {
        if self.qk_rows != RowOrder::Gguf {
            return Err("save_gguf: the adapter is in HF q/k row order; bind(.., RowOrder::Gguf) first".into());
        }
        let eff: Vec<f32> = self.targets.values().map(|t| t.scale * t.r as f32).collect();
        let a0 = eff[0];
        if eff.iter().any(|&e| (e - a0).abs() > 1e-6 * a0.abs().max(1.0)) {
            return Err("save_gguf: modules need different alpha/rank products; llama.cpp's adapter \
                        format carries one alpha".into());
        }
        let mut w = ferric_gguf::write::GgufWriter::new(arch);
        w.kv_str("general.type", "adapter").kv_str("adapter.type", "lora").kv_f32("adapter.lora.alpha", a0);
        if let Some(b) = &self.base_model { w.kv_str("general.base_model.0.name", b); }
        for (name, t) in &self.targets {
            w.tensor_f32(&format!("{name}.lora_a"), &[t.n_in as u64, t.r as u64], &t.a);
            w.tensor_f32(&format!("{name}.lora_b"), &[t.r as u64, t.n_out as u64], &t.b);
        }
        w.write_to(path)
    }
}

/// `LlamaModel.permute` (convert_hf_to_gguf.py) on the ROWS of `b` (`[n_out, r]`):
/// `w.reshape(heads, 2, n_out/heads/2, r).swapaxes(1, 2).reshape(n_out, r)`. `forward = false` is its
/// inverse (GGUF order back to HF).
fn permute_rows(b: &[f32], n_out: usize, r: usize, heads: usize, forward: bool) -> Vec<f32> {
    let hd = n_out / heads;
    let half = hd / 2;
    let mut o = vec![0f32; b.len()];
    for h in 0..heads {
        for j in 0..half {
            for s in 0..2 {
                let gguf_row = h * hd + 2 * j + s;   // (h, j, s) after swapaxes
                let hf_row = h * hd + s * half + j;  // (h, s, j) before
                let (dst, src) = if forward { (gguf_row, hf_row) } else { (hf_row, gguf_row) };
                o[dst * r..(dst + 1) * r].copy_from_slice(&b[src * r..(src + 1) * r]);
            }
        }
    }
    o
}

/// A minimal safetensors writer: F32 tensors, `{"format": "pt"}` metadata (what `safe_save_file` in
/// PEFT writes and `transformers` checks for).
fn write_safetensors(path: &Path, entries: &[(String, Vec<usize>, &[f32])]) -> Result<(), String> {
    let mut hdr = serde_json::Map::new();
    hdr.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let mut off = 0usize;
    for (name, shape, data) in entries {
        let n = data.len() * 4;
        hdr.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + n]}));
        off += n;
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
    // The header is padded with spaces to 8 bytes, as the reference writer does, so the data that
    // follows is aligned for readers that map it.
    while h.len() % 8 != 0 { h.push(b' '); }
    let mut out = Vec::with_capacity(8 + h.len() + off);
    out.extend((h.len() as u64).to_le_bytes());
    out.extend(h);
    for (_, _, data) in entries { for x in data.iter() { out.extend(x.to_le_bytes()); } }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// How [`LoraMerged`] stores a merged weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeDtype {
    /// Exact up to float32 rounding of `W + scale·B·A` — what the conformance gate checks (0.43-1.48x
    /// PEFT's own float32 floor). ⚠ On a quantized base this WIDENS every adapted weight to 4 bytes, and
    /// Ferric's dense f32 weights decode slowly: 15 tok/s vs 58 for the Q4_K_M base (Qwen2.5-0.5B, all
    /// seven projections adapted; `examples/lora_decode_bench.rs`).
    F32,
    /// Half of F32's memory, and fast at decode (81 tok/s median on the same run — the 16-bit kernels
    /// split the reduction); rounds the merged weight to float16 (nearest even): 3.2e-3 max logit error on
    /// the gate's adapter, 25x the floor. Use it when that rounding is acceptable.
    F16,
}

/// **A base checkpoint with LoRA adapters merged into its weights**, presented as a [`GgufSource`] —
/// so it works with EVERY runtime unchanged: `Qwen3::load(&ctx, &LoraMerged::new(&gguf, ..)?)`.
///
/// Each adapted weight is dequantized, `Σ user_scale · scale · B·A` is added, and the result is handed
/// to the loader as F32 (or F16) — whatever quant the base stored it in. Un-adapted weights pass
/// through untouched, bytes and file ranges alike, so a quantized base stays quantized everywhere the
/// adapter does not reach. The merge is computed lazily, when the loader asks for the tensor.
pub struct LoraMerged<'a, G: GgufSource> {
    base: &'a G,
    adapters: Vec<(LoraAdapter, f32)>,
    infos: HashMap<String, TensorInfo>,
    dtype: MergeDtype,
}

impl<'a, G: GgufSource> LoraMerged<'a, G> {
    /// `adapters`: each with a user multiplier (1.0 = as trained). Every adapter is bound to the base's
    /// q/k row order and checked against its shapes here, before any weight is touched.
    pub fn new(base: &'a G, adapters: &[(&LoraAdapter, f32)], dtype: MergeDtype) -> Result<Self, String> {
        let md = base.metadata();
        let arch = match md.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => return Err("base has no general.architecture".into()) };
        let u = |k: &str| match md.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("base: missing {arch}.{k}")) };
        let (nh, nkv) = (u("attention.head_count")?, u("attention.head_count_kv")?);
        let mut bound = Vec::new();
        let mut infos = HashMap::new();
        for (a, s) in adapters {
            let b = a.bind(&arch, nh, nkv, RowOrder::Gguf)?;
            b.check_base(base)?;
            for name in b.targets.keys() {
                let mut ti = base.tensor(name).unwrap().clone();
                ti.ggml_type = match dtype { MergeDtype::F32 => 0, MergeDtype::F16 => 1 };
                ti.offset = 0;
                infos.insert(name.clone(), ti);
            }
            bound.push((b, *s));
        }
        Ok(LoraMerged { base, adapters: bound, infos, dtype })
    }

    /// Names of the weights this view merges.
    pub fn merged_names(&self) -> impl Iterator<Item = &String> { self.infos.keys() }

    fn merged(&self, name: &str) -> Result<Vec<f32>, String> {
        let mut w = self.base.dequant(name)?;
        for (a, s) in &self.adapters {
            if let Some(t) = a.targets.get(name) {
                let d = t.delta();
                if d.len() != w.len() { return Err(format!("{name}: delta {} vs weight {}", d.len(), w.len())); }
                for (x, y) in w.iter_mut().zip(d) { *x += *s * y; }
            }
        }
        if self.dtype == MergeDtype::F16 {
            for x in &mut w { *x = half::f16::from_f32(*x).to_f32(); }
        }
        Ok(w)
    }
}

impl<G: GgufSource> GgufSource for LoraMerged<'_, G> {
    fn metadata(&self) -> &HashMap<String, Meta> { self.base.metadata() }
    fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.infos.get(name).or_else(|| self.base.tensor(name)) }
    fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        if !self.infos.contains_key(name) { return self.base.raw(name); }
        let w = self.merged(name)?;
        Ok(match self.dtype {
            MergeDtype::F32 => w.iter().flat_map(|x| x.to_le_bytes()).collect(),
            MergeDtype::F16 => w.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        })
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        if self.infos.contains_key(name) { self.merged(name) } else { self.base.dequant(name) }
    }
    // ⚠ A merged weight has NO file range: its bytes exist only here. Forwarding the base's range for
    // one would let a streaming loader read the UNMERGED bytes straight off disk.
    fn tensor_file_range(&self, name: &str) -> Option<(std::path::PathBuf, u64, u64)> {
        if self.infos.contains_key(name) { None } else { self.base.tensor_file_range(name) }
    }
    fn raw_range(&self, name: &str, off: u64, dst: &mut [u8]) -> Result<(), String> {
        if !self.infos.contains_key(name) { return self.base.raw_range(name, off, dst); }
        let all = self.raw(name)?;
        let end = off as usize + dst.len();
        if end > all.len() { return Err(format!("{name}: range {off}..{end} exceeds {} bytes", all.len())); }
        dst.copy_from_slice(&all[off as usize..end]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ferric-lora-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn toy(name: &str, r: usize, n_in: usize, n_out: usize, seed: f32) -> (String, Vec<f32>, Vec<f32>, usize, usize, usize) {
        let v = |n: usize, s: f32| (0..n).map(|i| (i as f32 * 0.37 + s).sin() * 0.5).collect::<Vec<f32>>();
        (name.to_string(), v(r * n_in, seed), v(n_out * r, seed + 1.0), r, n_in, n_out)
    }

    /// The delta is B·A times the scale, in PEFT's layouts — checked against a hand product.
    #[test]
    fn delta_is_scale_times_b_times_a() {
        let t = LoraTarget { r: 2, n_in: 3, n_out: 2, a: vec![1., 2., 3., 4., 5., 6.], b: vec![1., 0., 0., 1.], scale: 0.5 };
        assert_eq!(t.delta(), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        let t = LoraTarget { b: vec![1., 1., 2., -1.], ..t };
        // row 0 = a0 + a1 = [5,7,9]; row 1 = 2a0 - a1 = [-2,-1,0]; times 0.5
        assert_eq!(t.delta(), vec![2.5, 3.5, 4.5, -1.0, -0.5, 0.0]);
    }

    /// PEFT round trip through our own writer and reader: every value, every name, the scale.
    #[test]
    fn save_peft_then_open_peft_is_the_identity() {
        let d = tmp("peft");
        let a = LoraAdapter::from_pairs("t", 16.0, false, "qwen2", vec![
            toy("blk.0.attn_q.weight", 4, 8, 8, 0.0), toy("blk.0.attn_v.weight", 4, 8, 2, 1.0),
            toy("blk.3.ffn_down.weight", 4, 12, 8, 2.0)]).unwrap();
        let a = a.bind("qwen2", 2, 1, RowOrder::Hf).unwrap();
        a.save_peft(&d, "toy/base").unwrap();
        let b = LoraAdapter::open_peft(&d).unwrap();
        assert_eq!(a.targets.len(), b.targets.len());
        for (n, t) in &a.targets {
            let u = &b.targets[n];
            assert_eq!((t.r, t.n_in, t.n_out), (u.r, u.n_in, u.n_out), "{n}");
            assert_eq!(t.a, u.a, "{n} lora_A");
            assert_eq!(t.b, u.b, "{n} lora_B");
            assert_eq!(u.scale, 4.0, "alpha 16 / r 4");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// ⛔ The llama q/k permutation must be an exact inverse pair, and must actually move rows — a
    /// permutation that is the identity would pass the round trip and apply HF rows to a GGUF base.
    #[test]
    fn llama_qk_permutation_round_trips_and_is_not_the_identity() {
        let a = LoraAdapter::from_pairs("t", 8.0, false, "llama", vec![
            toy("blk.0.attn_q.weight", 2, 8, 16, 0.0), toy("blk.0.attn_k.weight", 2, 8, 8, 1.0),
            toy("blk.0.attn_v.weight", 2, 8, 8, 2.0)]).unwrap();
        let hf = a.bind("llama", 4, 2, RowOrder::Hf).unwrap();
        assert_ne!(hf.targets["blk.0.attn_q.weight"].b, a.targets["blk.0.attn_q.weight"].b, "q rows must move");
        assert_ne!(hf.targets["blk.0.attn_k.weight"].b, a.targets["blk.0.attn_k.weight"].b, "k rows must move");
        assert_eq!(hf.targets["blk.0.attn_v.weight"].b, a.targets["blk.0.attn_v.weight"].b, "v is never permuted");
        let back = hf.bind("llama", 4, 2, RowOrder::Gguf).unwrap();
        for (n, t) in &a.targets { assert_eq!(t.b, back.targets[n].b, "{n}"); }
        // qwen2 has no permutation at all
        let q = a.bind("qwen2", 4, 2, RowOrder::Hf).unwrap();
        for (n, t) in &a.targets { assert_eq!(t.b, q.targets[n].b, "{n}"); }
        // and an unaudited architecture is refused, not guessed
        assert!(a.bind("muse-glimmer", 4, 2, RowOrder::Hf).is_err());
    }

    /// The permutation is llama.cpp's: for one head of dim 4, HF rows [0 1 | 2 3] interleave to [0 2 1 3].
    #[test]
    fn permutation_matches_llama_cpp_on_one_head() {
        let b: Vec<f32> = (0..4).map(|x| x as f32).collect(); // 4 rows, r = 1
        assert_eq!(permute_rows(&b, 4, 1, 1, true), vec![0., 2., 1., 3.]);
        assert_eq!(permute_rows(&[0., 2., 1., 3.], 4, 1, 1, false), b);
    }

    fn write_cfg(d: &Path, extra: serde_json::Value) {
        let mut c = serde_json::json!({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none",
                                       "target_modules": ["q_proj"]});
        for (k, v) in extra.as_object().unwrap() { c[k] = v.clone(); }
        std::fs::write(d.join("adapter_config.json"), c.to_string()).unwrap();
    }

    /// Every config that changes the math is refused BY NAME, not loaded and ignored.
    #[test]
    fn configs_that_change_the_math_are_refused_by_name() {
        let d = tmp("refuse");
        let a = LoraAdapter::from_pairs("t", 4.0, false, "qwen2", vec![toy("blk.0.attn_q.weight", 2, 4, 4, 0.0)])
            .unwrap().bind("qwen2", 1, 1, RowOrder::Hf).unwrap();
        a.save_peft(&d, "b").unwrap();
        assert!(LoraAdapter::open_peft(&d).is_ok(), "the plain adapter must load, or the refusals prove nothing");
        for (k, v) in [("use_dora", serde_json::json!(true)), ("bias", serde_json::json!("all")),
                       ("modules_to_save", serde_json::json!(["lm_head"])), ("fan_in_fan_out", serde_json::json!(true)),
                       ("alpha_pattern", serde_json::json!({"q_proj": 8})), ("lora_bias", serde_json::json!(true)),
                       ("init_lora_weights", serde_json::json!("pissa")), ("alora_invocation_tokens", serde_json::json!([1, 2])),
                       ("layer_replication", serde_json::json!([[0, 1]])), ("peft_type", serde_json::json!("IA3"))] {
            write_cfg(&d, serde_json::json!({k: v}));
            let e = LoraAdapter::open_peft(&d).expect_err(&format!("{k} = {v} must be refused"));
            assert!(e.contains(k), "refusal for {k} must name it: {e}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A rank disagreement without rank_pattern is refused; with rank_pattern each module's own rank
    /// sets its scale (PEFT: scaling = alpha / r_module).
    #[test]
    fn rank_pattern_scales_by_each_modules_own_rank() {
        let d = tmp("rank");
        let a = LoraAdapter::from_pairs("t", 8.0, false, "qwen2", vec![
            toy("blk.0.attn_q.weight", 2, 4, 4, 0.0), toy("blk.1.attn_q.weight", 4, 4, 4, 1.0)]).unwrap();
        let mut a = a.bind("qwen2", 1, 1, RowOrder::Hf).unwrap();
        a.r = 2;
        a.save_peft(&d, "b").unwrap();
        let b = LoraAdapter::open_peft(&d).unwrap();
        assert_eq!(b.targets["blk.0.attn_q.weight"].scale, 4.0);
        assert_eq!(b.targets["blk.1.attn_q.weight"].scale, 2.0);
        write_cfg(&d, serde_json::json!({"r": 2, "lora_alpha": 8, "rank_pattern": {}}));
        assert!(LoraAdapter::open_peft(&d).unwrap_err().contains("rank"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Module names outside the decoder layers, and sub-modules the runtime has no weight for, are refused.
    #[test]
    fn unknown_modules_are_refused() {
        assert_eq!(gguf_name_of("base_model.model.model.layers.7.mlp.down_proj").unwrap().0, "blk.7.ffn_down.weight");
        assert_eq!(gguf_name_of("base_model.model.model.language_model.layers.2.self_attn.o_proj").unwrap().0,
                   "blk.2.attn_output.weight");
        assert!(gguf_name_of("base_model.model.lm_head").is_err());
        assert!(gguf_name_of("base_model.model.model.embed_tokens").is_err());
        assert!(gguf_name_of("base_model.model.model.layers.0.self_attn.rotary_emb").is_err());
        assert!(gguf_name_of("base_model.model.visual.blocks.0.attn.qkv").is_err());
    }

    /// GGUF round trip through our writer: same pairs, and the rslora scale survives as alpha·√r.
    #[test]
    fn save_gguf_then_open_gguf_keeps_pairs_and_rslora_scale() {
        let d = tmp("gguf");
        let a = LoraAdapter::from_pairs("t", 6.0, true, "qwen2", vec![
            toy("blk.0.attn_q.weight", 4, 8, 8, 0.0), toy("blk.2.ffn_down.weight", 4, 12, 8, 1.0)]).unwrap();
        let p = d.join("a.gguf");
        a.save_gguf(&p, "qwen2").unwrap();
        let b = LoraAdapter::open_gguf(&p).unwrap();
        assert_eq!(b.arch.as_deref(), Some("qwen2"));
        for (n, t) in &a.targets {
            let u = &b.targets[n];
            assert_eq!((t.a.clone(), t.b.clone()), (u.a.clone(), u.b.clone()), "{n}");
            assert!((u.scale - 3.0).abs() < 1e-6, "rslora 6/sqrt(4) = 3 must survive llama.cpp's alpha/rank, got {}", u.scale);
        }
        assert!(b.bind("llama", 1, 1, RowOrder::Gguf).is_err(), "a GGUF adapter for qwen2 must not bind to llama");
        let _ = std::fs::remove_dir_all(&d);
    }
}
