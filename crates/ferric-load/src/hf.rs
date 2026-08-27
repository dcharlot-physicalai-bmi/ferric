//! **Run a HuggingFace checkpoint directly** — no conversion step, no converter.
//!
//! Every Ferric runtime takes `&impl GgufSource`, so until now the only way to run a published
//! checkpoint was to convert it to GGUF first — with llama.cpp's Python converter. That is the last
//! and largest way this project depended on that one: not a line of code, not a crate in the tree,
//! but a mandatory step in front of every model. The dependency was invisible in `Cargo.lock` and
//! total in practice.
//!
//! [`HfCheckpoint`] implements `GgufSource` over `config.json` + `model.safetensors`, so it works
//! with EVERY runtime unchanged rather than one at a time. Three translations, and only the third
//! is interesting:
//!
//! 1. **Metadata.** `config.json`'s keys to GGUF's `<arch>.<key>` namespace.
//! 2. **Names.** `model.layers.3.self_attn.q_proj.weight` to `blk.3.attn_q.weight`.
//! 3. **Geometry — REVERSE THE SHAPE, KEEP THE BYTES.** GGUF reports `ne[]` fastest-varying first,
//!    so a `[out, in]` PyTorch weight is `[in, out]` in GGUF. Both store the same row-major bytes.
//!    ⚠ This is the trap: the shapes disagree and the DATA does not, so anyone who "fixes" the
//!    mismatch by transposing gets a model that loads, runs, and is wrong. LFM2's conv weight makes
//!    it concrete — HF `[1024, 1, 3]`, GGUF `[3, 1024]`, and `Lfm2::load` reads that as `[d, L]`
//!    row-major, which is exactly the HF bytes with the singleton dropped.
//!
//! ## What this is not
//!
//! It does not write GGUF files and does not want to. A conversion produces a second artifact that
//! can drift from the first; this reads the published one. Nor does it cover every architecture —
//! the maps below are per-`model_type` and each needs its weights checked against something, which
//! is the only reason to add one.

use crate::{is_scaled, SafeTensors};
use ferric_gguf::{GgufSource, Meta, TensorInfo};
use std::collections::HashMap;
use std::path::Path;

/// A published checkpoint presented as a `GgufSource`.
pub struct HfCheckpoint {
    st: SafeTensors,
    meta: HashMap<String, Meta>,
    infos: HashMap<String, TensorInfo>,
    /// GGUF name -> the safetensors name it actually lives under.
    src: HashMap<String, String>,
    pub arch: String,
}

/// safetensors dtype -> ggml type id, for the types a weight can be stored in.
///
/// ⚠ Only lossless mappings. An FP8 tensor has a companion scale and its ggml counterpart does not
/// mean the same thing, so it is refused here rather than handed over as if it were a block quant.
fn ggml_type_of(dtype: &str) -> Result<u32, String> {
    Ok(match dtype {
        "F32" => 0,
        "F16" => 1,
        "BF16" => 30,
        other => return Err(format!(
            "safetensors dtype '{other}' has no ggml equivalent that means the same thing{}",
            if is_scaled(other) { " (it is scale-carrying: the weight is this times a companion tensor)" } else { "" })),
    })
}

fn cfg_u(c: &serde_json::Value, k: &str) -> Option<u64> { c[k].as_u64() }
fn cfg_f(c: &serde_json::Value, k: &str) -> Option<f64> { c[k].as_f64() }

impl HfCheckpoint {
    /// Open a checkpoint directory holding `config.json` and safetensors (sharded or not).
    pub fn open(dir: impl AsRef<Path>) -> Result<HfCheckpoint, String> {
        let dir = dir.as_ref();
        let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("{}/config.json: {e}", dir.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_txt).map_err(|e| format!("config.json: {e}"))?;
        let model_type = cfg["model_type"].as_str()
            .ok_or("config.json has no model_type, so there is nothing to key the mapping on")?.to_string();
        let st = SafeTensors::open(dir)?;

        let (meta, name_map) = match model_type.as_str() {
            "lfm2" => lfm2_map(&cfg, &st)?,
            other => return Err(format!(
                "no HF mapping for model_type '{other}'. Adding one is a table of metadata keys and \
                 tensor names — see `lfm2_map` — but it is only worth adding alongside something \
                 that checks the weights land where the runtime thinks they do")),
        };

        // Reverse each shape and inherit the dtype. Offsets are meaningless here (bytes are fetched
        // by name), so they are left zero rather than faked into something a caller might trust.
        let mut infos = HashMap::new();
        let mut src = HashMap::new();
        for (gguf_name, hf_name) in name_map {
            let e = st.info(&hf_name).ok_or_else(|| format!(
                "{gguf_name} maps to {hf_name}, which this checkpoint does not contain"))?;
            let mut dims: Vec<u64> = e.shape.iter().rev().map(|&d| d as u64).collect();
            // A PyTorch depthwise conv is [C, 1, L] — reversed, [L, 1, C] — and the singleton
            // carries no information; GGUF stores [L, C]. Dropping it keeps the runtime's
            // dims[0]/dims[1] reads correct. Guarded to rank > 2 so a genuine [1, N] weight, where
            // the 1 IS the shape, is left alone.
            if dims.len() > 2 { dims.retain(|&d| d != 1); }
            infos.insert(gguf_name.clone(), TensorInfo {
                name: gguf_name.clone(), dims, ggml_type: ggml_type_of(&e.dtype)?, offset: 0,
            });
            src.insert(gguf_name, hf_name);
        }
        Ok(HfCheckpoint { st, meta, infos, src, arch: model_type })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> { self.infos.keys() }
}

impl GgufSource for HfCheckpoint {
    fn metadata(&self) -> &HashMap<String, Meta> { &self.meta }
    fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.infos.get(name) }
    fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let hf = self.src.get(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        self.st.raw(hf)
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        let hf = self.src.get(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        Ok(self.st.get(hf)?.data)
    }
}

/// **LFM2** — conv/attention hybrid. `layer_types` is the whole schedule.
///
/// ⭐ GGUF encodes the schedule as a per-layer `head_count_kv` array where **0 means this layer is a
/// conv block**, and HF encodes it as `layer_types: ["conv", "full_attention", ...]`. The two say
/// the same thing in different alphabets, which is what makes this translatable at all.
fn lfm2_map(cfg: &serde_json::Value, st: &SafeTensors)
    -> Result<(HashMap<String, Meta>, Vec<(String, String)>), String>
{
    let need_u = |k: &str| cfg_u(cfg, k).ok_or_else(|| format!("config.json: missing {k}"));
    let need_f = |k: &str| cfg_f(cfg, k).ok_or_else(|| format!("config.json: missing {k}"));
    let n_layer = need_u("num_hidden_layers")? as usize;
    let n_kv = need_u("num_key_value_heads")?;

    let types = cfg["layer_types"].as_array()
        .ok_or("config.json: lfm2 needs layer_types to know which blocks are attention")?;
    if types.len() != n_layer {
        return Err(format!("layer_types covers {} of {n_layer} blocks", types.len()));
    }
    let kv: Vec<Meta> = types.iter()
        .map(|t| Meta::U(if t.as_str() == Some("full_attention") { n_kv } else { 0 }))
        .collect();

    let mut m = HashMap::new();
    m.insert("general.architecture".into(), Meta::Str("lfm2".into()));
    m.insert("lfm2.block_count".into(), Meta::U(n_layer as u64));
    m.insert("lfm2.embedding_length".into(), Meta::U(need_u("hidden_size")?));
    m.insert("lfm2.attention.head_count".into(), Meta::U(need_u("num_attention_heads")?));
    m.insert("lfm2.attention.head_count_kv".into(), Meta::Arr(kv));
    m.insert("lfm2.attention.layer_norm_rms_epsilon".into(), Meta::F(need_f("norm_eps")?));
    m.insert("lfm2.rope.freq_base".into(), Meta::F(need_f("rope_theta")?));
    m.insert("lfm2.shortconv.l_cache".into(), Meta::U(need_u("conv_L_cache")?));

    let mut n: Vec<(String, String)> = vec![
        ("token_embd.weight".into(), "model.embed_tokens.weight".into()),
        // ⚠ NOT a norm on the embeddings — it is the FINAL norm, under a name llama.cpp routes
        // through a dedicated enum whose comment reads "fix for wrong tensor name". Mapping it to
        // anything else produces a model that runs and is subtly wrong at every layer.
        ("token_embd_norm.weight".into(), "model.embedding_norm.weight".into()),
    ];
    // Untied head only if the checkpoint actually has one; LFM2-350M ties it, and `Lfm2::load`
    // falls back to token_embd when output.weight is absent.
    if st.info("lm_head.weight").is_some() {
        n.push(("output.weight".into(), "lm_head.weight".into()));
    }
    for il in 0..n_layer {
        let attn = types[il].as_str() == Some("full_attention");
        let mut pairs: Vec<(&str, String)> = vec![
            ("attn_norm.weight", format!("model.layers.{il}.operator_norm.weight")),
            ("ffn_norm.weight", format!("model.layers.{il}.ffn_norm.weight")),
            // w1/w2/w3 are gate/down/up — NOT in that order, and the names give no hint. w2 is the
            // one whose in-dim is the FFN width, which is the only local way to tell them apart.
            ("ffn_gate.weight", format!("model.layers.{il}.feed_forward.w1.weight")),
            ("ffn_down.weight", format!("model.layers.{il}.feed_forward.w2.weight")),
            ("ffn_up.weight", format!("model.layers.{il}.feed_forward.w3.weight")),
        ];
        if attn {
            pairs.extend([
                ("attn_q.weight", format!("model.layers.{il}.self_attn.q_proj.weight")),
                ("attn_k.weight", format!("model.layers.{il}.self_attn.k_proj.weight")),
                ("attn_v.weight", format!("model.layers.{il}.self_attn.v_proj.weight")),
                ("attn_output.weight", format!("model.layers.{il}.self_attn.out_proj.weight")),
                ("attn_q_norm.weight", format!("model.layers.{il}.self_attn.q_layernorm.weight")),
                ("attn_k_norm.weight", format!("model.layers.{il}.self_attn.k_layernorm.weight")),
            ]);
        } else {
            pairs.extend([
                ("shortconv.in_proj.weight", format!("model.layers.{il}.conv.in_proj.weight")),
                ("shortconv.conv.weight", format!("model.layers.{il}.conv.conv.weight")),
                ("shortconv.out_proj.weight", format!("model.layers.{il}.conv.out_proj.weight")),
            ]);
        }
        for (suffix, hf) in pairs { n.push((format!("blk.{il}.{suffix}"), hf)); }
    }
    Ok((m, n))
}
