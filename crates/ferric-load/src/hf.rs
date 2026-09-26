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
            "qwen3_vl" => qwen3vl_map(&cfg, &st)?,
            "qwen2_5_vl" => qwen25vl_map(&cfg, &st)?,
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
/// **Qwen3-VL, text side.** The vision tower, the merger and the deepstack mergers are present in the
/// checkpoint and deliberately NOT mapped: llama.cpp zero-pads the text rows of its wider input
/// embedding, so every deepstack injection adds exactly 0 for a text token and omitting them is
/// arithmetically identical, not an approximation. Mapping them without the tower that feeds them
/// would be worse than leaving them out.
///
/// ⚠ Names are read from a real checkpoint (`Qwen3-VL-Embedding-2B`), not from a description: the
/// text stack sits under `model.language_model.*`, one level deeper than a text-only Qwen3, and the
/// attention output is `o_proj` (not `out_proj`, which is LFM2's spelling). Both are the kind of
/// difference that loads fine and produces confident nonsense.
fn qwen3vl_map(cfg: &serde_json::Value, st: &SafeTensors)
    -> Result<(HashMap<String, Meta>, Vec<(String, String)>), String>
{
    // Qwen3-VL nests everything real under `text_config`; a flat lookup silently finds nothing.
    let t = cfg.get("text_config").unwrap_or(cfg);
    let need_u = |k: &str| cfg_u(t, k).ok_or_else(|| format!("config.json text_config: missing {k}"));
    let need_f = |k: &str| cfg_f(t, k).ok_or_else(|| format!("config.json text_config: missing {k}"));
    let n_layer = need_u("num_hidden_layers")? as usize;
    let n_vocab = need_u("vocab_size")? as usize;

    // ⛔ The sections are the whole reason this arch exists as a separate mapping. Absent means the
    // runtime would fall back to ordinary RoPE and be wrong at every position — so this REFUSES
    // rather than defaulting.
    let sections: Vec<u64> = t.pointer("/rope_scaling/mrope_section")
        .and_then(|v| v.as_array())
        .ok_or("config.json: qwen3_vl needs rope_scaling.mrope_section; without it the model would \
                silently take ordinary RoPE")?
        .iter().filter_map(|x| x.as_u64()).collect();
    if sections.iter().sum::<u64>() == 0 {
        return Err("rope_scaling.mrope_section summed to 0".into());
    }

    let mut m = HashMap::new();
    m.insert("general.architecture".into(), Meta::Str("qwen3vl".into()));
    m.insert("qwen3vl.block_count".into(), Meta::U(n_layer as u64));
    m.insert("qwen3vl.embedding_length".into(), Meta::U(need_u("hidden_size")?));
    m.insert("qwen3vl.feed_forward_length".into(), Meta::U(need_u("intermediate_size")?));
    m.insert("qwen3vl.attention.head_count".into(), Meta::U(need_u("num_attention_heads")?));
    m.insert("qwen3vl.attention.head_count_kv".into(), Meta::U(need_u("num_key_value_heads")?));
    m.insert("qwen3vl.attention.key_length".into(), Meta::U(need_u("head_dim")?));
    m.insert("qwen3vl.attention.layer_norm_rms_epsilon".into(), Meta::F(need_f("rms_norm_eps")?));
    m.insert("qwen3vl.rope.freq_base".into(), Meta::F(need_f("rope_theta")?));
    m.insert("qwen3vl.rope.dimension_sections".into(),
             Meta::Arr(sections.iter().map(|s| Meta::U(*s)).collect()));
    // ⚠ The runtime takes n_vocab from the token list's LENGTH. This path feeds token IDs directly
    // and never tokenises text, so the strings are placeholders and only the count is load-bearing —
    // said here because a silently empty vocabulary would otherwise look like a tokenizer bug later.
    m.insert("tokenizer.ggml.tokens".into(),
             Meta::Arr(vec![Meta::Str(String::new()); n_vocab]));

    let mut n: Vec<(String, String)> = vec![
        ("token_embd.weight".into(), "model.language_model.embed_tokens.weight".into()),
        ("output_norm.weight".into(), "model.language_model.norm.weight".into()),
    ];
    // Untied head only if the checkpoint has one — Qwen3-VL-Embedding-2B sets tie_word_embeddings
    // and ships no `lm_head.weight`, and the runtime falls back to token_embd.
    if st.info("lm_head.weight").is_some() {
        n.push(("output.weight".into(), "lm_head.weight".into()));
    }
    for il in 0..n_layer {
        let p = format!("model.language_model.layers.{il}");
        for (suffix, hf) in [
            ("attn_norm.weight", format!("{p}.input_layernorm.weight")),
            ("attn_q.weight", format!("{p}.self_attn.q_proj.weight")),
            ("attn_k.weight", format!("{p}.self_attn.k_proj.weight")),
            ("attn_v.weight", format!("{p}.self_attn.v_proj.weight")),
            ("attn_output.weight", format!("{p}.self_attn.o_proj.weight")),
            ("attn_q_norm.weight", format!("{p}.self_attn.q_norm.weight")),
            ("attn_k_norm.weight", format!("{p}.self_attn.k_norm.weight")),
            ("ffn_norm.weight", format!("{p}.post_attention_layernorm.weight")),
            ("ffn_gate.weight", format!("{p}.mlp.gate_proj.weight")),
            ("ffn_up.weight", format!("{p}.mlp.up_proj.weight")),
            ("ffn_down.weight", format!("{p}.mlp.down_proj.weight")),
        ] {
            n.push((format!("blk.{il}.{suffix}"), hf));
        }
    }
    // Every mapped name must exist, or the failure surfaces as a missing tensor deep in load rather
    // than here where the mapping is visible.
    if let Some((g, h)) = n.iter().find(|(_, hf)| st.info(hf).is_none()) {
        return Err(format!("mapping points {g} at {h}, which the checkpoint does not contain"));
    }
    Ok((m, n))
}

/// **Qwen2.5-VL, text side** — MiMo-Embodied-7B's language model: a Qwen2 decoder (q/k/v biases, no
/// QK-norm) under CHUNKED multimodal rope, `mrope_section` `[16, 24, 24]`. The vision tower is loaded
/// separately (`ferric_llama::qwen25vl_vision`) from the same directory.
///
/// ⚠ Two key layouts exist for the same class, and both are real: checkpoints saved before
/// transformers 5 (MiMo-Embodied-7B, Qwen2.5-VL-7B-Instruct) keep the text model at `model.layers.N`,
/// the 5.x layout nests it at `model.language_model.layers.N`. Whichever the file holds is used;
/// a file holding both is refused rather than resolved by guessing.
///
/// ⚠ `sliding_window` is in the config and is NOT mapped: `use_sliding_window` is false, and the
/// authors' code then makes every layer `full_attention`. Mapping the width alone would window a
/// model that never windows.
fn qwen25vl_map(cfg: &serde_json::Value, st: &SafeTensors)
    -> Result<(HashMap<String, Meta>, Vec<(String, String)>), String>
{
    // 4.x configs are flat; 5.x ones nest under `text_config`. Read whichever carries the keys.
    let t = match cfg.get("text_config") { Some(tc) if tc.get("hidden_size").is_some() => tc, _ => cfg };
    let need_u = |k: &str| cfg_u(t, k).ok_or_else(|| format!("config.json: missing {k}"));
    let need_f = |k: &str| cfg_f(t, k).ok_or_else(|| format!("config.json: missing {k}"));
    let n_layer = need_u("num_hidden_layers")? as usize;
    let n_vocab = need_u("vocab_size")? as usize;
    if t["use_sliding_window"].as_bool() == Some(true) {
        return Err("use_sliding_window is true: the windowed layers are not mapped here".into());
    }
    let sections: Vec<u64> = t.pointer("/rope_scaling/mrope_section")
        .or_else(|| t.pointer("/rope_parameters/mrope_section"))
        .and_then(|v| v.as_array())
        .ok_or("config.json: qwen2_5_vl needs mrope_section; without it the model would silently \
                take ordinary RoPE")?
        .iter().filter_map(|x| x.as_u64()).collect();
    if sections.len() != 3 || sections.iter().sum::<u64>() == 0 {
        return Err(format!("mrope_section {sections:?} is not three non-empty sections"));
    }
    let theta = need_f("rope_theta")
        .or_else(|_| t.pointer("/rope_parameters/rope_theta").and_then(|x| x.as_f64())
            .ok_or_else(|| "config.json: missing rope_theta".to_string()))?;
    let (h, nh) = (need_u("hidden_size")?, need_u("num_attention_heads")?);
    // The sections cover head_dim/2 frequencies exactly; anything else leaves some unrotated or
    // rotates past the head, and neither fails.
    if sections.iter().sum::<u64>() * 2 != h / nh {
        return Err(format!("mrope_section {sections:?} does not cover head_dim {} / 2", h / nh));
    }
    let root = match (st.info("model.embed_tokens.weight"), st.info("model.language_model.embed_tokens.weight")) {
        (Some(_), None) => "model",
        (None, Some(_)) => "model.language_model",
        (Some(_), Some(_)) => return Err("checkpoint holds the text model under BOTH model.* and \
                                          model.language_model.* — refusing to pick one".into()),
        (None, None) => return Err("no embed_tokens in the checkpoint".into()),
    };

    let mut m = HashMap::new();
    m.insert("general.architecture".into(), Meta::Str("qwen2vl".into()));
    m.insert("qwen2vl.block_count".into(), Meta::U(n_layer as u64));
    m.insert("qwen2vl.embedding_length".into(), Meta::U(h));
    m.insert("qwen2vl.feed_forward_length".into(), Meta::U(need_u("intermediate_size")?));
    m.insert("qwen2vl.attention.head_count".into(), Meta::U(nh));
    m.insert("qwen2vl.attention.head_count_kv".into(), Meta::U(need_u("num_key_value_heads")?));
    m.insert("qwen2vl.attention.layer_norm_rms_epsilon".into(), Meta::F(need_f("rms_norm_eps")?));
    m.insert("qwen2vl.rope.freq_base".into(), Meta::F(theta));
    m.insert("qwen2vl.rope.dimension_sections".into(),
             Meta::Arr(sections.iter().map(|s| Meta::U(*s)).chain([Meta::U(0)]).collect()));
    // Only the COUNT is load-bearing: this path is fed token ids, never text. See qwen3vl_map.
    m.insert("tokenizer.ggml.tokens".into(), Meta::Arr(vec![Meta::Str(String::new()); n_vocab]));

    let mut n: Vec<(String, String)> = vec![
        ("token_embd.weight".into(), format!("{root}.embed_tokens.weight")),
        ("output_norm.weight".into(), format!("{root}.norm.weight")),
    ];
    if st.info("lm_head.weight").is_some() {
        n.push(("output.weight".into(), "lm_head.weight".into()));
    } else if t["tie_word_embeddings"].as_bool() != Some(true) {
        return Err("no lm_head.weight and tie_word_embeddings is not set — refusing to tie silently".into());
    }
    for il in 0..n_layer {
        let p = format!("{root}.layers.{il}");
        for (suffix, hf) in [
            ("attn_norm.weight", format!("{p}.input_layernorm.weight")),
            ("attn_q.weight", format!("{p}.self_attn.q_proj.weight")),
            ("attn_q.bias", format!("{p}.self_attn.q_proj.bias")),
            ("attn_k.weight", format!("{p}.self_attn.k_proj.weight")),
            ("attn_k.bias", format!("{p}.self_attn.k_proj.bias")),
            ("attn_v.weight", format!("{p}.self_attn.v_proj.weight")),
            ("attn_v.bias", format!("{p}.self_attn.v_proj.bias")),
            ("attn_output.weight", format!("{p}.self_attn.o_proj.weight")),
            ("ffn_norm.weight", format!("{p}.post_attention_layernorm.weight")),
            ("ffn_gate.weight", format!("{p}.mlp.gate_proj.weight")),
            ("ffn_up.weight", format!("{p}.mlp.up_proj.weight")),
            ("ffn_down.weight", format!("{p}.mlp.down_proj.weight")),
        ] {
            n.push((format!("blk.{il}.{suffix}"), hf));
        }
    }
    if let Some((g, hf)) = n.iter().find(|(_, hf)| st.info(hf).is_none()) {
        return Err(format!("mapping points {g} at {hf}, which the checkpoint does not contain"));
    }
    Ok((m, n))
}

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
