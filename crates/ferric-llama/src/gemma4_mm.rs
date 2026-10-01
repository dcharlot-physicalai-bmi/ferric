//! **Gemma 4's two towers share this**: where their weights come from, the CLIPPED linear both towers are
//! built of, and the embedder that projects a tower's output into the text width.
//!
//! Two sources, one set of names. The towers ask for every tensor by the AUTHORS' name
//! (`model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight`) and get it in the authors'
//! orientation, whichever file holds it:
//!
//! ```text
//! the authors' checkpoint directory   model.safetensors, read as stored (BF16)
//! ggml-org's mmproj GGUF              mmproj-gemma-4-*-it-{BF16,Q8_0}.gguf, names translated by `gguf_name`,
//!                                     the one re-laid-out tensor (the patch embedding) permuted back
//! ```
//!
//! The translation is a TABLE CHECKED BY VALUE, not a reading of llama.cpp's converter: every tower
//! tensor in the BF16 mmproj is bit-identical to the authors' file once mapped (`examples/gemma4_mm_weights`),
//! and every mmproj tensor is consumed exactly once. A wrong row in this table would load, run, and describe
//! the wrong image.
//!
//! ⛔ THE CLIPPED LINEAR. With `use_clipped_linears` (E2B, E4B, the audio tower everywhere) every projection
//! clamps its INPUT to `[input_min, input_max]` and its OUTPUT to `[output_min, output_max]` — four scalars
//! per linear, stored beside the weight as BUFFERS. They are not decoration: without them the vision tower's
//! output moves well outside the noise floor (the gate's `noclip` control). The 31B tower is built without
//! them (`use_clipped_linears: false`) and its checkpoint carries none, so absence is read from the CONFIG,
//! never inferred from a missing tensor.
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource};
use ferric_load::SafeTensors;
use ferric_tensor::{QMatrix, Tensor};
use std::sync::Arc;

/// Where a tower's weights live.
pub enum Src {
    /// The authors' checkpoint directory (`config.json` + safetensors).
    Hf { st: SafeTensors, cfg: serde_json::Value },
    /// A llama.cpp-style `mmproj` GGUF (`general.architecture = clip`).
    Gguf(Box<GgufFile>),
}

impl Src {
    /// A directory is the authors' checkpoint; a file is an mmproj GGUF.
    pub fn open(path: &str) -> Result<Src, String> {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            let cfg: serde_json::Value = serde_json::from_slice(
                &std::fs::read(p.join("config.json")).map_err(|e| format!("{path}/config.json: {e}"))?)
                .map_err(|e| format!("config.json: {e}"))?;
            if cfg["model_type"].as_str() != Some("gemma4") {
                return Err(format!("{path}: model_type {:?}, not gemma4", cfg["model_type"]));
            }
            Ok(Src::Hf { st: SafeTensors::open(p)?, cfg })
        } else {
            let g = GgufFile::open(p)?;
            match g.metadata().get("general.architecture") {
                Some(ferric_gguf::Meta::Str(a)) if a == "clip" => Ok(Src::Gguf(Box::new(g))),
                other => Err(format!("{path}: general.architecture {other:?}; an mmproj file says `clip`")),
            }
        }
    }

    pub fn is_gguf(&self) -> bool { matches!(self, Src::Gguf(_)) }

    /// The GGUF metadata, or `None` for a checkpoint directory.
    pub fn gguf(&self) -> Option<&GgufFile> { if let Src::Gguf(g) = self { Some(g) } else { None } }

    /// The authors' `config.json`, or `None` for an mmproj.
    pub fn config(&self) -> Option<&serde_json::Value> { if let Src::Hf { cfg, .. } = self { Some(cfg) } else { None } }

    /// The file's name for the authors' tensor `hf`.
    fn file_name(&self, hf: &str) -> Option<String> {
        match self {
            Src::Hf { .. } => Some(hf.to_string()),
            Src::Gguf(_) => gguf_name(hf),
        }
    }

    pub fn has(&self, hf: &str) -> bool {
        match (self, self.file_name(hf)) {
            (Src::Hf { st, .. }, Some(n)) => st.info(&n).is_some(),
            (Src::Gguf(g), Some(n)) => g.tensor(&n).is_some(),
            _ => false,
        }
    }

    /// Values and SHAPE IN THE AUTHORS' ORIENTATION (PyTorch order, slowest axis first).
    pub fn f32s(&self, hf: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        match self {
            Src::Hf { st, .. } => { let t = st.get(hf)?; Ok((t.data, t.shape)) }
            Src::Gguf(g) => {
                let n = gguf_name(hf).ok_or_else(|| format!("{hf}: no mmproj name for this tensor"))?;
                let info = g.tensor(&n).ok_or_else(|| format!("{n} (for {hf}) is not in the mmproj"))?;
                let mut shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
                let mut v = g.dequant(&n)?;
                if hf.ends_with("patch_embedder.input_proj.weight") {
                    // ⚠ The ONE re-laid-out tensor. The authors' input projection is a Linear over each
                    // patch flattened (row, col, channel) — channel fastest — because their processor
                    // patchifies before the model sees a pixel. The converter stores it as a CONVOLUTION
                    // kernel `[out, channel, row, col]`, so llama.cpp can patchify with conv2d. Same
                    // numbers, different order: permute back, or every patch reads its colours from the
                    // wrong pixels with no shape error anywhere.
                    if shape.len() != 4 || shape[1] != 3 || shape[2] != shape[3] {
                        return Err(format!("{n}: shape {shape:?}, expected [out, 3, p, p]"));
                    }
                    let (o, p) = (shape[0], shape[2]);
                    let mut w = vec![0f32; o * 3 * p * p];
                    for oi in 0..o { for c in 0..3 { for y in 0..p { for x in 0..p {
                        w[oi * 3 * p * p + (y * p + x) * 3 + c] = v[((oi * 3 + c) * p + y) * p + x];
                    } } } }
                    v = w;
                    shape = vec![o, 3 * p * p];
                }
                // A scalar is stored as a 1-element vector.
                if shape == [1] && ["input_min", "input_max", "output_min", "output_max"].iter().any(|b| hf.ends_with(b)) {
                    shape = vec![];
                }
                Ok((v, shape))
            }
        }
    }

    /// A 1-D tensor (a norm weight, a bias) checked to be `n` long.
    pub fn vec(&self, ctx: &Arc<Context>, hf: &str, n: usize) -> Result<Tensor, String> {
        let (v, s) = self.f32s(hf)?;
        if v.len() != n { return Err(format!("{hf}: {s:?}, expected [{n}]")); }
        Ok(Tensor::from_vec(ctx, &v, &[n]))
    }

    /// A `[out, in]` linear weight, kept in its stored type where the matmul kernels read it (BF16, Q8_0, …).
    pub fn qmat(&self, ctx: &Arc<Context>, hf: &str, out: usize, inp: usize) -> Result<QMatrix, String> {
        let (ty, raw, dims) = match self {
            Src::Hf { st, .. } => {
                let e = st.info(hf).ok_or_else(|| format!("no tensor '{hf}'"))?;
                let ty = match e.dtype.as_str() { "BF16" => 30, "F16" => 1, "F32" => 0, d => return Err(format!("{hf}: dtype {d}")) };
                (ty, st.raw(hf)?, e.shape.clone())
            }
            Src::Gguf(g) => {
                let n = gguf_name(hf).ok_or_else(|| format!("{hf}: no mmproj name"))?;
                let i = g.tensor(&n).ok_or_else(|| format!("{n} (for {hf}) is not in the mmproj"))?;
                (i.ggml_type, g.raw(&n)?, i.dims.iter().rev().map(|&d| d as usize).collect())
            }
        };
        if hf.ends_with("patch_embedder.input_proj.weight") && self.is_gguf() {
            let (v, s) = self.f32s(hf)?;
            if s != [out, inp] { return Err(format!("{hf}: {s:?}, expected [{out}, {inp}]")); }
            return Ok(QMatrix::from_dense(ctx, &v, out, inp));
        }
        if dims != [out, inp] { return Err(format!("{hf}: {dims:?}, expected [{out}, {inp}]")); }
        if ty != 0 && QMatrix::block_bytes(ty).is_some() {
            QMatrix::from_bytes(ctx, &raw, ty, out, inp)
        } else {
            Ok(QMatrix::from_dense(ctx, &self.f32s(hf)?.0, out, inp))
        }
    }

    /// One scalar buffer (a clipping bound).
    pub fn scalar(&self, hf: &str) -> Result<f32, String> {
        let (v, _) = self.f32s(hf)?;
        if v.len() != 1 { return Err(format!("{hf}: {} values, expected a scalar", v.len())); }
        Ok(v[0])
    }
}

/// The authors' tensor name → ggml-org's mmproj name. `None` = this tensor has no mmproj counterpart.
///
/// Read from a real mmproj (`mmproj-gemma-4-E2B-it-*.gguf`) and CHECKED BY VALUE against the authors' file,
/// tensor by tensor (`examples/gemma4_mm_weights.rs`). Two traps it encodes:
///   * the audio tower's `output_proj` is `a.pre_encode.out` — the name says "pre", the tensor is the LAST
///     projection (1024 -> 1536, with bias);
///   * the conformer's two feed-forwards are `ffn_*` and `ffn_*_1`, and its final norm is `ln2` — while the
///     VISION blocks' `ln2` is the PRE-feed-forward norm. Same short name, different place in the block.
pub fn gguf_name(hf: &str) -> Option<String> {
    // The clipped linear: `<module>.linear.weight` holds the weight, `<module>.{input,output}_{min,max}` the
    // bounds; the mmproj flattens both onto `<name>.weight` / `<name>.<bound>`.
    fn leaf(rest: &str, table: &[(&str, &str)]) -> Option<String> {
        for (h, g) in table {
            if let Some(tail) = rest.strip_prefix(h) {
                let tail = tail.strip_prefix('.').unwrap_or(tail);
                let tail = match tail {
                    "linear.weight" | "weight" | "" => "weight",
                    t @ ("input_min" | "input_max" | "output_min" | "output_max" | "bias") => t,
                    _ => return None,
                };
                return Some(format!("{g}.{tail}"));
            }
        }
        None
    }
    if let Some(r) = hf.strip_prefix("model.vision_tower.") {
        if r == "patch_embedder.input_proj.weight" { return Some("v.patch_embd.weight".into()); }
        if r == "patch_embedder.position_embedding_table" { return Some("v.position_embd.weight".into()); }
        if r == "std_bias" { return Some("v.std_bias".into()); }
        if r == "std_scale" { return Some("v.std_scale".into()); }
        let r = r.strip_prefix("encoder.layers.")?;
        let (i, rest) = r.split_once('.')?;
        let i: usize = i.parse().ok()?;
        return leaf(rest, &[
            ("input_layernorm", "ln1"), ("post_attention_layernorm", "attn_post_norm"),
            ("pre_feedforward_layernorm", "ln2"), ("post_feedforward_layernorm", "ffn_post_norm"),
            ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_out"), ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
            ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down"),
        ]).map(|s| format!("v.blk.{i}.{s}"));
    }
    if let Some(r) = hf.strip_prefix("model.audio_tower.") {
        for j in 0..2 {
            if r == format!("subsample_conv_projection.layer{j}.conv.weight") { return Some(format!("a.conv1d.{j}.weight")); }
            if r == format!("subsample_conv_projection.layer{j}.norm.weight") { return Some(format!("a.conv1d.{j}.norm.weight")); }
        }
        if r == "subsample_conv_projection.input_proj_linear.weight" { return Some("a.input_projection.weight".into()); }
        if r == "output_proj.weight" { return Some("a.pre_encode.out.weight".into()); }
        if r == "output_proj.bias" { return Some("a.pre_encode.out.bias".into()); }
        let r = r.strip_prefix("layers.")?;
        let (i, rest) = r.split_once('.')?;
        let i: usize = i.parse().ok()?;
        if rest == "self_attn.per_dim_scale" { return Some(format!("a.blk.{i}.per_dim_scale.weight")); }
        return leaf(rest, &[
            ("feed_forward1.ffw_layer_1", "ffn_up"), ("feed_forward1.ffw_layer_2", "ffn_down"),
            ("feed_forward1.pre_layer_norm", "ffn_norm"), ("feed_forward1.post_layer_norm", "ffn_post_norm"),
            ("feed_forward2.ffw_layer_1", "ffn_up_1"), ("feed_forward2.ffw_layer_2", "ffn_down_1"),
            ("feed_forward2.pre_layer_norm", "ffn_norm_1"), ("feed_forward2.post_layer_norm", "ffn_post_norm_1"),
            ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
            ("self_attn.post", "attn_out"), ("self_attn.relative_k_proj", "attn_k_rel"),
            ("norm_pre_attn", "attn_pre_norm"), ("norm_post_attn", "attn_post_norm"), ("norm_out", "ln2"),
            ("lconv1d.pre_layer_norm", "norm_conv"), ("lconv1d.conv_norm", "conv_norm"),
            ("lconv1d.linear_start", "conv_pw1"), ("lconv1d.linear_end", "conv_pw2"),
            ("lconv1d.depthwise_conv1d", "conv_dw"),
        ]).map(|s| format!("a.blk.{i}.{s}"));
    }
    match hf {
        "model.embed_vision.embedding_projection.weight" => Some("mm.input_projection.weight".into()),
        "model.embed_audio.embedding_projection.weight" => Some("mm.a.input_projection.weight".into()),
        _ => None,
    }
}

/// `torch.clamp(x, lo, hi)` — `min(max(x, lo), hi)` — built from `maximum`; an infinite bound is skipped.
pub fn clamp(x: &Tensor, lo: f32, hi: f32) -> Tensor {
    let x = if lo.is_finite() { x.maximum(&x.scalar(lo)) } else { x.clone() };
    if hi.is_finite() { x.neg().maximum(&x.scalar(-hi)).neg() } else { x }
}

/// A linear with optional clipping, `y = clamp(clamp(x, in) · Wᵀ, out)`.
pub struct Lin {
    pub w: QMatrix,
    /// `(input_min, input_max, output_min, output_max)` — present iff the config says `use_clipped_linears`.
    pub clip: Option<[f32; 4]>,
}

impl Lin {
    /// `module` is the authors' module path (`...self_attn.q_proj`). The weight sits at `.linear.weight`
    /// for a clippable linear and at `.weight` for a plain one.
    pub fn load(src: &Src, ctx: &Arc<Context>, module: &str, out: usize, inp: usize, clipped: bool) -> Result<Lin, String> {
        let wname = if src.has(&format!("{module}.linear.weight")) { format!("{module}.linear.weight") }
                    else { format!("{module}.weight") };
        let w = src.qmat(ctx, &wname, out, inp)?;
        let clip = if clipped {
            let b = |k: &str| src.scalar(&format!("{module}.{k}"));
            Some([b("input_min")?, b("input_max")?, b("output_min")?, b("output_max")?])
        } else { None };
        Ok(Lin { w, clip })
    }

    pub fn fwd(&self, x: &Tensor, use_clip: bool) -> Tensor {
        match (self.clip, use_clip) {
            (Some([a, b, c, d]), true) => clamp(&clamp(x, a, b).matmul_q(&self.w), c, d),
            _ => x.matmul_q(&self.w),
        }
    }
}

/// `Gemma4MultimodalEmbedder`: a WEIGHTLESS RMS norm at the tower's width, then a bias-free projection into
/// the text width. Its output rows replace the placeholder tokens' embeddings — and, unlike a text token's,
/// they are NOT multiplied by `sqrt(hidden)`.
pub struct Embedder {
    pub proj: QMatrix,
    pub eps: f32,
    pub text_d: usize,
}

impl Embedder {
    pub fn fwd(&self, x: &Tensor) -> Tensor { x.rmsnorm_weightless(self.eps).matmul_q(&self.proj) }
}

/// `x` with rows `start .. start + rows.len()` replaced by `rows` — the authors' `masked_scatter` over one
/// contiguous placeholder run.
pub fn splice_rows(x: &Tensor, start: usize, rows: &Tensor) -> Tensor {
    let (t, n) = (x.shape[0], rows.shape[0]);
    assert!(start + n <= t, "splice of {n} rows at {start} overruns {t}");
    assert_eq!(x.shape[1], rows.shape[1], "splice width");
    let mut parts: Vec<Tensor> = Vec::new();
    if start > 0 { parts.push(x.narrow(0, 0, start).contiguous()); }
    parts.push(rows.clone());
    if start + n < t { parts.push(x.narrow(0, start + n, t - start - n).contiguous()); }
    parts[1..].iter().fold(parts[0].clone(), |a, b| a.cat(b, 0))
}

/// Every mmproj tensor that `gguf_name` maps an authors' name ONTO — the reverse direction, for checking
/// that nothing in the file is left unread.
pub fn mmproj_names(g: &GgufFile) -> Vec<String> { g.tensors.iter().map(|t| t.name.clone()).collect() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_table_maps_the_two_traps_the_short_names_hide() {
        // vision `ln2` is the PRE-feed-forward norm; audio `ln2` is the block's OUTPUT norm.
        assert_eq!(gguf_name("model.vision_tower.encoder.layers.3.pre_feedforward_layernorm.weight").as_deref(), Some("v.blk.3.ln2.weight"));
        assert_eq!(gguf_name("model.audio_tower.layers.3.norm_out.weight").as_deref(), Some("a.blk.3.ln2.weight"));
        // the audio tower's LAST projection is called `pre_encode.out`.
        assert_eq!(gguf_name("model.audio_tower.output_proj.weight").as_deref(), Some("a.pre_encode.out.weight"));
        assert_eq!(gguf_name("model.audio_tower.output_proj.bias").as_deref(), Some("a.pre_encode.out.bias"));
        // the clipped linear flattens `.linear.weight` and keeps the bound names.
        assert_eq!(gguf_name("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight").as_deref(), Some("v.blk.0.attn_q.weight"));
        assert_eq!(gguf_name("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max").as_deref(), Some("v.blk.0.attn_q.input_max"));
        assert_eq!(gguf_name("model.audio_tower.layers.11.feed_forward2.ffw_layer_1.output_min").as_deref(), Some("a.blk.11.ffn_up_1.output_min"));
        // an unknown leaf is NOT silently mapped onto something near it.
        assert_eq!(gguf_name("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.bias_x"), None);
        assert_eq!(gguf_name("model.language_model.layers.0.self_attn.q_proj.weight"), None);
    }
}
