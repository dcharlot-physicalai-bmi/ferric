//! SmolVLA forward in f32.
//!
//! The graph is the LeRobot `a0eb860d` `sample_actions` path:
//! SigLIP-style vision, scale-4 pixel shuffle, connector, text prefix,
//! then the expert with self-attention on even layers and cross-attention
//! on odd layers, integrated for 10 Euler steps.
//!
//! Vision GELU is `gelu_pytorch_tanh`. Text and expert norms are RMSNorm
//! with eps `1e-5`. RoPE wavelength is 10000, the value `apply_rope` uses,
//! not the Llama config's `rope_theta`.

use ferric_core::cpu::{layernorm, silu, softmax};

use crate::gemm::{gemm_nn, gemm_nt, linear};
use crate::weights::Store;

const VISION_LAYERS: usize = 12;
const TEXT_LAYERS: usize = 16;
const VISION_DIM: usize = 768;
const TEXT_DIM: usize = 960;
const EXPERT_DIM: usize = 720;
const HEADS: usize = 15;
const KV_HEADS: usize = 5;
const HEAD_DIM: usize = 64;
const VISION_HEADS: usize = 12;
const VISION_INTER: usize = 3072;
const TEXT_INTER: usize = 2560;
const EXPERT_INTER: usize = 2048;
const PATCH: usize = 16;
const IMAGE: usize = 512;
const GRID: usize = IMAGE / PATCH;
const PATCHES: usize = GRID * GRID;
const CONNECTOR_SCALE: usize = 4;
const IMG_TOKENS: usize = PATCHES / (CONNECTOR_SCALE * CONNECTOR_SCALE);
const CHUNK: usize = 50;
const MAX_ACTION: usize = 32;
const MAX_STATE: usize = 32;
const ACTION_DIM: usize = 6;
const LANG: usize = 48;
const STEPS: usize = 10;
const RMS_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-6;
const ROPE_BASE: f32 = 10_000.0;

pub struct Sample {
    pub images: Vec<Vec<f32>>,
    pub image_hw: Vec<(usize, usize)>,
    pub state: Vec<f32>,
    pub token_ids: Vec<u32>,
    pub token_mask: Vec<bool>,
    pub noise: Vec<f32>,
    pub state_mean: Option<Vec<f32>>,
    pub state_std: Option<Vec<f32>>,
    pub action_mean: Option<Vec<f32>>,
    pub action_std: Option<Vec<f32>>,
    /// The schema is MEAN_STD. This flag is the only way to run the tensor
    /// path when those stats are absent, and the result is not executable.
    pub allow_unnormalized_network: bool,
    pub capture: bool,
}

pub struct VisionSnap {
    pub patch_plus_pos_token0: Vec<f32>,
    pub vision_layer0_token0: Vec<f32>,
    pub vision_post_token0: Vec<f32>,
    pub connector_token0: Vec<f32>,
}

pub struct ActivationTrace {
    pub prepared_image0: Vec<f32>,
    pub vision: VisionSnap,
    pub state_token: Vec<f32>,
    pub text_layer0_k_token0: Vec<f32>,
    pub velocity0: Vec<f32>,
    pub raw_actions: Vec<f32>,
}

pub struct Forward {
    pub actions: Vec<f32>,
    pub raw_actions: Vec<f32>,
    pub state_norm: &'static str,
    pub action_norm: &'static str,
    pub executable: bool,
    pub trace: Option<ActivationTrace>,
}

pub fn forward(store: &Store, sample: &Sample) -> Result<Forward, String> {
    if sample.images.len() != 3 {
        return Err(format!(
            "this checkpoint has 3 cameras, got {}",
            sample.images.len()
        ));
    }
    if sample.state.len() != ACTION_DIM {
        return Err(format!(
            "state length {} is not {ACTION_DIM}",
            sample.state.len()
        ));
    }
    if sample.token_ids.len() != LANG || sample.token_mask.len() != LANG {
        return Err("language tokens must be length 48".into());
    }
    if sample.noise.len() != CHUNK * MAX_ACTION {
        return Err(format!(
            "noise length {} is not {}",
            sample.noise.len(),
            CHUNK * MAX_ACTION
        ));
    }
    let (state32, state_norm) = normalize_state(
        &sample.state,
        sample.state_mean.as_deref(),
        sample.state_std.as_deref(),
        sample.allow_unnormalized_network,
    )?;
    let mut prefix = Vec::new();
    let mut pad = Vec::new();
    let mut att = Vec::new();
    let mut prepared_image0 = Vec::new();
    let mut vision = None;
    for image in 0..3 {
        eprintln!("vision camera {image}");
        let (h, w) = sample.image_hw[image];
        let pixels = prepare_image(&sample.images[image], h, w)?;
        if image == 0 && sample.capture {
            prepared_image0 = pixels.clone();
        }
        let (tokens, snap) = embed_image(store, &pixels, image == 0 && sample.capture)?;
        if image == 0 {
            vision = snap;
        }
        let n = tokens.len() / TEXT_DIM;
        if n != IMG_TOKENS {
            return Err(format!("image tokens {n}, expected {IMG_TOKENS}"));
        }
        prefix.extend_from_slice(&scale_rows(&tokens, n, TEXT_DIM));
        pad.extend(std::iter::repeat_n(true, n));
        att.extend(std::iter::repeat_n(0u8, n));
    }
    let lang = embed_tokens(store, &sample.token_ids)?;
    prefix.extend_from_slice(&scale_rows(&lang, LANG, TEXT_DIM));
    pad.extend_from_slice(&sample.token_mask);
    att.extend(std::iter::repeat_n(0u8, LANG));
    let state = linear(
        &state32,
        &store.tensor("model.state_proj.weight")?,
        Some(&store.tensor("model.state_proj.bias")?),
        1,
        MAX_STATE,
        TEXT_DIM,
    );
    prefix.extend_from_slice(&state);
    pad.push(true);
    att.push(1);
    let seq = pad.len();
    let mask = att_2d(&pad, &att);
    let pos = positions(&pad);
    let (cache, text_k0) = text_prefix(store, &prefix, seq, &mask, &pos, sample.capture)?;
    let mut x = sample.noise.clone();
    let dt = -1.0 / STEPS as f32;
    let mut velocity0 = Vec::new();
    for step in 0..STEPS {
        eprintln!("euler step {step}");
        let time = 1.0 + step as f32 * dt;
        let v = denoise(store, &cache, &pad, &x, time)?;
        if step == 0 && sample.capture {
            velocity0 = v[..MAX_ACTION].to_vec();
        }
        for (dst, vel) in x.iter_mut().zip(&v) {
            *dst += dt * vel;
        }
    }
    let raw_actions = x.clone();
    let (actions, action_norm) = unpad_and_unnormalize(
        &x,
        sample.action_mean.as_deref(),
        sample.action_std.as_deref(),
        sample.allow_unnormalized_network,
    )?;
    let executable = state_norm == "mean_std" && action_norm == "mean_std_inverse";
    let trace = if sample.capture {
        Some(ActivationTrace {
            prepared_image0,
            vision: vision.unwrap_or(VisionSnap {
                patch_plus_pos_token0: Vec::new(),
                vision_layer0_token0: Vec::new(),
                vision_post_token0: Vec::new(),
                connector_token0: Vec::new(),
            }),
            state_token: state.clone(),
            text_layer0_k_token0: text_k0,
            velocity0,
            raw_actions: raw_actions.clone(),
        })
    } else {
        None
    };
    Ok(Forward {
        actions,
        raw_actions,
        state_norm,
        action_norm,
        executable,
        trace,
    })
}

fn normalize_state(
    state: &[f32],
    mean: Option<&[f32]>,
    std: Option<&[f32]>,
    allow_unnormalized_network: bool,
) -> Result<(Vec<f32>, &'static str), String> {
    let mut out = vec![0.0f32; MAX_STATE];
    let mode = match (mean, std) {
        (Some(mean), Some(std)) => {
            if mean.len() != state.len() || std.len() != state.len() {
                return Err("state mean/std length does not match the state".into());
            }
            for i in 0..state.len() {
                out[i] = (state[i] - mean[i]) / (std[i] + 1e-8);
            }
            "mean_std"
        }
        (None, None) if allow_unnormalized_network => {
            out[..state.len()].copy_from_slice(state);
            "unnormalized_network_forward"
        }
        (None, None) => {
            return Err("MEAN_STD requires observation.state mean and std. The schema is not IDENTITY, and a missing key does not become an execution.".into());
        }
        _ => return Err("state mean and std must both be present".into()),
    };
    Ok((out, mode))
}

fn unpad_and_unnormalize(
    x: &[f32],
    mean: Option<&[f32]>,
    std: Option<&[f32]>,
    allow_unnormalized_network: bool,
) -> Result<(Vec<f32>, &'static str), String> {
    let mut actions = Vec::with_capacity(CHUNK * ACTION_DIM);
    for t in 0..CHUNK {
        actions.extend_from_slice(&x[t * MAX_ACTION..t * MAX_ACTION + ACTION_DIM]);
    }
    let mode = match (mean, std) {
        (Some(mean), Some(std)) => {
            if mean.len() != ACTION_DIM || std.len() != ACTION_DIM {
                return Err("action mean/std must have length 6".into());
            }
            for t in 0..CHUNK {
                for d in 0..ACTION_DIM {
                    let i = t * ACTION_DIM + d;
                    actions[i] = actions[i] * std[d] + mean[d];
                }
            }
            "mean_std_inverse"
        }
        (None, None) if allow_unnormalized_network => "not_unnormalized",
        (None, None) => {
            return Err("MEAN_STD requires action mean and std from the pinned embodiment map. A missing key does not become an execution.".into());
        }
        _ => return Err("action mean and std must both be present".into()),
    };
    Ok((actions, mode))
}

/// v0.7.0 PR #4292 path. `forward` does not call this.
/// uint8 pixels are divided by 255 before the existing `[0, 1]` resize.
pub fn prepare_uint8_v070(nchw: &[u8], h: usize, w: usize) -> Result<Vec<f32>, String> {
    if nchw.len() != 3 * h * w {
        return Err(format!(
            "uint8 image bytes are {}, shape is 3x{h}x{w}",
            nchw.len()
        ));
    }
    let unit: Vec<f32> = nchw.iter().map(|p| f32::from(*p) / 255.0).collect();
    prepare_image(&unit, h, w)
}

pub(crate) fn prepare_image(nchw: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
    if nchw.len() != 3 * h * w {
        return Err(format!(
            "image bytes are {}, shape is 3x{h}x{w}",
            nchw.len()
        ));
    }
    if nchw.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0) {
        return Err("image pixels must be finite and inside [0, 1]".into());
    }
    let resized = resize_with_pad(nchw, h, w, IMAGE, IMAGE)?;
    Ok(resized.iter().map(|v| v * 2.0 - 1.0).collect())
}

pub fn resize_with_pad(
    nchw: &[f32],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>, String> {
    if h == out_h && w == out_w {
        return Ok(nchw.to_vec());
    }
    let ratio = (w as f32 / out_w as f32).max(h as f32 / out_h as f32);
    let resized_h = (h as f32 / ratio) as usize;
    let resized_w = (w as f32 / ratio) as usize;
    if resized_h == 0 || resized_w == 0 || resized_h > out_h || resized_w > out_w {
        return Err(format!(
            "resize {h}x{w} -> {resized_h}x{resized_w} does not fit {out_h}x{out_w}"
        ));
    }
    let scaled = bilinear(nchw, h, w, resized_h, resized_w);
    let mut out = vec![0.0f32; 3 * out_h * out_w];
    let pad_h = out_h - resized_h;
    let pad_w = out_w - resized_w;
    for c in 0..3 {
        for y in 0..resized_h {
            for x in 0..resized_w {
                let src = c * resized_h * resized_w + y * resized_w + x;
                let dst = c * out_h * out_w + (y + pad_h) * out_w + (x + pad_w);
                out[dst] = scaled[src];
            }
        }
    }
    Ok(out)
}

fn bilinear(nchw: &[f32], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * out_h * out_w];
    for c in 0..3 {
        for oy in 0..out_h {
            let (y0, y1, ly0, ly1) = source_index(oy, h, out_h);
            for ox in 0..out_w {
                let (x0, x1, lx0, lx1) = source_index(ox, w, out_w);
                let at = |yy: usize, xx: usize| nchw[c * h * w + yy * w + xx];
                let v = ly0 * lx0 * at(y0, x0)
                    + ly0 * lx1 * at(y0, x1)
                    + ly1 * lx0 * at(y1, x0)
                    + ly1 * lx1 * at(y1, x1);
                out[c * out_h * out_w + oy * out_w + ox] = v;
            }
        }
    }
    out
}

fn source_index(
    output_index: usize,
    input_size: usize,
    output_size: usize,
) -> (usize, usize, f32, f32) {
    if output_size == input_size {
        return (output_index, output_index, 1.0, 0.0);
    }
    let scale = input_size as f32 / output_size as f32;
    let mut src = scale * (output_index as f32 + 0.5) - 0.5;
    if src < 0.0 {
        src = 0.0;
    }
    let mut i0 = src.floor() as usize;
    if i0 > input_size - 1 {
        i0 = input_size - 1;
    }
    let lambda1 = (src - i0 as f32).clamp(0.0, 1.0);
    let i1 = if i0 < input_size - 1 { i0 + 1 } else { i0 };
    (i0, i1, 1.0 - lambda1, lambda1)
}

fn embed_image(store: &Store, pixels: &[f32], capture: bool) -> Result<(Vec<f32>, Option<VisionSnap>), String> {
    let weight = store
        .tensor("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight")?;
    let bias = store
        .tensor("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias")?;
    let pos = store.tensor(
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight",
    )?;
    let mut hidden = patch_embed(pixels, &weight, &bias);
    if pos.len() != PATCHES * VISION_DIM {
        return Err(format!("position embedding length {}", pos.len()));
    }
    for (h, p) in hidden.iter_mut().zip(&pos) {
        *h += p;
    }
    let patch_plus_pos_token0 = if capture { hidden[..VISION_DIM].to_vec() } else { Vec::new() };
    let mut vision_layer0_token0 = Vec::new();
    for layer in 0..VISION_LAYERS {
        hidden = vision_layer(store, layer, &hidden)?;
        if capture && layer == 0 {
            vision_layer0_token0 = hidden[..VISION_DIM].to_vec();
        }
    }
    let nw = store.tensor("model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight")?;
    let nb = store.tensor("model.vlm_with_expert.vlm.model.vision_model.post_layernorm.bias")?;
    hidden = layernorm(&hidden, &nw, &nb, PATCHES, VISION_DIM, LN_EPS);
    let vision_post_token0 = if capture { hidden[..VISION_DIM].to_vec() } else { Vec::new() };
    let shuffled = pixel_shuffle(&hidden, PATCHES, VISION_DIM, CONNECTOR_SCALE)?;
    let proj = store
        .tensor("model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight")?;
    let inn = VISION_DIM * CONNECTOR_SCALE * CONNECTOR_SCALE;
    let tokens = linear(&shuffled, &proj, None, IMG_TOKENS, inn, TEXT_DIM);
    let snap = if capture {
        Some(VisionSnap {
            patch_plus_pos_token0,
            vision_layer0_token0,
            vision_post_token0,
            connector_token0: tokens[..TEXT_DIM].to_vec(),
        })
    } else {
        None
    };
    Ok((tokens, snap))
}

fn patch_embed(pixels: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let inn = 3 * PATCH * PATCH;
    let mut cols = vec![0.0f32; PATCHES * inn];
    for hy in 0..GRID {
        for wx in 0..GRID {
            let p = hy * GRID + wx;
            for c in 0..3 {
                for ky in 0..PATCH {
                    for kx in 0..PATCH {
                        let src = c * IMAGE * IMAGE + (hy * PATCH + ky) * IMAGE + (wx * PATCH + kx);
                        let dst = p * inn + c * PATCH * PATCH + ky * PATCH + kx;
                        cols[dst] = pixels[src];
                    }
                }
            }
        }
    }
    linear(&cols, weight, Some(bias), PATCHES, inn, VISION_DIM)
}

pub fn pixel_shuffle(x: &[f32], seq: usize, dim: usize, scale: usize) -> Result<Vec<f32>, String> {
    let side = (seq as f32).sqrt() as usize;
    if side * side != seq || side % scale != 0 {
        return Err(format!("pixel shuffle seq {seq} scale {scale}"));
    }
    let w_out = side / scale;
    let h_out = side / scale;
    let c1 = dim * scale;
    let mut perm1 = vec![0.0f32; seq * dim];
    for y in 0..side {
        for x_out in 0..w_out {
            for c1i in 0..c1 {
                let src_x = x_out * scale + c1i / dim;
                let src_c = c1i % dim;
                perm1[(x_out * side + y) * c1 + c1i] = x[(y * side + src_x) * dim + src_c];
            }
        }
    }
    let c2 = dim * scale * scale;
    let mut reshaped = vec![0.0f32; w_out * h_out * c2];
    for x_out in 0..w_out {
        for y in 0..side {
            for c1i in 0..c1 {
                let y0 = y / scale;
                let dst_c = (y % scale) * c1 + c1i;
                reshaped[(x_out * h_out + y0) * c2 + dst_c] = perm1[(x_out * side + y) * c1 + c1i];
            }
        }
    }
    let mut out = vec![0.0f32; reshaped.len()];
    for y0 in 0..h_out {
        for x_out in 0..w_out {
            for c in 0..c2 {
                out[(y0 * w_out + x_out) * c2 + c] = reshaped[(x_out * h_out + y0) * c2 + c];
            }
        }
    }
    Ok(out)
}

fn vision_layer(store: &Store, layer: usize, hidden: &[f32]) -> Result<Vec<f32>, String> {
    let p = format!("model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{layer}");
    let n1w = store.tensor(&format!("{p}.layer_norm1.weight"))?;
    let n1b = store.tensor(&format!("{p}.layer_norm1.bias"))?;
    let normed = layernorm(hidden, &n1w, &n1b, PATCHES, VISION_DIM, LN_EPS);
    let q = linear(
        &normed,
        &store.tensor(&format!("{p}.self_attn.q_proj.weight"))?,
        Some(&store.tensor(&format!("{p}.self_attn.q_proj.bias"))?),
        PATCHES,
        VISION_DIM,
        VISION_DIM,
    );
    let k = linear(
        &normed,
        &store.tensor(&format!("{p}.self_attn.k_proj.weight"))?,
        Some(&store.tensor(&format!("{p}.self_attn.k_proj.bias"))?),
        PATCHES,
        VISION_DIM,
        VISION_DIM,
    );
    let v = linear(
        &normed,
        &store.tensor(&format!("{p}.self_attn.v_proj.weight"))?,
        Some(&store.tensor(&format!("{p}.self_attn.v_proj.bias"))?),
        PATCHES,
        VISION_DIM,
        VISION_DIM,
    );
    let attn = attention(
        &q,
        &k,
        &v,
        PATCHES,
        PATCHES,
        VISION_HEADS,
        VISION_HEADS,
        HEAD_DIM,
        None,
    );
    let projected = linear(
        &attn,
        &store.tensor(&format!("{p}.self_attn.out_proj.weight"))?,
        Some(&store.tensor(&format!("{p}.self_attn.out_proj.bias"))?),
        PATCHES,
        VISION_DIM,
        VISION_DIM,
    );
    let mut residual = ferric_core::cpu::add(hidden, &projected);
    let n2w = store.tensor(&format!("{p}.layer_norm2.weight"))?;
    let n2b = store.tensor(&format!("{p}.layer_norm2.bias"))?;
    let normed = layernorm(&residual, &n2w, &n2b, PATCHES, VISION_DIM, LN_EPS);
    let mid = linear(
        &normed,
        &store.tensor(&format!("{p}.mlp.fc1.weight"))?,
        Some(&store.tensor(&format!("{p}.mlp.fc1.bias"))?),
        PATCHES,
        VISION_DIM,
        VISION_INTER,
    );
    let mid = gelu_tanh(&mid);
    let mid = linear(
        &mid,
        &store.tensor(&format!("{p}.mlp.fc2.weight"))?,
        Some(&store.tensor(&format!("{p}.mlp.fc2.bias"))?),
        PATCHES,
        VISION_INTER,
        VISION_DIM,
    );
    for (a, b) in residual.iter_mut().zip(&mid) {
        *a += b;
    }
    Ok(residual)
}

pub(crate) fn gelu_tanh(x: &[f32]) -> Vec<f32> {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
        .collect()
}

fn embed_tokens(store: &Store, ids: &[u32]) -> Result<Vec<f32>, String> {
    let table = store.tensor("model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight")?;
    let mut out = vec![0.0f32; ids.len() * TEXT_DIM];
    for (i, id) in ids.iter().enumerate() {
        let id = *id as usize;
        if id >= table.len() / TEXT_DIM {
            return Err(format!("token id {id} is outside the embedding table"));
        }
        out[i * TEXT_DIM..(i + 1) * TEXT_DIM]
            .copy_from_slice(&table[id * TEXT_DIM..(id + 1) * TEXT_DIM]);
    }
    Ok(out)
}

fn scale_rows(x: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let s = (dim as f32).sqrt();
    x.iter().take(rows * dim).map(|v| v * s).collect()
}

struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

fn text_prefix(
    store: &Store,
    hidden0: &[f32],
    seq: usize,
    mask: &[bool],
    pos: &[u32],
    capture: bool,
) -> Result<(Vec<LayerCache>, Vec<f32>), String> {
    let mut hidden = hidden0.to_vec();
    let mut cache = Vec::with_capacity(TEXT_LAYERS);
    let mut k0 = Vec::new();
    for layer in 0..TEXT_LAYERS {
        let p = format!("model.vlm_with_expert.vlm.model.text_model.layers.{layer}");
        let normed = rmsnorm(
            &store.tensor(&format!("{p}.input_layernorm.weight"))?,
            &hidden,
            seq,
            TEXT_DIM,
        );
        let q = linear(
            &normed,
            &store.tensor(&format!("{p}.self_attn.q_proj.weight"))?,
            None,
            seq,
            TEXT_DIM,
            HEADS * HEAD_DIM,
        );
        let k = linear(
            &normed,
            &store.tensor(&format!("{p}.self_attn.k_proj.weight"))?,
            None,
            seq,
            TEXT_DIM,
            KV_HEADS * HEAD_DIM,
        );
        let v = linear(
            &normed,
            &store.tensor(&format!("{p}.self_attn.v_proj.weight"))?,
            None,
            seq,
            TEXT_DIM,
            KV_HEADS * HEAD_DIM,
        );
        let (q, k) = (rope(&q, pos, HEADS)?, rope(&k, pos, KV_HEADS)?);
        if capture && layer == 0 {
            k0 = k[..KV_HEADS * HEAD_DIM].to_vec();
        }
        cache.push(LayerCache {
            k: k.clone(),
            v: v.clone(),
        });
        let attn = attention(&q, &k, &v, seq, seq, HEADS, KV_HEADS, HEAD_DIM, Some(mask));
        let projected = linear(
            &attn,
            &store.tensor(&format!("{p}.self_attn.o_proj.weight"))?,
            None,
            seq,
            TEXT_DIM,
            TEXT_DIM,
        );
        let mut residual = ferric_core::cpu::add(&hidden, &projected);
        hidden = mlp(
            &store,
            &format!("{p}.mlp"),
            &format!("{p}.post_attention_layernorm.weight"),
            &residual,
            seq,
            TEXT_DIM,
            TEXT_INTER,
        )?;
        for (a, b) in hidden.iter_mut().zip(&residual) {
            *a += b;
        }
        residual = hidden;
        hidden = residual;
    }
    let norm = store.tensor("model.vlm_with_expert.vlm.model.text_model.norm.weight")?;
    let _ = rmsnorm(&norm, &hidden, seq, TEXT_DIM);
    Ok((cache, k0))
}

fn denoise(
    store: &Store,
    cache: &[LayerCache],
    prefix_pad: &[bool],
    x_t: &[f32],
    time: f32,
) -> Result<Vec<f32>, String> {
    let prefix = prefix_pad.len();
    let suffix = embed_suffix(store, x_t, time)?;
    let mut hidden = suffix;
    let offset = prefix_pad.iter().filter(|p| **p).count() as u32;
    let pos: Vec<u32> = (0..CHUNK).map(|i| offset + i as u32).collect();
    let pos0: Vec<u32> = (0..CHUNK as u32).collect();
    for layer in 0..TEXT_LAYERS {
        let p = format!("model.vlm_with_expert.lm_expert.layers.{layer}");
        let normed = rmsnorm(
            &store.tensor(&format!("{p}.input_layernorm.weight"))?,
            &hidden,
            CHUNK,
            EXPERT_DIM,
        );
        let q = linear(
            &normed,
            &store.tensor(&format!("{p}.self_attn.q_proj.weight"))?,
            None,
            CHUNK,
            EXPERT_DIM,
            HEADS * HEAD_DIM,
        );
        let attn = if layer % 2 == 0 {
            let k = linear(
                &normed,
                &store.tensor(&format!("{p}.self_attn.k_proj.weight"))?,
                None,
                CHUNK,
                EXPERT_DIM,
                KV_HEADS * HEAD_DIM,
            );
            let v = linear(
                &normed,
                &store.tensor(&format!("{p}.self_attn.v_proj.weight"))?,
                None,
                CHUNK,
                EXPERT_DIM,
                KV_HEADS * HEAD_DIM,
            );
            let q = rope(&q, &pos, HEADS)?;
            let k = rope(&k, &pos, KV_HEADS)?;
            let (k, v) = concat_kv(&cache[layer].k, &k, &cache[layer].v, &v, prefix, CHUNK);
            let mask = self_mask(prefix_pad, CHUNK);
            attention(
                &q,
                &k,
                &v,
                CHUNK,
                prefix + CHUNK,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                Some(&mask),
            )
        } else {
            let flat_k = flatten_heads(&cache[layer].k, prefix, KV_HEADS, HEAD_DIM);
            let flat_v = flatten_heads(&cache[layer].v, prefix, KV_HEADS, HEAD_DIM);
            let k = linear(
                &flat_k,
                &store.tensor(&format!("{p}.self_attn.k_proj.weight"))?,
                None,
                prefix,
                KV_HEADS * HEAD_DIM,
                KV_HEADS * HEAD_DIM,
            );
            let v = linear(
                &flat_v,
                &store.tensor(&format!("{p}.self_attn.v_proj.weight"))?,
                None,
                prefix,
                KV_HEADS * HEAD_DIM,
                KV_HEADS * HEAD_DIM,
            );
            let q = rope(&q, &pos0, HEADS)?;
            let mask = cross_mask(prefix_pad);
            attention(
                &q,
                &k,
                &v,
                CHUNK,
                prefix,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                Some(&mask),
            )
        };
        let projected = linear(
            &attn,
            &store.tensor(&format!("{p}.self_attn.o_proj.weight"))?,
            None,
            CHUNK,
            HEADS * HEAD_DIM,
            EXPERT_DIM,
        );
        let residual = ferric_core::cpu::add(&hidden, &projected);
        let mut next = mlp(
            &store,
            &format!("{p}.mlp"),
            &format!("{p}.post_attention_layernorm.weight"),
            &residual,
            CHUNK,
            EXPERT_DIM,
            EXPERT_INTER,
        )?;
        for (a, b) in next.iter_mut().zip(&residual) {
            *a += b;
        }
        hidden = next;
    }
    let normed = rmsnorm(
        &store.tensor("model.vlm_with_expert.lm_expert.norm.weight")?,
        &hidden,
        CHUNK,
        EXPERT_DIM,
    );
    let v = linear(
        &normed,
        &store.tensor("model.action_out_proj.weight")?,
        Some(&store.tensor("model.action_out_proj.bias")?),
        CHUNK,
        EXPERT_DIM,
        MAX_ACTION,
    );
    Ok(v)
}

fn embed_suffix(store: &Store, noisy: &[f32], time: f32) -> Result<Vec<f32>, String> {
    let action = linear(
        noisy,
        &store.tensor("model.action_in_proj.weight")?,
        Some(&store.tensor("model.action_in_proj.bias")?),
        CHUNK,
        MAX_ACTION,
        EXPERT_DIM,
    );
    let time_row = sinusoidal(time, EXPERT_DIM);
    let mut fused = vec![0.0f32; CHUNK * EXPERT_DIM * 2];
    for t in 0..CHUNK {
        fused[t * EXPERT_DIM * 2..t * EXPERT_DIM * 2 + EXPERT_DIM]
            .copy_from_slice(&action[t * EXPERT_DIM..(t + 1) * EXPERT_DIM]);
        fused[t * EXPERT_DIM * 2 + EXPERT_DIM..(t + 1) * EXPERT_DIM * 2].copy_from_slice(&time_row);
    }
    let mid = linear(
        &fused,
        &store.tensor("model.action_time_mlp_in.weight")?,
        Some(&store.tensor("model.action_time_mlp_in.bias")?),
        CHUNK,
        EXPERT_DIM * 2,
        EXPERT_DIM,
    );
    let mid = silu(&mid);
    Ok(linear(
        &mid,
        &store.tensor("model.action_time_mlp_out.weight")?,
        Some(&store.tensor("model.action_time_mlp_out.bias")?),
        CHUNK,
        EXPERT_DIM,
        EXPERT_DIM,
    ))
}

pub fn sinusoidal(time: f32, dimension: usize) -> Vec<f32> {
    let half = dimension / 2;
    let mut emb = vec![0.0f32; dimension];
    for i in 0..half {
        let fraction = if half == 1 {
            0.0
        } else {
            i as f64 / (half - 1) as f64
        };
        let period = 0.004_f64 * (4.0_f64 / 0.004_f64).powf(fraction);
        let scaling = (1.0 / period) * 2.0 * std::f64::consts::PI;
        let angle = scaling * f64::from(time);
        emb[i] = angle.sin() as f32;
        emb[half + i] = angle.cos() as f32;
    }
    emb
}

fn mlp(
    store: &Store,
    prefix: &str,
    norm: &str,
    x: &[f32],
    seq: usize,
    dim: usize,
    inter: usize,
) -> Result<Vec<f32>, String> {
    let normed = rmsnorm(&store.tensor(norm)?, x, seq, dim);
    let gate = linear(
        &normed,
        &store.tensor(&format!("{prefix}.gate_proj.weight"))?,
        None,
        seq,
        dim,
        inter,
    );
    let up = linear(
        &normed,
        &store.tensor(&format!("{prefix}.up_proj.weight"))?,
        None,
        seq,
        dim,
        inter,
    );
    let gate = silu(&gate);
    let prod: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| a * b).collect();
    Ok(linear(
        &prod,
        &store.tensor(&format!("{prefix}.down_proj.weight"))?,
        None,
        seq,
        inter,
        dim,
    ))
}

fn rmsnorm(weight: &[f32], x: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    rmsnorm_cpu(x, weight, rows, dim, RMS_EPS)
}

fn rmsnorm_cpu(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    ferric_core::cpu::rmsnorm(x, w, rows, dim, eps)
}

pub(crate) fn rope(x: &[f32], pos: &[u32], heads: usize) -> Result<Vec<f32>, String> {
    let seq = pos.len();
    let mut o = x.to_vec();
    let half = HEAD_DIM / 2;
    for t in 0..seq {
        let p = pos[t] as f32;
        for h in 0..heads {
            let off = (t * heads + h) * HEAD_DIM;
            for c in 0..half {
                let timescale = ROPE_BASE.powf((2.0 * c as f32) / HEAD_DIM as f32);
                let ang = p / timescale;
                let (sin, cos) = ang.sin_cos();
                let x1 = x[off + c];
                let x2 = x[off + c + half];
                o[off + c] = x1 * cos - x2 * sin;
                o[off + c + half] = x2 * cos + x1 * sin;
            }
        }
    }
    crate::tolerance::rope_invariant_holds(x, &o, pos, heads, HEAD_DIM, ROPE_BASE)?;
    Ok(o)
}

pub(crate) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    hq: usize,
    hkv: usize,
    dh: usize,
    mask: Option<&[bool]>,
) -> Vec<f32> {
    let scale = (dh as f32).powf(-0.5);
    let groups = hq / hkv;
    let mut out = vec![0.0f32; seq_q * hq * dh];
    for head in 0..hq {
        let kv = head / groups;
        let mut qq = vec![0.0f32; seq_q * dh];
        let mut kk = vec![0.0f32; seq_k * dh];
        let mut vv = vec![0.0f32; seq_k * dh];
        for t in 0..seq_q {
            qq[t * dh..(t + 1) * dh]
                .copy_from_slice(&q[(t * hq + head) * dh..(t * hq + head + 1) * dh]);
        }
        for t in 0..seq_k {
            kk[t * dh..(t + 1) * dh]
                .copy_from_slice(&k[(t * hkv + kv) * dh..(t * hkv + kv + 1) * dh]);
            vv[t * dh..(t + 1) * dh]
                .copy_from_slice(&v[(t * hkv + kv) * dh..(t * hkv + kv + 1) * dh]);
        }
        let mut scores = gemm_nt(&qq, &kk, seq_q, dh, seq_k);
        for s in &mut scores {
            *s *= scale;
        }
        if let Some(mask) = mask {
            for i in 0..seq_q {
                for j in 0..seq_k {
                    if !mask[i * seq_k + j] {
                        scores[i * seq_k + j] = f32::MIN;
                    }
                }
            }
        }
        let probs = softmax(&scores, seq_q, seq_k);
        let ctx = gemm_nn(&probs, &vv, seq_q, seq_k, dh);
        for t in 0..seq_q {
            out[(t * hq + head) * dh..(t * hq + head + 1) * dh]
                .copy_from_slice(&ctx[t * dh..(t + 1) * dh]);
        }
    }
    out
}

pub(crate) fn att_2d(pad: &[bool], att: &[u8]) -> Vec<bool> {
    let n = pad.len();
    let mut cum = vec![0i32; n];
    let mut acc = 0i32;
    for i in 0..n {
        acc += i32::from(att[i]);
        cum[i] = acc;
    }
    let mut m = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = pad[i] && pad[j] && cum[j] <= cum[i];
        }
    }
    m
}

pub(crate) fn positions(pad: &[bool]) -> Vec<u32> {
    let mut c = 0u32;
    pad.iter()
        .map(|p| {
            if *p {
                c += 1;
            }
            c.saturating_sub(1)
        })
        .collect()
}

pub(crate) fn self_mask(prefix_pad: &[bool], suffix: usize) -> Vec<bool> {
    let prefix = prefix_pad.len();
    let cols = prefix + suffix;
    let mut m = vec![false; suffix * cols];
    for i in 0..suffix {
        for j in 0..prefix {
            m[i * cols + j] = prefix_pad[j];
        }
        for j in 0..=i {
            m[i * cols + prefix + j] = true;
        }
    }
    m
}

pub(crate) fn cross_mask(prefix_pad: &[bool]) -> Vec<bool> {
    let prefix = prefix_pad.len();
    let mut m = vec![false; CHUNK * prefix];
    for i in 0..CHUNK {
        for j in 0..prefix {
            m[i * prefix + j] = prefix_pad[j];
        }
    }
    m
}

fn concat_kv(
    pk: &[f32],
    sk: &[f32],
    pv: &[f32],
    sv: &[f32],
    prefix: usize,
    suffix: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dk = KV_HEADS * HEAD_DIM;
    let mut k = Vec::with_capacity((prefix + suffix) * dk);
    let mut v = Vec::with_capacity((prefix + suffix) * dk);
    k.extend_from_slice(&pk[..prefix * dk]);
    k.extend_from_slice(&sk[..suffix * dk]);
    v.extend_from_slice(&pv[..prefix * dk]);
    v.extend_from_slice(&sv[..suffix * dk]);
    (k, v)
}

fn flatten_heads(x: &[f32], seq: usize, heads: usize, dh: usize) -> Vec<f32> {
    x[..seq * heads * dh].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_tanh_matches_the_pytorch_formula() {
        let y = gelu_tanh(&[1.0, -0.5]);
        assert!((y[0] - 0.8411919906082768).abs() < 1e-6, "{}", y[0]);
        assert!((y[1] - -0.15428599017485606).abs() < 1e-6, "{}", y[1]);
    }

    #[test]
    fn time_embedding_matches_the_f64_outer_product() {
        let y = sinusoidal(1.0, 720);
        assert!((y[1] - 0.9958927469604786).abs() < 1e-6, "{}", y[1]);
        assert!((y[360] - 1.0).abs() < 1e-6, "{}", y[360]);
        assert!(y[0].abs() < 1e-5, "{}", y[0]);
    }

    #[test]
    fn pixel_shuffle_matches_the_connector_permutation() {
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let y = pixel_shuffle(&x, 16, 4, 2).unwrap();
        let expect = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 48.0, 49.0, 50.0, 51.0, 52.0,
            53.0, 54.0, 55.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 56.0, 57.0, 58.0,
            59.0, 60.0, 61.0, 62.0, 63.0,
        ];
        assert_eq!(y, expect);
    }

    #[test]
    fn resize_keeps_a_corner_when_the_source_index_is_negative() {
        let mut img = vec![0.0f32; 3 * 2 * 2];
        img[0] = 0.25;
        let out = resize_with_pad(&img, 2, 2, 4, 4).unwrap();
        assert!((out[0] - 0.25).abs() < 1e-6, "{}", out[0]);
    }

    #[test]
    fn state_token_is_hidden_from_the_prefix_block() {
        let pad = [true, true, true];
        let att = [0u8, 0, 1];
        let m = att_2d(&pad, &att);
        assert!(m[0]);
        assert!(m[1]);
        assert!(!m[2]);
        assert!(m[2 * 3 + 2]);
    }
}
