//! Device-resident SmolVLA forward. Every GEMM, norm, attention softmax, RoPE,
//! and tanh-GELU is a wgpu dispatch. `to_vec` is the completion readback.
//! Pixel shuffle and the patch unfold are index moves of a tensor that was
//! already produced on device. This module does not claim GPU energy.

use std::collections::HashMap;

use ferric_core::{Context, Tensor};

use crate::model::{
    att_2d, cross_mask, positions, prepare_image, self_mask, sinusoidal, ActivationTrace, Sample,
    VisionSnap,
};
use crate::weights::Store;

const VISION_LAYERS: usize = 12;
const TEXT_LAYERS: usize = 16;
const VISION_DIM: u32 = 768;
const TEXT_DIM: u32 = 960;
const EXPERT_DIM: u32 = 720;
const HEADS: u32 = 15;
const KV_HEADS: u32 = 5;
const HEAD_DIM: u32 = 64;
const VISION_HEADS: u32 = 12;
const VISION_INTER: u32 = 3072;
const TEXT_INTER: u32 = 2560;
const EXPERT_INTER: u32 = 2048;
const PATCH: usize = 16;
const IMAGE: usize = 512;
const GRID: usize = IMAGE / PATCH;
const PATCHES: u32 = (GRID * GRID) as u32;
const CONNECTOR_SCALE: usize = 4;
const IMG_TOKENS: usize = (GRID * GRID) / (CONNECTOR_SCALE * CONNECTOR_SCALE);
const CHUNK: u32 = 50;
const MAX_ACTION: usize = 32;
const MAX_STATE: u32 = 32;
const LANG: usize = 48;
const STEPS: usize = 10;

pub struct GpuForward {
    pub raw_actions: Vec<f32>,
    pub adapter: String,
    pub backend: String,
    pub software: bool,
    pub dispatches: u32,
    pub completions: u32,
    pub readbacks: u32,
    pub trace: ActivationTrace,
    pub host_index_moves: &'static [&'static str],
}

struct Dev<'a> {
    ctx: &'a Context,
    store: &'a Store,
    mats: HashMap<String, Tensor>,
    vecs: HashMap<String, Tensor>,
    embed0: Option<Tensor>,
    embed1: Option<Tensor>,
    embed_mid: u32,
    pub dispatches: u32,
    pub completions: u32,
    pub readbacks: u32,
}

impl<'a> Dev<'a> {
    fn flush(&mut self) {
        self.ctx.flush();
        self.completions += 1;
    }

    async fn read(&mut self, t: &Tensor) -> Result<Vec<f32>, String> {
        self.readbacks += 1;
        self.ctx.to_vec(t).await
    }

    fn mat(&mut self, name: &str, out: u32, inn: u32) -> Result<Tensor, String> {
        if let Some(t) = self.mats.get(name) {
            return Ok(Tensor { buf: t.buf.clone(), shape: t.shape.clone() });
        }
        let data = self.store.tensor(name)?;
        if data.len() != out as usize * inn as usize {
            return Err(format!("{name}: {} values, expected {}x{inn}", data.len(), out));
        }
        let mut bt = vec![0.0f32; data.len()];
        for o in 0..out as usize {
            for i in 0..inn as usize {
                bt[i * out as usize + o] = data[o * inn as usize + i];
            }
        }
        let t = self.ctx.tensor(&bt, &[inn as usize, out as usize]);
        self.mats.insert(name.to_string(), Tensor { buf: t.buf.clone(), shape: t.shape.clone() });
        Ok(t)
    }

    /// The token table is 189_235_200 bytes. Lavapipe's storage binding limit is 134_217_728.
    /// The lookup is two device gathers, each under that limit.
    async fn embed_rows(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if self.embed0.is_none() {
            let name = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight";
            let data = self.store.tensor(name)?;
            let d = TEXT_DIM as usize;
            let vocab = data.len() / d;
            let mid = vocab / 2;
            self.embed_mid = mid as u32;
            self.embed0 = Some(self.ctx.tensor(&data[..mid * d], &[mid, d]));
            self.embed1 = Some(self.ctx.tensor(&data[mid * d..], &[vocab - mid, d]));
        }
        let mid = self.embed_mid;
        let d = TEXT_DIM as usize;
        let mut out = vec![0.0f32; ids.len() * d];
        for (i, id) in ids.iter().enumerate() {
            let (local, shard) = if *id < mid {
                (*id, 0)
            } else {
                (*id - mid, 1)
            };
            let row = {
                let table = if shard == 0 { self.embed0.as_ref().unwrap() } else { self.embed1.as_ref().unwrap() };
                self.ctx.gather0(table, &[local], d)
            };
            self.dispatches += 1;
            let host = self.read(&row).await?;
            out[i * d..(i + 1) * d].copy_from_slice(&host);
        }
        Ok(out)
    }

    fn vec1(&mut self, name: &str) -> Result<Tensor, String> {
        if let Some(t) = self.vecs.get(name) {
            return Ok(Tensor { buf: t.buf.clone(), shape: t.shape.clone() });
        }
        let data = self.store.tensor(name)?;
        let t = self.ctx.tensor(&data, &[data.len()]);
        self.vecs.insert(name.to_string(), Tensor { buf: t.buf.clone(), shape: t.shape.clone() });
        Ok(t)
    }

    fn linear(&mut self, x: &Tensor, w: &str, b: Option<&str>, seq: u32, inn: u32, out: u32) -> Result<Tensor, String> {
        let weight = self.mat(w, out, inn)?;
        let mut y = self.ctx.mm(x, &weight, seq, inn, out);
        self.dispatches += 1;
        if let Some(b) = b {
            let bias = self.vec1(b)?;
            y = self.ctx.add_bias_t(&y, &bias);
            self.dispatches += 1;
        }
        Ok(y)
    }

    fn rms(&mut self, x: &Tensor, w: &str, rows: u32, dim: u32) -> Result<Tensor, String> {
        let weight = self.vec1(w)?;
        self.dispatches += 1;
        Ok(self.ctx.rmsnorm_t(x, &weight, rows, dim, 1e-5))
    }

    fn ln(&mut self, x: &Tensor, w: &str, b: &str, rows: u32, dim: u32) -> Result<Tensor, String> {
        let weight = self.vec1(w)?;
        let bias = self.vec1(b)?;
        self.dispatches += 1;
        Ok(self.ctx.layernorm_t(x, &weight, &bias, rows, dim, 1e-6))
    }

    fn attend(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_q: u32,
        seq_k: u32,
        hq: u32,
        hkv: u32,
        dh: u32,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let mut acc = self.ctx.tensor(&vec![0.0f32; (seq_q * hq * dh) as usize], &[seq_q as usize, (hq * dh) as usize]);
        let groups = hq / hkv;
        let scale = (dh as f32).powf(-0.5);
        for head in 0..hq {
            let kv = head / groups;
            let qq = self.ctx.head_read_t(q, seq_q, hq, dh, head);
            let kk = self.ctx.head_read_t(k, seq_k, hkv, dh, kv);
            let vv = self.ctx.head_read_t(v, seq_k, hkv, dh, kv);
            self.dispatches += 3;
            let mut scores = self.ctx.mm_bt(&qq, &kk, seq_q, seq_k, dh, scale);
            self.dispatches += 1;
            if let Some(mask) = mask {
                scores = self.ctx.add_t(&scores, mask);
                self.dispatches += 1;
            }
            let probs = self.ctx.softmax_t(&scores, seq_q, seq_k);
            let ctxv = self.ctx.mm(&probs, &vv, seq_q, seq_k, dh);
            self.dispatches += 2;
            acc = self.ctx.head_write_t(&acc, &ctxv, seq_q, hq, dh, head);
            self.dispatches += 1;
        }
        acc
    }
}

fn mask_tensor(ctx: &Context, mask: &[bool]) -> Tensor {
    let data: Vec<f32> = mask.iter().map(|keep| if *keep { 0.0 } else { f32::MIN }).collect();
    let cols = if mask.is_empty() { 1 } else { 1 };
    let _ = cols;
    ctx.tensor(&data, &[data.len()])
}

fn unfold_patches(pixels: &[f32]) -> Vec<f32> {
    let inn = 3 * PATCH * PATCH;
    let mut cols = vec![0.0f32; PATCHES as usize * inn];
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
    cols
}

/// Device-resident weights and the context they live on. A second `forward`
/// reuses the uploaded weights, so a measured steady window does not include
/// weight upload. The first `forward` uploads and compiles.
pub struct GpuSession {
    pub ctx: Context,
    pub software: bool,
    pub listed: Vec<String>,
    mats: HashMap<String, Tensor>,
    vecs: HashMap<String, Tensor>,
    embed0: Option<Tensor>,
    embed1: Option<Tensor>,
    embed_mid: u32,
}

impl GpuSession {
    pub async fn open() -> Result<Self, String> {
        let listed = Context::enumerate().await;
        let ctx = Context::new().await?;
        let name_l = ctx.adapter_name.to_lowercase();
        let software = name_l.contains("llvmpipe")
            || name_l.contains("lavapipe")
            || name_l.contains("swiftshader")
            || listed.iter().any(|(n, _, t)| n == &ctx.adapter_name && format!("{t:?}").contains("Cpu"));
        Ok(Self {
            ctx,
            software,
            listed: listed.iter().map(|(n, b, t)| format!("{n} | {b:?} | {t:?}")).collect(),
            mats: HashMap::new(),
            vecs: HashMap::new(),
            embed0: None,
            embed1: None,
            embed_mid: 0,
        })
    }

    /// Submit an empty batch, then `device.poll(wait_indefinitely)`.
    pub fn drain(&self) {
        self.ctx.flush();
    }

    pub async fn forward(&mut self, store: &Store, sample: &Sample) -> Result<GpuForward, String> {
        if !sample.allow_unnormalized_network {
            return Err("the wgpu graph runs only the declared non-executable network forward".into());
        }
        let mut dev = Dev {
            ctx: &self.ctx,
            store,
            mats: std::mem::take(&mut self.mats),
            vecs: std::mem::take(&mut self.vecs),
            embed0: self.embed0.take(),
            embed1: self.embed1.take(),
            embed_mid: self.embed_mid,
            dispatches: 0,
            completions: 0,
            readbacks: 0,
        };
        let out = run_graph(&mut dev, sample).await;
        self.mats = std::mem::take(&mut dev.mats);
        self.vecs = std::mem::take(&mut dev.vecs);
        self.embed0 = dev.embed0.take();
        self.embed1 = dev.embed1.take();
        self.embed_mid = dev.embed_mid;
        let (raw, trace) = out?;
        Ok(GpuForward {
            raw_actions: raw,
            adapter: self.ctx.adapter_name.clone(),
            backend: format!("{:?}", self.ctx.backend),
            software: self.software,
            dispatches: dev.dispatches,
            completions: dev.completions,
            readbacks: dev.readbacks,
            trace,
            host_index_moves: &["patch_unfold", "pixel_shuffle"],
        })
    }
}

pub async fn forward_gpu(store: &Store, sample: &Sample) -> Result<GpuForward, String> {
    let mut session = GpuSession::open().await?;
    session.forward(store, sample).await
}

async fn run_graph(dev: &mut Dev<'_>, sample: &Sample) -> Result<(Vec<f32>, ActivationTrace), String> {
    let mut dev = dev;
    let mut prefix_host = Vec::new();
    let mut pad = Vec::new();
    let mut att = Vec::new();
    let mut prepared_image0 = Vec::new();
    let mut vision = None;
    for image in 0..sample.images.len() {
        eprintln!("wgpu vision camera {image}");
        let (h, w) = sample.image_hw[image];
        let pixels = prepare_image(&sample.images[image], h, w)?;
        if image == 0 {
            prepared_image0 = pixels.clone();
        }
        let (tokens, snap) = vision_image(&mut dev, &pixels, image == 0).await?;
        if image == 0 {
            vision = snap;
        }
        let scale = (TEXT_DIM as f32).sqrt();
        for v in tokens {
            prefix_host.push(v * scale);
        }
        pad.extend(std::iter::repeat_n(true, IMG_TOKENS));
        att.extend(std::iter::repeat_n(0u8, IMG_TOKENS));
        dev.flush();
    }
    let ids: Vec<u32> = sample.token_ids.clone();
    let mut lang_host = dev.embed_rows(&ids).await?;
    let scale = (TEXT_DIM as f32).sqrt();
    for v in &mut lang_host {
        *v *= scale;
    }
    prefix_host.extend_from_slice(&lang_host);
    pad.extend_from_slice(&sample.token_mask);
    att.extend(std::iter::repeat_n(0u8, LANG));
    let mut state32 = vec![0.0f32; MAX_STATE as usize];
    state32[..sample.state.len()].copy_from_slice(&sample.state);
    let state_x = dev.ctx.tensor(&state32, &[1, MAX_STATE as usize]);
    let state = dev.linear(
        &state_x,
        "model.state_proj.weight",
        Some("model.state_proj.bias"),
        1,
        MAX_STATE,
        TEXT_DIM,
    )?;
    let state_host = dev.read(&state).await?;
    prefix_host.extend_from_slice(&state_host);
    pad.push(true);
    att.push(1);
    let seq = pad.len() as u32;
    let mask = att_2d(&pad, &att);
    let pos = positions(&pad);
    let mask_t = mask_tensor(dev.ctx, &mask);
    eprintln!("wgpu text prefix");
    let prefix = dev.ctx.tensor(&prefix_host, &[seq as usize, TEXT_DIM as usize]);
    let (cache, k0) = text_prefix(&mut dev, &prefix, seq, &mask_t, &pos).await?;
    dev.flush();
    let mut x = sample.noise.clone();
    let dt = -1.0 / STEPS as f32;
    let mut velocity0 = Vec::new();
    for step in 0..STEPS {
        eprintln!("wgpu euler step {step}");
        let time = 1.0 + step as f32 * dt;
        let v = denoise(&mut dev, &cache, &pad, &x, time).await?;
        if step == 0 {
            velocity0 = v[..MAX_ACTION].to_vec();
        }
        for (dst, vel) in x.iter_mut().zip(&v) {
            *dst += dt * vel;
        }
        dev.flush();
    }
    let trace = ActivationTrace {
        prepared_image0,
        vision: vision.unwrap_or(VisionSnap {
            patch_plus_pos_token0: Vec::new(),
            vision_layer0_token0: Vec::new(),
            vision_post_token0: Vec::new(),
            connector_token0: Vec::new(),
        }),
        state_token: state_host,
        text_layer0_k_token0: k0,
        velocity0,
        raw_actions: x.clone(),
    };
    Ok((x, trace))
}

async fn vision_image(dev: &mut Dev<'_>, pixels: &[f32], capture: bool) -> Result<(Vec<f32>, Option<VisionSnap>), String> {
    let cols = unfold_patches(pixels);
    let inn = (3 * PATCH * PATCH) as u32;
    let x = dev.ctx.tensor(&cols, &[PATCHES as usize, inn as usize]);
    let mut hidden = dev.linear(
        &x,
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight",
        Some("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias"),
        PATCHES,
        inn,
        VISION_DIM,
    )?;
    let pos = dev.vec1("model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight")?;
    hidden = dev.ctx.add_t(&hidden, &pos);
    dev.dispatches += 1;
    let patch_plus_pos_token0 = if capture {
        dev.read(&hidden).await?.into_iter().take(VISION_DIM as usize).collect()
    } else {
        Vec::new()
    };
    let mut layer0 = Vec::new();
    for layer in 0..VISION_LAYERS {
        hidden = vision_layer(dev, layer, &hidden)?;
        dev.flush();
        if capture && layer == 0 {
            layer0 = dev.read(&hidden).await?.into_iter().take(VISION_DIM as usize).collect();
        }
    }
    hidden = dev.ln(
        &hidden,
        "model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight",
        "model.vlm_with_expert.vlm.model.vision_model.post_layernorm.bias",
        PATCHES,
        VISION_DIM,
    )?;
    let post_host = dev.read(&hidden).await?;
    let vision_post_token0 = if capture { post_host[..VISION_DIM as usize].to_vec() } else { Vec::new() };
    let shuffled = crate::model::pixel_shuffle(&post_host, PATCHES as usize, VISION_DIM as usize, CONNECTOR_SCALE)?;
    let inn_c = VISION_DIM * (CONNECTOR_SCALE * CONNECTOR_SCALE) as u32;
    let shuffled_t = dev.ctx.tensor(&shuffled, &[IMG_TOKENS, inn_c as usize]);
    let tokens = dev.linear(
        &shuffled_t,
        "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
        None,
        IMG_TOKENS as u32,
        inn_c,
        TEXT_DIM,
    )?;
    let host = dev.read(&tokens).await?;
    let snap = if capture {
        Some(VisionSnap {
            patch_plus_pos_token0,
            vision_layer0_token0: layer0,
            vision_post_token0,
            connector_token0: host[..TEXT_DIM as usize].to_vec(),
        })
    } else {
        None
    };
    Ok((host, snap))
}

fn vision_layer(dev: &mut Dev<'_>, layer: usize, hidden: &Tensor) -> Result<Tensor, String> {
    let p = format!("model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{layer}");
    let normed = dev.ln(hidden, &format!("{p}.layer_norm1.weight"), &format!("{p}.layer_norm1.bias"), PATCHES, VISION_DIM)?;
    let q = dev.linear(&normed, &format!("{p}.self_attn.q_proj.weight"), Some(&format!("{p}.self_attn.q_proj.bias")), PATCHES, VISION_DIM, VISION_DIM)?;
    let k = dev.linear(&normed, &format!("{p}.self_attn.k_proj.weight"), Some(&format!("{p}.self_attn.k_proj.bias")), PATCHES, VISION_DIM, VISION_DIM)?;
    let v = dev.linear(&normed, &format!("{p}.self_attn.v_proj.weight"), Some(&format!("{p}.self_attn.v_proj.bias")), PATCHES, VISION_DIM, VISION_DIM)?;
    let attn = dev.attend(&q, &k, &v, PATCHES, PATCHES, VISION_HEADS, VISION_HEADS, HEAD_DIM, None);
    let projected = dev.linear(&attn, &format!("{p}.self_attn.out_proj.weight"), Some(&format!("{p}.self_attn.out_proj.bias")), PATCHES, VISION_DIM, VISION_DIM)?;
    let residual = dev.ctx.add_t(hidden, &projected);
    dev.dispatches += 1;
    let normed = dev.ln(&residual, &format!("{p}.layer_norm2.weight"), &format!("{p}.layer_norm2.bias"), PATCHES, VISION_DIM)?;
    let mid = dev.linear(&normed, &format!("{p}.mlp.fc1.weight"), Some(&format!("{p}.mlp.fc1.bias")), PATCHES, VISION_DIM, VISION_INTER)?;
    let mid = dev.ctx.gelu_tanh_t(&mid);
    dev.dispatches += 1;
    let mid = dev.linear(&mid, &format!("{p}.mlp.fc2.weight"), Some(&format!("{p}.mlp.fc2.bias")), PATCHES, VISION_INTER, VISION_DIM)?;
    let y = dev.ctx.add_t(&residual, &mid);
    dev.dispatches += 1;
    Ok(y)
}

async fn text_prefix(
    dev: &mut Dev<'_>,
    hidden0: &Tensor,
    seq: u32,
    mask: &Tensor,
    pos: &[u32],
) -> Result<(Vec<(Tensor, Tensor)>, Vec<f32>), String> {
    let mut hidden = Tensor { buf: hidden0.buf.clone(), shape: hidden0.shape.clone() };
    let mut cache = Vec::with_capacity(TEXT_LAYERS);
    let mut k0 = Vec::new();
    for layer in 0..TEXT_LAYERS {
        let p = format!("model.vlm_with_expert.vlm.model.text_model.layers.{layer}");
        let normed = dev.rms(&hidden, &format!("{p}.input_layernorm.weight"), seq, TEXT_DIM)?;
        let q = dev.linear(&normed, &format!("{p}.self_attn.q_proj.weight"), None, seq, TEXT_DIM, HEADS * HEAD_DIM)?;
        let k = dev.linear(&normed, &format!("{p}.self_attn.k_proj.weight"), None, seq, TEXT_DIM, KV_HEADS * HEAD_DIM)?;
        let v = dev.linear(&normed, &format!("{p}.self_attn.v_proj.weight"), None, seq, TEXT_DIM, KV_HEADS * HEAD_DIM)?;
        let q = dev.ctx.rope_positions_t(&q, pos, HEADS, HEAD_DIM, 10_000.0);
        let k = dev.ctx.rope_positions_t(&k, pos, KV_HEADS, HEAD_DIM, 10_000.0);
        dev.dispatches += 2;
        if layer == 0 {
            k0 = dev.read(&k).await?.into_iter().take((KV_HEADS * HEAD_DIM) as usize).collect();
        }
        let attn = dev.attend(&q, &k, &v, seq, seq, HEADS, KV_HEADS, HEAD_DIM, Some(mask));
        cache.push((k, v));
        let projected = dev.linear(&attn, &format!("{p}.self_attn.o_proj.weight"), None, seq, TEXT_DIM, TEXT_DIM)?;
        let residual = dev.ctx.add_t(&hidden, &projected);
        dev.dispatches += 1;
        hidden = text_mlp(dev, &p, &residual, seq, TEXT_DIM, TEXT_INTER)?;
        dev.flush();
    }
    let _ = dev.rms(&hidden, "model.vlm_with_expert.vlm.model.text_model.norm.weight", seq, TEXT_DIM)?;
    Ok((cache, k0))
}

fn text_mlp(dev: &mut Dev<'_>, prefix: &str, x: &Tensor, seq: u32, dim: u32, inter: u32) -> Result<Tensor, String> {
    let normed = dev.rms(x, &format!("{prefix}.post_attention_layernorm.weight"), seq, dim)?;
    let gate = dev.linear(&normed, &format!("{prefix}.mlp.gate_proj.weight"), None, seq, dim, inter)?;
    let up = dev.linear(&normed, &format!("{prefix}.mlp.up_proj.weight"), None, seq, dim, inter)?;
    let gate = dev.ctx.silu_t(&gate);
    dev.dispatches += 1;
    let prod = dev.ctx.mul_t(&gate, &up);
    dev.dispatches += 1;
    let mid = dev.linear(&prod, &format!("{prefix}.mlp.down_proj.weight"), None, seq, inter, dim)?;
    let y = dev.ctx.add_t(x, &mid);
    dev.dispatches += 1;
    Ok(y)
}

async fn denoise(dev: &mut Dev<'_>, cache: &[(Tensor, Tensor)], prefix_pad: &[bool], x_t: &[f32], time: f32) -> Result<Vec<f32>, String> {
    let prefix = prefix_pad.len() as u32;
    let suffix = embed_suffix(dev, x_t, time).await?;
    let mut hidden = suffix;
    let offset = prefix_pad.iter().filter(|p| **p).count() as u32;
    let pos: Vec<u32> = (0..CHUNK).map(|i| offset + i).collect();
    let pos0: Vec<u32> = (0..CHUNK).collect();
    let self_m = mask_tensor(dev.ctx, &self_mask(prefix_pad, CHUNK as usize));
    let cross_m = mask_tensor(dev.ctx, &cross_mask(prefix_pad));
    for layer in 0..TEXT_LAYERS {
        let p = format!("model.vlm_with_expert.lm_expert.layers.{layer}");
        let normed = dev.rms(&hidden, &format!("{p}.input_layernorm.weight"), CHUNK, EXPERT_DIM)?;
        let q = dev.linear(&normed, &format!("{p}.self_attn.q_proj.weight"), None, CHUNK, EXPERT_DIM, HEADS * HEAD_DIM)?;
        let attn = if layer % 2 == 0 {
            let k = dev.linear(&normed, &format!("{p}.self_attn.k_proj.weight"), None, CHUNK, EXPERT_DIM, KV_HEADS * HEAD_DIM)?;
            let v = dev.linear(&normed, &format!("{p}.self_attn.v_proj.weight"), None, CHUNK, EXPERT_DIM, KV_HEADS * HEAD_DIM)?;
            let q = dev.ctx.rope_positions_t(&q, &pos, HEADS, HEAD_DIM, 10_000.0);
            let k = dev.ctx.rope_positions_t(&k, &pos, KV_HEADS, HEAD_DIM, 10_000.0);
            dev.dispatches += 2;
            let (pk, pv) = &cache[layer];
            let k = concat_seq(dev, pk, &k, prefix, CHUNK, KV_HEADS * HEAD_DIM).await?;
            let v = concat_seq(dev, pv, &v, prefix, CHUNK, KV_HEADS * HEAD_DIM).await?;
            dev.attend(&q, &k, &v, CHUNK, prefix + CHUNK, HEADS, KV_HEADS, HEAD_DIM, Some(&self_m))
        } else {
            let (pk, pv) = &cache[layer];
            let flat_k = Tensor { buf: pk.buf.clone(), shape: vec![prefix as usize, (KV_HEADS * HEAD_DIM) as usize] };
            let flat_v = Tensor { buf: pv.buf.clone(), shape: vec![prefix as usize, (KV_HEADS * HEAD_DIM) as usize] };
            let k = dev.linear(&flat_k, &format!("{p}.self_attn.k_proj.weight"), None, prefix, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM)?;
            let v = dev.linear(&flat_v, &format!("{p}.self_attn.v_proj.weight"), None, prefix, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM)?;
            let q = dev.ctx.rope_positions_t(&q, &pos0, HEADS, HEAD_DIM, 10_000.0);
            dev.dispatches += 1;
            dev.attend(&q, &k, &v, CHUNK, prefix, HEADS, KV_HEADS, HEAD_DIM, Some(&cross_m))
        };
        let projected = dev.linear(&attn, &format!("{p}.self_attn.o_proj.weight"), None, CHUNK, HEADS * HEAD_DIM, EXPERT_DIM)?;
        let residual = dev.ctx.add_t(&hidden, &projected);
        dev.dispatches += 1;
        hidden = text_mlp(dev, &p, &residual, CHUNK, EXPERT_DIM, EXPERT_INTER)?;
    }
    let normed = dev.rms(&hidden, "model.vlm_with_expert.lm_expert.norm.weight", CHUNK, EXPERT_DIM)?;
    let v = dev.linear(&normed, "model.action_out_proj.weight", Some("model.action_out_proj.bias"), CHUNK, EXPERT_DIM, MAX_ACTION as u32)?;
    dev.read(&v).await
}

async fn embed_suffix(dev: &mut Dev<'_>, noisy: &[f32], time: f32) -> Result<Tensor, String> {
    let x = dev.ctx.tensor(noisy, &[CHUNK as usize, MAX_ACTION]);
    let action = dev.linear(&x, "model.action_in_proj.weight", Some("model.action_in_proj.bias"), CHUNK, MAX_ACTION as u32, EXPERT_DIM)?;
    let time_row = sinusoidal(time, EXPERT_DIM as usize);
    let action_host = dev.read(&action).await?;
    let mut fused = vec![0.0f32; CHUNK as usize * EXPERT_DIM as usize * 2];
    for t in 0..CHUNK as usize {
        let d = EXPERT_DIM as usize;
        fused[t * d * 2..t * d * 2 + d].copy_from_slice(&action_host[t * d..(t + 1) * d]);
        fused[t * d * 2 + d..(t + 1) * d * 2].copy_from_slice(&time_row);
    }
    let fused_t = dev.ctx.tensor(&fused, &[CHUNK as usize, EXPERT_DIM as usize * 2]);
    let mid = dev.linear(&fused_t, "model.action_time_mlp_in.weight", Some("model.action_time_mlp_in.bias"), CHUNK, EXPERT_DIM * 2, EXPERT_DIM)?;
    let mid = dev.ctx.silu_t(&mid);
    dev.dispatches += 1;
    dev.linear(&mid, "model.action_time_mlp_out.weight", Some("model.action_time_mlp_out.bias"), CHUNK, EXPERT_DIM, EXPERT_DIM)
}

async fn concat_seq(dev: &mut Dev<'_>, a: &Tensor, b: &Tensor, a_rows: u32, b_rows: u32, dim: u32) -> Result<Tensor, String> {
    let ah = dev.read(a).await?;
    let bh = dev.read(b).await?;
    let mut cat = Vec::with_capacity(((a_rows + b_rows) * dim) as usize);
    cat.extend_from_slice(&ah[..(a_rows * dim) as usize]);
    cat.extend_from_slice(&bh[..(b_rows * dim) as usize]);
    Ok(dev.ctx.tensor(&cat, &[(a_rows + b_rows) as usize, dim as usize]))
}
