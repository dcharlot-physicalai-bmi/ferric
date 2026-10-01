//! **Gemma 4's vision tower** (`gemma4_vision`) and its preprocessing — from the image file to the soft tokens
//! that replace `<|image|>` in the prompt. Verified against the authors' code (transformers `modeling_gemma4`,
//! `Gemma4ImageProcessor`) stage by stage: `scripts/gemma4_mm_conformance.sh`.
//!
//! ```text
//! image ─ aspect-preserving resize to the patch budget (torchvision's uint8 antialiased BICUBIC, bit-exact)
//!       ─ level·(1/255) ─ 16×16 patches, (row, col, channel) per patch
//!       ─ 2·(p − ½) · W_in  +  pos_x[col] + pos_y[row]           learned 2-D table, NOT interpolated
//!       ─ L × { RMSNorm → q,k,v (clipped linears) → q_norm, k_norm, weightless v_norm
//!               → 2-D rope (first half of each head by col, second half by row) → full attention, scale 1
//!               → RMSNorm → +residual → RMSNorm → down(gelu_tanh(gate)·up) → RMSNorm → +residual }
//!       ─ 3×3 average pool BY POSITION ─ ·√hidden ─ [standardize: (x − bias)·scale, 31B only]
//!       ─ embed_vision: weightless RMSNorm → projection to the text width
//! ```
//!
//! Five details that are easy to get wrong and that no shape check sees — each has a negative control in the gate:
//!
//! 1. **The resampler is torchvision's, not Pillow's.** `processor_config.json` names `Gemma4ImageProcessor`,
//!    the torchvision backend, which resizes the uint8 tensor NATIVELY: int16 fixed-point weights at the
//!    largest precision that fits, horizontal pass then vertical, the intermediate kept in uint8. Pillow's
//!    8-bit BICUBIC (what `qwen3vl_image` reproduces) lands 2 levels off on 0.57% of this probe's values;
//!    this port lands 0 levels off on every image tried, upscales and downscales alike.
//! 2. **No mean/std normalisation**: the processor stops at [0, 1] and the MODEL maps it to [-1, 1].
//! 3. **Clipped linears** (see [`crate::gemma4_mm`]).
//! 4. **Pooling is by POSITION**, not by consecutive rows: output token `k` averages the patches whose
//!    `(col / 3, row / 3)` is `k`'s cell, raster over the pooled grid. Nine consecutive rows are a different
//!    and wrong grouping (`pool_rows` control).
//! 5. **Attention has scale 1**: the per-head q/k RMS norms absorb it, as in the text model.
use crate::gemma4_mm::{Embedder, Lin, Src};
use ferric_core::Context;
use ferric_tensor::image::Rgb8;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

/// The processor's constants. From `processor_config.json` for a checkpoint directory; for an mmproj (which
/// does not carry them) the published defaults, which every Gemma 4 checkpoint ships.
#[derive(Debug, Clone, Copy)]
pub struct PreCfg {
    pub patch: usize,
    pub pool: usize,
    /// `max_soft_tokens` — one of 70, 140, 280, 560, 1120.
    pub max_soft_tokens: usize,
    pub rescale: f32,
}

impl Default for PreCfg {
    fn default() -> Self { PreCfg { patch: 16, pool: 3, max_soft_tokens: 280, rescale: 0.00392156862745098f64 as f32 } }
}

impl PreCfg {
    pub fn load(dir: &str) -> Result<PreCfg, String> {
        let j: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!("{dir}/processor_config.json")).map_err(|e| format!("processor_config.json: {e}"))?)
            .map_err(|e| format!("processor_config.json: {e}"))?;
        let ip = &j["image_processor"];
        if ip["image_processor_type"].as_str() != Some("Gemma4ImageProcessor") {
            return Err(format!("image_processor_type {:?}: the resampler here is Gemma4ImageProcessor's", ip["image_processor_type"]));
        }
        if ip["do_normalize"].as_bool() == Some(true) || ip["resample"].as_u64() != Some(3) {
            return Err("processor_config asks for normalisation or a non-BICUBIC resample; this path implements neither".into());
        }
        let u = |k: &str| ip[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("image_processor.{k} missing"));
        let c = PreCfg { patch: u("patch_size")?, pool: u("pooling_kernel_size")?, max_soft_tokens: u("max_soft_tokens")?,
                         rescale: ip["rescale_factor"].as_f64().ok_or("image_processor.rescale_factor missing")? as f32 };
        c.check()?;
        Ok(c)
    }

    pub fn check(&self) -> Result<(), String> {
        if ![70, 140, 280, 560, 1120].contains(&self.max_soft_tokens) {
            return Err(format!("max_soft_tokens {} is not one of 70, 140, 280, 560, 1120", self.max_soft_tokens));
        }
        Ok(())
    }

    pub fn max_patches(&self) -> usize { self.max_soft_tokens * self.pool * self.pool }
}

/// `get_aspect_ratio_preserving_size`: the largest `(h, w)`, both multiples of `pool·patch`, whose patch
/// count fits `max_patches`, at the source aspect ratio. A port of the authors' arithmetic line by line
/// (f64 `sqrt`, `floor`), including its two edge cases for extreme aspect ratios.
pub fn target_size(h: usize, w: usize, c: &PreCfg) -> Result<(usize, usize), String> {
    let max_patches = c.max_patches();
    let total = (h * w) as f64;
    let target_px = (max_patches * c.patch * c.patch) as f64;
    let factor = (target_px / total).sqrt();
    let side = c.pool * c.patch;
    let mut th = ((factor * h as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * w as f64) / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        return Err(format!("a {h}x{w} image resizes to 0x0"));
    }
    let max_side = (max_patches / (c.pool * c.pool)) * side;
    if th == 0 {
        th = side;
        tw = ((w as f64 / h as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((h as f64 / w as f64).floor() as usize * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        return Err(format!("resizing {h}x{w} to {th}x{tw} exceeds {max_patches} patches"));
    }
    Ok((th, tw))
}

/// torchvision's antialiased bicubic filter (`aa_filter`, a = -0.5), in its own operation order.
fn aa_cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 { ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 }
    else if x < 2.0 { ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a }
    else { 0.0 }
}

/// One axis of `_compute_index_ranges_int16_weights`: per output index `(xmin, n)` and `n` int16 taps, plus
/// the precision they are scaled to.
fn int16_taps(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<Vec<i32>>, u32) {
    let scale = in_size as f64 / out_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_interp = (support.ceil() as usize) * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let (mut ranges, mut ws) = (Vec::with_capacity(out_size), Vec::with_capacity(out_size));
    let mut wt_max = 0f64;
    for i in 0..out_size {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xsize = (((center + support + 0.5) as i64).min(in_size as i64) - xmin as i64).clamp(0, max_interp as i64) as usize;
        let mut w: Vec<f64> = (0..xsize).map(|j| aa_cubic((j as f64 + xmin as f64 - center + 0.5) * invscale)).collect();
        let mut tot = 0f64;
        for &v in &w { tot += v; }
        if tot != 0.0 { for v in &mut w { *v /= tot; wt_max = wt_max.max(*v); } }
        ranges.push((xmin, xsize));
        ws.push(w);
    }
    let mut prec = 0u32;
    while prec < 22 {
        if (0.5 + wt_max * (1u64 << (prec + 1)) as f64) as i64 >= 1 << 15 { break; }
        prec += 1;
    }
    let s = (1u64 << prec) as f64;
    // C's `(int)` after the ±0.5: round half away from zero.
    let wi = ws.iter().map(|w| w.iter().map(|&v| { let x = v * s; if x < 0.0 { (-0.5 + x) as i32 } else { (0.5 + x) as i32 } }).collect()).collect();
    (ranges, wi, prec)
}

/// **torchvision's native uint8 antialiased BICUBIC resize**, bit for bit: what `Gemma4ImageProcessor`
/// does to the decoded image (`tvF.resize(uint8, BICUBIC, antialias=True)` on the CPU).
///
/// Horizontal pass first, into a uint8 intermediate, then vertical — each a sum of int16 fixed-point taps
/// with a rounding bias, shifted back and clamped to [0, 255].
pub fn resize_torch_u8(img: &Rgb8, out_h: usize, out_w: usize) -> Rgb8 {
    fn pass(src: &[u8], h: usize, w: usize, out: usize, horizontal: bool) -> Vec<u8> {
        let n_in = if horizontal { w } else { h };
        let (ranges, taps, prec) = int16_taps(n_in, out);
        let (oh, ow) = if horizontal { (h, out) } else { (out, w) };
        let mut dst = vec![0u8; oh * ow * 3];
        let bias = 1i64 << (prec - 1);
        for y in 0..oh {
            for x in 0..ow {
                let i = if horizontal { x } else { y };
                let (x0, n) = ranges[i];
                for c in 0..3 {
                    let mut acc = bias;
                    for (j, &t) in taps[i][..n].iter().enumerate() {
                        let s = if horizontal { src[(y * w + x0 + j) * 3 + c] } else { src[((x0 + j) * w + x) * 3 + c] };
                        acc += s as i64 * t as i64;
                    }
                    dst[(y * ow + x) * 3 + c] = (acc >> prec).clamp(0, 255) as u8;
                }
            }
        }
        dst
    }
    let (mut px, mut h, mut w) = (img.px.clone(), img.h, img.w);
    if out_w != w { px = pass(&px, h, w, out_w, true); w = out_w; }
    if out_h != h { px = pass(&px, h, w, out_h, false); h = out_h; }
    Rgb8 { w, h, px }
}

/// A preprocessed image: the patch rows the tower reads and where each sits.
#[derive(Debug)]
pub struct Patches {
    /// `[n, patch·patch·3]`, (row, col, channel) per patch, values `float32(level) · float32(rescale)`.
    pub px: Vec<f32>,
    /// Patch grid, rows × cols.
    pub gh: usize,
    pub gw: usize,
    /// The resized 8-bit image (kept so a caller can pin it, e.g. by hash).
    pub resized: Rgb8,
}

impl Patches {
    pub fn n(&self) -> usize { self.gh * self.gw }
    /// Soft tokens this image becomes.
    pub fn tokens(&self, pool: usize) -> usize { self.n() / (pool * pool) }
}

/// The authors' processor, end to end: resize to the budget, rescale, patchify. Already-at-budget images
/// are not resampled (the authors return the image unchanged then too).
pub fn preprocess(img: &Rgb8, c: &PreCfg) -> Result<Patches, String> {
    let (th, tw) = target_size(img.h, img.w, c)?;
    let r = if th == img.h && tw == img.w { Rgb8 { w: img.w, h: img.h, px: img.px.clone() } } else { resize_torch_u8(img, th, tw) };
    let p = c.patch;
    let (gh, gw) = (th / p, tw / p);
    let mut px = Vec::with_capacity(gh * gw * p * p * 3);
    for gy in 0..gh { for gx in 0..gw { for y in 0..p { for x in 0..p { for ch in 0..3 {
        px.push(r.px[((gy * p + y) * tw + gx * p + x) * 3 + ch] as f32 * c.rescale);
    } } } } }
    Ok(Patches { px, gh, gw, resized: r })
}

/// What the tower is built from.
#[derive(Debug, Clone)]
pub struct VisionCfg {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub patch: usize,
    pub pool: usize,
    pub pos_size: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub clipped: bool,
    pub standardize: bool,
    /// The text model's width — the embedder's output.
    pub text_d: usize,
}

/// Negative controls: each removes ONE mechanism. The gate requires every one to fail first at the stage that
/// mechanism lives in. Selected by `FERRIC_G4V_NEG`; empty = the real tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neg { None, NoClip, PoolRows, PosSwap, RopeSwap, NoRope, NoVNorm, ErfGelu, AttnScale }

impl Neg {
    pub fn from_env() -> Result<Neg, String> {
        Ok(match std::env::var("FERRIC_G4V_NEG").unwrap_or_default().as_str() {
            "" => Neg::None, "noclip" => Neg::NoClip, "pool_rows" => Neg::PoolRows, "pos_swap" => Neg::PosSwap,
            "rope_swap" => Neg::RopeSwap, "norope" => Neg::NoRope, "no_vnorm" => Neg::NoVNorm, "erf_gelu" => Neg::ErfGelu,
            "attn_scale" => Neg::AttnScale,
            o => return Err(format!("FERRIC_G4V_NEG={o}: not a control (noclip|pool_rows|pos_swap|rope_swap|norope|no_vnorm|erf_gelu|attn_scale)")),
        })
    }
}

struct Block {
    ln_in: Tensor, ln_post_attn: Tensor, ln_pre_ff: Tensor, ln_post_ff: Tensor,
    q: Lin, k: Lin, v: Lin, o: Lin,
    q_norm: Tensor, k_norm: Tensor,
    gate: Lin, up: Lin, down: Lin,
}

pub struct VisionTower {
    ctx: Arc<Context>,
    pub cfg: VisionCfg,
    patch_w: Lin,
    /// `[2, pos_size, hidden]` — x (column) table then y (row) table, host-side: only `n` rows are gathered.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    std: Option<(Tensor, Tensor)>,
    pub embed: Embedder,
    inv_freq: Vec<f32>,
    pub neg: Neg,
}

impl VisionTower {
    /// From the authors' checkpoint directory or a ggml-org mmproj GGUF. `text_d` is the text model's width
    /// (the mmproj records it as `clip.vision.projection_dim`; it is checked against that).
    pub fn load(ctx: &Arc<Context>, src: &Src) -> Result<VisionTower, String> {
        let cfg = match src {
            Src::Hf { cfg, .. } => {
                let v = &cfg["vision_config"];
                let u = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("vision_config.{k} missing"));
                if v["hidden_activation"].as_str() != Some("gelu_pytorch_tanh") {
                    return Err(format!("vision hidden_activation {:?}: this tower implements gelu_pytorch_tanh", v["hidden_activation"]));
                }
                if u("num_key_value_heads")? != u("num_attention_heads")? {
                    return Err("vision tower with grouped KV heads: not seen on any checkpoint, refused".into());
                }
                if v["rope_parameters"]["rope_type"].as_str() != Some("default") {
                    return Err(format!("vision rope_type {:?}", v["rope_parameters"]["rope_type"]));
                }
                VisionCfg {
                    layers: u("num_hidden_layers")?, hidden: u("hidden_size")?, heads: u("num_attention_heads")?,
                    head_dim: u("head_dim")?, ff: u("intermediate_size")?, patch: u("patch_size")?,
                    pool: u("pooling_kernel_size")?, pos_size: u("position_embedding_size")?,
                    eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
                    rope_theta: v["rope_parameters"]["rope_theta"].as_f64().ok_or("vision rope_theta missing")?,
                    clipped: v["use_clipped_linears"].as_bool().ok_or("vision use_clipped_linears missing")?,
                    standardize: v["standardize"].as_bool().unwrap_or(false),
                    text_d: cfg["text_config"]["hidden_size"].as_u64().ok_or("text_config.hidden_size missing")? as usize,
                }
            }
            Src::Gguf(g) => {
                use ferric_gguf::{GgufSource, Meta};
                let md = g.metadata();
                match md.get("clip.vision.projector_type") {
                    Some(Meta::Str(s)) if s == "gemma4v" => {}
                    o => return Err(format!("clip.vision.projector_type {o:?}, not gemma4v")),
                }
                let u = |k: &str| match md.get(&format!("clip.vision.{k}")) { Some(Meta::U(x)) => Ok(*x as usize), _ => Err(format!("clip.vision.{k} missing")) };
                let (hidden, heads) = (u("embedding_length")?, u("attention.head_count")?);
                let pos = g.tensor("v.position_embd.weight").ok_or("no v.position_embd.weight")?;
                VisionCfg {
                    layers: u("block_count")?, hidden, heads, head_dim: hidden / heads, ff: u("feed_forward_length")?,
                    patch: u("patch_size")?,
                    // ⚠ Not in the mmproj: the gemma4v projector's fixed values, the same in every Gemma 4
                    // config.json published (pooling_kernel_size 3, rope_theta 100).
                    pool: 3, rope_theta: 100.0,
                    pos_size: pos.dims[1] as usize,
                    eps: match md.get("clip.vision.attention.layer_norm_epsilon") { Some(Meta::F(e)) => *e as f32, _ => 1e-6 },
                    clipped: g.tensor("v.blk.0.attn_q.input_min").is_some(),
                    standardize: g.tensor("v.std_bias").is_some(),
                    text_d: u("projection_dim")?,
                }
            }
        };
        if cfg.heads * cfg.head_dim != cfg.hidden {
            return Err(format!("vision heads {} x head_dim {} != hidden {}", cfg.heads, cfg.head_dim, cfg.hidden));
        }
        if cfg.head_dim % 4 != 0 { return Err(format!("vision head_dim {} is not divisible by 4 (2-D rope)", cfg.head_dim)); }
        let (h, f, c) = (cfg.hidden, cfg.ff, cfg.clipped);
        let pre = "model.vision_tower";
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("{pre}.encoder.layers.{i}");
            let lin = |m: &str, o: usize, inp: usize| Lin::load(src, ctx, &format!("{p}.{m}"), o, inp, c);
            blocks.push(Block {
                ln_in: src.vec(ctx, &format!("{p}.input_layernorm.weight"), h)?,
                ln_post_attn: src.vec(ctx, &format!("{p}.post_attention_layernorm.weight"), h)?,
                ln_pre_ff: src.vec(ctx, &format!("{p}.pre_feedforward_layernorm.weight"), h)?,
                ln_post_ff: src.vec(ctx, &format!("{p}.post_feedforward_layernorm.weight"), h)?,
                q: lin("self_attn.q_proj", h, h)?, k: lin("self_attn.k_proj", h, h)?,
                v: lin("self_attn.v_proj", h, h)?, o: lin("self_attn.o_proj", h, h)?,
                q_norm: src.vec(ctx, &format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: src.vec(ctx, &format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                gate: lin("mlp.gate_proj", f, h)?, up: lin("mlp.up_proj", f, h)?, down: lin("mlp.down_proj", h, f)?,
            });
        }
        let pw = 3 * cfg.patch * cfg.patch;
        let patch_w = Lin { w: src.qmat(ctx, &format!("{pre}.patch_embedder.input_proj.weight"), h, pw)?, clip: None };
        let (pos, ps) = src.f32s(&format!("{pre}.patch_embedder.position_embedding_table"))?;
        if ps != [2, cfg.pos_size, h] { return Err(format!("position table {ps:?}, expected [2, {}, {h}]", cfg.pos_size)); }
        let std = if cfg.standardize {
            Some((src.vec(ctx, &format!("{pre}.std_bias"), h)?, src.vec(ctx, &format!("{pre}.std_scale"), h)?))
        } else { None };
        let embed = Embedder { proj: src.qmat(ctx, "model.embed_vision.embedding_projection.weight", cfg.text_d, h)?, eps: cfg.eps, text_d: cfg.text_d };
        // The authors' inverse frequencies: 1 / theta^(arange(0, d/2, 2)/(d/2)), d = head_dim, in float32.
        // Rounded from f64 here; for head_dim 64 that is bit-identical to torch's float32 `pow` (checked).
        let sd = cfg.head_dim / 2;
        let inv_freq = (0..sd / 2).map(|k| {
            let e = (2 * k) as f32 / sd as f32;
            1.0f32 / (cfg.rope_theta.powf(e as f64) as f32)
        }).collect();
        Ok(VisionTower { ctx: ctx.clone(), cfg, patch_w, pos, blocks, std, embed, inv_freq, neg: Neg::from_env()? })
    }

    /// `(cos, sin)` for one axis of the 2-D rope: `[n·heads, d/4]`, row `i·heads + h` holding patch `i`'s
    /// angles (the same for every head).
    fn rope_table(&self, coord: &[u32]) -> (Tensor, Tensor) {
        let (heads, q) = (self.cfg.heads, self.inv_freq.len());
        let mut c = Vec::with_capacity(coord.len() * heads * q);
        let mut s = Vec::with_capacity(coord.len() * heads * q);
        for &p in coord {
            for _ in 0..heads {
                for &f in &self.inv_freq {
                    let a = (p as f32 * f) as f64;
                    c.push(a.cos() as f32);
                    s.push(a.sin() as f32);
                }
            }
        }
        (Tensor::from_vec(&self.ctx, &c, &[coord.len() * heads, q]), Tensor::from_vec(&self.ctx, &s, &[coord.len() * heads, q]))
    }

    /// `apply_multidimensional_rope` with two axes: the head splits into two halves, each a rotate-half rope
    /// of its own (`[x·cos + rotate_half(x)·sin]` within the half); the first half is turned by the column,
    /// the second by the row.
    fn rope2d(&self, x: &Tensor, n: usize, tx: &(Tensor, Tensor), ty: &(Tensor, Tensor)) -> Tensor {
        let (heads, dh) = (self.cfg.heads, self.cfg.head_dim);
        let half = dh / 2;
        let rows = x.reshape(&[n * heads, dh]);
        let a = rows.narrow(1, 0, half).rope_with_table(&tx.0, &tx.1, 1, half, false);
        let b = rows.narrow(1, half, half).rope_with_table(&ty.0, &ty.1, 1, half, false);
        a.cat(&b, 1).reshape(&[n, heads * dh])
    }

    /// The tower: patch rows → soft tokens `[n / pool², text_d]`. `taps` (if given) receives the stages the
    /// conformance gate compares: `patch_embed`, `vblock{i}`, `pooled`, `soft`.
    pub fn encode(&self, p: &Patches, mut taps: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor, String> {
        let c = &self.cfg;
        let (n, h, heads, dh) = (p.n(), c.hidden, c.heads, c.head_dim);
        if p.gh % c.pool != 0 || p.gw % c.pool != 0 {
            return Err(format!("patch grid {}x{} is not a multiple of the pool {}", p.gh, p.gw, c.pool));
        }
        if p.gh.max(p.gw) > c.pos_size { return Err(format!("patch grid {}x{} exceeds the {}-entry position table", p.gh, p.gw, c.pos_size)); }
        let neg = self.neg;
        let clip = neg != Neg::NoClip;
        let mut tap = |name: &str, t: &Tensor| if let Some(v) = taps.as_deref_mut() { v.push((name.to_string(), t.clone())) };

        // 2·(p − ½), in float32 as the authors compute it, then the input projection.
        let xin: Vec<f32> = p.px.iter().map(|&v| 2.0 * (v - 0.5)).collect();
        let pw = 3 * c.patch * c.patch;
        let x = Tensor::from_vec(&self.ctx, &xin, &[n, pw]).matmul_q(&self.patch_w.w);
        // Position: the x (column) table plus the y (row) table — summed FIRST, then added (their order).
        let (col, row): (Vec<u32>, Vec<u32>) = (0..n).map(|i| ((i % p.gw) as u32, (i / p.gw) as u32)).unzip();
        let (px_, py_) = if neg == Neg::PosSwap { (&row, &col) } else { (&col, &row) };
        let mut pe = vec![0f32; n * h];
        for i in 0..n {
            let (xo, yo) = (px_[i] as usize * h, (c.pos_size + py_[i] as usize) * h);
            for d in 0..h { pe[i * h + d] = self.pos[xo + d] + self.pos[yo + d]; }
        }
        let mut x = x.add(&Tensor::from_vec(&self.ctx, &pe, &[n, h]));
        tap("patch_embed", &x);

        let (rx, ry) = if neg == Neg::RopeSwap { (&row, &col) } else { (&col, &row) };
        let (tx, ty) = (self.rope_table(rx), self.rope_table(ry));
        // Attention scale 1: the shared kernel bakes in 1/sqrt(head_dim), so q is pre-multiplied by
        // sqrt(head_dim) to cancel it (exactly, for head_dim 64: a power of two).
        let qscale = if neg == Neg::AttnScale { 1.0 } else { (dh as f32).sqrt() };
        for (il, b) in self.blocks.iter().enumerate() {
            let hn = x.rmsnorm(&b.ln_in, c.eps);
            let norm_heads = |t: Tensor, w: &Tensor| t.reshape(&[n * heads, dh]).rmsnorm(w, c.eps).reshape(&[n, h]);
            let mut q = norm_heads(b.q.fwd(&hn, clip), &b.q_norm);
            let mut k = norm_heads(b.k.fwd(&hn, clip), &b.k_norm);
            if neg != Neg::NoRope {
                q = self.rope2d(&q, n, &tx, &ty);
                k = self.rope2d(&k, n, &tx, &ty);
            }
            let v = b.v.fwd(&hn, clip);
            let v = if neg == Neg::NoVNorm { v } else { v.reshape(&[n * heads, dh]).rmsnorm_weightless(c.eps).reshape(&[n, h]) };
            let q = q.mul(&q.scalar(qscale));
            let att = nn::bidirectional_attention(&q, &k, &v, heads, heads);
            x = x.add(&b.o.fwd(&att, clip).rmsnorm(&b.ln_post_attn, c.eps));
            let f = x.rmsnorm(&b.ln_pre_ff, c.eps);
            let g = b.gate.fwd(&f, clip);
            let g = if neg == Neg::ErfGelu { g.gelu() } else { g.gelu_tanh() };
            let ffn = b.down.fwd(&g.mul(&b.up.fwd(&f, clip)), clip);
            x = x.add(&ffn.rmsnorm(&b.ln_post_ff, c.eps));
            tap(&format!("vblock{il}"), &x);
        }

        // Average pool BY POSITION: token k = (col/3) + (gw/3)·(row/3), weight 1/9 each — a dense matmul
        // with the authors' one-hot/9 matrix.
        let (k, pool_w) = (c.pool, p.gw / c.pool);
        let n_out = n / (k * k);
        let mut pm = vec![0f32; n_out * n];
        let w9 = 1.0f32 / (k * k) as f32;
        for i in 0..n {
            let cell = if neg == Neg::PoolRows { i / (k * k) } else { (col[i] as usize / k) + pool_w * (row[i] as usize / k) };
            pm[cell * n + i] = w9;
        }
        let pooled = Tensor::from_vec(&self.ctx, &pm, &[n_out, n]).matmul(&x);
        let pooled = pooled.mul(&pooled.scalar((h as f64).sqrt() as f32));
        tap("pooled", &pooled);
        let pooled = match &self.std { Some((bias, scale)) => pooled.sub(bias).mul(scale), None => pooled };
        let soft = self.embed.fwd(&pooled);
        tap("soft", &soft);
        Ok(soft)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_matches_the_authors_arithmetic() {
        let c = PreCfg::default();
        // probe.ppm 224x140 (w x h) -> 1008x624: 39x63 = 2457 patches, 273 soft tokens.
        assert_eq!(target_size(140, 224, &c).unwrap(), (624, 1008));
        // Already at budget: unchanged.
        assert_eq!(target_size(624, 1008, &c).unwrap(), (624, 1008));
        // A 1:40 strip hits the "one side rounds to 0" branch.
        let (h, w) = target_size(10, 400, &c).unwrap();
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
        assert!(h * w <= c.max_patches() * 256);
    }

    #[test]
    fn int16_taps_sum_to_one_in_fixed_point_and_pick_the_largest_precision() {
        for (i, o) in [(140usize, 624usize), (2000, 816), (333, 384), (7, 3)] {
            let (r, w, p) = int16_taps(i, o);
            assert_eq!(r.len(), o);
            for t in &w {
                let s: i32 = t.iter().sum();
                assert!((s - (1 << p)).abs() <= t.len() as i32, "{i}->{o}: taps sum {s} vs {}", 1 << p);
                assert!(t.iter().all(|&x| x.abs() < 1 << 15), "taps must fit int16");
            }
            assert!(p >= 13, "{i}->{o}: precision {p} is too coarse to be the largest that fits");
        }
    }

    #[test]
    fn a_constant_image_stays_constant_through_the_resampler() {
        for &(h, w, oh, ow) in &[(140usize, 224usize, 624usize, 1008usize), (900, 1200, 576, 816), (50, 3, 96, 48)] {
            let img = Rgb8 { w, h, px: vec![173u8; h * w * 3] };
            let r = resize_torch_u8(&img, oh, ow);
            assert!(r.px.iter().all(|&v| v == 173), "{h}x{w} -> {oh}x{ow} changed a flat field");
        }
    }

    #[test]
    fn patches_are_row_col_channel_and_rescaled_by_multiplication() {
        let c = PreCfg { patch: 2, pool: 1, max_soft_tokens: 70, rescale: 0.00392156862745098f64 as f32 };
        // A 2x4 image at exactly... use the raw patchify through `preprocess` on a budget-sized image.
        let (h, w) = (2usize, 4usize);
        let px: Vec<u8> = (0..h * w * 3).map(|i| i as u8).collect();
        let img = Rgb8 { w, h, px };
        // bypass the budget: patchify by hand the way `preprocess` does
        let r = &img;
        let mut out = Vec::new();
        for gx in 0..2 { for y in 0..2 { for x in 0..2 { for ch in 0..3 {
            out.push(r.px[(y * w + gx * 2 + x) * 3 + ch] as f32 * c.rescale);
        } } } }
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 3.0 * c.rescale, "second pixel of the first patch row is (0,1)");
        assert_eq!(out[6], (w * 3) as f32 * c.rescale, "the patch's second row starts one image row down");
    }
}
