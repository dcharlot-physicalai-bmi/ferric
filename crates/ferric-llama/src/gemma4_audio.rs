//! **Gemma 4's audio tower** (`gemma4_audio`, E2B / E4B / 12B) and its feature extractor — from a waveform
//! to the soft tokens that replace `<|audio|>` in the prompt. Verified against the authors' code
//! (transformers `modeling_gemma4`, `Gemma4AudioFeatureExtractor`) stage by stage:
//! `scripts/gemma4_mm_conformance.sh`.
//!
//! ```text
//! 16 kHz pcm ─ pad to a multiple of 128 samples (≤ 30 s: the authors TRUNCATE at 480 000 samples)
//!            ─ 160 zeros in front ("semicausal") ─ 320-sample frames every 160, periodic Hann
//!            ─ |rfft 512| (MAGNITUDE) · HTK mel 128 (0–8 kHz, unnormalised) ─ ln(· + 0.001)
//!            ─ frame valid iff its last sample is real audio; invalid frames zeroed
//! SSCP       ─ 2 × { conv 3×3 stride 2 (no bias) → LayerNorm over channels (no bias) → ReLU },
//!              the time mask halved ([::2]) and applied to each conv's input; (t, freq·chan) → linear
//! 12 ×       ─ ½·FF → chunked local attention (12-frame blocks, 12 frames of left context, relative
//!              position logits, softcap 50) → LightConv (GLU → causal depthwise k5 → RMSNorm → SiLU)
//!              → ½·FF → RMSNorm; every projection a CLIPPED linear
//! output_proj (1024 → 1536, bias) ─ embed_audio: weightless RMSNorm → projection to the text width
//! ```
//!
//! What the module had to READ rather than assume — each has a negative control in the gate:
//!   * the spectrum is the MAGNITUDE (`abs(rfft)`), not the power, and the log adds 0.001 instead of
//!     clamping — Parakeet's and Whisper's front ends are both different, so neither is reused;
//!   * the attention is LOCAL and CAUSAL in effect: chunk 12, `context_left 13` means keys at distances
//!     0..=11 (the authors' sliding mask is `dist < context_left - 1`), right context 0;
//!   * q is scaled by `head_dim^-½ / ln 2 · softplus(per_dim_scale)` and k by `ln(1 + e) / ln 2`;
//!   * the relative term is `q · relative_k_proj(sinusoid(distance))`, the sinusoid laid out `[sin | cos]`
//!     and indexed by DISTANCE through the authors' block rel-shift;
//!   * the feed-forwards are half-step (`residual_weight 0.5`) and post-normed INSIDE the half step.
//!
//! The `gradient_clipping` clamps (±1e10 in every config) are not applied: no activation comes within
//! twenty orders of magnitude of them.
use crate::gemma4_mm::{Embedder, Lin, Src};
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

/// The feature extractor's constants (`processor_config.json` → `feature_extractor`).
#[derive(Debug, Clone, Copy)]
pub struct FeCfg {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub frame: usize,
    pub hop: usize,
    pub n_fft: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub mel_floor: f64,
    /// `max_length`: the authors truncate longer audio to this many samples.
    pub max_samples: usize,
    /// `pad_to_multiple_of`.
    pub pad_multiple: usize,
    /// `audio_seq_length`: the processor's cap on soft tokens per clip.
    pub max_tokens: usize,
}

impl Default for FeCfg {
    fn default() -> Self {
        FeCfg { sample_rate: 16000, n_mels: 128, frame: 320, hop: 160, n_fft: 512, fmin: 0.0, fmax: 8000.0,
                mel_floor: 1e-3, max_samples: 480_000, pad_multiple: 128, max_tokens: 750 }
    }
}

impl FeCfg {
    pub fn load(dir: &str) -> Result<FeCfg, String> {
        let j: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!("{dir}/processor_config.json")).map_err(|e| format!("processor_config.json: {e}"))?)
            .map_err(|e| format!("processor_config.json: {e}"))?;
        let f = &j["feature_extractor"];
        if f["feature_extractor_type"].as_str() != Some("Gemma4AudioFeatureExtractor") {
            return Err(format!("feature_extractor_type {:?}", f["feature_extractor_type"]));
        }
        // What this front end does NOT implement is refused by name, not ignored.
        for (k, off) in [("preemphasis", 0.0), ("dither", 0.0), ("input_scale_factor", 1.0)] {
            if f[k].as_f64().unwrap_or(off) != off { return Err(format!("feature_extractor.{k} = {} is not implemented", f[k])); }
        }
        if !f["per_bin_mean"].is_null() || !f["per_bin_stddev"].is_null() || f["fft_overdrive"].as_bool() == Some(true) {
            return Err("per-bin normalisation / fft_overdrive are not implemented".into());
        }
        let d = FeCfg::default();
        let u = |k: &str, dv: usize| f[k].as_u64().map(|x| x as usize).unwrap_or(dv);
        let c = FeCfg {
            sample_rate: u("sampling_rate", d.sample_rate), n_mels: u("feature_size", d.n_mels),
            frame: u("frame_length", d.frame), hop: u("hop_length", d.hop), n_fft: u("fft_length", d.n_fft),
            fmin: f["min_frequency"].as_f64().unwrap_or(d.fmin), fmax: f["max_frequency"].as_f64().unwrap_or(d.fmax),
            mel_floor: f["mel_floor"].as_f64().unwrap_or(d.mel_floor),
            max_samples: d.max_samples, pad_multiple: d.pad_multiple,
            max_tokens: j["audio_seq_length"].as_u64().map(|x| x as usize).unwrap_or(d.max_tokens),
        };
        if c.n_fft < c.frame || !c.n_fft.is_power_of_two() { return Err(format!("fft_length {} vs frame {}", c.n_fft, c.frame)); }
        Ok(c)
    }
}

/// A clip's log-mel features.
#[derive(Debug, Clone)]
pub struct Feats {
    /// `[frames, n_mels]`, invalid frames zeroed (as the authors return them).
    pub mel: Vec<f32>,
    pub frames: usize,
    /// Valid frames — always a prefix.
    pub valid: usize,
    /// Samples the authors' truncation dropped (0 for clips up to 30 s).
    pub truncated: usize,
}

impl Feats {
    /// Soft tokens this clip becomes: the valid-frame count halved (ceil) twice — the processor's
    /// `mask[::2]` twice — capped at `max_tokens`.
    pub fn tokens(&self) -> usize { self.valid.div_ceil(2).div_ceil(2) }
}

/// WAV (16-bit PCM or 32-bit float, any channel count averaged to mono) → samples and rate.
pub fn read_wav(b: &[u8]) -> Result<(Vec<f32>, usize), String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" { return Err("not a RIFF/WAVE file".into()); }
    let (mut i, mut fmt, mut data) = (12usize, None, None);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " && d.len() >= 16 {
            let u16at = |k: usize| u16::from_le_bytes([d[k], d[k + 1]]);
            let tag = if u16at(0) == 0xFFFE && d.len() >= 26 { u16at(24) } else { u16at(0) };
            fmt = Some((tag, u16at(2) as usize, u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize, u16at(14)));
        }
        if id == b"data" { data = Some(d); }
        i += 8 + sz + (sz & 1);
    }
    let ((tag, ch, rate, bits), d) = (fmt.ok_or("WAV without a fmt chunk")?, data.ok_or("WAV without a data chunk")?);
    if ch == 0 || rate == 0 { return Err("WAV declares 0 channels or 0 Hz".into()); }
    let s: Vec<f32> = match (tag, bits) {
        (1, 16) => d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        (3, 32) => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => return Err(format!("WAV format tag {tag} at {bits} bits: only 16-bit PCM and 32-bit float are read")),
    };
    Ok((s.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect(), rate))
}

/// In-place radix-2 FFT in f64.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j |= bit;
        if i < j { re.swap(i, j); im.swap(i, j); }
    }
    let mut len = 2;
    while len <= n {
        for i in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let a = -2.0 * std::f64::consts::PI * k as f64 / len as f64;
                let (c, s) = (a.cos(), a.sin());
                let (p, q) = (i + k, i + k + len / 2);
                let (vr, vi) = (re[q] * c - im[q] * s, re[q] * s + im[q] * c);
                re[q] = re[p] - vr; im[q] = im[p] - vi;
                re[p] += vr; im[p] += vi;
            }
        }
        len <<= 1;
    }
}

/// transformers' `mel_filter_bank(norm=None, mel_scale="htk")`: `[n_bins, n_mels]`, f64, the triangles laid
/// out in Hz between mel-spaced edges.
pub fn mel_filters(c: &FeCfg) -> Vec<f64> {
    let n_bins = c.n_fft / 2 + 1;
    let hz2mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel2hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (m0, m1) = (hz2mel(c.fmin), hz2mel(c.fmax));
    let n = c.n_mels + 2;
    // np.linspace(m0, m1, n): start + i*step, the last point exactly m1.
    let edges: Vec<f64> = (0..n).map(|i| if i == n - 1 { m1 } else { m0 + i as f64 * ((m1 - m0) / (n - 1) as f64) })
        .map(mel2hz).collect();
    let nyq = (c.sample_rate / 2) as f64;
    let fft_f: Vec<f64> = (0..n_bins).map(|i| if i == n_bins - 1 { nyq } else { i as f64 * (nyq / (n_bins - 1) as f64) }).collect();
    let mut w = vec![0f64; n_bins * c.n_mels];
    for (b, &f) in fft_f.iter().enumerate() {
        for m in 0..c.n_mels {
            let down = -(edges[m] - f) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            w[b * c.n_mels + m] = 0f64.max(down.min(up));
        }
    }
    w
}

/// `Gemma4AudioFeatureExtractor.__call__` on one clip.
pub fn log_mel(pcm: &[f32], c: &FeCfg) -> Feats {
    let truncated = pcm.len().saturating_sub(c.max_samples);
    let real = pcm.len() - truncated;
    let padded = real.div_ceil(c.pad_multiple) * c.pad_multiple;
    let left = c.frame / 2;
    let mut x = vec![0f32; left + padded];
    x[left..left + real].copy_from_slice(&pcm[..real]);
    let win: Vec<f32> = (0..c.frame).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / c.frame as f64).cos()) as f32).collect();
    let span = c.frame + 1;
    let frames = if x.len() >= span { (x.len() - span) / c.hop + 1 } else { 0 };
    let fb = mel_filters(c);
    let n_bins = c.n_fft / 2 + 1;
    let mut mel = vec![0f32; frames * c.n_mels];
    let mut valid = 0;
    let (mut re, mut im) = (vec![0f64; c.n_fft], vec![0f64; c.n_fft]);
    let mut mag = vec![0f64; n_bins];
    for t in 0..frames {
        // Valid iff the frame's LAST sample (index t·hop + frame in the padded stream) is real audio.
        let end = t * c.hop + span - 1;
        let ok = end >= left && end - left < real;
        if !ok { continue; }
        valid = t + 1;
        for k in 0..c.n_fft {
            re[k] = if k < c.frame { (x[t * c.hop + k] * win[k]) as f64 } else { 0.0 };
            im[k] = 0.0;
        }
        fft(&mut re, &mut im);
        for k in 0..n_bins { mag[k] = (re[k] * re[k] + im[k] * im[k]).sqrt() as f32 as f64; }
        for m in 0..c.n_mels {
            let mut s = 0f64;
            for k in 0..n_bins { s += mag[k] * fb[k * c.n_mels + m]; }
            mel[t * c.n_mels + m] = (s + c.mel_floor).ln() as f32;
        }
    }
    Feats { mel, frames, valid, truncated }
}

/// What the tower is built from (`audio_config`).
#[derive(Debug, Clone)]
pub struct AudioCfg {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub chunk: usize,
    pub ctx_left: usize,
    pub ctx_right: usize,
    pub logit_cap: f32,
    pub residual_weight: f32,
    pub conv_k: usize,
    pub sub_ch: [usize; 2],
    pub out_dims: usize,
    pub eps: f32,
    pub clipped: bool,
    pub text_d: usize,
    pub n_mels: usize,
}

/// Negative controls (`FERRIC_G4A_NEG`): each removes ONE mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neg { None, NoClip, NoRelPos, RelShiftOff, MaskWindow, ConvNonCausal, NoFfwHalf, NoSoftcap, NoPerDimScale, ReluSub }

impl Neg {
    pub fn from_env() -> Result<Neg, String> {
        Ok(match std::env::var("FERRIC_G4A_NEG").unwrap_or_default().as_str() {
            "" => Neg::None, "noclip" => Neg::NoClip, "no_relpos" => Neg::NoRelPos, "rel_shift_off" => Neg::RelShiftOff,
            "mask_window" => Neg::MaskWindow, "conv_noncausal" => Neg::ConvNonCausal, "no_ffw_half" => Neg::NoFfwHalf,
            "no_softcap" => Neg::NoSoftcap, "no_per_dim_scale" => Neg::NoPerDimScale, "relu_sub" => Neg::ReluSub,
            o => return Err(format!("FERRIC_G4A_NEG={o}: not a control (noclip|no_relpos|rel_shift_off|mask_window|\
                                     conv_noncausal|no_ffw_half|no_softcap|no_per_dim_scale|relu_sub)")),
        })
    }
}

struct Ffw { pre: Tensor, post: Tensor, l1: Lin, l2: Lin }

struct Layer {
    ff1: Ffw, ff2: Ffw,
    norm_pre_attn: Tensor, norm_post_attn: Tensor, norm_out: Tensor,
    q: Lin, k: Lin, v: Lin, post: Lin,
    rel_k: Lin,
    /// `softplus(per_dim_scale)` — host-computed once.
    pds: Vec<f32>,
    conv_pre: Tensor, conv_norm: Tensor, conv_start: Lin, conv_end: Lin,
    /// `[hidden, conv_k]`.
    conv_w: Tensor,
}

pub struct AudioTower {
    ctx: Arc<Context>,
    pub cfg: AudioCfg,
    /// HWIO conv kernels and the LayerNorm weights of the two subsampling blocks.
    sub_w: [Tensor; 2],
    sub_norm: [Tensor; 2],
    sub_zero: [Tensor; 2],
    sub_proj: Lin,
    layers: Vec<Layer>,
    out_proj: Lin,
    out_bias: Tensor,
    pub embed: Embedder,
    /// `[ctx_left, hidden]` sinusoid rows for distances `ctx_left-1 … 0` — the authors' `rel_pos_enc` output.
    pos: Tensor,
    pub neg: Neg,
}

impl AudioTower {
    pub fn load(ctx: &Arc<Context>, src: &Src) -> Result<AudioTower, String> {
        let cfg = match src {
            Src::Hf { cfg, .. } => {
                let a = &cfg["audio_config"];
                if a.is_null() { return Err("this checkpoint has no audio tower (no audio_config)".into()); }
                let u = |k: &str| a[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("audio_config.{k} missing"));
                let fl = |k: &str| a[k].as_f64().map(|x| x as f32).ok_or_else(|| format!("audio_config.{k} missing"));
                if a["hidden_act"].as_str() != Some("silu") { return Err(format!("audio hidden_act {:?}", a["hidden_act"])); }
                let sc = a["subsampling_conv_channels"].as_array().ok_or("subsampling_conv_channels missing")?;
                if sc.len() != 2 { return Err(format!("{} subsampling layers; the authors' SSCP has 2", sc.len())); }
                AudioCfg {
                    layers: u("num_hidden_layers")?, hidden: u("hidden_size")?, heads: u("num_attention_heads")?,
                    chunk: u("attention_chunk_size")?, ctx_left: u("attention_context_left")?, ctx_right: u("attention_context_right")?,
                    logit_cap: fl("attention_logit_cap")?, residual_weight: fl("residual_weight")?, conv_k: u("conv_kernel_size")?,
                    sub_ch: [sc[0].as_u64().unwrap_or(0) as usize, sc[1].as_u64().unwrap_or(0) as usize],
                    out_dims: u("output_proj_dims")?, eps: fl("rms_norm_eps")?,
                    clipped: a["use_clipped_linears"].as_bool().ok_or("use_clipped_linears missing")?,
                    text_d: cfg["text_config"]["hidden_size"].as_u64().ok_or("text hidden_size")? as usize,
                    n_mels: 128,
                }
            }
            Src::Gguf(g) => {
                use ferric_gguf::{GgufSource, Meta};
                let md = g.metadata();
                match md.get("clip.audio.projector_type") {
                    Some(Meta::Str(s)) if s == "gemma4a" => {}
                    o => return Err(format!("this mmproj has no Gemma 4 audio tower (clip.audio.projector_type {o:?})")),
                }
                let u = |k: &str| match md.get(&format!("clip.audio.{k}")) { Some(Meta::U(x)) => Ok(*x as usize), _ => Err(format!("clip.audio.{k} missing")) };
                let c0 = g.tensor("a.conv1d.0.weight").ok_or("no a.conv1d.0.weight")?.dims[3] as usize;
                let c1 = g.tensor("a.conv1d.1.weight").ok_or("no a.conv1d.1.weight")?.dims[3] as usize;
                let out_dims = g.tensor("a.pre_encode.out.weight").ok_or("no a.pre_encode.out.weight")?.dims[1] as usize;
                let conv_k = g.tensor("a.blk.0.conv_dw.weight").ok_or("no a.blk.0.conv_dw.weight")?.dims[0] as usize;
                AudioCfg {
                    layers: u("block_count")?, hidden: u("embedding_length")?, heads: u("attention.head_count")?,
                    // ⚠ Not in the mmproj: the gemma4a projector's fixed values, the same in every Gemma 4
                    // audio_config published (chunk 12, context 13/0, cap 50, residual 0.5).
                    chunk: 12, ctx_left: 13, ctx_right: 0, logit_cap: 50.0, residual_weight: 0.5,
                    conv_k, sub_ch: [c0, c1], out_dims,
                    eps: match md.get("clip.audio.attention.layer_norm_epsilon") { Some(Meta::F(e)) => *e as f32, _ => 1e-6 },
                    clipped: g.tensor("a.blk.0.attn_q.input_min").is_some(),
                    text_d: u("projection_dim")?, n_mels: u("num_mel_bins")?,
                }
            }
        };
        if cfg.ctx_right != 0 { return Err(format!("attention_context_right {}: only 0 is implemented", cfg.ctx_right)); }
        if cfg.chunk + 1 != cfg.ctx_left { return Err(format!("chunk {} / context_left {}: the relative table is built for chunk + 1 = context_left", cfg.chunk, cfg.ctx_left)); }
        let (h, c) = (cfg.hidden, cfg.clipped);
        let p = "model.audio_tower";
        // Subsampling: conv [O, I, 3, 3] → HWIO [3, 3, I, O].
        let conv = |j: usize, cin: usize, cout: usize| -> Result<(Tensor, Tensor, Tensor), String> {
            let (w, s) = src.f32s(&format!("{p}.subsample_conv_projection.layer{j}.conv.weight"))?;
            if s != [cout, cin, 3, 3] { return Err(format!("layer{j}.conv {s:?}, expected [{cout}, {cin}, 3, 3]")); }
            let mut hw = vec![0f32; 9 * cin * cout];
            for o in 0..cout { for i in 0..cin { for ky in 0..3 { for kx in 0..3 {
                hw[((ky * 3 + kx) * cin + i) * cout + o] = w[((o * cin + i) * 3 + ky) * 3 + kx];
            } } } }
            Ok((Tensor::from_vec(ctx, &hw, &[3, 3, cin, cout]),
                src.vec(ctx, &format!("{p}.subsample_conv_projection.layer{j}.norm.weight"), cout)?,
                Tensor::from_vec(ctx, &vec![0f32; cout], &[cout])))
        };
        let (w0, n0, z0) = conv(0, 1, cfg.sub_ch[0])?;
        let (w1, n1, z1) = conv(1, cfg.sub_ch[0], cfg.sub_ch[1])?;
        let f_after = cfg.n_mels.div_ceil(2).div_ceil(2);
        let sub_in = f_after * cfg.sub_ch[1];
        let sub_proj = Lin::load(src, ctx, &format!("{p}.subsample_conv_projection.input_proj_linear"), h, sub_in, false)?;
        let hd = h / cfg.heads;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let b = format!("{p}.layers.{i}");
            let lin = |m: &str, o: usize, inp: usize| Lin::load(src, ctx, &format!("{b}.{m}"), o, inp, c);
            let ffw = |n: &str| -> Result<Ffw, String> { Ok(Ffw {
                pre: src.vec(ctx, &format!("{b}.{n}.pre_layer_norm.weight"), h)?,
                post: src.vec(ctx, &format!("{b}.{n}.post_layer_norm.weight"), h)?,
                l1: lin(&format!("{n}.ffw_layer_1"), 4 * h, h)?, l2: lin(&format!("{n}.ffw_layer_2"), h, 4 * h)?,
            }) };
            let pds = src.softplus_per_dim_scale(&format!("{b}.self_attn.per_dim_scale"))?;
            if pds.len() != hd { return Err(format!("per_dim_scale has {} entries, expected {hd}", pds.len())); }
            let (cw, cs) = src.f32s(&format!("{b}.lconv1d.depthwise_conv1d.weight"))?;
            if cw.len() != h * cfg.conv_k { return Err(format!("depthwise conv {cs:?}, expected [{h}, 1, {}]", cfg.conv_k)); }
            layers.push(Layer {
                ff1: ffw("feed_forward1")?, ff2: ffw("feed_forward2")?,
                norm_pre_attn: src.vec(ctx, &format!("{b}.norm_pre_attn.weight"), h)?,
                norm_post_attn: src.vec(ctx, &format!("{b}.norm_post_attn.weight"), h)?,
                norm_out: src.vec(ctx, &format!("{b}.norm_out.weight"), h)?,
                q: lin("self_attn.q_proj", h, h)?, k: lin("self_attn.k_proj", h, h)?, v: lin("self_attn.v_proj", h, h)?,
                post: lin("self_attn.post", h, h)?,
                rel_k: Lin::load(src, ctx, &format!("{b}.self_attn.relative_k_proj"), h, h, false)?,
                pds,
                conv_pre: src.vec(ctx, &format!("{b}.lconv1d.pre_layer_norm.weight"), h)?,
                conv_norm: src.vec(ctx, &format!("{b}.lconv1d.conv_norm.weight"), h)?,
                conv_start: lin("lconv1d.linear_start", 2 * h, h)?, conv_end: lin("lconv1d.linear_end", h, h)?,
                conv_w: Tensor::from_vec(ctx, &cw, &[h, cfg.conv_k]),
            });
        }
        let out_proj = Lin::load(src, ctx, &format!("{p}.output_proj"), cfg.out_dims, h, false)?;
        let out_bias = src.vec(ctx, &format!("{p}.output_proj.bias"), cfg.out_dims)?;
        let embed = Embedder { proj: src.qmat(ctx, "model.embed_audio.embedding_projection.weight", cfg.text_d, cfg.out_dims)?,
                               eps: cfg.eps, text_d: cfg.text_d };
        // `Gemma4AudioRelPositionalEncoding`: timescales exp(-i·ln(1e4)/(h/2 - 1)) in float32, positions
        // ctx_left-1 … 0 (DESCENDING), rows [sin | cos].
        let half = h / 2;
        let inc = (10000.0f64 / 1.0).ln() / ((half as f64) - 1.0).max(1.0);
        let inv: Vec<f32> = (0..half).map(|i| ((i as f32) * (-inc as f32)).exp()).collect();
        let n_pos = cfg.ctx_left;
        let mut pe = vec![0f32; n_pos * h];
        for r in 0..n_pos {
            let pos = (n_pos - 1 - r) as f32;
            for i in 0..half {
                let a = (pos * inv[i]) as f64;
                pe[r * h + i] = a.sin() as f32;
                pe[r * h + half + i] = a.cos() as f32;
            }
        }
        Ok(AudioTower { ctx: ctx.clone(), cfg, sub_w: [w0, w1], sub_norm: [n0, n1], sub_zero: [z0, z1], sub_proj, layers,
                        out_proj, out_bias, embed, pos: Tensor::from_vec(ctx, &pe, &[n_pos, h]), neg: Neg::from_env()? })
    }

    fn ffw(&self, x: &Tensor, f: &Ffw) -> Tensor {
        let (eps, clip) = (self.cfg.eps, self.neg != Neg::NoClip);
        let hh = f.l2.fwd(&f.l1.fwd(&x.rmsnorm(&f.pre, eps), clip).silu(), clip).rmsnorm(&f.post, eps);
        let w = if self.neg == Neg::NoFfwHalf { 1.0 } else { self.cfg.residual_weight };
        x.add(&hh.mul(&hh.scalar(w)))
    }

    /// Chunked local attention with relative position logits, on `t` valid frames.
    fn attention(&self, x: &Tensor, l: &Layer, t: usize, mask: &Tensor) -> Tensor {
        let c = &self.cfg;
        let (h, nh) = (c.hidden, c.heads);
        let hd = h / nh;
        let clip = self.neg != Neg::NoClip;
        let q_scale = ((hd as f64).powf(-0.5) / std::f64::consts::LN_2) as f32;
        let k_scale = ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32;
        let q = l.q.fwd(x, clip);
        let q = q.mul(&q.scalar(q_scale));
        let q = if self.neg == Neg::NoPerDimScale { q } else {
            let s = Tensor::from_vec(&self.ctx, &l.pds, &[hd]);
            q.reshape(&[t * nh, hd]).mul(&s.broadcast_to(&[t * nh, hd])).reshape(&[t, h])
        };
        let k = l.k.fwd(x, clip);
        let k = k.mul(&k.scalar(k_scale));
        let v = l.v.fwd(x, clip);
        // Relative keys: [ctx_left, h] — row r is distance ctx_left-1-r.
        let rk = l.rel_k.fwd(&self.pos, false);
        let n_pos = c.ctx_left;
        let m = n_pos.min(t);
        let mut heads = Vec::with_capacity(nh);
        for hi in 0..nh {
            let qh = q.narrow(1, hi * hd, hd).contiguous();
            let kh = k.narrow(1, hi * hd, hd).contiguous();
            let vh = v.narrow(1, hi * hd, hd).contiguous();
            let ac = qh.matmul_bt(&kh); // [t, t]
            let s = if self.neg == Neg::NoRelPos { ac } else {
                // bd[i][p] = q_i · relk[p] (p ↔ distance ctx_left-1-p). Placed so `rel_shift` (row i reads
                // column (t-1) - (i - j)) lands distance d = i - j on p = ctx_left-1-d.
                let rkh = rk.narrow(1, hi * hd, hd).contiguous();
                let bd = qh.matmul_bt(&rkh); // [t, n_pos]
                let bd = bd.narrow(1, n_pos - m, m).contiguous();
                let bd = if self.neg == Neg::RelShiftOff {
                    // the unshifted table at fixed columns: every query reads the same distances
                    let z = Tensor::from_vec(&self.ctx, &vec![0f32; t * (t - m)], &[t, t - m]);
                    if t > m { z.cat(&bd, 1) } else { bd }
                } else {
                    let mut parts = Vec::new();
                    if t > m { parts.push(Tensor::from_vec(&self.ctx, &vec![0f32; t * (t - m)], &[t, t - m])); }
                    parts.push(bd);
                    if t > 1 { parts.push(Tensor::from_vec(&self.ctx, &vec![0f32; t * (t - 1)], &[t, t - 1])); }
                    parts[1..].iter().fold(parts[0].clone(), |a, b| a.cat(b, 1)).rel_shift()
                };
                ac.add(&bd)
            };
            let s = if self.neg == Neg::NoSoftcap { s } else { s.softcap(c.logit_cap) };
            heads.push(s.add(mask).softmax(1).matmul(&vh));
        }
        let att = heads[1..].iter().fold(heads[0].clone(), |a, b| a.cat(b, 1));
        l.post.fwd(&att, clip)
    }

    /// The tower: features → soft tokens `[valid frames / 4, text_d]`. `taps` receives `subsample`,
    /// `ablock{i}`, `output_proj`, `soft` (valid rows only).
    pub fn encode(&self, f: &Feats, mut taps: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor, String> {
        let c = &self.cfg;
        let (h, eps, nm) = (c.hidden, c.eps, c.n_mels);
        if f.valid == 0 { return Err("the clip has no complete frame".into()); }
        let mut tap = |name: &str, t: &Tensor| if let Some(v) = taps.as_deref_mut() { v.push((name.to_string(), t.clone())) };
        let clip = self.neg != Neg::NoClip;
        // SSCP. NHWC: H = time, W = mel/freq, C = channels. Each conv's input rows past the valid count
        // (the halved mask) are zero, exactly as the authors multiply by the mask.
        let mut x = Tensor::from_vec(&self.ctx, &f.mel, &[1, f.frames, nm, 1]);
        let mut valid = f.valid;
        let mut rows = f.frames;
        let mut width = nm;
        for j in 0..2 {
            let cin = if j == 0 { 1 } else { c.sub_ch[0] };
            if valid < rows {
                let keep = x.narrow(1, 0, valid).contiguous();
                let z = Tensor::from_vec(&self.ctx, &vec![0f32; (rows - valid) * width * cin], &[1, rows - valid, width, cin]);
                x = keep.cat(&z, 1);
            }
            let y = x.conv2d(&self.sub_w[j], (2, 2), (1, 1));
            let (r2, w2, co) = (y.shape[1], y.shape[2], y.shape[3]);
            let y = y.reshape(&[r2 * w2, co]).layernorm(&self.sub_norm[j], &self.sub_zero[j], eps);
            let y = if self.neg == Neg::ReluSub { y } else { y.relu() };
            x = y.reshape(&[1, r2, w2, co]);
            rows = r2; width = w2; valid = valid.div_ceil(2);
        }
        let t = valid.min(rows);
        let x = x.narrow(1, 0, t).contiguous().reshape(&[t, width * c.sub_ch[1]]);
        let mut x = self.sub_proj.fwd(&x, false);
        tap("subsample", &x);

        // The attention mask over the valid frames: key j visible to query i iff 0 <= i - j < ctx_left - 1.
        let win = if self.neg == Neg::MaskWindow { c.ctx_left } else { c.ctx_left - 1 };
        let mut mv = vec![-1e9f32; t * t];
        for i in 0..t { for j in i.saturating_sub(win - 1)..=i { mv[i * t + j] = 0.0; } }
        let mask = Tensor::from_vec(&self.ctx, &mv, &[t, t]);
        for (il, l) in self.layers.iter().enumerate() {
            x = self.ffw(&x, &l.ff1);
            let a = self.attention(&x.rmsnorm(&l.norm_pre_attn, eps), l, t, &mask);
            x = x.add(&a.rmsnorm(&l.norm_post_attn, eps));
            // LightConv: GLU (first half gated by the sigmoid of the second), causal depthwise conv.
            let y = l.conv_start.fwd(&x.rmsnorm(&l.conv_pre, eps), clip);
            let g = y.narrow(1, 0, h).contiguous().mul(&y.narrow(1, h, h).contiguous().sigmoid());
            let g = if self.neg == Neg::ConvNonCausal {
                // centred window: shift the causal output back by k/2
                let k2 = c.conv_k / 2;
                let pad = Tensor::from_vec(&self.ctx, &vec![0f32; k2 * h], &[k2, h]);
                g.cat(&pad, 0).depthwise_conv1d_causal(&l.conv_w, c.conv_k).narrow(0, k2, t).contiguous()
            } else { g.depthwise_conv1d_causal(&l.conv_w, c.conv_k) };
            let g = g.rmsnorm(&l.conv_norm, eps).silu();
            x = x.add(&l.conv_end.fwd(&g, clip));
            x = self.ffw(&x, &l.ff2);
            x = x.rmsnorm(&l.norm_out, eps);
            tap(&format!("ablock{il}"), &x);
        }
        let o = self.out_proj.fwd(&x, false).add(&self.out_bias.broadcast_to(&[t, c.out_dims]));
        tap("output_proj", &o);
        let soft = self.embed.fwd(&o);
        tap("soft", &soft);
        Ok(soft)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_validity_is_the_last_sample_rule_and_tokens_halve_twice() {
        let c = FeCfg::default();
        // 1 s of audio: 16000 samples, already a multiple of 128 -> no padding.
        let f = log_mel(&vec![0.01f32; 16000], &c);
        // (16000 + 160 - 321) / 160 + 1 = 99 frames; the last has its end sample at 98*160+320-160 = 15840 < 16000.
        assert_eq!(f.frames, 99);
        assert_eq!(f.valid, 99);
        assert_eq!(f.tokens(), 25);
        // 16001 samples pad to 16128: one more frame exists and is NOT valid (its end sample is padding).
        let f = log_mel(&vec![0.01f32; 16001], &c);
        assert_eq!((f.frames, f.valid), (100, 99));
        assert!(f.mel[99 * 128..].iter().all(|&v| v == 0.0), "an invalid frame is zeroed");
    }

    #[test]
    fn silence_is_ln_of_the_floor() {
        let f = log_mel(&vec![0f32; 3200], &FeCfg::default());
        let want = (1e-3f64).ln() as f32;
        assert!(f.mel[..f.valid * 128].iter().all(|&v| v == want));
    }

    #[test]
    fn htk_filters_peak_at_their_centres_and_are_unnormalised() {
        let c = FeCfg::default();
        let w = mel_filters(&c);
        let col_max = |m: usize| (0..257).map(|b| w[b * 128 + m]).fold(0f64, f64::max);
        // Unnormalised triangles peak near 1 (exactly 1 only when a bin lands on the centre).
        for m in [10, 60, 120] { assert!(col_max(m) > 0.5 && col_max(m) <= 1.0, "filter {m} peak {}", col_max(m)); }
    }

    #[test]
    fn a_tone_lands_in_the_mel_band_that_contains_it() {
        let c = FeCfg::default();
        let pcm: Vec<f32> = (0..16000).map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 16000.0).sin() as f32 * 0.5).collect();
        let f = log_mel(&pcm, &c);
        let row = &f.mel[50 * 128..51 * 128];
        let arg = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        // HTK mel of 1 kHz = 1000.0; the 0–8 kHz HTK axis is 0–2840 mel over 129 steps -> band ~45.
        let mel_1k = 2595.0 * (1.0 + 1000.0f64 / 700.0).log10();
        let step = 2595.0 * (1.0 + 8000.0f64 / 700.0).log10() / 129.0;
        let want = (mel_1k / step).round() as i64 - 1;
        assert!((arg as i64 - want).abs() <= 1, "1 kHz peaked in band {arg}, expected ~{want}");
    }
}
