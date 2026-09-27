//! **MiMo-V2.5-ASR** (Xiaomi, MIT) — speech recognition by a 7B language model, read from the authors'
//! checkpoints and checked against their code stage by stage (`examples/mimo_asr_stages.rs`,
//! `scripts/mimo_asr_conformance.sh`).
//!
//! Three models in a row, and nothing like a transducer:
//!
//! 1. **MiMo-Audio-Tokenizer** turns 24 kHz audio into DISCRETE codes: a log-mel frontend, two
//!    convolutions (100 -> 50 Hz), 32 bidirectional transformer layers with one long skip, a strided
//!    convolution (-> 25 Hz), then residual vector quantisation. ASR keeps the first 8 of 20 codebooks.
//! 2. A **patch encoder** sums the 8 code embeddings per frame and runs a 6-layer transformer inside
//!    each group of 4 frames (bidirectional within the group, nothing across), then folds the group into
//!    one 4096-wide row: 6.25 language-model positions per second of audio.
//! 3. The **language model** is a plain Qwen2 (served by [`crate::qwen3`]) reading those rows between
//!    `<|sosp|>` and `<|eosp|>`, then writing the transcript as text.
//!
//! ⚠ Every fact below was read from the authors' code (github.com/XiaomiMiMo/MiMo-V2.5-ASR @ 210ef16):
//!   - the resampler is torchaudio's sinc (Hann, width 6, rolloff 0.99), 16 -> 24 kHz;
//!   - the mel is a MAGNITUDE spectrogram (`power=1.0`), HTK mel, no filter normalisation, `log(max(x,
//!     1e-7))` — not the power spectrum Whisper-style frontends use;
//!   - the tokenizer's LayerNorms take torch's default eps 1e-5, its key projection has NO bias while
//!     query, value and output do, and its skip adds layer 3's output after layer 32;
//!   - audio longer than 30 s is tokenised in independent 30 s chunks (rope restarts in each);
//!   - the patch encoder's rope base is the LM's (640000), at positions 0..3 in every group.
//!
//! ⭐ PRECISION. Both checkpoints are the authors' files; their code rounds every weight to bfloat16 as
//! it loads (the tokenizer is stored bf16 already, its RVQ codebooks and the whole ASR model f32). Ferric
//! uses those rounded values — the weights the model is deployed with — at float32 arithmetic.
//! ⚠ RVQ codes are an argmin, so precision decides near-ties: the authors' own bf16 arithmetic flips 11%
//! of codes against float32 on the reference clip. Ferric is checked against float32.
//!
//! ⛔⛔ THE TOKENIZER'S ROPE FREQUENCIES ARE ROUNDED TOO. `.bfloat16()` converts every floating BUFFER,
//! and the rotary `inv_freq` table is one: in the authors' deployment it holds bf16 values, up to 0.37%
//! off the formula (the LM's and patch encoder's tables stay float32 — their loader never casts
//! buffers). Frequencies computed exactly put q 0.15 away from the authors' by position 292, an error
//! that grows linearly with position, and left the encoder 40-100x further from float64 than the
//! authors' own float32 run. Found by feeding one layer the same input on both sides, op by op.
use crate::qwen3::{Cache, Qwen3};
use ferric_core::Context;
use ferric_load::hf::{f32_to_bf16_bits, HfCheckpoint, HfOptions};
use ferric_load::SafeTensors;
use ferric_tensor::Tensor;
use std::sync::Arc;

/// Mechanism controls for the conformance gate, read once at load. Each removes one mechanism the
/// reference has; the gate requires each to fail first at the stage that mechanism lives in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Neg {
    /// Power spectrogram (|X|^2) instead of the magnitude the authors' `power=1.0` asks for.
    pub mel_power: bool,
    /// Zero padding instead of reflect padding at the STFT edges.
    pub pad_zero: bool,
    /// No long skip connection in the tokenizer encoder.
    pub no_skip: bool,
    /// The tokenizer's key projection given the query's bias.
    pub k_bias: bool,
    /// Causal attention inside each 4-frame patch.
    pub patch_causal: bool,
    /// Patch-encoder rope at the tokenizer's base (10000) instead of the LM's (640000).
    pub patch_rope_base: bool,
    /// Tokenizer rope at the EXACT float32 frequencies instead of the bf16-rounded table the authors'
    /// `.bfloat16()` leaves in its buffer.
    pub rope_exact: bool,
    /// The patch encoder's three bfloat16 islands removed (a float32 accumulator, an unrounded first
    /// norm, float32 rope tables) — what an f32 port would naturally write.
    pub patch_f32: bool,
}

impl Neg {
    fn from_env() -> Result<Neg, String> {
        let mut n = Neg::default();
        if let Ok(v) = std::env::var("FERRIC_ASR_NEG") {
            match v.as_str() {
                "mel_power" => n.mel_power = true,
                "pad_zero" => n.pad_zero = true,
                "no_skip" => n.no_skip = true,
                "k_bias" => n.k_bias = true,
                "patch_causal" => n.patch_causal = true,
                "patch_rope_base" => n.patch_rope_base = true,
                "rope_exact" => n.rope_exact = true,
                "patch_f32" => n.patch_f32 = true,
                other => return Err(format!("FERRIC_ASR_NEG={other} is not a control (mel_power|pad_zero|\
                                             no_skip|k_bias|patch_causal|patch_rope_base|rope_exact|patch_f32)")),
            }
        }
        Ok(n)
    }
}

// ================================================================================================
// Audio frontend
// ================================================================================================

/// `torchaudio.functional.resample` (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), in the
/// kernel's own float32 arithmetic, transcribed from torchaudio 2.6 `_get_sinc_resample_kernel` /
/// `_apply_sinc_resample_kernel`.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to { return x.to_vec(); }
    let g = gcd(from, to);
    let (orig, new) = (from / g, to / g);
    let width_f = 6.0f32;
    let base = orig.min(new) as f32 * 0.99;
    let width = ((6.0 * orig as f64) / base as f64).ceil() as usize;
    let taps = 2 * width + orig;
    // kernel[j][k], j = output phase, k = tap — each op in f32, in the order torch evaluates them
    let mut kernel = vec![0f32; new * taps];
    for j in 0..new {
        for k in 0..taps {
            let idx = (k as f32 - width as f32) / orig as f32;
            let mut t = (-(j as f32)) / new as f32 + idx;
            t *= base;
            t = t.clamp(-width_f, width_f);
            let w = (t * std::f32::consts::PI / width_f / 2.0).cos().powi(2);
            let t = t * std::f32::consts::PI;
            let s = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[j * taps + k] = s * (w * (base / orig as f32));
        }
    }
    // pad (width, width + orig), conv1d with stride orig, phases interleaved, trimmed to ceil(new*n/orig)
    let n = x.len();
    let mut padded = vec![0f32; width + n + width + orig];
    padded[width..width + n].copy_from_slice(x);
    let frames = (padded.len() - taps) / orig + 1;
    let target = (new * n).div_ceil(orig);
    let mut out = Vec::with_capacity(frames * new);
    for f in 0..frames {
        let src = &padded[f * orig..f * orig + taps];
        for j in 0..new {
            let kr = &kernel[j * taps..(j + 1) * taps];
            out.push(src.iter().zip(kr).map(|(a, b)| a * b).sum());
        }
    }
    out.truncate(target);
    out
}

fn gcd(a: usize, b: usize) -> usize { if b == 0 { a } else { gcd(b, a % b) } }

/// The tokenizer's mel frontend: torchaudio `MelSpectrogram(sample_rate, n_fft, hop, win=n_fft,
/// f_min, f_max=sr/2, n_mels, power=1.0, center=True, pad_mode="reflect")`, then `log(max(x, 1e-7))`.
#[derive(Debug, Clone)]
pub struct MelCfg {
    pub sr: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
}

/// `torchaudio.functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr, norm=None, mel_scale="htk")`,
/// `[n_freqs, n_mels]` row-major, in float32 as torch builds it.
pub fn mel_fbanks(c: &MelCfg) -> Vec<f32> {
    let n_freqs = c.n_fft / 2 + 1;
    let hz2mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    // torch.linspace in f32: start + i * step, step = (end - start) / (steps - 1)
    let lin = |a: f32, b: f32, n: usize| -> Vec<f32> {
        let step = (b - a) / (n - 1) as f32;
        (0..n).map(|i| if i < n / 2 { a + step * i as f32 } else { b - step * (n - 1 - i) as f32 }).collect()
    };
    let all_freqs = lin(0.0, (c.sr / 2) as f32, n_freqs);
    let m_pts = lin(hz2mel(c.fmin) as f32, hz2mel(c.fmax) as f32, c.n_mels + 2);
    let f_pts: Vec<f32> = m_pts.iter().map(|&m| 700.0 * (10f32.powf(m / 2595.0) - 1.0)).collect();
    let f_diff: Vec<f32> = f_pts.windows(2).map(|w| w[1] - w[0]).collect();
    let mut fb = vec![0f32; n_freqs * c.n_mels];
    for (i, &f) in all_freqs.iter().enumerate() {
        for m in 0..c.n_mels {
            let down = -(f_pts[m] - f) / f_diff[m];
            let up = (f_pts[m + 2] - f) / f_diff[m + 1];
            fb[i * c.n_mels + m] = down.min(up).max(0.0);
        }
    }
    fb
}

/// Log-mel `[frames, n_mels]` for one chunk. The DFT is a GPU matmul against cos/sin bases (960 is not
/// a power of two); the edge padding is torch's reflect, the window a periodic Hann.
pub fn log_mel(ctx: &Arc<Context>, x: &[f32], c: &MelCfg, fb: &Tensor, neg: &Neg) -> Tensor {
    let half = c.n_fft / 2;
    let n = x.len();
    let at = |i: isize| -> f32 {
        if (0..n as isize).contains(&i) { return x[i as usize]; }
        if neg.pad_zero { return 0.0; }
        // reflect, excluding the edge sample: x[-1] = x[1], x[n] = x[n-2]
        let r = if i < 0 { -i } else { 2 * (n as isize - 1) - i };
        x[r as usize]
    };
    let frames = 1 + n / c.hop;
    let window: Vec<f32> = (0..c.n_fft)
        .map(|k| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / c.n_fft as f64).cos()) as f32).collect();
    let mut fr = vec![0f32; frames * c.n_fft];
    for f in 0..frames {
        let s = (f * c.hop) as isize - half as isize;
        for k in 0..c.n_fft { fr[f * c.n_fft + k] = at(s + k as isize) * window[k]; }
    }
    let nf = half + 1;
    let mut basis = vec![0f32; c.n_fft * 2 * nf];
    for k in 0..c.n_fft {
        for b in 0..nf {
            // angle reduced mod n_fft in integers before the float, so large k*b stays exact
            let a = 2.0 * std::f64::consts::PI * ((k * b) % c.n_fft) as f64 / c.n_fft as f64;
            basis[k * 2 * nf + b] = a.cos() as f32;
            basis[k * 2 * nf + nf + b] = (-a.sin()) as f32;
        }
    }
    let spec = Tensor::from_vec(ctx, &fr, &[frames, c.n_fft]).matmul(&Tensor::from_vec(ctx, &basis, &[c.n_fft, 2 * nf]));
    let (re, im) = (spec.narrow(1, 0, nf).contiguous(), spec.narrow(1, nf, nf).contiguous());
    let pw = re.mul(&re).add(&im.mul(&im));
    let mag = if neg.mel_power { pw } else { pw.sqrt() };
    let mel = mag.matmul(fb);
    mel.maximum(&mel.scalar(1e-7)).log()
}

// ================================================================================================
// MiMo-Audio-Tokenizer, encoder side
// ================================================================================================

struct EncLayer {
    ln1_w: Tensor, ln1_b: Tensor,
    q_w: Tensor, q_b: Tensor, k_w: Tensor, v_w: Tensor, v_b: Tensor, o_w: Tensor, o_b: Tensor,
    ln2_w: Tensor, ln2_b: Tensor,
    fc1_w: Tensor, fc1_b: Tensor, fc2_w: Tensor, fc2_b: Tensor,
}

#[derive(Debug, Clone)]
pub struct TokCfg {
    pub d: usize,
    pub heads: usize,
    pub layers: usize,
    pub skip_after: usize,
    pub rope_theta: f32,
    pub pool: usize,
    pub mel: MelCfg,
}

pub struct AudioTokenizer {
    ctx: Arc<Context>,
    pub cfg: TokCfg,
    neg: Neg,
    fb: Tensor,
    conv1: Vec<Tensor>, conv1_b: Tensor,
    conv2: Vec<Tensor>, conv2_b: Tensor,
    layers: Vec<EncLayer>,
    ln_w: Tensor, ln_b: Tensor,
    down: Vec<Tensor>,
    dn_w: Tensor, dn_b: Tensor,
    /// Per codebook: `[1280, bins]` for the distance matmul, `[bins, 1280]` to subtract, and `|e|^2`.
    cb_t: Vec<Tensor>, cb: Vec<Tensor>, cb_sq: Vec<Vec<f32>>,
}

/// Where a stage's tensor goes when a caller asked to see it. `None` in production: a tap holds its
/// tensor's GPU buffer until the caller drops the list, so recording stages nobody reads grows memory
/// with every 30 s chunk.
pub type Taps<'a> = Option<&'a mut Vec<(String, Tensor)>>;

fn tap(taps: &mut Taps<'_>, name: &str, t: &Tensor) {
    if let Some(v) = taps.as_deref_mut() { v.push((name.to_string(), t.clone())); }
}

/// Bidirectional attention over one chunk, in query-row blocks small enough that the `[heads, rows,
/// keys]` score matrix stays under 64 MiB.
///
/// ⚠ A 30 s chunk is 1,500 frames, and all 20 heads' scores at once are 180 MB — past the 128 MiB
/// storage binding the WebGPU baseline guarantees, so the unblocked form ran on Metal and failed on
/// the adapters Ferric treats as the floor. Each query row is independent, so the blocks change which
/// dispatch computes a row, not what it computes.
fn attention_blocked(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Tensor {
    const CAP: usize = 64 << 20;
    let t = q.shape[0];
    if heads * t * t * 4 <= CAP { return ferric_tensor::nn::bidirectional_attention(q, k, v, heads, heads); }
    let rows = (CAP / (heads * t * 4)).max(1);
    let mut out: Option<Tensor> = None;
    let mut s = 0;
    while s < t {
        let n = rows.min(t - s);
        let o = ferric_tensor::nn::full_attention_kv(&q.narrow(0, s, n).contiguous(), k, v, heads, heads);
        out = Some(match out { None => o, Some(acc) => acc.cat(&o, 0) });
        s += n;
    }
    out.expect("a chunk with no frames")
}

fn round_bf16(v: &mut [f32]) { for x in v { *x = f32::from_bits((f32_to_bf16_bits(*x) as u32) << 16); } }

fn get_rounded(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let t = st.get(name)?;
    let mut d = t.data;
    round_bf16(&mut d);   // the authors' `.bfloat16()`; the identity on a BF16-stored tensor
    Ok((d, t.shape))
}

fn vec_t(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let (d, s) = get_rounded(st, name)?;
    Ok(Tensor::from_vec(ctx, &d, &s))
}

/// `[out, in]` -> the `[in, out]` matrix `matmul` wants.
fn lin_t(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let (d, s) = get_rounded(st, name)?;
    if s.len() != 2 { return Err(format!("{name}: expected 2-D, got {s:?}")); }
    Ok(Tensor::from_vec(ctx, &d, &[s[0], s[1]]).transpose(0, 1).contiguous())
}

/// A Conv1d weight `[out, in, k]` as `k` matrices `[in, out]`, one per tap.
fn conv_taps(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Vec<Tensor>, String> {
    let (d, s) = get_rounded(st, name)?;
    if s.len() != 3 { return Err(format!("{name}: expected [out, in, k], got {s:?}")); }
    let (o, i, k) = (s[0], s[1], s[2]);
    Ok((0..k).map(|t| {
        let mut m = vec![0f32; i * o];
        for oo in 0..o { for ii in 0..i { m[ii * o + oo] = d[(oo * i + ii) * k + t]; } }
        Tensor::from_vec(ctx, &m, &[i, o])
    }).collect())
}

/// Conv1d as a sum of shifted matmuls: `y[t] = b + sum_k x[t*stride + k - pad] @ W_k`, out-of-range rows
/// zero. The shift is a row gather from `x` with one zero row appended.
fn conv1d(x: &Tensor, taps: &[Tensor], bias: Option<&Tensor>, stride: usize, pad: usize, out_len: usize) -> Tensor {
    let (n, c) = (x.shape[0], x.shape[1]);
    let xz = x.cat(&Tensor::zeros(&x.ctx_arc(), &[1, c]), 0);
    let mut y: Option<Tensor> = None;
    for (k, w) in taps.iter().enumerate() {
        let idx: Vec<u32> = (0..out_len).map(|t| {
            let s = (t * stride + k) as isize - pad as isize;
            if (0..n as isize).contains(&s) { s as u32 } else { n as u32 }
        }).collect();
        let term = xz.gather_rows(&idx).matmul(w);
        y = Some(match y { None => term, Some(acc) => acc.add(&term) });
    }
    let y = y.expect("a conv with no taps");
    match bias { Some(b) => y.add(b), None => y }
}

impl AudioTokenizer {
    pub fn load(ctx: &Arc<Context>, dir: &str) -> Result<AudioTokenizer, String> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| format!("tokenizer config.json: {e}"))?).map_err(|e| e.to_string())?;
        let u = |k: &str| j[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("tokenizer config: missing {k}"));
        for (k, want) in [("activation_function", "gelu"), ("ln_type", "LayerNorm"), ("position_embedding_type", "rope")] {
            if j[k].as_str() != Some(want) { return Err(format!("tokenizer {k} = {}, this encoder implements {want}", j[k])); }
        }
        if j["encoder_causal"].as_bool() != Some(false) || j["encoder_attn_window_size"] != serde_json::json!([-1, -1]) {
            return Err("tokenizer encoder is causal or windowed; this one is full attention".into());
        }
        if u("kernel_size")? != 3 || u("stride_size")? != 2 {
            return Err("tokenizer conv geometry is not kernel 3 / stride 2".into());
        }
        let sr = u("sampling_rate")?;
        let mel = MelCfg {
            sr, n_fft: u("nfft")?, hop: u("hop_length")?, n_mels: u("n_mels")?,
            fmin: j["fmin"].as_f64().unwrap_or(0.0),
            fmax: j["fmax"].as_f64().unwrap_or((sr / 2) as f64),
        };
        if u("window_size")? != mel.n_fft { return Err("window_size != nfft: the window would be zero-padded".into()); }
        let cfg = TokCfg {
            d: u("d_model")?, heads: u("encoder_attention_heads")?, layers: u("encoder_layers")?,
            skip_after: u("encoder_skip_layer_id")?, rope_theta: j["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            pool: u("avg_pooler")?, mel,
        };
        let st = SafeTensors::open(format!("{dir}/model.safetensors"))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layers.{i}");
            if st.info(&format!("{p}.self_attn.k_proj.bias")).is_some() {
                return Err(format!("{p}.self_attn.k_proj has a bias; this encoder's key projection has none"));
            }
            layers.push(EncLayer {
                ln1_w: vec_t(ctx, &st, &format!("{p}.self_attn_layer_norm.weight"))?,
                ln1_b: vec_t(ctx, &st, &format!("{p}.self_attn_layer_norm.bias"))?,
                q_w: lin_t(ctx, &st, &format!("{p}.self_attn.q_proj.weight"))?,
                q_b: vec_t(ctx, &st, &format!("{p}.self_attn.q_proj.bias"))?,
                k_w: lin_t(ctx, &st, &format!("{p}.self_attn.k_proj.weight"))?,
                v_w: lin_t(ctx, &st, &format!("{p}.self_attn.v_proj.weight"))?,
                v_b: vec_t(ctx, &st, &format!("{p}.self_attn.v_proj.bias"))?,
                o_w: lin_t(ctx, &st, &format!("{p}.self_attn.out_proj.weight"))?,
                o_b: vec_t(ctx, &st, &format!("{p}.self_attn.out_proj.bias"))?,
                ln2_w: vec_t(ctx, &st, &format!("{p}.final_layer_norm.weight"))?,
                ln2_b: vec_t(ctx, &st, &format!("{p}.final_layer_norm.bias"))?,
                fc1_w: lin_t(ctx, &st, &format!("{p}.fc1.weight"))?,
                fc1_b: vec_t(ctx, &st, &format!("{p}.fc1.bias"))?,
                fc2_w: lin_t(ctx, &st, &format!("{p}.fc2.weight"))?,
                fc2_b: vec_t(ctx, &st, &format!("{p}.fc2.bias"))?,
            });
        }
        let (mut cb_t, mut cb, mut cb_sq) = (Vec::new(), Vec::new(), Vec::new());
        for q in 0..8 {
            let (e, s) = get_rounded(&st, &format!("encoder.quantizer.vq.layers.{q}._codebook.embed"))?;
            let (bins, d) = (s[0], s[1]);
            // |e|^2 in f32, summed in index order like `embed.pow(2).sum(0)`
            cb_sq.push((0..bins).map(|b| e[b * d..(b + 1) * d].iter().map(|v| v * v).sum()).collect());
            let t = Tensor::from_vec(ctx, &e, &[bins, d]);
            cb_t.push(t.transpose(0, 1).contiguous());
            cb.push(t);
        }
        let fb = Tensor::from_vec(ctx, &mel_fbanks(&cfg.mel), &[cfg.mel.n_fft / 2 + 1, cfg.mel.n_mels]);
        Ok(AudioTokenizer {
            ctx: ctx.clone(), neg: Neg::from_env()?, fb,
            conv1: conv_taps(ctx, &st, "encoder.conv1.weight")?, conv1_b: vec_t(ctx, &st, "encoder.conv1.bias")?,
            conv2: conv_taps(ctx, &st, "encoder.conv2.weight")?, conv2_b: vec_t(ctx, &st, "encoder.conv2.bias")?,
            layers,
            ln_w: vec_t(ctx, &st, "encoder.layer_norm.weight")?, ln_b: vec_t(ctx, &st, "encoder.layer_norm.bias")?,
            down: conv_taps(ctx, &st, "encoder.down_sample_layer.0.weight")?,
            dn_w: vec_t(ctx, &st, "encoder.down_sample_norm.weight")?, dn_b: vec_t(ctx, &st, "encoder.down_sample_norm.bias")?,
            cb_t, cb, cb_sq, cfg,
        })
    }

    /// One chunk (<= 30 s at 24 kHz) to 8 code channels `[8][frames]`. `taps` receives the stages the
    /// fixture records: `mel`, `conv_out`, `enc_layer{i}`, `enc_norm`, `pooled`.
    pub async fn encode_chunk(&self, x: &[f32], mut taps: Taps<'_>) -> Vec<Vec<u32>> {
        let c = &self.cfg;
        let mel = log_mel(&self.ctx, x, &c.mel, &self.fb, &self.neg);
        tap(&mut taps, "mel", &mel);
        let f = mel.shape[0];
        let h = conv1d(&mel, &self.conv1, Some(&self.conv1_b), 1, 1, f).gelu();
        let t2 = (f - 1) / 2 + 1;
        let mut h = conv1d(&h, &self.conv2, Some(&self.conv2_b), 2, 1, t2).gelu();
        tap(&mut taps, "conv_out", &h);
        let (heads, dh) = (c.heads, c.d / c.heads);
        let (cos, sin) = self.rope_tables(t2);
        let mut skip: Option<Tensor> = None;
        for (i, l) in self.layers.iter().enumerate() {
            let a = h.layernorm(&l.ln1_w, &l.ln1_b, 1e-5);
            let q = a.matmul(&l.q_w).add(&l.q_b).apply_rope_costable(&cos, &sin, heads, dh);
            let k = a.matmul(&l.k_w);
            let k = if self.neg.k_bias { k.add(&l.q_b) } else { k }.apply_rope_costable(&cos, &sin, heads, dh);
            let v = a.matmul(&l.v_w).add(&l.v_b);
            let att = attention_blocked(&q, &k, &v, heads);
            h = h.add(&att.matmul(&l.o_w).add(&l.o_b));
            let a = h.layernorm(&l.ln2_w, &l.ln2_b, 1e-5);
            h = h.add(&a.matmul(&l.fc1_w).add(&l.fc1_b).gelu().matmul(&l.fc2_w).add(&l.fc2_b));
            if i + 1 == c.skip_after { skip = Some(h.clone()); }
            if [0, 2, 15, 31].contains(&i) { tap(&mut taps, &format!("enc_layer{i}"), &h); }
        }
        if let (Some(s), false) = (skip, self.neg.no_skip) { h = h.add(&s); }
        let h = h.layernorm(&self.ln_w, &self.ln_b, 1e-5);
        tap(&mut taps, "enc_norm", &h);
        // 2x pooling: a kernel-2 stride-2 conv (no bias) over the sequence zero-padded to even length
        let t4 = t2.div_ceil(c.pool);
        let p = conv1d(&h, &self.down, None, c.pool, 0, t4).gelu().layernorm(&self.dn_w, &self.dn_b, 1e-5);
        tap(&mut taps, "pooled", &p);
        self.rvq(&p, t4).await
    }

    /// cos/sin `[t, head_dim]` for positions `0..t`, from the authors' DEPLOYED frequency table: the
    /// formula in float32 (`1 / base^(arange(0, d, 2) / d)`), then rounded to bf16 as their
    /// `.bfloat16()` rounds the buffer; `freqs = inv * pos` in float32, both halves equal (NEOX).
    fn rope_tables(&self, t: usize) -> (Tensor, Tensor) {
        let dh = self.cfg.d / self.cfg.heads;
        let inv: Vec<f32> = (0..dh / 2).map(|c| {
            let e = (2 * c) as f32 / dh as f32;
            let v = 1.0f32 / self.cfg.rope_theta.powf(e);
            if self.neg.rope_exact { v } else { f32::from_bits((f32_to_bf16_bits(v) as u32) << 16) }
        }).collect();
        let (mut cs, mut sn) = (vec![0f32; t * dh], vec![0f32; t * dh]);
        for p in 0..t {
            for c in 0..dh {
                let a = inv[c % (dh / 2)] * p as f32;
                cs[p * dh + c] = a.cos();
                sn[p * dh + c] = a.sin();
            }
        }
        (Tensor::from_vec(&self.ctx, &cs, &[t, dh]), Tensor::from_vec(&self.ctx, &sn, &[t, dh]))
    }

    /// Residual VQ, first 8 codebooks: `argmin_b |r|^2 - 2 r.e_b + |e_b|^2`, first index on a tie,
    /// evaluated in that order in f32 as the reference's `quantize` writes it; then `r -= e_code`.
    async fn rvq(&self, x: &Tensor, t: usize) -> Vec<Vec<u32>> {
        let mut r = x.clone();
        let mut codes = Vec::with_capacity(8);
        for q in 0..8 {
            let xe = r.matmul(&self.cb_t[q]).to_vec().await;
            let rv = r.to_vec().await;
            let d = self.cfg.d;
            let bins = self.cb_sq[q].len();
            let mut cq = Vec::with_capacity(t);
            for i in 0..t {
                let xx: f32 = rv[i * d..(i + 1) * d].iter().map(|v| v * v).sum();
                let mut best = (0u32, f32::INFINITY);
                for b in 0..bins {
                    let dist = xx - 2.0 * xe[i * bins + b] + self.cb_sq[q][b];
                    if dist < best.1 { best = (b as u32, dist); }
                }
                cq.push(best.0);
            }
            r = r.sub(&self.cb[q].gather_rows(&cq));
            codes.push(cq);
        }
        codes
    }

    /// The authors' `preprocess_input` on 24 kHz audio: 30 s chunks (a trailing piece shorter than
    /// n_fft merged into the chunk before it, a whole input shorter than n_fft zero-padded), codes
    /// concatenated, then padded to a multiple of `group` by repeating the last frame.
    pub async fn encode(&self, x: &[f32], group: usize, mut taps: Taps<'_>) -> Vec<Vec<u32>> {
        let chunk = 30 * self.cfg.mel.sr;
        let nfft = self.cfg.mel.n_fft;
        let mut codes: Vec<Vec<u32>> = vec![Vec::new(); 8];
        let mut start = 0;
        while start < x.len() {
            let mut end = (start + chunk).min(x.len());
            if x.len() - end > 0 && x.len() - end < nfft { end = x.len(); }
            let mut c = x[start..end].to_vec();
            if c.len() < nfft { c.resize(nfft, 0.0); }
            let cc = self.encode_chunk(&c, taps.as_deref_mut()).await;
            for (dst, src) in codes.iter_mut().zip(cc) { dst.extend(src); }
            start = end;
        }
        let n = codes[0].len();
        if !n.is_multiple_of(group) {
            for ch in &mut codes { let last = *ch.last().unwrap(); ch.resize(n.div_ceil(group) * group, last); }
        }
        codes
    }
}

// ================================================================================================
// Patch encoder: 8 code embeddings -> 6 layers inside each 4-frame group -> one LM row per group
// ================================================================================================

struct PatchLayer {
    ln1: Tensor, q_w: Tensor, q_b: Tensor, k_w: Tensor, k_b: Tensor, v_w: Tensor, v_b: Tensor, o_w: Tensor,
    ln2: Tensor, gate: Tensor, up: Tensor, down: Tensor,
}

pub struct PatchEncoder {
    ctx: Arc<Context>,
    neg: Neg,
    tables: Vec<Vec<f32>>,
    empty: Vec<u32>,
    dim: usize,
    heads: usize,
    group: usize,
    eps: f32,
    rope_theta: f32,
    layers: Vec<PatchLayer>,
    norm: Tensor,
    downcast: Tensor,
}

impl PatchEncoder {
    pub fn load(ctx: &Arc<Context>, dir: &str) -> Result<PatchEncoder, String> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| format!("config.json: {e}"))?).map_err(|e| e.to_string())?;
        let u = |k: &str| j[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("config: missing {k}"));
        if j["input_full_attention"].as_bool() != Some(true) {
            return Err("input_full_attention is not true; this patch encoder is bidirectional".into());
        }
        let parse = |k: &str| -> Result<Vec<u32>, String> {
            j[k].as_str().ok_or(format!("config: {k}"))?.split('-').map(|v| v.parse().map_err(|e| format!("{k}: {e}"))).collect()
        };
        let empty = parse("speech_zeroemb_idx")?;
        let channels = u("audio_channels")?;
        let (dim, heads) = (u("input_local_dim")?, u("local_attn_heads")?);
        let st = SafeTensors::open(dir)?;
        let mut tables = Vec::new();
        for c in 0..channels {
            let (d, s) = get_rounded(&st, &format!("speech_embeddings.{c}.weight"))?;
            if s[1] != dim { return Err(format!("speech_embeddings.{c} is {s:?}, not [_, {dim}]")); }
            tables.push(d);
        }
        let mut layers = Vec::new();
        for i in 0..u("input_local_layers")? {
            let p = format!("input_local_transformer.layers.{i}");
            layers.push(PatchLayer {
                ln1: vec_t(ctx, &st, &format!("{p}.input_layernorm.weight"))?,
                q_w: lin_t(ctx, &st, &format!("{p}.self_attn.q_proj.weight"))?,
                q_b: vec_t(ctx, &st, &format!("{p}.self_attn.q_proj.bias"))?,
                k_w: lin_t(ctx, &st, &format!("{p}.self_attn.k_proj.weight"))?,
                k_b: vec_t(ctx, &st, &format!("{p}.self_attn.k_proj.bias"))?,
                v_w: lin_t(ctx, &st, &format!("{p}.self_attn.v_proj.weight"))?,
                v_b: vec_t(ctx, &st, &format!("{p}.self_attn.v_proj.bias"))?,
                o_w: lin_t(ctx, &st, &format!("{p}.self_attn.o_proj.weight"))?,
                ln2: vec_t(ctx, &st, &format!("{p}.post_attention_layernorm.weight"))?,
                gate: lin_t(ctx, &st, &format!("{p}.mlp.gate_proj.weight"))?,
                up: lin_t(ctx, &st, &format!("{p}.mlp.up_proj.weight"))?,
                down: lin_t(ctx, &st, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        Ok(PatchEncoder {
            ctx: ctx.clone(), neg: Neg::from_env()?, tables, empty, dim, heads, group: u("group_size")?,
            eps: j["rms_norm_eps"].as_f64().ok_or("config: rms_norm_eps")? as f32,
            rope_theta: j["rope_theta"].as_f64().ok_or("config: rope_theta")? as f32,
            layers,
            norm: vec_t(ctx, &st, "input_local_transformer.norm.weight")?,
            downcast: lin_t(ctx, &st, "speech_group_downcast.weight")?,
        })
    }

    /// Codes `[channels][frames]` (frames a multiple of the group) to LM rows `[groups, hidden]`.
    /// `taps` receives `patch_encoder` (`[frames, dim]`, the transformer's output before the fold).
    ///
    /// ⛔⛔ THREE bfloat16 ISLANDS, in the authors' code at ANY weight dtype. `_prepare_input_embeds` sums
    /// the 8 code embeddings into a buffer it creates as `torch.bfloat16`, so every `+=` rounds; that bf16
    /// tensor then enters the patch transformer, where (1) layer 0's RMSNorm returns
    /// `weight * normed.to(input_dtype)` — the normalised row rounded to bf16 — and (2) the rotary
    /// embedding returns `cos.to(x.dtype)`, so every layer rotates by bf16-rounded cos/sin. A float32
    /// port does none of this and differs by ~1e-2 relative. An earlier version of the reference harness
    /// "corrected" the accumulator to float32, claiming the authors' code could not run otherwise; it
    /// runs, and that correction had silently removed all three islands from the reference too. Found by
    /// an adversarial review before commit.
    pub fn encode(&self, codes: &[Vec<u32>], mut taps: Taps<'_>) -> Result<Tensor, String> {
        let (d, g) = (self.dim, self.group);
        let frames = codes[0].len();
        if !frames.is_multiple_of(g) { return Err(format!("{frames} frames is not a multiple of the group {g}")); }
        let islands = !self.neg.patch_f32;
        let rb = |v: f32| if islands { f32::from_bits((f32_to_bf16_bits(v) as u32) << 16) } else { v };
        // Sum of the channel embeddings per frame, in channel order, into a bf16 buffer: each `+=` is
        // computed in f32 and rounded back (an in-place add into a bf16 tensor). An empty id adds 0.
        let mut x = vec![0f32; frames * d];
        for (c, ch) in codes.iter().enumerate() {
            for (f, &code) in ch.iter().enumerate() {
                if code == self.empty[c] { continue; }
                let row = &self.tables[c][code as usize * d..(code as usize + 1) * d];
                for (o, v) in x[f * d..(f + 1) * d].iter_mut().zip(row) { *o = rb(*o + v); }
            }
        }
        // Layer 0's input norm on the bf16 rows: `x * rsqrt(mean(x^2) + eps)` in f32, rounded to bf16,
        // THEN times the weight (in f32) — done here on the host, where the rows already are.
        //
        // ⚠ The mean is accumulated in f64 and rounded once. A rounding step right after a sum makes the
        // SUMMATION ORDER visible: a sequential f32 sum differs from torch's vectorised one by an ulp in
        // every row, and on the reference clip that flipped 31 bf16 roundings in 5 rows — 100x the
        // authors' own f32-vs-f64 distance at the patch encoder. The correctly rounded mean flips none
        // there (torch's pairwise sum is within an ulp of it).
        let norm0: Vec<f32> = x.chunks_exact(d).flat_map(|r| {
            let var = (r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64) as f32;
            let inv = 1.0 / (var + self.eps).sqrt();
            r.iter().map(move |v| v * inv).collect::<Vec<_>>()
        }).map(rb).collect();
        let norm0 = Tensor::from_vec(&self.ctx, &norm0, &[frames, d]);
        let mut h = Tensor::from_vec(&self.ctx, &x, &[frames, d]);
        let (nh, hd) = (self.heads, d / self.heads);
        let groups = frames / g;
        let base: f32 = if self.neg.patch_rope_base { 10_000.0 } else { self.rope_theta };
        // cos/sin per position 0..g-1, as the rotary module builds them in f32 — then `.to(bf16)`.
        let inv: Vec<f32> = (0..hd / 2).map(|c| 1.0f32 / base.powf((2 * c) as f32 / hd as f32)).collect();
        let (mut cs, mut sn) = (vec![0f32; frames * hd], vec![0f32; frames * hd]);
        for f in 0..frames {
            for c in 0..hd {
                let a = inv[c % (hd / 2)] * (f % g) as f32;
                cs[f * hd + c] = rb(a.cos());
                sn[f * hd + c] = rb(a.sin());
            }
        }
        let (cos, sin) = (Tensor::from_vec(&self.ctx, &cs, &[frames, hd]), Tensor::from_vec(&self.ctx, &sn, &[frames, hd]));
        // Attention inside each group only: [groups*heads, g, hd] batched, never across groups.
        let mask = if self.neg.patch_causal {
            let m: Vec<f32> = (0..g * g).map(|i| if i % g > i / g { f32::NEG_INFINITY } else { 0.0 }).collect();
            Some(Tensor::from_vec(&self.ctx, &m, &[1, g, g]).broadcast_to(&[groups * nh, g, g]))
        } else { None };
        let per_group = |t: &Tensor| t.reshape(&[groups, g, nh, hd]).permute(&[0, 2, 1, 3]).contiguous()
            .reshape(&[groups * nh, g, hd]);
        for (il, l) in self.layers.iter().enumerate() {
            let a = if il == 0 { norm0.mul(&l.ln1) } else { h.rmsnorm(&l.ln1, self.eps) };
            let q = a.matmul(&l.q_w).add(&l.q_b).apply_rope_costable(&cos, &sin, nh, hd);
            let k = a.matmul(&l.k_w).add(&l.k_b).apply_rope_costable(&cos, &sin, nh, hd);
            let v = a.matmul(&l.v_w).add(&l.v_b);
            let (q, k, v) = (per_group(&q), per_group(&k), per_group(&v));
            let s = q.matmul(&k.transpose(2, 1)).mul(&q.scalar(1.0 / (hd as f32).sqrt()));
            let s = match &mask { Some(m) => s.add(m), None => s };
            let o = s.softmax(2).matmul(&v).reshape(&[groups, nh, g, hd]).permute(&[0, 2, 1, 3]).contiguous()
                .reshape(&[frames, d]);
            h = h.add(&o.matmul(&l.o_w));
            let a = h.rmsnorm(&l.ln2, self.eps);
            h = h.add(&a.matmul(&l.gate).silu().mul(&a.matmul(&l.up)).matmul(&l.down));
        }
        let h = h.rmsnorm(&self.norm, self.eps);
        tap(&mut taps, "patch_encoder", &h);
        Ok(h.reshape(&[groups, g * d]).matmul(&self.downcast))
    }
}

// ================================================================================================
// The whole recogniser
// ================================================================================================

/// Special token ids the prompt uses — read from the tokenizer's `added_tokens.json`, not assumed.
#[derive(Debug, Clone)]
pub struct Specials { pub im_start: u32, pub im_end: u32, pub endoftext: u32, pub sosp: u32, pub eosp: u32, pub empty: u32 }

impl Specials {
    pub fn load(dir: &str) -> Result<Specials, String> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/added_tokens.json"))
            .map_err(|e| format!("added_tokens.json: {e}"))?).map_err(|e| e.to_string())?;
        let id = |k: &str| j[k].as_u64().map(|x| x as u32).ok_or_else(|| format!("added_tokens.json: no {k}"));
        Ok(Specials { im_start: id("<|im_start|>")?, im_end: id("<|im_end|>")?, endoftext: id("<|endoftext|>")?,
                      sosp: id("<|sosp|>")?, eosp: id("<|eosp|>")?, empty: id("<|empty|>")? })
    }
}

pub struct MimoAsr {
    pub tokenizer: AudioTokenizer,
    pub patch: PatchEncoder,
    pub lm: Qwen3,
    pub sp: Specials,
    pub group: usize,
}

impl MimoAsr {
    pub fn load(ctx: &Arc<Context>, asr_dir: &str, tok_dir: &str) -> Result<MimoAsr, String> {
        let hf = HfCheckpoint::open_with(asr_dir, HfOptions { narrow_f32_to_bf16: true })?;
        let lm = Qwen3::load(ctx, &hf)?;
        drop(hf);
        let patch = PatchEncoder::load(ctx, asr_dir)?;
        Ok(MimoAsr { group: patch.group, tokenizer: AudioTokenizer::load(ctx, tok_dir)?, patch, lm,
                     sp: Specials::load(asr_dir)? })
    }

    /// The LM input rows for a prompt: text embeddings, with the speech run (`<|empty|>` positions,
    /// one contiguous block) replaced by the patch encoder's rows.
    pub fn prompt_embeds(&self, text_ids: &[u32], speech: &Tensor) -> Result<Tensor, String> {
        let start = text_ids.iter().position(|&t| t == self.sp.empty).ok_or("no speech in the prompt")?;
        let n = text_ids.iter().filter(|&&t| t == self.sp.empty).count();
        if text_ids[start..start + n].iter().any(|&t| t != self.sp.empty) {
            return Err("the speech positions are not one contiguous run".into());
        }
        if speech.shape[0] != n {
            return Err(format!("{n} speech positions for {} patch rows", speech.shape[0]));
        }
        Ok(self.lm.splice_image_embeds(&self.lm.embed_tokens(text_ids), start, speech))
    }

    /// The prompt the authors' `get_asr_sft_prompt` builds, as TEXT ids per LM position, around `groups`
    /// speech positions: `encode` must be the Qwen2 BPE (special tokens are placed by id here).
    pub fn prompt_ids(&self, groups: usize, template: &str, tag: &str, encode: &dyn Fn(&str) -> Vec<u32>) -> Vec<u32> {
        let sp = &self.sp;
        let mut ids = vec![sp.im_start];
        ids.extend(encode("user\n"));
        ids.push(sp.sosp);
        ids.extend(std::iter::repeat_n(sp.empty, groups));
        ids.push(sp.eosp);
        ids.extend(encode(template));
        ids.push(sp.im_end);
        ids.extend(encode("\n"));
        ids.push(sp.im_start);
        ids.extend(encode("assistant\n"));
        ids.extend(encode(&format!("<think>\n\n</think>\n{tag}")));
        ids
    }

    /// Greedy transcription from 16-bit-derived PCM at `rate`: returns the generated ids and the stop
    /// token that ended them (`None` if `max_new` did). The authors decode text greedily
    /// (`MiMoSampler(do_sample=False)`) and stop ONLY on `<|im_end|>`: their `asr_sft` passes
    /// `[tokenizer.eos_token_id, im_end_idx]`, and this tokenizer's eos IS `<|im_end|>`. `<|endoftext|>`
    /// does not stop them, so it does not stop this either (it is decoded like any other token).
    ///
    /// ⛔ `<|empty|>` as a TEXT token means the model has started to SPEAK: the authors' loop then
    /// samples speech codes with a local transformer this port does not run. Continuing with the
    /// token's text embedding would silently diverge from them, so it is an error instead.
    pub async fn transcribe_ids(&self, pcm: &[f32], rate: usize, template: &str, tag: &str,
                                encode: &dyn Fn(&str) -> Vec<u32>, max_new: usize)
                                -> Result<(Vec<u32>, Option<u32>), String> {
        let x = resample(pcm, rate, self.tokenizer.cfg.mel.sr);
        let codes = self.tokenizer.encode(&x, self.group, None).await;
        let rows = self.patch.encode(&codes, None)?;
        let ids = self.prompt_ids(rows.shape[0], template, tag, encode);
        let x = self.prompt_embeds(&ids, &rows)?;
        let mut cache = Cache::new(&self.lm.cfg);
        let last = self.lm.forward_embeds_last(&x, &mut cache).to_vec().await;
        let v = last.len();
        let mut next = crate::parakeet::argmax_first(&last) as u32;
        let mut out = Vec::new();
        while out.len() < max_new && next != self.sp.im_end {
            if next == self.sp.empty {
                return Err(format!("the model emitted <|empty|> after {} tokens: it has switched to generating \
                                    speech, which this port does not run", out.len()));
            }
            out.push(next);
            let lg = self.lm.forward_cached(&[next], &mut cache).to_vec().await;
            next = crate::parakeet::argmax_first(&lg[lg.len() - v..]) as u32;
        }
        let stop = (next == self.sp.im_end).then_some(next);
        Ok((out, stop))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torchaudio's 16 -> 24 kHz kernel has width ceil(6*2/1.98) = 7, so 16 taps per phase and 3 phases;
    /// a constant signal away from the edges must come out constant to within the filter's passband
    /// ripple at DC, and the length is ceil(3n/2).
    #[test]
    fn resample_16k_to_24k_keeps_dc_and_length() {
        let x = vec![0.25f32; 1001];
        let y = resample(&x, 16_000, 24_000);
        assert_eq!(y.len(), 1502);
        for &v in &y[40..y.len() - 40] { assert!((v - 0.25).abs() < 2e-3, "DC drifted to {v}"); }
        // ⚠ and it is not the identity-by-repetition a naive 2:3 would produce
        let s = resample(&(0..64).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>(), 16_000, 24_000);
        assert!((s[3] - (2.0f32 * 0.3).sin()).abs() < 0.05, "sample 3 of 24 kHz is t = 2 at 16 kHz");
    }

    /// The HTK filterbank: 128 triangles over 481 bins, each peaking at 1 (norm=None), the first rising
    /// from bin 0.
    #[test]
    fn htk_filterbank_shape() {
        let c = MelCfg { sr: 24_000, n_fft: 960, hop: 240, n_mels: 128, fmin: 0.0, fmax: 12_000.0 };
        let fb = mel_fbanks(&c);
        assert_eq!(fb.len(), 481 * 128);
        for m in 0..128 {
            let peak = (0..481).map(|i| fb[i * 128 + m]).fold(0f32, f32::max);
            assert!(peak > 0.0 && peak <= 1.0, "filter {m} peaks at {peak}");
        }
        assert_eq!(fb[0], 0.0, "bin 0 sits exactly on the first filter's left edge");
    }
}
