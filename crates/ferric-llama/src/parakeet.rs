//! **NVIDIA Parakeet / Nemotron-ASR** (`general.architecture = "parakeet"`) — a Conformer encoder
//! with an RNN-T decoder. Ferric's first speech model.
//!
//! Two of the top-40 most-downloaded GGUF repos share this architecture
//! (`parakeet-unified-en-0.6b`, `nemotron-3.5-asr-streaming-0.6b`), and nothing in this runtime
//! could load either: not a missing registry row but a missing MODALITY. Text models here are
//! decoder-only stacks over token ids; this takes a waveform.
//!
//! ```text
//! audio → mel spectrogram (n_fft 512, hop 160, 128 mels, hann, pre-emphasis 0.97)
//!       → pre_encode: 2-D depthwise/pointwise convs, subsampling factor 8 → [T/8, 1024]
//!       → 24 × Conformer block:
//!             ½·FF1 → rel-pos MHSA (pos_bias_u/v) → conv module → ½·FF2 → norm_out
//!       → RNN-T: 2-layer LSTM predictor + joint network → 1025-token vocab (1024 = blank)
//! ```
//!
//! ## What is genuinely new here, and what already existed
//!
//! `conv2d` ([kh,kw,c,o]) and `layernorm(weight, bias, eps)` were already in `ferric-tensor` — the
//! pre-encode stack and every Conformer norm use them unchanged. LayerNorm carries a BIAS, unlike
//! the RMSNorm every text model here uses, and the conv module carries a BATCH norm whose
//! `running_mean`/`running_var` make it an affine op at inference.
//!
//! Still missing when this landed: the mel frontend, relative-position attention, the LSTM, and
//! RNN-T decoding. This file is the loader; those follow.
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::Tensor;
use std::sync::Arc;

/// Everything the graph needs, read from `stt.*` metadata rather than inferred from tensor shapes.
pub struct Cfg {
    // ---- frontend ----
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub num_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    pub pre_emphasis: f32,
    // ---- encoder ----
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    /// `xscaling`: multiply the pre-encode output by sqrt(d_model). Off for some variants.
    pub xscaling: bool,
    // ---- decoder ----
    pub pred_hidden: usize,
    pub pred_layers: usize,
    pub joint_hidden: usize,
    pub vocab: usize,
    /// The RNN-T blank id — `vocab - 1` on every published parakeet, but read, not assumed.
    pub blank_id: u32,
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let u = |k: &str| -> Result<usize, String> {
            match md.get(k) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing {k}")) }
        };
        let f = |k: &str, d: f32| -> f32 {
            match md.get(k) { Some(Meta::F(v)) => *v as f32, _ => d }
        };
        let b = |k: &str, d: bool| -> bool {
            match md.get(k) { Some(Meta::Bool(v)) => *v, _ => d }
        };
        // ⛔ REFUSE THE VARIANTS THIS FORWARD PASS DOES NOT IMPLEMENT.
        //
        // `general.architecture = "parakeet"` names a FAMILY, not a configuration.
        // nemotron-3.5-asr-streaming-0.6b shares the arch string and differs in six ways at once —
        // chunked-limited attention (left 56 / right 13, not full context), a CAUSAL depthwise conv
        // (conv_context_right = 0, not symmetric), LayerNorm in the conv module instead of
        // BatchNorm, no input scaling, no per-feature normalisation, and a prompt-conditioned
        // multilingual head (13088 tokens, prompt.field = target_lang).
        //
        // Loading it anyway would produce a transcript. The wrong one. `arch::resolve` guards
        // against near-miss ARCHITECTURES; nothing guarded against near-miss VARIANTS WITHIN one.
        // ⚠ GUARD ON THE CONTEXT VALUES, NOT THE STYLE STRING. parakeet-unified ALSO declares
        // `att_context_style = "chunked_limited_with_rc"` — with left = right = -1, meaning
        // UNLIMITED, which is full attention and transcribes correctly. A first version refused on
        // the style name and broke the model that already worked. The style names the SCHEME; the
        // -1s say it is not actually limited.
        let ctx_of = |k: &str| -> i64 {
            match md.get(k) { Some(Meta::U(v)) => *v as i64, Some(Meta::I(v)) => *v, _ => -1 }
        };
        let (cl, cr_att) = (ctx_of("stt.parakeet.encoder.att_context_left"),
                            ctx_of("stt.parakeet.encoder.att_context_right"));
        if cl >= 0 || cr_att >= 0 {
            let style = match md.get("stt.parakeet.encoder.att_context_style") {
                Some(Meta::Str(v)) => v.clone(), _ => "chunked".into(),
            };
            return Err(format!("parakeet variant limits attention context (style {style:?}, \
                left {cl}, right {cr_att}); this runtime implements FULL-CONTEXT offline \
                attention only. Streaming models need the chunk mask and cache — unimplemented"));
        }
            let cr = match md.get("stt.parakeet.encoder.conv_context_right") {
                Some(Meta::U(v)) => *v as i64, Some(Meta::I(v)) => *v, _ => -1,
            };
            let ck = match md.get("stt.parakeet.encoder.conv_kernel") { Some(Meta::U(v)) => *v as i64, _ => 0 };
            if cr >= 0 && ck > 0 && cr != ck / 2 {
                return Err(format!("conv_context_right is {cr} for kernel {ck}: this runtime \
                    implements the SYMMETRIC depthwise conv (right = k/2 = {}) only", ck / 2));
            }
            if matches!(md.get("stt.parakeet.encoder.conv_norm_type"), Some(Meta::Str(v)) if v != "batch_norm") {
                return Err("conv_norm_type is not batch_norm; this runtime folds BatchNorm's \
                            running statistics and has no LayerNorm path in the conv module".into());
            }
            if md.get("stt.parakeet.prompt.field").is_some() {
                return Err("prompt-conditioned (multilingual) parakeet: the decoder needs a \
                            language prompt token this runtime does not supply".into());
            }

        Ok(Cfg {
            sample_rate: u("stt.frontend.sample_rate")?,
            n_fft: u("stt.frontend.n_fft")?,
            hop_length: u("stt.frontend.hop_length")?,
            win_length: u("stt.frontend.win_length")?,
            num_mels: u("stt.frontend.num_mels")?,
            f_min: f("stt.frontend.f_min", 0.0),
            f_max: f("stt.frontend.f_max", 8000.0),
            pre_emphasis: f("stt.frontend.pre_emphasis", 0.97),
            n_layers: u("stt.parakeet.encoder.n_layers")?,
            d_model: u("stt.parakeet.encoder.d_model")?,
            n_heads: u("stt.parakeet.encoder.n_heads")?,
            d_ff: u("stt.parakeet.encoder.d_ff")?,
            conv_kernel: u("stt.parakeet.encoder.conv_kernel")?,
            subsampling_factor: u("stt.parakeet.encoder.subsampling_factor")?,
            subsampling_channels: u("stt.parakeet.encoder.subsampling_channels")?,
            xscaling: b("stt.parakeet.encoder.xscaling", true),
            pred_hidden: u("stt.parakeet.predictor.hidden")?,
            pred_layers: u("stt.parakeet.predictor.n_layers")?,
            joint_hidden: u("stt.parakeet.joint.hidden")?,
            vocab: u("stt.parakeet.predictor.vocab")?,
            blank_id: match md.get("tokenizer.ggml.blank_token_id") {
                Some(Meta::U(v)) => *v as u32,
                // ⚠ Not defaulted silently: the blank is what RNN-T decoding advances on, and
                // guessing it wrong yields an empty or infinitely-repeating transcript.
                _ => return Err("missing tokenizer.ggml.blank_token_id".into()),
            },
        })
    }
}

/// A linear layer with an optional bias. Weights arrive F16/F32 — there is no quantized parakeet
/// in the wild yet — so they are dequantized to f32 and run through the ordinary matmul.
pub struct Linear { pub w: Tensor, pub b: Option<Tensor> }

impl Linear {
    /// GGUF stores `[in, out]` with `in` fastest, so the row-major bytes ARE `[out, in]` — which is
    /// the layout `matmul_bt` wants. Getting this backwards transposes every projection and still
    /// produces finite numbers.
    /// `bias` is a REQUEST, not an assertion: the tensor is used if present and skipped if not.
    ///
    /// ⚠ ANOTHER FALSE UNIVERSAL, this one mine. parakeet-unified carries `ff1.linear1.bias`;
    /// nemotron-3.5-asr-streaming — SAME `general.architecture` — does not, and the reference makes
    /// it `bias=config.attention_bias`, a per-checkpoint flag. Demanding it refused a second model
    /// that the runtime otherwise handles completely. Presence is ground truth, like everywhere else
    /// in this tree.
    fn load(ctx: &Arc<Context>, g: &impl GgufSource, name: &str, bias: bool) -> Result<Linear, String> {
        let t = g.tensor(name).ok_or_else(|| format!("missing {name}"))?;
        // ⚠ POINTWISE CONVS ARE RANK 3. `conv.pointwise1.weight` is [1, 1024, 2048] — a Conv1d of
        // kernel width 1, which is a matmul, but stored with the kernel axis first. Reading dims[0]
        // as the input width made it a [1 → 1024] layer and the element count disagreed by 2048x.
        // Caught by `data len != shape product`; had the counts happened to agree it would have run.
        let (i, o) = match t.dims.len() {
            2 => (t.dims[0] as usize, t.dims[1] as usize),
            3 if t.dims[0] == 1 => (t.dims[1] as usize, t.dims[2] as usize),
            _ => return Err(format!("{name}: rank {} dims {:?} is not a linear or a \
                                     kernel-1 pointwise conv", t.dims.len(), t.dims)),
        };
        let w = Tensor::from_vec(ctx, &g.dequant(name)?, &[o, i]);
        let bn = name.replace(".weight", ".bias");
        let b = if bias && g.tensor(&bn).is_some() {
            Some(Tensor::from_vec(ctx, &g.dequant(&bn)?, &[o]))
        } else { None };
        Ok(Linear { w, b })
    }
}

/// LayerNorm with a bias — Conformer uses it throughout, unlike the RMSNorm of every text model here.
pub struct Norm { pub w: Tensor, pub b: Tensor }

impl Norm {
    fn load(ctx: &Arc<Context>, g: &impl GgufSource, base: &str, d: usize) -> Result<Norm, String> {
        Ok(Norm {
            w: Tensor::from_vec(ctx, &g.dequant(&format!("{base}.weight"))?, &[d]),
            b: Tensor::from_vec(ctx, &g.dequant(&format!("{base}.bias"))?, &[d]),
        })
    }
}

/// The conv module's BatchNorm. At inference it is affine — `running_*` are constants — so it is
/// kept as four vectors and folded at forward time rather than needing a BatchNorm op.
pub struct BatchNorm { pub w: Tensor, pub b: Tensor, pub mean: Tensor, pub var: Tensor }

/// Relative-position multi-head attention (Transformer-XL style): `linear_pos` projects the
/// positional encoding, and `pos_bias_u`/`pos_bias_v` are the learned content/position biases added
/// to Q before the two score terms.
pub struct RelPosAttn {
    pub q: Linear, pub k: Linear, pub v: Linear, pub out: Linear,
    pub pos: Linear,                       // no bias in the checkpoint
    pub bias_u: Tensor, pub bias_v: Tensor, // [head_dim, n_heads]
}

/// The Conformer convolution module: pointwise → GLU → depthwise (SYMMETRIC, not causal:
/// `conv_context_left == conv_context_right == 4` for kernel 9) → batch-norm → SiLU → pointwise.
pub struct ConvModule {
    pub pw1: Linear, pub dw_w: Tensor, pub dw_b: Tensor, pub bn: BatchNorm, pub pw2: Linear,
}

pub struct Block {
    pub norm_ff1: Norm, pub ff1: (Linear, Linear),
    pub norm_attn: Norm, pub attn: RelPosAttn,
    pub norm_conv: Norm, pub conv: ConvModule,
    pub norm_ff2: Norm, pub ff2: (Linear, Linear),
    pub norm_out: Norm,
}

/// The RNN-T predictor: an embedding over the output vocab plus `n_layers` LSTMs. `Wx`/`Wh` are
/// `[hidden, 4·hidden]` — the four gates (i, f, g, o) concatenated, in that order.
pub struct Predictor { pub embed: Tensor, pub lstm: Vec<LstmLayer> }
pub struct LstmLayer { pub wx: Tensor, pub wh: Tensor, pub b: Tensor }

/// The joint network: encoder and predictor states are each projected to `joint_hidden`, summed,
/// passed through the activation, and projected to the vocab.
pub struct Joint { pub enc: Linear, pub pred: Linear, pub out: Linear }

pub struct Parakeet {
    pub ctx: Arc<Context>,
    pub cfg: Cfg,
    pub pre_conv: Vec<(Tensor, Tensor)>,   // (weight [kh,kw,c,o], bias)
    pub pre_out: Linear,
    pub blocks: Vec<Block>,
    pub pred: Predictor,
    pub joint: Joint,
    pub tokens: Vec<String>,
}

impl Parakeet {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Parakeet, String> {
        let cfg = Cfg::from_gguf(g)?;
        let d = cfg.d_model;
        let t1 = |n: &str, shape: &[usize]| -> Result<Tensor, String> {
            Ok(Tensor::from_vec(ctx, &g.dequant(n)?, shape))
        };
        // An ABSENT bias is mathematically a ZERO bias, so optional-bias tensors load as zeros
        // rather than as Option plumbing through the forward pass. Which biases exist varies BY
        // CHECKPOINT within one architecture — parakeet-unified has ff/conv biases, and
        // nemotron-3.5-asr-streaming (same general.architecture) does not.
        let t1z = |n: &str, shape: &[usize]| -> Result<Tensor, String> {
            match g.tensor(n) {
                Some(_) => Ok(Tensor::from_vec(ctx, &g.dequant(n)?, shape)),
                None => Ok(Tensor::from_vec(ctx, &vec![0f32; shape.iter().product()], shape)),
            }
        };

        // pre_encode: the conv indices are NOT contiguous (0,2,3,5,6 — the gaps are activations in
        // the original nn.Sequential), so they are discovered rather than assumed.
        let mut pre_conv = Vec::new();
        for i in 0..12 {
            let wn = format!("enc.pre_encode.conv.{i}.weight");
            if g.tensor(&wn).is_none() { continue; }
            let t = g.tensor(&wn).expect("checked");
            let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
            let o = dims[3];
            pre_conv.push((t1(&wn, &dims)?, t1z(&format!("enc.pre_encode.conv.{i}.bias"), &[o])?));
        }
        if pre_conv.is_empty() { return Err("no enc.pre_encode.conv.* tensors".into()); }
        let pre_out = Linear::load(ctx, g, "enc.pre_encode.out.weight", true)?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let b = |s: &str| format!("enc.blocks.{il}.{s}");
            let hd = d / cfg.n_heads;
            blocks.push(Block {
                norm_ff1: Norm::load(ctx, g, &b("norm_ff1"), d)?,
                ff1: (Linear::load(ctx, g, &b("ff1.linear1.weight"), true)?,
                      Linear::load(ctx, g, &b("ff1.linear2.weight"), true)?),
                norm_attn: Norm::load(ctx, g, &b("norm_attn"), d)?,
                attn: RelPosAttn {
                    q: Linear::load(ctx, g, &b("attn.linear_q.weight"), true)?,
                    k: Linear::load(ctx, g, &b("attn.linear_k.weight"), true)?,
                    v: Linear::load(ctx, g, &b("attn.linear_v.weight"), true)?,
                    out: Linear::load(ctx, g, &b("attn.linear_out.weight"), true)?,
                    pos: Linear::load(ctx, g, &b("attn.linear_pos.weight"), false)?,
                    bias_u: t1(&b("attn.pos_bias_u"), &[cfg.n_heads, hd])?,
                    bias_v: t1(&b("attn.pos_bias_v"), &[cfg.n_heads, hd])?,
                },
                norm_conv: Norm::load(ctx, g, &b("norm_conv"), d)?,
                conv: ConvModule {
                    pw1: Linear::load(ctx, g, &b("conv.pointwise1.weight"), true)?,
                    // [k, 1, d] as stored — kernel-major. Reshaping it to [d, k] here would have
                    // the right element COUNT and the wrong order, which no assert would catch.
                    dw_w: t1(&b("conv.depthwise.weight"), &[cfg.conv_kernel, 1, d])?,
                    dw_b: t1z(&b("conv.depthwise.bias"), &[d])?,
                    bn: BatchNorm {
                        w: t1(&b("conv.bn.weight"), &[d])?,
                        b: t1(&b("conv.bn.bias"), &[d])?,
                        mean: t1(&b("conv.bn.running_mean"), &[d])?,
                        var: t1(&b("conv.bn.running_var"), &[d])?,
                    },
                    pw2: Linear::load(ctx, g, &b("conv.pointwise2.weight"), true)?,
                },
                norm_ff2: Norm::load(ctx, g, &b("norm_ff2"), d)?,
                ff2: (Linear::load(ctx, g, &b("ff2.linear1.weight"), true)?,
                      Linear::load(ctx, g, &b("ff2.linear2.weight"), true)?),
                norm_out: Norm::load(ctx, g, &b("norm_out"), d)?,
            });
        }

        let mut lstm = Vec::with_capacity(cfg.pred_layers);
        for i in 0..cfg.pred_layers {
            let h = cfg.pred_hidden;
            lstm.push(LstmLayer {
                wx: t1(&format!("pred.lstm.{i}.Wx"), &[4 * h, h])?,
                wh: t1(&format!("pred.lstm.{i}.Wh"), &[4 * h, h])?,
                b:  t1(&format!("pred.lstm.{i}.bias"), &[4 * h])?,
            });
        }
        let pred = Predictor {
            embed: t1("pred.embed.weight", &[cfg.vocab, cfg.pred_hidden])?,
            lstm,
        };
        let joint = Joint {
            enc: Linear::load(ctx, g, "joint.enc.weight", true)?,
            pred: Linear::load(ctx, g, "joint.pred.weight", true)?,
            out: Linear::load(ctx, g, "joint.out.weight", true)?,
        };
        let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
            _ => return Err("missing tokenizer.ggml.tokens".into()),
        };
        if tokens.len() != cfg.vocab {
            return Err(format!("{} tokens but predictor.vocab is {}", tokens.len(), cfg.vocab));
        }
        Ok(Parakeet { ctx: ctx.clone(), cfg, pre_conv, pre_out, blocks, pred, joint, tokens })
    }

    /// A load-time receipt. A schedule that silently collapsed shows up here rather than as a wrong
    /// transcript later — the same reason `nemotron_h::schedule` exists.
    pub fn describe(&self) -> String {
        format!("parakeet · {} conformer blocks · d_model {} · {} heads · d_ff {} · conv_k {} \
                 · subsample {}x · {} mels @ {} Hz · RNN-T {}L LSTM h{} · joint h{} · vocab {} (blank {})",
                self.blocks.len(), self.cfg.d_model, self.cfg.n_heads, self.cfg.d_ff,
                self.cfg.conv_kernel, self.cfg.subsampling_factor, self.cfg.num_mels,
                self.cfg.sample_rate, self.pred.lstm.len(), self.cfg.pred_hidden,
                self.cfg.joint_hidden, self.cfg.vocab, self.cfg.blank_id)
    }
}

/// **The mel frontend** — waveform in, `[frames, num_mels]` log-mel out.
///
/// Ferric had no audio path at all, so this is written from the `stt.frontend.*` metadata rather
/// than ported: `n_fft 512`, `hop 160`, `win 400`, `128` mels, hann, pre-emphasis `0.97`,
/// `per_feature` normalisation. CPU: a 512-point FFT per 10 ms hop is negligible beside 24
/// Conformer blocks, and a WGSL FFT can replace it later without changing this interface.
pub mod frontend {
    use super::Cfg;

    /// Radix-2 Cooley-Tukey, in place. `n_fft` is 512 on every published parakeet; a non-power-of-two
    /// would silently produce garbage, so it is refused.
    fn fft(re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        assert!(n.is_power_of_two(), "fft length {n} is not a power of two");
        // bit-reversal permutation
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 { j ^= bit; bit >>= 1; }
            j |= bit;
            if i < j { re.swap(i, j); im.swap(i, j); }
        }
        let mut len = 2;
        while len <= n {
            let ang = -2.0 * std::f32::consts::PI / len as f32;
            let (wr, wi) = (ang.cos(), ang.sin());
            for i in (0..n).step_by(len) {
                let (mut cr, mut ci) = (1.0f32, 0.0f32);
                for k in 0..len / 2 {
                    let (ur, ui) = (re[i + k], im[i + k]);
                    let (vr, vi) = (re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr);
                    re[i + k] = ur + vr; im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                    let nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr; cr = nr;
                }
            }
            len <<= 1;
        }
    }

    // ⚠ SLANEY, NOT HTK. `librosa.filters.mel` defaults to `htk=False` — a mel scale that is LINEAR
    // below 1 kHz and logarithmic above, not `2595·log10(1 + f/700)`. The first version used HTK and
    // passed a pure-tone test, because that test computed the expected bin FROM THE SAME filterbank:
    // it proved the FFT and the filterbank agreed with each other, not that either was right.
    const F_SP: f32 = 200.0 / 3.0;          // Hz per mel below the break
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;   // 15.0

    fn logstep() -> f32 { (6.4f32).ln() / 27.0 }

    fn hz_to_mel(f: f32) -> f32 {
        if f < MIN_LOG_HZ { f / F_SP } else { MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / logstep() }
    }
    fn mel_to_hz(m: f32) -> f32 {
        if m < MIN_LOG_MEL { F_SP * m } else { MIN_LOG_HZ * (logstep() * (m - MIN_LOG_MEL)).exp() }
    }

    /// Triangular mel filterbank, `[num_mels][n_fft/2 + 1]`, with **Slaney area normalisation**
    /// (`norm="slaney"`): each filter is scaled by `2 / (edge[i+2] - edge[i])`, so wider
    /// high-frequency filters do not collect proportionally more energy. Unit-peak triangles feed
    /// the encoder a spectrum tilted towards high frequencies.
    pub fn filterbank(cfg: &Cfg) -> Vec<Vec<f32>> {
        let n_bins = cfg.n_fft / 2 + 1;
        let (lo, hi) = (hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max));
        let edges: Vec<f32> = (0..cfg.num_mels + 2)
            .map(|i| mel_to_hz(lo + (hi - lo) * i as f32 / (cfg.num_mels + 1) as f32))
            .collect();
        let bin_hz = cfg.sample_rate as f32 / cfg.n_fft as f32;
        (0..cfg.num_mels).map(|m| {
            let enorm = 2.0 / (edges[m + 2] - edges[m]).max(1e-9);
            (0..n_bins).map(|b| {
                let f = b as f32 * bin_hz;
                let (l, c, r) = (edges[m], edges[m + 1], edges[m + 2]);
                let w = if f <= l || f >= r { 0.0 }
                        else if f <= c { (f - l) / (c - l).max(1e-9) }
                        else { (r - f) / (r - c).max(1e-9) };
                w * enorm
            }).collect()
        }).collect()
    }

    /// `[frames, num_mels]` log-mel, row-major. `pcm` is mono f32 at `cfg.sample_rate`.
    pub fn log_mel(pcm: &[f32], cfg: &Cfg) -> (Vec<f32>, usize) {
        // Pre-emphasis first: y[0] = x[0]; y[t] = x[t] - a·x[t-1].
        let mut x = Vec::with_capacity(pcm.len());
        x.push(pcm.first().copied().unwrap_or(0.0));
        for t in 1..pcm.len() { x.push(pcm[t] - cfg.pre_emphasis * pcm[t - 1]); }

        // ⚠ CENTRED, like `torch.stft(center=True)`: zero-pad n_fft/2 at BOTH ends. Without it every
        // frame is offset by half a window against what the encoder was trained on.
        let half = cfg.n_fft / 2;
        let mut padded = vec![0f32; half];
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat(0.0).take(half));

        // ⚠ SYMMETRIC Hann (`periodic=False`): divide by (N-1), not N.
        let win: Vec<f32> = (0..cfg.win_length)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32
                                  / (cfg.win_length - 1) as f32).cos())
            .collect();
        let fb = filterbank(cfg);
        let n_bins = cfg.n_fft / 2 + 1;
        let frames = 1 + (padded.len().saturating_sub(cfg.n_fft)) / cfg.hop_length;
        let mut out = vec![0f32; frames * cfg.num_mels];
        let (mut re, mut im) = (vec![0f32; cfg.n_fft], vec![0f32; cfg.n_fft]);
        // The window is win_length wide inside an n_fft transform — centre it in the frame.
        let woff = (cfg.n_fft - cfg.win_length) / 2;
        for f in 0..frames {
            let s = f * cfg.hop_length;
            re.iter_mut().for_each(|v| *v = 0.0);
            im.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..cfg.win_length {
                if s + woff + i < padded.len() { re[woff + i] = padded[s + woff + i] * win[i]; }
            }
            fft(&mut re, &mut im);
            let power: Vec<f32> = (0..n_bins).map(|b| re[b] * re[b] + im[b] * im[b]).collect();
            for m in 0..cfg.num_mels {
                let e: f32 = fb[m].iter().zip(&power).map(|(w, p)| w * p).sum();
                // ⚠ 2^-24, the reference's LOG_ZERO_GUARD_VALUE — not 1e-9. It sets the floor for
                // silent bins, which per-feature normalisation then spreads across the whole range.
                out[f * cfg.num_mels + m] = (e + LOG_ZERO_GUARD).ln();
            }
        }
        (out, frames)
    }

    /// The reference's `LOG_ZERO_GUARD_VALUE`.
    pub const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;   // 2^-24

    /// `per_feature` normalisation — zero mean, unit variance PER MEL BIN across time, which is what
    /// `stt.frontend.normalize` selects. Normalising across the whole matrix instead would leave a
    /// per-bin offset the encoder was never trained to see.
    pub fn normalize_per_feature(mel: &mut [f32], frames: usize, n_mels: usize) {
        if frames == 0 { return; }
        for m in 0..n_mels {
            let mean: f32 = (0..frames).map(|f| mel[f * n_mels + m]).sum::<f32>() / frames as f32;
            // ⚠ BESSEL: the reference divides by (n-1), and adds EPSILON to the STD after the sqrt
            // rather than to the variance inside it. Both differ from the obvious form.
            let denom = (frames.saturating_sub(1)).max(1) as f32;
            let var: f32 = (0..frames).map(|f| { let d = mel[f * n_mels + m] - mean; d * d })
                           .sum::<f32>() / denom;
            let sd = var.sqrt() + 1e-5;
            for f in 0..frames { mel[f * n_mels + m] = (mel[f * n_mels + m] - mean) / sd; }
        }
    }
}

// ============================================================================================
// Encoder forward
// ============================================================================================

impl Parakeet {
    /// **Pre-encode**: the `dw_striding` subsampling stack, on the CPU.
    ///
    /// Three stride-2 stages take `[T, 128]` mel to `[T/8, 16, 256]`, flattened to `[T/8, 4096]` for
    /// the output projection. It is CPU because Ferric's `conv2d` is a FULL convolution and two of
    /// these three stages are DEPTHWISE — `[3,3,1,256]` against 256-channel input, which that op
    /// asserts against. A depthwise WGSL kernel can replace this later; correctness first.
    ///
    /// ⚠ THE FLATTEN ORDER IS AN ASSUMPTION. NeMo does `transpose(1,2).reshape(b, t, -1)`, i.e.
    /// channel-major (`c*n_freq + f`). The other order has the SAME element count and produces a
    /// different model — the class of bug that costs an afternoon, so it is named here as the first
    /// thing to suspect if the transcript is fluent nonsense.
    fn pre_encode(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize, usize) {
        let c = &self.cfg;
        let (mut h, mut w, mut ch) = (frames, c.num_mels, 1usize);
        let mut cur = mel.to_vec();                      // [h][w][ch]
        let relu = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| if *x < 0.0 { *x = 0.0 });

        for (idx, (wt, bs)) in self.pre_conv.iter().enumerate() {
            let wd = &wt.shape;                          // [kh, kw, c_in, c_out]
            let (kh, kw, cin, cout) = (wd[0], wd[1], wd[2], wd[3]);
            let wv = pollster::block_on(wt.to_vec());
            let bv = pollster::block_on(bs.to_vec());
            if kh == 1 && kw == 1 {
                // Pointwise: a plain [in → out] matmul over every (t, f) position.
                let mut out = vec![0f32; h * w * cout];
                for p in 0..h * w {
                    for o in 0..cout {
                        let mut a = bv[o];
                        for k in 0..cin { a += wv[o * cin + k] * cur[p * ch + k]; }
                        out[p * cout + o] = a;
                    }
                }
                cur = out; ch = cout;
            } else {
                // 3x3, stride 2, pad 1. `cin == 1` marks DEPTHWISE: output channel o reads input
                // channel o. `cin == ch` is the ordinary full convolution (the first stage, 1→256).
                let depthwise = cin == 1 && ch > 1;
                let (oh, ow) = ((h + 1) / 2, (w + 1) / 2);
                let mut out = vec![0f32; oh * ow * cout];
                for y in 0..oh { for x in 0..ow {
                    for o in 0..cout {
                        let mut a = bv[o];
                        for i in 0..kh { for j in 0..kw {
                            let (sy, sx) = (y as isize * 2 + i as isize - 1, x as isize * 2 + j as isize - 1);
                            if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize { continue; }
                            let base = (sy as usize * w + sx as usize) * ch;
                            if depthwise {
                                a += wv[((o * cin) * kh + i) * kw + j] * cur[base + o];
                            } else {
                                for k in 0..cin { a += wv[((o * cin + k) * kh + i) * kw + j] * cur[base + k]; }
                            }
                        }}
                        out[(y * ow + x) * cout + o] = a;
                    }
                }}
                cur = out; h = oh; w = ow; ch = cout;
            }
            // ⚠ `idx` is the position in THIS vec, not the GGUF name index. The convs are named
            // 0,2,3,5,6 (the gaps were activations in the original Sequential) and are stored
            // densely here as 0..4. The first version tested the GGUF numbering against the vec
            // index and put a ReLU after the second DEPTHWISE instead of after a pointwise.
            //
            // dw_striding is [full] [dw, pw] [dw, pw], with the activation after the full conv and
            // after each pointwise — vec indices 0, 2, 4.
            if idx == 0 || idx == 2 || idx == 4 { relu(&mut cur); }
        }
        // [h][w][ch] → [h][ch * w], channel-major (see the warning above).
        let mut flat = vec![0f32; h * ch * w];
        for t in 0..h { for k in 0..ch { for f in 0..w {
            flat[t * ch * w + k * w + f] = cur[(t * w + f) * ch + k];
        }}}
        (flat, h, ch * w)
    }
}

impl Norm {
    fn apply(&self, x: &Tensor, eps: f32) -> Tensor { x.layernorm(&self.w, &self.b, eps) }
}

impl Linear {
    /// `x [T, in] · wᵀ + b`. The bias broadcasts over rows.
    fn apply(&self, x: &Tensor) -> Tensor {
        let y = x.matmul_bt(&self.w);
        match &self.b { Some(b) => y.add(&b.reshape(&[1, b.numel()]).broadcast_to(&y.shape)), None => y }
    }
}

impl Parakeet {
    const EPS: f32 = 1e-5;

    /// One Conformer block: ½·FF → rel-pos MHSA → conv module → ½·FF → final norm.
    ///
    /// The **halves are not decoration** — Conformer's macaron FFNs each contribute 0.5·output to
    /// the residual, and using 1.0 doubles the FFN's influence on every one of 24 blocks.
    fn block(&self, x: &Tensor, b: &Block, pos: &Tensor) -> Tensor {
        let d = self.cfg.d_model;
        // No scalar-multiply op on Tensor; `scalar` makes a [1] tensor to broadcast against.
        let half = |t: &Tensor| t.mul(&t.scalar(0.5).broadcast_to(&t.shape));

        // ---- macaron FF1 ----
        let h = b.norm_ff1.apply(x, Self::EPS);
        let h = b.ff1.1.apply(&b.ff1.0.apply(&h).silu());
        let x = x.add(&half(&h));

        // ---- relative-position self-attention ----
        let h = b.norm_attn.apply(&x, Self::EPS);
        let x = x.add(&self.rel_pos_attn(&h, &b.attn, pos));

        // ---- convolution module ----
        let h = b.norm_conv.apply(&x, Self::EPS);
        let x = x.add(&self.conv_module(&h, &b.conv));

        // ---- macaron FF2 ----
        let h = b.norm_ff2.apply(&x, Self::EPS);
        let h = b.ff2.1.apply(&b.ff2.0.apply(&h).silu());
        let x = x.add(&half(&h));

        // ⚠ Probe the LAST block's final norm specifically: its output sits 10x below its own
        // weight rms while every other block tracks at ~1.2x. Either its pre-norm residual has
        // collapsed, or its bias cancels the scale — these need different fixes.
        if std::env::var("FERRIC_ASR_DEBUG").is_ok() && std::ptr::eq(b, self.blocks.last().unwrap()) {
            let f = |t: &Tensor| { let v = pollster::block_on(t.to_vec());
                                   (v.iter().map(|z| z * z).sum::<f32>() / v.len() as f32).sqrt() };
            let bb = pollster::block_on(b.norm_out.b.to_vec());
            let br = (bb.iter().map(|z| z * z).sum::<f32>() / bb.len() as f32).sqrt();
            eprintln!("       [last] pre-norm residual rms={:.4}  norm_out bias rms={br:.4}", f(&x));
            // Two implementations of the same LayerNorm. If they agree, the op is right and the
            // 0.044 is what these weights genuinely produce — meaning my expectation (output ~
            // rms(w)) is what is wrong, because rms(w) is dominated by a few large entries while
            // most are small. If they disagree, the GPU op is wrong at this shape.
            let xv = pollster::block_on(x.to_vec());
            let wv = pollster::block_on(b.norm_out.w.to_vec());
            let dd = self.cfg.d_model;
            let rows = xv.len() / dd;
            let mut acc = 0f64;
            for r in 0..rows {
                let row = &xv[r * dd..(r + 1) * dd];
                let mu = row.iter().sum::<f32>() / dd as f32;
                let va = row.iter().map(|z| (z - mu) * (z - mu)).sum::<f32>() / dd as f32;
                let sd = (va + Self::EPS).sqrt();
                for i in 0..dd {
                    let y = (row[i] - mu) / sd * wv[i] + bb[i];
                    acc += (y * y) as f64;
                }
            }
            let cpu = (acc / (rows * dd) as f64).sqrt();
            let wmed = { let mut v: Vec<f32> = wv.iter().map(|z| z.abs()).collect();
                         v.sort_by(f32::total_cmp); v[v.len() / 2] };
            eprintln!("       [last] cpu layernorm rms={cpu:.4}   |w| median={wmed:.4} (vs rms {br:.4} bias)");
            // ⭐ THE ACTUAL SIGNATURE. rms(x)=48 with per-row VARIANCE ~1e-9 means every feature in a
            // row holds nearly the same value — a constant vector, not a representation. LayerNorm
            // then divides by sqrt(eps) and the output collapses. Print the row mean and spread so
            // the diagnosis is a measurement rather than an inference.
            let r0 = &xv[0..dd];
            let mu0 = r0.iter().sum::<f32>() / dd as f32;
            let va0 = r0.iter().map(|z| (z - mu0) * (z - mu0)).sum::<f32>() / dd as f32;
            let (lo, hi) = (r0.iter().cloned().fold(f32::MAX, f32::min),
                            r0.iter().cloned().fold(f32::MIN, f32::max));
            eprintln!("       [last] row0 mean={mu0:.4} var={va0:.3e} min={lo:.4} max={hi:.4}  \
                       first8={:?}", &r0[..8].iter().map(|z| (z * 1e3).round() / 1e3).collect::<Vec<_>>());
        }
        b.norm_out.apply(&x, Self::EPS)
    }

    /// Transformer-XL relative-position attention.
    ///
    /// `score = (q + u)·kᵀ + (q + v)·posᵀ`, the second term shifted so entry `(i, j)` reads relative
    /// offset `i - j`. `u` and `v` are the learned content and position biases (`pos_bias_u/v`);
    /// folding them into one term, or dropping the shift, both yield finite scores and a model that
    /// attends to the wrong places.
    fn rel_pos_attn(&self, x: &Tensor, a: &RelPosAttn, pos: &Tensor) -> Tensor {
        let (t, nh) = (x.shape[0], self.cfg.n_heads);
        let hd = self.cfg.d_model / nh;
        let scale = 1.0 / (hd as f32).sqrt();

        let q = a.q.apply(x).reshape(&[t, nh, hd]);
        let k = a.k.apply(x).reshape(&[t, nh, hd]);
        let v = a.v.apply(x).reshape(&[t, nh, hd]);
        // pos is [2T-1, d] — every relative offset from -(T-1) to +(T-1).
        let p = a.pos.apply(pos);
        let np = p.shape[0];
        let p = p.reshape(&[np, nh, hd]);

        let u = a.bias_u.reshape(&[1, nh, hd]).broadcast_to(&[t, nh, hd]);
        let v_b = a.bias_v.reshape(&[1, nh, hd]).broadcast_to(&[t, nh, hd]);
        let qu = q.add(&u);
        let qv = q.add(&v_b);

        let mut heads: Vec<Tensor> = Vec::with_capacity(nh);
        for hi in 0..nh {
            let qu_h = qu.narrow(1, hi, 1).reshape(&[t, hd]);
            let qv_h = qv.narrow(1, hi, 1).reshape(&[t, hd]);
            let k_h = k.narrow(1, hi, 1).reshape(&[t, hd]);
            let v_h = v.narrow(1, hi, 1).reshape(&[t, hd]);
            let p_h = p.narrow(1, hi, 1).reshape(&[np, hd]);

            let ac = qu_h.matmul_bt(&k_h);                 // [t, t]  content term
            let bd = qv_h.matmul_bt(&p_h);                 // [t, np] position term, unshifted
            // `rel_shift`: row i must read offset (i - j), which lives at column (T-1 + i - j).
            let bdv = pollster::block_on(bd.to_vec());
            let mut sh = vec![0f32; t * t];
            for i in 0..t { for j in 0..t {
                // ⭐ (T-1) - i + j, NOT (T-1) + i - j. Derived by unrolling the reference
                // `_rel_shift` (pad(1,0) → view(-1,T) → drop row 0 → view(T,P) → slice to T):
                // final[i,j] = bd[i, (T-1) - i + j]. The first version had the sign of the relative
                // offset FLIPPED, so every head attended to the mirror of its intended distance —
                // finite scores, plausible output, wrong model. Reading the code could not settle
                // this; unrolling the reference's index arithmetic could.
                sh[i * t + j] = bdv[i * np + (t - 1 - i + j)];
            }}
            let bd = Tensor::from_vec(&self.ctx, &sh, &[t, t]);
            let sum = ac.add(&bd);
            let s = sum.mul(&sum.scalar(scale).broadcast_to(&sum.shape)).softmax(1);
            heads.push(s.matmul(&v_h));                     // [t, hd]
        }
        let cat = heads.iter().skip(1).fold(heads[0].clone(), |a, b| a.cat(b, 1));
        a.out.apply(&cat.reshape(&[t, self.cfg.d_model]))
    }

    /// Conformer convolution module: pointwise → GLU → depthwise → batch-norm → SiLU → pointwise.
    fn conv_module(&self, x: &Tensor, c: &ConvModule) -> Tensor {
        let (t, d) = (x.shape[0], self.cfg.d_model);
        let k = self.cfg.conv_kernel;
        // pointwise1 doubles the width for the GLU: half gates the other half.
        let y = c.pw1.apply(x);                                     // [t, 2d]
        let yv = pollster::block_on(y.to_vec());
        let mut g = vec![0f32; t * d];
        for i in 0..t { for j in 0..d {
            let a = yv[i * 2 * d + j];
            let b = yv[i * 2 * d + d + j];
            g[i * d + j] = a * (1.0 / (1.0 + (-b).exp()));           // GLU: a ⊙ σ(b)
        }}
        let g = Tensor::from_vec(&self.ctx, &g, &[t, d]);

        // ⚠ SYMMETRIC, not causal. conv_context_left == conv_context_right == 4 for kernel 9, so the
        // window is centred. Ferric only has a CAUSAL depthwise conv1d, and y_sym[t] = y_causal[t+4]
        // once the signal is right-padded by 4 — so the existing kernel serves, with a shift.
        let pad = k / 2;
        let mut padded = pollster::block_on(g.to_vec());
        padded.extend(std::iter::repeat(0.0).take(pad * d));
        let gp = Tensor::from_vec(&self.ctx, &padded, &[t + pad, d]);
        // ⚠ NO TRANSPOSE. GGUF dims [9, 1, 1024] list ne0 FIRST and ne0 is the INNERMOST axis, so
        // the data is already w[c * 9 + k] — the [C, L] layout `depthwise_conv1d_causal` wants. The
        // first version read it as [K][C] and transposed, scrambling every one of the 1024 kernels
        // into a mix of nine different channels' taps. Same element count, same shapes, no assert
        // could fire — the model just convolved with noise.
        let dwv = pollster::block_on(c.dw_w.to_vec());
        debug_assert_eq!(dwv.len(), d * k, "depthwise weight is {} floats, expected {}", dwv.len(), d * k);
        let wk = Tensor::from_vec(&self.ctx, &dwv, &[d, k]);
        let conv = gp.depthwise_conv1d_causal(&wk, k).narrow(0, pad, t).contiguous();

        // BatchNorm at inference is affine over the running statistics — no op needed.
        let bnv = pollster::block_on(conv.to_vec());
        let (mw, mb) = (pollster::block_on(c.bn.w.to_vec()), pollster::block_on(c.bn.b.to_vec()));
        let (mm, mv) = (pollster::block_on(c.bn.mean.to_vec()), pollster::block_on(c.bn.var.to_vec()));
        let dwb = pollster::block_on(c.dw_b.to_vec());
        let mut o = vec![0f32; t * d];
        for i in 0..t { for j in 0..d {
            let z = bnv[i * d + j] + dwb[j];
            let n = (z - mm[j]) / (mv[j] + 1e-5).sqrt() * mw[j] + mb[j];
            o[i * d + j] = n * (1.0 / (1.0 + (-n).exp()));            // SiLU
        }}
        c.pw2.apply(&Tensor::from_vec(&self.ctx, &o, &[t, d]))
    }
}

// ============================================================================================
// Encoder entry point + RNN-T decoding
// ============================================================================================

impl Parakeet {
    /// Sinusoidal relative positional encoding for offsets `+(T-1) … -(T-1)`, `[2T-1, d]`.
    ///
    /// ⚠ DESCENDING. NeMo builds positions from `+(T-1)` down to `-(T-1)`, and `rel_shift` reads
    /// column `T-1+i-j` on that assumption. Building it ascending flips every relative offset — the
    /// model then attends backwards, fluently.
    fn rel_pos_encoding(&self, t: usize) -> Tensor {
        let d = self.cfg.d_model;
        let n = 2 * t - 1;
        let mut v = vec![0f32; n * d];
        for (r, p) in (0..n).map(|r| (r, (t as isize - 1) - r as isize)) {
            for i in (0..d).step_by(2) {
                let div = (-(i as f32) * (10000f32).ln() / d as f32).exp();
                let a = p as f32 * div;
                v[r * d + i] = a.sin();
                if i + 1 < d { v[r * d + i + 1] = a.cos(); }
            }
        }
        Tensor::from_vec(&self.ctx, &v, &[n, d])
    }

    /// Waveform → encoder states `[T/8, d_model]`.
    pub fn encode(&self, pcm: &[f32]) -> Result<Tensor, String> {
        let (mut mel, frames) = frontend::log_mel(pcm, &self.cfg);
        if frames == 0 { return Err("audio shorter than one analysis window".into()); }
        frontend::normalize_per_feature(&mut mel, frames, self.cfg.num_mels);
        let (flat, t, width) = self.pre_encode(&mel, frames);
        if t == 0 { return Err("audio too short to survive 8x subsampling".into()); }
        let x = Tensor::from_vec(&self.ctx, &flat, &[t, width]);
        let mut x = self.pre_out.apply(&x);
        if self.cfg.xscaling {
            let s = (self.cfg.d_model as f32).sqrt();
            x = x.mul(&x.scalar(s).broadcast_to(&x.shape));
        }
        let pos = self.rel_pos_encoding(t);
        let dbg = std::env::var("FERRIC_ASR_DEBUG").is_ok();
        let rms = |t: &Tensor| { let v = pollster::block_on(t.to_vec());
                                 (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() };
        if dbg { eprintln!("       pre_encode out rms={:.4} (after xscaling)", rms(&x)); }
        for (i, b) in self.blocks.iter().enumerate() {
            x = self.block(&x, b, &pos);
            // Where does the signal die? LayerNorm's eps floors any block whose residual variance
            // falls below ~1e-5, so a collapse shows as a step change here, not a gradual decay.
            if dbg {
                // Print the block's OWN norm_out weight beside its output: LayerNorm output should
                // track its weight, so the two diverging is the signature of an eps-floored input.
                let w = pollster::block_on(b.norm_out.w.to_vec());
                let wr = (w.iter().map(|z| z * z).sum::<f32>() / w.len() as f32).sqrt();
                eprintln!("       block {i:>2} out rms={:.4}  norm_out w rms={wr:.4}", rms(&x));
            }
        }
        Ok(x)
    }

    /// One LSTM step. `Wx`/`Wh` are `[4h, h]` with gates concatenated **i, f, g, o** — the PyTorch
    /// order. A different gate order still produces a running model with a broken memory cell.
    fn lstm_step(&self, l: &LstmLayer, x: &[f32], h: &mut [f32], c: &mut [f32],
                 wx: &[f32], wh: &[f32], b: &[f32]) {
        let n = h.len();
        let mut g = vec![0f32; 4 * n];
        for r in 0..4 * n {
            let mut a = b[r];
            for k in 0..n { a += wx[r * n + k] * x[k] + wh[r * n + k] * h[k]; }
            g[r] = a;
        }
        let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
        for j in 0..n {
            let (i, f, gg, o) = (sig(g[j]), sig(g[n + j]), g[2 * n + j].tanh(), sig(g[3 * n + j]));
            c[j] = f * c[j] + i * gg;
            h[j] = o * c[j].tanh();
        }
        let _ = l;
    }

    /// **RNN-T greedy decode.** For each encoder frame, emit non-blank tokens until the joint
    /// predicts blank, then advance. The inner loop is capped because a joint that never predicts
    /// blank would otherwise emit forever — a real failure mode when the blank id is wrong.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String, String> {
        let enc = self.encode(pcm)?;
        let (t, d) = (enc.shape[0], self.cfg.d_model);
        let encv = pollster::block_on(enc.to_vec());
        let h = self.cfg.pred_hidden;
        let nl = self.pred.lstm.len();

        // Predictor weights to host once — the LSTM is sequential and tiny beside the encoder.
        let lw: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = self.pred.lstm.iter()
            .map(|l| (pollster::block_on(l.wx.to_vec()),
                      pollster::block_on(l.wh.to_vec()),
                      pollster::block_on(l.b.to_vec())))
            .collect();
        let emb = pollster::block_on(self.pred.embed.to_vec());
        let (je_w, je_b) = (pollster::block_on(self.joint.enc.w.to_vec()),
                            pollster::block_on(self.joint.enc.b.as_ref().unwrap().to_vec()));
        let (jp_w, jp_b) = (pollster::block_on(self.joint.pred.w.to_vec()),
                            pollster::block_on(self.joint.pred.b.as_ref().unwrap().to_vec()));
        let (jo_w, jo_b) = (pollster::block_on(self.joint.out.w.to_vec()),
                            pollster::block_on(self.joint.out.b.as_ref().unwrap().to_vec()));
        let jh = self.cfg.joint_hidden;

        if std::env::var("FERRIC_ASR_DEBUG").is_ok() {
            let rms = (encv.iter().map(|x| x * x).sum::<f32>() / encv.len() as f32).sqrt();
            let nf = encv.iter().filter(|x| !x.is_finite()).count();
            // Is 0.044 the LayerNorm weight, or a collapsed encoder? The final norm's own scale
            // answers that: LN output rms ~= rms(weight), so if they match the encoder is behaving.
            let lnw = pollster::block_on(self.blocks.last().unwrap().norm_out.w.to_vec());
            let lnr = (lnw.iter().map(|x| x * x).sum::<f32>() / lnw.len() as f32).sqrt();
            // Does the encoder DISCRIMINATE across time? A collapsed encoder gives every frame the
            // same vector, which produces exactly this symptom: a constant prior, blank always.
            let fr: Vec<f32> = (0..t).map(|i| {
                let r = &encv[i * d..(i + 1) * d];
                (r.iter().map(|x| x * x).sum::<f32>() / d as f32).sqrt()
            }).collect();
            let (fmin, fmax) = (fr.iter().cloned().fold(f32::MAX, f32::min),
                                fr.iter().cloned().fold(f32::MIN, f32::max));
            // Cosine between the first two frames: ~1.0 means the encoder emits one vector.
            let (a, b) = (&encv[0..d], &encv[d..2 * d]);
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let (na, nb) = (a.iter().map(|x| x * x).sum::<f32>().sqrt(),
                            b.iter().map(|x| x * x).sum::<f32>().sqrt());
            eprintln!("  enc: {t} frames x {d}  rms={rms:.4}  non-finite={nf}");
            eprintln!("       final norm_out weight rms={lnr:.4}  (enc rms should track this)");
            eprintln!("       per-frame rms {fmin:.4}..{fmax:.4}   cos(f0,f1)={:.4}",
                      dot / (na * nb).max(1e-9));
        }
        let mut hs = vec![vec![0f32; h]; nl];
        let mut cs = vec![vec![0f32; h]; nl];
        // The predictor starts from the BLANK embedding, which is the RNN-T convention for "nothing
        // emitted yet" — starting from zeros gives the first token a state the model never saw.
        let mut last = self.cfg.blank_id as usize;
        let mut started = false;
        let mut pred_out = vec![0f32; h];
        let mut refresh = true;
        let mut out: Vec<u32> = Vec::new();

        for ti in 0..t {
            let mut emitted = 0;
            loop {
                if refresh {
                    // ⚠ NeMo's `predict()` feeds ZEROS before anything is emitted, not the blank
                    // embedding — `if y is not None: y = self.embed(y) else: y = torch.zeros(...)`.
                    // I asserted blank-init as "the RNN-T convention" without checking; it is a real
                    // fork between implementations and this checkpoint follows NeMo.
                    let mut inp: Vec<f32> = if started {
                        emb[last * h..(last + 1) * h].to_vec()
                    } else { vec![0f32; h] };
                    for li in 0..nl {
                        let (wx, wh, b) = &lw[li];
                        let (mut hh, mut cc) = (hs[li].clone(), cs[li].clone());
                        self.lstm_step(&self.pred.lstm[li], &inp, &mut hh, &mut cc, wx, wh, b);
                        hs[li] = hh.clone(); cs[li] = cc; inp = hh;
                    }
                    pred_out = inp; refresh = false;
                }
                // joint: relu(W_e·enc + W_p·pred) → W_o → argmax
                let mut z = vec![0f32; jh];
                let (mut se, mut sp, mut sb) = (0f32, 0f32, 0f32);
                for r in 0..jh {
                    let mut e = 0f32;
                    for k in 0..d { e += je_w[r * d + k] * encv[ti * d + k]; }
                    let mut pp = 0f32;
                    for k in 0..h { pp += jp_w[r * h + k] * pred_out[k]; }
                    se += e * e; sp += pp * pp; sb += (je_b[r] + jp_b[r]) * (je_b[r] + jp_b[r]);
                    let a = e + pp + je_b[r] + jp_b[r];
                    z[r] = if a > 0.0 { a } else { 0.0 };
                }
                if std::env::var("FERRIC_ASR_DEBUG").is_ok() && ti < 2 && emitted == 0 {
                    // WHICH TERM DECIDES? If the encoder term is negligible beside the predictor and
                    // bias, the joint is a constant prior and blank wins at every frame regardless
                    // of the audio — which is exactly the observed symptom.
                    eprintln!("  t={ti} joint terms rms: enc={:.4} pred={:.4} bias={:.4}",
                              (se / jh as f32).sqrt(), (sp / jh as f32).sqrt(), (sb / jh as f32).sqrt());
                }
                let mut best = (0usize, f32::NEG_INFINITY);
                for o in 0..self.cfg.vocab {
                    let mut a = jo_b[o];
                    for k in 0..jh { a += jo_w[o * jh + k] * z[k]; }
                    if a > best.1 { best = (o, a); }
                }
                if std::env::var("FERRIC_ASR_DEBUG").is_ok() && ti < 3 && emitted == 0 {
                    // The SHAPE of the distribution is the diagnostic: blank winning by a hair means
                    // the encoder is roughly right and something small is off; blank winning by
                    // orders of magnitude means the encoder output is not speech-like at all.
                    let mut all: Vec<(usize, f32)> = (0..self.cfg.vocab).map(|o| {
                        let mut a = jo_b[o];
                        for k in 0..jh { a += jo_w[o * jh + k] * z[k]; }
                        (o, a)
                    }).collect();
                    all.sort_by(|x, y| y.1.total_cmp(&x.1));
                    let top: Vec<String> = all.iter().take(5)
                        .map(|(o, v)| format!("{}={v:.2}", if *o as u32 == self.cfg.blank_id {
                            "<blk>".to_string() } else { self.tokens[*o].clone() })).collect();
                    eprintln!("  t={ti} top5: {}", top.join(" "));
                }
                if std::env::var("FERRIC_ASR_NOBLANK").is_ok() && emitted == 0 {
                    // Diagnostic: the best NON-blank token per frame. If these spell the utterance,
                    // the encoder/joint are right and only the blank calibration is off — a very
                    // different bug from "the encoder learned nothing".
                    let mut bb = (0usize, f32::NEG_INFINITY);
                    for o in 0..self.cfg.vocab {
                        if o as u32 == self.cfg.blank_id { continue; }
                        let mut a = jo_b[o];
                        for k in 0..jh { a += jo_w[o * jh + k] * z[k]; }
                        if a > bb.1 { bb = (o, a); }
                    }
                    eprint!("{}", self.tokens[bb.0].replace('▁', " "));
                }
                if best.0 as u32 == self.cfg.blank_id { break; }
                out.push(best.0 as u32);
                last = best.0; refresh = true; started = true;
                emitted += 1;
                // NeMo caps symbols per frame; without it a wrong blank id loops forever.
                if emitted >= 10 { break; }
            }
        }
        Ok(out.iter()
            .map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default().replace('▁', " "))
            .collect::<String>().trim().to_string())
    }
}
