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
        let b = if bias {
            let bn = name.replace(".weight", ".bias");
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

        // pre_encode: the conv indices are NOT contiguous (0,2,3,5,6 — the gaps are activations in
        // the original nn.Sequential), so they are discovered rather than assumed.
        let mut pre_conv = Vec::new();
        for i in 0..12 {
            let wn = format!("enc.pre_encode.conv.{i}.weight");
            if g.tensor(&wn).is_none() { continue; }
            let t = g.tensor(&wn).expect("checked");
            let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
            let o = dims[3];
            pre_conv.push((t1(&wn, &dims)?, t1(&format!("enc.pre_encode.conv.{i}.bias"), &[o])?));
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
                    dw_b: t1(&b("conv.depthwise.bias"), &[d])?,
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
