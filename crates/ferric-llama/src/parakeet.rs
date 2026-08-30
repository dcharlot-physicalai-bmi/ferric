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
    /// `(left, right)` attention context for a **cache-aware streaming** encoder, when the file
    /// declares `att_context_style = "chunked_limited"` with non-negative limits. `None` means
    /// full-context offline attention, which is what every unlimited (-1/-1) file wants.
    ///
    /// Decoding such a model OFFLINE needs the mask but NOT the cache: with the whole utterance in
    /// hand, the chunk structure is expressible as a single [T, T] additive mask. Streaming
    /// *inference* — feeding audio incrementally — additionally needs the KV cache, still unbuilt.
    pub att_ctx: Option<(usize, usize)>,
    /// Right-hand context of the depthwise conv: `k/2` for the SYMMETRIC offline conformer, `0` for
    /// the CAUSAL streaming one. Both run on the same causal kernel — symmetric right-pads by `k/2`
    /// and shifts, causal does neither. Read from the file, never assumed.
    pub conv_right: usize,
    /// The conv module's normaliser. Offline parakeet ships BatchNorm (folded to affine at load);
    /// the streaming encoder ships LayerNorm, which is a different op over a different axis.
    pub conv_layernorm: bool,
    // ---- decoder ----
    pub pred_hidden: usize,
    pub pred_layers: usize,
    pub joint_hidden: usize,
    pub vocab: usize,
    /// The RNN-T blank id — `vocab - 1` on every published parakeet, but read, not assumed.
    pub blank_id: u32,
    /// Which tensor-naming convention this file uses. The SAME architecture ships under two
    /// converters: `parakeet` (handy-computer: `enc.blocks.N.ff1.linear1`) and `asr` (NVIDIA's own:
    /// `encoder.layers.N.feed_forward1.linear1`). Names differ; the forward pass does not.
    pub naming: Naming,
    /// `ctc` files have no predictor or joint — one linear from encoder states to vocab.
    pub ctc: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Naming { Parakeet, Asr }

impl Naming {
    /// Per-layer tensor name for a logical role. Keeping the map in ONE place is what stops the two
    /// conventions from becoming two loaders.
    pub fn t(self, il: usize, role: &str) -> String {
        match self {
            Naming::Parakeet => format!("enc.blocks.{il}.{role}"),
            Naming::Asr => {
                let r = match role {
                    "norm_ff1" => "norm_feed_forward1", "norm_ff2" => "norm_feed_forward2",
                    "norm_attn" => "norm_self_att",
                    "ff1.linear1.weight" => "feed_forward1.linear1.weight",
                    "ff1.linear2.weight" => "feed_forward1.linear2.weight",
                    "ff2.linear1.weight" => "feed_forward2.linear1.weight",
                    "ff2.linear2.weight" => "feed_forward2.linear2.weight",
                    "conv.pointwise1.weight" => "conv.pointwise_conv1.weight",
                    "conv.pointwise2.weight" => "conv.pointwise_conv2.weight",
                    "conv.depthwise.weight" => "conv.depthwise_conv.weight",
                    "conv.depthwise.bias" => "conv.depthwise_conv.bias",
                    "conv.bn.weight" => "conv.batch_norm.weight",
                    "conv.bn.bias" => "conv.batch_norm.bias",
                    "conv.bn.running_mean" => "conv.batch_norm.running_mean",
                    "conv.bn.running_var" => "conv.batch_norm.running_var",
                    o if o.starts_with("attn.") => return format!("encoder.layers.{il}.self_{}", &o[..]),
                    o => o,
                };
                format!("encoder.layers.{il}.{r}")
            }
        }
    }
    /// Non-layer tensors.
    pub fn pre_conv(self, i: usize) -> String {
        match self { Naming::Parakeet => format!("enc.pre_encode.conv.{i}"),
                     Naming::Asr => format!("encoder.pre_encode.conv.{i}") }
    }
    pub fn pre_out(self) -> &'static str {
        match self { Naming::Parakeet => "enc.pre_encode.out.weight",
                     Naming::Asr => "encoder.pre_encode.out.weight" }
    }
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        // Two converters ship the same architecture. `asr` is NVIDIA's own (parakeet-ctc-1.1b),
        // `parakeet` is the community one. They differ in key namespace AND tensor names.
        let naming = match md.get("general.architecture") {
            Some(Meta::Str(a)) if a == "asr" => Naming::Asr,
            _ => Naming::Parakeet,
        };
        // Read a key from whichever namespace this file uses. `asr` also states the frontend in
        // SECONDS (window_size 0.025) where `parakeet` states SAMPLES (win_length 400).
        let alt = |parakeet: &str, asr: &str| -> Option<&Meta> {
            md.get(if naming == Naming::Asr { asr } else { parakeet })
        };
        let u = |k: &str| -> Result<usize, String> {
            match md.get(k) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing {k}")) }
        };
        let ua = |pk: &str, ak: &str| -> Result<usize, String> {
            match alt(pk, ak) { Some(Meta::U(v)) => Ok(*v as usize),
                                _ => Err(format!("missing {}", if naming == Naming::Asr { ak } else { pk })) }
        };
        let fa = |pk: &str, ak: &str, d: f32| -> f32 {
            match alt(pk, ak) { Some(Meta::F(v)) => *v as f32, _ => d }
        };
        let ba = |pk: &str, ak: &str, d: bool| -> bool {
            match alt(pk, ak) { Some(Meta::Bool(v)) => *v, _ => d }
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
        let mut att_ctx: Option<(usize, usize)> = None;
        let (mut conv_right, mut conv_layernorm) = (usize::MAX, false);   // MAX = "use k/2" below
        if cl >= 0 || cr_att >= 0 {
            let style = match md.get("stt.parakeet.encoder.att_context_style") {
                Some(Meta::Str(v)) => v.clone(), _ => "chunked".into(),
            };
            // Only the scheme whose mask rule was read from the reference is accepted. Any other
            // limited style still refuses: a near-miss mask produces a fluent WRONG transcript, and
            // that is precisely what this guard exists to prevent.
            if style == "chunked_limited" && cl >= 0 && cr_att >= 0 {
                att_ctx = Some((cl as usize, cr_att as usize));
            } else {
                return Err(format!("parakeet variant limits attention context (style {style:?}, \
                    left {cl}, right {cr_att}); this runtime implements full-context attention and \
                    the `chunked_limited` offline mask. Other limited styles are unimplemented"));
            }
        }
            let cr = match md.get("stt.parakeet.encoder.conv_context_right") {
                Some(Meta::U(v)) => *v as i64, Some(Meta::I(v)) => *v, _ => -1,
            };
            let ck = match md.get("stt.parakeet.encoder.conv_kernel") { Some(Meta::U(v)) => *v as i64, _ => 0 };
            // Two conv shapes are implemented: symmetric (right = k/2) and causal (right = 0).
            // Anything between is a scheme nobody has written down here, so it still refuses.
            if cr >= 0 && ck > 0 && cr != ck / 2 && cr != 0 {
                return Err(format!("conv_context_right is {cr} for kernel {ck}: this runtime \
                    implements the SYMMETRIC (right = k/2 = {}) and CAUSAL (right = 0) depthwise \
                    convs only", ck / 2));
            }
            // ⚠ ONLY WHEN THE FILE DECLARES IT, and from `cr` itself — never derived from `ck`.
            // `ck` is read under the `stt.parakeet.*` key, which an `asr`-naming file does not have,
            // so it reads 0 there; `ck / 2` then silently selected the CAUSAL conv for the offline
            // CTC model and took it from 0.0% to 100% WER. An absent key means "not declared", and
            // the naming-aware fallback below is what knows the real kernel width.
            if cr >= 0 { conv_right = cr as usize; }
            match md.get("stt.parakeet.encoder.conv_norm_type") {
                Some(Meta::Str(v)) if v == "layer_norm" => conv_layernorm = true,
                Some(Meta::Str(v)) if v != "batch_norm" => {
                    return Err(format!("conv_norm_type {v:?} in the conv module: this runtime \
                                        implements batch_norm (folded) and layer_norm only"));
                }
                _ => {}
            }
            if md.get("stt.parakeet.prompt.field").is_some() {
                return Err("prompt-conditioned (multilingual) parakeet: the decoder needs a \
                            language prompt token this runtime does not supply".into());
            }

        let sample_rate = ua("stt.frontend.sample_rate", "asr.preprocessor.sample_rate")?;
        // asr states window/stride in SECONDS; parakeet in SAMPLES. Convert once, here.
        let (win_length, hop_length) = if naming == Naming::Asr {
            ((fa("", "asr.preprocessor.window_size", 0.025) * sample_rate as f32).round() as usize,
             (fa("", "asr.preprocessor.window_stride", 0.010) * sample_rate as f32).round() as usize)
        } else {
            (u("stt.frontend.win_length")?, u("stt.frontend.hop_length")?)
        };
        let ctc = matches!(md.get("asr.head_type"), Some(Meta::Str(h)) if h == "ctc");
        // CTC files have no predictor/joint; those widths are unused and must not be required.
        let (pred_hidden, pred_layers, joint_hidden) = if ctc { (0, 0, 0) } else {
            (u("stt.parakeet.predictor.hidden")?, u("stt.parakeet.predictor.n_layers")?,
             u("stt.parakeet.joint.hidden")?)
        };
        // asr.ctc.num_classes EXCLUDES the blank; the head emits num_classes + 1.
        let vocab = if ctc { u("asr.ctc.num_classes")? + 1 } else { u("stt.parakeet.predictor.vocab")? };
        Ok(Cfg {
            naming, ctc, sample_rate, win_length, hop_length, att_ctx, conv_layernorm,
            conv_right: if conv_right == usize::MAX {
                ua("stt.parakeet.encoder.conv_kernel", "asr.encoder.conv_kernel_size")? / 2
            } else { conv_right },
            n_fft: ua("stt.frontend.n_fft", "asr.preprocessor.n_fft")?,
            num_mels: ua("stt.frontend.num_mels", "asr.preprocessor.features")?,
            f_min: f("stt.frontend.f_min", 0.0),
            f_max: fa("stt.frontend.f_max", "", sample_rate as f32 / 2.0),
            pre_emphasis: fa("stt.frontend.pre_emphasis", "asr.preprocessor.preemph", 0.97),
            n_layers: ua("stt.parakeet.encoder.n_layers", "asr.encoder.n_layers")?,
            d_model: ua("stt.parakeet.encoder.d_model", "asr.encoder.d_model")?,
            n_heads: ua("stt.parakeet.encoder.n_heads", "asr.encoder.n_heads")?,
            d_ff: ua("stt.parakeet.encoder.d_ff", "asr.encoder.d_ff")?,
            conv_kernel: ua("stt.parakeet.encoder.conv_kernel", "asr.encoder.conv_kernel_size")?,
            subsampling_factor: ua("stt.parakeet.encoder.subsampling_factor", "asr.encoder.subsampling_factor")?,
            subsampling_channels: ua("stt.parakeet.encoder.subsampling_channels", "asr.encoder.subsampling_conv_channels")?,
            xscaling: ba("stt.parakeet.encoder.xscaling", "asr.encoder.xscaling", true),
            pred_hidden, pred_layers, joint_hidden, vocab,
            blank_id: match alt("tokenizer.ggml.blank_token_id", "asr.ctc.blank_id") {
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
    pub pw1: Linear, pub dw_b: Tensor, pub pw2: Linear,
    /// Depthwise weights already in the `[C, L]` order the kernel binds — hoisted out of the
    /// forward pass, where re-reading them cost a readback per layer per call.
    pub dw_w_ck: Tensor,
    /// BatchNorm folded to an affine pair at LOAD time: `scale = w/sqrt(var+eps)` and
    /// `shift = b - mean*scale`. Inference BatchNorm is constant, so it never needs its own op.
    ///
    /// For a LayerNorm conv module these are that norm's `weight`/`bias` instead, and `Cfg::
    /// conv_layernorm` says which reading applies. The two are NOT interchangeable: BatchNorm's
    /// affine is applied directly, LayerNorm first normalises each frame across channels.
    pub bn_scale: Tensor, pub bn_shift: Tensor,
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

/// The RNN-T decoder's weights and encoder output, on the HOST. Read once per utterance; the decode
/// that consumes them is sequential CPU work and touches the GPU not at all.
pub struct RnntHost {
    pub encv: Vec<f32>,
    pub lw: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub emb: Vec<f32>,
    pub je: (Vec<f32>, Vec<f32>),
    pub jp: (Vec<f32>, Vec<f32>),
    pub jo: (Vec<f32>, Vec<f32>),
}

/// One stage of the pre-encode subsampling stack: `[kh, kw, c_in, c_out]` and its host-side weights.
pub struct PreConv {
    pub dims: Vec<usize>, pub w: Vec<f32>, pub b: Vec<f32>,
    /// The same weights resident on the GPU: `[kh,kw,c_in,c_out]` and `[c_out]`. Uploaded once at
    /// load — never per call, which is the mistake that deadlocked the browser. The host copies stay
    /// so the CPU path remains available as a differential reference for the GPU one.
    pub wt: Tensor, pub bt: Tensor,
}

pub struct Parakeet {
    pub ctx: Arc<Context>,
    pub cfg: Cfg,
    /// Collapse each encoder block's dispatches into one command buffer. ON by default: 2.3x on
    /// Apple Silicon (9.8 s vs 22.4 s per encode, PAIRED — both arms measured back to back in one
    /// process under identical load).
    ///
    /// ⚠ The pairing is the whole measurement. Comparing separate process launches said the
    /// opposite, because wall-clock for one encode ranged 2.0–5.6 s run to run on a busy machine —
    /// a spread far larger than the effect. Set `FERRIC_ASR_NOBATCH=1` to A/B it on another fabric.
    ///
    /// ⚠ AN ENV VAR IS NOT A TOGGLE IN THE BROWSER. `std::env::var` always returns `Err` on wasm32,
    /// so reading the flag that way pinned this ON in a tab with no way to turn it off — the arm
    /// that most needed measuring was the one that could not be measured. Set it directly instead.
    pub batch_blocks: bool,
    /// The subsampling stack's weights, held ON THE HOST because `pre_encode` is a CPU function.
    ///
    /// ⚠ These used to live as GPU `Tensor`s and be read back on every `encode()`. That is a
    /// DEADLOCK in a browser, not a slow path: `pollster::block_on` waits on a buffer-mapping
    /// future that can only complete when the JS event loop runs, and `block_on` is what stops it
    /// running. The tab pinned forever, identically for 0.25 s and 5.86 s of audio, before a single
    /// encoder block executed. Weights never change, so they are dequantised once, at load.
    pub pre_conv: Vec<PreConv>,
    pub pre_out: Linear,
    pub blocks: Vec<Block>,
    /// RNN-T head. `None` on CTC files, which have no autoregressive decoder at all.
    pub rnnt: Option<(Predictor, Joint)>,
    /// CTC head: one linear from encoder states to vocab+blank. NVIDIA ships it as a kernel-1
    /// Conv1d (`decoder.decoder_layers.0`, dims [1, d, vocab]), which `Linear::load` reads as a
    /// matmul — the same rank-3 case as the conv module's pointwise layers.
    pub ctc_head: Option<Linear>,
    /// The mel filterbank as SHIPPED (`preprocessor.fb`), when the file carries one. The `asr`
    /// converter embeds it, which removes the Slaney-vs-HTK and area-normalisation guesswork
    /// entirely — six frontend conventions I had to recover from a reference for the other format.
    pub fb: Option<Vec<Vec<f32>>>,
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
            let wn = format!("{}.weight", cfg.naming.pre_conv(i));
            if g.tensor(&wn).is_none() { continue; }
            let t = g.tensor(&wn).expect("checked");
            let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
            let o = dims[3];
            let bn = format!("{}.bias", cfg.naming.pre_conv(i));
            let b = if g.tensor(&bn).is_some() { g.dequant(&bn)? } else { vec![0.0; o] };
            let w = g.dequant(&wn)?;
            // ⚠ TWO DIFFERENT LAYOUTS FOR THE SAME NUMBERS. The buffer is `[c_out][c_in][kh][kw]`
            // (kw innermost) — which is what the CPU path indexes as `((o*cin + k)*kh + i)*kw + j`,
            // and it is right because that path scores 0.0% WER. Ferric's `conv2d` and
            // `depthwise_conv2d` both want `[kh][kw][c_in][c_out]` with c_out innermost.
            //
            // Handing the raw buffer over with `dims` attached declares a shape the data does not
            // have: same element count, same rank, every index in range, no assert possible — and
            // the differential check against the CPU path measured a RELATIVE error of 21x. Build it
            // in memory order, then permute once, at load.
            let mem = [dims[3], dims[2], dims[0], dims[1]];            // [c_out, c_in, kh, kw]
            let wt = Tensor::from_vec(ctx, &w, &mem).permute(&[2, 3, 1, 0]).contiguous();
            let bt = Tensor::from_vec(ctx, &b, &[o]);
            pre_conv.push(PreConv { w, b, dims, wt, bt });
        }
        if pre_conv.is_empty() { return Err("no enc.pre_encode.conv.* tensors".into()); }
        let pre_out = Linear::load(ctx, g, cfg.naming.pre_out(), true)?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let b = |s: &str| cfg.naming.t(il, s);
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
                conv: {
                    // Fold BatchNorm to an affine pair HERE, once, instead of reading four vectors
                    // back per layer per forward. `y = (x-mean)/sqrt(var+eps)*w + b` is exactly
                    // `x*scale + shift` with scale = w/sqrt(var+eps), shift = b - mean*scale.
                    let (bw, bb) = (g.dequant(&b("conv.bn.weight"))?, g.dequant(&b("conv.bn.bias"))?);
                    // A LayerNorm conv module ships weight/bias and NO running statistics — there is
                    // nothing to fold, so they pass through as the norm's own parameters.
                    let (scale, shift) = if cfg.conv_layernorm { (bw, bb) } else {
                        let (bm, bv) = (g.dequant(&b("conv.bn.running_mean"))?, g.dequant(&b("conv.bn.running_var"))?);
                        let scale: Vec<f32> = bw.iter().zip(&bv).map(|(w, v)| w / (v + 1e-5).sqrt()).collect();
                        let shift: Vec<f32> = bb.iter().zip(bm.iter().zip(&scale)).map(|(b0, (m, s0))| b0 - m * s0).collect();
                        (scale, shift)
                    };
                    // dw weights are stored [C, L] already (ne0 = 9 is INNERMOST); no transpose.
                    let dwv = g.dequant(&b("conv.depthwise.weight"))?;
                    ConvModule {
                        pw1: Linear::load(ctx, g, &b("conv.pointwise1.weight"), true)?,
                        dw_w_ck: Tensor::from_vec(ctx, &dwv, &[d, cfg.conv_kernel]),
                        dw_b: t1z(&b("conv.depthwise.bias"), &[d])?,
                        bn_scale: Tensor::from_vec(ctx, &scale, &[d]),
                        bn_shift: Tensor::from_vec(ctx, &shift, &[d]),
                        pw2: Linear::load(ctx, g, &b("conv.pointwise2.weight"), true)?,
                    }
                },
                norm_ff2: Norm::load(ctx, g, &b("norm_ff2"), d)?,
                ff2: (Linear::load(ctx, g, &b("ff2.linear1.weight"), true)?,
                      Linear::load(ctx, g, &b("ff2.linear2.weight"), true)?),
                norm_out: Norm::load(ctx, g, &b("norm_out"), d)?,
            });
        }

        // RNN-T head, or none on a CTC file.
        let rnnt = if cfg.ctc { None } else {
            let mut lstm = Vec::with_capacity(cfg.pred_layers);
            for i in 0..cfg.pred_layers {
                let h = cfg.pred_hidden;
                lstm.push(LstmLayer {
                    wx: t1(&format!("pred.lstm.{i}.Wx"), &[4 * h, h])?,
                    wh: t1(&format!("pred.lstm.{i}.Wh"), &[4 * h, h])?,
                    b:  t1(&format!("pred.lstm.{i}.bias"), &[4 * h])?,
                });
            }
            Some((Predictor { embed: t1("pred.embed.weight", &[cfg.vocab, cfg.pred_hidden])?, lstm },
                  Joint {
                      enc: Linear::load(ctx, g, "joint.enc.weight", true)?,
                      pred: Linear::load(ctx, g, "joint.pred.weight", true)?,
                      out: Linear::load(ctx, g, "joint.out.weight", true)?,
                  }))
        };
        let ctc_head = if cfg.ctc {
            Some(Linear::load(ctx, g, "decoder.decoder_layers.0.weight", true)?)
        } else { None };
        // The shipped filterbank, [n_bins, n_mels] on disk → [n_mels][n_bins] to match `filterbank`.
        let fb = match g.tensor("preprocessor.fb") {
            Some(t) => {
                let (nb, nm) = (t.dims[0] as usize, t.dims[1] as usize);
                let v = g.dequant("preprocessor.fb")?;
                Some((0..nm).map(|m| (0..nb).map(|b| v[m * nb + b]).collect()).collect())
            }
            None => None,
        };
        let tokens: Vec<String> = match g.metadata().get(
            if cfg.naming == Naming::Asr { "asr.tokenizer.vocab" } else { "tokenizer.ggml.tokens" }) {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
            _ => return Err("missing tokenizer.ggml.tokens".into()),
        };
        // CTC files list the vocab WITHOUT the blank; RNN-T files include it.
        let want = if cfg.ctc { cfg.vocab - 1 } else { cfg.vocab };
        if tokens.len() != want {
            return Err(format!("{} tokens but the config implies {want}", tokens.len()));
        }
        Ok(Parakeet { ctx: ctx.clone(), cfg, pre_conv, pre_out, blocks, rnnt, ctc_head, fb, tokens,
                      batch_blocks: cfg!(target_arch = "wasm32") || std::env::var("FERRIC_ASR_NOBATCH").is_err() })
    }

    /// A load-time receipt. A schedule that silently collapsed shows up here rather than as a wrong
    /// transcript later — the same reason `nemotron_h::schedule` exists.
    pub fn describe(&self) -> String {
        format!("parakeet · {} conformer blocks · d_model {} · {} heads · d_ff {} · conv_k {} \
                 · subsample {}x · {} mels @ {} Hz · {} {}L LSTM h{} · joint h{} · vocab {} (blank {})",
                self.blocks.len(), self.cfg.d_model, self.cfg.n_heads, self.cfg.d_ff,
                self.cfg.conv_kernel, self.cfg.subsampling_factor, self.cfg.num_mels,
                if self.cfg.ctc { "CTC" } else { "RNN-T" },
                self.cfg.sample_rate,
                self.rnnt.as_ref().map_or(0, |(p, _)| p.lstm.len()), self.cfg.pred_hidden,
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


/// `pollster::block_on`, but it REFUSES on wasm instead of hanging.
///
/// A blocking readback in a browser is not a slow path, it is a deadlock: the buffer-mapping future
/// completes only when the JS event loop runs, and `block_on` is what prevents it from running. The
/// tab pins with no error, no progress and no console output, and it looks exactly like slow
/// inference — which is how one such call in `pre_encode` survived long enough to burn an afternoon.
/// Every readback below is either load-time or behind `FERRIC_ASR_DEBUG`; this makes a future one
/// fail loudly at the first call rather than silently never returning.
#[inline]
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = f;
        panic!("parakeet: blocking GPU readback on wasm32 would deadlock the event loop — \
                use transcribe_ctc_async, or hoist the value to load time");
    }
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(f)
}


/// The `chunked_limited` additive mask as a flat `[T*T]` row-major buffer: `0.0` where a query may
/// attend, `-1e30` where it may not.
///
/// A free function so it can be checked WITHOUT a checkpoint — no streaming model is available here
/// to exercise this path end-to-end, so the mask rule is verified directly rather than taken on
/// trust. See `examples/chunked_mask_check.rs`, which compares it against a transcription of
/// `chunked_limited_mask_function` from `transformers/models/nemotron_asr_streaming` and separately
/// asserts that rule differs from a sliding band — so the check cannot pass while blind to the one
/// error it exists to catch.
pub fn chunked_limited_mask(t: usize, left: usize, right: usize) -> Vec<f32> {
    let chunk = right + 1;
    let left_chunks = left / chunk;
    let mut m = vec![0f32; t * t];
    for q in 0..t {
        let qc = q / chunk;
        for kv in 0..t {
            let d = qc as i64 - (kv / chunk) as i64;
            if d < 0 || d > left_chunks as i64 { m[q * t + kv] = -1e30; }
        }
    }
    m
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
    /// Test hooks: both implementations are private, and the differential check in
    /// `examples/pre_encode_ab.rs` needs to call them side by side. Exposed rather than duplicating
    /// either one in the test, which would let the copy drift from what actually runs.
    pub fn pre_encode_for_test(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize, usize) {
        self.pre_encode(mel, frames)
    }
    pub fn pre_encode_gpu_for_test(&self, mel: &[f32], frames: usize) -> (Tensor, usize, usize) {
        self.pre_encode_gpu(mel, frames)
    }

    /// **`pre_encode` on the GPU.** Returns `[T/8, C*F]`, the same tensor the CPU path flattens to.
    ///
    /// This is the last piece of the speech encoder that ran on the host, and it measured 38-65% of
    /// total encode time — 255-299 ms of 739 ms at 80 mels, 439-527 ms of 738 ms at 128. It stayed
    /// there because `conv2d` asserts `activations.c == weights.c_in`, which a depthwise stage
    /// (`c_in = 1`) can never satisfy; `depthwise_conv2d` is that missing op.
    ///
    /// The stack is `[full] [dw, pw] [dw, pw]` with ReLU after the full conv and after each
    /// pointwise — vec indices 0, 2, 4. ⚠ Those are positions in THIS vec, not GGUF name indices
    /// (the file numbers them 0,2,3,5,6; the gaps were activations in the original Sequential).
    fn pre_encode_gpu(&self, mel: &[f32], frames: usize) -> (Tensor, usize, usize) {
        let c = &self.cfg;
        let (mut h, mut w, mut ch) = (frames, c.num_mels, 1usize);
        let mut cur = Tensor::from_vec(&self.ctx, mel, &[1, h, w, ch]);
        for (idx, pc) in self.pre_conv.iter().enumerate() {
            let (kh, kw, cin, cout) = (pc.dims[0], pc.dims[1], pc.dims[2], pc.dims[3]);
            if kh == 1 && kw == 1 {
                // Pointwise = a matmul over every (t,f) position.
                //
                // ⚠ `pc.wt` was ALREADY permuted to `[kh,kw,c_in,c_out]` at load, so a 1x1 stage is
                // `[1,1,c_in,c_out]` — c_in-major, c_out innermost. That is exactly `[c_in, c_out]`,
                // which plain `matmul` consumes. Reshaping it to `[c_out, c_in]` and calling
                // `matmul_bt` (the layout the RAW file has, before the permute) transposes it: same
                // element count, same output shape, wrong numbers.
                let flat = cur.reshape(&[h * w, ch]);
                let wt = pc.wt.reshape(&[cin, cout]);
                cur = flat.matmul(&wt)
                          .add(&pc.bt.reshape(&[1, cout]).broadcast_to(&[h * w, cout]))
                          .reshape(&[1, h, w, cout]);
                ch = cout;
            } else {
                // `c_in == 1` with multi-channel activations marks DEPTHWISE; `c_in == ch` is the
                // ordinary full convolution (the first stage, 1 -> 256).
                let (oh, ow) = ((h + 1) / 2, (w + 1) / 2);
                cur = if cin == 1 && ch > 1 {
                    cur.depthwise_conv2d(&pc.wt, (2, 2), (1, 1))
                } else {
                    cur.conv2d(&pc.wt, (2, 2), (1, 1))
                }.add(&pc.bt.reshape(&[1, 1, 1, cout]).broadcast_to(&[1, oh, ow, cout]));
                h = oh; w = ow; ch = cout;
            }
            if idx == 0 || idx == 2 || idx == 4 { cur = cur.relu(); }
        }
        // [1,h,w,ch] -> [h, ch*w], CHANNEL-MAJOR. NeMo does `transpose(1,2).reshape(b,t,-1)`, so the
        // channel index is the OUTER one. The other order has the same element count.
        let flat = cur.reshape(&[h, w, ch]).permute(&[0, 2, 1]).contiguous().reshape(&[h, ch * w]);
        (flat, h, ch * w)
    }

    fn pre_encode(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize, usize) {
        let c = &self.cfg;
        let (mut h, mut w, mut ch) = (frames, c.num_mels, 1usize);
        let mut cur = mel.to_vec();                      // [h][w][ch]
        let relu = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| if *x < 0.0 { *x = 0.0 });

        for (idx, pc) in self.pre_conv.iter().enumerate() {
            let wd = &pc.dims;                           // [kh, kw, c_in, c_out]
            let (kh, kw, cin, cout) = (wd[0], wd[1], wd[2], wd[3]);
            let (wv, bv) = (&pc.w, &pc.b);
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
    fn block(&self, x: &Tensor, b: &Block, pos: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let d = self.cfg.d_model;
        // No scalar-multiply op on Tensor; `scalar` makes a [1] tensor to broadcast against.
        let half = |t: &Tensor| t.mul(&t.scalar(0.5).broadcast_to(&t.shape));

        // ---- macaron FF1 ----
        let h = b.norm_ff1.apply(x, Self::EPS);
        let h = b.ff1.1.apply(&b.ff1.0.apply(&h).silu());
        let x = x.add(&half(&h));

        // ---- relative-position self-attention ----
        let h = b.norm_attn.apply(&x, Self::EPS);
        let x = x.add(&self.rel_pos_attn(&h, &b.attn, pos, mask));

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
            let f = |t: &Tensor| { let v = block_on(t.to_vec());
                                   (v.iter().map(|z| z * z).sum::<f32>() / v.len() as f32).sqrt() };
            let bb = block_on(b.norm_out.b.to_vec());
            let br = (bb.iter().map(|z| z * z).sum::<f32>() / bb.len() as f32).sqrt();
            eprintln!("       [last] pre-norm residual rms={:.4}  norm_out bias rms={br:.4}", f(&x));
            // Two implementations of the same LayerNorm. If they agree, the op is right and the
            // 0.044 is what these weights genuinely produce — meaning my expectation (output ~
            // rms(w)) is what is wrong, because rms(w) is dominated by a few large entries while
            // most are small. If they disagree, the GPU op is wrong at this shape.
            let xv = block_on(x.to_vec());
            let wv = block_on(b.norm_out.w.to_vec());
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

    /// The `chunked_limited` additive attention mask, `[T, T]`, or `None` for full context.
    ///
    /// ⚠ CHUNK-WISE, NOT A SLIDING WINDOW. Transcribed from the reference
    /// (`transformers/models/nemotron_asr_streaming`, `chunked_limited_mask_function`):
    /// ```text
    /// chunk_size          = right + 1
    /// left_context_chunks = left / chunk_size
    /// allowed(q, kv)      = 0 <= (q/chunk_size - kv/chunk_size) <= left_context_chunks
    /// ```
    /// A sliding band `[i-left, i+right]` has the same flavour, the same shape and the same element
    /// count — and is a different model. Within its own chunk a query sees the WHOLE chunk, future
    /// frames included; that lookahead IS the right context. Guessing here would have produced a
    /// fluent wrong transcript, so the rule was read rather than inferred.
    ///
    /// `-1e30` rather than `-inf`: softmax subtracts the row max first, and `-inf - -inf` is NaN.
    /// Every row keeps its own chunk (`chunk_diff == 0` is always allowed), so no row is fully
    /// masked — but a finite floor costs nothing and removes the failure mode entirely.
    fn att_mask(&self, t: usize) -> Option<Tensor> {
        let (left, right) = self.cfg.att_ctx?;
        Some(Tensor::from_vec(&self.ctx, &chunked_limited_mask(t, left, right), &[t, t]))
    }

    fn rel_pos_attn(&self, x: &Tensor, a: &RelPosAttn, pos: &Tensor, mask: Option<&Tensor>) -> Tensor {
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
            // ⭐ ON-DEVICE. This was a readback PER HEAD PER LAYER — 336 GPU→CPU syncs for a
            // 42-layer 8-head encoder, the native latency ceiling and an absolute wasm blocker
            // (a browser cannot block on a readback). `Tensor::rel_shift` does the same gather,
            // `out[i,j] = bd[i, (T-1)-i+j]`, without leaving the device.
            let bd = bd.rel_shift();
            let sum = ac.add(&bd);
            let scaled = sum.mul(&sum.scalar(scale).broadcast_to(&sum.shape));
            // The reference scales the position term and then fills -inf, so the mask lands on the
            // SCALED logits — masking before the scale would multiply the floor by `scale`.
            let scaled = match mask { Some(m) => scaled.add(m), None => scaled };
            let s = scaled.softmax(1);
            heads.push(s.matmul(&v_h));                     // [t, hd]
        }
        let cat = heads.iter().skip(1).fold(heads[0].clone(), |a, b| a.cat(b, 1));
        a.out.apply(&cat.reshape(&[t, self.cfg.d_model]))
    }

    /// Conformer convolution module: pointwise → GLU → depthwise → batch-norm → SiLU → pointwise.
    fn conv_module(&self, x: &Tensor, c: &ConvModule) -> Tensor {
        let (t, d) = (x.shape[0], self.cfg.d_model);
        let k = self.cfg.conv_kernel;
        // GLU on-device: the first half gates on the sigmoid of the second. `narrow` + `sigmoid` +
        // `mul` are all GPU ops, so no readback — the host loop this replaces ran once per layer.
        let y = c.pw1.apply(x);                                     // [t, 2d]
        let g = y.narrow(1, 0, d).contiguous()
                 .mul(&y.narrow(1, d, d).contiguous().sigmoid());   // [t, d]

        // ⚠ SYMMETRIC, not causal. conv_context_left == conv_context_right == k/2, so the window is
        // centred. Ferric has only a CAUSAL depthwise conv1d, and y_sym[t] = y_causal[t+k/2] once
        // the signal is right-padded — so the existing kernel serves, with a shift, and the pad is
        // a `cat` with zeros rather than a host round-trip.
        // Symmetric: right-pad by k/2 and drop the first k/2 outputs, so the causal kernel's window
        // lands centred. Causal (streaming): the kernel is ALREADY what the model wants — no pad, no
        // shift. Same kernel, two framings; the file says which.
        let pad = self.cfg.conv_right;
        let conv = if pad == 0 {
            g.depthwise_conv1d_causal(&c.dw_w_ck, k)
        } else {
            let zeros = Tensor::from_vec(&self.ctx, &vec![0f32; pad * d], &[pad, d]);
            g.cat(&zeros, 0).depthwise_conv1d_causal(&c.dw_w_ck, k).narrow(0, pad, t).contiguous()
        };

        // BatchNorm at inference is affine over the running statistics, so it folds into ordinary
        // broadcast arithmetic: ((x + dw_bias) - mean) * inv_std * weight + bias, then SiLU.
        let row = |v: &Tensor| v.reshape(&[1, d]).broadcast_to(&[t, d]);
        let z = conv.add(&row(&c.dw_b));
        // ⚠ MULTIPLY THEN ADD. `(z + shift) * scale` has the same shapes, the same element count and
        // the same op count as `z * scale + shift` — and scales the bias by inv_std. It produced a
        // silent empty transcript, no assert, no NaN. Fold arithmetic is order-bearing.
        //
        // LayerNorm is NOT that affine with different numbers: it first centres and scales each
        // frame across channels. Applying the folded-BatchNorm path to a LayerNorm checkpoint would
        // run, and be wrong.
        let n = if self.cfg.conv_layernorm {
            Norm { w: c.bn_scale.clone(), b: c.bn_shift.clone() }.apply(&z, Self::EPS)
        } else {
            z.mul(&row(&c.bn_scale))         // inv_std * w
             .add(&row(&c.bn_shift))         // b - mean * inv_std * w, precomputed at load
        };
        c.pw2.apply(&n.silu())
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
        // ⚠ NATIVE ONLY. `Instant::now()` PANICS on wasm32 ("time not implemented on this
        // platform") — it is not a no-op and not a zero. This probe, added to attribute encode time,
        // is what broke browser speech while every native test stayed green: the tab panicked before
        // the first conv. A diagnostic must not be able to take the product down on a platform it
        // was never meant to run on.
        #[cfg(not(target_arch = "wasm32"))]
        let _t_pre = std::time::Instant::now();
        // GPU by default; `FERRIC_ASR_CPU_PRE=1` selects the host implementation it was ported from,
        // which stays as the differential reference (`examples/pre_encode_ab.rs`).
        let cpu_pre = cfg!(not(target_arch = "wasm32")) && std::env::var("FERRIC_ASR_CPU_PRE").is_ok();
        let (x, t, width) = if cpu_pre {
            let (flat, t, width) = self.pre_encode(&mel, frames);
            (Tensor::from_vec(&self.ctx, &flat, &[t, width]), t, width)
        } else {
            self.pre_encode_gpu(&mel, frames)
        };
        // Attribute the encode: the subsampling stack used to run on the CPU and scale with
        // num_mels, which was 38-65% of total encode time — 255-299 ms of 739 ms at 80 mels,
        // 439-527 ms of 738 ms at 128. Kept as a probe so a regression back onto the host is visible.
        #[cfg(not(target_arch = "wasm32"))]
        let pre_ms = _t_pre.elapsed().as_secs_f64() * 1000.0;
        if t == 0 { return Err("audio too short to survive 8x subsampling".into()); }
        let _ = width;
        let mut x = self.pre_out.apply(&x);
        if self.cfg.xscaling {
            let s = (self.cfg.d_model as f32).sqrt();
            x = x.mul(&x.scalar(s).broadcast_to(&x.shape));
        }
        let pos = self.rel_pos_encoding(t);
        // Once per encode, not per layer: it depends only on T and the declared context.
        let mask = self.att_mask(t);
        let dbg = std::env::var("FERRIC_ASR_DEBUG").is_ok();
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("FERRIC_ASR_TIME").is_ok() {
            eprintln!("       pre_encode({}) {pre_ms:.0} ms for {frames} frames x {} mels -> t={t}",
                      if cpu_pre { "CPU" } else { "GPU" }, self.cfg.num_mels);
        }
        let rms = |t: &Tensor| { let v = block_on(t.to_vec());
                                 (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() };
        if dbg { eprintln!("       pre_encode out rms={:.4} (after xscaling)", rms(&x)); }
        for (i, b) in self.blocks.iter().enumerate() {
            // ⭐ ONE COMMAND BUFFER PER BLOCK. `batch()` defers dispatches into a single submit, and
            // `readback()` flushes it — so while the encoder read tensors back per head per layer,
            // batching was a no-op by construction. Removing those readbacks is what made this
            // possible; the two changes only pay off together.
            //
            // ⚠ PER BLOCK, NOT PER ENCODER. A batch retains every intermediate buffer until it
            // flushes, and one wrapper around all 42 layers holds ~1 GB of [t, 2t-1] score matrices
            // alone (8 heads x 42 layers x 2.7 MB) plus everything else. That exhausted the device
            // and surfaced as `Buffer with 'staging' label is invalid` from the NEXT readback —
            // an allocation failure reported at an unrelated call, several ops later.
            x = if self.batch_blocks { ferric_tensor::batch(&self.ctx, || self.block(&x, b, &pos, mask.as_ref())) }
                else { self.block(&x, b, &pos, mask.as_ref()) };
            // Where does the signal die? LayerNorm's eps floors any block whose residual variance
            // falls below ~1e-5, so a collapse shows as a step change here, not a gradual decay.
            if dbg {
                // Print the block's OWN norm_out weight beside its output: LayerNorm output should
                // track its weight, so the two diverging is the signature of an eps-floored input.
                let w = block_on(b.norm_out.w.to_vec());
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
        if let Some(head) = &self.ctc_head { return self.decode_ctc(&enc, head); }
        self.decode_rnnt(&enc)
    }

    /// **CTC greedy decode**: argmax per frame, collapse runs of the same id, drop blanks.
    ///
    /// The collapse is over CONSECUTIVE frames, not the whole sequence — a genuine repeated letter
    /// is separated by a blank frame, which is exactly what the blank is for. Deduplicating
    /// globally would turn "little" into "litle".
    fn decode_ctc(&self, enc: &Tensor, head: &Linear) -> Result<String, String> {
        let logits = head.apply(enc);
        let v = block_on(logits.to_vec());
        self.collapse_ctc(&v, enc.shape[0])
    }

    /// The async half of the CTC path, for wasm — where `block_on` is not merely slow but illegal
    /// on the main thread. The encoder itself no longer reads anything back, so this ONE await is
    /// the entire GPU→CPU boundary of browser speech recognition.
    pub async fn transcribe_ctc_async(&self, pcm: &[f32]) -> Result<String, String> {
        let head = self.ctc_head.as_ref().ok_or("this model has no CTC head")?;
        let enc = self.encode(pcm)?;
        let v = head.apply(&enc).to_vec().await;
        self.collapse_ctc(&v, enc.shape[0])
    }

    /// Greedy argmax + run collapse. Shared by the sync and async entry points so there is exactly
    /// one implementation of the decode rule — two copies would be free to drift apart silently,
    /// and only one of them is exercised by the native tests.
    fn collapse_ctc(&self, v: &[f32], t: usize) -> Result<String, String> {
        let nv = self.cfg.vocab;
        let mut out: Vec<u32> = Vec::new();
        let mut prev = u32::MAX;
        for i in 0..t {
            let row = &v[i * nv..(i + 1) * nv];
            let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            if best != prev && best != self.cfg.blank_id { out.push(best); }
            prev = best;
        }
        Ok(out.iter()
            .map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default().replace('▁', " "))
            .collect::<String>().trim().to_string())
    }

    fn decode_rnnt(&self, enc: &Tensor) -> Result<String, String> {
        let host = block_on(self.rnnt_host(enc))?;
        self.decode_rnnt_with(&host)
    }

    /// Every GPU→CPU read the RNN-T decoder needs, gathered in one place and AWAITED.
    ///
    /// The decode itself is host-side and sequential — an LSTM over emitted tokens — so the only
    /// thing standing between it and a browser was `block_on`, which deadlocks the event loop it is
    /// waiting on. Splitting the reads out makes the same decode reachable from an async caller
    /// without duplicating a line of the decode rule.
    async fn rnnt_host(&self, enc: &Tensor) -> Result<RnntHost, String> {
        let (pred_w, joint_w) = self.rnnt.as_ref().ok_or("no RNN-T head")?;
        let encv = enc.to_vec().await;
        let mut lw = Vec::with_capacity(pred_w.lstm.len());
        for l in &pred_w.lstm {
            lw.push((l.wx.to_vec().await, l.wh.to_vec().await, l.b.to_vec().await));
        }
        let jb = |b: &Option<Tensor>| b.as_ref().expect("joint layers carry biases").clone();
        Ok(RnntHost {
            encv,
            lw,
            emb: pred_w.embed.to_vec().await,
            je: (joint_w.enc.w.to_vec().await, jb(&joint_w.enc.b).to_vec().await),
            jp: (joint_w.pred.w.to_vec().await, jb(&joint_w.pred.b).to_vec().await),
            jo: (joint_w.out.w.to_vec().await, jb(&joint_w.out.b).to_vec().await),
        })
    }

    /// **RNN-T greedy decode in a browser.** One await for the weights, then the same host decode.
    pub async fn transcribe_rnnt_async(&self, pcm: &[f32]) -> Result<String, String> {
        if self.rnnt.is_none() { return Err("this model has no RNN-T decoder".into()); }
        let enc = self.encode(pcm)?;
        let host = self.rnnt_host(&enc).await?;
        self.decode_rnnt_with(&host)
    }

    fn decode_rnnt_with(&self, host: &RnntHost) -> Result<String, String> {
        let (pred_w, joint_w) = self.rnnt.as_ref().ok_or("no RNN-T head")?;
        let d = self.cfg.d_model;
        let RnntHost { encv, lw, emb, je: (je_w, je_b), jp: (jp_w, jp_b), jo: (jo_w, jo_b) } = host;
        let t = encv.len() / d;
        let h = self.cfg.pred_hidden;
        let nl = pred_w.lstm.len();
        let jh = self.cfg.joint_hidden;

        if std::env::var("FERRIC_ASR_DEBUG").is_ok() {
            let rms = (encv.iter().map(|x| x * x).sum::<f32>() / encv.len() as f32).sqrt();
            let nf = encv.iter().filter(|x| !x.is_finite()).count();
            // Is 0.044 the LayerNorm weight, or a collapsed encoder? The final norm's own scale
            // answers that: LN output rms ~= rms(weight), so if they match the encoder is behaving.
            let lnw = block_on(self.blocks.last().unwrap().norm_out.w.to_vec());
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
                        self.lstm_step(&pred_w.lstm[li], &inp, &mut hh, &mut cc, wx, wh, b);
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
