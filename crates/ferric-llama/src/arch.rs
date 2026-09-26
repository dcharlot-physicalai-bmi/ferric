//! **Which checkpoints does this runtime actually run?** — the registry, and the refusal.
//!
//! Ferric commits to supporting current releases within 30 days, the same cadence llama.cpp, vLLM,
//! MLX and Ollama hold. A cadence is a process requirement, and the process needs one thing above all:
//! the gap has to be **visible continuously**, not discovered when someone points a new GGUF at the
//! server. This module is that visibility.
//!
//! ## The failure it exists to stop
//!
//! Before this, dispatch was:
//!
//! ```text
//!     if arch.starts_with("qwen35") || arch == "laguna" { Hybrid } else { Dense }
//! ```
//!
//! The `else` is a catch-all. Point it at a `gemma4`, `deepseek2`, `glm4`, `minimax` or `hunyuan`
//! checkpoint and it does not fail — it loads the file as a dense Qwen3, reads whichever metadata keys
//! happen to share names, defaults the rest, and generates **fluent, confident, wrong** text. Nothing
//! errors. No test catches it, because the code runs.
//!
//! The same shape appears in [`crate::qwen3::Cfg::from_gguf`], where `arch.starts_with("gemma")` is
//! true for `gemma4` — a 2026 architecture silently inheriting Gemma-2/3 assumptions about sliding
//! window pattern, embedding scale and softcapping.
//!
//! So: **an architecture this runtime has not been taught is an error, not a default.** [`resolve`]
//! returns [`ArchError::Unsupported`] and names what would have to be written. A runtime that refuses
//! is one you can trust the output of.
//!
//! ## Status is not a boolean
//!
//! "Supported" hides the distinction that matters. A loader can be written, tested, correct, and
//! produce wrong text because one convention differs from the reference — so this registry separates
//! *runs* from *was checked against the reference implementation*. See [`Status`].

use std::fmt;

/// Which runtime in this crate serves an architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    /// [`crate::qwen3`] — dense GQA transformer family.
    Dense,
    /// [`crate::qwen35`] — gated-delta-net / attention hybrid, dense or MoE FFN.
    Hybrid,
    /// [`crate::lfm2`] — short-conv / attention hybrid with real conv state.
    Lfm2,
    /// [`crate::gemma4`] — per-layer embeddings, shared KV, two head widths.
    Gemma4,
    /// [`crate::deepseek2`] — multi-head latent attention + DeepSeekMoE.
    DeepSeek2,
    /// [`crate::cosmos`] — loads from safetensors rather than GGUF.
    Cosmos,
    /// [`crate::bert`] — encoder-only. Embeddings and rerankers, not generation.
    Bert,
    /// [`crate::modern_bert`] — encoder-only, but structurally unlike [`Runtime::Bert`]: RoPE with
    /// TWO bases, alternating symmetric-band / global attention, pre-LayerNorm without bias, GeGLU.
    ModernBert,
    /// [`crate::nemotron_h`] — Mamba-2 state-space mixers with a few attention layers.
    NemotronH,
    /// [`crate::parakeet`] — Conformer encoder + RNN-T decoder. SPEECH: waveform in, text out.
    /// Not a generative text runtime; `ferric-serve` refuses it the way it refuses `Bert`.
    Parakeet,
    /// [`crate::hyv4`] — hyper-connections, gated MLA with a learnable sink, the DSA lightning
    /// indexer, and DeepSeekMoE with a clamped SwiGLU.
    ///
    /// ⛔ This row previously named `DeepSeek2` as a PLACEHOLDER, and that was a live mis-dispatch
    /// waiting on one edit: `ferric-serve` matches on this field, so promoting hyv4's status would
    /// have loaded a hyv4 checkpoint AS A DEEPSEEK2 MODEL — the exact "fluent, confident, wrong
    /// text" failure the registry exists to prevent. Only `Status::Untried` stood in the way.
    Hyv4,
}

impl Runtime {
    pub fn label(self) -> &'static str {
        match self {
            Runtime::Bert => "bert",
            Runtime::ModernBert => "modern_bert",
            Runtime::NemotronH => "nemotron_h",
            Runtime::Parakeet => "parakeet",
            Runtime::Dense => "dense",
            Runtime::Hybrid => "hybrid",
            Runtime::Lfm2 => "lfm2",
            Runtime::Gemma4 => "gemma4",
            Runtime::DeepSeek2 => "deepseek2",
            Runtime::Cosmos => "cosmos",
            Runtime::Hyv4 => "hyv4",
        }
    }
}

/// How far a given architecture has actually been taken.
///
/// The ordering is by trustworthiness, and the gap between [`Status::Loads`] and [`Status::Verified`]
/// is where every silent wrong-output bug in this codebase has lived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    /// Output was compared against the reference implementation on real weights and matched.
    Verified,
    /// Loads and generates coherent text, but has not been diffed against the reference. A wrong RoPE
    /// convention or a missed norm produces exactly this: plausible output, no error.
    Loads,
    /// The hard components exist as verified library code, but no GGUF loader wires them to this
    /// architecture's metadata and tensor names.
    Parts,
    /// A complete loader and forward exist and run end to end on a SYNTHETIC checkpoint, but no
    /// real one has ever been loaded — because none fits on the machine that wrote the code.
    ///
    /// This is a real category, not a hedge. A synthetic model proves the WIRING: that the tensor
    /// names resolve, the shapes agree end to end, the block schedule composes. It cannot prove
    /// fidelity, because the same conventions used to write the file are used to read it back. So
    /// this is strictly more than [`Status::Parts`] and strictly less than [`Status::Loads`], and
    /// collapsing it into either would misreport what is known.
    ///
    /// Not [`Status::runnable`]: a server must not serve a model whose output nobody has seen.
    Untried,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Verified => "verified",
            Status::Loads => "loads",
            Status::Parts => "parts",
            Status::Untried => "untried",
        }
    }
    /// Whether a user may point the server at this and trust what comes out.
    pub fn runnable(self) -> bool { matches!(self, Status::Verified | Status::Loads) }
}

/// One `general.architecture` value and what this runtime does with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    /// The exact `general.architecture` string in the GGUF. Matched exactly — never by prefix, which
    /// is how `gemma4` inherited Gemma-3's behaviour.
    pub name: &'static str,
    pub runtime: Runtime,
    pub status: Status,
    /// What remains to be done, or what was checked. Written for whoever picks this up next.
    pub note: &'static str,
}

/// Every architecture this runtime knows, and nothing it does not.
///
/// Adding a row here without a loader makes [`coverage`] honest and [`resolve`] still refuse, because
/// [`Status::Parts`] is not [`Status::runnable`]. That is deliberate: the registry is allowed to
/// describe work in progress, and is not allowed to let it serve traffic.
pub const REGISTRY: &[Arch] = &[
    // ---- dense GQA family ----------------------------------------------------------------
    Arch { name: "nemotron_h", runtime: Runtime::NemotronH, status: Status::Verified,
           note: "Mamba-2 / attention / MLP hybrid — the FIRST non-transformer runtime here. 42 blocks: \
                  21 state-space mixers, 17 ReLU^2 MLPs, 4 attention, and NO rotary anywhere. \
                  ⭐ Verified against the MODEL AUTHORS' OWN FILE — the modeling_nemotron_h.py NVIDIA ships \
                  in nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (transformers 5.7.0, float32, eager), on F32 \
                  converted from their weights: max |logit diff| 4.0e-5 over 138 positions x 128 sampled \
                  ids, argmax 138/138, full-row sum of squares within 8.2e-6. Normalising the SSM output \
                  over one group instead of 8 is 204,353x worse. It was first checked against llama.cpp \
                  on a Q4_K_M file, logprobs to ~0.01, which could not have seen what follows. \
                  ⛔⛔ THE REFERENCE SIDE HAD THREE DEFECTS, each of which first read as Ferric's: \
                  (1) transformers' BUILT-IN port floors dt at time_step_min (0.001, an init range) where \
                  the authors clamp to (0, inf): 0.27 off; (2) the authors' own CPU fallback TILES heads \
                  over B/C groups where their CUDA kernels index them contiguously (pid_h // ratio), \
                  corrected in the generator by --fix-group-tiling; (3) under transformers 5.x their \
                  remote code re-initialises dt_bias and out_proj in all 21 mixers AFTER loading (42 \
                  tensors, 0 'missing keys'), so every reference parameter is now checked BY VALUE \
                  against the file. Corrected, the authors' file and the built-in port (floor removed) \
                  agree to 5.0e-5: two paths, one documented correction each. \
                  Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/nemotron3-nano-4b.json. \
                  INCREMENTAL STATE WORKS and is the architecture's whole argument: 85.4 MB of \
                  conv+SSM+KV that does NOT grow with the conversation, against a transformer's KV \
                  cache that does. Verified by EQUALITY — a cache bug drifts rather than raising, so \
                  the reference-checked stateless path stays and the cached one must reproduce it \
                  token for token, which it does. ⚠ conv state is the PRE-convolution signal, not the \
                  conv output; storing the output drifts plausibly instead of failing" },
    // ---- speech ---------------------------------------------------------------------------
    //
    // Ferric's first non-text modality. Two of the top-40 most-downloaded GGUF repos share this
    // arch (parakeet-unified-en-0.6b, nemotron-3.5-asr-streaming-0.6b) and neither could be loaded.
    // NVIDIA's own converter emits `asr`; the community one emits `parakeet`. Same architecture,
    // different tensor names and key namespace — one `Naming` map holds every difference so this
    // stays one runtime rather than two loaders.
    Arch { name: "asr", runtime: Runtime::Parakeet, status: Status::Verified,
           note: "NVIDIA NeMo ASR export (parakeet-ctc-1.1b): the same Conformer encoder with a CTC \
                  head instead of RNN-T. Waveform in, text out — NOT a chat model. ⭐ Verified against \
                  the MODEL AUTHORS' implementation, NVIDIA NeMo 3.0.0, STAGE BY STAGE (log-mel, mel, \
                  subsampling, projection, positional table, every one of 42 blocks, encoder, raw head \
                  logits, tokens) on 4 LibriSpeech clips, NeMo running this file's own dequantised \
                  weights: encoder within 1.6e-4 of its rms, head logits within 9.2e-4, argmax equal on \
                  every frame. Every tensor audited by value: all 33,112,064 Q8_0 blocks are ggml's \
                  Q8_0 of the authors' fp32 checkpoint, bit for bit. ⚠ That rounding is not small: it \
                  moves the encoder by 0.10-0.38 of its rms against the authors' own weights (tokens \
                  unchanged on the 4 clips). ⚠ The checkpoint was trained under NeMo 1.19 (reflect STFT \
                  padding, n/hop+1 frames); the reference is the NeMo NVIDIA ships today, and the 1.19 \
                  frontend is reported, not gated. The file's precomputed pos_enc.pe and preprocessor.fb \
                  are not read: Ferric builds both, as NeMo does. \
                  Gate: scripts/parakeet_conformance.sh vs tests/fixtures/parakeet/" },
    Arch { name: "parakeet", runtime: Runtime::Parakeet, status: Status::Verified,
           note: "NVIDIA Parakeet / Nemotron-ASR: Conformer encoder + RNN-T. Waveform in, text out: NOT a chat model, \
                  and ferric-serve refuses it the way it refuses bert. ⭐ Verified against the MODEL \
                  AUTHORS' implementation, NVIDIA NeMo 3.0.0, STAGE BY STAGE on \
                  parakeet-unified-en-0.6b (F16, every tensor bit-exact f16 of the authors' checkpoint) \
                  over 4 LibriSpeech clips: encoder within 1.5e-5 of its rms (NeMo's own fp32-vs-fp64 \
                  floor is 4e-5 to 6.5e-5), joint logits TEACHER-FORCED along NeMo's own greedy path \
                  within 1.0e-4, argmax equal at all 1,392 joint calls, tokens identical. It used to be \
                  'Verified' on transcripts alone, and the transcripts were right while the frame \
                  semantics were not: Ferric counted every STFT frame where NeMo counts n/hop, \
                  normalised over the extra one, masked nothing and decoded an encoder frame NeMo \
                  treats as padding — 0.13-1.4x the encoder rms inside NeMo, tokens unchanged. Fixed. \
                  26 negative controls (mel scale, window, padding, rel_shift, LSTM gate order, SOS, \
                  blank rule, ...) each fail first at the stage their mechanism lives in. \
                  ⭐ nemotron-3.5-asr-streaming-0.6b (same arch string; 13088-token multilingual) is \
                  Verified the same way: limited-context attention (the file declares (56,13); NeMo \
                  restores at (56,3) — both checked, set_att_context selects), causal conv and causal \
                  SUBSAMPLING (CausalConv2D pads 2 left / 1 right on both axes: 128 mels -> 17 bins, a \
                  4352-wide projection), LayerNorm conv module, the RAW log-mel (NeMo's normalize: NA) \
                  and the language PROMPT (one-hot after the encoder, Linear-ReLU-Linear; auto = 101, \
                  set_target_lang). Encoder within 2.3e-5 of rms; the prompt output at or below \
                  NeMo's own fp32-vs-fp64 floor there; joint logits within 7.1e-5 of each row's scale; \
                  argmax equal at all 2,851 joint calls and tokens identical on 7 fixtures, incl. \
                  (56,3) and en-US. 30-31 controls each, incl. the prompt (off / first / wrong \
                  index), the context mask and causal subsampling. Offline only: streaming needs the \
                  cache-aware encoder, which NeMo shows gives the same tokens. \
                  Gate: scripts/parakeet_conformance.sh vs tests/fixtures/parakeet/" },

    Arch { name: "bert", runtime: Runtime::Bert, status: Status::Verified,
           note: "encoder-only: bidirectional, learned positions, post-LayerNorm, GELU FFN, no KV cache and no LM \
                  head. ⭐ Verified against the MODEL AUTHORS' implementation (transformers 5.7.0, float32, \
                  eager), NOT llama.cpp: bge-small-en-v1.5 F16 first-token row within 1.2e-6 to 1.4e-6 over \
                  9-254 tokens; XLM-R cross-encoder bge-reranker-v2-m3 (F16 converted from the authors' \
                  weights) scores EXACT to 4 dp. ⛔⛔ GELU is the authors' exact erf form: it defaulted to \
                  ggml's tanh approximation to match llama.cpp, which was ~1000x further from the model \
                  (1.4e-3 to 2.8e-3). The GGUF records no activation, so a tanh-trained checkpoint needs \
                  FERRIC_BERT_GELU_TANH=1. ⚠ A Q4_K_M file cannot verify the math — the reranker at 4 bits \
                  differs by 0.04-0.31 with either GELU. Gate: scripts/bert_conformance.sh vs \
                  tests/fixtures/bert/. EMBEDS AND SCORES — generation is refused, there is no LM head" },
    Arch { name: "modern-bert", runtime: Runtime::ModernBert, status: Status::Verified,
           note: "encoder-only and NOT the BERT above: RoPE (NeoX), PRE-LayerNorm with no bias, GeGLU \
                  over a fused {d, 2*n_ff} up, one fused qkv, layer 0's attn_norm absent (identity), and \
                  alternating SYMMETRIC-band / global attention on a dense-first 1-in-3 schedule. \
                  ⭐ Verified against the MODEL AUTHORS' implementation (transformers 5.7.0, float32, \
                  eager) on Alibaba-NLP/gte-reranker-modernbert-base — NOT against llama.cpp: encoder \
                  within 2.5e-6 over 222 tokens and head scores EXACT to 4 dp on F32 weights converted \
                  from the authors' own F32 files (1.1e-3 / 0.0003 on the third-party F16 file = rounding). Disabling the \
                  window is 297x worse and reading ONE rope base 74x worse, so both mechanisms are \
                  load-bearing. ⛔⛔ The head POOLS PER THE CHECKPOINT: this one's config says \
                  classifier_pooling=mean, the GGUF converter drops the key, and transformers' own \
                  default is cls. Ferric first pooled the first token, matched llama.cpp to 0.03-0.10, \
                  and that gap was misread as F16 precision; against the authors it was the pooling. \
                  llama.cpp is right here only by hardcoding mean for every modern-bert reranker. \
                  GELU is the authors' exact erf form in the encoder AND the head — tanh (llama.cpp's choice) was \
                  ~3800x further on F32. Head LayerNorm, as the authors' ModernBertPredictionHead. \
                  ⛔⛔ TWO ROPE BASES (160000 global / 10000 sliding here; identical on mmBERT-base). \
                  Gate: scripts/modern_bert_conformance.sh vs tests/fixtures/modern_bert/. \
                  EMBEDS AND SCORES — generation is refused, there is no LM head to generate from." },
    Arch { name: "qwen2", runtime: Runtime::Dense, status: Status::Verified,
           note: "the family this runtime was written against. ⭐ Verified against the MODEL AUTHORS' \
                  implementation (transformers 5.7.0, float32, eager) on Qwen/Qwen2.5-0.5B-Instruct, F32 \
                  converted from their files: max |logit diff| 2.6e-4 over 139 positions x 128 sampled \
                  ids, argmax 139/139, full-row sum of squares within 1.3e-5. The wrong rope pairing is \
                  52,151x worse. The earlier check compared greedy tokens only, which a small logit \
                  error almost never moves. Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/" },
    Arch { name: "qwen3", runtime: Runtime::Dense, status: Status::Verified,
           note: "per-head QK RMSNorm. ⭐ Verified against the MODEL AUTHORS' implementation \
                  (transformers 5.7.0, float32, eager) on Qwen/Qwen3-0.6B, F32 converted from their \
                  files: max |logit diff| 7.0e-5 over 139 positions x 128 sampled ids, argmax 139/139, \
                  full-row sum of squares within 4.3e-6. The wrong rope pairing is 222,647x worse. \
                  Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/" },
    // Qwen3-VL-8B-Instruct is the #2 most-downloaded model on Hugging Face (18.4M/30d, 2026-09).
    // ⚠ TEXT-ONLY, and that is not a hedge — it is what llama.cpp does. `n_embd_inp` is
    // `n_embd * (1 + n_deepstack_layers)`, but the token path ZERO-PADS to that width
    // (`llama-graph.cpp`: `ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0,0,0)`), so every
    // deepstack injection adds exactly 0 for a text token. Ferric therefore omits deepstack entirely
    // on this path and is arithmetically identical, NOT approximate.
    // ⛔ What is missing is images: the vision tower, the merger, and the deepstack rows it produces.
    // Feeding an image would need all three, so the loader must not pretend otherwise.
    Arch { name: "qwen3vl", runtime: Runtime::Dense, status: Status::Loads,
           note: "TEXT-ONLY. qwen3 + interleaved multimodal rope (rope.dimension_sections [24,20,20]). \
                  Text tokens set t=h=w=pos, e=0, so deepstack contributes zero and is omitted — \
                  identical, not approximate. ⛔ No vision tower yet: images are unsupported. \
                  ⚠ Ferric follows llama.cpp's sector rule, which leaves sectors 61/62 unrotated \
                  where HF rotates them (~4e-4 rad @pos 1000) — see Tensor::rope_mrope" },
    // MiMo-Embodied-7B (Xiaomi, MIT): driving + embodied-robotics VLM, arch Qwen2_5_VLForConditionalGeneration.
    // Served straight from the authors' safetensors (ferric_load::hf maps model_type qwen2_5_vl -> qwen2vl);
    // the vision tower is crate::qwen25vl_vision. The name is llama.cpp's for the same text model.
    Arch { name: "qwen2vl", runtime: Runtime::Dense, status: Status::Verified,
           note: "qwen2 (q/k/v biases) + CHUNKED multimodal rope (mrope_section [16,24,24]) + the \
                  Qwen2.5-VL vision tower: 28 of 32 blocks attend in 112-px windows, rows reordered into \
                  window order and back, RMSNorm, SwiGLU with biases, one merger. ⭐ Verified against the \
                  MODEL AUTHORS' implementation (transformers 5.7.0 modeling_qwen2_5_vl, float32, eager) \
                  on XiaomiMiMo/MiMo-Embodied-7B from the image FILE to the logits, on the authors' exact \
                  weights (tower F32, text BF16 kept 16-bit): pixels 5e-7, window order and mRoPE positions \
                  identical, vision blocks 0-8 within 4e-6, logits within 4x the authors' OWN float32 \
                  distance from float64 at every one of 83 positions, argmax 83/83, and a 64-token greedy \
                  decode identical to the authors' argmax chain. ⛔ From block 17 a few tokens carry \
                  massive activations (5e4) from cancelling sums: at block 31 the authors' f32 run is 7e-4 \
                  (ssq) from float64 and Ferric 1.7e-3 — rounding on both sides, so the gate measures against \
                  float64, not against f32. Controls: no windows 32,415x, all windowed 92,672x, rope \
                  row/col swap 6,299x, no reverse 10,103x, 1-D image positions 1,780x, interleaved \
                  sectors 1,449x; decode by cache index diverges at step 56. ⚠ tanh-vs-erf GELU in the \
                  merger is below the drift (1.4x) and rests on the code. ONE still image per prompt; \
                  video refused. Gate: scripts/vl_conformance.sh vs tests/fixtures/qwen25vl/" },
    Arch { name: "qwen3vlmoe", runtime: Runtime::Dense, status: Status::Parts,
           note: "the mrope and text path are shared with qwen3vl, but the MoE FFN is not wired to \
                  this arch's tensor names; refused rather than run half-configured" },
    Arch { name: "llama", runtime: Runtime::Dense, status: Status::Verified,
           note: "NORM (interleaved) rope in the GGUF, unlike the Qwen family sharing this loader. \
                  ⭐ Verified against the MODEL AUTHORS' implementation (transformers 5.7.0, float32, \
                  eager; LlamaForCausalLM) on Llama-3.2-1B-Instruct, F32 converted from the same \
                  files: max |logit diff| 1.1e-4 over 136 positions x 128 sampled ids, argmax 136/136, \
                  full-row sum of squares within 1.6e-5. The wrong (NeoX) pairing is 114,914x worse. \
                  It was first checked against llama-cli, greedy tokens only. ⚠ The weights are \
                  unsloth/Llama-3.2-1B-Instruct, an ungated re-upload, not Meta's gated repo. \
                  Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/" },
    Arch { name: "phi3", runtime: Runtime::Dense, status: Status::Loads,
           note: "shares the dense path and the SPM vocab; not diffed against the reference" },
    Arch { name: "gemma", runtime: Runtime::Dense, status: Status::Loads,
           note: "embd_scale = sqrt(n_embd); SPM vocab" },
    Arch { name: "gemma2", runtime: Runtime::Dense, status: Status::Loads,
           note: "alternating SWA (pattern 2) + attn/final logit softcapping" },
    Arch { name: "gemma3", runtime: Runtime::Dense, status: Status::Verified,
           note: "1-in-6 global attention; local layers use a 512 window and their own rope base \
                  (rope.freq_base_swa, 10000; global 1e6). ⭐ Verified against the MODEL AUTHORS' \
                  implementation (transformers 5.7.0, float32, eager) on google/gemma-3-1b-it, F32 \
                  converted from their files: max |logit diff| 1.2e-4 over 134 positions and 1.1e-4 \
                  over 666 tokens (140 positions compared), argmax agreeing everywhere, full-row sum of \
                  squares within 8.0e-6. On the 666-token input: wrong rope pairing 278,363x worse, ONE \
                  rope base 237,961x, window disabled 236,359x. So all three mechanisms are \
                  load-bearing. ⛔ The window is INVISIBLE below 512 tokens: on the short input a port \
                  with no window matches exactly, which is why the long fixture exists. \
                  Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/gemma-3-1b{,-long}.json" },

    // ---- Muse Glimmer (2026-08-09) --------------------------------------------------------
    //
    // ⚠ THIS ROW WAS MISSING while the loader carried FOUR architecture-specific branches for it —
    // `rope_is_interleaved` (NORM pairing), `nope_global`, `embd_rmsnorm`, `post_norm_eps = 1e-8` —
    // plus `logit_scale` placement audited against `muse-glimmer.cpp`, a complete 50-layer vision
    // tower in `glimmer_vision.rs`, and two examples. `resolve()` refused the string, so none of it
    // could be reached: the work existed and the model could not load.
    //
    // Status is `Loads`, and deliberately not higher. Every per-detail choice above was read off the
    // reference implementation, but no end-to-end run has ever happened — it could not, because this
    // row's absence is what stopped it. `examples/muse_glimmer_vl.rs` is the test that settles it:
    // caption an image and compare against llama-mtmd-cli. Until someone runs that with weights in
    // hand, "the branches are reference-checked" is a claim about the source, not about the output.
    Arch { name: "muse-glimmer", runtime: Runtime::Dense, status: Status::Loads,
           note: "NORM (interleaved) rope, NoPE on the global layers, RMSNorm on the embeddings, \
                  post-attn/post-FFN norms at eps 1e-8, and logit_scale applied AFTER the LM head \
                  (on the queries it would have produced fluent, wrong text). Vision is separate: \
                  glimmer_vision::VisionTower loads the mmproj and its rows splice into the text \
                  sequence via Qwen3::forward_embeds. NOT diffed end to end against the reference" },

    // ---- Gemma 4 (2026-04-02) -------------------------------------------------------------
    Arch { name: "gemma4", runtime: Runtime::Gemma4, status: Status::Loads,
           note: "E2B/E4B dense path: per-layer embeddings, shared KV (blocks >= n-shared reuse 13/14), \
                  head_dim 512 global / 256 swa, weightless V norm, GELU FFN, no attention scale. \
                  MoE variants (26B-A4B, 31B) are refused at load rather than silently ignored" },

    // ---- DeepSeek MLA + DeepSeekMoE -------------------------------------------------------
    Arch { name: "deepseek2", runtime: Runtime::DeepSeek2, status: Status::Loads,
           note: "MLA (legacy attn_kv_b) + DeepSeekMoE, lite direct-Q. Block-0 tensors diffed against \
                  llama-eval-callback (attn_norm/q/kv_cmpr/k_pe all match); generates correct text on \
                  factual and code prompts. Absorbed (attn_k_b/attn_v_b) and Q-LoRA variants refused \
                  at load" },

    Arch { name: "hyv4", runtime: Runtime::Hyv4, status: Status::Verified,
           note: "Tencent Hy4, 770B/49B, and SUPPORTED BY NO UPSTREAM RUNTIME -- llama.cpp does not \
                  have this architecture; the published GGUFs ship two out-of-tree patches. Not a \
                  port: an independent implementation from the format. crate::hyv4 wires \
                  hyper-connections (4 residual streams, a rank-4 factorised DenseNet over sublayer \
                  outputs), gated MLA with a learnable per-head sink, absorbed MLA + Q-LoRA (both of \
                  which deepseek2 refuses), the DSA lightning indexer with its 21-of-78 index-sharing \
                  schedule, and DeepSeekMoE with a clamped SwiGLU. \
                  ⭐ VERIFIED ON THE REAL 213.66 GiB WEIGHTS against Tencent's own hyv4.cpp: all 78 \
                  blocks agree to ~1e-3 relative on sum_abs (flat with depth, no step), 2.87e-03 \
                  end-to-end on Ferric's own routing, and BOTH IMPLEMENTATIONS EMIT THE SAME GREEDY \
                  TOKEN (220). That residual is 6.25x SMALLER than llama.cpp's own CPU-vs-Metal \
                  disagreement measured the same way, so it sits inside the noise floor of a \
                  cross-fabric comparison. Cached decode equals a full prefill of the same sequence \
                  on the real weights at 7.25e-07. Finding the truth here cost one real defect: the \
                  SwiGLU clamp is a ROUTED-expert rule that was being applied to the shared and \
                  dense FFN too, invisible to every test because the clamp is the identity below \
                  ±10. VERIFICATION.md §3d carries the whole record; scripts/hyv4_vs_reference.sh \
                  gates sum_abs AND the greedy pick on every commit. \
                  ⛔ THE BOUND THAT REMAINS: the real-weights reference comparison is ONE PROMPT and \
                  ONE TOKEN, at a winning margin of 0.104 where the reference's is 0.581 -- multi-token \
                  generation has NOT been compared against the reference, and near-ties reorder \
                  (ranks 3/4 of the top-5 already differ). ⚠ SERVING REQUIRES STREAMING: ferric-serve \
                  loads this runtime with load_streaming under FERRIC_STREAM_GIB (default 12 GiB) \
                  because a resident load of 213.66 GiB is killed by the OS with no error at all, \
                  and it costs ~250 s/token on a 256 GiB machine because every token rebuilds all \
                  78 blocks" },

    // ---- gated-delta-net hybrids ---------------------------------------------------------
    Arch { name: "qwen35", runtime: Runtime::Hybrid, status: Status::Verified,
           note: "3-in-4 gated delta net; ssm_a pre-negated, tiled head order. ⭐ Verified against the \
                  MODEL AUTHORS' implementation (transformers 5.7.0, float32, eager) on Qwen/Qwen3.5-0.8B, \
                  F32 converted from their files: max |logit diff| 7.5e-4 over 135 positions x 128 \
                  sampled ids, argmax 135/135, full-row sum of squares within 6.3e-5. That residual is \
                  ~10x the dense rows'; the authors run a CHUNKED delta rule and Ferric a recurrent one, \
                  which is the leading hypothesis and is UNMEASURED. The wrong rope pairing is 2,908x \
                  worse (argmax 121/135: rotary covers 64 of 256 dims on 1 layer in 4). ⛔⛔ Two traps \
                  in the REFERENCE, each of which first read as a Ferric defect: transformers ignores \
                  dtype=float32 for this composite config and loads bf16, rounding the checkpoint's 36 \
                  F32 tensors; and a guard that casts THEN asserts float32 cannot see it. The generator \
                  asserts as loaded. ⛔ The delta-rule query scale was 1/sqrt(d_v); the authors and \
                  llama.cpp use 1/sqrt(d_k). Equal on every shipped checkpoint, so no real file could \
                  show it; fixed (Cfg::q_scale). ⭐ Also verified on MiMo-V2.6-Distill-Qwen-9B (Xiaomi's \
                  fine-tune of Qwen3.5-9B) on the authors' bf16 weights EXACTLY — a BF16 file held 16-bit \
                  (18 GB, not 36), against the authors' model run one layer at a time (refgen/stream.py, \
                  identical to the whole-model run on 0.8B): 4.0e-4, argmax 135/135. It has 32 value \
                  heads on 16 key heads, so the converter's grouped-to-TILED V-head reorder is checked \
                  for the first time — reading them grouped is 33,804x worse; 0.8B (16 on 16) could \
                  never show it. ⚠ the YaRN long-rope SUB-PATH is unverified: it ran \
                  through rope_scaled, which applied no rotation at all until 2026-08-15. \
                  Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/" },
    // ---- Qwen3-era MoE -------------------------------------------------------------------
    //
    // ⚠ THE MOST-DOWNLOADED GGUF ON HUGGING FACE (Qwen3-Coder-30B-A3B, 12.5M) and this runtime
    // refused it — while supporting qwen3 dense, qwen35 dense AND qwen35moe. The gap was never a
    // forward pass: the mixer is presence-detected (no ssm_out → Attn), the FFN is presence-detected
    // (ffn_gate_exps → MoE), and the metadata prefix already follows general.architecture. It was
    // two fields. `rope.dimension_count` was REQUIRED and qwen3moe does not emit it, and `MoeFfn`
    // demanded a shared expert that qwen3moe does not have.
    Arch { name: "qwen3moe", runtime: Runtime::Hybrid, status: Status::Loads,
           note: "Qwen3-30B-A3B / Qwen3-Coder-30B-A3B: plain GQA (no gated-delta-net) + 128 routed \
                  experts, top-8, expert width 768, NO shared expert and NO selection bias — so a \
                  softmax router straight through moe_topk. head_dim comes from attention.key_length \
                  (128), which is NOT n_embd/n_head (64). Not diffed against a reference" },
    Arch { name: "qwen35moe", runtime: Runtime::Hybrid, status: Status::Verified,
           note: "as qwen35 with an MoE FFN. ⚠ The shared hybrid path is verified against the authors \
                  (see qwen35); the MoE FFN has NOT yet been compared against the authors' code, only \
                  against the earlier reference" },
    Arch { name: "laguna", runtime: Runtime::Hybrid, status: Status::Loads,
           note: "shares the qwen35 runtime; uses YaRN (factor 32, orig ctx 8192) so it DOES exercise \
                  the rope_scaled path fixed on 2026-08-15. ⚠ NO REFERENCE AVAILABLE: llama.cpp \
                  refuses this file with \"unknown model architecture: laguna\", so it cannot be \
                  diffed against anything and must not be promoted on output that merely looks right" },

    // ---- short-conv hybrid ---------------------------------------------------------------
    Arch { name: "lfm2", runtime: Runtime::Lfm2, status: Status::Verified,
           note: "Liquid LFM2/LFM2.5; per-layer kv array marks conv blocks, conv state is PRE-conv. \
                  ⭐ Verified against the MODEL AUTHORS' implementation (transformers 5.7.0, float32, \
                  eager) on LiquidAI/LFM2-350M, F32 converted from their files: max |logit diff| 5.0e-5 \
                  over 138 positions x 128 sampled ids, argmax 138/138, full-row sum of squares within \
                  9.9e-6; the wrong rope pairing is 206,465x worse. ⚠ That is ONE PREFILL: cached \
                  decode, where the conv state matters, is compared with a full re-prefill by token ids \
                  (examples/run_lfm2_cached.rs, run by hand), not against the authors. Gate: scripts/lm_conformance.sh vs tests/fixtures/lm/" },
    Arch { name: "lfm2moe", runtime: Runtime::Lfm2, status: Status::Loads,
           note: "LFM2.5-8B-A1B: the same conv/attention schedule as lfm2, with the FFN made a \
                  mixture after `leading_dense_block_count` dense blocks (2 of 24). 32 experts, \
                  top-4, expert width 1792 — which is NOT feed_forward_length (7168, the dense \
                  blocks'). Sigmoid router with an exp_probs_b selection bias, and NO shared \
                  expert, unlike qwen35moe and laguna. RUNS: \"The capital of France is the \
                  city of Paris.\" Its expert compute is checked against an independent per-expert \
                  implementation (FERRIC_MOE_REF) and agrees to 1.8e-8, but nothing has been diffed \
                  against a REFERENCE RUNTIME, so this is not Verified" },

    // ---- safetensors-only ----------------------------------------------------------------
    Arch { name: "cosmos3_edge", runtime: Runtime::Cosmos, status: Status::Loads,
           note: "AR text tower only; loads from safetensors, not GGUF" },
];

/// Why a checkpoint cannot be served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchError {
    /// The file declares no `general.architecture`.
    Missing,
    /// Not in [`REGISTRY`]. Refused rather than guessed at.
    Unsupported(String),
    /// Known, but not finished. Carries the note so the caller learns what is left.
    NotRunnable(&'static Arch),
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::Missing => write!(f, "no general.architecture in the GGUF header"),
            ArchError::Unsupported(a) => write!(
                f,
                "architecture {a:?} is not supported by this runtime.\n  \
                 Refusing rather than loading it down a similar path: a near-miss architecture loads \
                 without error and generates fluent, wrong text.\n  \
                 Supported: {}",
                REGISTRY.iter().filter(|x| x.status.runnable()).map(|x| x.name).collect::<Vec<_>>().join(", ")
            ),
            ArchError::NotRunnable(a) => write!(
                f, "architecture {:?} is known but not runnable ({}): {}",
                a.name, a.status.label(), a.note
            ),
        }
    }
}

impl std::error::Error for ArchError {}

/// Look up an architecture. **Exact match only.**
///
/// Prefix matching is what let `gemma4` be treated as a Gemma-3, so it is not offered here even as a
/// convenience.
pub fn lookup(arch: &str) -> Option<&'static Arch> {
    REGISTRY.iter().find(|a| a.name == arch)
}

/// Resolve an architecture to a runtime, or explain the refusal.
///
/// This is the function the server and every example must call. Anything that dispatches on
/// `general.architecture` without going through here can reintroduce the catch-all.
pub fn resolve(arch: &str) -> Result<&'static Arch, ArchError> {
    if arch.is_empty() { return Err(ArchError::Missing); }
    match lookup(arch) {
        None => Err(ArchError::Unsupported(arch.to_string())),
        Some(a) if !a.status.runnable() => Err(ArchError::NotRunnable(a)),
        Some(a) => Ok(a),
    }
}

/// A printable coverage table: what runs, what is half-built, at what confidence.
///
/// Meant to be run in CI and read by a human deciding what to port next.
pub fn coverage() -> String {
    let mut s = String::from("arch            runtime   status     note\n");
    s.push_str(&"-".repeat(96));
    s.push('\n');
    let mut rows: Vec<&Arch> = REGISTRY.iter().collect();
    rows.sort_by_key(|a| (a.status, a.runtime.label(), a.name));
    for a in rows {
        s.push_str(&format!("{:<15} {:<9} {:<10} {}\n", a.name, a.runtime.label(), a.status.label(), a.note));
    }
    let v = REGISTRY.iter().filter(|a| a.status == Status::Verified).count();
    let r = REGISTRY.iter().filter(|a| a.status.runnable()).count();
    s.push_str(&format!("\n{} architectures runnable, {v} reference-verified, {} total known\n",
                        r, REGISTRY.len()));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyv4_is_verified_and_names_the_bound_that_remains() {
        let a = REGISTRY.iter().find(|a| a.name == "hyv4").expect("hyv4 must be registered");
        assert_eq!(a.status, Status::Verified);
        assert!(a.status.runnable(), "Verified must be runnable");
        assert!(resolve("hyv4").is_ok(), "resolve must now accept hyv4");
        // ⭐ THIS ASSERTION HAS NOW FAILED THREE TIMES, AND EVERY TIME THAT WAS THE POINT. It first
        // demanded "NO REAL CHECKPOINT HAS BEEN LOADED", which stopped being true when real weights
        // ran in slices. It then demanded "THAT IS NOT FIDELITY", which stopped being true when
        // AngelSlim's llama.cpp built and Ferric's forward matched it on a synthetic file. It then
        // demanded "SYNTHETIC checkpoint", which stopped being true when all 78 blocks of the real
        // 213.66 GiB checkpoint were compared against that reference and both emitted token 220.
        // Each failure was the row's bound MOVING, not the guard breaking, and each time the fix is
        // to repoint it at what is STILL true — never to soften the note until it passes.
        //
        // What is still true: the real-weights comparison is one prompt and one token. Nothing has
        // compared a multi-token GENERATION against the reference, and the margin is not large.
        assert!(a.note.contains("ONE PROMPT") && a.note.contains("ONE TOKEN"),
                "the row must name the bound that remains: the real-weights reference comparison is                  a single prompt and a single token, not a generation");
        assert!(a.note.contains("SERVING REQUIRES STREAMING"),
                "a 213.66 GiB checkpoint is killed by the OS if loaded resident, and the row is                  where a reader finds that out before their server dies without a message");
        assert!(a.note.contains("hyv4_vs_reference.sh"),
                "the row claims a reference comparison; it must name the script that performs it");
    }

    #[test]
    fn an_unknown_architecture_is_refused_not_defaulted() {
        // THE bug this module exists for. Every one of these is a real 2026 architecture that the old
        // `else` branch would have loaded as a dense Qwen3 and generated fluent nonsense from.
        for a in ["glm4", "minimax", "hunyuan", "mimo", "kimi", "step3", "ernie4"] {
            let e = resolve(a).unwrap_err();
            assert!(matches!(e, ArchError::Unsupported(_)), "{a} was not refused: {e:?}");
            // The message has to say what IS supported, or the refusal is useless to whoever hit it.
            assert!(e.to_string().contains("qwen3"), "refusal for {a} does not list what works");
        }
    }

    #[test]
    fn gemma4_does_not_inherit_gemma3_by_prefix() {
        // `arch.starts_with("gemma")` is true for "gemma4". That is the exact mechanism by which a
        // 2026 model would silently adopt 2025 assumptions about SWA pattern, embedding scale and
        // logit softcapping — and produce plausible output while doing it.
        assert!("gemma4".starts_with("gemma"), "premise of this test");
        // Both exist now, and they must land on DIFFERENT runtimes. Prefix matching would have sent
        // gemma4 to the dense Gemma-3 path, which shares neither the KV schedule nor the head width.
        let g3 = lookup("gemma3").expect("gemma3");
        let g4 = lookup("gemma4").expect("gemma4");
        assert_ne!(g3.runtime, g4.runtime, "gemma3 and gemma4 must not share a runtime");
        assert_eq!(g4.runtime, Runtime::Gemma4);
    }

    #[test]
    #[ignore = "deepseek2 was promoted to `loads` once it generated correct text"]
    fn a_written_but_never_executed_loader_is_refused_by_name() {
        // deepseek2 has a complete loader whose config logic is unit-tested and whose forward has
        // never run on real weights. That is NOT the same as unsupported, and it is NOT servable:
        // the registry says so out loud instead of letting it take traffic on the strength of
        // compiling.
        let e = resolve("deepseek2").unwrap_err();
        match &e {
            ArchError::NotRunnable(a) => {
                assert_eq!(a.status, Status::Parts);
                assert!(a.note.contains("never executed"), "the note must say why: {}", a.note);
            }
            other => panic!("deepseek2 should be known-but-not-runnable, got {other:?}"),
        }
        assert!(lookup("deepseek2").is_some(), "it must still be discoverable in the coverage table");
    }

    #[test]
    fn a_known_but_unfinished_architecture_still_refuses_to_serve() {
        // The registry is allowed to describe work in progress. It is not allowed to let it take
        // traffic — a `Parts` row that dispatched would be the catch-all wearing a different hat.
        let wip = Arch { name: "wip", runtime: Runtime::Dense, status: Status::Parts, note: "no loader" };
        assert!(!wip.status.runnable());
        for a in REGISTRY {
            if !a.status.runnable() {
                assert!(matches!(resolve(a.name), Err(ArchError::NotRunnable(_))),
                        "{} is not runnable but resolve() let it through", a.name);
            }
        }
    }

    #[test]
    fn an_empty_architecture_is_missing_not_unsupported() {
        // A GGUF with no arch key is a different problem from one this runtime has not learned, and
        // conflating them sends the reader looking in the wrong place.
        assert_eq!(resolve(""), Err(ArchError::Missing));
    }

    #[test]
    fn every_runtime_declares_whether_it_generates_text() {
        // ⚠ THIS TEST USED TO BE `assert!(true)`. Its match returned `true` from all eight arms, so
        // `assert!(served)` could not fail — and it sat green while `nemotron_h` was Status::Verified
        // and ferric-serve panicked "its forward pass is not written yet".
        //
        // What it CAN check from this crate (which cannot see ferric-serve) is that every runtime
        // has a stated answer to "does this generate text from tokens?" — the property that decides
        // whether a row belongs on a chat endpoint at all. The registry↔dispatch agreement itself is
        // tested where it is visible, in ferric-serve's
        // `every_verified_registry_row_can_actually_be_loaded`.
        //
        // The exhaustive match is the real guard: adding `Parakeet` made this a COMPILE ERROR rather
        // than a silent pass, which is exactly how a new modality should arrive.
        for a in REGISTRY {
            let generates_text = match a.runtime {
                Runtime::Dense | Runtime::Hybrid | Runtime::Lfm2 | Runtime::Gemma4
                    | Runtime::DeepSeek2 | Runtime::NemotronH | Runtime::Cosmos
                    | Runtime::Hyv4 => true,
                // Encoder: embeddings and cross-encoder scores, no LM head.
                Runtime::Bert => false,
                // Encoder too — a different one, but the same answer: no LM head to generate from.
                Runtime::ModernBert => false,
                // Speech: takes a WAVEFORM, not tokens.
                Runtime::Parakeet => false,
            };
            // A row that does not generate text must say so in its note, because every summary of
            // this registry reads like a list of chat models otherwise.
            if !generates_text {
                let n = a.note.to_ascii_lowercase();
                assert!(n.contains("refus") || n.contains("not a chat model") || n.contains("waveform"),
                        "{} does not generate text but its note does not say so: {}", a.name, a.note);
            }
        }
    }

    #[test]
    fn the_encoder_row_is_marked_as_embedding_only() {
        // The registry now mixes generators with a runtime that CANNOT generate. A caller reading
        // `status: Verified` and reaching for `generate` must find that stated here, because the
        // refusal itself lives two crates away in ferric-web.
        let bert = REGISTRY.iter().find(|a| a.name == "bert").expect("bert row");
        assert_eq!(bert.runtime, Runtime::Bert);
        // Assert the PROPERTY, not one phrasing. The first version matched the literal "EMBEDS ONLY"
        // and went red when the note was corrected to "EMBEDS AND SCORES" — a real change in what the
        // runtime does (it gained cross-encoder scoring) that left the no-generation contract intact.
        // A test pinned to wording fails on edits and passes on substance, which is backwards.
        assert!(bert.note.contains("generation is refused"),
                "the encoder row must state that generation is refused; note reads: {}", bert.note);
        assert!(!bert.note.contains("VERIFIED FOR BERT ONLY"),
                "the XLM-R qualifier was retracted once the divergence turned out to be a test bug");
    }

    #[test]
    fn the_registry_has_no_duplicate_architectures() {
        // Two rows for one arch means `lookup` silently picks the first, and which one that is depends
        // on edit order.
        let mut seen = std::collections::HashSet::new();
        for a in REGISTRY {
            assert!(seen.insert(a.name), "{} appears twice in REGISTRY", a.name);
        }
    }

    #[test]
    fn coverage_reports_a_floor_and_names_the_verified_ones() {
        // An enumerating tool that can return nothing and still look fine is not a check. Assert a
        // floor so an empty or broken registry fails loudly rather than printing a tidy zero.
        let c = coverage();
        let runnable = REGISTRY.iter().filter(|a| a.status.runnable()).count();
        assert!(runnable >= 10, "only {runnable} runnable architectures — registry looks truncated");
        assert!(c.contains("qwen3"));
        assert!(c.contains("lfm2"));
        assert!(c.contains("reference-verified"));
    }
}
