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
    /// [`crate::nemotron_h`] — Mamba-2 state-space mixers with a few attention layers.
    NemotronH,
    /// [`crate::parakeet`] — Conformer encoder + RNN-T decoder. SPEECH: waveform in, text out.
    /// Not a generative text runtime; `ferric-serve` refuses it the way it refuses `Bert`.
    Parakeet,
}

impl Runtime {
    pub fn label(self) -> &'static str {
        match self {
            Runtime::Bert => "bert",
            Runtime::NemotronH => "nemotron_h",
            Runtime::Parakeet => "parakeet",
            Runtime::Dense => "dense",
            Runtime::Hybrid => "hybrid",
            Runtime::Lfm2 => "lfm2",
            Runtime::Gemma4 => "gemma4",
            Runtime::DeepSeek2 => "deepseek2",
            Runtime::Cosmos => "cosmos",
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
                  21 state-space mixers, 17 ReLU^2 MLPs, 4 attention. RUNS and reproduces the \
                  reference: from 'The capital of France is' it generates ' Paris.', matching \
                  llama.cpp. Embedding sum is exact and block 0 agrees to 0.10%. REFERENCE-CHECKED at the \
                  distribution, which is the level that settles it: on the same 8-token prompt the \
                  top-10 match the reference token for token IN THE SAME ORDER, logprobs agreeing to \
                  ~0.01 (-0.7843 vs -0.7757, -2.4986 vs -2.5101, -3.1770 vs -3.1766, -3.3060 vs \
                  -3.3102). An earlier greedy run diverged after ' Paris.' and that was a different \
                  token PATH, not a different distribution — the leader there holds under half the \
                  mass, so which of several near-ties wins is not a fidelity test. Per-block sums are \
                  not one either: their error tracks the cancellation ratio |sum|/max|v| (5.18 gives \
                  0.10%, 0.14 gives 13.85%), so they localise a gross defect and cannot grade \
                  fidelity. INCREMENTAL STATE WORKS and is the architecture's whole argument: 85.4 MB of \
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
    Arch { name: "asr", runtime: Runtime::Parakeet, status: Status::Loads,
           note: "NVIDIA NeMo ASR export (parakeet-ctc-1.1b): the same Conformer encoder with a CTC \
                  head instead of RNN-T. Ships its own mel filterbank (preprocessor.fb) and \
                  precomputed positional encoding. Waveform in, text out — NOT a chat model" },
    Arch { name: "parakeet", runtime: Runtime::Parakeet, status: Status::Verified,
           note: "NVIDIA Parakeet / Nemotron-ASR: Conformer encoder + RNN-T. TRANSCRIBES — every \
                  word correct on three LibriSpeech utterances (residual WER is punctuation the \
                  references do not carry). Waveform in, text out: NOT a chat model, and \
                  ferric-serve refuses it the way it refuses bert" },

    Arch { name: "bert", runtime: Runtime::Bert, status: Status::Verified,
           note: "encoder-only: bidirectional, learned positions, post-LayerNorm, GELU FFN, no KV \
                  cache and no LM head. Reference-checked against llama-embedding on bge-small-en-v1.5 \
                  at cosine 0.999999 (F16) and 0.999996 (Q4_K_M), AND XLM-RoBERTa (bge-reranker-v2-m3, 24L \
                  d=1024 Q4_K_M) at 0.999995-1.000000, over 3-to-39-token inputs. Cross-encoder \
                  scoring matches the reference to 0.24%. EMBEDS AND SCORES — generation is refused, \
                  there is no LM head to generate from" },
    Arch { name: "qwen2", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked; the family this runtime was written against" },
    Arch { name: "qwen3", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked, incl. per-head QK RMSNorm" },
    Arch { name: "llama", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked against llama-cli on Llama-3.2-1B-Instruct. NORM (interleaved) \
                  rope, unlike the Qwen family sharing this loader" },
    Arch { name: "phi3", runtime: Runtime::Dense, status: Status::Loads,
           note: "shares the dense path and the SPM vocab; not diffed against the reference" },
    Arch { name: "gemma", runtime: Runtime::Dense, status: Status::Loads,
           note: "embd_scale = sqrt(n_embd); SPM vocab" },
    Arch { name: "gemma2", runtime: Runtime::Dense, status: Status::Loads,
           note: "alternating SWA (pattern 2) + attn/final logit softcapping" },
    Arch { name: "gemma3", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked; 1-in-6 global attention, local rope base 10000" },

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

    Arch { name: "hyv4", runtime: Runtime::DeepSeek2, status: Status::Untried,
           note: "Tencent Hy4, 770B/49B, and SUPPORTED BY NO UPSTREAM RUNTIME — llama.cpp does not \
                  have this architecture; the published GGUFs ship two out-of-tree patches. Not a \
                  port: an independent implementation from the format. crate::hyv4 wires \
                  hyper-connections (4 residual streams, a rank-4 factorised DenseNet over sublayer \
                  outputs), gated MLA with a learnable per-head sink, absorbed MLA + Q-LoRA (both of \
                  which deepseek2 refuses), the DSA lightning indexer with its 21-of-78 index-sharing \
                  schedule, and DeepSeekMoE with a clamped SwiGLU. Components verified individually: \
                  the HC closed form and both absorption folds exactly over GF(2^61-1), the schedule \
                  by bounded model checking for every is_full pattern, STQ1_0/IQ2_XXS/IQ3_XXS by Kani \
                  plus an interop check against Tencent's own published weights. ⛔ NO REAL \
                  CHECKPOINT HAS BEEN LOADED: the smallest is 213.66 GiB against ~47 GB free here, so \
                  the forward is exercised only by examples/hyv4_synthetic.rs. That proves the wiring \
                  and cannot prove fidelity. The runtime field is a placeholder; resolve() refuses \
                  this string because Untried is not runnable" },

    // ---- gated-delta-net hybrids ---------------------------------------------------------
    Arch { name: "qwen35", runtime: Runtime::Hybrid, status: Status::Verified,
           note: "3-in-4 gated delta net; ssm_a pre-negated, tiled head order. ⚠ the YaRN long-rope \
                  SUB-PATH is unverified: it ran through rope_scaled, which applied no rotation at \
                  all until 2026-08-15, so any earlier check passed without exercising it" },
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
           note: "as qwen35 with an MoE FFN" },
    Arch { name: "laguna", runtime: Runtime::Hybrid, status: Status::Loads,
           note: "shares the qwen35 runtime; uses YaRN (factor 32, orig ctx 8192) so it DOES exercise \
                  the rope_scaled path fixed on 2026-08-15. ⚠ NO REFERENCE AVAILABLE: llama.cpp \
                  refuses this file with \"unknown model architecture: laguna\", so it cannot be \
                  diffed against anything and must not be promoted on output that merely looks right" },

    // ---- short-conv hybrid ---------------------------------------------------------------
    Arch { name: "lfm2", runtime: Runtime::Lfm2, status: Status::Verified,
           note: "Liquid LFM2/LFM2.5; per-layer kv array marks conv blocks, conv state is PRE-conv" },
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
    fn hyv4_is_registered_but_refused_until_a_real_checkpoint_loads() {
        let a = REGISTRY.iter().find(|a| a.name == "hyv4").expect("hyv4 must be registered");
        assert_eq!(a.status, Status::Untried);
        assert!(!a.status.runnable(), "Untried must not be runnable: a server must not serve a \
                                       model whose output nobody has seen");
        assert!(resolve("hyv4").is_err(), "resolve must refuse hyv4: {:?}", resolve("hyv4").map(|_| ()));
        // And the note must carry the bound, so nobody reads the row as a capability claim.
        assert!(a.note.contains("NO REAL CHECKPOINT HAS BEEN LOADED"),
                "the row must state that no real checkpoint has been loaded");
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
                    | Runtime::DeepSeek2 | Runtime::NemotronH | Runtime::Cosmos => true,
                // Encoder: embeddings and cross-encoder scores, no LM head.
                Runtime::Bert => false,
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
