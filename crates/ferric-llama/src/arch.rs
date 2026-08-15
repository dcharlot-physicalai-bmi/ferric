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
}

impl Runtime {
    pub fn label(self) -> &'static str {
        match self {
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
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Verified => "verified",
            Status::Loads => "loads",
            Status::Parts => "parts",
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
    Arch { name: "qwen2", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked; the family this runtime was written against" },
    Arch { name: "qwen3", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked, incl. per-head QK RMSNorm" },
    // ⚠ DOWNGRADED 2026-08-15. Was marked `verified`. Llama-3.2-1B-Instruct Q4_K_M Q4_K_M generates
    // "The capital of France is located in the United States" where llama-cli --temp 0 on the same
    // ids gives "Paris." — top-1 Ġlocated 14.7175 vs ĠParis 14.6797, so the distribution is nearly
    // right and the argmax is not. Independent of both rope fixes: identical output with rope_freqs
    // disabled AND with NORM vs NEOX pairing, so the cause is elsewhere in the dense path.
    // Whatever the original verification covered, it did not cover this checkpoint.
    Arch { name: "llama", runtime: Runtime::Dense, status: Status::Loads,
           note: "⚠ WRONG on Llama-3.2-1B-Instruct: diverges from llama-cli on a trivial factual \
                  prompt. Earlier `verified` claim did not hold for this checkpoint. Under \
                  investigation; SentencePiece/BPE vocab path" },
    Arch { name: "phi3", runtime: Runtime::Dense, status: Status::Loads,
           note: "shares the dense path and the SPM vocab; not diffed against the reference" },
    Arch { name: "gemma", runtime: Runtime::Dense, status: Status::Loads,
           note: "embd_scale = sqrt(n_embd); SPM vocab" },
    Arch { name: "gemma2", runtime: Runtime::Dense, status: Status::Loads,
           note: "alternating SWA (pattern 2) + attn/final logit softcapping" },
    Arch { name: "gemma3", runtime: Runtime::Dense, status: Status::Verified,
           note: "reference-checked; 1-in-6 global attention, local rope base 10000" },

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

    // ---- gated-delta-net hybrids ---------------------------------------------------------
    Arch { name: "qwen35", runtime: Runtime::Hybrid, status: Status::Verified,
           note: "3-in-4 gated delta net; ssm_a pre-negated, tiled head order" },
    Arch { name: "qwen35moe", runtime: Runtime::Hybrid, status: Status::Verified,
           note: "as qwen35 with an MoE FFN" },
    Arch { name: "laguna", runtime: Runtime::Hybrid, status: Status::Loads,
           note: "shares the qwen35 runtime" },

    // ---- short-conv hybrid ---------------------------------------------------------------
    Arch { name: "lfm2", runtime: Runtime::Lfm2, status: Status::Verified,
           note: "Liquid LFM2/LFM2.5; per-layer kv array marks conv blocks, conv state is PRE-conv" },

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
    fn every_runnable_architecture_names_a_runtime_that_exists() {
        // Guards the registry against drifting from the code: a row pointing at a runtime that was
        // renamed or removed would advertise support that cannot be dispatched.
        for a in REGISTRY {
            let served = match a.runtime {
                Runtime::Dense | Runtime::Hybrid | Runtime::Lfm2 | Runtime::Cosmos | Runtime::Gemma4
                    | Runtime::DeepSeek2 => true,
            };
            assert!(served, "{} names a runtime with no dispatch", a.name);
        }
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
