# What to ingest into Ferric — the 2026-09 architecture & scaling sweep

**Date: 2026-09-20.** Five parallel research agents: decode-time strategy, post-transformer
architectures, scaling laws after Chinchilla, precision/quantization, and non-Western (CN/JP/KR)
research.

## ⚠ Provenance — read this before quoting anything below

Every claim here comes from a **research agent's report**, not from my own reading of the source. Each
agent tagged its own claims VERIFIED (it fetched the primary artifact — a `config.json`, a model card, a
paper's HTML) or INFERRED (summary, secondary blog, or search excerpt). **I have not independently
re-checked any of them**, and one agent explicitly caught a content-farm page asserting llama.cpp
support that the repository contradicts. Treat the download rankings, the fitted coefficients and the
2026 arXiv identifiers as leads to verify before they enter a design doc or a public claim, and keep
[docs/determinism](determinism/) and the measurement discipline rules as the standard for anything
Ferric *asserts*. What follows is a map of where the field went, not a receipt.

**The one thing here that is first-hand:** Qwen3-VL sits at **#2 by 30-day downloads**, and Ferric's
Qwen3-VL vision tower was verified against the published implementation on the same day this sweep ran
(five seams, worst 6.75e-5 relative — `examples/qwen3vl_vision_ref.rs`).

---

## The one-line answer

**The mixer math is no longer the hard part; the state manager is.** Four of the five reports converge
on it independently. A 2026 model's per-layer state is no longer a uniform KV cache — it is a mix of KV
pages, depthwise-conv windows, matrix-valued RNN states, compressed latents and a *second* cached
pathway for a learned sparse index. The capability that separates runtimes is whether that state can be
**snapshotted and rewound**, because prefix caching, speculative rejection, beam search and multi-turn
edit all require truncate-and-replay — and a recurrent layer that keeps only its final state cannot do
it. This is an architectural property you design in or retrofit badly, which is exactly the kind of gap
a from-scratch runtime can win.

---

## 1. What converged across independent sweeps (highest confidence)

### Hybrid linear + full attention at 3:1 is the shipped default
Three reports reached this separately, from different sources.

| model | schedule | source status |
|---|---|---|
| Qwen3.5-9B | 32 layers, `[GDN,GDN,GDN,FullGatedAttn] × 8` | VERIFIED `config.json` |
| Qwen3.8-27B | 64 layers, 48 GDN + 16 full | VERIFIED `config.json` |
| Qwen3-Next-80B-A3B | 48 layers, 36 GDN + 12 gated attention | VERIFIED Qwen blog |
| Kimi Linear / K3 | 3 KDA : 1 (gated, **NoPE**) MLA | VERIFIED paper |
| Nemotron 3 | `Mamba-2, LatentMoE, Mamba-2, Attn, Mamba-2, LatentMoE` — attention 1-in-6 | VERIFIED NVIDIA |
| Gemma 4 | 5 SWA : 1 global, **final layer always global**, p-RoPE in global layers | VERIFIED model card |
| Ling 2.6 | 7 Lightning : 1 MLA | INFERRED |

**Gated DeltaNet (GDN)** won the linear-attention race: a causal depthwise Conv1d state (width 4) plus a
matrix-valued RNN state, delta-rule update with an exponential forget gate, L2-normalised Q/K.
**Kimi Delta Attention** is GDN with *channel-wise* rather than head-wise forget gates.

There is a measured explanation, and it is the most useful paper in the sweep: Tsinghua/OpenBMB
(arXiv 2606.15378) find long-range retrieval is carried **almost entirely by the full-attention
layers**, while efficient attention mainly shapes the optimization trajectory — and name
*"Large-Window Laziness"*, where larger sliding windows **delay** retrieval-head formation. That is why
everyone landed on 3:1 rather than tuning it per-model.

### Learned sparse indexers are the other branch, and they are licensed across labs
DeepSeek's DSA (lightning indexer → top-k 2048 → MLA over the selection) now also ships in **GLM-5**;
Qwen3.8 has QSA, MiniMax M3 has MSA. One indexer abstraction covers several families. GLM-5.2's
**IndexShare** is the sharpest result: adjacent DSA layers were measured to select **70–100% the same
tokens**, so one indexer's selection is reused by three downstream layers — 2.9× per-token FLOP
reduction at 1M context.

### 1M context and extreme MoE sparsity are the frontier default
DeepSeek-V4-Pro 1.6T/49B active = **3.1%**; Kimi K3 2.8T/104B with 16 of 896 routed experts. The MoE
scaling law's fitted optimum (granularity ≈12, ~3% activation) and the shipped models now agree — which
is either strong confirmation or a sign the labs fit the same curve. Expect total:active of 20–30×.

### Small changes that are now near-universal and cheap to miss
QK-Norm (Qwen uses a *zero-centered* variant); **output-gated attention** (`attn_output_gate: true`,
reported to suppress attention sinks); **partial RoPE** (Qwen3.5 rotates 64 of 256 head dims,
`rope_theta: 1e7`); **mRoPE in text-first models** because they are natively multimodal
(`mrope_interleaved: true`, `mrope_section: [11,11,10]`); **MTP / draft heads** on essentially every
flagship. A runtime that carries MTP tensors but ignores them loses the advertised 2–3× on agentic
decode — EXAONE 4.5 disables its own MTP at inference.

---

## 2. The counter-evidence, which is load-bearing

**MiniMax M2/M2.5 deliberately reverted to dense GQA on all 62 layers**, and published why. Their stated
reasons are all runtime concerns, not modelling ones:
- 128K RULER **90 (full) vs 72 (sliding-window)**;
- recurrent/compressed states are **precision-fragile**;
- linear/sparse state **breaks native prefix caching and speculative decoding** — decisive for coding
  agents.

That third point is the same finding as §0, arrived at from the opposite direction: a lab chose a worse
FLOP profile rather than lose rewindable state. **This is the strongest argument that solving state
rollback properly is worth more than the kernels.** It also means plain GQA + RoPE + QK-norm is not a
legacy path — it is a live design choice, and still the largest bucket by real usage.

---

## 3. Ranked by what people actually run

30-day HF downloads, 2026-09-18 (one secondary source, cited by two agents; the two disagree slightly on
Qwen3.5-9B, 9.17M vs 9.3M):

| rank | model | downloads | mechanism | Ferric status |
|---|---|---|---|---|
| #1 | Qwen3-0.6B | 22.5M | dense GQA + RoPE + QK-norm | **runs** |
| #2 | Qwen3-VL-8B | 19.1M | dense + mRoPE + vision tower + deepstack | **tower verified 2026-09-20** |
| #5 | Qwen3-8B | 13.0M | dense GQA | **runs** |
| #9 | Qwen2.5-7B | 9.7M | dense GQA | **runs** |
| #10 | Qwen3.5-9B | ~9.2M | **Gated DeltaNet 3:1** | ⛔ not implemented |
| — | Qwen3.5-4B / 2B | 6.8M / 4.8M | **Gated DeltaNet 3:1** | ⛔ not implemented |
| #38 | DeepSeek-V4-Flash | 4.4M | DSA/CSA + mHC + FP4 experts | ⛔ not implemented |
| #51 | Kimi-K3-DSpark | 3.3M | KDA + NoPE MLA | ⛔ not implemented |

Qwen is ≈240M downloads and 27 of the top 60 rows. **Gated DeltaNet is the single biggest uncovered
mechanism by download weight — roughly 21M/month across the Qwen3.5 sizes alone**, and it is the one
where the incumbent is visibly struggling: llama.cpp took ~11 months from issue to merge, the
Kimi-Linear request went stale, and issue **#28461 ("bounded recurrent-state rollback")** is open on
precisely the state-management problem above.

---

## 4. What the scaling laws change about a *runtime*

Most scaling-law work is about how to spend training compute and is irrelevant here. Three results are
not:

1. **Models are now massively overtrained, and that makes naive PTQ structurally worse over time.**
   Tokens-per-*active*-parameter went ~10 (2022) → ~300–437 (2025), growing **3.1×/yr**. The precision
   scaling law gives `δ_PTQ ∝ D^γ_D / N_eff^γ_N` — post-training quantization damage **grows with D/N**.
   So: consume the lab's **native low-precision checkpoint as shipped**; do not re-derive it from BF16.
   Gemma 4 QAT (Q4_0 GGUF), Kimi K2-Thinking (INT4 QAT), Nemotron-3-NVFP4 (trained in NVFP4) and
   DeepSeek-V4 (FP4 experts) are all artifacts where the calibration *is* the release.
2. **Attention, not parameter count, dominates test-time-scaling cost** (Kinetics). Small models are
   *overrated* once memory access is priced — which is the performance-per-watt framing, arrived at
   from the benchmark side.
3. **Marginal returns on thinking go negative** ("When More Thinking Hurts"): models abandon correct
   answers, and *easy* problems hit negative marginal utility earlier than hard ones. **Budget control
   and early stop are runtime features**, not a UX nicety.

Also relevant to the ternary line: **ParetoQ** finds a genuine regime shift **between 2 and 3 bits** —
≥3-bit stays near the pretrained distribution, ≤2-bit re-learns representations; ternary/2/3-bit are
jointly Pareto-optimal and beat both 4-bit and binary. But the largest published natively-ternary model
is still ~2B (BitNet b1.58-2B-4T); no ≥7B native ternary release was located. **Ternary at frontier
scale remains unevidenced** — worth saying plainly, because our own ternary work is easy to over-read.

---

## 5. Where Ferric's differentiator moved

Prior position (memory: *runtime landscape 2026-09*): **cross-vendor bit-identity is unclaimed by
anyone.** This sweep both threatens and sharpens that.

**The threat.** DeepSeek-V4 ships **end-to-end bitwise batch-invariant deterministic kernels as a stated
design goal**, to bit-align pre-training, post-training and inference — rejecting split-KV decode in
favour of a dual-kernel decode that fixes accumulation order. No Western frontier lab has published an
equivalent commitment, and vLLM's batch-invariant PR is marked DO NOT MERGE. But it is a *lab*, on
*one vendor*. "Deterministic inference" is no longer rhetorically unclaimed.

**The sharpening — and this is the most valuable paragraph in the sweep.** The reproducible unit is
**(file bytes, decode rule, accumulation schedule)**, not "the model". Five *documented, live*
cross-implementation divergences:

- **MLX decodes NVFP4 block scales as signed E4M3 where NVIDIA and llama.cpp use UE4M3** — max scale
  448 vs ~61,440, a **137× dynamic-range gap**. Open bug, MLX #2962.
- **MXFP4 is container type 4 in ollama and 39 in llama.cpp**; llama.cpp-converted gpt-oss GGUFs fail
  to load in ollama.
- **mlx-lm reads any unrecognized compressed-tensors checkpoint as 4-bit affine** — F8_E4M3 bytes get
  read as packed nibbles **and the model still generates**. (This is vacuous-output failure at the
  format layer; it belongs in the vacuous-test register.)
- **CDNA3/MI300 uses `e4m3fnuz`, everyone else OCP `e4m3fn`** — biases differ by one, so the byte `0x40`
  is 2.0 on one fabric and 1.0 on another.
- **OCP MX v1.0 only requires that roundTiesToEven be *supported*** and explicitly permits "other
  implementation-defined conversion recipes"; NVFP4's global-scale divisor is likewise free (448 vs 256
  vs adaptive). **Two conformant quantizers produce different files from identical BF16 weights.**

**A sixth, found first-hand while building against it (2026-09-20):** *"bicubic" does not name a
resampler.* Pillow's `BICUBIC` is the Keys kernel with **a = −0.5**; PyTorch/torchvision and
HuggingFace's own `_interpolation_axis_taps_weights` use **a = −0.75**. Measured here by impulse
response, not read from a document: residual **0.000e+00** against a = −0.5 and **3.69e-2** against
a = −0.75. Qwen3-VL's `preprocessor_config.json` says only `"resample": 3`, and its named processor
(`Qwen2VLImageProcessorFast`) is the torch one while the slow sibling is the PIL one — so the same
config selects two different kernels depending on which class loads it. This is the same failure
shape as the NVFP4 scale disagreement, one layer earlier: in the *pixels*, before any weight is read.

**And it is now measured, not argued (2026-09-20).** Running Qwen3-VL's whole image path twice —
once on Ferric's own preprocessed pixels, once on the reference processor's — attributes the error:

| pixels | final hidden state, vs HF |
|---|---|
| the reference processor's | **4.25e-6** relative — the path itself, f32 drift |
| Ferric's own | **3.63e-4** relative |

Ferric reproduces Pillow's 8-bit resampler *exactly* and sits 1 level from HF's on 0.46% of values —
the same distance Pillow itself sits from it. That disagreement, between two conformant bicubic
implementations, is worth **85× the arithmetic noise** 28 layers later. **A bit-exactness claim that
starts at the tensor starts too late**: the largest term is upstream of the first weight.

→ **Decode can be bit-exact; encode cannot be assumed to be.** That is a crisp, defensible thesis, it is
exactly the seam Ferric already owns (*cross-fabric identity boundary*: hashes differ per **adapter**),
and it hands us five concrete regression targets. Accumulation is also below IEEE independently of
format: Hopper FP8 tensor cores keep only the top 14 mantissa bits (≥34 needed for exact FP32 over 32
terms), and Tenstorrent's Math Fidelity LoFi…HiFi4 consumes different mantissa-bit counts for the same
BFP8 operands *on the same chip* — which lands directly on the tt-tier work.

---

## 6. Proposed ingest order

1. **Gated DeltaNet + gated full attention at 3:1, with rewindable state from day one.** Biggest
   uncovered download bucket; the incumbent's weakest point; and the rollback property is a design
   decision, not a kernel. Pair the conv window and the matrix RNN state with the KV pages in *one*
   allocator, sized so a full-attention page and a recurrent state occupy comparable physical memory
   (vLLM's pattern). **Snapshot/restore is the feature, not the mixer.**
2. **A heterogeneous per-layer state schedule.** V4 mixes `c4a`/`c128a`/SWA; K3 is 3:1; Gemma 4 is 5:1
   with a mandatory global final layer. Any allocator assuming a uniform per-layer cache mis-sizes all
   of them.
3. **Format conformance as a shipped artifact** — decode rules as data, plus published conformance
   vectors, tested against the five divergences above. This is the differentiator, and there is prior
   art to align to (arXiv 2606.09686 — a 109-format catalog with bit-exact conformance packs).
4. **MXFP4 and FP8 E4M3 native decode** (gpt-oss; DeepSeek block-scaled UE8M0), then NVFP4 with the
   UE4M3-vs-E4M3 trap handled explicitly rather than inherited.
5. **MTP / draft heads as a first-class model component**, with budget control and early stop.
6. **Sliding-window + global interleave** — cheap, large installed base (Gemma 4, Mistral lineage).

**Deprioritize** (tier 2, frontier self-hosting): DSA/QSA indexers — two families, but the indexer is
**quantization-fragile in a way normal attention is not** (llama.cpp measured ~70% → 95% accuracy on
fixing a Hadamard bug, and Q4_K_M degrades it far more than Q8_0). Also mHC multi-stream residuals
(V4 + Qwen3.8 only), Qwen3.8's 51B off-accelerator N-gram embedding tier, LongCat's zero-computation
experts, Peri-LN.

**Do not build** — located as dead or research-only: Llama 4 chunked attention / iRoPE (Behemoth
shelved, chunk-boundary blind spots cited); pure SSM/RWKV at frontier scale (every shipped SSM is a
hybrid with ≥1-in-6 attention); standalone Lightning Attention (MiniMax abandoned their own design);
YaRN as the long-context answer; MHA without GQA/MLA; DroPE/HoPE/Periodic RoPE and softpick (no shipped
checkpoint located).

---

## 7. Not located — the honest gaps

- Llama 5 architecture; a primary readable DeepSeek-V4 architecture section (one agent could not
  extract the PDF); an SB Intuitions Sarashina architecture report.
- A published closed-form law relating **bits-per-weight to loss below 2 bits at frontier scale**
  (ParetoQ is empirical and ≲3B).
- A fitted functional form for the **negative-returns** regime of test-time compute.
- Any independent replication of the T² overtraining coefficients.
- Any documented cross-implementation **Hadamard** convention disagreement (suspected exactness hazard
  for rotation-based PTQ, but not evidenced).

Per the absence-claims rule: these are *"this sweep did not locate"*, not *"these do not exist"*.
