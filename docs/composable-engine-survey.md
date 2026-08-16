# Composable engine blocks: a survey, and what Ferric should take

**Status:** first pass, 2026-08-15. Prompted by Velox's "universal engine block" article (IBM/Presto,
2026-08-13).

## Scope and honesty about it

The ask was "all libraries and runtimes and solution architectures like Velox, globally." That set is
unbounded and this document does not claim to be complete. What it does claim:

- the **taxonomy** below is the right decomposition of "composable engine block", and any system not
  listed should fall into one of its rows;
- entries marked ✅ were confirmed by search or by reading the source this session;
- entries marked ○ are from prior knowledge and **have not been re-verified**, so treat them as
  leads rather than facts;
- one finding is a **negative** and it is the most important line in the document. See §5.

This review did not locate a systematic academic survey of composable *inference* engines, as
distinct from composable *analytical* engines, which are well covered.

## 1. What "composable" actually decomposes into

Velox's own layering is the clearest available taxonomy, and it generalises past SQL. Five seams,
each of which some project has tried to make reusable:

| seam | question it answers | data world | AI world |
|---|---|---|---|
| **plan IR** | how does a host hand work to an engine? | Substrait, PlanNode | StableHLO, ONNX, PJRT |
| **execution kernels** | who computes the primitive? | Velox operators, Arrow compute | FlashInfer, CUTLASS, CK |
| **memory & spill** | what happens under pressure? | MemoryPool / MemoryArbitrator | ⚠ largely absent |
| **storage/format** | how are bytes read? | Parquet, ORC, Nimble, Vortex | GGUF, safetensors |
| **semantics** | do two engines agree on the answer? | SQL function conformance | ⚠ **nothing** — see §5 |

The AI column is markedly emptier than the data column. That is the whole finding.

## 2. Data-plane execution engines (the Velox cohort)

| project | lang | note |
|---|---|---|
| ✅ **Velox** (Meta) | C++ | The article's subject. Library, not server: no parser, no optimizer, no scheduler. Layers = operators / vectors / DWIO / memory / storage / acceleration. |
| ✅ **Apache DataFusion** | **Rust** | The Rust analogue, and the one that matters most here. Embeddable, trait-extensible, first-class Substrait producer. Search consensus: *"for most new data platform projects in 2026, DataFusion is the default choice"* over Velox, because a library-first Rust design is easier to build on than to integrate into. |
| ✅ **Apache Gluten** | JNI | Bridge, not engine: serialises Spark plans to **Substrait** and dispatches to a native backend. The seam itself is the product. |
| ✅ **Apache Comet** | Rust | Spark accelerator built on DataFusion — the DataFusion-side answer to Gluten+Velox. |
| ✅ **Substrait** | IR | Cross-engine physical/logical plan IR. The interchange standard that makes the above swappable. |
| ✅ **cuDF / RAPIDS** | CUDA | GPU kernels; now callable *from inside* Velox (IBM+NVIDIA). Velox also has **Wave** (its own CUDA framework), **Breeze** (SIMD primitives), **UCX** (RDMA shuffle). |
| ○ DuckDB, ClickHouse/chdb, Polars, Daft, Databend, Vortex, Sail | C++/Rust | Embeddable engines and formats; Polars/Daft/Vortex/Databend/Sail are Rust. Vortex is the next-gen columnar format play. |
| ○ Photon (Databricks), Theseus (Voltron), Hyper (Tableau), Umbra/CedarDB | — | Mostly proprietary; useful as design references only. |
| ○ Arrow / Acero, Calcite, Ibis | — | Memory format + streaming exec, planner, and multi-backend frontend respectively. |

## 3. AI/ML-plane blocks (the cohort that matters for Ferric)

| project | seam | note |
|---|---|---|
| ✅ **FlashInfer** (NVIDIA) | kernels | **The closest thing to a Velox for inference.** Explicitly *engine-agnostic*; integrated into **vLLM, SGLang, MLC Engine** and custom engines. `FlashInfer-Bench` closes the loop by letting engines evaluate and deploy kernels. |
| ✅ **AMD AITER** | kernels | Multi-backend by design (CK / ASM / Triton). |
| ✅ **AMD Composable Kernels (CK)** | kernels | C++ templates, tile-op building blocks; AMD's CUTLASS analogue. |
| ○ CUTLASS / CuTe | kernels | The NVIDIA original of "kernels as a composable template library". |
| ○ **MLIR / IREE / TVM** | compiler+runtime | The compiler-side answer to composability. IREE is compiler *and* runtime and is the nearest structural analogue to Velox in ML. |
| ○ **OpenXLA / StableHLO / PJRT** | plan IR | PJRT is literally a *plugin* seam for backends — the ML equivalent of handing a PlanNode down. |
| ○ ONNX / ONNX Runtime | IR + runtime | The long-standing interchange bet; strongest on mobile NPU / Windows / browser. |
| ○ ggml / llama.cpp | runtime | Not composable by design, but is the **de facto architecture spec** for local inference (see §5). |
| ○ Candle, Burn, tract, wonnx | Rust | The Rust ML runtime cohort — Ferric's actual peers. |
| ○ ExecuTorch, torch.compile/Inductor, Triton, TensorRT-LLM, OpenVINO, Core ML / MPS Graph | — | Vendor and framework runtimes. |
| ○ **WebNN / WebGPU (wgpu)** | browser | The browser standardisation seam; directly load-bearing for Ferric's browser-first thesis. |
| ✅ *Llamas on the Web* (arXiv 2605.20706) | paper | Memory-efficient, performance-portable, multi-precision LLM inference **with WebGPU**. Closest published work to Ferric's browser position — read before making browser claims. |
| ✅ *Embodied.cpp* (arXiv 2607.02501) | paper | *Portable inference runtime for embodied AI models on heterogeneous robots.* Directly adjacent to the Physical AI positioning. |

## 4. What Ferric should take, in priority order

1. **Memory arbitration and spill.** The one Velox layer with no Ferric analogue. Today a 1076-row
   prefill through a 52-layer model was **SIGKILLed (exit 137, no message)** and the workaround was a
   manual `FERRIC_PREFILL_CHUNK` env var. Velox's answer is a hierarchical `MemoryPool` plus a
   `MemoryArbitrator` that reclaims and spills automatically; DataFusion has the same concept in
   Rust and is therefore the better source to read. Q93's numbers show the prize: peak per-task
   memory 3.3 GB → 80 MB. **Ferric should degrade predictably instead of dying.**
2. **The conformance artifact in §5.** Highest strategic value, lowest cost, nobody else is doing it.
3. **FlashInfer's engine-agnostic kernel contract.** Not the CUDA kernels — Ferric is WebGPU/Rust —
   but the *interface* discipline that let one kernel library serve vLLM, SGLang and MLC. Ferric's
   `ferric-tensor` is already shaped this way; the question is whether its ops are a stable contract.
4. **DataFusion's trait-extensibility patterns.** Ferric's crate split is library-shaped already;
   DataFusion is the proof that a Rust engine can be *built on* rather than *integrated into*.
5. **Substrait/PJRT as a model for a plan seam** — only if Ferric ever wants a host to hand it a
   graph. Speculative; listed for completeness, not recommended yet.

## 5. ⭐ The empty lane: there is no model-architecture conformance standard

Searching for a shared architecture spec with conformance tests returned the negative explicitly:
*"there isn't currently a unified standard mechanism ensuring all runtimes support new architectures
simultaneously"*, and *"model labs ship weights on their own schedule, and your inference runtime
adds support for each new architecture after the weights appear."*

**Velox's second reason for existing was exactly this problem, one layer down.** Meta found multiple
implementations of the same string function with different indexing rules; `substr()`, `round()` and
`date_trunc()` disagreed across engines, which *"undermined trust in analytical results."*

Model architectures are in that pre-Velox state right now, and this session is the evidence. Five
silent convention divergences found in one day, every one a case of two implementations disagreeing
with no error raised:

| convention | Ferric had | correct |
|---|---|---|
| rope pairing (llama) | NEOX split-half | NORM interleaved |
| YaRN `attn_factor` (deepseek2) | 0.7305 | 1.0 |
| YaRN in the dense loader | absent entirely | ramp + `1 + 0.1·ln(f)` |
| `expert_weights_norm` (DeepSeek-V2) | renormalised | off |
| `rope_scaled` positions | read out of bounds → no rotation at all | — |

The failure mode is **worse than SQL's**: a wrong `substr()` returns a visibly wrong string, while a
wrong rope returns *fluent, plausible prose*. Every one of these passed shape checks, finiteness
checks and the whole test suite.

Velox's answer was a shared *implementation*. Inference cannot copy that — the de facto spec is
`llama.cpp/src/models/*.cpp`, it is C++, and every other runtime hand-translates it. Ferric ported
Gemma 4 and DeepSeek-V2 that way today.

**The achievable version is a conformance artifact, not a shared engine**: per architecture, a set of
reference tensor sums at named points (`attn_norm-0`, `Qcur` post-rope, `kv_cmpr-0`, …) that any
runtime can diff against. `llama-eval-callback` already emits exactly this, and Ferric's
`FERRIC_DUMP=<block>` already consumes the same names. Two of today's bugs were localised in a single
pass by that diff after hours of reasoning had produced wrong answers.

That is a genuinely unclaimed layer, it is cheap, and it is the one place where being late to a
model is a *bounded* cost rather than an open-ended debugging session.

## 6. What this survey has not done

- No Chinese-ecosystem sweep (StarRocks, Doris, ByteDance/Alibaba internal engines), and given that
  Chinese open models are 41% of HF downloads, that omission is likely to matter.
- No verification of the ○ rows.
- No read of Velox's newer siblings **Nimble, Axiom, Collagen** beyond their names.
- No cost or benchmark reproduction — every number quoted here is the vendor's.
