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

## 6. ⭐ The Chinese ecosystem inverts the picture

Swept 2026-08-15, because Chinese open models are 41% of HF downloads and the omission looked
dangerous. The finding is an **asymmetry**, and it is the opposite of the West's.

**Analytics: no composable block located.** StarRocks (Apache-2.0, forked from Doris 2020),
Apache Doris, and ByteDance's ByteHouse are vertically-integrated MPP databases — vectorized, SIMD,
CBO, 3-10x operator gains, but shipped as *systems*, not as libraries you link. This review did not
locate a Chinese Velox/DataFusion equivalent, and the search for "composable library reuse" in that
cohort returned nothing. That is a located absence, not proof of one.

**Edge inference: this is where the composability work is.** And it is exactly Ferric's ground.

| project | owner | why it matters here |
|---|---|---|
| ✅ **MNN** | Alibaba | *"A Universal and Efficient Inference Engine"* (arXiv 2002.12418), plus MNN-LLM (arXiv 2506.10443). Multi-backend by design — CPU / GPU / NPU, and 3.6.1 (Jul 2026) added a **Qualcomm Hexagon DSP** backend. On-device LLM + Edge AI, battle-tested at Alibaba scale. **Ferric's closest global peer by mission**: universal, multi-backend, on-device. Read the architecture paper before making "runs everywhere" claims. |
| ✅ **KTransformers** | kvcache-ai (Tsinghua) | *"A Flexible Framework for Experiencing Heterogeneous LLM Inference/Fine-tune Optimizations."* Module **injection** via YAML — the composability idea applied to inference optimisations rather than kernels. Runs 100B+ models on a single RTX 5090 (32 GB) via CPU/GPU heterogeneous compute; Ascend NPU support added Oct 2025. The architectural pattern is the ingest, not the CUDA. |
| ✅ **CANN** | Huawei | Driver + runtime + libraries, deliberately CUDA-analogous for Ascend NPUs; its Graph Engine compiles whole-graph representations into execution plans. The vertically-integrated counter-model to composability. |
| ○ ncnn, TNN | Tencent | Mobile ARM inference; ncnn shipped an x86 SIMD PixelShuffle speedup in 2026. |
| ○ Paddle Lite (Baidu), MegEngine Lite (Megvii), OpenPPL (SenseTime), Bolt (Huawei), LMDeploy (Shanghai AI Lab) | — | The rest of the edge cohort; unverified. |

**Why this matters for Ferric's positioning.** The West's composability push is analytics-first
(Velox, DataFusion, Substrait) and has barely reached inference. China's is **edge-inference-first**
and has barely reached analytics. Ferric sits in the quadrant China is contesting, not the one the
Velox article describes — so MNN, not Velox, is the sharper comparand for "universal on-device
engine", and the honest differentiators against it remain pure Rust, browser-first, and joules.

## 7. ⭐ MNN read in full — and it beats Velox as Ferric's model

Read arXiv 2002.12418 (pp. 3-7) rather than the abstract. Three things transfer, and one of them
replaces the §4 priority-1 recommendation.

### 7.1 Pre-inference: the answer to today's SIGKILL, and it is not Velox's

MNN's central claim rests on an observation Velox cannot make: **in inference the shapes are known
ahead of time.** "Since input size is determined or can be pre-processed to a target size, MNN can
infer the exact required memory for the entire graph by virtually walking through all operations and
summing up all allocation and freeing." It then pre-allocates one pool and reuses it every session.

Velox spills because analytical query shapes are *unpredictable*. MNN pre-allocates because inference
shapes are *predictable*. **Ferric is inference, so MNN's model fits and Velox's does not.** Today's
exit-137 SIGKILL was a 1076-row prefill through a 52-layer model — a shape fully knowable before the
first dispatch. The right fix is a pre-inference pass that computes peak memory and either
pre-allocates or chooses a chunk size, not a `MemoryArbitrator`, and not a hand-set env var.

**This supersedes §4 item 1.** Read MNN's pre-inference before DataFusion's memory pool.

### 7.2 Preparation-execution decoupling is worth 50-75% on GPU

Separating setup from compute cut MNN's inference time by **7-8% on CPU and 49.5-75.2% on GPU**
(Table 2: MI6 Vulkan 63.6 → 15.8 ms, ↓75.2%; P10 Vulkan 41.0 → 20.7 ms, ↓49.5%), because "setting up
command buffer and its related command descriptions is time-consuming."

WebGPU has exactly this cost — pipeline creation, bind groups, command encoding. Ferric's
64-224 ms/tok against llama.cpp's ~17 was unexplained, and this was the first specific hypothesis
with a published magnitude behind it.

**⚠ MEASURED, AND REFUTED.** `run_ids` now prints the split. On Llama-3.2-1B, steady-state decode
(counters reset after prefill):

```
decode 24 tok in 579.8 ms (24.2 ms/tok)
host-side setup: 6.0% of wall (35.0 ms of 579.8 ms)
  pipeline lookup     2.86 ms   bind groups     5.48 ms
  pass recording      1.13 ms   submit         25.57 ms
  5808 dispatches in 792 submits (242 dispatches/token)
```

Host-side setup is **6%**, not 50-75%. Pipelines are already cached by content hash and dispatches
are already batched ~7:1 into submits, so MNN's GPU win does not transfer — Ferric had mostly done
this already. The hypothesis was reasonable and wrong, and the measurement cost one run.

**What the same number does expose: 242 dispatches per token** on a 16-layer 1B model — roughly 15
per layer. Host time for those is cheap, but each carries GPU-side launch latency the host timer
cannot see; at a conservative 20 µs that is ~5 ms/token, ~20% of the 24.2 ms. And llama.cpp runs this
same checkpoint at 426.7 tok/s (2.3 ms/tok), a **10x gap**.

So the lead is **operator fusion / dispatch count**, not command-buffer setup — which is MNN's
*offline graph optimizer* (operator fusion, replacement) rather than its pre-inference stage.

### 7.2.1 The dispatch mix, traced

`FERRIC_TRACE_KERNELS=1`, Llama-3.2-1B, one decode token over 16 layers:

| kernel | /token | /layer | |
|---|---|---|---|
| `matmul_q4_k_splitk` | 48 | 3 | qkv, o, down |
| `rmsnorm` | 33 | ~2 | |
| **`rope_scaled`** | **32** | **2** | q and k dispatched SEPARATELY |
| `matmul_q6_k_splitk` | 17 | ~1 | + lm_head |
| `matmul_q4_k_swiglu` | 16 | 1 | gate\|up already fused ✔ |
| `fattn`, `kv_write2`, `add_rmsnorm` | 16 ea | 1 | already fused ✔ |
| **`gather`, `cat`, `binary`** | **16 ea** | **1 ea** | **pure data movement** |

Two concrete targets, both bounded:

1. **`gather` + `cat` + `binary` = 48 dispatches/token (20% of all dispatches) doing no arithmetic.**
   This is the same class of finding that once showed the per-token cost was three `gather` copies
   per layer rather than the attention everyone assumes; it is down to one, and the remaining three
   movement kernels per layer are the next slice.
2. **`rope_scaled` runs twice per layer.** A fused single-dispatch q+k rope already exists, but
   `fuse_rope` requires `self.rope_freqs.is_none()` — the guard's comment says rope-scaling "goes
   through a different kernel that has not been shown equivalent". That kernel has now been fixed and
   verified tensor-exact against `llama-eval-callback` (§ rope work, 2026-08-15), so the guard is
   revisitable and is worth 16 dispatches/token on every Llama-3.1+ and YaRN model.

### 7.2.2 Fusion done, and it prices the dispatch itself

The rope fusion is shipped (target 2 above). A/B on the same prompt, `FERRIC_NO_FUSE_ROPE=1` as the
control:

| | dispatches/token | ms/tok |
|---|---|---|
| split (old) | 242 | 25.6 |
| **fused** | **210** | **25.1** |

Generated text is **token-identical** between the two, which is the check that matters.

**13% fewer dispatches bought ~2% wall time.** Those two numbers together price a dispatch: 12 ms
saved over 24 tokens × 32 removed dispatches ≈ **15 µs per dispatch**. At 210 dispatches/token that
is ~3.2 ms of the 25.1 ms, i.e. **~13% of decode is launch overhead** — real, worth removing, and
nowhere near the gap.

**So the third hypothesis is bounded too.** Setup was 6%; dispatch overhead is ~13%; the remaining
~10x against llama.cpp's 2.3 ms/tok is **kernel efficiency**, not orchestration. Fusing every
remaining movement kernel would leave Ferric still ~8x behind. That is a more useful thing to know
than another plausible story, and it redirects effort from the graph layer to the kernels.

### 7.2.3 The diagnosis, quantified: the quantised matmul is ~10x off memory bandwidth

`FERRIC_PROFILE=1` (read ratios only — the profile device-syncs at every boundary and costs ~20%):

```
── profile (ms, 464.8 total) ──
  attn      216.9  46.7%      ffn       191.3  41.2%
  lm_head    49.9  10.7%      embed       6.7   1.4%
```

`attn` here is the whole block — qkv matmul, rope, `fattn`, o matmul — and at a 6-30 token context the
attention math itself is negligible, so **~88% of decode is matmul**, and `lm_head` alone (one
`[2048 × 128256]` Q6_K matmul) is 10.7%.

Decode of a 1B model is **memory-bandwidth-bound**: every weight is read once per token.
Llama-3.2-1B-Q4_K_M is ~0.77 GB, so

```
Ferric     0.77 GB / 25.1 ms  ≈   31 GB/s
llama.cpp  0.77 GB /  2.3 ms  ≈  335 GB/s
```

Apple-silicon unified memory offers on the order of 100-400 GB/s, so llama.cpp is running near peak
and **Ferric is achieving roughly a tenth of available bandwidth**. That ratio is the 10x, and it is
not distributed across the system — it is concentrated in `matmul_q4_k_splitk` and
`matmul_q6_k_splitk`.

### 7.2.4 ⚠⚠ CORRECTION, THEN RE-CORRECTION — and the second one is the lesson

**First correction (wrong).** One run of `examples/bandwidth.rs` reported a WGSL read ceiling of
111 GB/s and I concluded ~3x of the gap was "structural to the WGSL/wgpu path", committed a
positioning claim that Ferric should never promise throughput parity, and wrote it into durable
memory.

**Re-measured, three times, on an idle machine:**

```
  scalar u32, wg64    385.7 / 399.8 / 473.6 GB/s
  vec4<u32>, wg64     439.9 / 463.1 / 466.3 GB/s
  wg128 401.8 · wg256 375.3 · wg512 290.0 · wg1024 172.8
```

The 111 GB/s reading was an **outlier taken on a contended machine** — that run followed hours of
10 GB model loads, full test suites and repeated inference. The identical configuration measures
3.6x higher when the GPU is idle.

**So the real picture is the opposite of what I committed:**

| | |
|---|---|
| WGSL read ceiling | **~385-475 GB/s** |
| llama.cpp effective stream | ~326 GB/s |
| Ferric quantised matmul | ~31 GB/s |

**WGSL reads memory faster than llama.cpp streams weights.** There is no structural bandwidth
penalty. The matmul is ~13x below the read ceiling, and the *entire* 10x is addressable by kernel
work. §7.2.3's original conclusion was right; the "correction" that replaced it was not. And vec4 is
faster than scalar, as expected — that "anomaly" was the same artifact.

**The lesson, which is worth more than the number.** Twice in one thread I drove a *positioning*
claim from a *single unrepeated microbenchmark*. The first time the instrument already existed and I
reasoned instead of running it; the second time I ran it once, on a machine I had just spent hours
loading, and propagated the result into repo docs and durable memory inside ten minutes.

A microbenchmark taken on a busy machine is not a slow measurement, it is a **wrong** one, and
nothing about its output says so. Repeat before claiming, and repeat *especially* when the number
supports a conclusion that would change strategy.

## 8. What this survey has not done

- No verification of the ○ rows.
- No read of Velox's newer siblings **Nimble, Axiom, Collagen** beyond their names.
- No cost or benchmark reproduction — every number quoted here is the vendor's.
