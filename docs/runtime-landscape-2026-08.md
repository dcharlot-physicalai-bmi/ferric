# AI runtime & harness landscape, 45-day window to 2026-08-16

**Method.** Triaged before sweeping, per the multi-region sweep rule. Five axes searched: serving
runtimes, Rust cohort, browser/WebGPU, open-weight model frontier, harnesses. Numbers are vendors' or
third-party bloggers' unless marked measured-here. This did not locate a systematic benchmark run
under one methodology, so **treat cross-source token/s figures as indicative, never comparable**.

## 1. Serving runtimes — the window's releases

| runtime | in-window | what shipped |
|---|---|---|
| **vLLM 0.25.1** | Jul 2026 | **Model Runner V2** (throughput on newer GPU archs), **EAGLE 3.1** speculative decoding |
| **SGLang 0.5.15** | Jul 2026 | **RadixAttention** (automatic KV reuse across shared prefixes), multi-model serving in one process, native structured output. Reported **+29% throughput over vLLM on H100**, up to **6x on RAG** |
| **TensorRT-LLM 1.3.0rc22/23** | Jul 22 / Jul 30 | shared-expert combine fusion, paged MQA logits decode tuning, fused RMSNorm/RoPE, EAGLE3 dynamic-tree kernels |
| TensorRT-LLM | **Aug 5** | **day-0 support for OpenAI GPT-OSS-120B / GPT-OSS-20B** |
| **HuggingFace TGI** | Mar 2026 | ⚠ **moved to maintenance mode**; now points users at vLLM, SGLang, llama.cpp, MLX |

**Two structural reads.** First, TGI's retirement is consolidation: the field has settled on four
runtimes, and Ferric is not among the four anyone is redirected to. Second, **speculative decoding is
now table stakes** — EAGLE 3.1 in vLLM, EAGLE3 dynamic-tree kernels in TensorRT-LLM, in the same
six-week window. `ferric-serve` has a draft-cache path; whether it is EAGLE-class is unverified.

**RadixAttention is the single most ingestible idea here.** Automatic KV reuse across *shared
prefixes* is what makes agent loops and RAG cheap, and it is a data-structure change (a radix tree
over the cache), not a kernel. Ferric has a one-slot prompt-prefix cache in `ferric-serve`; a radix
tree generalises it to arbitrary branching — which is exactly the agentic workload shape.

## 2. Rust cohort — Ferric's actual peers

| project | note |
|---|---|
| **mistral.rs** | Candle-based, Metal/CUDA/CPU, OpenAI-compatible server, ~6.3k stars. The Rust runtime with mindshare. |
| **Candle** | HF's Rust framework; the substrate mistral.rs is built on. |
| **Burn** | Rust DL framework that **compiles to WGPU** — overlaps Ferric's fabric claim. |
| ⚠ **Ratchet** | **wgpu-based ML inference with an explicit focus on web support.** This is Ferric's closest direct competitor by design intent: same backend, same lane. |

Ferric's "pure Rust ∩ browser ∩ native" claim is narrower than it reads: Burn already compiles to
WGPU, and Ratchet is explicitly wgpu + web. The defensible intersection is **pure Rust ∩ one codebase
∩ native AND browser ∩ sound verification ∩ joules** — and only the last two are uncontested.

## 3. ⚠ The browser lane is contested, and Ferric is behind in it

This is the finding that matters, because browser-first is Ferric's stated differentiator.

| engine | model | tok/s | hardware |
|---|---|---|---|
| **WebLLM** | Llama 3.1 **8B** q4 | **41** | M3 Max — *~80% of the same model native via MLC-LLM* |
| WebLLM | Phi 3.5 Mini | 71 | — |
| Transformers.js v4 | 20B | ~60 | "capable hardware" |
| hand-tuned WGSL | Qwen 3.5 INT4 | **180** | — |
| **custom WebGPU kernels** | **LFM2.5-230M** | **1,400** | M4 Max |
| **Ferric (measured here)** | Llama-3.2 **1B** q4_K_M | **38.6** | M5 Max, 5 runs, 4% spread |

Ferric's figure is verified rather than asserted: 25.3 / 25.6 / 26.3 / 26.3 / 26.4 ms/tok across five
runs, a **4% spread**. Earlier in the same session the identical command produced 39.8-68.6 ms/tok — a
**1.7x spread** — and the difference was entirely methodological: `cargo run` in a loop re-checks and
relinks the binary on every iteration, so the build competes with the thing being timed. **Build once,
then time.** That single mistake produced every noisy measurement in this repo's recent performance
notes, including one that was written up as a structural conclusion and had to be retracted.

Two conclusions, and neither is comfortable:

1. **WebGPU has no structural penalty.** WebLLM reaches ~80% of native on the same model. This
   independently confirms the re-corrected finding in `composable-engine-survey.md` §7.2.4 — the
   earlier "~3x structural to WGSL" claim was an artifact of a contended machine, and the browser
   ecosystem's own numbers say the ceiling is not the API.
2. **Ferric is roughly an order of magnitude off the browser state of the art.** Ferric does ~40 tok/s
   on a **1B**; WebLLM does 41 tok/s on an **8B**. Same lane, ~8x the model, same speed. And
   LFM2.5-230M at 1,400 tok/s shows what hand-written WebGPU kernels achieve at small scale.

**The gap is not the fabric, and it is not novel.** Others reached 80% of native on the same API
Ferric uses. That is good news for the thesis and bad news for the current implementation.

Corroborating, from *Llamas on the Web* (arXiv 2605.20706): "existing browser inference engines
dynamically allocate GPU memory and load weights into WebGPU inefficiently, leading to slowdowns as
memory grows over the course of execution and even crashes as some browsers enforce hard-caps on
memory per-tab." That is Ferric's exit-137 SIGKILL described from the outside, and it is the same
pre-inference allocation problem MNN solves.

## 4. Open weights caught the frontier — in this window

- *"A wave of open-weight model releases in July 2026 has undercut predictions that AI development
  would consolidate into a handful of closed, proprietary labs."*
- **Chinese labs (DeepSeek, Moonshot, Z.ai, Alibaba, MiniMax) are "the effective owners of the
  open-weight frontier."**
- **DeepSeek V4 Flash** is *"the first open-weight model that teams immediately dropped into real
  agentic pipelines as a plausible substitute for an Anthropic- or OpenAI-class frontier model."*
- **GLM-5.2** (Z.ai) is *"only a few months behind OpenAI's GPT-5.5 and Anthropic's Claude Opus 4.7"*
  on cyber and bio capability evaluations.
- **OpenAI shipped open weights**: GPT-OSS-120B and GPT-OSS-20B, day-0 in TensorRT-LLM on Aug 5.

This validates the 30-day cadence rule and sharpens it: the models people will run locally are
arriving faster than Ferric adds architectures, and **the frontier ones are now agentic-capable**,
which raises the value of RadixAttention-style prefix reuse specifically.

## 5. What parity actually requires

Ranked by gap size, not by appeal:

1. **Kernel throughput.** ~8x behind WebLLM in Ferric's own lane. Everything else is secondary until
   this moves. Instruments and the target are in `composable-engine-survey.md` §7.2.
2. **Pre-inference memory planning.** Named by MNN, independently described by *Llamas on the Web* as
   an industry-wide browser failure, and experienced here as exit 137.
3. **RadixAttention-equivalent prefix reuse.** A data structure, not a kernel — cheapest real win, and
   aimed at the agentic workload the new open-weight frontier enables.
4. **EAGLE-class speculative decoding.** Now standard in both vLLM and TensorRT-LLM. Ferric's draft
   path exists but is unverified against that bar.
5. **Architecture coverage on cadence** — GPT-OSS, DeepSeek V4, GLM-5.2, MiniMax M3 are all absent.

## 6. Positioning, corrected by this sweep

"SOTA go-to based on memory safety, Rust performance, and universal compute fabric" does not survive
contact with the window's evidence *as a performance claim*:

- **Rust is not a differentiator** — mistral.rs, Candle, Burn and Ratchet are all Rust.
- **WebGPU is not a differentiator** — Burn compiles to WGPU; Ratchet is wgpu-and-web by design;
  WebLLM already hits 80% of native.
- **Performance is currently a deficit, not an edge** — ~8x behind in the browser lane.

What *is* uncontested, on the evidence gathered across this and the composable-engine survey:
**joules as the objective function** (zero mentions of joule/watt/kWh across NVIDIA Switchyard and
DeepSeek Harness), **sound verification**, and **one codebase spanning native and browser**. The
honest claim is a *different objective*, not a faster one — and it stays honest only while the
throughput deficit is stated alongside it.

## 7. Not covered

Harness/agent layer beyond DeepSeek Harness (already surveyed); Ollama/LM Studio/MLX in-window
release detail; no reproduction of any quoted benchmark; no Chinese-language sources.
