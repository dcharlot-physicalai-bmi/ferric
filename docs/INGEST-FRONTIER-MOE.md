# Ingest plan — frontier-MoE-on-consumer-hardware

Three independent projects converged in 2026 on the same capability Ferric lacks: **running models far
larger than memory by streaming weights from disk, where placement affects speed but never results.**

| project | scale achieved | core method |
|---|---|---|
| [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) | 2.78T params in **8.24 GB RSS** (1.56 TB ckpt) | C99, no BLAS/GPU; 4-bit experts, MLA, trunk streaming |
| [colibri](https://github.com/JustVugg/colibri) | 744B–2.8T | int4 trunk resident, 19,456 experts streamed, LRU + learned pinning, 1-layer prefetch |
| [ds4 / DwarfStar](https://github.com/antirez/ds4) | DeepSeek V4, GLM 5.2 | **asymmetric role-based 2-bit**, imatrix, SSD streaming, Thunderbolt-RDMA distributed |

A fourth, independent confirmation: **KTransformers** (Tsinghua, SOSP 2025) does expert-granularity
CPU↔GPU offload with the same shape. Four implementations, one technique — this is settled, not speculative.

## The property they all advertise is already Ferric's thesis

- kimi-k3-in-c: *"byte-identical output at every budget"* between 8 GB and 224 GB.
- colibri: *"placement only ever decides speed — the router's decisions and the weights' precision are the
  same whether an expert answered from VRAM or from disk."*

That is **determinism under placement**, and it is the same guarantee Ferric already makes across fabrics
(CPU/Metal/Vulkan/WebGPU). They rediscovered it per-project as a feature; for Ferric it is an existing
architectural invariant that simply needs to extend along a new axis — the memory hierarchy.

**This is the ingest's organizing principle: one guarantee, two axes.**

```
              deterministic across FABRIC   (have)
                          ×
              deterministic across PLACEMENT (this ingest)
```

## Measured gap in Ferric

Surveyed 2026-08-03 — none of these appear anywhere in `crates/`:

| mechanism | status |
|---|---|
| mmap / lazy streaming weight load | **absent** |
| prefetch | **absent** |
| distributed (pipeline / tensor parallel) | **absent** |
| imatrix calibration | **absent** |
| KV-cache persistence to disk | absent (MCP session state is unrelated) |
| expert-granularity offload | absent |
| MLA | present, but **only in `ferric-llama/examples/instella_*`**, not in `src/` |
| LRU | only the Metal pipeline cache |

## Architecture — `ferric-tier`

A new crate owning the memory hierarchy, so no model code learns about placement.

```rust
/// Where a weight currently lives. NEVER observable in results — only in latency.
enum Tier { Vram, Ram, Disk }

/// Content-addressed weight handle: (layer, role, expert_idx)
struct WeightId { layer: u32, role: Role, expert: Option<u32> }

/// The invariant: fetch() returns byte-identical data for a given WeightId
/// regardless of which Tier served it. Placement is a scheduling decision.
trait TieredStore {
    fn fetch(&self, id: WeightId) -> Arc<Weights>;
    fn prefetch(&self, ids: &[WeightId]);      // hint only; never affects output
    fn pin(&self, id: WeightId, hot: bool);    // policy only; never affects output
}
```

The invariant is testable, and the test is the deliverable that makes this Ferric's rather than a copy:
**run the same prompt at several memory budgets and assert bit-identical logits.** kimi-k3-in-c claims
this property; Ferric should *enforce* it in CI.

## Work items

Ordered by (value ÷ effort). Mechanisms marked ⏳ are being extracted from the source repos.

### 1. Role-based asymmetric quantization  ⭐ start here
ds4's headline idea: quantize **by role, not by position** — routed experts at 2-bit (`up`/`gate`
IQ2_XXS, `down` Q2_K), while **router, shared experts, projections, and norms stay full precision**.

Directly corrects my own ternary work: the 16B QAT ternarized *uniformly*, and the mixed-precision
experiment I ran split by *layer position* (first/last f32) and came back null. Role-based is a different
and better-motivated hypothesis, and it is what ships in production. Note `down` gets more bits than
`up`/`gate` — asymmetry within the expert itself. ⏳ exact mapping being extracted.

Cheap to test: the 16B Instella forward is already verified (corr 0.99958) and the QAT harness exists.

### 2. Tiered weight store + expert streaming
The core build. ⏳ read granularity, file layout, and I/O mechanism (mmap vs pread vs io_uring) being
extracted from colibri and kimi-k3-in-c.

### 3. LRU + learned pinning + one-layer-ahead prefetch
colibri's policy layer. The interesting question is how the next layer's expert set is predicted *before*
its router runs. ⏳ extracting.

### 4. Placement-invariance test in CI
The differentiator. Same prompt, several budgets, assert bit-identical output.

### 5. imatrix calibration
Importance-matrix-guided quantization + corpus generation. Absent from Ferric entirely. ⏳

### 6. Persistent KV sessions
Disk-backed KV with warm resume (colibri claims 57× compression; ds4 uses it for instant multi-turn).
Lands in `ferric-serve`.

### 7. Distributed: pipeline + tensor parallel
ds4 runs tensor-parallel across two Macs over Thunderbolt-5 RDMA (1.66× on long prefill) and splits
experts across GPU pairs (120 t/s on 8×L40S). Largest item; last.

### 8. Promote MLA into `src/`
Already implemented and verified in the Instella examples — this is the audit's orphaned-capability
finding, and streaming needs it as a library API, not example code.

## What Ferric brings that these don't

Worth stating so the ingest stays additive rather than imitative: all three are **C, single-model-family,
single-purpose engines**. Ferric is Rust, memory-safe, multi-model, trains *and* serves, runs in the
browser, and already guarantees cross-fabric bit-reproducibility. The ingest adds their memory-hierarchy
capability to that base — and makes their headline determinism property a *tested invariant* rather than
a claim in a README.
