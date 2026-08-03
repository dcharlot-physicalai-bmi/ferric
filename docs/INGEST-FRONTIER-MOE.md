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

---

# Extracted mechanisms — kimi-k3-in-c

Source read directly (5,600 LOC C). These are the load-bearing details; several invert what a
reasonable engineer would build by default.

## ⭐ Layer streaming: pinned prefix + 1-slot ring — **NOT LRU**

A transformer walks layers 0→N cyclically every token. That is the textbook LRU pathology: with fewer
slots than layers, the scan returns to layer 0 exactly when it has become least-recently-used and was
just evicted ⇒ **hit rate 0 regardless of RAM**. Their design instead pins layers `0..npin-1` and streams
the rest through **one** slot, giving a deterministic `npin/93` hit rate where every extra GB buys its
fair share.

Two refinements that matter:
- Pinned layers get **exact-size** allocations, not uniform slots (layer 0 is 2.34 GB from its 33792-wide
  dense MLP vs 1.27 GB typical — uniform slotting wastes ~half the budget).
- Ring size is computed from **streaming layers only**. Sizing it over all layers reserves room for
  layer 0, which prefix-pinning pins first — ~1.17 GB wasted at every budget.
- Ring and pin count are mutually dependent (smaller ring → more pins → smaller ring): resolved by a
  bounded 4-pass fixed-point loop that converges in 2–3.

**Corollary for Ferric: the expert cache and the layer cache need *different* policies.** Expert reuse is
data-dependent (LRU is defensible); layer access is cyclic (LRU is provably worthless).

## I/O: `pread` + `O_DIRECT`, no mmap anywhere

Deliberate: buffer-owned pages never become file-backed mappings, so peak RSS reflects what is actually
resident rather than the whole 1.56 TB. On-disk layout packs each layer's tensors as **one contiguous
4096-aligned run**, so loading a layer is exactly one `pread`.

**2 MB alignment + `MADV_HUGEPAGE` is not cosmetic.** `O_DIRECT` pins destination pages; a 2.37 GB slot
on 4 KB pages is ~578,000 pins *per read* × 93 reads/token ≈ 53.8M pin operations ≈ 10 s/token of kernel
bookkeeping — and it is invisible to an I/O timer that brackets only the `pread`.

**No I/O overlap exists** (their `prefetch` is inert under O_DIRECT, since `FADV_WILLNEED` warms the page
cache that O_DIRECT bypasses). Worth ~16 s/token. Because the access order is fixed and known forever
(0,1,…,92), double-buffering is trivially correct — **this is the biggest single win available to a Rust
port**, and Rust's ownership model makes the concurrent-writer hazard they cited tractable.

## Expert cache: LRU is defeated *by the model*

K3's router is trained with **Quantile Balancing to flatten expert usage** — and flat usage is exactly
what LRU cannot exploit. Trace replay over 100,096 requests:

| cache | LRU | Belady | gap |
|---|---:|---:|---:|
| 8–64 GB | **36.24% (flat)** | 39→62% | **up to 25.5 pts** |
| 128 GB | 49.19% | 84.59% | 35.4 pts |
| 192 GB+ | 90.00% | 90.00% | 0 (compulsory-miss ceiling) |

**Policy, not capacity, is the available win.** Worth pairing with colibri's "learned pinning".

## Budget allocation: trunk before expert cache (1.69× measured)

The trunk is re-read *entirely* every token (108.81 GB) while only ~25.8 GB of experts are touched, so a
GB given to the trunk removes ~1.17 GB/token of *guaranteed* traffic while a GB given to the expert cache
removes nothing measurable below ~36 GB.

## ⭐ The determinism recipe — why output is byte-identical from 8 GB to 224 GB

The budget decides **where bytes come from, never how they are combined**. Nothing is quantised, skipped,
or approximated as a function of budget. Concretely:

1. **Parallelism only over independent output rows/experts** — no reduction crosses threads ⇒ identical
   at any thread count.
2. **Hand-written accumulation order**: four f64 accumulators partitioned `i%4`, reduced
   `(a0+a1)+(a2+a3)`, plus a verbatim scalar tail. Written out because FP reduction may not be
   reassociated without `-ffast-math` — they refuse to let the compiler choose.
3. **SIMD bit-identical to scalar by construction**, not approximately: a `__m256d` holds exactly 4
   doubles, so lane `i%4` ≡ scalar accumulator `i%4`, same order within each lane, same reduction tree.
4. **Mul-then-add, never FMA** — explicit `add(v, mul(...))`, `-ffp-contract=off`, never `-ffast-math`.
5. **Streaming is lossless** — trunk copied byte-for-byte; experts multiplied in their native MXFP4.
6. **Greedy only** — strict `>` argmax, ties to lowest index; no RNG in the binary.
7. **Routing independent of cache state** — same prompt picks the same experts regardless of residency.

Evidence: a 12-rung ladder from 8 GB to 224 GB whose emitted token ids are character-identical, asserted
in-harness, with a guard that refuses to pass vacuously on an unreadable result file.

**Direct bearing on `FERRIC_COOP`** (measured today: 13× at max|Δ| 5.0e-8): that delta *is* reassociation.
Rule 2/3 is the constructive fix — a coop kernel with a hand-fixed accumulation order could plausibly be
both fast and bit-identical, which would let it become the default without weakening the guarantee. Worth
attempting before accepting the speed/determinism trade as inherent.

## Quantization: MXFP4, never materialised

Group 32, `packed[rows][in/2]` u8 two-per-byte with a **separate** scale plane, **low nibble = EVEN
element**, E8M0 scale (`2^(sb-127)`, `sb==255` ⇒ NaN ⇒ zero the group so one bad byte cannot poison a
row). The kernel never widens to fp32: one expert is 132 MB dequantised vs 17.55 MB packed, and a token
touches 1,472 experts ⇒ **194 GB/token materialised** if widened. Matrix-vector is memory-bound, so
reading 7.5× fewer bytes is *faster*. Inner loop **expands a group to a 64-float scratch, then does a
plain dot** — the split exists because the dot cannot vectorise with a table lookup inside the
accumulation. Scale applied **once per group**, not per element.

## Correctness traps worth inheriting

- **Mark a cache slot empty *before* reading into it.** Registering only on success leaves the slot's key
  naming the OLD layer over a buffer that no longer holds it ⇒ the next bind counts a **HIT** and returns
  corrupt weights.
- Slot keys need **three** states (`valid` / `EMPTY` / `INFLIGHT`); with only two, parallel prefetch reads
  raced into one buffer. Cost one wrong token on the real model and nothing in the fixtures.
- Slot size must be alignment-rounded, not just the arena base. On the real checkpoint it held *by
  coincidence*; with any other expert size every O_DIRECT read after the first returns 0 bytes silently.

## Their honesty markers — adopt these

- **Run-to-run spread is 33% on identical configuration.** "Differences smaller than 33% are not effects."
  (Ferric reached the same conclusion via the ternary kernel bench: a 10% "win" that flipped sign on
  re-measurement. Both projects independently landed on: report the spread, not a single sample.)
- Trace-replay hit rates are labelled an **upper bound, not a forecast**.
- The int4-vs-int8 study is **weight reconstruction error, not output quality** — no downstream logit or
  token comparison was ever run.

## Speed reality

8 GB → 32.69 s/token; 224 GB → 19.21 s/token. **28× the RAM buys ~1.7× the speed**; I/O is 41–77% of wall
clock. Their own framing: *"The point was never the speed."* The point is that it runs at all.
