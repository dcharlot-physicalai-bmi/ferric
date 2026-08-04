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

---

# Extracted mechanisms — ds4 / DwarfStar

## ⭐ Role-based quantization: the exact map, and *why* it is asymmetric

No role metadata exists in the file format. Roles are derived from **tensor name** at load time; precision
rides the ordinary per-tensor GGUF type id.

| role | 2-bit build | bpw |
|---|---|---|
| routed `gate` (w1), routed `up` (w3) | **IQ2_XXS** | 2.0625 |
| routed `down` (w2) | **Q2_K** | 2.625 |
| attention projections, shared expert, output head | Q8_0 | 8.5 |
| router / `ffn_gate_inp` | F16 | — |
| norms, biases, all 1-D | F32 | — |

**Why quantizing only routed experts is enough**: they are `3 × 43 × 256 × 4096 × 2048` = 277.0B of the
284B total = **97.5% of all parameters**. Leaving router, shared experts, projections and norms at full
precision costs ~2.5% of the weights and buys back the accuracy the 2-bit experts lose.

### The mechanism behind `down` > `up`/`gate` — this is the transferable idea

- IQ2_XXS is a **sign-symmetric codebook with no zero and no offset term** (magnitudes {1,3,5,7} with
  signs stored separately, one f16 block scale refined by a 4-bit per-32 nibble).
- Q2_K carries **both `d` and `dmin`** — an *affine* quantizer with per-16 scale *and* min.
- `gate`/`up` consume the **RMSNorm'd hidden state: near-symmetric about zero**.
  `down` consumes the **SwiGLU product: gated and one-sided.**

With symmetric inputs, symmetric weight error partially **cancels** across the dot product; with one-sided
inputs it **accumulates coherently**. So the layer whose *input distribution is one-sided* needs the lower
weight error, and an affine quantizer's offset is what buys it.

**Bearing on Ferric's ternary work.** Ternary {−1,0,+1} is a *symmetric* quantizer. The 16B QAT ternarized
**uniformly**, so this predicts `down_proj` was the weak link — a sharper and more mechanistic hypothesis
than the layer-position split (first/last f32) already tested, which came back null. Testable cheaply on
the existing harness: ternary gate/up, affine-or-higher `down`.

### The asymmetry is kernel-enabled, not merely statistical
`gate` and `up` are computed by one fused kernel that reads the activation **once** and applies it to two
weight rows (`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`) — which is why the loader *hard-exits* if their
types differ (`ds4.c:5043`). `down` is a separate kernel fused with the 6-expert weighted sum
(`..._q2_K_sum6_f32`), so giving it a different type costs nothing. **Design the quantization split and
the kernel fusion together, or the split has no place to live.**

## The same failure, finer grain — colibri's int8 MTP head

`eh_proj [D, 2D]` multiplies `[embedding_norm ; hidden_norm]`, whose two column halves differ **~20–30× in
scale** (embedding-half absmax ~0.05, hidden-half ~1.5). Per-row int4 uses **one** scale per row, so every
embedding-half weight lands below half a step and `rint` rounds the **entire half to exact zeros** (packed
`0x88`). MTP draft acceptance: **0–4% at int4, 39–59% at int8.**

**The escape hatch is the real lesson**: group-scaled (gs64) int4 does **not** fail. The pathology is
specifically *one scale spanning two populations*, not the bit width. Any Ferric quantizer that puts one
scale across heterogeneous columns inherits this bug, silently — the model still emits plausible tokens.

## imatrix calibration — what is actually measured

Two statistics, matching the two roles, collected by hooking the real prefill graph (inference math is
unchanged):
- **gate/up**: `Σ x²` over the FFN-normalized activation
- **down**: `Σ mid²` over the routed SwiGLU row **after route weighting**

Accumulated per `(layer, expert)`, divided by that expert's own visit count; **experts never selected get
an all-1.0 vector**. Entry into the quantizer is importance × local magnitude:
```c
sigma2    = Σ(x[j]²) / 256;
weight[i] = imatrix[i] * sqrtf(sigma2 + x[i]*x[i]);
```
Corpus: 4,690 records / ~2.92M tokens, exactly balanced think/nothink, pre-rendered **with the real chat
template** so calibration sees the special-token patterns agents actually produce.

Measured benefit is modest and honestly reported: **−1.95% NLL** (0.1774 → 0.1739) over 100 continuations.

## Determinism: ds4 takes the *opposite* side from kimi-k3

ds4 builds with `-ffast-math` and `--use_fast_math`; kimi-k3-in-c refuses both so its scalar / OpenMP /
AVX2 paths stay bit-identical. **Ferric sides with kimi-k3** — bit-reproducibility is the differentiator.
Worth noting ds4 still needs determinism where it *must* have it, and pays for it explicitly: fixed-tree
`__fadd_rn` combine kernels to match single-GPU reduction order, and a `moe_scatter_sorted_pairs_
deterministic_kernel` because "atomic append order changes logits."

## ⭐ Do not trust `cudaDeviceCanAccessPeer`

At init, ds4 probes **every ordered GPU pair** with 4 sizes {4 KiB, 256 KiB, 1 MiB, 16 MiB} × 4 iterations,
byte-compares, and demotes that **direction** to a pinned-host bounce on a single mismatch — because an
RTX 6000 Ada was observed **silently corrupting peer copies while returning success**. Ferric should adopt
this stance for any multi-device path: capability bits are a claim, a byte-compare is evidence.

## Speculative decoding: greedy prefix match, no rejection sampling

DSpark accepts a draft token only on **exact top-1 equality** (`if (row_tops[i-1] != drafts[i]) break;`).
No `max(0, p−q)` residual, no resampling; the speculative path is only reachable at `temperature <= 0`.
"Correctness-gated" therefore means **bit-identical committed token stream vs non-speculative decode**,
and the test enforces exactly that: run each prompt twice, `cmp -s`, fail on any byte difference.

Rollback detail worth copying: the raw sliding-window KV ring is **deliberately not restored** (it is
append-only and will be overwritten by replay); only compressor/indexer frontiers and row counters rewind.

## Honest reading of the quality claims

The README says the 2-bit quants are "verified to be actually high quality." What is actually committed:
- The **only** quantified 2-bit-vs-4-bit delta in the repo is for **GLM 5.2, not DeepSeek Flash**:
  first-token match 95→92 /100, API top-1 0.942→0.890, pair-order 0.880→0.800.
- **No** committed q2-vs-q4-vs-FP NLL for Flash; no per-benchmark drop; no aggregate eval score.
- The MODEL_CARD benchmark table is **upstream FP4+FP8 numbers, never re-measured on the quantized GGUFs.**
- Release blockers are *relative* (regression vs last-known-good), not an absolute bar.

Treat "2-bit is fine" as **supported in direction, unquantified in magnitude** — the same standard applied
to kimi-k3's single-sample timings. Ferric's role-based experiment should therefore report a real
perplexity delta, which is the number none of the three projects published.

## Corrections to earlier claims in this plan

Both verified against source, and both were overstated in the summary that opened this document:
- The **1.66× is pipeline parallelism over TCP, not RDMA tensor parallelism** — and the baseline column is
  a *Q2* single-process run against a *Q4* distributed run, which favours the baseline.
- **120 t/s on 8×L40S is aggregate over 16 concurrent sessions** (~7.5 t/s per stream), not single-stream.

## Bonus: DeepSeek V4 routing is not standard MoE

1. **The first 3 layers route by deterministic token-id hash**, not learned top-k —
   `ffn_gate_tid2eid` is an `I32[6, 129280]` lookup indexed directly by token id. Side effect worth
   exploiting: for those layers the expert set is knowable **from the token alone**, which is a free,
   exact prefetch signal for streaming.
2. **Router scores are `sqrt(softplus(logit))`, not softmax**, normalized only *after* top-k selection,
   then scaled by 1.5. Selection uses *biased* scores; combining weights use the *unbiased* ones — the
   same selection/weighting split kimi-k3 lists among its five silent-wrong-model invariants.

---

# TESTED: does role-based quantization transfer to ternary?

`cargo run -p ferric-llama --example ternary_by_role --release` — Qwen2.5-0.5B, 6 disjoint chunks ×
384 tokens of real corpus, single-plane group-128 ternary, metric = NLL in **nats/token**.

Qwen's FFN makes this a controlled test: `gate`/`up` are `[4864,896]` and `down` is `[896,4864]` —
**4,358,144 parameters each**. Ternarizing exactly one holds the bit budget fixed and varies only role.

| variant (one role at a time) | ΔNLL vs baseline |
|---|---|
| **gate only** | **+5.849 nats** |
| up only | +4.561 nats |
| down only | +4.237 nats |

*(baseline Q8_0 = 3.461 nats. Context: gate+up +7.720, all-FFN +9.740, attn-only +6.620.)*

## Result 1 — a real role effect, perfectly controlled

`gate` and `up` have identical shape, identical fan-in, and read the **identical** RMSNorm'd hidden state.
The only thing that differs is downstream: `gate`'s output passes through SiLU before multiplying `up`'s.

**`gate` is worse by 1.288 nats, on 6/6 chunks paired.** Nothing else varies, so this is a genuine
role effect — and the plausible mechanism is that gate error is *multiplicative* in the FFN output
(`silu'(g)·δg·up(x)`) while up error is merely additive (`silu(g)·δu`).

## Result 2 — ds4's stated premise does NOT transfer to ternary

`down` is the **least** sensitive of the three, not the most. The input-symmetry argument predicts the
opposite sign.

⚠️ **This corrects the framing written earlier in this document**, which said the mechanism "predicts
`down_proj` was the weak link" in the uniform 16B QAT. Measured, it is not.

Two caveats that keep this honest in both directions:
- **`down` vs `up` is only 4/6 paired** — weak. The robust statement is narrower than "down is best": it
  is *`gate` is uniquely fragile*, with `up` and `down` comparable.
- **`down` has 5.4× the fan-in** (4864 vs 896) at equal parameter count, and quantization error averages
  down as ~√N across summed terms. Input symmetry and fan-in are confounded in this architecture and this
  test cannot separate them. So this does not refute ds4's *choice* — only the *reason given for it*.

The likelier explanation for IQ2_XXS-on-gate/up + Q2_K-on-down is (a) **codebook shape** — IQ2_XXS has no
zero and literally cannot represent a pruned weight — and (b) the **kernel fusion** that makes the split
free, since gate/up share one activation read and are therefore forced to share a type anyway.

## Actionable for Ferric

A uniform ternary QAT is **not** obviously mis-spending bits at `down`. If bits are re-allocated by role,
spend them on **`gate`** — measurably the most fragile role at equal parameter count. That is the opposite
of the recipe a straight port of ds4 would have produced, and it is the kind of thing that only shows up
by testing the mechanism rather than copying the configuration.

*Scope: single-plane ternary at 0.5B, where absolute degradation is severe by design (the relative ordering
is what is under test). Whether the ordering holds at 16B and at gentler quantization is not established.*

---

# TESTED: imatrix calibration — and the layer it must NOT be applied to

`ferric-gguf::imatrix` closes a total capability gap (Ferric had no importance matrix; every production
2-bit engine depends on one). Importance is collected from **real activations** — Ferric's Qwen3 already
exposed the four projection inputs via `set_capture`/`take_capture`, and they line up exactly with what
ds4 measures, so no model change was needed:

| projection | its input |
|---|---|
| `ffn_gate` / `ffn_up` | the FFN-normalised hidden state |
| `ffn_down` | the SwiGLU product, *after* gating |
| `wqkv` / `wo` | attention-normalised hidden / attention output |

`cargo run -p ferric-llama --example imatrix_ternary --release` — Qwen2.5-0.5B, 8 calibration chunks
disjoint from 6 eval chunks, NLL in nats/token:

| role (ternary) | plain | imatrix | Δ | paired |
|---|---|---|---|---|
| gate only | +5.641 | **+9.798** | **+4.156** | imatrix better on **0/6** |
| up only | +3.844 | +3.582 | −0.262 | 4/6 |
| down only | +2.938 | +2.471 | −0.467 | **6/6** |
| all FFN | +10.417 | +10.028 | −0.389 | 4/6 |

The gains on `up`/`down` are in the range ds4 publishes (−1.95% NLL). **The `gate` regression is not.**

## Ruling out the obvious explanation first

A skewed importance vector can satisfy the weighted objective by preserving a few high-energy channels
and zeroing the rest — scoring well while destroying the tensor. Measured, this is **not** what happens:

| capture | max/median | top-1 share | zeros plain→imatrix |
|---|---|---|---|
| `l0.ffn_gu` | 23.2 | 2.3% | 52.9% → 54.2% |
| `l0.ffn_down` | 68.3 | 0.9% | 53.4% → 52.7% |

The quantizer is not degenerating. The result stands.

## What it means

`gate` and `up` share the **same importance vector, the same input, and the same shape**. Calibration
helps one and devastates the other, so the difference is entirely in how their outputs are consumed.

**Importance weighting minimises error in `Wx` weighted by input energy — the right proxy only when the
output is consumed LINEARLY.** `up`'s output is (it multiplies `silu(gate)`). `gate`'s is not: it passes
through SiLU, whose behaviour near zero decides which units gate *off*. Moving error toward low-energy
channels changes *which units cross zero*, which is a discrete change in the gating pattern rather than a
small perturbation of a linear map.

This is a limitation of imatrix calibration as a technique, not of this implementation — and none of the
three source projects states it. It also explains their configuration: ds4 applies imatrix to routed
experts of a **MoE**, where the analogous gate is the router (kept at F16 and never quantized at all).

## Actionable, and measured rather than inferred

    all-FFN ternary:   plain +10.417   uniform-imatrix +10.028   SELECTIVE +9.553 nats

**Apply the imatrix to `up`, `down` and the attention projections; do not apply it to `gate`.** Selective
calibration beats both uniform choices. Combined with the earlier role finding — `gate` is the most
fragile role at equal parameter count — the picture is consistent: `gate` is the layer to protect, and it
is protected by spending *bits* on it, not by calibrating it.

*Scope: single-plane ternary at 0.5B, one model family. The mechanism predicts this holds wherever a
gated activation is used, but that is a prediction, not a measurement.*

---

# BUILT: persistent KV sessions — and the premise, measured

`ferric_tier::kvstore`. A 25k-token agent prompt costs tens of seconds of prefill and is very nearly the
same bytes every request; storing the attention state turns that into a disk read. The payload is opaque,
so the module owns a checkpoint's *identity and safety* rather than its layout, and works for any model.

## The key is the rendered BYTE prefix, not the token sequence

Not the obvious choice, and everything else follows from it. A model samples a token whose text a client
may later send back as *two* differently-tokenised tokens. Key on tokens and that request misses,
re-prefills, and looks like a cache that "just doesn't work sometimes". Key on bytes and it hits, because
bytes are what both sides agree on. A checkpoint answers exactly one question: **are these bytes a prefix
of the incoming prompt?**

## ⭐ The premise, verified rather than assumed

The API returns a **byte range** for the un-cached suffix, never a token slice, because slicing an
already-tokenised prompt at a cache boundary is claimed to yield a sequence the model never cached. That
claim is load-bearing, so it was measured against the real Qwen tokenizer
(`cargo run -p ferric-llama --example bpe_boundary`):

```
split points tested: 163
concatenation matches the whole-text tokenisation:  27  (16.6%)
concatenation DIFFERS:                             136  (83.4%)
```

**83.4% — the common case, not an edge case.** A resume path that slices tokens is wrong most of the time
it is used, and wrong *silently*: no error, just degraded output. Returning a byte range makes the bug
inexpressible rather than merely discouraged.

(This also corrected my own first write-up, which asserted the failure rate was low and that "most split
points happen to be clean". The measurement says the reverse.)

## Everything else guards a silent failure

Each validation rung exists because the corresponding mistake produces plausible output rather than an
error, and every one is a test:

- **model id** — a checkpoint is meaningless to a different model, and nothing in the bytes reveals it.
- **payload ABI** — bytes from an older layout parse cleanly and are simply the wrong numbers.
- **re-hash the stored text against its own filename** — catches a copied or renamed checkpoint being
  served for bytes it does not represent.
- **literal prefix compare** — turns a hash collision from a wrong answer into an error.
- **only files named by their content hash are parsed** — a stray temp file is never a checkpoint.
- **temp-file + rename** — a crash mid-write leaves the previous checkpoint intact, not a half-file that
  parses.

## Store placement is what makes checkpoints shareable

`store_len` trims the tail (the last tokens are the most likely to retokenise once more text is appended)
and aligns down. Alignment is the part that matters: independent sessions then land on the **same**
offsets and can reuse each other's checkpoints, instead of each writing a unique frontier nothing else
can hit.

Eviction scores **token density with a decaying hit bonus** — `(hits·2^(−age/half_life) + 1) · tokens /
bytes`, doubled for anchors — because the cache's job is tokens-not-prefilled per byte held, and a
checkpoint popular yesterday should not hold space forever.

Dependency-free (SHA-1 inline, as the source engines also do), and `#[cfg]`-ed off for wasm.
