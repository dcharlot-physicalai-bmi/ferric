# Ferric vs the field, measured — September 2026

Every number in the first three sections was measured **on this machine, this week**, not quoted from
a README. Where a figure is external it is cited. `docs/SOTA.md` is the standing scorecard; this is a
dated snapshot with a ranked plan, written because "best cross-fabric runtime, latest models, best
performance" is three different scoreboards and Ferric is in a very different place on each.

## 1. Correctness — Ferric leads, and it is now measured rather than asserted

The comparison that matters is **same fabric**. Cross-fabric comparisons are dominated by the fabric.

| comparison | relative Δ (`sum_abs`, 151936 logits) | argmax |
|---|---|---|
| Ferric-Metal vs **llama.cpp-Metal** | **5.35e-05** | both 12095 |
| llama.cpp-Metal vs **llama.cpp-CPU** | **1.79e-02** | both 12095 |

⭐ **The fabric term is 335x the implementation term.** One model (`qwen3-0.6b-q5km`), one prompt,
identical token ids. This retires a question that stood open for a long stretch: the ~1e-3 residual
between Ferric and llama.cpp on hyv4 is **6.25x smaller than llama.cpp's own CPU-vs-Metal
disagreement**. Ferric is closer to llama.cpp-CPU than llama.cpp-Metal is.

Alongside that: Metal and browser WebGPU are **bit-identical**, NVIDIA Vulkan agrees to ~1e-5 with
identical argmax (`docs/SOTA.md`), and greedy decode is token-for-token identical to llama.cpp on
Qwen3-0.6B-Q4_K_M / Q5_K_M and Qwen2.5-0.5B-Q6_K.

**No other project ships proven same-code cross-vendor parity.** This is the moat. It is also the
thing least visible on a benchmark chart.

## 2. Performance — Ferric is ~10x behind, and the reason is NOT what the docs assumed

Same model, same machine, both on Metal:

| | tok/s | ms/token |
|---|---|---|
| llama.cpp (`llama-bench` tg24, 3 reps) | **369.25 ± 7.44** | 2.7 |
| Ferric (warm, 24 tokens) | **38.5** | 26 |

**9.6x.** Four explanations are ruled out by measurement, not argument:

| hypothesis | verdict | evidence |
|---|---|---|
| host-side command preparation | **false** | 1.1–2.5% of wall (MNN reports 49.5–75.2% on GPU; not us) |
| memory-bandwidth-bound | **false** | 424 MiB and 974 MiB models both decode at **26 ms/token** — 2.3x the bytes, same time |
| dispatch-count-bound | **false** | 478 vs 178 dispatches/token — 2.7x, same time |
| submit-count-bound | **false** | 57 vs 33 submits/token — 1.7x, same time |

⚠ **A fifth hypothesis was raised and then RETRACTED — by the instrument that raised it.** The
category profiler reported `attn` at 52% (qwen3) and 62% (llama3.2) of a decode step, which looked
like the answer and was written into the first version of this document. It is not sound. `prof()`
calls `device_sync()` at **every category boundary** — two per layer — and the profiled run costs
+25% (qwen3) and **+52%** (llama3.2) over the unprofiled one. The model with *more* sync overhead is
exactly the one reporting the *higher* attention share:

| | profiled | unprofiled | syncs/token | added ms per sync |
|---|---|---|---|---|
| qwen3-0.6b | 32.3 ms | 25.9 ms | 56 | 0.114 |
| llama3.2-1b | 39.8 ms | 26.2 ms | 32 | 0.425 |

Measured directly instead, without a readback in the timed loop
(`crates/ferric-tensor/examples/fattn_bench.rs`), the fused decode-attention kernel costs:

| S (cache length) | ms/call | MFLOP | GFLOP/s |
|---|---|---|---|
| 17 | 0.051 | 0.14 | 2.7 |
| 128 | 0.042 | 1.05 | 24.7 |
| 512 | 0.057 | 4.19 | 73.9 |
| 2048 | 0.090 | 16.78 | 186.0 |

⭐ **Flat to S=128 — a ~0.04 ms launch floor**, then it scales. At 28 layers that is **1.43 ms/token,
5.5% of the step** (llama3.2: 0.64 ms, 2.4%). The category profile said 52% and 62%. **The attention
core is not the bottleneck, and the ranked plan below was rewritten because of it.**

### Where the gap is: the k-quant unpack path, measured per format

⚠ **First, a correction to this document's own previous answer.** It derived a "71 GB/s marginal
bandwidth" from a controlled pair — qwen3-0.6b at q5km (424 MB, 25.70 ms) against q4km (378 MB,
25.05 ms) — holding the architecture and dispatch count fixed so that *only bytes* varied. The graph
was indeed fixed; **the kernel was not.** A q5km checkpoint runs the Q5_K matmul and a q4km one runs
Q4_K, and those two kernels do not stream at the same rate. That pair varied two things, so its
marginal figure is withdrawn.

The direct measurement (`crates/ferric-tensor/examples/matmul_q_bench.rs`, an FFN-shaped decode GEMV
`[1,2048]·[8192,2048]ᵀ`, three runs):

| format | bpw | GB/s | % of read ceiling |
|---|---|---|---|
| **Q8_0** | 8.50 | **295–352** | **69–82%** |
| IQ3_XXS | 3.06 | 110 | 25% |
| Q5_K | 5.50 | 91–93 | 22% |
| IQ2_XXS | 2.06 | 80 | 19% |
| Q6_K | 6.56 | 71 | 16% |
| **Q4_K** | 4.50 | **43–56** | **10–13%** |
| STQ1_0 | 1.31 | 43 | 10% |
| Q2_0 | 2.13 | 39 | 9% |

⭐⭐ **Q8_0 reaches 69–82% of the read ceiling — llama.cpp's own streaming rate is ~326 GB/s, and
Q8_0 is in that band.** So there is **no structural WGSL penalty and no fabric excuse**: the same
runtime, the same device, the same dispatch machinery moves weights at native speed when the unpack
is trivial. Every k-quant and i-quant format sits at 9–25%. **The gap is the unpack path, and
nothing else.**

⭐ **Q4_K is the worst of the mainstream formats, and it is slower in ABSOLUTE terms than Q5_K and
Q6_K while moving fewer bytes** — 43–56 GB/s against Q5_K's 91–93, stable across runs. A format that
reads 18% fewer bytes and takes ~2x as long is not bandwidth-bound; its kernel is. **Q4_K_M is the
most common quantisation on Hugging Face**, so this single kernel is the highest-leverage target in
the runtime.

⚠ Measured at one shape with synthetic blocks. It bounds the kernels, not any particular model: a
real checkpoint mixes formats per tensor, which is precisely why the model-pair inference above was
unsound.

### ⭐⭐ THE CAUSE: the split-K kernel leaves 88–94% of its lanes idle

The k-quant kernels stride whole blocks across a 64-lane workgroup:

    @compute @workgroup_size(64)
    for (var blk: u32 = t; blk < nblk; blk = blk + 64u) { ... }

A k-quant block holds **256 values**, so a matmul with `in_dim = 2048` has `nblk = 8` — **8 lanes
work and 56 idle.** Q8_0's blocks hold **32** values, so the same width gives it `nblk = 64`: full
occupancy. That is the whole difference, and it is not about unpack arithmetic at all — Q5_K's inner
loop does *strictly more* work than Q4_K's (an extra load plus four bit extractions) and runs faster.

**Controlled test.** `in=2048,out=8192` against `in=8192,out=2048` — **identical weight bytes**
(9.0 MiB for Q4_K), identical MAC count, only blocks-per-row differs:

| in_dim | blocks/row | lanes busy | Q4_K | Q5_K | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|
| 2048 | 8 | 12% | **48.2** | 88.3 | 71.1 | 333.3 |
| 8192 | 32 | 50% | **186.1** | 235.6 | 144.9 | 419.1 |
| 16384 | 64 | 100% | 243.1 | 211.1 | 112.1 | 708.4 |

⭐ **Q4_K goes 48.2 → 186.1 GB/s, 3.9x, at the same bytes.** ⚠ Read the 16384 row with care: at that
width the weight is cache-resident (Q8_0's 708 GB/s is *above* the DRAM read ceiling), so it shows
the trend, not a streaming rate. The 2048→8192 pair is the honest one.

**And real decode shapes are worse than the benchmark's:**

| model | typical matmul `in` | blocks | lanes busy |
|---|---|---|---|
| qwen3-0.6b | 1024 | 4 | **6%** |
| llama3.2-1b | 2048 | 8 | **12%** |
| qwen3-8b | 4096 | 16 | 25% |

⭐⭐ **This resolves every loose end in this section.** Why two models of very different size both
decode at ~26 ms/token: the smaller one (d=1024, 6% lanes) is proportionally more crippled, cancelling
its byte advantage — which is also why the earlier "controlled pair" produced a meaningless marginal
bandwidth. Why Q8_0 alone reaches llama.cpp's rate: 32-value blocks fill the lanes. Why cutting 29%
of dispatches changed nothing: the waste is *inside* the kernel, not at its launch. **It was never
bandwidth and never dispatch overhead — it is occupancy.**

**The fix** is a lane-assignment change, not new arithmetic: when `nblk < 64`, map several output
elements to one workgroup (`out_local = t / nblk`, `blk = t % nblk`) and reduce per group, or split
each block's 8 sub-blocks across lanes. The inner loop is untouched.

### ⛔ The obvious fix was tried and DID NOT WORK — and why is the useful part

Ferric already ships the alternative kernel shape: `flat`, one thread per **output**, walking all of
K (`FERRIC_Q2_0_KERNEL=flat`). Per shape it wins exactly where the occupancy argument predicts, and
loses exactly where it predicts (GB/s, split-K vs flat, real decode shapes):

| shape | in vs out | split-K | flat | winner |
|---|---|---|---|---|
| `ffn_gate_up` 1024→3072 (Q5_K) | out > in | 34.9 | **55.2** | flat |
| `ffn_down` 3072→1024 (Q5_K) | in > out | **66.3** | 35.9 | split-K |
| `qkv` 1024→4096 (Q6_K) | out > in | 33.7 | **90.4** | flat |
| `ffn_gate_up` 2048→8192 (Q4_K) | out > in | 88.1 | **200.5** | flat |
| `ffn_down` 8192→2048 (Q4_K) | in > out | **231.6** | 60.2 | split-K |

The crossover is `in_dim ≈ n_out`, which is what the two kernels' widths predict: flat's parallelism
is `n_out`, split-K's is `min(in/256, 64)` lanes per output. The shipped rule ignores `in_dim`
entirely and takes split-K for every decode — the wrong half of that table on `gate_up` and `qkv`.

**So the rule was implemented. End to end it bought nothing**: 26.1 / 25.5 / 26.2 ms/token against
the old rule's 25.9 / 25.2 / 26.1 on the same three models.

⭐ **The reason is the finding.** `matmul_q` does not route the model's largest matmul. The FFN goes
through `try_matmul_swiglu` — a fused matmul+SwiGLU kernel with its own dispatch — so `matmul_q`
carries only the attention projections and the LM head. **The benchmark measures a path the hot loop
does not take.** Forcing `flat` globally is likewise a regression (4–9% slower end to end), because
it starves on `ffn_down`.

The shape-aware rule is therefore kept, measured, and **off by default** behind `FERRIC_SHAPE_KERNEL=1`:
a change to core dispatch with no demonstrated end-to-end win is risk without payment. The live target
is the `MATMUL_Q*_SWIGLU_WGSL` family, which no benchmark in this repo has yet touched.

⚠ **Fourth instrument in this document to answer a question nobody asked** — after the syncing
profiler, the syncing microbenchmark, and the cold-cache run. The pattern is consistent enough to
state as a rule: **a kernel benchmark is only evidence if the model actually dispatches that kernel,
at that shape.** Check the call graph before trusting the curve.

### ⭐⭐ END-TO-END CONFIRMATION: 52% more bytes, 2x faster

Everything above is microbenchmarks. Two *whole models*, same family, same machine, same harness:

| model | quant | weight bytes | ms/token | effective |
|---|---|---|---|---|
| qwen2.5-0.5b-instruct | **Q8_0** | 644 MiB | **13.2** | 48.8 GB/s |
| qwen3-0.6b | **Q5_K** | 424 MiB | **26.1** | 16.2 GB/s |

⭐ **The bigger model is twice as fast.** 52% more weight bytes to move, half the time to move them.
No microbenchmark, no profiler, no synthetic shape — two real checkpoints through the ordinary decode
path. Ferric's decode is **not bandwidth-bound**, and the k-quant kernels are what separates these
two runs.

⚠ Not perfectly controlled: different architectures (24 vs 28 layers; qwen3 adds QK-norm, which is
two extra ops per layer). So read it as a large effect in a clear direction rather than a coefficient.
It agrees with the per-format table above, where Q8_0 measures 3–4x Q5_K's rate at decode width.

### ⛔→✅ A guard that fired into an empty room, and the production bug behind it

`examples/dispatch_budget.rs` carries a regression guard on dispatch count. **It was failing** —
13.1 dispatches/layer/token against its `<= 12.5` bound — and nobody knew, because an example is not
a test: `cargo test --workspace` does not run examples and CI runs only `ebm_cert_verify`.

Finding the cause needed an instrument that did not exist. A dispatch *total* says a budget moved;
only a **per-kernel census** says which kernel moved it. `FERRIC_CENSUS=1` now reports dispatches
keyed by the label already passed to `run()`:

    matmul_q8_0_splitk  101.00     kv_write2  25.00     <- the K/V fusion IS active
    rmsnorm              51.04     rope       25.00
    binary               50.00     fattn      24.00

It disproved the obvious hypothesis on sight. 314 is exactly the pre-fusion count from the commit
that fused K/V appends, so that fusion looked like the regression — but `kv_write2` fires 25/token
and is fine. The anomaly is `rmsnorm` at **51/token where 24 layers need ~25**.

**The cause**, `qwen3.rs:982`:

```rust
dump("attn_norm", il, &x.rmsnorm(&l.attn_norm, self.cfg.eps));
```

Rust evaluates arguments before the callee runs. So a full RMSNorm **GPU dispatch executed on every
layer of every token in production** and was discarded the moment `dump` read `FERRIC_DUMP` and
returned. ⭐ **A diagnostic that costs a dispatch when disabled is not disabled.**

Fixed structurally rather than at the call site — `dump_with(tag, il, || …)` takes a closure, so the
work cannot be evaluated unless the dump is on (`qwen3.rs` and `deepseek2.rs`, which has its own
`dump` and two sites of the same shape):

| | before | after |
|---|---|---|
| `rmsnorm` dispatches/token | 51.04 | **26.04** |
| total dispatches/token | 314 | **290** |
| submits/token | 49 | **25** |
| guard | 13.1 FAIL | **12.1 PASS** |

290 is exactly the number the K/V-fusion commit recorded. Submits halved too — the stray norm sat
outside the batch region and forced its own queue submission.

⚠ **No speed number is claimed.** Attempts read 31.2 / 34.5 / 22.3 ms/token where this model measures
13.2 on a quiet machine; load average was 23.7 from this session's own builds. The fix stands on the
counters, which are host-side and exact, and on the guard.

✅ **And the guard is now wired**: `scripts/perf_guards.sh` runs the guards that live in examples,
building before running (a stale example binary made the library fix look like a no-op once already),
skipping loudly when a checkpoint is absent, and failing the battery on a non-zero exit.
Mutation-verified: tightening the bound to 11.0 fails the battery with exit 1. CI cannot hold these
checkpoints, so this is a **local** battery — under the standing rule that the local battery is a
superset of the CI jobs, not a subset.

⛔ **This retires "we are bandwidth-bound", which this repo's docs said for months.** The 47 GB/s
figure was bytes ÷ wall clock, and `docs/RUNTIME-PARITY-2026.md` already flagged it as false; the
two-model experiment above settles it.

## 3. Model coverage — the honest count

`crates/ferric-llama/src/arch.rs` is a registry with a status per architecture, and the taxonomy is
the point: `Verified` means diffed against a reference implementation **on real weights**.

| status | count | meaning |
|---|---|---|
| **Verified** | **15** | compared against a reference on real weights and matched |
| Loads | 15 | coherent output, never diffed — a wrong RoPE looks exactly like this |
| Parts | 5 | components exist, no loader wires them |
| Untried | 4 | synthetic checkpoint only |

Against llama.cpp's **50+ text architectures** and ~45k GGUF checkpoints on Hugging Face. So on raw
breadth Ferric is behind roughly 2:1 — but on *evidence per architecture* nothing else publishes a
status taxonomy at all, and 15 reference-verified is a real number.

⭐ **And Ferric is ahead where it counts most**: `hyv4` (Tencent Hy4, 770B/49B) is **supported by no
upstream runtime** — llama.cpp needs two out-of-tree patches. As of today Ferric runs all 78 blocks
of the real 213.66 GiB checkpoint, streamed, and emits the same greedy token as Tencent's own
implementation. That row moved `Untried → Verified` this week.

## 4. The field, September 2026

**Server tier** (not Ferric's game, but sets the performance narrative): SGLang ~16,200 tok/s vs
vLLM ~12,500 on prefix-heavy Llama-3.1-8B (H100), TensorRT-LLM highest absolute with a compile step.

**Local tier** (Ferric's actual competitor): llama.cpp. v0.4.0 shipped **2026-09-04** with
Qwen3.8-Flash-Next (`qwen4exp`), Nemotron-3-Puzzle-75B-A9B, and **lazy tensor reading — weights
pulled from disk on demand**, which is the same idea as Ferric's `load_streaming`. ggml 0.23.0 added
sparse flash attention for DeepSeek-V4, GLM and qwen4exp.

**Browser tier** (Ferric's differentiator): WebLLM runs Llama-3.1-8B-4bit at ~41 tok/s on an M3 Max
(~80% of native MLC); transformers.js v4 was rewritten in C++ with a WebGPU backend for 3–10x over
v3. Browser is generally 5–10x slower than native. Ferric's measured browser figures (LFM2.5-1.2B:
6.8 tok/s cold, ~20 warm, 37.6 with q4_0 KV) are **behind WebLLM on a model 7x smaller** — the
cross-fabric *correctness* lead is not yet a cross-fabric *performance* lead.

**Rust tier**: candle (inference, cuBLAS-direct), mistral.rs (quantized, OpenAI-compatible server),
burn+CubeCL (the only true same-code cross-platform peer, no bit-exact parity guarantee), ratchet
(WebGPU, inference-only). None combines native + browser + training + verified parity.

**Research worth ingesting**: dispatch overhead is characterised across 4 vendors / 3 backends / 3
browsers (arXiv 2604.02344) — **wgpu-native on Metal is the worst measured at 71.1 µs/dispatch vs
Safari's 31.7**, and kernel fusion cutting 312 dispatches bought 53% end-to-end. ⚠ That paper's
batch=1 conclusion does **not** transfer here: Ferric measured a 29% dispatch cut at 0.00 ms. Read it
for the fusion result, not the diagnosis. Also live: speculative decoding × sparse attention (NSA,
EAGLE-series), NPU table-lookup low-bit inference (T-MAN), CPU-GPU cooperative execution with AMX.

## 5. Ranked plan — what actually closes the gap

Ordered by measured impact per unit of work, not by novelty.

1. **`matmul_q4_k` — 43–56 GB/s where Q8_0 does 295–352 on the same device.** Q4_K_M is the most
   common quantisation in the ecosystem and its kernel is the slowest mainstream one Ferric has,
   *slower in absolute terms than Q5_K while reading fewer bytes*. Q8_0 proves the ceiling is
   reachable; this is unpack work, not fabric. Then Q6_K (71) and the i-quants.
2. **Cut the per-token remainder** (12–20 ms). Its per-dispatch figure (41–69 µs) matches published
   wgpu-native-on-Metal cost, which is the **worst of four backends measured** — Safari's Metal is
   2.2x cheaper and Dawn's Vulkan 3x. Some of this is wgpu's, not Ferric's, and that is worth knowing
   before optimising around it.
3. **Build a non-syncing profiler** (timestamp queries) to separate the remainder's per-dispatch and
   per-token halves. Every attribution attempt here either synced per boundary — distorting the split
   badly enough to produce a retracted 52% claim — or aggregated to a whole step.
4. **A real flash-attention/GQA WGSL kernel** (online softmax, tiled) for PREFILL and long context,
   where the measured curve above does start to scale. Not for short-context decode: the fused
   kernel already costs 5.5% there and fusing it further cannot buy back 9.6x.
5. **Per-shape/per-device kernel autotuning** (roadmap #6). Measured elsewhere at +41%; matters more
   for Ferric than anyone because WebGPU spans the widest hardware range.
6. **Close the browser performance gap** to WebLLM. The correctness story is already better; the
   throughput story is 7x worse per parameter, and browser-first is the stated thesis.
7. **Latest-model cadence.** `qwen4exp` (Qwen3.8-Flash-Next) and Nemotron-3-Puzzle-75B-A9B landed
   upstream on 2026-09-04. The standing rule is support within 30 days.
8. **Promote `Loads` → `Verified`.** Fifteen architectures generate coherent text and have never been
   diffed against a reference. `scripts/hyv4_vs_reference.sh` is now a reusable shape: same file,
   both implementations, gate `sum_abs` **and** the greedy pick.

⛔ **What NOT to do**: chase server-tier throughput. vLLM/SGLang/TensorRT-LLM own batched serving on
NVIDIA and Ferric will not take that, nor should it. The defensible position is *one codebase, every
fabric, verified identical, latest models, browser included* — and of those five, only performance
is currently red.
