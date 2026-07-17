# ferric-flow — ingesting the live-data / RAG / MCP layer into Ferric

**Status:** phases 1–3 landed and green (2026-07-17). Crate: `crates/ferric-flow`. 6 tests + a
real-corpus example, wasm-clean.

## Why this exists

The Institute needs a live-data / RAG / MCP layer. **Pathway** is the obvious reference — but the
reason to study Pathway is not to adopt it, it is to notice *what is actually theirs*.

- **What Pathway does:** a Rust engine (a modified subset of **Differential Dataflow**, McSherry et al.)
  driven by a Python layer. It does *incremental computation* — refresh a result by computing only the
  change — over unified batch+streaming input, and layers a live vector index (RAG), an MCP server,
  and 350+ connectors on top.
- **How they monetize a non-proprietary method:** the hard part — Differential/Timely Dataflow — is
  public and MIT-licensed. Pathway's moat is **open-core + BSL**: a source-available Business Source
  License (converts to Apache-2.0 after 4 years, the CockroachDB/Redpanda pattern) that forbids
  reselling it as a hosted service, plus paid Scale/Enterprise tiers. They fence a public technique
  with a license and packaging, not with proprietary IP.

That is the whole opening. The method is open; the moat is licensing and DX. So we **ingest the method
and make our own**, native to Ferric — the same three-tier play as the Ferromotion porting program,
one layer up the stack. Our differentiation is not the algorithm (nobody's is); it is that ours is
**pure Rust, `std`-only, and compiles to `wasm32`** — so it runs on the *edge* (Cloudflare Workers)
and cross-fabric, unified with the rest of Ferric, where Pathway needs an always-on Python/Docker box.

## The method, restated

A collection is a multiset that evolves over time. A **delta** `(record, diff)` asserts (`diff > 0`)
or retracts (`diff < 0`); a correction to late or wrong data is a retraction of the old value plus an
assertion of the new. Operators consume delta batches and emit delta batches, holding only the state
they need:

- `map` / `filter` — stateless, per record.
- `Join` — keeps an arrangement of each input; a delta joins against the *other* side's current
  arrangement, so work is `|delta| × matches`, never `|left| × |right|`. The delta-join ordering
  (delta-left ⋈ old-right, then new-left ⋈ delta-right) counts the delta×delta term exactly once.
- `Reduce` — keeps each group's multiset and last emitted aggregate; only groups whose input changed
  are recomputed, and each emits a retraction of its old aggregate + an assertion of the new.

## Phase 1 — the make-or-break, proven (`crates/ferric-flow`, 4 tests green)

The property that makes the whole layer worth building — mirroring Ferric's own "KV-cache: exact vs
full recompute" milestone:

| test | result |
|---|---|
| `reduce_incremental_equals_full_recompute_under_random_stream` | 400-step random insert/retract/correct stream; accumulated incremental output is **bit-identical** to a from-scratch group-by at every step |
| `join_reduce_pipeline_batch_equals_incremental_and_saves_work` | join→reduce fed incrementally == fed as one batch, and the reducer did **5.0× less work** (300 group recomputes vs a 1505 full-recompute baseline) |
| `corrections_and_late_data_converge` | a wrong value, later corrected out of order, converges to the corrected truth |
| `insert_then_retract_nets_to_nothing` | full retraction of a join nets to empty |

Builds clean on `wasm32-unknown-unknown` (std-only, zero dependencies) — the edge claim, not a promise.

## Phase 2 — the live vector index, proven (`crates/ferric-flow/src/index.rs`, 2 tests green)

The retrieval half of the RAG layer: a pure-Rust **HNSW** (`Hnsw`) wrapped as a `ferric-flow` operator
(`LiveIndex`). Embeddings flow in as `(id, vector)` deltas — assertions insert, retractions to zero
tombstone — so a corrected or deleted document drops out of retrieval immediately, no rebuild.

| test | result |
|---|---|
| `incremental_index_matches_brute_force_and_is_sublinear` | 800 inserts + 15% retracted, streamed one at a time; **recall@10 = 1.000** vs a brute-force scan over the live set, at **3.9× fewer** distance evals (176/query vs 691), and **no retracted doc is ever returned** |
| `retract_then_reassert_returns_to_results` | a doc retracted drops out of results and, re-asserted, returns |

Two findings worth keeping: (i) uniform-random high-dim vectors are the adversarial worst case for any
ANN (concentration of measure) — the test uses *clustered* embeddings, the real RAG regime; (ii) the
recall ceiling was connectivity, not search depth — the HNSW **neighbour-selection heuristic** (keep a
candidate only if it is closer to the base than to the neighbours already kept) took recall 0.85 → 1.00.

## Phase 3 — the real corpus, proven (`crates/ferric-flow/examples/corpus_rag.rs`)

The Institute Assistant retrieves with bge-m3 embeddings over **Cloudflare Vectorize** — a managed,
external vector-DB service that rebuilds on ingest. Vectorize is exactly the kind of rented,
non-proprietary capability this program ingests (an ANN index is a public algorithm). This example
drives `LiveIndex` over **37 real Institute documents** (papers + Charlot Lab corpus) embedded with a
real sentence model (all-MiniLM-L6-v2, 384-d; fixture built by `v2/tools/nanovla/embed_corpus.mjs`),
streamed in one at a time:

| check | result |
|---|---|
| recall@5 vs an exhaustive cosine scan | **1.000** |
| same #1 result as the exhaustive scan, on 8 real questions | **8/8** — the index is faithful, so it can replace Vectorize with no loss |
| retrieval quality (eyeball) | correct top hit for magnetic-skin, printable-actuator, energy-native, microfluidic, contact-layer, multiply-free/ternary, space-logistics (one embedding-model miss: "playable simulator" → explainer instead of the world-model paper — the index is faithful; that is the embedding's call) |
| retract a top hit → gone from results, **no reindex** → re-assert restores it | ✓ / ✓ — the in-place edit Vectorize can't do |

Honest scale note: at 37 documents no ANN beats brute force (41 vs 37 evals); sub-linearity is a
scale property (the 691-doc synthetic test showed 3.9× fewer). The value *here* is faithfulness +
incrementality; the value *at scale* adds the speed.

## Roadmap (the rest of the layer)

1. ~~**Live vector index** for RAG~~ — **done (phases 2–3).** Proven faithful to exhaustive retrieval on
   the real corpus and live under edits, and **serializable** (`Hnsw::to_bytes` / `LiveIndex::save`) —
   a 400-doc graph is ~145 KB and rehydrates to bit-identical queries. That closes the only real
   objection to an in-Worker index: **Workers are stateless**, so you build the graph once (offline or
   on ingest), store the bytes, and rehydrate + query per isolate rather than rebuilding on cold start.

   **Cutover to retire Vectorize — the one architectural decision:**
   - **(A) Serialized-graph in KV/R2 (recommended):** on `/ingest`, embed the corpus and build the
     graph, `save()` the bytes to R2/KV. The chat Worker loads the bytes (cached per isolate) and calls
     `LiveIndex::load(...).query(...)` in wasm. Cheapest, fully edge-native, no managed vector DB. Cost
     is the per-isolate rehydrate (~145 KB parse) — trivial at our corpus size.
   - **(B) Durable Object holds the live index in memory:** one DO owns a `LiveIndex`, applies deltas as
     the corpus changes, answers queries by RPC. Truly incremental (no rebuild ever), at the cost of a
     DO on the query path. Right when the corpus is large or updates constantly.
   Both keep the bge-m3 embeddings and the bge-reranker; only `env.VECTORIZE.query` is replaced. This
   step touches the live Assistant (a deploy), so it is gated on an explicit go.

   **Local proof of A — done (2026-07-17).** `crates/ferric-flow-wasm` (wasm-bindgen) + `worker-poc/`
   prove the wasm path on `wrangler dev`: the serialized 37-doc corpus graph is loaded per isolate
   (async `init(wasmModule)`) and queried in wasm — `/health` → `{liveDocs:37}`, all 8 real queries
   correct, 0 errors.

   **Production cutover — SHIPPED (2026-07-17), live.** The Institute Assistant (`v2/assistant`) now
   retrieves with ferric-flow, Vectorize kept only as a fallback:
   - `ferric-flow-wasm` gained a `RagBuilder` (insert + `save()`); vendored into `v2/assistant/vendor/`.
   - `/ingest` builds a ferric-flow graph from the SAME bge-m3 embeddings (no extra embed calls) and
     stores the serialized graph + metadata in a KV namespace (`FERRIC_KV`). 363 docs indexed.
   - `/chat` retrieves top-20 from the wasm HNSW (rehydrated from KV per isolate), applies the same
     kind-filter + score threshold, then the unchanged bge-reranker + Claude generation. Any miss/error
     falls back to `env.VECTORIZE.query`.
   - Verified on prod: a `/search-ferric` A/B returns the **same top documents as Vectorize** on real
     queries (identical top-3 on "digital twin", "parametric CAD", etc.); full `/chat` streams correct
     grounded answers (`x-assistant-generator: claude`). Vectorize is now a fallback, not the primary —
     a managed dependency retired in the answer path.
2. **MCP surface** — expose flow outputs as tools (extend, don't replace, the existing `bmi-papers`
   MCP server).
3. **Connectors** — start with what the Institute actually has (the corpus, courses, Atlas/`makers.json`),
   add external feeds only when a real Atlas data-stream appears.
4. **Graph compaction + iteration** — tombstones accumulate graph cruft; periodic compaction is a later
   concern (results stay correct meanwhile). Phase 1 uses a single totally-ordered logical time;
   iterative dataflow and partially-ordered times are the next depth if a use case needs them.

The license posture is the mirror image of Pathway's: **Apache-2.0**, like the rest of Ferric. The
edge/cross-fabric integration is the moat, and it is one we own rather than one we rent from a license.
