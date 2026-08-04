# Ferric in the browser — streaming inference

Run a model larger than the tab can hold: weights arrive by HTTP **Range** request and execute on WebGPU,
through the same tier and the same GGUF reader the native path uses.

```bash
./web/build.sh                 # wasm-pack -> web/pkg
ln -s /path/to/model.gguf web/model.gguf
./web/serve.sh                 # http://localhost:8770
```

## Two requirements, and one trap

**The host must honour Range** (`206 Partial Content`). A host that ignores it answers `200` with the
*whole body*, so every "range request" downloads the entire checkpoint — streaming appears to work while
doing exactly the opposite. Ferric refuses a wrong-length response by name rather than staging it.

⚠️ **`python3 -m http.server` does NOT implement Range.** Verified here: a 64-byte request came back as
675,710,816 bytes. `serve.py` in this directory does implement it — that is why it exists.

**WebGPU** is required: Chrome/Edge 113+, Safari 18+, or Firefox with `dom.webgpu.enabled`.

## What a cold start actually costs

Measured against `serve.py` with a 675.7 MB Qwen2.5-0.5B checkpoint:

| request | bytes | why |
|---|---|---|
| `HEAD` | — | file size; the plan needs it |
| header probe, doubling | 16.78 MB | the GGUF header's length is not recorded in the format, so a reader grows a prefix until it parses. The tokenizer vocabulary is most of it. |
| per streamed layer | 15.86 MB | one request, because a layer's tensors are one contiguous run |

**37.9 MB of 675.7 MB** before the first token. `FerricStream.plan()` reports all of it as JSON, so a page
can show the cost before committing to a download.

## The app supplies the bytes

`FerricStream.open(fetcher, totalBytes, budgetBytes)` takes a JS
`(offset, length) => Promise<Uint8Array>`. That is deliberately more general than baking in `fetch`, and
it is why this crate needs no `web-sys`:

```js
const get = (off, len) =>
  fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` } })
    .then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
```

Swap that for the Cache API, a service worker, OPFS, IndexedDB, or an authenticated endpoint and nothing
inside Ferric changes.

## Why the browser needs staging rather than a cache

`Backing::read_at` is synchronous; `fetch` is not; a wasm main thread cannot block on a promise. Bytes are
therefore **staged before** the forward pass reads them, which the fixed layer walk (0, 1, … N−1, repeat)
makes exact rather than speculative. A read of un-staged bytes is `TierError::NotStaged` naming the range
— never zeros, because a model that runs and lies is worse than one that stops.

`cargo run -p ferric-llama --example stream_embodiments --release` asserts this path produces
**byte-identical logits** to a file handle and to an in-memory buffer (`max|Δ| = 0.000e0`, same argmax).

## Verified in a real browser

Chrome + WebGPU, driven over CDP against `serve.py`, Qwen2.5-0.5B, 64 MB layer budget:

```
opening with a 64 MB layer budget…
  ready in 0.4s
  layers 24, pinned 3 (12.5% reuse)
  layer weights 380.5 MB of 675.7 MB
  fetched so far 358.9 MB in 9 range requests

The capital of France is Paris. It is
  290 ms/token
```

It generates. The output is correct text, from Range-fetched weights, on WebGPU.

## ⚠️ What it does NOT yet do: reduce peak memory

`peak resident 686.5 MB` against a 64 MB budget. That is not a bug in the tier — it is a structural limit
worth naming precisely:

**`forward_cached` walks all layers synchronously, with no point at which it can await a fetch.** So every
streamed layer must be staged *before* a step begins, and peak residency within a step is the whole layer
set. Releasing between tokens bounds the *steady* state (`now 353.6 MB`) but not the peak.

So today the browser path buys **incremental loading and byte-identical output**, not a smaller footprint.
The budget bounds what the tier *pins*, not what the tab holds.

Fixing it needs a forward that **yields per layer** so a fetch can be awaited mid-step — an architectural
change to the model, not to this directory. The native path does not have this problem because a thread
can block on a read; the browser cannot.

This was found by running it, not by reading it. An earlier version also called the *resident* loader
after staging only the pinned prefix, and the browser caught that too — `NotStaged` naming the missing
range instead of returning zeros, which is exactly what that error exists for.
