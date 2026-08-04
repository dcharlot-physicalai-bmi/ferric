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

## Status

The tier, the reader, the staging seam and the plan are verified natively — that is the code that could be
subtly wrong. **The JS glue in `index.html` and `FerricStream`'s wasm-bindgen surface have not been
executed in a browser here**; they compile, the bundle builds (830 KB), the exports are present, and the
server serves correct 206 responses. Treat browser execution itself as unverified until someone runs it.
