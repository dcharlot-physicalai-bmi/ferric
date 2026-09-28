# ferric-ffi — the Ferric C ABI (`libferric`)

The universal on-ramp. Any language with C FFI drives the pure-Rust cross-fabric runtime:

```c
FerricHandle* h = ferric_load("model.gguf");
char* r = ferric_chat(h, "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 32}");
// r = the OpenAI chat.completion object as JSON, with usage and the joules it cost ("energy")
char* text = ferric_generate(h, "The capital of France is", 8);           // free text (greedy)
char* json = ferric_generate_json(h, "A person.", "{...json-schema...}", 40); // guaranteed-conformant
ferric_free_string(r); ferric_free_string(text); ferric_free_string(json); ferric_free(h);
```

The handle is **ferric-serve's own engine** (`ferric_serve::LocalModel`): every architecture the registry
serves, the model's chat template and tokenizer, every sampler, constrained decoding (JSON Schema, GBNF,
regex, choice), reasoning split and per-request energy. `ferric_chat` / `ferric_chat_stream` (a callback
per piece) / `ferric_complete` take and return the OpenAI JSON shapes; an error comes back as
`{"error": {"message": ...}}`. `scripts/ffi_check.py` holds a binding to answering exactly what the HTTP
server answers for the same request (Qwen2.5, Llama-3.2, Qwen3 with reasoning). Before this, the ABI
loaded every GGUF as a dense Qwen3 with Qwen's EOS ids and greedy decoding only.

**Python**: `bindings/python/ferric.py` — ctypes, no build step, no dependencies:
`ferric.Model(path).chat(messages, stream=print, max_tokens=64)`.

Build `libferric.{dylib,so,a}` with `cargo build -p ferric-ffi --release`. Reference programs in `bindings/`:

| Language | Route | Status |
|---|---|---|
| **Python** | `bindings/python/ferric.py` (ctypes) | ✅ verified = HTTP (`scripts/ffi_check.py`) |
| **C / C++** | `#include "libferric.h"` | ✅ verified |
| **Zig** | `@cImport("libferric.h")` — zero binding code | ✅ verified |
| **Swift** | Clang module map + idiomatic wrapper — **on-device Apple / Metal** | ✅ verified |
| **Go** | cgo | ✅ verified |
| **Mojo** | Python interop (`ferric` pkg) *or* `DLHandle` | written |
| **Java/Kotlin** | JNI / uniffi | via ABI |
| **C# / .NET** | P/Invoke (+ Unity) | via ABI |
| **Ruby / PHP / …** | their C FFI | via ABI |

Verified end-to-end on Qwen2.5-0.5B (Metal) from C, Zig, Swift, and Go: `ferric_generate` →
"Paris…", `ferric_generate_json` → `{"name":"John","age":30}`.
