# ferric

A pure-Rust cross-fabric AI runtime, adoptable from Python.

```python
from ferric import Ferric
llm = Ferric("Qwen/Qwen2.5-0.5B-Instruct-GGUF")     # HF repo id or local .gguf
print(llm.generate("The capital of France is"))

# Guaranteed schema-conformant structured output (guided decoding):
person = llm.generate_object("Invent a person.", {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
})
```

Deterministic by default. Runs the standard GGUF ecosystem across Metal / Vulkan / CPU, and the
*same* runtime runs in the browser (WebGPU). This package drives the `ferric-serve` binary; set
`FERRIC_SERVE_BIN` or pass `bin=` to point at it.
