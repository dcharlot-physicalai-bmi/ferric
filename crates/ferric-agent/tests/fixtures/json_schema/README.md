# JSON Schema to GBNF conformance fixtures

`llama_cpp_cases.json` holds llama.cpp's own tests for its JSON-Schema-to-grammar converter (ggml-org/llama.cpp
@ `4da6337767f9`, 2026-09-28), case for case:

- `cases`: every `test({...})` of `tests/test-json-schema-to-grammar.cpp`, which is 81 schemas: 78 with the
  exact grammar the converter must print and 3 it must refuse. The expected grammar is stored after the
  harness's own `trim()`, which strips the ends and each line's leading indentation. The Rust check requires the
  output to equal it plus the final newline, byte for byte.
- `extra`: that file's two hand-written blocks. A schema parsed up front converts the same as its JSON, and a
  property's `$ref` node converted alone through `build_grammar` names the ref rule.
- `schema_matching`: every `test_schema(...)` of `tests/test-grammar-integration.cpp`. There are 33 schemas,
  and each converted grammar must match all 126 passing strings and none of the 122 failing ones.

Regenerate with `python3 extract.py 4da6337767f9 llama_cpp_cases.json`, run beside copies of both .cpp files
(`curl -sfLO https://raw.githubusercontent.com/ggml-org/llama.cpp/4da6337767f9/tests/<file>`).
Checked by `crates/ferric-agent/tests/json_schema.rs`.
