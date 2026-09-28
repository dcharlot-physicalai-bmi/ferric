# JSON Schema to GBNF conformance fixtures

All are checked by `crates/ferric-agent/tests/json_schema.rs`.

## `llama_cpp_cases.json`: the authors' tests

These are llama.cpp's own tests for its JSON-Schema-to-grammar converter (ggml-org/llama.cpp @
`4da6337767f9`, 2026-09-28), case for case:

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

## `differential.json.gz`: llama.cpp's converter as the oracle

`reference/build.sh <dir>` builds llama.cpp's converter itself, standalone, from its sources at that commit.
Two stub headers stand in for `common.h` (the three string helpers it uses, verbatim) and `ggml.h`
(`GGML_ASSERT`). Before it was used as an oracle, this build reproduced all 81 of the authors' cases.
`make_differential.py` runs it over 1469 schemas and keeps its output:

- the authors' schemas;
- integer bounds across magnitudes, signs and the i64 edges;
- fractional and exclusive bounds, string lengths and formats;
- about 90 regex patterns, including every refusal and fallback;
- property names that stress rule naming and the additional-properties key rule;
- arrays, tuples, combinators and `$ref`s;
- 1200 float literals (nlohmann's Grisu2 printing);
- refused schemas;
- 350 seeded random compositions.

The port must print the same grammar byte for byte, or the same error message.

There are 1409 grammars and 57 errors, all identical, plus 3 cases marked `deviation`. In those, llama.cpp reads
a pattern byte by byte and splits a multi-byte character in front of a quantifier (`^é+$` becomes
`"\xC3" "\xA9"+`, which is invalid UTF-8). The port reads per code point.

To regenerate:

```
reference/build.sh /tmp/js2g-ref
cargo build -p ferric-agent --release --example json_schema_to_gbnf
python3 make_differential.py /tmp/js2g-ref/js2g differential.json.gz target/release/examples/json_schema_to_gbnf
```

With the port's binary as the third argument, `make_differential.py` also prints every difference.
