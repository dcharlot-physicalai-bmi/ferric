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

## `semantic.json.gz`: what the grammars mean, according to `jsonschema`

`make_semantic.py` covers 112 schemas: the authors' success schemas and 16 realistic API ones (nested objects,
arrays of objects, enums, optional and nullable fields, string lengths, integer ranges, formats, `$defs`, a
recursive tree, a tuple, a discriminated union, strict all-required). 3 of the authors' schemas are skipped
because they are not valid Draft 2020-12 (`prefixItems` given as a schema), so jsonschema cannot evaluate them.

The instances come from three sources:
- values generated from each schema, with keys in the order the grammar emits them;
- mutations of those values;
- probes aimed at each known departure.

Each instance is serialized in the ways the `space` rule allows and labelled by jsonschema 4.26.0 (Draft 2020-12,
or 2019-09 where `items` is a list), using that draft's format checker.

The grammar's verdict comes from `examples/json_schema_accepts`. Every disagreement must be explained by a
counterfactual, or the script stops. There are three kinds:
- **Respelling**: the same value written differently (whitespace, number spelling, key order, escapes) and
  accepted by the grammar.
- **Reinterpretation**: jsonschema, given the schema rewritten the way llama.cpp reads it, agrees with the
  grammar. Examples are a dropped keyword, allOf merged llama.cpp's way, and objects closed by default.
- **Limit**: 17-digit integers, cut to 16 digits.

`causes` in the file explains each one.

Result: 8316 instances (5436 valid and 2880 invalid according to jsonschema). The grammar agrees on 7087. The
other 1229 depart for a recorded cause:

| Cause | Instances |
|---|---:|
| whitespace | 265 |
| type-inferred | 194 |
| key-order | 191 |
| number-spelling | 137 |
| additional-properties-default | 121 |
| empty-name-is-root | 77 |
| tuple-exact | 64 |
| number-bounds-ignored | 50 |
| integer-digits | 40 |
| raw-del | 33 |
| pattern-widened | 31 |
| pattern-raw-chars | 8 |
| pattern-anchored-whole | 6 |
| keyword-ignored | 5 |
| keyword-ignored + number-bounds-ignored | 3 |
| astral-escape-length | 1 |
| pattern-dot-dialect | 1 |
| additional-key-prefix | 1 |
| all-of-merged | 1 |

The probes include the edges of the `space` rule that it accepts (two newlines; a newline and exactly 20
blanks), so a wrong space rule fails this check too, not only the byte-identical ones.

To regenerate (the output is deterministic):

```
cargo build -p ferric-agent --release --example json_schema_accepts --example json_schema_to_gbnf
<venv>/bin/python make_semantic.py target/release/examples/json_schema_accepts semantic.json.gz
```

## `mutate.py`: can these tests fail?

`mutate.py` breaks the port in 8 plausible ways, one at a time, runs the tests, and restores the source:
- an array's item rule misnamed;
- `required` dropped;
- a digit range reaching one past the bound;
- `exclusiveMinimum` taken as inclusive;
- the space rule allowing 19 blanks, or one newline at most;
- Grisu2's rounding step skipped;
- optional properties emitted before required ones.

Every mutation must be caught. Run it from the repository root:

```
python3 crates/ferric-agent/tests/fixtures/json_schema/mutate.py
```
