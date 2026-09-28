# GBNF conformance fixture

`llama_cpp_integration.json` is llama.cpp's own `tests/test-grammar-integration.cpp` (ggml-org/llama.cpp @
`4da6337767f9`, 2026-09-28), case for case: 14 grammars with the 70 strings each must accept and the 69 it must
reject (text segments, and `token` segments for `<[id]>` / `!<[id]>` token elements, fed as that harness feeds
them), plus the 9 build expectations of its failure tests (missing root, undefined reference, left recursion
x4, a missing or custom root symbol). GBNF is llama.cpp's format, so llama.cpp is its authors.

Regenerate: `python3 extract.py <llama.cpp commit> llama_cpp_integration.json` beside a copy of the .cpp file.
Its JSON-Schema cases are not included: they test llama.cpp's schema-to-grammar converter, not GBNF.
Checked by `grammar::tests::llama_cpp_grammar_integration_cases` in crates/ferric-agent/src/grammar.rs.
