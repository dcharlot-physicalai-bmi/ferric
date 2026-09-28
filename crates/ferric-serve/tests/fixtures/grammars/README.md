# Sample GBNF grammars

Copied unchanged from llama.cpp `grammars/` at commit `4da6337767f9` (MIT licence,
https://github.com/ggml-org/llama.cpp/tree/4da6337767f9/grammars): `json`, `json_arr`, `arithmetic`,
`c`, `chess`, `list`, `japanese`, `english`.

Used by `crates/ferric-serve/src/constrain.rs` tests: the cached token mask must equal the full
vocabulary-trie walk at every step of seeded random walks through each grammar
(`the_cached_mask_equals_the_full_walk`), and the ignored bench walks `json.gbnf` and `c.gbnf` on a real
152k-token vocabulary (`bench_grammar_mask_on_a_real_vocabulary`).
