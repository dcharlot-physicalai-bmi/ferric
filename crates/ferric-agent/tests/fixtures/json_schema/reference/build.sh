#!/bin/sh
# Build llama.cpp's own JSON-Schema-to-grammar converter, standalone, at the commit ferric_agent::json_schema
# ports (4da6337767f9): its converter sources fetched unchanged, nlohmann/json from its vendor/, and two stubs
# from this directory in place of headers that would pull in all of llama/ggml: common.h (the three string
# helpers the converter uses, bodies verbatim from common.cpp) and ggml.h (GGML_ASSERT). driver.cpp runs
# json_schema_to_grammar(schema, force_gbnf = true) over a JSON array of schema texts.
#
# usage: build.sh <out dir>        prints the path of the binary, <out dir>/js2g
set -e
C=4da6337767f9
OUT=${1:?usage: build.sh <out dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
B=https://raw.githubusercontent.com/ggml-org/llama.cpp/$C
mkdir -p "$OUT/inc/nlohmann"
for f in json-schema-to-grammar.cpp json-schema-to-grammar.h json-schema.cpp json-schema.h json.cpp json.h trie.cpp trie.h unicode.cpp unicode.h; do
    curl -sfL -o "$OUT/$f" "$B/common/$f"
done
for f in json.hpp json_fwd.hpp; do curl -sfL -o "$OUT/inc/nlohmann/$f" "$B/vendor/nlohmann/$f"; done
cp "$HERE/common.h" "$HERE/ggml.h" "$OUT/inc/"
cp "$HERE/driver.cpp" "$OUT/"
cd "$OUT"
c++ -std=c++17 -O1 -Iinc -I. -o js2g driver.cpp json-schema-to-grammar.cpp json-schema.cpp json.cpp trie.cpp unicode.cpp
echo "$OUT/js2g"
