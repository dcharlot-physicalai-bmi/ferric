#!/bin/bash
# Build the browser bundle. Requires wasm-pack.
set -e
cd "$(dirname "$0")/.."
wasm-pack build crates/ferric-web --target web --out-dir ../../web/pkg --release
echo "built web/pkg — now: ./web/serve.sh"
