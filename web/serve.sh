#!/bin/bash
# Serve the demo. Uses serve.py, NOT `python3 -m http.server`, because the latter ignores Range and
# answers 200 with the whole body — verified: a 64-byte range came back as the full 675 MB file.
set -e
cd "$(dirname "$0")"
[ -f pkg/ferric_web.js ] || { echo "run ./build.sh first"; exit 1; }
[ -e model.gguf ] || echo "note: no ./model.gguf — symlink or copy a checkpoint here first"
exec python3 serve.py "${1:-8770}"
