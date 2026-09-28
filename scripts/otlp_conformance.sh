#!/usr/bin/env bash
# OTLP trace export, both protocols, decoded by the OpenTelemetry protocol definitions (opentelemetry-proto).
# usage: scripts/otlp_conformance.sh <model.gguf>    (PYTHON = an interpreter with opentelemetry-proto)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?model.gguf}
PY=${PYTHON:-python3}
"$PY" -c "import opentelemetry.proto.collector.trace.v1.trace_service_pb2" 2>/dev/null || { echo "needs: $PY -m pip install opentelemetry-proto"; exit 2; }
cargo build --release -q -p ferric-serve
for p in http/json http/protobuf; do "$PY" scripts/otlp_conformance.py target/release/ferric-serve "$MODEL" "$p"; done
