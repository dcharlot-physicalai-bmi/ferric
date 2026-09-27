#!/usr/bin/env bash
# Per-request LoRA over HTTP vs HF PEFT: ferric-serve with two PEFT-made adapters must answer exactly as PEFT
# does (greedy, float32) for the base and each adapter, under every way of selecting one — `model: "<adapter>"`
# (vLLM), `lora: [{"id"|"name", "scale"}]` (llama-server) — alone AND interleaved in one batch, and refuse an
# unknown adapter by name. The engine math is gated separately (scripts/lora_conformance.sh); this gates the
# WIRING: selection → the sequence's cache, before prompt-cache seeding, per batched row.
# Mutation-checked: dropping the model-name selection → 3 mismatches; the batch ignoring selections → 8.
#
#   scripts/serve_lora_conformance.sh <qwen2.5-0.5b-instruct F32 .gguf>
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; M="${1:-}"
[ -f "$M" ] || { echo "usage: $0 <qwen2.5-0.5b-instruct F32 .gguf>"; exit 2; }
FX="$ROOT/crates/ferric-llama/tests/fixtures/lora"; PORT=${PORT:-18811}
cargo build -q --release -p ferric-serve || exit 2
"$ROOT/target/release/ferric-serve" "$M" --port "$PORT" --lora syn_a="$FX/syn_a" --lora syn_b="$FX/syn_b" >/dev/null 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 90); do curl -s -m 2 "localhost:$PORT/health" >/dev/null && break; sleep 1; done
python3 "$ROOT/scripts/serve_lora_check.py" "$PORT" "$ROOT/crates/ferric-serve/tests/fixtures/lora_serve/peft_greedy.json" | tee /dev/stderr | tail -1 | grep -q "ALL OK"
