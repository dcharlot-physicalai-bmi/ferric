#!/usr/bin/env bash
# Independent correctness check: Ferric's greedy decode must match llama.cpp (the reference) token
# for token on real GGUF models. Cross-fabric consistency proves Ferric agrees with *itself*; this
# proves it agrees with an independent implementation — matching argmax at every step means the
# logits are right, not merely that the text is coherent.
#
# Needs: llama.cpp's `llama-simple` on PATH (brew install llama.cpp), and the models below.
# Build first: cargo build -p ferric-llama --example run_qwen3 --release
set -u
PROMPT="The capital of France is"
N=14
BIN=./target/release/examples/run_qwen3
CACHE="$HOME/.cache/ferric"

norm() { tr -d '\0' | tr -s ' \n' ' ' | sed 's/^ *//;s/ *$//'; }

compare() {
  local model="$1" name="$2"
  [ -f "$model" ] || { echo "⏭  $name (model not present)"; return; }
  local ll fe
  ll=$(llama-simple -m "$model" -n "$N" "$PROMPT" </dev/null 2>/dev/null | norm)
  fe=$("$BIN" "$model" "$PROMPT" "$N" 2>/dev/null | grep -vE "^loaded|prompt |ms/token" | norm)
  if [ "$ll" = "$fe" ]; then echo "✅ $name — token-for-token identical to llama.cpp"
  else echo "❌ $name DIFFERS"; echo "   llama.cpp: $ll"; echo "   ferric:    $fe"; fi
}

echo "Ferric vs llama.cpp (greedy, $N tokens): '$PROMPT'"
compare "$CACHE/qwen3-0.6b-q4km.gguf"    "Qwen3-0.6B Q4_K_M   (Q4_K + Q6_K)"
compare "$CACHE/qwen3-0.6b-q5km.gguf"    "Qwen3-0.6B Q5_K_M   (Q5_K + Q6_K)"
compare "$CACHE/qwen2.5-0.5b-q6k.gguf"   "Qwen2.5-0.5B Q6_K   (qwen2: QKV bias, no QK-norm; Q8_0 + Q6_K)"
