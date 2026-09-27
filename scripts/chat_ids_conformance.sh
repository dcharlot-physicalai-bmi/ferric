#!/usr/bin/env bash
# The prompt a chat request becomes: ferric-serve's token ids vs Hugging Face's
# apply_chat_template(tokenize=True) with the MODEL AUTHORS' own tokenizer. Template and tokenizer together.
#
#   scripts/chat_ids_conformance.sh <python with transformers> <model.gguf> <hf-repo-with-the-authors-tokenizer> [port]
#
# Starts the server with FERRIC_DUMP_IDS=1, sends one multi-turn conversation with a system prompt and
# non-ASCII text, and compares the ids the model would see. This is the check that found three defects the
# template-string harness (scripts/chat_template_conformance.py) could not: serde_json's IndexMut adding
# `"tool_calls": null` to every message (Llama-3.2 refused all chats), Phi-3's rstrip after <|…|> tokens,
# and the SentencePiece space prefix owed to every text fragment after a special token.
set -u
PY="${1:?python}"; GGUF="${2:?gguf}"; REPO="${3:?hf repo}"; PORT="${4:-18299}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cargo build -q -p ferric-serve --release || exit 2
LOG="$(mktemp)"; trap 'pkill -f "ferric-serve.*--port $PORT" 2>/dev/null; rm -f "$LOG"' EXIT
FERRIC_DUMP_IDS=1 "$ROOT/target/release/ferric-serve" "$GGUF" --port "$PORT" > "$LOG" 2>&1 &
for _ in $(seq 1 120); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
BODY='{"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"Name a colour, «vite»."},{"role":"assistant","content":"Blue."},{"role":"user","content":"Another?"}],"max_tokens":1}'
curl -s "localhost:$PORT/v1/chat/completions" -d "$BODY" > /dev/null
FER=$(grep -m1 "prompt ids" "$LOG" | sed 's/.*: //')
[ -n "$FER" ] || { echo "⛔ the server produced no prompt ids:"; tail -5 "$LOG"; exit 1; }
"$PY" - "$REPO" "$FER" "$BODY" <<'PYEOF'
import json, sys
from transformers import AutoTokenizer
repo, fer, body = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
tok = AutoTokenizer.from_pretrained(repo)
hf = tok.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=True)
if not isinstance(hf, list): hf = hf["input_ids"]
p = next((i for i, (a, b) in enumerate(zip(hf, fer)) if a != b), None if len(hf) == len(fer) else min(len(hf), len(fer)))
if p is None:
    print(f"✅ {repo}: {len(fer)} ids identical to apply_chat_template"); sys.exit(0)
print(f"⛔ {repo}: first difference at {p}: HF {hf[max(0,p-3):p+4]} {tok.decode(hf[max(0,p-3):p+4])!r}\n"
      f"                          Ferric {fer[max(0,p-3):p+4]} {tok.decode(fer[max(0,p-3):p+4])!r}  ({len(fer)} vs {len(hf)})")
sys.exit(1)
PYEOF
