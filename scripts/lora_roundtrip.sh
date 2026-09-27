#!/usr/bin/env bash
# Ferric TRAINS a LoRA, EXPORTS it as a PEFT adapter, and Hugging Face PEFT — reading only the exported files —
# must reproduce the logits Ferric computed with the adapter IN MEMORY, within PEFT's own float32-vs-float64
# floor. That is the whole claim "Ferric emits adapters the ecosystem can use", measured.
#
#   1. examples/finetune_lora_peft: LoRA pairs on q_proj/v_proj of the last 2 blocks of Qwen2.5-0.5B-Instruct,
#      trained a few Adam steps by Ferric autograd; applied UNMERGED by the runtime to a 139-token text
#      (the ROW lines); written to <tmp>/peft (adapter_config.json + adapter_model.safetensors) and
#      <tmp>/adapter.gguf, both read back value for value.
#   2. refgen/lora_ref.py logits: `PeftModel.from_pretrained(<authors' base>, <tmp>/peft)` in float32 and
#      float64 on the same ids.
#   3. |Ferric in-memory - PEFT float64| must be within 3x PEFT's own floor at every position, argmax equal.
# Controls, each >= 20x the correct distance: PEFT with the adapter DISABLED (the trained delta must be
# visible at all), and the exported config with lora_alpha DOUBLED (the gate can see the config it wrote).
#
#   PEFT_PYTHON=<python with torch, transformers, peft> scripts/lora_roundtrip.sh <qwen2.5-0.5b-instruct F32 .gguf>
#
# (A scratch venv: `uv venv -p python3.12 v && VIRTUAL_ENV=v uv pip install torch transformers peft`.)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; BASE_ID="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
[ -f "$M" ] || { echo "usage: PEFT_PYTHON=... $0 <qwen2.5-0.5b-instruct F32 .gguf> [base_model_id]"; exit 2; }
[ -x "${PEFT_PYTHON:-}" ] || { echo "⛔ PEFT_PYTHON must name a python with torch + transformers + peft — the \
reference cannot be skipped"; exit 2; }
REFGEN="$ROOT/crates/ferric-llama/examples/refgen/lora_ref.py"
cargo build -q -p ferric-llama --release --example finetune_lora_peft || exit 2
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

IDS_SAMPLE="$("$PEFT_PYTHON" "$REFGEN" ids "$BASE_ID" 2>/dev/null)" || { echo "⛔ tokenizing with $BASE_ID failed"; exit 1; }
IDS="$(echo "$IDS_SAMPLE" | sed -n 1p)"; SAMPLE="$(echo "$IDS_SAMPLE" | sed -n 2p)"
"$ROOT/target/release/examples/finetune_lora_peft" "$M" "$OUT" "$BASE_ID" "$IDS" "$SAMPLE" > "$OUT/ferric.txt" \
  || { cat "$OUT/ferric.txt"; exit 1; }
grep -v '^ROW ' "$OUT/ferric.txt"
"$PEFT_PYTHON" "$REFGEN" logits "$BASE_ID" "$OUT/peft.json.gz" "$OUT/peft" --ids "$IDS" 2>"$OUT/peft.log" \
  || { tail -20 "$OUT/peft.log"; exit 1; }
# Control: the same export with lora_alpha doubled — a config the exporter could plausibly get wrong.
cp -r "$OUT/peft" "$OUT/peft_alpha2"
python3 - "$OUT/peft_alpha2/adapter_config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["lora_alpha"] *= 2; json.dump(c, open(sys.argv[1], "w"))
PY
"$PEFT_PYTHON" "$REFGEN" logits "$BASE_ID" "$OUT/peft_alpha2.json.gz" "$OUT/peft_alpha2" --ids "$IDS" 2>>"$OUT/peft.log" \
  || { tail -20 "$OUT/peft.log"; exit 1; }

python3 - "$OUT" <<'PY'
import gzip, json, os, sys
OUT = sys.argv[1]
ref = json.load(gzip.open(os.path.join(OUT, "peft.json.gz"), "rt"))
alt = json.load(gzip.open(os.path.join(OUT, "peft_alpha2.json.gz"), "rt"))
rows = {l.split()[1]: l.split() for l in open(os.path.join(OUT, "ferric.txt")) if l.startswith("ROW ")}
POS, V = ref["positions"], ref["variants"]
F = [[float(x) for x in rows[str(t)][3:]] for t in POS]
def S(r, k): return [x["sample"] for x in r["variants"][k]]
def dist(a, b): return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
fl = dist(S(ref, "adapted_f32"), S(ref, "adapted_f64"))
e = dist(F, S(ref, "adapted_f64"))
am = sum(int(rows[str(t)][2]) == r["top"][0][0] for t, r in zip(POS, V["adapted_f64"]))
cfg = ref["adapters"][0]["config"]
print(f"reference: PEFT {ref['peft']} loading Ferric's export onto {ref['model']} (transformers {ref['transformers']})")
print(f"export:    r={cfg['r']} alpha={cfg['lora_alpha']} targets={cfg['target_modules']} layers={cfg.get('layers_to_transform')}")
ok = e <= 3 * fl and am == len(POS)
print(f"  Ferric in memory vs PEFT from the files: |diff| {e:.3e} = {e / fl:.2f}x PEFT's floor ({fl:.3e})   "
      f"argmax {am}/{len(POS)}{'' if ok else '   <-- FAIL'}")
for label, other in [("adapter disabled in PEFT", S(ref, "base_f64")), ("exported lora_alpha doubled", S(alt, "adapted_f64"))]:
    c = dist(F, other)
    good = c >= 20 * e
    ok &= good
    print(f"  control: {label:28s} {c:.3e} = {c / e:10,.0f}x{'' if good else '   <-- CONTROL FAILED'}")
print("✅ PEFT reproduces Ferric's fine-tune from the exported adapter" if ok else "⛔ LoRA export round trip FAILED")
sys.exit(0 if ok else 1)
PY
