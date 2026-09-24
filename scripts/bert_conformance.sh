#!/usr/bin/env bash
# BERT / XLM-R: Ferric vs THE MODEL AUTHORS' OWN IMPLEMENTATION.
#
# ⛔⛔ THE REFERENCE IS THE AUTHORS' CODE, NOT ANOTHER PORT. `bert` was Verified against llama-embedding
# only. Re-checked against the authors' `transformers` (tests/fixtures/bert/bge_hf.json, produced by
# crates/ferric-llama/examples/refgen/bert_ref.py in float32, eager attention), the encoder turned out to
# carry an error the llama.cpp comparison could never show, because it WAS llama.cpp's choice: GELU
# defaulted to ggml's tanh approximation, while the authors' `hidden_act: "gelu"` is the exact erf form.
#     tanh   bge-small max|diff| 1.4e-3 – 2.8e-3; reranker scores off by up to 0.0151
#     erf    bge-small max|diff| 1.2e-6 – 1.4e-6; reranker scores exact to 4 dp
# The tanh form is kept as this gate's NEGATIVE CONTROL: if it is not clearly worse, the gate cannot see
# the defect it exists for.
#
# ⚠ A 4-bit file cannot verify the math. bge-reranker-v2-m3 at Q4_K_M differs from the authors by
# 0.04–0.31 with EITHER GELU — the quantisation swamps a real defect. Convert the authors' weights at
# full precision (convert_hf_to_gguf.py --outtype f16 from their snapshot) and pass that file.
#
#   scripts/bert_conformance.sh <bge-small-en-v1.5 .gguf> [bge-reranker-v2-m3 F16/F32 .gguf]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMB="${1:-}"; RR="${2:-}"
FX="$ROOT/crates/ferric-llama/tests/fixtures/bert/bge_hf.json"
[ -f "$EMB" ] || { echo "usage: $0 <bge-small-en-v1.5 .gguf> [bge-reranker-v2-m3 full-precision .gguf]"; exit 2; }
[ -f "$FX" ]  || { echo "no authors' reference at $FX — regenerate with examples/refgen/bert_ref.py"; exit 2; }
BIN="$ROOT/target/release/examples/bert_ref"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example bert_ref || exit 2

python3 - "$BIN" "$FX" "$EMB" "$RR" <<'PY'
import json, math, os, struct, subprocess, sys
BIN, FX, EMB, RR = sys.argv[1:5]
ref = json.load(open(FX))

def file_type(path):
    def u32(f): return struct.unpack('<I', f.read(4))[0]
    def u64(f): return struct.unpack('<Q', f.read(8))[0]
    def rstr(f): return f.read(u64(f)).decode('utf-8', 'replace')
    def skip(f, t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in sizes: f.read(sizes[t]); return None
        if t == 8: return rstr(f)
        if t == 9:
            et = u32(f); n = u64(f)
            for _ in range(n): skip(f, et)
    with open(path, 'rb') as f:
        f.read(4); u32(f); u64(f); nkv = u64(f)
        for _ in range(nkv):
            k = rstr(f); t = u32(f)
            if k == 'general.file_type' and t in (4, 5): return u32(f)
            skip(f, t)
    return None

def run(model, ids, tanh=False):
    env = dict(os.environ); env.pop('FERRIC_BERT_GELU_TANH', None)
    if tanh: env['FERRIC_BERT_GELU_TANH'] = '1'
    r = subprocess.run([BIN, model, ",".join(map(str, ids))], capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-800:]); sys.exit(1)
    return {l.split(" ", 1)[0]: [float(x) for x in l.split(" ", 1)[1].split()]
            for l in r.stdout.splitlines() if " " in l}
def cos(a, b): return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
def mad(a, b): return max(abs(x - y) for x, y in zip(a, b))

ok = True
print(f"reference: transformers {ref['transformers']}, torch {ref['torch']}, float32, eager")

# ── 1. the embedding model ───────────────────────────────────────────────────────────────────
E = ref["embed"]
ft = file_type(EMB)
if ft not in (0, 1):
    print(f"⛔ {os.path.basename(EMB)} is quantised (file_type {ft}) — it cannot verify the math; use F16/F32"); sys.exit(2)
print(f"\n{E['model']} ({os.path.basename(EMB)}, {'F32' if ft == 0 else 'F16'}); authors' hidden_act="
      f"{E['hidden_act']!r}, pooling={ {k for k, v in E['sentence_transformers_pooling'].items() if v is True} }")
for it in E["items"]:
    good = mad(run(EMB, it["ids"])["CLS"], it["cls_hidden"])
    bad = mad(run(EMB, it["ids"], tanh=True)["CLS"], it["cls_hidden"])
    flag = ""
    if good > 1e-4: flag += "  <-- FAIL (beyond tol 1e-4)"; ok = False
    if bad < 20 * good: flag += "  <-- control not ≥20x worse"; ok = False
    print(f"  {len(it['ids']):>4} tok  first-token row max|diff| {good:.2e}   control (tanh GELU) {bad:.2e}  "
          f"= {bad / max(good, 1e-12):,.0f}x{flag}")

# ── 2. the cross-encoder reranker ────────────────────────────────────────────────────────────
R = ref["rerank"]
if not RR:
    print(f"\n(no reranker file given — {R['model']} skipped)")
else:
    ft = file_type(RR)
    if ft not in (0, 1):
        print(f"\n⛔ {os.path.basename(RR)} is quantised (file_type {ft}) — a 4-bit file differs from the authors by "
              f"0.04–0.31 with either GELU and cannot verify the math. Pass an F16/F32 conversion."); ok = False
    else:
        print(f"\n{R['model']} ({os.path.basename(RR)}, {'F32' if ft == 0 else 'F16'}); head={R['classifier']}")
        for it in R["items"]:
            got = run(RR, it["ids"])["RANK"][0]
            bad = run(RR, it["ids"], tanh=True)["RANK"][0]
            want = it["logit"][0]
            d = abs(got - want)
            flag = "  <-- FAIL" if d > 1e-3 else ""
            if flag: ok = False
            print(f"  {it['query'][:16]:<16}| {it['doc'][:24]:<24} authors {want:+9.4f}  ferric {got:+9.4f}  "
                  f"|diff| {d:.4f}   (tanh control {abs(bad - want):.4f}){flag}")

print("\n" + ("✅ BERT and XLM-R agree with the authors; the tanh control is clearly worse"
              if ok else "⛔ BERT conformance FAILED"))
sys.exit(0 if ok else 1)
PY
