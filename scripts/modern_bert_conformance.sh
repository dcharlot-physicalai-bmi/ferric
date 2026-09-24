#!/usr/bin/env bash
# ModernBERT: Ferric vs THE MODEL AUTHORS' OWN IMPLEMENTATION, with the hard mechanisms proved
# load-bearing.
#
# ⛔⛔ THE REFERENCE IS THE AUTHORS' CODE, NOT ANOTHER PORT. This gate first compared against
# `llama-embedding`. That caught a real disagreement on the classifier head (0.03-0.10) — and the
# disagreement was then explained away as "F16 precision amplified by the head", because llama.cpp was
# being treated as the gold standard it is not. Checked against the authors' `transformers` instead:
# Ferric's head matched THEIR HEAD FED THE FIRST TOKEN, and llama.cpp matched their real output. The
# checkpoint's config says `classifier_pooling: "mean"`; the GGUF converter drops that key; llama.cpp
# gets it right only by HARDCODING mean for every modern-bert reranker. A peer implementation is a
# cross-check. It is never the arbiter.
#
# The reference numbers are COMMITTED (tests/fixtures/modern_bert/gte_reranker_hf.json), produced by
# crates/ferric-llama/examples/refgen/modern_bert_ref.py in float32 with eager attention. So this gate
# needs only the GGUF — no Python, no llama.cpp. Regenerate the fixture with any Python that has
# `transformers` + `torch` and keep the generator in the repo: a reference that lives in scratch is a
# verification nobody can repeat.
#
# ⛔ WHY THE NEGATIVE CONTROLS ARE NOT OPTIONAL. ModernBERT has two places a port fails silently, and a
# match proves nothing about a mechanism unless disabling it breaks the match:
#   1. the SYMMETRIC sliding-window band (two layers in three) — only visible past ~130 tokens, which
#      is why the controls run on the fixture's 222-token input and never on the short pairs;
#   2. TWO rope bases — global 160000, sliding 10000 here and on ModernBERT-large, but IDENTICAL on
#      mmBERT-base, so a one-base port is exact on mmBERT and wrong on the others.
#
#   scripts/modern_bert_conformance.sh <gte-reranker-modernbert-base .gguf> [fixture.json]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"
FX="${2:-$ROOT/crates/ferric-llama/tests/fixtures/modern_bert/gte_reranker_hf.json}"
[ -f "$M" ] || { echo "usage: $0 <gte-reranker-modernbert-base .gguf> [fixture.json]"; exit 2; }
[ -f "$FX" ] || { echo "no authors' reference at $FX — regenerate it with examples/refgen/modern_bert_ref.py"; exit 2; }

BIN="$ROOT/target/release/examples/modern_bert_ref"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example modern_bert_ref || exit 2

python3 - "$BIN" "$M" "$FX" <<'PY'
import json, math, os, subprocess, sys
BIN, M, FX = sys.argv[1:4]
ref = json.load(open(FX))

def run(ids, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    r = subprocess.run([BIN, M, ",".join(map(str, ids))], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stderr[-800:]); sys.exit(1)
    return {ln.split(" ", 1)[0]: ln.split(" ", 1)[1] for ln in r.stdout.splitlines() if " " in ln}

def vec(s): return [float(x) for x in s.split()]
def cos(a, b): return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
def mad(a, b): return max(abs(x - y) for x, y in zip(a, b))

def file_type(path):
    import struct
    def u32(f): return struct.unpack('<I', f.read(4))[0]
    def u64(f): return struct.unpack('<Q', f.read(8))[0]
    def rstr(f): return f.read(u64(f)).decode('utf-8', 'replace')
    def skip(f, t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in sizes: f.read(sizes[t]); return
        if t == 8: rstr(f); return
        if t == 9:
            et = u32(f); n = u64(f)
            for _ in range(n): skip(f, et)
    with open(path, 'rb') as f:
        f.read(4); u32(f); u64(f); nkv = u64(f)
        for _ in range(nkv):
            k = rstr(f); t = u32(f)
            if k == 'general.file_type' and t in (4, 5): return u32(f)
            skip(f, t)

# ⛔ The tolerance follows the WEIGHTS. A fixed 0.03 would pass a revert to tanh GELU (0.0081 on the
# F16 file) — the exact defect this gate exists to catch. Measured with the authors' erf GELU:
#   F32 weights (converted from the authors' own F32 safetensors): encoder 2.5e-6, head 0.0000
#   F16 weights (third-party file, authors' F32 rounded):           encoder 1.1e-3, head 0.0003
FT = file_type(M)
if FT not in (0, 1):
    print(f"⛔ {os.path.basename(M)} is quantised (file_type {FT}) — it cannot verify the math; use F16/F32")
    sys.exit(2)
ENC_TOL, HEAD_TOL = (1e-4, 1e-3) if FT == 0 else (5e-3, 3e-3)
print(f"reference: {ref['model']} — transformers {ref['transformers']}, torch {ref['torch']}, "
      f"float32, eager; classifier_pooling={ref['classifier_pooling']!r}")
print(f"weights: {os.path.basename(M)} ({'F32' if FT == 0 else 'F16'}) — tolerances encoder {ENC_TOL:g}, head {HEAD_TOL:g}")
ok = True

# ── 1. the encoder, long enough that the window actually masks ──────────────────────────────
L = ref["long"]
print(f"\nencoder, {len(L['ids'])} tokens (the band masks past ~130), vs the authors' mean hidden state:")
rows = {}
for label, env in (("as implemented", None),
                   ("control: window disabled", {"FERRIC_MB_NO_SWA": "1"}),
                   ("control: one rope base", {"FERRIC_MB_ONE_ROPE": "1"})):
    o = run(L["ids"], env)
    got = vec(o["MEAN"])
    rows[label] = mad(got, L["mean_hidden"])
    print(f"  {label:<26} cosine {cos(got, L['mean_hidden']):.8f}   max|diff| {rows[label]:.3e}")
base = rows["as implemented"]
for k in ("control: window disabled", "control: one rope base"):
    if rows[k] < 20 * base:
        print(f"  ⛔ {k}: {rows[k]:.3e} is not ≥20x the implemented {base:.3e} — that mechanism is NOT "
              f"load-bearing on this input, so the match proves nothing about it"); ok = False
if base > ENC_TOL:
    print(f"  ⛔ the implemented encoder is {base:.3e} from the authors — beyond tol {ENC_TOL:g}"); ok = False

# ── 2. the classifier head, on the authors' own pair tokenization ────────────────────────────
print("\nclassifier head vs the authors' score (their tokenizer, their pooling):")
worst = 0.0
for p in ref["pairs"]:
    o = run(p["ids"])
    got = float(o["RANK"].split()[0])
    want = p["logit_configured"][0]
    d = abs(got - want); worst = max(worst, d)
    flag = "" if (d <= HEAD_TOL and (got > 0) == (want > 0)) else "   <-- FAIL"
    print(f"  {p['query'][:16]:<16} | {p['doc'][:26]:<26} authors {want:+8.4f}  ferric {got:+8.4f}  |diff| {d:.4f}{flag}")
    if flag: ok = False
    # The first-token pooling this head used to have is a DIFFERENT function; it must be visibly worse.
    wrong = p["logit_cls"][0]
    if abs(wrong - want) < d:
        print(f"    ⛔ the authors' head fed the FIRST token ({wrong:+.4f}) is closer than Ferric — "
              f"the pooling check cannot tell the two rules apart on this pair"); ok = False
print(f"  worst |diff| {worst:.4f}  (tol {HEAD_TOL:g})")
# ⭐ NEGATIVE CONTROL: ggml's tanh GELU — what this port used until checked against the authors.
tanh_worst = max(abs(float(run(p["ids"], {"FERRIC_MB_GELU_TANH": "1"})["RANK"].split()[0]) - p["logit_configured"][0])
                 for p in ref["pairs"])
ratio = tanh_worst / max(worst, 1e-9)
print(f"  control: tanh GELU  worst |diff| {tanh_worst:.4f}  = {ratio:,.0f}x the implemented")
if ratio < 5:
    print("  ⛔ the tanh control is not ≥5x worse — this gate cannot see the activation"); ok = False

print("\n" + ("✅ encoder and head agree with the authors; window, rope bases and GELU are all load-bearing"
              if ok else "⛔ conformance FAILED"))
sys.exit(0 if ok else 1)
PY
rc=$?

# ── optional peer cross-check: reported, never gating ─────────────────────────────────────────
LE="$(command -v llama-embedding || true)"
if [ "$rc" = "0" ] && [ -n "$LE" ]; then
  echo
  echo "peer cross-check (llama.cpp $(llama-embedding --version 2>&1 | head -1 | awk '{print $2, $4}')) — informational only:"
  python3 - "$FX" <<'PY' > /tmp/.mb_pairs.tsv
import json, sys
for p in json.load(open(sys.argv[1]))["pairs"]: print(p["query"] + "\t" + p["doc"] + "\t" + str(p["logit_configured"][0]))
PY
  while IFS=$'\t' read -r q d want; do
    printf '%s\t%s' "$q" "$d" > /tmp/.mb_pair.txt
    got=$("$LE" -m "$M" -f /tmp/.mb_pair.txt --pooling rank -ngl 0 --embd-normalize -1 2>/dev/null \
          | grep -i "rerank score" | awk '{print $NF}')
    printf "  %-16s authors %+8.4f  llama.cpp %+8.3f\n" "${q:0:16}" "$want" "${got:-nan}"
  done < /tmp/.mb_pairs.tsv
fi
exit $rc
