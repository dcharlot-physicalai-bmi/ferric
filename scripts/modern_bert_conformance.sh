#!/usr/bin/env bash
# ModernBERT: Ferric vs llama.cpp on the SAME ids, with the two hard mechanisms proved load-bearing.
#
# ⛔ WHY THE NEGATIVE CONTROLS ARE NOT OPTIONAL. ModernBERT has exactly two places a port fails
# silently, and "it matches the reference" says nothing about either unless disabling them makes the
# number visibly worse:
#   1. the SYMMETRIC sliding-window band (two layers in three), and
#   2. TWO rope bases — global 160000, sliding 10000 on ModernBERT-large and
#      gte-reranker-modernbert-base, but IDENTICAL on mmBERT-base. A port that reads one base is
#      bit-exact on mmBERT and wrong on the others.
# Measured on 222 tokens, gte-reranker-modernbert-base-F16:
#   as implemented         cosine 0.99999965  max|diff| 2.0e-05
#   window disabled        cosine 0.99272915  max|diff| 1.6e-02   (~800x worse)
#   one rope base          cosine 0.99971104  max|diff| 5.1e-03   (~257x worse)
# ⚠ Note the last row READS AS FINE. 0.9997 cosine is what the trap looks like from the outside.
#
# ⛔ USE -f ON BOTH SIDES. `llama-tokenize -p` takes one line; mixing `-p` for the ids and `-f` for
# the embedding compares the answer on two DIFFERENT inputs. That mistake moved a number from 2e-5
# to 3e-3 here and read as a regression in the port.
#
#   scripts/modern_bert_conformance.sh <model.gguf> [prompt-file]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"
[ -f "$M" ] || { echo "usage: $0 <modern-bert.gguf> [prompt-file]"; exit 2; }
LT="$(command -v llama-tokenize || true)"; LE="$(command -v llama-embedding || true)"
[ -z "$LT" ] || [ -z "$LE" ] && { echo "needs llama-tokenize AND llama-embedding on PATH — the \
reference is the point of this gate, so this is a hard exit, never a silent pass"; exit 2; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
P="${2:-}"
if [ -z "$P" ]; then
  P="$TMP/p.txt"
  python3 - "$P" <<'PY'
import sys
w=['the','quick','brown','fox','jumps','over','lazy','dog','while','seven','ships','sail','past',
   'harbour','lights','before','dawn','breaks','across','water']
# >129 tokens, or the 129-wide band masks nothing and the window is never exercised.
open(sys.argv[1],'w').write(' '.join(w[i%20] for i in range(220)))
PY
fi

IDS="$($LT -m "$M" -f "$P" --ids 2>/dev/null | tail -1 | tr -d '[] ')"
N="$(echo "$IDS" | tr ',' '\n' | wc -l | tr -d ' ')"
[ "$N" -lt 130 ] && echo "⚠ only $N tokens — under 130 the symmetric band masks NOTHING and this run \
does not exercise the sliding window at all"
$LE -m "$M" -f "$P" --pooling cls -ngl 0 --embd-normalize 2 2>/dev/null \
  | grep "^embedding 0:" | sed 's/^embedding 0: *//' > "$TMP/ref.txt"
[ -s "$TMP/ref.txt" ] || { echo "llama-embedding produced no vector"; exit 2; }

BIN="$ROOT/target/release/examples/modern_bert_ref"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example modern_bert_ref || exit 2

echo "model: $(basename "$M")   tokens: $N"
run_one () {
  env $2 "$BIN" "$M" "$IDS" 2>/dev/null > "$TMP/o.txt"
  python3 - "$1" "$TMP/ref.txt" "$TMP/o.txt" <<'PY'
import sys
ref=[float(x) for x in open(sys.argv[2]).read().split()]
got=[float(x) for ln in open(sys.argv[3]) if ln.startswith("NRM ") for x in ln[4:].split()]
if not ref or len(ref)!=len(got):
    print(f"  {sys.argv[1]:<36} MISMATCH ref={len(ref)} ferric={len(got)}"); sys.exit(3)
md=max(abs(a-b) for a,b in zip(ref,got)); dot=sum(a*b for a,b in zip(ref,got))
print(f"  {sys.argv[1]:<36} cosine {dot:.8f}   max|diff| {md:.3e}")
open("/tmp/.mb_md","a").write(f"{sys.argv[1]}\t{md}\n")
PY
}
rm -f /tmp/.mb_md
run_one "as implemented"                 ""                      || exit 1
run_one "control: window disabled"       "FERRIC_MB_NO_SWA=1"    || exit 1
run_one "control: one rope base"         "FERRIC_MB_ONE_ROPE=1"  || exit 1

python3 - <<'PY'
import sys
rows=dict(l.split('\t') for l in open('/tmp/.mb_md').read().strip().split('\n'))
base=float(rows['as implemented'])
ok=True
for k,mult in (('control: window disabled',20.0),('control: one rope base',20.0)):
    r=float(rows[k])
    if r < base*mult:
        print(f"⛔ {k}: {r:.3e} is not >= {mult:g}x the implemented {base:.3e} — that mechanism is NOT "
              f"load-bearing here, so the conformance number proves nothing about it")
        ok=False
print("✅ both mechanisms are load-bearing; the match is about the hard parts" if ok else "")
sys.exit(0 if ok else 1)
PY
