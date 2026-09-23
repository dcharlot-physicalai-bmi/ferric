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

# ── the CLASSIFIER head, if this checkpoint has one ──────────────────────────
# ⛔⛔ THE EMBEDDING COSINE DOES NOT COVER THIS, AND THAT IS THE POINT. On the pair below the
# encoder's CLS row matches the reference at cosine 0.99999964 — and max|diff| 3.4e-3 in ABSOLUTE
# terms, because cosine is scale-invariant. The head is not: a 768-term pooler, a LayerNorm and a
# projection amplify that absolute error about 10x, to ~0.03-0.10 on scores spanning [-2.2, +2.4].
# A gate that stopped at the embedding cosine would report this path as verified to 1e-5.
#
# ⛔ And the pair must be built the way the reference builds it: `cls_sep` ("\t") splits query from
# document, then it concatenates query + EOS_TEXT + document and tokenizes THAT with specials
# (examples/embedding/embedding.cpp:187-208). Tokenizing the tab itself puts a TAB TOKEN where the
# separator belongs and moved a score from -2.16 to +0.24 here — a sign flip that looks like a
# broken head.
HAS_HEAD="$(python3 - "$M" <<'PYH'
import struct,sys
def u32(f): return struct.unpack('<I', f.read(4))[0]
def u64(f): return struct.unpack('<Q', f.read(8))[0]
def rstr(f): return f.read(u64(f)).decode('utf-8','replace')
def val(f,t):
    if t in (0,1): return struct.unpack('<b' if t==0 else '<B', f.read(1))[0]
    if t in (2,3): return struct.unpack('<h' if t==2 else '<H', f.read(2))[0]
    if t in (4,5): return struct.unpack('<i' if t==4 else '<I', f.read(4))[0]
    if t==6: return struct.unpack('<f', f.read(4))[0]
    if t==7: return struct.unpack('<?', f.read(1))[0]
    if t==8: return rstr(f)
    if t==9:
        et=u32(f); n=u64(f); return [val(f,et) for _ in range(n)]
    if t in (10,11): return struct.unpack('<q' if t==10 else '<Q', f.read(8))[0]
    if t==12: return struct.unpack('<d', f.read(8))[0]
f=open(sys.argv[1],'rb'); f.read(4); u32(f); nt=u64(f); nkv=u64(f)
for _ in range(nkv): rstr(f); t=u32(f); val(f,t)
names=[]
for _ in range(nt):
    n=rstr(f); nd=u32(f); [u64(f) for _ in range(nd)]; u32(f); u64(f); names.append(n)
print("yes" if "cls.output.weight" in names else "no")
PYH
)"
if [ "$HAS_HEAD" = "yes" ]; then
  echo
  echo "  classifier head (rerank):"
  SEPTXT="$(python3 - "$M" <<'PY2'
import struct,sys
def u32(f): return struct.unpack('<I', f.read(4))[0]
def u64(f): return struct.unpack('<Q', f.read(8))[0]
def rstr(f): return f.read(u64(f)).decode('utf-8','replace')
def val(f,t):
    if t in (0,1): return struct.unpack('<b' if t==0 else '<B', f.read(1))[0]
    if t in (2,3): return struct.unpack('<h' if t==2 else '<H', f.read(2))[0]
    if t in (4,5): return struct.unpack('<i' if t==4 else '<I', f.read(4))[0]
    if t==6: return struct.unpack('<f', f.read(4))[0]
    if t==7: return struct.unpack('<?', f.read(1))[0]
    if t==8: return rstr(f)
    if t==9:
        et=u32(f); n=u64(f); return [val(f,et) for _ in range(n)]
    if t in (10,11): return struct.unpack('<q' if t==10 else '<Q', f.read(8))[0]
    if t==12: return struct.unpack('<d', f.read(8))[0]
f=open(sys.argv[1],'rb'); f.read(4); u32(f); u64(f); nkv=u64(f)
toks=None; sep=None
for _ in range(nkv):
    k=rstr(f); t=u32(f); v=val(f,t)
    if k=="tokenizer.ggml.tokens": toks=v
    if k.endswith("seperator_token_id") or k.endswith("eos_token_id"):
        if sep is None: sep=v
print(toks[sep])
PY2
)"
  RANK_OK=1
  score_pair () {   # $1 query  $2 doc  $3 expect-sign (pos|neg)
    printf '%s%s%s' "$1" "$SEPTXT" "$2" > "$TMP/pp.txt"
    printf '%s\t%s' "$1" "$2" > "$TMP/pp_tab.txt"
    local ids ref fer
    ids="$($LT -m "$M" -f "$TMP/pp.txt" --ids 2>/dev/null | tail -1 | tr -d '[] ')"
    ref="$($LE -m "$M" -f "$TMP/pp_tab.txt" --pooling rank -ngl 0 --embd-normalize -1 2>/dev/null \
          | grep -i 'rerank score' | awk '{print $NF}')"
    fer="$("$BIN" "$M" "$ids" 2>/dev/null | grep '^RANK' | awk '{print $2}')"
    python3 - "$3" "$ref" "$fer" <<'PY3'
import sys
want,ref,fer=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
d=abs(ref-fer)
ok_sign = (ref>0)==(fer>0)
ok_mag  = d <= 0.15
flag = "" if (ok_sign and ok_mag) else "   <-- FAIL"
print(f"    {want:<10} ref {ref:+8.3f}   ferric {fer:+8.3f}   |diff| {d:.3f}{flag}")
sys.exit(0 if (ok_sign and ok_mag) else 1)
PY3
  }
  score_pair "what is panda?" "The giant panda is a bear species endemic to China." "relevant"   || RANK_OK=0
  score_pair "what is panda?" "The Eiffel Tower is a wrought-iron lattice tower in Paris." "irrelevant" || RANK_OK=0
  [ "$RANK_OK" = "1" ] && echo "    ✅ sign and magnitude agree with the reference (tol 0.15 absolute)" \
                       || { echo "    ⛔ classifier head disagrees with the reference"; exit 1; }
fi

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
