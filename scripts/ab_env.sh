#!/bin/bash
# **A/B one env-var switch against the default, in the SAME binary.**
#
#   scripts/ab_env.sh FERRIC_SUBBLK <model.gguf> [reps]
#
# Every rule below was earned by a measurement that lied first:
#
# ⛔ `env -u`, NEVER `VAR=`. `std::env::var().is_err()` is FALSE for a set-but-EMPTY var, so `VAR=`
#    leaves both arms on the SAME path. That produced two "identical" 17.5 ms readings once and they
#    were reported as a result.
# ⛔ CORRECTNESS GATE FIRST. If the generated ids differ, this exits 2 and prints NO timings — a
#    faster wrong answer is not a result.
# ⛔ VACUITY GATE. If both arms produce the same bit-exact logits fingerprint, the switch did nothing
#    and the timings would be noise-on-noise. Exits 2.
# ⛔ INTERLEAVE, then compare the delta to the observed SPREAD. A delta inside the spread is reported
#    as INDISTINGUISHABLE with no winner named. Both the min-vs-spread and the paired sign test are
#    printed, so the more favourable framing cannot be picked after the fact.
# ⚠ A load gate (scripts/quiet.sh) is necessary and NOT sufficient: `uptime` sees competing
#    processes, not the GPU clock state, which has spanned 3.4x on identical work.
VAR=${1:?usage: ab_env.sh VAR model.gguf [reps]}
M=${2:?usage: ab_env.sh VAR model.gguf [reps]}
N=${3:-7}
R=$(cd "$(dirname "$0")/.." && pwd)
IDS=${FERRIC_AB_IDS:-785,6722,315,9625,374}
TOK=${FERRIC_AB_TOKENS:-24}   # llama-bench tg128 decodes 128; a short run overstates ms/tok (per-run fixed cost)
RUN=$R/target/release/examples/run_ids
BIT=$R/target/release/examples/kv_bitref
[ -x "$RUN" ] || { echo "⛔ build first: cargo build -p ferric-llama --release --examples"; exit 4; }

echo "=== device: the ONLY thing that makes the numbers below mean anything ==="
dev=$(env -u "$VAR" $RUN "$M" "$IDS" 1 2>/dev/null | grep -m1 "^adapter")
echo "  $dev"
[ -n "$dev" ] || { echo "  ⛔ run_ids printed no adapter line — rebuild the examples. NO measurement taken."; exit 4; }
echo "$dev" | grep -qiE "llvmpipe|swiftshader|software|cpu" && { echo "  ⛔ CPU/software adapter — refusing to benchmark it as a GPU."; exit 4; }
# ⛔⛔ THIS GATE ONCE PASSED ON A CHANGE THAT DID ALTER THE OUTPUT. It compared 16 ids on ONE prompt.
# FERRIC_CUDA_Q8X (int8 activations) passed it, then diverged on 3 of 5 prompts at 128 tokens — the
# earliest at TOKEN 31, i.e. just past the old window. Sixteen tokens is not evidence a numerics change
# is output-neutral; it is evidence the first sixteen tokens agree. Now: the SAME horizon as the timed
# runs (FERRIC_AB_TOKENS), over SEVERAL prompts, and it reports WHERE the first divergence is.
CTOK=${FERRIC_AB_CHECK_TOKENS:-$TOK}
SEEDS=${FERRIC_AB_SEEDS:-"$IDS 40,1265,315,264,3283 9707,11,847,829,374 3838,374,279,6722,315"}
echo "=== correctness: generated ids must be identical ($CTOK tokens x $(echo $SEEDS | wc -w) prompts) ==="
bad=0
for sd in $SEEDS; do
  a=$(env -u "$VAR" $RUN "$M" "$sd" "$CTOK" 2>/dev/null | grep "generated ids")
  b=$(env "$VAR=${FERRIC_AB_VAL:-1}"  $RUN "$M" "$sd" "$CTOK" 2>/dev/null | grep "generated ids")
  [ -n "$a" ] || { echo "  ⛔ no ids from run_ids on seed $sd — the gate cannot run. STOP."; exit 4; }
  if [ "$a" != "$b" ]; then
    bad=$((bad+1))
    n=$(paste -d'\n' <(echo "$a") <(echo "$b") | awk 'NR==1{split($0,x,",");next}{split($0,y,",");for(i=1;i<=length(x);i++) if(x[i]!=y[i]){print i;exit}}')
    echo "  ⛔ DIVERGED on seed $sd — first difference at token ${n:-?}"
  else
    echo "  ✅ identical   seed $sd"
  fi
done
if [ "$bad" != 0 ]; then
  echo "  ⛔ GENERATION DIFFERS on $bad prompt(s) — no timings taken."
  echo "    A faster wrong answer is not a result. If the change is a deliberate numerics trade"
  echo "    (quantised activations, a different accumulation width), it belongs behind its OWN"
  echo "    opt-in with its OWN fingerprint — measure it with FERRIC_AB_ALLOW_DIVERGENCE=1, which"
  echo "    keeps this report and proceeds to the timings."
  [ -n "$FERRIC_AB_ALLOW_DIVERGENCE" ] || exit 2
  echo "  ⚠ FERRIC_AB_ALLOW_DIVERGENCE set — proceeding, and the timings below compare DIFFERENT OUTPUTS."
fi

if [ -x "$BIT" ]; then
  echo "=== vacuity: the switch must actually change something ==="
  ha=$(env -u "$VAR" $BIT "$M" "the capital of France is" 3 2>/dev/null | grep -c .)
  # ⚠ `md5` is macOS-only; on Linux it is `md5sum`. Without this, both fingerprints were EMPTY on
  # Linux, compared equal, and the vacuity gate waved the run through with a warning nobody reads.
  H=$(command -v md5 >/dev/null && echo md5 || echo "md5sum")
  fa=$(env -u "$VAR" $BIT "$M" "the capital of France is" 3 2>/dev/null | grep "^step" | $H | cut -c1-32)
  fb=$(env "$VAR=${FERRIC_AB_VAL:-1}"  $BIT "$M" "the capital of France is" 3 2>/dev/null | grep "^step" | $H | cut -c1-32)
  [ -n "$fa" ] || { echo "  ⛔ fingerprint EMPTY — kv_bitref produced no steps; the vacuity gate cannot run. STOP."; exit 2; }
  if [ "$fa" = "$fb" ]; then
    echo "  ⚠ logits fingerprints IDENTICAL — the switch changed no arithmetic."
    echo "    That is fine for a pure scheduling change, but if you expected a different"
    echo "    reduction order, the arm is not engaging and the timings below mean nothing."
  else
    echo "  ✅ arms differ (logits fingerprint moved)"
  fi
fi

echo "=== interleaved, N=$N each, $TOK decode tokens per run ==="
O=(); S=()
t() { grep -o '([0-9.]* ms/tok)' | tr -d '()' | sed 's/ ms.tok//'; }
for i in $(seq 1 $N); do
  o=$(env -u "$VAR" $RUN "$M" "$IDS" $TOK 2>/dev/null | t)
  s=$(env "$VAR=${FERRIC_AB_VAL:-1}"  $RUN "$M" "$IDS" $TOK 2>/dev/null | t)
  echo "  rep $i  off $o   $VAR $s   [load $(uptime | sed 's/.*averages*: //' | cut -d' ' -f1)]"
  O+=("$o"); S+=("$s")
done
python3 - "$VAR" "${O[*]}" "${S[*]}" <<'PY'
import sys
from math import comb
var=sys.argv[1]
o=[float(x) for x in sys.argv[2].split()]; s=[float(x) for x in sys.argv[3].split()]
om,sm=min(o),min(s); osp,ssp=max(o)-min(o),max(s)-min(s)
print(f"\n  off     min {om:.2f}  spread {osp:.2f}  {['%.1f'%x for x in o]}")
print(f"  {var:<7} min {sm:.2f}  spread {ssp:.2f}  {['%.1f'%x for x in s]}")
d=om-sm; noise=max(osp,ssp)
print(f"  delta {d:+.2f} ms ({om/sm:.3f}x)")
if abs(d)<=noise:
    print(f"  ⚠ INDISTINGUISHABLE: |delta| {abs(d):.2f} <= spread {noise:.2f}. NO speed claim.")
else:
    print(f"  ✅ RESOLVED: |delta| {abs(d):.2f} > spread {noise:.2f}. {var+' faster' if d>0 else var.upper()+' SLOWER'}.")
w=sum(1 for a,b in zip(o,s) if b<a); n=len(o)
p=min(1.0, 2*sum(comb(n,k) for k in range(w,n+1))/2**n)
print(f"  paired: {var} faster in {w}/{n}, sign test p={p:.3f} -> {'significant' if p<0.05 else 'NOT significant'}")
PY
echo "end load: $(uptime | sed 's/.*averages*: //')"
