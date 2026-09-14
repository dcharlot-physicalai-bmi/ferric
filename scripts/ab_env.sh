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
RUN=$R/target/release/examples/run_ids
BIT=$R/target/release/examples/kv_bitref
[ -x "$RUN" ] || { echo "⛔ build first: cargo build -p ferric-llama --release --examples"; exit 4; }

echo "=== correctness: generated ids must be identical ==="
a=$(env -u "$VAR" $RUN "$M" "$IDS" 16 2>/dev/null | grep "generated ids")
b=$(env "$VAR=${FERRIC_AB_VAL:-1}"  $RUN "$M" "$IDS" 16 2>/dev/null | grep "generated ids")
if [ "$a" != "$b" ]; then
  echo "  ⛔ GENERATION DIFFERS — correctness regression, no timings taken."
  echo "    off: $a"; echo "    on : $b"; exit 2
fi
echo "  ✅ identical   $a"

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
    echo "  ✅ arms differ (logits fingerprint moved; generation did not)"
  fi
fi

echo "=== interleaved, N=$N each ==="
O=(); S=()
t() { grep -o '([0-9.]* ms/tok)' | tr -d '()' | sed 's/ ms.tok//'; }
for i in $(seq 1 $N); do
  o=$(env -u "$VAR" $RUN "$M" "$IDS" 24 2>/dev/null | t)
  s=$(env "$VAR=${FERRIC_AB_VAL:-1}"  $RUN "$M" "$IDS" 24 2>/dev/null | t)
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
