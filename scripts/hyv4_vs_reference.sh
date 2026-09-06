#!/usr/bin/env bash
# Compare Ferric's hyv4 forward against AngelSlim's llama.cpp reference, ON THE SAME FILE.
#
# WHY THIS EXISTS. Every other hyv4 check in this repo is a SELF-comparison — decode against
# prefill, batched against solo, one quantisation against another. All of them are satisfied by a
# wrong-but-consistent implementation. This is the only check that is not: llama.cpp's `hyv4.cpp`
# is Tencent's own graph, written by other people from the same spec, and it reads the very file
# Ferric writes.
#
# The claim "no reference implementation builds on this machine" was repeated for a whole session
# and was simply untested: the patches apply cleanly at llama.cpp 0cea36222 and build CPU-only in
# a few minutes. Retest a blocker before quoting it.
#
#   scripts/hyv4_vs_reference.sh <path-to-llama.cpp-build>
set -uo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:?usage: hyv4_vs_reference.sh <llama.cpp build dir with the two hy4-preview patches applied>}"
EC="$BUILD/bin/llama-eval-callback"
[ -x "$EC" ] || { echo "no llama-eval-callback at $EC"; exit 2; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
GGUF="$TMP/tiny_hyv4.gguf"
cargo run --release -q -p ferric-llama --example hyv4_synthetic -- "$GGUF" >/dev/null || exit 1

# The alphabet the synthetic checkpoint's vocabulary uses, so a prompt maps to known ids.
ALPHA='abcdefghijklmnopqrstuvwxyz0123456789+-*/'
# ⚠ The gate is on the SUM of the last row's logits, because eval-callback prints only the first
# three and last three values of a row but a sum over all 40. It is sensitive to every element.
# Observed across the prompts below: 4.1e-5 .. 9.6e-4, which is f32 accumulation noise between a
# NEON CPU and a Metal GPU reducing in different orders. One observation set; widen only with a
# reason, and never to make a failure go away.
TOL=5e-3
fail=0
printf '%-18s %14s %14s %12s\n' prompt reference ferric delta
for P in dlh a mzq abcde k9x zzz q7; do
  ids=$(python3 -c "print(','.join(str('$ALPHA'.index(c)) for c in '$P'))") || { echo "bad prompt $P"; exit 2; }
  ref=$("$EC" -m "$GGUF" -p "$P" -n 1 -c 64 --no-escape 2>/dev/null \
        | /usr/bin/grep -A 6 "result_output = " | tail -1 | awk '{print $3}')
  fer=$(cargo run --release -q -p ferric-llama --example hyv4_synthetic -- --logits "$ids" 2>/dev/null \
        | tail -1 | awk '{print $3}')
  read -r d ok < <(python3 -c "
r,f='$ref','$fer'
try: d=abs(float(r)-float(f)); print(f'{d:.6f}', 'ok' if d < $TOL else 'FAIL')
except Exception: print('nan','FAIL')")
  printf '%-18s %14s %14s %12s %s\n' "$P ($ids)" "$ref" "$fer" "$d" "$([ "$ok" = ok ] || echo '  <-- FAIL')"
  [ "$ok" = ok ] || fail=$((fail+1))
done
echo
if [ "$fail" -eq 0 ]; then
  echo "✓ Ferric agrees with Tencent's reference on every prompt, on the same file."
  echo "  This is fidelity evidence, not self-consistency. What it does NOT cover: the REAL"
  echo "  weights, sequences past a few tokens, 256-way routing, and the sparse DSA path at a"
  echo "  top_k that actually selects — the synthetic checkpoint admits everything."
else
  echo "⛔ $fail prompt(s) diverge from the reference."
fi
exit $((fail > 0))
