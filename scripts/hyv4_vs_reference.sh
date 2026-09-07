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
# ⛔ THREE PATCHES, NOT TWO. AngelSlim's 0001-hyv4-architecture and 0002-stq1_0-quant-and-cuda make
# the reference *run* hyv4; docs/hyv4/0003-eval-callback-sum-abs-and-argmax.patch makes it *report
# what this gate compares*. Without the third, eval-callback prints neither `sum_abs` nor `argmax`
# and every prompt here fails to parse. That patch lives in this repo because `.reference/` is
# gitignored: an earlier commit said it had added sum_abs "to Ferric's dump and to the reference's"
# and committed only Ferric's half, so the reference side existed on exactly one machine. A gate
# whose oracle cannot be rebuilt is a gate that runs once.
#
#   cd .reference/llama.cpp && git apply ../../docs/hyv4/0003-eval-callback-sum-abs-and-argmax.patch
#   cmake --build build --target llama-eval-callback -j
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

# ⭐ ASK THE ORACLE WHETHER IT CAN ANSWER, BEFORE READING SEVEN ANSWERS FROM IT. A reference built
# without patch 0003 prints no `sum_abs` and no `argmax`; the per-prompt parse then yields empty
# strings, every row reads FAIL, and nothing on screen says why. Refusing here, once, with the
# command that fixes it, is the difference between a diagnosable gate and a mystery.
probe=$("$EC" -m "$GGUF" -p "a" -n 1 -c 64 --no-escape 2>/dev/null | /usr/bin/grep -A 12 "result_output = ")
for field in sum_abs argmax; do
  printf '%s\n' "$probe" | /usr/bin/grep -qE "^[[:space:]]*$field = " || {
    echo "⛔ the reference at $EC does not report '$field'."
    echo "   This gate compares it. Apply the third patch and rebuild:"
    echo "     cd .reference/llama.cpp && git apply ../../docs/hyv4/0003-eval-callback-sum-abs-and-argmax.patch"
    echo "     cmake --build build --target llama-eval-callback -j"
    exit 2
  }
done

# ⛔ THE GATE IS ON `sum_abs`, NOT `sum`. A sum CANCELS, and a cancelled quantity flatters a
# comparison: on the real 770B checkpoint the per-block `l_out` sums agreed to 8.4e-6 while the
# values behind them were ~1e-3 apart per element, because the errors happened to offset. `sum_abs`
# cannot cancel, so it says whether the VALUES agree rather than whether their errors lined up.
# `sum` is still printed, because a large sum delta under a small sum_abs delta is its own signal
# (the magnitudes match but the signs moved), but it is not what fails the build.
#
# ⚠ BOTH SIDES ARE PARSED BY FIELD NAME, never by line offset. This gate previously read
# `grep -A 6 ... | tail -1`, which happened to land on `sum`; adding one line to the reference's
# own dump would have silently repointed it at a different number with nothing failing.
# Measured ladder on the seven prompts below, Apple M3 Max [Metal] vs NEON CPU:
#   3.9e-8  2.3e-7  6.2e-6  6.5e-6  1.4e-5  1.5e-5  5.4e-5
# TOL is ~4x the worst of those. The old gate was 5e-3 ABSOLUTE on the cancelling `sum`, which is
# ~25x looser than this and on a quantity that could not see a per-element error at all. Adapters
# do differ (two Metal adapters disagree on the golden hash), so the headroom is deliberate —
# widen it only with a new measured ladder, never to make a red build go green.
TOL_REL=2e-4
#
# ⭐ AND THE GREEDY PICK MUST MATCH EXACTLY. sum_abs says the magnitudes agree; it does not say the
# two implementations would emit the same token, and the token is what anyone actually sees. Greedy
# decoding is an argmax over the whole 40-wide (120832-wide on the real model) logit row, and the
# winner is rarely among the six values eval-callback prints — so this was invisible until both
# sides were taught to report it (`common_debug_print_argmax` in the patched reference).
# This is a DECISION test, not a magnitude test, and it is exact: no tolerance.
fail=0
printf '%-18s %13s %13s %11s %8s %8s\n' prompt ref_sum_abs fer_sum_abs rel_delta ref_pick fer_pick
for P in dlh a mzq abcde k9x zzz q7; do
  ids=$(python3 -c "print(','.join(str('$ALPHA'.index(c)) for c in '$P'))") || { echo "bad prompt $P"; exit 2; }

  rblock=$("$EC" -m "$GGUF" -p "$P" -n 1 -c 64 --no-escape 2>/dev/null | /usr/bin/grep -A 12 "result_output = ")
  ref_sum=$(printf '%s\n' "$rblock" | awk '/^[[:space:]]*sum = /     {print $3; exit}')
  ref_abs=$(printf '%s\n' "$rblock" | awk '/^[[:space:]]*sum_abs = / {print $3; exit}')
  ref_arg=$(printf '%s\n' "$rblock" | awk '/^[[:space:]]*argmax = /  {print $3; exit}')

  fblock=$(cargo run --release -q -p ferric-llama --example hyv4_synthetic -- --logits "$ids" 2>/dev/null)
  fer_sum=$(printf '%s\n' "$fblock" | awk '/^sum = /     {print $3; exit}')
  fer_abs=$(printf '%s\n' "$fblock" | awk '/^sum_abs = / {print $3; exit}')
  fer_arg=$(printf '%s\n' "$fblock" | awk '/^argmax = /  {print $3; exit}')

  read -r rel sd ok < <(python3 -c "
ra,fa,rs,fs='$ref_abs','$fer_abs','$ref_sum','$fer_sum'
try:
    ra,fa,rs,fs=float(ra),float(fa),float(rs),float(fs)
    if ra == 0: print('nan','nan','FAIL')
    else:
        rel=abs(fa-ra)/abs(ra); print(f'{rel:.3e}', f'{abs(fs-rs):.6f}', 'ok' if rel < $TOL_REL else 'FAIL')
except Exception: print('nan','nan','FAIL')")
  # An unparsed argmax is a FAILURE, not a skip: an empty variable must never compare equal.
  pick=FAIL
  [ -n "$ref_arg" ] && [ -n "$fer_arg" ] && [ "$ref_arg" = "$fer_arg" ] && pick=ok
  note=""
  [ "$ok"   = ok ] || note="  <-- sum_abs FAIL"
  [ "$pick" = ok ] && [ "$ok" = ok ] || true
  [ "$pick" = ok ] || note="$note  <-- PICK DIFFERS"
  printf '%-18s %13s %13s %11s %8s %8s%s\n' "$P ($ids)" "${ref_abs:-?}" "${fer_abs:-?}" "$rel" \
         "${ref_arg:-?}" "${fer_arg:-?}" "$note"
  [ "$ok" = ok ] && [ "$pick" = ok ] || fail=$((fail+1))
done
echo
if [ "$fail" -eq 0 ]; then
  echo "✓ Ferric agrees with Tencent's reference on every prompt, on the same file —"
  echo "  both in magnitude (sum_abs) and in the GREEDY TOKEN each would emit."
  echo "  This is fidelity evidence, not self-consistency. What it does NOT cover: the REAL"
  echo "  weights, sequences past a few tokens, 256-way routing, and the sparse DSA path at a"
  echo "  top_k that actually selects — the synthetic checkpoint admits everything."
  echo "  On the REAL checkpoint the same comparison sits at ~1e-3 per block and is NOT covered"
  echo "  here; see VERIFICATION.md §3d for what that number is and is not."
else
  echo "⛔ $fail prompt(s) diverge from the reference."
fi
exit $((fail > 0))
