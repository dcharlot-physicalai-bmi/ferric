#!/usr/bin/env bash
# Text -> ids, Ferric vs llama.cpp, over EVERY local checkpoint llama.cpp can load.
#
# ⛔ WHY THIS EXISTS. Ferric's verification is logit-level: qwen3vl_hf, hyv4, the vision seams — all
# of them take TOKEN IDS as given and compare what the model does with them. Nothing compared TEXT to
# IDS. That is upstream of every logit, so a tokenizer defect is invisible to the entire suite, and
# three shipped ones were found by hand in one afternoon:
#   - gemma4/t5 routed to byte-level BPE in the server (0/4 prompts matched the browser)
#   - qwen35 falling open to GPT-2's pre-tokenizer (12/20 against llama.cpp)
#   - and the `is_spm` rule maintained separately in two front ends
#
# Needs `llama-tokenize` on PATH. Checkpoints llama.cpp cannot read (Ferric's own Q2_0/STQ1_0 ternary
# formats) are reported as SKIP with the reason, never counted as passes.
#
#   scripts/tokenizer_conformance.sh [corpus.txt]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${1:-$ROOT/crates/ferric-llama/tests/fixtures/qwen35_pretok_corpus.txt}"
LT="$(command -v llama-tokenize || true)"
[ -z "$LT" ] && { echo "llama-tokenize not on PATH — this gate needs the reference implementation"; exit 2; }
[ -f "$CORPUS" ] || { echo "no corpus at $CORPUS"; exit 2; }

BIN="$ROOT/target/release/examples/pretokenizer_conformance"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example pretokenizer_conformance || exit 2

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
N_OK=0; N_PART=0; N_SKIP=0; N_FALL=0; N_NA=0
printf "%-46s %-10s %-8s %s\n" "checkpoint" "pre" "ferric" "note"
printf "%-46s %-10s %-8s %s\n" "----------" "---" "------" "----"
while IFS= read -r f; do
  base="$(basename "$f")"
  # reference ids, one line per corpus string
  : > "$TMP/ref.txt"; loadable=1
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    ids="$("$LT" -m "$f" -p "$line" --no-bos --ids 2>/dev/null | tail -1)"
    case "$ids" in \[*\]) printf '%s|%s\n' "$line" "$ids" >> "$TMP/ref.txt" ;; *) loadable=0; break ;; esac
  done < "$CORPUS"
  if [ "$loadable" -eq 0 ] || [ ! -s "$TMP/ref.txt" ]; then
    printf "%-46s %-10s %-8s %s\n" "$base" "-" "SKIP" "llama.cpp cannot load it (Ferric-only format?)"
    N_SKIP=$((N_SKIP+1)); continue
  fi
  out="$("$BIN" "$f" "$TMP/ref.txt" 2>/dev/null)"
  pre="$(printf '%s' "$out" | sed -n 's/.*pre=\"\([^\"]*\)\".*/\1/p' | head -1)"
  status="$(printf '%s' "$out" | /usr/bin/grep -oE 'IMPLEMENTED|FALL-OPEN' | head -1)"
  score="$(printf '%s' "$out" | /usr/bin/grep '(current)' | /usr/bin/grep -oE '[0-9]+/[0-9]+' | head -1)"
  hit="${score%%/*}"; tot="${score##*/}"
  if [ -z "$score" ]; then
    # no BPE variant is "current" -> this model tokenizes through Spm/WordPiece, which this probe
    # does not model. Report it as UNEVALUATED, never as a pass and never as a divergence.
    printf "%-46s %-10s %-12s %s\n" "$base" "${pre:--}" "n/a" "not BPE — this gate does not cover Spm/WordPiece"
    N_NA=$((N_NA+1)); continue
  fi
  if [ "$status" = "FALL-OPEN" ]; then
    printf "%-46s %-10s %-12s %s\n" "$base" "${pre:--}" "$score" "⛔ FALL-OPEN: no rule for this pre; score is GPT-2's"
    N_FALL=$((N_FALL+1))
  elif [ "$hit" = "$tot" ]; then
    printf "%-46s %-10s %-12s %s\n" "$base" "${pre:--}" "$score" "exact"
    N_OK=$((N_OK+1))
  else
    printf "%-46s %-10s %-12s %s\n" "$base" "${pre:--}" "$score" "implemented but divergent"
    N_PART=$((N_PART+1))
  fi
done < <(find ~/.cache/ferric ~/.cache/huggingface -iname "*.gguf" 2>/dev/null | sort)

echo
echo "exact: $N_OK   implemented-but-divergent: $N_PART   FALL-OPEN (no rule): $N_FALL"
echo "not-BPE (this gate does not cover them): $N_NA   unloadable by the reference: $N_SKIP"
[ $((N_PART + N_FALL)) -gt 0 ] && exit 1
exit 0
