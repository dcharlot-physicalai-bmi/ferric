#!/usr/bin/env bash
# perf-guards — the regression guards that live in EXAMPLES, run as a battery.
#
# ⛔ WHY THIS EXISTS. `examples/dispatch_budget.rs` carries an assertion on dispatches per layer.
# It was FAILING — 13.1 against its <=12.5 bound — and nobody knew, because an example is not a
# test: `cargo test --workspace` does not run examples and CI runs only `ebm_cert_verify`. A guard
# written precisely to catch a regression had been failing into an empty room. The cause turned out
# to be a disabled debug dump that still ran an RMSNorm per layer per token in production
# (qwen3.rs:982, fixed by `dump_with`).
#
# These guards need real checkpoints, which CI does not have — so this is a LOCAL battery, and the
# standing rule is that the local battery is a SUPERSET of the CI jobs, not a subset. Run it before
# pushing anything that touches a forward pass or a kernel dispatch.
#
#   scripts/perf_guards.sh
set -uo pipefail
cd "$(dirname "$0")/.."
CACHE="${FERRIC_CACHE:-$HOME/.cache/ferric}"
fail=0; ran=0; skipped=0

guard() {                      # guard <name> <model-path> <example> [args...]
  local name="$1" model="$2" ex="$3"; shift 3
  if [ ! -f "$model" ]; then
    printf '⏭  %-24s (model not present: %s)\n' "$name" "$(basename "$model")"
    skipped=$((skipped+1)); return
  fi
  # ⚠ Build FIRST, then run. `cargo run` in a timing/counting loop relinks while the thing being
  # measured runs, and a stale binary is worse: `cargo build --workspace` does NOT build examples,
  # so a library fix can look like it did nothing. Both traps were hit getting here.
  cargo build --release -q -p ferric-llama --example "$ex" || { echo "⛔ $name: build failed"; fail=$((fail+1)); return; }
  local out rc
  out=$("./target/release/examples/$ex" "$@" 2>&1); rc=$?
  ran=$((ran+1))
  if [ $rc -eq 0 ]; then
    printf '✅ %-24s %s\n' "$name" "$(printf '%s' "$out" | grep -oE 'decode \(per token\) +[0-9.]+ +[0-9.]+ +[0-9.]+' | tail -1)"
  else
    printf '⛔ %-24s FAILED (exit %d)\n' "$name" "$rc"
    printf '%s\n' "$out" | tail -4 | sed 's/^/     /'
    fail=$((fail+1))
  fi
}

echo "perf guards — regression bounds that live in examples"
echo
guard "dispatch budget" \
      "$CACHE/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf" \
      dispatch_budget

echo
if [ "$fail" -eq 0 ]; then
  echo "✓ $ran guard(s) passed, $skipped skipped for missing checkpoints."
  [ "$skipped" -gt 0 ] && echo "  ⚠ A skipped guard is not a passing guard. Fetch the checkpoint to run it."
else
  echo "⛔ $fail guard(s) FAILED."
fi
exit $((fail > 0))
