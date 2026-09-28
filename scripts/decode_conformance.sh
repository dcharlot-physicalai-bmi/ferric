#!/usr/bin/env bash
# **Small-row decode and device token selection** — the two contracts the decode path now rests on, each
# shown able to fail.
#
#  1. Small-M GEMV (crates/ferric-tensor/src/dtype/mrgemv.rs): every row of a 2-32 row matmul equals the
#     ONE-ROW decode kernel's result bit for bit (5 formats, plain + fused gate|up, forced tiles), and sits
#     within f32 noise of an independent f64 dequant-matmul of the raw GGUF bytes, where each plausible
#     wrong decoding misses by >= 20x. Whole model: batched decode LOGITS equal solo decode's, to the bit,
#     at n = 2..16; the can-fail arm (FERRIC_MR=0, the old kernels) must differ.
#  2. Device token selection (crates/ferric-tensor/src/sample.rs + crates/ferric-serve/src/gpu_sample.rs):
#     for the same logits and RNG state, the token AND the RNG state after it equal `genopts::sample`'s on
#     the full row — synthetic LM-shaped rows, adversarial rows (ties, ±0, -inf, NaN), and a real model's
#     rows (one-row and verify-shaped), every setting (greedy, the pinned T=0.8/top_p .95 default, top-k,
#     min-p, penalties) and several seeds; plus the reduced row's Σp equal to the full row's to the bit.
#
#   scripts/decode_conformance.sh [qwen2.5-0.5b-instruct-q8_0.gguf]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
M="${1:-$HOME/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf}"
[ -f "$M" ] || { echo "usage: $0 <qwen2.5-0.5b q8_0 .gguf>"; exit 2; }
fail=0
echo "== small-M kernels vs one-row decode and vs the f64 format definition"
cargo test -q --release -p ferric-tensor --lib small_m 2>&1 | grep -E "test result|panicked|FAILED" ; [ ${PIPESTATUS[0]} -eq 0 ] || fail=1
echo "== batched decode logits == solo decode, n = 2..16"
cargo build -q --release -p ferric-llama --example batched_decode || exit 2
if ./target/release/examples/batched_decode "$M" 2>&1 | grep -E "n=|panicked"; then :; fi
./target/release/examples/batched_decode "$M" >/dev/null 2>&1 || { echo "FAIL: batched decode not bit-identical"; fail=1; }
echo "== can-fail: the old kernels (FERRIC_MR=0) must NOT be bit-identical"
if FERRIC_MR=0 BATCHED_FORCE_BITEXACT=1 ./target/release/examples/batched_decode "$M" >/dev/null 2>&1; then
  echo "FAIL: the bit check passed without the small-M kernels — it cannot see the difference"; fail=1
else echo "ok: without small-M the batched logits differ, as they must"; fi
echo "== device token selection == genopts::sample (synthetic + real rows)"
FERRIC_SAMPLE_MODEL="$M" cargo test -q --release -p ferric-serve --lib gpu_sample -- --include-ignored --nocapture 2>&1 \
  | grep -E "picks identical|test result|panicked|FAILED"; [ ${PIPESTATUS[0]} -eq 0 ] || fail=1
[ $fail -eq 0 ] && echo "ALL OK" || { echo "FAILED"; exit 1; }
