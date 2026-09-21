#!/usr/bin/env bash
# Find assertions that live in a test module, read like a test, and CANNOT FAIL because nothing
# ever runs them.
#
# ⛔ WHY THIS EXISTS. `crates/ferric-tokenizer/src/lib.rs` shipped
# `marks_join_the_letter_run_under_qwen35_and_not_under_qwen2` with no `#[test]` attribute. It was
# the ONE assertion proving that the real \p{M} table changed user-visible tokenization — the whole
# point of the commit that landed it. Its own doc comment said the previous version had asserted the
# WRONG behaviour deliberately, so that landing \p{M} would break it and the break would be the
# instruction to rewrite it. That happened: the tripwire fired, the body was rewritten, and the
# `#[test]` line was dropped in the same edit. The tripwire was disarmed by the act of responding
# to it. `cargo test --list` showed two tests in that module where the source reads like three.
#
# ⛔⛔ rustc ALREADY REPORTED IT — `warning: function ... is never used` — and the workspace gate
# grepped only `^error`. A compiler warning nobody greps is not a gate. This is that grep.
#
# What it will NOT flag, all verified by --self-test:
#   - methods inside `impl ... { }` (mock/fixture trait impls: Meter, Problem, BuildHasher, ...)
#   - helpers that something actually calls, anywhere in the same crate
#   - fns carrying #[test] / #[tokio::test] / #[bench] / #[rstest] / #[proptest] / #[quickcheck]
#
#   scripts/disarmed_tests.sh              # gate: exits 1 if any are found
#   scripts/disarmed_tests.sh --self-test  # prove the detector can fail, and does not over-fire
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# ⚠ /usr/bin/python3 on this mac is the CLT stub; prefer homebrew locally, plain python3 in CI.
PY="$(command -v python3 || true)"
[ -x /opt/homebrew/bin/python3 ] && PY=/opt/homebrew/bin/python3
[ -z "$PY" ] && { echo "no python3 on PATH"; exit 2; }
exec "$PY" "$ROOT/scripts/disarmed_tests.py" "$ROOT" "${1:-}"
