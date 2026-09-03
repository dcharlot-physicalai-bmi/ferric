#!/usr/bin/env bash
# Ferric's formal proofs — the one entry point.
#
# These are bounded model checks (Kani/CBMC), not tests: they hold for ALL inputs in range rather
# than for the values someone chose. They exist because every defect that cost real time in the
# low-bit formats was a same-count/wrong-order bug — a stride, a nibble, a bit order — which changes
# no length, no element count and no distribution, so no assert fires.
#
# WHAT THEY DO NOT COVER, so a green run is not read as more than it is:
#   * the WGSL kernels. Kani verifies Rust; shaders are strings handed to a driver.
#   * that Ferric's decode matches the publisher's. That is empirical, and it is settled by
#     examples/stq1_0_interop.rs against real Hy4 weights, not by a proof.
#   * floating point. Energy figures are measurements; the least-squares search is a heuristic.
#   * the f16 conversion, which is STUBBED in the layout proofs (half reaches runtime CPU-feature
#     detection that Kani cannot encode). Those theorems are about where bytes go.
#
# `-Z stubbing` is required by the layout proofs and is not optional; running `cargo kani` without
# it fails to compile rather than silently verifying less.
set -euo pipefail
cd "$(dirname "$0")/.."
FLAGS=(-Z stubbing)
# A harness that does not converge must FAIL, loudly, not hang a CI runner or a laptop for an
# hour. Twenty minutes is far above anything here (the whole set runs in seconds); a harness that
# needs it has a symbolic allocation in it and should be rewritten, not waited on.
LIMIT=${PROOF_TIMEOUT_SECS:-1200}
fail=0
for c in ferric-gguf ferric-llama; do
    if ! grep -rq "kani::proof" "crates/$c/src" 2>/dev/null; then continue; fi
    echo "=== $c ==="
    out=$(timeout "$LIMIT" cargo kani -p "$c" "${FLAGS[@]}" 2>&1); rc=$?
    echo "$out" | grep -E "Checking harness|VERIFICATION|Complete -"
    # The verdict is read off the summary line, NOT off cargo-kani's exit code and NOT off whether
    # grep found anything. The first version of this gate piped through grep, which matched the
    # summary line whether it said "0 failures" or "2 failures", so a run with failing harnesses
    # exited 0 and CI would have gone green -- with a comment beside it claiming the opposite.
    if [ $rc -eq 124 ]; then echo "TIMED OUT after ${LIMIT}s: $c"; fail=1
    elif ! echo "$out" | grep -qE "^Complete - [0-9]+ successfully verified harnesses, 0 failures,"; then
        echo "PROOF FAILURE in $c"; fail=1
    fi
done
exit $fail
