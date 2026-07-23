#!/usr/bin/env bash
# fabric-ci — the cross-fabric determinism gate.
#
# Runs the 9-row kernel probe on every reachable fabric and fails if any
# digest differs from the local (Metal) reference:
#   1. local native            (Metal on a Mac, Vulkan elsewhere)
#   2. Chrome / Dawn / Tint    (headless, WebGPU) — needs Chrome + node deps
#   3. remote native via ssh   (set FABRIC_CI_REMOTE=user@host, repo at ~/ferric)
#
# Usage:  scripts/fabric-ci.sh            # local + browser (+ remote if set)
#         FABRIC_CI_REMOTE=dcharlot@100.78.153.58 scripts/fabric-ci.sh
#
# Every row is an FNV-1a hash of raw output bits — see
# docs/determinism/WGPU-DETERMINISM.md for the method and its history.
set -euo pipefail
cd "$(dirname "$0")/.."

norm() { grep -E '^(mm|rmsnorm|sqrt|rope|mha|sigmoid|layernorm|softmax|rms-tree|rms-tcpu|ln-tree|ln-tcpu|sm-tree|sm-tcpu|rms-sg|rms-sgc|demo-lm) +[0-9a-f]{16}$' | sort; }

echo "── local native ──"
cargo run --release -p ferric-core --example fabric_probe 2>/dev/null | tee /tmp/fabric-local.txt | grep -E "^fabric"
LOCAL=$(norm < /tmp/fabric-local.txt)
[ -n "$LOCAL" ] || { echo "FAIL: local probe produced no rows"; exit 1; }
NROWS=$(echo "$LOCAL" | wc -l | tr -d ' ')

FAIL=0

if [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] && [ -d crates/ferric-web/node_modules ]; then
  echo "── browser (Dawn/Tint) ──"
  cargo build --release --target wasm32-unknown-unknown -p ferric-web >/dev/null 2>&1
  wasm-bindgen --target web --out-dir crates/ferric-web/pkg \
    target/wasm32-unknown-unknown/release/ferric_web.wasm
  ( cd crates/ferric-web && pkill -f "http.server 8799" 2>/dev/null || true
    python3 -m http.server 8799 >/dev/null 2>&1 & SRV=$!
    sleep 1
    node probe_test.mjs > /tmp/fabric-browser.txt 2>&1 || true
    kill $SRV 2>/dev/null || true )
  BROWSER=$(norm < /tmp/fabric-browser.txt)
  # capability-gated rows (e.g. subgroups) may be absent on a fabric: compare
  # the intersection, and NAME what was skipped — absence is honest, silence isn't.
  echo "$BROWSER" | awk '{print $1}' > /tmp/fabric-bnames.txt
  LSHARED=$(awk 'NR==FNR{keep[$1]=1;next} keep[$1]' /tmp/fabric-bnames.txt <(echo "$LOCAL"))
  SKIPPED=$(comm -23 <(echo "$LOCAL" | awk '{print $1}' | sort) <(sort /tmp/fabric-bnames.txt) | tr '\n' ' ')
  if [ "$BROWSER" = "$LSHARED" ]; then
    N=$(echo "$LSHARED" | wc -l | tr -d ' ')
    echo "browser: MATCH ($N/$N)${SKIPPED:+  [skipped, capability-gated: $SKIPPED]}"
  else
    echo "browser: MISMATCH"
    diff <(echo "$LSHARED") <(echo "$BROWSER") || true
    FAIL=1
  fi
else
  echo "── browser: skipped (Chrome or node deps not found) ──"
fi

if [ -n "${FABRIC_CI_REMOTE:-}" ]; then
  echo "── remote native ($FABRIC_CI_REMOTE) ──"
  rsync -az --exclude target --exclude .git crates/ferric-core/ "$FABRIC_CI_REMOTE":~/ferric/crates/ferric-core/
  rsync -az --exclude target --exclude .git forks/ "$FABRIC_CI_REMOTE":~/ferric/forks/
  ssh -o BatchMode=yes "$FABRIC_CI_REMOTE" \
    'find ~/ferric/crates/ferric-core ~/ferric/forks -name "*.rs" -exec touch {} + && cd ~/ferric && ~/.cargo/bin/cargo run --release -p ferric-core --example fabric_probe 2>/dev/null' \
    > /tmp/fabric-remote.txt || true
  grep -E "^fabric" /tmp/fabric-remote.txt || true
  REMOTE=$(norm < /tmp/fabric-remote.txt)
  if [ "$REMOTE" = "$LOCAL" ]; then
    echo "remote: MATCH ($NROWS/$NROWS)"
  else
    echo "remote: MISMATCH"
    diff <(echo "$LOCAL") <(echo "$REMOTE") || true
    FAIL=1
  fi
else
  echo "── remote: skipped (set FABRIC_CI_REMOTE=user@host) ──"
fi

if [ "$FAIL" = 0 ]; then
  echo "✅ fabric-ci: all probed fabrics bit-identical"
else
  echo "❌ fabric-ci: digest divergence — see diffs above"
fi
exit $FAIL
