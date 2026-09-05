#!/usr/bin/env bash
# Run every example CI runs, with the storage-binding limit clamped to a CONSTRAINED FABRIC.
#
# WHY THIS EXISTS. `Context::new` requests `adapter.limits()`. That is right for native — big models
# need big buffers — and it means NOTHING on a developer's Metal machine fails when a code path
# outgrows the 134,217,728-byte (128 MiB) `max_storage_buffer_binding_size` that lavapipe AND the
# WebGPU baseline enforce. "Run everywhere, browser first" makes that the target which must pass
# FIRST, and it was the one target nobody could see. `examples/flash.rs` sized a 288,000,000-byte
# buffer, ran green on every machine anyone here owns, and panicked the first time CI executed it on
# a software rasterizer.
#
# Fixing that class one CI round-trip at a time costs ~35 minutes per iteration and only ever finds
# the NEXT one. This finds them all in one local pass.
#
#   scripts/constrained_fabric.sh              # lavapipe's / the WebGPU baseline's 128 MiB
#   scripts/constrained_fabric.sh 67108864     # tighter still
#   scripts/constrained_fabric.sh --selftest   # prove the checker can FAIL
set -uo pipefail
cd "$(dirname "$0")/.."

CAP="${1:-134217728}"
SELFTEST=0
[ "$CAP" = "--selftest" ] && { SELFTEST=1; CAP=134217728; }

# ⛔ The list is EXTRACTED FROM ci.yml, never hardcoded. A checker whose discovery step cannot see a
# newly-added example reads exactly like a clean codebase — and the Metal job runs a DIFFERENT,
# smaller list, so "it passed CI" does not mean an example was covered.
# `mapfile` is bash 4+; macOS ships 3.2, so read into the array the portable way.
CMDS=()
while IFS= read -r l; do CMDS+=("$l"); done < <(
    sed -n '/Validate — general tensor runtime/,/^  macos-metal:/p' .github/workflows/ci.yml \
    | grep -oE "cargo run -p [a-z-]+ --example [a-z0-9_]+" | sort -u)
if [ "${#CMDS[@]}" -lt 10 ]; then
    echo "REFUSING: found only ${#CMDS[@]} example commands in ci.yml — the extraction pattern has drifted." >&2
    exit 2
fi

TO=$(command -v gtimeout || command -v timeout || true)
echo "Constrained-fabric sweep · max_storage_buffer_binding_size = $CAP ($((CAP >> 20)) MiB)"
echo "${#CMDS[@]} example commands, taken from .github/workflows/ci.yml"
echo

# ⛔ Build ONLY the examples ci.yml names. `--examples` builds every example in each crate, which
# for ferric-tensor is far more than CI runs and dominated this script's runtime (17 minutes, almost
# all of it here). Per-crate `--example X` flags keep it to the set actually under test.
echo "building..."
for c in $(printf '%s\n' "${CMDS[@]}" | sed -E 's/.* -p ([a-z-]+) .*/\1/' | sort -u); do
    flags=""
    for line in "${CMDS[@]}"; do
        case "$line" in *" -p $c "*) flags="$flags --example $(sed -E 's/.*--example ([a-z0-9_]+).*/\1/' <<<"$line")";; esac
    done
    # shellcheck disable=SC2086
    cargo build -q -p "$c" $flags 2>&1 | grep -E '^error' | head -5
done

fails=0
for line in "${CMDS[@]}"; do
    ex=$(sed -E 's/.*--example ([a-z0-9_]+).*/\1/' <<<"$line")
    # ⏱ Name each example BEFORE running it. Without this the script printed a header and then
    # nothing for seventeen minutes, so "still working" and "hung" looked identical and telling them
    # apart needed `ps`. A tool built to diagnose exactly that must not have it.
    printf '  %-26s ' "$ex"
    bin="target/debug/examples/$ex"
    [ -x "$bin" ] || { echo "NOT BUILT"; fails=$((fails + 1)); continue; }
    if [ "$SELFTEST" = 1 ] && [ "$ex" = "flash" ]; then
        # Damage the run deliberately: a 1 MiB cap cannot fit even one head's scores, and flash.rs
        # is written to REFUSE rather than proceed there. A selftest that has never been seen to go
        # red is the same evidence as no selftest at all.
        out=$(FERRIC_MAX_BINDING=1048576 "$TO" 300 "./$bin" 2>&1); rc=$?
    else
        out=$(FERRIC_MAX_BINDING="$CAP" "$TO" 300 "./$bin" 2>&1); rc=$?
    fi
    if [ $rc -eq 0 ]; then echo "ok"; continue; fi
    fails=$((fails + 1))
    # ⚠ Of the four reporting branches below, --selftest exercises only the last (generic exit
    # code). BINDING LIMIT, TIMEOUT and ABORT are unexercised — and after the fixes in 102cf2e and
    # 3b049d6 no example can currently reach them, which is the point of those fixes. They are here
    # to name a REGRESSION precisely, so treat them as untested formatting, not as tested logic.
    if grep -q 'max_\*_buffer_binding_size' <<<"$out"; then
        printf 'BINDING LIMIT: %s\n' "$(grep -oE 'range [0-9]+ exceeds .* limit [0-9]+' <<<"$out" | head -1)"
    elif [ $rc -eq 124 ]; then echo "TIMEOUT (300s)"
    elif [ $rc -eq 134 ]; then echo "ABORT (exit 134) — a panic in a Drop aborts AFTER correct output"
    else printf 'exit %s: %s\n' "$rc" "$(grep -m1 'panicked at' <<<"$out" | sed 's/.*panicked at //' | cut -c1-90)"
    fi
done

echo
if [ "$SELFTEST" = 1 ]; then
    if [ "$fails" -ge 1 ]; then echo "SELFTEST PASSED: the sweep went red on a deliberately impossible cap."; exit 0
    else echo "SELFTEST FAILED: a 1 MiB cap produced no failure — this checker cannot detect anything."; exit 1; fi
fi
echo "$fails failing at $((CAP >> 20)) MiB"
[ "$fails" -eq 0 ]
