#!/bin/bash
# Run $@ ONLY on a quiet machine. ⛔ On timeout it REFUSES -- it does not measure anyway.
# A gate that falls through to the measurement is not a gate; that flaw once produced a
# matmul_q_bench run whose "batched" column came out SLOWER than unbatched.
# ⚠ AND A PASSING GATE IS STILL NOT A REPRODUCIBILITY CHECK: `uptime` sees competing processes,
# not the GPU clock state. Identical work at loads 2.17-3.91 has spanned 3.4x. Read ratios, repeat.
ONE=${QUIET_ONE:-4.0}; FIVE=${QUIET_FIVE:-6.0}; TRIES=${QUIET_TRIES:-60}

# ⛔ THE CHECK THE LOAD AVERAGE CANNOT MAKE. This machine is shared with another agent that runs
# Ferric's own GPU examples. Two of those pinning a core each shows up as load ~3.3 — under the
# gate — while they contend for the GPU my benchmark is trying to measure. The load average counts
# RUNNABLE PROCESSES; it knows nothing about who holds the device.
# Any other `target/release/examples/*` process is therefore a hard refusal, regardless of load.
# ⚠ Match the EXECUTABLE, not the command line. `pgrep -f` on the bare path also matches any shell
# whose script happens to mention it — including another agent's wait-loop that is merely QUEUED to
# run a benchmark later, and including this script's own child. Anchor on the path as the argv[0].
others() {
  ps -Ao pid=,args= 2>/dev/null \
    | awk -v me=$$ '$1 != me { $1=""; sub(/^ /,""); if ($0 ~ /^[^ ]*target\/release\/examples\//) print }'
}

for w in $(seq 1 $TRIES); do
  read one five fifteen <<< "$(uptime | sed 's/.*averages*: //' | tr -d ',')"
  gpu=$(others | wc -l | tr -d ' ')
  if [ "$(python3 -c "print(1 if $one<$ONE and $five<$FIVE else 0)")" = 1 ] && [ "$gpu" = 0 ]; then
    echo "✅ GATE PASSED: load $one $five $fifteen, no other GPU example running"
    exec "$@"
  fi
  [ "$gpu" != 0 ] && [ $((w % 8)) = 1 ] && echo "   waiting: $gpu other GPU example(s) running — $(others | head -2 | sed 's/^[0-9]* //' | tr '\n' ' ')"
  sleep 15
done
echo "⛔ GATE REFUSED after $((TRIES*15))s: load $(uptime | sed 's/.*averages*: //'), $(others | wc -l | tr -d ' ') other GPU example(s)"
echo "⛔ NO MEASUREMENT TAKEN. A contended microbenchmark is a WRONG number, not a slow one."
exit 3
