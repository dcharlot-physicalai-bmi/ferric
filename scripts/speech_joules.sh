#!/bin/bash
# **Joules per transcription** — the standing metric, not tokens/sec.
#
# Samples Apple Silicon package power with `macmon` while a workload runs, integrates over wall
# time, and subtracts an idle baseline measured on the same machine moments earlier.
#
# ⚠ SUM cpu+gpu+ane+ram. `all_power` EXCLUDES DRAM, and on a speech encoder that is not a rounding
# error — the model is memory-bound.
#
# ⚠ IDLE BASELINE. Package power never reaches zero, so raw integrated power answers "what did this
# machine draw", not "what did this task cost". The marginal figure is the one that scales.
#
# ⚠ MEASURE THE STEADY STATE, NOT THE LOAD. `parakeet_transcribe` re-reads the whole checkpoint on
# every invocation; timing it charges each utterance for a 1.18 GB model load. `parakeet_bench`
# loads once and encodes N times, which is what a real deployment does — a first version of this
# script used the wrong binary and reported ~3x the true cost per utterance.
set -u
FERRIC=/Users/dcharlot/vibe-coding/ferric
MODEL="${1:?usage: speech_joules.sh <model.gguf> <audio.wav> [iters]}"
AUDIO="${2:?need audio.wav}"
ITERS="${3:-8}"

watts() {   # mean of (cpu+gpu+ane+ram) over the samples in $1 -> "<watts> <n>"
  python3 -c "
import json,sys
tot=n=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: continue
    p=sum(v for k in ('cpu_power','gpu_power','ane_power','ram_power')
          if isinstance(v:=d.get(k),(int,float)))
    if p>0: tot+=p; n+=1
print(f'{tot/n:.3f} {n}' if n else '0 0')
" "$1"
}

# ⚠ ABORT ON A BUSY MACHINE, BEFORE SPENDING THE RUN. Energy is unlike wall-clock here: contention
# inflates time but CORRUPTS the baseline, and the marginal figure is a small difference between two
# large contaminated numbers. Measured at load average 15.3 this harness reported idle 26.75 W
# against load 25.51 W — a NEGATIVE marginal, i.e. the workload apparently generating power.
MAXLOAD="${MAXLOAD:-3.0}"
LOAD1=$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')
if [ "$(python3 -c "print(1 if float('$LOAD1') > float('$MAXLOAD') else 0)")" = "1" ]; then
  echo "REFUSING: 1-minute load average is $LOAD1 (limit $MAXLOAD)." >&2
  echo "Energy needs a quiet machine — there is no idle to baseline against. Retry when idle," >&2
  echo "or set MAXLOAD=<n> to override and accept that the marginal figure is not attributable." >&2
  exit 2
fi
echo "load average $LOAD1 — ok"

# Idle: macmon exits on its own after -s samples, so run it in the FOREGROUND. Backgrounding it
# inside a function called via \$( ) makes it a child of a subshell that `wait` cannot reap.
echo "=== idle baseline (8 s) ==="
macmon pipe -i 250 -s 32 2>/dev/null > /tmp/j_idle.json
read IDLE_W IDLE_N < <(watts /tmp/j_idle.json)
echo "idle ${IDLE_W} W over ${IDLE_N} samples"

echo "=== ${ITERS} encodes (model loaded once) ==="
macmon pipe -i 250 -s 0 2>/dev/null > /tmp/j_load.json &
MACMON=$!
sleep 1
T0=$(python3 -c 'import time;print(time.time())')
"$FERRIC/target/release/examples/parakeet_bench" "$MODEL" "$AUDIO" "$ITERS" > /tmp/j_bench.txt 2>&1
T1=$(python3 -c 'import time;print(time.time())')
kill $MACMON 2>/dev/null; wait $MACMON 2>/dev/null
read LOAD_W LOAD_N < <(watts /tmp/j_load.json)
grep -E "encode min|realtime" /tmp/j_bench.txt || true

python3 - "$IDLE_W" /tmp/j_load.json "$T0" "$T1" "$ITERS" "$AUDIO" /tmp/j_bench.txt <<'PY'
import sys, struct, re, json, datetime
idle, t0, t1, iters = (float(sys.argv[1]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
d = open(sys.argv[6], 'rb').read()
i, rate, secs = 12, 0, 0
while i + 8 <= len(d):
    cid = d[i:i+4]; sz = struct.unpack('<I', d[i+4:i+8])[0]
    if cid == b'fmt ': rate = struct.unpack('<I', d[i+12:i+16])[0]
    if cid == b'data': secs = sz / 2 / max(rate, 1)
    i += 8 + sz + (sz & 1)
# The wall window includes model load; the bench reports per-encode ms, so charge energy only to
# the encodes themselves rather than to the one-time load.
txt = open(sys.argv[7]).read()
per = [float(m) for m in re.findall(r'iter \d+: (\d+) ms', txt)]
enc_s = sum(per) / 1000.0 if per else (t1 - t0)
wall = t1 - t0
# ⚠ Integrate power over the MEASURED LOOP the bench reports, not the process lifetime. The window
# otherwise includes model load — I/O-heavy and at a different power draw than the encodes the
# energy is attributed to. The browser twin of this harness hit the extreme form: a hung browser
# stretched the window to 72 minutes and produced a marginal of 0.08 J that the sign check passed.
ws = re.search(r'WINDOW_START ([\d.]+)', txt); we = re.search(r'WINDOW_END ([\d.]+)', txt)
w0, w1 = (float(ws.group(1)), float(we.group(1))) if ws and we else (t0, t1)
tot = cnt = 0.0
for ln in open(sys.argv[2]):
    ln = ln.strip()
    if not ln: continue
    try: d = json.loads(ln)
    except Exception: continue
    if not (w0 <= datetime.datetime.fromisoformat(d['timestamp']).timestamp() <= w1): continue
    p = sum(v for k in ('cpu_power','gpu_power','ane_power','ram_power')
            if isinstance(v := d.get(k), (int, float)))
    if p > 0: tot += p; cnt += 1
if cnt < 4:
    raise SystemExit(f"INVALID: only {cnt:.0f} power samples in the {w1-w0:.1f} s work window.")
load = tot / cnt
print(f"power over the work window only: {cnt:.0f} samples in {w1-w0:.1f} s")
print(f"\nwall {wall:.1f} s total | {len(per) or iters} encodes summing {enc_s:.1f} s | audio {secs:.2f} s each")
print(f"package power: idle {idle:.2f} W -> load {load:.2f} W  (marginal {load-idle:.2f} W)")
n = len(per) or iters
for label, w in (("total   ", load), ("marginal", load - idle)):
    j = w * enc_s / n
    print(f"{label} {j:7.2f} J per encode   ({j/max(secs,1e-9):.2f} J per second of audio)")

# ⚠ A physically impossible sign is a free validity check — assert it rather than leave it to be
# noticed. A workload cannot lower package power; a negative marginal means the baseline was taken
# against something that was not idle, and BOTH numbers then describe that other workload.
if load - idle <= 0:
    raise SystemExit(
        f"\nINVALID: marginal power is {load-idle:+.2f} W. A workload cannot reduce package draw —\n"
        f"the idle baseline ({idle:.2f} W) was contaminated. The totals above are real draw but are\n"
        f"NOT attributable to this workload. Re-run on a quiet machine.")
PY
