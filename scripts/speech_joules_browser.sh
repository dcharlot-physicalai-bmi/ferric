#!/bin/bash
# **Joules per transcription IN A BROWSER** — the same metric as speech_joules.sh, so the two are
# directly comparable. Answers the question the "$0 AI in a tab" thesis depends on: does browser
# delivery cost meaningfully more energy than native?
#
# ⚠ Chrome's own processes are INSIDE the measurement, deliberately. They are the real cost of
# browser delivery — excluding them would measure a runtime nobody actually ships.
#
# ⚠ The page loads the model once and warms up before the measured loop, mirroring the native
# harness. Charging each utterance for a 1.18 GB fetch would repeat the error that made the first
# native figure ~3x too high.
set -u
FERRIC=/Users/dcharlot/vibe-coding/ferric
ITERS="${1:-10}"
MAXLOAD="${MAXLOAD:-3.0}"

LOAD1=$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')
if [ "$(python3 -c "print(1 if float('$LOAD1') > float('$MAXLOAD') else 0)")" = "1" ]; then
  echo "REFUSING: load average $LOAD1 (limit $MAXLOAD) — no idle to baseline against." >&2
  exit 2
fi
echo "load average $LOAD1 — ok"

watts() {
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

echo "=== idle baseline (8 s, no browser running) ==="
macmon pipe -i 250 -s 32 2>/dev/null > /tmp/jb_idle.json
read IDLE_W IDLE_N < <(watts /tmp/jb_idle.json)
echo "idle ${IDLE_W} W over ${IDLE_N} samples"

echo "=== ${ITERS} transcriptions in Chrome (model loaded once) ==="
macmon pipe -i 250 -s 0 2>/dev/null > /tmp/jb_load.json &
MACMON=$!
# ⚠ BOUND THE RUN. puppeteer's `browser.close()` has hung here twice AFTER the bench printed its
# result, leaving macmon sampling for over an hour. A `pkill` placed after this line cannot rescue a
# hang — it never executes. The timeout is the guard; the pkill afterwards is only cleanup.
# The energy figure is safe either way now (integration is bounded to the reported work window), but
# an unbounded run wedges the harness.
BENCH_TIMEOUT="${BENCH_TIMEOUT:-300}"
( cd "$FERRIC/crates/ferric-web" && \
  perl -e 'alarm shift; exec @ARGV' "$BENCH_TIMEOUT" node speech_bench_test.mjs "$ITERS" \
) > /tmp/jb_bench.txt 2>/tmp/jb_bench.err
kill $MACMON 2>/dev/null; wait $MACMON 2>/dev/null
pkill -f "user-data-dir=/tmp/ferric-bench" 2>/dev/null || true

python3 - "$IDLE_W" /tmp/jb_bench.txt /tmp/jb_load.json <<'PY'
import sys, json, datetime
idle = float(sys.argv[1])
line = next((l for l in open(sys.argv[2]) if l.startswith('BENCH_JSON ')), None)
if not line:
    raise SystemExit("no BENCH_JSON line — the browser run failed; see /tmp/jb_bench.err")
r = json.loads(line.split(' ', 1)[1])
if 'error' in r:
    raise SystemExit(f"browser run errored: {r['error']}")
ms, secs = r['ms'], r['audio_s']
enc_s = sum(ms) / 1000.0
n = len(ms)

# Integrate power over the WORK WINDOW the page reported, not over the sampler's lifetime.
t0, t1 = r['t_start'] / 1000.0, r['t_end'] / 1000.0
tot = cnt = 0.0
for ln in open(sys.argv[3]):
    ln = ln.strip()
    if not ln: continue
    try: d = json.loads(ln)
    except Exception: continue
    ts = datetime.datetime.fromisoformat(d['timestamp']).timestamp()
    if not (t0 <= ts <= t1): continue
    p = sum(v for k in ('cpu_power','gpu_power','ane_power','ram_power')
            if isinstance(v := d.get(k), (int, float)))
    if p > 0: tot += p; cnt += 1
if cnt < 4:
    raise SystemExit(f"INVALID: only {cnt:.0f} power samples inside the {t1-t0:.1f} s work window — "
                     f"too few to integrate. Increase iterations or the sample rate.")
load = tot / cnt
print(f"power sampled over the work window only: {cnt:.0f} samples in {t1-t0:.1f} s")
print(f"\nmodel load {r['load_s']:.1f} s (excluded) | {n} transcriptions summing {enc_s:.1f} s"
      f" | audio {secs:.2f} s each | min {min(ms):.0f} ms = {secs*1000/min(ms):.1f}x realtime")
print(f"package power: idle {idle:.2f} W -> load {load:.2f} W  (marginal {load-idle:.2f} W)")
for label, w in (("total   ", load), ("marginal", load - idle)):
    j = w * enc_s / n
    print(f"{label} {j:7.2f} J per transcription   ({j/max(secs,1e-9):.2f} J per second of audio)")
if load - idle <= 0:
    raise SystemExit(
        f"\nINVALID: marginal power {load-idle:+.2f} W. A workload cannot reduce package draw —\n"
        f"the idle baseline was contaminated. Re-run on a quiet machine.")
PY
