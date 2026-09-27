#!/usr/bin/env bash
# **Joules per correctly transcribed word** — one ASR model over a prepared corpus, measured with
# `macmon` (power sampled every 250 ms) around `asr_corpus`, which loads the model and warms it OUTSIDE
# the measurement, then runs the corpus in chunks separated by idle gaps (`IDLE`/`RUN` lines).
#
#   scripts/asr_joules.sh <corpus-dir> parakeet <model.gguf>
#   scripts/asr_joules.sh <corpus-dir> mimo <asr-dir> <tokenizer-dir>
#
# ⭐ EACH CHUNK IS CHARGED AGAINST THE IDLE GAPS ON EITHER SIDE OF IT. This machine is shared with other
# sessions that start GPU and CPU work at will: a first version took one idle baseline before the run
# and measured the GPU at 40.7 W "idle" against 33.4 W working — a NEGATIVE marginal, because someone
# else's job had been running during the baseline. A load average cannot see that (it counts CPU run
# queues, and the GPU was the contended part). So: the two gaps around a chunk must agree with each other
# to within 25% of the work measured between them, or the run is REFUSED — never averaged away.
#
# Four rails, each marginal and each with its own verdict:
#   GPU     `gpu_power` — where Ferric's arithmetic runs.
#   CPU     `cpu_power` — host-side work (the RVQ search, the frontends, dispatch).
#   RAM     `ram_power` — DRAM.
#   SYSTEM  `sys_power` — everything the machine drew, including rails with no name.
#   ⚠ On this Mac `macmon` has reported `cpu_power`/`ram_power` as exactly 0 in some runs and not in
#   others; a rail that reads 0 throughout is reported as unseen, never as free.
# Raw samples and the runner's output are kept in $ASR_JOULES_KEEP when it is set, so a verdict can be
# re-derived without re-running.
#
# ⚠ WHAT THE GUARDS CAN AND CANNOT SEE. (1) Drift: the two idle gaps around a chunk must agree within 25%
# of the work between them. (2) A burst inside one chunk: every chunk's joules per second of audio must lie
# within 2x of the median chunk's. That comparison only means something when the chunks are ALIKE, so the
# run is refused unless every chunk's mean clip length is within 2x of the median chunk's: a corpus in
# length order (the ESB parquet shards are sorted longest-first) made one chunk 0.5x the median on clip
# length alone. (3) The meter itself: `macmon` has returned a single sample of 18,779 W CPU / 5,314 W RAM /
# 18,788 W SYSTEM on a machine that cannot draw 1 kW, and rails that read exactly 0 in every sample but
# that one — so a sample above $ASR_JOULES_CEILING_W (default 1000) refuses its chunk, and a rail that
# reads 0 in most samples WHILE THE MODEL WORKS has dropped out (the GPU reads 0 between clips and at
# idle; that is real). The first MiMo run's SYSTEM rail had three chunks at 4-7x the rest, blamed at the
# time on other sessions — a glitch sample would look the same, and those samples were not kept. Load that
# is spread evenly over every run window and absent from every gap cannot be told apart from the model.
#
# ⭐ TWO DENOMINATORS. "Correct words" = reference words minus edits, under LibriSpeech's own normalisation
# AND, when $ASR_NORMALIZER_PY (a python with transformers) and $ASR_NORMALIZER_JSON (Whisper's
# normalizer.json) are set, under the Whisper English normaliser the Open ASR Leaderboard uses — see
# scripts/asr_wer_normalized.py for why a model that writes "Mr." needs the second.
# ⚠ A DERIVED figure (sampled watts, integrated), not an energy counter read once.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${1:?usage: asr_joules.sh <corpus-dir> parakeet <gguf> | mimo <asr-dir> <tokenizer-dir>}"; shift
BIN="$ROOT/target/release/examples/asr_corpus"
cargo build -q -p ferric-llama --release --example asr_corpus || exit 2
# The load average is a coarse pre-check only (it cannot see GPU contention at all); the attributable
# guard is the idle-gap agreement below. 40 refuses a machine that is plainly saturated.
MAXLOAD="${MAXLOAD:-40.0}"
load1() { uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/'; }
L0=$(load1)
if [ "$(python3 -c "print(int(float('$L0') > float('$MAXLOAD')))")" = "1" ]; then
  echo "REFUSING: 1-minute load average is $L0 (limit $MAXLOAD) — there is no idle to baseline against." >&2
  exit 3
fi
TMP="$(mktemp -d)"; trap '[ -n "${ASR_JOULES_KEEP:-}" ] && cp "$TMP"/* "$ASR_JOULES_KEEP"/ 2>/dev/null; rm -rf "$TMP"' EXIT
macmon pipe -i 250 -s 0 2>/dev/null > "$TMP/run.json" &
MM=$!
sleep 1
"$BIN" "$@" "$CORPUS" > "$TMP/out.txt" 2> "$TMP/err.txt"; RC=$?
kill $MM 2>/dev/null; wait $MM 2>/dev/null
[ $RC -eq 0 ] || { tail -20 "$TMP/err.txt"; exit 1; }
ASR_WER_SCRIPT="$ROOT/scripts/asr_wer_normalized.py" ASR_CORPUS="$CORPUS" python3 - "$TMP" "$L0" "$*" <<'PY'
import json, os, sys, datetime
tmp, l0, args = sys.argv[1], float(sys.argv[2]), sys.argv[3]
CEIL = float(os.environ.get("ASR_JOULES_CEILING_W", "1000"))
samples = []
for line in open(f"{tmp}/run.json"):
    try: d = json.loads(line)
    except Exception: continue
    samples.append((datetime.datetime.fromisoformat(d["timestamp"]).timestamp(), d.get("gpu_power", 0.0), d.get("sys_power", 0.0),
                    d.get("cpu_power", 0.0), d.get("ram_power", 0.0)))
o = open(f"{tmp}/out.txt").read().splitlines()
idles = [tuple(map(float, l.split()[1:3])) for l in o if l.startswith("IDLE ")]
runs = [tuple(map(float, l.split()[1:3])) for l in o if l.startswith("RUN ")]
run_audio = [float(l.split()[3]) for l in o if l.startswith("RUN ")]
run_utts = [int(l.split()[4]) for l in o if l.startswith("RUN ")]
summ = dict(zip(*[iter(next(l for l in o if l.startswith("SUMMARY ")).split()[1:])] * 2))
if len(idles) != len(runs) + 1:
    print(f"⛔ {len(idles)} idle gaps for {len(runs)} chunks"); sys.exit(1)
clip = [s / n for s, n in zip(run_audio, run_utts)]
mc = sorted(clip)[len(clip) // 2]
if any(not (0.5 * mc <= c <= 2.0 * mc) for c in clip):
    print("⛔ the chunks are not alike — mean clip length per chunk: " + ", ".join(f"{c:.2f} s" for c in clip)
          + ". The corpus is in some order (length?); the per-chunk guard compares chunks, so shuffle refs.txt.")
    sys.exit(1)
def vals(a, b, k, settle=0.0):
    return [s[k] for s in samples if a + settle <= s[0] <= b]
def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
correct, words, audio = int(summ["correct"]), int(summ["words"]), float(summ["audio_s"])
run_s = sum(b - a for a, b in runs)
print(f"model      {args}")
print(f"corpus     {summ['utterances']} utterances, {audio:.0f} s of audio, {words} words, {summ['edits']} edits "
      f"(WER {100 * int(summ['edits']) / words:.2f}%), {correct} correct words (reference words minus edits)")
print(f"work       {run_s:.1f} s in {len(runs)} chunks ({audio / run_s:.2f}x realtime); load average before {l0:.2f}")
verdict, totals = {}, {}
for name, k in [("GPU", 1), ("CPU", 3), ("RAM", 4), ("SYSTEM", 2)]:
    if all(s[k] == 0 for s in samples):
        print(f"{name:7s}    ⚠ reads 0 throughout this run — unseen by macmon here, not free"); continue
    total, rows, ok, per_s = 0.0, [], True, []
    for i, (a, b) in enumerate(runs):
        xr = vals(a, b, k)
        x0, x1 = vals(*idles[i], k, settle=2.0), vals(*idles[i + 1], k, settle=2.0)   # 2 s for the rails to fall back
        p_run, p_i0, p_i1 = mean(xr), mean(x0), mean(x1)
        base = (p_i0 + p_i1) / 2
        j = (p_run - base) * (b - a)
        why = []
        if min(len(xr), len(x0), len(x1)) < 8:
            why.append("fewer than 8 samples in a window")
        else:
            top = max(xr + x0 + x1)
            if top > CEIL:
                why.append(f"one sample reads {top:,.0f} W, above the {CEIL:.0f} W ceiling — a meter glitch, not power")
            if 2 * sum(v == 0 for v in xr) > len(xr):
                why.append("reads exactly 0 W in most samples while the model works — the rail dropped out")
            if not p_run > base:
                why.append("the work does not exceed idle")
            elif abs(p_i0 - p_i1) > 0.25 * (p_run - base):
                why.append("the gaps disagree by more than 25% of the work")
        ok &= not why
        total += j
        per_s.append(j / run_audio[i])
        rows.append(f"      chunk {i}: {b - a:6.1f} s  work {p_run:6.2f} W  idle {p_i0:6.2f} / {p_i1:6.2f} W  -> {j:8.1f} J"
                    + "".join(f"   ⛔ {w}" for w in why))
    med = sorted(per_s)[len(per_s) // 2]
    if not med > 0:
        ok = False
        rows.append(f"      ⛔ the median chunk has no positive marginal — nothing to compare the chunks against")
    else:
        for i, v in enumerate(per_s):
            if not (0.5 * med <= v <= 2.0 * med):
                ok = False
                rows[i] += f"   ⛔ {v:.2f} J per audio second, {v / med:.1f}x the median chunk"
    verdict[name] = ok
    totals[name] = total
    print(f"{name:7s}    marginal {total:9.1f} J   {total / correct * 1e3:8.2f} mJ per correct word   "
          f"{total / audio:6.3f} J per audio second   {'✅ attributable' if ok else '⛔ NOT attributable'}")
    print("\n".join(rows))
# Each rail stands or falls on its own chunks: the GPU is often clean while the meter's CPU rail drops out
# or other sessions load the system. A rail that fails is printed with its reasons and must not be
# quoted; the run fails only if NO rail is attributable.
for name, good in verdict.items():
    if not good:
        print(f"⛔ {name}: refused on at least one chunk for the reasons above — do not quote this rail from this run")
import subprocess
npy, njs = os.environ.get("ASR_NORMALIZER_PY"), os.environ.get("ASR_NORMALIZER_JSON")
if npy and njs:
    r = subprocess.run([npy, os.environ["ASR_WER_SCRIPT"], njs, os.environ["ASR_CORPUS"] + "/refs.txt", f"{tmp}/out.txt"],
                       capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("NORMALIZED ")), None)
    if line is None:
        print(f"⛔ normalised scoring failed: {r.stderr[-400:]}"); sys.exit(1)
    kv = dict(zip(*[iter(line.split()[1:])] * 2))
    print(f"normalised (Whisper English, as the Open ASR Leaderboard): WER {kv['wer']}%, {kv['correct']} correct words")
    for name, good in verdict.items():
        if good and name in totals:
            print(f"{name:7s}    {totals[name] / int(kv['correct']) * 1e3:8.2f} mJ per correct word (normalised)")
sys.exit(0 if any(verdict.values()) else 4)
PY
