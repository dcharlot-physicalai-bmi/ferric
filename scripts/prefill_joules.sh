#!/usr/bin/env bash
# **Joules per prompt token of prefill** — one dense GGUF, one prompt length, measured with `macmon`
# (power sampled every 250 ms) around `prefill_joules`, which loads and warms the model OUTSIDE the
# measurement and then runs back-to-back prefills in chunks separated by idle gaps.
#
#   scripts/prefill_joules.sh <model.gguf> <prompt_tokens> [chunks 4] [run_s 4] [idle_s 4]
#   FERRIC_QGEMM=1 scripts/prefill_joules.sh ...      # the Metal-4 tensor-unit GEMM route
#
# The attribution is `asr_joules.sh`'s, for the same reason (read its header): this machine is shared,
# so EACH CHUNK IS CHARGED AGAINST THE IDLE GAPS ON EITHER SIDE OF IT, and a chunk whose two gaps
# disagree by more than 25% of the work measured between them is REFUSED, never averaged away. Every
# rail (GPU, CPU, RAM, SYSTEM) gets its own verdict; a rail that reads 0 throughout is reported unseen,
# a sample above $PREFILL_JOULES_CEILING_W (default 1000) refuses its chunk as a meter glitch, and every
# chunk's joules per token must lie within 2x of the median chunk's. Raw samples and the runner's
# output are kept in $PREFILL_JOULES_KEEP when it is set.
# ⚠ A DERIVED figure (sampled watts, integrated), not an energy counter read once.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: prefill_joules.sh <model.gguf> <prompt_tokens> [chunks] [run_s] [idle_s]}"
N="${2:?prompt_tokens}"
BIN="$ROOT/target/release/examples/prefill_joules"
cargo build -q -p ferric-llama --release --example prefill_joules || exit 2
TMP="$(mktemp -d)"; trap '[ -n "${PREFILL_JOULES_KEEP:-}" ] && cp "$TMP"/* "$PREFILL_JOULES_KEEP"/ 2>/dev/null; rm -rf "$TMP"' EXIT
macmon pipe -i 250 -s 0 2>/dev/null > "$TMP/run.json" &
MM=$!
sleep 1
"$BIN" "$MODEL" "$N" "${3:-4}" "${4:-4}" "${5:-4}" > "$TMP/out.txt" 2> "$TMP/err.txt"; RC=$?
kill $MM 2>/dev/null; wait $MM 2>/dev/null
[ $RC -eq 0 ] || { tail -20 "$TMP/err.txt"; exit 1; }
python3 - "$TMP" <<'PY'
import json, os, sys, datetime
tmp = sys.argv[1]
CEIL = float(os.environ.get("PREFILL_JOULES_CEILING_W", "1000"))
samples = []
for line in open(f"{tmp}/run.json"):
    try: d = json.loads(line)
    except Exception: continue
    samples.append((datetime.datetime.fromisoformat(d["timestamp"]).timestamp(), d.get("gpu_power", 0.0),
                    d.get("sys_power", 0.0), d.get("cpu_power", 0.0), d.get("ram_power", 0.0)))
o = open(f"{tmp}/out.txt").read().splitlines()
idles = [tuple(map(float, l.split()[1:3])) for l in o if l.startswith("IDLE ")]
runs = [tuple(map(float, l.split()[1:3])) for l in o if l.startswith("RUN ")]
toks = [int(l.split()[3]) for l in o if l.startswith("RUN ")]
print(open(f"{tmp}/err.txt").read().strip().splitlines()[-1])
print(next(l for l in o if l.startswith("SUMMARY ")))
if len(idles) != len(runs) + 1:
    print(f"⛔ {len(idles)} idle gaps for {len(runs)} chunks"); sys.exit(1)
def vals(a, b, k, settle=0.0): return [s[k] for s in samples if a + settle <= s[0] <= b]
def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
ok_any = False
for name, k in [("GPU", 1), ("CPU", 3), ("RAM", 4), ("SYSTEM", 2)]:
    if all(s[k] == 0 for s in samples):
        print(f"{name:7s} ⚠ reads 0 throughout this run — unseen by macmon here, not free"); continue
    total, per_tok, rows, ok = 0.0, [], [], True
    for i, (a, b) in enumerate(runs):
        xr = vals(a, b, k); x0, x1 = vals(*idles[i], k, settle=2.0), vals(*idles[i + 1], k, settle=2.0)
        p_run, p0, p1 = mean(xr), mean(x0), mean(x1); base = (p0 + p1) / 2; j = (p_run - base) * (b - a)
        why = []
        if min(len(xr), len(x0), len(x1)) < 8: why.append("fewer than 8 samples in a window")
        else:
            if max(xr + x0 + x1) > CEIL: why.append(f"a sample above the {CEIL:.0f} W ceiling — a meter glitch")
            if 2 * sum(v == 0 for v in xr) > len(xr): why.append("reads 0 W in most samples while working — rail dropped out")
            if not p_run > base: why.append("the work does not exceed idle")
            elif abs(p0 - p1) > 0.25 * (p_run - base): why.append("the gaps disagree by more than 25% of the work")
        ok &= not why; total += j; per_tok.append(j / toks[i])
        rows.append(f"      chunk {i}: {b - a:5.1f} s  work {p_run:6.2f} W  idle {p0:6.2f} / {p1:6.2f} W  -> {j:7.1f} J"
                    f"  {j / toks[i] * 1e3:7.3f} mJ/token" + "".join(f"   ⛔ {w}" for w in why))
    med = sorted(per_tok)[len(per_tok) // 2]
    if not med > 0: ok = False
    else:
        for i, v in enumerate(per_tok):
            if not (0.5 * med <= v <= 2.0 * med): ok = False; rows[i] += f"   ⛔ {v / med:.1f}x the median chunk"
    ok_any |= ok
    print(f"{name:7s} marginal {total:8.1f} J over {sum(toks)} prompt tokens = {total / sum(toks) * 1e3:7.3f} mJ/token"
          f"   (median chunk {med * 1e3:.3f})   {'✅ attributable' if ok else '⛔ NOT attributable — do not quote'}")
    print("\n".join(rows))
sys.exit(0 if ok_any else 4)
PY
