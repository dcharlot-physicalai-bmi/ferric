#!/usr/bin/env bash
# **Decode tokens/s and joules per token, PrismML Bonsai 2 — Ferric or the authors' fork, same protocol.**
# `macmon` (250 ms samples) runs around a runner that loads and warms OUTSIDE every window, then
# alternates IDLE gaps and RUN windows of k greedy decode tokens (full-logits read per step). Runners:
#   Ferric: target/release/examples/bonsai2_decode_joules  (crates/ferric-llama/examples/)
#   fork:   crates/ferric-llama/examples/refgen/bonsai2_decode_joules.cpp built against PrismML's llama.h
#
#   scripts/bonsai2_decode_joules.sh <model.gguf> [chunks 5] [k 128] [idle_s 4]
#   BONSAI2_RUNNER=/path/to/fork/bonsai2_decode_joules scripts/bonsai2_decode_joules.sh <model.gguf> ...
#
# Attribution is prefill_joules.sh's (read its header): each chunk is charged against the idle gaps on
# either side, a chunk whose gaps disagree by more than 25% of the work is REFUSED, a sample above
# $PREFILL_JOULES_CEILING_W (1000) refuses its chunk, and every chunk must lie within 2x of the median.
# ⚠ A DERIVED figure (sampled watts, integrated, differenced against idle) — not an energy counter.
# ⚠ On a shared machine the verdict line decides whether a number may be quoted; the tokens/s is always
# printed, the joules only with "attributable".
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: bonsai2_decode_joules.sh <model.gguf> [chunks] [k] [idle_s]}"
IDS="${BONSAI2_IDS:-760,6511,314,9338,369}"   # "The capital of France is" (the fork's tokenizer)
BIN="${BONSAI2_RUNNER:-$ROOT/target/release/examples/bonsai2_decode_joules}"
[ -n "${BONSAI2_RUNNER:-}" ] || cargo build -q -p ferric-llama --release --example bonsai2_decode_joules || exit 2
TMP="$(mktemp -d)"; trap '[ -n "${PREFILL_JOULES_KEEP:-}" ] && cp "$TMP"/* "$PREFILL_JOULES_KEEP"/ 2>/dev/null; rm -rf "$TMP"' EXIT
macmon pipe -i 250 -s 0 2>/dev/null > "$TMP/run.json" &
MM=$!
sleep 1
"$BIN" "$MODEL" "$IDS" "${2:-5}" "${3:-128}" "${4:-4}" ${BONSAI2_RUNNER_ARGS:-} > "$TMP/out.txt" 2> "$TMP/err.txt"; RC=$?
kill $MM 2>/dev/null; wait $MM 2>/dev/null
[ $RC -eq 0 ] || { tail -20 "$TMP/err.txt"; exit 1; }
grep '^HASH' "$TMP/err.txt" | awk '{print $2}' | sort -u | awk 'END { if (NR != 1) { print "⛔ chunks generated DIFFERENT tokens (" NR " hashes)"; exit 1 } }' || exit 1
grep '^HASH' "$TMP/err.txt" | head -1 | sed 's/^/  output /'
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
    print(f"{name:7s} marginal {total:8.1f} J over {sum(toks)} decode tokens = {total / sum(toks) * 1e3:7.3f} mJ/token"
          f"   (median chunk {med * 1e3:.3f})   {'✅ attributable' if ok else '⛔ NOT attributable — do not quote'}")
    print("\n".join(rows))
sys.exit(0 if ok_any else 4)
PY
