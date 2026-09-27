#!/usr/bin/env bash
# **The Metal-4 tensor-unit GEMM route (`FERRIC_QGEMM=1`) against the MODEL AUTHORS** — does running a
# quantized model's prefill GEMMs on the matrix units (fp16 operands, fp32 accumulate) keep its logits
# inside the band the portable f32 path already has against the authors' own implementation?
#
#   scripts/qgemm_conformance.sh <quantized.gguf> <fixture.json>
#   e.g. scripts/qgemm_conformance.sh qwen2.5-0.5b-instruct-q8_0.gguf \
#            crates/ferric-llama/tests/fixtures/lm/qwen2.5-0.5b-long.json
#
# The fixture is the authors' `transformers` float32 run (refgen/lm_logits_ref.py, min_tokens 512), so
# the prefill has ~550 rows — far above the route's 32-row floor — and every recorded position is
# compared: 128 fixed vocabulary ids, the argmax, and the full row's sum of squares.
#
# A quantized file cannot verify the MATH (lm_conformance.sh refuses one, rightly): its distance to the
# authors is quantization loss plus, for a published file, whatever revision it was converted from. So
# this gate does not ask "is it close to the authors"; it asks "is the tensor-unit route no further from
# the authors than the portable route on the SAME file", and "is the route's own perturbation small
# against that distance". Band, measured for the commit that added this (full vocabulary, 126 positions):
#   Qwen2.5-0.5B, Q8_0 of the authors' own weights (refgen/requant_gguf.py): KL to HF float64
#     5.99e-4 portable, 6.07e-4 route; route vs portable 1.39e-6. HF's float32-vs-float64: 3.9e-11.
#   Qwen2.5-1.5B, the published Q4_K_M: 1.70e-2 both; route vs portable 1.06e-6.
#   Qwen3-0.6B, Q4_K/Q6_K of the authors' weights: 8.38e-2 both; route vs portable 5.3e-7.
# ⚠ Use a file quantized from the authors' weights where you can: the published Qwen2.5-0.5B GGUFs
# were converted from a different checkpoint revision (see requant_gguf.py), so their band is wide.
#
# ⛔ VACUITY GUARDS. A route that never fires returns the portable logits and passes every band check,
# so the run must show tensor-unit dispatches (FERRIC_TRACE_KERNELS) and logits that DIFFER from the
# portable run. ⭐ NEGATIVE CONTROL: FERRIC_QGEMM_FAULT plants one plausible decoding error per format
# (the neighbouring Q8_0 block's scale, the Q4_K sub-block minimum dropped, Q6_K high bits from the
# wrong position); it must move the model by ≥ 20x the band tolerance or this gate cannot see a defect.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"
[ -f "$M" ] && [ -f "$FX" ] || { echo "usage: $0 <quantized.gguf> <fixture.json>"; exit 2; }
BIN="$ROOT/target/release/examples/lm_logits"
cargo build -q -p ferric-llama --release --example lm_logits 2>/dev/null || { cargo build -p ferric-llama --release --example lm_logits; exit 2; }

python3 - "$BIN" "$M" "$FX" <<'PY'
import json, math, os, subprocess, sys
BIN, M, FX = sys.argv[1:4]
ref = json.load(open(FX))
POS = ref.get("positions") or list(range(len(ref["ids"])))

def run(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FERRIC_QGEMM")}
    env.update(extra); env["FERRIC_TRACE_KERNELS"] = "1"
    r = subprocess.run([BIN, M, FX], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-1500:]); sys.exit(1)
    fired = sum(1 for l in r.stderr.splitlines() if l.startswith("KERNEL\tqmm_"))
    rows = [l.split(" ") for l in r.stdout.splitlines() if l.startswith("ROW ")]
    if len(rows) != len(ref["ids"]):
        print(f"⛔ {len(rows)} rows for {len(ref['ids'])} tokens"); sys.exit(1)
    vals = {t: [float(x) for x in rows[t][5:]] for t in POS}
    worst = tot = n = am = 0; ssq = 0.0
    for t, rr in zip(POS, ref["rows"]):
        d = [abs(a - b) for a, b in zip(vals[t], rr["sample"])]
        worst = max(worst, max(d)); tot += sum(d); n += len(d)
        am += int(rows[t][2]) == rr["top"][0][0]
        ssq = max(ssq, abs(float(rows[t][4]) - rr["ssq"]) / rr["ssq"])
    return dict(max=worst, mean=tot / n, argmax=am, ssq=ssq, fired=fired, vals=vals)

P = run({})
N = run({"FERRIC_QGEMM": "1"})
F = run({"FERRIC_QGEMM": "1", "FERRIC_QGEMM_FAULT": "1"})
k = len(POS)
between = [abs(a - b) for t in POS for a, b in zip(N["vals"][t], P["vals"][t])]
nb_mean, nb_max = sum(between) / len(between), max(between)
print(f"reference: {ref['model']} — {ref.get('code', 'transformers')}, float32, {len(ref['ids'])} tokens, {k} positions x "
      f"{len(ref['sample_ids'])} sampled ids")
print(f"weights:   {os.path.basename(M)}")
for name, r in [("portable (f32 WGSL)", P), ("tensor units (FERRIC_QGEMM)", N), ("negative control (FAULT)", F)]:
    print(f"  {name:29s} max|dlogit| {r['max']:.4f}  mean {r['mean']:.5f}  argmax {r['argmax']}/{k}  ssq rel {r['ssq']:.2e}"
          f"  tensor-unit dispatches {r['fired']}")
print(f"  route vs portable, same file:  max|dlogit| {nb_max:.4f}  mean {nb_mean:.5f}  "
      f"({nb_mean / P['mean']:.1%} of the portable route's mean distance to the authors)")
TOL = 0.05   # the route may sit at most 5% further from the authors (mean) than the portable path
fails = []
# ⛔ A NaN compares False against every bound below, so non-finite logits would PASS them all — the
# first version of this gate did exactly that on its own negative control. Finite first, then bands.
for name, r in [("portable", P), ("tensor-unit", N)]:
    if not all(math.isfinite(r[k]) for k in ("max", "mean", "ssq")):
        fails.append(f"the {name} run produced non-finite logits")
if P["fired"]: fails.append("the portable run dispatched tensor-unit kernels — FERRIC_QGEMM leaked into it")
if not N["fired"]: fails.append("FERRIC_QGEMM=1 dispatched no tensor-unit kernel — the route never fired (no Metal-4 device?)")
if nb_max == 0: fails.append("route and portable logits are bit-identical — the route changed nothing")
if N["mean"] > (1 + TOL) * P["mean"]: fails.append(f"mean distance {N['mean']:.5f} > {1 + TOL:.2f} x portable {P['mean']:.5f}")
if N["max"] > 1.25 * P["max"]: fails.append(f"max distance {N['max']:.4f} > 1.25 x portable {P['max']:.4f}")
if N["argmax"] < P["argmax"] - 1: fails.append(f"argmax {N['argmax']} < portable {P['argmax']} - 1")
if nb_mean > 0.25 * P["mean"]: fails.append(f"the route's own perturbation {nb_mean:.5f} is > 25% of the quantization band")
# The control must land >= 20x the tolerance beyond the portable band: (F - P) >= 20 * TOL * P.
# Non-finite logits from the planted fault are a fault SEEN (the model is destroyed), not a NaN ratio.
ctl = (F["mean"] - P["mean"]) / (TOL * P["mean"]) if math.isfinite(F["mean"]) else math.inf
print(f"  negative control moved the mean distance by {ctl:.0f}x the tolerance (needs >= 20x)"
      + ("  (non-finite logits: the planted fault destroyed the model)" if math.isinf(ctl) else ""))
if not F["fired"]: fails.append("the negative control never reached the tensor units")
if not ctl >= 20: fails.append("negative control: a planted decoding error was not seen — this gate cannot fail")
for f in fails: print(f"⛔ {f}")
print("✅ PASS — the tensor-unit route stays inside the portable path's band against the authors" if not fails else "⛔ FAIL")
sys.exit(1 if fails else 0)
PY
