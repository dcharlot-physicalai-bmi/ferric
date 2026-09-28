#!/usr/bin/env bash
# **The NVIDIA native tier's rope angles at position 30,000 — against the portable path, which is exact there.**
#
#   scripts/cuda_rope_conformance.sh <model.gguf> <floor-fixture.json.gz>
#
# The floor fixtures (tests/fixtures/lm_floor/*-pos30000.json.gz, used by scripts/lm_floor_conformance.sh)
# carry the authors' ids run at positions P..P+T-1 (P = 30,000) with no cache. On F32 weights the WGSL path
# reproduces them within 1.6-2.3x of the authors' own float32-vs-float64 distance, because its rope table is
# built on the host the way the authors build theirs (qwen3.rs `rope_inv_freq` / `rope_cos_sin`). The native
# tier runs quantised weights only, so it cannot be put against the authors directly (a Q4_K_M file sits
# logits away from them); instead the SAME quantised file runs both paths at the SAME positions, and the
# native tier must agree with the WGSL path to its usual reduction-order band. What differs between them
# at 30,000 is exactly what this gate is for: where the angles come from.
#
# Three native schedules, each against the WGSL run of the same schedule:
#   FULL      one native prefill of all T rows at P (the tensor-core path, host rope rows [T, dh]);
#   DECODE    an 8-row native prefill, then T-8 native decode steps at P+8.. (the CUDA-graph step);
#   HANDOVER  the 8-row prefill on WGSL, then native decode — the device K/V picks up an OFFSET cache.
# Engagement is checked from `lm_logits`' NATIVE line (a WGSL fallback prints the same rows).
#
# ⛔ NEGATIVE CONTROL: FERRIC_CUDA_ROPE_DEVICE=1 gives ONLY the native tier its old device formula
# (`expf(-2c/dh · logf(base))`); it must land >= 20x further from the WGSL path than the clean run, in both
# FULL and DECODE — else this input cannot see where the native angles come from.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"
[ -f "$M" ] && [ -f "$FX" ] || { echo "usage: $0 <model.gguf> <floor-fixture.json.gz>"; exit 2; }
BIN="$ROOT/target/release/examples/lm_logits"
cargo build -q -p ferric-llama --release --example lm_logits || exit 2

python3 - "$BIN" "$M" "$FX" <<'PY'
import gzip, json, os, subprocess, sys, tempfile
BIN, M, FX = sys.argv[1:4]
ref = json.load(gzip.open(FX, "rt"))
off, ids = ref.get("position_offset", 0), ref["ids"]
T, PRE = len(ids), 8
if off + T <= 4096:
    print(f"⛔ the fixture reaches only position {off + T - 1}: rope-angle precision needs a large position"); sys.exit(2)
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({"ids": ids, "sample_ids": ref["sample_ids"], "position_offset": off}, tmp); tmp.close()
KNOBS = ("FERRIC_CUDA", "FERRIC_CUDA_", "FERRIC_ROPE", "FERRIC_NEOX", "FERRIC_LONGROPE", "FERRIC_LM_DECODE_FROM", "FERRIC_KVQ")

def run(cuda, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(KNOBS) or k == "FERRIC_CUDA_PTX_DIR"}
    if cuda: env["FERRIC_CUDA"] = "1"
    env.update(extra or {})
    r = subprocess.run([BIN, M, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
    rows, nat = {}, (0, 0, 0)
    for l in r.stdout.splitlines():
        p = l.split(" ")
        if p[0] == "ROW": rows[int(p[1])] = (int(p[2]), [float(x) for x in p[5:]])
        elif p[0] == "NATIVE": nat = tuple(int(x) for x in p[1:4])
    if len(rows) != T: print(f"⛔ {len(rows)} rows for {T} ids"); sys.exit(1)
    dev = [l for l in r.stderr.splitlines() if l.startswith("native") or "rope angles" in l]
    return rows, nat, dev

def diff(a, b):
    w, wt = 0.0, -1
    for t in a:
        d = max(abs(x - y) for x, y in zip(a[t][1], b[t][1]))
        if d > w: w, wt = d, t
    return w, wt, sum(a[t][0] == b[t][0] for t in a)

F64 = {t: r for t, r in zip(ref["positions"], ref["float64"])}
def vs_authors(rows):
    return max(max(abs(x - y) for x, y in zip(rows[t][1], F64[t]["sample"])) for t in F64 if t in rows)

print(f"model:     {os.path.basename(M)}   ids: {ref['model']} (floor fixture), {T} tokens at positions {off}..{off + T - 1}")
DEC = {"FERRIC_LM_DECODE_FROM": str(PRE)}
w_full, _, _ = run(False)
w_dec, _, _ = run(False, DEC)
# (name, env, WGSL twin, expected (steps, prefill rows, graph replays))
SCHED = [("FULL", {}, w_full, (0, T, 0)),
         ("DECODE", DEC, w_dec, (T - PRE, PRE, T - PRE)),
         ("HANDOVER", {**DEC, "FERRIC_CUDA_NO_PREFILL": "1"}, w_dec, (T - PRE, 0, T - PRE))]
# ⭐ The band, MEASURED on the RTX 4050 (sampled logits print at 1e-5): see the commit that added this gate.
TOL = 2e-3
ok, clean = True, {}
for name, env, twin, want in SCHED:
    rows, nat, dev = run(True, env)
    if not dev: print("⛔ no CUDA device line: FERRIC_CUDA had no driver to open. NOTHING native was checked."); sys.exit(1)
    d, dt, arg = diff(rows, twin)
    clean[name] = (d, rows)
    print(f"  [{name:<8}] native vs WGSL max |Δ logit| {d:.3e} at position {off + dt}   argmax {arg}/{T}   "
          f"native served (steps, prefill rows, graph replays) {nat}, want {want}   (tol {TOL:g})")
    if nat != want: print("  ⛔ the native tier did not serve this schedule as it should have"); ok = False
    if d > TOL or arg < T - 1: print("  ⛔ native and WGSL disagree at this position"); ok = False
print(f"  for scale — max |Δ| from the authors' float64 over the sample (quantised weights, so NOT a gate): "
      f"WGSL {vs_authors(w_full):.3f}, native FULL {vs_authors(clean['FULL'][1]):.3f}")
for name, env, twin, _ in SCHED[:2]:
    rows, nat, _ = run(True, {**env, "FERRIC_CUDA_ROPE_DEVICE": "1"})
    d, dt, arg = diff(rows, twin)
    x = d / max(clean[name][0], 1e-5)
    print(f"  control: native angles from the device formula (FERRIC_CUDA_ROPE_DEVICE) [{name}]   native vs WGSL {d:.3e} = "
          f"{x:,.0f}x the clean run   argmax {arg}/{T}   ran natively: {nat[0] + nat[1] > 0}")
    if x < 20 or nat[0] + nat[1] == 0: print("  ⛔ this input cannot see where the native angles come from"); ok = False
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
