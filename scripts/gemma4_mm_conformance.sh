#!/usr/bin/env bash
# Gemma 4 IMAGE and AUDIO inputs — Ferric vs THE MODEL AUTHORS' OWN CODE, from the file to the logits, stage
# by stage: the preprocessing, every recorded tower stage, the projection into the text width, and the logits
# at every position of the multimodal prompt (plus, for images, the same question with no image).
#
# The fixture (tests/fixtures/gemma4_mm/*.json.gz, from crates/ferric-llama/examples/refgen/gemma4_mm_ref.py)
# is the authors' transformers code — `Gemma4ForConditionalGeneration.forward`, unmodified, hooks only — run
# twice on the same input: float32, and float64, THE NOISE FLOOR. Every tolerance is MEASURED: a stage (or a
# logit row) passes when Ferric is no further from the float64 run than 4x the authors' own float32 run is,
# with an absolute floor where that distance is below float32 resolution.
#
# ⛔ Every mechanism the gate claims to see is shown load-bearing: each negative control removes one and must
# FIRST fail at the stage that mechanism lives in, by >= 20x the clean error there. A control that does not
# fail is a gate that cannot see that mechanism, and the gate fails.
#
# The preprocessing is checked EXACTLY: an image's resized 8-bit levels must hash to the authors' (sha256);
# an audio clip's log-mel rows must sit within float32 resolution of theirs.
#
#   scripts/gemma4_mm_conformance.sh <tower: checkpoint-dir | mmproj.gguf> <text.gguf> <fixture.json.gz> <file>
#
# <text.gguf> must carry the authors' weights unrounded for the logits to be judged at the floor (ggml-org's
# BF16 GGUF is converted from the same snapshot; `examples/gemma4_mm_weights` checks it tensor by tensor).
# FERRIC_G4_GATE_QUICK=1 runs the clean check only.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOWER="${1:-}"; TEXT="${2:-}"; FX="${3:-}"; FILE="${4:-}"
[ -e "$TOWER" ] && [ -f "$FX" ] && [ -f "$FILE" ] || { echo "usage: $0 <tower dir|mmproj.gguf> <text.gguf> <fixture.json.gz> <file>"; exit 2; }
BIN="${CARGO_TARGET_DIR:-$ROOT/target}/release/examples/gemma4_mm_stages"
(cd "$ROOT" && cargo build -q --offline -p ferric-llama --release --example gemma4_mm_stages) || exit 2

python3 - "$BIN" "$TOWER" "$TEXT" "$FX" "$FILE" <<'PY'
import gzip, hashlib, json, os, subprocess, sys, tempfile
BIN, TOWER, TEXT, FX, FILE = sys.argv[1:6]
ref = json.load(gzip.open(FX, "rt") if FX.endswith(".gz") else open(FX))
kind = ref["kind"]
f64 = ref["float64"]
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(ref, tmp); tmp.close()
lv_path = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
STAGES = [s for s in (["patch_embed"] + sorted([k for k in ref["stages"] if k.startswith("vblock")], key=lambda k: int(k[6:]))
                      + ["subsample"] + sorted([k for k in ref["stages"] if k.startswith("ablock")], key=lambda k: int(k[6:]))
                      + ["pooled", "output_proj", "soft"]) if s in ref["stages"]]
ABS_FLOOR = 2e-6     # relative stage error below which float32 cannot arbitrate
LOGIT_FLOOR = 2e-4   # absolute logit error below which the authors' own f32 run cannot arbitrate

def run(env_extra=None, tower_only=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FERRIC_G4",))}
    env.update(env_extra or {})
    if tower_only: env["FERRIC_G4_TOWER_ONLY"] = "1"
    env["FERRIC_G4_LEVELS_OUT"] = lv_path
    r = subprocess.run([BIN, TOWER, TEXT, FILE, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-3000:]); sys.exit(1)
    o = {"stages": {}, "rows": [], "trows": [], "feat": {}, "stderr": r.stderr}
    for l in r.stdout.splitlines():
        p = l.split()
        if p[0] == "STAGE": o["stages"].setdefault(p[1], {})[int(p[2])] = [float(x) for x in p[3:]]
        elif p[0] == "FEAT": o["feat"][int(p[1])] = [float(x) for x in p[2:]]
        elif p[0] == "ROW": o["rows"].append((int(p[2]), float(p[4]), [float(x) for x in p[5:]]))
        elif p[0] == "TROW": o["trows"].append((int(p[2]), float(p[4]), [float(x) for x in p[5:]]))
        elif p[0] in ("GRID", "FRAMES"): o[p[0].lower()] = [int(x) for x in p[1:]]
    return o

def serr(rec, rows, got):
    """(max |d| over sampled columns / max |ref| there, worst per-row relative sum of squares)."""
    if got is None: return (float("inf"), float("inf"))
    scale = max(abs(v) for r in rec["sample"] for v in r)
    s = q = 0.0
    for i, r in enumerate(rows):
        g = got.get(r) if isinstance(got, dict) else got[i]
        if g is None: return (float("inf"), float("inf"))
        s = max(s, max(abs(a - b) for a, b in zip(g[2:], rec["sample"][i])) / scale)
        q = max(q, abs(g[1] - rec["ssq"][i]) / max(rec["ssq"][i], 1e-30))
    return (s, q)

def as_got(rec): return [[a, b] + c for a, b, c in zip(rec["sum"], rec["ssq"], rec["sample"])]

floor = {s: serr(f64["stages"][s], ref["stages"][s]["rows"], as_got(ref["stages"][s])) for s in STAGES}

def stage_E(o, s):
    e = serr(f64["stages"][s], ref["stages"][s]["rows"], o["stages"].get(s))
    return max(e[0] / max(floor[s][0], ABS_FLOOR), e[1] / max(floor[s][1], ABS_FLOOR)), e

def logit_E(rows, key):
    R, R64 = ref[key], f64[key]
    if len(rows) != len(R): return float("inf"), 0, 0.0
    worst, arg, mx = 0.0, 0, 0.0
    for (am, ssq, v), r, r64 in zip(rows, R, R64):
        d = max(abs(a - b) for a, b in zip(v, r64["sample"]))
        fl = max(abs(a - b) for a, b in zip(r["sample"], r64["sample"]))
        worst = max(worst, d / max(fl, LOGIT_FLOOR)); mx = max(mx, d)
        arg += am == r["top"][0][0]
    return worst, arg, mx

fails = []
print(f"Gemma 4 {kind} — {ref['snapshot']} — {ref['code']}, transformers {ref['transformers']}, torch {ref['torch']}")
print(f"  tower weights: {TOWER}")
clean = run()
print(f"  {clean['stderr'].strip().splitlines()[0]}")
# ---- preprocessing --------------------------------------------------------------------------------------
if kind == "image":
    px = ref["pixels"]
    got = hashlib.sha256(open(lv_path, "rb").read()).hexdigest()
    ok = got == px["levels_sha256"] and clean.get("grid") == px["patch_grid"]
    print(f"  pixels    resized {px['resized_hw']} grid {clean.get('grid')} sha256 {'EQUAL to' if got == px['levels_sha256'] else 'DIFFERS from'} the authors' processor"
          f"{'' if ok else '  <-- FAIL'}")
    if not ok: fails.append("pixels")
else:
    F = ref["features"]
    e = serr(F, F["rows"], clean["feat"])
    ok = clean.get("frames") == [F["shape"][0], F["valid_frames"]] and e[0] < 1e-5 and e[1] < 1e-5
    print(f"  log-mel   {clean.get('frames')} frames/valid vs the authors' {F['shape'][0]}/{F['valid_frames']}: sample {e[0]:.2e}, ssq {e[1]:.2e}"
          f"{'' if ok else '  <-- FAIL'}")
    if not ok: fails.append("features")
# ---- stages and logits ----------------------------------------------------------------------------------
print(f"  {'stage':12s} {'vs float64':>22s} {'authors f32 vs f64':>22s}   ratio (limit 4)")
clean_E = {}
for s in STAGES:
    r, e = stage_E(clean, s); clean_E[s] = r
    print(f"  {s:12s} {e[0]:10.2e} {e[1]:10.2e}  {floor[s][0]:10.2e} {floor[s][1]:10.2e}   {r:5.2f}{'' if r <= 4 else '  <-- FAIL'}")
    if r > 4: fails.append(s)
for key, rows, label in (("rows", clean["rows"], "logits"), ("text_rows", clean["trows"], "text-only")):
    if key not in ref: continue
    w, arg, mx = logit_E(rows, key)
    n = len(ref[key])
    print(f"  {label:12s} {n} rows: max |Δ| {mx:.2e}, worst row {w:.2f}x the authors' own f32 distance (limit 4); argmax {arg}/{n}"
          f"{'' if w <= 4 and arg == n else '  <-- FAIL'}")
    clean_E[key] = max(w, 1e-3)
    if w > 4 or arg != n: fails.append(key)

# ---- negative controls ----------------------------------------------------------------------------------
if os.environ.get("FERRIC_G4_GATE_QUICK"):
    print("FAIL: " + ", ".join(fails) if fails else "PASS (clean only; FERRIC_G4_GATE_QUICK)"); sys.exit(1 if fails else 0)
if kind == "image":
    CONTROLS = [("FERRIC_G4V_NEG", "pos_swap", "patch_embed"), ("FERRIC_G4V_NEG", "rope_swap", "vblock0"),
                ("FERRIC_G4V_NEG", "norope", "vblock0"), ("FERRIC_G4V_NEG", "no_vnorm", "vblock0"),
                ("FERRIC_G4V_NEG", "attn_scale", "vblock0"), ("FERRIC_G4V_NEG", "erf_gelu", "vblock0"),
                ("FERRIC_G4V_NEG", "noclip", None), ("FERRIC_G4V_NEG", "pool_rows", "pooled")]
else:
    CONTROLS = [("FERRIC_G4A_NEG", c, None) for c in ("noclip", "no_relpos", "rel_shift_off", "mask_window", "conv_noncausal",
                                                         "no_ffw_half", "no_softcap", "no_per_dim_scale", "relu_sub")]
CONTROLS += [("FERRIC_G4_SPLICE_NEG", c, "rows") for c in ("ple_mm_id", "scale_soft", "shift")]
print("  negative controls — each must FIRST fail at its own stage, by >= 20x the clean error there:")
for var, val, want in CONTROLS:
    lm = var == "FERRIC_G4_SPLICE_NEG"
    o = run({var: val}, tower_only=not lm)
    first, factor = None, 0.0
    for s in STAGES:
        r, _ = stage_E(o, s)
        if r >= 20 * max(clean_E[s], 1.0):
            first, factor = s, r / max(clean_E[s], 1.0); break
    if first is None and lm:
        w, arg, _ = logit_E(o["rows"], "rows")
        if w >= 20 * clean_E["rows"]: first, factor = "rows", w / clean_E["rows"]
    ok = first is not None and (want is None or first == want)
    print(f"    {var}={val:16s} first fails at {str(first):12s} ({factor:,.0f}x clean){'' if ok else '  <-- FAIL (want ' + (want or 'any stage') + ')'}")
    if not ok: fails.append(f"{var}={val}")
print("FAIL: " + ", ".join(fails) if fails else "PASS"); sys.exit(1 if fails else 0)
PY
