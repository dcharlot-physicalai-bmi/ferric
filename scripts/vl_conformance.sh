#!/usr/bin/env bash
# Qwen2.5-VL — MiMo-Embodied-7B — Ferric vs THE MODEL AUTHORS' OWN CODE, from the image FILE to the logits,
# stage by stage: pixels, the window order, every recorded vision block, the merger before and after its
# un-permutation, the mRoPE positions, and the logits at every position of the prompt.
#
# The fixture (tests/fixtures/qwen25vl/<model>.json.gz, from crates/ferric-llama/examples/refgen/
# qwen25vl_ref.py) is the authors' transformers code run twice on the same pixels: at float32, and at
# float64 — THE NOISE FLOOR.
#
# ⭐ WHY TWO REFERENCE RUNS. From vision block 17 on, a few tokens carry massive activations (one channel
# reaches 5e4) produced by sums that cancel. There the authors' OWN float32 run sits ~1e-3 (sum of squares)
# from float64, and Ferric's float32 run sits ~3e-3 from it: the same order, both rounding. A float32
# reference cannot arbitrate below its own distance from exact arithmetic, so every tolerance here is
# MEASURED, not chosen: a stage (or a logit row) passes when Ferric is no further than 4x the authors'
# float32 distance from the float64 run, with an absolute floor where that distance is below f32 noise.
#
# ⛔ Every mechanism the gate claims to see is shown load-bearing: each negative control removes one and
# must FIRST fail at the stage that mechanism lives in, by >= 20x the clean error there.
#
# ⭐ THE DECODE, TOO. A prefill check says nothing about the cached decode steps, whose mRoPE position is
# NOT the cache index after an image. If `<fixture>.greedy.json` sits beside the fixture (refgen
# `--greedy`: the authors' argmax at every step of a continuation, teacher-forced in one forward), Ferric
# must generate that continuation exactly — and must NOT when decode positions follow the cache index.
#
#   scripts/vl_conformance.sh <checkpoint-dir> <fixture.json.gz>
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-}"; FX="${2:-}"
[ -d "$DIR" ] && [ -f "$FX" ] || { echo "usage: $0 <checkpoint-dir> <fixture.json.gz>"; exit 2; }
BIN="$ROOT/target/release/examples/qwen25vl_stages"
ASK="$ROOT/target/release/examples/vl_ask"
cargo build -q -p ferric-llama --release --example qwen25vl_stages --example vl_ask || exit 2

python3 - "$BIN" "$DIR" "$FX" "$ASK" <<'PY'
import gzip, json, os, subprocess, sys, tempfile
BIN, DIR, FX, ASK = sys.argv[1:5]
ref = json.load(gzip.open(FX, "rt") if FX.endswith(".gz") else open(FX))
IMG = os.path.join(os.path.dirname(FX), ref["image"])
f64 = ref["float64"]
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(ref, tmp); tmp.close()
ORDER = ["patch_embed"] + sorted((k for k in ref["stages"] if k.startswith("block")), key=lambda k: int(k[5:])) \
        + ["merger_window_order", "merger"]
ORDER = [s for s in ORDER if s in ref["stages"]]

def run(env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FERRIC_VL_", "FERRIC_METAL4", "FERRIC_COOP"))}
    env.update(env_extra or {})
    r = subprocess.run([BIN, DIR, IMG, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-2000:]); sys.exit(1)
    o = {"stages": {}, "rows": [], "pixels": [], "pos": {}}
    for l in r.stdout.splitlines():
        p = l.split(" ")
        if p[0] == "GRID": o["grid"] = [int(x) for x in p[1:]]
        elif p[0] == "WINDOW": o["window"] = [int(x) for x in p[1:]]
        elif p[0] == "POS": o["pos"][p[1]] = [int(x) for x in p[2:]]
        elif p[0] == "PIXELS": o["pixels"].append([float(x) for x in p[2:]])
        elif p[0] == "STAGE": o["stages"].setdefault(p[1], []).append([float(x) for x in p[3:]])
        elif p[0] == "ROW": o["rows"].append((int(p[2]), float(p[4]), [float(x) for x in p[5:]]))
    return o

def stage_err(got, rec):
    """(max |d| over sampled columns / max |ref| there, worst per-row relative sum-of-squares) — `got`
    rows are [sum, ssq, v...]; `rec` is a fixture record."""
    if got is None or len(got) != len(rec["ssq"]): return (float("inf"), float("inf"))
    scale = max(abs(v) for r in rec["sample"] for v in r)
    s = max(abs(a - b) for g, r in zip(got, rec["sample"]) for a, b in zip(g[2:], r)) / scale
    q = max(abs(g[1] - r) / r for g, r in zip(got, rec["ssq"]))
    return (s, q)

def as_rows(rec):  # a fixture record in the shape `run` returns
    return [[a, b] + c for a, b, c in zip(rec["sum"], rec["ssq"], rec["sample"])]

def logit_err(rows, recs):
    return [max(abs(a - b) for a, b in zip(r[2], rr["sample"])) for r, rr in zip(rows, recs)]

def E(o, name):
    """One number per stage for the controls: the larger of the two stage metrics vs the authors' f32."""
    if name == "logits":
        return max(logit_err(o["rows"], ref["rows"])) if len(o["rows"]) == len(ref["rows"]) else float("inf")
    return max(stage_err(o["stages"].get(name), ref["stages"][name]))

o = run()
ok = True
print(f"reference: {ref['model']} ({ref['model_type']}) — {ref['code']}, transformers {ref['transformers']}, "
      f"torch {ref['torch']}; processor {ref['processor']}; tower {ref['checkpoint_dtype']['vision']} and text "
      f"{ref['checkpoint_dtype']['text']} weights, read by Ferric from the authors' safetensors unconverted")
print(f"harness self-test (streamed vs the authors' whole model, tiny config): logits "
      f"{ref['harness_selftest']['max_abs_logit_diff']:g}, image rows {ref['harness_selftest']['max_abs_image_row_diff']:g}")
exact = [("patch grid", o.get("grid") == ref["grid"]),
         ("window order", o.get("window") == ref["window_index"]),
         ("mRoPE positions (t, h, w)", [o["pos"].get(k) for k in "thw"] == ref["position_ids"])]
for label, good in exact:
    print(f"  {label:30s} {'identical' if good else '⛔ DIFFERS'}"); ok &= good
px = max(abs(a - b) for g, r in zip(o["pixels"], ref["pixels"]["sample"]) for a, b in zip(g[2:], r))
print(f"  pixels (Ferric's own preprocessing)  max |d| {px:.2e}  (tol 1e-5)"); ok &= px <= 1e-5

print(f"\n  {'stage':22s} {'vs authors f32':>22s} {'vs float64':>22s} {'authors f32 vs f64':>22s}   (sampled-rel / ssq-rel)")
for name in ORDER:
    rec, rec64 = ref["stages"][name], f64["stages"][name]
    a = stage_err(o["stages"].get(name), rec)
    b = stage_err(o["stages"].get(name), rec64)
    fl = stage_err(as_rows(rec), rec64)
    tol = [max(4 * f, 2e-5) for f in fl]
    good = all(x <= t for x, t in zip(b, tol))
    ok &= good
    print(f"  {name:22s} {a[0]:9.2e} / {a[1]:9.2e} {b[0]:9.2e} / {b[1]:9.2e} {fl[0]:9.2e} / {fl[1]:9.2e}  "
          f"{'' if good else '⛔ beyond 4x the floor'}")

# Logits, per row: each row's tolerance is 4x how far the authors' f32 run is from f64 ON THAT ROW, never
# below the 1e-3 band the text-only gate uses for exact weights. Image rows inherit the tower's conditioning.
if len(o["rows"]) != len(ref["rows"]):
    print(f"⛔ Ferric produced {len(o['rows'])} logit rows for {len(ref['rows'])} tokens"); sys.exit(1)
e32, e64 = logit_err(o["rows"], ref["rows"]), logit_err(o["rows"], f64["rows"])
fl = [max(abs(a - b) for a, b in zip(r["sample"], r64["sample"])) for r, r64 in zip(ref["rows"], f64["rows"])]
bad = [t for t, (x, f) in enumerate(zip(e64, fl)) if x > max(4 * f, 1e-3)]
# argmax: vs the float64 top-1 wherever its margin over the runner-up exceeds the row's tolerance
am = [t for t, (r, r64, f) in enumerate(zip(o["rows"], f64["rows"], fl))
      if r64["top"][0][1] - r64["top"][1][1] > max(4 * f, 1e-3) and r[0] != r64["top"][0][0]]
img = [t for t, x in enumerate(ref["types"]) if x == 1]
seg = {"text before the image": list(range(img[0])), "image": img,
       "text after the image": list(range(img[-1] + 1, len(ref["types"])))}
print(f"\n  logits, {len(e32)} positions x {len(ref['sample_ids'])} sampled ids (vocab {ref['vocab']}):")
for k, idx in seg.items():
    print(f"    {k:21s} ({len(idx):3d})  vs authors f32 {max(e32[t] for t in idx):.2e}   vs float64 "
          f"{max(e64[t] for t in idx):.2e}   authors f32 vs f64 {max(fl[t] for t in idx):.2e}")
print(f"    rows beyond max(4x floor, 1e-3): {len(bad)}   argmax disagreements where f64's margin is decisive: {len(am)}")
ok &= not bad and not am

# ⭐ NEGATIVE CONTROLS: each must FIRST fail at the stage its mechanism lives in, by >= 20x the clean error.
controls = [("no windows (every block full attention)", {"FERRIC_VL_NEG": "nowindow"}, "block0"),
            ("every block windowed (full-attention schedule ignored)", {"FERRIC_VL_NEG": "allwindow"}, "block7"),
            ("vision rope row/column swapped", {"FERRIC_VL_NEG": "rope_swap"}, "block0"),
            ("merged rows left in window order (no reverse)", {"FERRIC_VL_NEG": "noreverse"}, "merger"),
            ("1-D text positions for the image", {"FERRIC_VL_LM_NEG": "pos1d"}, "logits"),
            ("position advances by the image's TOKEN count", {"FERRIC_VL_LM_NEG": "advance_tokens"}, "logits"),
            ("interleaved mRoPE sectors (Qwen3-VL's rule)", {"FERRIC_VL_LM_NEG": "imrope"}, "logits")]
clean = {s: E(o, s) for s in ORDER + ["logits"]}
print("\n  controls (must first fail at their own stage, >= 20x the clean error there):")
for label, env, want in controls:
    tower_only = want != "logits"
    c = run({**env, **({"FERRIC_VL_TOWER_ONLY": "1"} if tower_only else {})})
    stages = ORDER + ([] if tower_only else ["logits"])
    ratios = {s: E(c, s) / max(clean[s], 1e-12) for s in stages}
    first = next((s for s in stages if ratios[s] >= 20), None)
    good = first == want
    ok &= good
    print(f"    {label:55s} first fails at {str(first):20s} {ratios.get(want, float('nan')):>12,.0f}x  "
          f"{'' if good else '⛔ expected ' + want}")
# ⚠ A mechanism this input CANNOT see is reported, never dropped: the merger's GELU. The authors' code uses
# nn.GELU() (exact erf) and so does Ferric, but tanh-vs-erf moves the merger output by less than the f32
# drift already arriving from block 31 (measured 1x) — so that choice rests on the code, not on this gate.
c = run({"FERRIC_VL_NEG": "gelu_tanh", "FERRIC_VL_TOWER_ONLY": "1"})
r = E(c, "merger_window_order") / max(clean["merger_window_order"], 1e-12)
print(f"    ⚠ NOT VISIBLE on this input: tanh GELU in the merger ({r:.1f}x) — below the f32 drift from block 31; "
      f"the erf form rests on the authors' code, not on this gate")
GREEDY = FX.replace(".json.gz", ".greedy.json")
if os.path.exists(GREEDY):
    g = json.load(open(GREEDY))
    n = len(g["continuation"])
    def ask(env_extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FERRIC_VL_")}
        env.update(env_extra or {})
        r = subprocess.run([ASK, DIR, IMG, g["question"], str(n), GREEDY], capture_output=True, text=True, env=env)
        if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
        m = [l for l in r.stdout.splitlines() if l.startswith("MATCH ")]
        return int(m[-1].split()[1].split("/")[0]) if m else -1
    if not all(g["agrees"]):
        print("⛔ the greedy fixture's continuation is not the authors' own argmax chain"); ok = False
    got = ask()
    bad = ask({"FERRIC_VL_LM_NEG": "decode_cachepos"})
    print(f"\n  greedy decode, {n} steps (the authors' argmax at every step; smallest margin {min(g['margin']):.3f}):"
          f" Ferric matches {got}/{n}")
    print(f"    control: decode positions = cache index (not prompt + delta)   matches {bad}/{n}"
          f"  {'' if bad < n else '⛔ indistinguishable — the gate cannot see decode positions'}")
    ok &= got == n and bad < n
else:
    print(f"\n  ⚠ no {os.path.basename(GREEDY)}: the cached DECODE path is not checked by this run")
print("\n✅ agrees with the authors, image file to logits" if ok else "\n⛔ VL conformance FAILED")
os.unlink(tmp.name)
sys.exit(0 if ok else 1)
PY
