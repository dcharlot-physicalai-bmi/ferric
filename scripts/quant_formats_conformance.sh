#!/usr/bin/env bash
# Quantized checkpoint formats — Ferric vs THE DEFINING LIBRARY of each format, and vs THE AUTHORS' MODEL.
#
# Formats: GPTQ (GPTQModel), AWQ (AutoAWQ's dequantize_gemm, as vendored by GPTQModel), compressed-tensors
# (neuralmagic's library), block-scaled FP8 (transformers' Fp8Dequantize), NVIDIA ModelOpt NVFP4, and the
# GGUF types IQ1_S / IQ1_M / IQ2_XS / IQ2_S / IQ3_S / NVFP4 (ggml — llama.cpp IS the defining
# implementation of its own types).
#
# 1. UNIT (always; no downloads): committed fixtures, bit for bit —
#      ferric-gguf  more_quants: real published blocks + crafted every-grid-entry blocks, decoded by
#                   libggml-base's own to_float (tests/fixtures/ggml_quants)
#      ferric-load  quant: one module of 15 real published checkpoints, sliced, beside what the defining
#                   library dequantized that mini-checkpoint to (tests/fixtures/quant)
#      ferric-tensor iq_raw + gq: the GPU kernels against the CPU decoders (SKIPPED without a GPU — and
#                   the test says so; nothing is checked then)
# 2. MODEL (per crates/ferric-llama/tests/fixtures/quant_lm/<name>.json.gz whose checkpoint is on disk):
#    the authors' transformers model carrying exactly the library-dequantized weights, float32 AND
#    float64 (crates/ferric-llama/examples/refgen/quant_ref.py logits). PASS: at every recorded position
#    |Ferric - float64| over the 128-id sample <= 4x the authors' own |float32 - float64| (clipped below at
#    FLOOR_MIN), argmax == the float64 argmax, full-row sum of squares within 4x the floor's.
#    ⛔ NEGATIVE CONTROL per model: one plausible misreading of the format (wrong nibble order, v1 zero
#    offset skipped, g_idx ignored, signed offset dropped, FP4 nibbles swapped, scale rows collapsed, IQ
#    signs / delta flipped) must land >= 20x the clean max error — or the gate has not shown it can see
#    that mechanism.
#
#   scripts/quant_formats_conformance.sh [--unit-only] [name ...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
UNIT_ONLY=0; NAMES=()
for a in "$@"; do [ "$a" = "--unit-only" ] && UNIT_ONLY=1 || NAMES+=("$a"); done
FAIL=0

# ⚠ The verdict is cargo's EXIT STATUS, never a grep of its output: `grep "test result"` also matches
# "test result: FAILED", so a grep-as-verdict passes a failing suite. And a GPU test that SKIPPED
# (no adapter) checked nothing, so a skip fails this gate rather than reading as a pass.
unit() {
    local out rc; out=$("$@" 2>&1); rc=$?
    echo "$out" | grep -E "test result|FAILED|panicked|SKIPPED|worst" | sed 's/^/   /'
    [ $rc = 0 ] || { echo "   ⛔ exit $rc"; FAIL=1; }
    if echo "$out" | grep -q "SKIPPED"; then echo "   ⛔ a GPU test was skipped — nothing was checked"; FAIL=1; fi
    # a filter that matches nothing is "ok. 0 passed" with exit 0 — vacuous, so refused
    if ! echo "$out" | grep -qE "test result: ok\. [1-9]"; then echo "   ⛔ no test ran"; FAIL=1; fi
}
echo "── unit: ferric-gguf more_quants + gq (libggml-decoded fixtures)"
unit cargo test -q --offline -p ferric-gguf --lib -- more_quants gq::
echo "── unit: ferric-load quant (15 published checkpoints vs their libraries)"
unit cargo test -q --offline -p ferric-load --lib -- quant::
echo "── unit: ferric-tensor iq_raw + gq kernels (GPU)"
unit cargo test -q --offline -p ferric-tensor --lib -- iq_raw gq:: --nocapture
[ $UNIT_ONLY = 1 ] && { [ $FAIL = 0 ] && echo "PASS (unit)" || echo "FAIL (unit)"; exit $FAIL; }

BIN="$ROOT/target/release/examples/quant_logits"
cargo build -q --offline -p ferric-llama --release --example quant_logits || exit 2
PY="${PYTHON:-python3}"
"$PY" - "$BIN" "$ROOT/crates/ferric-llama/tests/fixtures/quant_lm" "${NAMES[@]+"${NAMES[@]}"}" <<'PY' || FAIL=1
import glob, gzip, json, os, subprocess, sys, tempfile
BIN, FXD = sys.argv[1:3]
only = set(sys.argv[3:])
HUB = os.path.expanduser("~/.cache/huggingface/hub")
QF = os.path.expanduser("~/.cache/ferric/hub/quant-fixtures")
FLOOR_MIN = 2e-6
# name -> (checkpoint, control env, control value, what the control reads wrongly)
CONTROLS = {
    "gptq_int4":        ("FERRIC_QUANT_CONTROL", "gptq_zero",   "v1 zero-points without the +1"),
    "gptq_actorder":    ("FERRIC_QUANT_CONTROL", "gidx",        "g_idx ignored (group i / group_size)"),
    "gptq_int3":        ("FERRIC_QUANT_CONTROL", "gptq_zero",   "v1 zero-points without the +1"),
    "awq":              ("FERRIC_QUANT_CONTROL", "awq_order",   "nibbles in order (no AWQ_REVERSE_ORDER)"),
    "ct_w4a16_actorder":("FERRIC_QUANT_CONTROL", "gidx",        "weight_g_idx ignored"),
    "ct_w4a16_asym":    ("FERRIC_QUANT_CONTROL", "ct_offset",   "stored code read as the signed value"),
    "ct_fp8_channel":   ("FERRIC_QUANT_CONTROL", "scale_rows",  "per-channel scale read as per-tensor"),
    "ct_nvfp4":         ("FERRIC_QUANT_CONTROL", "fp4_nibbles", "FP4 high nibble first"),
    "fp8_block":        ("FERRIC_QUANT_CONTROL", "scale_rows",  "every row takes block-row 0's scales"),
    "modelopt_nvfp4":   ("FERRIC_QUANT_CONTROL", "fp4_nibbles", "FP4 high nibble first"),
    "gguf_IQ1_S":       ("FERRIC_IQ_CONTROL",    "iq_delta",    "IQ1 grid shift with the wrong sign"),
    "gguf_IQ1_M":       ("FERRIC_IQ_CONTROL",    "iq_delta",    "IQ1 grid shift with the wrong sign"),
    "gguf_IQ2_XS":      ("FERRIC_IQ_CONTROL",    "iq_signs",    "sign bits ignored"),
    "gguf_IQ2_S":       ("FERRIC_IQ_CONTROL",    "iq_signs",    "sign bits ignored"),
    "gguf_IQ3_S":       ("FERRIC_IQ_CONTROL",    "iq_signs",    "sign bits ignored"),
    "gguf_NVFP4":       ("FERRIC_IQ_CONTROL",    "nvfp4_half",  "UE4M3 scale without ggml's * 0.5"),
}

def locate(fx, name):
    m = fx["model"]
    if name.startswith("gguf_"):
        p = os.path.join(QF, os.path.basename(m))
        return p if os.path.exists(p) else None
    d = os.path.join(HUB, "models--" + m.replace("/", "--"), "snapshots")
    if not os.path.isdir(d): return None
    s = sorted(os.listdir(d))
    return os.path.join(d, s[0]) if s else None

def run(model, ids_json, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FERRIC_QUANT_CONTROL", "FERRIC_IQ_CONTROL"))}
    env.update(env_extra)
    r = subprocess.run([BIN, model, ids_json], capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"  ⛔ quant_logits exited {r.returncode}: {r.stderr[-800:]}"); return None
    return [l.split(" ") for l in r.stdout.splitlines() if l.startswith("ROW ")]

ok_all, seen = True, 0
for path in sorted(glob.glob(os.path.join(FXD, "*.json.gz"))):
    name = os.path.basename(path)[:-len(".json.gz")]
    if only and name not in only: continue
    fx = json.load(gzip.open(path, "rt"))
    model = locate(fx, name)
    if model is None:
        print(f"── {name}: checkpoint {fx['model']} not on disk — SKIPPED (not checked)"); continue
    seen += 1
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"ids": fx["ids"], "sample_ids": fx["sample_ids"]}, tmp); tmp.close()
    floor = [max(max(abs(a - b) for a, b in zip(x["sample"], y["sample"])), FLOOR_MIN) for x, y in zip(fx["float32"], fx["float64"])]
    floor_ssq = max(abs(x["ssq"] - y["ssq"]) / y["ssq"] for x, y in zip(fx["float32"], fx["float64"]))
    def measure(env):
        rows = run(model, tmp.name, env)
        if rows is None or len(rows) != len(fx["ids"]): return None
        err, arg, ssq = [], 0, 0.0
        for t, y in enumerate(fx["float64"]):
            row = rows[t]
            err.append(max(abs(float(v) - w) for v, w in zip(row[5:], y["sample"])))
            arg += int(row[2]) == y["top"][0][0]
            ssq = max(ssq, abs(float(row[4]) - y["ssq"]) / y["ssq"])
        return err, arg, ssq
    print(f"── {name}: {fx['model']}  ({len(fx['ids'])} positions)\n   dequantized by {fx['dequantizer']}; model: transformers {fx['transformers']}, float32 + float64")
    m = measure({})
    if m is None: ok_all = False; continue
    err, arg, ssq = m
    ratio = sorted(e / f for e, f in zip(err, floor))
    n = len(err)
    print(f"   |Ferric - float64| max {max(err):.3e}   authors' float32 floor max {max(floor):.3e}")
    print(f"   ratio to the floor: median {ratio[n // 2]:.2f}x  worst {ratio[-1]:.2f}x (band 4x)   argmax {arg}/{n}   "
          f"ssq rel {ssq:.2e} (floor {floor_ssq:.2e})")
    ok = ratio[-1] <= 4 and arg == n and ssq <= 4 * max(floor_ssq, 1e-7)
    env, val, what = CONTROLS.get(name, (None, None, None))
    if env:
        c = measure({env: val})
        cerr = max(c[0]) if c else float("inf")
        cx = cerr / max(max(err), 1e-30)
        cok = cx >= 20
        print(f"   control [{what}]: max err {cerr:.3e} = {cx:.0f}x the clean max  {'✅' if cok else '⛔ < 20x'}")
        ok = ok and cok
    else:
        print("   control: none at model level (the weight-level fixtures cover this type)")
    print(f"   {'✅ PASS' if ok else '⛔ FAIL'}")
    ok_all = ok_all and ok
print(f"{seen} model fixtures checked")
sys.exit(0 if ok_all and seen > 0 else 1)
PY
[ $FAIL = 0 ] && echo "PASS" || echo "FAIL"
exit $FAIL
