#!/usr/bin/env bash
# The CPU fabric (FERRIC_CPU=1) on QUANTIZED files: against Ferric's GPU path on the same file, and both
# against the model's AUTHORS — the gate for what `scripts/lm_conformance.sh` refuses to judge (a
# quantized file cannot verify the math against the authors; it can verify one fabric against another).
#
# Arms per file, each one `lm_logits` prefill of the fixture's ids (the authors' tokenization):
#   GPU       the GPU path (dequantized weights x f32 activations)
#   CPU       the CPU fabric as shipped (int16 activations: an int8 row plus its rounding residual)
#   CPU-f32   FERRIC_CPU_ACT=f32 (dequantized weights x f32 activations — the GPU's arithmetic, so it
#             must agree with the GPU to accumulation order)
#   CPU-int8  FERRIC_CPU_ACT=int8 (llama.cpp's CPU scheme) — REPORTED, not gated: it is what the int16
#             default exists to avoid (on Qwen3-0.6B Q8_0 it sat 1.64x the GPU's distance to the authors)
#
# PASS, per file:
#   1. CPU-f32 vs GPU, full vocabulary at every position: max |dlogit| <= 2e-3 and argmax identical.
#      (F32 files of the same models sit at 5e-5..3.3e-4 from the authors on both fabrics.)
#   2. CPU vs the authors is in the GPU's band: its mean |dlogit| over the fixture's 128-id sample is
#      within 1.15x of the GPU's, and its argmax agreement with the authors is at most 2 positions short
#      of the GPU's. (int8 activations are a second, small rounding; this bounds what it may cost.)
#   3. CPU vs GPU greedy continuation, 64 tokens after a 32-token prompt: reported (first divergence and
#      the GPU's top-2 margin there); CPU-f32 vs GPU must be identical.
# ⭐ Negative control (resolution): CPU-f32 on file A against the GPU on file B — the same weights in the
# NEXT listed format — must be >= 20x the criterion-1 distance. A comparison that could not tell two
# quantizations of one model apart could not see a mis-decoded format either.
#
#   scripts/cpu_fabric_conformance.sh <fixture.json> <A.gguf> <B.gguf> [<C.gguf> ...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FX="${1:-}"; shift
[ -f "$FX" ] && [ $# -ge 2 ] || { echo "usage: $0 <fixture.json> <A.gguf> <B.gguf> [...]  (two or more quantized files of ONE model)"; exit 2; }
cargo build -q -p ferric-llama --release --example lm_logits --example greedy_ids || exit 2
python3 - "$ROOT/target/release/examples" "$FX" "$@" <<'PY'
import array, json, os, subprocess, sys, tempfile
BIN, FX, FILES = sys.argv[1], sys.argv[2], sys.argv[3:]
ref = json.load(open(FX))
T, V = len(ref["ids"]), ref["vocab"]
POS = ref.get("positions") or list(range(T))
SAMPLE = ref["sample_ids"]
tmp = tempfile.mkdtemp()

def run(model, arm):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FERRIC_")}
    if arm != "GPU": env["FERRIC_CPU"] = "1"
    if arm == "CPU-f32": env["FERRIC_CPU_ACT"] = "f32"
    if arm == "CPU-int8": env["FERRIC_CPU_ACT"] = "int8"
    dump = os.path.join(tmp, f"{os.path.basename(model)}.{arm}.bin")
    env["LM_LOGITS_DUMP"] = dump
    r = subprocess.run([f"{BIN}/lm_logits", model, FX], capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
    a = array.array("f"); a.frombytes(open(dump, "rb").read()); os.remove(dump)
    assert len(a) % T == 0
    g = subprocess.run([f"{BIN}/greedy_ids", model, FX, "32", "64"], capture_output=True, text=True, env=env)
    if g.returncode: print(g.stderr[-1500:]); sys.exit(1)
    steps = [l.split() for l in g.stdout.splitlines() if l.startswith("STEP ")]
    return a, len(a) // T, [(int(s[2]), float(s[3])) for s in steps]

def full(a, b, v):
    worst, agree = 0.0, 0
    for t in range(T):
        ra, rb = a[t * v:(t + 1) * v], b[t * v:(t + 1) * v]
        worst = max(worst, max(abs(x - y) for x, y in zip(ra, rb)))
        agree += max(range(v), key=ra.__getitem__) == max(range(v), key=rb.__getitem__)
    return worst, agree

def authors(a, v):
    s, n, agree = 0.0, 0, 0
    for t, rr in zip(POS, ref["rows"]):
        row = a[t * v:(t + 1) * v]
        for i, want in zip(SAMPLE, rr["sample"]): s += abs(row[i] - want); n += 1
        agree += max(range(v), key=row.__getitem__) == rr["top"][0][0]
    return s / n, agree

def greedy(g, c):
    div = next((i for i, (x, y) in enumerate(zip(g, c)) if x[0] != y[0]), None)
    return "identical (64/64)" if div is None else f"diverge at step {div}: GPU top-2 margin there {g[div][1]:.4f}", div

ok = True
arms = {}
for f in FILES:
    arms[f] = {arm: run(f, arm) for arm in ("GPU", "CPU", "CPU-f32", "CPU-int8")}
print(f"fixture: {ref['model']} — {T} tokens, authors' float32 logits at {len(POS)} positions x {len(SAMPLE)} ids")
for i, f in enumerate(FILES):
    (g, v, gg), (c, _, cg), (c32, _, c32g) = arms[f]["GPU"], arms[f]["CPU"], arms[f]["CPU-f32"]
    (c8, _, c8g) = arms[f]["CPU-int8"]
    (m8, a8u) = authors(c8, v)
    g8txt, _ = greedy(gg, c8g)
    d32, a32 = full(c32, g, v)
    dc, ac = full(c, g, v)
    (mg, ag), (mc, acu), (m32, a32u) = authors(g, v), authors(c, v), authors(c32, v)
    other = FILES[(i + 1) % len(FILES)]
    dctl, _ = full(c32, arms[other]["GPU"][0], v)
    gtxt, gdiv = greedy(gg, cg)
    g32txt, g32div = greedy(gg, c32g)
    print(f"\n{os.path.basename(f)}")
    print(f"  CPU-f32 vs GPU, full vocab    max |d| {d32:.3e}   argmax {a32}/{T}          (need <= 2e-3, {T}/{T})")
    print(f"  CPU     vs GPU, full vocab    max |d| {dc:.3e}   argmax {ac}/{T}")
    print(f"  vs the authors (mean |d| over the sample, argmax):  GPU {mg:.4f} {ag}/{len(POS)}   CPU {mc:.4f} {acu}/{len(POS)}"
          f"   CPU-f32 {m32:.4f} {a32u}/{len(POS)}   (CPU/GPU {mc / mg:.3f}, need <= 1.15)")
    print(f"  greedy 64 after 32:  CPU vs GPU {gtxt};  CPU-f32 vs GPU {g32txt}")
    print(f"  (reported) CPU-int8: vs the authors {m8:.4f} {a8u}/{len(POS)} ({m8 / mg:.3f}x the GPU's); greedy vs GPU {g8txt}")
    print(f"  control: CPU-f32 on this file vs GPU on {os.path.basename(other)}   max |d| {dctl:.3e} = {dctl / max(d32, 1e-9):,.0f}x")
    bad = []
    if d32 > 2e-3 or a32 != T: bad.append("CPU-f32 does not match the GPU")
    if mc > 1.15 * mg or acu < ag - 2: bad.append("CPU (int8 activations) is outside the GPU's band vs the authors")
    if g32div is not None: bad.append("CPU-f32 greedy continuation differs from the GPU's")
    if dctl < 20 * max(d32, 1e-9): bad.append("the control is not >= 20x: this comparison cannot resolve two formats")
    for b in bad: print(f"  ⛔ {b}")
    ok &= not bad
print("\n✅ the CPU fabric agrees with the GPU path on every file, and sits in its band against the authors" if ok
      else "\n⛔ CPU fabric conformance FAILED")
sys.exit(0 if ok else 1)
PY
