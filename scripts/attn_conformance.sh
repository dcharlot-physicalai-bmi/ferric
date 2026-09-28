#!/usr/bin/env bash
# **Prefill attention on the matrix units (`native_attn`, opt-in with FERRIC_QGEMM) against the MODEL
# AUTHORS** — does running S = Q·Kᵀ and P·V with fp16 operands keep a model's logits inside the band the
# portable f32 path already has against the authors' own implementation?
#
#   scripts/attn_conformance.sh <model.gguf> <fixture.json>
#   e.g. scripts/attn_conformance.sh qwen2.5-0.5b-authors-Q8_0.gguf \
#            crates/ferric-llama/tests/fixtures/lm/qwen2.5-0.5b-long.json
#
# Same question and same bands as scripts/qgemm_conformance.sh (read its header), asked of the ATTENTION
# kernel alone. Five runs of the same file and fixture (the authors' transformers float32 run, ~550 tokens,
# so every layer's prefill attention is a 550-query block):
#   portable   the default f32 route (tiled WGSL attention, f32 GEMMs)
#   attention  FERRIC_FLASH=native — attention on the matrix units, GEMMs still f32
#   tier       FERRIC_QGEMM=1 — the whole tensor-unit tier: quantized GEMMs AND attention
#   gemm-only  FERRIC_QGEMM=1 FERRIC_FLASH=tiled — the tier with attention held portable, so the
#              attention kernel's own increment on top of the GEMMs is visible
#   control    FERRIC_FLASH=native FERRIC_ATTN_FAULT=1 — the online-softmax rescale dropped (attn.metal)
# PASS: the attention and tier runs sit no further from the authors than portable (mean within 5%, max
# within 1.25x, argmax within 1); the attention route's own perturbation is < 25% of the band; the planted
# fault moves the mean by >= 20x the tolerance.
#
# A quantized file is what this gate is for (its band is the quantization's). On an F32 file of the
# authors' weights the portable band is f32 rounding (~1e-4), which an fp16-operand route cannot meet by
# construction — the run then REPORTS the route's intrinsic cost (Qwen2.5-0.5B: max 3.5e-2, see the commit
# that added this) and exits 3, neither pass nor fail.
#
# ⛔ VACUITY GUARDS: the attention, tier and control runs must dispatch `flash_attn_mu` (FERRIC_TRACE_KERNELS)
# in every layer; the portable and gemm-only runs must not; the attention run must differ from portable.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"
[ -f "$M" ] && [ -f "$FX" ] || { echo "usage: $0 <model.gguf> <fixture.json>"; exit 2; }
BIN="$ROOT/target/release/examples/lm_logits"
cargo build -q -p ferric-llama --release --example lm_logits 2>/dev/null || { cargo build -p ferric-llama --release --example lm_logits; exit 2; }

python3 - "$BIN" "$M" "$FX" <<'PY'
import json, math, os, struct, subprocess, sys
BIN, M, FX = sys.argv[1:4]
ref = json.load(open(FX))
POS = ref.get("positions") or list(range(len(ref["ids"])))

def file_type(path):
    def u32(f): return struct.unpack('<I', f.read(4))[0]
    def u64(f): return struct.unpack('<Q', f.read(8))[0]
    def rstr(f): return f.read(u64(f)).decode('utf-8', 'replace')
    def skip(f, t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in sizes: f.read(sizes[t]); return
        if t == 8: rstr(f); return
        if t == 9:
            et = u32(f); n = u64(f)
            for _ in range(n): skip(f, et)
    with open(path, 'rb') as f:
        f.read(4); u32(f); u64(f); nkv = u64(f)
        for _ in range(nkv):
            k = rstr(f); t = u32(f)
            if k == 'general.file_type' and t in (4, 5): return u32(f)
            skip(f, t)
    return None

def run(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FERRIC_QGEMM", "FERRIC_FLASH", "FERRIC_ATTN"))}
    env.update(extra); env["FERRIC_TRACE_KERNELS"] = "1"
    r = subprocess.run([BIN, M, FX], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-1500:]); sys.exit(1)
    fired = sum(1 for l in r.stderr.splitlines() if l.startswith("KERNEL\tflash_attn_mu"))
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

def between(a, b):
    d = [abs(x - y) for t in POS for x, y in zip(a["vals"][t], b["vals"][t])]
    return sum(d) / len(d), max(d)

P = run({})
A = run({"FERRIC_FLASH": "native"})
N = run({"FERRIC_QGEMM": "1"})
G = run({"FERRIC_QGEMM": "1", "FERRIC_FLASH": "tiled"})
F = run({"FERRIC_FLASH": "native", "FERRIC_ATTN_FAULT": "1"})
k = len(POS)
print(f"reference: {ref['model']} — {ref.get('code', 'transformers')}, float32, {len(ref['ids'])} tokens, {k} positions x "
      f"{len(ref['sample_ids'])} sampled ids")
ft = file_type(M)
print(f"weights:   {os.path.basename(M)} (file_type {ft})")
for name, r in [("portable (f32 WGSL)", P), ("attention on matrix units", A), ("whole tier (FERRIC_QGEMM)", N),
                ("tier, attention held f32", G), ("control: no rescale", F)]:
    print(f"  {name:27s} max|dlogit| {r['max']:.4f}  mean {r['mean']:.5f}  argmax {r['argmax']}/{k}  ssq rel {r['ssq']:.2e}"
          f"  flash_attn_mu dispatches {r['fired']}")
a_mean, a_max = between(A, P)
n_g_mean, n_g_max = between(N, G)
print(f"  attention route vs portable:        max|dlogit| {a_max:.4f}  mean {a_mean:.5f}  ({a_mean / P['mean']:.1%} of the portable band)")
print(f"  tier vs tier-with-f32-attention:    max|dlogit| {n_g_max:.4f}  mean {n_g_mean:.5f}  (the attention kernel's increment)")
TOL = 0.05
fails = []
for name, r in [("portable", P), ("attention", A), ("tier", N)]:
    if not all(math.isfinite(r[x]) for x in ("max", "mean", "ssq")):
        fails.append(f"the {name} run produced non-finite logits")
layers = A["fired"]
if P["fired"] or G["fired"]: fails.append("a run meant to be portable dispatched flash_attn_mu — the opt-in leaked")
if not layers: fails.append("FERRIC_FLASH=native dispatched no flash_attn_mu — the route never fired (no passthrough-MSL device?)")
if N["fired"] != layers or F["fired"] != layers: fails.append(f"dispatch counts differ: attention {layers}, tier {N['fired']}, control {F['fired']}")
if a_max == 0: fails.append("attention route and portable logits are bit-identical — the route changed nothing")
if ft in (0, 1, 32):
    print(f"  (an unquantized file: the portable band is f32 rounding, which fp16 operands cannot meet — reported, not gated)")
    for f in fails: print(f"⛔ {f}")
    sys.exit(1 if fails else 3)
for name, r in [("attention", A), ("tier", N)]:
    if r["mean"] > (1 + TOL) * P["mean"]: fails.append(f"{name}: mean distance {r['mean']:.5f} > {1 + TOL:.2f} x portable {P['mean']:.5f}")
    if r["max"] > 1.25 * P["max"]: fails.append(f"{name}: max distance {r['max']:.4f} > 1.25 x portable {P['max']:.4f}")
    if r["argmax"] < P["argmax"] - 1: fails.append(f"{name}: argmax {r['argmax']} < portable {P['argmax']} - 1")
if a_mean > 0.25 * P["mean"]: fails.append(f"the attention route's own perturbation {a_mean:.5f} is > 25% of the quantization band")
ctl = (F["mean"] - P["mean"]) / (TOL * P["mean"]) if math.isfinite(F["mean"]) else math.inf
print(f"  negative control moved the mean distance by {ctl:.0f}x the tolerance (needs >= 20x)")
if not ctl >= 20: fails.append("negative control: a planted attention defect was not seen — this gate cannot fail")
for f in fails: print(f"⛔ {f}")
print("✅ PASS — attention on the matrix units stays inside the portable path's band against the authors" if not fails else "⛔ FAIL")
sys.exit(1 if fails else 0)
PY
