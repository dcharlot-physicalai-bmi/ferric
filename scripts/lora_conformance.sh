#!/usr/bin/env bash
# LoRA adapters on the Dense runtime: Ferric vs HUGGING FACE PEFT — the library that defines the adapter
# format and its math — over the model authors' own base (transformers), every recorded position.
#
# ⛔ THE REFERENCE IS PEFT, not llama.cpp. llama.cpp's GGUF adapter is a PEER format this gate also loads
# (the committed syn_a.gguf is llama.cpp's own `convert_lora_to_gguf.py` output), and it must land on the
# SAME logits as the PEFT directory it was converted from.
#
# The committed fixture (crates/ferric-llama/tests/fixtures/lora/, provenance in its README.md):
#   syn_a/      a PEFT adapter made BY PEFT (seeded, init_lora_weights=False so lora_B is nonzero): r=2,
#               alpha=4, all seven projections (q,k,v,o,gate,up,down), layers 0 and 23, stored float16
#   syn_a.gguf  syn_a through llama.cpp's convert_lora_to_gguf.py --outtype f16
#   syn_b/      a second one, rslora (alpha/sqrt(r) = 6/2 = 3), r=4, q,v,o,down on layers 5 and 17
#   ref.json.gz PEFT's logits for a 139-token text, float32 AND float64 (the floor), adapter on and off; and
#               PEFT's MIXED BATCH — forward(..., adapter_names=["a","b","__base__"]), one adapter per row
#
# Every comparison is max |Ferric - PEFT float64| over 30 positions x 128 sampled ids, against PEFT's own
# float32-vs-float64 distance at the same cells (the floor), and must be within FLOOR_X of it:
#   merged     adapter merged into the weights at load (LoraMerged, F32)
#   unmerged   y = Wx + s·B(Ax) at runtime (upload_lora + Cache::set_adapters)
#   gguf       the llama.cpp-converted adapter, unmerged — and identical to the PEFT directory's logits
#   batch      three sequences prefilled alone, then decoded TOGETHER in one forward_batch, each row with
#              its own selection (syn_a, syn_b, none) — against PEFT's mixed batch
# Negative controls (FERRIC_LORA_NEG), each a mistake that loads and runs, must be >= 20x worse than the
# correct run: `rslora` (the other scaling formula), `alpha` (not divided by r), `transpose` (A/B bytes
# read transposed), `off` (adapter not applied), and on the batch `rows` (every row takes row 0's adapter).
#
#   scripts/lora_conformance.sh <qwen2.5-0.5b-instruct F32 .gguf> [<adapter> <ref.json.gz>]
#
# With an adapter and its reference, the single-adapter checks and controls run on it instead (no batch):
#   the committed Llama-3.2-1B fixture, whose GGUF q/k rows are PERMUTED (adds the `noperm` control):
#     scripts/lora_conformance.sh llama-3.2-1b-F32-authors.gguf \
#         crates/ferric-llama/tests/fixtures/lora/llama_qk crates/ferric-llama/tests/fixtures/lora/llama_qk.ref.json.gz
#   a published adapter (fetch it first — see the fixture README):
#     scripts/lora_conformance.sh qwen2.5-0.5b-F32-authors.gguf <HF cache>/taronklm/... \
#         crates/ferric-llama/tests/fixtures/lora/taronklm-lora-chatbot.ref.json.gz
#
# ⛔ The base must be the authors' weights at F32 (convert_hf_to_gguf.py --outtype f32): a quantised base
# cannot verify the math. Measured 2026-09-27, Apple M-series / Metal: every correct path 0.8-1.1x the floor.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"
FXD="$ROOT/crates/ferric-llama/tests/fixtures/lora"
[ -f "$M" ] || { echo "usage: $0 <qwen2.5-0.5b-instruct F32 .gguf> [<adapter> <ref.json.gz>]"; exit 2; }
BIN="$ROOT/target/release/examples/lora_logits"
cargo build -q -p ferric-llama --release --example lora_logits || exit 2

python3 - "$BIN" "$M" "$FXD" "${2:-}" "${3:-}" <<'PY'
import gzip, json, os, subprocess, sys
BIN, M, FXD, ADAPTER, REF = sys.argv[1:6]
CUSTOM = bool(ADAPTER)
if CUSTOM and not (os.path.exists(ADAPTER) and os.path.exists(REF)):
    print(f"⛔ adapter {ADAPTER!r} or reference {REF!r} not found"); sys.exit(2)
ref = json.load(gzip.open(REF if CUSTOM else os.path.join(FXD, "ref.json.gz"), "rt"))
A = ADAPTER if CUSTOM else os.path.join(FXD, "syn_a")
FLOOR_X, CONTROL_X = 3.0, 20.0
ok = True

def run(mode, ids, adapters=(), neg=None, extra=None):
    env = dict(os.environ); env.pop("FERRIC_LORA_NEG", None)
    if neg: env["FERRIC_LORA_NEG"] = neg
    env.update(extra or {})
    r = subprocess.run([BIN, M, ids, ",".join(map(str, ref["sample_ids"])), mode, *adapters],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-2000:]); sys.exit(1)
    return {l.split()[1]: l.split() for l in r.stdout.splitlines() if l.startswith("ROW ")}

def gguf_arch(path):
    """general.architecture from the GGUF header (the first string KV that names it)."""
    import struct
    with open(path, "rb") as f:
        f.read(4); f.read(4); f.read(8); nkv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(nkv):
            k = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
            t = struct.unpack("<I", f.read(4))[0]
            if t != 8: return None       # the architecture KV comes first in every converter's output
            v = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
            if k == "general.architecture": return v
    return None

def dist(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))

IDS = ",".join(map(str, ref["ids"]))
POS = ref["positions"]
V = ref["variants"]
def ferric(rows): return [[float(x) for x in rows[str(t)][5:]] for t in POS]
def samples(key): return [r["sample"] for r in V[key]]
def argmax(rows, key): return sum(int(rows[str(t)][2]) == r["top"][0][0] for t, r in zip(POS, V[key]))

cfg = ref["adapters"][0]["config"]
print(f"reference: PEFT {ref['peft']} on {ref['model']} (transformers {ref['transformers']}, torch {ref['torch']}), "
      f"float32 and float64")
print(f"adapter:   {ref['adapters'][0]['path']}  r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')} "
      f"targets={sorted(cfg['target_modules']) if isinstance(cfg['target_modules'], list) else cfg['target_modules']}")
print(f"input:     {len(ref['ids'])} tokens, {len(POS)} positions x {len(ref['sample_ids'])} sampled ids")

fl_b = dist(samples("base_f32"), samples("base_f64"))
fl = dist(samples("adapted_f32"), samples("adapted_f64"))
eff = dist(samples("adapted_f64"), samples("base_f64"))
print(f"  PEFT's own floor (float32 vs float64): adapted {fl:.3e}, base {fl_b:.3e};  the adapter moves the "
      f"logits by up to {eff:.3f}")
if eff < 1000 * fl:
    print(f"  ⛔ the adapter's effect is < 1000x the floor — a loader that ignored it would nearly pass"); ok = False

base = ferric(run("base", IDS))
e = dist(base, samples("base_f64"))
print(f"  base (no adapter)        |Ferric - PEFT64| {e:.3e} = {e / fl_b:4.2f}x its floor   (the runtime itself)")

errs = {}
paths = [("merged", "merged", [A]), ("unmerged", "unmerged", [A])]
# The PEER format: `<adapter>.gguf` beside a PEFT directory is that directory through llama.cpp's own
# convert_lora_to_gguf.py. On llama the converter permutes q/k lora_B itself, so this path takes NO
# permutation in Ferric while the PEFT path takes one — the two must still agree.
GG = os.path.normpath(A) + ".gguf"
if os.path.isdir(A) and os.path.exists(GG):
    paths.append(("gguf (llama.cpp)", "unmerged", [GG]))
rowsets = {}
for label, mode, ad in paths:
    rows = run(mode, IDS, ad)
    rowsets[label] = rows
    f = ferric(rows)
    e = dist(f, samples("adapted_f64"))
    am = argmax(rows, "adapted_f64")
    bad = e > FLOOR_X * fl or am != len(POS)
    ok &= not bad
    errs[label] = e
    print(f"  {label:24s} |Ferric - PEFT64| {e:.3e} = {e / fl:4.2f}x the floor   argmax {am}/{len(POS)}"
          f"{'   <-- FAIL' if bad else ''}")
if "gguf (llama.cpp)" in rowsets:
    # The GGUF stores the same float16 values as the safetensors, so the two paths must agree EXACTLY —
    # any difference is the reader (or the q/k permutation), not arithmetic.
    d = dist(ferric(rowsets["gguf (llama.cpp)"]), ferric(rowsets["unmerged"]))
    ok &= d == 0.0
    print(f"  gguf vs PEFT directory, Ferric vs Ferric: max |diff| {d:.3e}{'' if d == 0.0 else '   <-- FAIL: must be identical'}")
else:
    print(f"  ⚠ no {os.path.basename(GG)} beside the adapter: llama.cpp's GGUF adapter format NOT checked on this one")

if not CUSTOM:
    B = ref["batch"]
    P, pos = B["prompt_len"], B["positions"]
    rows = None
    def batch_err(neg=None):
        rows = run("batch", ";".join(",".join(map(str, s)) for s in B["seqs"]), [A, os.path.join(FXD, "syn_b")],
                   neg=neg, extra={"LORA_P": str(P)})
        e = 0.0; am = 0; n = 0
        for i in range(3):
            f = [[float(x) for x in rows[f"{i}:{t}"][5:]] for t in pos]
            e = max(e, dist(f, [r["sample"] for r in B["f64"][i]]))
            am += sum(int(rows[f"{i}:{t}"][2]) == r["top"][0][0] for t, r in zip(pos, B["f64"][i])); n += len(pos)
        return e, am, n
    bfl = max(dist([r["sample"] for r in B["f32"][i]], [r["sample"] for r in B["f64"][i]]) for i in range(3))
    e, am, n = batch_err()
    bad = e > FLOOR_X * bfl or am != n
    ok &= not bad
    errs["batch"] = e
    print(f"  batch: rows a | b | none |Ferric - PEFT64| {e:.3e} = {e / bfl:4.2f}x the floor ({bfl:.3e})   argmax {am}/{n}"
          f"{'   <-- FAIL' if bad else ''}   ({len(pos)} decode steps x 3 rows, prompt {P})")

if not CUSTOM:
    # A prompt cache keyed on tokens alone would hand one request's adapted K/V to another. The entry
    # made under syn_a must seed a syn_a cache and MISS for the base and for syn_b.
    env = dict(os.environ); env.pop("FERRIC_LORA_NEG", None)
    r = subprocess.run([BIN, M, ",".join(map(str, ref["ids"][:48])), "0", "prefix", A, os.path.join(FXD, "syn_b")],
                       capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
    seeds = {l.split()[1]: int(l.split()[2]) for l in r.stdout.splitlines() if l.startswith("SEED ")}
    good = seeds.get("adapter", 0) >= 32 and seeds.get("none") == 0 and seeds.get("adapter_b") == 0
    ok &= good
    print(f"  prefix cache: entry made under syn_a reused {seeds.get('adapter')} tokens by syn_a, "
          f"{seeds.get('none')} by the base, {seeds.get('adapter_b')} by syn_b{'' if good else '   <-- FAIL'}")

print("controls (each must be >= 20x the correct run):")
for neg, why in [("rslora", "the other scaling formula"), ("alpha", "alpha not divided by r"),
                 ("transpose", "lora_A/lora_B bytes read transposed"), ("off", "adapter not applied")]:
    for label, mode in [("unmerged", "unmerged"), ("merged", "merged")]:
        f = ferric(run(mode, IDS, [A], neg=neg))
        e = dist(f, samples("adapted_f64"))
        good = e >= CONTROL_X * errs[label]
        ok &= good
        print(f"  {neg:9s} {label:9s} {e:.3e} = {e / errs[label]:10,.0f}x   ({why}){'' if good else '   <-- CONTROL FAILED'}")
# ⛔ On a base whose GGUF PERMUTED q/k (llama), leaving a PEFT adapter's q/k rows in HF order is the
# silent failure `LoraAdapter::bind` exists to prevent. Shown only where the base permutes: on qwen2
# the permutation is the identity and this control would measure 1x, correctly.
ARCH = gguf_arch(M)
if ARCH == "llama" and not A.endswith(".gguf"):
    for label, mode in [("unmerged", "unmerged"), ("merged", "merged")]:
        f = ferric(run(mode, IDS, [A], neg="noperm"))
        e = dist(f, samples("adapted_f64"))
        good = e >= CONTROL_X * errs[label]
        ok &= good
        print(f"  noperm    {label:9s} {e:.3e} = {e / errs[label]:10,.0f}x   (q/k lora_B left in HF row order on a "
              f"permuted llama GGUF){'' if good else '   <-- CONTROL FAILED'}")
if not CUSTOM:
    e, am, n = batch_err("rows")
    good = e >= CONTROL_X * errs["batch"]
    ok &= good
    print(f"  rows      batch     {e:.3e} = {e / errs['batch']:10,.0f}x   (every row takes row 0's adapter)"
          f"{'' if good else '   <-- CONTROL FAILED'}")
print("✅ LoRA agrees with PEFT on every path" if ok else "⛔ LoRA conformance FAILED")
sys.exit(0 if ok else 1)
PY
