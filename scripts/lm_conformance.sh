#!/usr/bin/env bash
# Causal LMs (llama / qwen2 / qwen3 dense, qwen35 gated-delta-net hybrid): Ferric vs THE MODEL AUTHORS'
# OWN IMPLEMENTATION, every position.
#
# ⛔⛔ THE REFERENCE IS THE AUTHORS' CODE. These architectures were marked Verified against llama.cpp
# (`scripts/validate_vs_llamacpp.sh`, which called llama.cpp "the reference") or against a reference the
# registry never named — and that check compared GREEDY ARGMAX only. A small logit error almost never
# changes an argmax, so "matches token for token" is close to vacuous about the logits themselves.
#
# The fixture (tests/fixtures/lm/<model>.json, from crates/ferric-llama/examples/refgen/lm_logits_ref.py
# in float32, eager attention) carries the authors' tokenizer output for ~140 tokens and, per position,
# their top-10, a fixed 128-id vocabulary sample, and the sum and sum of squares of the FULL row. So a
# defect anywhere in the vocabulary moves a number even though 128 ids are compared directly.
#
# ⛔ Compare against weights at the precision the authors ran. Convert THEIR files:
#     convert_hf_to_gguf.py <their snapshot> --outtype f32
# A quantised file cannot verify the math — the BERT reranker at 4 bits differed by 0.04–0.31 with a
# defect hidden inside that noise.
#
#   scripts/lm_conformance.sh <model.gguf> <fixture.json>
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"
[ -f "$M" ] && [ -f "$FX" ] || { echo "usage: $0 <model.gguf> <fixture.json>"; exit 2; }
BIN="$ROOT/target/release/examples/lm_logits"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example lm_logits || exit 2

python3 - "$BIN" "$M" "$FX" <<'PY'
import json, os, struct, subprocess, sys
BIN, M, FX = sys.argv[1:4]
ref = json.load(open(FX))

def header(path):
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
    out = {}
    with open(path, 'rb') as f:
        f.read(4); u32(f); u64(f); nkv = u64(f)
        for _ in range(nkv):
            k = rstr(f); t = u32(f)
            if k == 'general.file_type' and t in (4, 5): out['ft'] = u32(f)
            elif k == 'general.architecture' and t == 8: out['arch'] = rstr(f)
            else: skip(f, t)
    return out

H = header(M); FT = H.get('ft'); ARCH = H.get('arch', '?')
if FT not in (0, 1, 32):
    print(f"⛔ {os.path.basename(M)} is quantised (file_type {FT}) — it cannot verify the math; convert the "
          f"authors' weights at F32"); sys.exit(2)
# Measured on F32 weights converted from the authors' own files (max |logit diff|): Qwen3-0.6B 7.0e-5,
# Llama-3.2-1B 1.1e-4, Qwen2.5-0.5B 2.6e-4, Qwen3.5-0.8B 7.5e-4 — the hybrid sits at 3/4 of the band, so
# a regression there has little room. F16/BF16 files add weight rounding, so they get a looser band.
LOGIT_TOL, SSQ_TOL = (1e-3, 1e-4) if FT == 0 else (5e-2, 5e-3)

def measure(env_extra=None):
    env = dict(os.environ)
    for k in ("FERRIC_ROPE_NORM", "FERRIC_NEOX"): env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run([BIN, M, FX], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-1500:]); sys.exit(1)
    rows = [l.split(" ") for l in r.stdout.splitlines() if l.startswith("ROW ")]
    if len(rows) != len(ref["rows"]):
        print(f"⛔ Ferric produced {len(rows)} rows for {len(ref['rows'])} positions"); sys.exit(1)
    worst = worst_t = 0; argmax = 0; ssq = 0.0
    for t, (row, rr) in enumerate(zip(rows, ref["rows"])):
        best = int(row[2]); q = float(row[4]); vals = [float(x) for x in row[5:]]
        d = max(abs(a - b) for a, b in zip(vals, rr["sample"]))
        if d > worst: worst, worst_t = d, t
        argmax += best == rr["top"][0][0]
        ssq = max(ssq, abs(q - rr["ssq"]) / rr["ssq"])
    return worst, worst_t, argmax, ssq, len(rows)

worst, worst_t, argmax, ssq, n = measure()
rows = [None] * n
# ⭐ NEGATIVE CONTROL: the WRONG rotary pairing — the classic silent RoPE failure, which still produces
# fluent text. llama's GGUF weights are permuted for interleaved (NORM) pairing; the Qwen family uses
# split-half (NEOX). Flip whichever this architecture uses; if the gate cannot see that, it sees nothing.
# ⛔ The Qwen3.5 runtime once ignored FERRIC_ROPE_NORM: the control measured 1x and the gate refused to
# pass, correctly. A control is a claim about the runtime too — check it moves before trusting it.
flip = {"FERRIC_NEOX": "1"} if ARCH in ("llama", "muse-glimmer") else {"FERRIC_ROPE_NORM": "1"}
c_worst, _, c_argmax, _, _ = measure(flip)

print(f"reference: {ref['model']} ({ref['model_type']}) — transformers {ref['transformers']}, torch "
      f"{ref['torch']}, float32, eager")
print(f"weights:   {os.path.basename(M)} ({ {0:'F32',1:'F16',32:'BF16'}[FT] }, arch {ARCH})   positions {n}   vocab {ref['vocab']}")
print(f"  max |logit diff|, {len(rows)} x {len(ref['sample_ids'])} sampled   {worst:.3e}  (at position {worst_t}; tol {LOGIT_TOL:g})")
print(f"  argmax agreement                          {argmax}/{n}")
print(f"  full-row sum of squares, worst rel diff   {ssq:.3e}  (tol {SSQ_TOL:g})")
print(f"  control: wrong rope pairing ({list(flip)[0]})     max |logit diff| {c_worst:.3e}  "
      f"= {c_worst / max(worst, 1e-9):,.0f}x   argmax {c_argmax}/{n}")
ok = worst <= LOGIT_TOL and ssq <= SSQ_TOL and argmax == n
if c_worst < 20 * worst:
    print("  ⛔ the wrong-rope control is not ≥20x worse — this gate cannot see a rotary error"); ok = False
print("✅ agrees with the authors at every position" if ok else "⛔ LM conformance FAILED")
sys.exit(0 if ok else 1)
PY
