#!/usr/bin/env bash
# nomic-embed-text: Ferric vs THE MODEL AUTHORS' OWN IMPLEMENTATION, stage by stage.
#
# The reference (tests/fixtures/nomic_bert/, from crates/ferric-llama/examples/refgen/nomic_bert_ref.py) is
# nomic's `modeling_hf_nomic_bert.py` run through transformers' trust_remote_code, in float32 AND float64.
# Every stage — the embedding LayerNorm and each block's output — must lie within 4x of the authors' own
# float32-vs-float64 distance, measured at the same rows and columns; so must the sentence embedding
# (mean pooling from the authors' 1_Pooling/config.json, then L2 normalisation).
#
# Controls (FERRIC_BERT_NEG), each of which must be >= 20x the floor and must first fail in the component it
# names: gate_swap (silu on fc11 instead of fc12), rope_interleaved, rope_base_10000, no_rope — all in
# block 0 — and no_type (token-type row not added) — in the embedding LayerNorm.
#
#   scripts/nomic_bert_conformance.sh <nomic-embed-text-v1.5 HF snapshot dir | F32 .gguf> [fixture.json.gz]
#
# ⚠ An F16 or quantised file cannot pass a float32 floor: F16 differs by its own rounding (cos >= 0.99999976),
# nomic's Q4_K_M by 0.94-0.965 — the second is the quantisation (the authors' code on the same dequantised
# weights gives the same cosines). Give the authors' snapshot or nomic's F32 GGUF.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: nomic_bert_conformance.sh <hf-dir | F32 gguf> [fixture]}"
FX="${2:-$ROOT/crates/ferric-llama/tests/fixtures/nomic_bert/nomic-embed-text-v1.5.json.gz}"
cargo build -q -p ferric-llama --release --example bert_stages || exit 2
BIN="$ROOT/target/release/examples/bert_stages"
python3 - "$BIN" "$MODEL" "$FX" <<'PY'
import gzip, json, os, subprocess, sys
BIN, MODEL, FX = sys.argv[1:4]
ref = json.load(gzip.open(FX, "rt"))
print(f"reference: {ref['model']} via {ref['modeling']} (transformers {ref['transformers']}, torch {ref['torch']})")

def run(it, neg=None):
    env = dict(os.environ); env.pop("FERRIC_BERT_NEG", None)
    if neg: env["FERRIC_BERT_NEG"] = neg
    r = subprocess.run([BIN, MODEL, ",".join(map(str, it["ids"])), ",".join(map(str, it["rows"])), ",".join(map(str, ref["cols"]))],
                       capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
    F = {}
    for l in r.stdout.splitlines():
        tag, name, *v = l.split(); F[(tag, name)] = [float(x) for x in v]
    return F

def stages(it, F):
    """(stage, ferric-vs-f64, floor) in network order; the floor is the authors' f32-vs-f64 at the same cells."""
    out = []
    for st, rv in it["stages"].items():
        fname = "inp_norm" if st == "emb_ln" else f"{st}.layer_out_norm"
        a = [x for row in rv["f32"]["rows"] for x in row]; b = [x for row in rv["f64"]["rows"] for x in row]
        f = F[("ROWS", fname)]
        if len(f) != len(b): print(f"⛔ {st}: {len(f)} values for {len(b)}"); sys.exit(1)
        out.append((st, max(abs(x - y) for x, y in zip(f, b)), max(abs(x - y) for x, y in zip(a, b))))
    return out

ok = True
for it in ref["items"]:
    F = run(it)
    S = stages(it, F)
    worst = max(d / fl for _, d, fl in S)
    e64, e32, fe = it["emb_norm"]["f64"], it["emb_norm"]["f32"], F[("NORM", "pooled")]
    ed = max(abs(x - y) for x, y in zip(fe, e64)); efl = max(abs(x - y) for x, y in zip(e32, e64))
    bad = worst > 4 or ed > 4 * efl
    ok &= not bad
    print(f"  {len(it['ids']):4d} tok  stages: worst {worst:5.2f}x the floor   embedding {ed:.2e} vs floor {efl:.2e} "
          f"({ed / efl:.2f}x)  cos {sum(x * y for x, y in zip(fe, e64)):.9f}{'   <-- FAIL' if bad else ''}")

print("controls (on the second text; each must be >= 20x and first fail where it names):")
it = ref["items"][1]
for neg, where in [("gate_swap", "l0"), ("rope_interleaved", "l0"), ("rope_base_10000", "l0"), ("no_rope", "l0"), ("no_type", "emb_ln")]:
    S = stages(it, run(it, neg))
    first = next((st for st, d, fl in S if d > 4 * fl), None)
    ratio = max(d / fl for _, d, fl in S)
    good = ratio >= 20 and first == where
    ok &= good
    print(f"  {neg:17s} {ratio:14,.0f}x  first fails at {first} (expected {where}){'' if good else '   <-- CONTROL FAILED'}")
print("✅ nomic-bert matches the authors at every stage" if ok else "⛔ nomic-bert conformance FAILED")
sys.exit(0 if ok else 1)
PY
