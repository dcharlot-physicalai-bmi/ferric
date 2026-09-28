#!/usr/bin/env bash
# PrismML **Bonsai 2** (Ternary-Bonsai-2-27B: qwen35 hybrid, ternary weights in a Hadamard-ROTATED basis)
# on Ferric vs THE AUTHORS' OWN RUNTIME — PrismML's llama.cpp fork — on identical token ids.
#
# ⛔ A Bonsai 2 file decodes like any ternary file and then produces fluent gibberish unless every folded
# matmul sees H(s ⊙ x) (blockwise normalized Walsh–Hadamard, block 1024, ±1 signs per width, and a
# tiled→grouped head permutation before ssm_out) and the embedding lookup is inverted. That is PrismML's
# own warning for mainline llama.cpp. So this gate checks, in order:
#   1. the decoders: PQ2_0 / PTQ1_0 / group-64 Q2_0 rows bit-identical to the fork's `to_float`
#      (committed rows), the lossless GPU carriers, the reader lock, the contract validation, and the GPU
#      FWHT bit-identical to the host reference (blocks 2..16384);
#      with BONSAI2_FULL_DEQ=1 also EVERY quantized tensor of the file, hashed, against the fork's hashes
#      (26.87B values; measured 0 mismatches for all three published files, 2026-09-28);
#   2. whole-model logits at every recorded position of 3 prompts (5, 63 and 441 tokens) and a 32-token
#      greedy continuation each, against tests/fixtures/bonsai2/<packing>.json from
#      crates/ferric-llama/examples/refgen/bonsai2_ref.py (the fork, decode path, flash attention off);
#   3. NEGATIVE CONTROLS, each ≥20x worse than the real run or the gate fails: no Hadamard, no sign
#      flips, no GDN head permutation, and the transform at block 64 instead of 1024
#      (FERRIC_PRISM_OFF=hadamard|signs|perm|block64). "Group-128 read as group-64" is covered in step 1
#      (the decode control in crates/ferric-gguf/tests/bonsai2_dequant.rs) — at the model level a stride
#      mismatch is refused by the reader before any matmul runs.
#
# Tolerance: the fixture's `noise_floor` — how far the fork's own two attention kernels (flash on vs off,
# same decode path) disagree over the same rows. Ferric must agree with the authors at least that well.
#
#   scripts/bonsai2_conformance.sh <Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0,Q2_0...}.gguf> [fixture.json]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"
[ -f "$M" ] || { echo "usage: $0 <Ternary-Bonsai-2-27B-*.gguf> [fixture.json]"; exit 2; }
case "$(basename "$M")" in
  *PTQ1_0*) PK=ptq1_0 ;; *PQ2_0*) PK=pq2_0 ;; *Q2_0*) PK=q2_0 ;; *) PK=unknown ;;
esac
FX="${2:-$ROOT/crates/ferric-llama/tests/fixtures/bonsai2/$PK.json}"
[ -f "$FX" ] || { echo "⛔ no fixture $FX"; exit 2; }
cd "$ROOT"
echo "── 1. decoders, carriers, lock, contract, FWHT"
cargo test -q --release -p ferric-gguf --test bonsai2_dequant 2>&1 | tail -2 | grep -q "test result: ok" \
  || { echo "⛔ ferric-gguf bonsai2_dequant failed"; exit 1; }
cargo test -q --release -p ferric-gguf --lib prism 2>&1 | grep -q "test result: ok. [1-9]" \
  || { echo "⛔ ferric-gguf prism unit tests failed"; exit 1; }
cargo test -q --release -p ferric-tensor --lib fwht 2>&1 | grep -q "test result: ok. [1-9]" \
  || { echo "⛔ GPU FWHT tests failed"; exit 1; }
echo "  ✓ rows bit-identical to the fork's to_float (3 packings); carriers lossless; lock + contract; GPU FWHT = host"
if [ "${BONSAI2_FULL_DEQ:-0}" = 1 ]; then
  cargo build -q --release -p ferric-gguf --example deqhash || exit 2
  "$ROOT/target/release/examples/deqhash" "$M" | awk '{print $1, $3}' | sort > /tmp/bonsai2_deq_ferric.$$
  awk '{print $1, $4}' crates/ferric-gguf/tests/fixtures/bonsai2/deqhash_fork_all_packings.txt | sort > /tmp/bonsai2_deq_fork.$$
  if cmp -s /tmp/bonsai2_deq_ferric.$$ /tmp/bonsai2_deq_fork.$$; then
    echo "  ✓ every quantized tensor ($(wc -l < /tmp/bonsai2_deq_fork.$$)) bit-identical to the fork's dequantization"
  else
    echo "⛔ whole-file dequant differs from the fork: $(diff /tmp/bonsai2_deq_ferric.$$ /tmp/bonsai2_deq_fork.$$ | grep -c '^<') tensors"; exit 1
  fi
  rm -f /tmp/bonsai2_deq_*.$$
fi
BIN="$ROOT/target/release/examples/bonsai2_logits"
cargo build -q --release -p ferric-llama --example bonsai2_logits || exit 2

python3 - "$BIN" "$M" "$FX" <<'PY'
import array, hashlib, json, os, subprocess, sys, tempfile
BIN, M, FX = sys.argv[1:4]
ref = json.load(open(FX)); nv = ref["n_vocab"]; S = ref["sample_ids"]
sha = hashlib.sha256()
with open(M, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""): sha.update(c)
if sha.hexdigest() != ref["sha256"]:
    print(f"⛔ {os.path.basename(M)} is not the file the fixture was made from ({sha.hexdigest()[:12]} vs {ref['sha256'][:12]})"); sys.exit(2)
TOL = ref["noise_floor"]["max_abs"]; SSQ_TOL = 1e-5

def run(p, env_extra=None, gen=32):
    env = dict(os.environ); env.pop("FERRIC_PRISM_OFF", None); env.update(env_extra or {})
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "l.bin")
        r = subprocess.run([BIN, M, ",".join(map(str, p["ids"])), "--out", out, "--gen", str(gen)],
                           capture_output=True, text=True, env=env)
        if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
        g = [int(x) for x in r.stdout.split("gen=")[1].split()[0].split(",")] if gen else []
        worst = ssq = 0.0; wpos = am = 0; n = 0
        with open(out, "rb") as f:
            for r_, rr in zip(p["positions"], p["rows"]):
                f.seek(r_ * nv * 4); row = array.array("f"); b = f.read(nv * 4)
                if len(b) < nv * 4: break
                row.frombytes(b); n += 1
                vals = [row[i] for i in S] + [row[i] for i, _ in rr["top"]]
                want = rr["sample"] + [v for _, v in rr["top"]]
                d = max(abs(a - b) for a, b in zip(vals, want))
                if d > worst: worst, wpos = d, r_
                am += max(range(nv), key=row.__getitem__) == rr["top"][0][0]
                ssq = max(ssq, abs(sum(x * x for x in row) - rr["ssq"]) / rr["ssq"])
        return worst, wpos, am, n, ssq, g

print(f"── 2. logits vs {ref['reference']}")
print(f"   file {ref['model']} (sha256 {ref['sha256'][:12]}…), tolerance = the fork's own flash-on/off spread {TOL:.3e}")
ok = True; worst_all = 0.0
for p in ref["prompts"]:
    w, wp, am, n, ssq, g = run(p)
    worst_all = max(worst_all, w)
    same = g == p["gen"]
    first_diff = next((i for i, (a, b) in enumerate(zip(g, p["gen"])) if a != b), None)
    print(f"  {p['name']:7s} {len(p['ids']):4d} ids, {n:3d} rows: max|Δ| {w:.3e} (row {wp})  argmax {am}/{n}  "
          f"ssq rel {ssq:.1e}  greedy {len(p['gen'])} tokens {'identical' if same else f'DIFFER at step {first_diff}'}"
          f"  (fork floor on this prompt {p['noise_floor']:.2e}, min greedy margin {min(p['gen_margins']):.2f})")
    ok &= w <= TOL and am == n and ssq <= SSQ_TOL and same and n == len(p["rows"])
print("── 3. negative controls (france, prompt rows)")
p0 = dict(ref["prompts"][0]); k = len(p0["ids"])
p0["positions"], p0["rows"] = p0["positions"][:k], p0["rows"][:k]
base, *_ = run(p0, gen=0)
for c, what in [("hadamard", "no Hadamard transform"), ("signs", "no sign flips"), ("perm", "no GDN tiled→grouped permutation"),
                ("block64", "transform at block 64, not the contract's")]:
    w, _, am, n, _, _ = run(p0, {"FERRIC_PRISM_OFF": c}, gen=0)
    ratio = w / max(worst_all, 1e-9)
    print(f"  {what:44s} max|Δ| {w:.3e} = {ratio:,.0f}x the real run's worst  argmax {am}/{n}")
    if ratio < 20: print("  ⛔ not ≥20x worse — the gate cannot see this mechanism"); ok = False
print("✅ Bonsai 2 agrees with PrismML's fork at every recorded position" if ok else "⛔ Bonsai 2 conformance FAILED")
sys.exit(0 if ok else 1)
PY
