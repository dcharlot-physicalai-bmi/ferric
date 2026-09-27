#!/usr/bin/env bash
# Causal LM (dense runtime): Ferric vs THE MODEL AUTHORS' OWN CODE, gated on THEIR OWN float32-vs-float64
# distance — for inputs where a fixed tolerance cannot be justified in advance: long context, large rope
# positions, 3.8B-parameter models.
#
# The fixture (tests/fixtures/lm_floor/<name>.json.gz, from crates/ferric-llama/examples/refgen/lm_floor_ref.py)
# carries the authors' transformers forward run twice, float32 and float64, at every recorded position: the
# logits at a fixed 128-id vocabulary sample, the sum of squares of the FULL row, the float64 top-10, the
# rotary table their module actually used, and the position the ids ran at (`position_offset`: the authors'
# `position_ids = P..P+T-1` with no cache, which Ferric reproduces as an empty cache starting at P).
#
# PASS: at every recorded position Ferric is within 4x of the authors' float32 distance from float64 (a
# floor below 5e-6 — five counts of the fixture's 6-decimal recording — is clipped there), the argmax is
# the float64 argmax, and the worst full-row sum-of-squares error is within 4x of the floor's worst.
#
# ⛔ Negative controls, each of which must be >= 20x the clean max error, so the gate has SHOWN it can see
# the mechanism: the wrong rotary pairing, always; on a LongRoPE fixture the other table and no attention
# factor. And on an input reaching position 4096 or beyond, a CAN-FAIL demonstration: the rope angles
# derived on the device (FERRIC_ROPE_DEVICE=1, the code before host tables) must leave the 4x band — at
# position 30,000 they sat at 18x (Qwen2.5-0.5B), 46x (Llama-3.2-1B) and 109x (Qwen3-0.6B).
#
#   scripts/lm_floor_conformance.sh <model.gguf> <fixture.json.gz> [--decode-from N --upto M]
#
# `--decode-from N --upto M` (the LongRoPE crossing): the first M ids, prefilled to N and then decoded ONE
# TOKEN AT A TIME; only positions N..M-1 are compared, and the control is the stale cache
# (FERRIC_LONGROPE_NO_REFILL=1) instead of the table controls.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"; shift 2 2>/dev/null
[ -f "$M" ] && [ -f "$FX" ] || { echo "usage: $0 <model.gguf> <fixture.json.gz> [--decode-from N --upto M]"; exit 2; }
BIN="$ROOT/target/release/examples/lm_logits"
cargo build -q -p ferric-llama --release --example lm_logits || exit 2

python3 - "$BIN" "$M" "$FX" "$@" <<'PY'
import gzip, json, os, struct, subprocess, sys, tempfile
BIN, M, FX = sys.argv[1:4]
args = sys.argv[4:]
DECODE_FROM = int(args[args.index("--decode-from") + 1]) if "--decode-from" in args else None
UPTO = int(args[args.index("--upto") + 1]) if "--upto" in args else None
ref = json.load(gzip.open(FX, "rt"))
off = ref.get("position_offset", 0)

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

H = header(M); ARCH = H.get('arch', '?')
if H.get('ft') not in (0, 1, 32):
    print(f"⛔ {os.path.basename(M)} is quantised (file_type {H.get('ft')}) — it cannot verify the math; convert the "
          f"authors' weights at F32"); sys.exit(2)
ids = ref["ids"][:UPTO] if UPTO else ref["ids"]
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({"ids": ids, "sample_ids": ref["sample_ids"], "position_offset": off}, tmp); tmp.close()
# (position, float32 row, float64 row) — the crossing compares only the rows past the prefill.
REC = [(t, a, b) for t, a, b in zip(ref["positions"], ref["float32"], ref["float64"])
       if t < len(ids) and (DECODE_FROM is None or t >= DECODE_FROM)]
if not REC:
    print("⛔ no recorded position in range"); sys.exit(2)
FLOOR_MIN = 5e-6
floor = [max(max(abs(x - y) for x, y in zip(a["sample"], b["sample"])), FLOOR_MIN) for _, a, b in REC]
floor_ssq = max(abs(a["ssq"] - b["ssq"]) / b["ssq"] for _, a, b in REC)

def measure(env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FERRIC_ROPE", "FERRIC_NEOX", "FERRIC_LONGROPE",
                                                                    "FERRIC_LM_DECODE_FROM", "FERRIC_KVQ"))}
    if DECODE_FROM is not None: env["FERRIC_LM_DECODE_FROM"] = str(DECODE_FROM)
    env.update(env_extra or {})
    r = subprocess.run([BIN, M, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode < 0:
        # Killed by a signal, not a verdict: on a shared machine a 15 GB F32 model plus a 4,650-token
        # forward can be killed under memory pressure (seen: SIGKILL mid-gate, swap 20 of 21.5 GB used).
        # Once, and said out loud — a second death fails the gate.
        print(f"  ⚠ lm_logits killed by signal {-r.returncode} ({', '.join(env_extra or {}) or 'clean run'}) — retrying once")
        r = subprocess.run([BIN, M, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"⛔ lm_logits exited with status {r.returncode}"); print(r.stderr[-1500:]); sys.exit(1)
    rows = [l.split(" ") for l in r.stdout.splitlines() if l.startswith("ROW ")]
    if len(rows) != len(ids):
        print(f"⛔ Ferric produced {len(rows)} rows for {len(ids)} tokens"); sys.exit(1)
    err, arg, ssq = [], 0, 0.0
    for (t, a, b) in REC:
        row = rows[t]
        err.append(max(abs(float(x) - y) for x, y in zip(row[5:], b["sample"])))
        arg += int(row[2]) == b["top"][0][0]
        ssq = max(ssq, abs(float(row[4]) - b["ssq"]) / b["ssq"])
    ratio = [e / f for e, f in zip(err, floor)]
    return err, ratio, arg, ssq

rope = ref.get("rope", {})
print(f"reference: {ref['model']} ({ref['model_type']}) — {ref['code']}, transformers {ref['transformers']}, torch "
      f"{ref['torch']}, float32 AND float64, {ref['attn_implementation']} attention")
print(f"weights:   {os.path.basename(M)} (arch {ARCH})   tokens {len(ids)} at positions {off}..{off + len(ids) - 1}"
      + (f", prefilled to {DECODE_FROM} then decoded one token at a time" if DECODE_FROM else "")
      + f"   rope {rope.get('rope_type')} {rope.get('factors_used', '')}   positions compared {len(REC)}")
err, ratio, arg, ssq = measure()
n = len(REC); s = sorted(ratio); w = max(range(n), key=lambda i: ratio[i])
print(f"  |Ferric - float64| over the sample: max {max(err):.3e}   the authors' float32: max {max(floor):.3e}")
print(f"  ratio to the floor, per position: median {s[n // 2]:.2f}x  p90 {s[int(n * 0.9)]:.2f}x  worst {s[-1]:.2f}x "
      f"(position {REC[w][0] + off})   band 4x")
print(f"  argmax agreement with float64            {arg}/{n}")
print(f"  full-row sum of squares, worst rel diff  {ssq:.3e}  (floor {floor_ssq:.3e}, band 4x)")
ok = s[-1] <= 4 and arg == n and ssq <= 4 * max(floor_ssq, 1e-7)

controls = []
if DECODE_FROM is not None:
    controls.append(("stale cache across the crossing (the short-table rows kept)", {"FERRIC_LONGROPE_NO_REFILL": "1"}))
else:
    # llama.cpp's llama / muse-glimmer GGUFs are permuted for NORM pairing; every other dense arch is NEOX.
    controls.append(("wrong rope pairing", {"FERRIC_NEOX": "1"} if ARCH in ("llama", "muse-glimmer") else {"FERRIC_ROPE_NORM": "1"}))
    if rope.get("rope_type") == "longrope":
        other = "short" if rope.get("factors_used") == "long_factor" else "long"
        controls.append((f"the {other} LongRoPE table", {"FERRIC_LONGROPE": other}))
        controls.append(("no LongRoPE attention factor", {"FERRIC_LONGROPE_NO_ATTN_FACTOR": "1"}))
for label, env in controls:
    e, r, a, _ = measure(env)
    x = max(e) / max(max(err), 1e-9)
    print(f"  control: {label} ({', '.join(env)})   max {max(e):.3e} = {x:,.0f}x   worst {max(r):,.0f}x the floor   argmax {a}/{n}")
    if x < 20:
        print("  ⛔ not >= 20x worse — this gate cannot see that mechanism on this input"); ok = False

if DECODE_FROM is None and off + len(ids) > 4096:
    e, r, a, _ = measure({"FERRIC_ROPE_DEVICE": "1"})
    left = max(r) > 4
    print(f"  can-fail: rope angles derived on the device (FERRIC_ROPE_DEVICE)   worst {max(r):.2f}x the floor, median "
          f"{sorted(r)[n // 2]:.2f}x — {'leaves the band, as it must' if left else 'STAYS IN THE BAND'}")
    if not left:
        print("  ⛔ this input cannot see rope-angle precision"); ok = False
print("✅ within 4x of the authors' own float32 error at every position" if ok else "⛔ LM floor conformance FAILED")
sys.exit(0 if ok else 1)
PY
