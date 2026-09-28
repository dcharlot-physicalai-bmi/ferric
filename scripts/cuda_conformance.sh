#!/usr/bin/env bash
# **The NVIDIA native tier, one decode step at a time, against Ferric's portable path AND the authors.**
#
#   scripts/cuda_conformance.sh <model.gguf> <fixture.json> [long_total]
#
# Runs `examples/lm_decode_logits` twice on the SAME quantised file — once with FERRIC_CUDA unset (the
# WGSL path, which `scripts/lm_conformance.sh` verifies against the authors on F32 weights) and once with
# it set — and compares every position's logits (128 sampled ids + the full row's sum of squares):
#
# in TWO native modes: DECODE (FERRIC_CUDA_NO_PREFILL=1: multi-token calls stay on WGSL, so the K/V cache
# is handed device -> WGSL -> device mid-sequence) and FULL (prefill on the tensor cores too).
#
#   1. ENGAGEMENT: every single-token call must have been a native step (`NATIVE_STEPS n OF n`) replayed
#      from the captured CUDA graph (`NATIVE_GRAPH n`), and in FULL mode every multi-token row a native
#      prefill row. The graph's eager twin (FERRIC_CUDA_NO_GRAPH=1) must print the SAME logits to the last
#      printed digit — the two launch the same kernels on the same inputs. A WGSL fallback prints the same kind of
#      rows, so without this the gate would pass on a tier that never ran — the fate of the first CUDA
#      check, whose "reference" had been routed to CUDA too.
#   2. NATIVE vs WGSL, same weights: max |Δ logit| within TOL_NW (decode: two f32 reduction orders) or
#      TOL_PF (full: prefill on the tensor cores). Both bands are measured, below.
#   3. vs THE AUTHORS (transformers, float32, the fixture): a quantised file cannot match them — its
#      distance is the quantisation noise — so the claim is RELATIVE: the native tier sits no further
#      from the authors than the portable path does (plus the native-vs-WGSL band), with the same argmax
#      agreement up to near-ties.
#   4. NEGATIVE CONTROLS on the native path: each flips one mechanism the native kernels implement to a
#      known-wrong form (rope pairing; Llama-3 rope_freqs; Qwen2 q/k/v bias — whichever the FILE has).
#      Under the control the native run must (a) still agree with WGSL-under-the-same-control within the
#      band — the native path really honours that mechanism — and (b) move ≥ 20x the band away from the
#      normal run, while running natively. Otherwise this gate has not shown it can see the mechanism.
#   5. LONG (optional `long_total`, e.g. 2300): native vs WGSL only, past the old 2048-token cap.
#
# The multi-token chunk mid-sequence (lm_decode_logits' `chunk_at`) runs on WGSL, so the K/V cache is
# handed device → WGSL and back inside every run; a stale row anywhere shows up in (2).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"; LONG="${3:-0}"
[ -f "$M" ] && [ -f "$FX" ] || { echo "usage: $0 <model.gguf> <fixture.json> [long_total]"; exit 2; }
BIN="$ROOT/target/release/examples/lm_decode_logits"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example lm_decode_logits || exit 2

python3 - "$BIN" "$M" "$FX" "$LONG" <<'PY'
import json, os, struct, subprocess, sys
BIN, M, FX, LONG = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ref = json.load(open(FX))

def gguf(path):
    """architecture + the tensor names (so the controls are chosen from what the FILE has)."""
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
    arch, names = '?', []
    with open(path, 'rb') as f:
        f.read(4); u32(f); nt = u64(f); nkv = u64(f)
        for _ in range(nkv):
            k = rstr(f); t = u32(f)
            if k == 'general.architecture' and t == 8: arch = rstr(f)
            else: skip(f, t)
        for _ in range(nt):
            names.append(rstr(f)); nd = u32(f); f.read(8 * nd + 4 + 8)
    return arch, set(names)
ARCH, NAMES = gguf(M)

# ⚠ Clear every knob that changes the math, so an inherited shell variable cannot make both arms wrong alike.
KNOBS = ("FERRIC_CUDA", "FERRIC_CUDA_Q8X", "FERRIC_CUDA_NO_PREFILL", "FERRIC_CUDA_NO_GRAPH", "FERRIC_CUDA_ROPE_DEVICE", "FERRIC_ROPE_DEVICE", "FERRIC_NEOX", "FERRIC_ROPE_NORM", "FERRIC_NO_ROPE_FREQS", "FERRIC_NO_QKV_BIAS",
         "FERRIC_KVQ", "FERRIC_NOFUSE", "FERRIC_NO_QK_FUSE", "FERRIC_ONE_ROPE", "FERRIC_NO_SWA")
def run(cuda, extra=None, total=0, prefill=8, chunk_at=64, chunk=5):
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    if cuda: env["FERRIC_CUDA"] = "1"
    env.update(extra or {})
    r = subprocess.run([BIN, M, FX, str(prefill), str(chunk_at), str(chunk), str(total)], capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
    rows, steps, pf, graph = {}, None, None, None
    for l in r.stdout.splitlines():
        p = l.split(" ")
        if p[0] == "ROW": rows[int(p[1])] = (int(p[2]), float(p[4]), [float(x) for x in p[5:]])
        elif p[0] == "NATIVE_STEPS": steps = (int(p[1]), int(p[3]))
        elif p[0] == "NATIVE_PREFILL": pf = (int(p[1]), int(p[3]))
        elif p[0] == "NATIVE_GRAPH": graph = int(p[1])
    dev = [l for l in r.stderr.splitlines() if l.startswith(("adapter", "native"))]
    return rows, (steps, pf, graph), dev, r.stderr

def diff(a, b):
    """max |Δ| over the sampled logits, worst position, worst relative Δ of the full-row sum of squares."""
    worst, wt, ssq = 0.0, -1, 0.0
    for t in a:
        d = max(abs(x - y) for x, y in zip(a[t][2], b[t][2]))
        if d > worst: worst, wt = d, t
        ssq = max(ssq, abs(a[t][1] - b[t][1]) / max(abs(b[t][1]), 1e-30))
    return worst, wt, ssq

POS = ref.get("positions") or list(range(len(ref["ids"])))
AUTH = {t: rr for t, rr in zip(POS, ref["rows"])}
def vs_authors(rows):
    worst, agree, n = 0.0, 0, 0
    for t, rr in AUTH.items():
        if t not in rows: continue
        worst = max(worst, max(abs(x - y) for x, y in zip(rows[t][2], rr["sample"])))
        agree += rows[t][0] == rr["top"][0][0]; n += 1
    return worst, agree, n

# ⭐ The two bands, MEASURED on the RTX 4050 (sampled logits print at 1e-5 resolution):
#   DECODE native vs WGSL, max |Δ logit|: over the ~140-token fixtures Qwen3-0.6B Q5_K_M (the tier as
#   it was accepted) 4.0e-5, Qwen2.5-0.5B Q4_K_M 1.2e-4, Qwen2.5-0.5B Q8_0 5.0e-5, Llama-3.2-1B Q4_K_M
#   2.0e-5 — but the band GROWS WITH CONTEXT (two f32 attention sums over more keys): at 2300 positions
#   Qwen2.5 reached 3.2e-4 and Llama 1.35e-3, which a 1e-3 tolerance refused. TOL_NW = 5e-3 is ~4x that.
#   FULL (tensor-core prefill: integer weight codes + split hi/lo f16 activations, nothing rounded but
#   f32 accumulation), fixture / 1000-row prompt: Qwen2.5 Q4_K_M 1.9e-4 / 1.3e-4, Qwen2.5 Q8_0 1.1e-4 /
#   1.4e-4, Qwen3 Q5_K_M 2.0e-4 / 5.1e-4, Llama Q4_K_M 2.0e-5 / 1.3e-4 — the decode band's order, so
#   TOL_PF = TOL_NW. (The two forms it replaced: f16 activations + f16-rounded weights, 0.219 on Qwen3's
#   1000 rows; split activations + rounded weights, 0.028 there and 4.7e-2 on Qwen2.5 Q8_0.)
#   For scale: every one of these files sits 2.4-11.7 logits from the authors (quantisation), and the
#   native tier's distance from them matched WGSL's to within 0.01 in every run.
TOL_NW = 5e-3
TOL_PF = TOL_NW
ok = True
print(f"model:     {os.path.basename(M)} (arch {ARCH})   fixture: {ref['model']} — transformers {ref['transformers']}, float32")
wr, _, wdev, _ = run(False)
for l in wdev: print(f"  {l}")
for mode, extra, tol in (("DECODE", {"FERRIC_CUDA_NO_PREFILL": "1"}, TOL_NW), ("FULL", {}, TOL_PF)):
    cr, cs, cdev, cerr = run(True, extra)
    if not any(l.startswith("native") for l in cdev):
        print("⛔ no CUDA device line: FERRIC_CUDA had no driver to open. NOTHING native was checked."); sys.exit(1)
    (n_steps, n_single), (n_pf, n_multi), n_graph = cs
    want_pf = n_multi if mode == "FULL" else 0
    print(f"  [{mode}] engagement: {n_steps}/{n_single} decode steps native ({n_graph} replayed from the CUDA graph), "
          f"{n_pf}/{n_multi} prompt rows native (want {want_pf})")
    if n_steps == 0 or n_steps != n_single or n_pf != want_pf or n_graph != n_steps:
        print("  ⛔ the native tier did not serve what it should have"); print("\n".join(l for l in cerr.splitlines() if "cuda" in l)[-800:]); ok = False
    if mode == "DECODE":
        er, es, _, _ = run(True, {**extra, "FERRIC_CUDA_NO_GRAPH": "1"})
        ed, et, essq = diff(er, cr)
        print(f"  [{mode}] eager launches (FERRIC_CUDA_NO_GRAPH) vs graph replay: max |Δ logit| {ed:.1e}   "
              f"({es[0][0]} steps, {es[2]} replays; want identical printed logits and 0 replays)")
        if ed != 0.0 or essq != 0.0 or es[2] != 0 or es[0][0] != n_steps: print("  ⛔ the graph replay and its eager twin differ"); ok = False
    d, dt, ssq = diff(cr, wr)
    arg = sum(cr[t][0] == wr[t][0] for t in cr)
    print(f"  [{mode}] native vs WGSL (same weights)   max |Δ logit| {d:.3e} at position {dt}   ssq rel {ssq:.2e}   argmax {arg}/{len(cr)}   (tol {tol:g})")
    ok &= d <= tol
    wa, wagree, na = vs_authors(wr); ca, cagree, _ = vs_authors(cr)
    print(f"  [{mode}] vs the authors: WGSL max |Δ| {wa:.3f}, argmax {wagree}/{na}   native max |Δ| {ca:.3f}, argmax {cagree}/{na}")
    if ca > 1.25 * wa + tol: print("  ⛔ the native tier sits further from the authors than the portable path"); ok = False
    if abs(cagree - wagree) > max(1, na // 50): print("  ⛔ argmax agreement with the authors differs beyond near-ties"); ok = False

controls = []
if ARCH not in ("nemotron_h",):
    controls.append(("wrong rope pairing", {"FERRIC_NEOX": "1"} if ARCH in ("llama", "muse-glimmer") else {"FERRIC_ROPE_NORM": "1"}))
if "rope_freqs.weight" in NAMES: controls.append(("rope_freqs dropped", {"FERRIC_NO_ROPE_FREQS": "1"}))
if "blk.0.attn_q.bias" in NAMES: controls.append(("q/k/v bias dropped", {"FERRIC_NO_QKV_BIAS": "1"}))
if not controls: print("  ⛔ no negative control applies — this gate has shown nothing"); ok = False
# DECODE-mode controls run on the fixture. FULL-mode controls run on a LONG PROMPT — 1024 positions, the
# first 1000 one native prefill (two 512-row chunks) — because some mechanisms only act at distance:
# dropping Llama-3's rope_freqs moved a 136-token prompt by 0.46 and 1024 positions by 2.40. (With the
# 0.1 band the rounded-f16 prefill needed, the short prompt could not show it: 5x, refused.) The long
# run is also where the chunked prefill is checked against WGSL at all.
short = dict(total=len(ref["ids"]), chunk_at=10**9)
longpf = dict(total=1024, prefill=1000, chunk_at=10**9)
lw_pf, _, _, _ = run(False, **longpf)
lc_pf, lps, _, _ = run(True, None, **longpf)
dl, tl, sl = diff(lc_pf, lw_pf)
print(f"  [FULL] 1000-row prefill (2 chunks) + 24 decode steps: native vs WGSL max |Δ| {dl:.3e} at {tl}   ssq rel {sl:.2e}   "
      f"prompt rows {lps[1][0]}/{lps[1][1]} native   (tol {TOL_PF:g})")
ok &= dl <= TOL_PF and lps[1][0] == lps[1][1] == 1000
for name, env in controls:
    for mode, extra, tol in (("DECODE", {"FERRIC_CUDA_NO_PREFILL": "1"}, TOL_NW), ("FULL", {}, TOL_PF)):
        cfg, base = (short, wr) if mode == "DECODE" else (longpf, lw_pf)
        xw, _, _, _ = run(False, env, **cfg)
        xc, xs, _, _ = run(True, {**env, **extra}, **cfg)
        same, _, _ = diff(xc, xw)
        moved, _, _ = diff(xc, {t: base[t] for t in xc})
        ran = xs[0][0] == xs[0][1] > 0 and xs[1][0] == (xs[1][1] if mode == "FULL" else 0)
        follow = tol
        print(f"  control '{name}' [{mode}]: native vs WGSL-under-control {same:.2e} (≤ {follow:.2e});  moved from normal "
              f"{moved:.2e} = {moved / tol:.0f}x tol;  ran natively: {ran}")
        if same > follow or moved < 20 * tol or not ran: print("  ⛔ the gate cannot see this mechanism on the native path"); ok = False

if LONG:
    lw, _, _, _ = run(False, total=LONG, chunk_at=LONG // 2)
    for mode, extra, tol in (("DECODE", {"FERRIC_CUDA_NO_PREFILL": "1"}, TOL_NW), ("FULL", {}, TOL_PF)):
        lc, ls, _, _ = run(True, extra, total=LONG, chunk_at=LONG // 2)
        d2, t2, s2 = diff(lc, lw)
        print(f"  LONG {LONG} positions [{mode}]: native vs WGSL max |Δ| {d2:.3e} at {t2}   ssq rel {s2:.2e}   "
              f"decode {ls[0][0]}/{ls[0][1]} native, prompt rows {ls[1][0]}/{ls[1][1]}")
        ok &= d2 <= tol and ls[0][0] == ls[0][1] > 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
