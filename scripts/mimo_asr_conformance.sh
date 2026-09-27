#!/usr/bin/env bash
# MiMo-V2.5-ASR: Ferric vs THE MODEL AUTHORS' OWN CODE (github.com/XiaomiMiMo/MiMo-V2.5-ASR in its pinned
# environment), from a 16 kHz WAV to the logits, stage by stage: the 24 kHz resample, the log-mel, the
# tokenizer encoder's recorded layers, the RVQ CODES (discrete — compared exactly), the patch encoder, the
# prompt ids Ferric tokenises itself, the LM input rows and the logits at every prompt position; then the
# greedy transcript against the authors' argmax chain.
#
# The fixture (tests/fixtures/mimo_asr/<clip>.json.gz, from crates/ferric-llama/examples/refgen/
# mimo_asr_ref.py) carries the authors' code run twice on the same audio: at float32 and at float64 — the
# noise floor. A stage passes when Ferric is within 4x of the authors' own float32 distance from float64,
# so every tolerance is measured. The only absolute floors are the fixture's own recording precision —
# logits are written to 5-6 decimals, so a floor below ~1e-5 cannot be resolved — and the gate prints how
# many checks that floor, rather than the measurement, decided. (An earlier version floored logits at
# 1e-3, which set 60 of 62 row tolerances instead of the measured 4x; found by review before commit.)
#
# ⛔ Every mechanism the gate claims to see is shown load-bearing: each negative control removes one and
# must FIRST fail (>= 20x the clean error) inside the component that mechanism belongs to.
#
#   scripts/mimo_asr_conformance.sh <asr-dir> <tokenizer-dir> <fixture.json.gz> <audio_16k.wav>
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASR="${1:-}"; TOK="${2:-}"; FX="${3:-}"; WAV="${4:-}"
[ -d "$ASR" ] && [ -d "$TOK" ] && [ -f "$FX" ] && [ -f "$WAV" ] || {
  echo "usage: $0 <asr-dir> <tokenizer-dir> <fixture.json.gz> <audio_16k.wav>"; exit 2; }
BIN="$ROOT/target/release/examples/mimo_asr_stages"
ASK="$ROOT/target/release/examples/mimo_asr_wer"
cargo build -q -p ferric-llama --release --example mimo_asr_stages --example mimo_asr_wer || exit 2

python3 - "$BIN" "$ASK" "$ASR" "$TOK" "$FX" "$WAV" <<'PY'
import gzip, hashlib, json, os, subprocess, sys, tempfile, wave
BIN, ASK, ASR, TOK, FX, WAV = sys.argv[1:7]
ref = json.load(gzip.open(FX, "rt") if FX.endswith(".gz") else open(FX))
f64 = ref["float64"]
with wave.open(WAV, "rb") as w:
    pcm = w.readframes(w.getnframes())
if hashlib.sha256(pcm).hexdigest() != ref["pcm_sha256"]:
    print("⛔ the audio is not the clip the reference ran on (int16 PCM sha256 differs)"); sys.exit(2)
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(ref, tmp); tmp.close()
TOKSTAGES = ["mel", "conv_out"] + sorted((k for k in ref["stages"] if k.startswith("enc_layer")), key=lambda k: int(k[9:])) \
            + ["enc_norm", "pooled"]
LMSTAGES = ["patch_encoder", "inputs_embeds"]

def run(env_extra=None, tok_only=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FERRIC_ASR_", "FERRIC_ROPE_NORM", "FERRIC_NEOX",
                                                                     "FERRIC_METAL4", "FERRIC_COOP"))}
    env.update(env_extra or {})
    if tok_only: env["FERRIC_ASR_TOK_ONLY"] = "1"
    r = subprocess.run([BIN, ASR, TOK, WAV, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
    o = {"stages": {}, "codes": {}, "rows": []}
    for l in r.stdout.splitlines():
        p = l.split(" ")
        if p[0] == "WAV24": o["wav24"] = (int(p[1]), float(p[3]), [float(x) for x in p[4:]])
        elif p[0] == "STAGE": o["stages"].setdefault(p[1], []).append([float(x) for x in p[3:]])
        elif p[0] == "CODES": o["codes"][int(p[1])] = [int(x) for x in p[2:]]
        elif p[0] == "TEXTIDS": o["text_ids"] = [int(x) for x in p[1:]]
        elif p[0] == "ROW": o["rows"].append((int(p[2]), [float(x) for x in p[5:]]))
    return o

def stage_err(got, rec):
    if got is None or len(got) != len(rec["ssq"]): return (float("inf"), float("inf"))
    scale = max(abs(v) for r in rec["sample"] for v in r)
    s = max(abs(a - b) for g, r in zip(got, rec["sample"]) for a, b in zip(g[2:], r)) / scale
    q = max(abs(g[1] - r) / max(r, 1e-30) for g, r in zip(got, rec["ssq"]))
    return (s, q)

def as_rows(rec): return [[a, b] + c for a, b, c in zip(rec["sum"], rec["ssq"], rec["sample"])]
def logit_err(rows, recs): return [max(abs(a - b) for a, b in zip(r[1], rr["sample"])) for r, rr in zip(rows, recs)]

def E(o, name):
    if name == "codes":
        return float(sum(a != b for c in range(8) for a, b in zip(o["codes"].get(c, []), ref["codes"][c])))
    if name == "logits":
        return max(logit_err(o["rows"], ref["rows"])) if len(o["rows"]) == len(ref["rows"]) else float("inf")
    return max(stage_err(o["stages"].get(name), ref["stages"][name]))

o = run()
ok = True
print(f"reference: {ref['model']} — {ref['code']}\n           authors' commit {ref['env']['authors_commit'][:10]}, torch "
      f"{ref['env']['torch']}, torchaudio {ref['env']['torchaudio']}, transformers {ref['env']['transformers']}; "
      f"{ref['precision']}")
print(f"           shims: {'; '.join(ref['shims'])}\n           corrections: {'; '.join(ref['corrections'])}")
print(f"harness self-test (streamed vs the authors' forward, tiny config): {ref['harness_selftest']['max_abs_last_logit_diff']:g}")
n, q, vals = o["wav24"]
wd = max(abs(a - b) for a, b in zip(vals, ref["wav24"]["values"]))
print(f"  resample 16 -> 24 kHz              {n} samples, max |d| {wd:.1e}   (tol 1e-6)"); ok &= n == ref["wav24"]["n"] and wd <= 1e-6
ncodes = sum(len(c) for c in ref["codes"])
bad_codes = int(E(o, "codes"))
print(f"  RVQ codes (8 channels, exact)      {ncodes - bad_codes}/{ncodes} equal   (the authors' own bf16 run: "
      f"{100 * ref['codes_bf16_agreement']:.1f}%; float64: {100 * f64['codes_agree_with_f32']:.1f}%)")
ok &= bad_codes == 0
tid = o.get("text_ids") == ref["text_ids"]
print(f"  prompt ids (Ferric's own tokenizer) {'identical' if tid else '⛔ DIFFER'} ({len(ref['text_ids'])} positions)"); ok &= tid

print(f"\n  {'stage':15s} {'vs authors f32':>22s} {'vs float64':>22s} {'authors f32 vs f64':>22s}   (sampled-rel / ssq-rel)")
for name in TOKSTAGES + LMSTAGES:
    rec, rec64 = ref["stages"][name], f64["stages"][name]
    a = stage_err(o["stages"].get(name), rec); b = stage_err(o["stages"].get(name), rec64); fl = stage_err(as_rows(rec), rec64)
    tol = [max(4 * x, 1e-7) for x in fl]      # sampled values carry 8 significant digits
    good = all(x <= t for x, t in zip(b, tol)); ok &= good
    print(f"  {name:15s} {a[0]:9.2e} / {a[1]:9.2e} {b[0]:9.2e} / {b[1]:9.2e} {fl[0]:9.2e} / {fl[1]:9.2e}  {'' if good else '⛔ beyond 4x the floor'}")

if len(o["rows"]) != len(ref["rows"]):
    print(f"⛔ Ferric produced {len(o['rows'])} logit rows for {len(ref['rows'])} positions"); sys.exit(1)
e32, e64 = logit_err(o["rows"], ref["rows"]), logit_err(o["rows"], f64["rows"])
fl = [max(abs(a - b) for a, b in zip(r["sample"], r64["sample"])) for r, r64 in zip(ref["rows"], f64["rows"])]
REC = 2e-5   # two recording quanta: f32 rows at 5 decimals, f64 rows at 6
bad = [t for t, (x, f) in enumerate(zip(e64, fl)) if x > max(4 * f, REC)]
by_floor = sum(4 * f < REC for f in fl)
worst_ratio = max(x / max(f, 1e-12) for x, f in zip(e64, fl))
am = [t for t, (r, r64, f) in enumerate(zip(o["rows"], f64["rows"], fl))
      if r64["top"][0][1] - r64["top"][1][1] > max(4 * f, REC) and r[0] != r64["top"][0][0]]
sp = [t for t, x in enumerate(ref["is_speech"]) if x]
seg = {"text before the speech": [t for t in range(len(e32)) if t < sp[0]], "speech": sp,
       "text after the speech": [t for t in range(len(e32)) if t > sp[-1]]}
print(f"\n  logits, {len(e32)} positions x {len(ref['sample_ids'])} sampled ids (vocab {ref['vocab']}):")
for k, idx in seg.items():
    print(f"    {k:23s} ({len(idx):3d})  vs authors f32 {max(e32[t] for t in idx):.2e}   vs float64 "
          f"{max(e64[t] for t in idx):.2e}   authors f32 vs f64 {max(fl[t] for t in idx):.2e}")
print(f"    rows beyond 4x the floor: {len(bad)} (worst row at {worst_ratio:.2f}x; {by_floor} rows set by the {REC:g} "
      f"recording floor instead)   argmax disagreements where f64's margin is decisive: {len(am)}")
ok &= not bad and not am

# ⭐ NEGATIVE CONTROLS: (label, env, first stage allowed, last stage allowed)
ORDER = TOKSTAGES + ["codes"] + LMSTAGES + ["logits"]
controls = [("magnitude -> POWER spectrum", {"FERRIC_ASR_NEG": "mel_power"}, "mel", "mel"),
            ("zero padding at the STFT edges (not reflect)", {"FERRIC_ASR_NEG": "pad_zero"}, "mel", "mel"),
            ("key projection given a bias", {"FERRIC_ASR_NEG": "k_bias"}, "enc_layer0", "enc_layer0"),
            ("rope at EXACT frequencies (not their bf16 table)", {"FERRIC_ASR_NEG": "rope_exact"}, "enc_layer0", "pooled"),
            ("no long skip in the encoder", {"FERRIC_ASR_NEG": "no_skip"}, "enc_norm", "enc_norm"),
            ("causal attention inside a patch", {"FERRIC_ASR_NEG": "patch_causal"}, "patch_encoder", "patch_encoder"),
            ("patch rope at base 10000 (not 640000)", {"FERRIC_ASR_NEG": "patch_rope_base"}, "patch_encoder", "patch_encoder"),
            ("patch encoder's three bf16 islands removed", {"FERRIC_ASR_NEG": "patch_f32"}, "patch_encoder", "patch_encoder"),
            ("LM rope pairing interleaved (not NEOX)", {"FERRIC_ROPE_NORM": "1"}, "logits", "logits")]
clean = {s: E(o, s) for s in ORDER}
print("\n  controls (must first fail >= 20x the clean error inside their own component):")
for label, env, lo, hi in controls:
    tok_only = ORDER.index(hi) <= ORDER.index("codes")
    c = run(env, tok_only=tok_only)
    stages = ORDER[:ORDER.index("codes") + 1] if tok_only else ORDER
    def ratio(s):
        if s == "codes": return float("inf") if E(c, s) > 0 and clean[s] == 0 else (E(c, s) / clean[s] if clean[s] else 1.0)
        return E(c, s) / max(clean[s], 1e-12)
    ratios = {s: ratio(s) for s in stages}
    first = next((s for s in stages if ratios[s] >= 20), None)
    good = first is not None and ORDER.index(lo) <= ORDER.index(first) <= ORDER.index(hi)
    ok &= good
    shown = ratios.get(first, float("nan")) if first else float("nan")
    print(f"    {label:50s} first fails at {str(first):14s} {shown:>12,.0f}x  {'' if good else '⛔ expected ' + lo + '..' + hi}")

GREEDY = FX[:-len(".json.gz")] if FX.endswith(".json.gz") else FX[:-len(".json")] if FX.endswith(".json") else FX
GREEDY += ".greedy.json"
if os.path.exists(GREEDY):
    g = json.load(open(GREEDY))
    r = subprocess.run([ASK, ASR, TOK, WAV, GREEDY], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if not k.startswith("FERRIC_")})
    # `MATCH <prefix>/<n> LEN <ferric> <authors>`: the prefix alone cannot see a run that fails to STOP
    # (zip ends at the shorter list), so both lengths must equal n too.
    m = [l for l in r.stdout.splitlines() if l.startswith("MATCH ")]
    lens = [int(x) for x in m[-1].split()[3:5]] if m else [-1, -1]
    got = int(m[-1].split()[1].split("/")[0]) if m else -1
    n = len(g["continuation"])
    print(f"\n  greedy transcript, {n} tokens (the authors' argmax at every step; smallest margin {min(g['margin']):.3f}):"
          f" Ferric matches {got}/{n}\n    \"{g['decoded'].strip()}\"")
    ok &= got == n and lens == [n, n] and all(g["agrees"])
    if lens != [n, n]: print(f"    ⛔ Ferric produced {lens[0]} tokens (stop included) where the authors' chain has {n}")
else:
    print(f"\n  ⛔ no {os.path.basename(GREEDY)}: the decode cannot be checked, so this gate cannot pass"); ok = False
os.unlink(tmp.name)
print("\n✅ agrees with the authors, waveform to transcript" if ok else "\n⛔ MiMo-ASR conformance FAILED")
sys.exit(0 if ok else 1)
PY
