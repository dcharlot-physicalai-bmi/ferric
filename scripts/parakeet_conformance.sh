#!/usr/bin/env bash
# Parakeet speech (arch "parakeet" = Conformer + RNN-T, arch "asr" = Conformer + CTC): Ferric vs THE MODEL
# AUTHORS' OWN IMPLEMENTATION, NVIDIA NeMo, at every stage — with every silent-failure mechanism proved
# load-bearing by a negative control. Three checkpoints: parakeet-unified-en-0.6b (RNN-T, offline),
# parakeet-ctc-1.1b (CTC) and nemotron-3.5-asr-streaming-0.6b (RNN-T; limited-context attention, causal
# conv and subsampling, raw log-mel, and a language PROMPT — fixtures at both its declared context (56,13)
# and NeMo's default (56,3), and with prompt en-US as well as auto).
#
# ⛔⛔ A TRANSCRIPT IS THE LEAST SENSITIVE PLACE TO LOOK. `parakeet` was Verified on "every word correct on
# three LibriSpeech utterances" and a 1.77% corpus WER. Checked against NeMo stage by stage, Ferric counted
# every STFT frame as audio where NeMo counts n/hop of them, normalised over the extra frame, masked
# nothing, and decoded an encoder frame NeMo treats as padding. Inside NeMo, that rule moves the encoder by
# 0.13-1.4x its rms on the unified model and 0.17-3.6x on the CTC one — and the tokens were identical on
# every clip in the fixtures. Only a comparison of numbers can see it.
#
# THE REFERENCE is NeMo 3.0.0 running THIS GGUF's own dequantised weights ("NeMo-G"), injected after a
# by-value audit against the authors' checkpoint: every F16 tensor is bit-exact f16(checkpoint), F32 is
# exact, the LSTM bias is exactly bias_ih + bias_hh. So a gap here is a CODE difference, never rounding.
# The weight-rounding cost itself is reported in each fixture's "arms" (NeMo-ckpt), not gated: 5e-3 of the
# encoder rms for the F16 unified file, 0.10-0.38 for the Q8_0 CTC file. Fixtures are written by
# crates/ferric-llama/examples/refgen/parakeet_ref.py; the reference numbers are committed, so this gate
# needs only the GGUF and the audio — no Python ML stack, no NeMo.
#
#   scripts/parakeet_conformance.sh <model.gguf> <fixture.json[.gz]> <audio.wav|flac> [--no-controls]
#
# The audio is the LibriSpeech utterance the fixture names (clip.path_hint; OpenSLR test-clean, CC-BY 4.0)
# or the 16-bit WAV of exactly those samples the generator writes. The int16 PCM hash must match. The
# fixture's "config" (attention context, prompt index) is applied to Ferric before it runs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-}"; FX="${2:-}"; AUDIO="${3:-}"; OPT="${4:-}"
[ -f "$M" ] && [ -f "$FX" ] && [ -f "$AUDIO" ] || {
  echo "usage: $0 <model.gguf> <fixture.json[.gz]> <audio.wav|flac> [--no-controls]"; exit 2; }
BIN="$ROOT/target/release/examples/parakeet_stages"
[ -x "$BIN" ] || cargo build -q -p ferric-llama --release --example parakeet_stages || exit 2
# Both switch in reduced-precision kernels; the tolerances below are for f32 arithmetic.
unset FERRIC_METAL4 FERRIC_COOP

python3 - "$BIN" "$M" "$FX" "$AUDIO" "$OPT" <<'PY'
import gzip, json, math, os, struct, subprocess, sys, tempfile
BIN, M, FX, AUDIO, OPT = sys.argv[1:6]

raw = gzip.open(FX, "rt").read() if FX.endswith(".gz") else open(FX).read()
ref = json.loads(raw)
if ref.get("format") != "parakeet-conformance/1":
    print(f"⛔ {FX}: not a parakeet-conformance/1 fixture"); sys.exit(2)
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); tmp.write(raw); tmp.close()

def arch(path):
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
            if k == 'general.architecture' and t == 8: return rstr(f)
            skip(f, t)
ARCH = arch(M)
if ARCH not in ("parakeet", "asr"):
    print(f"⛔ {os.path.basename(M)} is arch {ARCH!r}, not a parakeet speech model"); sys.exit(2)
# The settings the reference ran at — which also say which mechanisms this model HAS, and so which
# controls can apply. Required: a fixture without them would have the gate assume a configuration.
CFG = ref.get("config")
if CFG is None:
    print("⛔ fixture has no \"config\" (attention context, prompt, subsampling, norms) — regenerate it"); sys.exit(2)
PROMPT = CFG.get("prompt_id") is not None
CAUSAL_CONV = str(CFG.get("conv_context")) == "causal"
RAW_MEL = str(CFG.get("normalize")) in ("NA", "none")

# ⭐ TOLERANCES, per stage family: (max|d| over the sampled values / the stage's rms, worst relative
# difference in a recorded row's sum of squares over its FULL width). Each is ~3x the worst CLEAN residual
# measured on the four fixture clips, against the same weights — so what remains is arithmetic:
#   unified (F16):  logmel 5.8e-5 / 8.1e-6 — Ferric's own f32 FFT sets this floor in low-energy bins;
#                   mel 3.0e-4 / 7.4e-6, pre_conv 3.0e-4 / 4.9e-6, pre_out 7.7e-5 / 3.6e-6,
#                   pe 3.2e-6 / 1.5e-8, block 4.6e-5 / 1.4e-6, enc 1.5e-5 / 3.7e-6; joint logits 1.0e-4 abs
#   ctc (Q8_0):     logmel 1.7e-5 / 6.8e-6, mel 8.6e-5 / 6.7e-6, pre_conv 2.1e-4 / 1.0e-5,
#                   pre_out 6.2e-5 / 3.7e-6, pe 1.3e-6 / 1.5e-8, block 7.5e-5 / 1.9e-6, enc 1.6e-4 / 1.9e-5
#                   (42 layers); head logits 9.2e-4 abs
#   nemotron-3.5-asr-streaming (F16, prompt model): encoder 2.3e-5 / 2.5e-6, then the language-prompt MLP
#                   amplifies: prompt 1.3e-4 / 3.3e-5 — and NeMo's OWN fp32-vs-fp64 floor there is 1.5e-4..8.0e-4
#                   of the prompt rms. Ferric sits at or below it; the looser pair is the model's arithmetic.
# HEAD LOGITS are compared RELATIVE TO EACH ROW'S SCALE, max|d| / max(1, |row max|): measured 9.7e-5 (unified),
# 1.8e-5 (ctc), 7.1e-5 (nemotron). ⚠ An absolute tolerance was the wrong measure: nemotron's third joint call
# on a 33 s clip (a near-silent opening frame) has logits near -783, and its 1.67e-2 absolute gap — 3x NeMo's
# own fp32-vs-fp64 gap on that FIXTURE — is 2.1e-5 of that row, like every other row. Absolute, the tolerance
# had to be wide enough for that row, and was then 60x loose on the ordinary ones.
# NeMo's own fp32-vs-fp64 floor at the encoder is 4e-5 to 6.5e-5 (unified) and 1e-5 to 2e-5 (ctc) of the
# rms, so Ferric sits at or below the reference's own arithmetic on the unified model.
# ⚠ The SUM-OF-SQUARES term is not decoration: a pure scale error — population instead of Bessel variance
# on a 33 s clip, 1/L = 3e-4 — hides under the FFT floor on max|d| and stands 40x clear of it here.
TOL = {
  "parakeet": {"logmel": (2e-4, 3e-5), "mel": (1e-3, 3e-5), "pre_conv": (1e-3, 2e-5), "pre_out": (3e-4, 1.5e-5),
               "pe": (1e-5, 1e-7), "block": (2e-4, 5e-6), "enc": (5e-5, 1.5e-5), "logits": 3e-4},
  "asr":      {"logmel": (6e-5, 2e-5), "mel": (3e-4, 2e-5), "pre_conv": (7e-4, 3e-5), "pre_out": (2e-4, 1.5e-5),
               "pe": (1e-5, 1e-7), "block": (3e-4, 6e-6), "enc": (5e-4, 6e-5), "logits": 6e-5},
  "prompt":   {"logmel": (2e-4, 3e-5), "mel": (1e-3, 3e-5), "pre_conv": (1e-3, 2e-5), "pre_out": (3e-4, 1.5e-5),
               "pe": (1e-5, 1e-7), "block": (2e-4, 5e-6), "enc": (6e-5, 1.5e-5), "prompt": (4e-4, 1e-4),
               "logits": 2.5e-4},
}["prompt" if PROMPT else ARCH]
fam = lambda s: "block" if s.startswith("block.") else s
ORDER = [s for s in ["logmel", "mel", "pre_conv", "pre_out", "pe"] if s in ref["stages"]] + \
        sorted([s for s in ref["stages"] if s.startswith("block.")], key=lambda s: int(s.split(".")[1])) + \
        [s for s in ["enc", "prompt"] if s in ref["stages"]]

def run(env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FERRIC_ASR_NEG_")}
    env.update(env_extra or {})
    r = subprocess.run([BIN, M, AUDIO, tmp.name], capture_output=True, text=True, env=env)
    if r.returncode == 3:
        print("⛔ the audio is not the clip the reference ran on (int16 PCM sha256 differs)"); sys.exit(2)
    if r.returncode:
        return None, r.stderr[-800:]
    o = {"stages": {}, "steps": [], "ctc": {}, "tokens": None, "valid": None, "inert": None, "shape": {}}
    for l in r.stdout.splitlines():
        p = l.split(" ")
        if p[0] == "STAGE": o["stages"].setdefault(p[1], {})[int(p[2])] = (float(p[3]), float(p[4]), [float(x) for x in p[5:]])
        elif p[0] == "SHAPE": o["shape"][p[1]] = (int(p[2]), int(p[3]))
        elif p[0] == "STEP": o["steps"].append((int(p[4]), [float(x) for x in p[7:]]))
        elif p[0] == "CTC": o["ctc"][int(p[1])] = (int(p[2]), [float(x) for x in p[5:]])
        elif p[0] == "TOKENS": o["tokens"] = [int(x) for x in p[1:]]
        elif p[0] == "VALID": o["valid"] = (int(p[1]), int(p[2]))
        elif p[0] == "TAPS_INERT": o["inert"] = float(p[1])
    return o, None

def bad(x): return not (x == x and abs(x) != float("inf"))

def checks(o):
    """Every check in pipeline order: (name, family, measured, tolerance, ok). A NaN or a missing row is
    a failure, never a comparison that happens to evaluate False."""
    out = []
    for name in ORDER:
        s, got = ref["stages"][name], o["stages"].get(name, {})
        tol_d, tol_q = TOL[fam(name)]
        d = q = 0.0
        for i, r in enumerate(s["rows"]):
            g = got.get(r)
            if g is None or bad(g[1]) or any(bad(v) for v in g[2]) or len(g[2]) != len(s["cols"]):
                d = q = float("inf"); break
            d = max(d, max(abs(a - b) for a, b in zip(g[2], s["vals"][i])) / s["rms"])
            q = max(q, abs(g[1] - s["row_ssq"][i]) / max(1e-30, s["row_ssq"][i]))
        if not s["rows"]: d = q = float("inf")          # a stage with no rows checks nothing
        ok = d <= tol_d and q <= tol_q
        out.append((name, fam(name), max(d / tol_d, q / tol_q), f"{d:.2e} / {q:.2e}", ok))
    if "rnnt" in ref:
        R = ref["rnnt"]
        if len(o["steps"]) != len(R["steps"]) or not R["steps"]:
            out.append(("rnnt logits", "head", float("inf"), f"{len(o['steps'])} of {len(R['steps'])} steps", False))
        else:
            w = max(max(abs(a - b) if not (bad(a) or bad(b)) else float("inf") for a, b in zip(v, R["vals"][k]))
                    / max(1.0, abs(R["row_max"][k])) for k, (_, v) in enumerate(o["steps"]))
            am = sum(a == R["steps"][k][2] for k, (a, _) in enumerate(o["steps"]))
            out.append(("rnnt logits", "head", w / TOL["logits"], f"{w:.2e} of row", w <= TOL["logits"]))
            out.append(("rnnt argmax", "head", float(len(R["steps"]) - am), f"{am}/{len(R['steps'])}", am == len(R["steps"])))
        out.append(("tokens", "tokens", 0.0 if o["tokens"] == R["tokens"] else float("inf"),
                    f"{len(o['tokens'] or [])} vs {len(R['tokens'])}", o["tokens"] == R["tokens"]))
    else:
        C = ref["ctc"]; rows = {r: i for i, r in enumerate(C["rows"])}
        if len(o["ctc"]) != len(C["argmax"]):
            out.append(("ctc logits", "head", float("inf"), f"{len(o['ctc'])} of {len(C['argmax'])} frames", False))
        else:
            w = max((max(abs(a - b) if not (bad(a) or bad(b)) else float("inf") for a, b in zip(o["ctc"][t][1], C["vals"][i]))
                     / max(1.0, abs(C["row_max"][t])) for t, i in rows.items()), default=float("inf"))
            am = sum(o["ctc"][t][0] == a for t, a in enumerate(C["argmax"]))
            out.append(("ctc logits", "head", w / TOL["logits"], f"{w:.2e} of row", w <= TOL["logits"]))
            out.append(("ctc argmax", "head", float(len(C["argmax"]) - am), f"{am}/{len(C['argmax'])}", am == len(C["argmax"])))
        out.append(("tokens", "tokens", 0.0 if o["tokens"] == C["tokens"] else float("inf"),
                    f"{len(o['tokens'] or [])} vs {len(C['tokens'])}", o["tokens"] == C["tokens"]))
    return out

print(f"reference: {ref['model']} — {ref['reference']} (NeMo {ref['nemo']}, torch {ref['torch']}) on {ref['gguf']}")
print(f"config:    {json.dumps(CFG)}")
print(f"clip:      {ref['clip']['id']}  {ref['clip']['n_samples']} samples  valid {ref['valid']['mel_frames']} mel / "
      f"{ref['valid']['enc_frames']} encoder frames   weights audit: {json.dumps(ref['weights_audit'].get('rules', {}))}")
o, err = run()
if o is None:
    print("⛔ Ferric failed to run:\n" + err); sys.exit(1)
ok = True
if o["valid"] != (ref["valid"]["mel_frames"], ref["valid"]["enc_frames"]):
    print(f"⛔ valid lengths: Ferric {o['valid']}, NeMo {(ref['valid']['mel_frames'], ref['valid']['enc_frames'])}"); ok = False
if o["inert"] != 0.0:
    print(f"⛔ the taps changed the encoder output by {o['inert']} — every number below describes another run"); ok = False
clean = checks(o)
first = next((c for c in clean if not c[4]), None)
print(f"\n{'stage':12} {'max|d|/rms / ssq-rel':>24}   result")
for name, f_, ratio, shown, good in clean:
    if name.startswith("block.") and good and name not in ("block.0", ORDER[-2]): continue
    print(f"  {name:12} {shown:>24}   {'ok' if good else '⛔ FAIL (x%.1f tol)' % ratio}")
nblk = sum(1 for c in clean if c[0].startswith("block."))
if nblk: print(f"  ({nblk} blocks checked; the first and last are shown, any failing one is shown)")
if first:
    print(f"⛔ FIRST FAILING STAGE: {first[0]}"); ok = False

# ⭐ NEGATIVE CONTROLS. Each re-introduces ONE known-wrong convention in Ferric (crates/ferric-llama/src/
# parakeet.rs, `Neg`) and must fail FIRST at the stage where its mechanism lives — not merely somewhere,
# which a control that broke an unrelated upstream stage would also do. Every one names a convention some
# version of this port, or another port, actually got wrong while the transcript stayed fluent.
ENC = "encoder"
# `where` is the family the control must fail FIRST in. VALIDLEN's extra frame is invisible in the mel of a
# RAW-log-mel model (no statistics to shift, and the fixture records only valid rows), so there it first
# shows wherever the frame reaches a valid row downstream — which depends on the length class. When it adds
# a whole ENCODER frame (causal subsampling, n/hop ≡ 1 mod 8), the first thing it changes is `pe`: the
# relative-position table is built for T frames, and T is now one more.
CONTROLS = [
    ("MEL", "htk", "logmel"), ("FBNORM", "peak", "logmel"), ("WINDOW", "periodic", "logmel"),
    ("PAD", "reflect", "logmel"), ("POWER", "magnitude", "logmel"), ("PREEMPH", "0", "logmel"),
    ("LOGGUARD", "1e-9", "logmel"), ("NORM", "biased", "mel"), ("NORMALIZE", "flip", "mel"),
    ("VALIDLEN", "all_frames", ("pre_conv", "pre_out", "pe", ENC, "prompt") if RAW_MEL else "mel"),
    ("FLATTEN", "freq_major", "pre_conv"), ("PRECONV", "swap_axes", "pre_conv"), ("RELU", "dw", "pre_conv"),
    ("SUBPAD", "time_symmetric", "pre_conv"),
    ("XSCALE", "flip", "pre_out"), ("PE", "ascending", "pe"),
    ("RELSHIFT", "none", ENC), ("POSBIAS", "swap_uv", ENC), ("CONVPAD", "symmetric" if CAUSAL_CONV else "causal", ENC),
    ("CONVNORM", "bn_eps1e-3", ENC), ("CONVNORM", "ln_affine", ENC), ("ATTCTX", "full", ENC),
    ("GLU", "swap", ENC), ("MACARON", "1.0", ENC), ("DW1D", "flip", ENC),
    ("PROMPT", "off", "prompt"), ("PROMPT", "concat_first", "prompt"), ("PROMPT", "id0", "prompt"),
    ("LSTM_GATES", "ifog", "head"), ("SOS", "no_step", "head"), ("BLANK", "update_state", "head"),
    ("JOINT_ACT", "tanh", "head"), ("MAXSYM", "1", "tokens"), ("CTC_TIE", "last", "head"),
]
# A control must clear its stage's tolerance by this factor. The tolerance is ~3x the clean residual, so
# this asks for ~6x the residual: a control that lands just over the line could pass or fail on the next
# clip by arithmetic alone, and proves nothing either way. The weakest measured is CONVNORM=bn_eps1e-3
# (the TF/Keras BatchNorm default) at ~x3 on the unified model: real, and the smallest one here.
MARGIN = 2.0
def not_applicable(name, val):
    if name == "NORM" and RAW_MEL: return "the model takes the RAW log-mel: there is no variance to compute"
    if name == "SUBPAD" and not CFG.get("causal_subsampling"): return "symmetric subsampling has no causal padding to move"
    if name == "CONVNORM" and val == "bn_eps1e-3" and CFG.get("conv_norm") == "layer_norm":
        return "the conv module is LayerNorm: no BatchNorm eps"
    if name == "CONVNORM" and val == "ln_affine" and CFG.get("conv_norm") != "layer_norm":
        return "the conv module is BatchNorm, folded to an affine by construction"
    if name == "ATTCTX" and not CFG.get("att_context"): return "full-context attention: there is no mask to remove"
    if name == "PROMPT" and not PROMPT: return "this model takes no language prompt"
    if name == "PROMPT" and val == "id0" and CFG.get("prompt_id") == 0: return "the fixture already uses index 0"
    rnnt_only = {"LSTM_GATES", "SOS", "BLANK", "JOINT_ACT", "MAXSYM"}
    if name in rnnt_only and "rnnt" not in ref: return "a CTC model has no predictor or joint"
    if name == "MAXSYM" and ref.get("decode_checks", {}).get("max_emissions_at_one_frame", 0) < 2:
        return "no frame of this clip emits 2 or more tokens, so a cap of 1 cannot bind"
    if name == "CTC_TIE":
        return ("needs an EXACT tie between two logits, which real audio does not produce; the tie rule is "
                "covered by parakeet::tests::ctc_argmax_keeps_the_first_maximum_on_a_tie" if "ctc" in ref
                else "an RNN-T model has no CTC head")
    if name == "CONVPAD" and "block.0" not in ref["stages"] and "enc" not in ref["stages"]:
        return "no encoder stage recorded"
    return None

if OPT != "--no-controls":
    print(f"\nnegative controls (each must fail FIRST where its mechanism lives):")
    applied = 0
    for name, val, where in CONTROLS:
        na = not_applicable(name, val)
        if na:
            print(f"  {name + '=' + val:<22} not applicable: {na}"); continue
        applied += 1
        co, err = run({f"FERRIC_ASR_NEG_{name}": val})
        if co is None:
            print(f"  {name + '=' + val:<22} ⛔ Ferric failed to run: {err.strip().splitlines()[-1] if err.strip() else '?'}"); ok = False; continue
        cf = next((c for c in checks(co) if not c[4]), None)
        fam_of = lambda c: ENC if c[1] in ("block", "enc") else c[1]
        lab = f"{name}={val}"
        wheres = where if isinstance(where, tuple) else (where,)
        if cf is None:
            print(f"  {lab:<22} ⛔ PASSES THE GATE — this gate cannot see that mechanism"); ok = False
        elif fam_of(cf) not in wheres:
            print(f"  {lab:<22} ⛔ fails first at {cf[0]}, not at {where} — it is not testing what it names"); ok = False
        elif cf[2] < MARGIN:
            print(f"  {lab:<22} ⛔ fails at {cf[0]} by only x{cf[2]:.1f} tol — too close to the clean residual to "
                  f"count"); ok = False
        else:
            print(f"  {lab:<22} fails at {cf[0]:<12} {cf[3]:>22}  (x{cf[2]:,.0f} tol)")
    if applied == 0:
        print("  ⛔ no negative control applies — a match with nothing shown load-bearing proves little"); ok = False

os.unlink(tmp.name)
if not ok:
    print("\n⛔ parakeet conformance FAILED")
elif OPT == "--no-controls":
    print("\n✅ agrees with NeMo (the authors' code) at every stage — ⚠ negative controls NOT run, so this run "
          "does not show the gate can fail")
else:
    print("\n✅ agrees with NeMo (the authors' code) at every stage; every applicable control is load-bearing")
sys.exit(0 if ok else 1)
PY
