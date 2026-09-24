#!/usr/bin/env python3
"""Stage-by-stage Parakeet reference from the MODEL AUTHORS' OWN implementation, NVIDIA NeMo, for the
`parakeet` (Conformer + RNN-T) and `asr` (Conformer + CTC) archs. Writes one fixture per model x clip, read by
`examples/parakeet_stages.rs` and the gate.

⛔⛔ WHY THIS EXISTS. `parakeet` was marked Verified on transcripts and WER alone, and `asr` only Loads. A
transcript is the least sensitive place to look: inside NeMo, Ferric's frame semantics (all 1+N//160 STFT
frames valid, no subsampling / attention / conv masks, every encoder frame decoded) move the CTC encoder by up
to 3.6x its rms and its log-probs by 25 — and on every clip tried, the collapsed tokens did not change. So the
reference is recorded at every stage, and the gate compares numbers, not words.

⛔ THIS LIVES IN THE REPO ON PURPOSE: a generator kept in scratch is a verification nobody can repeat.

Run (NeMo 3.0.0 on CPU; the 1.1B CTC model needs ~25 GB of RAM):
    <nemo-venv>/bin/python parakeet_ref.py <model.gguf> <model.nemo> <audio_root> <out_dir> [clip-id ...]
e.g. parakeet_ref.py ~/.cache/ferric/parakeet-0.6b-f16.gguf .../parakeet-unified-en-0.6b.nemo .../audio out/
`audio_root` holds LibriSpeech/test-clean/... (OpenSLR, CC-BY 4.0) and dev-clean-dummy/1272-128104-0000.wav
(PCM16 from hf-internal-testing/librispeech_asr_dummy). Each clip's int16 PCM sha256 goes into the fixture, and
a 16-bit WAV of exactly those samples is written next to it for the Ferric side. Regenerate by re-running: the
column samples are seeded (20260924), so the same NeMo gives the same fixture.

THE REFERENCE ("NeMo-G"): NeMo 3.0.0 as installed, running the GGUF's own dequantised weights, injected after a
by-value audit against the authors' checkpoint (parakeet_weights.py). That isolates CODE differences from WEIGHT
differences. NeMo 3.0.0's valid-length semantics are the reference semantics: L = N//160 valid mel frames
(stats over L, Bessel; frames >= L zeroed), masked subsampling, attention pad mask, conv pad zeroing, and only
encoded_len frames decoded.

ARMS, reported and never gated, each as max|d| and max|d|/rms over the reference's valid domain:
    NeMo-ckpt  the authors' fp32 checkpoint: the cost of the GGUF's weight rounding
    G64        model.double(), preprocessor kept fp32: the reference's own arithmetic floor
    F64        the parts NeMo computes in fp32 even under G64: the featurizer in fp64, and the exact pe table
    B          Ferric's OLD semantics (get_seq_len + 1, so no masks and every frame decoded): A - B is the
               defect, measured inside the authors' code
    V119       (CTC) B + reflect-padded STFT: the NeMo 1.19 frontend parakeet-ctc-1.1b was trained under

GUARDS, each added because the trap was measured on this model, in this NeMo:
  - transcribe() is never called, and calling it raises. Its teardown ends in module.train() (unfreeze at
    parts/mixins/transcription.py:795-801 -> core/classes/module.py:66): dropout goes live in encoder, decoder
    and joint while m.training still reads False. The encoder moved by 2.09 and 'Mister' became 'Mr'.
  - no module may be in training mode at any capture (checked module by module, and shown to fire).
  - every input is a copy, and the caller's PCM is hashed before and after: a train-mode featurizer dithers
    the CALLER's waveform in place (features.py:427; 4.6e-5 measured). The trap is re-demonstrated each run.
  - a featurizer is never deep-copied as is: copy.deepcopy(FilterbankFeatures) silently runs the ORIGINAL
    (featurizer_copy). This run's first dither demonstration "passed" that way, on the eval-mode original.
  - raw joint logits need m.joint.log_softmax = False: with None (the config's value) the joint applies
    log_softmax ON CPU ONLY (rnnt.py joint_after_projection), and every logit moves by its logsumexp.
    Recorded rows are checked to be raw (a log-softmaxed row has logsumexp 0).
  - the RNN-T visit order is re-implemented explicitly (label looping: blank advances t and keeps the
    predictor state; after max_symbols emissions at one t, t advances without a blank), its tokens must EQUAL
    m.decoding.rnnt_decoder_predictions_tensor on the same encoder output, and the logits TEACHER-FORCED along
    NeMo's own token prefix must reproduce every greedy argmax. The fixture records the teacher-forced ones.
  - the CTC head's raw logits are taken BEFORE ConvASRDecoder's unconditional log_softmax (conv_asr.py).
  - pe is NeMo's live table (a non-persistent buffer, recomputed at init), never the GGUF's
    encoder.pos_enc.pe; the audit compares the two and reports the difference.

FIXTURE (format parakeet-conformance/1): per stage the sampled rows and columns, their values (6 to 8
significant digits, per DIGITS and recorded as "digits"), and row_sum / row_ssq over the FULL width of every
recorded row. Rows are the reference's frame
indices over VALID frames; "pe" row r is relative position (T-1)-r ("pos_first"). The fixture must stay under
400 KB, so rows and columns follow the task's plan (32 columns; logmel/mel every 8th frame + the last 16;
encoder-rate stages every frame; blocks every 4th + the last 4) until it does not fit, and then a fixed ladder
of reductions (LADDER) applies in order; the fixture records how far down it went ("sampling"). Full-precision
arrays of every reference stage are saved beside the fixtures (<out>/full/*.npz) for debugging a failure.
"""
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parakeet_weights as pw  # noqa: E402

SEED = 20260924
LIMIT = 400_000                    # bytes per fixture
BLOCK_CLIP = "1089-134686-0000"    # the only clip whose every Conformer block is recorded
CLIPS = {                          # id -> (file under audio_root, LibriSpeech path hint)
    "1089-134686-0000": ("LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac", None),
    "7021-79730-0003": ("LibriSpeech/test-clean/7021/79730/7021-79730-0003.flac", None),
    "4507-16021-0026": ("LibriSpeech/test-clean/4507/16021/4507-16021-0026.flac", None),
    "1272-128104-0000": ("dev-clean-dummy/1272-128104-0000.wav", "LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"),
}
# Reductions applied cumulatively, in this order, only while the fixture is over LIMIT. Blocks go first (the
# short clip carries 24/42 of them), then column counts, then interior rows. Tails are never thinned: the last
# frames are where valid-length semantics live.
LADDER = [
    ("block cols 16", {"block_cols": 16}),
    ("block cols 8", {"block_cols": 8}),
    ("cols 24", {"cols": 24, "vocab_cols": 24}),
    ("cols 16", {"cols": 16, "vocab_cols": 16}),
    ("pe rows <= 64", {"pe_rows": 64}),
    ("logmel/mel every 16th (+ last 16)", {"mel_stride": 16}),
    ("pre_conv every 2nd (+ last 8)", {"pre_conv_stride": 2}),
    ("blocks every 8th (+ last 4)", {"block_stride": 8}),
    ("pre_conv every 4th (+ last 8)", {"pre_conv_stride": 4}),
    ("pre_out every 2nd (+ last 8)", {"pre_out_stride": 2}),
    ("cols 12", {"cols": 12, "vocab_cols": 12, "block_cols": 6}),
    ("enc every 2nd (+ last 8)", {"enc_stride": 2}),
    ("logmel/mel every 32nd (+ last 16)", {"mel_stride": 32}),
]
# Significant digits per stage, from the gate each stage faces and its measured max/rms: the rounding of the
# LARGEST value must sit well under the tolerance. pre_conv/pre_out face max/rms <= 1e-5 with max/rms ~13-40
# (xscale 32 makes pre_out reach 1128 against an rms of 89): 6 digits would round by 5.6e-5 x rms there, so 8.
# Blocks/enc (max/rms <= 1e-4 stage-local, max/rms ~40) and logmel (|v| to 16.6) get 7; CTC logits reach ~500
# against a 5e-3 gate, 7. mel (|v| <= ~5, p99 <= 1e-4), pe (|v| <= 1) and RNN-T logits (1e-2) are fine at 6.
DIGITS = {"logmel": 7, "mel": 6, "pre_conv": 8, "pre_out": 8, "pe": 6, "block": 7, "enc": 7, "rnnt": 6, "ctc": 7}
PLAN0 = {"cols": 32, "vocab_cols": 32, "block_cols": 32, "pe_rows": 128, "mel_stride": 8, "pre_conv_stride": 1,
         "pre_out_stride": 1, "enc_stride": 1, "block_stride": 4}


def g6(v):
    return float("%.6g" % v)


def gd(v, d):
    return float("%.*g" % (d, v))


def g10(v):
    return float("%.10g" % v)


# ------------------------------------------------------------------------------------------------ hygiene
def assert_eval(m):
    live = [n or "<root>" for n, mod in m.named_modules() if mod.training]
    if live:
        raise SystemExit(f"{len(live)} modules in TRAINING mode (e.g. {live[:3]}) — dropout would be live; refusing")


def pcm_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_clip(root, cid):
    rel, hint = CLIPS[cid]
    path = os.path.join(root, rel)
    x16, sr = sf.read(path, dtype="int16")
    if sr != 16000 or x16.ndim != 1:
        raise SystemExit(f"{path}: {sr} Hz, {x16.ndim} channels — the models take 16 kHz mono")
    pcm = x16.astype(np.float32) / np.float32(32768.0)
    if not np.array_equal(pcm, sf.read(path, dtype="float32")[0]):
        raise SystemExit(f"{path}: soundfile's float32 is not int16/32768 — the two sides would read different audio")
    sha = hashlib.sha256(x16.astype("<i2").tobytes()).hexdigest()
    return x16, pcm, sha, hint or rel


def featurizer_copy(fz):
    """⛔ copy.deepcopy(featurizer) is NOT a copy. FilterbankFeatures.__init__ stores
    self.forward = torch.no_grad()(self.forward), a closure over the ORIGINAL's bound method, and deepcopy shares
    it — so calling the "copy" runs the original. Measured here: a train-mode deep copy did not dither, because
    the original it silently ran was in eval. Dropping the instance attribute gives the copy its class forward."""
    c = copy.deepcopy(fz)
    c.__dict__.pop("forward", None)
    if c.forward.__self__ is not c:
        raise SystemExit("the featurizer copy's forward is still bound to another module; refusing")
    return c


# ------------------------------------------------------------------------------------------------ capture
def _reflect_stft(fz):
    """NeMo 1.19's STFT: torch.stft with its default pad_mode, 'reflect' (v1.19.0 features.py:310-318)."""
    def stft(x, *_, **__):
        return torch.stft(x, n_fft=fz.n_fft, hop_length=fz.hop_length, win_length=fz.win_length, center=True,
                          window=fz.window.to(dtype=torch.float, device=x.device), return_complex=True,
                          pad_mode="reflect")
    return stft


def run_stages(m, pcm, *, plus1=False, reflect=False, dtype=torch.float32, blocks=False):
    """One forward through NeMo's own modules, with every fixture stage captured by hooks.
    plus1/reflect select arm B / V119. Returns float64 arrays over the WHOLE tensors, plus the lengths."""
    from nemo.collections.asr.parts.preprocessing import features
    assert_eval(m)
    before = pcm_hash(pcm)
    fz, enc_mod = m.preprocessor.featurizer, m.encoder
    got, hooks = {}, []
    nb0, g0 = features.normalize_batch, fz.get_seq_len
    if "get_seq_len" in fz.__dict__ or "stft" in fz.__dict__:
        raise SystemExit("featurizer carries a patched get_seq_len/stft from an earlier arm; refusing")

    def nb(x, seq_len, normalize_type):          # its input is the raw log-mel, all 1+N//160 frames
        got["logmel"] = x[0].T.double().clone()
        return nb0(x, seq_len, normalize_type)
    features.normalize_batch = nb
    if plus1:
        fz.get_seq_len = lambda s: g0(s) + 1
    if reflect:
        fz.stft = _reflect_stft(fz)
    try:
        with torch.no_grad():
            sig = torch.from_numpy(pcm.copy())[None]
            mel, ml = m.preprocessor(input_signal=sig, length=torch.tensor([len(pcm)]))
    finally:
        features.normalize_batch = nb0
        fz.__dict__.pop("get_seq_len", None)     # both are methods: dropping the instance patch restores them
        fz.__dict__.pop("stft", None)
    if "logmel" not in got:
        raise SystemExit("normalize_batch was never called — the raw log-mel was not captured")
    mask0 = enc_mod._create_masks

    def masks(*a, **k):
        pm, am = mask0(*a, **k)
        got["valid_after_masks"] = int((~pm[0]).sum())
        return pm, am
    enc_mod._create_masks = masks
    hooks.append(enc_mod.pre_encode.conv.register_forward_hook(
        lambda mod, i, o: got.__setitem__("pre_conv", o[0].transpose(1, 2).reshape(o[0].shape[2], -1).double().clone())))
    hooks.append(enc_mod.pos_enc.register_forward_hook(
        lambda mod, i, o: got.update(pre_out=o[0][0].double().clone(), pe_hook=o[1][0].double().clone())))
    if blocks:
        for i, layer in enumerate(enc_mod.layers):
            hooks.append(layer.register_forward_hook(
                lambda mod, inp, o, i=i: got.__setitem__(f"block.{i}", o[0].double().clone())))
    try:
        assert_eval(m)
        with torch.no_grad():
            enc, el = enc_mod(audio_signal=mel.to(dtype), length=ml)
    finally:
        for h in hooks:
            h.remove()
        del enc_mod._create_masks
    if pcm_hash(pcm) != before:
        raise SystemExit("the caller's PCM changed during the forward — an in-place dither ran; refusing")
    got.update(mel=mel[0].T.double().clone(), mel_len=int(ml[0]), enc=enc[0].T.double().clone(), enc_len=int(el[0]),
               enc_t=enc.clone(), enc_len_t=el.clone())
    if got["valid_after_masks"] != got["enc_len"]:
        raise SystemExit(f"pad mask keeps {got['valid_after_masks']} frames, encoded_len is {got['enc_len']}")
    return got


def pe_valid(pe_hook, T):
    """Rows of a hooked pos_emb [2Tt-1, d] (positions Tt-1 .. -(Tt-1)) for positions T-1 .. -(T-1)."""
    Tt = (pe_hook.shape[0] + 1) // 2
    return pe_hook[Tt - T: Tt - T + 2 * T - 1]


# ------------------------------------------------------------------------------------------------ decoding
def _first(h):
    h = h[0] if isinstance(h, tuple) else h
    return h[0]


def _ids(ys):
    return [int(v) for v in (ys.tolist() if torch.is_tensor(ys) else ys)]


def nemo_rnnt(m, enc, el):
    """The authors' decode, exactly as shipped (joint.log_softmax as configured)."""
    h = _first(m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=el, return_hypotheses=True))
    return _ids(h.y_sequence), h.text


def rnnt_greedy(m, f, blank, max_symbols):
    """NeMo's greedy label-looping visit order (rnnt_label_looping.py torch_impl), one utterance, raw logits."""
    dt = f.dtype
    g, state = m.decoder.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)   # SOS = blank
    gp = m.joint.project_prednet(g)
    t, u, at_t, steps, tokens, rows = 0, 0, 0, [], [], []
    while t < f.shape[0]:
        z = m.joint.joint_after_projection(f[t][None, None], gp)[0, 0, 0]
        k = int(z.argmax())
        steps.append((t, u, k))
        rows.append(z.to(dt))
        if k == blank:
            t, at_t = t + 1, 0
            continue
        tokens.append(k)
        u, at_t = u + 1, at_t + 1
        g, state = m.decoder.predict(torch.tensor([[k]]), state, add_sos=False, batch_size=1)
        gp = m.joint.project_prednet(g)
        if at_t >= max_symbols:                 # forced advance: no blank is queried at this t
            t, at_t = t + 1, 0
    return steps, tokens, torch.stack(rows)


def rnnt_teacher_forced(m, f, steps, tokens, blank):
    """Raw joint logits at every (t, u) of `steps`, with the predictor fed `tokens`' prefix (one batched LSTM)."""
    G, _ = m.decoder.predict(torch.tensor([[blank] + tokens]), None, add_sos=False, batch_size=1)
    Gp = m.joint.project_prednet(G)[0]
    ts = torch.tensor([s[0] for s in steps])
    us = torch.tensor([s[1] for s in steps])
    return m.joint.joint_after_projection(f[ts][:, None], Gp[us][:, None])[:, 0, 0]


def rnnt_path(m, cap):
    blank = m.joint.num_classes_with_blank - 1
    max_symbols = int(m.cfg.decoding.greedy.get("max_symbols", 10))
    live = getattr(getattr(m.decoding, "decoding", None), "max_symbols", max_symbols)
    if live != max_symbols:
        raise SystemExit(f"config max_symbols {max_symbols} but the live decoder uses {live}; refusing")
    T = cap["enc_len"]
    saved = m.joint.log_softmax
    with torch.no_grad():
        nemo_tokens, text = nemo_rnnt(m, cap["enc_t"], cap["enc_len_t"])
        f = m.joint.project_encoder(cap["enc_t"].transpose(1, 2)[:, :T])[0]
        # the trap, demonstrated: as configured, the CPU joint returns log-probs (every row's logsumexp is 0)
        z_cfg = m.joint.joint_after_projection(f[:1][None], m.joint.project_prednet(
            m.decoder.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)[0]))[0, 0, 0]
        m.joint.log_softmax = False
        try:
            raw_tokens, _ = nemo_rnnt(m, cap["enc_t"], cap["enc_len_t"])
            steps, tokens, greedy_rows = rnnt_greedy(m, f, blank, max_symbols)
            tf = rnnt_teacher_forced(m, f, steps, nemo_tokens, blank).double()
        finally:
            m.joint.log_softmax = saved
    if tokens != nemo_tokens:
        raise SystemExit(f"re-implemented greedy ({len(tokens)} tokens) != m.decoding ({len(nemo_tokens)}) — the visit "
                         f"order is wrong, and the recorded steps would not be NeMo's; refusing")
    am = [int(v) for v in tf.argmax(-1)]
    if am != [s[2] for s in steps]:
        bad = [i for i, (a, s) in enumerate(zip(am, steps)) if a != s[2]]
        raise SystemExit(f"teacher-forced logits do not reproduce the greedy argmax at steps {bad[:6]}; refusing")
    lse = torch.logsumexp(tf, -1)
    if lse.abs().max() < 1e-3:
        raise SystemExit("recorded joint rows have logsumexp 0 — they are log-probs, not raw logits; refusing")
    top2 = tf.topk(2, -1).values
    return {"steps": steps, "tokens": nemo_tokens, "text": text, "tf": tf, "lse": lse, "margin": top2[:, 0] - top2[:, 1],
            "greedy_vs_teacher_forced_max_abs": float((greedy_rows.double() - tf).abs().max()),
            "tokens_from_raw_logits_equal": raw_tokens == nemo_tokens,
            "configured_joint_is_log_softmax": bool(abs(float(torch.logsumexp(z_cfg.double(), -1))) < 1e-4),
            "max_emissions_at_one_frame": max([sum(1 for s in steps if s[0] == t and s[2] != blank)
                                                for t in range(T)] or [0])}


def ctc_path(m, cap):
    """Raw head logits (Conv1d, BEFORE ConvASRDecoder's unconditional log_softmax), NeMo's greedy on the log-probs."""
    T = cap["enc_len"]
    with torch.no_grad():
        raw = m.decoder.decoder_layers(cap["enc_t"]).transpose(1, 2)
        lp = m.decoder(encoder_output=cap["enc_t"])
        if not torch.equal(torch.nn.functional.log_softmax(raw, dim=-1), lp):
            raise SystemExit("log_softmax(decoder_layers(enc)) != m.decoder(enc) — the raw-logit tap is not the head; refusing")
        h = _first(m.decoding.ctc_decoder_predictions_tensor(lp, decoder_lengths=cap["enc_len_t"], return_hypotheses=True))
    raw = raw[0, :T].double()
    am = [int(v) for v in lp[0, :T].argmax(-1)]          # NeMo greedy: prediction.max(-1) on log-probs, first max
    blank = raw.shape[1] - 1
    tokens, prev = [], None
    for k in am:
        if k != prev and k != blank:
            tokens.append(k)
        prev = k
    if m.tokenizer.ids_to_text(tokens) != h.text:
        raise SystemExit(f"collapsed argmax text != NeMo's hypothesis text ({h.text!r}); refusing")
    top2 = raw.topk(2, -1).values
    return {"raw": raw, "argmax": am, "tokens": tokens, "text": h.text, "lse": torch.logsumexp(raw, -1),
            "margin": top2[:, 0] - top2[:, 1], "raw_argmax_equals_logprob_argmax": [int(v) for v in raw.argmax(-1)] == am}


# ------------------------------------------------------------------------------------------------ stages
def stage_domains(cap, T, L):
    """name -> float64 [rows, width] over the reference's VALID domain."""
    d = {"logmel": cap["logmel"], "mel": cap["mel"][:L], "pre_conv": cap["pre_conv"][:T],
         "pre_out": cap["pre_out"][:T], "pe": pe_valid(cap["pe_hook"], T)}
    for k in sorted((k for k in cap if k.startswith("block.")), key=lambda s: int(s.split(".")[1])):
        d[k] = cap[k][:T]
    d["enc"] = cap["enc"][:T]
    return d


def diff(a, b):
    """max|a-b| over the whole domain; rel = that / rms(a); p99 of |a-b| (the mel gate is p99-based)."""
    dd = (a - b).abs()
    rms = float(a.pow(2).mean().sqrt())
    return {"max_abs_diff_vs_ref": g6(float(dd.max())), "rel": g6(float(dd.max()) / max(rms, 1e-30)),
            "p99": g6(float(torch.quantile(dd.flatten().float(), 0.99)))}


def compare_arm(ref, arm_cap, T, L):
    """Every stage of an arm against the reference, over the reference's valid rows (full width)."""
    rd, ad = stage_domains(ref["cap"], T, L), stage_domains(arm_cap, T, L)
    out = {}
    for k, a in rd.items():
        if k in ad and ad[k].shape == a.shape:
            out[k] = diff(a, ad[k])
    e = (rd["enc"] - ad["enc"]).abs().max(1).values
    out["enc"].update(first_frame=g6(float(e[0])), last_valid_frame=g6(float(e[-1])), arm_enc_frames=arm_cap["enc_len"],
                      arm_mel_frames=arm_cap["mel_len"])
    return out


# ------------------------------------------------------------------------------------------------ fixture
def _rows(n, stride, tail):
    return sorted(set(range(0, n, stride)) | set(range(max(0, n - tail), n)))


def _cols(stage, width, k):
    return sorted(random.Random(f"{SEED}/{stage}/{width}").sample(range(width), min(k, width)))


def _stage(full, rows, cols, digits, extra=None):
    sub = full[rows][:, cols]
    r = {"width": int(full.shape[1]), "shape": list(full.shape), "rows": rows, "cols": cols, "digits": digits,
         "vals": [[gd(v, digits) for v in row] for row in sub.tolist()],
         "row_sum": [g10(v) for v in full[rows].sum(1).tolist()],
         "row_ssq": [g10(v) for v in full[rows].pow(2).sum(1).tolist()],
         "max_abs": g10(float(full.abs().max())), "rms": g10(float(full.pow(2).mean().sqrt()))}
    r.update(extra or {})
    return r


def _vocab_cols(V, blank, k):
    return sorted(set(random.Random(f"{SEED}/vocab/{V}").sample([i for i in range(V) if i != blank], k - 1)) | {blank})


def build_fixture(head, ref, plan):
    T, L = ref["T"], ref["L"]
    dom = stage_domains(ref["cap"], T, L)
    st = {}
    for k, full in dom.items():
        n, w = full.shape
        if k in ("logmel", "mel"):
            rows, cols = _rows(n, plan["mel_stride"], 16), _cols(k, w, plan["cols"])
        elif k == "pe":
            stride = max(1, math.ceil(n / plan["pe_rows"]))
            rows = sorted(set(range(0, n, stride)) | set(range(4)) | set(range(n - 4, n)) | {T - 1})
            cols = _cols(k, w, plan["cols"])
        elif k.startswith("block."):
            rows, cols = _rows(n, plan["block_stride"], 4), _cols(k, w, plan["block_cols"])
        else:
            stride = {"pre_conv": plan["pre_conv_stride"], "pre_out": plan["pre_out_stride"], "enc": plan["enc_stride"]}[k]
            rows, cols = _rows(n, stride, 8 if stride > 1 else 0), _cols(k, w, plan["cols"])
        st[k] = _stage(full, rows, cols, DIGITS[k.split(".")[0]], {"pos_first": T - 1} if k == "pe" else None)
    fx = {"stages": st}
    if head == "rnnt":
        r = ref["rnnt"]
        V = r["tf"].shape[1]
        cols = _vocab_cols(V, V - 1, plan["vocab_cols"])
        fx["rnnt"] = {"blank": V - 1, "steps": [list(s) for s in r["steps"]], "cols": cols, "digits": DIGITS["rnnt"],
                      "vals": [[gd(v, DIGITS["rnnt"]) for v in row] for row in r["tf"][:, cols].tolist()],
                      "row_max": [g10(v) for v in r["tf"].max(-1).values.tolist()],
                      "row_lse": [g10(v) for v in r["lse"].tolist()],
                      "margin": [g6(v) for v in r["margin"].tolist()],
                      "tokens": r["tokens"], "text": r["text"]}
    else:
        c = ref["ctc"]
        V = c["raw"].shape[1]
        cols = _vocab_cols(V, V - 1, plan["vocab_cols"])
        rows = list(range(T))
        fx["ctc"] = {"blank": V - 1, "rows": rows, "cols": cols, "digits": DIGITS["ctc"],
                     "vals": [[gd(v, DIGITS["ctc"]) for v in row] for row in c["raw"][rows][:, cols].tolist()],
                     "row_max": [g10(v) for v in c["raw"].max(-1).values.tolist()],
                     "row_lse": [g10(v) for v in c["lse"].tolist()],
                     "margin": [g6(v) for v in c["margin"].tolist()],
                     "argmax": c["argmax"], "tokens": c["tokens"], "text": c["text"]}
    return fx


def fit(head, ref, header, tail):
    plan, applied = dict(PLAN0), []
    for step in [None] + LADDER:
        if step is not None:
            plan.update(step[1])
            applied.append(step[0])
        doc = dict(header)
        doc.update(build_fixture(head, ref, plan))
        doc.update(tail)
        doc["sampling"] = {"seed": SEED, "plan": dict(plan), "reductions_applied": list(applied),
                           "limit_bytes": LIMIT}
        s = json.dumps(doc, separators=(",", ":"))
        if len(s) <= LIMIT:
            return s, applied
    raise SystemExit(f"fixture is {len(s)} bytes even after every reduction in LADDER; refusing")


# ------------------------------------------------------------------------------------------------ main
def main():
    gguf_path, nemo_path, audio_root, out_dir, *only = sys.argv[1:]
    os.makedirs(os.path.join(out_dir, "full"), exist_ok=True)
    clips = only or list(CLIPS)
    t0 = time.time()
    import nemo
    from nemo.collections.asr.models import ASRModel

    torch.manual_seed(0)
    m = ASRModel.restore_from(nemo_path, map_location="cpu")
    m.eval()
    m.transcribe = lambda *a, **k: (_ for _ in ()).throw(SystemExit(
        "transcribe() leaves encoder/decoder/joint in train mode (unfreeze -> module.train()); never in a dump process"))
    head = "rnnt" if hasattr(m, "joint") else "ctc"
    repo = "nvidia/" + os.path.basename(nemo_path)[: -len(".nemo")]
    print(f"[{time.time() - t0:.0f}s] restored {type(m).__name__} ({head}) from {nemo_path}", flush=True)

    # ---- weights: checkpoint file == restored model; GGUF == checkpoint under each storage rule
    ck = pw.load_checkpoint(nemo_path)
    restore = pw.restore_audit(m, ck)
    if restore["differ"] or restore["only_in_model"] or restore["only_in_file"]:
        raise SystemExit(f"restore_from() did not produce the checkpoint file's weights: {restore}")
    import librosa
    fz = m.preprocessor.featurizer
    fb_rebuild = torch.tensor(librosa.filters.mel(sr=16000, n_fft=fz.n_fft, n_mels=fz.nfilt, fmin=0, fmax=8000,
                                                  norm="slaney"), dtype=torch.float32).unsqueeze(0)
    gg, meta = pw.read_gguf(gguf_path)
    summary, per_tensor, target = pw.audit(gg, ck, live_pe=m.encoder.pos_enc.pe.detach().clone(), live_fb=fb_rebuild,
                                           live_window=torch.hann_window(fz.win_length, periodic=False))
    del gg
    summary["gguf_sha256"] = pw.sha256_file(gguf_path)
    summary["restore_audit"] = restore
    print(f"[{time.time() - t0:.0f}s] weight audit: {json.dumps(summary['rules'])}", flush=True)

    # ---- NeMo-G: the GGUF's values in the authors' code, then audited by value, and the audit shown to fire
    summary["injected"] = pw.inject(m, target)
    summary["injected_selftest"] = pw.injected_audit_selftest(m, target)
    m.eval()
    guards = {"injected_audit": summary["injected_selftest"]}
    probe = m.joint if head == "rnnt" else m.decoder
    probe.train()
    try:
        assert_eval(m)
        raise RuntimeError("assert_eval did not fire on a module in training mode")
    except SystemExit as e:
        guards["train_mode"] = f"fired on {type(probe).__name__}.train(): {str(e)[:60]}..."
    probe.eval()
    noise = np.random.default_rng(SEED).standard_normal(16000).astype(np.float32) * np.float32(0.1)
    fz_train, kept = featurizer_copy(fz).train(), noise.copy()
    with torch.no_grad():
        fz_train(torch.from_numpy(noise)[None], torch.tensor([16000]))    # from_numpy SHARES the caller's memory
    if pcm_hash(noise) == pcm_hash(kept):
        raise SystemExit("a train-mode featurizer did not dither its input in place — the trap this run guards "
                         "against could not be reproduced, so the guard's description is unfounded; look first")
    guards["dither_mutates_caller_input"] = {"max_abs_change": g6(float(np.abs(noise - kept).max())),
                                             "caught_by": "every capture hashes the caller's PCM before and after"}
    del fz_train
    json.dump({"summary": summary, "tensors": per_tensor},
              open(os.path.join(out_dir, os.path.basename(gguf_path)[:-5] + "__weights_audit.json"), "w"), indent=1)

    # ---- A: the reference, every clip
    refs = {}
    for cid in clips:
        x16, pcm, sha, hint = read_clip(audio_root, cid)
        wav = os.path.join(out_dir, cid + ".wav")
        sf.write(wav, x16, 16000, subtype="PCM_16")
        if not np.array_equal(sf.read(wav, dtype="int16")[0], x16):
            raise SystemExit(f"{wav} does not read back as the same int16 samples")
        cap = run_stages(m, pcm, blocks=(cid == BLOCK_CLIP))
        T, L = cap["enc_len"], cap["mel_len"]
        if L != len(pcm) // 160 or cap["logmel"].shape[0] != 1 + len(pcm) // 160:
            raise SystemExit(f"{cid}: NeMo gave {L} valid / {cap['logmel'].shape[0]} frames, not N//160 / 1+N//160")
        refs[cid] = {"cap": cap, "T": T, "L": L, "pcm": pcm, "sha": sha, "hint": hint, "wav": wav}
        refs[cid][head] = rnnt_path(m, cap) if head == "rnnt" else ctc_path(m, cap)
        refs[cid]["arms"] = {}
        print(f"[{time.time() - t0:.0f}s] A {cid}: L={L} T={T} (tensor {cap['enc'].shape[0]}) "
              f"text={refs[cid][head]['text']!r}", flush=True)

    def arm(name, mdl, dtype=torch.float32, **kw):
        for cid, ref in refs.items():
            cap = run_stages(mdl, ref["pcm"], dtype=dtype, blocks=(cid == BLOCK_CLIP), **kw)
            a = compare_arm(ref, cap, ref["T"], ref["L"])
            if head == "rnnt":
                r = ref["rnnt"]
                with torch.no_grad():
                    saved = mdl.joint.log_softmax
                    toks, _ = nemo_rnnt(mdl, cap["enc_t"], cap["enc_len_t"])
                    mdl.joint.log_softmax = False
                    f = mdl.joint.project_encoder(cap["enc_t"].transpose(1, 2)[:, :ref["T"]])[0]
                    blank = mdl.joint.num_classes_with_blank - 1
                    tf = rnnt_teacher_forced(mdl, f, r["steps"], r["tokens"], blank).double()
                    mdl.joint.log_softmax = saved
                a["rnnt_logits"] = diff(r["tf"], tf)
                a["rnnt_logits"]["argmax_equal_steps"] = int(sum(int(x) == s[2] for x, s in zip(tf.argmax(-1), r["steps"])))
                a["tokens_equal"] = toks == r["tokens"]
                a["n_tokens"] = len(toks)
            else:
                c = ctc_path(mdl, cap)
                a["ctc_logits"] = diff(ref["ctc"]["raw"], c["raw"][: ref["T"]])
                a["ctc_logits"]["argmax_frames_differ"] = int(sum(x != y for x, y in zip(ref["ctc"]["argmax"], c["argmax"])))
                a["tokens_equal"] = c["tokens"] == ref["ctc"]["tokens"]
                a["n_tokens"] = len(c["tokens"])
            ref["arms"][name] = a
            print(f"[{time.time() - t0:.0f}s] {name} {cid}: enc {a['enc']['max_abs_diff_vs_ref']:.3e} "
                  f"(rel {a['enc']['rel']:.2e}) tokens_equal {a['tokens_equal']}", flush=True)

    arm("B", m, plus1=True)
    if head == "ctc":
        arm("V119", m, plus1=True, reflect=True)
    m.double()
    m.preprocessor.float()                       # AudioPreprocessor runs fp32 in NeMo whatever the model's dtype
    arm("G64", m, dtype=torch.float64)
    m.float()                                    # f32 -> f64 -> f32 is exact; the re-audit proves it
    back = pw.injected_audit(m, target)
    if back["n_differ"]:
        raise SystemExit(f"NeMo-G did not survive the fp64 round trip: {back}")
    # F64: what NeMo computes in fp32 even under model.double() — the featurizer, and create_pe's table
    from nemo.collections.asr.parts.preprocessing import features
    fz64 = featurizer_copy(fz).double()
    for cid, ref in refs.items():
        got, nb0 = {}, features.normalize_batch

        def nb(x, seq_len, normalize_type, got=got, nb0=nb0):
            got["logmel"] = x[0].T.clone()
            return nb0(x, seq_len, normalize_type)
        features.normalize_batch = nb
        try:
            with torch.no_grad():
                mel64, _ = fz64(torch.from_numpy(ref["pcm"].astype(np.float64))[None], torch.tensor([len(ref["pcm"])]))
        finally:
            features.normalize_batch = nb0
        T, d = ref["T"], m.encoder.pos_enc.d_model
        pos = torch.arange(T - 1, -T, -1, dtype=torch.float64)[:, None]
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float64) * -(math.log(10000.0) / d))
        pe64 = torch.zeros(2 * T - 1, d, dtype=torch.float64)
        pe64[:, 0::2], pe64[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        dom = stage_domains(ref["cap"], T, ref["L"])
        ref["arms"]["F64"] = {"logmel": diff(dom["logmel"], got["logmel"]), "mel": diff(dom["mel"], mel64[0].T[: ref["L"]]),
                              "pe": diff(dom["pe"], pe64)}
    del fz64
    # NeMo-ckpt: the authors' own fp32 weights, audited again as loaded
    m.load_state_dict(ck, strict=True)
    again = pw.restore_audit(m, ck)
    if again["differ"]:
        raise SystemExit(f"the checkpoint did not load back by value: {again}")
    arm("NeMo-ckpt", m)

    # ---- fixtures
    report = {}
    for cid, ref in refs.items():
        r = ref[head]
        header = {"format": "parakeet-conformance/1", "model": repo, "gguf": os.path.basename(gguf_path),
                  "reference": "NeMo-G", "nemo": nemo.__version__, "torch": torch.__version__, "weights_audit": summary,
                  "guards": guards,
                  "clip": {"id": cid, "path_hint": ref["hint"], "wav": os.path.basename(ref["wav"]),
                           "n_samples": len(ref["pcm"]), "pcm_sha256": ref["sha"]},
                  "valid": {"mel_frames": ref["L"], "enc_frames": ref["T"], "logmel_frames": ref["cap"]["logmel"].shape[0],
                            "enc_tensor_frames": ref["cap"]["enc"].shape[0]}}
        checks = {k: r[k] for k in ("greedy_vs_teacher_forced_max_abs", "tokens_from_raw_logits_equal",
                                    "configured_joint_is_log_softmax", "max_emissions_at_one_frame",
                                    "raw_argmax_equals_logprob_argmax") if k in r}
        tail = {"decode_checks": checks, "arms": ref["arms"]}
        s, applied = fit(head, ref, header, tail)
        path = os.path.join(out_dir, f"{repo.split('/')[1]}__{cid}.json")
        open(path, "w").write(s)
        np.savez(os.path.join(out_dir, "full", f"{repo.split('/')[1]}__{cid}.npz"),
                 **{k: v.float().numpy() for k, v in stage_domains(ref["cap"], ref["T"], ref["L"]).items()},
                 **({"rnnt_logits": r["tf"].float().numpy(), "rnnt_steps": np.array(r["steps"])} if head == "rnnt"
                    else {"ctc_logits": r["raw"].float().numpy()}))
        report[cid] = {"fixture": path, "bytes": len(s), "reductions": applied, "stages": list(json.loads(s)["stages"]),
                       "n_tokens": len(r["tokens"]), "text": r["text"], "decode_checks": checks,
                       "arms": {a: {k: v[k] for k in ("enc", "tokens_equal", "rnnt_logits", "ctc_logits", "mel", "logmel", "pe")
                                    if k in v}
                                for a, v in ref["arms"].items()}}
    print(json.dumps({"model": repo, "weights": {k: summary[k] for k in ("rules", "controls_must_differ", "gguf_extras",
                                                                         "dropped_checkpoint_entries", "buffers", "injected")},
                      "q8_0": summary.get("q8_0"), "clips": report, "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
