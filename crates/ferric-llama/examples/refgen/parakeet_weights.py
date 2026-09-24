"""The Parakeet GGUF, checked BY VALUE against the MODEL AUTHORS' .nemo checkpoint, and then injected into the
authors' own code (NeMo) so that the reference runs on exactly the numbers Ferric reads. Used by parakeet_ref.py.

Why "NeMo on the GGUF's weights" (NeMo-G) is the gating reference, not NeMo on its own checkpoint: the GGUF
rounds every matrix (F16 in the unified file, Q8_0 in the CTC one). Measured inside NeMo, that rounding alone
moves the unified encoder by 5.6e-4 and the CTC log-probs by 4.8e-2 — as large as the frame-semantics defects
(V1-V3) the comparison exists to find. With both effects in one residual, neither can be seen. So the weights
are audited here, the audited values are loaded into NeMo, and the loaded model is audited again.

Every guard below exists because a "reference" was once wrong in exactly that way (see refload.py):
  - a GGUF tensor is mapped to the checkpoint by NAME and SHAPE, and every tensor on both sides must be
    accounted for — an unmapped or silently dropped tensor refuses;
  - each tensor is checked under its file's STORAGE RULE, not by closeness: F32 exact, F16 = f16(ckpt) bit
    for bit, LSTM bias = bias_ih + bias_hh exactly, Q8_0 = ggml's block_q8_0 re-quantisation byte for byte.
    A closeness test (the earlier 3.8e-3 check) would pass a file quantised from an F16 intermediate; each
    rule therefore carries a CONTROL that must FAIL (raw f32 vs F16, bias_ih alone, Q8_0 via f16);
  - "0 missing keys" is not "the weights are the file's": the injected model is compared, parameter by
    parameter, against the arrays it was given, and that audit is itself shown to catch a 1-ulp change.
"""
import hashlib
import io
import os
import re
import sys
import tarfile

import numpy as np
import torch

# The vendored llama.cpp's gguf-py, read-only — resolved from this file (examples/refgen -> repo root).
_GGUF_PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "..", "..", ".reference", "llama.cpp", "gguf-py"))


def _gguf():
    if _GGUF_PY not in sys.path:
        sys.path.insert(0, _GGUF_PY)
    import gguf  # noqa: E402
    return gguf


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def load_checkpoint(nemo_path):
    """model_weights.ckpt straight out of the .nemo tar — the file, not what restore_from made of it."""
    mode = "r:gz" if open(nemo_path, "rb").read(2) == b"\x1f\x8b" else "r:"
    with tarfile.open(nemo_path, mode) as tf:
        mem = next(m for m in tf.getmembers() if m.name.endswith("model_weights.ckpt"))
        f = tf.extractfile(mem) if mode == "r:" else io.BytesIO(tf.extractfile(mem).read())
        ck = torch.load(f, map_location="cpu", weights_only=True)
    return ck["state_dict"] if isinstance(ck.get("state_dict"), dict) else ck


def restore_audit(model, ck):
    """⛔ restore_from() reporting success is not the checkpoint: every state_dict entry vs the file, by value."""
    sd = model.state_dict()
    only_model, only_file = sorted(set(sd) - set(ck)), sorted(set(ck) - set(sd))
    bad = [k for k in sorted(set(sd) & set(ck)) if sd[k].dtype != ck[k].dtype or not torch.equal(sd[k], ck[k])]
    return {"entries": len(sd), "equal": len(sd) - len(bad) - len(only_model), "differ": bad[:8],
            "only_in_model": only_model, "only_in_file": only_file}


# GGUF name -> NeMo key. The unified file comes from a third-party converter with its own short names; the CTC
# file is NVIDIA's own q8_0 conversion and keeps NeMo's names. Layouts need no permutation in either: every
# tensor is the torch tensor written row-major, so ggml's ne order reversed IS the torch shape (checked below).
_UNIFIED = [
    (r"enc\.pre_encode\.(.*)", r"encoder.pre_encode.\1"),
    (r"enc\.blocks\.(\d+)\.norm_ff1\.(.*)", r"encoder.layers.\1.norm_feed_forward1.\2"),
    (r"enc\.blocks\.(\d+)\.norm_ff2\.(.*)", r"encoder.layers.\1.norm_feed_forward2.\2"),
    (r"enc\.blocks\.(\d+)\.ff1\.(.*)", r"encoder.layers.\1.feed_forward1.\2"),
    (r"enc\.blocks\.(\d+)\.ff2\.(.*)", r"encoder.layers.\1.feed_forward2.\2"),
    (r"enc\.blocks\.(\d+)\.norm_attn\.(.*)", r"encoder.layers.\1.norm_self_att.\2"),
    (r"enc\.blocks\.(\d+)\.attn\.(.*)", r"encoder.layers.\1.self_attn.\2"),
    (r"enc\.blocks\.(\d+)\.conv\.pointwise1\.(.*)", r"encoder.layers.\1.conv.pointwise_conv1.\2"),
    (r"enc\.blocks\.(\d+)\.conv\.pointwise2\.(.*)", r"encoder.layers.\1.conv.pointwise_conv2.\2"),
    (r"enc\.blocks\.(\d+)\.conv\.depthwise\.(.*)", r"encoder.layers.\1.conv.depthwise_conv.\2"),
    (r"enc\.blocks\.(\d+)\.conv\.bn\.(.*)", r"encoder.layers.\1.conv.batch_norm.\2"),
    (r"enc\.blocks\.(\d+)\.(norm_conv|norm_out)\.(.*)", r"encoder.layers.\1.\2.\3"),
    (r"pred\.embed\.weight", r"decoder.prediction.embed.weight"),
    (r"pred\.lstm\.(\d+)\.Wx", r"decoder.prediction.dec_rnn.lstm.weight_ih_l\1"),
    (r"pred\.lstm\.(\d+)\.Wh", r"decoder.prediction.dec_rnn.lstm.weight_hh_l\1"),
    (r"joint\.(enc|pred)\.(.*)", r"joint.\1.\2"),
    (r"joint\.out\.(.*)", r"joint.joint_net.2.\1"),
]
_LSTM_BIAS = re.compile(r"pred\.lstm\.(\d+)\.bias")
# GGUF tensors that are not checkpoint parameters. Compared too, and never injected: NeMo's pe is a
# non-persistent buffer recomputed at init (multi_head_attention.py create_pe), so no checkpoint has one.
_EXTRAS = {"encoder.pos_enc.pe": None, "preprocessor.fb": "preprocessor.featurizer.fb"}
# Checkpoint entries no GGUF carries, and why none of them can move an inference result.
DROPPED_WHY = {
    "num_batches_tracked": "int64 BatchNorm step counter; only BatchNorm's TRAINING update reads it "
                           "(momentum=None cumulative average). Eval uses running_mean/var, which ARE carried.",
    "preprocessor.featurizer.window": "torch.hann_window(400, periodic=False), rebuilt from the config at "
                                      "load; the checkpoint copy is compared to that rebuild below.",
    "preprocessor.featurizer.fb": "librosa slaney mel filterbank, rebuilt from the config at load; the "
                                  "checkpoint copy is compared to that rebuild below.",
}


def _map_one(gname):
    """-> (kind, [nemo keys]) with kind one of 'param', 'lstm_bias', 'extra', or (None, []) if unmapped."""
    m = _LSTM_BIAS.fullmatch(gname)
    if m:
        l = m.group(1)
        return "lstm_bias", [f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}", f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}"]
    if gname in _EXTRAS:
        return "extra", [_EXTRAS[gname]] if _EXTRAS[gname] else []
    for pat, rep in _UNIFIED:
        if re.fullmatch(pat, gname):
            return "param", [re.sub(pat, rep, gname)]
    return "param", [gname]          # the CTC file: already NeMo's name (existence is checked by the caller)


def read_gguf(path):
    """name -> dict(type, shape (torch order), raw (the file's bytes/elements), deq (float32, torch shape))."""
    gguf = _gguf()
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        shape = tuple(int(x) for x in reversed(list(t.shape)))
        ty = t.tensor_type.name
        raw = np.asarray(t.data)
        if ty == "F32":
            deq = raw.reshape(shape)
        elif ty == "F16":
            deq = raw.reshape(shape).astype(np.float32)                    # exact
        elif ty == "Q8_0":
            deq = gguf.quants.dequantize(raw, t.tensor_type).reshape(shape)  # int8 x f16 scale: exact in f32
        else:
            raise SystemExit(f"{path}: tensor {t.name} has storage {ty}, which this audit has no rule for")
        out[t.name] = {"type": ty, "shape": shape, "raw": raw, "deq": deq}
    meta = {k: f.contents() for k, f in r.fields.items() if not k.startswith(("tokenizer.", "GGUF."))}
    return out, meta


def _q8_0(x):
    gguf = _gguf()
    return gguf.quants.quantize(np.ascontiguousarray(x, dtype=np.float32).reshape(x.shape[0], -1),
                                gguf.GGMLQuantizationType.Q8_0).reshape(-1, 34)


def _q8_compare(raw_blocks, mine):
    a, b = raw_blocks.reshape(-1, 34), mine
    qa, qb = a[:, 2:].view(np.int8).astype(np.int16), b[:, 2:].view(np.int8).astype(np.int16)
    return {"blocks": int(a.shape[0]), "blocks_exact": int((a == b).all(1).sum()),
            "scales_exact": int((a[:, :2] == b[:, :2]).all(1).sum()),
            "q_mismatch": int((qa != qb).sum()), "max_abs_dq": int(np.abs(qa - qb).max())}


def audit(gg, ck, live_pe=None, live_fb=None, live_window=None):
    """Every GGUF tensor against the checkpoint under its storage rule, every checkpoint entry accounted for.

    Returns (summary, per_tensor, nemo_target) where nemo_target maps EVERY checkpoint key to the float32 (or
    original-dtype) tensor NeMo-G must hold: the GGUF's dequantised value where the file carries it, the
    checkpoint's own value where it does not (window, num_batches_tracked, a fb the file omits)."""
    per, used, unmapped, shape_bad, squeezed = [], set(), [], [], []
    target = {}
    rules = {"F32_exact": [0, 0], "F16_exact": [0, 0], "lstm_bias_sum_exact": [0, 0], "Q8_0_requant_exact": [0, 0]}
    controls = {"F16_vs_unrounded_f32_differs": [0, 0], "lstm_bias_vs_bias_ih_only_differs": [0, 0],
                "Q8_0_via_f16_intermediate_differs": [0, 0]}
    q8_tot = {"blocks": 0, "blocks_exact": 0, "scales_exact": 0, "q_mismatch": 0, "max_abs_dq": 0}
    q8_alt = {"blocks": 0, "blocks_exact": 0, "scales_exact": 0, "q_mismatch": 0, "max_abs_dq": 0}
    extras = {}
    for gname, g in gg.items():
        kind, keys = _map_one(gname)
        if kind == "extra":
            if gname == "encoder.pos_enc.pe" and live_pe is not None:
                lp = live_pe.reshape(-1, live_pe.shape[-1]).numpy()
                same = lp.shape == g["deq"].shape
                d = np.abs(lp - g["deq"]) if same else None
                c0 = lp.shape[0] // 2                                      # the row of position 0
                extras[gname] = {"type": g["type"], "shape": list(g["shape"]),
                                 "vs": "NeMo 3.0.0's live create_pe table (never injected)",
                                 "exact": bool(same and (d == 0).all()),
                                 "max_abs_diff": float(d.max()) if same else None,
                                 "max_abs_diff_within_1024_positions": float(d[c0 - 1023: c0 + 1024].max()) if same else None,
                                 "n_differ": int((d != 0).sum()) if same else None}
            elif gname == "preprocessor.fb":
                c = ck["preprocessor.featurizer.fb"].numpy()
                d = np.abs(c.reshape(g["deq"].shape) - g["deq"])
                extras[gname] = {"type": g["type"], "shape": list(g["shape"]), "vs": "checkpoint preprocessor.featurizer.fb",
                                 "exact": bool((d == 0).all()), "max_abs_diff": float(d.max()), "n_differ": int((d != 0).sum())}
                if live_fb is not None:
                    d2 = np.abs(live_fb.reshape(g["deq"].shape) - g["deq"])
                    extras[gname].update({"vs_librosa_rebuild_exact": bool((d2 == 0).all()),
                                          "vs_librosa_rebuild_max_abs_diff": float(d2.max())})
                target["preprocessor.featurizer.fb"] = torch.from_numpy(np.ascontiguousarray(g["deq"]).reshape(c.shape).copy())
                used.add("preprocessor.featurizer.fb")
            continue
        if any(k not in ck for k in keys):
            unmapped.append(gname)
            continue
        if kind == "lstm_bias":
            bih, bhh = ck[keys[0]], ck[keys[1]]
            want = (bih + bhh).numpy()                                   # torch f32 add, as the converter did
            ok = g["type"] == "F32" and np.array_equal(g["deq"].view(np.uint32), want.view(np.uint32))
            rules["lstm_bias_sum_exact"][0 if ok else 1] += 1
            ctrl = not np.array_equal(g["deq"], bih.numpy())
            controls["lstm_bias_vs_bias_ih_only_differs"][0 if ctrl else 1] += 1
            per.append({"gguf": gname, "nemo": keys, "type": g["type"], "rule": "lstm_bias_sum", "exact": bool(ok)})
            target[keys[0]] = torch.from_numpy(np.array(g["deq"], dtype=np.float32))
            target[keys[1]] = torch.zeros_like(bhh)                     # the sum carries both; NeMo adds b_ih + b_hh
            used.update(keys)
            continue
        key = keys[0]
        c = ck[key]
        if tuple(c.shape) != g["shape"]:
            # NVIDIA's CTC converter drops the unit kernel axis of the pointwise Conv1d weights ([out,in,1] ->
            # [out,in]). Row-major, the elements are the same; any other shape difference is a layout question.
            if [d for d in c.shape if d != 1] != [d for d in g["shape"] if d != 1]:
                shape_bad.append((gname, list(g["shape"]), list(c.shape)))
                continue
            squeezed.append(gname)
            g["deq"] = g["deq"].reshape(c.shape)
        cf = c.float().numpy()
        rec = {"gguf": gname, "nemo": key, "type": g["type"]}
        if g["type"] == "F32":
            ok = np.array_equal(g["raw"].reshape(-1).view(np.uint32), cf.reshape(-1).view(np.uint32))
            rules["F32_exact"][0 if ok else 1] += 1
            rec.update(rule="F32_exact", exact=bool(ok))
            if not ok:
                rec["max_abs_diff"] = float(np.abs(g["deq"] - cf).max())
        elif g["type"] == "F16":
            ok = np.array_equal(g["raw"].reshape(-1).view(np.uint16), cf.astype(np.float16).reshape(-1).view(np.uint16))
            rules["F16_exact"][0 if ok else 1] += 1
            ctrl = not np.array_equal(g["deq"], cf)
            controls["F16_vs_unrounded_f32_differs"][0 if ctrl else 1] += 1
            rec.update(rule="F16_exact", exact=bool(ok))
            if not ok:
                rec["max_abs_diff_vs_f16"] = float(np.abs(g["deq"] - cf.astype(np.float16).astype(np.float32)).max())
        else:   # Q8_0
            cmp = _q8_compare(g["raw"], _q8_0(cf))
            ok = cmp["blocks_exact"] == cmp["blocks"]
            rules["Q8_0_requant_exact"][0 if ok else 1] += 1
            for k in q8_tot:
                q8_tot[k] = max(q8_tot[k], cmp[k]) if k == "max_abs_dq" else q8_tot[k] + cmp[k]
            alt = _q8_compare(g["raw"], _q8_0(cf.astype(np.float16).astype(np.float32)))
            for k in q8_alt:
                q8_alt[k] = max(q8_alt[k], alt[k]) if k == "max_abs_dq" else q8_alt[k] + alt[k]
            controls["Q8_0_via_f16_intermediate_differs"][0 if alt["blocks_exact"] < alt["blocks"] else 1] += 1
            rec.update(rule="Q8_0_requant_exact", exact=bool(ok), **{"q8": cmp})
            d = np.abs(g["deq"] - cf)
            rec["rel_err_max"] = float(d.max() / max(np.abs(cf).max(), 1e-30))
        per.append(rec)
        target[key] = torch.from_numpy(np.array(g["deq"], dtype=np.float32).reshape(c.shape))
        used.add(key)
    if unmapped or shape_bad:
        raise SystemExit(f"GGUF tensors with no checkpoint counterpart {unmapped[:6]} or a shape that is not the "
                         f"torch shape {shape_bad[:6]} — refusing: NeMo-G would not be the file's model")
    dropped = {}
    for k in sorted(set(ck) - used):
        why = next((w for pat, w in DROPPED_WHY.items() if k.endswith(pat) or k == pat), None)
        if why is None:
            raise SystemExit(f"checkpoint entry {k} is carried by no GGUF tensor and has no recorded reason — refusing")
        dropped.setdefault(why, []).append(k)
        target[k] = ck[k].clone()
    buffers = {}
    if "preprocessor.featurizer.window" in ck and live_window is not None:
        d = (ck["preprocessor.featurizer.window"] - live_window).abs()
        buffers["window_ckpt_vs_rebuild_max_abs_diff"] = float(d.max())
        buffers["window_ckpt_vs_rebuild_n_differ"] = int((d != 0).sum())
    if "preprocessor.featurizer.fb" in ck and live_fb is not None:
        d = (ck["preprocessor.featurizer.fb"] - torch.as_tensor(live_fb)).abs()
        buffers["fb_ckpt_vs_librosa_rebuild_max_abs_diff"] = float(d.max())
        buffers["fb_ckpt_vs_librosa_rebuild_n_differ"] = int((d != 0).sum())
    summary = {
        "gguf_tensors": len(gg), "checkpoint_entries": len(ck),
        "gguf_by_type": {t: sum(1 for g in gg.values() if g["type"] == t) for t in sorted({g["type"] for g in gg.values()})},
        "rules": {k: {"pass": v[0], "fail": v[1]} for k, v in rules.items() if sum(v)},
        "controls_must_differ": {k: {"differs": v[0], "does_not": v[1]} for k, v in controls.items() if sum(v)},
        "mapped_checkpoint_entries": len(used),
        "unit_dims_dropped_by_converter": {"count": len(squeezed), "examples": squeezed[:2]},
        "dropped_checkpoint_entries": {why: {"count": len(ks), "examples": ks[:3]} for why, ks in dropped.items()},
        "gguf_extras": extras,
        "buffers": buffers,
    }
    if q8_tot["blocks"]:
        summary["q8_0"] = {"rule": "ggml quantize_row_q8_0_ref (gguf-py quants.Q8_0, documented bit-exact to it) "
                                   "applied to the checkpoint's float32 tensor",
                           **q8_tot,
                           "control_via_f16_intermediate": q8_alt}
    return summary, per, target


def inject(model, target):
    """Load NeMo-G (strict), then audit it BY VALUE against the arrays it was given. Refuses on any difference."""
    missing, unexpected = model.load_state_dict(target, strict=True)
    rep = injected_audit(model, target)
    if rep["differ"] or missing or unexpected:
        raise SystemExit(f"NeMo-G is not the GGUF's model: {rep} missing={missing} unexpected={unexpected}")
    return rep


def injected_audit(model, target):
    sd = model.state_dict()
    bad = [k for k in sd if k not in target or sd[k].dtype != target[k].dtype or not torch.equal(sd[k], target[k])]
    return {"entries": len(sd), "equal": len(sd) - len(bad), "differ": bad[:8], "n_differ": len(bad)}


def injected_audit_selftest(model, target):
    """⛔ A by-value audit that cannot fail is decoration. Move ONE element of ONE parameter by one ulp and require
    the audit to find exactly that one; put it back and require it to find nothing."""
    name, p = next((n, p) for n, p in model.named_parameters() if p.dim() == 2)
    with torch.no_grad():
        old = p.view(-1)[7].clone()
        p.view(-1)[7] = torch.nextafter(old, old + 1)
        caught = injected_audit(model, target)
        p.view(-1)[7] = old
    clean = injected_audit(model, target)
    ok = caught["n_differ"] == 1 and caught["differ"] == [name] and clean["n_differ"] == 0
    if not ok:
        raise SystemExit(f"the injected-model audit did not catch a 1-ulp change in {name}: {caught} / {clean}")
    return {"perturbed": f"{name}[7] by 1 ulp", "caught": caught["differ"], "after_restore_differ": clean["n_differ"]}
