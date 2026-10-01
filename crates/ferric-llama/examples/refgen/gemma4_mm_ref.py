#!/usr/bin/env python3
"""Gemma 4 (E2B / E4B) IMAGE and AUDIO inputs — from the MODEL AUTHORS' OWN CODE, file to logits, stage by stage.

The authors' implementation is transformers' `modeling_gemma4.py` (Google ship no modeling file of their own:
the checkpoint names `Gemma4ForConditionalGeneration`) and the processor classes their `processor_config.json`
names — `Gemma4ImageProcessor` (torchvision backend) and `Gemma4AudioFeatureExtractor`. Every stage below runs
THAT code, UNMODIFIED, through the whole-model `forward`: no streaming, no re-wiring, so there is no harness of
our own to get wrong. Stages are read with forward hooks.

  image   the authors' processor on an image FILE -> pixels (the resized 8-bit image is pinned by sha256),
          the patch embedding, vision blocks, the pooled soft tokens, the projection into the text width,
          and the logits at every position of an image+text prompt — plus the same question WITHOUT the
          image (`text_rows`), so a text-decoder defect is told apart from a splice defect
  audio   the authors' feature extractor on a WAV -> log-mel (every frame), the subsampling convolutions,
          conformer blocks, the output projection, the projection into the text width, and the logits

Everything twice: float32, and float64 — THE NOISE FLOOR. The gate asks whether Ferric is as close to the
float64 run as the authors' own float32 run is. ⚠ Not pure float64: the authors cast to float32 inside every
RMSNorm, the vision rope, the pooler's scaling and the attention softmax, and those casts are kept — this is
their code at the highest precision it runs at.

⛔ Loaded at float32 on the config AND every sub-config, and every parameter AND every persistent BUFFER is
checked BY VALUE against the safetensors file as loaded. The buffers matter here: the clipped linears' bounds
(`input_min/input_max/output_min/output_max`) are buffers that `_init_weights` sets to -inf/+inf, so a loader
that re-initialised them would silently run the model WITHOUT its clipping — exactly the negative control the
gate uses on Ferric.

    <py> gemma4_mm_ref.py image <snapshot> <image> "<question>"         | gzip -9 > fx.image.json.gz
    <py> gemma4_mm_ref.py audio <snapshot> <audio.wav> "<question>"     | gzip -9 > fx.audio.json.gz
    <py> gemma4_mm_ref.py greedy image|audio <snapshot> <file> "<q>" <id,id,...>   > fx.greedy.json
    <py> gemma4_mm_ref.py generate image|audio <snapshot> <file> "<q>" <n>         > fx.generate.json

`greedy` checks a DECODE: given a continuation some runtime generated greedily, it runs the authors' forward
ONCE over prompt + continuation and records their argmax at every continuation position with the top-1/top-2
margin. If each argmax is the next token, the authors' own greedy decode produces exactly this continuation.
`generate` runs the authors' `generate(do_sample=False)` — the task-level answer, as text.
"""
import gc
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

from transformers.models.gemma4 import modeling_gemma4 as M  # noqa: E402

VISION_TAPS = [0, 7, -1]   # first, a middle block, last (resolved against the config's depth)
AUDIO_TAPS = [0, 5, -1]


def cfg_at(snap, dtype):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snap)
    for c in (cfg, cfg.text_config, getattr(cfg, "vision_config", None), getattr(cfg, "audio_config", None)):
        if c is not None:
            c.dtype = dtype
            c._attn_implementation = "eager"
    return cfg


def load(snap, dtype):
    """The authors' class at `dtype`, every parameter and persistent buffer checked against the FILE."""
    from safetensors import safe_open
    cfg = cfg_at(snap, dtype)
    m = M.Gemma4ForConditionalGeneration.from_pretrained(snap, config=cfg, dtype=dtype, attn_implementation="eager")
    m.eval()
    bad = sorted({str(p.dtype) for p in m.parameters() if p.dtype != dtype})
    if bad:
        raise SystemExit(f"loaded as {bad}, not {dtype} — refusing to emit a fixture")
    f = safe_open(os.path.join(snap, "model.safetensors"), "pt")
    keys = set(f.keys())
    live = dict(m.named_parameters())
    live.update({n: b for n, b in m.named_buffers()})
    checked = mismatched = 0
    examples = []
    used = set()
    for name, t in live.items():
        if name not in keys:
            continue
        used.add(name)
        want = f.get_tensor(name).to(torch.float64)
        got = t.detach().to(torch.float64)
        checked += 1
        if want.shape != got.shape or not torch.equal(want, got):
            mismatched += 1
            if len(examples) < 4:
                examples.append(name)
    # Which file tensors the model never took: only the K/V projections and norms of the shared-KV blocks
    # (they are dead weights; the authors list them in `_keys_to_ignore_on_load_unexpected`).
    unused = sorted(keys - used)
    surprising = [k for k in unused if not any(s in k for s in (".self_attn.k_proj.", ".self_attn.v_proj.",
                                                                ".self_attn.k_norm.", ".self_attn.v_norm."))]
    clip_bufs = [n for n in live if n.endswith(("input_min", "input_max", "output_min", "output_max"))]
    finite = sum(1 for n in clip_bufs if torch.isfinite(live[n]).all())
    rep = {"dtype": str(dtype), "checked_by_value": checked, "mismatched": mismatched, "examples": examples,
           "file_tensors_unused": len(unused), "unused_not_shared_kv": surprising,
           "clip_buffers": len(clip_bufs), "clip_buffers_finite": finite}
    if mismatched or surprising:
        raise SystemExit(f"load check failed: {rep}")
    if clip_bufs and finite == 0:
        raise SystemExit("every clipping bound is infinite — the clipped linears were re-initialised, refusing")
    return m, rep


def g(v, d):
    return float(f"{float(v):.{d}g}")


def rows_record(x, rows, cols):
    """Per recorded row: sum and sum of squares over the WHOLE row, and the values at `cols`."""
    x = x.detach().to(torch.float64)
    return {"sum": [g(x[r].sum(), 10) for r in rows], "ssq": [g((x[r] * x[r]).sum(), 10) for r in rows],
            "sample": [[g(v, 8) for v in x[r, cols]] for r in rows]}


def pick_rows(n, k, seed):
    if n <= k:
        return list(range(n))
    rest = sorted(random.Random(seed).sample(range(4, n - 4), k - 8))
    return list(range(4)) + rest + list(range(n - 4, n))


def logit_rows(lg, sample):
    out = []
    for t in range(lg.shape[0]):
        row = lg[t]
        top = torch.topk(row, 10)
        out.append({"top": [[int(i), round(float(v), 6)] for v, i in zip(top.values, top.indices)],
                    "sample": [round(float(row[i]), 6) for i in sample],
                    "sum": float(row.double().sum()), "ssq": float((row.double() ** 2).sum())})
    return out


def wav_read(path):
    """16-bit PCM / float32 WAV, mono — the same reader contract as Ferric's (ferric-serve audio.rs)."""
    import wave
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"{path}: {8 * sw}-bit WAV; this generator reads 16-bit PCM")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    a = a.reshape(-1, ch).mean(1) if ch > 1 else a
    return a, sr, hashlib.sha256(np.frombuffer(raw, dtype="<i2").tobytes()).hexdigest()


def prepare(kind, snap, path, question, extra_ids=None):
    """The authors' chat template and processor -> (enc, prompt ids, info)."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(snap)
    info = {"processor": type(proc).__name__}
    if kind == "image":
        from PIL import Image
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], images=[img], return_tensors="pt")
        info.update(image_processor=type(proc.image_processor).__name__, image_size=[img.height, img.width],
                    cpu_capability=torch.backends.cpu.get_cpu_capability())
    else:
        a, sr, sha = wav_read(path)
        fe = proc.feature_extractor
        if sr != fe.sampling_rate:
            raise SystemExit(f"{path} is {sr} Hz; the reference takes {fe.sampling_rate} Hz audio as-is "
                             f"(resampling is a separate, separately-checked step)")
        # Text FIRST, then the audio: the order of the authors' own audio example (model card).
        msgs = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "audio"}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], audio=[a], return_tensors="pt")
        info.update(feature_extractor=type(fe).__name__, samples=len(a), sample_rate=sr, pcm16_sha256=sha)
    info["prompt_text"] = text
    ids = enc["input_ids"][0].tolist()
    if extra_ids:
        enc["input_ids"] = torch.tensor([ids + extra_ids])
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        if "mm_token_type_ids" in enc:
            enc["mm_token_type_ids"] = torch.cat(
                [enc["mm_token_type_ids"], torch.zeros(1, len(extra_ids), dtype=enc["mm_token_type_ids"].dtype)], 1)
    return proc, enc, ids, info


def run(kind, m, enc, dtype, taps_on=True):
    """The authors' whole-model forward, with hooks at the stages Ferric reports."""
    taps, hooks = {}, []
    mm = m.model

    def keep(name, pick=None):
        def h(_m, _a, o):
            t = o if pick is None else pick(o)
            taps[name] = t.detach().clone()
        return h
    if taps_on and kind == "image":
        vt = mm.vision_tower
        hooks.append(vt.patch_embedder.register_forward_hook(keep("patch_embed")))
        n = len(vt.encoder.layers)
        for i in VISION_TAPS:
            j = i % n
            hooks.append(vt.encoder.layers[j].register_forward_hook(keep(f"vblock{j}")))
        hooks.append(vt.pooler.register_forward_hook(keep("pooled", lambda o: o[0][o[1]])))
        hooks.append(mm.embed_vision.register_forward_hook(keep("soft")))
    if taps_on and kind == "audio":
        at = mm.audio_tower
        hooks.append(at.subsample_conv_projection.register_forward_hook(keep("subsample", lambda o: o[0])))
        n = len(at.layers)
        for i in AUDIO_TAPS:
            j = i % n
            hooks.append(at.layers[j].register_forward_hook(keep(f"ablock{j}")))
        hooks.append(at.output_proj.register_forward_hook(keep("output_proj")))
        hooks.append(mm.embed_audio.register_forward_hook(keep("soft")))
    kw = {k: v for k, v in enc.items() if k not in ("num_soft_tokens_per_image",)}
    for k in ("pixel_values", "input_features"):
        if k in kw:
            kw[k] = kw[k].to(dtype)
    with torch.no_grad():
        out = m(**kw, use_cache=False)
    for h in hooks:
        h.remove()
    return out.logits[0].detach(), taps


def stage_records(taps, taps64, n_real, kind):
    stages, stages64 = {}, {}
    for name, t in taps.items():
        t = t.reshape(-1, t.shape[-1])
        t64 = taps64[name].reshape(-1, t.shape[-1])
        if name in ("patch_embed",) or name.startswith("vblock"):
            t, t64 = t[:n_real], t64[:n_real]          # the real patches; padding rows are never read
        if kind == "audio" and name in ("subsample", "output_proj") or name.startswith("ablock") or \
                (kind == "audio" and name == "soft"):
            t, t64 = t[:n_real], t64[:n_real]          # the audio frames the mask keeps
        rows = pick_rows(t.shape[0], 128, f"{name}/rows/20261001")
        cols = sorted(random.Random(f"{name}/20261001").sample(range(t.shape[1]), 24))
        stages[name] = {"shape": list(t.shape), "rows": rows, "cols": cols,
                        "max_abs": g(t.abs().max(), 8), **rows_record(t, rows, cols)}
        stages64[name] = rows_record(t64, rows, cols)
    return stages, stages64


def main():
    args = [a for a in sys.argv[1:]]
    mode = args.pop(0)
    import transformers
    meta = {"code": f"transformers built-in: {M.__name__} (the checkpoint ships no modeling file)",
            "transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__}
    if mode in ("greedy", "generate"):
        kind, snap, path, question, arg = args
        extra = [int(x) for x in arg.split(",")][:-1] if mode == "greedy" else None
        proc, enc, ids, info = prepare(kind, snap, path, question, extra)
        m, rep = load(snap, torch.float32)
        if mode == "greedy":
            cont = [int(x) for x in arg.split(",")]
            lg, _ = run(kind, m, enc, torch.float32, taps_on=False)
            lg = lg[len(ids) - 1:]
            top2 = torch.topk(lg, 2, dim=1)
            am = top2.indices[:, 0].tolist()
            json.dump({**meta, "mode": "greedy", "kind": kind, "snapshot": os.path.basename(snap.rstrip("/")),
                       "question": question, "file": os.path.basename(path), "load": rep, **info,
                       "prompt_ids": ids, "continuation": cont, "authors_argmax": am,
                       "margin": [round(float(a - b), 5) for a, b in top2.values.tolist()],
                       "agrees": [a == c for a, c in zip(am, cont)]}, sys.stdout)
        else:
            kw = {k: v for k, v in enc.items() if k not in ("num_soft_tokens_per_image",)}
            with torch.no_grad():
                out = m.generate(**kw, do_sample=False, max_new_tokens=int(arg), top_k=None, top_p=None,
                                 temperature=None)
            new = out[0, len(ids):].tolist()
            json.dump({**meta, "mode": "generate", "kind": kind, "snapshot": os.path.basename(snap.rstrip("/")),
                       "question": question, "file": os.path.basename(path), "load": rep, **info,
                       "prompt_ids": ids, "generated": new,
                       "text": proc.tokenizer.decode(new, skip_special_tokens=False)}, sys.stdout)
        return

    kind = mode
    snap, path, question = args[:3]
    proc, enc, ids, info = prepare(kind, snap, path, question)
    cfg = cfg_at(snap, torch.float32)
    fx = {**meta, "kind": kind, "snapshot": os.path.basename(snap.rstrip("/")), "model_type": cfg.model_type,
          "question": question, "file": os.path.basename(path), **info, "ids": ids,
          "image_token_id": cfg.image_token_id, "audio_token_id": cfg.audio_token_id}
    if kind == "image":
        pv, pos = enc["pixel_values"][0], enc["image_position_ids"][0]
        n_real = int((pos[:, 0] >= 0).sum())
        ph, pw = int(pos[:n_real, 1].max()) + 1, int(pos[:n_real, 0].max()) + 1
        ps = cfg.vision_config.patch_size
        img = pv[:n_real].reshape(ph, pw, ps, ps, 3).permute(0, 2, 1, 3, 4).reshape(ph * ps, pw * ps, 3)
        lv = img * 255.0
        if (lv - lv.round()).abs().max() > 1e-3:
            raise SystemExit("pixel_values are not 8-bit levels / 255 — the resize no longer quantises")
        # The rescale, bit for bit: the torchvision backend multiplies the float32 level by float32(1/255)
        # — NOT level/255 (that differs by an ulp on some levels). Asserted so the fixture says which.
        rf = torch.tensor(proc.image_processor.rescale_factor, dtype=torch.float32)
        if not torch.equal(pv[:n_real], (img * 255.0).round().reshape(ph, ps, pw, ps, 3).permute(0, 2, 1, 3, 4)
                           .reshape(n_real, -1).to(torch.float32) * rf):
            raise SystemExit("pixel_values != float32(level) * float32(rescale_factor) — the rescale changed")
        lv = lv.round().to(torch.uint8).numpy()
        fx["pixels"] = {"rescale": "float32(level) * float32(rescale_factor), bit-exact","resized_hw": [ph * ps, pw * ps], "patch_grid": [ph, pw], "real_patches": n_real,
                        "padded_patches": int(pv.shape[0]), "soft_tokens": n_real // cfg.vision_config.pooling_kernel_size ** 2,
                        "levels_sha256": hashlib.sha256(lv.tobytes()).hexdigest(),
                        "levels_note": "sha256 of the resized image's 8-bit levels, HWC row-major (pixel_values * 255)"}
        positions = pos[:n_real].tolist()
        if positions != [[x, y] for y in range(ph) for x in range(pw)]:
            raise SystemExit("image_position_ids are not the raster (x, y) grid this fixture assumes")
    else:
        feats, fmask = enc["input_features"][0], enc["input_features_mask"][0]
        n_frames = int(fmask.sum())
        cols = sorted(random.Random("features/20261001").sample(range(feats.shape[1]), 24))
        rows = list(range(feats.shape[0]))
        fx["features"] = {"shape": list(feats.shape), "valid_frames": n_frames, "mask": fmask.int().tolist(),
                          "rows": rows, "cols": cols, **rows_record(feats, rows, cols)}
        n_audio = sum(1 for x in ids if x == cfg.audio_token_id)
        n_real = n_audio
    fx["mm_token_type_ids"] = enc["mm_token_type_ids"][0].tolist() if "mm_token_type_ids" in enc else None

    sample = sorted(random.Random(20260924).sample(range(cfg.text_config.vocab_size), 128))
    fx["sample_ids"] = sample

    m, rep = load(snap, torch.float32)
    lg, taps = run(kind, m, enc, torch.float32)
    text_lg = None
    if kind == "image":
        # The same question with no image: the text decoder alone, so a defect there is not read as a splice one.
        tmsgs = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        ttext = proc.apply_chat_template(tmsgs, add_generation_prompt=True, tokenize=False)
        tids = proc.tokenizer(ttext, add_special_tokens=False)["input_ids"]
        fx["text_ids"] = tids
        with torch.no_grad():
            text_lg = m(input_ids=torch.tensor([tids]), use_cache=False).logits[0]
    del m
    gc.collect()
    m64, rep64 = load(snap, torch.float64)
    lg64, taps64 = run(kind, m64, enc, torch.float64)
    text_lg64 = None
    if kind == "image":
        with torch.no_grad():
            text_lg64 = m64(input_ids=torch.tensor([fx["text_ids"]]), use_cache=False).logits[0]
    del m64
    gc.collect()

    stages, stages64 = stage_records(taps, taps64, n_real, kind)
    fx.update({"load": rep, "stages": stages, "vocab": int(lg.shape[1]), "rows": logit_rows(lg, sample),
               "float64": {"note": "the same code at float64 (its explicit float32 casts kept): the noise floor",
                           "load": rep64, "stages": stages64, "rows": logit_rows(lg64, sample)}})
    if text_lg is not None:
        fx["text_rows"] = logit_rows(text_lg, sample)
        fx["float64"]["text_rows"] = logit_rows(text_lg64, sample)
    json.dump(fx, sys.stdout)


if __name__ == "__main__":
    main()
