#!/usr/bin/env python3
"""Qwen2.5-VL — MiMo-Embodied-7B — from the MODEL AUTHORS' OWN CODE, image file to logits, stage by stage.

The authors' implementation is transformers' `modeling_qwen2_5_vl.py` (Xiaomi ship no modeling file of
their own: the checkpoint names `Qwen2_5_VLForConditionalGeneration`). Every stage below runs THAT code:

  pixels   the authors' processor (`AutoProcessor` from their snapshot), on a committed PPM
  tower    `Qwen2_5_VisionTransformerPretrainedModel`, float32, eager — the checkpoint stores the tower in
           F32, so this is the authors' weights exactly; loaded strict and checked BY VALUE
  merger   its output BEFORE and AFTER the window un-permutation, so a missing reverse is attributable
  rope     the authors' `get_rope_index` on their own token-type ids
  LM       `Qwen2_5_VLTextModel` STREAMED one decoder layer at a time (refgen/stream.py) — 30 GB in
           float32 does not fit beside anything else — with the image rows spliced in where the authors'
           `masked_scatter` puts them, and the LM head applied in vocabulary chunks

⛔ THE HARNESS IS ITSELF CHECKED. The streamed path re-wires the authors' forward (splice, positions,
head) by hand, and a wiring error there would read as a Ferric defect. `--selftest` builds a TINY
random Qwen2.5-VL with the authors' class, saves it in this checkpoint's key layout, and requires the
streamed path to reproduce the authors' own `Qwen2_5_VLForConditionalGeneration.forward` — logits and
the tower's output — to the last bit. It runs before every fixture is written.

Per stage the fixture records, compactly: per-row sum and sum of squares over the WHOLE row, plus a fixed
seeded sample of columns. Logits use lm_logits_ref.py's record (top-10, 128 sampled ids, sum, ssq).

    <python> qwen25vl_ref.py <model_id> <image.ppm> "<question>" | gzip -9 > fixture.json.gz
    <python> qwen25vl_ref.py --selftest
    <python> qwen25vl_ref.py --greedy <model_id> <image.ppm> "<question>" <id,id,...> > greedy.json

`--greedy` checks a DECODE, not a prefill: given a continuation some runtime generated greedily, it runs
the authors' forward ONCE over prompt + continuation and records their argmax at every continuation
position, with the top-1/top-2 margin. If each argmax is the next token, the authors' own greedy decode
produces exactly this continuation (by induction on the steps) — without 64 streamed forwards.
"""
import gc
import json
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream  # noqa: E402
from refload import _f32_config  # noqa: E402

from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M  # noqa: E402

# Vision blocks whose outputs are recorded: 0 is WINDOWED, 7 is the first FULL block, 8 the first windowed
# block after one, 31 the last. A defect in the window schedule shows at 0 or 7; a drift over depth at 31.
TAP_BLOCKS = [0, 7, 8, 31]


def f32_everywhere(cfg):
    cfg.dtype = torch.float32
    for sub in ("text_config", "vision_config"):
        if getattr(cfg, sub, None) is not None:
            getattr(cfg, sub).dtype = torch.float32
    return cfg


def load_tower(cfg, ckpt, report, dtype=torch.float32):
    """The authors' vision tower, float32, every tensor from the file, checked by value AS LOADED."""
    vc = cfg.vision_config
    vc._attn_implementation = "eager"
    tower = M.Qwen2_5_VisionTransformerPretrainedModel._from_config(vc, dtype=torch.float32)
    sd = {}
    for name, t in tower.state_dict().items():
        key = "visual." + name
        if key not in ckpt.where:
            raise SystemExit(f"tower parameter {name} has no checkpoint key {key} — refusing")
        v = ckpt.get(key)
        if tuple(v.shape) != tuple(t.shape):
            raise SystemExit(f"{key}: checkpoint {tuple(v.shape)} != tower {tuple(t.shape)} — refusing")
        sd[name] = v.to(torch.float32)
    tower.load_state_dict(sd, strict=True)
    if dtype != torch.float32:
        tower = tower.to(dtype)
        sd = {k: v.to(dtype) for k, v in sd.items()}
    live = tower.state_dict()
    bad = [n for n, v in sd.items() if not torch.equal(live[n], v)]
    if bad:
        raise SystemExit(f"{len(bad)} tower tensors differ from the file after loading ({bad[:3]})")
    if any(p.dtype != dtype for p in tower.parameters()):
        raise SystemExit(f"tower is not {dtype} as loaded — refusing")
    # The rotary table is a NON-persistent buffer: no checkpoint value to compare, so compare it to the
    # formula. transformers 5.x re-initialises buffers in `_init_weights`; this is where a re-init that
    # changed it would surface instead of reading as a Ferric rope defect.
    rp = tower.rotary_pos_emb
    want = 1.0 / (rp.theta ** (torch.arange(0, rp.dim, 2, dtype=torch.float) / rp.dim))
    if not torch.equal(rp.inv_freq, want.to(rp.inv_freq.dtype)):
        raise SystemExit("vision rotary inv_freq is not the formula's — refusing")
    tower.eval()
    if any(m.training for m in tower.modules()):
        raise SystemExit("tower has a module in training mode — refusing")
    report["tower_tensors"] = len(sd)
    report["tower_dtypes_in_file"] = sorted({ckpt.handles[ckpt.where[k]].get_slice(k).get_dtype()
                                             for k in ckpt.where if k.startswith("visual.")})
    return tower


def streamed_vl(snap, cfg, ids, mm, pv, grid, dtype=torch.float32):
    """The authors' Qwen2.5-VL forward with one LM decoder layer resident at a time.

    Returns the tower's stages, the window index, the authors' positions and the logits [T, V].
    `dtype=torch.float64` is the NOISE-FLOOR run (see main)."""
    ckpt = stream.Checkpoint(snap)
    report = {"mode": f"tower resident ({dtype}); LM streamed, one decoder layer resident",
              "layers_streamed": 0, "tensors_loaded": 0}

    # ---- vision: the authors' tower, with hooks at the stages Ferric reports ------------------------
    tower = load_tower(cfg, ckpt, report, dtype)
    taps = {}
    hooks = [tower.patch_embed.register_forward_hook(lambda m, a, o: taps.__setitem__("patch_embed", o.detach().clone())),
             tower.merger.register_forward_hook(lambda m, a, o: taps.__setitem__("merger_window_order", o.detach().clone()))]
    for b in TAP_BLOCKS:
        if b < len(tower.blocks):
            hooks.append(tower.blocks[b].register_forward_hook(
                lambda m, a, o, b=b: taps.__setitem__(f"block{b}", o.detach().clone())))
    with torch.no_grad():
        vout = tower(pv.to(dtype), grid_thw=grid)
        widx, cu = tower.get_window_index(grid)
    for h in hooks:
        h.remove()
    taps["merger"] = vout.pooler_output.detach().clone()
    del tower
    gc.collect()

    # ---- LM skeleton on the META device: structure only, the authors' own forward -------------------
    with torch.device("meta"):
        skel = M.Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation="eager")
    skel.eval()
    tcfg = skel.config.text_config
    body = skel.model.language_model
    layers = body.layers
    if len(layers) != tcfg.num_hidden_layers:
        raise SystemExit(f"skeleton has {len(layers)} layers, config says {tcfg.num_hidden_layers}")
    # ⚠ This checkpoint keeps the pre-5.x key layout (`model.layers.N`, `visual.*`), where the 5.x class
    # nests them (`model.language_model.layers.N`, `model.visual.*`). The prefixes are given EXACTLY
    # here, so every lookup is an exact key and stream.Checkpoint's suffix fallback is never consulted.
    for i in range(len(layers)):
        layers[i] = stream.StreamedLayer(type(layers[i]), tcfg, i, f"model.layers.{i}.", ckpt, report, dtype)
    body.rotary_emb = type(body.rotary_emb)(config=tcfg)
    body.norm = type(body.norm)(tcfg.hidden_size, eps=tcfg.rms_norm_eps).to(dtype)
    body.norm.load_state_dict({"weight": ckpt.get("model.norm.weight").to(dtype)}, strict=True)

    # The authors' position rule, on the authors' token types. get_rope_index touches no parameter, so
    # it runs on the meta skeleton; its tensors live on input_ids' device (CPU).
    input_ids = torch.tensor([ids])
    pos, delta = skel.model.get_rope_index(input_ids, mm_token_type_ids=mm, image_grid_thw=grid)
    if torch.equal(pos[0], pos[1]) and torch.equal(pos[1], pos[2]):
        raise SystemExit("the three position axes agree — no image was found and mRoPE never fired")

    # The splice: the authors' `masked_scatter` fills the image-token rows, in order, with the tower's
    # merged rows. Done here by index; the self-test proves it equals theirs.
    emb = ckpt.get("model.embed_tokens.weight")[input_ids[0]].to(dtype)
    img_rows = (input_ids[0] == cfg.image_token_id).nonzero().flatten()
    if img_rows.numel() != taps["merger"].shape[0]:
        raise SystemExit(f"{img_rows.numel()} image tokens for {taps['merger'].shape[0]} merged rows")
    emb[img_rows] = taps["merger"]
    with torch.no_grad():
        h = body(inputs_embeds=emb[None], position_ids=pos, use_cache=False).last_hidden_state[0]
    head_key = "lm_head.weight" if "lm_head.weight" in ckpt.where else "model.embed_tokens.weight"
    W = ckpt.get(head_key)
    logits = torch.empty(h.shape[0], W.shape[0], dtype=dtype)
    with torch.no_grad():
        for c in range(0, W.shape[0], 16384):
            logits[:, c:c + 16384] = h @ W[c:c + 16384].to(dtype).T
    if report["layers_streamed"] != len(layers):
        raise SystemExit(f"{report['layers_streamed']} of {len(layers)} layers ran — refusing")
    report["lm_head"] = head_key
    report["attn_implementation"] = {"text": tcfg._attn_implementation,
                                     "vision": cfg.vision_config._attn_implementation}
    return {"taps": taps, "window_index": widx.tolist(), "cu_window_seqlens": [int(x) for x in cu],
            "position_ids": pos[:, 0].tolist(), "rope_delta": int(delta.flatten()[0]),
            "logits": logits, "report": report}


def selftest():
    """Tiny random Qwen2.5-VL: the streamed harness must equal the authors' whole-model forward EXACTLY."""
    from safetensors.torch import save_file
    from transformers import Qwen2_5_VLConfig
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, rope_theta=640000.0, rms_norm_eps=1e-5,
                         max_position_embeddings=4096, rope_scaling={"type": "default", "mrope_section": [4, 2, 2]}),
        vision_config=dict(depth=4, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=112,
                           fullatt_block_indexes=[1, 3], in_channels=3),
        image_token_id=500, video_token_id=501, vision_start_token_id=502, vision_end_token_id=503)
    f32_everywhere(cfg)
    full = M.Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation="eager", dtype=torch.float32).eval()
    with torch.no_grad():   # non-trivial norms and biases, so a skipped one cannot hide
        for n, p in full.named_parameters():
            p.copy_(torch.randn_like(p) * (0.5 if p.ndim > 1 else 0.2) + (1.0 if n.endswith("norm.weight") or "ln_q" in n else 0.0))
    # A 10x14 patch grid: 5x7 merged tokens, windows of 4x4 merged -> 4 windows, two of them partial.
    gh, gw = 10, 14
    grid = torch.tensor([[1, gh, gw]])
    pv = torch.randn(gh * gw, 3 * 2 * 14 * 14)
    n_img = gh * gw // 4
    ids = [1, 2, 502] + [500] * n_img + [503, 7, 8, 9, 10, 11]
    mm = torch.tensor([[1 if x == 500 else 0 for x in ids]])
    with torch.no_grad():
        want = full(input_ids=torch.tensor([ids]), pixel_values=pv, image_grid_thw=grid,
                    mm_token_type_ids=mm).logits[0]
        want_img = full.model.visual(pv, grid_thw=grid).pooler_output
    # Save in THIS checkpoint's (pre-5.x) key layout — the layout the real run reads.
    sd = {}
    for k, v in full.state_dict().items():
        if k.startswith("model.visual."): k = "visual." + k[len("model.visual."):]
        elif k.startswith("model.language_model."): k = "model." + k[len("model.language_model."):]
        sd[k] = v.contiguous()
    with tempfile.TemporaryDirectory() as d:
        save_file(sd, os.path.join(d, "model.safetensors"))
        got = streamed_vl(d, cfg, ids, mm, pv, grid)
    dl = (got["logits"] - want).abs().max().item()
    di = (got["taps"]["merger"] - want_img).abs().max().item()
    if dl != 0.0 or di != 0.0:
        raise SystemExit(f"⛔ harness self-test: streamed != the authors' whole model (logits {dl:.3e}, "
                         f"image rows {di:.3e}) — the wiring is wrong, refusing to emit a fixture")
    return {"tiny_config": "vision depth 4 (full [1,3]), text 2 layers, grid 10x14 -> 4 windows",
            "max_abs_logit_diff": dl, "max_abs_image_row_diff": di}


def rows_record(x, cols):
    # 8 significant digits on the samples (5e-9 relative, far below the ~1e-6 f32 floor this measures),
    # 10 on the row sums — full `repr` doubled the fixture for digits no comparison can use.
    g = lambda v, d: float(f"{float(v):.{d}g}")
    x = x.detach().to(torch.float64)
    return {"sum": [g(v, 10) for v in x.sum(1)], "ssq": [g(v, 10) for v in (x * x).sum(1)],
            "sample": [[g(v, 8) for v in r] for r in x[:, cols]]}


def main():
    if sys.argv[1:] == ["--selftest"]:
        print(json.dumps(selftest()))
        return
    greedy = sys.argv[1] == "--greedy"
    if greedy:
        del sys.argv[1]
    model_id, img_path, question = sys.argv[1:4]
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    import transformers

    st = selftest()
    snap = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    cfg = f32_everywhere(_f32_config(snap))
    proc = AutoProcessor.from_pretrained(snap)
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=[img], return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    pv, grid, mm = enc["pixel_values"], enc["image_grid_thw"], enc["mm_token_type_ids"]
    if greedy:
        cont = [int(x) for x in sys.argv[4].split(",")]
        full = ids + cont[:-1]
        mm_full = torch.cat([mm, torch.zeros(1, len(cont) - 1, dtype=mm.dtype)], 1)
        r = streamed_vl(snap, cfg, full, mm_full, pv, grid)
        lg = r["logits"][len(ids) - 1:]
        top2 = torch.topk(lg, 2, dim=1)
        am = top2.indices[:, 0].tolist()
        json.dump({"model": model_id, "snapshot": os.path.basename(snap), "question": question,
                   "image": os.path.basename(img_path), "code": f"transformers built-in: {M.__name__}",
                   "transformers": transformers.__version__, "torch": torch.__version__, "dtype": "torch.float32",
                   "harness_selftest": st, "load": r["report"], "prompt_ids": ids, "continuation": cont,
                   "authors_argmax": am,
                   "margin": [round(float(a - b), 5) for a, b in top2.values.tolist()],
                   "agrees": [a == c for a, c in zip(am, cont)]}, sys.stdout)
        return
    r = streamed_vl(snap, cfg, ids, mm, pv, grid)
    # ⭐ THE NOISE FLOOR. The same code at float64 on the same pixels. From block 17 on, a few tokens carry
    # "massive activations" (|x| up to 5e4 in one channel), and the sums that produce them cancel: the
    # authors' own float32 run is ~1e-3 (ssq) away from float64 there. A float32 reference cannot then
    # arbitrate at 1e-4, and a gate that pretends it can either fails correct code or is loosened by
    # hand. So the fixture carries the float64 run, and the gate asks the objective question: is Ferric
    # as close to it as the authors' own float32 run is?
    # ⚠ Not pure float64: the authors' code casts to float32 explicitly inside every RMSNorm, the vision
    # rope, the attention softmax and the text rope's angles, and those casts are kept — this is their
    # code at the highest precision it runs at. The ill-conditioned sums are matmuls, which run at f64.
    r64 = streamed_vl(snap, cfg, ids, mm, pv, grid, dtype=torch.float64)
    if r64["window_index"] != r["window_index"] or r64["position_ids"] != r["position_ids"]:
        raise SystemExit("the float64 run disagrees on the window index or positions — refusing")

    rng = random.Random(20260926)
    px_cols = sorted(rng.sample(range(pv.shape[1]), 24))
    stages, stages64 = {}, {}
    for name, t in r["taps"].items():
        cols = sorted(random.Random(f"{name}/20260926").sample(range(t.shape[1]), 24))
        order = "window" if name.startswith("block") or name == "merger_window_order" else "sweep"
        stages[name] = {"order": order, "shape": list(t.shape), "cols": cols, **rows_record(t, cols)}
        stages64[name] = rows_record(r64["taps"][name], cols)
    lg = r["logits"]
    V = lg.shape[1]
    sample = sorted(random.Random(20260924).sample(range(min(V, cfg.text_config.vocab_size)), 128))
    def logit_rows(lg):
        rows = []
        for t in range(lg.shape[0]):
            row = lg[t]
            top = torch.topk(row, 10)
            rows.append({"top": [[int(i), round(float(v), 6)] for v, i in zip(top.values, top.indices)],
                         "sample": [round(float(row[i]), 6) for i in sample],
                         "sum": float(row.double().sum()), "ssq": float((row.double() ** 2).sum())})
        return rows
    rows, rows64 = logit_rows(lg), logit_rows(r64["logits"])
    json.dump({
        "model": model_id, "snapshot": os.path.basename(snap),
        "architectures": cfg.architectures, "model_type": cfg.model_type,
        "code": f"transformers built-in: {M.__name__} (the checkpoint ships no modeling file)",
        "processor": f"{type(proc).__name__} / {type(proc.image_processor).__name__}",
        "transformers": transformers.__version__, "torch": torch.__version__,
        "dtype": "torch.float32", "checkpoint_dtype": {"text": "bfloat16", "vision": "float32"},
        "harness_selftest": st, "load": r["report"],
        "question": question, "image": os.path.basename(img_path), "image_size": [img.height, img.width],
        "ids": ids, "types": mm[0].tolist(), "grid": grid[0].tolist(),
        "window_index": r["window_index"], "cu_window_seqlens": r["cu_window_seqlens"],
        "position_ids": r["position_ids"], "rope_delta": r["rope_delta"],
        "pixels": {"shape": list(pv.shape), "cols": px_cols, **rows_record(pv, px_cols)},
        "stages": stages, "vocab": V, "sample_ids": sample, "rows": rows,
        "float64": {"note": "the same code at float64 (its explicit float32 casts kept): the noise floor",
                    "load": r64["report"], "stages": stages64, "rows": rows64},
    }, sys.stdout)


if __name__ == "__main__":
    main()
