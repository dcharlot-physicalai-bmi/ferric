#!/usr/bin/env python3
"""A causal LM's logits from the MODEL AUTHORS' OWN implementation, at float32 AND float64 — the noise floor.

`lm_logits_ref.py` records one float32 run and its gate uses a fixed tolerance (1e-3). That is enough for
a short prompt at small positions. It is NOT enough for the two questions this generator exists for,
where the size of an honest difference is unknown until it is measured:

  * Phi-3 / Phi-3.5 LongRoPE (`--min-tokens`): a prompt past `original_max_position_embeddings` switches
    the whole rotary table to the long factors. 4,600 tokens through a 3.8B model accumulate more
    float32 rounding than 150 do, so the tolerance must come from the authors' own float32 distance.
  * Rope angle precision at large positions (`--pos-offset`): the authors' rotary module builds its
    angles in float32 — `inv_freq(f32) x position(f32)` — so their float32 run carries its own angle
    rounding at position 30,000. Is Ferric further from float64 than that?

Both runs are the authors' code UNMODIFIED. ⚠ Their float64 run is not pure float64 and must not be made
so: their rotary module casts to float32 (`inv_freq.float() @ position_ids.float()`) and their RMSNorm
computes its variance in float32 — so the float64 run SHARES the float32 angles. That is the right floor
for the question: a port whose angles differ from the authors' shows up as distance the floor does not
contain (nomic-bert: 4.48x the floor with device-computed angles, 2.11x with the authors' table).

Every decoder layer is streamed (refgen/stream.py): one layer resident at a time, built by the authors'
class, loaded strict from the checkpoint and checked by value — so a 3.8B model runs at float64 in a few
GB, beside the runtime under test.

`--pos-offset P` feeds `position_ids = P .. P+T-1` to the authors' forward with no cache: the same T x T
causal attention over the same tokens, rotated as if they sat at position P. Ferric's side sets its cache
position to P before an empty-cache prefill, which is the identical computation. A short sequence thus
reaches angles of 30,000 rad without a 30k x 30k attention matrix on either side.

The fixture records, per recorded position and per dtype: the logits at a fixed seeded sample of 128
vocabulary ids, and the sum and sum of squares of the FULL row; plus the float64 top-10. And the rotary
table the authors' forward actually used (read back from their module AFTER the forward, so a dynamic
switch like LongRoPE's is recorded as it happened, not as expected).

  <python> lm_floor_ref.py <model_id> <out.json.gz> [--min-tokens N] [--pos-offset P] [--attn sdpa|eager]
                           [--stride S] [--tail K]
"""
import argparse
import gc
import gzip
import json
import os
import random
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream  # noqa: E402

# The same text lm_logits_ref.py uses, so the two families of fixtures exercise the same tokens.
TEXT = ("The heron stood motionless in the shallows while the tide turned. A fisherman on the far bank "
        "counted his nets twice, then a third time, because the numbers never agreed. Measurements, he "
        "thought, are only as honest as the instrument and the person reading it. In 1854 John Snow "
        "mapped cholera deaths around a single water pump on Broad Street; the pattern was plain once the "
        "points were on paper. def mean(xs): return sum(xs) / len(xs)  # an empty list divides by zero. "
        "Über den Wolken muss die Freiheit wohl grenzenlos sein. 月が綺麗ですね。 The answer is 42.")

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--min-tokens", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=0, help="truncate the ids to this many (0 = no cut)")
ap.add_argument("--pos-offset", type=int, default=0)
ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
ap.add_argument("--stride", type=int, default=0, help="record every S-th position (0 = every position)")
ap.add_argument("--tail", type=int, default=64, help="with --stride, also record the last K positions")
A = ap.parse_args()

tok = AutoTokenizer.from_pretrained(A.model)
text = TEXT
while A.min_tokens and len(tok(text, add_special_tokens=True)["input_ids"]) < A.min_tokens:
    text += " " + TEXT
ids = tok(text, add_special_tokens=True)["input_ids"]
if A.max_tokens:
    ids = ids[:A.max_tokens]
T = len(ids)
positions = (list(range(T)) if not A.stride
             else sorted(set(range(0, T, A.stride)) | set(range(max(0, T - A.tail), T))))
snap = snapshot_download(A.model, allow_patterns=["*.json", "*.safetensors"])
cfg0 = AutoConfig.from_pretrained(A.model)
CHECKPOINT_DTYPE = str(getattr(cfg0, "dtype", None) or "unknown").replace("torch.", "")


def streamed(dtype):
    """The authors' forward, one decoder layer resident, every tensor at `dtype` (upcasts are exact)."""
    cfg = AutoConfig.from_pretrained(A.model)
    cfg.dtype = dtype
    ckpt = stream.Checkpoint(snap)
    report = {"mode": f"streamed, one decoder layer resident, {dtype}", "layers_streamed": 0, "tensors_loaded": 0}
    with torch.device("meta"):
        skel = AutoModelForCausalLM.from_config(cfg, attn_implementation=A.attn)
    skel.eval()
    # The skeleton's OWN config from here on: it carries the attention implementation the masks and every
    # layer's forward consult, which the config passed in may not (from_config can copy it).
    cfg = skel.config
    if cfg._attn_implementation != A.attn:
        raise SystemExit(f"skeleton runs {cfg._attn_implementation} attention, not {A.attn} — refusing")
    mod_name = type(skel).__module__
    body = skel.model
    layers = body.layers
    if len(layers) != cfg.num_hidden_layers:
        raise SystemExit(f"skeleton has {len(layers)} layers, config says {cfg.num_hidden_layers}")
    prefix = next(n for n, m in skel.named_modules() if m is layers) + "."
    for i in range(len(layers)):
        layers[i] = stream.StreamedLayer(type(layers[i]), cfg, i, f"{prefix}{i}.", ckpt, report, dtype=dtype)
    # Non-persistent buffers are COMPUTED by the authors' constructor; rebuilt on the CPU (the skeleton's
    # are on the meta device and hold nothing).
    body.rotary_emb = type(body.rotary_emb)(config=cfg)
    body.norm = type(body.norm)(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dtype)
    body.norm.load_state_dict({"weight": ckpt.get(ckpt.resolve(prefix[: -len("layers.")] + "norm.weight")).to(dtype)},
                              strict=True)
    # ⛔ The rows are gathered from the checkpoint, which is the authors' embedding ONLY when their module
    # is a plain lookup. Gemma 3's `Gemma3TextScaledWordEmbedding` multiplies by sqrt(hidden) in its
    # forward; gathering skipped it, and the "reference" moved 20.8 logits — caught because their full
    # model is translation-invariant (position_ids 30000.. vs 0..: 7.1e-3) and the streamed run was not.
    if type(body.embed_tokens) is not torch.nn.Embedding:
        raise SystemExit(f"{type(body.embed_tokens).__name__} is not a plain lookup — gathering its weight rows "
                         f"would skip what its forward does; refusing")
    emb_key = ckpt.resolve(prefix[: -len("layers.")] + "embed_tokens.weight")
    emb = ckpt.get(emb_key)[torch.tensor(ids)].to(dtype)[None]
    pos = torch.arange(A.pos_offset, A.pos_offset + T)[None]
    with torch.no_grad():
        h = body(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state[0]
    if report["layers_streamed"] != len(layers):
        raise SystemExit(f"{report['layers_streamed']} of {len(layers)} layers ran — refusing")
    head_key = emb_key if getattr(cfg, "tie_word_embeddings", False) else ckpt.resolve("lm_head.weight")
    W = ckpt.get(head_key)
    logits = torch.empty(T, W.shape[0], dtype=torch.float64)
    with torch.no_grad():
        for c in range(0, W.shape[0], 8192):
            logits[:, c:c + 8192] = (h @ W[c:c + 8192].to(dtype).T).to(torch.float64)
    rot = body.rotary_emb
    # AFTER the forward: what it used. A model with per-layer-type rotaries (Gemma 3: sliding / full)
    # keeps one table per type, `<type>_inv_freq`.
    tables = {"": rot.inv_freq} if hasattr(rot, "inv_freq") else {
        n[: -len("_inv_freq")]: b for n, b in rot.named_buffers() if n.endswith("_inv_freq") and not n.endswith("original_inv_freq")}
    rt = getattr(rot, "rope_type", "default")
    rope = {"rope_type": rt if isinstance(rt, str) else dict(rt),
            "attention_scaling": (float(rot.attention_scaling) if hasattr(rot, "attention_scaling") else
                                  {k: float(getattr(rot, f"{k}_attention_scaling", 1.0)) for k in tables}),
            "inv_freq": ([float(x) for x in tables[""].float()] if "" in tables else
                         {k: [float(x) for x in v.float()] for k, v in tables.items()})}
    rp = getattr(cfg, "rope_parameters", None) or {}
    if rope["rope_type"] == "longrope":
        base = float(rp["rope_theta"])
        dim = len(rope["inv_freq"]) * 2
        e = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        used = torch.tensor(rope["inv_freq"], dtype=torch.float32)
        for name in ("short_factor", "long_factor"):
            if torch.equal(used, 1.0 / (torch.tensor(rp[name], dtype=torch.float32) * base ** e)):
                rope["factors_used"] = name
        rope["original_max_position_embeddings"] = int(rp["original_max_position_embeddings"])
    report["lm_head"] = head_key
    del skel, body, W, h
    gc.collect()
    return logits, rope, report, cfg, mod_name


runs = {}
for name, dt in (("float32", torch.float32), ("float64", torch.float64)):
    print(f"{A.model}: {T} tokens at positions {A.pos_offset}..{A.pos_offset + T - 1}, {name}, {A.attn}", file=sys.stderr)
    runs[name] = streamed(dt)
lg32, rope32, rep32, cfg, MOD = runs["float32"]
lg64, rope64, rep64, _, _ = runs["float64"]
if rope32 != rope64:
    raise SystemExit("the float32 and float64 runs used different rotary tables — refusing")
V = int(lg32.shape[1])
sample = sorted(random.Random(20260924).sample(range(min(V, cfg.vocab_size)), 128))


def rows(lg, top):
    out = []
    for t in positions:
        r = lg[t]
        row = {"sample": [round(float(r[i]), 6) for i in sample],
               "sum": float(r.sum()), "ssq": float((r ** 2).sum())}
        if top:
            tv = torch.topk(r, 10)
            row["top"] = [[int(i), round(float(v), 6)] for v, i in zip(tv.values, tv.indices)]
        out.append(row)
    return out


fx = {
    "model": A.model,
    "model_type": cfg.model_type,
    "code": f"transformers built-in: {MOD} — layers streamed one at a time (refgen/stream.py)",
    "attn_implementation": A.attn,
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "checkpoint_dtype": CHECKPOINT_DTYPE,
    "load": {"float32": rep32, "float64": rep64},
    "note": "float64 = the same code at float64, the authors' explicit float32 casts kept (rotary angles, "
            "RMSNorm variance): the noise floor",
    "vocab": V,
    "ids": ids,
    "position_offset": A.pos_offset,
    "positions": positions,
    "sample_ids": sample,
    "rope": rope32,
    "float32": rows(lg32, False),
    "float64": rows(lg64, True),
}
with gzip.open(A.out, "wt") as f:
    json.dump(fx, f)
d = [float((lg32[t, sample] - lg64[t, sample]).abs().max()) for t in positions]
print(f"wrote {A.out}: {T} tokens, {len(positions)} positions recorded; float32-vs-float64 over the sample: "
      f"median {sorted(d)[len(d) // 2]:.2e}, max {max(d):.2e}; rope {rope32.get('rope_type')} "
      f"{rope32.get('factors_used', '')} attention_scaling {rope32['attention_scaling']}", file=sys.stderr)
