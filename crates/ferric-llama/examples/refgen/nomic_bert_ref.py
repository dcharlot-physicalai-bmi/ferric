#!/usr/bin/env python3
"""nomic-embed-text reference from the MODEL AUTHORS' OWN implementation — `modeling_hf_nomic_bert.py`
(nomic-ai/nomic-bert-2048, loaded through `trust_remote_code`), not llama.cpp.

Records, for each text, the tokenizer's ids and the encoder at every stage — the embedding LayerNorm and
each block's output — at sampled rows and columns, plus every row's sum and sum of squares, plus the
sentence embedding the authors ship (sentence-transformers mean pooling from 1_Pooling/config.json,
then L2 normalisation). Everything TWICE: float32, and float64 — the second is the noise floor the gate
measures against, because a float32 reference cannot arbitrate below its own distance from exact
arithmetic. ⚠ The authors' rotary module builds its angles in float32 even in a float64 model
(`pos_idx_in_fp32=True`, then `.to(dtype)`); that is their code, left as it is.

    <python with torch, transformers 5.3, einops> nomic_bert_ref.py nomic-embed-text-v1.5.json.gz

⚠ The JSON goes to a FILE, not stdout: transformers prints its remote-code prompt on stdout, and a first
version's reference began with that prompt instead of JSON.
"""
import gzip, json, sys

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

MODEL = "nomic-ai/nomic-embed-text-v1.5"
TEXTS = [
    "search_query: What is TSNE?",
    "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van der Maaten",
    "classification: the new phone's battery lasts two full days, and I love it!",
    # long: RoPE positions far past anything a short query reaches (and past BERT's 512)
    "search_document: " + " ".join(
        f"Section {i}: the harbour lights at dawn guided {i * 7 % 13} ships past the breakwater while "
        f"gulls circled the {['north', 'south', 'east', 'west'][i % 4]} pier." for i in range(40)),
]
COLS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 400, 511, 512, 600, 700, 760, 764, 765, 766, 767]


def rows_for(t):
    if t <= 64:
        return list(range(t))
    return sorted(set(list(range(8)) + list(range(8, t - 4, 37)) + list(range(t - 4, t))))


def run(model, ids):
    taps = []
    hooks = [model.emb_ln.register_forward_hook(lambda m, i, o: taps.append(("emb_ln", o[0])))]
    for il, blk in enumerate(model.encoder.layers):
        hooks.append(blk.register_forward_hook(
            lambda m, i, o, il=il: taps.append((f"l{il}", (o[0] if isinstance(o, tuple) else o)[0]))))
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in hooks:
        h.remove()
    last = out.last_hidden_state[0]
    emb = last.mean(0)  # every row is real: no padding, so mean pooling is the plain mean
    return taps, emb, torch.nn.functional.normalize(emb, dim=0)


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m32 = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()
m64 = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval().double()
cfg = m32.config
out = {
    "model": MODEL, "transformers": transformers.__version__, "torch": torch.__version__,
    "modeling": type(m32).__module__,
    "config": {k: getattr(cfg, k, None) for k in ["activation_function", "rotary_emb_base", "rotary_emb_fraction",
                                                  "rotary_emb_interleaved", "rotary_scaling_factor", "prenorm",
                                                  "qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias", "layer_norm_epsilon"]},
    "cols": COLS, "items": [],
}
for text in TEXTS:
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"]
    t = ids.shape[1]
    rows = rows_for(t)
    item = {"text": text, "ids": ids[0].tolist(), "rows": rows, "stages": {}}
    for tag, model in (("f32", m32), ("f64", m64)):
        taps, emb, nrm = run(model, ids)
        for name, x in taps:
            x = x.double()
            st = item["stages"].setdefault(name, {})
            st[tag] = {
                "rows": [[x[r, c].item() for c in COLS] for r in rows],
                "sum": x.sum(1).tolist(), "ssq": (x * x).sum(1).tolist(),
            }
        item.setdefault("emb", {})[tag] = emb.double().tolist()
        item.setdefault("emb_norm", {})[tag] = nrm.double().tolist()
    out["items"].append(item)
    print(f"{t:4d} tokens  {text[:60]!r}", file=sys.stderr)
with gzip.open(sys.argv[1], "wt") as f:
    json.dump(out, f)
