#!/usr/bin/env python3
"""BERT / XLM-R reference from the MODEL AUTHORS' OWN implementation (HF `transformers`), not llama.cpp.

⛔⛔ WHY THIS EXISTS. `bert` was marked Verified against `llama-embedding` only. llama.cpp is a port, as
Ferric is; the reference is what the model's authors run. The same move on ModernBERT (same day) found
a real defect that the llama.cpp comparison had been explained away: the head pooled the wrong token.

⛔ One choice in bert.rs was made TO MATCH llama.cpp rather than the model: GELU defaults to the tanh
approximation because ggml's `ggml_gelu` is tanh. The authors' configs say `hidden_act: "gelu"`, which
in `transformers` is the EXACT erf form. This reference records the authors' activation so the two can
be compared rather than assumed.

⛔ THIS LIVES IN THE REPO ON PURPOSE: a generator kept in scratch is a verification nobody can repeat.

Writes JSON to stdout:
  embed:  BAAI/bge-small-en-v1.5 (BertModel) — per text: ids, cls_hidden, mean_hidden, and the
          sentence-transformers pooling the AUTHORS ship (read from 1_Pooling/config.json, not assumed)
  rerank: BAAI/bge-reranker-v2-m3 (XLMRobertaForSequenceClassification) — per pair: ids, logit,
          cls_hidden (the classifier head reads token 0 by construction)

Run in float32 with eager attention:  <python-with-transformers> bert_ref.py > ref.json
"""
import json
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# ⛔⛔ LOAD AT FULL PRECISION, AND CHECK IT *AS LOADED* — NEVER AFTER A CAST.
#
# Two failures, found in order on Qwen/Qwen3.5-0.8B under transformers 5.7.0, both of which produced
# a "reference" that read as a defect in Ferric:
#   1. `from_pretrained(dtype=torch.float32)` is IGNORED for a COMPOSITE config (a multimodal checkpoint
#      whose text model carries `text_config.dtype = bfloat16`). The model loads in bf16 and every one
#      of the checkpoint's 36 F32-STORED tensors (the norms, the gate parameters) is ROUNDED on load —
#      the gated-norm weight arrived 3.9e-3 away from the file. Ferric, reading the file, was right.
#   2. The first guard written for (1) was VACUOUS: it called `.to(float32)` and THEN asserted float32,
#      which is always true. It checked the LABEL; the values had been rounded before the cast.
# So: set float32 on the config AND every sub-config, pass it in, and assert on the parameters exactly
# as `from_pretrained` returned them.
from transformers import AutoConfig

def _f32_config(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    def walk(c, seen):
        if id(c) in seen: return
        seen.add(id(c)); c.dtype = torch.float32
        for name in list(getattr(type(c), "sub_configs", {}) or {}) + ["text_config", "vision_config", "audio_config"]:
            sub = getattr(c, name, None)
            if sub is not None and hasattr(sub, "to_dict"): walk(sub, seen)
    walk(cfg, set())
    return cfg

def load_f32(cls, model_id, **kw):
    m = cls.from_pretrained(model_id, config=_f32_config(model_id), dtype=torch.float32, **kw)
    bad = sorted({str(p.dtype) for p in m.parameters() if p.dtype != torch.float32})   # AS LOADED
    if bad:
        raise SystemExit(f"{model_id} loaded as {bad}, not float32 — refusing to emit a fixture")
    return m


EMBED = "BAAI/bge-small-en-v1.5"
RERANK = "BAAI/bge-reranker-v2-m3"
TEXTS = [
    "The giant panda is a bear species endemic to China.",
    "how do I bake bread?",
    # a long one: learned positions go to 512, so exercise far rows, not only the first dozen
    " ".join(["the quick brown fox jumps over the lazy dog while seven ships sail past harbour "
              "lights before dawn breaks across water"] * 12),
]
PAIRS = [
    ("what is panda?", "The giant panda is a bear species endemic to China."),
    ("what is panda?", "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
    ("how do I bake bread?", "Mix flour, water, salt and yeast, then proof and bake."),
    ("how do I bake bread?", "The mitochondrion is the powerhouse of the cell."),
]

out = {"transformers": __import__("transformers").__version__, "torch": torch.__version__}

with torch.no_grad():
    tok = AutoTokenizer.from_pretrained(EMBED)
    m = load_f32(AutoModel, EMBED, attn_implementation="eager").eval()
    try:
        pool_cfg = json.load(open(hf_hub_download(EMBED, "1_Pooling/config.json")))
    except Exception as e:  # recorded, not guessed
        pool_cfg = {"error": str(e)}
    out["embed"] = {
        "model": EMBED,
        "hidden_act": m.config.hidden_act,
        "layer_norm_eps": m.config.layer_norm_eps,
        "sentence_transformers_pooling": pool_cfg,
        "items": [],
    }
    for t in TEXTS:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
        h = m(**enc).last_hidden_state[0]
        msk = enc["attention_mask"][0].unsqueeze(-1).to(h.dtype)
        out["embed"]["items"].append({
            "text": t,
            "ids": enc["input_ids"][0].tolist(),
            "cls_hidden": h[0].tolist(),
            "mean_hidden": ((h * msk).sum(0) / msk.sum()).tolist(),
        })

    tok = AutoTokenizer.from_pretrained(RERANK)
    r = load_f32(AutoModelForSequenceClassification, RERANK, attn_implementation="eager").eval()
    out["rerank"] = {
        "model": RERANK,
        "hidden_act": r.config.hidden_act,
        "layer_norm_eps": r.config.layer_norm_eps,
        "classifier": type(r.classifier).__name__,
        "items": [],
    }
    for q, d in PAIRS:
        enc = tok(q, d, return_tensors="pt")
        base = getattr(r, r.base_model_prefix)
        h = base(**enc).last_hidden_state[0]
        out["rerank"]["items"].append({
            "query": q,
            "doc": d,
            "ids": enc["input_ids"][0].tolist(),
            "logit": r(**enc).logits[0].tolist(),
            "cls_hidden": h[0].tolist(),
        })

json.dump(out, sys.stdout)
