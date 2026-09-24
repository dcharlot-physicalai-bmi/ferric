#!/usr/bin/env python3
"""ModernBERT reference from the MODEL AUTHORS' OWN implementation (HF `transformers`), not llama.cpp.

⛔⛔ WHY THIS EXISTS. `modern-bert` was first marked Verified against `llama-embedding`. llama.cpp is a
port, not the reference, and on the checkpoint used — Alibaba-NLP/gte-reranker-modernbert-base — it is
WRONG in a way no llama.cpp-vs-Ferric comparison can see: the authors' config sets
`classifier_pooling: "mean"`, llama.cpp's converter never reads that key, the GGUF carries no
pooling_type, and llama.cpp's rank path pools the FIRST token. Ferric matched llama.cpp, so the two
agreed with each other while both computed a function the model was not trained to compute.

A reference is whatever the model's authors run. Everything else — llama.cpp included — is a peer.

⛔ THIS LIVES IN THE REPO ON PURPOSE (see qwen3vl_vision_ref.py): a generator kept in scratch is a
verification that cannot be repeated.

Writes JSON to stdout with, per (query, document) pair:
    ids               the authors' tokenizer on the PAIR (the rank path's real input)
    logit_configured  the score with the checkpoint's OWN classifier_pooling  ← the reference
    logit_cls         the same head fed the first token instead               ← what llama.cpp computes
    logit_mean        the same head fed the masked mean
    cls_hidden        last_hidden_state[0] — encoder output, pooling-independent
    mean_hidden       masked mean of last_hidden_state

Run:  <python-with-transformers> modern_bert_ref.py [model_id] > ref.json
"""
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


MODEL = sys.argv[1] if len(sys.argv) > 1 else "Alibaba-NLP/gte-reranker-modernbert-base"
PAIRS = [
    ("what is panda?", "The giant panda is a bear species endemic to China."),
    ("what is panda?", "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
    ("how do I bake bread?", "Mix flour, water, salt and yeast, then proof and bake."),
    ("how do I bake bread?", "The mitochondrion is the powerhouse of the cell."),
]

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
# float32 and eager attention: the reference should be the model's math, not a fused kernel's
# accumulation order. (transformers 5.x removed ModernBERT's `reference_compile` kwarg.)
model = load_f32(AutoModelForSequenceClassification, MODEL, attn_implementation="eager").eval()
cfg = model.config


def head(pooled):
    # ModernBertForSequenceClassification: logits = classifier(drop(head(pooled))),
    # head = norm(act(dense(x))). Dropout is 0.0 and the model is in eval mode.
    return model.classifier(model.drop(model.head(pooled)))


out = {
    "model": MODEL,
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "classifier_pooling": cfg.classifier_pooling,
    "classifier_activation": getattr(cfg, "classifier_activation", None),
    "pairs": [],
}
with torch.no_grad():
    for q, d in PAIRS:
        enc = tok(q, d, return_tensors="pt")
        o = model.model(**enc)
        h = o.last_hidden_state[0]  # [T, d]
        m = enc["attention_mask"][0].unsqueeze(-1).to(h.dtype)
        cls = h[0]
        mean = (h * m).sum(0) / m.sum()
        configured = model(**enc).logits[0]
        out["pairs"].append({
            "query": q,
            "doc": d,
            "ids": enc["input_ids"][0].tolist(),
            "logit_configured": configured.tolist(),
            "logit_cls": head(cls.unsqueeze(0))[0].tolist(),
            "logit_mean": head(mean.unsqueeze(0))[0].tolist(),
            "cls_hidden": cls.tolist(),
            "mean_hidden": mean.tolist(),
        })

# ⛔ A LONG SINGLE SEQUENCE, because every pair above is under 24 tokens and ModernBERT's sliding
# window is 128 wide (129 visible positions): below ~130 tokens the local layers see everything and a
# port with the window DISABLED is indistinguishable from a correct one. This is the input the
# negative controls in scripts/modern_bert_conformance.sh are measured on.
W = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "while", "seven", "ships",
     "sail", "past", "harbour", "lights", "before", "dawn", "breaks", "across", "water"]
LONG = " ".join(W[i % 20] for i in range(220))
with torch.no_grad():
    enc = tok(LONG, return_tensors="pt")
    h = model.model(**enc).last_hidden_state[0]
    m = enc["attention_mask"][0].unsqueeze(-1).to(h.dtype)
    out["long"] = {
        "text": LONG,
        "ids": enc["input_ids"][0].tolist(),
        "cls_hidden": h[0].tolist(),
        "mean_hidden": ((h * m).sum(0) / m.sum()).tolist(),
    }

json.dump(out, sys.stdout)
