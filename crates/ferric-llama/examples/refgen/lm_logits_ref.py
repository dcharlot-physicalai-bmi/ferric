#!/usr/bin/env python3
"""Causal-LM logits from the MODEL AUTHORS' OWN implementation (HF `transformers`), for any checkpoint.

⛔⛔ WHY THIS EXISTS. Ferric's dense LMs (llama, qwen2, qwen3, qwen3.5) were marked Verified against
llama.cpp (`llama-cli`, `scripts/validate_vs_llamacpp.sh`) or against an unnamed reference. That check
also compared GREEDY ARGMAX only — and a small logit error almost never flips an argmax, so "matching
token for token" says little about the logits. The reference is what the model's authors run.

The same move on BERT and ModernBERT (2026-09-24) found three defects the llama.cpp comparisons had
hidden or explained away — every one of them a choice made to match llama.cpp instead of the model.

⛔ THIS LIVES IN THE REPO ON PURPOSE: a generator kept in scratch is a verification nobody can repeat.

Per position the fixture records what a full-vocabulary comparison needs, compactly:
    top      the authors' top-10 (id, logit)
    sample   logits at a FIXED seeded sample of 128 vocabulary ids (same ids at every position)
    sum/ssq  sum and sum of squares of the FULL vocabulary row — a defect ANYWHERE moves these
The input is ~150 tokens of the authors' own tokenizer output, so RoPE is exercised well past the
first handful of positions where a wrong rotation is nearly invisible.

Run in float32, eager:  <python> lm_logits_ref.py <model_id> [min_tokens] > fixture.json

⛔ A SLIDING WINDOW IS INVISIBLE BELOW ITS WIDTH. Gemma 3's local layers see the last 512 tokens, and at
~140 tokens they see everything — a port with no window at all matches exactly. Pass `min_tokens` above
the window: the text repeats until it is that long, so the global layers can copy from a repetition
the local ones can no longer reach. A long input records a stride of positions plus the whole tail
(listed in `positions`), not every row, to keep the fixture small.
"""
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


MODEL = sys.argv[1]
TEXT = ("The heron stood motionless in the shallows while the tide turned. A fisherman on the far bank "
        "counted his nets twice, then a third time, because the numbers never agreed. Measurements, he "
        "thought, are only as honest as the instrument and the person reading it. In 1854 John Snow "
        "mapped cholera deaths around a single water pump on Broad Street; the pattern was plain once the "
        "points were on paper. def mean(xs): return sum(xs) / len(xs)  # an empty list divides by zero. "
        "Über den Wolken muss die Freiheit wohl grenzenlos sein. 月が綺麗ですね。 The answer is 42.")

MIN_T = int(sys.argv[2]) if len(sys.argv) > 2 else 0

tok = AutoTokenizer.from_pretrained(MODEL)
model = load_f32(AutoModelForCausalLM, MODEL, attn_implementation="eager").eval()
text = TEXT
while MIN_T and len(tok(text, add_special_tokens=True)["input_ids"]) < MIN_T:
    text += " " + TEXT
ids = tok(text, add_special_tokens=True)["input_ids"]
positions = (list(range(len(ids))) if not MIN_T
             else sorted(set(range(0, len(ids), 8)) | set(range(len(ids) - 64, len(ids)))))
V = model.config.vocab_size
rng = random.Random(20260924)
sample = sorted(rng.sample(range(min(V, model.get_output_embeddings().weight.shape[0])), 128))

with torch.no_grad():
    logits = model(input_ids=torch.tensor([ids])).logits[0].float()   # [T, V]

rows = []
for t in positions:
    r = logits[t]
    top = torch.topk(r, 10)
    rows.append({
        # 5 decimals: the measured Ferric-vs-authors gap is ~1e-4, so 4 decimals (a 5e-5 step) would
        # be the same size as the thing being measured.
        "top": [[int(i), round(float(v), 5)] for v, i in zip(top.values, top.indices)],
        "sample": [round(float(r[i]), 5) for i in sample],
        "sum": float(r.double().sum()),
        "ssq": float((r.double() ** 2).sum()),
    })

json.dump({
    "model": MODEL,
    "model_type": model.config.model_type,
    "architectures": model.config.architectures,
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "dtype": str(next(model.parameters()).dtype),   # asserted float32 AS LOADED by load_f32, recorded anyway
    "vocab": int(logits.shape[1]),
    "ids": ids,
    "positions": positions,
    "sample_ids": sample,
    "rows": rows,
}, sys.stdout)
