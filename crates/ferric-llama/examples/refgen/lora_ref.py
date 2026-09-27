#!/usr/bin/env python3
"""LoRA reference from THE LIBRARY THAT DEFINES THE FORMAT — Hugging Face PEFT — applied to the model
authors' own base (transformers), for `scripts/lora_conformance.sh` and `scripts/lora_roundtrip.sh`.

PEFT is the oracle, not llama.cpp: it owns `adapter_config.json` / `adapter_model.safetensors`, and its
`LoraLayer.forward` is the definition of the math — `result + lora_B(lora_A(dropout(x))) * scaling`, with
`scaling = lora_alpha / r`, or `lora_alpha / sqrt(r)` under `use_rslora` (peft/tuners/lora/layer.py).

Three commands:

  make <base_id> <out_dir> --seed S --r R --alpha A [--rslora] [--targets q_proj,v_proj] [--layers 0,23]
                           [--b-scale 1.0] [--f16]
      A synthetic adapter built BY PEFT (`get_peft_model` + `save_pretrained`), with a fixed seed and
      `init_lora_weights=False`, so lora_B is NONZERO. PEFT's default init zeroes lora_B, and an all-zero
      delta is an adapter every loader — including one that applies nothing — reproduces exactly.
      `--f16` stores the adapter in float16: the values PEFT then loads are exact float32 upcasts of what
      is on disk, so both sides of the comparison see the same numbers, at half the committed bytes.

  ref <base_id> <out.json.gz> <adapter_dir> [<adapter_dir_b>]
      Logits for a fixed text with the adapter applied by `PeftModel.from_pretrained`, in float32 AND in
      float64 (the whole PeftModel `.double()`d — the noise floor: a float32 reference cannot arbitrate
      below its own distance from exact arithmetic), and the base with the adapter DISABLED
      (`disable_adapter()`), both precisions. With a second adapter it also records PEFT's own MIXED
      BATCH — `forward(..., adapter_names=["a", "b", "__base__"])`, one adapter per ROW — which is the
      reference for per-request adapter selection in a batched decode.

  logits <base_id> <out.json.gz> <adapter_dir> --ids 1,2,3
      As `ref` for a caller-supplied token sequence and no mixed batch — the round-trip check's reference.

  ids <base_id>
      Two lines: the authors' tokenizer's ids for the fixed text, and the fixed 128-id vocabulary sample —
      what the Ferric side of the round trip evaluates on.

⚠ The JSON goes to a FILE: transformers and PEFT print warnings on stdout.
"""
import argparse
import gzip
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refload import LOAD_REPORT, load_f32  # noqa: E402

import peft  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TEXT = ("The heron stood motionless in the shallows while the tide turned. A fisherman on the far bank "
        "counted his nets twice, then a third time, because the numbers never agreed. Measurements, he "
        "thought, are only as honest as the instrument and the person reading it. In 1854 John Snow "
        "mapped cholera deaths around a single water pump on Broad Street; the pattern was plain once the "
        "points were on paper. def mean(xs): return sum(xs) / len(xs)  # an empty list divides by zero. "
        "Über den Wolken muss die Freiheit wohl grenzenlos sein. 月が綺麗ですね。 The answer is 42.")
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def base(model_id):
    torch.manual_seed(0)  # nothing below samples, but a dropout left on would; see the eval() calls
    return load_f32(AutoModelForCausalLM, model_id, attn_implementation="eager").eval()


def make(a):
    torch.manual_seed(a.seed)
    m = base(a.base)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, use_rslora=a.rslora, target_modules=a.targets.split(","),
                     layers_to_transform=[int(x) for x in a.layers.split(",")] if a.layers else None,
                     init_lora_weights=False, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(m, cfg)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if ".lora_B." in n:
                p.mul_(a.b_scale)
            if a.f16 and ".lora_" in n:
                p.data = p.data.half()
    pm.save_pretrained(a.out)
    n_a = sum(1 for n, _ in pm.named_parameters() if ".lora_A." in n)
    print(f"wrote {a.out}: {n_a} adapted linears, r={a.r}, alpha={a.alpha}, rslora={a.rslora}, "
          f"scaling={a.alpha / (a.r ** 0.5 if a.rslora else a.r):g}", file=sys.stderr)


def hub_id(path):
    """`org/name` for an adapter in the Hugging Face cache, else the directory name."""
    parts = os.path.normpath(path).split(os.sep)
    for x in parts:
        if x.startswith("models--"):
            return x[len("models--"):].replace("--", "/")
    return parts[-1]


def rows(logits, positions, sample):
    out = []
    for t in positions:
        r = logits[t]
        top = torch.topk(r.float(), 10)
        out.append({
            "top": [[int(i), round(float(v), 6)] for v, i in zip(top.values, top.indices)],
            "sample": [round(float(r[i]), 6) for i in sample],
            "sum": float(r.double().sum()),
            "ssq": float((r.double() ** 2).sum()),
        })
    return out


def peft_pair(base_id, adapters):
    """The adapter(s) on the authors' base: (float32 PeftModel, float64 PeftModel)."""
    out = []
    for dt in (torch.float32, torch.float64):
        pm = PeftModel.from_pretrained(base(base_id), adapters[0], adapter_name="a")
        if len(adapters) > 1:
            pm.load_adapter(adapters[1], adapter_name="b")
        pm = pm.to(dt).eval()
        bad = sorted({str(p.dtype) for p in pm.parameters() if p.dtype != dt})
        if bad:
            raise SystemExit(f"PeftModel has {bad} parameters, not {dt} — refusing to emit a reference")
        pm.set_adapter("a")
        out.append(pm)
    return out


def logits_of(pm, ids, adapter_names=None, disable=False):
    x = torch.tensor(ids if isinstance(ids[0], list) else [ids])
    kw = {"adapter_names": adapter_names} if adapter_names else {}
    with torch.no_grad():
        if disable:
            with pm.disable_adapter():
                return pm(input_ids=x).logits
        return pm(input_ids=x, **kw).logits


def ref(a, ids=None, batch=True):
    tok = AutoTokenizer.from_pretrained(a.base)
    ids = ids or tok(TEXT, add_special_tokens=True)["input_ids"]
    T = len(ids)
    positions = sorted(set(range(0, T, 6)) | set(range(max(0, T - 8), T)))
    adapters = [a.adapter] + ([a.adapter_b] if getattr(a, "adapter_b", None) else [])
    p32, p64 = peft_pair(a.base, adapters)
    V = int(p32.base_model.model.config.vocab_size)
    rng = random.Random(20260924)
    sample = sorted(rng.sample(range(V), 128))
    doc = {
        "model": a.base,
        "adapters": [{"path": hub_id(p),
                      "config": json.load(open(os.path.join(p, "adapter_config.json")))} for p in adapters],
        "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
        "load": LOAD_REPORT,
        "ids": ids, "positions": positions, "sample_ids": sample, "vocab": V,
        "variants": {},
    }
    for tag, pm in (("f32", p32), ("f64", p64)):
        doc["variants"][f"adapted_{tag}"] = rows(logits_of(pm, ids)[0], positions, sample)
        doc["variants"][f"base_{tag}"] = rows(logits_of(pm, ids, disable=True)[0], positions, sample)
    if batch and len(adapters) > 1:
        # PEFT's own per-row adapter selection. Three rows of equal length (so no padding enters the
        # math), different tokens, adapters a / b / none. A runtime that crosses rows — applies row 0's
        # adapter to row 1 — cannot match all three.
        L, P = 40, 32
        seqs = [ids[3 * i: 3 * i + L] for i in range(3)]
        names = ["a", "b", "__base__"]
        doc["batch"] = {"adapter_names": names, "seqs": seqs, "prompt_len": P,
                        "positions": list(range(P - 1, L))}
        for tag, pm in (("f32", p32), ("f64", p64)):
            lg = logits_of(pm, seqs, adapter_names=names)
            doc["batch"][tag] = [rows(lg[i], doc["batch"]["positions"], sample) for i in range(3)]
    json.dump(doc, gzip.open(a.out, "wt"))
    print(f"wrote {a.out}: {T} tokens, {len(positions)} positions, adapters {[d['path'] for d in doc['adapters']]}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("base"); m.add_argument("out")
    m.add_argument("--seed", type=int, required=True); m.add_argument("--r", type=int, required=True)
    m.add_argument("--alpha", type=float, required=True); m.add_argument("--rslora", action="store_true")
    m.add_argument("--targets", default=",".join(ALL)); m.add_argument("--layers", default="")
    m.add_argument("--b-scale", type=float, default=1.0); m.add_argument("--f16", action="store_true")
    r = sub.add_parser("ref")
    r.add_argument("base"); r.add_argument("out"); r.add_argument("adapter"); r.add_argument("adapter_b", nargs="?")
    ti = sub.add_parser("ids")
    ti.add_argument("base")
    lo = sub.add_parser("logits")
    lo.add_argument("base"); lo.add_argument("out"); lo.add_argument("adapter"); lo.add_argument("--ids", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a)
    elif a.cmd == "ref":
        ref(a)
    elif a.cmd == "ids":
        from transformers import AutoConfig
        ids = AutoTokenizer.from_pretrained(a.base)(TEXT, add_special_tokens=True)["input_ids"]
        V = AutoConfig.from_pretrained(a.base).vocab_size
        print(",".join(map(str, ids)))
        print(",".join(map(str, sorted(random.Random(20260924).sample(range(V), 128)))))
    else:
        ref(a, ids=[int(x) for x in a.ids.split(",")], batch=False)


if __name__ == "__main__":
    main()
