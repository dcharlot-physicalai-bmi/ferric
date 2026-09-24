"""The ONE guarded loader every reference generator here uses: the authors' checkpoint, at float32,
checked BY VALUE against the checkpoint file before a single logit is computed.

Every trap below produced a "reference" that read as a defect in Ferric. Each was found the hard way,
on this repo's own verification work (2026-09-24), and each guard was then shown to REFUSE the bad
load, not merely to exist.
"""
import torch

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

def _f32_config(model_id, **kw):
    cfg = AutoConfig.from_pretrained(model_id, **kw)
    def walk(c, seen):
        if id(c) in seen: return
        seen.add(id(c)); c.dtype = torch.float32
        for name in list(getattr(type(c), "sub_configs", {}) or {}) + ["text_config", "vision_config", "audio_config"]:
            sub = getattr(c, name, None)
            if sub is not None and hasattr(sub, "to_dict"): walk(sub, seen)
    walk(cfg, set())
    return cfg

def load_f32(cls, model_id, restore_from_file=False, **kw):
    rc = {k: kw[k] for k in ("trust_remote_code",) if k in kw}
    m = cls.from_pretrained(model_id, config=_f32_config(model_id, **rc), dtype=torch.float32, **kw)
    bad = sorted({str(p.dtype) for p in m.parameters() if p.dtype != torch.float32})   # AS LOADED
    if bad:
        raise SystemExit(f"{model_id} loaded as {bad}, not float32 — refusing to emit a fixture")
    entry = LOAD_REPORT.setdefault(model_id, {})
    if restore_from_file:
        entry["restored_from_file"] = _against_file(m, model_id, restore=True)["mismatched"]
    rep = _against_file(m, model_id)
    entry.update({k: v for k, v in rep.items() if k != "examples"})
    if rep["mismatched"]:
        raise SystemExit(f"{model_id}: {rep['mismatched']} parameters differ from the checkpoint FILE as loaded "
                         f"(e.g. {rep['examples']}) — refusing to emit a fixture")
    return m


# ⛔⛔ "0 MISSING KEYS" IS NOT "THE WEIGHTS ARE THE FILE'S". Under transformers 5.7.0 the Nemotron-H
# authors' remote code loads every tensor, reports 0 missing / 0 unexpected, and then RE-RUNS its own
# `_init_weights` over `dt_bias` and `out_proj.weight` in all 21 Mamba layers — 42 of 263 tensors
# replaced by a fresh initialisation. The logits it then produced missed Ferric by 10 with 4/138 argmax
# agreement, and read as a verdict on Ferric. So every parameter is compared, BY VALUE, against the
# checkpoint file; f32 upcasts of bf16/f16 are exact, so any difference at all is a load defect.
LOAD_REPORT = {}   # model id -> what the by-value check found, recorded in every fixture

def _against_file(m, model_id, restore=False):
    import glob, os
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    snap = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    files = [safe_open(p, "pt") for p in sorted(glob.glob(os.path.join(snap, "*.safetensors")))]
    keys = [(f, k) for f in files for k in f.keys()]
    params = dict(m.named_parameters())
    matched = mismatched = 0; examples = []; unmatched = 0; total = 0
    for name, p in params.items():
        total += p.numel()
        # An exact name wins. Otherwise the file key must END with the name minus its first component
        # (`model.` / `backbone.` / ...), which is how a composite checkpoint nests its text model
        # (`model.language_model.layers.0...`). ⛔ Suffix alone is NOT enough: Qwen3.5 also ships a
        # multi-token-prediction block, `mtp.layers.0.mlp...`, and the first version of this check
        # compared `model.layers.0.mlp` against IT and refused a correct load. So among several
        # candidates, keep those sharing the parameter's first component. Still ambiguous: unverified.
        exact = [(f, k) for f, k in keys if k == name]
        first, tail = (name.split(".", 1) + [""])[:2]
        cands = exact or [(f, k) for f, k in keys if k.endswith("." + tail) or k == tail]
        if len(cands) > 1:
            cands = [(f, k) for f, k in cands if k.startswith(first + ".")]
        if len(cands) != 1:
            unmatched += p.numel(); continue
        f, k = cands[0]
        t = f.get_tensor(k).float()
        if t.shape != p.shape:
            unmatched += p.numel(); continue
        matched += p.numel()
        if not torch.equal(p.detach().float(), t):
            mismatched += 1
            if len(examples) < 4: examples.append(name)
            if restore:
                with torch.no_grad(): p.copy_(t)
    return {"mismatched": mismatched, "examples": examples,
            "verified_fraction": round(matched / max(1, total), 6), "unverified_elements": unmatched}
