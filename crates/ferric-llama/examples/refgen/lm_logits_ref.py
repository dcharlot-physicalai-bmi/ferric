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

Run in float32, eager:  <python> lm_logits_ref.py <model_id> [min_tokens] [--remote-code] > fixture.json

`--remote-code` runs the modeling file the AUTHORS SHIP IN THEIR REPO (trust_remote_code) instead of
transformers' built-in port of it — NVIDIA's Nemotron-H repo carries its own `modeling_nemotron_h.py`,
and that file is the authors' implementation in the most literal sense. The fixture records which ran.

⛔⛔ THE BUILT-IN PORT IS NOT THE AUTHORS' CODE, and on Nemotron-H they disagree by 0.27 in the logits.
transformers' NemotronH torch path floors dt at `time_step_min` (0.001, an INITIALISATION range), where
the authors' file, and transformers' own kernel path, clamp to `time_step_limit` = (0, inf): no floor.
Ferric matched the built-in to 0.27 and the built-in-without-the-floor to 3.6e-5.

`--restore-from-file` re-copies every parameter from the checkpoint file after loading, for a loader known
to overwrite them (the Nemotron-H remote code under transformers 5.x: see refload.py). The count is
recorded; the by-value check then runs again and must find nothing.

`--fix-group-tiling` corrects ONE line of the authors' Nemotron-H CPU fallback, and refuses to run if it
finds nothing to correct. That fallback maps heads onto B/C groups with `B.repeat(1, 1, heads_per_group, 1)`,
which TILES them (head h -> group h mod 8). The authors' own CUDA kernels, which is how the model is
trained and served, index `pid_h // nheads_ngroups_ratio`, CONTIGUOUS groups (head h -> group h // 12;
state-spaces/mamba `ssd_chunk_state.py`, and its reference `"b l g d -> b l (g h) d"`). Corrected, the
authors' file agrees with transformers' built-in port (its dt floor removed) to 5.0e-5 over the full
vocabulary: two independent paths, one documented correction each, one answer.

`--mamba-ref-shim` lets the authors' Nemotron-H file run without CUDA: it imports mamba_ssm's Triton
`rmsnorm_fn` even on its torch path, and `shims/mamba_ssm` supplies the kernel authors' own pure-torch
`rms_norm_ref` in its place (see that file). Recorded in the fixture's `shims`.

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

# The loader, its float32 handling and its by-value check against the checkpoint file: refload.py.
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from refload import LOAD_REPORT, _f32_config, load_f32  # noqa: E402


POS_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
REMOTE = "--remote-code" in sys.argv[1:]
SHIMS = []
if "--mamba-ref-shim" in sys.argv[1:]:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shims"))
    SHIMS.append("mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn -> rms_norm_ref (the kernel authors' "
                 "pure-torch reference, state-spaces/mamba)")
    if not torch.cuda.is_available():
        # The authors' block forward runs inside `torch.cuda.stream(...)`, which a CPU build refuses. A
        # stream ORDERS GPU work and changes no arithmetic; with no GPU there is nothing to order.
        import contextlib
        torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()
        torch.cuda.default_stream = lambda *a, **k: None
        SHIMS.append("torch.cuda.stream -> nullcontext (CPU build: a stream orders GPU work, no arithmetic)")
MODEL = POS_ARGS[0]
TEXT = ("The heron stood motionless in the shallows while the tide turned. A fisherman on the far bank "
        "counted his nets twice, then a third time, because the numbers never agreed. Measurements, he "
        "thought, are only as honest as the instrument and the person reading it. In 1854 John Snow "
        "mapped cholera deaths around a single water pump on Broad Street; the pattern was plain once the "
        "points were on paper. def mean(xs): return sum(xs) / len(xs)  # an empty list divides by zero. "
        "Über den Wolken muss die Freiheit wohl grenzenlos sein. 月が綺麗ですね。 The answer is 42.")

MIN_T = int(POS_ARGS[1]) if len(POS_ARGS) > 1 else 0

tok = AutoTokenizer.from_pretrained(MODEL)
model = load_f32(AutoModelForCausalLM, MODEL, attn_implementation="eager",
                 restore_from_file="--restore-from-file" in sys.argv[1:],
                 **({"trust_remote_code": True} if REMOTE else {})).eval()
CORRECTIONS = []
if "--fix-group-tiling" in sys.argv[1:]:
    import inspect, textwrap
    fixed = set()
    for mod in model.modules():
        cls = type(mod)
        if cls in fixed or not hasattr(cls, "torch_forward"): continue
        code = textwrap.dedent(inspect.getsource(cls.torch_forward))
        n = 0
        for v in ("B", "C"):
            tiled = f"{v}.repeat(1, 1, self.num_heads // self.n_groups, 1)"
            n += code.count(tiled)
            code = code.replace(tiled, f"{v}.repeat_interleave(self.num_heads // self.n_groups, dim=2)")
        if n:
            ns = {}; exec(compile(code, f"<{cls.__name__}.torch_forward, contiguous groups>", "exec"),
                          sys.modules[cls.__module__].__dict__, ns)
            cls.torch_forward = ns["torch_forward"]; fixed.add(cls)
            CORRECTIONS.append(f"{cls.__name__}.torch_forward: {n} x B/C `repeat` (tiled heads) -> "
                               f"`repeat_interleave` (contiguous), as the authors' CUDA kernels index")
    if not CORRECTIONS:
        raise SystemExit("--fix-group-tiling found no tiled group mapping to correct — refusing, the flag "
                         "would otherwise be recorded while changing nothing")
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
    "code": (f"the authors' repo: {type(model).__module__}" if REMOTE
             else f"transformers built-in: {type(model).__module__}"),
    "shims": SHIMS,
    "corrections": CORRECTIONS,
    "load": LOAD_REPORT,   # parameters compared BY VALUE to the checkpoint file, as loaded
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
