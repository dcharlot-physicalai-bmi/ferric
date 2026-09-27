"""PEFT's own greedy continuations (float32) for the base and each fixture adapter on fixed raw prompts — the
reference scripts/serve_lora_conformance.sh holds ferric-serve to, over HTTP.

    HF_HUB_OFFLINE=1 python make_ref.py peft_greedy.json     # run from the repo root; torch, transformers, peft

Generated with peft 0.21.0, transformers 5.17.0. ⛔ repetition_penalty=1.0 is passed EXPLICITLY: Qwen2.5-Instruct's
generation_config.json sets 1.1, and generate() applies it even with do_sample=False — the first reference built
without it disagreed with Ferric's plain greedy on the BASE model, adapters or not."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
FX = "crates/ferric-llama/tests/fixtures/lora"
prompts = ["The three laws of motion state that", "def fibonacci(n):\n    ", "Paris is the capital of"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
out = {}
for name in ["base", "syn_a", "syn_b"]:
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
    if name != "base": m = PeftModel.from_pretrained(m, f"{FX}/{name}")
    m.eval()
    out[name] = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            g = m.generate(ids, max_new_tokens=24, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        out[name].append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False))
json.dump({"prompts": prompts, "peft": out}, open(sys.argv[1], "w"), indent=1)
print({k: [v[:40] for v in vs] for k, vs in out.items()})
