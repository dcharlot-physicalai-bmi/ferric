#!/usr/bin/env python3
"""The model's chat template: Ferric vs Hugging Face's apply_chat_template environment, byte for byte.

    <python with transformers + gguf> chat_template_conformance.py <render_template binary> <cases.json> <model.gguf>...

For each GGUF: its `tokenizer.chat_template` is compiled by transformers' own `_compile_jinja_template`
(jinja2 ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols, raise_exception,
strftime_now, the json.dumps tojson) and rendered over every case with the same context Ferric passes
(messages, tools, add_generation_prompt, bos_token, eos_token, kwargs). A case passes when both render the
same string, or both refuse. Exit 1 on any mismatch.
"""
import json, subprocess, sys
from gguf import GGUFReader
from transformers.utils.chat_template_utils import _compile_jinja_template

BIN, CASES, MODELS = sys.argv[1], sys.argv[2], sys.argv[3:]
cases = json.load(open(CASES))

def field(r, k):
    f = r.fields.get(k)
    if f is None: return None
    return f.contents()

bad = 0
for path in MODELS:
    r = GGUFReader(path)
    src = field(r, "tokenizer.chat_template")
    name = path.rsplit("/", 1)[-1]
    if not src:
        print(f"-- {name}: no chat template"); continue
    toks = field(r, "tokenizer.ggml.tokens") or []
    tok = lambda k: toks[field(r, k)] if field(r, k) is not None and field(r, k) < len(toks) else ""
    bos, eos = tok("tokenizer.ggml.bos_token_id"), tok("tokenizer.ggml.eos_token_id")
    out = subprocess.run([BIN, path, CASES], capture_output=True, text=True)
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    if lines and "compile_err" in lines[0]:
        print(f"⛔ {name}: Ferric cannot compile the template: {lines[0]['compile_err'][:200]}"); bad += 1; continue
    fer = {l["i"]: l for l in lines}
    tmpl = _compile_jinja_template(src)
    ok = 0
    for i, c in enumerate(cases):
        ctx = dict(messages=c["messages"], add_generation_prompt=c.get("add_generation_prompt", True),
                   bos_token=bos, eos_token=eos, **c.get("kwargs", {}))
        if c.get("tools") is not None: ctx["tools"] = c["tools"]
        try: want = ("out", tmpl.render(**ctx))
        except Exception as e: want = ("err", str(e))
        got = fer.get(i, {})
        got = ("out", got["out"]) if "out" in got else ("err", got.get("err", "no output"))
        if want[0] == got[0] and (want[0] == "err" or want[1] == got[1]):
            ok += 1
        else:
            bad += 1
            a, b = want[1], got[1]
            p = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            print(f"  ⛔ {name} case {i} ({c['name']}): HF {want[0]} vs Ferric {got[0]} — first difference at {p}:\n"
                  f"       HF     {a[max(0, p - 40):p + 40]!r}\n       Ferric {b[max(0, p - 40):p + 40]!r}")
    print(f"{'✅' if ok == len(cases) else '⛔'} {name}: {ok}/{len(cases)} cases identical to HF")
print("✅ every template renders as Hugging Face renders it" if bad == 0 else f"⛔ {bad} mismatches")
sys.exit(1 if bad else 0)
