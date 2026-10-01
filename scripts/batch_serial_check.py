#!/usr/bin/env python3
"""The batched path answers exactly as the serial path, on a real model and its real template.

Runs ferric-serve twice on the same model — continuous batching ON, then --no-batch — and sends the same
requests to both; the batched server gets them all AT ONCE (so they share decode steps). Every field a
client reads must match: content, reasoning_content, finish_reason, usage, logprobs, and for streamed
requests the concatenated content and reasoning deltas. The request set covers what the batched path must
render and split like the serial one: template kwargs (enable_thinking), reasoning_effort, a thinking
model's reasoning split, the `developer` role, stop strings, logprobs, streaming, and plain completions.

Logprob VALUES are compared to 1e-4 (the max deviation is printed; "bitwise" when zero); everything else —
text, reasoning, finish, usage, the logprob tokens and bytes — exactly. Why: a decode step of 3+ rows takes
different matmul kernels than a solo step (a speed heuristic, ferric-tensor dtype.rs `q2_0_split_k`, and at
least one more on qwen3), whose reduction order moves logits ~1e-5. Both servers run with
FERRIC_Q2_0_KERNEL=splitk, which makes Qwen2.5 bitwise; the rest is the kernels' batch variance, not this
path's — tracked as E27 in docs/parity.

usage: batch_serial_check.py <ferric-serve binary> <model.gguf> [model2.gguf ...]
"""
import os
import json, subprocess, sys, threading, time, urllib.request

BIN, MODELS, PORT = sys.argv[1], sys.argv[2:], 18491
Q = [{"role": "user", "content": "What is 17 * 3? Answer briefly."}]
REQS = [
    ("chat", {"messages": Q, "max_tokens": 160}),
    ("chat enable_thinking=false", {"messages": Q, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False}}),
    ("chat reasoning_effort=none", {"messages": Q, "max_tokens": 60, "reasoning_effort": "none"}),
    ("chat developer role", {"messages": [{"role": "developer", "content": "Answer in French."}] + Q, "max_tokens": 60,
                             "chat_template_kwargs": {"enable_thinking": False}}),
    ("chat stop", {"messages": Q, "max_tokens": 80, "stop": ["."], "chat_template_kwargs": {"enable_thinking": False}}),
    ("chat logprobs", {"messages": Q, "max_tokens": 24, "logprobs": True, "top_logprobs": 2}),
    ("chat stream", {"messages": Q, "max_tokens": 160, "stream": True, "stream_options": {"include_usage": True}}),
    ("completion", {"prompt": "The capital of France is", "max_tokens": 16}),
]

def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f:
        return f.read().decode()

def ask(name, body):
    path = "/v1/completions" if name == "completion" else "/v1/chat/completions"
    raw = post(path, body)
    if body.get("stream"):
        content, reasoning, finish, usage = "", "", None, None
        for line in raw.splitlines():
            if not line.startswith("data: {"): continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                content += c["delta"].get("content") or ""
                reasoning += c["delta"].get("reasoning_content") or ""
                finish = c.get("finish_reason") or finish
        return {"content": content, "reasoning": reasoning, "finish": finish, "usage": usage}
    v = json.loads(raw)
    c = v["choices"][0]
    if name == "completion":
        return {"text": c["text"], "finish": c["finish_reason"], "usage": v["usage"]}
    return {"content": c["message"]["content"], "reasoning": c["message"].get("reasoning_content", ""),
            "finish": c["finish_reason"], "usage": v["usage"], "logprobs": c.get("logprobs")}

def serve(model, nobatch):
    args = [BIN, model, "--port", str(PORT)] + (["--no-batch"] if nobatch else [])
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                         env=dict(os.environ, FERRIC_Q2_0_KERNEL=os.environ.get("FERRIC_Q2_0_KERNEL", "splitk")))
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); return p
        except Exception: time.sleep(0.5)
    raise SystemExit("server did not start")

fail = []
for model in MODELS:
    print(f"== {model.split('/')[-1]}")
    p = serve(model, False)
    batched = {}
    th = [threading.Thread(target=lambda n=n, b=b: batched.__setitem__(n, ask(n, b))) for n, b in REQS]
    [t.start() for t in th]; [t.join() for t in th]
    p.terminate(); err = p.communicate(timeout=30)[1]
    if "continuous batching ON" not in err:
        print("  FAIL the first server did not batch"); fail.append(model)
    p = serve(model, True)
    serial = {n: ask(n, b) for n, b in REQS}
    p.terminate(); p.wait(timeout=30)
    def split_lp(r):
        """(the result with logprob floats removed, the floats)"""
        r = json.loads(json.dumps(r))
        vals = []
        for e in ((r.get("logprobs") or {}).get("content") or []):
            vals.append(e.pop("logprob"))
            for t in e.get("top_logprobs", []): vals.append(t.pop("logprob"))
        return r, vals
    for n, _ in REQS:
        (b0, bv), (s0, sv) = split_lp(batched[n]), split_lp(serial[n])
        dev = max((abs(x - y) for x, y in zip(bv, sv)), default=0.0)
        same = b0 == s0 and len(bv) == len(sv) and dev <= 1e-4
        if sv: n_ = n; print(f"         logprob values: max |Δ| {dev:.2e}" + (" (bitwise)" if dev == 0 else ""))
        extra = f" (reasoning {len(serial[n].get('reasoning') or '')} chars)" if serial[n].get("reasoning") else ""
        print(("  ok   " if same else "  FAIL ") + n + extra)
        if not same:
            fail.append(f"{model}: {n}")
            for k in serial[n]:
                b_, s_ = batched[n].get(k), serial[n].get(k)
                if b_ == s_: continue
                if k == "logprobs" and b_ and s_:
                    bl, sl = b_["content"], s_["content"]
                    i = next((i for i in range(min(len(bl), len(sl))) if bl[i] != sl[i]), min(len(bl), len(sl)))
                    print(f"         logprobs: {len(bl)} batched vs {len(sl)} serial entries; first difference at {i}:")
                    print(f"           batched {json.dumps(bl[i]) if i < len(bl) else None}"[:300])
                    print(f"           serial  {json.dumps(sl[i]) if i < len(sl) else None}"[:300])
                else:
                    print(f"         {k}: batched {str(b_)[:90]!r}\n         {k}: serial  {str(s_)[:90]!r}")
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
