#!/usr/bin/env python3
"""The reasoning budget, against llama-server's own (the feature is llama.cpp's: common/reasoning-budget.cpp).

The same thinking model, the same greedy requests, `thinking_budget_tokens` 0 / 8 / 24 with the same
`--reasoning-budget-message` on both servers: the reasoning must stop where the budget says (the message and
the end marker forced), the answer must follow, and the two servers' reasoning and answers are compared
text for text (they share the state machine; their logits differ in the last bits, so a greedy
divergence — reported with its position — is a numerics difference, not a budget one). A request without
a budget must think past the largest budget, so the check can fail.

usage: reasoning_budget_check.py <ferric-serve> <llama-server> <thinking model.gguf>
"""
import json, subprocess, sys, time, urllib.request

FERRIC, LLAMA, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
PF, PL, MSG = 18511, 18512, "Okay, I have to answer now."
Q = [{"role": "user", "content": "How many prime numbers are there between 10 and 30? List them."}]
fail = []
def check(c, what):
    print(("  ok   " if c else "  FAIL ") + what)
    if not c: fail.append(what)

def post(port, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    v = json.loads(urllib.request.urlopen(r, timeout=900).read())
    m = v["choices"][0]["message"]
    return (m.get("reasoning_content") or "").strip(), (m.get("content") or "").strip(), v["usage"]["completion_tokens"]

def up(port):
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); return
        except Exception: time.sleep(0.5)
    raise SystemExit(f"no server on {port}")

fs = subprocess.Popen([FERRIC, MODEL, "--port", str(PF), "--reasoning-budget-message", MSG], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ls = subprocess.Popen([LLAMA, "-m", MODEL, "--port", str(PL), "--jinja", "--reasoning-format", "deepseek", "-ngl", "99", "-c", "4096",
                       "--reasoning-budget-message", MSG, "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    up(PF); up(PL)
    base = {"messages": Q, "max_tokens": 400, "temperature": 0}
    fr, fc, _ = post(PF, base)
    check(len(fr.split()) > 40, f"without a budget the model thinks at length ({len(fr.split())} words) — the budgets below must cut it")
    for budget in [0, 8, 24]:
        body = dict(base, thinking_budget_tokens=budget)
        (fr, fc, fn), (lr, lc, ln) = post(PF, body), post(PL, body)
        print(f"  budget {budget}: ferric reasoning {fr[-60:]!r}\n             answer {fc[:60]!r}")
        check(fr.endswith(MSG) or (budget == 0 and fr in ("", MSG)), f"budget {budget}: the reasoning ends with the forced message")
        check(len(fc) > 0, f"budget {budget}: an answer follows the forced end")
        check(fr == lr, f"budget {budget}: reasoning == llama-server's" + ("" if fr == lr else f"\n         llama {lr[-80:]!r}"))
        same = fc == lc
        if not same:
            k = next((i for i in range(min(len(fc), len(lc))) if fc[i] != lc[i]), min(len(fc), len(lc)))
            print(f"  info budget {budget}: answers diverge at char {k} (greedy numerics): ferric {fc[k:k+30]!r} vs llama {lc[k:k+30]!r}")
        else:
            print(f"  ok   budget {budget}: answer == llama-server's")
finally:
    fs.terminate(); ls.terminate(); fs.wait(timeout=30); ls.wait(timeout=30)
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
