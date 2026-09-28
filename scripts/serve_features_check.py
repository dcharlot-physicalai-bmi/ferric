#!/usr/bin/env python3
"""Live checks of ferric-serve request features against a running server, each against its definition:

  n > 1        choice 0 == the n = 1 answer; choice i == the request with seed + i; streamed == non-streamed
  constraints  json_object parses; guided_regex satisfies Python re.fullmatch; guided_choice is a choice;
               GBNF (llama.cpp's json.gbnf) output parses when it finishes; two constraints -> 400
  refusals     n out of range, n with tools, streamed completion with n > 1 -> 400

usage: serve_features_check.py <port> <json.gbnf path>
"""
import json, re, sys, urllib.request

U = f"http://127.0.0.1:{sys.argv[1]}"
GBNF = open(sys.argv[2]).read()
fail = []


def post(path, body, raw=False):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            d = f.read()
        return 200, (d.decode() if raw else json.loads(d))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(c, what):
    print(("  ok   " if c else "  FAIL ") + what)
    if not c: fail.append(what)


msgs = [{"role": "user", "content": "Write one short sentence about the sea."}]
base = {"messages": msgs, "temperature": 1.0, "max_tokens": 24, "seed": 7}
_, one = post("/v1/chat/completions", base)
_, three = post("/v1/chat/completions", dict(base, n=3))
texts = [c["message"]["content"] for c in three["choices"]]
check(texts[0] == one["choices"][0]["message"]["content"], "n=3: choice 0 is the n = 1 answer")
_, s9 = post("/v1/chat/completions", dict(base, seed=9))
check(s9["choices"][0]["message"]["content"] == texts[2], "n=3: choice 2 is the seed + 2 request")
check(len(set(texts)) == 3 and [c["index"] for c in three["choices"]] == [0, 1, 2], "n=3: three distinct choices, indexed")
check(three["usage"]["prompt_tokens"] == one["usage"]["prompt_tokens"], "n=3: prompt counted once")
_, raw = post("/v1/chat/completions", dict(base, n=3, stream=True, stream_options={"include_usage": True}), raw=True)
acc, usage = {}, None
for line in raw.splitlines():
    if not line.startswith("data: {"): continue
    d = json.loads(line[6:])
    usage = d.get("usage") or usage
    for c in d["choices"]:
        acc[c["index"]] = acc.get(c["index"], "") + (c["delta"].get("content") or "")
check([acc.get(i) for i in range(3)] == texts and usage == three["usage"], "n=3 streamed == non-streamed, same usage")
_, cp = post("/v1/completions", {"prompt": "The sea is", "n": 2, "temperature": 1.0, "max_tokens": 12})
check(len(cp["choices"]) == 2 and cp["choices"][0]["text"] != cp["choices"][1]["text"], "completions n=2: two distinct choices")

def chat(extra, q, n):
    return post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0, **extra})

c, v = chat({"response_format": {"type": "json_object"}}, "Give a JSON object describing a cat with name and age.", 64)
out = v["choices"][0]["message"]["content"]
try: json.loads(out); ok = True
except Exception: ok = False
check(ok and v["choices"][0]["finish_reason"] == "stop", f"json_object parses: {out[:50]!r}")
c, v = chat({"guided_regex": r"\d{3}-\d{4}"}, "Make up a phone number.", 16)
out = v["choices"][0]["message"]["content"]
check(re.fullmatch(r"\d{3}-\d{4}", out, re.ASCII) is not None, f"guided_regex satisfies re.fullmatch: {out!r}")
c, v = chat({"guided_choice": ["positive", "negative", "neutral"]}, "Sentiment of: 'I love this product!'", 8)
check(v["choices"][0]["message"]["content"] in ["positive", "negative", "neutral"], "guided_choice answers a choice")
for i in range(2):  # the second request reuses the grammar's cached masks
    c, v = chat({"grammar": GBNF}, "Describe Ada Lovelace as a JSON object with name and born.", 200)
    out, fr = v["choices"][0]["message"]["content"], v["choices"][0]["finish_reason"]
    try: json.loads(out); ok = True
    except Exception: ok = False
    check(ok if fr == "stop" else out.startswith("{"), f"json.gbnf #{i + 1} ({fr}): {out[:50]!r}")
check(chat({"grammar": "root ::= \"a\"", "guided_regex": "b"}, "x", 4)[0] == 400, "two constraints -> 400")
check(post("/v1/chat/completions", dict(base, n=17))[0] == 400, "n = 17 -> 400")
check(post("/v1/chat/completions", dict(base, n=2, tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]))[0] == 400, "n > 1 with tools -> 400")
check(post("/v1/completions", {"prompt": "x", "n": 2, "stream": True})[0] == 400, "streamed completion n > 1 -> 400")
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
