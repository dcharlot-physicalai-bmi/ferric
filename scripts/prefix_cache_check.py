#!/usr/bin/env python3
"""Prompt caching must not change an answer. Run a 4-turn chat behind a ~1,800-token system prompt against a
ferric-serve on PORT; run it once with FERRIC_PREFIX_CACHE=0 and once with the default, and compare: every
answer must be identical, and the follow-up turns should cost a fraction of the time and joules.

    python3 scripts/prefix_cache_check.py PORT > out.json
"""
import json, sys, time, urllib.request
port = sys.argv[1]
sysmsg = "You are a meticulous assistant for a robotics lab. " + " ".join(f"Rule {i}: always state units, cite the sensor, and keep answers under three sentences; the lab's arm has 7 joints and a 5 kg payload at a reach of 850 mm." for i in range(40))
msgs = [{"role": "system", "content": sysmsg}]
out = []
for q in ["What payload can the arm lift?", "And at what reach?", "How many joints does it have?", "Summarise the three facts in one sentence."]:
    msgs.append({"role": "user", "content": q})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
        json.dumps({"messages": msgs, "max_tokens": 60}).encode(), {"content-type": "application/json"})))
    dt = time.time() - t0
    a = r["choices"][0]["message"]["content"]
    msgs.append({"role": "assistant", "content": a})
    e = r.get("energy", {})
    out.append({"q": q, "a": a, "prompt": r["usage"]["prompt_tokens"], "secs": round(dt, 3), "joules": e.get("joules")})
print(json.dumps(out))
