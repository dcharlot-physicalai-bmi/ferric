#!/usr/bin/env python3
"""libferric (the C ABI, through the Python ctypes binding) answers exactly what ferric-serve answers.

For each model: the same requests to the binding and to a ferric-serve HTTP server on the same file —
content, reasoning_content, finish_reason and usage must be identical (greedy); the streamed pieces must
concatenate to the final answer; a json_schema request must return JSON that validates by construction
(parsed and key-checked); an error comes back as an exception carrying the server's message. Run it on a
non-Qwen model too: the old ABI loaded every GGUF as a dense Qwen3 with Qwen's EOS ids.

usage: ffi_check.py <ferric-serve binary> <model.gguf> [model2.gguf ...]   (build: cargo build --release -p ferric-ffi -p ferric-serve)
"""
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "ferric-ffi", "bindings", "python"))
import ferric

BIN, MODELS, PORT = sys.argv[1], sys.argv[2:], 18501
Q = [{"role": "user", "content": "Name two rivers in Europe, briefly."}]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"]}
CHATS = [
    ("chat", {"messages": Q, "max_tokens": 48, "temperature": 0}),
    ("chat sampled, seeded", {"messages": Q, "max_tokens": 32, "temperature": 0.9, "seed": 11, "top_p": 0.9}),
    ("chat json_schema", {"messages": [{"role": "user", "content": "Give a city and its population as JSON."}], "max_tokens": 64, "temperature": 0,
                          "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}}),
]
fail = []
def check(c, what):
    print(("  ok   " if c else "  FAIL ") + what)
    if not c: fail.append(what)

def http(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())

def key(v):
    c = v["choices"][0]
    m = c.get("message") or {}
    return {"content": m.get("content", c.get("text")), "reasoning": m.get("reasoning_content", ""), "finish": c["finish_reason"], "usage": v["usage"]}

for model in MODELS:
    print(f"== {model.split('/')[-1]}")
    srv = subprocess.Popen([BIN, model, "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
            except Exception: time.sleep(0.5)
        want = {n: key(http("/v1/chat/completions", b)) for n, b in CHATS}
        want["completion"] = key(http("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 12, "temperature": 0}))
    finally:
        srv.terminate(); srv.wait(timeout=30)
    m = ferric.Model(model)
    for n, b in CHATS:
        got = key(m.chat(b["messages"], **{k: v for k, v in b.items() if k != "messages"}))
        check(got == want[n], f"{n}: binding == HTTP" + ("" if got == want[n] else f"\n         binding {json.dumps(got)[:160]}\n         http    {json.dumps(want[n])[:160]}"))
    pieces = {False: [], True: []}
    r = m.chat(Q, stream=lambda t, reasoning: pieces[reasoning].append(t), max_tokens=48, temperature=0)
    k = key(r)
    check("".join(pieces[False]).strip() == (k["content"] or "").strip() and "".join(pieces[True]).strip() == (k["reasoning"] or "").strip(),
          f"streamed pieces ({len(pieces[False])} answer, {len(pieces[True])} reasoning) concatenate to the final answer")
    check(k == want["chat"], "streamed call's final object == HTTP")
    js = json.loads(want["chat json_schema"]["content"]) if want["chat json_schema"]["finish"] == "stop" else None
    check(js is not None and set(js) == {"city", "population"} and isinstance(js["population"], int), f"json_schema answer is the schema's shape: {js}")
    got = key(m.complete("The capital of France is", max_tokens=12, temperature=0))
    check(got == want["completion"], f"completion: binding == HTTP: {got['content']!r}")
    try:
        m.chat(Q, n=99); check(False, "an invalid request raises")
    except ferric.FerricError as e:
        check("`n`" in str(e), f"an invalid request raises with the server's message: {str(e)[:60]}")
    m.close()
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
