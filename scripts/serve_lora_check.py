import json, sys, threading, urllib.request
P = sys.argv[1]; ref = json.load(open(sys.argv[2]))
def comp(body):
    r = urllib.request.Request(f"http://127.0.0.1:{P}/v1/completions", data=json.dumps({**body, "max_tokens": 24, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
    try: return json.loads(urllib.request.urlopen(r, timeout=600).read())["choices"][0]["text"]
    except urllib.error.HTTPError as e: return "HTTP %d %s" % (e.code, e.read().decode()[:160])
clean = lambda s: s.split("<|endoftext|>")[0].split("<|im_end|>")[0]
ids = [m["id"] for m in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{P}/v1/models").read())["data"]]
print("listed:", [i for i in ids if "syn" in i or "F32" in i])
ok = True
for k, p in enumerate(ref["prompts"]):
    for name, spellings in [("base", [{}]), ("syn_a", [{"model": "syn_a"}, {"lora": [{"id": 0, "scale": 1}]}, {"lora": [{"name": "syn_a"}]}]),
                            ("syn_b", [{"model": "syn_b"}, {"lora": [{"id": 1}]}])]:
        want = clean(ref["peft"][name][k])
        for sp in spellings:
            got = comp({"prompt": p, **sp})
            same = got == want; ok &= same
            if not same: print(f"MISMATCH {name} {sp}: got {got!r}\n   want {want!r}")
    print(f"prompt {k}: base/syn_a/syn_b all spellings == PEFT")
# concurrency: one batch holding rows base / syn_a / syn_b; each must equal its solo (PEFT) answer
res = {}
def w(i, name, sp): res[i] = (name, comp({"prompt": ref["prompts"][0], **sp}))
ts = [threading.Thread(target=w, args=(i, n, sp)) for i, (n, sp) in enumerate([("base", {}), ("syn_a", {"model": "syn_a"}), ("syn_b", {"model": "syn_b"})] * 3)]
[t.start() for t in ts]; [t.join() for t in ts]
conc = all(got == clean(ref["peft"][n][0]) for n, got in res.values()); ok &= conc
print("9 concurrent requests, rows base/syn_a/syn_b interleaved: each == PEFT:", conc)
print("unknown adapter:", comp({"prompt": "x", "lora": [{"name": "nope"}]})[:120])
print("ALL OK" if ok else "FAILED")
