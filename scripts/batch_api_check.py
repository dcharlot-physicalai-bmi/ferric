#!/usr/bin/env python3
"""The Batch API, driven by the official OpenAI Python SDK (the client people use against it).

Uploads a JSONL of chat requests, creates a batch, polls it to completion, downloads the output and error
files, and checks: every successful line's answer equals the same request made live (greedy), a request the
server refuses lands in the error file with its 400, request_counts and usage add up, a malformed file fails
the batch before anything runs (OpenAI's validation codes and line numbers), cancel stops a running batch,
and list/retrieve/delete work. Runs with --api-key, so the lines the server sends itself carry the key.

usage: batch_api_check.py <ferric-serve binary> <model.gguf>      needs: pip install openai
"""
import json, subprocess, sys, time, urllib.request
from openai import OpenAI

BIN, MODEL, PORT, KEY = sys.argv[1], sys.argv[2], 18481, "batch-key"
fail = []
def check(c, what):
    print(("  ok   " if c else "  FAIL ") + what)
    if not c: fail.append(what)

srv = subprocess.Popen([BIN, MODEL, "--port", str(PORT), "--api-key", KEY], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(240):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
        except Exception: time.sleep(0.5)
    c = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key=KEY)
    model = c.models.list().data[0].id
    qs = ["Name a primary color.", "What is 2 + 2?", "Name a planet.", "Say hello in French.", "Name a fruit.", "What is the capital of Japan?"]
    lines = [{"custom_id": f"q{i}", "method": "POST", "url": "/v1/chat/completions",
              "body": {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 32, "temperature": 0}} for i, q in enumerate(qs)]
    lines.insert(3, {"custom_id": "refused", "method": "POST", "url": "/v1/chat/completions",
                     "body": {"model": model, "messages": [{"role": "user", "content": "x"}], "n": 99}})
    f = c.files.create(file=("in.jsonl", "\n".join(json.dumps(l) for l in lines).encode()), purpose="batch")
    check(f.id.startswith("file-") and f.bytes > 0 and f.purpose == "batch", f"files.create: {f.id}")
    b = c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"run": "check"})
    check(b.status in ("validating", "in_progress") and b.metadata == {"run": "check"}, f"batches.create: {b.id} {b.status}")
    t0 = time.time()
    while b.status not in ("completed", "failed", "cancelled") and time.time() - t0 < 600:
        time.sleep(0.5); b = c.batches.retrieve(b.id)
    check(b.status == "completed", f"batch completed ({time.time() - t0:.1f} s)")
    rc = b.request_counts
    check((rc.total, rc.completed, rc.failed) == (7, 6, 1), f"request_counts {rc.total}/{rc.completed}/{rc.failed}")
    out = [json.loads(l) for l in c.files.content(b.output_file_id).text.splitlines()]
    err = [json.loads(l) for l in c.files.content(b.error_file_id).text.splitlines()]
    check([o["custom_id"] for o in out] == [l["custom_id"] for l in lines if l["custom_id"] != "refused"], "output keeps input order, refused line absent")
    check(len(err) == 1 and err[0]["custom_id"] == "refused" and err[0]["response"]["status_code"] == 400, "the refused request is in the error file with its 400")
    same = 0
    for o, l in zip(out, [l for l in lines if l["custom_id"] != "refused"]):
        live = c.chat.completions.create(**l["body"])
        same += o["response"]["body"]["choices"][0]["message"]["content"] == live.choices[0].message.content
    check(same == 6, f"each batch answer == the same request made live: {same}/6")
    u_in = sum(o["response"]["body"]["usage"]["prompt_tokens"] for o in out)
    u_out = sum(o["response"]["body"]["usage"]["completion_tokens"] for o in out)
    bu = b.model_extra.get("usage") if b.model_extra else None
    bu = bu or (b.usage.model_dump() if getattr(b, "usage", None) else None)
    check(bu and bu["input_tokens"] == u_in and bu["output_tokens"] == u_out, f"batch usage == summed responses ({u_in} in, {u_out} out)")
    fe = (b.model_extra or {}).get("ferric_energy")
    j = [o["response"]["body"].get("energy", {}).get("joules") for o in out]
    j = [x for x in j if x is not None]
    check(fe is not None and fe["responses_attributed"] == len(j) and (not j or abs(fe["joules"] - sum(j)) < 1e-6),
          f"ferric_energy sums the responses' joules ({len(j)} attributed)")
    # Validation fails the whole batch, with OpenAI's codes and 1-based lines.
    bad = c.files.create(file=("bad.jsonl", (json.dumps(lines[0]) + "\n" + json.dumps(lines[0]) + "\n{oops\n").encode()), purpose="batch")
    bb = c.batches.create(input_file_id=bad.id, endpoint="/v1/chat/completions", completion_window="24h")
    for _ in range(40):
        if bb.status in ("failed", "completed"): break
        time.sleep(0.25); bb = c.batches.retrieve(bb.id)
    codes = [(e.code, e.line) for e in (bb.errors.data if bb.errors else [])]
    check(bb.status == "failed" and codes == [("duplicate_custom_id", 2), ("invalid_json_line", 3)], f"malformed file fails validation: {codes}")
    # Cancel a long batch.
    long = [{"custom_id": f"l{i}", "method": "POST", "url": "/v1/chat/completions",
             "body": {"model": model, "messages": [{"role": "user", "content": f"Write a long story number {i}."}], "max_tokens": 400}} for i in range(40)]
    lf = c.files.create(file=("long.jsonl", "\n".join(json.dumps(l) for l in long).encode()), purpose="batch")
    lb = c.batches.create(input_file_id=lf.id, endpoint="/v1/chat/completions", completion_window="24h")
    time.sleep(2)
    lb = c.batches.cancel(lb.id)
    check(lb.status in ("cancelling", "cancelled"), f"cancel answers {lb.status}")
    for _ in range(600):
        if lb.status == "cancelled": break
        time.sleep(0.5); lb = c.batches.retrieve(lb.id)
    check(lb.status == "cancelled" and lb.request_counts.completed < 40, f"cancelled with {lb.request_counts.completed}/40 done")
    ids = [x.id for x in c.batches.list().data]
    check(b.id in ids and bb.id in ids and lb.id in ids, "batches.list has all three")
    check(c.files.delete(f.id).deleted and f.id not in [x.id for x in c.files.list().data], "files.delete")
    enc = c.embeddings.create  # noqa: F841 (embeddings batches share the path; not every model serves them)
finally:
    srv.terminate(); srv.wait(timeout=30)
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
