#!/usr/bin/env python3
"""Live checks of ferric-serve request features against a running server, each against its definition:

  n > 1        choice 0 == the n = 1 answer; choice i == the request with seed + i; streamed == non-streamed
  constraints  json_object parses; guided_regex satisfies Python re.fullmatch; guided_choice is a choice;
               GBNF (llama.cpp's json.gbnf) output parses when it finishes; two constraints -> 400
  json schema  response_format json_schema / guided_json / structured_outputs.json: every answer finishes
               and validates with the `jsonschema` package (nested objects, arrays of objects, $defs, enums,
               nullable, a discriminated union, a tuple, string lengths, integer ranges, pattern, date, uuid);
               a schema the converter refuses -> 400 naming why, streamed too; decode tok/s with and without
  refusals     n out of range, n with tools, streamed completion with n > 1 -> 400

usage: serve_features_check.py <port> <json.gbnf path>
(the json schema section needs `jsonschema`: uv venv v && uv pip install --python v/bin/python jsonschema,
then run this with v/bin/python)
"""
import json, re, sys, time, urllib.request

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

# JSON Schema: compiled to GBNF as llama.cpp compiles it; every answer must finish and validate.
try:
    import jsonschema
except ImportError:
    jsonschema = None
    check(False, "json_schema: the `jsonschema` package is needed for this section (see usage)")
PERSON = {"type": "object", "properties": {
    "name": {"type": "string", "maxLength": 40}, "born": {"type": "integer", "minimum": 1000, "maximum": 2100},
    "fields": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 3}},
    "required": ["name", "born", "fields"], "additionalProperties": False}
SCHEMAS = [
    ("person: string length, integer range, bounded array", PERSON, "Describe Ada Lovelace as JSON: name, year born, fields of work."),
    ("classification: enum, number, boolean, nullable", {"type": "object", "properties": {
        "sentiment": {"enum": ["positive", "negative", "neutral"]}, "confidence": {"type": "number"}, "sarcastic": {"type": "boolean"},
        "topic": {"type": ["string", "null"], "maxLength": 30}}, "required": ["sentiment", "confidence", "sarcastic", "topic"],
        "additionalProperties": False}, "Classify this review: 'I love this product, it works perfectly!'"),
    ("order: nested object, array of $ref objects, pattern, date", {"$defs": {"item": {"type": "object", "properties": {
        "sku": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"}, "qty": {"type": "integer", "minimum": 1, "maximum": 99}},
        "required": ["sku", "qty"], "additionalProperties": False}}, "type": "object", "properties": {
        "customer": {"type": "object", "properties": {"name": {"type": "string", "maxLength": 30}, "city": {"type": "string", "maxLength": 30}},
                     "required": ["name", "city"], "additionalProperties": False},
        "items": {"type": "array", "items": {"$ref": "#/$defs/item"}, "minItems": 1, "maxItems": 3},
        "date": {"type": "string", "format": "date"}}, "required": ["customer", "items", "date"], "additionalProperties": False},
     "Make up an order: a customer in Paris buying two items, with the order date."),
    ("shape: anyOf of objects told apart by const", {"anyOf": [
        {"type": "object", "properties": {"kind": {"const": "circle"}, "radius": {"type": "number"}}, "required": ["kind", "radius"], "additionalProperties": False},
        {"type": "object", "properties": {"kind": {"const": "rect"}, "w": {"type": "number"}, "h": {"type": "number"}}, "required": ["kind", "w", "h"], "additionalProperties": False}]},
     "Describe a circle of radius 2 as JSON."),
    ("record: tuple of uuid, small integer, boolean", {"type": "array", "prefixItems": [
        {"type": "string", "format": "uuid"}, {"type": "integer", "minimum": -5, "maximum": 5}, {"type": "boolean"}]},
     "Give a record: an id, a number from -5 to 5, and a flag."),
]
for name, schema, q in (SCHEMAS if jsonschema else []):
    c, v = chat({"response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}}, q, 300)
    out, fr = (v["choices"][0]["message"]["content"], v["choices"][0]["finish_reason"]) if c == 200 else (str(v), None)
    try:
        jsonschema.validate(json.loads(out), schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER); ok = True
    except Exception as e:
        ok = False; out = f"{out} <- {type(e).__name__}"
    check(c == 200 and fr == "stop" and ok, f"json_schema {name} ({fr}): {out[:90]!r}")
for spelling in ([{"guided_json": PERSON}, {"guided_json": json.dumps(PERSON)}, {"structured_outputs": {"json": PERSON}}] if jsonschema else []):
    c, v = chat(spelling, "Describe Grace Hopper as JSON: name, year born, fields of work.", 300)
    out = v["choices"][0]["message"]["content"] if c == 200 else str(v)
    try: jsonschema.validate(json.loads(out), PERSON); ok = True
    except Exception: ok = False
    check(c == 200 and ok, f"{list(spelling)[0]}{' (as text)' if isinstance(spelling.get('guided_json'), str) else ''} validates: {out[:60]!r}")
c, v = chat({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "kaboom"}}}}, "x", 4)
check(c == 400 and "unrecognized type kaboom" in v, f"json_schema with an unknown type -> 400 naming it: {c} {v[:90]!r}")
c, v = chat({"guided_json": {"$ref": "https://example.com/schema.json"}}, "x", 4)
check(c == 400 and "same document" in v, f"a remote $ref -> 400, never fetched: {c}")
c, v = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 4, "stream": True,
                                     "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "kaboom"}}}}, raw=True)
check(c == 400, f"streamed, a refused schema is still a 400 (not a 200 stream): {c}")


def decode_rate(extra, q, n):
    """Decode tokens/s of one streamed chat: tokens after the first over the time from first to last chunk."""
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, **extra}
    r = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    stamps, toks = [], 0
    with urllib.request.urlopen(r, timeout=600) as f:
        for line in f:
            line = line.decode()
            if not line.startswith("data: {"): continue
            d = json.loads(line[6:])
            if d.get("usage"): toks = d["usage"]["completion_tokens"]
            if any((ch.get("delta") or {}).get("content") for ch in d.get("choices", [])): stamps.append(time.perf_counter())
    return toks, ((toks - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else float("nan"))


RATE = dict(PERSON, properties=dict(PERSON["properties"], name={"type": "string", "maxLength": 41}))  # a schema not used above: cold
q = "Describe Ada Lovelace as JSON: name, year born, fields of work."
for label, extra in [("unconstrained", {}), ("json_schema, cold", {"response_format": {"type": "json_schema", "json_schema": {"schema": RATE}}}),
                     ("json_schema, warm", {"response_format": {"type": "json_schema", "json_schema": {"schema": RATE}}}),
                     ("json_object", {"response_format": {"type": "json_object"}})]:
    toks, rate = decode_rate(extra, q, 120)
    print(f"  info decode {rate:6.1f} tok/s over {toks} tokens: {label}")
check(post("/v1/chat/completions", dict(base, n=17))[0] == 400, "n = 17 -> 400")
check(post("/v1/chat/completions", dict(base, n=2, tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]))[0] == 400, "n > 1 with tools -> 400")
check(post("/v1/completions", {"prompt": "x", "n": 2, "stream": True})[0] == 400, "streamed completion n > 1 -> 400")
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
