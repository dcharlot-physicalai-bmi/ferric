"""Semantic check of the JSON-Schema grammars: which JSON texts a schema's grammar accepts, against which the
`jsonschema` package (the JSON Schema reference validator for Python) says are valid.

For every schema (the authors' test schemas and realistic API ones), instances are generated: values built
from the schema (keys in the order the grammar emits them: required, then optional, then additional), then
mutated (types swapped, integers stepped across bounds, strings and arrays grown and shrunk, keys dropped,
added and reordered), plus probes aimed at each place llama.cpp's grammar is known to depart from JSON Schema.
Each is serialized the ways the grammar's `space` rule allows (", " / ": ", compact, indented), and labelled by
jsonschema (Draft 2020-12, 2019-09 where `items` is a list; with that draft's format checker).

The grammar's verdict comes from the port (examples/json_schema_accepts). Where they disagree, the cause must be
one of llama.cpp's documented departures, and it is established by a counterfactual, not asserted: the
departure is undone (the schema keyword it ignores removed, the property order it demands restored, the
whitespace it refuses normalized, ...) and the two verdicts must then agree. A disagreement no cause explains
stops the script. The fixture records each instance, jsonschema's verdict and the cause (if any);
tests/json_schema.rs requires the grammar to agree exactly where no cause is recorded and to disagree exactly
where one is.

usage: <venv>/bin/python make_semantic.py <json_schema_accepts binary> semantic.json.gz
(the venv: `uv venv -p python3.12 venv && uv pip install --python venv/bin/python jsonschema rfc3339-validator rstr`)
"""
import copy, gzip, json, math, random, re, subprocess, sys
from importlib.metadata import version
import rstr
import jsonschema
from jsonschema import Draft201909Validator, Draft202012Validator

here = __file__.rsplit("/", 1)[0]
authors = json.load(open(f"{here}/llama_cpp_cases.json"))
ACCEPTS = sys.argv[1]

# ------------------------------------------------------------------------------------------------ schemas
schemas = []  # (name, text)
seen = set()


def add(name, s):
    t = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    if t not in seen:
        seen.add(t)
        schemas.append((name, t))


for c in authors["cases"]:
    if c["status"] == "SUCCESS": add("authors: " + c["name"], c["schema"])
for i, c in enumerate(authors["schema_matching"]):
    add(f"authors (integration): {c['desc'] or i}", c["schema"])

REALISTIC = {
    "person": {"type": "object", "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40}, "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "pattern": "^[a-z0-9.]+@[a-z0-9]+\\.[a-z]{2,4}$"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
        "required": ["name", "age"], "additionalProperties": False},
    "order (nested objects, arrays of objects, formats)": {"type": "object", "properties": {
        "id": {"type": "string", "format": "uuid"},
        "customer": {"type": "object", "properties": {"name": {"type": "string"}, "address": {"type": "object", "properties": {
            "street": {"type": "string"}, "city": {"type": "string"}, "zip": {"type": "string", "pattern": "^[0-9]{5}$"}},
            "required": ["street", "city", "zip"], "additionalProperties": False}}, "required": ["name", "address"], "additionalProperties": False},
        "items": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "object", "properties": {
            "sku": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"}, "qty": {"type": "integer", "minimum": 1, "maximum": 99},
            "price": {"type": "number"}}, "required": ["sku", "qty", "price"], "additionalProperties": False}},
        "status": {"enum": ["pending", "shipped", "delivered"]}, "notes": {"type": "string", "maxLength": 20},
        "created": {"type": "string", "format": "date-time"}},
        "required": ["id", "customer", "items", "status"], "additionalProperties": False},
    "tool call: get_weather": {"type": "object", "properties": {
        "location": {"type": "string", "description": "City and country"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 14}}, "required": ["location"], "additionalProperties": False},
    "classification": {"type": "object", "properties": {
        "label": {"enum": ["positive", "negative", "neutral"]}, "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string", "maxLength": 12}, "maxItems": 3}},
        "required": ["label", "confidence"], "additionalProperties": False},
    "event (date, time, bools, small ranges)": {"type": "object", "properties": {
        "title": {"type": "string", "minLength": 3}, "date": {"type": "string", "format": "date"}, "start": {"type": "string", "format": "time"},
        "attendees": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "rsvp": {"type": "boolean"}},
                                                  "required": ["name", "rsvp"], "additionalProperties": False}},
        "priority": {"type": "integer", "minimum": -2, "maximum": 2}}, "required": ["title", "date"], "additionalProperties": False},
    "all optional": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "null"}}, "additionalProperties": False},
    "nullable fields": {"type": "object", "properties": {
        "middle_name": {"type": ["string", "null"]}, "score": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 100}, {"type": "null"}]}},
        "required": ["middle_name", "score"], "additionalProperties": False},
    "$defs reused": {"type": "object", "properties": {"billing": {"$ref": "#/$defs/address"}, "shipping": {"$ref": "#/$defs/address"}},
                     "required": ["billing"], "additionalProperties": False,
                     "$defs": {"address": {"type": "object", "properties": {"line": {"type": "string"}, "country": {"enum": ["US", "FR", "JP"]}},
                                           "required": ["line", "country"], "additionalProperties": False}}},
    "recursive tree": {"$ref": "#/$defs/node", "$defs": {"node": {"type": "object", "properties": {
        "value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "maxItems": 2}},
        "required": ["value"], "additionalProperties": False}}},
    "tuple (prefixItems)": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]},
    "string lengths": {"type": "object", "properties": {"code": {"type": "string", "minLength": 3, "maxLength": 3},
                                                        "desc": {"type": "string", "maxLength": 10}}, "required": ["code"], "additionalProperties": False},
    "integer ranges": {"type": "object", "properties": {"a": {"type": "integer", "minimum": -100, "maximum": 100},
                                                        "b": {"type": "integer", "minimum": 1000}, "c": {"type": "integer", "exclusiveMaximum": 10}},
                       "required": ["a", "b", "c"], "additionalProperties": False},
    "typed additional properties": {"type": "object", "properties": {"id": {"type": "integer"}},
                                    "required": ["id"], "additionalProperties": {"type": "integer", "minimum": 0}},
    "array of enums with bounds": {"type": "array", "items": {"enum": ["red", "green", "blue"]}, "minItems": 2, "maxItems": 4},
    "discriminated union (const)": {"anyOf": [
        {"type": "object", "properties": {"type": {"const": "circle"}, "r": {"type": "number"}}, "required": ["type", "r"], "additionalProperties": False},
        {"type": "object", "properties": {"type": {"const": "rect"}, "w": {"type": "number"}, "h": {"type": "number"}}, "required": ["type", "w", "h"], "additionalProperties": False}]},
    "OpenAI strict style (all required, nested)": {"type": "object", "properties": {
        "answer": {"type": "string"}, "steps": {"type": "array", "items": {"type": "object", "properties": {
            "explanation": {"type": "string"}, "output": {"type": "string"}}, "required": ["explanation", "output"], "additionalProperties": False}},
        "final": {"type": ["number", "null"]}}, "required": ["answer", "steps", "final"], "additionalProperties": False},
}
for k, s in REALISTIC.items(): add("realistic: " + k, s)


# ------------------------------------------------------------------------------------------------ helpers
def pointer(root, ref):
    t = root
    for sel in ref[1:].split("/")[1:]:
        if isinstance(t, dict) and sel in t: t = t[sel]
        elif isinstance(t, list): t = t[int(sel)]
        else: raise KeyError(ref)
    return t


def deref(root, s, n=0):
    while isinstance(s, dict) and isinstance(s.get("$ref"), str) and s["$ref"].startswith("#/") and n < 50:
        s = pointer(root, s["$ref"]); n += 1
    return s


def validator(schema):
    lists = []
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("items"), list): lists.append(1)
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(schema)
    # A list under `items` is a tuple up to Draft 2019-09; 2020-12 spells it prefixItems. Each draft's own format
    # checker: jsonschema's catch-all FormatChecker() keeps Draft 3's `time` (no offset allowed).
    cls = Draft201909Validator if lists else Draft202012Validator
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def is_valid(schema, value):
    return validator(schema).is_valid(value)


ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,:é日ß"


def rand_str(r, lo, hi):
    n = r.randint(lo, hi)
    s = "".join(r.choice(ALPHA) for _ in range(n))
    if n and r.random() < 0.15:  # characters JSON must escape, and one outside the BMP
        i = r.randrange(n); s = s[:i] + r.choice(['"', "\\", "\n", "\t", "😀", "/"]) + s[i + 1:]
    return s


def rand_any(r, depth=0):
    k = r.choice(["int", "num", "str", "bool", "null"] + (["arr", "obj"] if depth < 2 else []))
    if k == "int": return r.randint(-1000, 1000)
    if k == "num": return round(r.uniform(-100, 100), r.randint(0, 3))
    if k == "str": return rand_str(r, 0, 6)
    if k == "bool": return r.random() < 0.5
    if k == "null": return None
    if k == "arr": return [rand_any(r, depth + 1) for _ in range(r.randint(0, 3))]
    return {rand_str(r, 1, 4): rand_any(r, depth + 1) for _ in range(r.randint(0, 3))}


def int_bounds(s):
    lo = hi = None
    if "minimum" in s and isinstance(s["minimum"], (int, float)) and not isinstance(s["minimum"], bool): lo = math.ceil(s["minimum"])
    elif "exclusiveMinimum" in s and isinstance(s["exclusiveMinimum"], (int, float)): lo = math.floor(s["exclusiveMinimum"]) + 1
    if "maximum" in s and isinstance(s["maximum"], (int, float)) and not isinstance(s["maximum"], bool): hi = math.floor(s["maximum"])
    elif "exclusiveMaximum" in s and isinstance(s["exclusiveMaximum"], (int, float)): hi = math.ceil(s["exclusiveMaximum"]) - 1
    return lo, hi


def kind_of(s):  # the node kind llama.cpp's builder picks (json-schema.cpp build_node)
    t = s.get("type")
    if t in ("object", "array", "string", "integer", "number", "boolean", "null"): return t
    if "properties" in s or ("additionalProperties" in s and s["additionalProperties"] is not True): return "object"
    if "allOf" in s: return "allOf"
    if "items" in s or "prefixItems" in s: return "array"
    if any(k in s for k in ("pattern", "minLength", "maxLength", "format")): return "string"
    return "any"


def gen(root, s, r, depth=0):
    s = deref(root, s)
    if not isinstance(s, dict): return None
    for key in ("oneOf", "anyOf"):
        if isinstance(s.get(key), list) and s[key]: return gen(root, r.choice(s[key]), r, depth)
    if isinstance(s.get("type"), list) and s["type"]: return gen(root, {**s, "type": r.choice(s["type"])}, r, depth)
    if "const" in s: return copy.deepcopy(s["const"])
    if isinstance(s.get("enum"), list) and s["enum"]: return copy.deepcopy(r.choice(s["enum"]))
    k = kind_of(s)
    if k == "allOf":
        merged = {"type": "object", "properties": {}, "required": []}
        for c in s["allOf"]:
            c = deref(root, c)
            for alt in (c.get("anyOf", [c]) if isinstance(c, dict) else []):
                alt = deref(root, alt)
                if isinstance(alt, dict):
                    merged["properties"].update(alt.get("properties", {}))
                    if alt is c: merged["required"] += list(alt.get("properties", {}))
                    if "enum" in alt: return copy.deepcopy(r.choice(alt["enum"]))
        return gen(root, merged, r, depth)
    if k == "object":
        props = s.get("properties", {}) if isinstance(s.get("properties"), dict) else {}
        req = [p for p in props if p in (s.get("required") or [])]
        out = {}
        for p in req: out[p] = gen(root, props[p], r, depth + 1)
        for p in props:
            if p not in req and r.random() < 0.6: out[p] = gen(root, props[p], r, depth + 1)
        addl = s.get("additionalProperties", None if "properties" in s else True)
        if addl not in (None, False) and r.random() < 0.4:
            for _ in range(r.randint(1, 2)):
                key = r.choice(["extra", "zq", "note_1", "X", "日付", "a b"])
                if key in props or any(p.startswith(key) for p in props): continue
                out[key] = gen(root, addl, r, depth + 1) if isinstance(addl, dict) else rand_any(r, 1)
        return out
    if k == "array":
        tup = s.get("items") if isinstance(s.get("items"), list) else s.get("prefixItems") if isinstance(s.get("prefixItems"), list) else None
        if tup is not None: return [gen(root, x, r, depth + 1) for x in tup]
        items = s.get("items", s.get("prefixItems", {}))
        lo = s.get("minItems", 0) if isinstance(s.get("minItems"), int) else 0
        hi = s.get("maxItems", lo + 3) if isinstance(s.get("maxItems"), int) else lo + 3
        n = r.randint(lo, max(lo, min(hi, lo + 3))) if depth < 4 else lo
        return [gen(root, items, r, depth + 1) for _ in range(n)]
    if k == "string":
        if isinstance(s.get("pattern"), str) and s["pattern"]:
            try: return rstr.Rstr(r).xeger(s["pattern"])
            except Exception: return rand_str(r, 0, 6)
        f = s.get("format")
        if f == "date": return f"{r.randint(1900, 2099)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
        if f == "time": return f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}" + r.choice(["Z", ".123Z", "+05:30", "-08:00"])
        if f == "date-time": return f"{r.randint(1900, 2099)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}" + r.choice(["Z", ".500Z", "+01:00"])
        if f in ("uuid", "uuid1", "uuid4"): return "-".join("".join(r.choice("0123456789abcdefABCDEF") for _ in range(n)) for n in (8, 4, 4, 4, 12))
        lo = s.get("minLength", 0) if isinstance(s.get("minLength"), int) else 0
        hi = s.get("maxLength", lo + 8) if isinstance(s.get("maxLength"), int) else lo + 8
        return rand_str(r, lo, max(lo, min(hi, lo + 8)))
    if k == "integer":
        lo, hi = int_bounds(s)
        if lo is not None and hi is not None and lo <= hi:
            return r.choice([lo, hi, r.randint(lo, hi), r.randint(lo, hi)])
        if lo is not None: return r.choice([lo, lo + 1, lo + r.randint(0, 10 ** r.randint(1, 6))])
        if hi is not None: return r.choice([hi, hi - 1, hi - r.randint(0, 10 ** r.randint(1, 6))])
        return r.randint(-10 ** r.randint(1, 9), 10 ** r.randint(1, 9))
    if k == "number": return r.choice([r.randint(-1000, 1000), round(r.uniform(-1000, 1000), r.randint(1, 4))])
    if k == "boolean": return r.random() < 0.5
    if k == "null": return None
    return rand_any(r)


def paths(v, p=()):
    yield p
    if isinstance(v, dict):
        for k, x in v.items(): yield from paths(x, p + (k,))
    elif isinstance(v, list):
        for i, x in enumerate(v): yield from paths(x, p + (i,))


def get_at(v, p):
    for k in p: v = v[k]
    return v


def set_at(v, p, x):
    if not p: return x
    get_at(v, p[:-1])[p[-1]] = x
    return v


MUTATIONS = ["type", "int-step", "str-grow", "str-shrink", "arr-grow", "arr-shrink", "obj-drop", "obj-extra", "obj-shuffle", "enum-miss", "to-float", "null"]


def mutate(v, r):
    """(mutated value, mutation name, path) or None when this mutation does not apply anywhere."""
    v = copy.deepcopy(v)
    m = r.choice(MUTATIONS)
    ps = list(paths(v))
    r.shuffle(ps)
    for p in ps:
        x = get_at(v, p)
        if m == "type":
            y = r.choice([0, 1.5, "s", True, None, [], {}])
            if type(y) != type(x): return set_at(v, p, y), m, p
        elif m == "int-step" and isinstance(x, int) and not isinstance(x, bool): return set_at(v, p, x + r.choice([-1, 1])), m, p
        elif m == "str-grow" and isinstance(x, str): return set_at(v, p, x + r.choice(["a", "zz", "é", " "])), m, p
        elif m == "str-shrink" and isinstance(x, str) and x: return set_at(v, p, x[:-1]), m, p
        elif m == "arr-grow" and isinstance(x, list) and x: return set_at(v, p, x + [copy.deepcopy(x[-1])]), m, p
        elif m == "arr-shrink" and isinstance(x, list) and x: return set_at(v, p, x[:-1]), m, p
        elif m == "obj-drop" and isinstance(x, dict) and x:
            k = r.choice(list(x)); del x[k]; return v, m, p + (k,)
        elif m == "obj-extra" and isinstance(x, dict) and "extra_key" not in x:
            x["extra_key"] = r.choice([1, "x", None]); return v, m, p
        elif m == "obj-shuffle" and isinstance(x, dict) and len(x) >= 2:
            items = list(x.items()); items.reverse(); x.clear(); x.update(items); return v, m, p
        elif m == "enum-miss" and isinstance(x, str): return set_at(v, p, "not-a-member"), m, p
        elif m == "to-float" and isinstance(x, int) and not isinstance(x, bool): return set_at(v, p, x + 0.5), m, p
        elif m == "null" and x is not None: return set_at(v, p, None), m, p
    return None


def dumps(v, style):
    if style == "compact": return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if style == "indent": return json.dumps(v, ensure_ascii=False, indent=2)
    return json.dumps(v, ensure_ascii=False)


def depth(v):
    return 1 + max((depth(x) for x in (v.values() if isinstance(v, dict) else v)), default=0) if isinstance(v, (dict, list)) else 0


# ------------------------------------------------------------------------------------------------ instances
cases = []
for si, (name, text) in enumerate(schemas):
    schema = json.loads(text)
    r = random.Random(7919 * si + 17)
    inst = []  # (value or None, text, how)
    try:
        type(validator(schema)).check_schema(schema)
    except Exception as e:
        cases.append({"name": name, "schema": text, "skipped": f"not a valid schema for {type(validator(schema)).__name__}: {str(e).splitlines()[0]}"[:300]})
        continue
    base = []
    for _ in range(40):
        try: base.append(gen(schema, schema, r))
        except (RecursionError, KeyError, ValueError, TypeError): pass
    for v in base:
        style = r.choice(["default", "default", "compact", "indent"] if depth(v) <= 9 else ["default", "compact"])
        inst.append((v, dumps(v, style), "generated/" + style))
    for v in base:
        for _ in range(3):
            m = mutate(v, r)
            if m: inst.append((m[0], dumps(m[0], r.choice(["default", "compact"])), f"mutated/{m[1]}@{'/'.join(map(str, m[2]))}"))
    for t in ["null", "true", "0", "-1", "1.5", '""', '"x"', "[]", "{}", "[1, 2]", '{"a": 1}']:
        inst.append((json.loads(t), t, "generic"))
    # probes: the places llama.cpp's grammar departs from JSON Schema
    v = next((b for b in base if isinstance(b, (dict, list)) and len(b) >= 2), None)
    if v is not None:
        t = dumps(v, "default")
        inst.append((v, t.replace(", ", ",  ", 1), "probe/two spaces after a comma"))
        inst.append((v, t.replace(", ", " , ", 1), "probe/a space before a comma"))
        inst.append((v, " " + t, "probe/leading space"))
        inst.append((v, t + "\n", "probe/trailing newline"))
        inst.append((v, dumps(v, "indent").replace("\n  ", "\n" + " " * 22, 1) if "\n  " in dumps(v, "indent") else t, "probe/22 blanks of indent"))
        inst.append((v, dumps(v, "default").replace(", ", ",\n\n\n", 1), "probe/three newlines"))
    for b in base:
        if isinstance(b, dict) and len(b) >= 2:
            w = dict(reversed(list(b.items()))); inst.append((w, dumps(w, "default"), "probe/keys reversed")); break
    for b in base:
        for p in paths(b):
            x = get_at(b, p)
            if isinstance(x, int) and not isinstance(x, bool):
                w = set_at(copy.deepcopy(b), p, float(x)); inst.append((w, dumps(w, "default"), "probe/integer written as a float@" + "/".join(map(str, p)))); break
        else: continue
        break
    for b in base:
        for p in paths(b):
            x = get_at(b, p)
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                for spelled in ["1e-05", "0.30000000000000004", "12345678901234567", "1E+2", "-0.0"]:
                    w = set_at(copy.deepcopy(b), p, json.loads(spelled))
                    inst.append((w, dumps(set_at(copy.deepcopy(b), p, "@@N@@"), "default").replace('"@@N@@"', spelled), f"probe/number spelled {spelled}@" + "/".join(map(str, p))))
                break
        else: continue
        break
    for b in base:
        for p in paths(b):
            x = get_at(b, p)
            if isinstance(x, str):
                for s2, how in [(x + "\x7f", "a raw DEL character"), (x + "😀", "a character outside the BMP, \\u-escaped as a surrogate pair")]:
                    w = set_at(copy.deepcopy(b), p, s2)
                    t = dumps(w, "default") if "DEL" in how else json.dumps(w)
                    inst.append((w, t, "probe/" + how + "@" + "/".join(map(str, p))))
                break
        else: continue
        break
    for b in base:
        if isinstance(b, dict):
            w = dict(b); w["zz_extra"] = 1; inst.append((w, dumps(w, "default"), "probe/an extra key")); break
    # dedupe by text
    uniq, texts = [], set()
    for v, t, how in inst:
        if t not in texts: texts.add(t); uniq.append((v, t, how))
    val = validator(schema)
    rows = []
    try:
        for v, t, how in uniq:
            try: value = json.loads(t)
            except json.JSONDecodeError: continue
            rows.append({"text": t, "valid": val.is_valid(value), "how": how})
    except Exception as e:
        cases.append({"name": name, "schema": text, "skipped": f"jsonschema fails evaluating it: {type(e).__name__}: {e}"[:300]})
        continue
    cases.append({"name": name, "schema": text, "instances": rows})

# ------------------------------------------------------------------------------------------------ verdicts
def accepts(queries):
    """[(schema text, [texts])] -> [[bool]] from the port's grammar (None where the schema does not convert)."""
    out = json.loads(subprocess.run([ACCEPTS], input=json.dumps([{"schema": s, "instances": ts} for s, ts in queries]),
                                    capture_output=True, text=True, check=True).stdout)
    return [None if isinstance(o, dict) else o for o in out]


live = [c for c in cases if "instances" in c]
for c, vs in zip(live, accepts([(c["schema"], [i["text"] for i in c["instances"]]) for c in live])):
    assert vs is not None, f"{c['name']}: no grammar"
    for i, g in zip(c["instances"], vs): i["grammar"] = g


# ------------------------------------------------------------------------------------------------ causes
# Respellings: the same JSON value written another way. The grammar's verdict on the respelled text (same
# schema) must equal jsonschema's verdict on the value. A small tokenizer keeps every token's spelling except
# the one being changed.
TOKEN = re.compile(r'\s+|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null|[{}\[\],:]', re.S)


def tokens(t):
    out, p = [], 0
    while p < len(t):
        m = TOKEN.match(t, p)
        if not m: return None
        out.append(m.group(0)); p = m.end()
    return out


def respace(t, item=", ", colon=": "):
    toks = [x for x in (tokens(t) or []) if not x.isspace()]
    return "".join(item if x == "," else colon if x == ":" else x for x in toks) if toks else None


NUMBER_OK = re.compile(r"-?(0|[1-9]\d{0,15})(\.\d{1,16})?([eE][-+]?(0|[1-9]\d{0,15}))?$")  # the grammar's number rule


def number_spelling(x):
    if not NUMBER_OK.match(x) or (("." in x or "e" in x.lower()) and float(x).is_integer() and abs(float(x)) < 1e16):
        v = float(x) if ("." in x or "e" in x.lower()) else int(x)
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e16: return str(int(v))
        if isinstance(v, int): return x  # an integer of 17+ digits has no other spelling
        r_ = repr(v)
        m = re.match(r"(-?)(\d)(?:\.(\d+))?e([-+])(\d+)$", r_) or re.match(r"(-?)(\d)\.(\d+)$", r_)
        if "e" in r_:
            sign, d, frac, es, ed = m.groups()
            return f"{sign}{d}{'.' + frac if frac else ''}e{es}{int(ed)}"
        ip, fp = r_.lstrip("-").split(".")
        if len(fp) <= 16: return r_
        digits = (ip + fp).lstrip("0"); exp = len(ip) - 1 if ip != "0" else -(len(fp) - len(fp.lstrip("0")) + 1)
        return f"{'-' if r_.startswith('-') else ''}{digits[0]}.{digits[1:] or '0'}e{exp}"
    return x


def respell_numbers(t):
    toks = tokens(t)
    return "".join(number_spelling(x) if re.match(r"-?\d", x) else x for x in toks) if toks else None


def respell_strings(t, fn):
    toks = tokens(t)
    return "".join(fn(x) if x.startswith('"') else x for x in toks) if toks else None


def escape_del(s): return s.replace("\x7f", "\\u007f")


def raw_astral(s): return re.sub(r"\\u(d[89ab][0-9a-f]{2})\\u(d[c-f][0-9a-f]{2})", lambda m: json.loads(f'"{m.group(0)}"'), s, flags=re.I)


def raw_chars(s):  # a string's escapes written as the characters they stand for (not JSON any more)
    return '"' + json.loads(s) + '"'


def grammar_order(v, s, root):
    """The value with every object's keys in the order the grammar emits them (required, then optional, in schema
    order, then the rest); one candidate per anyOf branch at the top."""
    s = deref(root, s)
    if not isinstance(s, dict): return [v]
    for key in ("oneOf", "anyOf"):
        if isinstance(s.get(key), list): return [w for alt in s[key] for w in grammar_order(v, alt, root)]
    if isinstance(v, dict):
        props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
        if "allOf" in s and not props:
            props = {}
            for c in s["allOf"]:
                for alt in (deref(root, c).get("anyOf", [c]) if isinstance(deref(root, c), dict) else []):
                    alt = deref(root, alt)
                    if isinstance(alt, dict): props.update(alt.get("properties", {}))
        req = [p for p in props if p in (s.get("required") or [])]
        order = req + [p for p in props if p not in req]
        out = {k: grammar_order(v[k], props[k], root)[0] for k in order if k in v}
        ap = s.get("additionalProperties")
        out.update({k: (grammar_order(x, ap, root)[0] if isinstance(ap, dict) else x) for k, x in v.items() if k not in out})
        return [out]
    if isinstance(v, list):
        it = s.get("items") if isinstance(s.get("items"), dict) else {}
        return [[grammar_order(x, it, root)[0] for x in v]]
    return [v]


class Raw(str): pass


def dumps_raw(v):  # default separators, numbers as they were spelled
    if isinstance(v, Raw): return str(v)
    if isinstance(v, dict): return "{" + ", ".join(json.dumps(k, ensure_ascii=False) + ": " + dumps_raw(x) for k, x in v.items()) + "}"
    if isinstance(v, list): return "[" + ", ".join(dumps_raw(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def reorder(t, schema):
    v = json.loads(t, parse_int=Raw, parse_float=Raw)
    return [dumps_raw(w) for w in grammar_order(v, schema, schema)]


def truncate_long_integers(t):
    toks = tokens(t)
    return "".join(re.sub(r"^(-?[1-9]\d{15})\d+$", r"\1", x) for x in toks) if toks else None


# Limits: the text cannot be written in the grammar at all; the counterfactual changes the value minimally (it
# must keep jsonschema's verdict) and the grammar must then accept it.
LIMIT = {
    "integer-digits": ("a number's integral part has at most 16 digits (integral-part ::= [0] | [1-9] [0-9]{0,15}), so an "
                       "integer of 17 or more digits has no spelling; the counterfactual cuts it to its first 16 digits",
                       lambda t, s: [truncate_long_integers(t)]),
}

RESPELL = {  # id: (explanation, text -> [candidate texts])
    "whitespace": ("whitespace outside the grammar's `space` rule (nothing, one space, or 1-2 newlines and up to 20 blanks; "
                   "none before a comma or at either end)", lambda t, s: [respace(t)]),
    "compact-literal": ("a const/enum object or array must be written as its compact serialization (no blanks)",
                        lambda t, s: [respace(t, ",", ":")]),
    "number-spelling": ("a number spelled outside the grammar's number syntax: an integer written as a float (1.0, 1E+2), "
                        "a fraction of more than 16 digits, an exponent with a leading zero", lambda t, s: [respell_numbers(t)]),
    "raw-del": ("the grammar's char rule excludes a raw U+007F, which JSON allows unescaped",
                lambda t, s: [respell_strings(t, escape_del)]),
    "astral-escape-length": ("maxLength/minLength count the grammar's char units: a \\u-escaped surrogate pair is two, "
                             "a character outside the BMP is one code point to JSON Schema", lambda t, s: [respell_strings(t, raw_astral)]),
    "key-order": ("properties must come in the grammar's order: required ones, then optional ones, each in schema order, "
                  "then additional ones", lambda t, s: reorder(t, s)),
    "pattern-raw-chars": ("a pattern is matched against the raw text between the quotes: a character JSON must escape "
                          "(\" \\ control characters) cannot match as the pattern says, its escape sequence is seen instead",
                          lambda t, s: [respell_strings(t, raw_chars)] if "pattern" in json.dumps(s) else []),
}

IGNORED_NUMBER = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")
IGNORED = ("uniqueItems", "contains", "minContains", "maxContains", "minProperties", "maxProperties", "patternProperties",
           "propertyNames", "dependentRequired", "dependentSchemas", "dependencies", "not", "if", "then", "else",
           "unevaluatedProperties", "unevaluatedItems")
LLAMA_FORMAT = {
    "date": r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])$",
    "time": r"^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]{3})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$",
    "date-time": r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]{3})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
}
SUB_ONE = ("additionalProperties", "items", "not", "if", "then", "else", "contains", "propertyNames")
SUB_MAP = ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas")
SUB_LIST = ("anyOf", "oneOf", "allOf", "prefixItems", "items")


def walk(s, fn):
    """A copy of the schema with fn applied to every subschema, outermost first."""
    if not isinstance(s, dict): return s
    s = fn(copy.deepcopy(s))
    if not isinstance(s, dict): return s
    for k in SUB_ONE:
        if isinstance(s.get(k), dict): s[k] = walk(s[k], fn)
    for k in SUB_MAP:
        if isinstance(s.get(k), dict): s[k] = {n: walk(x, fn) for n, x in s[k].items()}
    for k in SUB_LIST:
        if isinstance(s.get(k), list): s[k] = [walk(x, fn) for x in s[k]]
    return s


def drop(keys, cond=lambda n: True):
    def f(n):
        if cond(n):
            for k in keys: n.pop(k, None)
        return n
    return f


def is_number_node(n): return n.get("type") == "number" or (isinstance(n.get("type"), list) and "number" in n["type"])


def close_objects(n):
    if isinstance(n.get("properties"), dict) and "additionalProperties" not in n and "$ref" not in n: n["additionalProperties"] = False
    return n


def anchor_pattern(n):
    p = n.get("pattern")
    if isinstance(p, str) and len(p) >= 2 and p[0] == "^" and p[-1] == "$": n["pattern"] = "^(?:" + p[1:-1] + ")$"
    return n


def pattern_wins(n):
    if isinstance(n.get("pattern"), str) and n["pattern"]:
        for k in ("minLength", "maxLength", "format"): n.pop(k, None)
    elif n.get("format") in LLAMA_FORMAT or (isinstance(n.get("format"), str) and re.fullmatch(r"uuid[1-5]", n["format"])):
        n.pop("minLength", None); n.pop("maxLength", None)
    return n


def llama_formats(n):
    f = n.get("format")
    if isinstance(f, str) and re.fullmatch(r"uuid[1-5]", f): f = "uuid"
    if f in LLAMA_FORMAT and not n.get("pattern"): n.pop("format"); n["pattern"] = LLAMA_FORMAT[f]
    return n


def close_tuples(n):
    for k in ("prefixItems", "items"):
        if isinstance(n.get(k), list):
            if k == "prefixItems" and "items" in n and not isinstance(n["items"], list): continue
            size = len(n[k]); n["minItems"] = size; n["maxItems"] = size
            n.pop("additionalItems", None)
    return n


def declared_required(n):
    if isinstance(n.get("required"), list):
        props = n.get("properties") if isinstance(n.get("properties"), dict) else {}
        n["required"] = [r_ for r_ in n["required"] if r_ in props]
    return n


def one_of_as_any_of(n):
    if "oneOf" in n: n["anyOf"] = n.pop("oneOf")
    return n


def only_first_keyword(n):
    keep = next((k for k in ("$ref", "oneOf", "anyOf") if k in n), None)
    if keep is None and not isinstance(n.get("type"), list):
        keep = next((k for k in ("const", "enum") if k in n), None)
    if keep: return {k: v for k, v in n.items() if k in (keep, "$defs", "definitions")}
    return n


def merge_all_of(root):
    def f(n):
        k = kind_of(n) if not any(x in n for x in ("$ref", "oneOf", "anyOf", "const", "enum")) and not isinstance(n.get("type"), list) else None
        has_props = "properties" in n or ("additionalProperties" in n and n["additionalProperties"] is not True)
        if k == "allOf" or ("allOf" in n and (n.get("type") == "string" or (n.get("type") == "object" and not has_props))):
            props, req, enums = {}, [], {}
            for c in n["allOf"]:
                opt = isinstance(c, dict) and "anyOf" in c
                for alt in (c["anyOf"] if opt else [c]):
                    alt = deref(root, alt)
                    if isinstance(alt, dict) and isinstance(alt.get("properties"), dict):
                        props.update(alt["properties"]); req += [] if opt else list(alt["properties"])
                    if isinstance(alt, dict) and isinstance(alt.get("enum"), list):
                        for v in alt["enum"]: enums[json.dumps(v)] = enums.get(json.dumps(v), 0) + 1
            both = [json.loads(v) for v, m in enums.items() if m == len(n["allOf"])]
            out = {"enum": both} if both else {"type": "object", "properties": props, "required": req, "additionalProperties": False}
            for d in ("$defs", "definitions"):
                if d in n: out[d] = n[d]
            return out
        return n
    return f


def empty_name_root(s):
    props = s.get("properties") if isinstance(s, dict) and isinstance(s.get("properties"), dict) else {}
    if "" in props and not any(k in s for k in ("$ref", "oneOf", "anyOf")):
        out = dict(props[""])
        for d in ("$defs", "definitions"):
            if d in s: out[d] = s[d]
        return out
    return s


def ecma_dot(n):
    p = n.get("pattern")
    if isinstance(p, str):
        out, i, cls = [], 0, False
        while i < len(p):
            ch = p[i]
            if ch == "\\" and i + 1 < len(p): out.append(p[i:i + 2]); i += 2; continue
            if ch == "[": cls = True
            elif ch == "]": cls = False
            out.append("[^\\n\\r]" if ch == "." and not cls else ch); i += 1
        n["pattern"] = "".join(out)
    return n


def no_prefix_keys(n):
    props = n.get("properties") if isinstance(n.get("properties"), dict) else {}
    if props and n.get("additionalProperties", False) is not False:
        pre = sorted({k[:j] for k in props for j in range(1, len(k)) if k[:j] not in props})
        if pre: n["propertyNames"] = {"not": {"enum": pre}}
    return n


def infer_type(n):
    if "type" not in n and not any(k in n for k in ("$ref", "oneOf", "anyOf", "const", "enum")):
        k = kind_of(n)
        if k in ("object", "array", "string"): n["type"] = k
    return n


REINTERPRET = {  # id: (explanation, schema -> schema)
    "pattern-dot-dialect": ("a pattern's `.` excludes \\n and \\r, as in ECMA-262 (JSON Schema's regex dialect); jsonschema "
                            "uses Python's re, whose `.` matches \\r", lambda s: walk(s, ecma_dot)),
    "additional-key-prefix": ("an additional key that is a proper prefix of a declared name is refused (llama.cpp's trie "
                              "rule for other keys wants one more character there)", lambda s: walk(s, no_prefix_keys)),
    "type-inferred": ("without `type`, llama.cpp takes the type from the keywords (properties/additionalProperties: object; "
                      "items/prefixItems: array; pattern/minLength/maxLength/format: string), where JSON Schema applies them "
                      "only to values of that type and accepts every other value", lambda s: walk(s, infer_type)),
    "additional-properties-default": ("once `properties` is given, llama.cpp allows no other key unless additionalProperties says so "
                                      "(JSON Schema's default allows any)", lambda s: walk(s, close_objects)),
    "number-bounds-ignored": ("minimum/maximum/exclusive*/multipleOf are not enforced on a number", lambda s: walk(s, drop(IGNORED_NUMBER, is_number_node))),
    "keyword-ignored": ("a keyword llama.cpp does not convert (" + ", ".join(IGNORED) + ") constrains nothing",
                        lambda s: walk(s, drop(IGNORED))),
    "pattern-anchored-whole": ("a pattern ^…$ is read as a whole-string match of what lies between (JSON Schema searches: "
                               "^A|B$ is (^A)|(B$))", lambda s: walk(s, anchor_pattern)),
    "pattern-widened": ("a pattern llama.cpp cannot translate (unanchored, lookaround, \\d \\w \\s, …) accepts any string",
                        lambda s: walk(s, lambda n: (n.pop("pattern") if isinstance(n.get("pattern"), str) and n["pattern"] in WIDENED else None, n)[1])),
    "pattern-wins": ("beside a pattern, minLength/maxLength/format are ignored; beside a format, minLength/maxLength are",
                     lambda s: walk(s, pattern_wins)),
    "format-rules": ("formats are llama.cpp's own rules, not RFC 3339's: any day 01-31 in any month, fractional seconds of "
                     "exactly 3 digits, upper-case Z and T only", lambda s: walk(s, llama_formats)),
    "all-of-merged": ("allOf merges its object components into one closed object, every property of a component outside "
                      "anyOf required (or intersects enums)", lambda s: walk(s, merge_all_of(s))),
    "tuple-exact": ("a tuple (prefixItems, or a list under items) takes exactly its listed items; JSON Schema allows fewer and, "
                    "without items: false, more", lambda s: walk(s, close_tuples)),
    "required-undeclared-ignored": ("a required name with no property is not required", lambda s: walk(s, declared_required)),
    "one-of-as-any-of": ("oneOf is read as anyOf (no exclusivity)", lambda s: walk(s, one_of_as_any_of)),
    "keywords-beside-ignored": ("beside $ref, anyOf/oneOf, const or enum every other keyword is ignored", lambda s: walk(s, only_first_keyword)),
    "empty-name-is-root": ("a top-level property named \"\" gets the rule name `root`, so the grammar's start symbol is that "
                           "property's schema (llama.cpp's name clash)", empty_name_root),
}

# Which patterns the port widens to any string: its grammar for {"type": "string", "pattern": p} is `root ::= string`.
pats = sorted({p for _, t in schemas for p in re.findall(r'"pattern":\s*("(?:[^"\\]|\\.)*")', t)})
pats = [json.loads(p) for p in pats]
GBNF = ACCEPTS.replace("json_schema_accepts", "json_schema_to_gbnf")
g = json.loads(subprocess.run([GBNF], input=json.dumps([json.dumps({"type": "string", "pattern": p}) for p in pats]),
                              capture_output=True, text=True, check=True).stdout)
WIDENED = {p for p, o in zip(pats, g) if "root ::= string\n" in o.get("ok", "")}  # llama.cpp's fallback: the plain string rule


def same_value(t, orig, rid):
    if rid == "pattern-raw-chars":  # raw characters are not JSON by design: the same string, its escapes decoded
        return json.loads(orig) is not None and tokens(t) is None or respell_strings(orig, raw_chars) == t
    try: return json.loads(t) == json.loads(orig)
    except json.JSONDecodeError: return False


def jsonschema_verdict(schema, text):
    try: return validator(schema).is_valid(json.loads(text))
    except Exception: return None


# every disagreement, and every single or paired cause that could undo it
dis = [(c, i) for c in live for i in c["instances"] if i["grammar"] != i["valid"]]
queries = {}  # schema text -> [texts] (respelled texts the grammar must judge)
cands = []
for c, i in dis:
    schema = json.loads(c["schema"])
    opts = []
    for rid, (_, fn) in {**RESPELL, **LIMIT}.items():
        for t in fn(i["text"], schema):
            if t and t != i["text"]: opts.append(((rid,), t, "limit" if rid in LIMIT else None))
    for rid, (_, fn) in REINTERPRET.items():
        opts.append(((rid,), i["text"], fn(schema)))
    for a in RESPELL:
        for b in REINTERPRET:
            for t in RESPELL[a][1](i["text"], schema):
                if t and t != i["text"]: opts.append(((a, b), t, REINTERPRET[b][1](schema)))
    for a in REINTERPRET:
        for b in REINTERPRET:
            if a < b: opts.append(((a, b), i["text"], REINTERPRET[b][1](REINTERPRET[a][1](schema))))
    for a in RESPELL:
        for b in RESPELL:
            if a < b:
                for t in RESPELL[a][1](i["text"], schema):
                    for t2 in (RESPELL[b][1](t, schema) if t else []):
                        if t2 and t2 != i["text"]: opts.append(((a, b), t2, None))
    for _, t, _ in opts: queries.setdefault(c["schema"], set()).add(t)
    cands.append((c, i, opts))
qs = [(s, sorted(ts)) for s, ts in queries.items()]
judged = {}
for (s, ts), vs in zip(qs, accepts(qs)):
    for t, v in zip(ts, vs): judged[(s, t)] = v

unexplained = []
for c, i, opts in cands:
    schema = json.loads(c["schema"])
    for size in (1, 2):
        hit = None
        for ids, t, rs in opts:
            if len(ids) != size: continue
            grammar_says = judged[(c["schema"], t)]
            if rs is None:  # a respelling: same value, and the grammar now agrees with jsonschema's verdict on it
                if grammar_says != i["valid"] or not same_value(t, i["text"], ids[-1] if ids[-1] in RESPELL else ids[0]): continue
            elif rs == "limit":  # a limit: a minimally changed value, same verdict, and the grammar agrees
                if grammar_says != i["valid"] or jsonschema_verdict(schema, t) != i["valid"]: continue
            else:  # a reinterpretation: jsonschema, reading the schema as llama.cpp does, agrees with the grammar
                json_says = jsonschema_verdict(rs, t)
                if json_says is None or grammar_says is None or grammar_says != json_says: continue
            hit = (ids, t)
            break
        if hit: break
    if not hit:
        unexplained.append((c, i)); continue
    i["cause"] = "+".join(hit[0])
    if hit[1] != i["text"]: i["respelled"] = hit[1]

n = sum(len(c["instances"]) for c in live)
print(f"{len(cases)} schemas ({len(cases) - len(live)} skipped), {n} instances, {n - len(dis)} agree, {len(dis)} disagree, {len(unexplained)} unexplained")
for c in cases:
    if "skipped" in c: print("SKIPPED", c["name"], "-", c["skipped"])
count = {}
for c, i in dis:
    if "cause" in i: count[i["cause"]] = count.get(i["cause"], 0) + 1
for k, m in sorted(count.items(), key=lambda x: -x[1]): print(f"{m:5d}  {k}")
pos = sum(i["valid"] for c in live for i in c["instances"])
print(f"jsonschema: {pos} valid, {n - pos} invalid; grammar agrees on {sum(i['valid'] for c in live for i in c['instances'] if i['grammar'] == i['valid'])} "
      f"valid and {sum(not i['valid'] for c in live for i in c['instances'] if i['grammar'] == i['valid'])} invalid")
for c, i in unexplained:
    print(f"UNEXPLAINED [{c['name']}] valid={i['valid']} grammar={i['grammar']} {i['how']}: {i['text'][:200]!r}")
if unexplained: sys.exit(1)

for c in live:
    for i in c["instances"]: del i["grammar"]
causes = {k: v[0] for k, v in {**RESPELL, **LIMIT, **REINTERPRET}.items()}
body = json.dumps({"source": "make_semantic.py; jsonschema " + version("jsonschema") + ", rfc3339-validator " + version("rfc3339-validator"),
                   "causes": causes, "schemas": cases}, indent=0, ensure_ascii=False).encode()
with open(sys.argv[2], "wb") as f: f.write(gzip.compress(body, mtime=0))
