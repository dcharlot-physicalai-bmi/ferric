"""A differential corpus: schemas converted by llama.cpp's own converter (built by reference/build.sh), its
output kept as the expectation for ferric_agent::json_schema, byte for byte (tests/json_schema.rs).

The authors' tests pin 81 schemas. These add the rest of the surface: integer bounds across magnitudes and
signs (llama.cpp's digit-range builder), fractional and exclusive bounds, string lengths and formats, ~70
regex patterns (its own translator, including every refusal and every fallback), property names that stress
rule naming and the additional-properties key rule, arrays and tuples, combinators and $refs, const/enum
literals (nlohmann's number printing, string escaping), the schemas it refuses, and seeded random
compositions of all of it.

usage: python3 make_differential.py <reference js2g binary> differential.json.gz [<ferric json_schema_to_gbnf binary>]
Build the reference with `reference/build.sh <dir>`, the port with
`cargo build -p ferric-agent --release --example json_schema_to_gbnf`. With the third argument it also runs the
port and prints every difference.
"""
import json, random, subprocess, sys

here = __file__.rsplit("/", 1)[0]
authors = json.load(open(f"{here}/llama_cpp_cases.json"))
schemas = []  # (group, schema text)


def add(group, s):
    schemas.append((group, s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)))


for c in authors["cases"]: add("authors", c["schema"])
for c in authors["schema_matching"]: add("authors", c["schema"])

# --- integers: llama.cpp's build_min_max_int ---------------------------------------------------------------
V = [0, 1, 2, 3, 5, 8, 9, 10, 11, 15, 19, 20, 25, 29, 30, 42, 89, 90, 99, 100, 101, 109, 110, 123, 199, 200, 255, 256, 300, 456, 909, 990,
     999, 1000, 1001, 1234, 5000, 9999, 10000, 12345, 65535, 99999, 100000, 123456, 999999, 1000000, 2147483647, 4294967295,
     900719925474091, 9007199254740991, 1234567890123456, 9999999999999999, 10000000000000000, 99999999999999999,
     123456789012345678, 9223372036854775806, 9223372036854775807]
for v in V:
    add("int-min", {"type": "integer", "minimum": v})
    add("int-max", {"type": "integer", "maximum": v})
    if v:
        add("int-min", {"type": "integer", "minimum": -v})
        add("int-max", {"type": "integer", "maximum": -v})
add("int-edge", {"type": "integer", "minimum": -9223372036854775808})
add("int-edge", {"type": "integer", "maximum": -9223372036854775808})
add("int-edge", {"type": "integer", "minimum": 9223372036854775808})
add("int-edge", {"type": "integer", "minimum": -9223372036854775807})
add("int-edge", {"type": "integer", "minimum": -9223372036854775807, "maximum": 5})
r = random.Random(1)
for _ in range(160):
    def bound():
        d = r.choice([1, 1, 2, 2, 3, 3, 4, 5, 6, 8, 12, 16, 17])
        v = r.randrange(10 ** (d - 1) if d > 1 else 0, 10 ** d)
        return -v if r.random() < 0.3 else v
    a, b = bound(), bound()
    if r.random() < 0.9: a, b = min(a, b), max(a, b)
    add("int-range", {"type": "integer", "minimum": a, "maximum": b})
for a, b in [(0, 0), (5, 5), (-5, -5), (-1, 0), (0, 1), (-1, 1), (-10, 10), (-99, 99), (-100, -1), (-999, -100), (1, 9), (10, 99),
             (100, 999), (0, 9), (0, 10), (9, 10), (99, 100), (1, 1000000), (-1000000, 1), (123, 456), (-456, -123), (19, 21), (7, 3)]:
    add("int-range", {"type": "integer", "minimum": a, "maximum": b})
for s in [{"exclusiveMinimum": 0}, {"exclusiveMaximum": 0}, {"exclusiveMinimum": -1, "exclusiveMaximum": 1}, {"minimum": 1.5},
          {"maximum": 10.2}, {"exclusiveMinimum": 0.5}, {"exclusiveMaximum": 99.9}, {"minimum": -2.5, "maximum": -0.5},
          {"minimum": 3, "exclusiveMinimum": 10}, {"maximum": 3, "exclusiveMaximum": 1}, {"minimum": 1e3}, {"maximum": 1e3},
          {"minimum": "3"}, {"maximum": None}, {"minimum": True}, {"exclusiveMinimum": 1e300}]:
    add("int-bounds", {"type": "integer", **s})
add("number-bounds", {"type": "number", "minimum": 0, "maximum": 1, "exclusiveMinimum": 0, "multipleOf": 0.5})

# --- strings ---------------------------------------------------------------------------------------------
L = [None, 0, 1, 2, 3, 5, 10, 100, 2000, 2001]
for a in L:
    for b in L:
        s = {"type": "string"}
        if a is not None: s["minLength"] = a
        if b is not None: s["maxLength"] = b
        add("string-length", s)
for s in [{"minLength": -1}, {"minLength": 1.5}, {"maxLength": "3"}, {"maxLength": 4294967299}, {"minLength": 3}, {"maxLength": 0}]:
    add("string-length", {"type": "string", **s})
    add("string-length", s)
for f in ["date", "time", "date-time", "uuid", "uuid1", "uuid5", "uuid6", "uuid0", "UUID", "email", "uri", ""]:
    add("format", {"type": "string", "format": f})
    add("format", {"format": f})
add("format", {"type": "string", "format": 5})
add("format", {"type": "string", "format": "date", "pattern": "^[0-9]+$"})
add("format", {"type": "string", "format": "uuid", "minLength": 3})
add("format", {"type": "string", "pattern": "", "minLength": 2})

# --- patterns: llama.cpp's regex translation, its refusals and its fallbacks -------------------------------
P = ["^abc?d*efg+(hij)?kl$", "^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$", "^\"$", "^A|B|C|D$", "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$",
     "^(?:foo|bar)baz$", "^(?:(?:ab)+c)?d$", "[0-9]+", "^[0-9]{3}\\w$", "^[a-z\\-]+$", "^a\\-b$", "^(a$", "^(?=a)a$",
     "^[a-zA-Z0-9_-]*$", "^a\\^\\$\\.\\[\\]\\(\\)\\|\\{\\}\\*\\+\\?b$", "^[a-z]+$", "^\\d+$", "^(foo|bar)+$", "^a{2}$", "^a{2,}$",
     "^a{,3}$", "^(ab){2,3}$", "^(ab){2,3}(ab){2,3}$", "^([a-z]{2}){3}$", "^[^0-9]*$", "^\\.$", "^.*$", "^.+x.?$", "^(a|b|)$",
     "^héllo$", "^日本語$", "^\\x41$", "^\\x4$", "^\\u00e9+$", "^\\U0001F600$", "^a\\nb$", "^[\\n\\t]+$", "^\\\\$", "^\"quoted\"$",
     "^*a$", "^a{x}$", "^a{1,2,3}$", "^a{$", "^[a$", "^a)$", "^a\\$", "^(?<n>a)$", "^(?i)a$", "^a$b$", "^ab.+c$", "^[a-z]{3}\\d$",
     "^(a(b(c(d)))){2}$", "^(a|bc|def)*$", "^[\\[\\]]$", "^[\\d]$", "^\\s*$", "^a{0}$", "^a{0,0}$", "^a{1}b{1,1}$", "^a{3,2}$",
     "^a{-1}$", "^a{ 2 }$", "^a{2, 3}$", "^a{99999999999}$", "^$", "^", "$", "^^$", "^[]]$", "^]$", "^}$", "^a|$", "^(|a)$",
     "^a.b$", "^ab.$", "^a*b+c?$", "^(a)(b)(c)$", "^((a))$", "^(ab)+$", "^\\t\\r$", "^\\-$", "^[a\\]]$", "^(" * 101 + "a" + ")" * 101 + "$",
     "^(" * 99 + "a" + ")" * 99 + "$", "^é+$", "^aé{2}$", "^日本+x$", "^[é-ë]+$", "^(é)+$"]
for p in P:
    add("pattern", {"type": "string", "pattern": p})
add("pattern", {"type": "object", "properties": {"a": {"type": "string", "pattern": "^\\d+$"}, "b": {"type": "string", "pattern": "^[0-9]{2}$"}},
                "required": ["a", "b"]})
add("pattern", {"type": "array", "items": {"type": "string", "pattern": "^[a-f0-9]{8}$"}, "minItems": 1})
add("pattern", {"type": "string", "pattern": 5})

# --- objects -----------------------------------------------------------------------------------------------
NAMES = ["a", "b", "c", "name", "age", "id", "tags", "x-y", "a b", "a_b", "", "root", "number", "string", "value", "char", "space",
         "dot", "é", "日本", "aa", "ab", "abc", "a1", "1a", "additional", "item", "tuple-0", "alternative-0", "ref"]
for n in NAMES:
    add("object-names", {"type": "object", "properties": {n: {"type": "integer"}}, "required": [n]})
    add("object-names", {"type": "object", "properties": {n: {"type": "string"}, "z": {"type": "boolean"}}, "additionalProperties": {"type": "integer"}})
for n in ["a\"b", "a\\b", "a\nb", "a]b", "a-b", "]", "\\", "-", "^", "a[b"]:
    add("object-names", {"type": "object", "properties": {n: {"type": "integer"}}, "required": [n]})
    add("object-names", {"type": "object", "properties": {n: {"type": "integer"}, "q": {"type": "null"}}, "additionalProperties": True})
for props, req, addl in [
    (["a"], [], None), (["a"], ["a"], None), (["a", "b"], ["b"], False), (["a", "b", "c", "d", "e"], [], False),
    (["a", "b", "c", "d", "e"], ["c"], {"type": "string"}), (["a", "b", "c"], ["a", "b", "c"], True),
    ([f"p{i}" for i in range(12)], ["p3", "p7"], None), ([f"p{i}" for i in range(12)], [], True),
    (["x"], ["x", "y"], False), ([], ["x"], None), ([], [], False), ([], [], True), ([], [], {}), ([], [], {"type": "number"}),
    (["and", "also", "any"], ["any"], {"type": "number"}), (["ab", "ac", "b"], [], True), (["", "a"], [], {"type": "integer"})]:
    s = {"type": "object"}
    if props: s["properties"] = {p: {"type": "string"} for p in props}
    if req: s["required"] = req
    if addl is not None: s["additionalProperties"] = addl
    add("object", s)
    del s["type"]
    add("object", s)
add("object", {"type": "object", "properties": {"a": {"type": "object", "properties": {"a": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}, "required": ["a"]}}, "required": ["a"]})
add("object", {"properties": {"*": {"type": "integer"}}, "additionalProperties": {"type": "string"}})
add("object", {"type": "object", "properties": 5})
add("object", {"type": "object", "additionalProperties": 5})
add("object", {"type": "object", "required": "a", "properties": {"a": {}}})
add("object", {"type": "object", "properties": {"a": True}})
add("object", {"properties": {"a": {"type": "integer"}}, "patternProperties": {"^x": {}}, "minProperties": 1})

# --- arrays and tuples -------------------------------------------------------------------------------------
for mn in [None, 0, 1, 2, 3]:
    for mx in [None, 0, 1, 2, 5]:
        for items in [{"type": "integer"}, {}, None]:
            s = {"type": "array"}
            if items is not None: s["items"] = items
            if mn is not None: s["minItems"] = mn
            if mx is not None: s["maxItems"] = mx
            add("array", s)
for s in [{"items": [{"type": "string"}, {"type": "integer"}], "minItems": 5}, {"prefixItems": [{"type": "boolean"}], "items": {"type": "null"}},
          {"prefixItems": [], "type": "array"}, {"items": []}, {"items": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}},
          {"type": "array", "minItems": -1}, {"type": "array", "maxItems": 1.0}, {"type": "array", "items": True},
          {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}, "maxItems": 3},
          {"type": "array", "uniqueItems": True, "items": {"enum": [1, 2, 3]}}, {"type": "array", "contains": {"type": "integer"}}]:
    add("array", s)

# --- combinators and $refs -----------------------------------------------------------------------------------
for s in [
    {"anyOf": [{"type": "string"}, {"type": "integer"}]}, {"oneOf": [{"type": "null"}, {"type": "number"}], "anyOf": [{"type": "string"}]},
    {"anyOf": []}, {"anyOf": {}}, {"oneOf": [{"const": 1}, {"const": "1"}]}, {"type": ["string", "null"], "maxLength": 3},
    {"type": ["integer", "string"], "minimum": 5, "minLength": 2}, {"type": []}, {"type": ["kaboom"]}, {"type": [5]},
    {"type": ["object", "array"], "properties": {"a": {}}, "items": {"type": "integer"}},
    {"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}, {"properties": {"b": {"type": "string"}}}]},
    {"allOf": [{"enum": [1, 2, 3]}, {"enum": [2, 3, 4]}, {"enum": [3, 2]}]}, {"allOf": [{"enum": ["a", "a"]}]},
    {"allOf": [{"enum": [1]}, {"enum": [2]}]}, {"allOf": [{"type": "string"}, {"minLength": 2}]}, {"allOf": []},
    {"type": "object", "allOf": [{"properties": {"a": {"type": "integer"}}}, {"anyOf": [{"properties": {"b": {"type": "null"}}}, {"properties": {"c": {"type": "boolean"}}}]}]},
    {"type": "string", "allOf": [{"enum": ["x", "y"]}, {"enum": ["y", "z"]}]},
    {"$ref": "#/$defs/a", "$defs": {"a": {"type": "integer", "minimum": 3}}},
    {"$ref": "#/definitions/a", "definitions": {"a": {"$ref": "#/definitions/b"}, "b": {"type": "string"}}},
    {"type": "object", "properties": {"x": {"$ref": "#/$defs/p"}, "y": {"$ref": "#/$defs/p"}}, "$defs": {"p": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"]}}},
    {"$ref": "#/$defs/node", "$defs": {"node": {"type": "object", "properties": {"value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}}, "required": ["value"]}}},
    {"$ref": "#/$defs/node", "$defs": {"node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}, "leaf": {}}, "additionalProperties": False}}},
    {"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"},
    {"prefixItems": [{"type": "integer"}, {"$ref": "#/prefixItems/0"}]}, {"items": [{"type": "integer"}, {"$ref": "#/items/0"}]},
    {"anyOf": [{"$ref": "#/anyOf/1"}, {"type": "boolean"}]}, {"anyOf": [{"$ref": "#/anyOf/9"}, {"type": "boolean"}]},
    {"anyOf": [{"$ref": "#/anyOf/x"}, {"type": "boolean"}]}, {"anyOf": [{"$ref": "#/anyOf/-0"}, {"type": "null"}]},
    {"anyOf": [{"$ref": "#/anyOf/ 1"}, {"type": "null"}]}, {"anyOf": [{"$ref": "#/anyOf/1abc"}, {"type": "null"}]},
    {"$ref": "#/$defs/missing", "$defs": {}}, {"$ref": "https://example.com/schema.json"}, {"$ref": "other.json#/a"}, {"$ref": "#"},
    {"$ref": 5}, {"$ref": "#/$defs/a~1b", "$defs": {"a~1b": {"type": "null"}, "a/b": {"type": "boolean"}}},
    {"$ref": "#/$defs/x", "$defs": {"x": 5}}, {"$ref": "#/$defs/x/type", "$defs": {"x": {"type": "string"}}},
    {"allOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}], "$defs": {"a": {"properties": {"p": {"type": "integer"}}}, "b": {"enum": [1, 2]}}},
]:
    add("combinators", s)

# --- const / enum literals: nlohmann's dump() --------------------------------------------------------------
LIT = [0, 1, -1, 42, 123456789012345678, -9223372036854775808, 18446744073709551615, 1.5, 1.0, -1.0, 100.0, 0.1, 0.5, 3.14159, 1e15,
       1e16, 1e17, 1e20, 1e21, 1e100, 1e-4, 1e-5, 1.5e-5, 1e-7, 123456.789, 0.000123, 2.5e-300, 1.7976931348623157e308, 5e-324,
       9007199254740993.0, 0.30000000000000004, 1234567890.0987654, "", "x", "a\nb", "a\"b", "a\\b", "tab\there", "\u0001\u001f", "\u007f",
       "é", "日本", "😀", "</script>", "a/b", True, False, None, [], {}, [1, "a", None], {"k": [1, {"z": 2.5}], "a": "b"}]
add("literal", {"enum": LIT})
for v in LIT:
    add("literal", {"const": v})
r = random.Random(2)
fl = []
for _ in range(600):  # short decimals across magnitudes
    e = r.randrange(-30, 30)
    fl.append(float(f"{r.random():.{r.randrange(1, 18)}g}e{e}"))
import math, struct
while len(fl) < 1200:  # random bit patterns: 17-digit values, subnormals, extremes
    x = struct.unpack("<d", r.getrandbits(64).to_bytes(8, "little"))[0]
    if math.isfinite(x): fl.append(x)
for i in range(0, len(fl), 50):
    add("literal-floats", {"enum": fl[i:i + 50]})
add("literal", {"enum": []})
add("literal", {"enum": 5})
add("literal", {"type": "string", "const": 5})
add("literal", {"const": None, "enum": [1]})

# --- refused -------------------------------------------------------------------------------------------------
for s in ["[]", "5", "null", "true", "\"x\"", {"type": "kaboom"}, {"type": 123}, {"type": None}, {"type": "array", "items": False},
          {"type": "object", "additionalProperties": "yes"}, {"not": {"type": "integer"}}, {"if": {}, "then": {}, "else": {}}]:
    add("refused", s)


# --- seeded random compositions ------------------------------------------------------------------------------
def rnd(r, depth, defs):
    leaf = ["string", "integer", "number", "boolean", "null", "enum", "const", "pattern", "format", "any"]
    kinds = leaf + (["object", "object", "array", "anyOf", "allOf", "typelist", "tuple"] + (["ref"] if defs else []) if depth < 3 else [])
    k = r.choice(kinds)
    if k == "string":
        s = {"type": "string"}
        if r.random() < 0.4: s["minLength"] = r.randrange(0, 4)
        if r.random() < 0.4: s["maxLength"] = r.randrange(0, 8)
        return s
    if k == "integer":
        s = {"type": "integer"}
        if r.random() < 0.5: s["minimum"] = r.randrange(-500, 500)
        if r.random() < 0.5: s["maximum"] = r.randrange(-500, 5000)
        return s
    if k == "number": return {"type": "number"}
    if k == "boolean": return {"type": "boolean"}
    if k == "null": return {"type": "null"}
    if k == "enum": return {"enum": r.sample(["red", "green", "blue", 1, 2.5, None, True, "a b", "é"], r.randrange(1, 5))}
    if k == "const": return {"const": r.choice(["x", 7, -1.25, False, None, [1, 2], {"a": 1}])}
    if k == "pattern": return {"type": "string", "pattern": r.choice(P[:30])}
    if k == "format": return {"type": "string", "format": r.choice(["date", "time", "date-time", "uuid"])}
    if k == "any": return {}
    if k == "ref": return {"$ref": "#/$defs/" + r.choice(sorted(defs))}
    if k == "object":
        names = r.sample(NAMES, r.randrange(0, 5))
        s = {"type": "object", "properties": {n: rnd(r, depth + 1, defs) for n in names}}
        req = [n for n in names if r.random() < 0.5]
        if req: s["required"] = req
        x = r.random()
        if x < 0.4: s["additionalProperties"] = False
        elif x < 0.55: s["additionalProperties"] = True
        elif x < 0.7: s["additionalProperties"] = rnd(r, depth + 1, defs)
        return s
    if k == "array":
        s = {"type": "array", "items": rnd(r, depth + 1, defs)}
        if r.random() < 0.4: s["minItems"] = r.randrange(0, 3)
        if r.random() < 0.4: s["maxItems"] = r.randrange(0, 6)
        return s
    if k == "anyOf": return {r.choice(["anyOf", "oneOf"]): [rnd(r, depth + 1, defs) for _ in range(r.randrange(1, 4))]}
    if k == "allOf": return {"allOf": [rnd(r, depth + 1, defs) for _ in range(r.randrange(1, 4))]}
    if k == "typelist": return {"type": r.sample(["string", "integer", "null", "boolean", "number", "array", "object"], r.randrange(1, 4))}
    if k == "tuple": return {r.choice(["items", "prefixItems"]): [rnd(r, depth + 1, defs) for _ in range(r.randrange(1, 4))]}


for seed in range(350):
    r = random.Random(1000 + seed)
    defs = {f"d{i}": rnd(r, 3, {}) for i in range(r.randrange(0, 3))}  # leaf targets, so no $ref cycle through allOf
    s = rnd(r, 0, defs)
    if defs and isinstance(s, dict): s["$defs"] = defs
    add("random", s)

texts = [t for _, t in schemas]
ref = json.loads(subprocess.run([sys.argv[1]], input=json.dumps(texts), capture_output=True, text=True, check=True).stdout)
cases = []
for (group, t), o in zip(schemas, ref):
    c = {"group": group, "schema": t}
    c.update(o)
    if "ok" in o and "�" in o["ok"] and "�" not in t:
        # llama.cpp printed invalid UTF-8 (its byte-wise regex reader split a character); the port reads per
        # code point, so this case documents the difference instead of pinning llama.cpp's bytes.
        c["deviation"] = "llama.cpp splits a multi-byte character in front of a quantifier; the port keeps it whole"
    cases.append(c)
import gzip
body = json.dumps({"source": "llama.cpp @ 4da6337767f9 common/json-schema-to-grammar.cpp + json-schema.cpp, built by reference/build.sh",
                   "cases": cases}, indent=0, ensure_ascii=False).encode()
with open(sys.argv[2], "wb") as f:  # gzip, no timestamp: the same corpus gives the same bytes
    f.write(gzip.compress(body, mtime=0))
groups = {}
for c in cases: groups.setdefault(c["group"], [0, 0]); groups[c["group"]][0 if "ok" in c else 1] += 1
print(len(cases), "schemas:", ", ".join(f"{g} {a}+{b}" for g, (a, b) in groups.items()), "(grammar+error)")
print(sum("deviation" in c for c in cases), "documented deviations")

if len(sys.argv) > 3:
    mine = json.loads(subprocess.run([sys.argv[3]], input=json.dumps(texts), capture_output=True, text=True, check=True).stdout)
    diff = 0
    for c, m in zip(cases, mine):
        if "deviation" in c: continue
        if c.get("ok") != m.get("ok") or c.get("err") != m.get("err"):
            diff += 1
            print(f"--- DIFF [{c['group']}] {c['schema'][:200]}\n  llama.cpp: {json.dumps({k: c[k] for k in ('ok', 'err') if k in c})[:600]}\n  ferric:    {json.dumps(m)[:600]}")
    print(diff, "differences")
