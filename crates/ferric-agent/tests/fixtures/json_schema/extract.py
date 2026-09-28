"""llama.cpp's JSON-Schema-to-grammar tests -> one JSON fixture, case for case.

  tests/test-json-schema-to-grammar.cpp: every test({SUCCESS|FAILURE, name, schema, expected_grammar}), the
      expected grammar trimmed exactly as that harness's trim() does before it compares; plus its two extra
      blocks (a pre-parsed document converts the same as the JSON; a property's $ref node, converted alone
      through build_grammar, names the ref rule).
  tests/test-grammar-integration.cpp: every test_schema(desc, schema, passing, failing) - the grammar
      converted from the schema must match every passing string and no failing one.

usage: python3 extract.py <llama.cpp commit> <out.json>   (run beside copies of both .cpp files)
"""
import json, re, sys

ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "?": "?"}


class Src:
    def __init__(self, path):
        self.s = open(path, encoding="utf-8").read()

    def skip_ws(self, p):
        s = self.s
        while True:
            while p < len(s) and s[p] in " \t\r\n": p += 1
            if s.startswith("//", p): p = s.index("\n", p)
            elif s.startswith("/*", p): p = s.index("*/", p) + 2
            else: return p

    def c_string(self, p):  # p at the opening quote; returns (text, p after)
        s = self.s
        assert s[p] == '"'; p += 1; out = bytearray()
        while s[p] != '"':
            if s[p] == "\\":
                c = s[p + 1]
                if c == "x":
                    m = re.match(r"[0-9a-fA-F]+", s[p + 2:]); out.append(int(m.group(0), 16) & 0xFF); p += 2 + len(m.group(0)); continue
                if c == "u":
                    out += chr(int(s[p + 2:p + 6], 16)).encode(); p += 6; continue
                out += ESC[c].encode(); p += 2; continue
            out += s[p].encode(); p += 1
        return out.decode("utf-8"), p + 1

    def raw_string(self, p):  # R"delim( ... )delim"
        s = self.s
        assert s.startswith('R"', p); q = s.index("(", p); delim = s[p + 2:q]; end = s.index(")" + delim + '"', q)
        return s[q + 1:end], end + len(delim) + 2

    def expr(self, p):  # a string expression: adjacent literals concatenate
        out = None
        while True:
            p = self.skip_ws(p)
            if self.s.startswith('R"', p): t, p = self.raw_string(p)
            elif self.s[p] == '"': t, p = self.c_string(p)
            else: break
            out = t if out is None else out + t
        assert out is not None, self.s[p:p + 60]
        return out, p

    def expect(self, p, ch):
        p = self.skip_ws(p)
        assert self.s[p] == ch, (ch, self.s[p:p + 60])
        return p + 1

    def str_list(self, p):
        p = self.expect(p, "{"); out = []
        while True:
            p = self.skip_ws(p)
            if self.s[p] == "}": return out, p + 1
            e, p = self.expr(p); out.append(e); p = self.skip_ws(p)
            if self.s[p] == ",": p += 1


def trim(s):  # the harness's trim(): strip the ends, then leading blanks at the start of every line
    s = s.strip(" \n\r\t")
    return re.sub(r"(^|\n)[ \t]+", r"\1", s)


conv = Src("test-json-schema-to-grammar.cpp")
cases = []
body = conv.s[conv.s.index("static void test_all("):conv.s.index("int main()")]
start = conv.s.index("static void test_all(")
for m in re.finditer(r"\btest\(\{", body):
    p = start + m.end()
    p = conv.skip_ws(p)
    status = re.match(r"SUCCESS|FAILURE", conv.s[p:]).group(0); p += len(status)
    p = conv.expect(p, ",")
    name, p = conv.expr(p); p = conv.expect(p, ",")
    schema, p = conv.expr(p); p = conv.expect(p, ",")
    expected, p = conv.expr(p)
    p = conv.expect(p, "}"); p = conv.expect(p, ")")
    json.loads(schema)  # every schema is JSON
    cases.append({"name": name, "status": status, "schema": schema, "expected": trim(expected) if status == "SUCCESS" else ""})

main = conv.s[conv.s.index("int main()"):]
mstart = conv.s.index("int main()")
extra = []
# a document parsed up front gives the same grammar as the JSON
m = re.search(r'fprintf\(stderr, "- parsed document\\n"\);\s*auto schema = common_json::parse\(', main)
schema, _ = conv.expr(mstart + m.end())
extra.append({"name": "parsed document", "kind": "document_equals_json", "schema": schema})
# a property node carries its $ref target, so its grammar names the ref rule
m = re.search(r'fprintf\(stderr, "- sub-schema \$ref\\n"\);\s*auto parameters = common_json::parse\(', main)
params, p = conv.expr(mstart + m.end())
m2 = re.search(r'TestCase tc \{\s*SUCCESS,\s*', conv.s[p:])
p += m2.end()
name, p = conv.expr(p); p = conv.expect(p, ",")
_empty, p = conv.expr(p); p = conv.expect(p, ",")
expected, p = conv.expr(p)
assert "properties.at(0)" in conv.s[p:p + 600] and 'add_schema("root", *item.schema)' in conv.s[p:p + 600]
extra.append({"name": name, "kind": "property_0_as_root", "schema": params, "expected": trim(expected)})

integ = Src("test-grammar-integration.cpp")
schema_cases = []
for m in re.finditer(r"\btest_schema\(", integ.s):
    if integ.s[m.start() - 5:m.start()] == "void ": continue
    p = m.end()
    desc, p = integ.expr(p); p = integ.expect(p, ",")
    schema, p = integ.expr(p); p = integ.expect(p, ",")
    ok, p = integ.str_list(p); p = integ.expect(p, ",")
    bad, p = integ.str_list(p)
    json.loads(schema)
    schema_cases.append({"desc": desc, "schema": schema, "pass": ok, "fail": bad})

commit = sys.argv[1]
json.dump({
    "source": "llama.cpp @ " + commit + ": tests/test-json-schema-to-grammar.cpp (cases, extra), tests/test-grammar-integration.cpp (schema_matching)",
    "cases": cases, "extra": extra, "schema_matching": schema_cases,
}, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(cases), "conversion cases (", sum(c["status"] == "SUCCESS" for c in cases), "success,", sum(c["status"] == "FAILURE" for c in cases), "failure ),",
      len(extra), "extra,", len(schema_cases), "schema matching cases with", sum(len(c["pass"]) for c in schema_cases), "passing and",
      sum(len(c["fail"]) for c in schema_cases), "failing strings")
