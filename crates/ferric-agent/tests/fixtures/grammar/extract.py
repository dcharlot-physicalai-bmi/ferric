"""llama.cpp tests/test-grammar-integration.cpp -> JSON fixture: every test_grammar(desc, grammar, passing, failing)
and every test_build_grammar_fails(grammar). Strings are sequences of {"text"} / {"token"} segments."""
import json, re, sys
src = open("test-grammar-integration.cpp", encoding="utf-8").read()
i = 0
def skip_ws(p):
    while True:
        while p < len(src) and src[p] in " \t\r\n": p += 1
        if src.startswith("//", p): p = src.index("\n", p)
        elif src.startswith("/*", p): p = src.index("*/", p) + 2
        else: return p
ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "?": "?"}
def c_string(p):  # p at opening quote; returns (bytes, p after)
    assert src[p] == '"'; p += 1; out = bytearray()
    while src[p] != '"':
        if src[p] == "\\":
            c = src[p + 1]
            if c == "x":
                m = re.match(r"[0-9a-fA-F]+", src[p + 2:]); out.append(int(m.group(0), 16) & 0xFF); p += 2 + len(m.group(0)); continue
            if c == "u":
                out += chr(int(src[p + 2:p + 6], 16)).encode(); p += 6; continue
            if c == "U":
                out += chr(int(src[p + 2:p + 10], 16)).encode(); p += 10; continue
            if c in "01234567" and c != "0" or (c == "0" and src[p + 2] in "01234567"):
                m = re.match(r"[0-7]{1,3}", src[p + 1:]); out.append(int(m.group(0), 8)); p += 1 + len(m.group(0)); continue
            out += ESC[c].encode(); p += 2; continue
        out += src[p].encode(); p += 1
    return bytes(out), p + 1
def raw_string(p):  # R"delim( ... )delim"
    assert src.startswith('R"', p); q = src.index("(", p); delim = src[p + 2:q]; end = src.index(")" + delim + '"', q)
    return src[q + 1:end].encode(), end + len(delim) + 2
def expr(p):  # string expression: literals / raw / token(N) joined by +; adjacent literals concatenate
    segs = []
    while True:
        p = skip_ws(p)
        if src.startswith('R"', p): s, p = raw_string(p); segs.append({"text": s.decode("utf-8", "surrogateescape")})
        elif src[p] == '"': s, p = c_string(p); segs.append({"text": s.decode("utf-8", "surrogateescape")})
        elif src.startswith("token(", p):
            q = src.index(")", p); segs.append({"token": int(src[p + 6:q])}); p = q + 1
        else: break
        p = skip_ws(p)
        if src[p] == "+": p += 1; continue
        if src[p] in '"R' and (src[p] == '"' or src.startswith('R"', p)): continue
        break
    merged = []
    for s in segs:
        if merged and "text" in s and "text" in merged[-1]: merged[-1]["text"] += s["text"]
        else: merged.append(dict(s))
    return merged, p
def str_list(p):
    p = skip_ws(p); assert src[p] == "{", src[p:p + 40]; p += 1; out = []
    while True:
        p = skip_ws(p)
        if src[p] == "}": return out, p + 1
        e, p = expr(p); out.append(e); p = skip_ws(p)
        if src[p] == ",": p += 1
cases, fails = [], []
for m in re.finditer(r"\btest_grammar\(", src):
    if src[m.start() - 5:m.start()] == "void ": continue
    p = m.end(); d, p = expr(p); p = skip_ws(p) + 1; g, p = expr(p); p = skip_ws(p) + 1
    ok, p = str_list(p); p = skip_ws(p) + 1; bad, p = str_list(p)
    cases.append({"desc": d[0]["text"], "grammar": g[0]["text"], "pass": ok, "fail": bad})
def var_in(fn, name):  # the raw string bound to `const std::string name` inside function `fn`
    body = src[src.index("static void " + fn + "("):]
    m = re.search(r"const std::string " + name + r" =\s*", body)
    g, _ = expr(src.index(body[m.start():m.end()], src.index("static void " + fn + "(")) + (m.end() - m.start()))
    return g[0]["text"]
# (function, variable, root symbol, builds?) — what each test asserts
for fn, var, root, builds in [
    ("test_failure_missing_root", "grammar_str", "root", False),
    ("test_failure_missing_reference", "grammar_str", "root", False),
    ("test_failure_left_recursion", "simple_str", "root", False),
    ("test_failure_left_recursion", "medium_str", "root", False),
    ("test_failure_left_recursion", "hard_str", "root", False),
    ("test_failure_left_recursion", "hardest_str", "root", False),
    ("test_failure_missing_root_symbol", "grammar_str", "nonexistent", False),
    ("test_custom_root_symbol_check", "custom_root_grammar_str", "root", False),
    ("test_custom_root_symbol_check", "custom_root_grammar_str", "foobar", True)]:
    fails.append({"test": fn + "/" + var, "grammar": var_in(fn, var), "root": root, "builds": builds})
json.dump({"source": "llama.cpp tests/test-grammar-integration.cpp @ " + sys.argv[1], "cases": cases, "build_fails": fails}, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(cases), "grammar cases,", sum(len(c["pass"]) for c in cases), "passing,", sum(len(c["fail"]) for c in cases), "failing strings,", len(fails), "build failures")
