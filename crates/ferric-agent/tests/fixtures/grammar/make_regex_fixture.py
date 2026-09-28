"""regex_fullmatch.json: patterns x strings with the verdict of Python's re.fullmatch(pattern, s, re.ASCII) —
the meaning of a regex constraint in outlines / vLLM. Seeded, deterministic.   python3 make_regex_fixture.py"""
import json, random, re, sys
random.seed(20260928)
patterns = {
    r"[a-z]+@[a-z]+\.com": ["bob@example.com", "a@b.com", "@b.com", "a@b.co", "A@b.com", "ab@cd.comm"],
    r"\d{3}-\d{4}": ["555-1234", "55-1234", "5551234", "555-12345", "abc-defg"],
    r"(yes|no|maybe)": ["yes", "no", "maybe", "yesno", "", "Yes"],
    r"(?:ab)+c?": ["ab", "ababc", "abc", "c", "aba", ""],
    r"[^aeiou\s]{2,4}": ["bcd", "bc", "b", "bcdfg", "bad", "b c"],
    r"\w+\s\w+": ["hello world", "a_1 b2", "hello  world", "hello", "héllo world"],
    r"(a|b)*abb": ["abb", "aabb", "babb", "ab", "abba", "aaaabb"],
    r"x{0,2}y{2,}z?": ["yy", "xyyyz", "xxyy", "xxxyy", "y", "yyz"],
    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?": ["0", "-12.50", "01", "3.", ".5", "1234567890", "-0"],
    r"\(\d+\)": ["(12)", "()", "12", "(a)"],
    r"[A-Za-z_][A-Za-z0-9_]*": ["x", "_a1", "1a", "a-b", "CamelCase"],
    r".{3}": ["abc", "a\nc", "ab", "abcd", "   "],
    r"a|b|": ["a", "b", "", "ab"],
    r"[\d.]+": ["1.2.3", "...", "1a", ""],
    r"(\w+,)*\w+": ["a,b,c", "a", "a,", ",a", "a,,b"],
    r"[-+]?[0-9]+": ["+1", "-12", "1-", "+", "007"],
    r"\{\"k\": \d+\}": ['{"k": 1}', '{"k": }', '{"k":1}'],
    r"^\S+$": ["abc", "a b", ""],
    r"(ab?){2}": ["aa", "abab", "aba", "a", "ababab"],
    r"[a-c]{2}[x-z]?": ["ab", "abx", "abz", "abw", "a"],
    r"colou?r": ["color", "colour", "colouur", "colr"],
    r"\t\n": ["\t\n", "\t", "tn"],
}
alphabet = "abcxyz019 _-.,@(){}\"\n\tABé"
out = []
for p, hand in patterns.items():
    rx = re.compile(p, re.ASCII)
    pool = set(hand)
    for _ in range(40):
        n = random.randint(0, 8)
        pool.add("".join(random.choice(alphabet) for _ in range(n)))
    # near-misses of every positive: drop, duplicate or swap one character
    for s in list(hand):
        if rx.fullmatch(s) and s:
            i = random.randrange(len(s)); pool.add(s[:i] + s[i + 1:]); pool.add(s[:i] + s[i] + s[i:])
    strings = sorted(pool)
    out.append({"pattern": p, "strings": [[s, rx.fullmatch(s) is not None] for s in strings]})
json.dump({"python": sys.version.split()[0], "flags": "re.ASCII, fullmatch", "cases": out}, open("regex_fullmatch.json", "w"), indent=1, ensure_ascii=False)
print(len(out), "patterns,", sum(len(c["strings"]) for c in out), "strings,", sum(t[1] for c in out for t in c["strings"]), "matching")
