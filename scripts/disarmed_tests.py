#!/usr/bin/env python3
"""Detector for tests that cannot fail because nothing runs them. See disarmed_tests.sh for why.

The signature of the defect: a fn that sits inside a `#[cfg(test)]` module, is not a method of an
`impl` block, carries no test attribute, and has no caller anywhere in its crate. Such a fn is
either dead scaffolding or -- the case this was written for -- an assertion whose `#[test]` was
lost, which reads in review exactly like coverage and is incapable of failing.
"""
import re, sys, pathlib


def strip_literals(ln):
    """Blank out string/char literals and line comments so brace counting is structural.

    ⛔ Without this the counter reads `ok("{\\"a\\":1}", true)` as real braces and the module-depth
    tracking drifts, which silently changes WHICH fns are considered to be inside a test module.
    A depth tracker that counts braces in strings is a detector that reports on the wrong scope.
    """
    out, i, n = [], 0, len(ln)
    while i < n:
        c = ln[i]
        if c == '/' and i + 1 < n and ln[i+1] == '/':
            break
        if c == 'r' and i + 1 < n and ln[i+1] in '#"':          # raw string r"..." / r#"..."#
            j = i + 1; hashes = 0
            while j < n and ln[j] == '#': hashes += 1; j += 1
            if j < n and ln[j] == '"':
                close = '"' + '#' * hashes
                k = ln.find(close, j + 1)
                i = n if k < 0 else k + len(close)
                out.append(' '); continue
        if c == '"':
            j = i + 1
            while j < n:
                if ln[j] == '\\': j += 2; continue
                if ln[j] == '"': break
                j += 1
            i = j + 1; out.append(' '); continue
        if c == "'":                                            # char literal, not a lifetime
            j = i + 1
            if j < n and ln[j] == '\\':
                j += 2
                while j < n and ln[j] != "'": j += 1
                i = j + 1; out.append(' '); continue
            if j + 1 < n and ln[j+1] == "'":
                i = j + 2; out.append(' '); continue
        out.append(c); i += 1
    return ''.join(out)


FN = re.compile(r'^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?'
                r'(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)')
ATTR = re.compile(r'^\s*#!?\[')
DOC = re.compile(r'^\s*//')
IMPL = re.compile(r'^\s*(?:unsafe\s+)?impl\b')
MOD = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{')
TEST_ATTR = re.compile(r'#\[\s*(?:\w+\s*::\s*)*(?:test|bench|rstest|proptest|quickcheck|test_case|'
                       r'should_panic|ignore|tokio\s*::\s*test|googletest\s*::\s*test)\b')

def scan(src):
    """Yield (line_no, fn_name) for every fn matching the defect signature, ignoring callers."""
    lines = src.splitlines()
    depth = 0
    test_mod_depths = []   # brace depths at which a #[cfg(test)] mod opened
    impl_depths = []       # brace depths at which an impl block opened
    saw_cfg_test = False
    out = []
    for i, ln in enumerate(lines):
        stripped = strip_literals(ln).strip()
        if '#[cfg(test)]' in ln:
            saw_cfg_test = True
        opened_here = None
        if MOD.search(ln):
            if saw_cfg_test:
                opened_here = 'test_mod'
            saw_cfg_test = False
        elif IMPL.match(ln):
            opened_here = 'impl'
        elif stripped and not ATTR.match(ln) and not DOC.match(ln):
            # any other real statement ends the attribute's reach
            if not FN.match(ln):
                saw_cfg_test = False if 'mod ' not in ln else saw_cfg_test

        m = FN.match(ln)
        if m and test_mod_depths and not impl_depths:
            name = m.group(2)
            j, attrs = i - 1, []
            while j >= 0:
                s = lines[j]
                if ATTR.match(s):
                    attrs.append(s.strip()); j -= 1; continue
                if DOC.match(s) or not s.strip():
                    if not s.strip() and attrs:
                        break
                    j -= 1; continue
                break
            if not any(TEST_ATTR.search(a) for a in attrs):
                out.append((i + 1, name))

        if opened_here == 'test_mod':
            test_mod_depths.append(depth)
        elif opened_here == 'impl':
            impl_depths.append(depth)
        code = strip_literals(ln)
        depth += code.count('{') - code.count('}')
        while test_mod_depths and depth <= test_mod_depths[-1]:
            test_mod_depths.pop()
        while impl_depths and depth <= impl_depths[-1]:
            impl_depths.pop()
    return out

def find(src, whole):
    """The gate's ACTUAL decision, used by both main() and the self-test.

    ⛔ The self-test must run this, not `scan` alone. The first version of this file self-tested
    `scan` while the gate ran scan + a caller filter, so the self-test could pass on a detector the
    gate did not use. It caught itself: it reported `helper` as disarmed. Kept as the reason the two
    paths are now one function.
    """
    return [(line, name) for line, name in scan(src)
            if len(re.findall(r'\b' + re.escape(name) + r'\s*[(:]', whole)) <= 1]


FIXTURE = r'''
#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;
    impl Meter for Fake {
        fn read_joules(&self) -> Option<f64> { Some(1.0) }   // trait impl: MUST NOT flag
        fn class(&self) -> Class { Class::Measured }         // trait impl: MUST NOT flag
    }

    fn helper(x: u32) -> u32 { x + 1 }                       // called below: MUST NOT flag

    #[test]
    fn a_real_test() { assert_eq!(helper(1), 2); }           // attributed: MUST NOT flag

    /// doc comment, and an attribute that is not a test attribute
    #[allow(dead_code)]
    fn disarmed_with_allow() { assert!(false); }             // MUST FLAG

    /// ⭐ the real case
    fn disarmed_plain() { assert!(false); }                  // MUST FLAG

    #[test]
    fn braces_inside_a_string_literal() {                    // MUST NOT shift the scope
        let s = "{\\"a\\":1}";
        let c = '}';
        assert!(s.len() > 0 && c == '}');
    }
}

// OUTSIDE the test module: test-only, but at file scope, and nothing calls it.
#[cfg(test)]
pub(crate) fn orphan_helper(x: u32) -> u32 { x }             // MUST NOT flag (not in a test mod)
'''

def self_test():
    got = {n for _, n in find(FIXTURE, FIXTURE)}   # the SAME call the gate makes
    want = {'disarmed_with_allow', 'disarmed_plain'}
    print("self-test fixture contains: a trait impl, a called helper, an attributed test,")
    print("                            and two disarmed assertions.")
    print(f"  detector flagged: {sorted(got) or '(nothing)'}")
    print(f"  must flag:        {sorted(want)}")
    if got != want:
        print(f"\n⛔ SELF-TEST FAILED — {'over-fires on ' + str(sorted(got - want)) if got - want else ''}"
              f"{' misses ' + str(sorted(want - got)) if want - got else ''}")
        return 1
    print("\n✅ the detector can fail, and does not fire on the three legitimate shapes.")
    return 0

def main():
    root = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == '--self-test':
        return self_test()
    crates = sorted(p for p in (root / 'crates').iterdir() if p.is_dir()) if (root / 'crates').is_dir() else []
    hits, n_files = [], 0
    for crate in crates:
        files = [p for p in crate.rglob('*.rs') if '/target/' not in str(p)]
        if not files:
            continue
        whole = "\n".join(p.read_text(encoding='utf-8', errors='replace') for p in files)
        for p in files:
            n_files += 1
            src = p.read_text(encoding='utf-8', errors='replace')
            for line, name in find(src, whole):
                hits.append((p.relative_to(root), line, name))
    print(f"scanned {n_files} .rs files across {len(crates)} crates in crates/")
    if not hits:
        print("✅ no disarmed tests: every fn in a #[cfg(test)] module is attributed, a method, or called.")
        return 0
    print(f"\n⛔ {len(hits)} fn(s) in a test module that nothing runs and nothing calls:\n")
    for f, l, n in hits:
        print(f"  {f}:{l}  fn {n}()")
    print("\nEach is either dead scaffolding (delete it) or an assertion missing #[test] (wire it).")
    print("Wire it FIRST and watch it run before you trust what it claims to prove.")
    return 1

sys.exit(main())
