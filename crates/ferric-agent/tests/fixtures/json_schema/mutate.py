"""Can the JSON-Schema tests fail? Break the port in plausible ways, one at a time, and require the tests to catch each.

Each mutation replaces one exact piece of crates/ferric-agent/src/json_schema.rs (it must occur exactly once),
runs `cargo test -p ferric-agent --test json_schema`, records which tests failed, and restores the file from
memory (in a finally, so an interrupted run leaves the source as it was). A mutation no test catches is a FAIL.

usage (from the repository root): python3 crates/ferric-agent/tests/fixtures/json_schema/mutate.py
"""
import os, re, subprocess, sys

SRC = "crates/ferric-agent/src/json_schema.rs"
MUTATIONS = [
    ("a wrong rule name: an array's item rule named `items`",
     'let item_rule_name = self.visit(items, &format!("{sub_name}item"))?;',
     'let item_rule_name = self.visit(items, &format!("{sub_name}items"))?;'),
    ("`required` dropped: every property optional",
     "properties.push(Property { name: name.clone(), schema: node, required: required.contains(name.as_str()) });",
     "properties.push(Property { name: name.clone(), schema: node, required: false });"),
    ("off by one in the integer range builder: a digit range reaching the bound's digit",
     "                digit_range(out, fi, ti - 1);\n                out.push(' ');\n                more_digits(out, sub_len as i32, sub_len as i32);",
     "                digit_range(out, fi, ti);\n                out.push(' ');\n                more_digits(out, sub_len as i32, sub_len as i32);"),
    ("off by one in an exclusive bound: exclusiveMinimum taken as inclusive",
     'minimum = Self::get_bound(schema, "exclusiveMinimum", path, false)?.wrapping_add(1);',
     'minimum = Self::get_bound(schema, "exclusiveMinimum", path, false)?.wrapping_add(0);'),
    ("a wrong space rule: 19 blanks of indent instead of 20",
     r'''pub const SPACE_RULE: &str = r#"| " " | "\n"{1,2} [ \t]{0,20}"#;''',
     r'''pub const SPACE_RULE: &str = r#"| " " | "\n"{1,2} [ \t]{0,19}"#;'''),
    ("a wrong space rule: one newline at most",
     r'''pub const SPACE_RULE: &str = r#"| " " | "\n"{1,2} [ \t]{0,20}"#;''',
     r'''pub const SPACE_RULE: &str = r#"| " " | "\n" [ \t]{0,20}"#;'''),
    ("Grisu2's final rounding step skipped",
     "    exp10 -= m;\n    round(&mut buf, dist, delta, p2, one.f);",
     "    exp10 -= m;\n    let _ = (dist, delta, p2, &round);"),
    ("optional properties before required ones",
     '        for (i, k) in required_props.iter().enumerate() {\n            if i > 0 { rule.push_str(" \\",\\" space "); }',
     '        required_props.reverse();\n        for (i, k) in required_props.iter().enumerate() {\n            if i > 0 { rule.push_str(" \\",\\" space "); }'),
]

orig = open(SRC).read()
failed_to_catch = 0
env = dict(os.environ, CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "4"))
try:
    for what, old, new in MUTATIONS:
        n = orig.count(old)
        if n != 1:
            print(f"FAIL  {what}: the original text occurs {n} times"); failed_to_catch += 1; continue
        open(SRC, "w").write(orig.replace(old, new))
        r = subprocess.run(["cargo", "test", "-p", "ferric-agent", "--offline", "--test", "json_schema"], capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        failing = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED", out, re.M)))
        passing = sorted(set(re.findall(r"^test (\S+) \.\.\. ok", out, re.M)))
        if "error[E" in out:
            print(f"FAIL  {what}: does not compile"); failed_to_catch += 1
        elif r.returncode != 0 and failing:
            print(f"ok    {what}: caught by {len(failing)} test(s): {', '.join(failing)}")
        else:
            print(f"FAIL  {what}: NOT CAUGHT (tests passing: {', '.join(passing)})"); failed_to_catch += 1
finally:
    open(SRC, "w").write(orig)
print("ALL CAUGHT" if not failed_to_catch else f"{failed_to_catch} NOT CAUGHT")
sys.exit(1 if failed_to_catch else 0)
