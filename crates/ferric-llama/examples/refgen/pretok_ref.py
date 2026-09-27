#!/usr/bin/env python3
"""Pre-tokenization pieces from the MODEL AUTHORS' OWN tokenizer files, for strings WITH NEWLINES.

Ferric's text->ids gate (`scripts/tokenizer_conformance.sh`) reads its corpus one LINE at a time, so no
string it ever checked contained a newline — and two of the regex's seven alternatives only fire on
one: ` ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*` (punctuation keeps the newlines that follow it) and `\\s*[\\r\\n]+`
(a whitespace run is cut after its LAST newline). Both were missing from Ferric's Qwen/Llama-3 path;
MiMo-V2.5-ASR's prompt `<think>\\n\\n</think>\\n` exposed it: Ferric built 64 ids against the authors' 62.

The reference is the authors' `tokenizer.json`, run by HuggingFace `tokenizers` (the library those
files are written for). Pieces are recorded as the ORIGINAL substrings (by offset), not byte-level
symbols, so the Rust test needs no vocabulary.

    <python> pretok_ref.py > crates/ferric-tokenizer/tests/fixtures/pretok_newlines.json
"""
import glob, json, os, sys
from tokenizers import Tokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
SOURCES = {   # Ferric's Pre variant -> a checkpoint whose tokenizer.json carries that regex
    "qwen2": "models--Qwen--Qwen3-0.6B",
    "qwen35": "models--Qwen--Qwen3.5-0.8B",
    "llama3": "models--unsloth--Llama-3.2-1B-Instruct",
}
CORPUS = [
    "<think>\n\n</think>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n", "a:\n", "x;\r\n",
    "\n  x", "foo  \n bar", "\n\n\n", "a \n", "end.\n\nNext", "  \n\t\n  y", "a\r\n\r\nb", "\t(x)\n",
    "def f(x):\n    return x  # ok\n\n\nprint(f(1))\n", "- item one\n- item two\n\n1. first\n2. second",
    "key: value\r\nother:\tvalue2 \r\n", "line with trailing spaces   \nnext", " \n ", "\n", " ", "  ",
    "x \n\n y", "}\n}\n", "!!\n\n??", "a.\n b", "Q:\nA:\n", "IT'S we'RE they'Ll", "tab\tsep\tvalues\n",
    "2026\n1999 12345", "héllo\nwörld", "こんにちは\n世界", "नमस्ते\nदुनिया", "Hello-Reyes\n-Reyes", "a--b\n--c",
    "url: https://x.y/z?a=1\n", "\u00a0\n\u2028x", "   leading", "trailing   ", "mixed \t \n \t end",
    "'\u017ft it'\u017felf o'\u017ftrich",   # U+017F folds to 's' under (?i:) — lowercase does not
]


def pieces(tok, s):
    return [s[a:b] for _, (a, b) in tok.pre_tokenizer.pre_tokenize_str(s)]


out = {"note": __doc__.split("\n\n")[0], "tokenizers": __import__("tokenizers").__version__, "rules": {}}
for pre, repo in SOURCES.items():
    f = sorted(glob.glob(os.path.join(HUB, repo, "snapshots", "*", "tokenizer.json")))
    if not f:
        sys.exit(f"no tokenizer.json for {repo}")
    tok = Tokenizer.from_file(f[0])
    j = json.load(open(f[0]))
    out["rules"][pre] = {"source": repo.replace("models--", "").replace("--", "/"),
                         "regex": [p["pattern"]["Regex"] for p in j["pre_tokenizer"]["pretokenizers"]
                                   if p.get("type") == "Split"],
                         "cases": [{"text": s, "pieces": pieces(tok, s)} for s in CORPUS]}
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
