#!/usr/bin/env python3
"""Corpus WER under the Whisper English text normaliser — the one the Open ASR Leaderboard scores with.

`asr_corpus` scores LibriSpeech's own way (uppercase, letters and apostrophes), which counts a WRITTEN
form as an error against the corpus's SPOKEN one: MiMo-V2.5-ASR writes "Mr." where LibriSpeech has
"MISTER", and the transcript the conformance gate certifies as the authors' exact output scores 1 edit in
17 words. A transducer trained on LibriSpeech-style text never pays that. Comparing two such models on the
LibriSpeech normaliser alone biases the comparison, so both are reported.

    <python-with-transformers> asr_wer_normalized.py <normalizer.json> <refs.txt> <asr_corpus output>

`normalizer.json` is the spelling map Whisper ships (e.g. openai/whisper-base); the normaliser class is
transformers' `EnglishTextNormalizer`. Prints `NORMALIZED words <n> edits <e> correct <n-e> wer <pct>`.
"""
import json, sys
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

norm = EnglishTextNormalizer(json.load(open(sys.argv[1])))
refs = dict(l.rstrip("\n").split(" ", 1) for l in open(sys.argv[2]) if " " in l)
hyps = {}
for l in open(sys.argv[3]):
    if l.startswith("UTT "):
        head, _, hyp = l.rstrip("\n").partition(" | ")
        hyps[head.split()[1]] = hyp
missing = sorted(set(refs) - set(hyps))
if missing:
    sys.exit(f"⛔ {len(missing)} references have no hypothesis (e.g. {missing[:3]})")

def edits(r, h):
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1]

words = errs = 0
for k, ref in refs.items():
    r, h = norm(ref).split(), norm(hyps[k]).split()
    words += len(r); errs += edits(r, h)
print(f"NORMALIZED words {words} edits {errs} correct {max(words - errs, 0)} wer {100 * errs / max(words, 1):.2f}")
