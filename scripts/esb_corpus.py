#!/usr/bin/env python3
"""An `asr_corpus` directory from one of the Open ASR Leaderboard's ESB test sets.

    <python with pyarrow + transformers> esb_corpus.py <parquet-dir> <out-dir> <normalizer.json> [n]

`<parquet-dir>` is one set of `hf-audio/esb-datasets-test-only-sorted` (e.g. its `ami/` folder: 15 shards,
12,643 rows of `audio{bytes,path}, text, id, audio_length_s`). Writes `<id>.wav` — the parquet's WAV
bytes untouched (32-bit float for AMI; `asr_corpus` reads 16-bit PCM and 32-bit float and refuses
anything else) — and `refs.txt`, then prints what it kept.

⭐ WHICH ROWS. Every k-th row of the whole set (k = rows // n), so each length range is represented: the
shards are sorted LONGEST FIRST, and the first n rows of AMI are all 20-26 s clips from a set whose
median is under 2 s. A reference that normalises to nothing under the Whisper English normaliser is
dropped, as the leaderboard drops it (`is_target_text_in_range` in its `normalizer/data_utils.py`): a
filler-only reference ("Mm.", "Um") has no words to get right.

⭐ WHICH ORDER. `asr_joules.sh` compares its chunks with each other, so contiguous blocks of `refs.txt`
must be alike. Rows are sorted by length and then placed by the golden-ratio sequence (rank r goes to
key frac(r * 0.618...)): any contiguous block samples the whole length range evenly, for any number of
chunks. Written in length order, one chunk held the long clips and another the "Yeah."s, and the harness
now refuses that.
"""
import glob, json, struct, sys
import pyarrow.parquet as pq
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

src, out, njs = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
norm = EnglishTextNormalizer(json.load(open(njs)))


def wav_format(b):
    """(tag, channels, rate, bits, seconds) from a RIFF/WAVE header, WAVE_FORMAT_EXTENSIBLE resolved."""
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "not a RIFF/WAVE file"
    i, fmt, data = 12, None, None
    while i + 8 <= len(b):
        cid, sz = b[i:i + 4], struct.unpack_from("<I", b, i + 4)[0]
        if cid == b"fmt ":
            tag, ch, rate = struct.unpack_from("<HHI", b, i + 8)
            bits = struct.unpack_from("<H", b, i + 22)[0]
            if tag == 0xFFFE:
                tag = struct.unpack_from("<H", b, i + 32)[0]
            fmt = (tag, ch, rate, bits)
        if cid == b"data":
            data = sz
        i += 8 + sz + (sz & 1)
    tag, ch, rate, bits = fmt
    return tag, ch, rate, bits, data / (ch * bits // 8) / rate


shards = sorted(glob.glob(f"{src}/*.parquet"))
rows = [(f, i, r) for f in shards
        for i, r in enumerate(pq.read_table(f, columns=["id", "text", "audio_length_s"]).to_pylist())]
k = max(len(rows) // n, 1)
pick = rows[::k][:n]
keep = [p for p in pick if norm(p[2]["text"]).strip()]
dropped = [p[2]["text"] for p in pick if not norm(p[2]["text"]).strip()]
print(f"{len(rows)} rows in {len(shards)} shards; every {k}th -> {len(pick)}; "
      f"{len(dropped)} normalise to nothing and are dropped (e.g. {dropped[:4]})")

got = {}
for f in sorted({p[0] for p in keep}):
    t = pq.read_table(f, columns=["id", "audio"])
    ids, aud = t.column("id").to_pylist(), t.column("audio").to_pylist()
    for _, i, r in (p for p in keep if p[0] == f):
        assert ids[i] == r["id"] and " " not in r["id"], r["id"]
        b = aud[i]["bytes"]
        tag, ch, rate, bits, secs = wav_format(b)
        if ch != 1 or rate != 16000 or (tag, bits) not in [(1, 16), (3, 32)]:
            sys.exit(f"⛔ {r['id']}: format tag {tag}, {ch} channels, {rate} Hz, {bits} bits — asr_corpus reads mono "
                     f"16 kHz 16-bit PCM or 32-bit float")
        open(f"{out}/{r['id']}.wav", "wb").write(b)
        got[r["id"]] = (secs, r["text"])

phi = (5 ** 0.5 - 1) / 2
by_len = sorted(got, key=lambda i: (got[i][0], i))
order = [i for _, i in sorted((((r + 1) * phi) % 1.0, i) for r, i in enumerate(by_len))]
with open(f"{out}/refs.txt", "w") as o:
    for i in order:
        o.write(f"{i} {got[i][1]}\n")
lens = sorted(s for s, _ in got.values())
print(f"wrote {len(got)} clips, {sum(lens):.1f} s of audio: min {lens[0]:.2f} s, median {lens[len(lens) // 2]:.2f} s, "
      f"max {lens[-1]:.2f} s -> {out}")
