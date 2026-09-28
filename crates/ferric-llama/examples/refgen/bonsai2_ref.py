#!/usr/bin/env python3
"""PrismML **Bonsai 2** logits from the AUTHORS' OWN runtime — their llama.cpp fork — on fixed token ids.

⛔ THE REFERENCE IS THE AUTHORS' CODE. Bonsai 2 ships only as GGUFs whose weights sit in a Hadamard-rotated
basis, and the runtime that defines the model is PrismML's fork (PrismML-Eng/llama.cpp, release
prism-b10743-adfffbe — the build the Bonsai-demo pins): its graph applies the activation transform and its
converter wrote the contract. There is no transformers checkpoint for these files to compare against.

Reference program: `dumplogits` (bonsai2_dumplogits.cpp beside this file), ~80 lines of C++ against the
fork's public C API (llama.h), built inside a checkout of the fork (examples/dumplogits, -DGGML_METAL=ON).
It feeds EXACT token ids — a tokenizer difference cannot pass for a model one — writes float32 logits for
every prompt position, then continues greedily.

⭐ WHICH FORK PATH, measured 2026-09-28 on the PQ2_0 file (max |logit diff|):
  * default batched prefill vs the fork's own decode path: 2.4e-2 (chat) and 0.375 (long) — its mul_mm
    converts activations to f16. Unusable as a tight reference.
  * decode path (`--ubatch 1`, f32 activations) flash attention ON vs OFF: 3.0e-3 (chat), 6.7e-2 at one
    sensitive row of the long prompt — Metal flash attention runs K/V in half precision.
  * decode path, flash OFF — what this fixture records. Ferric agrees with it to 1.1e-4 (chat).
The flash-ON-vs-OFF spread over the recorded rows is stored as `noise_floor`: how far the authors' own
two attention kernels disagree. The gate holds Ferric within it.

Per recorded row, like refgen/lm_logits_ref.py: top-10 (id, logit), logits at 128 fixed vocabulary ids,
and the sum / sum-of-squares of the FULL row. Rows: every prompt position (every 16th + the last 8 for a
long prompt), then one row per greedy step. Each step's top-1/top-2 margin is kept so a near-tie shows.

  python3 bonsai2_ref.py <dumplogits> <model.gguf> > fixture.json
"""
import array, hashlib, json, os, subprocess, sys, tempfile

N_GEN = 32
HERE = os.path.dirname(os.path.abspath(__file__))
# Token ids from the fork's own tokenizer (`llama-tokenize -f <text> --ids`; add_bos_token=false), and the
# texts they came from, in bonsai2_prompts.json.
PROMPTS = json.load(open(os.path.join(HERE, "bonsai2_prompts.json")))


def run(binary, model, ids, extra):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "l.bin")
        r = subprocess.run([binary, model, ",".join(map(str, ids)), p, "--gen", str(N_GEN), "--ubatch", "1", *extra],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        gen = [int(x) for x in r.stdout.split("gen=")[1].split()[0].split(",")]
        a = array.array("f"); a.frombytes(open(p, "rb").read())
        return a, gen


def main():
    binary, model = sys.argv[1], sys.argv[2]
    nv = 248320
    sample_ids = sorted({(i * 1939 + 7) % nv for i in range(128)})
    sha = hashlib.sha256()
    with open(model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): sha.update(chunk)
    out = {"model": os.path.basename(model), "sha256": sha.hexdigest(), "n_vocab": nv, "sample_ids": sample_ids,
           "reference": "PrismML-Eng/llama.cpp @ adfffbe (release prism-b10743), Metal, dumplogits via llama.h, "
                        "n_ubatch=1 (decode kernels, f32 activations), flash attention OFF, KV f32",
           "prompts": []}
    floor = 0.0
    for p in PROMPTS:
        ids = p["ids"]
        a, gen = run(binary, model, ids, ["--noflash"])
        b, gen_fa = run(binary, model, ids, [])
        t, n_rows = len(ids), len(a) // nv
        # Every position of a short prompt, every 2nd of a medium one, every 16th of a long one — each
        # with its tail — and every greedy step. Logits kept to 1e-5, far below the 3e-3 floor.
        stride = 1 if t <= 16 else 2 if t <= 64 else 16
        pos = sorted(set(range(0, t, stride)) | set(range(max(0, t - 8), t)))
        pos += list(range(t, n_rows))
        rows, margins, pfloor = [], [], 0.0
        for r in pos:
            row = a[r * nv:(r + 1) * nv]
            order = sorted(range(nv), key=row.__getitem__, reverse=True)[:10]
            rd = lambda v: round(v, 5)
            rows.append({"top": [[i, rd(row[i])] for i in order], "sample": [rd(row[i]) for i in sample_ids],
                         "sum": rd(float(sum(row))), "ssq": rd(float(sum(x * x for x in row)))})
            pfloor = max(pfloor, max(abs(x - y) for x, y in zip(row, b[r * nv:(r + 1) * nv])))
            if r >= t - 1:
                margins.append(round(row[order[0]] - row[order[1]], 5))
        floor = max(floor, pfloor)
        out["prompts"].append({"name": p["name"], "ids": ids, "gen": gen, "gen_flash_equal": gen == gen_fa,
                               "positions": pos, "rows": rows, "gen_margins": margins, "noise_floor": round(pfloor, 7)})
        print(f"{p['name']}: {t} ids, {len(pos)} rows, floor {pfloor:.3e}, min greedy margin {min(margins):.3f}",
              file=sys.stderr)
    out["noise_floor"] = {"what": "the same fork and path with flash attention ON vs OFF: max |logit diff| over "
                                  "every recorded row of every prompt", "max_abs": round(floor, 7)}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
