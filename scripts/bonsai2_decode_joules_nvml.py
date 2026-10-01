#!/usr/bin/env python3
"""**Decode tokens/s and BOARD joules per token on an NVIDIA GPU** — Bonsai 2, Ferric or PrismML's fork,
the same IDLE/RUN/SUMMARY protocol as scripts/bonsai2_decode_joules.sh (macmon), metered with NVML.

    scripts/bonsai2_decode_joules_nvml.py <runner> <model.gguf> [chunks 5] [k 128] [idle_s 6] [ENV=VAL ...]

Runners (both load and warm OUTSIDE every window, then per chunk: prefill untimed, an IDLE gap, a RUN
window of k greedy decode tokens with a full-logits read per step, a HASH of the generated ids):
  Ferric  target/release/examples/bonsai2_decode_joules          (FERRIC_STREAM_GIB=<GiB> to stream)
  fork    crates/ferric-llama/examples/refgen/bonsai2_decode_joules.cpp against the fork's libllama
          (BONSAI2_NGL=<gpu layers>, BONSAI2_THREADS=<cpu threads> for partial offload)

⛔ NOT THE ENERGY COUNTER. On the RTX 4050 Laptop box `nvmlDeviceGetTotalEnergyConsumption` is broken
(sawtooth deltas implying hundreds of watts; scripts/cuda_phase_joules.py), so board POWER is sampled
every 20 ms and integrated. Its limits, stated: the reading refreshes about once a second, so a RUN window
under 5 s is refused; the first reading after the GPU wakes has been seen at 590 W on a 1.5 W idle board,
so any sample above BONSAI2_NVML_CEILING_W (default 200 W — the laptop part's TGP is far below) refuses its
chunk rather than being silently dropped. BOARD power = GPU + its GDDR; the host CPU and DRAM are NOT in it,
which matters for a runner that leaves layers on the CPU (the fork with partial offload): its CPU work is
unpriced here, and this script says so in its output.

Attribution as bonsai2_decode_joules.sh: each chunk is charged against the idle gaps either side (2 s
settle); refused if the gaps disagree by more than 25% of the work, or if a chunk is > 2x from the median.
Both the GROSS board energy in the RUN window and the MARGINAL energy over idle are printed.
"""
import ctypes, os, subprocess, sys, threading, time


def nvml():
    n = ctypes.CDLL("libnvidia-ml.so.1")
    if n.nvmlInit_v2() != 0: sys.exit("NVML init failed — no power measurement possible")
    h = ctypes.c_void_p()
    if n.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0: sys.exit("no NVML device 0")
    p = ctypes.c_uint()
    return lambda: p.value / 1000.0 if n.nvmlDeviceGetPowerUsage(h, ctypes.byref(p)) == 0 else None


def main():
    a = [x for x in sys.argv[1:] if "=" not in x]
    if len(a) < 2: sys.exit(__doc__)
    env = dict(os.environ)
    for kv in (x for x in sys.argv[1:] if "=" in x):
        k, v = kv.split("=", 1); env[k] = v
    runner, model = a[0], a[1]
    chunks, k, idle = (a[2] if len(a) > 2 else "5"), (a[3] if len(a) > 3 else "128"), (a[4] if len(a) > 4 else "6")
    ids = env.get("BONSAI2_IDS", "760,6511,314,9338,369")  # "The capital of France is" (fork tokenizer)
    ceil = float(env.get("BONSAI2_NVML_CEILING_W", "200"))
    read = nvml()
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            w = read()
            if w is not None: samples.append((time.time(), w))
            time.sleep(0.02)

    th = threading.Thread(target=sampler); th.start()
    r = subprocess.run([runner, model, ids, chunks, k, idle], capture_output=True, text=True, env=env)
    stop.set(); th.join()
    if r.returncode: print(r.stderr[-3000:]); sys.exit(1)
    for l in r.stderr.splitlines():
        if l.startswith(("adapter:", "streaming:")) or "CUDA" in l and "device" in l.lower(): print("  " + l.strip())
    hashes = sorted({l.split()[1] for l in r.stderr.splitlines() if l.startswith("HASH")})
    if len(hashes) != 1: print(f"⛔ chunks generated DIFFERENT tokens ({len(hashes)} hashes)"); sys.exit(1)
    print(f"  output hash {hashes[0]}")
    o = r.stdout.splitlines()
    idles = [tuple(map(float, l.split()[1:3])) for l in o if l.startswith("IDLE ")]
    runs = [tuple(map(float, l.split()[1:3])) for l in o if l.startswith("RUN ")]
    toks = [int(l.split()[3]) for l in o if l.startswith("RUN ")]
    print("  " + next(l for l in o if l.startswith("SUMMARY ")))
    if len(idles) != len(runs) + 1: print(f"⛔ {len(idles)} idle gaps for {len(runs)} chunks"); sys.exit(1)

    def window(lo, hi):  # time-weighted mean of piecewise-constant power over [lo, hi]
        e = 0.0
        for (t0, w), (t1, _) in zip(samples, samples[1:]):
            x, y = max(lo, t0), min(hi, t1)
            if y > x: e += w * (y - x)
        return e / (hi - lo), [w for t, w in samples if lo <= t <= hi]

    rows, gross, marg, ok = [], [], [], True
    for i, (s0, s1) in enumerate(runs):
        p_run, xr = window(s0, s1)
        p0, x0 = window(idles[i][0] + 2.0, idles[i][1])
        p1, x1 = window(idles[i + 1][0] + 2.0, idles[i + 1][1])
        base = (p0 + p1) / 2
        why = []
        if s1 - s0 < 5.0: why.append("RUN window under 5 s (power refreshes ~1/s)")
        if max(xr + x0 + x1, default=0) > ceil: why.append(f"a sample above {ceil:.0f} W — the wake glitch")
        if not p_run > base: why.append("the work does not exceed idle")
        elif abs(p0 - p1) > 0.25 * (p_run - base): why.append("the gaps disagree by more than 25% of the work")
        ok &= not why
        gross.append(p_run * (s1 - s0) / toks[i]); marg.append((p_run - base) * (s1 - s0) / toks[i])
        rows.append(f"    chunk {i}: {s1 - s0:6.1f} s  {toks[i] / (s1 - s0):6.2f} tok/s  board {p_run:6.2f} W  "
                    f"idle {p0:5.2f}/{p1:5.2f} W  gross {gross[-1]:6.3f} J/tok  marginal {marg[-1]:6.3f} J/tok"
                    + "".join(f"   ⛔ {w}" for w in why))
    med = sorted(marg)[len(marg) // 2]
    for i, v in enumerate(marg):
        if not (0.5 * med <= v <= 2.0 * med): ok = False; rows[i] += f"   ⛔ {v / med:.1f}x the median chunk"
    print("\n".join(rows))
    n = sum(toks); secs = sum(b - a for a, b in runs)
    print(f"  BOARD {n} tokens in {secs:.1f} s = {n / secs:.2f} tok/s;  gross {sum(g * t for g, t in zip(gross, toks)) / n:.3f} "
          f"J/token, marginal over idle {sum(m * t for m, t in zip(marg, toks)) / n:.3f} J/token   "
          f"{'✅ attributable' if ok else '⛔ NOT attributable — do not quote'}")
    print("  ⚠ board power only (GPU + GDDR): host CPU/DRAM work — a CPU-resident layer, a host-side weight "
          "copy — is NOT in these joules")
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
