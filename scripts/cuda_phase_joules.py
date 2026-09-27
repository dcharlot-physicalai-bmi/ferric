#!/usr/bin/env python3
"""**Board joules per token, by phase, on an NVIDIA GPU** — around `examples/phase_bench`'s MARK lines.

    scripts/cuda_phase_joules.py <phase_bench binary> <model.gguf> <prompt> <decode> <reps> [ENV=VAL ...]

Samples NVML board power (`nvmlDeviceGetPowerUsage`) every 20 ms on a thread while the bench runs, and
integrates it over each phase's marked windows. Also records an idle baseline first.

⛔ WHY NOT THE ENERGY COUNTER. `nvmlDeviceGetTotalEnergyConsumption` is the right instrument in
principle (a counter read once beats a sampler), and on the RTX 4050 Laptop box it is BROKEN: while
board power read 1.8–3.5 W, its 100 ms deltas ramped in a sawtooth from 0.3 J to 55 J — hundreds of
watts implied. So this samples power instead, and states the sampler's limits:
  • the reading refreshes about once a second (measured: it held one value for ~1 s at a time), so a
    phase window shorter than a few seconds cannot be resolved — this script REFUSES to report a phase
    whose marked windows total under 5 s, rather than print a number that is mostly edge error;
  • it is BOARD power (GPU + its memory), not the host CPU or DRAM.
Run long single-phase configurations: decode-heavy (prompt 16, decode 1024+) and prefill-heavy
(prompt 512, decode 0, many reps).
"""
import ctypes, os, subprocess, sys, threading, time

def nvml():
    n = ctypes.CDLL("libnvidia-ml.so.1")
    if n.nvmlInit_v2() != 0: sys.exit("NVML init failed — no power measurement possible")
    h = ctypes.c_void_p()
    if n.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0: sys.exit("no NVML device 0")
    p = ctypes.c_uint()
    def read():
        return p.value / 1000.0 if n.nvmlDeviceGetPowerUsage(h, ctypes.byref(p)) == 0 else None
    return read

def main():
    if len(sys.argv) < 6: sys.exit(__doc__)
    binp, model, prompt, decode, reps = sys.argv[1:6]
    env = dict(os.environ)
    for kv in sys.argv[6:]:
        k, v = kv.split("=", 1); env[k] = v
    read = nvml()
    idle = []
    t_end = time.time() + 3.0
    while time.time() < t_end:
        w = read(); idle.append(w) if w is not None else None; time.sleep(0.05)
    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            w = read()
            if w is not None: samples.append((time.time(), w))
            time.sleep(0.02)
    th = threading.Thread(target=sampler); th.start()
    r = subprocess.run([binp, model, prompt, decode, reps], capture_output=True, text=True, env=env)
    stop.set(); th.join()
    if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
    print(r.stdout, end="")
    wins = {}
    for l in r.stderr.splitlines():
        p = l.split()
        if len(p) == 4 and p[0] == "MARK":
            wins.setdefault(p[1], []).append((p[2], int(p[3]) / 1e9))
    toks = {"prefill": int(prompt), "decode": int(decode)}
    print(f"  idle board power {sum(idle) / len(idle):.2f} W (3 s before the run)")
    for ph, edges in wins.items():
        spans = [(edges[i][1], edges[i + 1][1]) for i in range(0, len(edges) - 1, 2)]
        dur = sum(b - a for a, b in spans)
        if dur < 5.0:
            print(f"  {ph}: marked windows total {dur:.2f} s < 5 s — NOT MEASURED (power refreshes ~1/s)"); continue
        # piecewise-constant power between samples, clipped to each window
        e = 0.0
        for a, b in spans:
            for (t0, w), (t1, _) in zip(samples, samples[1:]):
                lo, hi = max(a, t0), min(b, t1)
                if hi > lo: e += w * (hi - lo)
        n = toks[ph] * len(spans)
        print(f"  {ph}: {dur:.2f} s marked, board {e / dur:.2f} W mean, {e:.1f} J for {n} tokens = {e / n * 1000:.2f} mJ/token")

if __name__ == "__main__":
    main()
