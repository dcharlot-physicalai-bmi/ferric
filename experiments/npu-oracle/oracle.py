#!/usr/bin/env python3
"""npu-oracle — the oracle-digest methodology for opaque accelerator stacks.

For CPUs and GPUs, Ferric pins float semantics at the source/compiler level
and demands ONE digest across substrates (docs/determinism/). NPUs sit behind
opaque compilers (CoreML -> ANE), so the honest contract is different:
MEASURE what a device family produces for a fixed (model, input) pair and
pin THAT as the family's oracle digest. A fleet device of the same family
must reproduce its family oracle bit-exactly; different families keep
different oracles. This script establishes the measurements the contract
needs:

  1. run-to-run stability within a process, per compute unit
  2. process-to-process stability (fresh CoreML compile/load each time)
  3. which compute units agree with each other (CPU vs GPU vs ANE paths)
  4. fp16 vs fp32 program behavior

Model: L layers of (matmul + relu); default 2x(64x64) on a 12x64 input, or
--big for the ANE-engagement grid: 4x(1024x1024) on 256x1024 (small models
never leave the CPU, so timing at --big is the engagement witness).
Weights and inputs come from
the same integer-xorshift generator as ferric's fabric probe (no libm, no
RNG state — bit-identical inputs everywhere, forever).

Usage:  python3 oracle.py            # full grid + report
        python3 oracle.py child P U  # one prediction, print digest (internal)
"""
import hashlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402


def det(n: int, seed: int) -> np.ndarray:
    """xorshift64* -> f32 in [-0.5, 0.5) — bit-exact twin of ferric's det()."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    s = np.uint64((seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) | np.uint64(1)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s ^ (s << np.uint64(13))) & mask
        s = s ^ (s >> np.uint64(7))
        s = (s ^ (s << np.uint64(17))) & mask
        bits = np.uint32(0x3F800000) | np.uint32(s >> np.uint64(41))
        out[i] = np.float32(bits.view(np.float32)) - np.float32(1.5)
    return out


BIG = "--big" in sys.argv
T, D = (256, 1024) if BIG else (12, 64)
L = 4 if BIG else 2
X = det(T * D, 1).reshape(T, D)
WS = [det(D * D, 2 + i).reshape(D, D) for i in range(L)]


def build(precision: str, path: str) -> None:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    @mb.program(input_specs=[mb.TensorSpec(shape=(T, D))])
    def prog(x):
        y = x
        for i in range(L):
            y = mb.matmul(x=y, y=WS[i])
            if i < L - 1:
                y = mb.relu(x=y)
        return y

    prec = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=prec,
        minimum_deployment_target=ct.target.macOS14,
    )
    m.save(path)


def digest_once(path: str, unit: str):
    import coremltools as ct, time

    m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, unit))
    m.predict({"x": X})  # warm (compile/dispatch)
    t0 = time.perf_counter()
    for _ in range(8):
        out = m.predict({"x": X})
    ms = (time.perf_counter() - t0) * 1000 / 8
    y = np.ascontiguousarray(list(out.values())[0], dtype=np.float32)
    return hashlib.sha256(y.tobytes()).hexdigest()[:16], ms


def main() -> None:
    import tempfile, os

    tmp = tempfile.mkdtemp(prefix="npu-oracle-")
    units = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]
    print(f"chip: Apple M5 Max · model {L}x(mm+relu) {T}x{D} · timing = engagement witness")
    for precision in ["fp32", "fp16"]:
        path = os.path.join(tmp, f"m-{precision}.mlpackage")
        build(precision, path)
        print(f"\n── {precision} program ──")
        for unit in units:
            # in-process repeats
            pairs = [digest_once(path, unit) for _ in range(3)]
            runs = [p[0] for p in pairs]
            ms = min(p[1] for p in pairs)
            stable = "stable" if len(set(runs)) == 1 else "UNSTABLE " + str(runs)
            # fresh-process repeats (fresh CoreML compile/load)
            procs = []
            for _ in range(2):
                r = subprocess.run(
                    [sys.executable, __file__, "child", path, unit] + (["--big"] if BIG else []),
                    capture_output=True,
                    text=True,
                )
                procs.append(r.stdout.strip().splitlines()[-1] if r.stdout else "ERR")
            pstable = "stable" if len(set(procs + runs[:1])) == 1 else "PROC-VARIES " + str(procs)
            print(f"{unit:12} {runs[0]}  {ms:7.2f} ms  run-to-run: {stable:8}  process: {pstable}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "child":
        print(digest_once(sys.argv[2], sys.argv[3])[0])
    else:
        main()
