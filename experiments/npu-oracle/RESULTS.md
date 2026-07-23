# NPU oracle digests — first measurements (Apple M5 Max, 2026-07-23)

Model: 4×(matmul 1024×1024 + relu), input 256×1024, weights/inputs from the
fabric-probe integer generator. coremltools 9.0, macOS 15 target. 3
in-process runs + 2 fresh-process runs per cell; timing = min of 8 warm
predictions (the ANE-engagement witness).

| program | compute units | digest             | ms    | stability |
|---------|---------------|--------------------|-------|-----------|
| fp32    | CPU_ONLY      | 5e153a39c4d77c7f   | 1.38  | run+process stable |
| fp32    | CPU_AND_GPU   | d8e41a1f771ca33e   | 1.28  | run+process stable |
| fp32    | CPU_AND_NE    | 5e153a39c4d77c7f   | 1.36  | = CPU (ANE not engaged) |
| fp32    | ALL           | d8e41a1f771ca33e   | 1.25  | scheduler → GPU |
| fp16    | CPU_ONLY      | 323687e47203ea8d   | 0.69  | run+process stable |
| fp16    | CPU_AND_GPU   | cb412e4f8c4c5442   | 0.66  | run+process stable |
| fp16    | CPU_AND_NE    | **8a7ca6f8438985dc** | **0.30** | run+process stable |
| fp16    | ALL           | 8a7ca6f8438985dc   | 0.30  | scheduler → ANE |

## Findings

1. **The ANE engaged, twice-witnessed**: fp16/CPU_AND_NE is 2.3× faster
   than CPU AND carries a digest distinct from both CPU and GPU paths —
   three execution families on one chip, each with its own numerics.
2. **Every family is deterministic** within-process and across fresh
   compiles/processes. The necessary condition for oracle digests holds.
3. **fp32 silently never reaches the ANE** (digest+timing identical to
   CPU). Family oracles must record program precision.
4. **`ALL` is scheduler-dependent** (chose GPU for fp32, ANE for fp16
   here): oracle contracts must pin explicit compute units, never ALL.

## The contract this enables (Ferrite verified-behavior on NPUs)

A pack targeting an NPU family declares (family, precision, compute-units)
and carries the family's oracle digests for its eval vectors — measured
once on family hardware, not derived. A fleet device of that family
re-runs the vectors and must reproduce the family oracle bit-exactly;
cross-family identity is neither promised nor needed. Open (needs more
hardware): whether digests hold across devices of one family (M5 vs M5
Max vs M4) and across OS/CoreML compiler updates — the oracle store must
version by (family, OS, compiler) until measured otherwise.
