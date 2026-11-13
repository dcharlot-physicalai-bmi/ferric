# Lavapipe on the GitHub Ubuntu 26.04 runner

`ubuntu-latest` on 2026-11-13 is Ubuntu 26.04.1. Its lavapipe is Mesa 26, built against LLVM 21.1.8. Two faults show up there and not on this repo's Ubuntu 24.04 lavapipe (Mesa 25.2.8, LLVM 20.1.2). Neither is fixed by skipping a test or changing an assertion.

## AVX-512 codegen

On an AVX-512 host that LLVM tunes as znver4, the JIT assigns the VEX-only `AVX2_SETALLONES` pseudo to ymm16–31. The shader SIGSEGVs. LLVM 22 fixes the pseudo ([llvm/llvm-project#169009](https://github.com/llvm/llvm-project/issues/169009), [mesa#16702](https://gitlab.freedesktop.org/mesa/mesa/-/issues/16702)). AVX2 codegen does not hit it.

`GALLIUM_OVERRIDE_CPU_CAPS` is a ceiling, not a feature name. In Mesa 26.0.8 `src/util/u_cpu_detect.c`, `avx` clears `has_avx512f` and leaves `has_avx` / `has_avx2` set. `avx2` matches no string, so it is a no-op and AVX-512 stays on. The documented x86 values are `nosse`, `sse`, `sse2`, `sse3`, `ssse3`, `sse4.1`, and `avx`. The software-GPU job sets `avx` and `LP_NATIVE_VECTOR_WIDTH=256`.

## Concurrent shader compilation

LLVMContext is not thread safe. Mesa says so in [mesa!29397](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/29397): sample-function compilation was racing the context used by shaders. ORC's thread-safe context lock covers adding a module to the JIT, not the IRBuilder phase, so concurrent compiles still race inside LLVM.

On this runner the race is visible as signal 11 with no assertion, inside a `ferric-llama` test process, after tests that do not compile those shaders have printed `ok`. Run [43456837947](https://github.com/dcharlot-physicalai-bmi/ferric/actions/runs/43456837947) died that way with the default cargo test thread pool. The same tree with one test thread at a time did not: run [43458775739](https://github.com/dcharlot-physicalai-bmi/ferric/actions/runs/43458775739) reports `ferric-llama` `168 passed; 0 failed; 1 ignored` and no SIGSEGV.

Minimal repro on that runner, with the AVX-512 pin already applied so this is the concurrency fault and not the one above:

```sh
export GALLIUM_OVERRIDE_CPU_CAPS=avx
export LP_NATIVE_VECTOR_WIDTH=256
cargo test -p ferric-llama --lib          # SIGSEGV
cargo test -p ferric-llama --lib -- --test-threads=1   # 168 passed, 1 ignored
```

The software-GPU job runs `cargo test --workspace --verbose -- --test-threads=1`. Every assertion still runs. The Metal job stays parallel; it does not use lavapipe. On the measurement VM (Mesa 25.2.8, LLVM 20.1.2) the workspace passed 1164 and ignored 39 both in parallel (`cargo test --workspace --no-fail-fast`) and under `--test-threads=1`. The serial runner does not drop a test.
