//! **The tracked `.ptx` is a build artifact, and it has gone stale twice.**
//!
//! `5a3fe60` (after the attention rewrite) and `b924c6a`, which added the three `FERRIC_CUDA_Q8X`
//! kernels to `cuda_decode.cu` and shipped an 8-entry `.ptx` for an 11-kernel `.cu`. The second was
//! live on `main`. It matters because `Driver::decode_kernels()` resolves **every** name and
//! `load_ptx` bails on the FIRST miss, so a checkout without a local `nvcc` got **no tier-2 decode
//! at all** — not a slower path, no path.
//!
//! ⛔ And it was invisible: the runtime falls back to WGSL and generates the SAME ids, so the only
//! trace was one stderr line (`cuModuleGetFunction failed: named symbol not found (500)`). Verified
//! on the RTX 4050 by putting the stale artifact back. **No correctness test could catch it** — which
//! is precisely why it shipped, and why this gate reads the artifacts instead of the output.
//!
//! This lives OUTSIDE `cuda.rs` on purpose: that module is `#![cfg(linux/windows)]`, so a test inside
//! it cannot run on the Mac where most commits are made. Pure text, no GPU, no driver, every platform.

/// ⚠ Keeps NO list of its own. Both sides are read from the artifacts, so adding a kernel to the
/// `.cu` without rebuilding the `.ptx` fails HERE rather than at someone else's runtime.
#[test]
fn tracked_ptx_exports_every_kernel_the_cu_defines() {
    let cu = include_str!("../src/cuda_decode.cu");
    let ptx = include_str!("../src/cuda_decode.ptx");

    let names: Vec<&str> = cu
        .lines()
        .filter_map(|l| {
            let l = l.trim_start();
            let r = l.strip_prefix("extern \"C\" ").unwrap_or(l).strip_prefix("__global__ void ")?;
            let n = r.split(|c: char| !(c.is_alphanumeric() || c == '_')).next()?;
            (!n.is_empty()).then_some(n)
        })
        .collect();

    // ⚠ A parser that silently matches nothing would make every assertion below vacuous
    // (vacuous-test mechanism: "a loop that asserts nothing"). Floor it on the real count.
    assert!(
        names.len() >= 8,
        "parsed only {} kernels out of cuda_decode.cu — the `__global__ void` parser drifted, and \
         this test would have passed while checking nothing",
        names.len()
    );

    // ⚠ `.entry q5k_gemv` is a PREFIX of `.entry q5k_gemv_q8`. Without the trailing `(` this passes
    // on the wrong kernel and a missing one reads as present (vacuous-test mechanism "the needle is
    // a PREFIX"). PTX emits `.visible .entry NAME(`.
    let missing: Vec<&str> =
        names.iter().copied().filter(|n| !ptx.contains(&format!(".entry {n}("))).collect();

    assert!(
        missing.is_empty(),
        "cuda_decode.ptx is STALE: {:?} defined in cuda_decode.cu but not exported by the tracked \
         PTX ({} of {} present). decode_kernels() resolves all of them and gives up on the first \
         miss, so this disables the ENTIRE native decode tier while still generating correct ids. \
         Rebuild it on a CUDA box:\n  \
         nvcc -O3 -arch=compute_75 -ptx crates/ferric-tensor/src/cuda_decode.cu -o crates/ferric-tensor/src/cuda_decode.ptx\n  \
         ptxas -arch=sm_89 -o /dev/null crates/ferric-tensor/src/cuda_decode.ptx",
        missing,
        names.len() - missing.len(),
        names.len()
    );
}
