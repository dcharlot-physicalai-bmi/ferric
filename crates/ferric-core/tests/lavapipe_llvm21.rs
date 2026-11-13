//! Lavapipe in Mesa 26.0.8 JITs shaders with LLVM 21.1.8. On an AVX-512 host that
//! LLVM tunes as znver4 (the machines behind `ubuntu-latest`), it assigns
//! `AVX2_SETALLONES` to ymm16–31. VEX cannot encode those registers, so the
//! generated shader SIGSEGVs. LLVM 22 fixes the pseudo (llvm/llvm-project#169009);
//! codegen limited to AVX2 does not hit it.
//!
//! The software-GPU job therefore sets `GALLIUM_OVERRIDE_CPU_CAPS=avx2` and
//! `LP_NATIVE_VECTOR_WIDTH=256` before any Ferric process starts. These tests are
//! what makes deleting that pin fail `cargo test` on every machine, and what makes
//! running against that exact driver without the pin fail on the runner.

use std::path::PathBuf;

fn software_gpu_job() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push(".github/workflows/ci.yml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let start = yaml.find("\n  build-and-validate:\n").expect("software GPU job");
    let rest = &yaml[start + 1..];
    let end = rest.find("\n  macos-metal:\n").expect("metal job follows the software GPU job");
    rest[..end].to_string()
}

#[test]
fn the_software_gpu_job_pins_lavapipe_off_avx512() {
    let job = software_gpu_job();
    assert!(
        job.contains("GALLIUM_OVERRIDE_CPU_CAPS: avx2"),
        "the software GPU job lost the LLVM 21 AVX-512 pin"
    );
    assert!(
        job.contains("LP_NATIVE_VECTOR_WIDTH: \"256\""),
        "the software GPU job lost the 256-bit llvmpipe width pin"
    );
    assert!(
        !job.contains("continue-on-error"),
        "the pin must not sit next to a skipped failure"
    );
}

#[test]
fn llvm21_lavapipe_is_not_left_on_its_native_target() {
    let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
        eprintln!("no GPU context — the workflow pin is still checked by the other test");
        return;
    };
    let name = ctx.adapter_name;
    if !name.contains("llvmpipe") || !name.contains("LLVM 21") {
        eprintln!("adapter {name} is not the LLVM 21 lavapipe this pin exists for");
        return;
    }
    let caps = std::env::var("GALLIUM_OVERRIDE_CPU_CAPS").unwrap_or_default();
    assert!(
        caps == "avx2" || caps == "sse4_2" || caps == "sse2",
        "llvmpipe ({name}) still JITs with GALLIUM_OVERRIDE_CPU_CAPS={caps:?}; LLVM 21.1.8 SIGSEGVs on AVX-512"
    );
    let width = std::env::var("LP_NATIVE_VECTOR_WIDTH").unwrap_or_default();
    assert!(
        width == "128" || width == "256",
        "LP_NATIVE_VECTOR_WIDTH={width:?} still allows a 512-bit JIT on {name}"
    );
}
