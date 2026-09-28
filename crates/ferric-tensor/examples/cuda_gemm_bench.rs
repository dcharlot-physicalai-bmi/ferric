//! **Native-tier PREFILL GEMM microbench at the real prompt shapes** — µs per GEMM and useful TFLOPS
//! (2·M·N·K / time) for the tensor-core kernel of `cuda_prefill.cu`, per weight format. Needs
//! `FERRIC_CUDA=1` and an NVIDIA driver. `FERRIC_CUDA_PTX_DIR` points it at a variant PTX, so a kernel
//! change can be timed against the tracked one without rebuilding anything.
//!
//! ⚠ "Useful" FLOPs: the kernel runs every product TWICE on the tensor cores (the activation's hi and lo
//! f16 halves — it rounds nothing the f32 path does not), so the tensor-core work is 2x what is printed.
//!
//!   FERRIC_CUDA=1 cargo run -p ferric-tensor --release --example cuda_gemm_bench [-- M]
#[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
fn main() {
    use ferric_tensor::{cuda, dtype::QMatrix};
    use std::sync::Arc;
    let Some(name) = cuda::device_name() else { eprintln!("no CUDA driver / FERRIC_CUDA unset. NOTHING measured."); return; };
    let m: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let ctx = Arc::new(pollster::block_on(ferric_core::Context::new()).expect("wgpu ctx (for QMatrix::from_bytes)"));
    println!("native    : {name} [CUDA tier]   M = {m} prompt rows   PTX dir: {}",
             std::env::var("FERRIC_CUDA_PTX_DIR").unwrap_or_else(|_| "tracked".into()));
    let mut seed = 0x5eed_u64;
    let mut blk = |n: usize, ty: u32, bpb: usize| -> Vec<u8> {
        let mut b = vec![0u8; n];
        for x in b.iter_mut() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; *x = (seed >> 40) as u8; }
        for (i, c) in b.chunks_exact_mut(bpb).enumerate() {
            let d = half::f16::from_f32(0.01 + 0.003 * (i % 7) as f32).to_le_bytes();
            match ty {
                14 => c[208..210].copy_from_slice(&d),
                6 | 8 => c[0..2].copy_from_slice(&d),
                _ => { c[0..2].copy_from_slice(&d); c[2..4].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes()); }
            }
        }
        b
    };
    // (label, K = in, N = out, ggml type) — the weights a 512-row prompt multiplies, per model.
    let shapes: [(&str, usize, usize, u32); 11] = [
        ("L1B q+k     (Q4_K)", 2048, 2560, 12), ("L1B wo      (Q4_K)", 2048, 2048, 12),
        ("L1B gate|up (Q4_K)", 2048, 16384, 12), ("L1B down    (Q6_K)", 8192, 2048, 14),
        ("Q2.5 qkv    (Q5_0)", 896, 1152, 6), ("Q2.5 gate|up (Q5_0)", 896, 9728, 6), ("Q2.5 down   (Q8_0)", 4864, 896, 8),
        ("Q3 q+k      (Q5_K)", 1024, 3072, 13), ("Q3 gate|up  (Q5_K)", 1024, 6144, 13), ("Q3 down     (Q6_K)", 3072, 1024, 14),
        ("Q8 gate|up  (Q8_0)", 896, 9728, 8),
    ];
    let a: Vec<f32> = { let mut s = 0x1234_5678u64; (0..m * 8192).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect() };
    println!("{:<22} {:>10} {:>10} {:>12}", "shape", "MiB", "us/GEMM", "useful TFLOPS");
    for (label, k, n, ty) in shapes {
        let (vals, bpb) = QMatrix::block_bytes(ty).unwrap();
        let bytes = n * (k / vals) * bpb;
        let reps = (2 * 24 * 1024 * 1024 / bytes).max(1);
        let qms: Vec<_> = (0..reps).map(|_| QMatrix::from_bytes(&ctx, &blk(bytes, ty, bpb), ty, n, k).expect("qmatrix")).collect();
        let ws: Vec<_> = qms.iter().map(|q| q.native_weight().expect("native mirror")).collect();
        let iters = (4e9 / (2.0 * (m * n * k) as f64)).clamp(5.0, 200.0) as usize;
        let Some((us, c)) = cuda::bench_gemm(&ws, &a[..m * k], m, iters) else { println!("{label:<22} FAILED"); continue };
        assert!(c.iter().all(|v| v.is_finite()));
        println!("{label:<22} {:>10.2} {us:>10.1} {:>12.2}", bytes as f64 / 1048576.0, 2.0 * (m * n * k) as f64 / (us * 1e-6) / 1e12);
    }
}
#[cfg(not(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32"))))]
fn main() { eprintln!("cuda_gemm_bench: NVIDIA tier is linux/windows only."); }
