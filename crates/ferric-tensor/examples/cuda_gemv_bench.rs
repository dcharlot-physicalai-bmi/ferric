//! **Native-tier GEMV microbench at the real decode shapes.** Which kernel/shape is furthest from the
//! bandwidth floor? That is where kernel work starts. Needs `FERRIC_CUDA=1` and an NVIDIA driver.
//!
//!   FERRIC_CUDA=1 cargo run -p ferric-tensor --release --example cuda_gemv_bench
#[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
fn main() {
    use ferric_tensor::{cuda, dtype::QMatrix};
    use std::sync::Arc;
    let Some(name) = cuda::device_name() else { eprintln!("no CUDA driver / FERRIC_CUDA unset. NOTHING measured."); return; };
    let ctx = Arc::new(pollster::block_on(ferric_core::Context::new()).expect("wgpu ctx (for QMatrix::from_bytes)"));
    println!("native    : {name} [CUDA tier]");
    let mut seed = 0x5eed_u64;
    let mut blk = |n: usize, ty: u32, bpb: usize| -> Vec<u8> {
        let mut b = vec![0u8; n];
        for x in b.iter_mut() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; *x = (seed >> 40) as u8; }
        for (i, c) in b.chunks_exact_mut(bpb).enumerate() {
            let d = half::f16::from_f32(0.01 + 0.003 * (i % 7) as f32).to_le_bytes();
            if ty == 14 { c[208..210].copy_from_slice(&d); } else { c[0..2].copy_from_slice(&d); c[2..4].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes()); }
        }
        b
    };
    // qwen3-0.6b Q5_K_M decode shapes (after consecutive-format grouping): (label, in, out, ggml type)
    let shapes: [(&str, usize, usize, u32); 6] = [
        ("attn q+k  (Q5_K)", 1024, 3072, 13), ("attn v    (Q6_K)", 1024, 1024, 14), ("attn wo   (Q5_K)", 2048, 1024, 13),
        ("ffn gate|up swiglu (Q5_K)", 1024, 6144, 13), ("ffn down  (Q6_K)", 3072, 1024, 14), ("lm_head   (Q6_K)", 1024, 151936, 14),
    ];
    println!("{:<28} {:>9} {:>10} {:>9}   (floor: bytes / ~192 GB/s)", "shape", "MiB", "us/call", "GB/s");
    for (label, inn, out, ty) in shapes {
        let (vals, bpb) = QMatrix::block_bytes(ty).unwrap();
        let bytes = blk(out * (inn / vals) * bpb, ty, bpb);
        let x: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.37).sin()).collect();
        let qm = QMatrix::from_bytes(&ctx, &bytes, ty, out, inn).expect("qmatrix");
        let w = qm.native_weight().expect("native mirror (FERRIC_CUDA set?)");
        let r = if label.contains("swiglu") { cuda::bench_swiglu(&w, &x, 200) } else { cuda::bench_gemv(&w, &x, 200) };
        if ty == 13 && !label.contains("swiglu") {
            if let Some((us8, _)) = cuda::bench_gemv_q8(&w, &x, 200) {
                println!("{:<28} {:>9.2} {us8:>10.1} {:>9.1}   ← int8 activations + dp4a (quantise incl.)",
                         format!("  ↳ q8x  {}", label.trim()), bytes.len() as f64 / 1048576.0, bytes.len() as f64 / (us8 * 1e-6) / 1e9);
            }
        }
        let Some((us, o)) = r else { println!("{label:<28}  FAILED"); continue };
        assert!(o.iter().all(|v| v.is_finite()));
        let mib = bytes.len() as f64 / 1048576.0;
        println!("{label:<28} {mib:>9.2} {us:>10.1} {:>9.1}   floor {:.1} us", bytes.len() as f64 / (us * 1e-6) / 1e9, bytes.len() as f64 / 192e9 * 1e6);
    }
}
#[cfg(not(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32"))))]
fn main() { eprintln!("cuda_gemv_bench: NVIDIA tier is linux/windows only."); }
