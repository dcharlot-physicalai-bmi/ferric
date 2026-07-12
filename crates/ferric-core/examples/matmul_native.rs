// Native validation: run the Ferric matmul kernel on this machine's GPU (Metal via wgpu) and
// confirm it matches the plain-Rust CPU reference. This proves the L0/L1 core computes correctly
// on a real fabric — the same kernel source that will run in the browser on WebGPU.
use ferric_core::{matmul_cpu, max_abs_diff, Context};

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = Context::new().await.expect("compute context");
    println!("Ferric · fabric: {:?} · adapter: {}", ctx.backend, ctx.adapter_name);

    // deterministic test matrices
    let (m, k, n) = (64usize, 48usize, 32usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();

    let t = std::time::Instant::now();
    let gpu = ctx.matmul(&a, &b, m as u32, k as u32, n as u32).await.expect("matmul");
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

    let cpu = matmul_cpu(&a, &b, m, k, n);
    let diff = max_abs_diff(&gpu, &cpu);

    println!("matmul {m}x{k}x{n} → gpu {:.2}ms · max|gpu-cpu| = {:.3e}", gpu_ms, diff);
    println!("gpu[:6] = {:?}", &gpu[..6.min(gpu.len())]);
    println!("cpu[:6] = {:?}", &cpu[..6.min(cpu.len())]);
    assert!(diff < 1e-4, "GPU and CPU disagree ({diff})");
    println!("✅ VALIDATED — Ferric matmul matches CPU on {:?}", ctx.backend);
}
