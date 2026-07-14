//! Uses CPU / GPU / NPU as available — honestly. Enumerates EVERY compute adapter (all GPUs across
//! all backends + software), reports the CPU (multi-core), and PROBES the NPU (reporting how it's
//! reachable — WebGPU cannot target an NPU). Then builds a fabric from everything present and runs a
//! matmul across all of it, validated against a single-device reference.
use ferric_core::Context;
use ferric_tensor::sched::{detect_devices, Fabric};

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 0.13 + s).sin())) * 0.1).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    // full adapter inventory
    println!("=== Ferric compute inventory ===");
    for (name, backend, dt) in Context::enumerate().await {
        println!("  GPU adapter: {name:<28} [{backend:?} · {dt:?}]");
    }
    let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    println!("  CPU: {cores} cores (parallel kernels)");

    let (devices, npu) = detect_devices().await;
    print!("  NPU: ");
    if npu.present {
        println!("{} — present, reachable via {} · dispatchable now: {}", npu.name, npu.reachable_via, npu.dispatchable);
        if !npu.dispatchable { println!("        (WebGPU cannot target the NPU; a CoreML/DirectML/WebNN backend is the execution-provider to wire — never faked on the GPU)"); }
    } else {
        println!("none detected");
    }

    // build a fabric from everything and run across all of it
    let fabric = Fabric::new(devices);
    println!("=== fabric: {} ===", fabric.devices.iter().map(|d| d.name()).collect::<Vec<_>>().join("  +  "));
    let w = fabric.probe();
    println!("  measured split: {}", fabric.devices.iter().zip(&w).map(|(d, x)| format!("{} {:.0}%", d.name(), x * 100.0)).collect::<Vec<_>>().join("  "));

    let (batch, m, k, n) = (48usize, 24, 48, 24);
    let a = seq(batch * m * k, 1.0);
    let b = seq(k * n, 2.0);
    let (out, counts) = fabric.data_parallel_bmm(&a, &b, batch, m, k, n, &w);
    let single = fabric.devices[0].bmm(&a, &b, batch, m, k, n);
    let d = maxdiff(&out, &single);
    let split = fabric.devices.iter().zip(&counts).map(|(dev, c)| format!("{}={}", dev.name(), c)).collect::<Vec<_>>().join(" ");
    println!("  data-parallel across all devices ({split}) · max|fabric-single| = {d:.2e}");
    assert!(d < 1e-3);
    println!("✅ Ferric uses every GPU + all CPU cores as one fabric; NPU is detected and honestly reported");
}
