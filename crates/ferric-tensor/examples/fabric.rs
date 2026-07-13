// L7 heterogeneous scheduler: the GPU and the CPU as one fabric. Partition a batched matmul across
// both by measured throughput (run concurrently), and pipeline an MLP's layers across devices —
// both validated to equal single-device execution. Host-buffer transfers = the cloud/browser path.
use ferric_core::Context;
use ferric_tensor::sched::{Device, Fabric};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 0.13 + s).sin())) * 0.1).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let fabric = Fabric::new(vec![Device::Gpu(ctx.clone()), Device::Cpu]);
    println!("Fabric devices: {}", fabric.devices.iter().map(|d| d.name()).collect::<Vec<_>>().join("  +  "));

    let w = fabric.probe();
    println!("  measured split: {}", fabric.devices.iter().zip(&w).map(|(d, x)| format!("{} {:.0}%", d.name(), x * 100.0)).collect::<Vec<_>>().join("  "));

    // ---- data-parallel batched matmul across GPU+CPU vs single-device ----
    let (batch, m, k, n) = (48usize, 24, 48, 24);
    let a = seq(batch * m * k, 1.0);
    let b = seq(k * n, 2.0);
    let (out, counts) = fabric.data_parallel_bmm(&a, &b, batch, m, k, n, &w);
    let single = fabric.devices[0].bmm(&a, &b, batch, m, k, n); // all on GPU
    let d1 = maxdiff(&out, &single);
    let split = fabric.devices.iter().zip(&counts).map(|(d, c)| format!("{}={}", d.name(), c)).collect::<Vec<_>>().join(" ");
    let ok1 = d1 < 1e-3;
    println!("  {} data-parallel bmm ({batch} batch split {split})  max|fabric-single| = {:.2e}", if ok1 { "✅" } else { "❌" }, d1);

    // ---- pipeline an MLP across devices (layer 0 GPU, layer 1 CPU, ...) vs all-on-GPU ----
    let rows = 16usize;
    let dims = [32usize, 48, 48, 32];
    let x = seq(rows * dims[0], 3.0);
    let layers: Vec<(Vec<f32>, usize, usize)> = (0..3).map(|i| (seq(dims[i] * dims[i + 1], 4.0 + i as f32), dims[i], dims[i + 1])).collect();
    let (piped, trace) = fabric.pipeline_mlp(&x, rows, &layers);
    let gpu_only = Fabric::new(vec![Device::Gpu(ctx.clone())]).pipeline_mlp(&x, rows, &layers).0;
    let d2 = maxdiff(&piped, &gpu_only);
    let ok2 = d2 < 1e-3;
    println!("  {} pipeline MLP across [{}]  max|fabric-single| = {:.2e}", if ok2 { "✅" } else { "❌" }, trace.join(" → "), d2);

    println!("{}", if ok1 && ok2 { "✅ Heterogeneous scheduler: work partitioned across GPU+CPU, results identical to single-device" } else { "❌ scheduler mismatch" });
    assert!(ok1 && ok2);
}
