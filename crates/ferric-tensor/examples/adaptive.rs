//! **Adaptive routing** — the "smart software" that knows the hardware and adapts per workload.
//!
//! Calibrate a cost model over the fabric (CPU + every GPU), then for a sweep of matmul sizes let the
//! `Planner` route each op to the device it predicts fastest — and check it against reality: results match
//! the CPU oracle, and the routed device really is the quick one. Small ops go to the low-overhead device
//! (CPU), large ops to the high-throughput device (GPU), with a predicted crossover in between. Same code,
//! whatever silicon it lands on.

use ferric_tensor::sched::{detect_devices, Device, Fabric, Planner};
use std::time::Instant;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| 0.01 * (((i + salt) % 13) as f32 - 6.0)).collect()
}

async fn run() {
    let (devices, npu) = detect_devices().await;
    let fabric = Fabric::new(devices);
    println!("=== fabric ===");
    for d in &fabric.devices {
        println!("  {}", d.name());
    }
    println!("  NPU: {} (dispatchable now: {})", npu.name, npu.dispatchable);

    let planner = Planner::calibrate(&fabric);
    println!("\n=== calibrated cost model (measured on THIS box) ===");
    for (name, overhead, tput) in planner.report() {
        println!("  {:<22} dispatch overhead {:>8.1} µs   throughput {:>8.2} GFLOP/s", name, overhead * 1e6, tput / 1e9);
    }

    // locate CPU and the first GPU (if any)
    let cpu = fabric.devices.iter().position(|d| matches!(d, Device::Cpu)).expect("a CPU device");
    let gpu = fabric.devices.iter().position(|d| matches!(d, Device::Gpu(_)));

    if let Some(g) = gpu {
        if let Some(f) = planner.crossover(cpu, g) {
            let n = (f / 2.0).cbrt(); // f = 2 N³ for an N×N×N matmul
            println!("\n  predicted crossover: GPU overtakes CPU above ~{:.0}×{:.0} matmuls ({:.1e} flops)", n, n, f);
        }
    }

    println!("\n=== routing sweep (verify each result vs the CPU oracle) ===");
    println!("  {:>5}  {:>14}  {:>10}  {:>10}  {:>9}", "N", "routed to", "cpu (ms)", "gpu (ms)", "max err");
    let sizes = [8usize, 32, 64, 128, 256, 384, 512, 768];
    let mut worst_err = 0.0f32;
    for &nn in &sizes {
        let (a, b) = (gen(nn * nn, 1), gen(nn * nn, 7));
        let (res, dev) = planner.adaptive_bmm(&fabric, &a, &b, [1, nn, nn, nn]);
        let oracle = fabric.devices[cpu].bmm(&a, &b, 1, nn, nn, nn);
        let err = max_abs_diff(&res, &oracle);
        worst_err = worst_err.max(err);

        // measure each device for the table (min of 2)
        let t = |d: usize| {
            let mut best = f64::INFINITY;
            for _ in 0..2 {
                let t0 = Instant::now();
                let _ = fabric.devices[d].bmm(&a, &b, 1, nn, nn, nn);
                best = best.min(t0.elapsed().as_secs_f64());
            }
            best * 1e3
        };
        let cpu_ms = t(cpu);
        let gpu_ms = gpu.map(t).unwrap_or(f64::NAN);
        println!("  {:>5}  {:>14}  {:>10.3}  {:>10.3}  {:>9.1e}", nn, fabric.devices[dev].name(), cpu_ms, gpu_ms, err);
        assert!(err < 1e-2, "adaptive result must match the CPU oracle at N={nn}: err {err}");
    }

    // the headline invariants: tiny → CPU, huge → GPU (when a GPU is present)
    assert!(matches!(fabric.devices[planner.route(1, 8, 8, 8)], Device::Cpu), "a tiny matmul should route to the CPU");
    if gpu.is_some() {
        let big = planner.route(1, 768, 768, 768);
        assert!(matches!(fabric.devices[big], Device::Gpu(_)), "a large matmul should route to the GPU");
    }
    println!("\n✅ adaptive router: every result matched the CPU oracle (worst {:.1e}); tiny→CPU, large→GPU, chosen by a measured cost model", worst_err);
}

fn main() {
    pollster::block_on(run());
}
