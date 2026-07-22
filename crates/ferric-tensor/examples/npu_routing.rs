//! **NPU routing readiness** — proves the fabric's `Device::Npu` dispatch path and the adaptive `Planner`
//! route correctly across a CPU + GPU + NPU device set, so a real NPU execution-provider drops straight in.
//!
//! HONESTY (the same contract as `probe_npu`): no faked ANE dispatch, ever. On Apple Silicon the
//! **real CoreML ANE execution-provider** (`ferric_tensor::npu_coreml`) is auto-added by
//! `detect_devices` — but only after `MLComputePlan` confirms the Neural Engine runs the model's
//! matmul (Apple's own scheduler receipt). Where no real EP is available, this example falls back to
//! a **reference EP that computes on the CPU**, present only to exercise `Device::Npu` dispatch and
//! N-way routing — and never reported as the ANE.

use ferric_tensor::sched::{detect_devices, Device, Fabric, NpuBackend, Planner};
use std::sync::Arc;

/// A reference NPU execution-provider: implements `NpuBackend` but computes on the CPU. It exists ONLY to
/// verify the fabric's NPU dispatch + routing path — it is NOT the Apple Neural Engine and never claims to be.
struct ReferenceNpu;

fn naive_bmm(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for bt in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for l in 0..k {
                    acc += a[bt * m * k + i * k + l] * b[l * n + j];
                }
                out[bt * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

impl NpuBackend for ReferenceNpu {
    fn name(&self) -> String {
        "reference-EP (CPU-backed; for routing tests — NOT the ANE)".into()
    }
    fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        naive_bmm(a, b, batch, m, k, n)
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| 0.01 * (((i + salt) % 13) as f32 - 6.0)).collect()
}

async fn run() {
    let (mut devices, npu_info) = detect_devices().await;
    // real ANE present? detect_devices added it (plan-receipt-gated). Otherwise fall back to the
    // reference EP so the dispatch path is still exercised.
    let real_ane = devices.iter().any(|d| matches!(d, Device::Npu(_)));
    if !real_ane {
        devices.push(Device::Npu(Arc::new(ReferenceNpu)));
    }
    let fabric = Fabric::new(devices);

    println!("=== fabric ===");
    for d in &fabric.devices {
        println!("  {}", d.name());
    }
    println!("  probe: {} — {}, dispatchable now: {}", npu_info.name, npu_info.reachable_via, npu_info.dispatchable);
    if real_ane {
        println!("  → the NPU device above is the REAL Neural Engine (compute-plan receipt in `reachable via`)");
    } else {
        println!("  → no real EP here; the CPU-backed reference EP exercises the dispatch path only");
    }

    // find the NPU device index; confirm the Planner calibrates + can route to it
    let npu = fabric.devices.iter().position(|d| matches!(d, Device::Npu(_))).unwrap();
    let cpu = fabric.devices.iter().position(|d| matches!(d, Device::Cpu)).unwrap();

    let planner = Planner::calibrate(&fabric);
    println!("\n=== calibrated cost model over ALL devices (incl. the NPU EP) ===");
    for (name, overhead, tput) in planner.report() {
        println!("  {:<52} overhead {:>8.1} µs   throughput {:>7.2} GFLOP/s", name, overhead * 1e6, tput / 1e9);
    }

    // the routing/dispatch proof: run ops through the fabric, verify each vs the CPU oracle
    println!("\n=== dispatch + routing across the fabric (verify vs CPU oracle) ===");
    let mut worst = 0.0f32;
    let mut npu_ran = false;
    // (a) direct NPU dispatch: prove Device::Npu actually executes and is correct
    for &nn in &[16usize, 48, 96] {
        let (a, b) = (gen(nn * nn, 1), gen(nn * nn, 7));
        let via_npu = fabric.devices[npu].bmm(&a, &b, 1, nn, nn, nn);
        let oracle = fabric.devices[cpu].bmm(&a, &b, 1, nn, nn, nn);
        let e = max_abs_diff(&via_npu, &oracle);
        worst = worst.max(e);
        npu_ran = true;
        println!("  N={nn:>3}  direct NPU-EP dispatch → max err vs oracle {e:.1e}");
        // the real ANE is an fp16-input device (like Metal4); the reference EP is exact f32
        let tol = if real_ane { 1e-2 } else { 1e-4 };
        assert!(e < tol, "NPU dispatch must be correct (tol {tol})");
    }
    // (b) adaptive routing: the Planner picks a device per size across the whole fabric
    for &nn in &[8usize, 64, 256, 512] {
        let (a, b) = (gen(nn * nn, 2), gen(nn * nn, 5));
        let (res, dev) = planner.adaptive_bmm(&fabric, &a, &b, [1, nn, nn, nn]);
        let oracle = fabric.devices[cpu].bmm(&a, &b, 1, nn, nn, nn);
        let e = max_abs_diff(&res, &oracle);
        worst = worst.max(e);
        println!("  N={nn:>3}  adaptive route → {:<52} err {e:.1e}", fabric.devices[dev].name());
        assert!(e < 1e-2, "adaptive result must match the oracle");
    }

    println!("\n✅ the fabric dispatches through Device::Npu and the Planner routes across CPU+GPU+NPU (worst err {worst:.1e}).");
    if real_ane {
        println!("   That NPU is the REAL Apple Neural Engine — CoreML EP, ANE dispatch confirmed by MLComputePlan.");
    }
    println!("   Any other NPU (WebNN 'npu', DirectML/QNN) implements the same NpuBackend and is routed identically —");
    println!("   that platform binding is the remaining work; the routing/dispatch path is proven ready.");
    assert!(npu_ran && worst < 1e-2);
}

fn main() {
    pollster::block_on(run());
}
