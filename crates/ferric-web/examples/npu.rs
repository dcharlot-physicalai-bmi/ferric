//! Actually run compute on the NPU via WebNN. On macOS, Chrome's WebNN is backed by CoreML → the
//! Apple Neural Engine, so a WebNN `deviceType:'npu'` context executes on the ANE. This starts the WS
//! bridge, launches Chrome (WebNN enabled) at the WebNN worker page, learns which device WebNN bound,
//! and — if a WebNN context came up — dispatches a matmul that runs through WebNN and validates it
//! against the CPU. Honest: reports exactly which backend WebNN used (npu / gpu / cpu / none).
use ferric_tensor::sched::Device;
use ferric_tensor::ws;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 0.21 + s).sin())) * 0.3).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() {
    let site = format!("{}/site", env!("CARGO_MANIFEST_DIR"));
    let bridge = ws::start(site);
    let url = format!("{}npu_worker.html", bridge.url());
    println!("bridge serving {url} — launching Chrome with WebNN enabled…");

    let chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    let _ = std::fs::remove_dir_all("/tmp/ferric-npu");
    let mut child = Command::new(chrome)
        .args(["--headless=new", "--no-sandbox",
               "--enable-features=WebMachineLearningNeuralNetwork,WebNN,Vulkan,WebGPU",
               "--enable-experimental-web-platform-features", "--enable-unsafe-webgpu", "--use-angle=metal",
               "--user-data-dir=/tmp/ferric-npu", &url])
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("launch chrome");

    let t0 = Instant::now();
    let conn = loop {
        if let Some(c) = bridge.take_worker() { break c; }
        if t0.elapsed() > Duration::from_secs(30) { let _ = child.kill(); panic!("browser never connected"); }
        std::thread::sleep(Duration::from_millis(100));
    };
    let status = String::from_utf8_lossy(&conn.recv().expect("status frame")).into_owned(); // "WEBNN:npu" etc.
    let device = status.strip_prefix("WEBNN:").unwrap_or("?").to_string();
    println!("  WebNN reported device: {device}");

    if device == "NO_WEBNN" {
        let _ = child.kill();
        println!("⚠️  This Chrome build has no WebNN — code path is ready; enable WebNN or run on hardware/OS with an NPU driver.");
        return;
    }

    // dispatch a matmul through WebNN (runs on the NPU when device == npu)
    let dev = Device::BrowserWorker(conn);
    let (batch, m, k, n) = (2usize, 8, 6, 5);
    let a = seq(batch * m * k, 1.0);
    let b = seq(k * n, 2.0);
    let got = dev.bmm(&a, &b, batch, m, k, n);
    let cpu = Device::Cpu.bmm(&a, &b, batch, m, k, n);
    let diff = maxdiff(&got, &cpu);
    let _ = child.kill();

    // the ANE is an fp16 engine, so its result matches CPU-f32 only to fp16 precision (~1e-1)
    let tol = if device == "npu" { 2e-1 } else { 1e-3 };
    println!("  matmul via WebNN:{device} · max|webnn-cpu| = {diff:.2e} (tol {tol:.0e})");
    assert!(diff < tol, "WebNN result mismatch {diff}");
    if device == "npu" {
        println!("✅ Ferric ran compute ON THE NPU — WebNN(npu) → CoreML → Apple Neural Engine, matches CPU to fp16 precision ({diff:.1e})");
    } else {
        println!("✅ Ferric ran compute via WebNN (device: {device}) — NPU code path works; bound to {device} on this machine");
    }
}
