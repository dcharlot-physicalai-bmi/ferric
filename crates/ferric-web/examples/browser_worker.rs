//! The heterogeneous fabric physically spanning the BROWSER. Native Rust starts a WebSocket bridge
//! serving the worker page, launches a headless browser at it, and — once the tab connects — treats
//! it as `Device::BrowserWorker`. A batched matmul dispatched from native is computed on the tab's
//! WebGPU and returned, validated against the local CPU. Local-only (needs a real browser); not in CI.
use ferric_tensor::sched::Device;
use ferric_tensor::ws;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| (((i as f32 * 0.21 + s).sin())) * 0.3).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() {
    let site = format!("{}/site", env!("CARGO_MANIFEST_DIR"));
    let bridge = ws::start(site);
    println!("bridge serving {} — launching headless browser worker…", bridge.url());

    let chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    let _ = std::fs::remove_dir_all("/tmp/ferric-bw");
    let mut child = Command::new(chrome)
        .args(["--headless=new", "--no-sandbox", "--enable-unsafe-webgpu",
               "--enable-features=Vulkan,WebGPU", "--use-angle=metal",
               "--user-data-dir=/tmp/ferric-bw", &bridge.url()])
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("launch chrome");

    // wait for the tab to connect and signal ready
    let t0 = Instant::now();
    let conn = loop {
        if let Some(c) = bridge.take_worker() { break c; }
        if t0.elapsed() > Duration::from_secs(30) { let _ = child.kill(); panic!("browser never connected"); }
        std::thread::sleep(Duration::from_millis(100));
    };
    let ready = conn.recv().expect("ready handshake");
    println!("  browser worker connected ({}B ready frame)", ready.len());

    // dispatch a batched matmul to the browser's WebGPU
    let dev = Device::BrowserWorker(conn);
    let (batch, m, k, n) = (4usize, 8, 6, 5);
    let a = seq(batch * m * k, 1.0);
    let b = seq(k * n, 2.0);
    let got = dev.bmm(&a, &b, batch, m, k, n);
    let cpu = Device::Cpu.bmm(&a, &b, batch, m, k, n);
    let diff = maxdiff(&got, &cpu);
    let _ = child.kill();

    println!("  matmul [{batch},{m},{k}]·[{k},{n}] on {} · max|browser-cpu| = {:.2e}", dev.name(), diff);
    assert!(diff < 1e-3, "browser worker mismatch {diff}");
    println!("✅ The scheduler physically spans the BROWSER — a tab's WebGPU served an op dispatched from native Rust");
}
