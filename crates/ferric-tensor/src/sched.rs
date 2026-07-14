//! L7 — the heterogeneous scheduler. Presents whatever compute is present as one fabric and
//! partitions work across it. Here the devices are a **GPU** (wgpu) and the **CPU** (plain Rust) —
//! two genuinely different fabrics on any machine. Data crosses device boundaries as host buffers,
//! which is exactly the format that would serialize to a cloud node or `postMessage` to a browser
//! worker, so the same `Device`/`Fabric` abstraction extends to cloud+local+browser.
//!
//! Two scheduling modes, both validated to equal single-device execution:
//!   • data-parallel — split a batched matmul across devices proportional to measured throughput,
//!     run concurrently (GPU on the main thread, CPU on a worker), concatenate.
//!   • pipeline — assign a model's layers to devices round-robin; activations hop across boundaries.

use crate::Tensor;
use ferric_core::Context;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

/// A compute device in the fabric.
pub enum Device {
    Gpu(Arc<Context>),
    Cpu,
    /// A remote worker reached over TCP — a cloud node, or (with a WS bridge) a browser tab. Work
    /// crosses the boundary as host buffers, exactly like the GPU/CPU hop, so it's the same fabric.
    Remote(String),
    /// A browser tab reached over a WebSocket, computing on the tab's WebGPU. Same op frames as
    /// Remote, different transport — this is the fabric physically spanning cloud+local+browser.
    BrowserWorker(Arc<crate::ws::WsConn>),
    /// An NPU reached through an execution-provider backend (CoreML/ANE, DirectML-QNN, OpenVINO,
    /// WebNN). WebGPU cannot target an NPU, so this only exists when a real backend is plugged in —
    /// it never falls back to the GPU/CPU silently.
    Npu(Arc<dyn NpuBackend>),
}

/// An NPU execution-provider backend. Implement over CoreML (Apple ANE), DirectML/QNN, OpenVINO, or
/// WebNN to make that NPU a real device in the fabric. Ferric ships the abstraction + detection; the
/// platform binding is the EP to wire — kept honest, no fake NPU dispatch on the GPU.
pub trait NpuBackend: Send + Sync {
    fn name(&self) -> String;
    fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32>;
    fn linear_relu(&self, x: &[f32], rows: usize, w: &[f32], in_: usize, out: usize) -> Vec<f32> {
        self.bmm(x, w, 1, rows, in_, out).iter().map(|v| v.max(0.0)).collect()
    }
}

impl Device {
    pub fn name(&self) -> String {
        match self {
            Device::Gpu(c) => format!("GPU:{}", c.adapter_name),
            Device::Cpu => "CPU".into(),
            Device::Remote(addr) => format!("Remote:{addr}"),
            Device::BrowserWorker(_) => "BrowserWorker".into(),
            Device::Npu(b) => format!("NPU:{}", b.name()),
        }
    }

    /// Batched matmul [batch,m,k] · [k,n] → [batch,m,n] on host data.
    pub fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        match self {
            Device::Gpu(ctx) => {
                let ta = Tensor::from_vec(ctx, a, &[batch, m, k]);
                let tb = Tensor::from_vec(ctx, b, &[k, n]);
                pollster::block_on(ta.matmul(&tb).to_vec())
            }
            Device::Remote(addr) => remote_call(addr, 0, &[batch as u32, m as u32, k as u32, n as u32], a, b),
            Device::BrowserWorker(conn) => browser_call(conn, 0, &[batch as u32, m as u32, k as u32, n as u32], a, b),
            Device::Npu(back) => back.bmm(a, b, batch, m, k, n),
            Device::Cpu => cpu_bmm(a, b, batch, m, k, n), // parallel across all cores
        }
    }

    /// One MLP layer: relu(x[rows,in] · w[in,out]).
    pub fn linear_relu(&self, x: &[f32], rows: usize, w: &[f32], in_: usize, out: usize) -> Vec<f32> {
        match self {
            Device::Gpu(ctx) => {
                let tx = Tensor::from_vec(ctx, x, &[rows, in_]);
                let tw = Tensor::from_vec(ctx, w, &[in_, out]);
                pollster::block_on(tx.matmul(&tw).relu().to_vec())
            }
            Device::Remote(addr) => remote_call(addr, 1, &[rows as u32, in_ as u32, out as u32, 0], x, w),
            Device::BrowserWorker(conn) => browser_call(conn, 1, &[rows as u32, in_ as u32, out as u32, 0], x, w),
            Device::Npu(back) => back.linear_relu(x, rows, w, in_, out),
            Device::Cpu => cpu_bmm(x, w, 1, rows, in_, out).iter().map(|v| v.max(0.0)).collect(),
        }
    }
}

/// Multi-threaded CPU batched matmul — splits the batch across all available cores.
fn cpu_bmm(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(batch.max(1));
    let chunk = batch.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (ci, slab) in out.chunks_mut(chunk * m * n).enumerate() {
            let lo = ci * chunk;
            s.spawn(move || {
                for bt_local in 0..slab.len() / (m * n) {
                    let bt = lo + bt_local;
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0.0;
                            for l in 0..k { acc += a[bt * m * k + i * k + l] * b[l * n + j]; }
                            slab[bt_local * m * n + i * n + j] = acc;
                        }
                    }
                }
            });
        }
    });
    out
}

/// What an NPU probe found. Honest: `dispatchable` is true only when a real NPU backend is wired.
/// WebGPU/wgpu CANNOT target an NPU — ANE/Intel/AMD/Qualcomm NPUs need CoreML/DirectML-QNN/OpenVINO/WebNN.
pub struct NpuInfo {
    pub present: bool,
    pub name: String,
    pub reachable_via: String,
    pub dispatchable: bool,
}

/// Probe the platform for an NPU — reports presence + how it's reachable, never that we run on it.
pub fn probe_npu() -> NpuInfo {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    { return NpuInfo { present: true, name: "Apple Neural Engine (ANE)".into(), reachable_via: "CoreML".into(), dispatchable: false }; }
    #[cfg(target_os = "windows")]
    { return NpuInfo { present: false, name: "Windows NPU (if present)".into(), reachable_via: "DirectML / QNN / OpenVINO EP".into(), dispatchable: false }; }
    #[allow(unreachable_code)]
    NpuInfo { present: false, name: "none".into(), reachable_via: "n/a".into(), dispatchable: false }
}

/// Enumerate a Device for EVERY GPU adapter present (all backends) + the CPU, and probe the NPU.
/// "Use CPU/GPU/NPU as available": every GPU is a device, CPU is always a device (all cores), NPU is
/// reported (added as a compute device only when a backend is wired — never faked).
#[cfg(not(target_arch = "wasm32"))]
pub async fn detect_devices() -> (Vec<Device>, NpuInfo) {
    let mut devices = Vec::new();
    for (idx, (_name, _backend, dt)) in Context::enumerate().await.into_iter().enumerate() {
        if dt != wgpu::DeviceType::Cpu {
            if let Ok(ctx) = Context::for_adapter(idx).await { devices.push(Device::Gpu(Arc::new(ctx))); }
        }
    }
    devices.push(Device::Cpu);
    (devices, probe_npu())
}

/// The fabric: a set of devices + a scheduler over them.
pub struct Fabric {
    pub devices: Vec<Device>,
}

impl Fabric {
    pub fn new(devices: Vec<Device>) -> Fabric { Fabric { devices } }

    /// Measure each device's throughput on a probe matmul → relative weights (sum = 1).
    pub fn probe(&self) -> Vec<f32> {
        let (batch, m, k, n) = (64usize, 32, 64, 32);
        let a = vec![0.01f32; batch * m * k];
        let b = vec![0.02f32; k * n];
        let mut rates = vec![];
        for d in &self.devices {
            let t0 = Instant::now();
            let _ = d.bmm(&a, &b, batch, m, k, n);
            let secs = t0.elapsed().as_secs_f32().max(1e-6);
            rates.push((batch * m * k * n) as f32 / secs); // ~flops/s
        }
        let sum: f32 = rates.iter().sum();
        rates.iter().map(|r| r / sum).collect()
    }

    /// Data-parallel: split the batch across devices by `weights`, run concurrently, concatenate.
    pub fn data_parallel_bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize, weights: &[f32]) -> (Vec<f32>, Vec<usize>) {
        // assign a contiguous batch range to each device, sized by weight
        let mut counts: Vec<usize> = weights.iter().map(|w| (w * batch as f32).round() as usize).collect();
        let assigned: isize = counts.iter().map(|&c| c as isize).sum();
        counts[0] = (counts[0] as isize + (batch as isize - assigned)).max(0) as usize; // reconcile rounding
        let mut ranges = vec![]; let mut off = 0;
        for &c in &counts { ranges.push((off, off + c)); off += c; }

        let mut out = vec![0.0f32; batch * m * n];
        std::thread::scope(|s| {
            let mut handles = vec![];
            for (di, &(lo, hi)) in ranges.iter().enumerate().skip(1) {
                if hi <= lo { continue; }
                let dev = &self.devices[di];
                let aslice = &a[lo * m * k..hi * m * k];
                handles.push((di, lo, hi, s.spawn(move || dev.bmm(aslice, b, hi - lo, m, k, n))));
            }
            // device 0 runs on this thread (overlaps the workers)
            let (lo0, hi0) = ranges[0];
            if hi0 > lo0 {
                let r = self.devices[0].bmm(&a[lo0 * m * k..hi0 * m * k], b, hi0 - lo0, m, k, n);
                out[lo0 * m * n..hi0 * m * n].copy_from_slice(&r);
            }
            for (_di, lo, hi, h) in handles {
                let r = h.join().unwrap();
                out[lo * m * n..hi * m * n].copy_from_slice(&r);
            }
        });
        (out, counts)
    }

    /// Pipeline: run an MLP whose layers are assigned to devices round-robin; activations cross
    /// device boundaries as host buffers. `layers` = (weight[in*out], in, out).
    pub fn pipeline_mlp(&self, x: &[f32], rows: usize, layers: &[(Vec<f32>, usize, usize)]) -> (Vec<f32>, Vec<String>) {
        let mut act = x.to_vec();
        let mut trace = vec![];
        for (li, (w, in_, out)) in layers.iter().enumerate() {
            let dev = &self.devices[li % self.devices.len()];
            act = dev.linear_relu(&act, rows, w, *in_, *out);
            trace.push(dev.name());
        }
        (act, trace)
    }
}

// ---------- Remote transport: a tiny binary op-server over TCP ----------
// Request:  op:u8 · dims[4]:u32 · lenA:u32 · A(f32 LE) · lenB:u32 · B(f32 LE)
// Response: len:u32 · result(f32 LE)
// The same frames would ride a WebSocket to a browser tab computing on WebGPU (Device::BrowserWorker).

fn wr_u32(v: &mut Vec<u8>, x: u32) { v.extend_from_slice(&x.to_le_bytes()); }
fn wr_f32s(v: &mut Vec<u8>, f: &[f32]) { wr_u32(v, f.len() as u32); v.extend_from_slice(bytemuck::cast_slice(f)); }
fn rd_exact(s: &mut impl Read, n: usize) -> std::io::Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    s.read_exact(&mut b)?;
    Ok(b)
}
fn rd_u32(s: &mut impl Read) -> std::io::Result<u32> {
    Ok(u32::from_le_bytes(rd_exact(s, 4)?.try_into().unwrap()))
}
fn rd_f32s(s: &mut impl Read) -> std::io::Result<Vec<f32>> {
    let n = rd_u32(s)? as usize;
    Ok(bytemuck::cast_slice(&rd_exact(s, n * 4)?).to_vec())
}

/// Build an op request frame (op · dims · A · B) — shared by the TCP and WebSocket transports.
fn op_frame(op: u8, dims: &[u32; 4], a: &[f32], b: &[f32]) -> Vec<u8> {
    let mut req = vec![op];
    for &d in dims { wr_u32(&mut req, d); }
    wr_f32s(&mut req, a);
    wr_f32s(&mut req, b);
    req
}

/// Client side: dispatch one op to a browser tab over the WebSocket; the tab computes on WebGPU.
fn browser_call(conn: &crate::ws::WsConn, op: u8, dims: &[u32; 4], a: &[f32], b: &[f32]) -> Vec<f32> {
    conn.send(&op_frame(op, dims, a, b)).expect("browser worker send");
    let resp = conn.recv().expect("browser worker response");
    bytemuck::cast_slice(&resp).to_vec()
}

/// Client side: send one op to a remote worker and block for the result.
fn remote_call(addr: &str, op: u8, dims: &[u32; 4], a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut req = vec![op];
    for &d in dims { wr_u32(&mut req, d); }
    wr_f32s(&mut req, a);
    wr_f32s(&mut req, b);
    let mut s = TcpStream::connect(addr).expect("remote worker unreachable");
    s.write_all(&req).unwrap();
    rd_f32s(&mut s).expect("remote worker response")
}

/// Handle one request on the worker, computing with `backend` (a local GPU or CPU device).
fn serve_one(s: &mut TcpStream, backend: &Device) -> std::io::Result<()> {
    let op = rd_exact(s, 1)?[0];
    let dims = [rd_u32(s)?, rd_u32(s)?, rd_u32(s)?, rd_u32(s)?];
    let a = rd_f32s(s)?;
    let b = rd_f32s(s)?;
    let out = match op {
        0 => backend.bmm(&a, &b, dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize),
        _ => backend.linear_relu(&a, dims[0] as usize, &b, dims[1] as usize, dims[2] as usize),
    };
    let mut resp = Vec::new();
    wr_f32s(&mut resp, &out);
    s.write_all(&resp)
}

/// Spin up a worker on an ephemeral localhost port, backed by `backend`; returns its address.
/// (Localhost stands in for a cloud node — same wire path across a real network.)
pub fn spawn_worker(backend: Device) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind worker");
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            if let Ok(mut s) = stream { let _ = serve_one(&mut s, &backend); }
        }
    });
    addr
}
