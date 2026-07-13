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
use std::sync::Arc;
use std::time::Instant;

/// A compute device in the fabric.
pub enum Device {
    Gpu(Arc<Context>),
    Cpu,
}

impl Device {
    pub fn name(&self) -> String {
        match self {
            Device::Gpu(c) => format!("GPU:{}", c.adapter_name),
            Device::Cpu => "CPU".into(),
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
            Device::Cpu => {
                let mut out = vec![0.0f32; batch * m * n];
                for bt in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut s = 0.0;
                            for l in 0..k { s += a[bt * m * k + i * k + l] * b[l * n + j]; }
                            out[bt * m * n + i * n + j] = s;
                        }
                    }
                }
                out
            }
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
            Device::Cpu => {
                let mut o = vec![0.0f32; rows * out];
                for i in 0..rows {
                    for j in 0..out {
                        let mut s = 0.0;
                        for l in 0..in_ { s += x[i * in_ + l] * w[l * out + j]; }
                        o[i * out + j] = s.max(0.0);
                    }
                }
                o
            }
        }
    }
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
        let assigned: usize = counts.iter().sum();
        if assigned != batch { counts[0] = counts[0] as isize as usize + (batch as isize - assigned as isize) as usize; }
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
