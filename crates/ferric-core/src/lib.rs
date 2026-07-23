//! Ferric core — L0 (portable device/fabric abstraction) + L1 (kernel dispatch), on `wgpu`.
//!
//! `wgpu` gives us one API over WebGPU (browser), Vulkan, Metal, DX12, and GL. This crate wraps it
//! into a tiny `Context` and the first real kernel (matmul), written ONCE in WGSL and run on any
//! fabric. The same source compiles to native (Metal/Vulkan/DX12) and to `wasm32` for the browser.
//! Numerics are validated against a plain-Rust CPU reference so "runs everywhere" also means
//! "computes the same everywhere".

use std::borrow::Cow;

pub type Result<T> = std::result::Result<T, String>;

mod kernels;
pub use kernels::cpu; // CPU references for validation
pub use kernels::rmsnorm_tree_cpu;
pub mod demo; // a small deterministic Llama-style LM, same code native + browser

/// A compute context bound to one GPU adapter on whatever fabric is available.
pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub backend: wgpu::Backend,
    pub adapter_name: String,
    /// Whether the `subgroups` WGSL feature is enabled on this device — lets kernels use
    /// `subgroupAdd`/`subgroupBroadcast` (a hardware warp op) instead of a shared-memory barrier
    /// tree. Shipped natively (Vulkan/Metal/DX12) and in browsers (Chrome 134+); kernels that use it
    /// must guard on this flag and keep a barrier-tree fallback for the portable floor.
    pub subgroups: bool,
    /// `maxStorageBufferBindingSize` for this device — the ceiling on a single bound storage buffer
    /// (WebGPU baseline 128 MB; Safari 256 MB–1 GB; native GPUs much higher). A weight above this must
    /// be sharded across buffers, so packed-quant loaders split oversized tensors along output rows.
    pub max_binding: u64,
    /// Whether `EXPERIMENTAL_COOPERATIVE_MATRIX` is enabled — the WGSL `coop_mat` types that lower to
    /// the hardware matrix units (Metal `simdgroup_matrix`, Vulkan `KHR_cooperative_matrix` → tensor
    /// cores / MFMA). Native-only (no browser spec yet); the path to real GEMM throughput on the
    /// M5's matrix hardware and NVIDIA tensor cores. Kernels using it are feature-gated + fp-order
    /// dependent, so like subgroups they stay off the bit-identical cross-fabric default path.
    pub coop_matrix: bool,
    /// Whether `SHADER_F16` is enabled — `enable f16;` + `array<f16>` in WGSL. Required for the NVIDIA
    /// tensor-core coop path: the RTX 4050 (and Intel) enumerate only f16-input cooperative-matrix
    /// configs (A/B = f16, C = f32), never f32×f32, so mixed-precision coop needs native f16 storage.
    pub shader_f16: bool,
}

/// An f32 tensor living in GPU memory. Ops chain Tensor→Tensor with no host readback until `to_vec`,
/// so a whole model runs on-device — this is the L2/L3 substrate (the graph executor works on these).
pub struct Tensor {
    pub buf: wgpu::Buffer,
    pub shape: Vec<usize>,
}
impl Tensor {
    pub fn len(&self) -> usize { self.shape.iter().product() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl Context {
    /// Submit an empty batch and wait for the device to go idle — commits any pending buffer
    /// initializations. Call periodically during huge allocation bursts (tens of GB across tens of
    /// thousands of `create_buffer_init` calls, e.g. loading a mixture-of-experts model), where Metal
    /// otherwise silently drops later buffers' contents (they read back as zeros, with no error).
    pub fn flush(&self) {
        self.queue.submit(std::iter::empty());
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Whether the cooperative-matrix GEMM path produces **correct** results here. Metal only.
    /// Metal (MSL → simdgroup_matrix) is exact f32 — verified with constant operands (a=0.01, b=0.02,
    /// every output = K·2e-4, relΔ ~1e-5 across M=8…2048, square and non-square). On Vulkan the kernel
    /// now *compiles* (after let-binding the coop pointer indices to dodge the naga "Expression is not
    /// cached" panic) but every `coopLoad`/`coopMultiplyAdd`/`coopStore` chain returns **all zeros** on
    /// the RTX 4050 — both the RB and plain 8×8 kernels, at every shape. A separate, unfixed naga
    /// SPIR-V codegen bug for KHR_cooperative_matrix, distinct from the compile panic. The old
    /// "9.4/21.7 TFLOP/s on NVIDIA" numbers were zero-output false passes: coop_gemm.rs validated coop
    /// vs naive with sin inputs that cancel to ~1e-2 under a 6e-2 threshold, so an all-zero result
    /// "passed". Gated off on Vulkan until the coop ops actually compute. (The cross-fabric MODEL parity
    /// is unaffected — it runs the scalar quant matmul, never coop.)
    pub fn coop_gemm_ok(&self) -> bool {
        self.coop_matrix && matches!(self.backend, wgpu::Backend::Metal)
    }

    /// Whether the **16×16 f16-input** cooperative-matrix path is usable (NVIDIA tensor cores / Intel
    /// XMX). Those vendors enumerate only f16 A/B configs (A=B=f16, C=f32), never f32×f32, and at
    /// 16×16 (NVIDIA) — so the Metal 8×8-f32 kernel matches no supported config there and runs as
    /// zeros. This path converts A/B to f16 and uses `coop_mat16x16<f16>` with an f32 accumulator.
    /// Vulkan-only (Metal keeps the exact-f32 8×8 path); needs both coop matrix and f16.
    pub fn coop16_ok(&self) -> bool {
        self.coop_matrix && self.shader_f16 && matches!(self.backend, wgpu::Backend::Vulkan)
    }

    /// Whether the dequant-tile→`coopLoad` quant-coop prefill path is worth taking here. The kernels
    /// dequantize a weight tile into shared memory and `coopLoad` it onto the matrix unit.
    /// **Measured on an RTX 4050 (Vulkan), M = 64…2048:** it is now **correct** (rel|Δ| = 0.0 vs the
    /// scalar kernel — the old column-major coop-load garbage was fixed by transposing in the dequant so
    /// the load is row-major) but **exactly 1.0× at every M**. Both the coop path and the scalar split-K
    /// path plateau at ~1.2 TFLOP/s: reading + unpacking the 2-bit weights into shared is the ceiling, so
    /// the tensor cores (9.4 TFLOP/s on f32-coop here) sit idle waiting on dequant, and no M amortizes it
    /// (dequant scales with the weight, not with M). So on NVIDIA there is simply **no win to gate in** —
    /// this is Metal-only because Metal's scalar quant path is the slower baseline coop beats (1.6–3.3× on
    /// real models), not because NVIDIA is broken. The only tensor-core-prefill route on NVIDIA is a
    /// one-time dequant to f16 in VRAM (8× footprint) feeding the f32/f16 coop GEMM — a separate trade.
    /// `FERRIC_COOP_SHARED_FORCE` overrides the gate for debugging.
    pub fn coop_shared_ok(&self) -> bool {
        self.coop_matrix && (matches!(self.backend, wgpu::Backend::Metal) || std::env::var("FERRIC_COOP_SHARED_FORCE").is_ok())
    }
}

impl Context {
    /// Acquire the best available compute device on this fabric (native GPU or browser WebGPU).
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no compute adapter: {e:?}"))?;
        let info = adapter.get_info();
        // Opt into subgroups when the adapter has it (native GPUs + modern browsers); harmless to
        // omit where absent, and the flag lets kernels pick a subgroup path or the barrier fallback.
        let af = adapter.features();
        let subgroups = af.contains(wgpu::Features::SUBGROUP);
        let coop_matrix = af.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let shader_f16 = af.contains(wgpu::Features::SHADER_F16);
        let max_binding = adapter.limits().max_storage_buffer_binding_size as u64;
        let mut want = wgpu::Features::empty();
        if subgroups { want |= wgpu::Features::SUBGROUP; }
        if coop_matrix { want |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX; }
        if shader_f16 { want |= wgpu::Features::SHADER_F16; }
        // wgpu gates EXPERIMENTAL_* features behind an explicit acknowledgment token (WIP APIs that
        // may have UB). We opt in only when the adapter advertises cooperative matrix.
        let experimental = if coop_matrix {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ferric"),
                required_features: want,
                // request the adapter's real limits: native gets big buffers (big models); in a
                // browser this resolves to the WebGPU baseline, so cross-fabric portability holds.
                required_limits: adapter.limits(),
                experimental_features: experimental,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no compute device: {e:?}"))?;
        Ok(Self { device, queue, backend: info.backend, adapter_name: info.name, subgroups, max_binding, coop_matrix, shader_f16 })
    }

    /// Enumerate EVERY compute adapter present (all GPUs across all backends + software/CPU adapters),
    /// so the scheduler can use all of them, not just one. Returns (name, backend, device_type).
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn enumerate() -> Vec<(String, wgpu::Backend, wgpu::DeviceType)> {
        let instance = wgpu::Instance::default();
        instance.enumerate_adapters(wgpu::Backends::all()).await.iter()
            .map(|a| { let i = a.get_info(); (i.name, i.backend, i.device_type) })
            .collect()
    }

    /// Build a Context on a specific enumerated adapter (by index into `enumerate()`), so each GPU in
    /// the machine can be its own device in the fabric.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn for_adapter(idx: usize) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        let adapter = adapters.into_iter().nth(idx).ok_or_else(|| format!("no adapter at index {idx}"))?;
        let info = adapter.get_info();
        let subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        let max_binding = wgpu::Limits::downlevel_defaults().max_storage_buffer_binding_size as u64;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ferric"),
                required_features: if subgroups { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() },
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no compute device: {e:?}"))?;
        Ok(Self { device, queue, backend: info.backend, adapter_name: info.name, subgroups, max_binding, coop_matrix: false, shader_f16: false })
    }

    pub(crate) fn storage(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }
    pub(crate) fn storage_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }
    /// Copy a buffer into a fresh one of the same byte length (used for Reshape/Cast — data unchanged).
    pub(crate) fn copy_buf(&self, src: &wgpu::Buffer, len: usize) -> wgpu::Buffer {
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dup"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &dst, 0, (len * 4) as u64);
        self.queue.submit([enc.finish()]);
        dst
    }
    pub(crate) fn uniform_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
    /// An empty output storage buffer of `len` f32s (readable back via `readback`).
    pub(crate) fn out_buffer(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
    /// Compile a WGSL compute pipeline (entry `main`, auto bind-group layout).
    /// Kernels that reference `det_` get the deterministic-math preamble
    /// prepended — the exact-ops transcendentals that keep every kernel on the
    /// cross-fabric bit-identical path (see kernels::DET_MATH_WGSL).
    pub(crate) fn pipeline(&self, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
        let src = if wgsl.contains("det_") {
            format!("{}\n{wgsl}", kernels::DET_MATH_WGSL)
        } else {
            wgsl.to_string()
        };
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }
    /// Dispatch a compute pipeline over `binds` with `groups` workgroups (queue-ordered before readback).
    pub(crate) fn dispatch(&self, pipeline: &wgpu::ComputePipeline, binds: &[&wgpu::Buffer], groups: (u32, u32, u32)) {
        let entries: Vec<wgpu::BindGroupEntry> = binds.iter().enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.queue.submit([enc.finish()]);
    }
    /// Read `len` f32s back from a storage buffer (works native + wasm).
    pub(crate) async fn readback(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        let bytes = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = flume::bounded(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv_async().await.map_err(|e| format!("recv: {e:?}"))?.map_err(|e| format!("map: {e:?}"))?;
        let data = staging.slice(..).get_mapped_range().map_err(|e| format!("map range: {e:?}"))?;
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// C = A(m×k) · B(k×n), row-major, f32. One kernel, any fabric.
    pub async fn matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Result<Vec<f32>> {
        assert_eq!(a.len(), (m * k) as usize);
        assert_eq!(b.len(), (k * n) as usize);
        let out_len = (m * n) as usize;
        let out_bytes = (out_len * 4) as u64;

        let a_buf = self.storage("a", a);
        let b_buf = self.storage("b", b);
        let dims_buf = self.uniform_u32("dims", &[m, k, n, 0]);
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            // Same det-math preamble rule as Context::pipeline — this raw path
            // must never drift from the portable-det kernels.
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{}\n{MATMUL_WGSL}",
                kernels::DET_MATH_WGSL
            ))),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul-bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = (m + 15) / 16;
            let gy = (n + 15) / 16;
            pass.dispatch_workgroups(gx, gy, 1);
        }
        // copy to a mappable staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit([enc.finish()]);

        // async readback that works on native (poll blocks) and wasm (browser drives the queue)
        let (tx, rx) = flume::bounded(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv_async().await.map_err(|e| format!("recv: {e:?}"))?.map_err(|e| format!("map: {e:?}"))?;
        let data = staging.slice(..).get_mapped_range().map_err(|e| format!("map range: {e:?}"))?;
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

pub(crate) const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>; // m, k, n, _

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = dims.x; let k = dims.y; let n = dims.z;
    let row = gid.x; let col = gid.y;
    if (row >= m || col >= n) { return; }
    var acc: f32 = 0.0;
    // portable-det: barriered MAC — forces the plain rounded sequence even on
    // compilers we can't configure (the browser's). Value-identical under
    // strict compilers; see kernels::DET_MATH_WGSL.
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        acc = det_bar(acc + det_bar(a[row * k + i] * b[i * n + col], dims.w), dims.w);
    }
    out[row * n + col] = acc;
}
"#;

/// Plain-Rust CPU reference (the source of truth for validation).
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

/// Max absolute difference between two equal-length vectors.
pub fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}
