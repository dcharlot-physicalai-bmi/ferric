//! **Metal 4 tensor-unit GEMM backend** — the M5-class tensor cores as a fabric device. Runs the
//! MetalPerformancePrimitives cooperative `tensor_ops::matmul2d` kernel (fp16 inputs → fp32 accumulate)
//! through the MTL4 command model, at ~23–31 TFLOP/s on an M5 Max vs ~0.1 TFLOP/s for the portable
//! wgpu-WGSL matmul (~280×). Host-boundary contract like every scheduler device: `&[f32]` in, `Vec<f32>`
//! out — upload converts to fp16, so this is an explicitly **reduced-precision** backend (the standard ML
//! GEMM trade); results match an fp16-input oracle to fp32 rounding, not the f32 oracle to 1e-7. The
//! adaptive [`crate::sched::Planner`] measures it like any device and routes accordingly.
//!
//! Per-call overhead is amortized with a **shape-keyed cache**: buffers, tensor views, argument tables,
//! residency set, command allocator and command buffer are all built once per shape and reused while the
//! same shape repeats (the common training/inference pattern) — a repeat call is just convert-upload →
//! dispatch → wait → readback. Conversion is row-wise via `half`'s slice converter (hardware `fcvt` on
//! aarch64), writing straight into the mapped buffers. Arbitrary shapes are handled by zero-padding M to
//! 64 and N to 32 (the kernel's tile) and slicing the result back out; batches encode one dispatch per
//! element over offset tensor views in a single submit.
//!
//! The precompiled kernel is embedded (`metal4_gemm.metallib`; source `metal4_gemm.metal` alongside —
//! rebuild with `xcrun metal -std=metal4.0 -c … && xcrun metallib …`).
#![cfg(all(target_os = "macos", not(target_arch = "wasm32")))]

use core::ptr::NonNull;
use half::f16;
use half::slice::HalfFloatSliceExt;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const METALLIB: &[u8] = include_bytes!("metal4_gemm.metallib");
const TILE_M: usize = 64;
const TILE_N: usize = 32;

type Obj<T> = Retained<ProtocolObject<T>>;

// Everything rebuilt on a shape change and reused while the shape repeats.
struct ShapeCache {
    key: (usize, usize, usize, usize), // batch, m, k, n
    buf_a: Obj<dyn MTLBuffer>,
    buf_b: Obj<dyn MTLBuffer>,
    buf_c: Obj<dyn MTLBuffer>,
    argtabs: Vec<Obj<dyn MTL4ArgumentTable>>,
    _tensors: Vec<Obj<dyn MTLTensor>>, // kept alive for the argtables' resource IDs
    rset: Obj<dyn MTLResidencySet>,
    alloc: Obj<dyn MTL4CommandAllocator>,
    cb: Obj<dyn MTL4CommandBuffer>,
}

// Resident-path objects (wgpu-buffer operands), rebuilt on shape change like ShapeCache. The wgpu
// buffers themselves are per-call: their GPU addresses go into the convert/unpad argument tables and
// a small per-call residency set; everything else here is shape-static.
struct ResidentCache {
    key: (usize, usize, usize, usize, bool, u32), // batch, m, k, n, b-transposed, act
    // scratch kept alive for the argtables' baked GPU addresses (f16 A/B pads, f32 C pad)
    _scr: [Obj<dyn MTLBuffer>; 3],
    at_cv_a: Obj<dyn MTL4ArgumentTable>, // padConvert A: wgpu src → scr_a
    at_cv_b: Obj<dyn MTL4ArgumentTable>, // padConvert B: wgpu src → scr_b
    at_up: Obj<dyn MTL4ArgumentTable>,   // unpad: scr_c → wgpu out
    mm_argtabs: Vec<Obj<dyn MTL4ArgumentTable>>,
    _params: Vec<Obj<dyn MTLBuffer>>,  // kernel param blocks (shape-static, written at build)
    _tensors: Vec<Obj<dyn MTLTensor>>, // kept alive for the argtables' resource IDs
    rset: Obj<dyn MTLResidencySet>,
}

/// A deferred resident GEMM — the op-DAG's unit of work. Owns `Arc`s to its wgpu buffers so a
/// pooled output can't be recycled (and an input can't be dropped) before the op executes.
pub struct ResOp {
    pub a: std::sync::Arc<wgpu::Buffer>,
    pub a_off: usize,
    pub b: std::sync::Arc<wgpu::Buffer>,
    pub b_off: usize,
    pub out: std::sync::Arc<wgpu::Buffer>,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub bt: bool,
    pub act: u32,
}

impl ResOp {
    fn key(&self) -> (usize, usize, usize, usize, bool, u32) {
        (self.batch, self.m, self.k, self.n, self.bt, self.act)
    }
}

/// A deferred resident conv2d — same ownership contract as [`ResOp`].
pub struct ConvOp {
    pub x: std::sync::Arc<wgpu::Buffer>,
    pub x_off: usize,
    pub w: std::sync::Arc<wgpu::Buffer>,
    pub w_off: usize,
    pub out: std::sync::Arc<wgpu::Buffer>,
    pub n: usize,
    pub h: usize,
    pub wd: usize,
    pub c: usize,
    pub kh: usize,
    pub kw: usize,
    pub o: usize,
    pub stride: (usize, usize),
    pub pad: (usize, usize),
}

impl ConvOp {
    #[allow(clippy::type_complexity)]
    fn key(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize) {
        (self.n, self.h, self.wd, self.c, self.kh, self.kw, self.o, self.stride.0, self.stride.1, self.pad.0, self.pad.1)
    }
}

/// A deferred external-queue op — the op-DAG's unit of work (GEMM or conv).
pub enum ExtOp {
    Gemm(ResOp),
    Conv(ConvOp),
}

// Resident conv2d objects — same shape-static doctrine as ResidentCache. The conv PSO itself is
// per-config (the MPP descriptor is fully constexpr), runtime-compiled and cached in `conv_psos`.
struct ConvCache {
    key: (usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize), // n,h,w,c,kh,kw,o,sh,sw,ph,pw
    at_cv_a: Obj<dyn MTL4ArgumentTable>, // padConvertNHWC: wgpu x → scr_a
    at_cv_w: Obj<dyn MTL4ArgumentTable>, // padConvert (identity): wgpu w → scr_w
    at_up: Obj<dyn MTL4ArgumentTable>,   // unpad: scr_c → wgpu out
    at_conv: Obj<dyn MTL4ArgumentTable>, // the three tensor views
    pso_conv: Obj<dyn MTLComputePipelineState>,
    dims: ConvDims,
    _scr: [Obj<dyn MTLBuffer>; 3],
    _params: Vec<Obj<dyn MTLBuffer>>,
    _tensors: Vec<Obj<dyn MTLTensor>>,
    rset: Obj<dyn MTLResidencySet>,
}

#[derive(Clone, Copy)]
struct ConvDims {
    ho: usize,
    wo: usize,
    tiles_x: usize,
    tiles_y: usize,
    n_a: usize, // activation element count (pad-convert dispatch)
    n_w: usize, // weight element count
    n_o: usize, // output element count (unpad dispatch)
}

const CONV_TILE: usize = 16;

/// The Metal-4 tensor-unit GEMM device. Create with [`Metal4Gemm::new`]; `None` when the platform has no
/// Metal-4 tensor support (kernel load or pipeline creation fails), so detection stays honest.
pub struct Metal4Gemm {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTL4CommandQueue>,
    pso: Obj<dyn MTLComputePipelineState>,
    pso_bt: Obj<dyn MTLComputePipelineState>,
    pso_pad: Obj<dyn MTLComputePipelineState>,
    pso_padnhwc: Obj<dyn MTLComputePipelineState>,
    pso_unpad: Obj<dyn MTLComputePipelineState>,
    event: Obj<dyn MTLSharedEvent>,
    ticket: AtomicU64,
    cache: Mutex<Vec<ShapeCache>>,      // MRU-front, capped — alternating shapes must not thrash
    rcache: Mutex<Vec<ResidentCache>>,  // (a real model's q/k/v projections are 3 different shapes)
    ccache: Mutex<Vec<ConvCache>>,
    // Allocator/command-buffer pairs for resident runs. A multi-run flush encodes each run into
    // its own pair and chains them with QUEUE-side event waits (ordering proven by the
    // floor_probe test), paying the ~170 µs commit→completion floor ONCE per flush instead of
    // once per run (~35 µs marginal per chained cb, measured). Guarded by the mutex; the flush's
    // final host wait completes every pair before release, so reuse is race-free.
    mcb: Mutex<Vec<(Obj<dyn MTL4CommandAllocator>, Obj<dyn MTL4CommandBuffer>)>>,
    conv_psos: Mutex<std::collections::HashMap<(usize, usize, usize, usize, usize, usize, usize, usize, usize), Obj<dyn MTLComputePipelineState>>>,
    pub adapter_name: String,
}

// SAFETY: MTLDevice / MTL4CommandQueue / MTLComputePipelineState / MTLSharedEvent are thread-safe Metal
// objects. The non-thread-safe encode objects (allocator / command buffer / argument tables) live inside
// `cache` and are only ever touched while holding its Mutex, and every call waits for GPU completion
// before releasing the lock — so no cross-thread concurrent use is possible.
unsafe impl Send for Metal4Gemm {}
unsafe impl Sync for Metal4Gemm {}

// MRU-front bounded cache lookup: find-or-build, move to front, evict past `cap`. Returns a
// reference into the vec's front slot (stable for the borrow's lifetime — the caller holds the lock).
fn lru_entry<'a, T, K: PartialEq + Copy>(
    v: &'a mut Vec<T>,
    key: K,
    cap: usize,
    build: impl FnOnce(K) -> T,
    key_of: impl Fn(&T) -> K,
) -> &'a T {
    if let Some(pos) = v.iter().position(|e| key_of(e) == key) {
        if pos != 0 {
            let e = v.remove(pos);
            v.insert(0, e);
        }
    } else {
        v.insert(0, build(key));
        v.truncate(cap);
    }
    &v[0]
}

fn make_extents(vals: &[isize]) -> Retained<MTLTensorExtents> {
    unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), vals.len(), vals.as_ptr()) }
        .expect("tensor extents")
}

// The descriptor's storage mode must MATCH the backing buffer's (Metal validates it) — wgpu storage
// buffers are Private, our own staging buffers Shared, so the mode is the caller's to state.
fn tensor_desc(dt: MTLTensorDataType, dims: &[isize], strides: &[isize], mode: MTLStorageMode) -> Retained<MTLTensorDescriptor> {
    let d = MTLTensorDescriptor::new();
    d.setDataType(dt);
    d.setUsage(MTLTensorUsage::Compute);
    d.setDimensions(&make_extents(dims));
    d.setStrides(Some(&make_extents(strides)));
    d.setStorageMode(mode);
    d
}

/// The raw `MTLBuffer` behind a wgpu buffer (Metal backend only) — the interop handle that lets the
/// tensor units read/write wgpu-resident data with no host copy. Caller owns queue synchronization.
pub fn wgpu_buffer_raw(buf: &wgpu::Buffer) -> Option<Obj<dyn MTLBuffer>> {
    let hal = unsafe { buf.as_hal::<wgpu::hal::api::Metal>() }?;
    Some(hal.raw_handle().clone())
}

impl Metal4Gemm {
    /// Probe + build the tensor-unit device on the system default `MTLDevice` (standalone use —
    /// host-boundary `bmm`). Returns `None` if unavailable (no faked capability).
    pub fn new() -> Option<Metal4Gemm> {
        Self::from_raw_device(MTLCreateSystemDefaultDevice()?)
    }

    /// Build on **wgpu's own** `MTLDevice`, so wgpu-resident buffers are directly usable as tensor
    /// operands (Metal resources are per-`MTLDevice`; sharing requires the same device object).
    pub fn for_wgpu(device: &wgpu::Device) -> Option<Metal4Gemm> {
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }?;
        Self::from_raw_device(hal.raw_device().clone())
    }

    /// Build the queue/pipeline/event on an existing `MTLDevice`.
    pub fn from_raw_device(device: Obj<dyn MTLDevice>) -> Option<Metal4Gemm> {
        let path = std::env::temp_dir().join("ferric_metal4_gemm.metallib");
        std::fs::write(&path, METALLIB).ok()?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(path.to_str()?));
        let lib = device.newLibraryWithURL_error(&url).ok()?;
        let mut psos = ["matMul", "matMulBT", "padConvert", "padConvertNHWC", "unpad"].into_iter().map(|name| {
            let func = lib.newFunctionWithName(&NSString::from_str(name))?;
            device.newComputePipelineStateWithFunction_error(&func).ok()
        });
        let (pso, pso_bt, pso_pad, pso_padnhwc, pso_unpad) =
            (psos.next()??, psos.next()??, psos.next()??, psos.next()??, psos.next()??);
        let queue = device.newMTL4CommandQueue()?;
        let event = device.newSharedEvent()?;
        let adapter_name = device.name().to_string();
        Some(Metal4Gemm {
            device,
            queue,
            pso,
            pso_bt,
            pso_pad,
            pso_padnhwc,
            pso_unpad,
            event,
            ticket: AtomicU64::new(0),
            cache: Mutex::new(Vec::new()),
            rcache: Mutex::new(Vec::new()),
            ccache: Mutex::new(Vec::new()),
            mcb: Mutex::new(Vec::new()),
            conv_psos: Mutex::new(std::collections::HashMap::new()),
            adapter_name,
        })
    }

    // Completion wait: bounded spin on the shared event's counter first — completion latency is
    // ~100 µs-scale and the kernel wakeup inside waitUntilSignaledValue costs tens of µs, which a
    // short GEMM feels. Falls back to the blocking wait for anything longer than 2 ms.
    fn wait_ticket(&self, ticket: u64) {
        let t0 = std::time::Instant::now();
        while self.event.signaledValue() < ticket {
            if t0.elapsed().as_millis() >= 2 {
                assert!(self.event.waitUntilSignaledValue_timeoutMS(ticket, 60_000), "Metal4 GEMM timed out");
                return;
            }
            std::hint::spin_loop();
        }
    }

    // Build (or rebuild) all shape-dependent objects.
    fn build_cache(&self, batch: usize, m: usize, k: usize, n: usize) -> ShapeCache {
        let mp = m.div_ceil(TILE_M) * TILE_M;
        let np = n.div_ceil(TILE_N) * TILE_N;
        let dev = &self.device;

        let buf_a = dev.newBufferWithLength_options(batch * mp * k * 2, MTLResourceOptions::StorageModeShared).expect("A buf");
        let buf_b = dev.newBufferWithLength_options(k * np * 2, MTLResourceOptions::StorageModeShared).expect("B buf");
        let buf_c = dev.newBufferWithLength_options(batch * mp * np * 4, MTLResourceOptions::StorageModeShared).expect("C buf");
        // zero once: pad rows/cols stay zero forever (uploads only overwrite the data regions)
        unsafe {
            std::ptr::write_bytes(buf_a.contents().as_ptr() as *mut u8, 0, batch * mp * k * 2);
            std::ptr::write_bytes(buf_b.contents().as_ptr() as *mut u8, 0, k * np * 2);
            std::ptr::write_bytes(buf_c.contents().as_ptr() as *mut u8, 0, batch * mp * np * 4);
        }

        let t_b = unsafe {
            buf_b.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[np as isize, k as isize], &[1, np as isize], MTLStorageMode::Shared),
                0,
            )
        }
        .expect("B tensor");
        let mut argtabs = Vec::with_capacity(batch);
        let mut tensors: Vec<Obj<dyn MTLTensor>> = Vec::with_capacity(batch * 2 + 1);
        for bt in 0..batch {
            let t_a = unsafe {
                buf_a.newTensorWithDescriptor_offset_error(
                    &tensor_desc(MTLTensorDataType::Float16, &[k as isize, mp as isize], &[1, k as isize], MTLStorageMode::Shared),
                    bt * mp * k * 2,
                )
            }
            .expect("A tensor");
            let t_c = unsafe {
                buf_c.newTensorWithDescriptor_offset_error(
                    &tensor_desc(MTLTensorDataType::Float32, &[np as isize, mp as isize], &[1, np as isize], MTLStorageMode::Shared),
                    bt * mp * np * 4,
                )
            }
            .expect("C tensor");
            let atd = MTL4ArgumentTableDescriptor::new();
            atd.setMaxBufferBindCount(3);
            let at = self.device.newArgumentTableWithDescriptor_error(&atd).expect("arg table");
            unsafe {
                at.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
                at.setResource_atBufferIndex(t_b.gpuResourceID(), 1);
                at.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
            }
            argtabs.push(at);
            tensors.push(t_a);
            tensors.push(t_c);
        }
        tensors.push(t_b);

        // residency: buffers AND tensor objects (tensor metadata must be resident too)
        let rset = dev.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).expect("residency set");
        rset.addAllocation(ProtocolObject::from_ref(&*buf_a));
        rset.addAllocation(ProtocolObject::from_ref(&*buf_b));
        rset.addAllocation(ProtocolObject::from_ref(&*buf_c));
        for t in &tensors {
            rset.addAllocation(ProtocolObject::from_ref(&**t));
        }
        rset.commit();
        rset.requestResidency();

        let alloc = dev.newCommandAllocator().expect("allocator");
        let cb = dev.newCommandBuffer().expect("command buffer");
        ShapeCache { key: (batch, m, k, n), buf_a, buf_b, buf_c, argtabs, _tensors: tensors, rset, alloc, cb }
    }

    /// Batched matmul `[batch,m,k] · [k,n] → [batch,m,n]` on the tensor units (fp16 inputs, fp32 out).
    pub fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut guard = self.cache.lock().unwrap();
        let c = lru_entry(&mut guard, (batch, m, k, n), 8, |key| self.build_cache(key.0, key.1, key.2, key.3), |e| e.key);
        let mp = m.div_ceil(TILE_M) * TILE_M;
        let np = n.div_ceil(TILE_N) * TILE_N;

        // upload: row-wise vectorized f32→f16 straight into the mapped buffers (pads stay zero)
        unsafe {
            let pa = std::slice::from_raw_parts_mut(c.buf_a.contents().as_ptr() as *mut f16, batch * mp * k);
            for bt in 0..batch {
                for i in 0..m {
                    pa[bt * mp * k + i * k..bt * mp * k + (i + 1) * k]
                        .convert_from_f32_slice(&a[bt * m * k + i * k..bt * m * k + (i + 1) * k]);
                }
            }
            let pb = std::slice::from_raw_parts_mut(c.buf_b.contents().as_ptr() as *mut f16, k * np);
            for i in 0..k {
                pb[i * np..i * np + n].convert_from_f32_slice(&b[i * n..(i + 1) * n]);
            }
        }

        // encode: allocator reset (all prior work completed under this lock), re-begin, re-attach residency
        c.alloc.reset();
        c.cb.beginCommandBufferWithAllocator(&c.alloc);
        c.cb.useResidencySet(&c.rset); // must be re-called after every begin
        let enc = c.cb.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.pso);
        let tew = self.pso.threadExecutionWidth();
        let grid = MTLSize { width: np / TILE_N, height: mp / TILE_M, depth: 1 };
        let tg = MTLSize { width: tew * 4, height: 1, depth: 1 };
        for at in &c.argtabs {
            enc.setArgumentTable(Some(at));
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
        enc.endEncoding();
        c.cb.endCommandBuffer();

        let bufs = [NonNull::from(&*c.cb)];
        unsafe { self.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ticket = self.ticket.fetch_add(1, Ordering::SeqCst) + 1;
        let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
        self.queue.signalEvent_value(ev, ticket);
        self.wait_ticket(ticket);

        // readback: slice [m, n] out of the padded [mp, np] per batch
        let mut out = vec![0.0f32; batch * m * n];
        unsafe {
            let pc = c.buf_c.contents().as_ptr() as *const f32;
            for bt in 0..batch {
                for i in 0..m {
                    std::ptr::copy_nonoverlapping(pc.add(bt * mp * np + i * np), out.as_mut_ptr().add(bt * m * n + i * n), n);
                }
            }
        }
        out
    }

    // Build (or rebuild) the resident-path shape-static objects. `bt` = B is a [n, k] weight
    // consumed transposed (NT); `act` = fused activation code applied by the unpad epilogue.
    fn build_rcache(&self, batch: usize, m: usize, k: usize, n: usize, bt: bool, act: u32) -> ResidentCache {
        let mp = m.div_ceil(TILE_M) * TILE_M;
        let np = n.div_ceil(TILE_N) * TILE_N;
        let dev = &self.device;

        let scr_a = dev.newBufferWithLength_options(batch * mp * k * 2, MTLResourceOptions::StorageModeShared).expect("A scratch");
        let scr_b = dev.newBufferWithLength_options(k * np * 2, MTLResourceOptions::StorageModeShared).expect("B scratch"); // same byte size either orientation
        let scr_c = dev.newBufferWithLength_options(batch * mp * np * 4, MTLResourceOptions::StorageModeShared).expect("C scratch");
        // zero once: pads stay zero (padConvert only writes data regions; matmul rewrites all of C)
        unsafe {
            std::ptr::write_bytes(scr_a.contents().as_ptr() as *mut u8, 0, batch * mp * k * 2);
            std::ptr::write_bytes(scr_b.contents().as_ptr() as *mut u8, 0, k * np * 2);
            std::ptr::write_bytes(scr_c.contents().as_ptr() as *mut u8, 0, batch * mp * np * 4);
        }

        // kernel param blocks — shape-only, written once here
        let mk_params = |vals: &[u32]| {
            let b = dev
                .newBufferWithLength_options(vals.len() * 4, MTLResourceOptions::StorageModeShared)
                .expect("param buf");
            unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), b.contents().as_ptr() as *mut u32, vals.len()) };
            b
        };
        let par_a = mk_params(&[(batch * m * k) as u32, (m * k) as u32, (mp * k) as u32]);
        // NT: W's [n, k] rows are contiguous — the padded copy is the identity map, pads = tail rows
        let par_b = if bt {
            mk_params(&[(n * k) as u32, (n * k) as u32, (n * k) as u32])
        } else {
            mk_params(&[(k * n) as u32, n as u32, np as u32])
        };
        let par_c = mk_params(&[(batch * m * n) as u32, n as u32, np as u32, m as u32, mp as u32, act]);

        // convert/unpad argument tables: scratch + param addresses are static; the wgpu-side
        // addresses (index 0 of cv_a/cv_b, index 1 of up) are set per call.
        let mk_at = |count: usize| {
            let atd = MTL4ArgumentTableDescriptor::new();
            atd.setMaxBufferBindCount(count);
            self.device.newArgumentTableWithDescriptor_error(&atd).expect("arg table")
        };
        let (at_cv_a, at_cv_b, at_up) = (mk_at(3), mk_at(3), mk_at(3));
        unsafe {
            at_cv_a.setAddress_atIndex(scr_a.gpuAddress(), 1);
            at_cv_a.setAddress_atIndex(par_a.gpuAddress(), 2);
            at_cv_b.setAddress_atIndex(scr_b.gpuAddress(), 1);
            at_cv_b.setAddress_atIndex(par_b.gpuAddress(), 2);
            at_up.setAddress_atIndex(scr_c.gpuAddress(), 0);
            at_up.setAddress_atIndex(par_c.gpuAddress(), 2);
        }

        // matmul tensor views over the scratch buffers — NN keeps the host-boundary layout; NT views
        // the weight rows as extents [k, np] (k innermost), which matmul2d consumes transposed
        let t_b = unsafe {
            let desc = if bt {
                tensor_desc(MTLTensorDataType::Float16, &[k as isize, np as isize], &[1, k as isize], MTLStorageMode::Shared)
            } else {
                tensor_desc(MTLTensorDataType::Float16, &[np as isize, k as isize], &[1, np as isize], MTLStorageMode::Shared)
            };
            scr_b.newTensorWithDescriptor_offset_error(&desc, 0)
        }
        .expect("B tensor");
        let mut mm_argtabs = Vec::with_capacity(batch);
        let mut tensors: Vec<Obj<dyn MTLTensor>> = Vec::with_capacity(batch * 2 + 1);
        for bt in 0..batch {
            let t_a = unsafe {
                scr_a.newTensorWithDescriptor_offset_error(
                    &tensor_desc(MTLTensorDataType::Float16, &[k as isize, mp as isize], &[1, k as isize], MTLStorageMode::Shared),
                    bt * mp * k * 2,
                )
            }
            .expect("A tensor");
            let t_c = unsafe {
                scr_c.newTensorWithDescriptor_offset_error(
                    &tensor_desc(MTLTensorDataType::Float32, &[np as isize, mp as isize], &[1, np as isize], MTLStorageMode::Shared),
                    bt * mp * np * 4,
                )
            }
            .expect("C tensor");
            let at = mk_at(3);
            unsafe {
                at.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
                at.setResource_atBufferIndex(t_b.gpuResourceID(), 1);
                at.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
            }
            mm_argtabs.push(at);
            tensors.push(t_a);
            tensors.push(t_c);
        }
        tensors.push(t_b);

        let rset = dev.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).expect("residency set");
        for b in [&scr_a, &scr_b, &scr_c, &par_a, &par_b, &par_c] {
            rset.addAllocation(ProtocolObject::from_ref(&**b));
        }
        for t in &tensors {
            rset.addAllocation(ProtocolObject::from_ref(&**t));
        }
        rset.commit();
        rset.requestResidency();

        ResidentCache {
            key: (batch, m, k, n, bt, act),
            _scr: [scr_a, scr_b, scr_c],
            at_cv_a,
            at_cv_b,
            at_up,
            mm_argtabs,
            _params: vec![par_a, par_b, par_c],
            _tensors: tensors,
            rset,
        }
    }

    /// **Resident** batched matmul `[batch,m,k]·[k,n]`: operands and result are **wgpu buffers**
    /// (f32, contiguous), and the whole pipeline — pad+f16-convert, `matmul2d` on the tensor units,
    /// unpad — runs as one MTL4 command buffer on wgpu's own `MTLDevice`. No byte crosses the host.
    /// Offsets are in bytes.
    ///
    /// The caller must ensure prior wgpu work producing `a`/`b` has completed (submit **then** poll —
    /// staged uploads only flush on submit) — this call blocks until the GPU finishes, so wgpu
    /// submissions issued after it return safely observe `out`. Returns `None` (touching nothing) if
    /// the buffers aren't Metal-backed.
    #[allow(clippy::too_many_arguments)] // a GEMM signature: three operands + offsets + four dims
    pub fn bmm_resident(
        &self,
        a: &std::sync::Arc<wgpu::Buffer>,
        a_off: usize,
        b: &std::sync::Arc<wgpu::Buffer>,
        b_off: usize,
        out: &std::sync::Arc<wgpu::Buffer>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<()> {
        let op = ResOp { a: a.clone(), a_off, b: b.clone(), b_off, out: out.clone(), batch, m, k, n, bt: false, act: 0 };
        self.run_resident_many(&[ExtOp::Gemm(op)])
    }

    /// **Resident** linear layer `y = act(x·Wᵀ)` with W in the HF `[out_f, in]` layout, consumed
    /// transposed by the tensor units (NT — no transpose materialization) and the activation fused
    /// into the unpad epilogue (codes match `matmul_bt_act`: 0 id, 1 relu, 2 silu, 3 gelu,
    /// 4 sigmoid). Same residency/sync contract as [`Self::bmm_resident`].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_resident(
        &self,
        x: &std::sync::Arc<wgpu::Buffer>,
        x_off: usize,
        w: &std::sync::Arc<wgpu::Buffer>,
        w_off: usize,
        out: &std::sync::Arc<wgpu::Buffer>,
        rows: usize,
        inn: usize,
        out_f: usize,
        act: u32,
    ) -> Option<()> {
        let op = ResOp { a: x.clone(), a_off: x_off, b: w.clone(), b_off: w_off, out: out.clone(), batch: 1, m: rows, k: inn, n: out_f, bt: true, act };
        self.run_resident_many(&[ExtOp::Gemm(op)])
    }

    /// Execute a sequence of deferred external-queue ops (GEMMs and convs) — the op-DAG executor.
    /// Ops split into runs at key repeats (same-config ops share scratch and argument tables, so
    /// they cannot coexist in one command buffer) and at cache capacity; each run encodes every
    /// op's stages into ONE MTL4 command buffer with inter-op barriers, then pays a single
    /// commit + completion wait — instead of a full round-trip per op.
    pub fn run_resident_many(&self, ops: &[ExtOp]) -> Option<()> {
        #[derive(PartialEq, Clone)]
        enum K {
            G((usize, usize, usize, usize, bool, u32)),
            C((usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize)),
        }
        let key_of = |op: &ExtOp| match op {
            ExtOp::Gemm(g) => K::G(g.key()),
            ExtOp::Conv(c) => K::C(c.key()),
        };
        // split into runs at key repeats / cap
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < ops.len() {
            let mut keys: Vec<K> = Vec::new();
            let mut j = i;
            while j < ops.len() && keys.len() < 8 && !keys.contains(&key_of(&ops[j])) {
                keys.push(key_of(&ops[j]));
                j += 1;
            }
            runs.push((i, j));
            i = j;
        }
        // encode each run into its own cb; chain with queue-side event waits (GPU-ordered, proven
        // by floor_probe); ONE host wait at the end — the dispatch floor is paid once per flush.
        const POOL: usize = 8;
        let mut pairs = self.mcb.lock().unwrap();
        // hold BOTH cache locks across the whole chained flush: committed-but-unfinished cbs
        // reference cache-owned scratch/argtables, and an LRU eviction mid-flight would free them
        let mut rguard = self.rcache.lock().unwrap();
        let mut cguard = self.ccache.lock().unwrap();
        let mut last_ticket = 0u64;
        for (ri, &(a, b)) in runs.iter().enumerate() {
            if ri > 0 && ri % POOL == 0 {
                self.wait_ticket(last_ticket); // recycle the pair pool
            }
            let idx = ri % POOL;
            if pairs.len() <= idx {
                pairs.push((self.device.newCommandAllocator()?, self.device.newCommandBuffer()?));
            }
            let (alloc, cb) = &pairs[idx];
            self.encode_run(alloc, cb, &ops[a..b], &mut rguard, &mut cguard)?;
            if ri > 0 {
                self.queue.waitForEvent_value(ProtocolObject::from_ref(&*self.event), last_ticket);
            }
            let bufs = [NonNull::from(&**cb)];
            unsafe { self.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
            last_ticket = self.ticket.fetch_add(1, Ordering::SeqCst) + 1;
            self.queue.signalEvent_value(ProtocolObject::from_ref(&*self.event), last_ticket);
        }
        if last_ticket > 0 {
            self.wait_ticket(last_ticket);
        }
        // safe to evict now — nothing is in flight
        rguard.truncate(16);
        cguard.truncate(8);
        Some(())
    }

    fn encode_run(
        &self,
        alloc: &ProtocolObject<dyn MTL4CommandAllocator>,
        cb: &ProtocolObject<dyn MTL4CommandBuffer>,
        ops: &[ExtOp],
        rguard: &mut Vec<ResidentCache>,
        cguard: &mut Vec<ConvCache>,
    ) -> Option<()> {
        // raw handles first (fallible) — also what the per-run residency set needs
        let raws: Vec<[Obj<dyn MTLBuffer>; 3]> = ops
            .iter()
            .map(|op| -> Option<[Obj<dyn MTLBuffer>; 3]> {
                match op {
                    ExtOp::Gemm(g) => Some([wgpu_buffer_raw(&g.a)?, wgpu_buffer_raw(&g.b)?, wgpu_buffer_raw(&g.out)?]),
                    ExtOp::Conv(c) => Some([wgpu_buffer_raw(&c.x)?, wgpu_buffer_raw(&c.w)?, wgpu_buffer_raw(&c.out)?]),
                }
            })
            .collect::<Option<Vec<_>>>()?;

        // phase 1: ensure every op's cache entry exists (mutating; distinct keys ≤ run cap < caps)
        for op in ops {
            match op {
                ExtOp::Gemm(g) => {
                    // no eviction here — earlier runs of this flush may still be in flight on the
                    // GPU referencing their entries; the flush truncates after its final wait
                    let k = g.key();
                    let _ = lru_entry(rguard, k, usize::MAX, |key| self.build_rcache(key.0, key.1, key.2, key.3, key.4, key.5), |e| e.key);
                }
                ExtOp::Conv(cv) => {
                    let key = cv.key();
                    if !cguard.iter().any(|e| e.key == key) {
                        let built = self.build_conv_cache(cv.n, cv.h, cv.wd, cv.c, cv.kh, cv.kw, cv.o, cv.stride, cv.pad)?;
                        cguard.insert(0, built);
                    }
                }
            }
        }
        // phase 2: per-op address binding + the per-run residency set
        let prset = self.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).ok()?;
        for (op, raw) in ops.iter().zip(&raws) {
            match op {
                ExtOp::Gemm(g) => {
                    let c = rguard.iter().find(|e| e.key == g.key()).unwrap();
                    unsafe {
                        c.at_cv_a.setAddress_atIndex(raw[0].gpuAddress() + g.a_off as u64, 0);
                        c.at_cv_b.setAddress_atIndex(raw[1].gpuAddress() + g.b_off as u64, 0);
                        c.at_up.setAddress_atIndex(raw[2].gpuAddress(), 1);
                    }
                }
                ExtOp::Conv(cv) => {
                    let cc = cguard.iter().find(|e| e.key == cv.key()).unwrap();
                    unsafe {
                        cc.at_cv_a.setAddress_atIndex(raw[0].gpuAddress() + cv.x_off as u64, 0);
                        cc.at_cv_w.setAddress_atIndex(raw[1].gpuAddress() + cv.w_off as u64, 0);
                        cc.at_up.setAddress_atIndex(raw[2].gpuAddress(), 1);
                    }
                }
            }
            for b in raw {
                prset.addAllocation(ProtocolObject::from_ref(&**b));
            }
        }
        prset.commit();
        prset.requestResidency();

        alloc.reset();
        cb.beginCommandBufferWithAllocator(alloc);
        for op in ops {
            match op {
                ExtOp::Gemm(g) => cb.useResidencySet(&rguard.iter().find(|e| e.key == g.key()).unwrap().rset),
                ExtOp::Conv(cv) => cb.useResidencySet(&cguard.iter().find(|e| e.key == cv.key()).unwrap().rset),
            }
        }
        cb.useResidencySet(&prset);
        let enc = cb.computeCommandEncoder()?;
        for (idx, op) in ops.iter().enumerate() {
            if idx > 0 {
                // conservative inter-op dependency barrier (op N may read op N-1's output)
                enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            }
            match op {
                ExtOp::Gemm(g) => self.encode_stages(&enc, rguard.iter().find(|e| e.key == g.key()).unwrap(), g),
                ExtOp::Conv(cv) => self.encode_conv_stages(&enc, cguard.iter().find(|e| e.key == cv.key()).unwrap()),
            }
        }
        enc.endEncoding();
        cb.endCommandBuffer();
        Some(())
    }

    // One conv's three stages (NHWC pad-convert + weight convert, tiled convolution2d, unpad),
    // with the stage barriers, into an open encoder.
    fn encode_conv_stages(&self, enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>, cc: &ConvCache) {
        let d = cc.dims;
        let tg256 = MTLSize { width: 256, height: 1, depth: 1 };
        let lin = |count: usize| MTLSize { width: count.div_ceil(256), height: 1, depth: 1 };
        enc.setComputePipelineState(&self.pso_padnhwc);
        enc.setArgumentTable(Some(&cc.at_cv_a));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(d.n_a), tg256);
        enc.setComputePipelineState(&self.pso_pad);
        enc.setArgumentTable(Some(&cc.at_cv_w));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(d.n_w), tg256);
        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        enc.setComputePipelineState(&cc.pso_conv);
        enc.setArgumentTable(Some(&cc.at_conv));
        let tew = cc.pso_conv.threadExecutionWidth();
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: d.tiles_x, height: d.tiles_y, depth: cc.key.0 },
            MTLSize { width: tew * 4, height: 1, depth: 1 },
        );
        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        enc.setComputePipelineState(&self.pso_unpad);
        enc.setArgumentTable(Some(&cc.at_up));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(d.n_o), tg256);
    }

    // One op's three stages (pad-convert both operands, tensor-unit GEMM, unpad-gather), with the
    // stage barriers, into an open encoder.
    fn encode_stages(&self, enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>, c: &ResidentCache, op: &ResOp) {
        let (batch, m, k, n) = (op.batch, op.m, op.k, op.n);
        let mp = m.div_ceil(TILE_M) * TILE_M;
        let np = n.div_ceil(TILE_N) * TILE_N;
        let tg256 = MTLSize { width: 256, height: 1, depth: 1 };
        let lin = |count: usize| MTLSize { width: count.div_ceil(256), height: 1, depth: 1 };
        enc.setComputePipelineState(&self.pso_pad);
        enc.setArgumentTable(Some(&c.at_cv_a));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(batch * m * k), tg256);
        enc.setArgumentTable(Some(&c.at_cv_b));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(k * n), tg256);
        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        let mm = if op.bt { &self.pso_bt } else { &self.pso };
        enc.setComputePipelineState(mm);
        let tew = mm.threadExecutionWidth();
        let grid = MTLSize { width: np / TILE_N, height: mp / TILE_M, depth: 1 };
        let tgm = MTLSize { width: tew * 4, height: 1, depth: 1 };
        for at in &c.mm_argtabs {
            enc.setArgumentTable(Some(at));
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgm);
        }
        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        enc.setComputePipelineState(&self.pso_unpad);
        enc.setArgumentTable(Some(&c.at_up));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(batch * m * n), tg256);
    }

    // Runtime-compile (and cache) the per-config conv PSO. The MPP descriptor is fully constexpr,
    // so every (dims, kernel, stride) combination is its own pipeline; the contract baked here is
    // the one the conv_probe tests pin: VALID windows via set_offsets(k/2 + tile·stride), per-tile
    // dest slices, grid over ceil(dest/16) tiles.
    #[allow(clippy::too_many_arguments)]
    fn conv_pso(
        &self,
        n: usize,
        hp: usize,
        wp: usize,
        c: usize,
        kh: usize,
        kw: usize,
        o: usize,
        sh: usize,
        sw: usize,
    ) -> Option<Obj<dyn MTLComputePipelineState>> {
        let key = (n, hp, wp, c, kh, kw, o, sh, sw);
        if let Some(p) = self.conv_psos.lock().unwrap().get(&key) {
            return Some(p.clone());
        }
        let t = CONV_TILE;
        let (cx, cy) = (kw / 2, kh / 2);
        let (a_bind, a_use, c_z) = if n == 1 {
            ("", "A", "0")
        } else {
            ("auto tA = A.slice(0, 0, 0, int(tgid.z));", "tA", "int(tgid.z)")
        };
        let src = format!(
            r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void conv(tensor<device half,  dextents<int32_t, 4>> A,
                 tensor<device half,  dextents<int32_t, 4>> W,
                 tensor<device float, dextents<int32_t, 4>> C,
                 uint3 tgid [[threadgroup_position_in_grid]])
{{
    // batch rides the grid's z (descriptor N = 1, per-batch slices) so batches parallelize
    // across threadgroups instead of serializing inside each tile; the n = 1 build keeps the
    // unsliced activation (slicing A measurably defeats an internal fast path)
    constexpr auto desc = convolution2d_descriptor(
        int4({o}, {t}, {t}, 1), int4({c}, {wp}, {hp}, 1), int2({kw}, {kh}),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2({sw}, {sh}), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<desc, execution_simdgroups<4>> op;
    op.set_offsets(int2({cx} + int(tgid.x) * {t} * {sw}, {cy} + int(tgid.y) * {t} * {sh}));
    {a_bind}
    auto tC = C.slice(0, int(tgid.x) * {t}, int(tgid.y) * {t}, {c_z});
    op.run({a_use}, W, tC);
}}
"#
        );
        let opts = MTLCompileOptions::new();
        unsafe { opts.setLanguageVersion(MTLLanguageVersion::Version4_0) };
        let lib = self.device.newLibraryWithSource_options_error(&NSString::from_str(&src), Some(&opts)).ok()?;
        let func = lib.newFunctionWithName(&NSString::from_str("conv"))?;
        let pso = self.device.newComputePipelineStateWithFunction_error(&func).ok()?;
        self.conv_psos.lock().unwrap().insert(key, pso.clone());
        Some(pso)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_conv_cache(
        &self,
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        kh: usize,
        kw: usize,
        o: usize,
        stride: (usize, usize),
        pad: (usize, usize),
    ) -> Option<ConvCache> {
        let (sh, sw) = stride;
        let (ph, pw) = pad;
        let (hp, wp) = (h + 2 * ph, w + 2 * pw);
        let ho = (hp - kh) / sh + 1;
        let wo = (wp - kw) / sw + 1;
        let (ho_p, wo_p) = (ho.div_ceil(CONV_TILE) * CONV_TILE, wo.div_ceil(CONV_TILE) * CONV_TILE);
        let pso_conv = self.conv_pso(n, hp, wp, c, kh, kw, o, sh, sw)?;
        let dev = &self.device;

        let scr_a = dev.newBufferWithLength_options(n * hp * wp * c * 2, MTLResourceOptions::StorageModeShared).expect("A scratch");
        let scr_w = dev.newBufferWithLength_options(kh * kw * c * o * 2, MTLResourceOptions::StorageModeShared).expect("W scratch");
        let scr_c = dev.newBufferWithLength_options(n * ho_p * wo_p * o * 4, MTLResourceOptions::StorageModeShared).expect("C scratch");
        unsafe {
            std::ptr::write_bytes(scr_a.contents().as_ptr() as *mut u8, 0, n * hp * wp * c * 2);
            std::ptr::write_bytes(scr_c.contents().as_ptr() as *mut u8, 0, n * ho_p * wo_p * o * 4);
        }

        let mk_params = |vals: &[u32]| {
            let b = dev.newBufferWithLength_options(vals.len() * 4, MTLResourceOptions::StorageModeShared).expect("param buf");
            unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), b.contents().as_ptr() as *mut u32, vals.len()) };
            b
        };
        let par_a = mk_params(&[(n * h * w * c) as u32, h as u32, w as u32, c as u32, hp as u32, wp as u32, ph as u32, pw as u32]);
        let par_w = mk_params(&[(kh * kw * c * o) as u32, (kh * kw * c * o) as u32, (kh * kw * c * o) as u32]);
        let par_c = mk_params(&[(n * ho * wo * o) as u32, (wo * o) as u32, (wo_p * o) as u32, ho as u32, ho_p as u32, 0]);

        let mk_at = || {
            let atd = MTL4ArgumentTableDescriptor::new();
            atd.setMaxBufferBindCount(3);
            self.device.newArgumentTableWithDescriptor_error(&atd).expect("arg table")
        };
        let (at_cv_a, at_cv_w, at_up, at_conv) = (mk_at(), mk_at(), mk_at(), mk_at());
        unsafe {
            at_cv_a.setAddress_atIndex(scr_a.gpuAddress(), 1);
            at_cv_a.setAddress_atIndex(par_a.gpuAddress(), 2);
            at_cv_w.setAddress_atIndex(scr_w.gpuAddress(), 1);
            at_cv_w.setAddress_atIndex(par_w.gpuAddress(), 2);
            at_up.setAddress_atIndex(scr_c.gpuAddress(), 0);
            at_up.setAddress_atIndex(par_c.gpuAddress(), 2);
        }

        // tensor views over the scratch (extents innermost-first)
        let (ci, oi) = (c as isize, o as isize);
        let t_a = unsafe {
            scr_a.newTensorWithDescriptor_offset_error(
                &tensor_desc(
                    MTLTensorDataType::Float16,
                    &[ci, wp as isize, hp as isize, n as isize],
                    &[1, ci, ci * wp as isize, ci * (wp * hp) as isize],
                    MTLStorageMode::Shared,
                ),
                0,
            )
        }
        .expect("A tensor");
        let t_w = unsafe {
            scr_w.newTensorWithDescriptor_offset_error(
                &tensor_desc(
                    MTLTensorDataType::Float16,
                    &[oi, ci, kw as isize, kh as isize],
                    &[1, oi, oi * ci, oi * ci * kw as isize],
                    MTLStorageMode::Shared,
                ),
                0,
            )
        }
        .expect("W tensor");
        let t_c = unsafe {
            scr_c.newTensorWithDescriptor_offset_error(
                &tensor_desc(
                    MTLTensorDataType::Float32,
                    &[oi, wo_p as isize, ho_p as isize, n as isize],
                    &[1, oi, oi * wo_p as isize, oi * (wo_p * ho_p) as isize],
                    MTLStorageMode::Shared,
                ),
                0,
            )
        }
        .expect("C tensor");
        unsafe {
            at_conv.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
            at_conv.setResource_atBufferIndex(t_w.gpuResourceID(), 1);
            at_conv.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
        }

        let rset = dev.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).expect("residency set");
        for b in [&scr_a, &scr_w, &scr_c, &par_a, &par_w, &par_c] {
            rset.addAllocation(ProtocolObject::from_ref(&**b));
        }
        for t in [&t_a, &t_w, &t_c] {
            rset.addAllocation(ProtocolObject::from_ref(&**t));
        }
        rset.commit();
        rset.requestResidency();

        Some(ConvCache {
            key: (n, h, w, c, kh, kw, o, sh, sw, ph, pw),
            at_cv_a,
            at_cv_w,
            at_up,
            at_conv,
            pso_conv,
            dims: ConvDims {
                ho,
                wo,
                tiles_x: wo_p / CONV_TILE,
                tiles_y: ho_p / CONV_TILE,
                n_a: n * h * w * c,
                n_w: kh * kw * c * o,
                n_o: n * ho * wo * o,
            },
            _scr: [scr_a, scr_w, scr_c],
            _params: vec![par_a, par_w, par_c],
            _tensors: vec![t_a, t_w, t_c],
            rset,
        })
    }

    /// **Resident conv2d**: NHWC f32 wgpu activations × HWIO f32 wgpu weights → NHWO f32 wgpu
    /// output on the conv tensor units — one-op run of the op-DAG executor (same sync contract as
    /// [`Self::bmm_resident`]; fp16-input by contract like the whole device).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_resident(
        &self,
        x: &std::sync::Arc<wgpu::Buffer>,
        x_off: usize,
        w: &std::sync::Arc<wgpu::Buffer>,
        w_off: usize,
        out: &std::sync::Arc<wgpu::Buffer>,
        n: usize,
        h: usize,
        wd: usize,
        c: usize,
        kh: usize,
        kw: usize,
        o: usize,
        stride: (usize, usize),
        pad: (usize, usize),
    ) -> Option<()> {
        let op = ConvOp {
            x: x.clone(),
            x_off,
            w: w.clone(),
            w_off,
            out: out.clone(),
            n,
            h,
            wd,
            c,
            kh,
            kw,
            o,
            stride,
            pad,
        };
        self.run_resident_many(&[ExtOp::Conv(op)])
    }
}

/// The process-wide resident tensor-unit device bound to a wgpu `Context`'s `MTLDevice`. Built on
/// first use (one per process); guarded by raw-device identity so a second `Context` on a different
/// device falls back to the portable path instead of crossing Metal devices.
pub fn resident_for(ctx: &ferric_core::Context) -> Option<&'static Metal4Gemm> {
    use std::sync::OnceLock;
    static DEV: OnceLock<Option<Metal4Gemm>> = OnceLock::new();
    let raw = {
        let hal = unsafe { ctx.device.as_hal::<wgpu::hal::api::Metal>() }?;
        Retained::as_ptr(hal.raw_device()) as usize
    };
    let g = DEV.get_or_init(|| Metal4Gemm::for_wgpu(&ctx.device)).as_ref()?;
    (Retained::as_ptr(&g.device) as usize == raw).then_some(g)
}

/// Whether the resident fast path would fire for a workload of `flops` on this context — the
/// opt-in env (`FERRIC_METAL4=1`), the ~1e8-flop floor (below it the ~0.2 ms tensor-unit dispatch
/// loses to the portable WGSL kernels), and device availability, in one place. Callers that stage
/// extra work for the tensor units (e.g. a dequant pass) should gate on this so a declined route
/// never pays the staging cost.
pub fn resident_ready(ctx: &std::sync::Arc<ferric_core::Context>, flops: usize) -> bool {
    flops >= 100_000_000 && std::env::var("FERRIC_METAL4").is_ok() && resident_for(ctx).is_some()
}

/// Ring pool of matmul output buffers for the resident path. A pooled buffer has already been
/// clear_buffer'd once — wgpu's init tracker marks it initialized forever — so reuse skips the
/// ~170 µs clear-submit round trip that a fresh buffer needs (returns `fresh = true` when the
/// caller must still clear). Reuse requires `strong_count == 1` (the tensor that borrowed it was
/// dropped); content races with in-flight wgpu readers are excluded by the fast path's
/// submit-then-poll drain, which runs before the external queue touches the buffer. Keyed per
/// `Context` identity (Weak-checked, so a recycled address can't resurrect a dead pool) and element
/// count; at most 4 buffers pooled per key, extra demand allocates transient un-pooled buffers.
pub fn pooled_out(ctx: &std::sync::Arc<ferric_core::Context>, n: usize) -> (std::sync::Arc<wgpu::Buffer>, bool) {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, Weak};
    type Pool = Mutex<HashMap<(usize, usize), (Weak<ferric_core::Context>, Vec<Arc<wgpu::Buffer>>)>>;
    static POOL: OnceLock<Pool> = OnceLock::new();
    let mut map = POOL.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    let entry = map.entry((Arc::as_ptr(ctx) as usize, n)).or_insert_with(|| (Arc::downgrade(ctx), Vec::new()));
    if entry.0.upgrade().is_none() {
        *entry = (Arc::downgrade(ctx), Vec::new()); // dead Context recycled this address
    }
    if let Some(buf) = entry.1.iter().find(|b| Arc::strong_count(b) == 1) {
        return (buf.clone(), false);
    }
    let buf = Arc::new(crate::empty(ctx, n));
    if entry.1.len() < 4 {
        entry.1.push(buf.clone());
    }
    (buf, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_ref_f16(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let af: Vec<f32> = a.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let bf: Vec<f32> = b.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let mut c = vec![0.0f32; batch * m * n];
        for bt in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        acc += af[bt * m * k + i * k + l] * bf[l * n + j];
                    }
                    c[bt * m * n + i * n + j] = acc;
                }
            }
        }
        c
    }

    fn check(g: &Metal4Gemm, batch: usize, m: usize, k: usize, n: usize, salt: usize) {
        let a: Vec<f32> = (0..batch * m * k).map(|i| 0.05 * (((i + 1 + salt) % 13) as f32 - 6.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| 0.05 * (((i + 7 + salt) % 11) as f32 - 5.0)).collect();
        let gpu = g.bmm(&a, &b, batch, m, k, n);
        let cpu = cpu_ref_f16(&a, &b, batch, m, k, n);
        let err = gpu.iter().zip(&cpu).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "batch={batch} m={m} k={k} n={n} salt={salt}: err {err}");
    }

    #[test]
    fn tensor_unit_bmm_matches_the_fp16_oracle_across_shapes_and_cache_reuse() {
        let Some(g) = Metal4Gemm::new() else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };
        // exact tiles, ragged (padding), batched — and REPEATED shapes with fresh data (cache-reuse path)
        check(&g, 1, 128, 64, 64, 0);
        check(&g, 1, 128, 64, 64, 3); // reuse cached buffers with new data
        check(&g, 1, 100, 37, 50, 0); // ragged → rebuild + padding
        check(&g, 4, 32, 64, 32, 0); // batch → rebuild + offset views
        check(&g, 4, 32, 64, 32, 5); // batch reuse
        check(&g, 1, 128, 64, 64, 9); // shape switch back → rebuild again
    }

    /// The interop proof for the resident path: MTLTensor views created directly on **wgpu-created**
    /// buffers (via `as_hal`), dispatched on a Metal-4 queue built from **wgpu's own MTLDevice**, with
    /// the result read back through wgpu — no host copy of the operands anywhere.
    #[test]
    fn tensor_views_on_wgpu_buffers_feed_the_tensor_units_directly() {
        use std::sync::Arc;
        use wgpu::util::DeviceExt;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            eprintln!("backend is {:?}, not Metal — skipping", ctx.backend);
            return;
        }
        let Some(g) = Metal4Gemm::for_wgpu(&ctx.device) else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };

        let (m, k, n) = (64usize, 16, 32); // one exact tile — isolates interop, not padding
        let a: Vec<f32> = (0..m * k).map(|i| 0.05 * ((i % 13) as f32 - 6.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| 0.05 * (((i + 7) % 11) as f32 - 5.0)).collect();
        let ah: Vec<u16> = a.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let bh: Vec<u16> = b.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let mk_buf = |bytes: &[u8]| {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        let buf_a = mk_buf(bytemuck::cast_slice(&ah));
        let buf_b = mk_buf(bytemuck::cast_slice(&bh));
        let buf_c = mk_buf(&vec![0u8; m * n * 4]);
        ctx.queue.submit([]);
        crate::device_sync(&ctx); // wgpu writes must land before the external queue reads

        let ra = wgpu_buffer_raw(&buf_a).expect("raw A");
        let rb = wgpu_buffer_raw(&buf_b).expect("raw B");
        let rc = wgpu_buffer_raw(&buf_c).expect("raw C");
        let t_a = unsafe {
            ra.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[k as isize, m as isize], &[1, k as isize], ra.storageMode()),
                0,
            )
        }
        .expect("A tensor on wgpu buffer");
        let t_b = unsafe {
            rb.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[n as isize, k as isize], &[1, n as isize], rb.storageMode()),
                0,
            )
        }
        .expect("B tensor on wgpu buffer");
        let t_c = unsafe {
            rc.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float32, &[n as isize, m as isize], &[1, n as isize], rc.storageMode()),
                0,
            )
        }
        .expect("C tensor on wgpu buffer");

        let atd = MTL4ArgumentTableDescriptor::new();
        atd.setMaxBufferBindCount(3);
        let at = g.device.newArgumentTableWithDescriptor_error(&atd).expect("arg table");
        unsafe {
            at.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
            at.setResource_atBufferIndex(t_b.gpuResourceID(), 1);
            at.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
        }
        let rset = g.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).expect("rset");
        for buf in [&ra, &rb, &rc] {
            rset.addAllocation(ProtocolObject::from_ref(&**buf));
        }
        for t in [&t_a, &t_b, &t_c] {
            rset.addAllocation(ProtocolObject::from_ref(&**t));
        }
        rset.commit();
        rset.requestResidency();

        let alloc = g.device.newCommandAllocator().expect("allocator");
        let cb = g.device.newCommandBuffer().expect("command buffer");
        cb.beginCommandBufferWithAllocator(&alloc);
        cb.useResidencySet(&rset);
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(&g.pso);
        enc.setArgumentTable(Some(&at));
        let tew = g.pso.threadExecutionWidth();
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n / TILE_N, height: m / TILE_M, depth: 1 },
            MTLSize { width: tew * 4, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cb.endCommandBuffer();
        let bufs = [NonNull::from(&*cb)];
        unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ticket = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
        g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), ticket);
        assert!(g.event.waitUntilSignaledValue_timeoutMS(ticket, 60_000), "interop GEMM timed out");

        // readback THROUGH wgpu — proves wgpu sees the external queue's writes
        let arc_ctx = Arc::new(ctx);
        let t = crate::Tensor::from_arc(&arc_ctx, Arc::new(buf_c), &[m, n]);
        let got = pollster::block_on(t.to_vec());
        let want = cpu_ref_f16(&a, &b, 1, m, k, n);
        let err = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "wgpu-resident interop GEMM mismatch: err {err}");
    }

    /// The full resident pipeline (pad+convert → matmul2d → unpad, one command buffer, zero host
    /// copies) against the fp16 oracle — exact tiles, cache reuse, ragged shapes, batches.
    #[test]
    fn resident_bmm_on_wgpu_tensors_matches_the_fp16_oracle() {
        use std::sync::Arc;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            eprintln!("backend is {:?}, not Metal — skipping", ctx.backend);
            return;
        }
        let ctx = Arc::new(ctx);
        let Some(g) = resident_for(&ctx) else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };

        let rcheck = |batch: usize, m: usize, k: usize, n: usize, salt: usize| {
            let a: Vec<f32> = (0..batch * m * k).map(|i| 0.05 * (((i + 1 + salt) % 13) as f32 - 6.0)).collect();
            let b: Vec<f32> = (0..k * n).map(|i| 0.05 * (((i + 7 + salt) % 11) as f32 - 5.0)).collect();
            let ta = crate::Tensor::from_vec(&ctx, &a, &[batch, m, k]);
            let tb = crate::Tensor::from_vec(&ctx, &b, &[k, n]);
            let out = crate::Tensor::zeros(&ctx, &[batch, m, n]);
            ctx.queue.submit([]); // flush pending staged uploads (poll alone never runs them)
            crate::device_sync(&ctx);
            g.bmm_resident(&ta.buf, 0, &tb.buf, 0, &out.buf, batch, m, k, n)
                .expect("resident dispatch");
            let got = pollster::block_on(out.to_vec());
            let want = cpu_ref_f16(&a, &b, batch, m, k, n);
            let err = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-3, "resident batch={batch} m={m} k={k} n={n} salt={salt}: err {err}");
        };
        rcheck(1, 128, 64, 64, 0); // exact tiles
        rcheck(1, 128, 64, 64, 3); // cache reuse, fresh data
        rcheck(1, 100, 37, 50, 0); // ragged → rebuild + padding
        rcheck(4, 32, 64, 32, 0); // batch → offset views
        rcheck(1, 128, 64, 64, 9); // shape switch back
    }

    /// The NT (transpose-right) linear path with the fused activation epilogue: y = act(x·Wᵀ),
    /// W in the HF [out,in] layout, against an fp16-input oracle mirroring the WGSL act formulas.
    #[test]
    fn resident_linear_bt_and_act_match_the_fp16_oracle() {
        use std::sync::Arc;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            eprintln!("backend is {:?}, not Metal — skipping", ctx.backend);
            return;
        }
        let ctx = Arc::new(ctx);
        let Some(g) = resident_for(&ctx) else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };

        let act_f = |v: f32, a: u32| -> f32 {
            match a {
                1 => v.max(0.0),
                2 => v / (1.0 + (-v).exp()),
                4 => 1.0 / (1.0 + (-v).exp()),
                _ => v,
            }
        };
        let lcheck = |rows: usize, inn: usize, out_f: usize, act: u32, salt: usize| {
            let x: Vec<f32> = (0..rows * inn).map(|i| 0.05 * (((i + 1 + salt) % 13) as f32 - 6.0)).collect();
            let w: Vec<f32> = (0..out_f * inn).map(|i| 0.05 * (((i + 7 + salt) % 11) as f32 - 5.0)).collect();
            let tx = crate::Tensor::from_vec(&ctx, &x, &[rows, inn]);
            let tw = crate::Tensor::from_vec(&ctx, &w, &[out_f, inn]);
            let out = crate::Tensor::zeros(&ctx, &[rows, out_f]);
            ctx.queue.submit([]);
            crate::device_sync(&ctx);
            g.linear_resident(&tx.buf, 0, &tw.buf, 0, &out.buf, rows, inn, out_f, act)
                .expect("resident linear dispatch");
            let got = pollster::block_on(out.to_vec());
            // fp16-input oracle: y[i,j] = act(Σ_l x16[i,l]·w16[j,l])
            let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&t| f16::from_f32(t).to_f32()).collect() };
            let (xf, wf) = (q(&x), q(&w));
            let err = (0..rows * out_f)
                .map(|i| {
                    let (r, j) = (i / out_f, i % out_f);
                    let acc: f32 = (0..inn).map(|l| xf[r * inn + l] * wf[j * inn + l]).sum();
                    (got[i] - act_f(acc, act)).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(err < 1e-3, "linear rows={rows} in={inn} out={out_f} act={act} salt={salt}: err {err}");
        };
        lcheck(128, 64, 64, 0, 0); // exact tiles, identity
        lcheck(128, 64, 64, 0, 3); // cache reuse, fresh data
        lcheck(100, 37, 50, 0, 0); // ragged → padding (weight tail rows)
        lcheck(100, 37, 50, 2, 0); // ragged + fused silu
        lcheck(128, 64, 64, 1, 0); // relu → act is part of the cache key (rebuild)
        lcheck(128, 64, 64, 4, 1); // sigmoid
    }

    /// The resident conv2d pipeline (NHWC pad+convert → tiled convolution2d → unpad) against the
    /// fp16 CPU oracle — exact tiles, ragged edge tiles, stride, larger kernels, batch, reuse.
    #[test]
    fn resident_conv2d_matches_the_fp16_oracle() {
        use std::sync::Arc;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            return;
        }
        let ctx = Arc::new(ctx);
        let Some(g) = resident_for(&ctx) else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };

        let ccheck = |n: usize, h: usize, w: usize, c: usize, kh: usize, kw: usize, o: usize,
                      stride: (usize, usize), pad: (usize, usize), salt: usize| {
            let x: Vec<f32> = (0..n * h * w * c).map(|i| 0.1 * (((i + 3 + salt) % 11) as f32 - 5.0)).collect();
            let wt: Vec<f32> = (0..kh * kw * c * o).map(|i| 0.1 * (((i + 5 + salt) % 7) as f32 - 3.0)).collect();
            let tx = crate::Tensor::from_vec(&ctx, &x, &[n, h, w, c]);
            let tw = crate::Tensor::from_vec(&ctx, &wt, &[kh, kw, c, o]);
            let ho = (h + 2 * pad.0 - kh) / stride.0 + 1;
            let wo = (w + 2 * pad.1 - kw) / stride.1 + 1;
            let out = crate::Tensor::zeros(&ctx, &[n, ho, wo, o]);
            ctx.queue.submit([]);
            crate::device_sync(&ctx);
            g.conv2d_resident(&tx.buf, 0, &tw.buf, 0, &out.buf, n, h, w, c, kh, kw, o, stride, pad)
                .expect("resident conv dispatch");
            let got = pollster::block_on(out.to_vec());
            // fp16-input oracle
            let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&t| f16::from_f32(t).to_f32()).collect() };
            let (xf, wf) = (q(&x), q(&wt));
            let mut err = 0.0f32;
            for b in 0..n {
                for yo in 0..ho {
                    for xo in 0..wo {
                        for oc in 0..o {
                            let mut acc = 0.0f32;
                            for ky in 0..kh {
                                let yi = (yo * stride.0 + ky) as isize - pad.0 as isize;
                                if yi < 0 || yi >= h as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let xi = (xo * stride.1 + kx) as isize - pad.1 as isize;
                                    if xi < 0 || xi >= w as isize {
                                        continue;
                                    }
                                    for cc in 0..c {
                                        acc += xf[((b * h + yi as usize) * w + xi as usize) * c + cc]
                                            * wf[((ky * kw + kx) * c + cc) * o + oc];
                                    }
                                }
                            }
                            err = err.max((got[((b * ho + yo) * wo + xo) * o + oc] - acc).abs());
                        }
                    }
                }
            }
            assert!(err < 1e-2, "resident conv n={n} {h}x{w}x{c} k={kh}x{kw} o={o} s={stride:?} p={pad:?} salt={salt}: err {err}");
        };
        ccheck(1, 18, 18, 16, 3, 3, 32, (1, 1), (0, 0), 0); // exact 16x16 dest tile
        ccheck(1, 18, 18, 16, 3, 3, 32, (1, 1), (0, 0), 3); // cache reuse, fresh data
        ccheck(1, 13, 11, 8, 3, 3, 8, (1, 1), (1, 1), 0); // ragged → edge tiles + spatial pad
        ccheck(1, 17, 17, 8, 3, 3, 8, (2, 2), (1, 1), 0); // stride 2
        ccheck(1, 20, 20, 4, 5, 5, 8, (1, 1), (2, 2), 0); // k=5 (offset cancel = 2)
        ccheck(2, 12, 12, 4, 3, 3, 8, (1, 1), (1, 1), 0); // batch
    }

    /// An open `batch()` records dispatches WITHOUT submitting them — an external-queue path that
    /// only does `queue.submit([])` would read inputs that haven't been computed yet. The fast
    /// paths flush the batch first; this pins the contract at the `bmm_resident` level: inputs
    /// produced inside an open batch are visible after `flush_batch` + poll.
    #[test]
    fn resident_path_sees_ops_recorded_in_an_open_batch() {
        use std::sync::Arc;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            return;
        }
        let ctx = Arc::new(ctx);
        let Some(g) = resident_for(&ctx) else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };
        let (m, k, n) = (64usize, 64, 64);
        let base: Vec<f32> = (0..m * k).map(|i| 0.02 * (((i + 1) % 13) as f32 - 6.0)).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| 0.02 * (((i + 7) % 11) as f32 - 5.0)).collect();
        let tb = crate::Tensor::from_vec(&ctx, &bv, &[k, n]);
        let out = crate::Tensor::zeros(&ctx, &[m, n]);
        let got = crate::batch(&ctx, || {
            // `a` is PRODUCED INSIDE the open batch — recorded, not yet submitted
            let t_base = crate::Tensor::from_vec(&ctx, &base, &[m, k]);
            let two = crate::Tensor::from_vec(&ctx, &[2.0], &[1]);
            let a = t_base.mul(&two);
            // what the fast paths do before handing to the external queue:
            crate::flush_batch(&ctx);
            ctx.queue.submit([]);
            crate::device_sync(&ctx);
            g.bmm_resident(&a.buf, 0, &tb.buf, 0, &out.buf, 1, m, k, n).expect("dispatch");
            pollster::block_on(out.to_vec())
        });
        // oracle over the DOUBLED input (fp16 contract)
        let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16::from_f32(x).to_f32()).collect() };
        let (af, bf) = (q(&base.iter().map(|v| v * 2.0).collect::<Vec<_>>()), q(&bv));
        let mut err = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let acc: f32 = (0..k).map(|l| af[i * k + l] * bf[l * n + j]).sum();
                err = err.max((got[i * n + j] - acc).abs());
            }
        }
        assert!(err < 1e-3, "batched-input resident bmm: err {err}");
    }

    /// The op-DAG end to end: inside ONE `batch()`, a dependency chain (h = x·W1ᵀ then y = h·W2ᵀ —
    /// a deferred output consumed as a deferred input) and the q/k/v pattern (three different
    /// shapes) all defer into segments, execute as multi-op command buffers at flush, and match the
    /// fp16 oracle. Env-gated route, exercised exactly as a model would.
    #[test]
    fn deferred_dag_chains_match_the_oracle() {
        use std::sync::Arc;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            return;
        }
        let ctx = Arc::new(ctx);
        if resident_for(&ctx).is_none() {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        }
        std::env::set_var("FERRIC_METAL4", "1");
        let (rows, d, dk) = (128usize, 512usize, 256usize);
        let xv: Vec<f32> = (0..rows * d).map(|i| 0.02 * (((i + 1) % 13) as f32 - 6.0)).collect();
        let w1v: Vec<f32> = (0..d * d).map(|i| 0.02 * (((i + 7) % 11) as f32 - 5.0)).collect();
        let w2v: Vec<f32> = (0..d * d).map(|i| 0.02 * (((i + 3) % 7) as f32 - 3.0)).collect();
        let wkv: Vec<f32> = (0..dk * d).map(|i| 0.02 * (((i + 5) % 9) as f32 - 4.0)).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[rows, d]);
        let w1 = crate::Tensor::from_vec(&ctx, &w1v, &[d, d]);
        let w2 = crate::Tensor::from_vec(&ctx, &w2v, &[d, d]);
        let wk = crate::Tensor::from_vec(&ctx, &wkv, &[dk, d]);

        let (y, q, kk, v) = crate::batch(&ctx, || {
            let h = x.matmul_bt(&w1); // deferred
            let y = h.matmul_bt(&w2); // deferred, READS a deferred output
            let q = x.matmul_bt(&w1); // qkv-style same-shape (forces a run split)
            let kkk = x.matmul_bt(&wk); // different shape in the same segment
            let v = x.matmul_bt(&wk);
            (y, q, kkk, v)
        });
        std::env::remove_var("FERRIC_METAL4");
        let (gy, gq, gk, gv) = (
            pollster::block_on(y.to_vec()),
            pollster::block_on(q.to_vec()),
            pollster::block_on(kk.to_vec()),
            pollster::block_on(v.to_vec()),
        );

        // fp16 oracle (two-level for y: h is itself fp16-rounded through the device)
        let qz = |vv: &[f32]| -> Vec<f32> { vv.iter().map(|&t| f16::from_f32(t).to_f32()).collect() };
        let bt_ref = |xin: &[f32], w: &[f32], r: usize, inn: usize, of: usize| -> Vec<f32> {
            let (xf, wf) = (qz(xin), qz(w));
            let mut out = vec![0.0f32; r * of];
            for i in 0..r {
                for j in 0..of {
                    out[i * of + j] = (0..inn).map(|l| xf[i * inn + l] * wf[j * inn + l]).sum();
                }
            }
            out
        };
        let href = bt_ref(&xv, &w1v, rows, d, d);
        let yref = bt_ref(&href, &w2v, rows, d, d);
        let qref = href.clone();
        let kref = bt_ref(&xv, &wkv, rows, d, dk);
        let ck = |got: &[f32], want: &[f32], tol: f32, label: &str| {
            let e = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(e < tol, "{label}: err {e}");
        };
        ck(&gy, &yref, 5e-2, "chained y"); // two fp16 hops accumulate error
        ck(&gq, &qref, 1e-2, "q");
        ck(&gk, &kref, 1e-2, "k");
        ck(&gv, &kref, 1e-2, "v");
    }

    /// Mixed deferred segments: conv and GEMM ops interleave in one batch and execute through the
    /// shared multi-op engine; results match the respective immediate paths bit-for-bit.
    #[test]
    fn deferred_mixed_conv_and_gemm_match_immediate() {
        use std::sync::Arc;
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("no GPU context — skipping");
            return;
        };
        if ctx.backend != wgpu::Backend::Metal {
            return;
        }
        let ctx = Arc::new(ctx);
        if resident_for(&ctx).is_none() {
            return;
        }
        std::env::set_var("FERRIC_METAL4", "1");
        let (n, hw, c, o) = (1usize, 34usize, 32usize, 64usize);
        let xv: Vec<f32> = (0..n * hw * hw * c).map(|i| 0.05 * (((i + 3) % 11) as f32 - 5.0)).collect();
        let wv: Vec<f32> = (0..3 * 3 * c * o).map(|i| 0.05 * (((i + 5) % 7) as f32 - 3.0)).collect();
        let (rows, d) = (128usize, 512usize);
        let mv: Vec<f32> = (0..rows * d).map(|i| 0.02 * (((i + 1) % 13) as f32 - 6.0)).collect();
        let wl: Vec<f32> = (0..d * d).map(|i| 0.02 * (((i + 7) % 11) as f32 - 5.0)).collect();
        let xc = crate::Tensor::from_vec(&ctx, &xv, &[n, hw, hw, c]);
        let wc = crate::Tensor::from_vec(&ctx, &wv, &[3, 3, c, o]);
        let xm = crate::Tensor::from_vec(&ctx, &mv, &[rows, d]);
        let wm = crate::Tensor::from_vec(&ctx, &wl, &[d, d]);

        // immediate references
        let conv_i = pollster::block_on(xc.conv2d(&wc, (1, 1), (1, 1)).to_vec());
        let mm_i = pollster::block_on(xm.matmul_bt(&wm).to_vec());

        // deferred, interleaved in one batch
        let (cv, mm, cv2) = crate::batch(&ctx, || {
            let a = xc.conv2d(&wc, (1, 1), (1, 1));
            let b = xm.matmul_bt(&wm);
            let a2 = xc.conv2d(&wc, (1, 1), (1, 1)); // same config again → forces a run split
            (a, b, a2)
        });
        std::env::remove_var("FERRIC_METAL4");
        let (gv, gm, gv2) = (
            pollster::block_on(cv.to_vec()),
            pollster::block_on(mm.to_vec()),
            pollster::block_on(cv2.to_vec()),
        );
        let ck = |got: &[f32], want: &[f32], label: &str| {
            let e = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(e == 0.0, "{label}: deferred must be bit-identical to immediate, err {e}");
        };
        ck(&gv, &conv_i, "conv");
        ck(&gm, &mm_i, "gemm");
        ck(&gv2, &conv_i, "conv repeat (run split)");
    }
}


/// Bench-only: measure per-call residency-set construction (used by examples/m4prof.rs).
#[doc(hidden)]
pub fn bench_prset(g: &Metal4Gemm, bufs: &[&Obj<dyn MTLBuffer>]) {
    let prset = g.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).unwrap();
    for b in bufs {
        prset.addAllocation(ProtocolObject::from_ref(&***b));
    }
    prset.commit();
    prset.requestResidency();
}

// The MPP `convolution2d` contract, established empirically (the header ships no worked example):
//  - runtime MSL compilation with MTLLanguageVersion::Version4_0 works (per-config constexpr
//    descriptors → per-config PSOs);
//  - the op computes CROSS-CORRELATION with the source window implicitly shifted by -k/2 per axis
//    (SAME-centering, zero-padded edges) — `set_offsets(k/2 + tile_origin)` recovers corner-anchored
//    VALID windows exactly;
//  - multi-threadgroup tiling = per-tile `C.slice(...)` + per-tile offsets; grid (wo/T, ho/T).
// The tests below pin each of those facts.
#[cfg(test)]
mod conv_probe {
    use super::*;

    /// Decisive semantics experiment: c=1, o=1, asymmetric 3x3 kernel — compare the op's output
    /// against every plausible index-mapping variant to identify the actual contract.
    #[test]
    fn conv2d_semantics_experiment() {
        let Some(g) = Metal4Gemm::new() else { return };
        let (h, w, k) = (6usize, 6usize, 3usize);
        let (ho, wo) = (h - k + 1, w - k + 1);
        // delta input: single 1.0 at (2,3) → output IS the (possibly transformed) kernel imprint
        let mut a = vec![0.0f32; h * w];
        a[2 * w + 3] = 1.0;
        let wt: Vec<f32> = (0..k * k).map(|i| (i + 1) as f32).collect(); // 1..9, fully asymmetric
        let got = probe_raw(&g, &a, &wt, h, w, 1, k, 1);
        eprintln!("kernel (hw): {:?}", wt);
        for yo in 0..ho {
            let row: Vec<f32> = (0..wo).map(|xo| got[yo * wo + xo]).collect();
            eprintln!("  out y{yo}: {row:?}");
        }
        // Measured contract: got[y][x] = w[3-y][4-x] — cross-correlation with the source window
        // shifted by -k/2 on both axes (implicit SAME-centering; zero-padded edges). Assert it so a
        // toolchain change that alters the semantics fails loudly here.
        for y in 0..ho as i32 {
            for x in 0..wo as i32 {
                let want = if (1..=3).contains(&y) && (2..=3).contains(&x) { wt[((3 - y) * k as i32 + (4 - x)) as usize] } else { 0.0 };
                let g_ = got[(y * wo as i32 + x) as usize];
                assert!((g_ - want).abs() < 1e-2, "delta imprint mismatch at ({y},{x}): {g_} vs {want}");
            }
        }
    }

    /// Confirmations: (a) set_offsets(1,1) recovers VALID cross-correlation; (b) dest = src dims
    /// gives exact SAME cross-correlation with zero-pad.
    #[test]
    fn conv2d_offsets_recover_valid_and_same() {
        let Some(g) = Metal4Gemm::new() else { return };
        let (h, w, k) = (6usize, 6usize, 3usize);
        let a: Vec<f32> = (0..h * w).map(|i| 0.1 * (((i + 3) % 11) as f32 - 5.0)).collect();
        let wt: Vec<f32> = (0..k * k).map(|i| (i + 1) as f32 * 0.1).collect();
        let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16::from_f32(x).to_f32()).collect() };
        let (af, wf) = (q(&a), q(&wt));
        let corr = |yo: i32, xo: i32| -> f32 {
            let mut acc = 0.0f32;
            for ky in 0..k as i32 {
                for kx in 0..k as i32 {
                    let (yi, xi) = (yo + ky, xo + kx);
                    if yi >= 0 && yi < h as i32 && xi >= 0 && xi < w as i32 {
                        acc += af[(yi * w as i32 + xi) as usize] * wf[(ky * k as i32 + kx) as usize];
                    }
                }
            }
            acc
        };
        // (a) valid via offsets (1,1): out[y][x] should be corr(y, x) with corner-anchored windows
        let (ho, wo) = (h - k + 1, w - k + 1);
        let got = probe_raw_ex(&g, &a, &wt, h, w, 1, k, 1, ho, wo, 1, 1);
        let mut err = 0.0f32;
        for y in 0..ho {
            for x in 0..wo {
                err = err.max((got[y * wo + x] - corr(y as i32, x as i32)).abs());
            }
        }
        eprintln!("valid-via-offsets err: {err:.3e}");
        assert!(err < 1e-2, "offsets(1,1) must recover valid correlation: err {err}");
        // (b) SAME: dest = src dims, offsets 0 → centered windows, zero-padded
        let got = probe_raw_ex(&g, &a, &wt, h, w, 1, k, 1, h, w, 0, 0);
        let mut err = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                err = err.max((got[y * w + x] - corr(y as i32 - 1, x as i32 - 1)).abs());
            }
        }
        eprintln!("same-centered err: {err:.3e} (informational — integration uses explicit-pad + valid + offsets)");
    }

    /// Tiling experiment: compute an 8x8 valid dest as a 2x2 grid of 4x4 tiles — dest tensor sliced
    /// per threadgroup, source offsets = cancel + tile origin. This is the multi-threadgroup recipe.
    #[test]
    fn conv2d_tiles_assemble_the_full_output() {
        let Some(g) = Metal4Gemm::new() else { return };
        let (h, w, k, tile) = (10usize, 10usize, 3usize, 4usize);
        let (ho, wo) = (h - k + 1, w - k + 1); // 8x8
        let a: Vec<f32> = (0..h * w).map(|i| 0.1 * (((i + 3) % 11) as f32 - 5.0)).collect();
        let wt: Vec<f32> = (0..k * k).map(|i| (i + 1) as f32 * 0.1).collect();
        let src = format!(
            r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void conv(tensor<device half,  dextents<int32_t, 4>> A,
                 tensor<device half,  dextents<int32_t, 4>> W,
                 tensor<device float, dextents<int32_t, 4>> C,
                 uint2 tgid [[threadgroup_position_in_grid]])
{{
    constexpr auto desc = convolution2d_descriptor(
        int4(1, {tile}, {tile}, 1), int4(1, {w}, {h}, 1), int2({k}, {k}),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<desc, execution_simdgroups<4>> op;
    op.set_offsets(int2(1 + int(tgid.x) * {tile}, 1 + int(tgid.y) * {tile}));
    auto tC = C.slice(0, int(tgid.x) * {tile}, int(tgid.y) * {tile}, 0);
    op.run(A, W, tC);
}}
"#
        );
        let opts = MTLCompileOptions::new();
        unsafe { opts.setLanguageVersion(MTLLanguageVersion::Version4_0) };
        let lib = g.device.newLibraryWithSource_options_error(&NSString::from_str(&src), Some(&opts)).unwrap();
        let func = lib.newFunctionWithName(&NSString::from_str("conv")).unwrap();
        let pso = g.device.newComputePipelineStateWithFunction_error(&func).unwrap();
        let dev = &g.device;
        let mk = |bytes: usize| dev.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).unwrap();
        let (ba, bw, bc) = (mk(a.len() * 2), mk(wt.len() * 2), mk(ho * wo * 4));
        unsafe {
            let pa = std::slice::from_raw_parts_mut(ba.contents().as_ptr() as *mut f16, a.len());
            pa.convert_from_f32_slice(&a);
            let pw = std::slice::from_raw_parts_mut(bw.contents().as_ptr() as *mut f16, wt.len());
            pw.convert_from_f32_slice(&wt);
            std::ptr::write_bytes(bc.contents().as_ptr() as *mut u8, 0, ho * wo * 4);
        }
        let ta = unsafe {
            ba.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[1, w as isize, h as isize, 1],
                    &[1, 1, w as isize, (w * h) as isize], MTLStorageMode::Shared), 0)
        }.unwrap();
        let tw = unsafe {
            bw.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[1, 1, k as isize, k as isize],
                    &[1, 1, 1, k as isize], MTLStorageMode::Shared), 0)
        }.unwrap();
        let tc = unsafe {
            bc.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float32, &[1, wo as isize, ho as isize, 1],
                    &[1, 1, wo as isize, (wo * ho) as isize], MTLStorageMode::Shared), 0)
        }.unwrap();
        let atd = MTL4ArgumentTableDescriptor::new();
        atd.setMaxBufferBindCount(3);
        let at = g.device.newArgumentTableWithDescriptor_error(&atd).unwrap();
        unsafe {
            at.setResource_atBufferIndex(ta.gpuResourceID(), 0);
            at.setResource_atBufferIndex(tw.gpuResourceID(), 1);
            at.setResource_atBufferIndex(tc.gpuResourceID(), 2);
        }
        let rset = g.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).unwrap();
        for b in [&ba, &bw, &bc] { rset.addAllocation(ProtocolObject::from_ref(&**b)); }
        for t in [&ta, &tw, &tc] { rset.addAllocation(ProtocolObject::from_ref(&**t)); }
        rset.commit();
        rset.requestResidency();
        let alloc = g.device.newCommandAllocator().unwrap();
        let cb = g.device.newCommandBuffer().unwrap();
        cb.beginCommandBufferWithAllocator(&alloc);
        cb.useResidencySet(&rset);
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso);
        enc.setArgumentTable(Some(&at));
        let tew = pso.threadExecutionWidth();
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 2, height: 2, depth: 1 },
            MTLSize { width: tew * 4, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cb.endCommandBuffer();
        let bufs = [NonNull::from(&*cb)];
        unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ticket = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
        g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), ticket);
        assert!(g.event.waitUntilSignaledValue_timeoutMS(ticket, 30_000));

        let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16::from_f32(x).to_f32()).collect() };
        let (af, wf) = (q(&a), q(&wt));
        let got = unsafe { std::slice::from_raw_parts(bc.contents().as_ptr() as *const f32, ho * wo) };
        let mut err = 0.0f32;
        for yo in 0..ho {
            for xo in 0..wo {
                let mut acc = 0.0f32;
                for ky in 0..k {
                    for kx in 0..k {
                        acc += af[(yo + ky) * w + xo + kx] * wf[ky * k + kx];
                    }
                }
                err = err.max((got[yo * wo + xo] - acc).abs());
            }
        }
        eprintln!("tiled valid-conv err: {err:.3e}");
        assert!(err < 1e-2, "tiled conv must assemble exactly: err {err}");
    }

    fn probe_raw(g: &Metal4Gemm, a: &[f32], wt: &[f32], h: usize, w: usize, c: usize, k: usize, o: usize) -> Vec<f32> {
        let (ho, wo) = (h - k + 1, w - k + 1);
        probe_raw_ex(g, a, wt, h, w, c, k, o, ho, wo, 0, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn probe_raw_ex(g: &Metal4Gemm, a: &[f32], wt: &[f32], h: usize, w: usize, c: usize, k: usize, o: usize,
                    ho: usize, wo: usize, ox: i32, oy: i32) -> Vec<f32> {
        let src = format!(
            r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void conv(tensor<device half,  dextents<int32_t, 4>> A,
                 tensor<device half,  dextents<int32_t, 4>> W,
                 tensor<device float, dextents<int32_t, 4>> C)
{{
    constexpr auto desc = convolution2d_descriptor(
        int4({o}, {wo}, {ho}, 1), int4({c}, {w}, {h}, 1), int2({k}, {k}),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<desc, execution_simdgroups<4>> op;
    op.set_offsets(int2({ox}, {oy}));
    op.run(A, W, C);
}}
"#
        );
        let opts = MTLCompileOptions::new();
        unsafe { opts.setLanguageVersion(MTLLanguageVersion::Version4_0) };
        let lib = g.device.newLibraryWithSource_options_error(&NSString::from_str(&src), Some(&opts)).unwrap();
        let func = lib.newFunctionWithName(&NSString::from_str("conv")).unwrap();
        let pso = g.device.newComputePipelineStateWithFunction_error(&func).unwrap();
        let dev = &g.device;
        let mk = |bytes: usize| dev.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).unwrap();
        let (ba, bw, bc) = (mk(a.len() * 2), mk(wt.len() * 2), mk(ho * wo * o * 4));
        unsafe {
            let pa = std::slice::from_raw_parts_mut(ba.contents().as_ptr() as *mut f16, a.len());
            pa.convert_from_f32_slice(a);
            let pw = std::slice::from_raw_parts_mut(bw.contents().as_ptr() as *mut f16, wt.len());
            pw.convert_from_f32_slice(wt);
            std::ptr::write_bytes(bc.contents().as_ptr() as *mut u8, 0, ho * wo * o * 4);
        }
        let (ci, oi) = (c as isize, o as isize);
        let ta = unsafe {
            ba.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[ci, w as isize, h as isize, 1],
                    &[1, ci, ci * w as isize, ci * (w * h) as isize], MTLStorageMode::Shared), 0)
        }.unwrap();
        let tw = unsafe {
            bw.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[oi, ci, k as isize, k as isize],
                    &[1, oi, oi * ci, oi * ci * k as isize], MTLStorageMode::Shared), 0)
        }.unwrap();
        let tc = unsafe {
            bc.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float32, &[oi, wo as isize, ho as isize, 1],
                    &[1, oi, oi * wo as isize, oi * (wo * ho) as isize], MTLStorageMode::Shared), 0)
        }.unwrap();
        let atd = MTL4ArgumentTableDescriptor::new();
        atd.setMaxBufferBindCount(3);
        let at = g.device.newArgumentTableWithDescriptor_error(&atd).unwrap();
        unsafe {
            at.setResource_atBufferIndex(ta.gpuResourceID(), 0);
            at.setResource_atBufferIndex(tw.gpuResourceID(), 1);
            at.setResource_atBufferIndex(tc.gpuResourceID(), 2);
        }
        let rset = g.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).unwrap();
        for b in [&ba, &bw, &bc] { rset.addAllocation(ProtocolObject::from_ref(&**b)); }
        for t in [&ta, &tw, &tc] { rset.addAllocation(ProtocolObject::from_ref(&**t)); }
        rset.commit();
        rset.requestResidency();
        let alloc = g.device.newCommandAllocator().unwrap();
        let cb = g.device.newCommandBuffer().unwrap();
        cb.beginCommandBufferWithAllocator(&alloc);
        cb.useResidencySet(&rset);
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso);
        enc.setArgumentTable(Some(&at));
        let tew = pso.threadExecutionWidth();
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: tew * 4, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cb.endCommandBuffer();
        let bufs = [NonNull::from(&*cb)];
        unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ticket = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
        g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), ticket);
        assert!(g.event.waitUntilSignaledValue_timeoutMS(ticket, 30_000));
        let got = unsafe { std::slice::from_raw_parts(bc.contents().as_ptr() as *const f32, ho * wo * o) };
        got.to_vec()
    }
}

#[cfg(test)]
mod floor_probe {
    use super::*;
    use std::time::Instant;

    /// Decompose the per-run dispatch floor: what does an EMPTY command buffer round trip cost,
    /// what does host-waiting per cb cost vs chaining cbs with QUEUE-side event waits and paying
    /// one host wait at the end — and is cross-cb ordering via queue wait/signal actually correct?
    #[test]
    fn dispatch_floor_decomposition_and_queue_ordering() {
        let Some(g) = Metal4Gemm::new() else { return };
        let dev = &g.device;
        let mk_pair = || (dev.newCommandAllocator().unwrap(), dev.newCommandBuffer().unwrap());
        let pairs: Vec<_> = (0..8).map(|_| mk_pair()).collect();

        // (a) single empty cb, host wait — the irreducible platform floor
        let round = |reps: usize| -> f64 {
            let t0 = Instant::now();
            for _ in 0..reps {
                let (alloc, cb) = &pairs[0];
                alloc.reset();
                cb.beginCommandBufferWithAllocator(alloc);
                cb.endCommandBuffer();
                let bufs = [NonNull::from(&**cb)];
                unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
                let t = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
                g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), t);
                g.wait_ticket(t);
            }
            t0.elapsed().as_secs_f64() / reps as f64 * 1e6
        };
        round(20); // warm
        eprintln!("empty cb, host wait each:      {:>7.1} µs/cb", round(100));

        // (b) 8 empty cbs, host wait per cb
        let t0 = Instant::now();
        let reps = 25;
        for _ in 0..reps {
            for (alloc, cb) in &pairs {
                alloc.reset();
                cb.beginCommandBufferWithAllocator(alloc);
                cb.endCommandBuffer();
                let bufs = [NonNull::from(&**cb)];
                unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
                let t = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
                g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), t);
                g.wait_ticket(t);
            }
        }
        eprintln!("8 cbs, host wait each:         {:>7.1} µs total", t0.elapsed().as_secs_f64() / reps as f64 * 1e6);

        // (c) 8 empty cbs, queue-side signal/wait chain, ONE host wait
        let t0 = Instant::now();
        for _ in 0..reps {
            let mut last = 0u64;
            for (i, (alloc, cb)) in pairs.iter().enumerate() {
                alloc.reset();
                cb.beginCommandBufferWithAllocator(alloc);
                cb.endCommandBuffer();
                if i > 0 {
                    g.queue.waitForEvent_value(ProtocolObject::from_ref(&*g.event), last);
                }
                let bufs = [NonNull::from(&**cb)];
                unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
                last = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
                g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), last);
            }
            g.wait_ticket(last);
        }
        eprintln!("8 cbs, queue-chained, 1 wait:  {:>7.1} µs total", t0.elapsed().as_secs_f64() / reps as f64 * 1e6);

        // (d) ORDERING PROOF: cb1 doubles a buffer (padConvert identity f32→f16 won't do —
        // use the unpad kernel as a copy: src f32 → dst f32 with act=0, 1 "row"), cb2 doubles
        // again reading cb1's output. Chain via queue wait; ONE host wait; check = 4×input.
        let n = 4096usize;
        let mkbuf = |bytes: usize| dev.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).unwrap();
        let (b0, b1, b2) = (mkbuf(n * 4), mkbuf(n * 4), mkbuf(n * 4));
        let par = mkbuf(6 * 4);
        unsafe {
            let p0 = std::slice::from_raw_parts_mut(b0.contents().as_ptr() as *mut f32, n);
            for (i, v) in p0.iter_mut().enumerate() { *v = i as f32 * 0.25; }
            let pp = std::slice::from_raw_parts_mut(par.contents().as_ptr() as *mut u32, 6);
            pp.copy_from_slice(&[n as u32, n as u32, n as u32, 1, 1, 0]); // count,n,np,m,mp,act — identity copy
        }
        // "double" kernel: reuse padConvert? it converts to f16. Simplest true f32 op we have with
        // raw pointers is unpad (identity copy). Copy proves ORDERING (cb2 output == input) if we
        // chain b0→b1→b2 and check b2 == b0 with only one host wait.
        let atd = MTL4ArgumentTableDescriptor::new();
        atd.setMaxBufferBindCount(3);
        let (at1, at2) = (
            dev.newArgumentTableWithDescriptor_error(&atd).unwrap(),
            dev.newArgumentTableWithDescriptor_error(&atd).unwrap(),
        );
        unsafe {
            at1.setAddress_atIndex(b0.gpuAddress(), 0);
            at1.setAddress_atIndex(b1.gpuAddress(), 1);
            at1.setAddress_atIndex(par.gpuAddress(), 2);
            at2.setAddress_atIndex(b1.gpuAddress(), 0);
            at2.setAddress_atIndex(b2.gpuAddress(), 1);
            at2.setAddress_atIndex(par.gpuAddress(), 2);
        }
        let rset = dev.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).unwrap();
        for b in [&b0, &b1, &b2, &par] { rset.addAllocation(ProtocolObject::from_ref(&**b)); }
        rset.commit();
        rset.requestResidency();
        let mut last = 0u64;
        for (i, at) in [&at1, &at2].iter().enumerate() {
            let (alloc, cb) = &pairs[i];
            alloc.reset();
            cb.beginCommandBufferWithAllocator(alloc);
            cb.useResidencySet(&rset);
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&g.pso_unpad);
            enc.setArgumentTable(Some(at));
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n.div_ceil(256), height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cb.endCommandBuffer();
            if i > 0 {
                g.queue.waitForEvent_value(ProtocolObject::from_ref(&*g.event), last);
            }
            let bufs = [NonNull::from(&**cb)];
            unsafe { g.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
            last = g.ticket.fetch_add(1, Ordering::SeqCst) + 1;
            g.queue.signalEvent_value(ProtocolObject::from_ref(&*g.event), last);
        }
        g.wait_ticket(last);
        let err = unsafe {
            let p2 = std::slice::from_raw_parts(b2.contents().as_ptr() as *const f32, n);
            (0..n).map(|i| (p2[i] - i as f32 * 0.25).abs()).fold(0.0f32, f32::max)
        };
        eprintln!("cross-cb chained copy err: {err:.1e}");
        assert!(err == 0.0, "queue-chained cbs must execute in order: err {err}");
    }
}
