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
    key: (usize, usize, usize, usize), // batch, m, k, n
    // scratch kept alive for the argtables' baked GPU addresses (f16 A/B pads, f32 C pad)
    _scr: [Obj<dyn MTLBuffer>; 3],
    at_cv_a: Obj<dyn MTL4ArgumentTable>, // padConvert A: wgpu src → scr_a
    at_cv_b: Obj<dyn MTL4ArgumentTable>, // padConvert B: wgpu src → scr_b
    at_up: Obj<dyn MTL4ArgumentTable>,   // unpad: scr_c → wgpu out
    mm_argtabs: Vec<Obj<dyn MTL4ArgumentTable>>,
    _params: Vec<Obj<dyn MTLBuffer>>,  // kernel param blocks (shape-static, written at build)
    _tensors: Vec<Obj<dyn MTLTensor>>, // kept alive for the argtables' resource IDs
    rset: Obj<dyn MTLResidencySet>,
    alloc: Obj<dyn MTL4CommandAllocator>,
    cb: Obj<dyn MTL4CommandBuffer>,
}

/// The Metal-4 tensor-unit GEMM device. Create with [`Metal4Gemm::new`]; `None` when the platform has no
/// Metal-4 tensor support (kernel load or pipeline creation fails), so detection stays honest.
pub struct Metal4Gemm {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTL4CommandQueue>,
    pso: Obj<dyn MTLComputePipelineState>,
    pso_pad: Obj<dyn MTLComputePipelineState>,
    pso_unpad: Obj<dyn MTLComputePipelineState>,
    event: Obj<dyn MTLSharedEvent>,
    ticket: AtomicU64,
    cache: Mutex<Option<ShapeCache>>,
    rcache: Mutex<Option<ResidentCache>>,
    pub adapter_name: String,
}

// SAFETY: MTLDevice / MTL4CommandQueue / MTLComputePipelineState / MTLSharedEvent are thread-safe Metal
// objects. The non-thread-safe encode objects (allocator / command buffer / argument tables) live inside
// `cache` and are only ever touched while holding its Mutex, and every call waits for GPU completion
// before releasing the lock — so no cross-thread concurrent use is possible.
unsafe impl Send for Metal4Gemm {}
unsafe impl Sync for Metal4Gemm {}

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
        let mut psos = ["matMul", "padConvert", "unpad"].into_iter().map(|name| {
            let func = lib.newFunctionWithName(&NSString::from_str(name))?;
            device.newComputePipelineStateWithFunction_error(&func).ok()
        });
        let (pso, pso_pad, pso_unpad) = (psos.next()??, psos.next()??, psos.next()??);
        let queue = device.newMTL4CommandQueue()?;
        let event = device.newSharedEvent()?;
        let adapter_name = device.name().to_string();
        Some(Metal4Gemm {
            device,
            queue,
            pso,
            pso_pad,
            pso_unpad,
            event,
            ticket: AtomicU64::new(0),
            cache: Mutex::new(None),
            rcache: Mutex::new(None),
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
        if guard.as_ref().map(|c| c.key) != Some((batch, m, k, n)) {
            *guard = Some(self.build_cache(batch, m, k, n));
        }
        let c = guard.as_ref().unwrap();
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

    // Build (or rebuild) the resident-path shape-static objects.
    fn build_rcache(&self, batch: usize, m: usize, k: usize, n: usize) -> ResidentCache {
        let mp = m.div_ceil(TILE_M) * TILE_M;
        let np = n.div_ceil(TILE_N) * TILE_N;
        let dev = &self.device;

        let scr_a = dev.newBufferWithLength_options(batch * mp * k * 2, MTLResourceOptions::StorageModeShared).expect("A scratch");
        let scr_b = dev.newBufferWithLength_options(k * np * 2, MTLResourceOptions::StorageModeShared).expect("B scratch");
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
        let par_b = mk_params(&[(k * n) as u32, n as u32, np as u32]);
        let par_c = mk_params(&[(batch * m * n) as u32, n as u32, np as u32, m as u32, mp as u32]);

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

        // matmul tensor views over the scratch buffers — identical layout to the host-boundary path
        let t_b = unsafe {
            scr_b.newTensorWithDescriptor_offset_error(
                &tensor_desc(MTLTensorDataType::Float16, &[np as isize, k as isize], &[1, np as isize], MTLStorageMode::Shared),
                0,
            )
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

        let alloc = dev.newCommandAllocator().expect("allocator");
        let cb = dev.newCommandBuffer().expect("command buffer");
        ResidentCache {
            key: (batch, m, k, n),
            _scr: [scr_a, scr_b, scr_c],
            at_cv_a,
            at_cv_b,
            at_up,
            mm_argtabs,
            _params: vec![par_a, par_b, par_c],
            _tensors: tensors,
            rset,
            alloc,
            cb,
        }
    }

    /// **Resident** batched matmul: operands and result are **wgpu buffers** (f32, contiguous), and the
    /// whole pipeline — pad+f16-convert, `matmul2d` on the tensor units, unpad — runs as one MTL4
    /// command buffer on wgpu's own `MTLDevice`. No byte crosses the host. Offsets are in bytes.
    ///
    /// The caller must ensure prior wgpu work producing `a`/`b` has completed (e.g.
    /// [`ferric_core`-level] device poll) — this call blocks until the GPU finishes, so wgpu
    /// submissions issued after it return safely observe `out`. Returns `None` (touching nothing) if
    /// the buffers aren't Metal-backed.
    #[allow(clippy::too_many_arguments)] // a GEMM signature: three operands + offsets + four dims
    pub fn bmm_resident(
        &self,
        a: &wgpu::Buffer,
        a_off: usize,
        b: &wgpu::Buffer,
        b_off: usize,
        out: &wgpu::Buffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<()> {
        let ra = wgpu_buffer_raw(a)?;
        let rb = wgpu_buffer_raw(b)?;
        let rc = wgpu_buffer_raw(out)?;
        let mut guard = self.rcache.lock().unwrap();
        if guard.as_ref().map(|c| c.key) != Some((batch, m, k, n)) {
            *guard = Some(self.build_rcache(batch, m, k, n));
        }
        let c = guard.as_ref().unwrap();
        let mp = m.div_ceil(TILE_M) * TILE_M;
        let np = n.div_ceil(TILE_N) * TILE_N;

        unsafe {
            c.at_cv_a.setAddress_atIndex(ra.gpuAddress() + a_off as u64, 0);
            c.at_cv_b.setAddress_atIndex(rb.gpuAddress() + b_off as u64, 0);
            c.at_up.setAddress_atIndex(rc.gpuAddress(), 1);
        }
        // the wgpu buffers change every call → a small per-call residency set alongside the cached one
        let prset = self.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).ok()?;
        for buf in [&ra, &rb, &rc] {
            prset.addAllocation(ProtocolObject::from_ref(&**buf));
        }
        prset.commit();
        prset.requestResidency();

        c.alloc.reset();
        c.cb.beginCommandBufferWithAllocator(&c.alloc);
        c.cb.useResidencySet(&c.rset);
        c.cb.useResidencySet(&prset);
        let enc = c.cb.computeCommandEncoder()?;
        let tg256 = MTLSize { width: 256, height: 1, depth: 1 };
        let lin = |count: usize| MTLSize { width: count.div_ceil(256), height: 1, depth: 1 };
        // stage 1: pad-convert both operands (independent dispatches)
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
        // stage 2: the tensor-unit GEMM (one dispatch per batch element, disjoint C)
        enc.setComputePipelineState(&self.pso);
        let tew = self.pso.threadExecutionWidth();
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
        // stage 3: gather the [m,n] regions into the wgpu output
        enc.setComputePipelineState(&self.pso_unpad);
        enc.setArgumentTable(Some(&c.at_up));
        enc.dispatchThreadgroups_threadsPerThreadgroup(lin(batch * m * n), tg256);
        enc.endEncoding();
        c.cb.endCommandBuffer();

        let bufs = [NonNull::from(&*c.cb)];
        unsafe { self.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ticket = self.ticket.fetch_add(1, Ordering::SeqCst) + 1;
        self.queue.signalEvent_value(ProtocolObject::from_ref(&*self.event), ticket);
        self.wait_ticket(ticket);
        Some(())
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
