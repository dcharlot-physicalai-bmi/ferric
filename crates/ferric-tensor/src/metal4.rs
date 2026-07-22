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

/// The Metal-4 tensor-unit GEMM device. Create with [`Metal4Gemm::new`]; `None` when the platform has no
/// Metal-4 tensor support (kernel load or pipeline creation fails), so detection stays honest.
pub struct Metal4Gemm {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTL4CommandQueue>,
    pso: Obj<dyn MTLComputePipelineState>,
    event: Obj<dyn MTLSharedEvent>,
    ticket: AtomicU64,
    cache: Mutex<Option<ShapeCache>>,
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

fn tensor_desc(dt: MTLTensorDataType, dims: &[isize], strides: &[isize]) -> Retained<MTLTensorDescriptor> {
    let d = MTLTensorDescriptor::new();
    d.setDataType(dt);
    d.setUsage(MTLTensorUsage::Compute);
    d.setDimensions(&make_extents(dims));
    d.setStrides(Some(&make_extents(strides)));
    d
}

impl Metal4Gemm {
    /// Probe + build the tensor-unit device. Returns `None` if unavailable (no faked capability).
    pub fn new() -> Option<Metal4Gemm> {
        let device = MTLCreateSystemDefaultDevice()?;
        let path = std::env::temp_dir().join("ferric_metal4_gemm.metallib");
        std::fs::write(&path, METALLIB).ok()?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(path.to_str()?));
        let lib = device.newLibraryWithURL_error(&url).ok()?;
        let func = lib.newFunctionWithName(&NSString::from_str("matMul"))?;
        let pso = device.newComputePipelineStateWithFunction_error(&func).ok()?;
        let queue = device.newMTL4CommandQueue()?;
        let event = device.newSharedEvent()?;
        let adapter_name = device.name().to_string();
        Some(Metal4Gemm { device, queue, pso, event, ticket: AtomicU64::new(0), cache: Mutex::new(None), adapter_name })
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
                &tensor_desc(MTLTensorDataType::Float16, &[np as isize, k as isize], &[1, np as isize]),
                0,
            )
        }
        .expect("B tensor");
        let mut argtabs = Vec::with_capacity(batch);
        let mut tensors: Vec<Obj<dyn MTLTensor>> = Vec::with_capacity(batch * 2 + 1);
        for bt in 0..batch {
            let t_a = unsafe {
                buf_a.newTensorWithDescriptor_offset_error(
                    &tensor_desc(MTLTensorDataType::Float16, &[k as isize, mp as isize], &[1, k as isize]),
                    bt * mp * k * 2,
                )
            }
            .expect("A tensor");
            let t_c = unsafe {
                buf_c.newTensorWithDescriptor_offset_error(
                    &tensor_desc(MTLTensorDataType::Float32, &[np as isize, mp as isize], &[1, np as isize]),
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
        assert!(self.event.waitUntilSignaledValue_timeoutMS(ticket, 60_000), "Metal4 GEMM timed out");

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
}
