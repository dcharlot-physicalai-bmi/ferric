//! **Metal 4 tensor-unit GEMM backend** — the M5-class tensor cores as a fabric device. Runs the
//! MetalPerformancePrimitives cooperative `tensor_ops::matmul2d` kernel (fp16 inputs → fp32 accumulate)
//! through the MTL4 command model, at ~23–31 TFLOP/s on an M5 Max vs ~0.1 TFLOP/s for the portable
//! wgpu-WGSL matmul (~280×). Host-boundary contract like every scheduler device: `&[f32]` in, `Vec<f32>`
//! out — upload converts to fp16, so this is an explicitly **reduced-precision** backend (the standard ML
//! GEMM trade); results match an fp16-input oracle to fp32 rounding, not the f32 oracle to 1e-7. The
//! adaptive [`crate::sched::Planner`] measures it like any device and routes accordingly.
//!
//! Inputs of any shape are handled by padding M to 64 and N to 32 (the kernel's tile sizes) with zeros and
//! slicing the result back out. Batches encode one dispatch per element (offset tensor views, one submit).
//! The precompiled kernel is embedded (`metal4_gemm.metallib`; source `metal4_gemm.metal` alongside —
//! rebuild with `xcrun metal -std=metal4.0 -c … && xcrun metallib …`).
#![cfg(all(target_os = "macos", not(target_arch = "wasm32")))]

use core::ptr::NonNull;
use half::f16;
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

/// The Metal-4 tensor-unit GEMM device. Create with [`Metal4Gemm::new`]; `None` when the platform has no
/// Metal-4 tensor support (kernel load or pipeline creation fails), so detection stays honest.
pub struct Metal4Gemm {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    ticket: AtomicU64,
    // MTL4CommandAllocator is not Sync; serialize encoding (fine: one GPU anyway).
    encode_lock: Mutex<()>,
    pub adapter_name: String,
}

// SAFETY: MTLDevice / MTL4CommandQueue / MTLComputePipelineState / MTLSharedEvent are thread-safe Metal
// objects; the non-thread-safe encode objects (allocator/command buffer/encoder) are created per call
// under `encode_lock`.
unsafe impl Send for Metal4Gemm {}
unsafe impl Sync for Metal4Gemm {}

fn make_extents(vals: &[isize]) -> Retained<MTLTensorExtents> {
    unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), vals.len(), vals.as_ptr()) }
        .expect("tensor extents")
}

impl Metal4Gemm {
    /// Probe + build the tensor-unit device. Returns `None` if unavailable (no faked capability).
    pub fn new() -> Option<Metal4Gemm> {
        let device = MTLCreateSystemDefaultDevice()?;
        // write the embedded metallib to a temp file and load it
        let path = std::env::temp_dir().join("ferric_metal4_gemm.metallib");
        std::fs::write(&path, METALLIB).ok()?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(path.to_str()?));
        let lib = device.newLibraryWithURL_error(&url).ok()?;
        let func = lib.newFunctionWithName(&NSString::from_str("matMul"))?;
        let pso = device.newComputePipelineStateWithFunction_error(&func).ok()?;
        let queue = device.newMTL4CommandQueue()?;
        let event = device.newSharedEvent()?;
        let adapter_name = device.name().to_string();
        Some(Metal4Gemm { device, queue, pso, event, ticket: AtomicU64::new(0), encode_lock: Mutex::new(()), adapter_name })
    }

    /// Batched matmul `[batch,m,k] · [k,n] → [batch,m,n]` on the tensor units (fp16 inputs, fp32 out).
    pub fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let _guard = self.encode_lock.lock().unwrap();
        let mp = m.div_ceil(TILE_M) * TILE_M; // padded M
        let np = n.div_ceil(TILE_N) * TILE_N; // padded N
        let dev = &self.device;

        // upload buffers: A padded per batch [batch, mp, k] fp16; B [k, np] fp16; C [batch, mp, np] fp32
        let buf_a = dev.newBufferWithLength_options(batch * mp * k * 2, MTLResourceOptions::StorageModeShared).expect("A buf");
        let buf_b = dev.newBufferWithLength_options(k * np * 2, MTLResourceOptions::StorageModeShared).expect("B buf");
        let buf_c = dev.newBufferWithLength_options(batch * mp * np * 4, MTLResourceOptions::StorageModeShared).expect("C buf");
        unsafe {
            let pa = buf_a.contents().as_ptr() as *mut u16;
            std::ptr::write_bytes(pa, 0, batch * mp * k);
            for bt in 0..batch {
                for i in 0..m {
                    for j in 0..k {
                        *pa.add(bt * mp * k + i * k + j) = f16::from_f32(a[bt * m * k + i * k + j]).to_bits();
                    }
                }
            }
            let pb = buf_b.contents().as_ptr() as *mut u16;
            std::ptr::write_bytes(pb, 0, k * np);
            for i in 0..k {
                for j in 0..n {
                    *pb.add(i * np + j) = f16::from_f32(b[i * n + j]).to_bits();
                }
            }
            std::ptr::write_bytes(buf_c.contents().as_ptr() as *mut u8, 0, batch * mp * np * 4);
        }

        // tensors: extents[0] = innermost dim; per-batch A/C views via byte offsets
        let t_b = {
            let d = MTLTensorDescriptor::new();
            d.setDataType(MTLTensorDataType::Float16);
            d.setUsage(MTLTensorUsage::Compute);
            d.setDimensions(&make_extents(&[np as isize, k as isize]));
            d.setStrides(Some(&make_extents(&[1, np as isize])));
            unsafe { buf_b.newTensorWithDescriptor_offset_error(&d, 0) }.expect("B tensor")
        };
        let mut argtabs = Vec::with_capacity(batch);
        let mut tensors = Vec::with_capacity(batch * 2 + 1);
        for bt in 0..batch {
            let da = MTLTensorDescriptor::new();
            da.setDataType(MTLTensorDataType::Float16);
            da.setUsage(MTLTensorUsage::Compute);
            da.setDimensions(&make_extents(&[k as isize, mp as isize]));
            da.setStrides(Some(&make_extents(&[1, k as isize])));
            let t_a = unsafe { buf_a.newTensorWithDescriptor_offset_error(&da, bt * mp * k * 2) }.expect("A tensor");
            let dc = MTLTensorDescriptor::new();
            dc.setDataType(MTLTensorDataType::Float32);
            dc.setUsage(MTLTensorUsage::Compute);
            dc.setDimensions(&make_extents(&[np as isize, mp as isize]));
            dc.setStrides(Some(&make_extents(&[1, np as isize])));
            let t_c = unsafe { buf_c.newTensorWithDescriptor_offset_error(&dc, bt * mp * np * 4) }.expect("C tensor");

            let atd = MTL4ArgumentTableDescriptor::new();
            atd.setMaxBufferBindCount(3);
            let at = dev.newArgumentTableWithDescriptor_error(&atd).expect("arg table");
            unsafe {
                at.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
                at.setResource_atBufferIndex(t_b.gpuResourceID(), 1);
                at.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
            }
            argtabs.push(at);
            tensors.push(t_a);
            tensors.push(t_c);
        }
        tensors.push(t_b.clone());

        // residency (mandatory in MTL4)
        let rsd = MTLResidencySetDescriptor::new();
        let rset = dev.newResidencySetWithDescriptor_error(&rsd).expect("residency set");
        rset.addAllocation(ProtocolObject::from_ref(&*buf_a));
        rset.addAllocation(ProtocolObject::from_ref(&*buf_b));
        rset.addAllocation(ProtocolObject::from_ref(&*buf_c));
        for t in &tensors {
            rset.addAllocation(ProtocolObject::from_ref(&**t));
        }
        rset.commit();
        rset.requestResidency();

        // encode: one dispatch per batch element (disjoint C regions — no barriers needed)
        let alloc = dev.newCommandAllocator().expect("allocator");
        let cb = dev.newCommandBuffer().expect("command buffer");
        cb.beginCommandBufferWithAllocator(&alloc);
        cb.useResidencySet(&rset); // per-command-buffer residency (queue-level sets cap at 32 — would leak)
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.pso);
        let tew = self.pso.threadExecutionWidth();
        let grid = MTLSize { width: np / TILE_N, height: mp / TILE_M, depth: 1 };
        let tg = MTLSize { width: tew * 4, height: 1, depth: 1 };
        for at in &argtabs {
            enc.setArgumentTable(Some(at));
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
        enc.endEncoding();
        cb.endCommandBuffer();

        let bufs = [NonNull::from(&*cb)];
        unsafe { self.queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ticket = self.ticket.fetch_add(1, Ordering::SeqCst) + 1;
        let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
        self.queue.signalEvent_value(ev, ticket);
        assert!(self.event.waitUntilSignaledValue_timeoutMS(ticket, 60_000), "Metal4 GEMM timed out");

        // readback: slice [m, n] out of the padded [mp, np] per batch
        let mut out = vec![0.0f32; batch * m * n];
        unsafe {
            let pc = buf_c.contents().as_ptr() as *const f32;
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

    #[test]
    fn tensor_unit_bmm_matches_the_fp16_oracle_including_padding_and_batches() {
        let Some(g) = Metal4Gemm::new() else {
            eprintln!("no Metal 4 tensor support — skipping");
            return;
        };
        // ragged dims (exercise padding) and a batch
        for &(batch, m, k, n) in &[(1usize, 128usize, 64usize, 64usize), (1, 100, 37, 50), (4, 32, 64, 32)] {
            let a: Vec<f32> = (0..batch * m * k).map(|i| 0.05 * (((i + 1) % 13) as f32 - 6.0)).collect();
            let b: Vec<f32> = (0..k * n).map(|i| 0.05 * (((i + 7) % 11) as f32 - 5.0)).collect();
            let gpu = g.bmm(&a, &b, batch, m, k, n);
            let cpu = cpu_ref_f16(&a, &b, batch, m, k, n);
            let err = gpu.iter().zip(&cpu).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-3, "batch={batch} m={m} k={k} n={n}: err {err}");
        }
    }
}
