//! **Metal 4 tensor-unit GEMM** — the M5-class tensor cores, driven from pure Rust (vendored
//! objc2-metal), verified against a CPU fp16 oracle. This is the fastest matmul path on Apple
//! silicon and it is NOT reachable through wgpu/WGSL: the kernel uses MetalPerformancePrimitives'
//! cooperative `tensor_ops::matmul2d` (Metal 4), dispatched via the MTL4 command model (argument
//! tables + residency sets + batched commit + event completion).
//!
//! Measured on an M5 Max: 10.4 TFLOP/s @ 512^3, 23.5 @ 1024^3, 29.0 @ 2048x1024x2048 —
//! vs ~0.104 TFLOP/s for the portable wgpu-WGSL matmul on the same box (~280x). Correctness:
//! max |tensor-unit - cpu(f16 oracle)| at fp32 rounding (~4e-7).
//!
//! macOS-only (Metal 4 / macOS 26+); on other platforms this example prints a note and exits.

#[cfg(target_os = "macos")]
mod mac {
    use core::ptr::NonNull;
use half::f16;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;

fn gen(rows: usize, cols: usize, salt: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| 0.05 * (((i + salt) % 13) as f32 - 6.0)).collect()
}
fn cpu_ref_f16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let af: Vec<f32> = a.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let bf: Vec<f32> = b.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += af[i * k + l] * bf[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn to_f16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn make_extents(vals: &[isize]) -> objc2::rc::Retained<MTLTensorExtents> {
    unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), vals.len(), vals.as_ptr()) }
        .expect("extents")
}


#[allow(clippy::too_many_arguments)]
fn bench(
    device: &ProtocolObject<dyn MTLDevice>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    m: usize, k: usize, n: usize, reps: usize,
) -> f64 {
    let buf_a = device.newBufferWithLength_options(m * k * 2, MTLResourceOptions::StorageModeShared).unwrap();
    let buf_b = device.newBufferWithLength_options(k * n * 2, MTLResourceOptions::StorageModeShared).unwrap();
    let buf_c = device.newBufferWithLength_options(m * n * 4, MTLResourceOptions::StorageModeShared).unwrap();
    // fill A,B with small fp16 values (pattern; correctness proven separately at small size)
    unsafe {
        let pa = buf_a.contents().as_ptr() as *mut u16;
        let pb = buf_b.contents().as_ptr() as *mut u16;
        for i in 0..m * k { *pa.add(i) = f16::from_f32(0.01 * ((i % 7) as f32 - 3.0)).to_bits(); }
        for i in 0..k * n { *pb.add(i) = f16::from_f32(0.01 * ((i % 5) as f32 - 2.0)).to_bits(); }
    }
    let mk_tensor = |buf: &ProtocolObject<dyn MTLBuffer>, dims: &[isize], strides: &[isize], dt: MTLTensorDataType| {
        let d = MTLTensorDescriptor::new();
        d.setDataType(dt);
        d.setUsage(MTLTensorUsage::Compute);
        d.setDimensions(&make_extents(dims));
        d.setStrides(Some(&make_extents(strides)));
        unsafe { buf.newTensorWithDescriptor_offset_error(&d, 0) }.expect("tensor")
    };
    let t_a = mk_tensor(&buf_a, &[k as isize, m as isize], &[1, k as isize], MTLTensorDataType::Float16);
    let t_b = mk_tensor(&buf_b, &[n as isize, k as isize], &[1, n as isize], MTLTensorDataType::Float16);
    let t_c = mk_tensor(&buf_c, &[n as isize, m as isize], &[1, n as isize], MTLTensorDataType::Float32);

    let atd = MTL4ArgumentTableDescriptor::new();
    atd.setMaxBufferBindCount(3);
    let argtab = device.newArgumentTableWithDescriptor_error(&atd).unwrap();
    unsafe {
        argtab.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
        argtab.setResource_atBufferIndex(t_b.gpuResourceID(), 1);
        argtab.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
    }
    let rsd = MTLResidencySetDescriptor::new();
    let rset = device.newResidencySetWithDescriptor_error(&rsd).unwrap();
    for al in [
        ProtocolObject::from_ref(&*buf_a), ProtocolObject::from_ref(&*buf_b), ProtocolObject::from_ref(&*buf_c),
    ] { rset.addAllocation(al); }
    rset.commit();
    rset.requestResidency();

    let queue = device.newMTL4CommandQueue().unwrap();
    queue.addResidencySet(&rset);
    let alloc = device.newCommandAllocator().unwrap();
    let event = device.newSharedEvent().unwrap();
    let tew = pso.threadExecutionWidth();
    let grid = MTLSize { width: n / 32, height: m / 64, depth: 1 };
    let tg = MTLSize { width: tew * 4, height: 1, depth: 1 };

    let mut best = f64::INFINITY;
    for round in 0..3u64 {
        let cb = device.newCommandBuffer().unwrap();
        cb.beginCommandBufferWithAllocator(&alloc);
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso);
        enc.setArgumentTable(Some(&argtab));
        for r in 0..reps {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            if r + 1 < reps {
                enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch, MTLStages::Dispatch, MTL4VisibilityOptions::None);
            }
        }
        enc.endEncoding();
        cb.endCommandBuffer();
        let t0 = std::time::Instant::now();
        let bufs = [NonNull::from(&*cb)];
        unsafe { queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };
        let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
        queue.signalEvent_value(ev, round + 1);
        assert!(event.waitUntilSignaledValue_timeoutMS(round + 1, 30_000), "bench timeout");
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        best = best.min(dt);
        alloc.reset();
    }
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    flops / best / 1e9 // GFLOP/s
}

pub fn run() {
    // tile constraints: M % 64 == 0, N % 32 == 0
    let (m, k, n) = (128usize, 64usize, 64usize);
    let a = gen(m, k, 1);
    let b = gen(k, n, 7);
    let cpu = cpu_ref_f16(&a, &b, m, k, n);

    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    println!("device: {}", device.name());

    // library from the precompiled metallib (Metal 4 tensor kernel)
    // embedded precompiled Metal 4 kernel (source: examples/metal4_gemm.metal; rebuild with
    //   xcrun metal -std=metal4.0 -c metal4_gemm.metal -o t.air && xcrun metallib t.air -o metal4_gemm.metallib)
    const METALLIB: &[u8] = include_bytes!("metal4_gemm.metallib");
    let lib_path = std::env::temp_dir().join("ferric_metal4_gemm.metallib");
    std::fs::write(&lib_path, METALLIB).expect("write metallib");
    let url = NSURL::fileURLWithPath(&NSString::from_str(lib_path.to_str().unwrap()));
    let lib = device.newLibraryWithURL_error(&url).expect("load metallib");
    let func = lib.newFunctionWithName(&NSString::from_str("matMul")).expect("kernel fn");
    let pso = device.newComputePipelineStateWithFunction_error(&func).expect("pipeline");

    // shared buffers: A fp16 [M,K], B fp16 [K,N], C fp32 [M,N]
    let a16 = to_f16_bits(&a);
    let b16 = to_f16_bits(&b);
    let buf_a = device.newBufferWithLength_options(a16.len() * 2, MTLResourceOptions::StorageModeShared).unwrap();
    let buf_b = device.newBufferWithLength_options(b16.len() * 2, MTLResourceOptions::StorageModeShared).unwrap();
    let buf_c = device.newBufferWithLength_options(m * n * 4, MTLResourceOptions::StorageModeShared).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(a16.as_ptr() as *const u8, buf_a.contents().as_ptr() as *mut u8, a16.len() * 2);
        std::ptr::copy_nonoverlapping(b16.as_ptr() as *const u8, buf_b.contents().as_ptr() as *mut u8, b16.len() * 2);
        std::ptr::write_bytes(buf_c.contents().as_ptr() as *mut u8, 0, m * n * 4);
    }

    // tensors over the buffers (extents[0] = innermost/contiguous dim)
    let mk_tensor = |buf: &ProtocolObject<dyn MTLBuffer>, dims: &[isize], strides: &[isize], dt: MTLTensorDataType| {
        let d = MTLTensorDescriptor::new();
        d.setDataType(dt);
        d.setUsage(MTLTensorUsage::Compute);
        d.setDimensions(&make_extents(dims));
        d.setStrides(Some(&make_extents(strides)));
        unsafe { buf.newTensorWithDescriptor_offset_error(&d, 0) }.expect("tensor")
    };
    let t_a = mk_tensor(&buf_a, &[k as isize, m as isize], &[1, k as isize], MTLTensorDataType::Float16);
    let t_b = mk_tensor(&buf_b, &[n as isize, k as isize], &[1, n as isize], MTLTensorDataType::Float16);
    let t_c = mk_tensor(&buf_c, &[n as isize, m as isize], &[1, n as isize], MTLTensorDataType::Float32);

    // argument table: kernel tensor args A,B,C -> buffer slots 0,1,2
    let atd = MTL4ArgumentTableDescriptor::new();
    atd.setMaxBufferBindCount(3);
    let argtab = device.newArgumentTableWithDescriptor_error(&atd).expect("arg table");
    unsafe {
        argtab.setResource_atBufferIndex(t_a.gpuResourceID(), 0);
        argtab.setResource_atBufferIndex(t_b.gpuResourceID(), 1);
        argtab.setResource_atBufferIndex(t_c.gpuResourceID(), 2);
    }

    // residency (mandatory in MTL4)
    let rsd = MTLResidencySetDescriptor::new();
    let rset = device.newResidencySetWithDescriptor_error(&rsd).expect("residency set");
    rset.addAllocation(ProtocolObject::from_ref(&*buf_a));
    rset.addAllocation(ProtocolObject::from_ref(&*buf_b));
    rset.addAllocation(ProtocolObject::from_ref(&*buf_c));
    rset.addAllocation(ProtocolObject::from_ref(&*t_a));
    rset.addAllocation(ProtocolObject::from_ref(&*t_b));
    rset.addAllocation(ProtocolObject::from_ref(&*t_c));
    rset.commit();
    rset.requestResidency();

    // MTL4 lifecycle
    let queue = device.newMTL4CommandQueue().expect("MTL4 queue");
    queue.addResidencySet(&rset);
    let alloc = device.newCommandAllocator().expect("allocator");
    let cb = device.newCommandBuffer().expect("command buffer");
    cb.beginCommandBufferWithAllocator(&alloc);
    let enc = cb.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&pso);
    enc.setArgumentTable(Some(&argtab));
    let tew = pso.threadExecutionWidth();
    let grid = MTLSize { width: n / 32, height: m / 64, depth: 1 };
    let tg = MTLSize { width: tew * 4, height: 1, depth: 1 };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cb.endCommandBuffer();

    let bufs = [NonNull::from(&*cb)];
    unsafe { queue.commit_count(NonNull::new(bufs.as_ptr() as *mut _).unwrap(), 1) };

    // completion: shared event signaled by the queue after the committed work
    let event = device.newSharedEvent().expect("shared event");
    let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
    queue.signalEvent_value(ev, 1);
    assert!(event.waitUntilSignaledValue_timeoutMS(1, 10_000), "GPU timed out");

    // readback + verify
    let mut gpu = vec![0.0f32; m * n];
    unsafe { std::ptr::copy_nonoverlapping(buf_c.contents().as_ptr() as *const f32, gpu.as_mut_ptr(), m * n) };
    let err = max_abs_diff(&gpu, &cpu);
    println!("M={m} K={k} N={n}  max|tensor-unit − cpu(f16-ref)| = {err:.5}");
    println!("gpu[0..4] = {:?}", &gpu[..4]);
    println!("cpu[0..4] = {:?}", &cpu[..4]);
    assert!(err < 5e-2, "tensor-unit GEMM must match the fp16 oracle: {err}");
    println!("✅ Metal 4 tensor-unit matmul (MPPTensorOps matmul2d) ran and matched the CPU oracle");

    println!("\n=== tensor-unit GEMM throughput (min of 3 rounds, 20 reps/round) ===");
    for &(bm, bk, bn) in &[(512usize, 512usize, 512usize), (1024, 1024, 1024), (2048, 1024, 2048)] {
        let gflops = bench(&device, &pso, bm, bk, bn, 20);
        println!("  {}x{}x{}: {:.0} GFLOP/s", bm, bk, bn, gflops);
    }
    println!("  (Ferric wgpu-WGSL path measured ~85-104 GFLOP/s fp32 on this box)");
}

}

fn main() {
    #[cfg(target_os = "macos")]
    mac::run();
    #[cfg(not(target_os = "macos"))]
    println!("Metal 4 tensor-unit GEMM is macOS-only (Metal 4).");
}
