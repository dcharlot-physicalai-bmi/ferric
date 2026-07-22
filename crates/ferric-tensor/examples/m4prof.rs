//! **Metal-4 overhead profiler** — decomposes the resident path's per-call cost so optimizations
//! target the real bottleneck: full call vs wgpu flush+poll vs clear-submit vs buffer creation vs
//! the pure dispatch floor (tiny shape) vs residency-set construction. This is how the buffer pool
//! and spin-wait were justified; rerun it before optimizing further.
use ferric_core::Context;
use ferric_tensor::{device_sync, metal4, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let ctx = Arc::new(pollster::block_on(Context::new()).unwrap());
    let g = metal4::resident_for(&ctx).expect("metal4");
    let nn = 1024usize;
    let a = Tensor::from_vec(&ctx, &vec![0.01f32; nn * nn], &[nn, nn]);
    let b = Tensor::from_vec(&ctx, &vec![0.02f32; nn * nn], &[nn, nn]);
    let out = Tensor::zeros(&ctx, &[nn, nn]);
    ctx.queue.submit([]);
    device_sync(&ctx);

    let time = |label: &str, reps: usize, f: &mut dyn FnMut()| {
        f(); // warm
        let t0 = Instant::now();
        for _ in 0..reps { f() }
        println!("  {:<34} {:>9.1} µs", label, t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
    };

    println!("=== per-call cost breakdown (N=1024, same shape, cache hot) ===");
    time("full bmm_resident", 50, &mut || {
        g.bmm_resident(a.buffer(), 0, b.buffer(), 0, out.buffer(), 1, nn, nn, nn).unwrap();
    });
    time("wgpu submit([]) + poll-wait", 200, &mut || {
        ctx.queue.submit([]);
        device_sync(&ctx);
    });
    time("wgpu clear_buffer submit + poll", 200, &mut || {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.clear_buffer(out.buffer(), 0, None);
        ctx.queue.submit([enc.finish()]);
        device_sync(&ctx);
    });
    time("empty-buffer create (1024^2 f32)", 200, &mut || {
        let _ = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (nn * nn * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    });
    // the pieces inside bmm_resident, via its public surface: a second identical call after the
    // cache is hot IS (setAddress + prset + encode + commit + signal + wait) — subtract the GEMM
    // by timing a tiny shape (all overhead, ~zero flops)
    let t1 = Tensor::from_vec(&ctx, &vec![0.01f32; 64 * 64], &[64, 64]);
    let t2 = Tensor::from_vec(&ctx, &vec![0.02f32; 64 * 64], &[64, 64]);
    let t3 = Tensor::zeros(&ctx, &[64, 64]);
    ctx.queue.submit([]);
    device_sync(&ctx);
    time("bmm_resident tiny (pure overhead)", 200, &mut || {
        g.bmm_resident(t1.buffer(), 0, t2.buffer(), 0, t3.buffer(), 1, 64, 64, 64).unwrap();
    });
    // isolate: per-call residency-set construction on wgpu buffers
    let ra = metal4::wgpu_buffer_raw(t1.buffer()).unwrap();
    let rb = metal4::wgpu_buffer_raw(t2.buffer()).unwrap();
    let rc = metal4::wgpu_buffer_raw(t3.buffer()).unwrap();
    time("prset create+add3+commit+request", 200, &mut || {
        metal4::bench_prset(g, &[&ra, &rb, &rc]);
    });
    time("wgpu_buffer_raw x3 (as_hal)", 200, &mut || {
        let _ = metal4::wgpu_buffer_raw(t1.buffer()).unwrap();
        let _ = metal4::wgpu_buffer_raw(t2.buffer()).unwrap();
        let _ = metal4::wgpu_buffer_raw(t3.buffer()).unwrap();
    });
}
