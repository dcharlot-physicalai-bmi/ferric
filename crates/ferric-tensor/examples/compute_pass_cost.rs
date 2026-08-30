//! **What does a compute pass cost?** The measurement behind a 2.2x native-vs-browser gap.
//!
//! Ferric's `run()` opens a fresh compute pass per dispatch. On the same model, same kernels, the
//! browser issues 6319 dispatches in 61 submits and finishes in 282 ms; native issues 6317 in 59
//! and takes 629 ms. Identical work, 2.2x apart — so the cost is per-dispatch, and the one thing
//! native does per dispatch is `begin_compute_pass`, which on Metal is a fresh
//! `MTLComputeCommandEncoder`.
//!
//! This isolates that: the SAME N dispatches of the SAME trivial kernel, recorded (a) one pass each
//! and (b) all in one pass. Everything else — pipeline, bind group, buffer, submit count — is held
//! equal, so a difference is the pass boundary and nothing else.
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL: &str = r#"
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 64u) { x[gid.x] = x[gid.x] * 1.0000001 + 0.0000001; }
}
"#;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let dev = &ctx.device;
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tiny"), source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tiny"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"), contents: bytemuck::cast_slice(&[1.0f32; 64]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let bgl = pipe.get_bind_group_layout(0);
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });

    // The dispatch count Ferric actually issues for one 5.86 s speech encode.
    for &n in &[1000usize, 6300] {
        let mut per_pass = f64::MAX;
        let mut one_pass = f64::MAX;
        // Min of several reps: on a shared machine the fast run is the least perturbed one.
        for _ in 0..5 {
            // (a) one compute pass per dispatch — what `run()` does today.
            let t = std::time::Instant::now();
            let mut enc = dev.create_command_encoder(&Default::default());
            for _ in 0..n {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None, timestamp_writes: None });
                p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            let _ = dev.poll(wgpu::PollType::wait_indefinitely());
            per_pass = per_pass.min(t.elapsed().as_secs_f64() * 1e3);

            // (b) all dispatches inside ONE pass. Same pipeline, same bind group, same submit.
            let t = std::time::Instant::now();
            let mut enc = dev.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None, timestamp_writes: None });
                p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]);
                for _ in 0..n { p.dispatch_workgroups(1, 1, 1); }
            }
            ctx.queue.submit([enc.finish()]);
            let _ = dev.poll(wgpu::PollType::wait_indefinitely());
            one_pass = one_pass.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let ratio = per_pass / one_pass.max(1e-9);
        let each_us = (per_pass - one_pass) * 1000.0 / n as f64;
        println!("n={n:<6} per-dispatch pass {per_pass:7.2} ms   single pass {one_pass:7.2} ms");
        println!("       ratio {ratio:.2}x   pass overhead {each_us:.1} us each");
    }
    println!("\nIf the ratio is large, `run()`'s per-dispatch `begin_compute_pass` is the native gap.");
}
