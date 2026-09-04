//! **What is the actual read-bandwidth ceiling of a WGSL compute shader here?**
//!
//! Ferric's Q2_0 matmul streams cold weights at ~70 GB/s while llama.cpp's Metal kernels reach
//! ~326 GB/s on the same machine. That gap is only worth attacking once it's known *which* wall it
//! is: if a shader that does nothing but read also tops out near 70 GB/s, the ceiling is the
//! memory path (or how we drive it) and no amount of ALU cleverness helps. If a pure read runs far
//! faster, the matmul is ALU/latency-bound and the fix is in the inner loop.
//!
//! The buffer is deliberately far larger than the SLC so every pass is a cold DRAM stream — the
//! mistake that made the earlier microbenchmarks (24 MB, re-read 20×) report gains that vanished
//! end-to-end.
use ferric_core::Context;
use ferric_tensor::probe_read_bandwidth;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Ferric · WGSL compute read-bandwidth probe (cold, buffer >> cache)\n");
    // ⛔ The premise here is "buffer >> last-level cache", NOT the literal 512 MiB — and asking for
    // 512 MiB as a single storage binding panicked in `Device::create_bind_group` the first time CI
    // ran this on Linux: lavapipe and the WebGPU baseline cap a binding at 128 MiB. Clamp to what
    // the device will actually bind and SAY SO, because a bandwidth number measured on a smaller
    // buffer is a different measurement and must not be reported as if it were the same one.
    let want_mb = 512usize;
    let bytes = ((ctx.max_binding as usize) & !0xFFF).min(want_mb << 20);
    if bytes < want_mb << 20 {
        println!("  (this device binds at most {} MiB of storage; probing there rather than {want_mb} MiB.\n\
                   Still far beyond any last-level cache, so the cold-DRAM-stream premise holds.)\n",
                 bytes >> 20);
    }
    println!("  {:<26} {:>10} {:>11} {:>12}", "variant", "bytes", "time", "GB/s");
    for mb in [bytes >> 20] {
        for (name, per_thread) in [("scalar u32, wg64", 1u32), ("vec4<u32>, wg64", 4),
                                    ("scalar u32, wg128", 128), ("scalar u32, wg256", 256),
                                    ("scalar u32, wg512", 512), ("scalar u32, wg1024", 1024)] {
            let (dt, bytes) = probe_read_bandwidth(&ctx, mb << 20, per_thread).await;
            println!("  {:<26} {:>9.1}M {:>9.2}ms {:>11.1}", name, bytes as f64 / 1e6, dt * 1e3, bytes as f64 / dt / 1e9);
        }
    }
    println!("\n  Reference on this machine: llama.cpp Metal decodes Bonsai-27B at 22 ms/token,");
    println!("  i.e. it streams 7.17 GB of weights at ~326 GB/s (~90% of the M5 Max roofline).");
}
