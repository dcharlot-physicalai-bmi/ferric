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
    println!("  {:<26} {:>10} {:>11} {:>12}", "variant", "bytes", "time", "GB/s");
    for mb in [512usize] {
        for (name, per_thread) in [("scalar u32 (1 word/iter)", 1u32), ("vec4<u32> (4 words/iter)", 4)] {
            let (dt, bytes) = probe_read_bandwidth(&ctx, mb << 20, per_thread).await;
            println!("  {:<26} {:>9.1}M {:>9.2}ms {:>11.1}", name, bytes as f64 / 1e6, dt * 1e3, bytes as f64 / dt / 1e9);
        }
    }
    println!("\n  Reference on this machine: llama.cpp Metal decodes Bonsai-27B at 22 ms/token,");
    println!("  i.e. it streams 7.17 GB of weights at ~326 GB/s (~90% of the M5 Max roofline).");
}
