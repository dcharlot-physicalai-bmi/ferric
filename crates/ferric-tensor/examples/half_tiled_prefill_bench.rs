//! 16-bit weight matmul at PREFILL shapes: the flat kernel (one thread per output, every row re-reads
//! the weights) against the row-blocked one `FERRIC_HALF_KERNEL=tiled` selects (one thread per output
//! and block of 8 rows, each weight word read once per 8 rows). Both accumulate in the same order, so
//! their outputs are bit-identical and the column to watch is the time.
//!
//!   cargo run -p ferric-tensor --release --example half_tiled_prefill_bench
//!
//! ⚠ A rate: meaningless on a busy machine. The load average is printed and a run above 3 says so.
use ferric_tensor::dtype::HalfWeights;
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let load = std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    println!("adapter {} — loadavg {}", ctx.adapter_name, load.trim());
    for &(rows, out, inn) in &[(62usize, 4096usize, 4096usize), (62, 11008, 4096), (62, 4096, 11008), (62, 151680, 4096),
                               (200, 11008, 4096), (8, 11008, 4096)] {
        let x: Vec<f32> = (0..rows * inn).map(|i| ((i % 97) as f32 - 48.0) / 97.0).collect();
        let bytes: Vec<u8> = (0..out * inn).flat_map(|i| (0x3c00u16 ^ (i as u16 & 0x03ff)).to_le_bytes()).collect();
        let w = HalfWeights::from_bytes(&ctx, &bytes, out, inn, true);
        let xt = Tensor::from_vec(&ctx, &x, &[rows, inn]);
        let mut line = format!("{rows:4} x {out:6} x {inn:5}");
        let mut outs = Vec::new();
        for kernel in ["flat", "tiled"] {
            // SAFETY: single-threaded here; the kernel choice is read from the environment on each call
            unsafe { std::env::set_var("FERRIC_HALF_KERNEL", kernel); }
            let _ = xt.matmul_half(&w).to_vec().await;             // compile + warm
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let y = xt.matmul_half(&w).to_vec().await;
                best = best.min(t.elapsed().as_secs_f64());
                if outs.len() < 2 { outs.push(y); }
            }
            let gb = (out * inn * 2) as f64 / 1e9;
            line += &format!("   {kernel}: {:7.2} ms ({:6.1} GB/s of weights)", best * 1e3, gb / best);
        }
        let dev = outs[0].iter().zip(&outs[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{line}   max |flat - tiled| {dev:.2e}");
    }
}
