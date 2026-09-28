//! **Prefill attention kernels, head to head** — the per-query kernel (`rows`), the tiled WGSL kernel
//! (`tiled`) and the matrix-unit MSL kernel (`native`, Metal only), on real model shapes, interleaved.
//!
//!   cargo run -p ferric-tensor --release --example flash_prefill_bench [reps 5]
//!
//! Each (shape, kernel) cell times 10 back-to-back dispatches ending in one readback, repeated `reps`
//! times in ROUND-ROBIN order across kernels (a contended machine drifts; interleaving spreads the drift
//! over every arm instead of charging it to whichever ran last), and reports min / median ms per call and
//! the TFLOP/s of the median: 4·dh FLOPs per (head, causal query-key pair), masked pairs not counted.
use ferric_core::Context;
use ferric_tensor::{flash_tiled::Kernel, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }

async fn run() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let ctx = Arc::new(Context::new().await.expect("gpu"));
    println!("adapter {}  native_msl {}", ctx.adapter_name, ctx.native_msl);
    // (label, T, off, nh, nkv, dh)
    let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("qwen2.5-0.5b", 512, 0, 14, 2, 64),
        ("qwen2.5-0.5b", 2048, 0, 14, 2, 64),
        ("qwen2.5-0.5b", 8192, 0, 14, 2, 64),
        ("qwen2.5-0.5b chunk", 512, 1536, 14, 2, 64),
        ("qwen2.5-0.5b suffix", 16, 2000, 14, 2, 64),
        ("qwen2.5-0.5b suffix", 64, 2000, 14, 2, 64),
        ("qwen3-0.6b", 512, 0, 16, 8, 128),
        ("qwen3-0.6b", 2048, 0, 16, 8, 128),
        ("qwen2.5-1.5b", 2048, 0, 12, 2, 128),
        ("llama-3.2-1b", 2048, 0, 32, 8, 64),
    ];
    let only: Option<String> = std::env::var("BENCH_ONLY").ok();
    // BENCH_SWEEP=1: the small-block crossover instead (short blocks, fresh and continuing a long cache).
    let sweep: Vec<(&str, usize, usize, usize, usize, usize)> = [(14usize, 2usize, 64usize), (16, 8, 128), (32, 8, 64)].iter()
        .flat_map(|&(nh, nkv, dh)| [2usize, 8, 16, 32, 64, 128, 256].into_iter()
            .flat_map(move |t| [0usize, 2000].into_iter().map(move |off| ("sweep", t, off, nh, nkv, dh))))
        .collect();
    let shapes = if std::env::var_os("BENCH_SWEEP").is_some() { &sweep[..] } else { shapes };
    for &(label, t, off, nh, nkv, dh) in shapes {
        if only.as_deref().is_some_and(|o| !label.contains(o) && o != t.to_string()) { continue; }
        let s = off + t;
        let mk = |n: usize, seed: u32| -> Vec<f32> {
            let mut x = seed.wrapping_mul(2654435761) | 1;
            (0..n).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; (x as f32 / u32::MAX as f32) * 2.0 - 1.0 }).collect()
        };
        let q = Tensor::from_vec(&ctx, &mk(t * nh * dh, 1), &[t, nh * dh]);
        let k = Tensor::from_vec(&ctx, &mk(s * nkv * dh, 2), &[s, nkv * dh]);
        let v = Tensor::from_vec(&ctx, &mk(s * nkv * dh, 3), &[s, nkv * dh]);
        let pairs: f64 = (0..t).map(|i| (off + i + 1) as f64).sum();
        let flops = 4.0 * dh as f64 * nh as f64 * pairs;
        let kernels: Vec<Kernel> = [Kernel::Rows, Kernel::Tiled, Kernel::Native].into_iter()
            .filter(|&kk| q.flash_attention_prefill_with(&k, &v, nh, nkv, dh, off, kk).is_some())
            .collect();
        // warm: compile every pipeline
        for &kk in &kernels { let _ = q.flash_attention_prefill_with(&k, &v, nh, nkv, dh, off, kk).unwrap().to_vec().await; }
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
        let iters = if t * s > 16_000_000 { 3 } else { 10 };
        for _ in 0..reps {
            for (ki, &kk) in kernels.iter().enumerate() {
                let t0 = Instant::now();
                let mut last = None;
                for _ in 0..iters { last = q.flash_attention_prefill_with(&k, &v, nh, nkv, dh, off, kk); }
                let _ = last.unwrap().to_vec().await;
                times[ki].push(t0.elapsed().as_secs_f64() / iters as f64);
            }
        }
        let mut line = format!("{label:<20} T={t:<5} off={off:<5} nh={nh:<2} nkv={nkv} dh={dh:<3}");
        let mut med0 = 0.0;
        for (ki, &kk) in kernels.iter().enumerate() {
            let mut ts = times[ki].clone();
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ts[ts.len() / 2];
            if ki == 0 { med0 = med; }
            line += &format!(" | {kk:?} {:.2}/{:.2} ms {:.2} TF/s {:.1}x", ts[0] * 1e3, med * 1e3, flops / med / 1e12, med0 / med);
        }
        println!("{line}");
    }
}
