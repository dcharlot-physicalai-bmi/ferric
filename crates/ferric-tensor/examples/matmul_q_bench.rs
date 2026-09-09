//! **How fast does each quantised matmul actually move its weights, against the read ceiling?**
//!
//! A controlled pair measured Ferric's marginal weight-streaming rate at **71 GB/s** during decode,
//! against a WGSL read ceiling of 385–475 GB/s and llama.cpp's ~326 GB/s
//! (docs/sota-assessment-2026-09.md §2). That is a whole-model figure. This attributes it PER
//! FORMAT, at decode shapes, so the kernel work has a target and a baseline instead of an average.
//!
//! Reported as **GB/s of weight bytes moved**, because that is the quantity the roofline is in and
//! the one `examples/bandwidth.rs` measures the ceiling for. GFLOP/s would flatter the low-bit
//! formats, which do the same arithmetic against fewer bytes — bytes are the bound here, not flops.
//!
//! ⛔ NO READBACK IN THE TIMED LOOP. A `to_vec()` per call forces a GPU round trip and puts a
//! ~0.14 ms sync floor on every sample, which is enough to make a fast kernel look latency-bound —
//! that mistake is recorded in `fattn_bench`. Issue the calls, then `device_sync` ONCE.
//! ⚠ Every buffer is touched before timing. A cold first run reads 39% slow here and says nothing
//! about the kernel.
use ferric_core::Context;
use ferric_tensor::{dtype::QMatrix, Tensor};
use std::sync::Arc;
use std::time::Instant;

/// Random block bytes with a SANE f16 scale in the first two bytes. Arbitrary bytes there are
/// happily NaN or inf, which costs nothing in time but makes a correctness check meaningless.
fn blocks(n: usize, seed: u64, bpb: usize) -> Vec<u8> {
    let mut s = seed;
    let mut v: Vec<u8> = (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u8
    }).collect();
    for (i, blk) in v.chunks_exact_mut(bpb).enumerate() {
        let d = half::f16::from_f32(0.01 + 0.003 * (i % 7) as f32);
        blk[0..2].copy_from_slice(&d.to_le_bytes());
    }
    v
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let Ok(ctx) = Context::new().await else { eprintln!("no GPU"); return };
    let ctx = Arc::new(ctx);
    println!("adapter: {} [{:?}]", ctx.adapter_name, ctx.backend);
    println!("ceiling: examples/bandwidth reports 385-475 GB/s scalar on this class of device");
    println!("llama.cpp streams weights at ~326 GB/s decoding\n");

    // ⭐⭐ REAL DECODE SHAPES, BOTH KERNELS. An earlier version of this benchmark swept only the
    // inner dimension at a FIXED out=8192 and concluded the `flat` kernel was 1.5-2.2x faster than
    // `splitk` for k-quants. End to end on the actual models `flat` is 4-9% SLOWER — because
    // out=8192 flattered it. `flat` runs one thread per OUTPUT, so its parallelism IS `out`; a real
    // model also has `ffn_down` at out=2048 with in=8192, where `flat` starves and `splitk` wins 3x.
    // A benchmark whose shape is not the workload's shape answers a question nobody asked.
    //
    // Set FERRIC_Q2_0_KERNEL=flat|splitk to force one; the table below is read as "which wins here".
    for (inn, out, who) in [
        (1024usize, 3072usize, "qwen3-0.6b ffn_gate_up"),
        (3072, 1024, "qwen3-0.6b ffn_down"),
        (1024, 4096, "qwen3-0.6b qkv"),
        (2048, 8192, "llama3.2-1b ffn_gate_up"),
        (8192, 2048, "llama3.2-1b ffn_down"),
        (2048, 3072, "llama3.2-1b qkv"),
    ] {
    println!("\n=== {who}: in={inn} out={out}  (k-quant blocks/row = {}, of 64 lanes) ===",
             inn / 256);
    let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
    println!("{:>10}  {:>6}  {:>10}  {:>9}  {:>9}  {:>7}", "format", "bpw", "MiB", "ms/call", "GB/s", "% ceil");
    for (ty, name) in [(12u32, "Q4_K"), (13, "Q5_K"), (14, "Q6_K"), (8, "Q8_0"),
                       (43, "STQ1_0"), (16, "IQ2_XXS"), (18, "IQ3_XXS"), (42, "Q2_0")] {
        let Some((vals, bpb)) = QMatrix::block_bytes(ty) else { continue };
        if inn % vals != 0 { println!("{name:>10}  (in={inn} not a multiple of {vals}) — skipped"); continue }
        let bytes = blocks(out * (inn / vals) * bpb, 1234 + ty as u64, bpb);
        let nbytes = bytes.len();
        let Ok(m) = QMatrix::from_bytes(&ctx, &bytes, ty, out, inn) else {
            println!("{name:>10}  from_bytes refused — no packed kernel"); continue };

        // Warm: compile the pipeline, fault the buffers, settle the clocks.
        for _ in 0..5 { let _ = x.matmul_q(&m).to_vec().await; }
        let n = 100;
        ferric_tensor::device_sync(&ctx);
        let t0 = Instant::now();
        let mut sink = None;
        for _ in 0..n { sink = Some(x.matmul_q(&m)); }
        ferric_tensor::device_sync(&ctx);
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let _ = sink;

        let gbs = nbytes as f64 / (ms * 1e-3) / 1e9;
        println!("{name:>10}  {:>6.4}  {:>10.1}  {ms:>9.3}  {gbs:>9.1}  {:>6.0}%",
                 bpb as f64 * 8.0 / vals as f64, nbytes as f64 / 1048576.0, 100.0 * gbs / 430.0);
    }
    }

    // ⭐ THE FUSED SwiGLU PATH, which is where the FFN's gate_up actually goes. `matmul_q` carries
    // qkv / wo / down / lm_head; `try_matmul_swiglu` carries gate_up, the single largest weight in
    // a layer. It is a DIFFERENT kernel shape — one thread per output walking all of K serially,
    // rather than split-K — so the lane-occupancy fix does not touch it and it needs its own number.
    println!("\n=== fused matmul+SwiGLU (gate_up), the other half of the FFN ===");
    println!("{:>10}  {:>16}  {:>10}  {:>9}  {:>9}", "format", "in -> 2*n_ff", "MiB", "ms/call", "GB/s");
    for (inn, n_ff, who) in [(1024usize, 3072usize, "qwen3-0.6b"), (2048, 8192, "llama3.2-1b")] {
        let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
        for (ty, name) in [(12u32, "Q4_K"), (13, "Q5_K"), (14, "Q6_K")] {
            let Some((vals, bpb)) = QMatrix::block_bytes(ty) else { continue };
            let out = 2 * n_ff;                      // gate and up are one fused weight
            let bytes = blocks(out * (inn / vals) * bpb, 99 + ty as u64, bpb);
            let nbytes = bytes.len();
            let Ok(m) = QMatrix::from_bytes(&ctx, &bytes, ty, out, inn) else { continue };
            if x.try_matmul_swiglu(&m).is_none() { println!("{name:>10}  (no fused kernel)"); continue }
            for _ in 0..5 { let _ = x.try_matmul_swiglu(&m).unwrap().to_vec().await; }
            let n = 100;
            ferric_tensor::device_sync(&ctx);
            let t0 = Instant::now();
            let mut sink = None;
            for _ in 0..n { sink = x.try_matmul_swiglu(&m); }
            ferric_tensor::device_sync(&ctx);
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            let _ = sink;
            println!("{name:>10}  {:>16}  {:>10.1}  {ms:>9.3}  {:>9.1}   {who}",
                     format!("{inn} -> {out}"), nbytes as f64 / 1048576.0,
                     nbytes as f64 / (ms * 1e-3) / 1e9);
        }
    }
    // ⛔⛔ IS ANY OF THE ABOVE A STREAMING RATE? Every number so far re-reads the SAME weight 100
    // times, and a 4-18 MiB weight fits in this machine's cache — so those are CACHE-RESIDENT rates,
    // not DRAM streaming rates. A real decode step streams the whole model once: 424 MB for
    // qwen3-0.6b-q5km, which cannot be cached. Sweep the weight past cache and watch it fall.
    println!("\n=== the same kernel as the weight grows past cache (Q5_K, in=2048) ===");
    println!("{:>10}  {:>10}  {:>9}  {:>9}", "out", "MiB", "ms/call", "GB/s");
    {
        let inn = 2048usize;
        let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
        let (vals, bpb) = QMatrix::block_bytes(13).unwrap();
        for out in [2048usize, 8192, 32768, 131072] {
            let bytes = blocks(out * (inn / vals) * bpb, 7, bpb);
            let nbytes = bytes.len();
            let Ok(m) = QMatrix::from_bytes(&ctx, &bytes, 13, out, inn) else { continue };
            for _ in 0..3 { let _ = x.matmul_q(&m).to_vec().await; }
            let n = 20;
            ferric_tensor::device_sync(&ctx);
            let t0 = Instant::now();
            let mut sink = None;
            for _ in 0..n { sink = Some(x.matmul_q(&m)); }
            ferric_tensor::device_sync(&ctx);
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            let _ = sink;
            println!("{out:>10}  {:>10.1}  {ms:>9.3}  {:>9.1}", nbytes as f64 / 1048576.0,
                     nbytes as f64 / (ms * 1e-3) / 1e9);
        }
    }
    println!("\n% ceil is against 430 GB/s, the midpoint of the measured scalar read ceiling.");
}
