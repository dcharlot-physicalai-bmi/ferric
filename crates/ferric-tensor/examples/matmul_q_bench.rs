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
    // ⭐ THE LM HEAD. For qwen3-0.6b it is [151936, 1024] Q6_K = 126 MB — THIRTY PERCENT of the whole
    // checkpoint in a single matmul, run once per token. Ablating both sublayers still leaves
    // 5.7 ms/token, and this is most of what remains, so its rate matters more than any layer's.
    println!("\n=== the LM head, the single biggest matmul in the model ===");
    println!("{:>12}  {:>10}  {:>9}  {:>9}", "shape", "MiB", "ms/call", "GB/s");
    for (inn, out, ty, name) in [(1024usize, 151936usize, 14u32, "qwen3-0.6b Q6_K"),
                                 (2048, 128256, 14, "llama3.2-1b Q6_K")] {
        let Some((vals, bpb)) = QMatrix::block_bytes(ty) else { continue };
        let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
        let bytes = blocks(out * (inn / vals) * bpb, 5, bpb);
        let nbytes = bytes.len();
        let Ok(m) = QMatrix::from_bytes(&ctx, &bytes, ty, out, inn) else { continue };
        for _ in 0..3 { let _ = x.matmul_q(&m).to_vec().await; }
        let n = 20;
        ferric_tensor::device_sync(&ctx);
        let t0 = Instant::now();
        let mut sink = None;
        for _ in 0..n { sink = Some(x.matmul_q(&m)); }
        ferric_tensor::device_sync(&ctx);
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let _ = sink;
        println!("{:>12}  {:>10.1}  {ms:>9.3}  {:>9.1}   {name}", format!("{inn}x{out}"),
                 nbytes as f64 / 1048576.0, nbytes as f64 / (ms * 1e-3) / 1e9);
    }

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
    // ⛔⛔ THE ACCESS PATTERN THE MODEL ACTUALLY HAS. Every figure above re-reads ONE weight in a
    // tight loop, so it is cache-resident and measures a rate no decode step ever sees. A real token
    // sweeps ~112 DIFFERENT weights of 1-4 MiB each, touching every byte exactly once — nothing is
    // reused, nothing is warm. That is a different machine-level problem, and this measures it:
    // allocate N distinct weights and walk them once per pass.
    println!("\n=== ONE PASS over N DISTINCT weights — no reuse, the decode pattern ===");
    println!("{:>6}  {:>9}  {:>10}  {:>9}  {:>9}", "N", "each MiB", "total MiB", "ms/pass", "GB/s");
    {
        let inn = 1024usize;
        let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
        let (vals, bpb) = QMatrix::block_bytes(13).unwrap();
        for (n_w, out) in [(28usize, 4096usize), (28, 6144), (112, 3072)] {
            let ws: Vec<QMatrix> = (0..n_w).map(|j| {
                let bytes = blocks(out * (inn / vals) * bpb, 1000 + j as u64, bpb);
                QMatrix::from_bytes(&ctx, &bytes, 13, out, inn).unwrap()
            }).collect();
            let each = out * (inn / vals) * bpb;
            let total = each * n_w;
            // Warm the pipeline, not the data: one pass before timing, then time a fresh pass. With
            // `total` far above cache the second pass is still cold, which is the point.
            for w in &ws { let _ = x.matmul_q(w); }
            ferric_tensor::device_sync(&ctx);
            let t0 = Instant::now();
            let mut sink = None;
            for w in &ws { sink = Some(x.matmul_q(w)); }
            ferric_tensor::device_sync(&ctx);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let _ = sink;
            println!("{n_w:>6}  {:>9.1}  {:>10.1}  {ms:>9.3}  {:>9.1}",
                     each as f64 / 1048576.0, total as f64 / 1048576.0,
                     total as f64 / (ms * 1e-3) / 1e9);
        }
    }

    // ⭐ IS THERE A FIXED COST PER MATMUL DISPATCH? The sweep above shows 2.1 MiB and 4.1 MiB weights
    // taking the SAME ~47 us, which says the bytes are nearly free and the dispatch is the bill — and
    // that contradicts the ~7 us/dispatch the QK-norm+RoPE A/B implied. One of those is wrong about
    // what a dispatch costs, so measure it across a wide size range, cold, with no reuse.
    println!("\n=== cost of ONE cold matmul vs its size (Q5_K, in=1024, 24 distinct weights each) ===");
    println!("{:>10}  {:>10}  {:>11}  {:>11}  {:>9}", "out", "each MiB", "us unbatched", "us batched", "GB/s");
    {
        let inn = 1024usize;
        let (mut bad, mut prev) = (0usize, None::<f64>);
        let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
        let (vals, bpb) = QMatrix::block_bytes(13).unwrap();
        for out in [256usize, 1024, 4096, 16384, 65536] {
            let n_w = 24usize;
            let ws: Vec<QMatrix> = (0..n_w).map(|j| {
                let bytes = blocks(out * (inn / vals) * bpb, 2000 + (out * 31 + j) as u64, bpb);
                QMatrix::from_bytes(&ctx, &bytes, 13, out, inn).unwrap()
            }).collect();
            let each = out * (inn / vals) * bpb;
            for w in &ws { let _ = x.matmul_q(w); }
            // ⛔ BATCHED vs NOT is the whole question. Outside a `batch` region every op is its own
            // queue submission, so a per-dispatch cost measured that way is a SUBMIT cost wearing a
            // dispatch's name — and the model batches ~1 submit per layer. Measure both.
            ferric_tensor::device_sync(&ctx);
            let t0 = Instant::now();
            let mut sink = None;
            for w in &ws { sink = Some(x.matmul_q(w)); }
            ferric_tensor::device_sync(&ctx);
            let us_un = t0.elapsed().as_secs_f64() * 1e6 / n_w as f64;
            ferric_tensor::device_sync(&ctx);
            let t1 = Instant::now();
            ferric_tensor::batch(&ctx, || { for w in &ws { sink = Some(x.matmul_q(w)); } });
            ferric_tensor::device_sync(&ctx);
            let us_b = t1.elapsed().as_secs_f64() * 1e6 / n_w as f64;
            let _ = sink;
            // ⭐ SELF-REFUTATION CHECK. Batching only removes queue submissions; it cannot add work, so
            // batched > unbatched is IMPOSSIBLE and means the run was contended. Likewise a bigger
            // weight must not be faster than a smaller one at the same shape. A contended benchmark is
            // a WRONG number, not a slow one, and nothing else in this output says so — a load gate
            // cannot, because the dominant variation is a GPU clock state `uptime` cannot see.
            let mut flag = "";
            if us_b > us_un * 1.15 { flag = "  ⛔ BATCHED SLOWER THAN UNBATCHED — CONTENDED, DISCARD"; bad += 1; }
            // ⚠ The loop walks out SMALL -> LARGE, so cost RISING is correct and must not be flagged.
            // The impossible direction is a BIGGER weight coming out materially FASTER than a smaller
            // one. Writing this the other way round rejected a hand-built clean table on its first
            // replay, which is the only reason the inversion was caught.
            if let Some(pus) = prev {
                if us_b < pus * 0.7 { flag = "  ⛔ FASTER THAN A SMALLER WEIGHT — CONTENDED, DISCARD"; bad += 1; }
            }
            prev = Some(us_b);
            println!("{out:>10}  {:>10.2}  {us_un:>11.1}  {us_b:>11.1}  {:>9.1}{flag}",
                     each as f64 / 1048576.0, each as f64 / (us_b * 1e-6) / 1e9);
        }
        if bad > 0 {
            println!("\n  ⛔⛔ {bad} IMPOSSIBLE ORDERING(S) ABOVE. THIS RUN IS CONTENDED — DO NOT QUOTE IT.");
            println!("      Re-run on a quiet machine, and read RATIOS across >=3 whole runs, not one table.");
        } else {
            println!("\n  ✅ no impossible orderings — batched <= unbatched and cost is monotone in size.");
        }
    }

    // ⭐ THE CONTROL THAT DECIDES WHAT ~50 us MEANS. If a tiny rmsnorm also costs ~50 us, the number
    // is a UNIVERSAL per-dispatch cost on this fabric and the 7 us implied by the QK-norm+RoPE A/B is
    // wrong. If rmsnorm is cheap, then matmul dispatches specifically are expensive and the cause is
    // in that path, not in wgpu. Same harness, same sync discipline, distinct buffers, no reuse.
    println!("\n=== control: cost of a tiny NON-matmul dispatch, measured identically ===");
    {
        let d = 4096usize;
        let w = Tensor::from_vec(&ctx, &vec![1.0f32; d], &[d]);
        let n = 24usize;
        let xs: Vec<Tensor> = (0..n).map(|j| Tensor::from_vec(
            &ctx, &(0..d).map(|i| ((i + j * 7) as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, d])).collect();
        for t in &xs { let _ = t.rmsnorm(&w, 1e-6); }
        ferric_tensor::device_sync(&ctx);
        let t0 = Instant::now();
        let mut sink = None;
        for t in &xs { sink = Some(t.rmsnorm(&w, 1e-6)); }
        ferric_tensor::device_sync(&ctx);
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        let _ = sink;
        println!("  rmsnorm [1,{d}]  {us:.1} us/dispatch  (16 KiB read)");

        // And an equally tiny MATMUL, so the only variable is which kernel runs.
        let (vals, bpb) = QMatrix::block_bytes(13).unwrap();
        let inn = 1024usize; let out = 256usize;
        let xm = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[1, inn]);
        let ws: Vec<QMatrix> = (0..n).map(|j| {
            let b = blocks(out * (inn / vals) * bpb, 4000 + j as u64, bpb);
            QMatrix::from_bytes(&ctx, &b, 13, out, inn).unwrap()
        }).collect();
        for q in &ws { let _ = xm.matmul_q(q); }
        ferric_tensor::device_sync(&ctx);
        let t1 = Instant::now();
        let mut sink2 = None;
        for q in &ws { sink2 = Some(xm.matmul_q(q)); }
        ferric_tensor::device_sync(&ctx);
        let us2 = t1.elapsed().as_secs_f64() * 1e6 / n as f64;
        let _ = sink2;
        println!("  matmul_q [1,{inn}]x[{out},{inn}]  {us2:.1} us/dispatch  (176 KiB read)");
    }

    println!("\n% ceil is against 430 GB/s, the midpoint of the measured scalar read ceiling.");
}
