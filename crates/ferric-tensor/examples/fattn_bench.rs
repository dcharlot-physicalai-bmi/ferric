//! **What does one decode attention step actually cost, and is that the arithmetic or the launch?**
//!
//! Profiling a real decode found attention at **52–62% of the step** on two models whose attention
//! projections are a FIFTH of their FFN's parameters — a ~10x anomaly against what the arithmetic
//! predicts (docs/sota-assessment-2026-09.md §2). That measurement attributes by category, so it
//! cannot say whether the cost is the fused kernel, the small ops around it, or launch latency.
//! This one times the kernel alone, against a matmul of comparable size on the same device.
//!
//! ⚠ Read the RATIO, not the absolute ms. The point is whether the kernel's cost tracks its work.
//! A kernel whose time is flat in S is latency-bound; one that scales with S is doing arithmetic.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.13 + s).sin()) * 0.3).collect() }

fn main() { pollster::block_on(run()); }

async fn run() {
    let Ok(ctx) = Context::new().await else { eprintln!("no GPU"); return };
    let ctx = Arc::new(ctx);
    println!("adapter: {} [{:?}]\n", ctx.adapter_name, ctx.backend);

    // Two real decode configurations, from the models the assessment measured.
    for (label, nh, nkv, dh) in [("qwen3-0.6b", 16usize, 8usize, 128usize),
                                 ("llama3.2-1b", 32, 8, 64)] {
        println!("── {label}: nh={nh} nkv={nkv} dh={dh}");
        println!("   {:>6}  {:>10}  {:>12}  {:>10}", "S", "ms/call", "MFLOP", "GFLOP/s");
        for s in [17usize, 128, 512, 2048] {
            let q = Tensor::from_vec(&ctx, &seq(nh * dh, 1.0), &[1, nh * dh]);
            let k = Tensor::from_vec(&ctx, &seq(s * nkv * dh, 2.0), &[s, nkv * dh]);
            let v = Tensor::from_vec(&ctx, &seq(s * nkv * dh, 3.0), &[s, nkv * dh]);
            // ⛔ NO `.to_vec()` IN THE TIMED LOOP. A readback per call forces a GPU round trip, and
            // the first version of this benchmark did exactly that: every row carried a ~0.14 ms sync
            // floor, so S=17 and S=2048 differed by 2.2x for 120x the work and the kernel looked
            // latency-bound when the HARNESS was. Issue the calls, then sync ONCE.
            for _ in 0..3 { let _ = q.fused_decode_attention(&k, &v, nh, nkv, dh).to_vec().await; }
            let n = 200;
            ferric_tensor::device_sync(&ctx);
            let t0 = Instant::now();
            let mut sink = None;
            for _ in 0..n { sink = Some(q.fused_decode_attention(&k, &v, nh, nkv, dh)); }
            ferric_tensor::device_sync(&ctx);
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            let _ = sink;
            // scores + weighted V-sum: 2 * nh * S * dh MACs = 4*nh*S*dh flops
            let fl = 4.0 * nh as f64 * s as f64 * dh as f64;
            println!("   {s:>6}  {ms:>10.3}  {:>12.2}  {:>10.2}", fl / 1e6, fl / (ms * 1e6));
        }
        // The thing it competes with: an FFN-sized matmul on the same device.
        let (inn, out) = (dh * nh, 4 * dh * nh);
        let x = Tensor::from_vec(&ctx, &seq(inn, 1.0), &[1, inn]);
        let w = Tensor::from_vec(&ctx, &seq(inn * out, 2.0), &[out, inn]);
        for _ in 0..3 { let _ = x.matmul_bt(&w).to_vec().await; }
        let n = 200;
        ferric_tensor::device_sync(&ctx);
        let t0 = Instant::now();
        let mut sink = None;
        for _ in 0..n { sink = Some(x.matmul_bt(&w)); }
        ferric_tensor::device_sync(&ctx);
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let _ = sink;
        let fl = 2.0 * inn as f64 * out as f64;
        println!("   {:>6}  {ms:>10.3}  {:>12.2}  {:>10.2}   <- f32 matmul [{inn}x{out}], for scale\n",
                 "mm", fl / 1e6, fl / (ms * 1e6));
    }
}
