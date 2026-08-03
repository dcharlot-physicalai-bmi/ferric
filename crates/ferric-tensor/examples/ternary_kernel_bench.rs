//! TERNARY KERNEL: does "multiply-free" actually buy anything on a GPU?
//!
//! BitNet.cpp and T-MAC get 4-6× and −70% energy from multiply-free / lookup-table ternary — but those are
//! **CPU** wins, where a multiply is expensive and table lookup + SIMD shuffle is cheap. A GPU has fused
//! multiply-add as a SINGLE instruction, so removing the multiply may buy nothing, and branching to do it can
//! cost more than it saves. Ferric's existing ternary kernel even admits this in a comment: it converts each
//! trit to f32 and multiplies ("multiply-free in spirit").
//!
//! This measures three paths on identical data:
//!   f32 dense           — the baseline every quantization scheme is trying to beat
//!   ternary (multiply)  — packed 2-bit weights, trit→f32 then FMA   [existing kernel]
//!   ternary (mult-FREE) — packed 2-bit weights, add/sub/skip via branchless select   [new kernel]
//! and verifies all three agree numerically before comparing speed.
//!   cargo run -p ferric-tensor --example ternary_kernel_bench --release
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let mut seed = 0xBEEF_1234u64;
    let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 40) as f32 + 0.5) / (1u32 << 24) as f32 };

    // decode-shaped workload (rows=1) and a prefill-shaped one (rows=64), both at a realistic hidden size.
    // Bigger workload: under background contention (a macOS daemon spiking to 80% CPU) a ~2ms kernel is the
    // same order as scheduler jitter, so a 7% difference is unmeasurable. Scaling the matmul up makes the
    // kernel time dominate the noise — the right way to measure a small effect on a busy machine.
    for (rows, inn, outn, label) in [(256usize, 4096usize, 4096usize, "batch-256 [256 x 4096 x 4096]"),
                                     (256, 8192, 4096, "deep-K    [256 x 8192 x 4096]")] {
        let xv: Vec<f32> = (0..rows * inn).map(|_| u() * 2.0 - 1.0).collect();
        let wv: Vec<f32> = (0..outn * inn).map(|_| (u() * 2.0 - 1.0) * 0.05).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[rows, inn]);
        let wf = Tensor::from_vec(&ctx, &wv, &[outn, inn]);
        let wt = wf.quantize_ternary();

        // correctness: the two ternary kernels must agree with each other exactly (same math, different ops),
        // and both must track the f32 dense result to within ternary's inherent quantization error.
        let y_dense = x.matmul_bt(&wf).to_vec().await;
        let y_mul = x.matmul_ternary(&wt).to_vec().await;
        let y_mf = x.matmul_ternary_mf(&wt).to_vec().await;
        let y_coop = x.matmul_ternary_coop(&wt).to_vec().await;
        let y_c4 = x.matmul_ternary_coop4(&wt).to_vec().await;
        let y_n16 = x.matmul_ternary_coop_n16(&wt).to_vec().await;
        let y_n32 = x.matmul_ternary_coop_n32(&wt).to_vec().await;
        let y_n64 = x.matmul_ternary_coop_n64(&wt).to_vec().await;
        let maxdiff = y_mul.iter().zip(&y_mf).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let rel = |a: &[f32], b: &[f32]| {
            let n: f32 = a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt();
            let d: f32 = a.iter().map(|p| p * p).sum::<f32>().sqrt();
            n / d.max(1e-9)
        };

        // A single timing run is NOT a benchmark. The first version of this bench reported multiply-free as a
        // 10% WIN and then a 11% LOSS on the next run — pure noise on a contended GPU, and I reported it as a
        // result. Fix: many independent trials, take the MEDIAN (robust to scheduler hiccups), and report the
        // spread so a "speedup" smaller than the noise floor is visibly not a speedup.
        let bench = |f: &dyn Fn() -> Tensor| -> (f64, f64) {
            for _ in 0..5 { let _ = pollster::block_on(f().to_vec()); }        // warm up
            let trials = 9; let iters = 15;
            let mut ts: Vec<f64> = Vec::new();
            for _ in 0..trials {
                let t0 = Instant::now();
                for _ in 0..iters { let _ = pollster::block_on(f().to_vec()); }
                ts.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            ts.sort_by(f64::total_cmp);
            let med = ts[trials / 2];
            (med, 100.0 * (ts[trials - 1] - ts[0]) / med)                       // (median ms, spread %)
        };
        let (t_dense, s_dense) = bench(&|| x.matmul_bt(&wf));
        let (t_mul, s_mul) = bench(&|| x.matmul_ternary(&wt));
        let (t_mf, s_mf) = bench(&|| x.matmul_ternary_mf(&wt));
        let (t_coop, s_coop) = bench(&|| x.matmul_ternary_coop(&wt));
        let (t_c4, s_c4) = bench(&|| x.matmul_ternary_coop4(&wt));
        let (t_n16, s_n16) = bench(&|| x.matmul_ternary_coop_n16(&wt));
        let (t_n32, s_n32) = bench(&|| x.matmul_ternary_coop_n32(&wt));
        let (t_n64, s_n64) = bench(&|| x.matmul_ternary_coop_n64(&wt));

        println!("\n{label}");
        println!("  correctness: ternary-mul vs ternary-mult-free  max|Δ| {maxdiff:.2e} {}",
                 if maxdiff < 1e-3 { "✓ identical math" } else { "✗ KERNELS DISAGREE" });
        assert!(maxdiff < 1e-3, "ternary multiply vs multiply-free kernels DISAGREE (max|Δ| {maxdiff:.2e})");
        println!("               ternary vs f32 dense              rel err {:.3}  (ternary's inherent loss)", rel(&y_mul, &y_dense));
        println!("  (median of 9 trials; spread = (max-min)/median — a 'win' inside the spread is noise)");
        println!("  f32 dense            {t_dense:7.3} ms  ±{s_dense:4.1}%");
        println!("  ternary (multiply)   {t_mul:7.3} ms  ±{s_mul:4.1}%   {:.2}× vs dense", t_dense / t_mul);
        let delta = 100.0 * (t_mf - t_mul) / t_mul;
        println!("  ternary (mult-FREE)  {t_mf:7.3} ms  ±{s_mf:4.1}%   {:.2}× vs dense   {delta:+.1}% vs multiply {}",
                 t_dense / t_mf, if delta.abs() < s_mul.max(s_mf) { "← INSIDE NOISE, not a real difference" } else { "" });
        let mdc = y_coop.iter().zip(&y_mul).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(mdc < 1e-3, "coop kernel disagrees with scalar reference (max|Δ| {mdc:.2e})");
        let dc = 100.0 * (t_coop - t_mf) / t_mf;
        println!("  ternary (COOP/tensor){t_coop:7.3} ms  ±{s_coop:4.1}%   {:.2}× vs dense   {:.2}× vs mult-free {}  [Δ {mdc:.1e}]",
                 t_dense / t_coop, t_mf / t_coop, if dc.abs() < s_coop.max(s_mf) { "← inside noise" } else { "← REAL" });
        let d4 = 100.0 * (t_c4 - t_coop) / t_coop;
        let md4 = y_c4.iter().zip(&y_coop).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  ternary (COOP K=32)  {t_c4:7.3} ms  ±{s_c4:4.1}%   {:.2}× vs dense   {d4:+.1}% vs COOP K=8 {}  [Δ {md4:.1e}]",
                 t_dense / t_c4, if d4.abs() < s_c4.max(s_coop) { "← inside noise" } else { "← REAL" });
        let dn = 100.0 * (t_n16 - t_coop) / t_coop;
        let mdn = y_n16.iter().zip(&y_coop).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  ternary (COOP N=16)  {t_n16:7.3} ms  ±{s_n16:4.1}%   {:.2}× vs dense   {dn:+.1}% vs COOP N=8 {}  [Δ {mdn:.1e}]",
                 t_dense / t_n16, if dn.abs() < s_n16.max(s_coop) { "← inside noise" } else if dn < 0.0 { "← REAL WIN" } else { "← REAL regression" });
        let d32 = 100.0 * (t_n32 - t_n16) / t_n16;
        let md32 = y_n32.iter().zip(&y_n16).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  ternary (COOP N=32)  {t_n32:7.3} ms  ±{s_n32:4.1}%   {:.2}× vs dense   {d32:+.1}% vs COOP N=16 {}  [Δ {md32:.1e}]",
                 t_dense / t_n32, if d32.abs() < s_n32.max(s_n16) { "← inside noise" } else if d32 < 0.0 { "← REAL WIN" } else { "← REAL regression" });
        let d64 = 100.0 * (t_n64 - t_n32) / t_n32;
        let md64 = y_n64.iter().zip(&y_n32).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  ternary (COOP N=64)  {t_n64:7.3} ms  ±{s_n64:4.1}%   {:.2}× vs dense   {d64:+.1}% vs COOP N=32 {}  [Δ {md64:.1e}]",
                 t_dense / t_n64, if d64.abs() < s_n64.max(s_n32) { "← inside noise" } else if d64 < 0.0 { "← REAL WIN" } else { "← REAL regression (limit found)" });
    }
    println!("\nWeight memory: f32 4096×4096 = {:.1} MB → ternary 2-bit = {:.1} MB (16× smaller)",
             4096.0 * 4096.0 * 4.0 / 1e6, 4096.0 * 4096.0 * 2.0 / 8.0 / 1e6);
    println!("Read the result honestly: on a GPU, FMA is one instruction, so removing the multiply may buy");
    println!("nothing — ternary's GPU win is MEMORY BANDWIDTH (16× fewer weight bytes), not arithmetic.");
    println!("The multiply-free win that BitNet.cpp/T-MAC report is a CPU phenomenon (LUT + SIMD).");
}
