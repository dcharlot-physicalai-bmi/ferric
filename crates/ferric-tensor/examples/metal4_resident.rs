//! **Resident tensor-unit GEMM** — `Tensor::matmul` routed through the Metal-4 tensor units with
//! ZERO host copies: pad+f16-convert, `matmul2d`, and unpad all run as one MTL4 command buffer on
//! wgpu's own `MTLDevice`, reading and writing the wgpu-resident tensor buffers directly.
//!
//! Opt-in per the crate's precision doctrine (`FERRIC_METAL4=1`, like `FERRIC_COOP`): the tensor
//! units take fp16 inputs, so results are verified against an **fp16-input** CPU oracle, not the f32
//! one. The sweep times the portable WGSL path vs the resident tensor-unit path on the same shapes,
//! checks both against the oracle, and demonstrates the training pattern (repeated same-shape calls
//! hitting the shape cache).

use ferric_core::Context;
use ferric_tensor::{device_sync, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| 0.01 * (((i + salt) % 13) as f32 - 6.0)).collect()
}

/// fp16-input CPU oracle (the tensor units' contract: fp16 operands, fp32 accumulate).
fn cpu_ref_f16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect() };
    let (af, bf) = (q(a), q(b));
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|l| af[i * k + l] * bf[l * n + j]).sum();
        }
    }
    c
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() {
    let ctx = Arc::new(pollster::block_on(Context::new()).expect("a GPU context"));
    println!("adapter: {} ({:?})", ctx.adapter_name, ctx.backend);

    let time_matmul = |a: &Tensor, b: &Tensor, reps: usize| {
        let _ = pollster::block_on(a.matmul(b).to_vec()); // warm (kernel/pipeline/shape caches)
        let t0 = Instant::now();
        let mut last = None;
        for _ in 0..reps {
            last = Some(a.matmul(b));
        }
        let out = last.unwrap();
        let res = pollster::block_on(out.to_vec());
        (t0.elapsed().as_secs_f64() / reps as f64, res)
    };

    println!("\n=== portable WGSL vs resident Metal-4 tensor units (same tensors, same call) ===");
    println!("  {:>5}  {:>10}  {:>10}  {:>9}  {:>10}  {:>9}", "N", "wgsl (ms)", "GFLOP/s", "m4 (ms)", "GFLOP/s", "speedup");
    for &nn in &[512usize, 1024, 2048] {
        let (av, bv) = (gen(nn * nn, 1), gen(nn * nn, 7));
        let a = Tensor::from_vec(&ctx, &av, &[nn, nn]);
        let b = Tensor::from_vec(&ctx, &bv, &[nn, nn]);
        let flops = 2.0 * (nn as f64).powi(3);

        std::env::remove_var("FERRIC_METAL4");
        let (t_wgsl, r_wgsl) = time_matmul(&a, &b, 3);
        std::env::set_var("FERRIC_METAL4", "1");
        let (t_m4, r_m4) = time_matmul(&a, &b, 3);

        // verify: ≤1024 against the fp16 CPU oracle; 2048 against the WGSL f32 result (fp16 tol)
        let err = if nn <= 1024 {
            let oracle = cpu_ref_f16(&av, &bv, nn, nn, nn);
            max_abs_diff(&r_m4, &oracle)
        } else {
            max_abs_diff(&r_m4, &r_wgsl)
        };
        let tol = if nn <= 1024 { 1e-3 } else { 1e-1 };
        assert!(err < tol, "resident result off at N={nn}: err {err}");
        println!(
            "  {:>5}  {:>10.3}  {:>10.1}  {:>9.3}  {:>10.1}  {:>8.1}x",
            nn,
            t_wgsl * 1e3,
            flops / t_wgsl / 1e9,
            t_m4 * 1e3,
            flops / t_m4 / 1e9,
            t_wgsl / t_m4
        );
        assert!(t_m4 < t_wgsl, "tensor units should beat WGSL at N={nn}");
    }

    // the training pattern: same shape over and over → every call after the first reuses the cache
    println!("\n=== cache-reuse cadence (training pattern, N=1024, 20 back-to-back calls) ===");
    let nn = 1024;
    let a = Tensor::from_vec(&ctx, &gen(nn * nn, 3), &[nn, nn]);
    let b = Tensor::from_vec(&ctx, &gen(nn * nn, 9), &[nn, nn]);
    let _ = pollster::block_on(a.matmul(&b).to_vec());
    let t0 = Instant::now();
    for _ in 0..20 {
        let _ = a.matmul(&b);
    }
    device_sync(&ctx);
    let per = t0.elapsed().as_secs_f64() / 20.0;
    println!("  {:.3} ms/call  ({:.1} GFLOP/s sustained)", per * 1e3, 2.0 * (nn as f64).powi(3) / per / 1e9);

    // the inference hot path: y = silu(x·Wᵀ), W in the HF [out,in] layout (a llama-class FFN
    // projection) — NT on the tensor units, activation fused into the unpad epilogue
    println!("\n=== linear layers (x·Wᵀ, HF layout — the ferric-llama hot path) ===");
    println!("  {:>22}  {:>10}  {:>10}  {:>9}  {:>10}  {:>9}", "shape", "wgsl (ms)", "GFLOP/s", "m4 (ms)", "GFLOP/s", "speedup");
    for &(rows, inn, out_f) in &[(32usize, 2048usize, 8192usize), (64, 4096, 4096)] {
        let x = Tensor::from_vec(&ctx, &gen(rows * inn, 1), &[rows, inn]);
        let w = Tensor::from_vec(&ctx, &gen(out_f * inn, 7), &[out_f, inn]);
        let flops = 2.0 * (rows * inn * out_f) as f64;

        std::env::remove_var("FERRIC_METAL4");
        let time_bt = |reps: usize| {
            let _ = pollster::block_on(x.matmul_bt_act(&w, 2).to_vec());
            let t0 = Instant::now();
            let mut last = None;
            for _ in 0..reps {
                last = Some(x.matmul_bt_act(&w, 2));
            }
            let res = pollster::block_on(last.unwrap().to_vec());
            (t0.elapsed().as_secs_f64() / reps as f64, res)
        };
        let (t_wgsl, r_wgsl) = time_bt(3);
        std::env::set_var("FERRIC_METAL4", "1");
        let (t_m4, r_m4) = time_bt(3);
        let err = max_abs_diff(&r_m4, &r_wgsl);
        assert!(err < 5e-2, "resident linear off at {rows}x{inn}x{out_f}: err {err}");
        assert!(t_m4 < t_wgsl, "tensor units should beat WGSL at {rows}x{inn}x{out_f}");
        println!(
            "  {:>22}  {:>10.3}  {:>10.1}  {:>9.3}  {:>10.1}  {:>8.1}x",
            format!("[{rows},{inn}]·[{out_f},{inn}]ᵀ"),
            t_wgsl * 1e3,
            flops / t_wgsl / 1e9,
            t_m4 * 1e3,
            flops / t_m4 / 1e9,
            t_wgsl / t_m4
        );
    }

    // quantized prefill — Bonsai-27B's real FFN shape, packed Q2_0 ternary. Decode (1 token) stays
    // on the fused scalar kernel (bandwidth-optimal on packed bytes); prefill routes dequant-once →
    // tensor-unit GEMM. The 32-row threshold in matmul_q2_0 is what this section justifies (1.3x there, 9x at 512).
    println!("\n=== Q2_0 ternary prefill (Bonsai ffn_gate/up shape: 5120→17408) ===");
    println!("  {:>5}  {:>11}  {:>10}  {:>10}  {:>10}  {:>9}", "toks", "fused (ms)", "GFLOP/s", "m4 (ms)", "GFLOP/s", "speedup");
    {
        let (inn, out_f) = (5120usize, 17408usize);
        let wsrc: Vec<f32> = (0..out_f * inn).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
        let mut packed = Vec::with_capacity(out_f * (inn / 128) * 34);
        for r in 0..out_f {
            packed.extend(ferric_gguf::quant_q2_0(&wsrc[r * inn..(r + 1) * inn]));
        }
        let qw = ferric_tensor::Q2_0Weights::from_bytes(&ctx, &packed, out_f, inn);
        for toks in [32usize, 128, 512] {
            let x = Tensor::from_vec(&ctx, &gen(toks * inn, 5), &[toks, inn]);
            let flops = 2.0 * (toks * inn * out_f) as f64;
            std::env::remove_var("FERRIC_METAL4");
            let time_q = |reps: usize| {
                let _ = pollster::block_on(x.matmul_q2_0(&qw).to_vec());
                let t0 = Instant::now();
                let mut last = None;
                for _ in 0..reps {
                    last = Some(x.matmul_q2_0(&qw));
                }
                let res = pollster::block_on(last.unwrap().to_vec());
                (t0.elapsed().as_secs_f64() / reps as f64, res)
            };
            let (t_fused, r_fused) = time_q(3);
            std::env::set_var("FERRIC_METAL4", "1");
            let (t_m4, r_m4) = time_q(3);
            // fp16-contract tolerance, relative to the result scale
            let scale = r_fused.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let err = max_abs_diff(&r_m4, &r_fused);
            assert!(err < (1e-2 * scale).max(1e-3), "Q2_0 metal4 off at {toks} toks: err {err} (scale {scale})");
            println!(
                "  {:>5}  {:>11.3}  {:>10.1}  {:>10.3}  {:>10.1}  {:>8.1}x",
                toks,
                t_fused * 1e3,
                flops / t_fused / 1e9,
                t_m4 * 1e3,
                flops / t_m4 / 1e9,
                t_fused / t_m4
            );
        }
    }

    println!("\n✅ resident tensor-unit path: correct vs the fp16 oracle, faster than WGSL, zero host copies");
}
