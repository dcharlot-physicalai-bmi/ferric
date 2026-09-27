//! **Quantized GEMM throughput at a model's prefill shapes** — the portable WGSL kernels against the
//! Metal-4 tensor-unit route (`FERRIC_QGEMM=1`), same process layout, same random weights.
//!
//!   FERRIC_QGEMM=1 cargo run -p ferric-tensor --release --example qgemm_bench -- <fmt> <rows> <k:n>...
//!   e.g. qgemm_bench q8_0 512 896:1152 896:896 896:9728 4864:896      (Qwen2.5-0.5B, Q8_0)
//!        qgemm_bench q4_k 512 1536:2048 1536:1536 1536:17920 8960:1536 (Qwen2.5-1.5B, Q4_K)
//!
//! `rows` may be a comma list (e.g. `1,8,16,32,64,512`) to find where the tensor units start to pay.
//! Each shape is timed as REPS back-to-back GEMMs inside one `batch()` (one submit, one wait), best of
//! 5, so the number is device time plus one submit — the way a prefill issues them. The route is read
//! once per process, so compare two runs; the env is echoed so a pair can't be mislabelled.
use ferric_tensor::dtype::{Q4_KWeights, Q6_KWeights, Q8_0Weights};
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let fmt = a.first().expect("usage: qgemm_bench <q8_0|q4_k|q6_k> <rows[,rows..]> <k:n>...").clone();
    let rows_list: Vec<usize> = a[1].split(',').map(|s| s.parse().unwrap()).collect();
    let shapes: Vec<(usize, usize)> = a[2..].iter().map(|s| {
        let (k, n) = s.split_once(':').unwrap();
        (k.parse().unwrap(), n.parse().unwrap())
    }).collect();
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter {}  FERRIC_QGEMM={}  fmt {fmt}", ctx.adapter_name, std::env::var("FERRIC_QGEMM").unwrap_or_default());
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    for &rows in &rows_list {
        let mut tot_flop = 0f64;
        let mut tot_s = 0f64;
        for &(k, n) in &shapes {
            let (bs, bb) = match fmt.as_str() { "q8_0" => (32, 34), "q4_k" => (256, 144), _ => (256, 210) };
            let mut raw = vec![0u8; n * k / bs * bb];
            for (i, b) in raw.iter_mut().enumerate() { *b = rnd() as u8; if fmt == "q6_k" && i % bb == bb - 1 { *b = 0x10; } }
            // keep the f16 scales finite and small: overwrite each block's scale bytes
            for blk in raw.chunks_mut(bb) {
                let h = half::f16::from_f32(0.001 + (rnd() % 100) as f32 * 1e-5).to_bits().to_le_bytes();
                match fmt.as_str() {
                    "q8_0" => blk[..2].copy_from_slice(&h),
                    "q4_k" => { blk[..2].copy_from_slice(&h); blk[2..4].copy_from_slice(&h); }
                    _ => blk[208..210].copy_from_slice(&h),
                }
            }
            let x: Vec<f32> = (0..rows * k).map(|_| (rnd() % 2001) as f32 * 1e-3 - 1.0).collect();
            let xt = Tensor::from_vec(&ctx, &x, &[rows, k]);
            enum W { Q8(Q8_0Weights), Q4(Q4_KWeights), Q6(Q6_KWeights) }
            let w = match fmt.as_str() {
                "q8_0" => W::Q8(Q8_0Weights::from_bytes(&ctx, &raw, n, k)),
                "q4_k" => W::Q4(Q4_KWeights::from_bytes(&ctx, &raw, n, k)),
                _ => W::Q6(Q6_KWeights::from_bytes(&ctx, &raw, n, k)),
            };
            let mm = |t: &Tensor| match &w { W::Q8(w) => t.matmul_q8_0(w), W::Q4(w) => t.matmul_q4_k(w), W::Q6(w) => t.matmul_q6_k(w) };
            let go = |r: usize| {
                let outs = ferric_tensor::batch(&ctx, || (0..r).map(|_| mm(&xt)).collect::<Vec<_>>());
                ferric_tensor::device_sync(&ctx);
                drop(outs);
            };
            go(2);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                go(reps);
                best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
            }
            let flop = 2.0 * (rows * k * n) as f64;
            tot_flop += flop;
            tot_s += best;
            println!("  rows {rows:5}  K {k:5} N {n:6}: {:8.3} ms  {:7.2} TFLOP/s", best * 1e3, flop / best / 1e12);
        }
        println!("rows {rows:5}: all shapes {:.3} ms, {:.2} TFLOP/s", tot_s * 1e3, tot_flop / tot_s / 1e12);
    }
}
