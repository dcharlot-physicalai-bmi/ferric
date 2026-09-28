//! **Small-M quantized matmul, kernel by kernel** — ms per call at M = 1..16 rows on real decode shapes,
//! the small-M kernels against the previous ones, INTERLEAVED in one process (`dtype::set_small_m`).
//!
//! No readback in the timed loop (one `device_sync` per arm-rep), every buffer touched first — the two
//! traps `matmul_q_bench.rs` records. Shapes are Qwen2.5-0.5B's (in 896, ffn 4864, vocab 151936).
//!
//!   cargo run -p ferric-tensor --release --example small_m_bench [reps] [calls]
use ferric_core::Context;
use ferric_tensor::{dtype::QMatrix, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn blocks(n: usize, seed: u64, bpb: usize, scale_at: usize) -> Vec<u8> {
    let mut s = seed;
    let mut v: Vec<u8> = (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u8 }).collect();
    for (i, blk) in v.chunks_exact_mut(bpb).enumerate() {
        let d = half::f16::from_f32(0.01 + 0.003 * (i % 7) as f32);
        blk[scale_at..scale_at + 2].copy_from_slice(&d.to_le_bytes());
    }
    v
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(Context::new().await.expect("gpu"));
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let calls: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    println!("adapter: {} [{:?}]  reps {reps} x {calls} calls", ctx.adapter_name, ctx.backend);
    let fmts: &[(u32, &str, usize)] = &[(8, "Q8_0", 0), (6, "Q5_0", 0), (12, "Q4_K", 0), (14, "Q6_K", 208), (13, "Q5_K", 0)];
    for &(inn, out, who) in &[(896usize, 151936usize, "lm_head"), (896, 9728, "gate_up"), (4864, 896, "ffn_down"),
                              (896, 1152, "qkv"), (896, 896, "attn_out"), (1024, 6144, "q3-0.6b gate_up"), (3072, 1024, "q3-0.6b down")] {
        for &(ty, name, sat) in fmts {
            let Some((vals, bpb)) = QMatrix::block_bytes(ty) else { continue };
            if inn % vals != 0 { continue }
            let m = QMatrix::from_bytes(&ctx, &blocks(out * (inn / vals) * bpb, 99 + ty as u64, bpb, sat), ty, out, inn).unwrap();
            let gbytes = m.nbytes() as f64 / 1e9;
            println!("\n{who} {name}: in={inn} out={out} ({:.1} MB)", gbytes * 1e3);
            println!("{:>4} {:>22} {:>22} {:>8}", "M", "small-M ms (min/med)", "old ms (min/med)", "old/new");
            for mrows in [1usize, 2, 3, 4, 8, 9, 12, 16, 24, 32] {
                let x = Tensor::from_vec(&ctx, &(0..mrows * inn).map(|i| ((i * 7) as f32 * 0.013).sin()).collect::<Vec<_>>(), &[mrows, inn]);
                let mut t: [Vec<f64>; 2] = [vec![], vec![]];
                for rep in 0..=reps {
                    for (k, on) in [true, false].into_iter().enumerate() {
                        ferric_tensor::dtype::set_small_m(on);
                        let _ = x.matmul_q(&m).to_vec().await; // compile + touch
                        ferric_tensor::device_sync(&ctx);
                        let t0 = Instant::now();
                        // One batch, as a forward records them: the calls share a command buffer.
                        let sink: Vec<Tensor> = ferric_tensor::batch(&ctx, || (0..calls).map(|_| x.matmul_q(&m)).collect());
                        ferric_tensor::device_sync(&ctx);
                        let ms = t0.elapsed().as_secs_f64() * 1e3 / calls as f64;
                        drop(sink);
                        if rep > 0 { t[k].push(ms); }
                    }
                }
                for v in t.iter_mut() { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); }
                let (a, b) = (&t[0], &t[1]);
                println!("{mrows:>4} {:>10.3} / {:>9.3} {:>10.3} / {:>9.3} {:>7.2}x", a[0], a[a.len() / 2], b[0], b[b.len() / 2], b[b.len() / 2] / a[a.len() / 2]);
            }
        }
    }
}
