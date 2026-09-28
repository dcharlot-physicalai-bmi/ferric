//! **Which small-M tile wins, per shape** — the sweep the planner (`dtype::mrgemv::plan`) is fitted to.
//! For each decode shape, format and row count: ms per call of the previous kernels, the planner's
//! tile, and each forced `(R outputs, M rows)` tile, interleaved rep by rep in one process.
//!
//!   cargo run -p ferric-tensor --release --example small_m_sweep [reps] [calls]
use ferric_core::Context;
use ferric_tensor::{dtype::{set_small_m, set_small_m_one, set_small_m_tile, QMatrix}, Tensor};
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
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let calls: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let only: Option<String> = std::env::args().nth(3);
    println!("adapter: {} [{:?}]  reps {reps} x {calls} calls (median ms per call)", ctx.adapter_name, ctx.backend);
    let tiles: Vec<Option<(usize, usize)>> = vec![None, Some((1, 1)), Some((1, 2)), Some((1, 4)), Some((1, 8)), Some((1, 16)),
        Some((2, 2)), Some((2, 4)), Some((2, 8)), Some((2, 16)), Some((4, 2)), Some((4, 4)), Some((4, 8)), Some((8, 4))];
    let fmts: &[(u32, &str, usize)] = &[(8, "Q8_0", 0), (6, "Q5_0", 0), (12, "Q4_K", 0), (14, "Q6_K", 208), (13, "Q5_K", 0), (42, "Q2_0", 0)];
    let rows_list: Vec<usize> = std::env::var("SWEEP_ROWS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3, 4, 8, 9, 16]);
    for &(inn, out, who, fused) in &[(896usize, 151936usize, "lm_head", false), (896, 9728, "gate_up", false), (4864, 896, "ffn_down", false),
                              (896, 1152, "qkv", false), (1024, 6144, "q3-0.6b gate_up", false), (1024, 6144, "q3-0.6b gate_up+swiglu", true),
                              (3072, 1024, "q3-0.6b down", false), (1536, 17920, "q2.5-1.5b gate_up", false),
                              (1536, 17920, "q2.5-1.5b gate_up+swiglu", true), (8960, 1536, "q2.5-1.5b down", false),
                              (5120, 34816, "bonsai2 gate_up", false), (17408, 5120, "bonsai2 down", false),
                              (5120, 10240, "bonsai2 gdn qkv", false), (5120, 12288, "bonsai2 attn q", false),
                              (6144, 5120, "bonsai2 gdn out", false), (5120, 248320, "bonsai2 lm_head", false)] {
        for &(ty, name, sat) in fmts {
            if fused && !matches!(ty, 12 | 13 | 14) { continue } // the fused portable kernels are the k-quants'
            if let Some(o) = &only { if !format!("{who} {name}").contains(o.as_str()) { continue } }
            let Some((vals, bpb)) = QMatrix::block_bytes(ty) else { continue };
            if inn % vals != 0 { continue }
            let m = QMatrix::from_bytes(&ctx, &blocks(out * (inn / vals) * bpb, 99 + ty as u64, bpb, sat), ty, out, inn).unwrap();
            println!("\n{who} {name}: in={inn} out={out}");
            print!("{:>4} {:>8}", "M", "old");
            for t in &tiles { match t { None => print!(" {:>7}", "plan"), Some((r, mm)) => print!(" {:>7}", format!("{r}x{mm}")) } }
            println!("   best");
            for &mrows in &rows_list {
                let x = Tensor::from_vec(&ctx, &(0..mrows * inn).map(|i| ((i * 7) as f32 * 0.013).sin()).collect::<Vec<_>>(), &[mrows, inn]);
                let arms: Vec<(bool, Option<(usize, usize)>)> = std::iter::once((false, None)).chain(tiles.iter().map(|&t| (true, t))).collect();
                let mut t: Vec<Vec<f64>> = vec![vec![]; arms.len()];
                for rep in 0..=reps {
                    for (k, &(on, tile)) in arms.iter().enumerate() {
                        set_small_m(on); set_small_m_tile(tile); set_small_m_one(on && mrows == 1);
                        let call = || if fused { x.try_matmul_swiglu(&m).expect("fused k-quant") } else { x.matmul_q(&m) };
                        if rep == 0 { let _ = call().to_vec().await; continue; }
                        ferric_tensor::device_sync(&ctx);
                        let t0 = Instant::now();
                        let sink: Vec<Tensor> = ferric_tensor::batch(&ctx, || (0..calls).map(|_| call()).collect());
                        ferric_tensor::device_sync(&ctx);
                        t[k].push(t0.elapsed().as_secs_f64() * 1e3 / calls as f64);
                        drop(sink);
                    }
                }
                let med: Vec<f64> = t.iter_mut().map(|v| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }).collect();
                print!("{mrows:>4} {:>8.3}", med[0]);
                for v in &med[1..] { print!(" {:>7.3}", v); }
                let (bi, bv) = med.iter().enumerate().skip(1).fold((0, f64::MAX), |a, (i, &v)| if v < a.1 { (i, v) } else { a });
                let bn = match arms[bi].1 { None => "plan".to_string(), Some((r, mm)) => format!("{r}x{mm}") };
                println!("   {bn} {:.2}x old, plan {:.2}x old", med[0] / bv, med[0] / med[1]);
            }
        }
    }
    set_small_m_tile(None);
}
