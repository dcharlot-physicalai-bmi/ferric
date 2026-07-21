//! Q5_0-native matmul (weights stay packed at 5.5 bpw): validate against dequantize-then-f32-matmul.
//! Q5_0 is llama.cpp's symmetric 5-bit — a 4-bit nibble plus a 5th bit pulled from a per-block `qh`.
//! The in-kernel dequant reassembles the 5-bit code on the fly, so those GGUFs run packed, ~6× lighter.
use ferric_core::Context;
use ferric_gguf::{deq_raw, quant_q5_0};
use ferric_tensor::{Q5_0Weights, Tensor};
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    for (rows, inn, m) in [(8usize, 256usize, 4usize), (17, 128, 1), (5120, 2048, 1)] {
        let wf: Vec<f32> = (0..rows * inn).map(|i| ((i as f32 * 0.021).sin()) * 0.7).collect();
        let mut packed = Vec::new();
        for r in 0..rows { packed.extend(quant_q5_0(&wf[r * inn..(r + 1) * inn])); }
        let qw = Q5_0Weights::from_bytes(&ctx, &packed, rows, inn);
        // reference: dequant the packed bytes back to f32 and do a plain matmul (same bytes both paths)
        let wdq = deq_raw(&packed, rows * inn, 6 /*Q5_0*/).unwrap();
        let wt = Tensor::from_vec(&ctx, &wdq, &[rows, inn]);
        let x = Tensor::from_vec(&ctx, &(0..m * inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[m, inn]);
        let got = x.matmul_q5_0(&qw).to_vec().await;
        let refv = x.matmul_bt(&wt).to_vec().await;
        let e = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = refv.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let p = e / scale < 1e-5; ok &= p;
        let bpw = 22.0 * 8.0 / 32.0;
        println!("{} [{m},{inn}]·[{inn},{rows}]ᵀ  Q5_0 ({bpw:.2} bpw) vs dequant  max|Δ|/scale = {:.1e}", if p { "✅" } else { "❌" }, e / scale);
    }
    // throughput: native packed vs the dequant-then-f32 path it replaces, at a decode shape
    let (rows, inn) = (4096usize, 4096usize);
    let wf: Vec<f32> = (0..rows * inn).map(|i| ((i as f32 * 0.021).sin()) * 0.7).collect();
    let mut packed = Vec::new();
    for r in 0..rows { packed.extend(quant_q5_0(&wf[r * inn..(r + 1) * inn])); }
    let qw = Q5_0Weights::from_bytes(&ctx, &packed, rows, inn);
    let wdq = Tensor::from_vec(&ctx, &deq_raw(&packed, rows * inn, 6).unwrap(), &[rows, inn]);
    let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[1, inn]);
    let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t = std::time::Instant::now();
        for _ in 0..30 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64() * 1e3 / 30.0 };
    let _ = pollster::block_on(x.matmul_q5_0(&qw).to_vec()); // warm
    let nat = bench(&|| x.matmul_q5_0(&qw));
    let deq = bench(&|| x.matmul_bt(&wdq));
    println!("  {rows}→{inn} GEMV: native Q5_0 {nat:.3} ms vs dequant→f32 {deq:.3} ms  ({:.1}× faster, 5.8× less memory)", deq / nat);

    println!("{}", if ok { "✅ Q5_0-native matmul is exact — standard 5-bit GGUFs run packed" } else { "❌ Q5_0 failed" });
    assert!(ok);
}
