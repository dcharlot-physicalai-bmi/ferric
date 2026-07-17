//! Q8_0-native matmul (8-bit, 8.5 bpw) vs the dequant-then-f32 path.
use ferric_core::Context;
use ferric_gguf::{deq_raw, quant_q8_0};
use ferric_tensor::{Q8_0Weights, Tensor};
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    for (rows, inn, m) in [(8usize, 256usize, 4usize), (13, 128, 1), (4096, 2048, 1)] {
        let wf: Vec<f32> = (0..rows * inn).map(|i| ((i as f32 * 0.021).sin()) * 0.7).collect();
        let mut packed = Vec::new();
        for r in 0..rows { packed.extend(quant_q8_0(&wf[r * inn..(r + 1) * inn])); }
        let qw = Q8_0Weights::from_bytes(&ctx, &packed, rows, inn);
        let wdq = Tensor::from_vec(&ctx, &deq_raw(&packed, rows * inn, 8 /*Q8_0*/).unwrap(), &[rows, inn]);
        let x = Tensor::from_vec(&ctx, &(0..m * inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[m, inn]);
        let got = x.matmul_q8_0(&qw).to_vec().await;
        let refv = x.matmul_bt(&wdq).to_vec().await;
        let e = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = refv.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let p = e / scale < 1e-5; ok &= p;
        println!("{} [{m},{inn}]·[{inn},{rows}]ᵀ  Q8_0 vs dequant  max|Δ|/scale = {:.1e}", if p { "✅" } else { "❌" }, e / scale);
    }
    println!("{}", if ok { "✅ Q8_0-native matmul is exact" } else { "❌ Q8_0 failed" });
    assert!(ok);
}
