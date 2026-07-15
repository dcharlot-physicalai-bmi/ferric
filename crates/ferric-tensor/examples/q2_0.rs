//! Q2_0-native matmul: PrismML ternary weights stay PACKED on the GPU (2.125 bpw) and are
//! dequantized per-128-group inside the kernel. Validated against dequantize-then-f32-matmul —
//! identical math, but a 27B model needs ~7 GB instead of the 108 GB f32 would demand.
use ferric_core::Context;
use ferric_gguf::quant_q2_0;
use ferric_tensor::{Q2_0Weights, Tensor};
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.013 + s).sin())).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (rows, inn, outf) = (4usize, 256usize, 8usize); // in must be a multiple of 128
    let x = Tensor::from_vec(&ctx, &seq(rows * inn, 1.0), &[rows, inn]);
    let wf = seq(outf * inn, 2.0); // [out, in]

    // pack each row's `in` values into Q2_0 group-128 blocks (row-major, as the GGUF stores them)
    let mut bytes = Vec::new();
    for o in 0..outf { bytes.extend_from_slice(&quant_q2_0(&wf[o * inn..(o + 1) * inn])); }
    let qw = Q2_0Weights::from_bytes(&ctx, &bytes, outf, inn);

    // reference: dequantize the same blocks to f32, then a normal matmul
    let deq: Vec<f32> = {
        let mut v = Vec::with_capacity(outf * inn);
        for o in 0..outf {
            for blk in 0..inn / 128 {
                let b = &bytes[(o * (inn / 128) + blk) * 34..][..34];
                let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..128 { v.push((((b[2 + j / 4] >> ((j % 4) * 2)) & 3) as i32 - 1) as f32 * d); }
            }
        }
        v
    };
    let wref = Tensor::from_vec(&ctx, &deq, &[outf, inn]);
    let expect = x.matmul_bt(&wref).to_vec().await;
    let got = x.matmul_q2_0(&qw).to_vec().await;

    let e = got.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let f32_bytes = outf * inn * 4;
    println!("Ferric · Q2_0-native matmul (PrismML ternary, weights stay packed) · {:?}", ctx.backend);
    println!("  [{rows},{inn}]·[{inn},{outf}]ᵀ · max|packed - dequantized| = {e:.2e}");
    println!("  weights: {f32_bytes} B f32 → {} B packed ({:.3} bpw)", qw.nbytes(), qw.nbytes() as f32 * 8.0 / (outf * inn) as f32);
    assert!(e < 1e-4, "Q2_0 matmul mismatch {e}");
    println!("✅ Q2_0-native matmul is exact — a 27B ternary model fits in ~7 GB, not 108 GB");
}
