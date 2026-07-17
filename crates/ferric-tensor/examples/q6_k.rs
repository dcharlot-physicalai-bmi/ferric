//! Q6_K-native matmul (6-bit K-quant, ~6.5 bpw) vs dequant-then-f32. Q6_K is required to run real
//! Q4_K_M models (their embedding/output tensors are Q6_K). Synthesize valid blocks, dequant with
//! the reference (deq_raw), require the packed matmul to match.
use ferric_core::Context;
use ferric_gguf::deq_raw;
use ferric_tensor::{Q6_KWeights, Tensor};
use std::sync::Arc;
use half::f16;
fn q6k_block(seed: u32) -> Vec<u8> {
    let mut b = vec![0u8; 210];
    for i in 0..128u32 { b[i as usize] = ((seed.wrapping_add(i * 2246822519)) % 256) as u8; }       // ql
    for i in 0..64u32 { b[128 + i as usize] = ((seed.wrapping_mul(40503).wrapping_add(i * 97)) % 256) as u8; } // qh
    for i in 0..16u32 { b[192 + i as usize] = (((seed.wrapping_add(i * 7)) % 64) as i32 - 32) as i8 as u8; }   // int8 scales
    let d = f16::from_f32(0.04 + 0.01 * ((seed % 6) as f32));
    b[208..210].copy_from_slice(&d.to_le_bytes());
    b
}
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    for (rows, inn, m) in [(8usize, 256usize, 4usize), (7, 512, 1), (2048, 1024, 1)] {
        let nblk = inn / 256;
        let mut packed = Vec::new();
        for r in 0..rows { for blk in 0..nblk { packed.extend(q6k_block((r * nblk + blk) as u32 + 1)); } }
        let qw = Q6_KWeights::from_bytes(&ctx, &packed, rows, inn);
        let wdq = Tensor::from_vec(&ctx, &deq_raw(&packed, rows * inn, 14).unwrap(), &[rows, inn]);
        let x = Tensor::from_vec(&ctx, &(0..m * inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[m, inn]);
        let got = x.matmul_q6_k(&qw).to_vec().await;
        let refv = x.matmul_bt(&wdq).to_vec().await;
        let e = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = refv.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let p = e / scale < 1e-5; ok &= p;
        println!("{} [{m},{inn}]·[{inn},{rows}]ᵀ  Q6_K vs dequant  max|Δ|/scale = {:.1e}", if p { "✅" } else { "❌" }, e / scale);
    }
    println!("{}", if ok { "✅ Q6_K-native matmul is exact — Q4_K_M models (Q4_K+Q6_K) can run packed" } else { "❌ Q6_K failed" });
    assert!(ok);
}
