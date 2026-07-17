//! Q5_K-native matmul (5-bit K-quant) vs dequant reference. Q5_K = Q4_K + a 1-bit-per-value qh array.
use ferric_core::Context;
use ferric_gguf::deq_raw;
use ferric_tensor::{Q5_KWeights, Tensor};
use std::sync::Arc;
use half::f16;
fn q5k_block(seed: u32) -> Vec<u8> {
    let mut b = vec![0u8; 176]; // d, dmin, scales[12], qh[32], qs[128]
    b[0..2].copy_from_slice(&f16::from_f32(0.05 + 0.01 * ((seed % 7) as f32)).to_le_bytes());
    b[2..4].copy_from_slice(&f16::from_f32(0.02 + 0.005 * ((seed % 5) as f32)).to_le_bytes());
    let sc = |j: u32| ((seed.wrapping_mul(2654435761).wrapping_add(j * 40503)) % 64) as u8;
    let mn = |j: u32| ((seed.wrapping_mul(40503).wrapping_add(j * 2654435761)) % 64) as u8;
    for j in 0..8u32 {
        if j < 4 { b[4 + j as usize] |= sc(j) & 63; b[4 + (j + 4) as usize] |= mn(j) & 63; }
        else {
            let (scv, mnv) = (sc(j), mn(j));
            b[4 + (j + 4) as usize] |= (scv & 0x0F) | ((mnv & 0x0F) << 4);
            b[4 + (j - 4) as usize] |= (scv >> 4) << 6;
            b[4 + j as usize] |= (mnv >> 4) << 6;
        }
    }
    for i in 0..32u32 { b[16 + i as usize] = ((seed.wrapping_add(i * 2246822519)) % 256) as u8; } // qh
    for i in 0..128u32 { b[48 + i as usize] = ((seed.wrapping_mul(97).wrapping_add(i * 40503)) % 256) as u8; } // qs
    b
}
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    for (rows, inn, m) in [(8usize, 256usize, 4usize), (11, 512, 1), (2048, 1024, 1)] {
        let nblk = inn / 256;
        let mut packed = Vec::new();
        for r in 0..rows { for blk in 0..nblk { packed.extend(q5k_block((r * nblk + blk) as u32 + 1)); } }
        let qw = Q5_KWeights::from_bytes(&ctx, &packed, rows, inn);
        let wdq = Tensor::from_vec(&ctx, &deq_raw(&packed, rows * inn, 13).unwrap(), &[rows, inn]);
        let x = Tensor::from_vec(&ctx, &(0..m * inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[m, inn]);
        let got = x.matmul_q5_k(&qw).to_vec().await;
        let refv = x.matmul_bt(&wdq).to_vec().await;
        let e = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = refv.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let p = e / scale < 1e-5; ok &= p;
        println!("{} [{m},{inn}]·[{inn},{rows}]ᵀ  Q5_K vs dequant  max|Δ|/scale = {:.1e}", if p { "✅" } else { "❌" }, e / scale);
    }
    println!("{}", if ok { "✅ Q5_K-native matmul is exact" } else { "❌ Q5_K failed" });
    assert!(ok);
}
