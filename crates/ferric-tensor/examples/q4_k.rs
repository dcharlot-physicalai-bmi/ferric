//! Q4_K-native matmul (the default llama.cpp quant, ~4.5 bpw) validated against the reference
//! dequantizer. Q4_K is the most common format on Hugging Face, so a native packed path is the
//! biggest single "run standard models fast + light" win. No quantizer needed: synthesize valid
//! Q4_K super-block bytes, dequant them with the reference (deq_raw), and require the packed matmul
//! to match dequant-then-f32.
use ferric_core::Context;
use ferric_gguf::deq_raw;
use ferric_tensor::{Q4_KWeights, Tensor};
use std::sync::Arc;
use half::f16;

/// Build one synthetic Q4_K super-block (144 bytes, 256 values) with seeded-but-varied contents.
fn q4k_block(seed: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    let d = 0.05 + 0.01 * ((seed % 7) as f32);
    let dmin = 0.02 + 0.005 * ((seed % 5) as f32);
    b.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    b.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
    // 12 scale bytes: 8 six-bit (scale,min) pairs, packed exactly as llama.cpp expects.
    let sc = |j: u32| ((seed.wrapping_mul(2654435761).wrapping_add(j * 40503)) % 64) as u8; // 0..63
    let mn = |j: u32| ((seed.wrapping_mul(40503).wrapping_add(j * 2654435761)) % 64) as u8;
    let mut s = [0u8; 12];
    for j in 0..8u32 {
        if j < 4 {
            s[j as usize] |= sc(j) & 63;
            s[(j + 4) as usize] |= mn(j) & 63;
        } else {
            let (scv, mnv) = (sc(j), mn(j));
            s[(j + 4) as usize] |= (scv & 0x0F) | ((mnv & 0x0F) << 4);
            s[(j - 4) as usize] |= (scv >> 4) << 6;
            s[j as usize] |= (mnv >> 4) << 6;
        }
    }
    b.extend_from_slice(&s);
    for i in 0..128u32 { b.push((((seed.wrapping_add(i * 2246822519)) % 256) as u8) & 0xff); } // 128 quant bytes
    b
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    for (rows, inn, m) in [(8usize, 256usize, 4usize), (13, 512, 1), (4096, 2048, 1)] {
        let nblk = inn / 256;
        let mut packed = Vec::new();
        for r in 0..rows { for blk in 0..nblk { packed.extend(q4k_block((r * nblk + blk) as u32 + 1)); } }
        let qw = Q4_KWeights::from_bytes(&ctx, &packed, rows, inn);
        let wdq = Tensor::from_vec(&ctx, &deq_raw(&packed, rows * inn, 12 /*Q4_K*/).unwrap(), &[rows, inn]);
        let x = Tensor::from_vec(&ctx, &(0..m * inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[m, inn]);
        let got = x.matmul_q4_k(&qw).to_vec().await;
        let refv = x.matmul_bt(&wdq).to_vec().await;
        let e = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = refv.iter().map(|v| v.abs()).fold(1e-3, f32::max);
        let p = e / scale < 1e-5; ok &= p;
        println!("{} [{m},{inn}]·[{inn},{rows}]ᵀ  Q4_K vs dequant  max|Δ|/scale = {:.1e}", if p { "✅" } else { "❌" }, e / scale);
    }
    // throughput vs the dequant-then-f32 path
    let (rows, inn) = (4096usize, 4096usize);
    let mut packed = Vec::new();
    for i in 0..rows * (inn / 256) { packed.extend(q4k_block(i as u32 + 1)); }
    let qw = Q4_KWeights::from_bytes(&ctx, &packed, rows, inn);
    let wdq = Tensor::from_vec(&ctx, &deq_raw(&packed, rows * inn, 12).unwrap(), &[rows, inn]);
    let x = Tensor::from_vec(&ctx, &(0..inn).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>(), &[1, inn]);
    let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t = std::time::Instant::now();
        for _ in 0..30 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64() * 1e3 / 30.0 };
    let _ = pollster::block_on(x.matmul_q4_k(&qw).to_vec());
    let nat = bench(&|| x.matmul_q4_k(&qw));
    let deq = bench(&|| x.matmul_bt(&wdq));
    println!("  {rows}→{inn} GEMV: native Q4_K {nat:.3} ms vs dequant→f32 {deq:.3} ms  ({:.1}× faster, ~7× less memory)", deq / nat);
    println!("{}", if ok { "✅ Q4_K-native matmul is exact — the default llama.cpp quant runs packed" } else { "❌ Q4_K failed" });
    assert!(ok);
}
