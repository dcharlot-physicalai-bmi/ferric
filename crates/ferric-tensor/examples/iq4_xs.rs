//! **IQ4_XS packed kernel == the CPU dequant reference.**
//!
//! Ferric had no native kernel for this format: the loader dequantised to f32 on load and ran a dense
//! matmul, correct but an 8x memory blow-up over the 4.25 bits/weight the format actually costs. The
//! quant_crossover measurement made that visible as the slowest format in the table despite being the
//! smallest, which is what a missing kernel looks like from the outside.
//!
//! The only thing that matters here is exactness. A 4-bit codebook kernel that is subtly wrong produces
//! plausible weights and fluent, wrong text, so it is checked against `ferric_gguf`'s CPU dequant, which
//! is the authority.
use ferric_core::Context;
use ferric_tensor::{Iq4XsWeights, Tensor};
use std::sync::Arc;

/// Build a synthetic IQ4_XS super-block stream: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128].
fn blocks(nblk: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
    let mut out = Vec::with_capacity(nblk * 136);
    for _ in 0..nblk {
        // f16 around 0.01: exponent/mantissa chosen to stay well inside range.
        out.extend_from_slice(&half_bits(0.005 + (rnd() % 40) as f32 * 0.0007).to_le_bytes());
        out.extend_from_slice(&rnd().to_le_bytes());              // scales_h, all 16 bits used
        for _ in 0..4 { out.push((rnd() & 0xFF) as u8); }         // scales_l
        for _ in 0..128 { out.push((rnd() & 0xFF) as u8); }       // qs
    }
    out
}

/// Minimal f32 -> f16 bits, enough for the values used above.
fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let man = ((b >> 13) & 0x3FF) as u16;
    if exp <= 0 { sign } else { sign | ((exp as u16) << 10) | man }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("IQ4_XS packed kernel vs the CPU dequant reference\n");
    println!("  {:>6} {:>7} {:>9} {:>14} {:>13}", "rows", "cols", "MB packed", "MB as dense f32", "max|Δ|");
    println!("  {:-<56}", "");

    let mut worst = 0f32;
    for &(rows, cols) in &[(8usize, 256usize), (32, 512), (7, 1024), (64, 256)] {
        let nblk = rows * (cols / 256);
        let raw = blocks(nblk, (rows * 31 + cols) as u32);

        // Reference: the CPU dequant this workspace already trusts, run through a plain dense matmul.
        let deq = ferric_gguf::deq_raw(&raw, rows * cols, 23).expect("IQ4_XS dequant");
        assert_eq!(deq.len(), rows * cols);

        let xv: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let x = Tensor::from_vec(&ctx, &xv, &[1, cols]);

        // Dense reference y[o] = sum_i x[i] * W[o,i]
        let want: Vec<f32> = (0..rows)
            .map(|o| (0..cols).map(|i| xv[i] * deq[o * cols + i]).sum::<f32>())
            .collect();

        let w = Iq4XsWeights::from_bytes(&ctx, &raw, rows, cols);
        let got = x.matmul_iq4_xs(&w).to_vec().await;

        assert_eq!(got.len(), rows, "shape mismatch");
        // Relative tolerance: the sums run over `cols` terms, so absolute error scales with magnitude.
        let scale = want.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
        let d = got.iter().zip(&want).fold(0f32, |a, (&g, &w)| a.max((g - w).abs())) / scale;
        worst = worst.max(d);
        let packed_mb = w.nbytes() as f64 / 1e6;
        println!("  {rows:>6} {cols:>7} {packed_mb:>9.4} {:>14.4} {d:>13.3e}",
                 (rows * cols * 4) as f64 / 1e6);
        assert!(d < 1e-5, "rows={rows} cols={cols}: packed kernel differs from the CPU reference by {d:.3e} \
                           relative. A 4-bit codebook kernel that is subtly wrong emits fluent, wrong text.");
    }

    println!("\n  ✅ Packed IQ4_XS matches the CPU dequant reference, worst relative Δ {worst:.3e}.");
    println!("     Weight memory is now 136 bytes per 256 values (4.25 bits/weight) instead of f32,");
    println!("     which is the 8x the dequant-on-load fallback was costing. Speed is NOT claimed here:");
    println!("     the machine is loaded and this example asserts correctness only. Time it with");
    println!("     ferric-llama/examples/quant_crossover.rs on a quiet box.");
}
