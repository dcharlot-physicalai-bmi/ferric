//! Fused FFN gate/up + SwiGLU (Q4_K / Q5_K / Q6_K), the whole-block megafusion step: one kernel
//! computes the gate and up projections and applies SwiGLU, never materializing the [t, 2·n_ff]
//! intermediate. Validate each against the un-fused `matmul_qk(w).swiglu(n_ff)`, then microbench
//! both at a decode (t=1) and a prefill (t=8) shape — measured, not assumed.
use ferric_core::Context;
use ferric_tensor::{Q4_KWeights, Q5_KWeights, Q6_KWeights, Tensor};
use std::sync::Arc;
use half::f16;

fn q4k_block(seed: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    let d = 0.05 + 0.01 * ((seed % 7) as f32);
    let dmin = 0.02 + 0.005 * ((seed % 5) as f32);
    b.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    b.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
    let sc = |j: u32| ((seed.wrapping_mul(2654435761).wrapping_add(j * 40503)) % 64) as u8;
    let mn = |j: u32| ((seed.wrapping_mul(40503).wrapping_add(j * 2654435761)) % 64) as u8;
    let mut s = [0u8; 12];
    for j in 0..8u32 {
        if j < 4 { s[j as usize] |= sc(j) & 63; s[(j + 4) as usize] |= mn(j) & 63; }
        else {
            let (scv, mnv) = (sc(j), mn(j));
            s[(j + 4) as usize] |= (scv & 0x0F) | ((mnv & 0x0F) << 4);
            s[(j - 4) as usize] |= (scv >> 4) << 6;
            s[j as usize] |= (mnv >> 4) << 6;
        }
    }
    b.extend_from_slice(&s);
    for i in 0..128u32 { b.push((((seed.wrapping_add(i * 2246822519)) % 256) as u8) & 0xff); }
    b
}
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
    for i in 0..32u32 { b[16 + i as usize] = ((seed.wrapping_add(i * 2246822519)) % 256) as u8; }
    for i in 0..128u32 { b[48 + i as usize] = ((seed.wrapping_mul(97).wrapping_add(i * 40503)) % 256) as u8; }
    b
}
fn q6k_block(seed: u32) -> Vec<u8> {
    let mut b = vec![0u8; 210];
    for i in 0..128u32 { b[i as usize] = ((seed.wrapping_add(i * 2246822519)) % 256) as u8; }
    for i in 0..64u32 { b[128 + i as usize] = ((seed.wrapping_mul(40503).wrapping_add(i * 97)) % 256) as u8; }
    for i in 0..16u32 { b[192 + i as usize] = (((seed.wrapping_add(i * 7)) % 64) as i32 - 32) as i8 as u8; }
    b[208..210].copy_from_slice(&f16::from_f32(0.04 + 0.01 * ((seed % 6) as f32)).to_le_bytes());
    b
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (n_ff, n_embd) = (3072usize, 1024usize); // Qwen3-0.6B FFN dims
    let rows_w = 2 * n_ff; // gate rows 0..n_ff, up rows n_ff..2n_ff
    let nblk = n_embd / 256;
    let mut ok = true;

    for fmt in ["Q4_K", "Q5_K", "Q6_K"] {
        // build a gate_up weight and closures: fused kernel + un-fused reference, for this format
        let (fused, refv): (
            Box<dyn Fn(&Tensor) -> Tensor>,
            Box<dyn Fn(&Tensor) -> Tensor>,
        ) = match fmt {
            "Q4_K" => {
                let mut p = Vec::new();
                for r in 0..rows_w { for blk in 0..nblk { p.extend(q4k_block((r * nblk + blk) as u32 + 1)); } }
                let w = Q4_KWeights::from_bytes(&ctx, &p, rows_w, n_embd);
                let w2 = Q4_KWeights::from_bytes(&ctx, &p, rows_w, n_embd);
                (Box::new(move |h: &Tensor| h.matmul_q4_k_swiglu(&w)),
                 Box::new(move |h: &Tensor| h.matmul_q4_k(&w2).swiglu(n_ff)))
            }
            "Q5_K" => {
                let mut p = Vec::new();
                for r in 0..rows_w { for blk in 0..nblk { p.extend(q5k_block((r * nblk + blk) as u32 + 1)); } }
                let w = Q5_KWeights::from_bytes(&ctx, &p, rows_w, n_embd);
                let w2 = Q5_KWeights::from_bytes(&ctx, &p, rows_w, n_embd);
                (Box::new(move |h: &Tensor| h.matmul_q5_k_swiglu(&w)),
                 Box::new(move |h: &Tensor| h.matmul_q5_k(&w2).swiglu(n_ff)))
            }
            _ => {
                let mut p = Vec::new();
                for r in 0..rows_w { for blk in 0..nblk { p.extend(q6k_block((r * nblk + blk) as u32 + 1)); } }
                let w = Q6_KWeights::from_bytes(&ctx, &p, rows_w, n_embd);
                let w2 = Q6_KWeights::from_bytes(&ctx, &p, rows_w, n_embd);
                (Box::new(move |h: &Tensor| h.matmul_q6_k_swiglu(&w)),
                 Box::new(move |h: &Tensor| h.matmul_q6_k(&w2).swiglu(n_ff)))
            }
        };

        for t in [1usize, 8] {
            let h = Tensor::from_vec(&ctx, &(0..t * n_embd).map(|i| (i as f32 * 0.017).cos() * 0.3).collect::<Vec<_>>(), &[t, n_embd]);
            let f = fused(&h).to_vec().await;
            let rv = refv(&h).to_vec().await;
            let e = f.iter().zip(&rv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = rv.iter().map(|v| v.abs()).fold(1e-3, f32::max);
            let p = e / scale < 1e-5; ok &= p;
            println!("{} {fmt} t={t}: fused gate/up+SwiGLU vs matmul+swiglu  max|Δ|/scale = {:.1e}", if p { "✅" } else { "❌" }, e / scale);
        }

        // microbench: fused (1 kernel) vs un-fused (matmul + swiglu = 2 kernels)
        let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t0 = std::time::Instant::now();
            for _ in 0..50 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t0.elapsed().as_secs_f64() * 1e3 / 50.0 };
        for t in [1usize, 8] {
            let h = Tensor::from_vec(&ctx, &(0..t * n_embd).map(|i| (i as f32 * 0.017).cos() * 0.3).collect::<Vec<_>>(), &[t, n_embd]);
            let _ = pollster::block_on(fused(&h).to_vec()); // warm
            let fu = bench(&|| fused(&h));
            let un = bench(&|| refv(&h));
            println!("  {fmt} t={t}: fused {fu:.3} ms  vs  matmul+swiglu {un:.3} ms   ({:+.0}%)", (un / fu - 1.0) * 100.0);
        }
    }
    println!("{}", if ok { "✅ fused FFN gate/up+SwiGLU is exact for Q4_K, Q5_K, Q6_K" } else { "❌ mismatch" });
    assert!(ok);
}
