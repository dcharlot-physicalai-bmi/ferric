//! **What does forcing one zero in four actually cost?**
//!
//! Tencent's Hy4 introduced STQ1_0: ternary weights where the container *guarantees* exactly one
//! zero in every group of four. That constraint is what buys the rate — a free ternary group needs
//! `log2(3^4) ≈ 6.34` bits, a 3:4 group needs 5 — so STQ1_0 lands at 1.3125 bpw against roughly
//! 1.585 for free ternary, a 17% saving.
//!
//! The saving is arithmetic and certain. The cost is not, and it is the whole question: a weight
//! matrix does not owe anyone a quarter of its entries being negligible. This measures both sides
//! on real expert slabs — the very tensors hyv4 puts in STQ1_0 (`ffn_gate_exps`, `ffn_up_exps`) —
//! against free ternary fitted by the same least-squares machinery, so the only difference between
//! the two arms is the constraint.
//!
//! ⚠ **The reference is itself quantized.** These slabs are Q4_K on disk, so every number here is
//! an error against a Q4_K reconstruction, not against the trained weights. That is fine for
//! comparing two encoders on the same reference and wrong for quoting either as an absolute
//! degradation, so no absolute claim is made.
//!
//! ```text
//! cargo run --release -p ferric-gguf --example stq1_0_experts -- [model.gguf] [n_tensors] [max_MElem]
//! ```

use ferric_gguf::quantize::{quantize_stq1_0, quantize_stq1_0_amax};
use ferric_gguf::{deq_raw, GgufFile};

/// Free ternary: `s ∈ {−1,0,+1}` with no structural constraint, fitted the same way STQ1_0 is —
/// a magnitude threshold picks the zeros, weighted least squares picks the scale, and the two
/// alternate. The threshold sweep is what STQ1_0 does not get to have.
fn free_ternary(xb: &[f32], w: &[f32]) -> (Vec<i8>, f32, usize) {
    let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    if !(amax > 0.0) { return (vec![0; xb.len()], 0.0, xb.len()) }
    let (mut best_sel, mut best_d, mut best_err) = (vec![0i8; xb.len()], 0.0f32, f32::INFINITY);
    for step in 0..=40 {
        let t = amax * step as f32 / 80.0;
        let sel: Vec<i8> = xb.iter().map(|&v| if v.abs() <= t { 0 } else if v < 0.0 { -1 } else { 1 }).collect();
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for j in 0..xb.len() {
            let q = sel[j] as f32;
            num += w[j] * q * xb[j];
            den += w[j] * q * q;
        }
        if !(den > 0.0) { continue }
        let d = num / den;
        if !(d > 0.0) { continue }
        let err: f32 = (0..xb.len()).map(|j| { let r = xb[j] - d * sel[j] as f32; w[j] * r * r }).sum();
        if err < best_err { best_err = err; best_d = d; best_sel = sel; }
    }
    let zeros = best_sel.iter().filter(|&&s| s == 0).count();
    (best_sel, best_d, zeros)
}

fn ls_weights(xb: &[f32]) -> Vec<f32> {
    let sumx2: f32 = xb.iter().map(|v| v * v).sum();
    let sigma2 = 2.0 * sumx2 / xb.len() as f32;
    xb.iter().map(|v| (sigma2 + v * v).sqrt()).collect()
}

fn main() -> Result<(), String> {
    let mut a = std::env::args().skip(1);
    let path = a.next().unwrap_or_else(|| {
        format!("{}/.lux-studio/models/apodex_Apodex-1.1-mini-Q4_K_M.gguf", std::env::var("HOME").unwrap())
    });
    let want: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    // An expert slab is hundreds of millions of weights and the free-ternary arm sweeps 40
    // thresholds over every one of them. Cap the sample and SAY what was dropped — an uncapped run
    // and a silently truncated one print the same table.
    let cap: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8) * 1_000_000;

    let g = GgufFile::open(&path)?;
    let names: Vec<String> = g.tensors.iter()
        .filter(|t| t.name.ends_with("ffn_gate_exps.weight") || t.name.ends_with("ffn_up_exps.weight"))
        .map(|t| t.name.clone()).take(want).collect();
    if names.is_empty() { return Err(format!("no expert slabs in {path} — this study needs an MoE model")) }

    println!("model: {path}");
    println!("{} expert slabs, the tensors hyv4 stores as STQ1_0\n", names.len());
    println!("{:<34} {:>9}  {:>8} {:>8} {:>8}  {:>7} {:>7}",
             "tensor", "elems", "amax", "stq1_0", "free-3", "0-frac", "cost");

    let (mut tot_stq, mut tot_free, mut tot_amax, mut tot_n, mut tot_zeros) = (0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
    let mut skipped = 0usize;

    for name in &names {
        let full = g.dequant(name)?;
        let n = full.len().min(cap) / 256 * 256;
        let x = &full[..n];
        if n < full.len() { skipped += full.len() - n }

        let mut b_ls = Vec::new();   quantize_stq1_0(x, None, &mut b_ls);
        let mut b_am = Vec::new();   quantize_stq1_0_amax(x, &mut b_am);
        let y_ls = deq_raw(&b_ls, n, 43)?;
        let y_am = deq_raw(&b_am, n, 43)?;

        let mut sse_free = 0.0f64;
        let mut zeros = 0usize;
        for xb in x.chunks_exact(256) {
            let w = ls_weights(xb);
            let (sel, d, z) = free_ternary(xb, &w);
            zeros += z;
            for j in 0..256 { let r = (xb[j] - d * sel[j] as f32) as f64; sse_free += r * r }
        }

        let sse = |y: &[f32]| -> f64 { x.iter().zip(y).map(|(a, b)| { let r = (a - b) as f64; r * r }).sum() };
        let (s_ls, s_am) = (sse(&y_ls), sse(&y_am));
        let rms = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64).sqrt();
        let rel = |s: f64| (s / n as f64).sqrt() / rms;

        println!("{:<34} {:>9} {:>8.3} {:>8.3} {:>8.3}  {:>6.1}% {:>6.2}x",
                 name, n, rel(s_am), rel(s_ls), rel(sse_free),
                 100.0 * zeros as f64 / n as f64, rel(s_ls) / rel(sse_free));

        tot_stq += s_ls; tot_free += sse_free; tot_amax += s_am; tot_n += n; tot_zeros += zeros;
    }

    let rel = |s: f64| (s / tot_n as f64).sqrt();
    let (e_stq, e_free, e_amax) = (rel(tot_stq), rel(tot_free), rel(tot_amax));
    let zfrac = tot_zeros as f64 / tot_n as f64;

    // Free ternary's rate is its actual symbol entropy at the zero fraction it chose, not log2(3).
    let p0 = zfrac.clamp(1e-9, 1.0 - 1e-9);
    let h_free = -(p0 * p0.log2() + (1.0 - p0) * ((1.0 - p0) / 2.0).log2());

    println!("\n─── over {} weights ({} sampled from each slab, {} not read) ───", tot_n, cap, skipped);
    println!("  reference encoder (d = amax)   RMSE {:.4}   {:.2}x the least-squares fit", e_amax, e_amax / e_stq);
    println!("  STQ1_0, forced 3:4             RMSE {:.4}   at 1.3125 bpw", e_stq);
    println!("  free ternary, same fit         RMSE {:.4}   at {:.4} bpw (its own entropy)", e_free, h_free);
    println!("  free ternary chose {:.1}% zeros; STQ1_0 forces 25.0%", 100.0 * zfrac);

    // Two framings, and they disagree, so both are printed. The entropy one asks whether the
    // constraint is information-theoretically justified; the fixed-rate one asks whether it is a
    // good engineering choice given that no GGUF ternary format ships an entropy coder.
    println!("\n  vs free ternary AT ITS OWN ENTROPY ({:.4} bpw, not a format that exists):", h_free);
    println!("      the 3:4 constraint costs {:+.1}% RMSE to save {:.1}% of the bits",
             100.0 * (e_stq / e_free - 1.0), 100.0 * (1.0 - 1.3125 / h_free));
    println!("  vs free ternary AS SHIPPED (llama.cpp TQ1_0 packs 5 trits per 8 bits = 1.6875 bpw):");
    println!("      the 3:4 constraint costs {:+.1}% RMSE to save {:.1}% of the bits",
             100.0 * (e_stq / e_free - 1.0), 100.0 * (1.0 - 1.3125 / 1.6875));
    println!("\n  ⚠ these are Apodex-1.1-mini experts, not Hy4's — the zero fraction is a property");
    println!("    of the trained weights, and it is the number the whole trade turns on.");
    Ok(())
}
