//! **Does the signed `dmin` buy anything on a real checkpoint, or only on synthetic offset data?**
//!
//! Ferric's Q2_K encoder tries both floor polarities per super-block and keeps the better one, which
//! no conventional quantizer does. On synthetic data with a DC offset that took the cost of the
//! offset from x3.50 to x1.00 — but weight matrices are centred on zero, so the honest question is
//! whether a centred matrix contains enough ONE-SIDED sub-blocks for the choice to matter. A
//! sub-block is sixteen CONTIGUOUS weights; if those are usually mixed-sign, the second polarity is
//! dead code carrying a doubled encode cost.
//!
//! ## ⛔ THE ANSWER IS ZERO, AND IT IS THE POINT OF THE FILE
//!
//! Measured on Llama-3.2-1B, 8 largest eligible tensors, 379 M parameters:
//!
//! ```text
//!   weighted NRMSE  dmin>=0 0.2725   both polarities 0.2725   improvement 0.00%
//!   super-blocks choosing a NEGATIVE dmin: 0/1484800 (0.0%)
//! ```
//!
//! Not "small" — **zero out of 1.48 million**. The reasoning that predicted otherwise was about
//! SUB-blocks, and the polarity is chosen per SUPER-block: 256 weights share one `dmin`, so a single
//! negative weight anywhere among them forces its sub-block's floor to zero under σ=−1 and makes
//! that encoding strictly worse. On a matrix centred on zero the negative branch is unreachable.
//!
//! So the capability is correct, it is worth **x3.50 → x1.00 on data with a DC offset**, and it is
//! worth **nothing at all on a weight matrix**. `quantize_q2_k` now scans for a negative weight
//! before paying for the second encode, which on real checkpoints means never paying for it. Where
//! it should still matter is anything NOT centred on zero — post-softmax attention values in a
//! quantized KV cache being the obvious candidate, and untested here.
//!
//! The measurement, per tensor: error with the conventional non-negative `dmin`, error when both
//! polarities are priced, and how many super-blocks actually chose the negative one.
//!
//!   cargo run -p ferric-gguf --example quant_real_weights --release -- <model.gguf>
use ferric_gguf::quantize::{encode_q2_k_super, quantize_q3_k};
use ferric_gguf::{deq_raw, GgufFile, GgufSource};

const Q2_K: u32 = 10;
const Q3_K: u32 = 11;

fn nrmse(x: &[f32], y: &[f32]) -> f32 {
    let se: f32 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum();
    let sx: f32 = x.iter().map(|a| a * a).sum();
    if sx <= 0.0 { 0.0 } else { (se / sx).sqrt() }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: quant_real_weights <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");

    // Only 2-D weights whose row length is a multiple of 256 — the K-quants' own requirement, and
    // the reason a file named "Q2_K" is really a mixture: rows that do not divide fall to another
    // type entirely. Reporting an average over tensors the format cannot even hold would be
    // measuring the fallback, not the format.
    let mut picks: Vec<(String, Vec<u64>)> = g.tensors.iter()
        .filter(|t| t.dims.len() == 2 && t.dims[0] % 256 == 0)
        .map(|t| (t.name.clone(), t.dims.clone()))
        .collect();
    picks.sort_by_key(|(_, d)| std::cmp::Reverse(d.iter().product::<u64>()));
    picks.truncate(8);
    assert!(!picks.is_empty(), "no 2-D tensor in {path} has a row length divisible by 256");

    println!("Q2_K floor polarity on real weights — {path}\n");
    println!("{:<34} {:>10} {:>10} {:>10} {:>9} {:>9}",
             "tensor", "params", "dmin>=0", "both", "neg%", "Q3_K");

    let (mut tot_a, mut tot_b, mut tot_n, mut tot_neg, mut tot_sb) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize);
    for (name, dims) in &picks {
        let w = g.dequant(name).expect("dequant");
        let n = w.len() - w.len() % 256;
        let w = &w[..n];

        // Conventional: floor pinned at or below zero, exactly one polarity considered.
        let mut conv = Vec::with_capacity(n / 256 * 84);
        // Ferric: both priced, better kept — and count how often the unconventional one wins.
        let mut both = Vec::with_capacity(n / 256 * 84);
        let mut neg = 0usize;
        for sb in w.chunks_exact(256) {
            let (ba, sa) = encode_q2_k_super(sb, 3, 1.0);
            let (bb, sbb) = encode_q2_k_super(sb, 3, -1.0);
            conv.extend_from_slice(&ba);
            if sbb < sa { neg += 1; both.extend_from_slice(&bb) } else { both.extend_from_slice(&ba) }
        }
        let ea = nrmse(w, &deq_raw(&conv, n, Q2_K).unwrap());
        let eb = nrmse(w, &deq_raw(&both, n, Q2_K).unwrap());
        let mut q3 = Vec::new();
        quantize_q3_k(w, &mut q3);
        let e3 = nrmse(w, &deq_raw(&q3, n, Q3_K).unwrap());

        let sbs = n / 256;
        println!("{:<34} {:>10} {:>10.4} {:>10.4} {:>8.1}% {:>9.4}",
                 if name.len() > 34 { &name[name.len() - 34..] } else { name },
                 dims.iter().product::<u64>(), ea, eb, 100.0 * neg as f32 / sbs as f32, e3);
        tot_a += ea as f64 * n as f64; tot_b += eb as f64 * n as f64;
        tot_n += n; tot_neg += neg; tot_sb += sbs;
    }

    let (wa, wb) = (tot_a / tot_n as f64, tot_b / tot_n as f64);
    println!("\n  weighted NRMSE  dmin>=0 {wa:.4}   both polarities {wb:.4}   \
              improvement {:.2}%", 100.0 * (wa - wb) / wa);
    println!("  super-blocks choosing a NEGATIVE dmin: {tot_neg}/{tot_sb} ({:.1}%)",
             100.0 * tot_neg as f32 / tot_sb as f32);
    println!("\n  A negative dmin is legal Q2_K — the file is ordinary and any reader that multiplies\n  \
              by a signed f16, which every reader must, decodes it unchanged. It costs nothing at\n  \
              inference. It is also, on centred weights, never chosen: see this file's header.");
}
