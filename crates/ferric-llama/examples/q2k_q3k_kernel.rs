//! **The Q2_K / Q3_K packed kernels against the CPU dequant reference, on real weights.**
//!
//! A block-quant kernel that is subtly wrong does not fail. It reconstructs finite, plausibly-scaled
//! weights and the model emits fluent, wrong text — so the only useful check is exact agreement with
//! an independent dequantizer, and the only useful weights are real ones.
//!
//! Two probes, and the first is the sharper of the two:
//!
//!   1. **One-hot x.** With `x = e_k`, `y = x·Wᵀ` reduces to column `k` of `W` — every other term is
//!      a hard zero, so accumulation order cannot contribute anything. The output IS the kernel's
//!      dequantization of one element per row, comparable element-for-element against `deq_raw`.
//!      That separates "does the kernel decode the layout" from "does it sum correctly", which a
//!      random-vector test conflates: an error in the interleave and an error in the reduction both
//!      show up as one wrong dot product.
//!
//!      The probe positions are chosen to straddle every boundary the layouts have — 0/1 (first
//!      sub-block), 15/16/17 (sub-block edge), 31/32 (group edge), 127/128/129 (the HALF edge, where
//!      Q3_K's high-bit selector continues rather than restarting), 255 (last).
//!
//!   2. **Random x**, against an f64 reference. Now accumulation is in scope, and the bar is loose
//!      enough to allow reordering and tight enough to catch a wrong weight.
//!
//! Weights come from a real checkpoint, quantized by Ferric's own quantizer — no external binary is
//! involved anywhere in this file.
//!
//! ## The one-hot probe reads 0.00e0, so it was mutation-tested
//!
//! An assertion that passes at EXACTLY zero is the one most worth doubting: bit-exact agreement and
//! a test that cannot fail look identical from the outside. Two mutations, each the trap its format
//! actually has, both caught by the one-hot probe alone:
//!
//! ```text
//!   Q3_K high-bit sense inverted (set means subtract)      1.688e11 relative
//!   Q2_K qs read sequentially instead of (index, shift)    7.919e10 relative
//! ```
//!
//! Those magnitudes are the point. A wrong block layout does not drift by a few percent — it
//! reconstructs a completely different number, which is exactly why nothing downstream can catch it
//! by looking reasonable.
//!
//!   cargo run -p ferric-llama --example q2k_q3k_kernel --release -- <model.gguf>
use ferric_core::Context;
use ferric_gguf::quantize::{quantize_q2_k, quantize_q3_k};
use ferric_gguf::{deq_raw, GgufFile, GgufSource};
use ferric_tensor::{QMatrix, Tensor};
use std::sync::Arc;

const Q2_K: u32 = 10;
const Q3_K: u32 = 11;
const PROBES: [usize; 11] = [0, 1, 15, 16, 17, 31, 32, 127, 128, 129, 255];

fn main() { pollster::block_on(run()); }

async fn run() {
    let path = std::env::args().nth(1).expect("usage: q2k_q3k_kernel <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");
    let ctx = Arc::new(Context::new().await.expect("gpu"));

    // ⚠ THE WIRING IS THE FIRST THING TO CHECK, not the arithmetic. A written, correct, UNREACHABLE
    // kernel produces no error anywhere — the model just runs on the f32 fallback, slower and much
    // fatter, exactly as if the kernel had never been written. That is how IQ4_XS shipped once.
    for (ty, name) in [(Q2_K, "Q2_K"), (Q3_K, "Q3_K")] {
        assert!(QMatrix::block_bytes(ty).is_some(),
                "{name} (type {ty}) is not in QMatrix::block_bytes, so every {name} weight would \
                 load through the f32 dense fallback and nothing below would be testing the kernel");
    }

    let mut picks: Vec<(String, usize, usize)> = g.tensors.iter()
        .filter(|t| t.dims.len() == 2 && t.dims[0] % 256 == 0)
        .map(|t| (t.name.clone(), t.dims[0] as usize, t.dims[1] as usize))
        .collect();
    picks.sort_by_key(|(_, c, r)| std::cmp::Reverse(c * r));
    // Largest, plus one from deeper in the file, so a bug that needs a particular offset can appear.
    let picks: Vec<_> = picks.into_iter().take(4).collect();
    assert!(!picks.is_empty(), "no 2-D tensor in {path} has a row length divisible by 256");

    println!("Q2_K / Q3_K packed kernels vs CPU dequant — {path}\n");
    println!("  {:>26} {:>6} {:>6} {:>6} {:>13} {:>13} {:>10} {:>10}",
             "tensor", "fmt", "in", "out", "MB packed", "MB as f32", "onehot Δ", "random Δ");
    println!("  {:-<98}", "");

    let (mut worst_hot, mut worst_rnd, mut checked) = (0f64, 0f64, 0usize);
    let (mut packed, mut dense) = (0u64, 0u64);

    for (name, cols, rows) in &picks {
        let (cols, rows) = (*cols, *rows);
        let w32 = g.dequant(name).expect("dequant source");
        assert_eq!(w32.len(), rows * cols);

        for (ty, tag) in [(Q2_K, "Q2_K"), (Q3_K, "Q3_K")] {
            // Ferric quantizes it, Ferric unpacks it on the CPU, Ferric unpacks it on the GPU.
            let mut raw = Vec::new();
            if ty == Q2_K { quantize_q2_k(&w32, &mut raw) } else { quantize_q3_k(&w32, &mut raw) }
            let deq = deq_raw(&raw, rows * cols, ty).expect("cpu dequant");
            let qm = QMatrix::from_bytes(&ctx, &raw, ty, rows, cols).expect("pack for gpu");

            // ---- probe 1: one-hot, batched into a single dispatch ----
            let mut xh = vec![0f32; PROBES.len() * cols];
            for (r, &k) in PROBES.iter().enumerate() { xh[r * cols + k] = 1.0; }
            let got = Tensor::from_vec(&ctx, &xh, &[PROBES.len(), cols]).matmul_q(&qm).to_vec().await;
            assert_eq!(got.len(), PROBES.len() * rows, "{name}/{tag}: one-hot output shape");
            let mut dhot = 0f64;
            for (r, &k) in PROBES.iter().enumerate() {
                for o in 0..rows {
                    // Sum of `cols−1` exact zeros and one weight: this is the weight, not a sum.
                    let want = deq[o * cols + k] as f64;
                    let scale = want.abs().max(1e-12);
                    dhot = dhot.max((got[r * rows + o] as f64 - want).abs() / scale);
                }
            }
            assert!(dhot < 1e-5,
                    "{name}/{tag}: the kernel's dequantization disagrees with deq_raw by {dhot:.3e} \
                     relative on a ONE-HOT probe, where no summation happens at all — this is the \
                     block layout being read differently on GPU and CPU, not float noise");

            // ---- probe 2: random x against an f64 reference ----
            let xv: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let want: Vec<f64> = (0..rows)
                .map(|o| (0..cols).map(|i| xv[i] as f64 * deq[o * cols + i] as f64).sum())
                .collect();
            let got = Tensor::from_vec(&ctx, &xv, &[1, cols]).matmul_q(&qm).to_vec().await;
            let scale = want.iter().fold(1e-9f64, |a, &v| a.max(v.abs()));
            let drnd = got.iter().zip(&want).fold(0f64, |a, (&gv, &wv)| a.max((gv as f64 - wv).abs())) / scale;
            assert!(drnd < 1e-5, "{name}/{tag}: packed matmul differs from the CPU reference by \
                                  {drnd:.3e} relative");

            worst_hot = worst_hot.max(dhot);
            worst_rnd = worst_rnd.max(drnd);
            checked += 1;
            packed += qm.nbytes() as u64;
            dense += (rows * cols * 4) as u64;

            let short = name.strip_suffix(".weight").unwrap_or(name);
            let short = if short.len() > 26 { &short[short.len() - 26..] } else { short };
            println!("  {short:>26} {tag:>6} {cols:>6} {rows:>6} {:>13.3} {:>13.3} {dhot:>10.2e} {drnd:>10.2e}",
                     qm.nbytes() as f64 / 1e6, (rows * cols * 4) as f64 / 1e6);
        }
    }

    assert!(checked > 0, "nothing was checked — this example asserted nothing");
    println!("\n  ✅ {checked} tensor/format pairs. Worst one-hot Δ {worst_hot:.2e}, worst random Δ {worst_rnd:.2e}.");
    println!("     {:.1} MB packed vs {:.1} MB on the f32 fallback ({ratio:.1}x) — which is what wiring\n     \
              block_bytes() buys: without it these load CORRECT and {ratio:.1}x fatter, with no error\n     \
              anywhere, which is how IQ4_XS shipped unreachable once already.",
             packed as f64 / 1e6, dense as f64 / 1e6, ratio = dense as f64 / packed as f64);
}
