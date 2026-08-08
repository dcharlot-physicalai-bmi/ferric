//! **The IQ4 packed kernels against the CPU dequant reference, on real checkpoint weights.**
//!
//! `ferric-tensor/examples/iq4_xs.rs` already checks the IQ4_XS kernel on synthetic blocks. Two things
//! that check cannot see:
//!
//! 1. **Whether the model loader ever reaches the kernel.** It did not. `Iq4XsWeights` and
//!    `matmul_iq4_xs` existed and were correct, but `QMatrix::block_bytes` did not list ggml type 23,
//!    so `qm()` took the `from_dense` branch and every IQ4 weight was dequantised to f32 on load. A
//!    written, tested, unreachable kernel produces no error anywhere — the model just runs, slower and
//!    8x fatter, exactly as if the kernel had never been written.
//!
//! 2. **What is actually in a file named IQ4_XS.** Measured on
//!    `bartowski/Qwen2.5-0.5B-Instruct-IQ4_XS.gguf`: **IQ4_XS is 24 tensors / 104.6 M params**, and
//!    **IQ4_NL is 120 tensors / 250.5 M params**. IQ4_XS needs rows divisible by 256; this model's
//!    `n_embd` is 896, so only `ffn_down` (in-dim 4864) qualifies and everything else falls to the
//!    32-block IQ4_NL. Wiring IQ4_XS alone would have left the majority of the model on the f32
//!    fallback while the table said the format was supported.
//!
//! Synthetic blocks would not have shown either. Both are properties of the real checkpoint.
//!
//! Exactness is the only thing asserted here. A 4-bit codebook kernel that is subtly wrong yields
//! plausible weights and fluent, wrong text, which no perplexity-free smoke test catches.
//!
//!   cargo run -p ferric-llama --example iq4_real_weights --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource};
use ferric_tensor::{Iq4NlWeights, Iq4XsWeights, Tensor};
use std::sync::Arc;

const MODEL: &str = "bartowski_Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-IQ4_XS.gguf";

const IQ4_NL: u32 = 20;
const IQ4_XS: u32 = 23;

fn main() { pollster::block_on(run()); }

async fn run() {
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/{MODEL}");
    let Ok(g) = GgufFile::open(&path) else {
        println!("model not present at {path} — skipping");
        return;
    };
    let ctx = Arc::new(Context::new().await.unwrap());

    // ---- what the file is actually made of ----
    let mut counts: std::collections::BTreeMap<u32, (usize, u64)> = Default::default();
    for t in &g.tensors {
        let e = counts.entry(t.ggml_type).or_default();
        e.0 += 1;
        e.1 += t.dims.iter().product::<u64>();
    }
    println!("IQ4 packed kernels vs the CPU dequant reference — real weights\n");
    println!("  Composition of a file distributed as \"IQ4_XS\":");
    for (ty, (n, params)) in &counts {
        let name = match *ty { 0 => "F32", 1 => "F16", 7 => "Q5_1", 8 => "Q8_0", 20 => "IQ4_NL", 23 => "IQ4_XS", _ => "other" };
        let native = ferric_tensor::QMatrix::block_bytes(*ty).is_some();
        println!("    type {ty:>3} {name:>7}: {n:>4} tensors, {:>7.1} M params   {}",
                 *params as f64 / 1e6, if native { "packed kernel" } else { "f32 dense fallback" });
    }

    // ---- every IQ4 tensor's packed kernel must equal the CPU dequant ----
    println!("\n  {:>24} {:>7} {:>6} {:>6} {:>10} {:>12} {:>11}",
             "tensor", "format", "in", "out", "MB packed", "MB as f32", "rel max|Δ|");
    println!("  {:-<82}", "");

    let mut worst = 0f64;
    let mut checked = 0usize;
    let (mut packed_bytes, mut dense_bytes) = (0u64, 0u64);

    for ty in [IQ4_NL, IQ4_XS] {
        // Two tensors per format: the first of its kind and one from a later layer, so a bug that only
        // shows up at a particular offset into the file has a chance to appear.
        let names: Vec<String> = g.tensors.iter().filter(|t| t.ggml_type == ty)
            .map(|t| t.name.clone()).take(2).collect();
        for name in &names {
            let t = g.tensor(name).unwrap();
            let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
            let raw = g.raw(name).expect("raw bytes");

            // The authority: ferric-gguf's CPU dequant, then a plain dot product.
            let deq = ferric_gguf::deq_raw(&raw, rows * cols, ty).expect("dequant");
            assert_eq!(deq.len(), rows * cols);

            let xv: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let want: Vec<f64> = (0..rows)
                .map(|o| (0..cols).map(|i| xv[i] as f64 * deq[o * cols + i] as f64).sum())
                .collect();

            let x = Tensor::from_vec(&ctx, &xv, &[1, cols]);
            let (got, nbytes) = if ty == IQ4_XS {
                let w = Iq4XsWeights::from_bytes(&ctx, &raw, rows, cols);
                let n = w.nbytes();
                (x.matmul_iq4_xs(&w).to_vec().await, n)
            } else {
                let w = Iq4NlWeights::from_bytes(&ctx, &raw, rows, cols);
                let n = w.nbytes();
                (x.matmul_iq4_nl(&w).to_vec().await, n)
            };
            assert_eq!(got.len(), rows, "{name}: shape mismatch");

            // Relative to the largest output, because the sums run over `cols` terms and absolute
            // error scales with magnitude.
            let scale = want.iter().fold(1e-9f64, |a, &v| a.max(v.abs()));
            let d = got.iter().zip(&want).fold(0f64, |a, (&gv, &wv)| a.max((gv as f64 - wv).abs())) / scale;
            worst = worst.max(d);
            checked += 1;
            packed_bytes += nbytes as u64;
            dense_bytes += (rows * cols * 4) as u64;

            let short = name.strip_suffix(".weight").unwrap_or(name);
            let fmt = if ty == IQ4_XS { "IQ4_XS" } else { "IQ4_NL" };
            println!("  {short:>24} {fmt:>7} {cols:>6} {rows:>6} {:>10.3} {:>12.3} {d:>11.3e}",
                     nbytes as f64 / 1e6, (rows * cols * 4) as f64 / 1e6);
            assert!(d < 1e-5, "{name}: packed kernel differs from the CPU dequant by {d:.3e} relative. \
                               A 4-bit codebook kernel that is subtly wrong emits fluent, WRONG text.");
        }
    }

    assert!(checked > 0, "no IQ4 tensors were checked — this asserted nothing");
    println!("\n  ✅ {checked} real tensors, both formats, worst relative Δ {worst:.3e}.");
    println!("     On these tensors alone: {:.1} MB packed vs {:.1} MB as the f32 fallback ({:.1}x).",
             packed_bytes as f64 / 1e6, dense_bytes as f64 / 1e6,
             dense_bytes as f64 / packed_bytes as f64);

    // ---- what the wiring changes for the whole model ----
    let iq4: u64 = counts.get(&IQ4_NL).map(|c| c.1).unwrap_or(0) + counts.get(&IQ4_XS).map(|c| c.1).unwrap_or(0);
    let total: u64 = counts.values().map(|c| c.1).sum();
    println!("     Model-wide, {:.1} M of {:.1} M params ({:.0}%) move off the f32 dense fallback.",
             iq4 as f64 / 1e6, total as f64 / 1e6, 100.0 * iq4 as f64 / total as f64);
    println!("\n  ⚠ Correctness only. No speed or energy claim is made here — see quant_crossover.rs,");
    println!("    whose IQ4_XS column was measuring the missing kernel and must be re-run now that the");
    println!("    loader reaches a real one.");
}
