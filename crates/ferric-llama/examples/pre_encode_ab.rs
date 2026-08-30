//! **Does the GPU `pre_encode` compute what the CPU one did?** A differential check on a real model.
//!
//! `pre_encode` is a 5-stage subsampling stack whose output feeds every encoder layer. A wrong
//! flatten order, a wrong ReLU position, or a depthwise stage reading the wrong channel all produce
//! a finite tensor of the right shape — and a fluent wrong transcript. The CPU implementation is
//! kept precisely so the GPU one can be diffed against it rather than trusted.
//!
//! ⚠ This compares two implementations of the SAME spec, so it cannot catch an error present in
//! both (e.g. if the flatten order were wrong in the original). It catches PORTING errors, which is
//! what the port can introduce. End-to-end WER is what covers the rest.
use ferric_gguf::GgufFile;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let gguf = a.get(1).expect("usage: pre_encode_ab <model.gguf> [frames]");
    let frames: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(586);

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(gguf).expect("open gguf");
    let m = ferric_llama::parakeet::Parakeet::load(&ctx, &g).expect("load");
    let n_mels = m.cfg.num_mels;

    // Deterministic pseudo-mel with real dynamic range; a constant input would hide channel mixups.
    let mel: Vec<f32> = (0..frames * n_mels)
        .map(|i| ((i * 37 % 211) as f32 / 211.0 - 0.5) * 4.0).collect();

    let (cpu, t_c, w_c) = m.pre_encode_for_test(&mel, frames);
    let (gpu_t, t_g, w_g) = m.pre_encode_gpu_for_test(&mel, frames);
    let gpu = gpu_t.to_vec().await;

    println!("cpu -> [{t_c}, {w_c}]   gpu -> [{t_g}, {w_g}]");
    assert_eq!((t_c, w_c), (t_g, w_g), "shape mismatch");
    assert_eq!(cpu.len(), gpu.len(), "element count mismatch");

    let max_err = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let rms = (cpu.iter().map(|v| v * v).sum::<f32>() / cpu.len() as f32).sqrt();
    // Relative to signal scale: an absolute threshold means nothing without knowing the magnitude.
    println!("rms {rms:.4}  max|cpu-gpu| {max_err:.3e}  relative {:.2e}", max_err / rms.max(1e-9));

    // A permuted-order bug would leave max_err comparable to rms. Require it far below.
    assert!(max_err / rms.max(1e-9) < 1e-4, "GPU pre_encode diverges from the CPU reference");
    println!("\nGPU pre_encode matches the CPU implementation");
}
