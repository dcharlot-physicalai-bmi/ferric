//! **Does `layernorm` normalise?** A unit check, because a norm that under-scales is invisible
//! downstream — every activation is merely smaller, never wrong-looking.
//!
//! With weight=1 and bias=0, LayerNorm output must have per-row rms ≈ 1 for ANY input scale: that
//! is the whole point of the op. A parakeet encoder whose output rms was 10x below its final norm's
//! weight rms is what prompted this.
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (rows, d) = (4usize, 1024usize);
    let mut seed = 12345u64;
    let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 33) as f32 / 2147483648.0) - 0.5 };

    // ⚠ Scales chosen so var >> eps. At scale 0.01 the variance is 8.3e-6, COMPARABLE to eps=1e-5,
    // and (x-mu)/sqrt(var+eps) legitimately shrinks to ~0.68. That is correct LayerNorm behaviour,
    // not a bug — a first version of this test asserted against it and nearly reported one.
    for scale in [1.0f32, 10.0, 100.0] {
        let v: Vec<f32> = (0..rows * d).map(|_| rnd() * scale).collect();
        let x = Tensor::from_vec(&ctx, &v, &[rows, d]);
        let w = Tensor::from_vec(&ctx, &vec![1.0f32; d], &[d]);
        let b = Tensor::from_vec(&ctx, &vec![0.0f32; d], &[d]);
        let y = pollster::block_on(x.layernorm(&w, &b, 1e-5).to_vec());
        let r: Vec<f32> = (0..rows).map(|i| {
            let row = &y[i * d..(i + 1) * d];
            (row.iter().map(|z| z * z).sum::<f32>() / d as f32).sqrt()
        }).collect();
        let mean = r.iter().sum::<f32>() / rows as f32;
        println!("  input scale {scale:>6} → output rms {mean:.4}  (want ~1.0)");
        assert!((mean - 1.0).abs() < 0.05,
                "layernorm produced rms {mean:.4} at input scale {scale}: it is not normalising");
    }
    println!("\n  layernorm normalises at every input scale");
}
