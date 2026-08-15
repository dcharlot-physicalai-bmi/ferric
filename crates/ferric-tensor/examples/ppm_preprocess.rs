//! Read a real P6 PPM and preprocess it to a vision encoder's input, end to end.
//!
//!   ffmpeg -i photo.jpg -pix_fmt rgb24 /tmp/x.ppm
//!   cargo run -p ferric-tensor --example ppm_preprocess --release -- /tmp/x.ppm 896
use ferric_tensor::image::{preprocess, read_ppm};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: ppm_preprocess <file.ppm> [size]");
    let size: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(896);
    let bytes = std::fs::read(path).expect("read");
    let img = match read_ppm(&bytes) { Ok(i) => i, Err(e) => { eprintln!("{e}"); std::process::exit(1); } };
    println!("decoded {}x{} ({:.1} MB of pixels)", img.w, img.h, img.px.len() as f64 / 1048576.0);

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    // Muse Glimmer's clip.vision.image_mean / image_std are 0.5 — read them from the mmproj in a real
    // pipeline rather than hardcoding; a model trained on ImageNet stats fed 0.5s is silently wrong.
    let t = preprocess(&ctx, &img, size, [0.5; 3], [0.5; 3]);
    let v = t.to_vec().await;
    let el = t0.elapsed();

    let (mut mn, mut mx, mut sum) = (f32::MAX, f32::MIN, 0f64);
    for &x in &v { mn = mn.min(x); mx = mx.max(x); sum += x as f64; }
    println!("-> [{size}, {size}, 3] = {} values in {el:.2?}", v.len());
    println!("   min {mn:.4}  max {mx:.4}  mean {:.4}", sum / v.len() as f64);
    assert_eq!(v.len(), size * size * 3, "shape must be [size, size, 3]");
    assert!(v.iter().all(|x| x.is_finite()), "preprocessing produced non-finite values");
    assert!(mn >= -1.001 && mx <= 1.001, "mean/std 0.5 must map into [-1, 1], got [{mn}, {mx}]");
    println!("   ✅ finite, in [-1,1], correct shape — ready for a patch embedding");
}
