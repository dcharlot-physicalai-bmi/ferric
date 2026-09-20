//! **Ferric's image preprocessing against the REAL `Qwen2VLImageProcessorFast`.**
//!
//! This is the seam the unit tests could not reach: `patchify` was checked against a derivation and
//! `resize_cubic` against Pillow, but the two had never been run together against the processor that
//! actually feeds the tower. It also settles the two questions left open when the resampler shipped —
//! which cubic coefficient the fast path applies, and whether it resizes in uint8 or float.
//!
//!   cargo run -p ferric-llama --example qwen3vl_preproc_ref --release -- <ckpt-dir> <img.ppm> <ref-prefix>
use ferric_llama::qwen3vl_image::{plan, preprocess, PreprocCfg, KEYS_PILLOW, KEYS_TORCH};
use ferric_tensor::image::read_ppm;

fn read_f32s(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    assert_eq!(b.len(), 4 + n * 4, "{path}: header says {n} floats");
    b[4..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: <ckpt-dir> <img.ppm> <ref-prefix>");
    let imgp = a.next().expect("usage: <ckpt-dir> <img.ppm> <ref-prefix>");
    let pfx = a.next().expect("usage: <ckpt-dir> <img.ppm> <ref-prefix>");

    let cfg = PreprocCfg::load(&dir).expect("preprocessor_config.json");
    let img = read_ppm(&std::fs::read(&imgp).expect("read image")).expect("P6 PPM");
    println!("image     : {}x{} (w,h)", img.w, img.h);
    println!("config    : patch {} merge {} tps {} pixels [{}, {}]",
             cfg.patch, cfg.merge, cfg.temporal_patch, cfg.min_pixels, cfg.max_pixels);

    let p = plan(img.h, img.w, &cfg).expect("plan");
    let want_grid: Vec<usize> = std::fs::read_to_string(format!("{pfx}.grid")).expect("grid")
        .trim().split(',').map(|v| v.parse().unwrap()).collect();
    println!("plan      : resize to {}x{}, grid {}x{}, {} patches, {} tokens",
             p.out_h, p.out_w, p.grid_h, p.grid_w, p.patches(), p.tokens(cfg.merge));
    // ⛔ The grid is the first thing to agree on: it is the token count, so a mismatch here is a
    // different prompt, not a different image. 80/32 = 2.5 exactly on this probe, so this line is
    // also the end-to-end check on banker's rounding.
    assert_eq!((p.grid_h, p.grid_w), (want_grid[1], want_grid[2]),
               "grid disagrees with the real processor — smart_resize is wrong");
    println!("            grid matches the real processor ✓");

    // optional dump, so the result can be diffed against Pillow's own 8-bit resampler off-line
    if let Ok(d) = std::env::var("FERRIC_PREPROC_DUMP") {
        let (rows, _) = preprocess(&img, &cfg, KEYS_PILLOW).expect("preprocess");
        let mut b: Vec<u8> = (rows.len() as u32).to_le_bytes().to_vec();
        for v in &rows { b.extend(v.to_le_bytes()); }
        std::fs::write(&d, &b).unwrap();
        println!("wrote {d} ({} floats, a = -0.5)", rows.len());
    }

    let want = read_f32s(&format!("{pfx}.pixel_values"));
    println!("\nvs Qwen2VLImageProcessorFast:");
    for (name, a) in [("a = -0.75 (torch/HF)", KEYS_TORCH), ("a = -0.50 (Pillow)", KEYS_PILLOW)] {
        let (got, _) = preprocess(&img, &cfg, a).expect("preprocess");
        assert_eq!(got.len(), want.len(), "{name}: {} vs {} values", got.len(), want.len());
        let (mut worst, mut n_diff) = (0f32, 0usize);
        for (g, w) in got.iter().zip(&want) {
            let d = (g - w).abs();
            worst = worst.max(d);
            if d > 1e-6 { n_diff += 1; }
        }
        // one 8-bit level is 2/255 after normalisation, so report the error in LEVELS: that is the
        // unit the disagreement actually lives in, and "0.0078" means "off by one", not "tiny".
        println!("  {name:<22} max |Δ| = {worst:.6}  = {:.2} levels   ({n_diff} of {} values differ, {:.2}%)",
                 worst * 255.0 / 2.0, want.len(), 100.0 * n_diff as f32 / want.len() as f32);
    }
}
