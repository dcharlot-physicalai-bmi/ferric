//! **Qwen3-VL end to end from an image FILE** — preprocessing, tower, splice, mRoPE, deepstack, LM.
//!
//! Every other check in this family supplies synthesised pixels, on purpose, so that a mismatch has
//! one cause. This one closes the remaining seam: Ferric's own `preprocess` output feeding
//! `encode_patches`. The two agree with HuggingFace individually at the boundary they share; this
//! runs them together.
//!
//! ⭐ It also ATTRIBUTES the error. The same forward is run twice — once on Ferric's pixels and once
//! on the reference processor's — so the resampler's contribution is measured rather than argued
//! about. Ferric matches Pillow's 8-bit resampler exactly and sits 1 level from the HF one on 0.46%
//! of values; the question this answers is what that is worth 28 layers later.
//!
//!   cargo run -p ferric-llama --example qwen3vl_e2e --release -- <ckpt-dir>
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::qwen3vl_image::{preprocess, PreprocCfg, KEYS_DEFAULT};
use ferric_llama::qwen3vl_rope::rope_index;
use ferric_llama::qwen3vl_vision::VisionTower;
use ferric_load::hf::HfCheckpoint;
use ferric_tensor::image::read_ppm;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn read_f32s(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    assert_eq!(b.len(), 4 + n * 4, "{path}: header says {n} floats");
    b[4..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn dev(got: &[f32], want: &[f32]) -> (f32, f32) {
    let (mut worst, mut scale) = (0f32, 0f32);
    for (g, w) in got.iter().zip(want) { worst = worst.max((g - w).abs()); scale = scale.max(w.abs()); }
    (worst, worst / scale.max(1e-9))
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let dir = std::env::args().nth(1).expect("usage: qwen3vl_e2e <ckpt-dir>");
    let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3vl_e2e");
    let pfx = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3vl_preproc");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter   : {} [{:?}]", ctx.adapter_name, ctx.backend);

    let meta = std::fs::read_to_string(format!("{fx}/ref.meta")).expect("ref.meta");
    let field = |k: &str| -> String {
        meta.lines().find(|l| l.starts_with(&format!("{k} "))).unwrap_or_else(|| panic!("no {k}"))
            [k.len() + 1..].trim().to_string()
    };
    let ids: Vec<u32> = field("ids").split(',').map(|v| v.parse().unwrap()).collect();
    let types: Vec<u8> = field("types").bytes().map(|c| c - b'0').collect();
    let img_pos: Vec<usize> = field("image_positions").split(',').map(|v| v.parse().unwrap()).collect();
    let g: Vec<usize> = field("grid").split(',').map(|v| v.parse().unwrap()).collect();
    let (gt, gh, gw) = (g[0], g[1], g[2]);
    let img_start = img_pos[0];
    assert!(ids.contains(&field("vision_start_token_id").parse::<u32>().unwrap()),
            "no <vision_start> — mRoPE would never fire and this check would be vacuous");

    // ---- 1. preprocessing, from the image FILE ---------------------------------------------
    let cfg = PreprocCfg::load(&dir).expect("preprocessor_config.json");
    let img = read_ppm(&std::fs::read(format!("{pfx}/probe.ppm")).expect("probe.ppm")).expect("P6");
    let (rows, plan) = preprocess(&img, &cfg, KEYS_DEFAULT).expect("preprocess");
    println!("image     : {}x{} -> {}x{}, grid {}x{}, {} tokens",
             img.w, img.h, plan.out_h, plan.out_w, plan.grid_h, plan.grid_w, plan.tokens(cfg.merge));
    assert_eq!((plan.grid_h, plan.grid_w), (gh, gw), "grid disagrees with the reference processor");

    let hf_px = read_f32s(&format!("{pfx}/probe.pixel_values"));
    let (px_abs, px_rel) = dev(&rows, &hf_px);
    println!("pixels    : vs the real processor  max |Δ| = {px_abs:.6} = {:.2} 8-bit levels (rel {px_rel:.2e})",
             px_abs * 255.0 / 2.0);

    // ---- 2. the same forward twice, on Ferric's pixels and on the reference's ---------------
    let tower = VisionTower::load(&ctx, &dir).expect("vision tower");
    let hf = HfCheckpoint::open(&dir).expect("HF checkpoint");
    let model = Qwen3::load(&ctx, &hf).expect("Qwen3::load");
    let ri = rope_index(&types, &[(gt, gh, gw)], cfg.merge).expect("rope_index");
    let mut mrope: Vec<u32> = Vec::with_capacity(4 * ids.len());
    for v in [&ri.t, &ri.h, &ri.w] { mrope.extend(v.iter().map(|&x| x as u32)); }
    mrope.extend(std::iter::repeat_n(0u32, ids.len()));
    let want = read_f32s(&format!("{fx}/ref.final"));
    let row_w = 3 * cfg.temporal_patch * cfg.patch * cfg.patch;

    let mut out = Vec::new();
    for (name, px) in [("Ferric's own pixels", &rows), ("the reference's pixels", &hf_px)] {
        let pxt = Tensor::from_vec(&ctx, px, &[gh * gw, row_w]);
        let (pooled, deep, _) = tower.encode_patches(&pxt, gh, gw).expect("encode");
        let spliced = model.splice_image_embeds(&model.embed_tokens(&ids), img_start, &pooled);
        let mut cache = Cache::new(&model.cfg);
        let mut taps = Vec::new();
        let h = model.forward_embeds_mm(&spliced, &mut cache, &mrope, &deep, img_start, &mut taps);
        out.push((name, dev(&h.to_vec().await, &want)));
    }

    println!("\nfinal hidden state vs HuggingFace transformers 5.1.0:");
    for (name, (a, r)) in &out {
        println!("  from {name:<24} max |Δ| = {a:.4e}   rel {r:.2e}");
    }
    let (_, (_, rel_ferric)) = out[0];
    let (_, (_, rel_ref)) = out[1];
    println!("\n⭐ the resampler's whole contribution, 28 layers later: {:.2e} -> {:.2e} relative",
             rel_ref, rel_ferric);

    // ⛔ Two different bars. On identical pixels this is arithmetic and must be f32 drift. On
    // Ferric's own pixels it carries a real 1-level image difference through a 24-block tower and
    // 28 decoder layers, which is a DIFFERENT question and gets a looser, stated bound.
    assert!(rel_ref < 5e-3, "on identical pixels the path must be exact, got {rel_ref:.3e}");
    assert!(rel_ferric < 5e-2, "end-to-end deviation {rel_ferric:.3e} is larger than a 1-level \
                                image difference can explain");
    println!("\n✅ image file -> hidden state, end to end");
}
