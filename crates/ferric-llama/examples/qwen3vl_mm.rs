//! **Qwen3-VL end to end: image rows into the language model**, against the published implementation.
//!
//! The tower is already checked on its own (`qwen3vl_vision_ref`). This checks the WIRING: that the
//! merger's rows land on the image placeholder tokens, that mRoPE positions come from
//! `qwen3vl_rope::rope_index` rather than the sequence index, and that the three deepstack features
//! are added after LM layers 0, 1, 2 — not the vision blocks 5, 11, 17 they came from.
//!
//! Seams, in pipeline order, so a mismatch localises:
//!   embed    layer-0 input — the splice alone, before any layer runs
//!   layer0/1/2  after each deepstack injection
//!   final    last_hidden_state, post-norm
//!
//!   cargo run -p ferric-llama --example qwen3vl_mm --release -- <ckpt-dir>
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::qwen3vl_rope::rope_index;
use ferric_llama::qwen3vl_vision::VisionTower;
use ferric_load::hf::HfCheckpoint;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn read_f32s(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    assert_eq!(b.len(), 4 + n * 4, "{path}: header says {n} floats");
    b[4..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn compare(label: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}: {} vs {} values", got.len(), want.len());
    let (mut worst, mut scale) = (0f32, 0f32);
    for (g, w) in got.iter().zip(want) { worst = worst.max((g - w).abs()); scale = scale.max(w.abs()); }
    let rel = worst / scale.max(1e-9);
    println!("  {label:<8} max |Δ| = {worst:.4e} on |x| up to {scale:.4e}   rel {rel:.2e}");
    rel
}

/// ⭐ PER-ROW, because "which tokens are wrong" answers a different question from "how wrong".
/// Image rows only -> the splice or the deepstack add. Text rows too -> the POSITIONS, since an
/// image shifts every token after it. Rows before the image only -> neither, look earlier.
fn per_row(label: &str, got: &[f32], want: &[f32], d: usize, img: &[usize]) {
    let t = want.len() / d;
    print!("  {label:<8} per row:");
    for r in 0..t {
        let mut m = 0f32;
        for k in 0..d { m = m.max((got[r * d + k] - want[r * d + k]).abs()); }
        let tag = if img.contains(&r) { "*" } else { " " };
        print!(" {r}{tag}{m:.1e}");
    }
    println!("   (* = image row)");
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let dir = std::env::args().nth(1).expect("usage: qwen3vl_mm <ckpt-dir>");
    let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3vl_lm");
    let vfx = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3vl_vision");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter   : {} [{:?}]", ctx.adapter_name, ctx.backend);

    // ---- meta: the exact sequence the reference ran on -------------------------------------
    let meta = std::fs::read_to_string(format!("{fx}/ref.meta")).expect("ref.meta");
    let field = |k: &str| -> String {
        meta.lines().find(|l| l.starts_with(&format!("{k} "))).unwrap_or_else(|| panic!("no {k}"))
            [k.len() + 1..].trim().to_string()
    };
    let ids: Vec<u32> = field("ids").split(',').map(|v| v.parse().unwrap()).collect();
    let img_pos: Vec<usize> = field("image_positions").split(',').map(|v| v.parse().unwrap()).collect();
    let grid: Vec<usize> = field("grid").split(',').map(|v| v.parse().unwrap()).collect();
    let (gt, gh, gw) = (grid[0], grid[1], grid[2]);
    let img_start = img_pos[0];
    // ⚠ contiguity is what makes narrow+cat legal instead of a scatter; assert it rather than assume
    assert!(img_pos.windows(2).all(|w| w[1] == w[0] + 1), "image positions are not one run: {img_pos:?}");
    println!("sequence  : {} tokens, image at {}..{} , grid {gt}x{gh}x{gw}",
             ids.len(), img_start, img_start + img_pos.len());

    // ---- tower ------------------------------------------------------------------------------
    let tower = VisionTower::load(&ctx, &dir).expect("vision tower");
    let c = &tower.cfg;
    let row = 3 * c.temporal_patch * c.patch * c.patch;
    // ⛔ the SAME pixels the tower fixtures use — byte-identical, so this file is not duplicated
    let px = read_f32s(&format!("{vfx}/ref.px"));
    let pxt = Tensor::from_vec(&ctx, &px, &[gh * gw, row]);
    let (pooled, deep, _hidden) = tower.encode_patches(&pxt, gh, gw).expect("encode");
    assert_eq!(pooled.shape[0], img_pos.len(), "tower produced {} rows for {} image tokens",
               pooled.shape[0], img_pos.len());

    // ---- language model ---------------------------------------------------------------------
    let hf = HfCheckpoint::open(&dir).expect("open HF checkpoint");
    let model = Qwen3::load(&ctx, &hf).expect("Qwen3::load");
    println!("text model: {} layers, n_embd {}", model.cfg.n_layer, model.cfg.n_embd);

    // splice the merger's rows in place of the placeholder embeddings
    let raw = model.embed_tokens(&ids);
    let spliced = model.splice_image_embeds(&raw, img_start, &pooled);

    // mRoPE positions: text tokens get the sequence position in all three, image tokens their grid
    // coordinates, and the image advances the position by max(h, w) / merge.
    let types: Vec<u8> = (0..ids.len()).map(|i| u8::from(img_pos.contains(&i))).collect();
    // ⛔ Cross-checked against the fixture's own `types` line. The reference is only a multimodal
    // one if the image run is bracketed by <vision_start>/<vision_end>: transformers locates images
    // by scanning for that marker, and without it `get_rope_index` finds ZERO images and hands the
    // rotary plain text positions on all three axes. The forward still runs and still splices the
    // image embeddings, so the fixture looks multimodal and silently is not.
    let meta_types: Vec<u8> = field("types").bytes().map(|c| c - b'0').collect();
    assert_eq!(types, meta_types, "derived modality ids disagree with the fixture's own");
    assert!(ids.contains(&field("vision_start_token_id").parse::<u32>().unwrap()),
            "no <vision_start> in the fixture sequence — mRoPE would never fire");
    let ri = rope_index(&types, &[(gt, gh, gw)], c.merge).expect("rope_index");
    let mut mrope: Vec<u32> = Vec::with_capacity(4 * ids.len());
    for v in [&ri.t, &ri.h, &ri.w] { mrope.extend(v.iter().map(|&x| x as u32)); }
    mrope.extend(std::iter::repeat_n(0u32, ids.len()));

    // ⛔ DIAGNOSTIC SPLIT: with FERRIC_PRE_REF set, the deepstack features are replaced by ZEROS, so
    // the taps capture each layer's own output with nothing added. Compared against HF post-hooks
    // (which fire before `_deepstack_process`), that separates "the layer computes the wrong thing"
    // from "the injection is wrong" — two hypotheses the combined seam cannot tell apart.
    let pre_ref = std::env::var("FERRIC_PRE_REF").unwrap_or_default();
    let zeros: Vec<Tensor> = deep.iter()
        .map(|d| Tensor::from_vec(&ctx, &vec![0f32; d.shape[0] * d.shape[1]], &d.shape)).collect();
    let feed: &[Tensor] = if pre_ref.is_empty() { &deep } else { &zeros };

    let mut cache = Cache::new(&model.cfg);
    let mut taps: Vec<Tensor> = Vec::new();
    let final_h = model.forward_embeds_mm(&spliced, &mut cache, &mrope, feed, img_start, &mut taps);
    if !pre_ref.is_empty() {
        println!("\nDEEPSTACK ZEROED — taps are each layer's own output, vs HF post-hooks:");
        for (k, t) in taps.iter().enumerate() {
            let g = t.to_vec().await;
            let w = read_f32s(&format!("{pre_ref}.pre{k}"));
            compare(&format!("pre{k}"), &g, &w);
            per_row(&format!("pre{k}"), &g, &w, model.cfg.n_embd, &img_pos);
        }
        return;
    }

    println!("\nvs HuggingFace transformers 5.1.0 (M3 Ultra):");
    let mut worst = 0f32;
    // the layer-0 input is the spliced rows after whatever the model applies before layer 0
    let embed_in = if model.cfg.embd_rmsnorm { spliced.rmsnorm_weightless(model.cfg.eps) } else { spliced.clone() };
    let ge = embed_in.to_vec().await; let we = read_f32s(&format!("{fx}/ref.embed"));
    worst = worst.max(compare("embed", &ge, &we));
    per_row("embed", &ge, &we, model.cfg.n_embd, &img_pos);
    assert_eq!(taps.len(), deep.len(), "one tap per deepstack feature");
    for (k, t) in taps.iter().enumerate() {
        let g = t.to_vec().await;
        let w = read_f32s(&format!("{fx}/ref.layer{k}"));
        let r = compare(&format!("layer{k}"), &g, &w);
        per_row(&format!("layer{k}"), &g, &w, model.cfg.n_embd, &img_pos);
        worst = worst.max(r);
    }
    worst = worst.max(compare("final", &final_h.to_vec().await, &read_f32s(&format!("{fx}/ref.final"))));

    assert!(worst < 5e-3, "worst relative deviation {worst:.3e} is too large to be f32 drift");
    println!("\n✅ image path matches the published implementation to {worst:.2e} relative");
}
