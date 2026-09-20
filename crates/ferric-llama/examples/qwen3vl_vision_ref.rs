//! **The Qwen3-VL vision tower against the published implementation, at five seams.**
//!
//! ⛔ PIXELS ARE SUPPLIED, not produced by an image preprocessor. Preprocessing has its own rules
//! (smart-resize to patch multiples, normalisation constants) and the two references are known to
//! disagree on the pixel budget. Testing tower and preprocessing together would leave a mismatch with
//! two possible causes; this way it has one.
//!
//! Five seams, so a failure LOCALISES instead of merely existing:
//!   deep0/1/2   [4, 2048]  the deepstack projections from blocks 5, 11, 17
//!   hidden      [16,1024]  per-patch rows after all 24 blocks, BEFORE the merger
//!   pooled      [4, 2048]  merged 2x2 + projected to text width — what the LM consumes
//! A mismatch at deep1 but not deep0 points at blocks 6-11; one only at `pooled` points at the merger.
//!
//! ⛔ THE PIXEL ROWS ARE ALREADY IN SPATIAL-MERGE-BLOCK ORDER and must not be reordered. This check
//! first ran with the tower gathering them into block order a second time, which is invisible to
//! every shape assertion, leaves attention (permutation-equivariant) and the merger reshape (four
//! consecutive rows either way) perfectly happy, and still scrambles the image. It read as
//! rel 5.70e-1 at `pooled` and 7.71e-2 already at `deep0`; correct is ~5e-6.
//!
//!   cargo run -p ferric-llama --example qwen3vl_vision_ref --release -- <ckpt-dir>
use ferric_llama::qwen3vl_vision::VisionTower;
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

fn main() { pollster::block_on(run()); }

async fn run() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: qwen3vl_vision_ref <ckpt-dir> [ref-prefix]");
    // ⛔ The reference seams live IN THE REPO, and the default points at them. They were once kept
    // in a scratch directory; it was wiped between sessions and took both the data and the
    // generator with it, leaving a check that could not be re-run. Regenerate with
    // `examples/refgen/qwen3vl_vision_ref.py` only to re-derive them, never to make a run pass.
    let pfx = a.next().unwrap_or_else(|| format!(
        "{}/tests/fixtures/qwen3vl_vision/ref", env!("CARGO_MANIFEST_DIR")));
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter   : {} [{:?}]", ctx.adapter_name, ctx.backend);

    let tower = VisionTower::load(&ctx, &dir).expect("load vision tower");
    let c = &tower.cfg;
    println!("tower     : depth {} hidden {} heads {} ff {} merge {} deepstack {:?}",
             c.depth, c.hidden, c.heads, c.ff, c.merge, c.deepstack);

    // the same 4x4 patch grid the reference was run on
    let (gh, gw) = (4usize, 4usize);
    let px = read_f32s(&format!("{pfx}.px"));
    let row = 3 * c.temporal_patch * c.patch * c.patch;
    assert_eq!(px.len(), gh * gw * row, "pixel buffer is not {}x{row}", gh * gw);
    let pxt = Tensor::from_vec(&ctx, &px, &[gh * gw, row]);

    let (pooled, deep, hidden) = tower.encode_patches(&pxt, gh, gw).expect("encode");
    assert_eq!(deep.len(), c.deepstack.len(), "one deepstack output per index");

    println!("\nvs HuggingFace transformers:");
    let mut worst_rel = 0f32;
    // ⚠ Checked in PIPELINE ORDER so the first failure is the earliest divergence, not the loudest.
    // deep0 is block 5, deep1 block 11, deep2 block 17, `hidden` all 24, `pooled` the main merger.
    for (k, d) in deep.iter().enumerate() {
        let got = pollster::block_on(d.to_vec());
        worst_rel = worst_rel.max(compare(&format!("deep{k}"), &got, &read_f32s(&format!("{pfx}.deep{k}"))));
    }
    // ⛔ The pre-merger rows. The generator has always emitted these and this check did not read
    // them, so the header advertised a seam that did not exist: a mismatch at `pooled` alone could
    // not be separated from one carried in from the blocks. It is the only seam at VISION width
    // (1024) rather than text width (2048), so it is also the only one that can catch a merger
    // that is wrong in a way the projection happens to absorb.
    let got = pollster::block_on(hidden.to_vec());
    worst_rel = worst_rel.max(compare("hidden", &got, &read_f32s(&format!("{pfx}.hidden"))));
    let got = pollster::block_on(pooled.to_vec());
    worst_rel = worst_rel.max(compare("pooled", &got, &read_f32s(&pfx)));

    // f32 over 24 blocks plus a merger; a transposed or swapped weight moves this by O(1), not 1e-3.
    assert!(worst_rel < 5e-3, "worst relative deviation {worst_rel:.3e} is too large to be f32 drift");
    println!("\n✅ vision tower matches the published implementation to {worst_rel:.2e} relative");
}
