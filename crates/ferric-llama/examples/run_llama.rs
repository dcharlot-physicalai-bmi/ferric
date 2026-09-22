// Load a Llama/SmolLM-layout safetensors checkpoint (GQA, tied embeddings) with Ferric's pure-Rust
// loader + bridge, run the forward pass on-GPU, and validate the logits against an independent
// numpy Llama reference. Proves the HF name-mapping, [out,in]→mm_bt transpose, and GQA are correct.
use ferric_core::max_abs_diff;
use ferric_core::Context;
use ferric_llama::{Config, Llama};
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect() }
fn main() { pollster::block_on(run()); }
async fn run() {
    // ⛔ THIS EXAMPLE TAKES NO ARGUMENTS, AND SAYING SO IS THE POINT. It runs a FIXED 2-layer
    // synthetic safetensors fixture from `testdata/`. Verifying an unrelated change on 2026-09-22 I
    // ran `run_llama -- gemma-3-1b.gguf` and `-- gemma2-2b.gguf`, got the identical `1.431e-6` from
    // both, and believed I had regression-tested two real checkpoints. A nonexistent path prints the
    // same number. An example that silently ignores an argument reports on a model you did not test.
    let extra: Vec<String> = std::env::args().skip(1).collect();
    assert!(extra.is_empty(),
            "run_llama takes NO arguments — it validates a fixed synthetic fixture in testdata/ and \
             would have ignored {extra:?}, printing a number that says nothing about it. \
             To exercise a real GGUF use `--example generate`, `--example swa_probe` (per-layer \
             sliding-window schedule) or `--example cls_head` (classifier head).");
    let ctx = Context::new().await.unwrap();
    let dir = env!("CARGO_MANIFEST_DIR");
    let cfg = Config { n_layers: 2, d: 64, n_heads: 8, n_kv_heads: 2, head_dim: 8, hidden: 128, vocab: 32, rope_theta: 10000.0, eps: 1e-5 };
    let bytes = std::fs::read(format!("{dir}/testdata/llama.safetensors")).unwrap();
    let model = Llama::from_safetensors(&bytes, cfg).unwrap();
    let ids: Vec<u32> = std::fs::read_to_string(format!("{dir}/testdata/llama.ids")).unwrap()
        .trim().split(',').map(|s| s.parse().unwrap()).collect();
    let refl = f32s(&std::fs::read(format!("{dir}/testdata/llama.ref.bin")).unwrap());

    let logits = model.forward(&ctx, &ids).await.unwrap();
    let d = max_abs_diff(&logits, &refl);
    let vocab = model.cfg.vocab;
    let last = &logits[(ids.len()-1)*vocab..];
    let next = last.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
    println!("Ferric Llama bridge · {:?} · {} layers · GQA {}→{} heads · tied-emb", ctx.backend, model.cfg.n_layers, model.cfg.n_heads, model.cfg.n_kv_heads);
    println!("  loaded {} tokens → next-token argmax = {next}", ids.len());
    println!("  max|ferric - numpy-llama| over {}×{vocab} logits = {d:.3e}", ids.len());
    assert!(d < 2e-3, "llama bridge mismatch {d}");
    println!("✅ A Llama/SmolLM-layout safetensors checkpoint LOADS + RUNS in Ferric — matches numpy reference");
}
