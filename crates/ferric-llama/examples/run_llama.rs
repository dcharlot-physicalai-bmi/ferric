// Load a Llama/SmolLM-layout safetensors checkpoint (GQA, tied embeddings) with Ferric's pure-Rust
// loader + bridge, run the forward pass on-GPU, and validate the logits against an independent
// numpy Llama reference. Proves the HF name-mapping, [out,in]→mm_bt transpose, and GQA are correct.
use ferric_core::max_abs_diff;
use ferric_core::Context;
use ferric_llama::{Config, Llama};
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect() }
fn main() { pollster::block_on(run()); }
async fn run() {
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
