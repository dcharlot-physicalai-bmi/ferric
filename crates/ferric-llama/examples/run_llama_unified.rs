//! ferric-llama, ported onto the unified `ferric-tensor` runtime: load the SAME Llama/SmolLM-layout
//! safetensors checkpoint and run its forward pass using only general-runtime ops (matmul, broadcast,
//! fused rmsnorm/rope/softmax, GQA attention composed from primitives, HF [out,in] linears). Validated
//! against the same numpy reference the ferric-core path matched — proving the model runs on the one
//! substrate, no bespoke kernels.
use ferric_load::safetensors;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (n_layers, d, nh, nkv, dh, hidden, vocab, base, eps) = (2usize, 64usize, 8usize, 2usize, 8usize, 128usize, 32usize, 10000.0f32, 1e-5f32);
    let dir = env!("CARGO_MANIFEST_DIR");
    let w = safetensors(&std::fs::read(format!("{dir}/testdata/llama.safetensors")).unwrap()).unwrap();
    let ids: Vec<u32> = std::fs::read_to_string(format!("{dir}/testdata/llama.ids")).unwrap().trim().split(',').map(|s| s.parse().unwrap()).collect();
    let refl = f32s(&std::fs::read(format!("{dir}/testdata/llama.ref.bin")).unwrap());

    // upload every checkpoint tensor to the fabric
    let g: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), Tensor::from_vec(&ctx, &t.data, &t.shape))).collect();
    let get = |n: &str| g.get(n).unwrap_or_else(|| panic!("missing {n}"));

    // forward, entirely on ferric-tensor
    let mut x = get("model.embed_tokens.weight").gather_rows(&ids); // [T, d]
    for i in 0..n_layers {
        let p = format!("model.layers.{i}");
        let h = x.rmsnorm(get(&format!("{p}.input_layernorm.weight")), eps);
        let q = nn::linear_hf(&h, get(&format!("{p}.self_attn.q_proj.weight"))).rope(nh, dh, base, 0);
        let k = nn::linear_hf(&h, get(&format!("{p}.self_attn.k_proj.weight"))).rope(nkv, dh, base, 0);
        let v = nn::linear_hf(&h, get(&format!("{p}.self_attn.v_proj.weight")));
        let attn = nn::causal_attention(&q, &k, &v, nh, nkv);
        x = x.add(&nn::linear_hf(&attn, get(&format!("{p}.self_attn.o_proj.weight"))));
        let h2 = x.rmsnorm(get(&format!("{p}.post_attention_layernorm.weight")), eps);
        let gate = nn::linear_hf(&h2, get(&format!("{p}.mlp.gate_proj.weight")));
        let up = nn::linear_hf(&h2, get(&format!("{p}.mlp.up_proj.weight")));
        x = x.add(&nn::linear_hf(&gate.silu().mul(&up), get(&format!("{p}.mlp.down_proj.weight"))));
    }
    let x = x.rmsnorm(get("model.norm.weight"), eps);
    let head = if g.contains_key("lm_head.weight") { "lm_head.weight" } else { "model.embed_tokens.weight" };
    let logits = nn::linear_hf(&x, get(head)).to_vec().await; // [T, vocab], tied embeddings

    let diff = logits.iter().zip(&refl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let last = &logits[(ids.len() - 1) * vocab..];
    let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    println!("ferric-llama on the UNIFIED runtime · {:?} · GQA {nh}→{nkv} · {n_layers} layers", ctx.backend);
    println!("  {} tokens → next-token argmax {next} · max|ferric-tensor - numpy| = {diff:.2e}", ids.len());
    assert!(diff < 2e-3, "unified llama mismatch {diff}");
    println!("✅ A real Llama/SmolLM checkpoint runs on the general ferric-tensor runtime — matches numpy");
}
