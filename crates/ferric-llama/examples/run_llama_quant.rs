//! The real Llama/SmolLM checkpoint run with INT8-QUANTIZED WEIGHTS on the unified runtime: every
//! attention/MLP projection is per-output-channel int8 (weights at 1/4 memory, stay packed and
//! dequant on the fly in the matmul); norms/embeddings stay f32. Validated against the same numpy
//! reference — argmax must still match, logits within int8-quantization tolerance.
use ferric_load::safetensors;
use ferric_tensor::{nn, QRow, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (n_layers, nh, nkv, dh, base, eps, vocab) = (2usize, 8usize, 2usize, 8usize, 10000.0f32, 1e-5f32, 32usize);
    let dir = env!("CARGO_MANIFEST_DIR");
    let w = safetensors(&std::fs::read(format!("{dir}/testdata/llama.safetensors")).unwrap()).unwrap();
    let ids: Vec<u32> = std::fs::read_to_string(format!("{dir}/testdata/llama.ids")).unwrap().trim().split(',').map(|s| s.parse().unwrap()).collect();
    let refl = f32s(&std::fs::read(format!("{dir}/testdata/llama.ref.bin")).unwrap());

    let t: HashMap<String, Tensor> = w.iter().map(|(k, v)| (k.clone(), Tensor::from_vec(&ctx, &v.data, &v.shape))).collect();
    let ft = |n: &str| t.get(n).unwrap();
    // pre-quantize every projection weight to per-row int8 (stays packed at 1/4 memory)
    let q: HashMap<String, QRow> = t.iter()
        .filter(|(k, _)| k.ends_with("_proj.weight"))
        .map(|(k, v)| (k.clone(), v.quantize_rowwise(8)))
        .collect();
    let fq = |n: &str| q.get(n).unwrap();

    let mut x = ft("model.embed_tokens.weight").gather_rows(&ids);
    for i in 0..n_layers {
        let p = format!("model.layers.{i}");
        let h = x.rmsnorm(ft(&format!("{p}.input_layernorm.weight")), eps);
        let qh = nn::linear_hf_q(&h, fq(&format!("{p}.self_attn.q_proj.weight"))).rope(nh, dh, base, 0);
        let kh = nn::linear_hf_q(&h, fq(&format!("{p}.self_attn.k_proj.weight"))).rope(nkv, dh, base, 0);
        let vh = nn::linear_hf_q(&h, fq(&format!("{p}.self_attn.v_proj.weight")));
        let attn = nn::causal_attention(&qh, &kh, &vh, nh, nkv);
        x = x.add(&nn::linear_hf_q(&attn, fq(&format!("{p}.self_attn.o_proj.weight"))));
        let h2 = x.rmsnorm(ft(&format!("{p}.post_attention_layernorm.weight")), eps);
        let gate = nn::linear_hf_q(&h2, fq(&format!("{p}.mlp.gate_proj.weight")));
        let up = nn::linear_hf_q(&h2, fq(&format!("{p}.mlp.up_proj.weight")));
        x = x.add(&nn::linear_hf_q(&gate.silu().mul(&up), fq(&format!("{p}.mlp.down_proj.weight"))));
    }
    let x = x.rmsnorm(ft("model.norm.weight"), eps);
    let logits = nn::linear_hf(&x, ft("model.embed_tokens.weight")).to_vec().await; // tied head, f32

    let den = refl.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = logits.iter().zip(&refl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / den;
    let last = &logits[(ids.len() - 1) * vocab..];
    let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let ref_last = &refl[(ids.len() - 1) * vocab..];
    let ref_next = ref_last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    println!("ferric-llama · INT8 weights on the unified runtime · GQA {nh}→{nkv}");
    println!("  next-token argmax {next} (ref {ref_next}) · rel logit err vs f32 numpy = {rel:.2e}");
    assert!(next == ref_next && rel < 0.1, "int8 weight-quant llama off (argmax {next} vs {ref_next}, rel {rel})");
    println!("✅ A real checkpoint runs with INT8-quantized weights (1/4 memory) on the fabric — same prediction as f32");
}
