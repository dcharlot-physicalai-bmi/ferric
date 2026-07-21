//! GENERATES REAL TEXT from the real SmolLM2-135M on Ferric: loads the HF tokenizer.json, encodes a
//! prompt, runs greedy autoregressive decode entirely on the ferric-tensor unified runtime, and
//! decodes back to text. The tokenizer (vs HF `tokenizers`) and the generated ids (vs a numpy forward)
//! are both validated. Weights in ~/.cache/ferric/smollm2-135m (download first); local-only.
use ferric_load::safetensors;
use ferric_tensor::{nn, Tensor};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

// SmolLM2-135M config
const NL: usize = 30; const D: usize = 576; const NH: usize = 9; const NKV: usize = 3;
const DH: usize = 64; const VOCAB: usize = 49152; const BASE: f32 = 100000.0; const EPS: f32 = 1e-5;

fn logits_last(ctx: &Arc<ferric_core::Context>, g: &HashMap<String, Tensor>, ids: &[u32]) -> Tensor {
    let get = |n: &str| g.get(n).unwrap();
    let mut x = get("model.embed_tokens.weight").gather_rows(ids);
    for i in 0..NL {
        let p = format!("model.layers.{i}");
        let h = x.rmsnorm(get(&format!("{p}.input_layernorm.weight")), EPS);
        let q = nn::linear_hf(&h, get(&format!("{p}.self_attn.q_proj.weight"))).rope(NH, DH, BASE, 0);
        let k = nn::linear_hf(&h, get(&format!("{p}.self_attn.k_proj.weight"))).rope(NKV, DH, BASE, 0);
        let v = nn::linear_hf(&h, get(&format!("{p}.self_attn.v_proj.weight")));
        let attn = nn::causal_attention(&q, &k, &v, NH, NKV, 0.0);
        x = x.add(&nn::linear_hf(&attn, get(&format!("{p}.self_attn.o_proj.weight"))));
        let h2 = x.rmsnorm(get(&format!("{p}.post_attention_layernorm.weight")), EPS);
        let gate = h2.matmul_bt_act(get(&format!("{p}.mlp.gate_proj.weight")), 2); // fused silu(x·Wᵀ)
        let up = nn::linear_hf(&h2, get(&format!("{p}.mlp.up_proj.weight")));
        x = x.add(&nn::linear_hf(&gate.mul(&up), get(&format!("{p}.mlp.down_proj.weight"))));
    }
    let x = x.rmsnorm(get("model.norm.weight"), EPS);
    let _ = ctx;
    nn::linear_hf(&x, get("model.embed_tokens.weight"))
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let dir = format!("{}/.cache/ferric/smollm2-135m", std::env::var("HOME").unwrap());
    let bpe = Bpe::from_tokenizer_json(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
    let w = safetensors(&std::fs::read(format!("{dir}/model.safetensors")).unwrap()).unwrap();
    let g: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), Tensor::from_vec(&ctx, &t.data, &t.shape))).collect();

    let prompt = "The capital of France is";
    let mut ids = bpe.encode(prompt);
    println!("prompt {prompt:?} → ids {ids:?}");
    assert_eq!(ids, vec![504, 3575, 282, 4649, 314], "tokenizer disagrees with HF");

    let mut gen = Vec::new();
    for _ in 0..8 {
        let logits = logits_last(&ctx, &g, &ids).to_vec().await;
        let last = &logits[(ids.len() - 1) * VOCAB..];
        let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        ids.push(next); gen.push(next);
    }
    println!("generated ids {gen:?}");
    assert_eq!(gen, vec![260, 3575, 282, 260, 1798, 30, 198, 198], "generation disagrees with numpy reference");
    println!("  TEXT: {:?}", bpe.decode(&ids));
    println!("✅ Ferric GENERATES REAL TEXT from SmolLM2-135M — tokenizer matches HF, generation matches numpy");
}
