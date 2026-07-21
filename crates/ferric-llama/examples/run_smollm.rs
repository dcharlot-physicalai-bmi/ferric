//! Runs the REAL published SmolLM2-135M checkpoint (HuggingFaceTB/SmolLM2-135M — a 30-layer Llama,
//! GQA 9→3, tied embeddings, bf16) entirely on the Ferric unified runtime, and validates its logits
//! against a numpy forward on the same weights. Same next-token argmax ⇒ Ferric correctly runs a real
//! frontier-lab model. Weights are downloaded to ~/.cache/ferric/smollm2-135m (not in the repo).
use ferric_load::safetensors;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// SmolLM2-135M config (config.json)
const NL: usize = 30;
const D: usize = 576;
const NH: usize = 9;
const NKV: usize = 3;
const DH: usize = 64;
const VOCAB: usize = 49152;
const BASE: f32 = 100000.0;
const EPS: f32 = 1e-5;

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let dir = format!("{}/.cache/ferric/smollm2-135m", std::env::var("HOME").unwrap());
    let bytes = std::fs::read(format!("{dir}/model.safetensors")).expect("download SmolLM2-135M first");
    println!("loading SmolLM2-135M ({:.0} MB safetensors)…", bytes.len() as f64 / 1e6);
    let w = safetensors(&bytes).unwrap();
    let g: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), Tensor::from_vec(&ctx, &t.data, &t.shape))).collect();
    let get = |n: &str| g.get(n).unwrap_or_else(|| panic!("missing {n}"));
    let ids: Vec<u32> = std::fs::read_to_string(format!("{dir}/ids.txt")).unwrap().trim().split(',').map(|s| s.parse().unwrap()).collect();
    let refl = f32s(&std::fs::read(format!("{dir}/ref_logits.bin")).unwrap()); // last-token logits [VOCAB]

    // forward, entirely on ferric-tensor
    let mut x = get("model.embed_tokens.weight").gather_rows(&ids); // [T, D]
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
    let logits = nn::linear_hf(&x, get("model.embed_tokens.weight")).to_vec().await; // [T, VOCAB], tied head
    let last = &logits[(ids.len() - 1) * VOCAB..];

    let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let ref_next = refl.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let den = refl.iter().map(|v| v.abs()).fold(1e-6, f32::max);
    let rel = last.iter().zip(&refl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / den;
    println!("Ferric · REAL SmolLM2-135M · 30 layers · GQA {NH}→{NKV} · tied · {:?}", ctx.backend);
    println!("  input ids {ids:?} → next-token argmax {next} (numpy ref {ref_next}) · rel logit err = {rel:.2e}");
    assert_eq!(next, ref_next, "argmax disagrees with the numpy reference");
    println!("✅ Ferric runs a REAL published model (SmolLM2-135M) — same next-token as the numpy reference");
}
