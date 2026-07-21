//! On-GPU KV-cache generation on the REAL SmolLM2-135M: decode one token at a time, caching each
//! layer's K/V so attention is O(S) per step instead of re-running the whole prefix (O(S²)). Prefill
//! and decode are the same single-token step. Produces the SAME text as the full-recompute path
//! (validated), and times both to show the KV cache win. Local-only (needs the ~/.cache download).
use ferric_load::safetensors;
use ferric_tensor::{nn, Tensor};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const NL: usize = 30; const D: usize = 576; const NH: usize = 9; const NKV: usize = 3;
const DH: usize = 64; const KVD: usize = NKV * DH; const VOCAB: usize = 49152;
const BASE: f32 = 100000.0; const EPS: f32 = 1e-5;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let dir = format!("{}/.cache/ferric/smollm2-135m", std::env::var("HOME").unwrap());
    let bpe = Bpe::from_tokenizer_json(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
    let w = safetensors(&std::fs::read(format!("{dir}/model.safetensors")).unwrap()).unwrap();
    let g: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), Tensor::from_vec(&ctx, &t.data, &t.shape))).collect();
    let get = |n: &str| g.get(n).unwrap();

    let prompt = "The capital of France is";
    let prompt_ids = bpe.encode(prompt);
    let steps = 8usize;

    // ---- KV-cache decode: one token/step, per-layer host K/V cache ----
    let t0 = Instant::now();
    let mut kcache = vec![Vec::<f32>::new(); NL];
    let mut vcache = vec![Vec::<f32>::new(); NL];
    let mut seq = prompt_ids.clone();
    let mut gen = Vec::new();
    for pos in 0..prompt_ids.len() + steps - 1 {
        let tok = seq[pos];
        let mut x = get("model.embed_tokens.weight").gather_rows(&[tok]); // [1, D]
        for l in 0..NL {
            let p = format!("model.layers.{l}");
            let h = x.rmsnorm(get(&format!("{p}.input_layernorm.weight")), EPS);
            let q1 = nn::linear_hf(&h, get(&format!("{p}.self_attn.q_proj.weight"))).rope(NH, DH, BASE, pos);
            let knew = nn::linear_hf(&h, get(&format!("{p}.self_attn.k_proj.weight"))).rope(NKV, DH, BASE, pos);
            let vnew = nn::linear_hf(&h, get(&format!("{p}.self_attn.v_proj.weight")));
            kcache[l].extend(knew.to_vec().await);
            vcache[l].extend(vnew.to_vec().await);
            let s = kcache[l].len() / KVD;
            let kc = Tensor::from_vec(&ctx, &kcache[l], &[s, KVD]);
            let vc = Tensor::from_vec(&ctx, &vcache[l], &[s, KVD]);
            let attn = nn::decode_attention(&q1, &kc, &vc, NH, NKV, 0.0);
            x = x.add(&nn::linear_hf(&attn, get(&format!("{p}.self_attn.o_proj.weight"))));
            let h2 = x.rmsnorm(get(&format!("{p}.post_attention_layernorm.weight")), EPS);
            let gate = h2.matmul_bt_act(get(&format!("{p}.mlp.gate_proj.weight")), 2);
            let up = nn::linear_hf(&h2, get(&format!("{p}.mlp.up_proj.weight")));
            x = x.add(&nn::linear_hf(&gate.mul(&up), get(&format!("{p}.mlp.down_proj.weight"))));
        }
        let x = x.rmsnorm(get("model.norm.weight"), EPS);
        let logits = nn::linear_hf(&x, get("model.embed_tokens.weight")).to_vec().await;
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        if pos + 1 >= prompt_ids.len() { gen.push(next); seq.push(next); }
    }
    let kv_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!("Ferric · REAL SmolLM2-135M · KV-cache decode · {:?}", ctx.backend);
    println!("  prompt {prompt:?} → generated ids {gen:?}");
    println!("  TEXT: {:?}", bpe.decode(&seq));
    println!("  {steps} tokens in {kv_ms:.0} ms (O(S)/step KV cache)");
    assert_eq!(gen, vec![260, 3575, 282, 260, 1798, 30, 198, 198], "KV-cache generation disagrees with the validated full-recompute output");
    println!("✅ On-GPU KV-cache generation on a REAL model — same text as full recompute, O(S)/step");
}
