//! Runs the REAL Liquid AI LFM2-350M — a NOVEL conv/attention hybrid, not a plain transformer —
//! entirely on Ferric, validated against Hugging Face `transformers` (the ground truth). 16 layers:
//! gated short-conv blocks (in_proj→B/C/x, B⊙x → causal depthwise conv1d(L=3) → C⊙ → out_proj) at
//! most layers, GQA attention with QK-norm at layers {2,5,8,10,12,14}, SwiGLU MLP, untied lm_head.
//! Exercises the conv1d + gating primitives built for LFM2, on real published weights.
use ferric_load::safetensors;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const NL: usize = 16; const D: usize = 1024; const NH: usize = 16; const NKV: usize = 8;
const DH: usize = 64; const CONV_L: usize = 3; const VOCAB: usize = 65536;
const BASE: f32 = 1_000_000.0; const EPS: f32 = 1e-5;
const ATTN: [usize; 6] = [2, 5, 8, 10, 12, 14];

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let dir = format!("{}/.cache/ferric/lfm2-350m", std::env::var("HOME").unwrap());
    let bytes = std::fs::read(format!("{dir}/model.safetensors")).expect("save LFM2-350M via transformers first");
    println!("loading LFM2-350M ({:.0} MB)…", bytes.len() as f64 / 1e6);
    let w = safetensors(&bytes).unwrap();
    let g: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), Tensor::from_vec(&ctx, &t.data, &t.shape))).collect();
    let get = |n: &str| g.get(n).unwrap_or_else(|| panic!("missing {n}"));
    let ids: Vec<u32> = std::fs::read_to_string(format!("{dir}/ids.txt")).unwrap().trim().split(',').map(|s| s.parse().unwrap()).collect();
    let refl = f32s(&std::fs::read(format!("{dir}/ref_logits.bin")).unwrap());
    let t = ids.len();

    let mut x = get("model.embed_tokens.weight").gather_rows(&ids); // [T, D]
    for l in 0..NL {
        let p = format!("model.layers.{l}");
        let xn = x.rmsnorm(get(&format!("{p}.operator_norm.weight")), EPS);
        let op = if ATTN.contains(&l) {
            // GQA attention with per-head QK-norm
            let q = nn::linear_hf(&xn, get(&format!("{p}.self_attn.q_proj.weight")));
            let k = nn::linear_hf(&xn, get(&format!("{p}.self_attn.k_proj.weight")));
            let v = nn::linear_hf(&xn, get(&format!("{p}.self_attn.v_proj.weight")));
            let ql = get(&format!("{p}.self_attn.q_layernorm.weight"));
            let kl = get(&format!("{p}.self_attn.k_layernorm.weight"));
            let q = q.reshape(&[t * NH, DH]).rmsnorm(ql, EPS).reshape(&[t, NH * DH]).rope(NH, DH, BASE, 0);
            let k = k.reshape(&[t * NKV, DH]).rmsnorm(kl, EPS).reshape(&[t, NKV * DH]).rope(NKV, DH, BASE, 0);
            let attn = nn::causal_attention(&q, &k, &v, NH, NKV, 0.0);
            nn::linear_hf(&attn, get(&format!("{p}.self_attn.out_proj.weight")))
        } else {
            // gated short conv: in_proj → B,C,x ; B⊙x → conv1d(L=3) → C⊙ → out_proj
            let inw = get(&format!("{p}.conv.in_proj.weight")); // [3D, D]
            let slice = |lo: u32| inw.gather_rows(&(lo..lo + D as u32).collect::<Vec<_>>());
            let (bw, cw, xw) = (slice(0), slice(D as u32), slice(2 * D as u32));
            let b = xn.matmul_bt(&bw);
            let cc = xn.matmul_bt(&cw);
            let xin = xn.matmul_bt(&xw);
            let convw = get(&format!("{p}.conv.conv.weight")).reshape(&[D, CONV_L]); // [C,1,L]→[C,L]
            let xc = b.mul(&xin).depthwise_conv1d_causal(&convw, CONV_L);
            nn::linear_hf(&cc.mul(&xc), get(&format!("{p}.conv.out_proj.weight")))
        };
        x = x.add(&op);
        // SwiGLU MLP: w2( silu(w1·x) ⊙ (w3·x) )
        let h = x.rmsnorm(get(&format!("{p}.ffn_norm.weight")), EPS);
        let gate = h.matmul_bt_act(get(&format!("{p}.feed_forward.w1.weight")), 2); // fused silu
        let up = nn::linear_hf(&h, get(&format!("{p}.feed_forward.w3.weight")));
        x = x.add(&nn::linear_hf(&gate.mul(&up), get(&format!("{p}.feed_forward.w2.weight"))));
    }
    let x = x.rmsnorm(get("model.embedding_norm.weight"), EPS);
    let head = if g.contains_key("lm_head.weight") { "lm_head.weight" } else { "model.embed_tokens.weight" };
    let logits = x.matmul_bt(get(head)).to_vec().await; // tied to embeddings
    let last = &logits[(t - 1) * VOCAB..];

    let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let ref_next = refl.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let rel = last.iter().zip(&refl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / refl.iter().map(|v| v.abs()).fold(1e-6, f32::max);
    println!("Ferric · REAL Liquid AI LFM2-350M (conv/attention hybrid) · {:?}", ctx.backend);
    println!("  next-token argmax {next} (transformers ref {ref_next}) · rel logit err = {rel:.2e}");
    assert_eq!(next, ref_next, "argmax disagrees with transformers");
    println!("✅ Ferric runs a REAL LFM2 (Liquid AI conv-hybrid) — same next-token as HF transformers");
}
