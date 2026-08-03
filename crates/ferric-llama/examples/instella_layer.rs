//! AMD Instella-MoE — a FULL FarSkip decoder layer in pure Rust: input_layernorm → Gated-MLA → residual →
//! post_attention_layernorm → DeepSeekMoE → the FarSkip two-stream residual (main = stock DeepSeek-V3 residual
//! `+routed+shared`; routed-free `no_routed = +shared` feeds the next block's attention). Verified against a
//! golden composed from AMD's real MLAGatedAttention + DeepseekV3MoE with AMD's exact two-stream wiring.
//!   cargo run -p ferric-llama --example instella_layer --release
use ferric_core::{max_abs_diff, Context};
use ferric_load::{safetensors, STensor};
use ferric_tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;
const SCALING: f32 = 0.16562687709876717;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let w = safetensors(&std::fs::read(format!("{home}/.cache/ferric/instella_ref/layer.safetensors")).unwrap()).unwrap();
    let w: HashMap<String, STensor> = w.into_iter().collect();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let (h, qk, nope, rope, vh, kvl) = (16usize, 128usize, 96usize, 32usize, 128usize, 512usize);
    let (hdim, inter, e, topk, scale) = (2048usize, 1408usize, 8usize, 6usize, 2.5f32);

    let x = g("x");                       // [S, H]
    let s = x.shape[0];
    let (cos, sin) = (g("cos"), g("sin"));

    // ================= input_layernorm → Gated-MLA attention =================
    let hs = x.rmsnorm(&g("in_ln"), EPS);
    let deint = |t: &Tensor, rows: usize| t.reshape(&[rows, rope / 2, 2]).transpose(1, 2).contiguous().reshape(&[rows, rope]);
    let q = hs.matmul_bt(&g("attn.q_proj.weight")).reshape(&[s, h, qk]);
    let q_pass = q.narrow(2, 0, nope).contiguous();
    let q_rot = deint(&q.narrow(2, nope, rope).contiguous(), s * h).reshape(&[s, h * rope])
        .apply_rope_costable(&cos, &sin, h, rope).reshape(&[s, h, rope]);
    let ckv = hs.matmul_bt(&g("attn.kv_a_proj_with_mqa.weight"));
    let k_passc = ckv.narrow(1, 0, kvl).contiguous();
    let k_rot = ckv.narrow(1, kvl, rope).contiguous();
    let kb = k_passc.rmsnorm(&g("attn.kv_a_layernorm.weight"), EPS)
        .matmul_bt(&g("attn.kv_b_proj.weight")).reshape(&[s, h, nope + vh]);
    let k_nope = kb.narrow(2, 0, nope).contiguous();
    let value = kb.narrow(2, nope, vh).contiguous();
    let k_rot = deint(&k_rot, s).apply_rope_costable(&cos, &sin, 1, rope)
        .reshape(&[s, 1, rope]).broadcast_to(&[s, h, rope]).contiguous();
    let qh = q_pass.cat(&q_rot, 2).reshape(&[s, h * qk]);
    let kh = k_nope.cat(&k_rot, 2).reshape(&[s, h * qk]);
    let vv = value.reshape(&[s, h * vh]);
    let qh = qh.mul(&qh.scalar(SCALING * (qk as f32).sqrt()));
    let ao = ferric_tensor::nn::causal_attention(&qh, &kh, &vv, h, h, 0.0);
    let gate = hs.matmul_bt(&g("attn.gate_proj.weight")).sigmoid();
    let attn_out = ao.mul(&gate).matmul_bt(&g("attn.o_proj.weight"));

    let residual = x.add(&attn_out);                     // residual after attention

    // ================= post_attention_layernorm → DeepSeekMoE =================
    let hm = residual.rmsnorm(&g("post_ln"), EPS);
    let logits = hm.matmul_bt(&g("gate_w")).to_vec().await;
    let bias = g("gate_bias").to_vec().await;
    let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
    let mut routed = vec![0f32; s * hdim];
    for t in 0..s {
        let scores: Vec<f32> = (0..e).map(|j| sig(logits[t * e + j])).collect();
        let sfc: Vec<f32> = (0..e).map(|j| scores[j] + bias[j]).collect();
        let mut ord: Vec<usize> = (0..e).collect();
        ord.sort_by(|&a, &b| sfc[b].total_cmp(&sfc[a]));
        let top = &ord[..topk];
        let wsum: f32 = top.iter().map(|&j| scores[j]).sum::<f32>() + 1e-20;
        let ht = hm.narrow(0, t, 1);
        for &ex in top {
            let wt = scores[ex] / wsum * scale;
            let gu = ht.matmul_bt(&g(&format!("e{ex}_gate_up")));
            let hh = gu.narrow(1, 0, inter).silu().mul(&gu.narrow(1, inter, inter));
            let o = hh.matmul_bt(&g(&format!("e{ex}_down"))).to_vec().await;
            for i in 0..hdim { routed[t * hdim + i] += wt * o[i]; }
        }
    }
    let shared = hm.matmul_bt(&g("shared_gate")).silu().mul(&hm.matmul_bt(&g("shared_up")))
        .matmul_bt(&g("shared_down")).to_vec().await;

    // ================= FarSkip two-stream residual =================
    let res = residual.to_vec().await;
    let main: Vec<f32> = (0..s * hdim).map(|i| res[i] + routed[i] + shared[i]).collect();
    let no_routed: Vec<f32> = (0..s * hdim).map(|i| res[i] + shared[i]).collect();

    let d_main = max_abs_diff(&main, &g("main").to_vec().await);
    let d_nr = max_abs_diff(&no_routed, &g("no_routed").to_vec().await);
    println!("Instella FarSkip decoder layer in Ferric vs AMD (S={s}):");
    println!("  main stream (stock DS-V3 residual)  maxΔ = {d_main:.3e}");
    println!("  no_routed stream (→ next attention) maxΔ = {d_nr:.3e}");
    let ok = d_main < 3e-4 && d_nr < 3e-4;
    println!("  -> {}", if ok { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(ok, "Instella FarSkip layer diverged: main {d_main}, no_routed {d_nr}");
    println!("\n✅ A full AMD Instella-MoE FarSkip decoder layer (Gated-MLA + DeepSeekMoE + two-stream residual)\n   runs layer-exact in pure Rust Ferric — the complete decoder layer is done.");
}
