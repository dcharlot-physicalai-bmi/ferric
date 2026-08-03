//! AMD Instella-MoE — the FULL model assembly in pure Rust: embed → [dense layer-0, FarSkip-MoE layers] →
//! final norm → LM head. Verified logit-exact against AMD's real InstellaMoEForCausalLM.forward (patched to
//! run on the installed transformers). Reduced scale (256 vocab, 3 layers, 8 experts) at REAL MLA dims —
//! identical assembly to the 16B; real weights plug into this same code (quantized) next.
//! Unified FarSkip update (per layer): residual=stream0; attn reads stream_attn; MLP reads post_ln(stream0);
//! dense → (out,out); MoE → (residual+routed+shared, residual+shared).
//!   cargo run -p ferric-llama --example instella_model --release
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
    let w = safetensors(&std::fs::read(format!("{home}/.cache/ferric/instella_ref/model.safetensors")).unwrap()).unwrap();
    let w: HashMap<String, STensor> = w.into_iter().collect();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let ids: Vec<u32> = vec![139, 188, 36, 181];   // from model.json golden
    let (nl, e, topk, _inter, scale) = (3usize, 8usize, 6usize, 512usize, 2.5f32);
    let (h, qk, nope, rope, vh, kvl) = (16usize, 128usize, 96usize, 32usize, 128usize, 512usize);
    let (cos, sin) = (g("cos"), g("sin"));
    let s = ids.len();

    // ---- gated-MLA attention (reads {p}.self_attn.* ) ----
    let gmla = |g: &dyn Fn(&str) -> Tensor, hs: &Tensor, p: &str| -> Tensor {
        let deint = |t: &Tensor, rows: usize| t.reshape(&[rows, rope / 2, 2]).transpose(1, 2).contiguous().reshape(&[rows, rope]);
        let q = hs.matmul_bt(&g(&format!("{p}.self_attn.q_proj.weight"))).reshape(&[s, h, qk]);
        let q_pass = q.narrow(2, 0, nope).contiguous();
        let q_rot = deint(&q.narrow(2, nope, rope).contiguous(), s * h).reshape(&[s, h * rope])
            .apply_rope_costable(&cos, &sin, h, rope).reshape(&[s, h, rope]);
        let ckv = hs.matmul_bt(&g(&format!("{p}.self_attn.kv_a_proj_with_mqa.weight")));
        let k_passc = ckv.narrow(1, 0, kvl).contiguous();
        let k_rot = ckv.narrow(1, kvl, rope).contiguous();
        let kb = k_passc.rmsnorm(&g(&format!("{p}.self_attn.kv_a_layernorm.weight")), EPS)
            .matmul_bt(&g(&format!("{p}.self_attn.kv_b_proj.weight"))).reshape(&[s, h, nope + vh]);
        let k_nope = kb.narrow(2, 0, nope).contiguous();
        let value = kb.narrow(2, nope, vh).contiguous();
        let k_rot = deint(&k_rot, s).apply_rope_costable(&cos, &sin, 1, rope)
            .reshape(&[s, 1, rope]).broadcast_to(&[s, h, rope]).contiguous();
        let qh = q_pass.cat(&q_rot, 2).reshape(&[s, h * qk]);
        let kh = k_nope.cat(&k_rot, 2).reshape(&[s, h * qk]);
        let vv = value.reshape(&[s, h * vh]);
        let qh = qh.mul(&qh.scalar(SCALING * (qk as f32).sqrt()));
        let ao = ferric_tensor::nn::causal_attention(&qh, &kh, &vv, h, h, 0.0);
        let gate = hs.matmul_bt(&g(&format!("{p}.self_attn.gate_proj.weight"))).sigmoid();
        ao.mul(&gate).matmul_bt(&g(&format!("{p}.self_attn.o_proj.weight")))
    };

    // ---- embed ----
    let mut stream0 = g("embed").gather_rows(&ids);   // [S, H]
    let mut stream_attn = stream0.clone();            // first layer: both streams = input

    for l in 0..nl {
        let p = format!("L{l}");
        let residual = stream0.clone();
        let attn_out = gmla(&g, &stream_attn.rmsnorm(&g(&format!("{p}.input_layernorm.weight")), EPS), &p);
        let residual = residual.add(&attn_out);                 // stream0 + attn
        let mlp_in = stream0.rmsnorm(&g(&format!("{p}.post_attention_layernorm.weight")), EPS); // post_ln(stream0)

        let is_moe = w.contains_key(&format!("{p}.gate_w"));
        if !is_moe {
            // dense MLP
            let mlp = mlp_in.matmul_bt(&g(&format!("{p}.mlp.gate_proj"))).silu()
                .mul(&mlp_in.matmul_bt(&g(&format!("{p}.mlp.up_proj"))))
                .matmul_bt(&g(&format!("{p}.mlp.down_proj")));
            stream0 = residual.add(&mlp); stream_attn = stream0.clone();
        } else {
            // DeepSeekMoE (router on host; experts + shared on device)
            let logits = mlp_in.matmul_bt(&g(&format!("{p}.gate_w"))).to_vec().await;
            let bias = g(&format!("{p}.gate_bias")).to_vec().await;
            let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
            let hdim = stream0.shape[1];
            let mut routed = vec![0f32; s * hdim];
            for t in 0..s {
                let sc: Vec<f32> = (0..e).map(|j| sig(logits[t * e + j])).collect();
                let sfc: Vec<f32> = (0..e).map(|j| sc[j] + bias[j]).collect();
                let mut ord: Vec<usize> = (0..e).collect();
                ord.sort_by(|&a, &b| sfc[b].total_cmp(&sfc[a]));
                let top = &ord[..topk];
                let wsum: f32 = top.iter().map(|&j| sc[j]).sum::<f32>() + 1e-20;
                let xt = mlp_in.narrow(0, t, 1);
                for &ex in top {
                    let wt = sc[ex] / wsum * scale;
                    let hh = xt.matmul_bt(&g(&format!("{p}.e{ex}.gate_proj"))).silu()
                        .mul(&xt.matmul_bt(&g(&format!("{p}.e{ex}.up_proj"))));
                    let o = hh.matmul_bt(&g(&format!("{p}.e{ex}.down_proj"))).to_vec().await;
                    for i in 0..hdim { routed[t * hdim + i] += wt * o[i]; }
                }
            }
            let shared = mlp_in.matmul_bt(&g(&format!("{p}.shared.gate_proj"))).silu()
                .mul(&mlp_in.matmul_bt(&g(&format!("{p}.shared.up_proj"))))
                .matmul_bt(&g(&format!("{p}.shared.down_proj"))).to_vec().await;
            let res = residual.to_vec().await;
            let main: Vec<f32> = (0..s * hdim).map(|i| res[i] + routed[i] + shared[i]).collect();
            let nr: Vec<f32> = (0..s * hdim).map(|i| res[i] + shared[i]).collect();
            stream0 = Tensor::from_vec(&ctx, &main, &[s, hdim]);
            stream_attn = Tensor::from_vec(&ctx, &nr, &[s, hdim]);
        }
    }

    // ---- final norm + LM head (take main stream) ----
    let hn = stream0.rmsnorm(&g("norm"), EPS);
    let logits = hn.matmul_bt(&g("lm_head")).to_vec().await;   // [S, vocab]
    let d = max_abs_diff(&logits, &g("logits").to_vec().await);
    let vocab = w["lm_head"].shape[0];
    let am = |v: &[f32]| (0..vocab).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    let mine: Vec<usize> = (0..s).map(|t| am(&logits[t * vocab..(t + 1) * vocab])).collect();
    let refv = g("logits").to_vec().await;
    let refa: Vec<usize> = (0..s).map(|t| am(&refv[t * vocab..(t + 1) * vocab])).collect();
    println!("Instella-MoE FULL MODEL in Ferric vs AMD real forward (S={s}, {nl} layers: 0 dense, 1-2 FarSkip-MoE):");
    println!("  logits maxΔ = {d:.3e}");
    println!("  argmax  Ferric {mine:?}  vs  AMD {refa:?}");
    let ok = d < 3e-4 && mine == refa;
    println!("  -> {}", if ok { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(ok, "Instella full model diverged: {d}");
    println!("\n✅ The FULL AMD Instella-MoE model assembly (embed → dense layer + FarSkip-MoE layers → norm → LM head)\n   runs logit-exact in pure Rust Ferric. Real 16B weights plug into this same code (quantized) next.");
}
