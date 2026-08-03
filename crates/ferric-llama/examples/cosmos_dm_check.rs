//! Verify Cosmos 3 Edge's DIFFUSION/generator tower — the dual-stream joint attention (tech report
//! Eq 7-8): AR tokens attend causally over K_AR; DM tokens attend fully over [K_AR; K_DM]. One
//! layer-0 forward with real weights + synthetic tokens, compared to a numpy reference.
//! usage: cosmos_dm_check <cosmos3-edge-dir>
use ferric_core::Context;
use ferric_load::safetensors_filtered;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let dir = std::env::args().nth(1).expect("usage: cosmos_dm_check <dir>");
    let ctx = Arc::new(Context::new().await.unwrap());
    // layer-0 AR + DM weights
    let keep = |n: &str| n.starts_with("layers.0.");
    let mut w = std::collections::HashMap::new();
    for e in std::fs::read_dir(format!("{dir}/transformer")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            w.extend(safetensors_filtered(&std::fs::read(&p).unwrap(), keep).unwrap());
        }
    }
    let t = |n: &str| { let s = &w[&format!("layers.0.{n}")]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let (nh, nkv, hd, eps, base) = (16usize, 8usize, 128usize, 1e-5f32, 1e8f32);
    let (ta, td) = (4usize, 3usize);
    // deterministic synthetic AR + DM tokens (numpy regenerates identically)
    let mk = |t: usize, s: f32| -> Vec<f32> { (0..t * 2048).map(|i| ((i as f32 * 0.01 + s).sin() * 0.1)).collect() };
    let xa = Tensor::from_vec(&ctx, &mk(ta, 0.0), &[ta, 2048]);
    let xd = Tensor::from_vec(&ctx, &mk(td, 1.0), &[td, 2048]);

    let mlp = |h: &Tensor, up: &Tensor, dn: &Tensor| h.matmul_bt(up).relu2().matmul_bt(dn);
    // AR stream: causal self-attention, k-norm only
    let ha = xa.rmsnorm(&t("input_layernorm.weight"), eps);
    let qa = ha.matmul_bt(&t("self_attn.to_q.weight")).rope(nh, hd, base, 0);
    let ka = ha.matmul_bt(&t("self_attn.to_k.weight")).reshape(&[ta, nkv, hd]).rmsnorm(&t("self_attn.k_norm_und_for_gen.weight"), eps).reshape(&[ta, nkv * hd]).rope(nkv, hd, base, 0);
    let va = ha.matmul_bt(&t("self_attn.to_v.weight"));
    let oa = nn::causal_attention(&qa, &ka, &va, nh, nkv, 0.0);
    let xa1 = xa.add(&oa.matmul_bt(&t("self_attn.to_out.weight")));
    let _xa2 = xa1.add(&mlp(&xa1.rmsnorm(&t("post_attention_layernorm.weight"), eps), &t("mlp.up_proj.weight"), &t("mlp.down_proj.weight")));

    // DM stream: q AND k normed, full attention over [K_AR; K_DM]
    let hd_ = xd.rmsnorm(&t("input_layernorm_moe_gen.weight"), eps);
    let qd = hd_.matmul_bt(&t("self_attn.add_q_proj.weight")).reshape(&[td, nh, hd]).rmsnorm(&t("self_attn.norm_added_q.weight"), eps).reshape(&[td, nh * hd]).rope(nh, hd, base, 0);
    let kd = hd_.matmul_bt(&t("self_attn.add_k_proj.weight")).reshape(&[td, nkv, hd]).rmsnorm(&t("self_attn.norm_added_k.weight"), eps).reshape(&[td, nkv * hd]).rope(nkv, hd, base, 0);
    let vd = hd_.matmul_bt(&t("self_attn.add_v_proj.weight"));
    let kcat = ka.cat(&kd, 0); // [Ta+Td, 1024]
    let vcat = va.cat(&vd, 0);
    let od = nn::full_attention_kv(&qd, &kcat, &vcat, nh, nkv);
    let xd1 = xd.add(&od.matmul_bt(&t("self_attn.to_add_out.weight")));
    let xd2 = xd1.add(&mlp(&xd1.rmsnorm(&t("post_attention_layernorm_moe_gen.weight"), eps), &t("mlp_moe_gen.up_proj.weight"), &t("mlp_moe_gen.down_proj.weight")));

    let o = xd2.to_vec().await;
    println!("Ferric DM layer-0 out (token0[:6]) = {:?}", &o[..6].iter().map(|v| (v * 1e5).round() / 1e5).collect::<Vec<_>>());
    println!("  DM out sum = {:.5} · finite = {}", o.iter().sum::<f32>(), o.iter().all(|x| x.is_finite()));
}
