//! AMD Instella-MoE — the DeepSeekMoE block (noaux_tc sigmoid router + top-k experts + shared experts),
//! pure Rust, verified layer-exact against stock `DeepseekV3MoE` (the real MoE Instella wraps; a step-by-step
//! reference matched it at maxΔ=0.0). Router: sigmoid(logits)+bias → top-k by (score+bias) → weights are the
//! ORIGINAL sigmoid scores, normalized, ×routed_scaling. Experts + shared are SwiGLU FFNs.
//! (Reduced to 8 experts at real dims — same math as the 64-expert model; real weights come at full-model.)
//!   cargo run -p ferric-llama --example instella_moe --release
use ferric_core::{max_abs_diff, Context};
use ferric_load::safetensors;
use ferric_tensor::{Tensor};
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let w = safetensors(&std::fs::read(format!("{home}/.cache/ferric/instella_ref/moe.safetensors")).unwrap()).unwrap();
    let w: HashMap<String, ferric_load::STensor> = w.into_iter().collect();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let (hdim, inter, e, topk, scale) = (2048usize, 1408usize, 8usize, 6usize, 2.5f32);

    let x = g("x");                       // [S, H]
    let s = x.shape[0];

    // ---- router (host-side: top-k is a gather/sort) ----
    let logits = x.matmul_bt(&g("gate_w")).to_vec().await;    // [S, E]
    let bias = g("gate_bias").to_vec().await;                 // [E]
    let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
    let mut sel: Vec<Vec<(usize, f32)>> = Vec::new();         // per token: (expert, weight)
    for t in 0..s {
        let scores: Vec<f32> = (0..e).map(|j| sig(logits[t * e + j])).collect();
        let sfc: Vec<f32> = (0..e).map(|j| scores[j] + bias[j]).collect();
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| sfc[b].total_cmp(&sfc[a]));
        let top: Vec<usize> = order[..topk].to_vec();
        let mut wsum = 0.0; for &j in &top { wsum += scores[j]; }
        wsum += 1e-20;
        sel.push(top.iter().map(|&j| (j, scores[j] / wsum * scale)).collect());
    }

    // ---- experts (device matmuls per selected expert) + host accumulate into routed ----
    let mut routed = vec![0f32; s * hdim];
    for t in 0..s {
        let xt = x.narrow(0, t, 1);                          // [1, H]
        for &(ex, wt) in &sel[t] {
            let gu = xt.matmul_bt(&g(&format!("e{ex}_gate_up"))); // [1, 2*INTER]
            let gate = gu.narrow(1, 0, inter);
            let up = gu.narrow(1, inter, inter);
            let h = gate.silu().mul(&up);                     // [1, INTER]
            let o = h.matmul_bt(&g(&format!("e{ex}_down"))).to_vec().await; // [1, H]
            for i in 0..hdim { routed[t * hdim + i] += wt * o[i]; }
        }
    }

    // ---- shared experts (SwiGLU, inter = INTER*n_shared) ----
    let sh = x.matmul_bt(&g("shared_gate")).silu().mul(&x.matmul_bt(&g("shared_up")))
        .matmul_bt(&g("shared_down")).to_vec().await;        // [S, H]

    // ---- combine + verify ----
    let out: Vec<f32> = (0..s * hdim).map(|i| routed[i] + sh[i]).collect();
    let d_routed = max_abs_diff(&routed, &g("routed").to_vec().await);
    let d_shared = max_abs_diff(&sh, &g("shared").to_vec().await);
    let d_out = max_abs_diff(&out, &g("out").to_vec().await);
    println!("Instella DeepSeekMoE in Ferric vs stock DeepseekV3MoE (S={s}, {e} experts, top-{topk}):");
    println!("  routed maxΔ = {d_routed:.3e}");
    println!("  shared maxΔ = {d_shared:.3e}");
    println!("  OUTPUT maxΔ = {d_out:.3e}  ->  {}", if d_out < 2e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(d_out < 2e-4, "Instella MoE diverged: {d_out}");
    println!("\n✅ AMD Instella-MoE's DeepSeekMoE block (noaux_tc router + experts + shared) runs layer-exact in Ferric.");
}
