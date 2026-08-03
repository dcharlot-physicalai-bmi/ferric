//! AMD Instella-MoE — the novel **Gated Multi-head Latent Attention** block, in pure Rust, verified
//! layer-exact against AMD's real `MLAGatedAttention` (a step-by-step reference that matched the HF module
//! at maxΔ=0.0). This is DeepSeek-V3 MLA (KV-compression, nope/rope split, interleaved YaRN rope) plus the
//! Instella delta: `attn_output * sigmoid(gate_proj(x))` before o_proj. Weights + I/O + intermediates come
//! from `~/.cache/ferric/instella_ref/gmla.safetensors`.
//!   cargo run -p ferric-llama --example instella_gmla --release
use ferric_core::{max_abs_diff, Context};
use ferric_load::safetensors;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;
const SCALING: f32 = 0.16562687709876717; // DeepSeek-V3 MLA scaling with YaRN mscale (from the reference)

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let bytes = std::fs::read(format!("{home}/.cache/ferric/instella_ref/gmla.safetensors")).unwrap();
    let w = safetensors(&bytes).unwrap();
    let w: HashMap<String, ferric_load::STensor> = w.into_iter().collect();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };

    let (h, qk, nope, rope, vh, kvl) = (16usize, 128usize, 96usize, 32usize, 128usize, 512usize);
    let hs = g("hs");                    // [S, 2048]
    let s = hs.shape[0];
    let (cos, sin) = (g("cos"), g("sin"));  // [S, 32]

    // transformers' apply_rotary_pos_emb_interleave = de-interleave (a0,b0,a1,b1..)→(a0..,b0..) then
    // split-half rope with the doubled cos/sin table. So: de-interleave, then apply_rope_costable.
    let deint = |x: &Tensor, rows: usize| x.reshape(&[rows, rope / 2, 2]).transpose(1, 2).contiguous().reshape(&[rows, rope]);

    // ---- Q: full projection, split nope/rope, rope on the rope part ----
    let q = hs.matmul_bt(&g("q_proj.weight")).reshape(&[s, h, qk]);   // [S,H,128]
    let q_pass = q.narrow(2, 0, nope).contiguous();                  // [S,H,96]
    let q_rot = deint(&q.narrow(2, nope, rope).contiguous(), s * h)  // de-interleave each head's 32
        .reshape(&[s, h * rope])
        .apply_rope_costable(&cos, &sin, h, rope).reshape(&[s, h, rope]); // [S,H,32]

    // ---- KV: compress → layernorm → up-project → split nope/value ; rope on the shared MQA rope part ----
    let ckv = hs.matmul_bt(&g("kv_a_proj_with_mqa.weight"));         // [S,544]
    let k_passc = ckv.narrow(1, 0, kvl).contiguous();               // [S,512]
    let k_rot = ckv.narrow(1, kvl, rope).contiguous();             // [S,32]
    let kb = k_passc.rmsnorm(&g("kv_a_layernorm.weight"), EPS)
        .matmul_bt(&g("kv_b_proj.weight")).reshape(&[s, h, nope + vh]); // [S,H,224]
    let k_nope = kb.narrow(2, 0, nope).contiguous();                // [S,H,96]
    let value = kb.narrow(2, nope, vh).contiguous();                // [S,H,128]
    let k_rot_roped = deint(&k_rot, s).apply_rope_costable(&cos, &sin, 1, rope);   // [S,32] (shared across heads)
    let d_krot = max_abs_diff(&k_rot_roped.to_vec().await, &g("krot_post").to_vec().await);
    println!("  [dbg] k_rot post-rope maxΔ = {d_krot:.3e}  (isolates the rope convention)");
    let k_rot = k_rot_roped.reshape(&[s, 1, rope]).broadcast_to(&[s, h, rope]).contiguous(); // [S,H,32]

    // ---- assemble Q,K,V as [S, H*128] and run causal attention with the custom scaling ----
    let qh = q_pass.cat(&q_rot, 2).reshape(&[s, h * qk]);
    let kh = k_nope.cat(&k_rot, 2).reshape(&[s, h * qk]);
    let vv = value.reshape(&[s, h * vh]);
    let qh = qh.mul(&qh.scalar(SCALING * (qk as f32).sqrt()));       // pre-scale so 1/√dh · qh = SCALING·qh
    let ao = nn::causal_attention(&qh, &kh, &vv, h, h, 0.0);         // [S,2048]

    // ---- the Instella gate + output projection ----
    let gate = hs.matmul_bt(&g("gate_proj.weight")).sigmoid();       // [S,2048]
    let ao_gated = ao.mul(&gate);
    let out = ao_gated.matmul_bt(&g("o_proj.weight"));               // [S,2048]

    // ---- verify against AMD's real module (and a couple intermediates) ----
    let d_gate = max_abs_diff(&gate.to_vec().await, &g("gate").to_vec().await);
    let d_ao = max_abs_diff(&ao_gated.to_vec().await, &g("ao_pregate").to_vec().await); // note: ao_pregate is pre-gate
    let d_out = max_abs_diff(&out.to_vec().await, &g("out").to_vec().await);
    println!("Instella Gated-MLA in Ferric vs AMD real module (S={s}, {h} heads):");
    println!("  gate  maxΔ = {d_gate:.3e}");
    println!("  attn·gate vs pre-gate maxΔ = {d_ao:.3e}  (sanity — differs by the gate, expected nonzero)");
    println!("  OUTPUT maxΔ = {d_out:.3e}  ->  {}", if d_out < 2e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(d_out < 2e-4, "Instella Gated-MLA diverged: {d_out}");
    println!("\n✅ AMD Instella-MoE's Gated Multi-head Latent Attention runs layer-exact in pure Rust Ferric.");
}
