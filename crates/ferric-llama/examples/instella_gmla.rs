//! AMD Instella-MoE — **Gated Multi-head Latent Attention**, verified layer-exact against AMD's real
//! `MLAGatedAttention` module (a step-by-step reference that matched the HF module at maxΔ=0.0).
//!
//! This is DeepSeek-V3 MLA (KV-compression, nope/rope split, interleaved YaRN rope) plus the Instella
//! delta: `attn_output * sigmoid(gate_proj(x))` before `o_proj`.
//!
//! **The implementation now lives in `ferric_llama::mla`**, not in this file. That is deliberate: MLA
//! exists to shrink the KV cache, which makes it the attention a memory-tiered engine wants, so it has to
//! be a library API rather than example code. This example is what keeps the promotion honest — it runs
//! the same comparison against the same reference tensors, but **through the shipped library path**, so
//! the layer-exactness claim is about the code that ships and not about a copy of it that has since
//! drifted.
//!
//! Weights + I/O + intermediates come from `~/.cache/ferric/instella_ref/gmla.safetensors`.
//!   cargo run -p ferric-llama --example instella_gmla --release
use ferric_core::{max_abs_diff, Context};
use ferric_llama::mla::{CachePolicy, Mla, MlaConfig, MlaWeights};
use ferric_load::safetensors;
use ferric_tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let bytes = std::fs::read(format!("{home}/.cache/ferric/instella_ref/gmla.safetensors")).unwrap();
    let w = safetensors(&bytes).unwrap();
    let w: HashMap<String, ferric_load::STensor> = w.into_iter().collect();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };

    let cfg = MlaConfig {
        n_heads: 16,
        qk_nope_dim: 96,
        qk_rope_dim: 32,
        v_head_dim: 128,
        kv_lora_rank: 512,
        // DeepSeek-V3 MLA scaling with YaRN mscale, taken from the reference rather than derived —
        // 1/sqrt(128) would be 0.0884, which is a different model.
        scaling: 0.16562687709876717,
        eps: 1e-6,
        rope_interleaved: true,
    };
    let mla = Mla::new(
        cfg,
        MlaWeights {
            q_proj: g("q_proj.weight"),
            kv_a_proj_with_mqa: g("kv_a_proj_with_mqa.weight"),
            kv_a_layernorm: g("kv_a_layernorm.weight"),
            kv_b_proj: g("kv_b_proj.weight"),
            o_proj: g("o_proj.weight"),
            gate_proj: Some(g("gate_proj.weight")),
        },
    );

    let hs = g("hs");
    let (cos, sin) = (g("cos"), g("sin"));
    let s = hs.shape[0];

    let out = mla.forward(&hs, &cos, &sin);

    let d_out = max_abs_diff(&out.to_vec().await, &g("out").to_vec().await);
    println!("Instella Gated-MLA via ferric_llama::mla vs AMD's real module (S={s}, {} heads):", cfg.n_heads);
    println!("  OUTPUT maxΔ = {d_out:.3e}  ->  {}", if d_out < 2e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(d_out < 2e-4, "Instella Gated-MLA diverged after promotion to src/: {d_out}");

    // The reason MLA is worth having in a tiered engine, in the model's own numbers.
    println!(
        "\n  KV cache per position per layer:  dense {} floats  ->  latent {} floats  ({:.1}x smaller)",
        cfg.dense_cache_floats(), cfg.latent_cache_floats(), cfg.latent_compression()
    );
    println!(
        "  at f32: latent {} B/pos, expanded {} B/pos — the trade `ferric_llama::mla::CachePolicy` makes\n  \
         explicit rather than baking in, because it sets the caller's context-length ceiling.",
        CachePolicy::Latent.bytes_per_position(&cfg),
        CachePolicy::Expanded.bytes_per_position(&cfg)
    );

    println!("\n✅ Gated-MLA runs layer-exact in pure Rust Ferric — from the LIBRARY, not from this example.");
}
