//! **DeepSeek-V2 / Coder-V2 on Ferric** — the correctness run.
//!
//! Takes token ids so a mismatch is the forward pass and not the tokenizer.
//!
//! ```text
//!   llama-tokenize -m DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf -p "def fibonacci(n):" --ids
//!   cargo run -p ferric-llama --example run_deepseek2 --release -- <model.gguf> <ids> [n]
//! ```
//!
//! This module carries more unvalidated convention-dependent constants than most: YaRN appears in two
//! places (`attn_factor` on the rope lanes, `mscale²` in the attention scale), the MoE routed weights
//! are NOT renormalised on V2, and the query/key head is wider than the value head. Each of those
//! produces fluent, wrong text on its own, so the only real check is against the reference.
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::deepseek2::{Cache, DeepSeek2};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: run_deepseek2 <model.gguf> <comma,separated,ids> [n_gen]");
    let ids: Vec<u32> = a.get(2).expect("token ids")
        .split(',').map(|s| s.trim().parse().expect("id")).collect();
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(path).expect("open");
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    assert_eq!(arch, "deepseek2", "this loader is for deepseek2, got {arch:?}");

    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let eos: Vec<u32> = ["eos_token_id"].iter()
        .filter_map(|k| match g.metadata.get(&format!("tokenizer.ggml.{k}")) { Some(Meta::U(v)) => Some(*v as u32), _ => None })
        .collect();

    let t0 = std::time::Instant::now();
    let m = DeepSeek2::load(&ctx, &g).expect("load deepseek2");
    let c = &m.cfg;
    println!("deepseek2 · {} blocks, d={}, {} heads", c.n_layer, c.d, c.n_head);
    println!("  MLA: qk {} = {} nope + {} rope, v {}, latent {}",
             c.qk_head(), c.qk_nope, c.qk_rope, c.v_head, c.kv_lora_rank);
    println!("  MoE: {} experts, {} used, {} shared, ff {} ({} dense lead, dense ff {})",
             c.n_expert, c.n_expert_used, c.n_expert_shared, c.expert_ff, c.dense_lead, c.n_ff);
    println!("  routing: {} gate, renorm {}, scale {}",
             if c.sigmoid_gate { "sigmoid" } else { "softmax" }, c.expert_norm, c.routed_scale);
    println!("  YaRN: factor {} → mscale {:.5}, attn_factor {:.5}, q prescale {:.5}",
             c.yarn_factor, c.mscale(), c.attn_factor(), c.q_prescale());
    println!("  loaded in {:.2?}", t0.elapsed());

    let vn = c.n_vocab;
    let argmax = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;

    let t0 = std::time::Instant::now();
    let mut cache = Cache::new(c);
    let logits = m.forward(&ids, &mut cache).to_vec().await;
    let last = &logits[logits.len() - vn..];
    let bad = last.iter().filter(|v| !v.is_finite()).count();
    let (mn, mx) = last.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("  prefill {} ids in {:.2?} · {bad} non-finite, min {mn:.3} max {mx:.3}", ids.len(), t0.elapsed());
    assert_eq!(bad, 0, "non-finite logits out of prefill");

    let mut next = argmax(&logits);
    let mut out = Vec::new();
    let t0 = std::time::Instant::now();
    for _ in 0..n_gen {
        if eos.contains(&next) { break; }
        out.push(next);
        next = argmax(&m.forward(&[next], &mut cache).to_vec().await);
    }
    let dt = t0.elapsed();

    // GPT-2 byte-level vocab: map the printable-unicode aliases back to raw bytes.
    let mut u2b = std::collections::HashMap::new();
    let mut n = 0u32;
    for b in 0u32..256 {
        let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let ch = if p { b } else { let c = 256 + n; n += 1; c };
        u2b.insert(char::from_u32(ch).unwrap(), b as u8);
    }
    let bytes: Vec<u8> = out.iter().flat_map(|&t| tokens[t as usize].chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).collect();
    println!("\n--- Ferric ---\n{}\n", String::from_utf8_lossy(&bytes));
    println!("{} tokens in {:.2?} ({:.1} ms/tok)", out.len(), dt, dt.as_secs_f64() * 1000.0 / out.len().max(1) as f64);
    println!("ids: {out:?}");
    println!("\nCompare against llama-cli --temp 0 on the same prompt. Fluent text about the wrong");
    println!("thing means one of: YaRN mscale, YaRN attn_factor, MoE renormalisation, or the");
    println!("split query/value head widths.");
}
