//! **A published HuggingFace checkpoint, straight into a Ferric runtime, with no conversion step.**
//!
//! Until now every runtime here took `&impl GgufSource`, so running a published checkpoint meant
//! converting it to GGUF first — with llama.cpp's Python converter. That was the last and largest
//! way this project depended on that one: not a line of code and not a crate in the tree, but a
//! mandatory step in front of every model, invisible in `Cargo.lock` and total in practice.
//!
//! `ferric_load::hf::HfCheckpoint` implements `GgufSource` over `config.json` + `model.safetensors`,
//! so this drives the SAME `Lfm2::load` a GGUF file drives — the runtime cannot tell the difference,
//! which is the point. Nothing about LFM2 is special-cased below.
//!
//! ## The oracle is not another runtime
//!
//! `ref_logits.bin` beside the checkpoint is the last-token logit vector from **HuggingFace
//! `transformers`** — the implementation the weights were published against. Agreeing with it is a
//! statement about the model; agreeing with a second runtime is a statement about two runtimes.
//!
//! ⚠ A conversion bug does NOT look like a crash. Swap two of `w1`/`w2`/`w3`, or transpose a weight
//! whose shape looks wrong, and the model loads, runs, and produces confident nonsense. Which is why
//! the check is against published logits and not against "does it emit words".
//!
//!   cargo run -p ferric-llama --example hf_direct --release -- ~/.cache/ferric/lfm2-350m
use ferric_llama::lfm2::{Cache, Lfm2};
use ferric_load::hf::HfCheckpoint;
use std::sync::Arc;

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }

fn main() { pollster::block_on(run()); }

async fn run() {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| format!("{}/.cache/ferric/lfm2-350m", std::env::var("HOME").unwrap()));
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));

    let hf = HfCheckpoint::open(&dir).expect("open HF checkpoint");
    println!("HuggingFace checkpoint -> Ferric runtime, no conversion\n");
    println!("  {dir}");
    println!("  model_type {} · {} tensors presented as GGUF names", hf.arch, hf.names().count());

    // The whole claim in one line: a runtime written against GGUF, loading safetensors.
    let m = Lfm2::load(&ctx, &hf).expect("load");
    let n_attn = m.cfg.kv.iter().filter(|&&k| k > 0).count();
    println!("  loaded: {} blocks ({} conv + {n_attn} attn) · d={} · vocab={}",
             m.cfg.n_layer, m.cfg.n_layer - n_attn, m.cfg.d, m.cfg.n_vocab);
    // ⭐ The conv/attention schedule is the sharpest thing the mapping has to get right: GGUF says it
    // with a per-layer kv array where 0 means conv, HF says it with layer_types strings. Getting it
    // wrong runs a conv block as attention, which loads fine and is wrong everywhere.
    assert_eq!(n_attn, 6, "LFM2-350M has attention at 6 of 16 blocks; the schedule did not survive \
                           translation from layer_types");

    let ids: Vec<u32> = std::fs::read_to_string(format!("{dir}/ids.txt"))
        .expect("ids.txt").trim().split(',').map(|s| s.parse().unwrap()).collect();
    let refl = f32s(&std::fs::read(format!("{dir}/ref_logits.bin")).expect("ref_logits.bin"));
    assert_eq!(refl.len(), m.cfg.n_vocab, "reference logits are {} wide, vocab is {}",
               refl.len(), m.cfg.n_vocab);

    let mut cache = Cache::new(&m.cfg);
    let out = m.forward(&ids, &mut cache).to_vec().await;
    let got = &out[out.len() - m.cfg.n_vocab..];

    let (mut max_abs, mut argmax_got, mut argmax_ref) = (0f32, 0usize, 0usize);
    for i in 0..refl.len() {
        max_abs = max_abs.max((got[i] - refl[i]).abs());
        if got[i] > got[argmax_got] { argmax_got = i }
        if refl[i] > refl[argmax_ref] { argmax_ref = i }
    }
    // Correlation, because a uniform scale or offset would leave argmax intact and is exactly the
    // kind of error a wrong final norm produces.
    let n = refl.len() as f64;
    let (mg, mr) = (got.iter().map(|&v| v as f64).sum::<f64>() / n, refl.iter().map(|&v| v as f64).sum::<f64>() / n);
    let (mut cov, mut vg, mut vr) = (0f64, 0f64, 0f64);
    for i in 0..refl.len() {
        let (a, b) = (got[i] as f64 - mg, refl[i] as f64 - mr);
        cov += a * b; vg += a * a; vr += b * b;
    }
    let corr = cov / (vg.sqrt() * vr.sqrt());

    println!("\n  {} prompt tokens {ids:?}", ids.len());
    println!("  argmax  ferric {argmax_got}  ·  transformers {argmax_ref}");
    println!("  max |Δ| {max_abs:.4}   ·   correlation {corr:.6}");

    assert_eq!(argmax_got, argmax_ref, "different next token than transformers — the weights are \
                                        not where the runtime thinks they are");
    assert!(corr > 0.9999, "logits correlate at only {corr:.6} with transformers; argmax agreeing \
                            is not enough, a wrong final norm preserves it");
    assert!(max_abs < 0.05, "max |Δ| {max_abs:.4} against transformers is too large for f32 ordering");

    println!("\n  ✅ A checkpoint as published, run by a runtime written against GGUF, verified\n     \
              against the reference implementation the weights shipped with. No converter ran.");
}
