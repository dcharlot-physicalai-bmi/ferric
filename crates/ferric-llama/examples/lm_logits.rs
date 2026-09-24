//! **A dense LM's logits on ids the model's AUTHORS produced** — the Ferric side of
//! `scripts/lm_conformance.sh`, compared against `tests/fixtures/lm/` (their `transformers`).
//!
//!   cargo run -p ferric-llama --release --example lm_logits -- <model.gguf> <fixture.json>
//!
//! Reads the authors' token ids and their fixed vocabulary sample from the fixture, runs ONE
//! stateless prefill (`Qwen3::forward` for llama / qwen2 / qwen3 / gemma3, `Qwen35::forward` for the
//! gated-delta-net hybrids, `Lfm2::forward` from an empty cache for the short-conv hybrids —
//! dispatched on the file's `general.architecture`), and prints per
//! position: the sampled logits, the argmax, and the sum and sum-of-squares of the FULL row — so a
//! defect anywhere in the vocabulary moves a number even though only 128 ids are compared directly.
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::{lfm2::{self, Lfm2}, qwen3::Qwen3, qwen35::Qwen35};
use std::sync::Arc;

fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\": [");
    let i = s.find(&k).unwrap_or_else(|| panic!("fixture has no \"{key}\"")) + k.len();
    let j = s[i..].find(']').unwrap() + i;
    &s[i..j]
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: lm_logits <model.gguf> <fixture.json>");
    let fx = std::fs::read_to_string(a.get(2).expect("fixture.json")).expect("read fixture");
    let parse = |key: &str| -> Vec<u32> {
        field(&fx, key).split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect()
    };
    let ids = parse("ids");
    let sample = parse("sample_ids");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    // Dispatch on the file's own architecture: the gated-delta-net hybrids have their own runtime.
    let arch = match g.metadata().get("general.architecture") { Some(Meta::Str(a)) => a.clone(), _ => String::new() };
    eprintln!("arch: {arch}");
    let lg = if arch.starts_with("qwen35") {
        Qwen35::load(&ctx, &g).expect("load qwen35").forward(&ids).to_vec().await
    } else if arch.starts_with("lfm2") {
        let m = Lfm2::load(&ctx, &g).expect("load lfm2");
        let mut cache = lfm2::Cache::new(&m.cfg);
        m.forward(&ids, &mut cache).to_vec().await
    } else {
        Qwen3::load(&ctx, &g).expect("load").forward(&ids).to_vec().await
    };
    let v = lg.len() / ids.len();
    for t in 0..ids.len() {
        let r = &lg[t * v..(t + 1) * v];
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        let (mut sum, mut ssq) = (0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            sum += x as f64; ssq += (x as f64) * (x as f64);
        }
        let s: Vec<String> = sample.iter().map(|&i| format!("{:.5}", r[i as usize])).collect();
        println!("ROW {t} {best} {sum:.4} {ssq:.4} {}", s.join(" "));
    }
}
