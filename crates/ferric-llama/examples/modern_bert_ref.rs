//! **ModernBERT vs llama.cpp**, on ids the reference itself produced.
//!
//!   cargo run -p ferric-llama --release --example modern_bert_ref -- <model.gguf> <id,id,id,...>
//!
//! Takes explicit token ids rather than text ON PURPOSE: it isolates the MODEL from the tokenizer,
//! so a divergence here cannot be blamed on pre-tokenization. `scripts/modern_bert_conformance.sh`
//! gets the ids from `llama-tokenize` and the target vector from `llama-embedding`.
//!
//! Prints the CLS row (position 0) both raw and L2-normalised — `llama-embedding` normalises by
//! default (`--embd-normalize 2`), so comparing raw against normalised is a units mistake that looks
//! like a small error.
use ferric_gguf::GgufFile;
use ferric_llama::modern_bert::ModernBert;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: modern_bert_ref <model.gguf> <comma-separated ids>");
    let ids: Vec<u32> = a.get(2).expect("token ids").split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("id")).collect();
    assert!(!ids.is_empty(), "give at least one token id");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    let m = ModernBert::load(&ctx, &g).expect("load modern-bert");
    let c = &m.cfg;
    eprintln!("layers={} d={} heads={} ff={} eps={:.3e}", c.n_layer, c.d, c.n_head, c.n_ff, c.eps);
    eprintln!("n_swa={} rope_base={} rope_base_swa={}", c.n_swa, c.rope_base, c.rope_base_swa);
    eprintln!("global layers: {:?}", c.swa.iter().enumerate().filter(|(_, s)| !**s).map(|(i, _)| i).collect::<Vec<_>>());
    eprintln!("labels: {:?}", c.labels);
    eprintln!("ids ({}): {:?}", ids.len(), ids);

    let h = m.forward(&ids).expect("forward");
    let v = h.to_vec().await;
    let d = c.d;
    let cls = &v[0..d];
    let n = cls.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    println!("RAW {}", cls.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" "));
    println!("NRM {}", cls.iter().map(|x| format!("{:.6}", x / n)).collect::<Vec<_>>().join(" "));
}
