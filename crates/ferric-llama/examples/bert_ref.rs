//! **BERT / XLM-R on ids the model's AUTHORS produced** — the Ferric side of
//! `scripts/bert_conformance.sh`, compared against `tests/fixtures/bert/` (their `transformers`).
//!
//!   cargo run -p ferric-llama --release --example bert_ref -- <model.gguf> <id,id,...>
//!
//! Takes token ids rather than text ON PURPOSE: the fixture carries the authors' own tokenizer output,
//! so this isolates the MODEL from Ferric's tokenizer. Prints the first-token row and the mean of all
//! rows (raw, unnormalised) and, when the checkpoint carries a `cls.*` head, every logit it produces.
//!
//! bert.rs uses the exact erf GELU — the authors' `hidden_act: "gelu"`. `FERRIC_BERT_GELU_TANH=1`
//! selects ggml's tanh approximation instead, which is the conformance gate's negative control.
use ferric_gguf::GgufFile;
use ferric_llama::bert::Bert;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: bert_ref <model.gguf> <comma-separated ids>");
    let ids: Vec<u32> = a.get(2).expect("token ids").split(',')
        .filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().expect("id")).collect();
    assert!(!ids.is_empty(), "give at least one token id");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    let m = Bert::load(&ctx, &g).expect("load bert");
    let d = m.cfg.d;
    let v = m.forward(&ids).expect("forward").to_vec().await;
    let t = ids.len();
    println!("CLS {}", v[0..d].iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" "));
    let mean: Vec<f32> = (0..d).map(|j| (0..t).map(|r| v[r * d + j]).sum::<f32>() / t as f32).collect();
    println!("MEAN {}", mean.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" "));
    if m.n_cls_out().is_some() {
        let s = m.score_all(&ids).await.expect("score_all");
        println!("RANK {}", s.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" "));
    }
}
