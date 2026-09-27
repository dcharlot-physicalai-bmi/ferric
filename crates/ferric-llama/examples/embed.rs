//! **Embed text with a BERT-family encoder, the way a server would** — tokenizer, declared pooling,
//! L2 normalisation.
//!
//!   embed <model.gguf> <text> [<text> ...]
//!
//! Prints `IDS` (the checkpoint's own tokenisation) and `EMB` for each text.
use ferric_llama::bert::Embedder;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() >= 3, "usage: embed <model.gguf> <text>...");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let e = Embedder::load(&ctx, &ferric_gguf::GgufFile::open(&a[1]).expect("open")).expect("load");
    for t in &a[2..] {
        println!("IDS {}", e.encode(t).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        let (v, n) = e.embed(t, false).await.expect("embed");
        println!("EMB {n} {}", v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" "));
    }
}
