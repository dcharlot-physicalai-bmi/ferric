//! **Does the parakeet loader find every weight it needs?**
//!
//! Milestone one of speech support. It transcribes nothing yet — it proves the 961-tensor
//! inventory maps onto a struct with no missing name and no guessed shape, which is the part that
//! silently poisons everything downstream if it is wrong.
use ferric_gguf::GgufFile;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let path = std::env::args().nth(1).expect("usage: parakeet_load <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let m = ferric_llama::parakeet::Parakeet::load(&ctx, &g).expect("load parakeet");
    println!("{}", m.describe());
    println!("  pre_encode convs: {}", m.pre_conv.len());
    // Every count is asserted, not printed and eyeballed: a loader that silently built 0 blocks
    // would otherwise "succeed".
    assert_eq!(m.blocks.len(), m.cfg.n_layers, "block count");
    assert_eq!(m.pred.lstm.len(), m.cfg.pred_layers, "lstm layer count");
    assert!(!m.pre_conv.is_empty(), "no pre-encode convs");
    assert_eq!(m.tokens.len(), m.cfg.vocab, "vocab");
    println!("  all {} blocks, {} lstm layers, {} tokens — no missing tensor",
             m.blocks.len(), m.pred.lstm.len(), m.tokens.len());
}
