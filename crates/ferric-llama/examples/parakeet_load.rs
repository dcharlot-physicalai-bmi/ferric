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
    assert!(!m.pre_conv.is_empty(), "no pre-encode convs");
    // ⚠ `vocab` COUNTS THE BLANK; the token table does not. A CTC head emits `vocab` logits where the
    // last index is the blank — a control symbol with no piece string — so a 1025-wide head carries
    // 1024 pieces. Asserting equality failed on every CTC file; asserting nothing would have let a
    // genuinely truncated table through. Both accepted shapes are named instead.
    assert!(m.tokens.len() == m.cfg.vocab || m.tokens.len() + 1 == m.cfg.vocab,
            "token table is {} for a {}-wide head; expected {} (no blank piece) or {}",
            m.tokens.len(), m.cfg.vocab, m.cfg.vocab - 1, m.cfg.vocab);
    // ⚠ The decoder is OPTIONAL. A CTC export carries no predictor at all, so asserting on one
    // unconditionally fails the whole family — `rnnt` is an Option precisely because half the
    // shipped Parakeet files are CTC-only, and a file with NEITHER head is the real defect.
    let lstms = match &m.rnnt {
        Some((pred, _)) => {
            assert_eq!(pred.lstm.len(), m.cfg.pred_layers, "lstm layer count");
            pred.lstm.len()
        }
        None => { assert!(m.ctc_head.is_some(), "file has neither an RNN-T decoder nor a CTC head"); 0 }
    };
    println!("  all {} blocks, {lstms} lstm layers, {} tokens — no missing tensor",
             m.blocks.len(), m.tokens.len());
}
