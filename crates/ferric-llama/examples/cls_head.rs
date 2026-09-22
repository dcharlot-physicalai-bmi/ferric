//! **The classifier head's FULL output vector** — and the regression that the scalar `score()` still
//! returns exactly what it used to.
//!
//!   cargo run -p ferric-llama --example cls_head --release -- <reranker-or-classifier.gguf> "query" "doc"
//!
//! ⛔⛔ WHY THIS EXISTS. `Bert::score` computed the whole `cls.output` vector and returned
//! `out.first()`. For a cross-encoder reranker that is correct — `n_cls_out` is 1. For any other
//! checkpoint carrying the same `cls.*` tensors (guardrails, moderation taxonomies, NLI) it returned
//! the FIRST class's logit as though it were "the score", with every other class silently dropped.
//! Nothing errored. The number was real; it answered a question nobody asked.
//!
//! ⚠⚠ WHAT THIS EXAMPLE CAN AND CANNOT PROVE, STATED UP FRONT. Run against a k=1 checkpoint such as
//! `bge-reranker-v2-m3` it proves exactly one thing: `score()` is bit-identical to `score_all()[0]`,
//! so the refactor changed no existing answer. **It CANNOT prove the k-way behaviour**, because with
//! one class the old code and the new code are indistinguishable by construction. A k>1 checkpoint
//! is required for that, and this review did not locate one in GGUF form on this machine — so the
//! k-way path is IMPLEMENTED AND UNVERIFIED until one is converted. Do not read a green run here as
//! evidence for multi-class output.
use ferric_gguf::GgufFile;
use ferric_llama::bert::{Bert, Reranker};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: cls_head <model.gguf> <query> <doc>");
    let q = a.get(2).map(String::as_str).unwrap_or("what is panda?");
    let doc = a.get(3).map(String::as_str)
        .unwrap_or("The giant panda is a bear species endemic to China.");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");

    let bert = Bert::load(&ctx, &g).expect("load bert");
    let k = bert.n_cls_out();
    println!("checkpoint : {mp}");
    println!("n_cls_out  : {k:?}   (None = no cls.* head; this file embeds but cannot classify)");
    if k.is_none() {
        println!("\nthis checkpoint has no classifier head — nothing further to check here.");
        return;
    }

    // Build the pair exactly as the reranker does, so the two calls below see identical ids.
    let rr = Reranker::load(&ctx, &g).expect("load reranker");
    let ids = rr.pair(q, doc);

    let all = bert.score_all(&ids).await.expect("score_all");
    let one = bert.score(&ids).await.expect("score");

    println!("\nfull head output ({} logit{}):", all.len(), if all.len() == 1 { "" } else { "s" });
    for (i, v) in all.iter().enumerate() { println!("  [{i}] {v:+.6}"); }
    println!("\nscore()       : {one:+.6}");
    println!("score_all()[0]: {:+.6}", all[0]);

    // ⭐ THE REGRESSION. Bit-identical, not approximately equal — this is the same arithmetic reached
    // through one more function call, so any difference at all means the refactor moved a number.
    assert_eq!(one.to_bits(), all[0].to_bits(),
               "score() must be BIT-IDENTICAL to score_all()[0]; it is the same matmul reached through \
                one extra call. A difference here means the refactor changed an answer.");

    // ⭐ And the width must come from the tensor, not from a guess.
    assert_eq!(Some(all.len()), k,
               "n_cls_out() reads cls.output.weight's leading dim; it MUST equal the length of the \
                vector the matmul actually produced, or one of the two is reading the wrong axis.");

    match k {
        Some(1) => println!("\n✅ k=1: score() unchanged, bit-identical. \
                             ⚠ the k-way path is NOT exercised by this checkpoint."),
        Some(n) => println!("\n✅ k={n}: all {n} logits returned where the old API returned 1."),
        None => unreachable!(),
    }
}
