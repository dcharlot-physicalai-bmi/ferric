//! **Rank documents against a query with a cross-encoder.**
//!   cargo run -p ferric-llama --example rerank --release -- <reranker.gguf> "query" "doc1" "doc2" ...
use ferric_gguf::GgufFile;
use ferric_llama::bert::Reranker;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: rerank <reranker.gguf> <query> <doc>...");
    let q = a.get(2).expect("query");
    let docs: Vec<String> = a[3..].to_vec();
    assert!(!docs.is_empty(), "give at least one document");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    let rr = Reranker::load(&ctx, &g).expect("load reranker");
    let ranked = rr.rank(q, &docs).await.expect("rank");
    println!("query: {q}");
    for (rank, (i, score)) in ranked.iter().enumerate() {
        println!("  {}. [{:+.4}] doc {i}: {}", rank + 1, score, &docs[*i][..docs[*i].len().min(64)]);
    }
}
