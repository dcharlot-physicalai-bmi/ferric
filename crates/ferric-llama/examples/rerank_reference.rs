//! **Cross-encoder reranking, diffed against llama.cpp.**
//!   llama-embedding -m rr.gguf --pooling rank -p "query\tpassage" --embd-output-format array
//!   cargo run -p ferric-llama --example rerank_reference --release -- rr.gguf "query" "passage"
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::bert::Bert;
use ferric_tokenizer::Spm;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (mp, q, p) = (a.get(1).expect("model"), a.get(2).expect("query"), a.get(3).expect("passage"));
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let scores: Vec<f32> = match g.metadata.get("tokenizer.ggml.scores") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::F(f) = x { *f as f32 } else { 0.0 }).collect(),
        _ => Vec::new(),
    };
    let u = |k: &str, d: u32| match g.metadata.get(k) { Some(Meta::U(v)) => *v as u32, _ => d };
    let (cls, sep) = (u("tokenizer.ggml.cls_token_id", 0), u("tokenizer.ggml.seperator_token_id", 2));
    let spm = Spm::new(toks, scores);

    let m = Bert::load(&ctx, &g).expect("load");
    println!("bert {} layers · d={} · reranker={}", m.cfg.n_layer, m.cfg.d, m.is_reranker());
    assert!(m.is_reranker(), "this checkpoint has no cls.* head");

    // **BOS query EOS doc EOS** — ONE separator, no second BOS, and this was deduced from the
    // reference's own token count rather than guessed. llama-server reported 39 prompt_tokens for two
    // pairs whose content is 6 query tokens and 10/11 doc tokens; the three candidate formats give
    // 41, 41 and 39. Only the last fits, and it is the one that also happens to be right.
    //
    // Two wrong formats were tried first. Their scores stayed plausible and kept the correct ORDERING
    // both times, which is exactly why a rank bench would not have caught either.
    let _ = (cls, sep);
    let bos = u("tokenizer.ggml.bos_token_id", 0);
    let eos = u("tokenizer.ggml.eos_token_id", 2);
    let mut ids = vec![bos];
    ids.extend(spm.encode_piece(q, true));
    ids.push(eos);
    ids.extend(spm.encode_piece(p, true));
    ids.push(eos);
    // Print the ids, because a logit that is close but wrong cannot say WHY and these can. The
    // reference is `llama-tokenize -m M -p TEXT --ids` on each half.
    println!("  {} tokens: {:?}", ids.len(), ids);
    let s = m.score(&ids).await.expect("score");
    println!("  ferric logit: {s:.6}");
}
