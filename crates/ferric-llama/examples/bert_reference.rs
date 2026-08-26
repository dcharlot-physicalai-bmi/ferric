//! **Reference-diff the BERT encoder against llama.cpp**, end to end: WordPiece → forward → pool.
//!   llama-embedding -m bge.gguf -p TEXT --embd-output-format array > ref.json
//!   cargo run -p ferric-llama --example bert_reference --release -- bge.gguf ref.json "TEXT"
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::bert::Bert;
use ferric_tokenizer::WordPiece;
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (mp, rp, text) = (a.get(1).expect("model"), a.get(2).expect("ref.json"), a.get(3).expect("text"));
    let raw = std::fs::read_to_string(rp).expect("read reference");
    let want: Vec<f32> = raw.trim().trim_matches(|c| c == '[' || c == ']')
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert!(!want.is_empty(), "reference parsed to zero floats");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let u = |k: &str, d: u32| match g.metadata.get(k) { Some(Meta::U(v)) => *v as u32, _ => d };
    let wp = WordPiece::new(toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect::<HashMap<_, _>>(),
                            u("tokenizer.ggml.cls_token_id", 101),
                            u("tokenizer.ggml.seperator_token_id", 102),
                            u("tokenizer.ggml.unknown_token_id", 100));
    let ids = wp.encode(text);
    let m = Bert::load(&ctx, &g).expect("load bert");
    println!("bert {} layers · d={} · heads={} · pooling={} · causal={}",
             m.cfg.n_layer, m.cfg.d, m.cfg.n_head, m.cfg.pooling, m.cfg.causal);
    println!("  {} tokens", ids.len());

    let h = m.forward(&ids).expect("forward").to_vec().await;
    let (t, d) = (ids.len(), m.cfg.d);
    // Pool where the checkpoint says: 2 = CLS (index 0), 1 = MEAN over tokens.
    // FERRIC_BERT_POOLER=1 applies BERT's pooler — tanh(dense(CLS)) — instead of returning the raw
    // CLS state. Which of the two a reference tool means by "cls pooling" is genuinely ambiguous for a
    // checkpoint that HAS the pooler tensors, and comparing the wrong one blames the encoder for a
    // difference that is entirely in the last two ops.
    let use_pooler = std::env::var("FERRIC_BERT_POOLER").ok().as_deref() == Some("1");
    let pooled: Vec<f32> = if use_pooler {
        let hs = ferric_tensor::Tensor::from_vec(&ctx, &h, &[t, d]);
        m.pooler(&hs).await.expect("pooler")
    } else {
        match m.cfg.pooling {
            1 => (0..d).map(|c| (0..t).map(|r| h[r * d + c]).sum::<f32>() / t as f32).collect(),
            _ => h[0..d].to_vec(),
        }
    };
    let dot: f32 = pooled.iter().zip(&want).map(|(a, b)| a * b).sum();
    let (na, nb): (f32, f32) = (pooled.iter().map(|x| x * x).sum::<f32>().sqrt(),
                                want.iter().map(|x| x * x).sum::<f32>().sqrt());
    let cos = dot / (na * nb);
    println!("  reference {} dims, ferric {} dims", want.len(), pooled.len());
    println!("  cosine vs llama.cpp: {cos:.6}");
    assert_eq!(pooled.len(), want.len(), "width differs — wrong pooling or a different model");
    assert!(cos > 0.99,
            "BERT ENCODER DISAGREES WITH llama.cpp (cosine {cos:.6}). Candidate causes, in the order \
             they are worth checking: a causal mask left on (each token would see only its left \
             context), the wrong pooling position, post-norm applied as pre-norm, or a missing bias.");
    println!("  ✅ agrees with an independent implementation");
}
