//! **Reference-diff `FerricModel::embed` against llama.cpp.**
//!
//! Every retrieval number this project has published flows through `embed`, and until now nothing
//! compared it to an independent implementation. Sensible output is not correct output: this session
//! has already produced two embedding paths that returned ordinary-looking cosine scores and ranked
//! near-arbitrarily, and both were found by measurement rather than by reading.
//!
//!   llama-embedding -m M -p TEXT --embd-output-format array > ref.json
//!   cargo run -p ferric-web --example embed_reference --release -- M ref.json "TEXT"
//!
//! Cosine, not bit-equality, is the right test and the reason is measured: kernel selection is
//! capability-dependent, so even Ferric-vs-Ferric across fabrics differs at ulp order. The bar is
//! therefore what a DIFFERENT implementation of the same arithmetic should achieve — very close to
//! 1.0 — and the failure it must catch is a wrong pooling position or a wrong token sequence, both of
//! which land far below any reasonable threshold rather than just outside it.
fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (mp, rp, text) = (a.get(1).expect("model.gguf"), a.get(2).expect("ref.json"), a.get(3).expect("text"));

    let raw = std::fs::read_to_string(rp).expect("read reference");
    let nums: Vec<f32> = raw.trim().trim_start_matches('[').trim_end_matches(']')
        .trim().trim_start_matches('[').trim_end_matches(']')
        .split(',').filter_map(|s| s.trim().parse::<f32>().ok()).collect();
    assert!(!nums.is_empty(), "reference parsed to zero floats — check --embd-output-format array");

    let m = ferric_web::FerricModel::load(std::fs::read(mp).expect("read model")).await.expect("load");
    let ours = m.embed(text.clone()).await.expect("embed");

    println!("reference: {} dims   ferric: {} dims", nums.len(), ours.len());
    assert_eq!(nums.len(), ours.len(), "width disagrees — different pooling or a different model");

    // Both sides may or may not be L2-normalised; cosine is invariant to that, which is why it is the
    // comparison rather than a per-element diff.
    let dot: f32 = nums.iter().zip(&ours).map(|(a, b)| a * b).sum();
    let (na, nb): (f32, f32) = (nums.iter().map(|x| x * x).sum::<f32>().sqrt(),
                                ours.iter().map(|x| x * x).sum::<f32>().sqrt());
    let cos = dot / (na * nb);
    let maxdiff = nums.iter().map(|x| x / na).zip(ours.iter().map(|x| x / nb))
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("  cosine vs llama.cpp: {cos:.6}");
    println!("  max |Δ| after L2 normalising both: {maxdiff:.6}");
    println!("  reference norm {na:.4}, ferric norm {nb:.4}");

    assert!(cos > 0.99,
            "FERRIC'S EMBEDDING DISAGREES WITH llama.cpp (cosine {cos:.6}). Two implementations of \
             the same arithmetic on the same weights and the same text should be far closer than \
             this. The usual causes are a different pooling position, a different token sequence \
             (BOS/EOS handling), or a missing normalisation — all of which move the vector far more \
             than kernel-selection noise does. Every retrieval figure this project publishes flows \
             through this function.");
    println!("  ✅ agrees with an independent implementation");
}
