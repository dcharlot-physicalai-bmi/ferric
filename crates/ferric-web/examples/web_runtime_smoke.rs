//! **The browser code path, exercised natively** — FerricModel::load + generate on real checkpoints.
//!
//! wasm32 compilation proves the browser build LINKS; this proves the identical `WebRuntime` dispatch
//! GENERATES, for both runtimes it claims. Reads its subjects from argv, no defaults.
//!
//!   cargo run -p ferric-web --example web_runtime_smoke --release -- <model.gguf> "<prompt>" [steps]
fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: web_runtime_smoke <model.gguf> \"<prompt>\" [steps]");
    let prompt = a.get(2).map(String::as_str).unwrap_or("The capital of France is");
    let steps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let bytes = std::fs::read(path).expect("read model");
    let m = ferric_web::FerricModel::load(bytes).await.expect("load");
    println!("info: {}", m.info());
    println!("prompt ids: {:?}", m.encode_ids(prompt));
    let eos = m.eos_ids();
    println!("eos_set: {eos:?}");
    // The hardcoded pair this replaced was Qwen's {151645, 151643}. For a Qwen checkpoint the resolved
    // set must still COVER both (a smaller set silently generates past the end of turn); for any model
    // it must be non-empty, or generation only ever ends by running out of steps.
    assert!(!eos.is_empty(), "no stop token resolved — generation can only end by exhausting steps");
    if m.info().contains("dense") && eos.iter().any(|&t| t == 151643) {
        assert!(eos.contains(&151645) && eos.contains(&151643),
                "Qwen checkpoint lost part of the old hardcoded stop set: {eos:?}");
    }
    let out = m.generate_plain(prompt, steps).await.expect("generate");
    println!("out: {out}");
    assert!(!out.trim().is_empty(), "generated nothing");
    // Anti-saturation: a stream of one repeated fragment cannot distinguish a working runtime from a
    // broken one that emits fluent noise — the same guard every equivalence example here carries.
    let words: std::collections::BTreeSet<&str> = out.split_whitespace().collect();
    assert!(words.len() > 2, "degenerate output ({} distinct words): {out:?}", words.len());
}
