//! Verify GPTQ activation capture: run one forward with capture on, confirm every linear's real input is grabbed.
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let g = GgufFile::open(format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf")).unwrap();
    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s)=m {s.clone()} else {String::new()}).collect(), _=>panic!() };
    let vocab: HashMap<String,u32> = toks.iter().enumerate().map(|(i,t)|(t.clone(),i as u32)).collect();
    let merges: Vec<(String,String)> = match g.metadata().get("tokenizer.ggml.merges") { Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s)=m {s.split_once(' ').map(|(x,y)|(x.to_string(),y.to_string()))} else {None}).collect(), _=>panic!() };
    let bpe = Bpe::new(vocab, &merges);
    let ids = bpe.encode("The quick brown fox jumps over the lazy dog and then runs away quickly.");
    let m = Qwen3::load(&ctx, &g).unwrap();
    m.set_capture(true);
    let _ = m.forward_cached(&ids, &mut Cache::new(&m.cfg)).to_vec().await;
    let cap = m.take_capture();
    println!("captured {} activations over {} tokens, {} layers", cap.len(), ids.len(), m.cfg.n_layer);
    for (name, t) in cap.iter().take(5) { println!("  {name:<14} shape {:?}", t.shape); }
    let kinds: std::collections::BTreeSet<_> = cap.iter().map(|(n,_)| n.split('.').nth(1).unwrap()).collect();
    println!("  linear kinds captured: {:?}", kinds);
    println!("  expected {} (= n_layer×4)", m.cfg.n_layer*4);
}
