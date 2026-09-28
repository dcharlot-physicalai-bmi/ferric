//! **A greedy continuation through the Dense runtime's public entry points** — the same calls on
//! either fabric (`FERRIC_CPU=1` selects the CPU at load), so two runs can be diffed token for token.
//!
//!   cargo run -p ferric-llama --release --example greedy_ids -- <model.gguf> <fixture.json> [prompt] [gen]
//!
//! Prompt: the first `prompt` (default 32) ids of an `lm_logits` fixture (the authors' tokenization).
//! Then `gen` (default 64) greedy steps, each through `forward_cached` with one token. Prints one line
//! per step: `STEP i token margin`, where `margin` is the top-1 minus top-2 logit — so a divergence
//! between fabrics can be judged against how close the choice was.
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\": [");
    let i = s.find(&k).unwrap_or_else(|| panic!("fixture has no \"{key}\"")) + k.len();
    let j = s[i..].find(']').unwrap() + i;
    &s[i..j]
}

fn top2(r: &[f32]) -> (u32, f32) {
    let (mut b, mut bv, mut sv) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &x) in r.iter().enumerate() {
        if x > bv { sv = bv; bv = x; b = i; } else if x > sv { sv = x; }
    }
    (b as u32, bv - sv)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: greedy_ids <model.gguf> <fixture.json> [prompt] [gen]");
    let fx = std::fs::read_to_string(a.get(2).expect("fixture.json")).expect("read fixture");
    let np: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(32);
    let ng: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(64);
    let ids: Vec<u32> = field(&fx, "ids").split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect();
    let prompt = &ids[..np.min(ids.len())];
    let ctx = Arc::new(pollster::block_on(ferric_core::Context::new()).expect("context"));
    let g = GgufFile::open(mp).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    eprintln!("fabric: {}", if m.on_cpu() { "CPU" } else { "GPU" });
    let mut c = Cache::new(&m.cfg);
    let lg = pollster::block_on(m.forward_cached_last(prompt, &mut c).to_vec());
    let (mut next, mut margin) = top2(&lg);
    for i in 0..ng {
        println!("STEP {i} {next} {margin:.4}");
        let lg = pollster::block_on(m.forward_cached(&[next], &mut c).to_vec());
        (next, margin) = top2(&lg);
    }
}
