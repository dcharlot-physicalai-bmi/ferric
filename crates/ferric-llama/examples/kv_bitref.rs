//! **Bit-exact decode fingerprint** — the regression that guards every change to the KV cache path.
//!
//! Prefill a prompt, then greedily decode `n` tokens, and print for every step a FNV-1a-64 hash of the
//! **raw bit patterns of the whole logits row** plus the chosen id. Nothing is rounded on the way out:
//! a one-ulp change anywhere in attention moves the hash.
//!
//! This exists to be run BEFORE and AFTER a change to `qwen3::Cache` / `qwen3::attn` with the new
//! feature switched off. Identical output is the claim "the default path is untouched", and it is a
//! claim a `{:.4}` print cannot make.
//!
//! Reads its subject from argv, with no default — a hardcoded model path hid a live divergence in
//! this tree for a day.
//!
//!   cargo run -p ferric-llama --example kv_bitref --release -- <model.gguf> "<prompt>" <n>
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// FNV-1a over the little-endian bit patterns of `xs`. Chosen over a float checksum because summing
/// f32s reorders error and would hide exactly the one-ulp differences this exists to catch.
fn fnv_bits(xs: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: kv_bitref <model.gguf> <prompt> <n>");
    let prompt = args.get(2).expect("usage: kv_bitref <model.gguf> <prompt> <n>");
    let n: usize = args.get(3).expect("usage: kv_bitref <model.gguf> <prompt> <n>").parse().expect("n");

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(path).unwrap();

    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens in {path}"),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m {
            s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string()))
        } else { None }).collect(),
        _ => panic!("no merges in {path}"),
    };
    let bpe = Bpe::new(vocab, &merges);

    let m = Qwen3::load(&ctx, &g).unwrap();
    let c = &m.cfg;
    let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
    let mut ids = bpe.encode(prompt);
    if add_bos { if let Some(b) = bos_id { ids.insert(0, b); } }

    println!("# model={path}");
    println!("# layers={} n_head_kv={} head_dim={} vocab={}", c.n_layer, c.n_head_kv, c.head_dim, c.n_vocab);
    println!("# prompt_ids={ids:?}");

    let mut cache = Cache::new(c);
    let mut seq = ids.clone();
    let mut hist: u64 = 0xcbf2_9ce4_8422_2325;
    for step in 0..n {
        let logits = if step == 0 { m.forward_cached(&ids, &mut cache) }
                     else { m.forward_cached(&seq[seq.len() - 1..], &mut cache) };
        let v = logits.to_vec().await;
        let row = &v[v.len() - c.n_vocab..];
        let h = fnv_bits(row);
        let next = (0..c.n_vocab).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() as u32;
        // Fold every step's hash into a running one so a divergence at ANY step survives to the end.
        hist ^= h;
        hist = hist.wrapping_mul(0x0000_0100_0000_01b3);
        println!("step {step:>4} id {next:>6} logits_fnv {h:016x} argmax_bits {:08x}", row[next as usize].to_bits());
        seq.push(next);
    }
    println!("TOTAL_FNV {hist:016x}");
    println!("TOTAL_IDS {:?}", &seq[ids.len()..]);
}
