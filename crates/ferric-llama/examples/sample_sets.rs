//! **How much of a logits row does exact sampling actually need?** Records real decode rows and measures,
//! per row and setting, the two sets an exact device/host split rests on:
//!   * R — elements that can change the host's sequential `Σ probs` (p_i ≥ ulp(prefix max)/2);
//!   * C — the sorted prefix nucleus/top-k/min-p reads.
//!
//!   cargo run -p ferric-llama --release --example sample_sets [model.gguf] [steps]
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let home = std::env::var("HOME").unwrap();
    let path = std::env::args().nth(1).unwrap_or_else(|| format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"));
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(&path).unwrap();
    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => panic!() };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") { Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => panic!() };
    let bpe = Bpe::new(vocab, &merges);
    let m = Qwen3::load(&ctx, &g).unwrap();
    let vn = m.cfg.n_vocab;
    let prompts = ["<|im_start|>user\nWrite a short poem about rain on a tin roof.<|im_end|>\n<|im_start|>assistant\n",
                   "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n<|im_start|>assistant\n",
                   "The lighthouse at Point Reyes was built in 1870 and"];
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for p in prompts {
        let ids = bpe.encode(p);
        let mut c = Cache::new(&m.cfg);
        let mut v = m.forward_cached_last(&ids, &mut c).to_vec().await;
        for _ in 0..steps {
            let row = v[v.len() - vn..].to_vec();
            let next = row.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;
            rows.push(row);
            v = m.forward_cached_last(&[next], &mut c).to_vec().await;
        }
    }
    if let Ok(out) = std::env::var("SAMPLE_SETS_DUMP") {
        let mut bytes = Vec::with_capacity(rows.len() * vn * 4);
        for r in &rows { for x in r { bytes.extend(x.to_le_bytes()); } }
        std::fs::write(&out, bytes).unwrap();
        println!("wrote {} rows x {vn} to {out}", rows.len());
    }
    println!("{} rows, vocab {vn}", rows.len());
    for &t in &[0.6f32, 0.8, 1.0, 1.5] {
        let (mut r_true, mut r_lb, mut c95, mut c90, mut k40, mut mp05) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for row in &rows {
            let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<f32> = row.iter().map(|&l| ((l - maxl) / t).exp()).collect();
            // true R: additions that changed the accumulator
            let (mut acc, mut n_true, mut n_lb, mut pmax) = (-0.0f32, 0usize, 0usize, 0.0f32);
            for &p in &probs {
                let half = if pmax < 2e-30 { 0.0 } else { f32::from_bits(((pmax.to_bits() >> 23) - 24) << 23) };
                if pmax < 2e-30 || p >= half { n_lb += 1; }
                let a2 = acc + p; if a2 != acc { n_true += 1; } acc = a2;
                pmax = pmax.max(p);
            }
            let sum = acc;
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let cut = |tp: f32| { let mut cum = 0f32; for (k, &i) in idx.iter().enumerate() { cum += probs[i] / sum; if cum >= tp { return k + 1; } } idx.len() };
            r_true.push(n_true); r_lb.push(n_lb); c95.push(cut(0.95)); c90.push(cut(0.9)); k40.push(40usize);
            mp05.push(probs.iter().filter(|&&p| p >= 0.05).count());
        }
        let st = |v: &mut Vec<usize>| { v.sort(); format!("med {:>6} p90 {:>6} max {:>6}", v[v.len() / 2], v[v.len() * 9 / 10], v[v.len() - 1]) };
        println!("T={t}: R_true {} | R_lb {} | C(top_p .95) {} | C(.90) {} | min_p .05 {}", st(&mut r_true), st(&mut r_lb), st(&mut c95), st(&mut c90), st(&mut mp05));
        let _ = k40;
    }
}
