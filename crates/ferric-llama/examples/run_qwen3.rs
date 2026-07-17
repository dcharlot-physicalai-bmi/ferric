//! Run a PrismML Ternary Bonsai **1.7B** (Qwen3 dense, Q2_0) natively and greedily continue a text
//! prompt. Coherent output is the correctness signal (a wrong forward produces token soup). This is
//! the same `Qwen3` code the browser path compiles to wasm.
//!
//!   cargo run -p ferric-llama --example run_qwen3 --release -- <model.gguf> "The capital of France is" 40
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// GPT-2 byte↔printable-unicode map, inverted — turn a vocab entry back into raw bytes for display.
fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new();
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: run_qwen3 <model.gguf> [prompt] [n]");
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("The capital of France is");
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);

    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let g = GgufFile::open(path).unwrap();

    // Build the exact BPE from the tokenizer embedded in the GGUF (tokens + merges).
    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m {
            s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string()))
        } else { None }).collect(),
        _ => panic!("no merges"),
    };
    let bpe = Bpe::new(vocab, &merges);
    let u2b = byte_decoder();
    let detok = |ids: &[u32]| -> String {
        let s: String = ids.iter().map(|&i| tokens.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };

    let m = Qwen3::load(&ctx, &g).unwrap();
    let c = &m.cfg;
    println!("loaded in {:?} · {} layers · d={} ff={} · {}h/{}kv × {} · vocab={}",
        t0.elapsed(), c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab);

    let ids = bpe.encode(prompt);

    // `--dump`: forward the prompt ONCE and print a deterministic fingerprint of the last-position
    // logits (top-8 + a fixed probe set + a sum). This is the native (Metal) reference for the
    // cross-fabric check — the browser (WebGPU) runs the SAME Qwen3 code and must reproduce it.
    if args.iter().any(|a| a == "--dump") {
        let v = m.forward_cached(&ids, &mut Cache::new(c)).to_vec().await;
        let row = &v[v.len() - c.n_vocab..];
        let mut idx: Vec<usize> = (0..c.n_vocab).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        println!("prompt ids: {ids:?}");
        println!("top-8 (id, logit):");
        for &i in idx.iter().take(8) { println!("  {i:>6}  {:+.5}  {:?}", row[i], detok(&[i as u32])); }
        let sum: f64 = row.iter().map(|&x| x as f64).sum();
        println!("probe[0,100,1000,10000]: {:+.5} {:+.5} {:+.5} {:+.5}", row[0], row[100], row[1000], row[10000]);
        println!("argmax={} sum={:.3}", idx[0], sum);
        return;
    }

    print!("\n{prompt}");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let mut cache = Cache::new(c);
    let mut seq = ids.clone();
    let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
    let t1 = std::time::Instant::now();
    let mut prompt_ms = 0.0;
    for step in 0..n {
        let logits = if step == 0 { m.forward_cached(&ids, &mut cache) } else { m.forward_cached(&seq[seq.len() - 1..], &mut cache) };
        let v = logits.to_vec().await;
        let next = argmax(&v[v.len() - c.n_vocab..]);
        if step == 0 { prompt_ms = t1.elapsed().as_secs_f64() * 1e3; }
        if step == 0 { ferric_tensor::prof_report(); } // prompt-step breakdown
        print!("{}", detok(&[next]));
        std::io::stdout().flush().ok();
        seq.push(next);
        if next == 151645 || next == 151643 { break; } // im_end / endoftext
    }
    ferric_tensor::prof_report(); // decode-step breakdown (accumulated)
    let decode_ms = (t1.elapsed().as_secs_f64() * 1e3 - prompt_ms) / (n - 1).max(1) as f64;
    println!("\n\n  prompt {:.0}ms · {:.0} ms/token", prompt_ms, decode_ms);
}
