//! **LFM2 / LFM2.5 with real incremental state** — and proof it matches the naive path.
//!
//! `run_lfm2_gguf` re-runs the whole prefix for every token: correct, and O(n²), which makes a
//! 128k-context model's context a number on a spec sheet. This carries a KV cache for the 8 attention
//! blocks and a 2-timestep window for the 22 conv blocks.
//!
//! The check is not "it is faster" — it is that the cached path emits the SAME TOKEN IDS as the naive
//! one. A state bug (wrong rope offset, storing post-conv instead of pre-conv signal, a stale window)
//! produces fluent text that slowly diverges, which no speed measurement would catch.
//!
//!   cargo run -p ferric-llama --example run_lfm2_cached --release -- <lfm2.gguf> "prompt" [n]
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::lfm2::{Cache, Lfm2};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: run_lfm2_cached <model.gguf> [prompt] [n]");
    let prompt = a.get(2).map(|s| s.as_str()).unwrap_or("The capital of France is");
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(path).expect("open");
    let m = Lfm2::load(&ctx, &g).expect("load");
    let n_attn = m.cfg.kv.iter().filter(|&&k| k > 0).count();
    println!("LFM2 · {} blocks ({} conv + {n_attn} attn) · d={} · vocab={}",
             m.cfg.n_layer, m.cfg.n_layer - n_attn, m.cfg.d, m.cfg.n_vocab);

    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let vocab: std::collections::HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|x| if let Meta::Str(s) = x { s.split_once(' ').map(|(p, q)| (p.into(), q.into())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);
    let mut ids: Vec<u32> = bpe.encode(prompt);
    if let Some(Meta::U(bos)) = g.metadata.get("tokenizer.ggml.bos_token_id") {
        // ⚠ ABSENT is not FALSE. Neither LFM2.5 GGUF declares `add_bos_token`, while both declare
        // `bos_token_id` and their chat template opens with `{{- bos_token -}}`. Requiring the flag
        // to be explicitly true fed LFM2.5-8B-A1B no BOS, and it answered
        //   "The capital of France is France is France is ..."
        // With one — "the city of Paris." Ten hypotheses were eliminated inside the MoE (routing,
        // slab addressing, byte counts, expert compute vs a per-expert reference at 1.8e-8) before
        // the fault turned out to be here, in the harness. `ferric-serve` had it right all along:
        // `_ => bos_id.is_some()`. This now matches, because two policies for one question is how
        // the server and its own examples came to disagree.
        let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(Meta::Bool(b)) => *b, _ => true,
        };
        if add_bos { ids.insert(0, *bos as u32); }
    }
    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;

    // --- cached: prompt once, then one token at a time ---
    let t0 = std::time::Instant::now();
    let mut cache = Cache::new(&m.cfg);
    let mut next = am(&m.forward(&ids, &mut cache).to_vec().await);
    let t_prefill = t0.elapsed();
    let t0 = std::time::Instant::now();
    let mut cached: Vec<u32> = Vec::new();
    for _ in 0..n_gen {
        cached.push(next);
        next = am(&m.forward(&[next], &mut cache).to_vec().await);
    }
    let t_cached = t0.elapsed();

    // --- naive: re-run the whole prefix every step, no state carried ---
    let t0 = std::time::Instant::now();
    let mut seq = ids.clone();
    let mut naive: Vec<u32> = Vec::new();
    for _ in 0..n_gen {
        let mut c = Cache::new(&m.cfg);
        let nx = am(&m.forward(&seq, &mut c).to_vec().await);
        naive.push(nx);
        seq.push(nx);
    }
    let t_naive = t0.elapsed();

    let dec = |v: &[u32]| -> String {
        let mut mp = std::collections::HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
            let c = if p { b } else { let c = 256 + n; n += 1; c };
            mp.insert(char::from_u32(c).unwrap(), b as u8);
        }
        let s: String = v.iter().map(|&t| tokens[t as usize].clone()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| mp.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };
    println!("\n  cached: {prompt}{}", dec(&cached));
    println!("  naive : {prompt}{}", dec(&naive));
    println!("\n  prefill {t_prefill:.2?} · cached {n_gen} tok in {t_cached:.2?} ({:.1} ms/tok)",
             t_cached.as_secs_f64() * 1000.0 / n_gen as f64);
    println!("  naive  {n_gen} tok in {t_naive:.2?} ({:.1} ms/tok) · speedup {:.1}x",
             t_naive.as_secs_f64() * 1000.0 / n_gen as f64,
             t_naive.as_secs_f64() / t_cached.as_secs_f64().max(1e-9));

    let first_diff = cached.iter().zip(&naive).position(|(a, b)| a != b);
    match first_diff {
        None => println!("\n  ✅ identical token ids ({n_gen}/{n_gen}) — the state carries exactly what re-running recomputes"),
        Some(i) => {
            println!("\n  ❌ diverged at token {i}: cached {} ({:?}) vs naive {} ({:?})",
                     cached[i], tokens[cached[i] as usize], naive[i], tokens[naive[i] as usize]);
            std::process::exit(1);
        }
    }
}
