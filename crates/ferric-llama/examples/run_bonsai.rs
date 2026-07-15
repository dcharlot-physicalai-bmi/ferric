//! Run **PrismML Ternary Bonsai-27B** (Qwen3.5 hybrid, Q2_0 ternary) on Ferric.
//!
//! Takes token IDs directly rather than a prompt string, so a comparison against PrismML's
//! llama.cpp fork isolates the *model* — a tokenizer discrepancy can't be mistaken for a
//! numerical one. Prints the top-k next-token distribution.
//!
//!   cargo run --release -p ferric-llama --example run_bonsai -- <model.gguf> <id,id,...> [--layers N]
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen35::Qwen35;
use std::sync::Arc;

/// GPT-2 byte↔printable-unicode bijection, inverted — turns a vocab entry back into raw bytes.
fn unicode_to_byte() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u16> = Vec::new();
    bs.extend(b'!' as u16..=b'~' as u16);
    bs.extend(0xA1u16..=0xAC);
    bs.extend(0xAEu16..=0xFF);
    let mut m = std::collections::HashMap::new();
    let mut extra = 0u32;
    for b in 0u16..256 {
        let c = if bs.contains(&b) { char::from_u32(b as u32).unwrap() } else { let c = char::from_u32(256 + extra).unwrap(); extra += 1; c };
        m.insert(c, b as u8);
    }
    m
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: run_bonsai <model.gguf> <id,id,...> [--layers N]");
    let ids: Vec<u32> = args.get(2).map(|s| s.split(',').map(|x| x.trim().parse().expect("bad token id")).collect()).unwrap_or_default();
    let layers = args.iter().position(|a| a == "--layers").map(|i| args[i + 1].parse::<usize>().unwrap());

    let t0 = std::time::Instant::now();
    let g = GgufFile::open(path).unwrap();
    println!("GGUF: {} tensors, opened in {:?}", g.tensors.len(), t0.elapsed());

    // vocab (for decoding the result only)
    let vocab: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    };
    let u2b = unicode_to_byte();
    let detok = |s: &str| -> String { String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned() };

    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let m = Qwen35::load(&ctx, &g).unwrap();
    let c = &m.cfg;
    let n_recr = (0..c.n_layer).filter(|&i| c.is_recurrent(i)).count();
    println!(
        "loaded in {:?}\n  {} layers ({} gated-delta-net + {} full-attn) · d={} · ff={} · vocab={}\n  GDN: {} k-heads × {} v-heads, head_k={} head_v={}, conv={} · attn: {}q/{}kv × {}",
        t0.elapsed(), c.n_layer, n_recr, c.n_layer - n_recr, c.n_embd, c.n_ff, c.n_vocab,
        c.n_k_heads, c.n_v_heads, c.head_k_dim, c.head_v_dim(), c.conv_kernel, c.n_head, c.n_head_kv, c.head_dim
    );

    if ids.is_empty() { println!("(no token ids given — load-only)"); return; }
    println!("\ntokens: {ids:?}");
    for &i in &ids { print!("{}", detok(vocab.get(i as usize).map(|s| s.as_str()).unwrap_or("?"))); }
    println!();

    let t0 = std::time::Instant::now();
    let logits = m.forward_upto(&ids, layers.unwrap_or(c.n_layer));
    let v = logits.to_vec().await;
    let dt = t0.elapsed();

    let last = &v[(ids.len() - 1) * c.n_vocab..];
    let finite = last.iter().all(|x| x.is_finite());
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("\nforward {:?} ({} tok){}", dt, ids.len(), if layers.is_some() { format!(" [first {} layers]", layers.unwrap()) } else { String::new() });
    println!("  finite: {finite}");
    println!("  top-8 next-token:");
    for &i in idx.iter().take(8) {
        println!("    {:>10.5}  [{:>6}] {:?}", last[i], i, detok(vocab.get(i).map(|s| s.as_str()).unwrap_or("?")));
    }

    // `--ref id=logit,id=logit,...` — compare against reference logits (PrismML's llama.cpp fork,
    // via examples/dumplogits) on the SAME token ids, so this is a pure model-vs-model check.
    if let Some(p) = args.iter().position(|a| a == "--ref") {
        let mut worst = 0.0f32;
        println!("\n  vs reference (PrismML llama.cpp fork):");
        for pair in args[p + 1].split(',') {
            let (id, r) = pair.split_once('=').expect("--ref wants id=logit");
            let (id, r): (usize, f32) = (id.parse().unwrap(), r.parse().unwrap());
            let d = (last[id] - r).abs();
            worst = worst.max(d);
            println!("    [{:>6}] ferric {:>10.5}   ref {:>10.5}   Δ {:.2e}  {:?}", id, last[id], r, d, detok(vocab.get(id).map(|s| s.as_str()).unwrap_or("?")));
        }
        // logits run to ~15; f32 GPU reductions over 64 layers in a different op order than ggml
        // will not be bit-identical, so agreement at ~1e-2 on this scale is the real bar.
        let pass = worst < 5e-2;
        println!("\n{} max|Δ| = {worst:.2e} across {} layers / 27B ternary params",
            if pass { "✅ Ferric matches the reference implementation" } else { "❌ MISMATCH" }, c.n_layer);
        assert!(pass, "logits diverge from reference: {worst:.2e}");
    }
    assert!(finite, "non-finite logits");

    // `--gen N` — greedy continuation. Re-prefills each step (no KV/recurrent-state cache yet), so
    // cost grows with the sequence; it demonstrates the model, not the decode speed.
    if let Some(p) = args.iter().position(|a| a == "--gen") {
        let n: usize = args[p + 1].parse().unwrap();
        let mut seq = ids.clone();
        print!("\ngenerating: ");
        for &i in &ids { print!("{}", detok(vocab.get(i as usize).map(|s| s.as_str()).unwrap_or("?"))); }
        use std::io::Write;
        std::io::stdout().flush().ok();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let v = m.forward(&seq).to_vec().await;
            let row = &v[(seq.len() - 1) * c.n_vocab..];
            let next = (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
            print!("{}", detok(vocab.get(next as usize).map(|s| s.as_str()).unwrap_or("?")));
            std::io::stdout().flush().ok();
            seq.push(next);
        }
        println!("\n  ({} tokens in {:?})", n, t0.elapsed());
    }
}
