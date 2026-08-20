//! **KV-cache quantization, end to end on a real model** — what it costs in accuracy and what it buys
//! in memory, measured rather than argued.
//!
//! `ferric_tensor::kvquant` ships the blocks; `qwen3::Cache::with_kvq` / `FERRIC_KVQ` is the switch that
//! puts them under the attention path. This example is the evidence for that switch.
//!
//! Reads its subject from argv, with no default.
//!
//!   cargo run -p ferric-llama --example kv_quant_wire --release -- <model.gguf> accuracy <n_gen>
//!   cargo run -p ferric-llama --example kv_quant_wire --release -- <model.gguf> memory <n_tokens>
//!
//! **accuracy** — teacher-forces a real passage **one token at a time**, which is the case that matters:
//! one row appended per step is the constraint that picked the block scheme, and a whole-sequence prefill
//! would never exercise it. Reports, against the f32 cache on the identical token sequence:
//!   * perplexity of each cache format on the passage (the headline accuracy number)
//!   * mean KL(P_f32 ‖ P_q) per position, in nats — the distribution-level divergence
//!   * max and mean |Δ logit| over every position × every vocabulary entry
//!   * top-1 agreement, and the step at which free greedy generation first diverges
//!
//! **memory** — grows a cache to `n_tokens` and reports the device bytes it actually occupies.
mod kvw {
    pub fn softmax_ln(row: &[f32]) -> Vec<f32> {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f64;
        for &x in row { s += ((x - m) as f64).exp(); }
        let ls = s.ln();
        row.iter().map(|&x| ((x - m) as f64 - ls) as f32).collect()
    }
}

use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tensor::kvquant::KvqFmt;
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

/// Resident set size of this process, in bytes. `ps` rather than a crate: no new dependency, and the
/// number is the one an operator would read.
fn rss_bytes() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout).trim().parse::<u64>().unwrap_or(0) * 1024
}

const PASSAGE: &str = "The Rhine rises in the Swiss Alps and flows north through Germany to the North Sea. \
Its valley has carried trade since the Roman period, and the river remains one of the busiest inland \
waterways in the world. Barges leaving Rotterdam reach Basel in about a week, moving coal, ore, chemicals \
and containers past Cologne, Koblenz and Mainz. In 2018 a drought dropped the water level at Kaub below \
thirty centimetres and the traffic stopped; the effect on German industrial output was measurable in the \
national accounts for that quarter. Engineers have since dredged sections of the channel and the fleet has \
begun to shift toward shallow-draft hulls, but the underlying exposure has not gone away: a river is a \
piece of infrastructure whose capacity is set by the weather.";

fn main() { pollster::block_on(run()); }

async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: kv_quant_wire <model.gguf> <accuracy|memory> <n>");
    let mode = args.get(2).expect("usage: kv_quant_wire <model.gguf> <accuracy|memory> <n>").as_str();
    let n: usize = args.get(3).map(|s| s.parse().expect("n")).expect("usage: ... <n>");

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
    let width = c.n_head_kv * c.head_dim;
    println!("model  {path}");
    println!("shape  {} layers · {} kv-heads × {} = {width}/row · vocab {}", c.n_layer, c.n_head_kv, c.head_dim, c.n_vocab);
    assert_eq!(width % 32, 0, "KV row width {width} is not a multiple of 32; QKvCache cannot store it");

    let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };

    match mode {
        "accuracy" => accuracy(&ctx, &m, &bpe, bos_id, add_bos, n).await,
        "memory" => memory(&ctx, &m, &bpe, bos_id, add_bos, n).await,
        other => panic!("unknown mode {other:?}; use `accuracy` or `memory`"),
    }
}

/// Teacher-force `ids` one token at a time and return every position's full logits row.
///
/// One token per `forward_cached` call, deliberately: that is one `QKvCache::append` of one row per
/// step, which is the append pattern the whole scheme is designed around. Passing the sequence in one
/// call would quantize it in a single shot and prove nothing about decode.
async fn teacher_force(m: &Qwen3, ids: &[u32], fmt: Option<KvqFmt>) -> (Vec<Vec<f32>>, Cache) {
    let c = &m.cfg;
    let mut cache = Cache::with_kvq(c, fmt);
    let mut rows = Vec::with_capacity(ids.len());
    for &t in ids {
        let v = m.forward_cached(&[t], &mut cache).to_vec().await;
        rows.push(v[v.len() - c.n_vocab..].to_vec());
    }
    (rows, cache)
}

/// Greedy continuation of `ids` for `n_gen` steps: prompt in one call, then one token per call.
async fn greedy(m: &Qwen3, ids: &[u32], fmt: Option<KvqFmt>, n_gen: usize) -> Vec<u32> {
    let c = &m.cfg;
    let mut cache = Cache::with_kvq(c, fmt);
    let mut seq = ids.to_vec();
    let mut out = Vec::with_capacity(n_gen);
    for step in 0..n_gen {
        let logits = if step == 0 { m.forward_cached(ids, &mut cache) }
                     else { m.forward_cached(&seq[seq.len() - 1..], &mut cache) };
        let v = logits.to_vec().await;
        let row = &v[v.len() - c.n_vocab..];
        let next = (0..c.n_vocab).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() as u32;
        seq.push(next);
        out.push(next);
    }
    out
}

async fn accuracy(ctx: &Arc<Context>, m: &Qwen3, bpe: &Bpe, bos: Option<u32>, add_bos: bool, n_gen: usize) {
    let _ = ctx;
    let c = &m.cfg;
    let mut ids = bpe.encode(PASSAGE);
    if add_bos { if let Some(b) = bos { ids.insert(0, b); } }
    println!("passage {} tokens · teacher-forced ONE TOKEN PER STEP (the append path)", ids.len());

    // Reference: the f32 cache, same code path a default run takes.
    let (ref_rows, ref_cache) = teacher_force(m, &ids, None).await;
    assert!(ref_cache.kvq_fmt().is_none(), "reference run must be the f32 cache");
    let ref_lp: Vec<Vec<f32>> = ref_rows.iter().map(|r| kvw::softmax_ln(r)).collect();
    // Perplexity over positions 1.. (position i predicts token i+1).
    let nll = |lp: &[Vec<f32>]| -> f64 {
        let mut s = 0.0f64;
        for i in 0..ids.len() - 1 { s -= lp[i][ids[i + 1] as usize] as f64; }
        s / (ids.len() - 1) as f64
    };
    let ref_nll = nll(&ref_lp);
    println!("\n{:<7} {:>10} {:>10} {:>11} {:>11} {:>11} {:>8} {:>9}",
             "fmt", "bits/val", "ppl", "mean KL", "max|Δlgt|", "mean|Δlgt|", "top1", "gen ==");
    println!("{:<7} {:>10} {:>10.4} {:>11} {:>11} {:>11} {:>8} {:>9}",
             "f32", 32.0, ref_nll.exp(), "0", "0", "0", "100.00%", "—");

    // Free greedy generation from the passage, per format, for a divergence step.
    let ref_gen = greedy(m, &ids, None, n_gen).await;

    for fmt in KvqFmt::ALL {
        let (rows, cache) = teacher_force(m, &ids, Some(fmt)).await;
        assert_eq!(cache.kvq_fmt(), Some(fmt), "cache did not take the requested format");
        let lp: Vec<Vec<f32>> = rows.iter().map(|r| kvw::softmax_ln(r)).collect();
        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f64;
        let mut kl = 0.0f64;
        let mut agree = 0usize;
        for i in 0..ids.len() {
            let (a, b) = (&ref_rows[i], &rows[i]);
            for j in 0..c.n_vocab {
                let d = (a[j] - b[j]).abs();
                if d > max_d { max_d = d; }
                sum_d += d as f64;
            }
            // KL(P_f32 ‖ P_q) = Σ P (ln P − ln Q). Positive by construction; a sanity check that a
            // sign slip cannot pass.
            let (pa, pb) = (&ref_lp[i], &lp[i]);
            let mut k = 0.0f64;
            for j in 0..c.n_vocab { k += (pa[j] as f64).exp() * (pa[j] - pb[j]) as f64; }
            assert!(k >= -1e-6, "KL went negative ({k}) at position {i} — the comparison is wrong, not the quantizer");
            kl += k;
            let ta = (0..c.n_vocab).max_by(|&x, &y| a[x].total_cmp(&a[y])).unwrap();
            let tb = (0..c.n_vocab).max_by(|&x, &y| b[x].total_cmp(&b[y])).unwrap();
            if ta == tb { agree += 1; }
        }
        let np = ids.len() as f64;
        let g = greedy(m, &ids, Some(fmt), n_gen).await;
        let first_div = ref_gen.iter().zip(&g).position(|(a, b)| a != b);
        println!("{:<7} {:>10.1} {:>10.4} {:>11.3e} {:>11.4} {:>11.2e} {:>7.2}% {:>9}",
                 fmt.name(), fmt.bits_per_value(), nll(&lp).exp(), kl / np, max_d,
                 sum_d / (np * c.n_vocab as f64),
                 100.0 * agree as f64 / np,
                 match first_div { None => format!("all {n_gen}"), Some(s) => format!("diverge@{s}") });
    }
    println!("\nppl is on the SAME token sequence for every row, so the columns are comparable directly.");
}

async fn memory(ctx: &Arc<Context>, m: &Qwen3, bpe: &Bpe, bos: Option<u32>, add_bos: bool, n_tokens: usize) {
    let _ = ctx;
    let c = &m.cfg;
    let mut base = bpe.encode(PASSAGE);
    if add_bos { if let Some(b) = bos { base.insert(0, b); } }
    let ids: Vec<u32> = base.iter().copied().cycle().take(n_tokens).collect();
    const CHUNK: usize = 256;
    println!("growing a {n_tokens}-token cache in {CHUNK}-token chunks\n");
    println!("{:<7} {:>12} {:>12} {:>9} {:>12} {:>12}", "fmt", "cache MB", "as f32 MB", "ratio", "RSS MB", "ΔRSS MB");
    let mut rows = Vec::new();
    for fmt in [None, Some(KvqFmt::Q8_0), Some(KvqFmt::Q4_0), Some(KvqFmt::Q4_1)] {
        let before = rss_bytes();
        let mut cache = Cache::with_kvq(c, fmt);
        for ch in ids.chunks(CHUNK) { let _ = m.forward_cached(ch, &mut cache).to_vec().await; }
        let (b, f) = (cache.kv_bytes(), cache.kv_f32_bytes());
        assert!(b > 0, "cache reports 0 bytes after {n_tokens} tokens — it is not being filled");
        let after = rss_bytes();
        let name = fmt.map(|f| f.name()).unwrap_or("f32");
        println!("{:<7} {:>12.1} {:>12.1} {:>8.2}x {:>12.1} {:>12.1}",
                 name, b as f64 / 1e6, f as f64 / 1e6, f as f64 / b as f64,
                 after as f64 / 1e6, (after.saturating_sub(before)) as f64 / 1e6);
        rows.push((name, b, f));
        drop(cache);
    }
    println!("\n`cache MB` is allocated code+scale words (capacity, incl. doubling slack); `as f32 MB` is\n\
              live rows × width × 4 (no slack). The ratio therefore UNDERSTATES the saving.");
    let f32b = rows[0].1;
    for (name, b, _) in &rows[1..] {
        println!("  {name}: {:.1} MB vs f32 {:.1} MB — {:.2}x more context per byte",
                 *b as f64 / 1e6, f32b as f64 / 1e6, f32b as f64 / *b as f64);
    }
}
