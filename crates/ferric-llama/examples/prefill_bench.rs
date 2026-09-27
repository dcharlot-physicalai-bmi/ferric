//! **Prefill and decode throughput for a dense GGUF** — the tokens-per-second a server's time-to-first-token
//! and streaming rate come from, measured the same way for any file.
//!
//!   prefill_bench <model.gguf> [prompt_tokens 1024] [decode_tokens 64]
//!
//! Warms the pipelines, then times one prefill of N tokens and a greedy decode, reading the logits back
//! each step as a server does. Wall clock, one run: a throughput estimate, not an energy measurement.
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: prefill_bench <model.gguf> [prompt_tokens] [decode_tokens]");
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let d: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let g = ferric_gguf::GgufFile::open(path).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    let prompt: Vec<u32> = (0..n as u32).map(|i| 100 + (i * 7919) % 20000).collect();
    // warm-up: compile every pipeline both phases use
    { let mut c = Cache::new(&m.cfg); let _ = m.forward_cached(&prompt[..32.min(n)], &mut c).to_vec().await;
      let _ = m.forward_cached(&[5], &mut c).to_vec().await; }
    // The last-row head must give EXACTLY the last row of the full head, or it is a different model.
    let full = { let mut c = Cache::new(&m.cfg); m.forward_cached(&prompt[..64.min(n)], &mut c).to_vec().await };
    let last = { let mut c = Cache::new(&m.cfg); m.forward_cached_last(&prompt[..64.min(n)], &mut c).to_vec().await };
    let nv0 = m.cfg.n_vocab;
    let fr = &full[full.len() - nv0..];
    let md = fr.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = fr.iter().map(|a| a.abs()).fold(0f32, f32::max);
    let am = |r: &[f32]| (0..r.len()).max_by(|&x, &y| r[x].partial_cmp(&r[y]).unwrap()).unwrap();
    println!("  last-row head vs the full head's last row: max|diff| {md:.2e} of max|logit| {scale:.1}; argmax {} vs {}; bit-identical {}",
             am(fr), am(&last), fr == &last[..]);
    // PREFILL_REPS=n times n fresh prefills and reports the median (a single wall-clock sample on a
    // shared machine is one draw from a wide distribution); the last one's cache feeds the decode.
    let reps: usize = std::env::var("PREFILL_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let mut times = Vec::with_capacity(reps);
    let (mut c, mut v) = (Cache::new(&m.cfg), Vec::new());
    for _ in 0..reps {
        c = Cache::new(&m.cfg);
        let t0 = Instant::now();
        v = if std::env::var("FULL_HEAD").is_ok() { m.forward_cached(&prompt, &mut c).to_vec().await } else { m.forward_cached_last(&prompt, &mut c).to_vec().await };
        times.push(t0.elapsed().as_secs_f64());
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tp = sorted[reps / 2];
    if reps > 1 {
        println!("  prefill {n} tokens x{reps}: min {:.1} / median {:.1} / max {:.1} ms = {:.0} / {:.0} / {:.0} tok/s",
                 sorted[0] * 1e3, tp * 1e3, sorted[reps - 1] * 1e3,
                 n as f64 / sorted[0], n as f64 / tp, n as f64 / sorted[reps - 1]);
    }
    ferric_tensor::prof_report(); // FERRIC_PROFILE=1: where the prefill went, by category
    let nv = m.cfg.n_vocab;
    let mut tok = (0..nv).max_by(|&x, &y| v[v.len() - nv + x].partial_cmp(&v[v.len() - nv + y]).unwrap()).unwrap() as u32;
    let t1 = Instant::now();
    for _ in 0..d {
        let v = m.forward_cached(&[tok], &mut c).to_vec().await;
        tok = (0..nv).max_by(|&x, &y| v[x].partial_cmp(&v[y]).unwrap()).unwrap() as u32;
    }
    let td = t1.elapsed().as_secs_f64();
    ferric_tensor::prof_report(); // …and the decode
    println!("{path}: {} layers, adapter {}", m.cfg.n_layer, ctx.adapter_name);
    println!("  prefill {n} tokens  {tp:.3} s  = {:.0} tok/s", n as f64 / tp);
    println!("  decode  {d} tokens   {td:.3} s  = {:.1} tok/s", d as f64 / td);
}
