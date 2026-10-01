//! **Decode tokens/s and joules per token for PrismML Bonsai 2** — the Ferric runner of
//! `scripts/bonsai2_decode_joules.sh`, speaking the IDLE/RUN/SUMMARY protocol of `prefill_joules`.
//!
//! The model loads and warms OUTSIDE every window. Each chunk: reset the cache and prefill the prompt
//! (untimed), sleep `idle_s` (an IDLE window), then decode `k` tokens greedily with a full-logits
//! readback per step (a RUN window) — exactly what the fork's runner (`refgen/bonsai2_decode_joules.cpp`)
//! does through `llama_decode` + `llama_get_logits_ith`, so the two windows hold the same work. Every
//! chunk's generated ids are hashed (`HASH`): a power trace over wrong tokens has no denominator.
//!
//!   cargo run --release -p ferric-llama --example bonsai2_decode_joules -- <model.gguf> <id,id,...> \
//!       [chunks 5] [k 128] [idle_s 4]
//!
//! `FERRIC_STREAM_GIB=<GiB>` runs it with layer streaming (`Qwen35::load_streaming`).
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen35::{Cache, Qwen35};
use std::sync::Arc;

fn now() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() }
fn argmax(v: &[f32]) -> u32 { let mut b = 0; for (i, x) in v.iter().enumerate() { if *x > v[b] { b = i; } } b as u32 }

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let ids: Vec<u32> = a[2].split(',').map(|x| x.trim().parse().unwrap()).collect();
    let arg = |i: usize, d: f64| a.get(i).map(|s| s.parse().unwrap()).unwrap_or(d);
    let (chunks, k, idle) = (arg(3, 5.0) as usize, arg(4, 128.0) as usize, arg(5, 4.0));
    let ctx = Arc::new(Context::new().await.expect("GPU"));
    eprintln!("adapter: {} ({:?})", ctx.adapter_name, ctx.backend);
    // FERRIC_STREAM_GIB=<device GiB for layer weights>: stream the blocks beyond that budget.
    let m = match std::env::var("FERRIC_STREAM_GIB").ok().and_then(|v| v.parse::<f64>().ok()) {
        Some(gib) => Qwen35::load_streaming(&ctx, &a[1], (gib * (1u64 << 30) as f64) as u64),
        None => Qwen35::load(&ctx, &GgufFile::open(&a[1]).expect("open")),
    }.unwrap_or_else(|e| panic!("load: {e}"));
    if let Some(s) = &m.stream {
        eprintln!("streaming: {}/{} blocks resident, {:.2} GiB re-read per pass", s.npin, m.cfg.n_layer,
                  s.streamed_bytes_per_pass as f64 / (1u64 << 30) as f64);
    }
    let nl = m.cfg.n_layer;
    // One decode run: prefill (untimed by the caller), then `k` greedy steps with readback.
    let decode = |cache: &mut Cache, first: u32, k: usize| -> Vec<u32> {
        let mut out = vec![first];
        let mut next = first;
        for _ in 1..k {
            let row = pollster::block_on(m.forward_cached(&[next], cache, nl).to_vec());
            next = argmax(&row);
            out.push(next);
        }
        out
    };
    let prefill = |cache: &mut Cache| -> u32 {
        let lg = pollster::block_on(m.forward_cached(&ids, cache, nl).to_vec());
        argmax(&lg[(ids.len() - 1) * m.cfg.n_vocab..])
    };
    { let mut c = Cache::new(&m.cfg); let f = prefill(&mut c); decode(&mut c, f, 16); } // warm-up
    let (mut tok, mut secs) = (0usize, 0.0f64);
    for _ in 0..chunks {
        let mut c = Cache::new(&m.cfg);
        let first = prefill(&mut c);
        let t0 = now(); std::thread::sleep(std::time::Duration::from_secs_f64(idle)); println!("IDLE {t0:.3} {:.3}", now());
        let t0 = now();
        let out = decode(&mut c, first, k);
        let t1 = now();
        println!("RUN {t0:.3} {t1:.3} {} 1", out.len());
        let h = out.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
        eprintln!("HASH {h:016x} {} tokens {:.1} ms/token", out.len(), (t1 - t0) * 1e3 / out.len() as f64);
        tok += out.len(); secs += t1 - t0;
    }
    let t0 = now(); std::thread::sleep(std::time::Duration::from_secs_f64(idle)); println!("IDLE {t0:.3} {:.3}", now());
    println!("SUMMARY tokens {tok} seconds {secs:.3} tok_per_s {:.2}", tok as f64 / secs);
}
