//! **The runner half of `scripts/prefill_joules.sh`** — repeated prefills of one prompt length in
//! timed chunks separated by idle gaps, so a power meter sampling every 250 ms can charge each chunk
//! against the idle on either side of it (the `asr_joules.sh` method).
//!
//!   prefill_joules <model.gguf> <prompt_tokens> [chunks 4] [run_s 4] [idle_s 4]
//!
//! One prefill of 512 tokens on the tensor-unit route takes ~50 ms, far below the meter's sample
//! interval, so a chunk is as many back-to-back prefills (fresh cache each, last-row head, logits read
//! back — what a server's time-to-first-token pays) as fit in `run_s`. Prints `IDLE a b`,
//! `RUN a b tokens prefills` (UNIX seconds), then `SUMMARY`. The model loads and warms OUTSIDE every
//! window.
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn now() -> f64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: prefill_joules <model.gguf> <prompt_tokens> [chunks] [run_s] [idle_s]");
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let chunks: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let run_s: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let idle_s: f64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let g = ferric_gguf::GgufFile::open(path).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    // Same synthetic prompt as prefill_bench, so the two tools measure the same work.
    let prompt: Vec<u32> = (0..n as u32).map(|i| 100 + (i * 7919) % 20000).collect();
    let prefill = || async {
        let mut c = Cache::new(&m.cfg);
        m.forward_cached_last(&prompt, &mut c).to_vec().await
    };
    for _ in 0..2 { let _ = prefill().await; } // compile every pipeline, touch every weight
    eprintln!("{path}: {n}-token prefills, adapter {}, FERRIC_QGEMM={}", ctx.adapter_name,
              std::env::var("FERRIC_QGEMM").unwrap_or_default());
    let idle = || {
        let t0 = now();
        std::thread::sleep(Duration::from_secs_f64(idle_s));
        println!("IDLE {t0:.3} {:.3}", now());
    };
    idle();
    let (mut tot_tok, mut tot_s) = (0usize, 0f64);
    for _ in 0..chunks {
        let (t0, w0) = (now(), Instant::now());
        let mut reps = 0usize;
        while w0.elapsed().as_secs_f64() < run_s {
            let _ = prefill().await;
            reps += 1;
        }
        let dt = w0.elapsed().as_secs_f64();
        println!("RUN {t0:.3} {:.3} {} {reps}", now(), reps * n);
        tot_tok += reps * n;
        tot_s += dt;
        idle();
    }
    println!("SUMMARY tokens {tot_tok} seconds {tot_s:.3} tok_per_s {:.1}", tot_tok as f64 / tot_s);
}
