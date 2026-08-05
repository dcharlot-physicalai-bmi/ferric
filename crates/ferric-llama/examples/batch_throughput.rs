//! **Does concurrency buy throughput?** Aggregate tokens/sec against the number of live sequences.
//!
//! Profiling decode found no hot function: 69% of main-thread time is blocked, and a step is
//! ~10.8 ms/token = 5.8 ms host + 5.0 ms device, **fully serialised**. Single-sequence greedy decode has
//! a hard dependency (token *N+1* needs token *N*), so the host cannot run ahead and the device idles
//! while commands are built.
//!
//! The standard escape is concurrency: with many live sequences, one host build should serve many
//! tokens. `ferric_kv::Scheduler` already decides *what* runs each step, so it is tempting to assume
//! throughput follows. **It does not, and this measures the gap.**
//!
//! `Qwen3::forward_cached(tokens, cache)` takes exactly one sequence's cache, so a step with N live
//! sequences issues **N separate forward passes** — N host builds, N GPU round trips. The scheduler
//! removes head-of-line blocking (measured elsewhere: a 3-token request waits 16 steps instead of 272)
//! but cannot amortise work the execution layer has no way to share.
//!
//! What would fix it is a **batched decode**: stack N sequences' single tokens into `[N, d]` and run one
//! forward. That needs each sequence's KV attended separately inside one kernel — which is exactly what
//! paged attention provides, and why `ferric-kv`'s `PagedKv` is complete, tested, and unwired.
//!
//! This benchmark exists to give that work a number to beat.
//!
//!   cargo run -p ferric-llama --example batch_throughput --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();
    let m = Qwen3::load(&ctx, &g).unwrap();
    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    println!("Aggregate decode throughput vs live sequences — Qwen2.5-0.5B\n");
    println!("  If concurrency amortised the host build, tokens/sec would rise with the sequence count.");
    println!("  Each step below runs N sequences as N SEPARATE forward passes, because `forward_cached`");
    println!("  takes a single sequence's cache.\n");
    println!("  {:>6} {:>12} {:>14} {:>12}", "seqs", "ms/step", "tokens/sec", "vs 1 seq");
    println!("  {:-<48}", "");

    let mut base = 0.0f64;
    let mut scaling = Vec::new();
    for &nseq in &[1usize, 2, 4, 8] {
        let mut caches: Vec<Cache> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        for i in 0..nseq {
            let mut c = Cache::new(&m.cfg);
            let seed: Vec<u32> = (0..6u32).map(|j| 100 + j + 37 * i as u32).collect();
            let l = m.forward_cached(&seed, &mut c).to_vec().await;
            next.push(am(&l));
            caches.push(c);
        }

        const STEPS: usize = 20;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..STEPS {
                for s in 0..nseq {
                    let l = m.forward_cached(&[next[s]], &mut caches[s]).to_vec().await;
                    next[s] = am(&l);
                }
            }
            samples.push(t0.elapsed().as_secs_f64() * 1000.0 / STEPS as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms_step = samples[2]; // median
        let tps = nseq as f64 / (ms_step / 1000.0);
        if nseq == 1 { base = tps; }
        scaling.push(tps / base);
        println!("  {nseq:>6} {ms_step:>12.2} {tps:>14.1} {:>11.2}x", tps / base);
    }

    let best = scaling.iter().cloned().fold(0.0f64, f64::max);
    println!("\n  Best scaling at 8 sequences: {:.2}x (perfect would be 8.00x).\n", scaling[scaling.len() - 1]);
    println!("  ⚠ Throughput is FLAT in the sequence count, and that is the finding. N sequences cost N");
    println!("  times as long: each is its own host build and its own GPU round trip. The scheduler");
    println!("  removes head-of-line blocking but cannot amortise work the execution layer has no way");
    println!("  to share.\n");
    println!("  The fix is a BATCHED decode: stack N sequences' tokens into [N, d] and run one forward.");
    println!("  That needs each sequence's KV attended separately inside one kernel, which is what paged");
    println!("  attention provides — and is why `ferric-kv`'s PagedKv is complete, tested, and unwired.");
    println!("  This number is what that work has to beat.");

    // Guard the claim in both directions. If scaling ever becomes real, this example's text is stale and
    // should fail rather than keep asserting a gap that has been closed.
    assert!(
        best < 1.6,
        "throughput now scales {best:.2}x with concurrency — batched decode appears to have landed, so \
         this example's conclusion is out of date and must be rewritten"
    );
    assert!(base > 1.0, "baseline {base:.1} tok/s is implausible — the benchmark measured nothing");
}
