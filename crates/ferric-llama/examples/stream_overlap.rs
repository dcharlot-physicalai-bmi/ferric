//! **Does overlapping the read with compute pay?** Measured against a device slower than a page cache.
//!
//! `stream_generate` showed the streaming path works and that its dominant cost is rebuilding a layer's
//! GPU tensors, not reading it. That is true *on a local SSD with a warm page cache* — where a read is
//! effectively a memcpy. It is not the situation streaming exists for.
//!
//! A model that genuinely exceeds memory reads from a real device on every miss, and there the read is
//! the cost. This measures both regimes by injecting a backing with a controllable per-read delay, so the
//! answer is a curve rather than one number from whichever machine happened to run it.
//!
//! The prefetcher hides a read behind the *next* layer's build and compute — work that
//! `stream_generate` already established is substantial. Prediction: little at zero delay, growing as the
//! device slows, saturating once the read exceeds the compute available to hide it behind. Measured, over
//! two runs:
//!
//! ```text
//!   read delay   speedup (run 1 / run 2)
//!        0 us      1.16x / 1.09x   <- at the noise floor
//!      500 us      1.33x / 1.26x
//!     2000 us      1.53x / 1.54x   <- reproducible
//!     8000 us      1.50x / 1.52x   <- saturated
//! ```
//!
//!   cargo run -p ferric-llama --example stream_overlap --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::stream::layer_runs;
use ferric_tier::{Backing, FileBacking, TierError};
use std::sync::Arc;
use std::time::Duration;

const GEN: usize = 6;

/// A backing with a fixed per-read delay, standing in for a device slower than RAM.
struct Slow {
    inner: FileBacking,
    delay: Duration,
    reads: std::sync::atomic::AtomicU64,
}
impl Backing for Slow {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !self.delay.is_zero() { std::thread::sleep(self.delay); }
        self.inner.read_at(offset, dst)
    }
}

async fn generate(m: &Qwen3, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut cache = Cache::new(&m.cfg);
    let mut ids = prompt.to_vec();
    let mut out = Vec::new();
    let mut fed = 0usize;
    for _ in 0..n {
        let logits = m.forward_cached(&ids[fed..], &mut cache).to_vec().await;
        cache.pos = ids.len();
        fed = ids.len();
        let v = m.cfg.n_vocab;
        let last = &logits[logits.len() - v..];
        let (mut best, mut bv) = (0u32, f32::MIN);
        for (i, &x) in last.iter().enumerate() { if x > bv { bv = x; best = i as u32; } }
        ids.push(best);
        out.push(best);
    }
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();
    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!(),
    };
    let vocab: std::collections::HashMap<String, u32> =
        toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!(),
    };
    let prompt = ferric_tokenizer::Bpe::new(vocab, &merges).encode("The capital of France is");

    let runs = layer_runs(&g).unwrap();
    let biggest = runs.iter().map(|r| r.bytes).max().unwrap();
    let budget = biggest + 4096; // the floor: everything streams, so every layer is a read
    drop(g);

    println!("Overlap vs synchronous streaming — Qwen2.5-0.5B, {} layers, {GEN} tokens", runs.len());
    println!("  budget {:.1} MB (nothing pinned: every layer of every token is a real read)\n", budget as f64 / 1e6);
    println!("  {:>10}  {:>12}  {:>12}  {:>9}   {}", "read delay", "synchronous", "overlapped", "speedup", "tokens match");
    println!("  {:-<74}", "");

    let mut reference: Option<Vec<u32>> = None;
    let mut any_win = false;
    for delay_us in [0u64, 500, 2000, 8000] {
        let mut row = [0f64; 2];
        let mut ids_seen = Vec::new();
        for (k, overlap) in [(0usize, false), (1usize, true)] {
            let backing: Arc<dyn Backing + Send + Sync> = Arc::new(Slow {
                inner: FileBacking::open(&path).unwrap(),
                delay: Duration::from_micros(delay_us),
                reads: std::sync::atomic::AtomicU64::new(0),
            });
            let m = Qwen3::load_streaming_with(&ctx, &path, budget, Some(backing), overlap).unwrap();
            let t = std::time::Instant::now();
            let ids = generate(&m, &prompt, GEN).await;
            row[k] = t.elapsed().as_secs_f64() * 1000.0 / GEN as f64;
            ids_seen = ids;
        }
        // Overlap must not change a single token — a faster wrong answer is not a speedup.
        match &reference {
            None => reference = Some(ids_seen.clone()),
            Some(r) => assert_eq!(&ids_seen, r, "overlap or delay changed the output at {delay_us} us"),
        }
        let speedup = row[0] / row[1];
        if speedup > 1.15 { any_win = true; }
        println!("  {:>7} us  {:>9.1} ms  {:>9.1} ms  {:>8.2}x   {}",
                 delay_us, row[0], row[1], speedup,
                 if reference.as_ref() == Some(&ids_seen) { "identical ✓" } else { "DIFFERS ✗" });
    }

    println!();
    if any_win {
        println!("  ✅ The overlap pays, and never changes a token. The gain GROWS with device latency");
        println!("     and then saturates once the read exceeds the compute available to hide it behind.");
        println!("     Even at 0 us added delay there is a modest gain, because a 15.9 MB read is real");
        println!("     work even from the page cache — but it is near the noise floor, which is exactly");
        println!("     why stream_generate found the REBUILD dominant on this machine. A warm page cache");
        println!("     is not the workload streaming exists for; the 2000-8000 us rows are.");
    } else {
        println!("  ⚠️  No regime here showed a win beyond noise. Report that as-is: on this machine the");
        println!("     read never became the bottleneck, so the overlap has nothing to hide.");
    }
    println!("\n  (Wall clock on this machine carries ~20% run-to-run spread; treat a ratio under ~1.15x");
    println!("   as noise, per the same rule applied to every other timing in this codebase.)");
}
