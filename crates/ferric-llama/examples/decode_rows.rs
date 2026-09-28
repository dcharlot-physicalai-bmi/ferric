//! **What a forward of a few rows costs** — the number prompt lookup and small continuous batches live on.
//!
//! A verify forward (prompt lookup: the pending token plus up to k drafts) and a batched decode step
//! (N sequences, one token each) both push 2-16 rows through every projection. Decode is weight
//! streaming, so the question is whether those rows read the weights once or once each. This times,
//! on one loaded model and INTERLEAVED so a contended machine hits both arms alike:
//!
//!   * `verify t`: `forward_cached` of `t` rows continuing a cache, all rows' logits read back (what
//!     `generate_lookup` does), then `Cache::truncate` back — so every rep sees the same position;
//!   * `batch n`: `forward_batch` of `n` sequences one token each, logits read back;
//!   * `serial`: one-row `forward_cached_last` + readback — the plain loop's step.
//!
//! Each configuration runs in both arms of the small-M switch (`ferric_tensor::dtype::set_small_m`), alternating
//! rep by rep. Ranges are min..max over reps; the median is the headline.
//!
//!   cargo run -p ferric-llama --release --example decode_rows [model.gguf] [reps]
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;
use std::time::Instant;

fn main() { pollster::block_on(run()); }

fn stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf")
    });
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    println!("model: {path}\nadapter: {} [{:?}]", ctx.adapter_name, ctx.backend);
    let g = GgufFile::open(&path).unwrap();
    let m = Qwen3::load(&ctx, &g).unwrap();
    let vn = m.cfg.n_vocab;
    let prompt: Vec<u32> = (0..160).map(|j| (100 + (j * 37) % 5000) as u32).collect();
    let feed: Vec<u32> = (0..16).map(|j| (300 + j * 11) as u32).collect();

    let mut c = Cache::new(&m.cfg);
    let _ = m.forward_cached_last(&prompt, &mut c).to_vec().await;
    let base = c.pos;
    let arms = [true, false];
    let label = |on: bool| if on { "small-M" } else { "prefill" };

    println!("\n{:>10} {:>8}  {:>26}  {:>26}", "shape", "rows", "small-M ms (min..med..max)", "old ms (min..med..max)");
    // warm every pipeline in both arms
    for &on in &arms {
        ferric_tensor::dtype::set_small_m(on);
        for t in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16] {
            let _ = m.forward_cached(&feed[..t], &mut c).to_vec().await;
            c.truncate(base);
        }
    }
    let serial = |m: &Qwen3, c: &mut Cache| {
        let t0 = Instant::now();
        let _ = pollster::block_on(m.forward_cached_last(&feed[..1], c).to_vec());
        c.truncate(base);
        t0.elapsed().as_secs_f64() * 1e3
    };
    let verify = |m: &Qwen3, c: &mut Cache, t: usize| {
        let t0 = Instant::now();
        let v = pollster::block_on(m.forward_cached(&feed[..t], c).to_vec());
        assert_eq!(v.len(), t * vn);
        c.truncate(base);
        t0.elapsed().as_secs_f64() * 1e3
    };
    {
        let mut s: [Vec<f64>; 2] = [vec![], vec![]];
        for _ in 0..reps { for (k, &on) in arms.iter().enumerate() { ferric_tensor::dtype::set_small_m(on); s[k].push(serial(&m, &mut c)); } }
        let (a, b) = (stats(&mut s[0]), stats(&mut s[1]));
        println!("{:>10} {:>8}  {:>8.2}..{:>6.2}..{:>7.2}  {:>8.2}..{:>6.2}..{:>7.2}", "serial", 1, a.0, a.1, a.2, b.0, b.1, b.2);
    }
    for t in [2usize, 3, 4, 6, 8, 9, 12, 16] {
        let mut s: [Vec<f64>; 2] = [vec![], vec![]];
        for _ in 0..reps { for (k, &on) in arms.iter().enumerate() { ferric_tensor::dtype::set_small_m(on); s[k].push(verify(&m, &mut c, t)); } }
        let (a, b) = (stats(&mut s[0]), stats(&mut s[1]));
        println!("{:>10} {:>8}  {:>8.2}..{:>6.2}..{:>7.2}  {:>8.2}..{:>6.2}..{:>7.2}", "verify", t, a.0, a.1, a.2, b.0, b.1, b.2);
    }
    for n in [2usize, 4, 8, 16] {
        let mut caches: Vec<Cache> = (0..n).map(|i| {
            let mut ci = Cache::new(&m.cfg);
            let _ = pollster::block_on(m.forward_cached_last(&prompt[..40 + 7 * i], &mut ci).to_vec());
            ci
        }).collect();
        let bases: Vec<usize> = caches.iter().map(|c| c.pos).collect();
        let toks: Vec<u32> = (0..n).map(|i| (500 + i) as u32).collect();
        let mut s: [Vec<f64>; 2] = [vec![], vec![]];
        for rep in 0..reps + 1 {
            for (k, &on) in arms.iter().enumerate() {
                ferric_tensor::dtype::set_small_m(on);
                let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let t0 = Instant::now();
                let v = m.forward_batch(&toks, &mut refs).to_vec().await;
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                assert_eq!(v.len(), n * vn);
                for (ci, &b) in caches.iter_mut().zip(&bases) { ci.truncate(b); }
                if rep > 0 { s[k].push(ms); } // rep 0 warms this batch size's pipelines
            }
        }
        let (a, b) = (stats(&mut s[0]), stats(&mut s[1]));
        println!("{:>10} {:>8}  {:>8.2}..{:>6.2}..{:>7.2}  {:>8.2}..{:>6.2}..{:>7.2}   agg tok/s {:.0} vs {:.0}",
                 "batch", n, a.0, a.1, a.2, b.0, b.1, b.2, n as f64 * 1e3 / a.1, n as f64 * 1e3 / b.1);
    }
    let _ = label;
}
