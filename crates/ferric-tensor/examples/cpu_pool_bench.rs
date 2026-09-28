//! **What one CPU-fabric dispatch costs**, with no work in it: the round trip a decode step pays ~120
//! times per token on a 24-layer model (4 matmuls + attention per layer, + the head).
//!
//!   FERRIC_CPU_THREADS=24 cargo run -p ferric-tensor --release --example cpu_pool_bench
//!
//! Reports the median and p90 round trip of `for_each(items, no-op)` for a few item counts, and the
//! round trip with ~2 us of real work per item.
use std::time::Instant;

fn main() {
    let pool = ferric_tensor::cpu_q::pool();
    println!("threads {}", pool.threads());
    let sink = std::sync::atomic::AtomicU64::new(0);
    for &(items, work) in &[(1usize, 0u32), (24, 0), (96, 0), (384, 0), (96, 2000)] {
        let mut v = Vec::new();
        for _ in 0..3000 {
            let t = Instant::now();
            pool.for_each(items, |i| {
                let mut x = i as u64;
                for _ in 0..work { x = x.wrapping_mul(6364136223846793005).wrapping_add(1); }
                if work > 0 { sink.fetch_add(x & 1, std::sync::atomic::Ordering::Relaxed); }
            });
            v.push(t.elapsed().as_secs_f64() * 1e6);
        }
        v.sort_by(|a, b| a.total_cmp(b));
        println!("items {items:4} work/item {work:5} iters: round trip median {:6.2} us  p90 {:6.2} us  min {:6.2} us",
                 v[v.len() / 2], v[v.len() * 9 / 10], v[0]);
    }
    let _ = sink.load(std::sync::atomic::Ordering::Relaxed);
}
