//! **Batched decode: N sequences, one forward pass** — identical output, and faster.
//!
//! Decode is a weight-streaming problem. One token costs a full pass over ~525 MB of Q8_0 weights, and
//! `examples/batch_throughput.rs` measured the consequence: running N sequences as N separate forwards
//! costs exactly N times as long — **1.00× scaling at 8 sequences**, no amortisation whatsoever.
//!
//! `forward_batch` stacks N sequences' tokens into `[N, d]` so the weights are read **once for N tokens**.
//! The projections (qkv, o, gate_up, down, lm_head) batch directly; attention cannot, because sequence
//! `i` attends its own history at its own position and those histories differ in length, so that part
//! stays a loop. Collapsing that loop too is what paged attention would buy.
//!
//! ## Correctness first, and it is not optional here
//!
//! Every sequence's logits must be **identical** to decoding it alone. A batched path that leaked between
//! sequences — a shared RoPE position, a KV row appended to the wrong cache — still emits fluent text.
//! There is no crash to catch it. So the tokens are compared directly, for many steps, with sequences at
//! deliberately *different* history lengths so a position mix-up cannot cancel out.
//!
//!   cargo run -p ferric-llama --example batched_decode --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

const STEPS: usize = 24;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    // ⚠ This used to HARDCODE a Qwen path and ignore argv entirely, so passing any other checkpoint
    // silently tested Qwen instead. Qwen has no rope_freqs, so the equivalence claim never covered the
    // scaled-rope branch of attn_batch — a check that passed for the wrong reason.
    let home = std::env::var("HOME").unwrap();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf")
    });
    println!("model: {path}");
    let g = GgufFile::open(&path).unwrap();
    let m = Qwen3::load(&ctx, &g).unwrap();
    let vn = m.cfg.n_vocab;
    let am = |row: &[f32]| row.iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    println!("Batched decode — N sequences in one forward pass\n");

    for &n in &[2usize, 4, 8] {
        // Deliberately UNEQUAL prompt lengths, so every sequence sits at a different position. If the
        // batched path shared a position or crossed a cache, equal lengths could hide it.
        let prompts: Vec<Vec<u32>> = (0..n)
            .map(|i| (0..(5 + 3 * i)).map(|j| (100 + j + 29 * i) as u32).collect())
            .collect();

        // ---- reference: each sequence decoded entirely on its own ----
        let mut ref_out: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward_cached(p, &mut c).to_vec().await;
            let mut tok = am(&l[l.len() - vn..]);
            let mut r#gen = vec![tok];
            for _ in 1..STEPS {
                let l = m.forward_cached(&[tok], &mut c).to_vec().await;
                tok = am(&l[l.len() - vn..]);
                r#gen.push(tok);
            }
            ref_out.push(r#gen);
        }

        // ---- batched: same sequences, one forward per step ----
        let mut caches: Vec<Cache> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        for p in &prompts {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward_cached(p, &mut c).to_vec().await;
            next.push(am(&l[l.len() - vn..]));
            caches.push(c);
        }
        let mut bat_out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
        for _ in 1..STEPS {
            let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let logits = m.forward_batch(&next, &mut refs).to_vec().await;
            for i in 0..n {
                next[i] = am(&logits[i * vn..(i + 1) * vn]);
                bat_out[i].push(next[i]);
            }
        }

        for i in 0..n {
            assert_eq!(
                ref_out[i], bat_out[i],
                "sequence {i} of {n} diverged. Batching must change only HOW the work is scheduled — a \
                 crossed cache or a shared RoPE position produces fluent, wrong text with no error."
            );
        }
        println!("  n={n}: all {n} sequences IDENTICAL to solo decode over {STEPS} steps \
                  (prompt lengths {:?})", prompts.iter().map(|p| p.len()).collect::<Vec<_>>());
    }

    // ---- throughput ----
    let Some(load) = load_avg() else { return };
    println!("\n  machine load average: {load:.2}");
    if load >= 8.0 {
        println!("  ⚠ too loaded to time anything — correctness above still holds. Re-run when quiet.");
        return;
    }

    println!("\n  {:>6} {:>13} {:>13} {:>11}", "seqs", "N separate", "1 batched", "speedup");
    println!("  {:-<48}", "");
    let mut best = 0.0f64;
    for &n in &[1usize, 2, 4, 8] {
        let mk = |m: &Qwen3| -> Vec<Cache> { (0..n).map(|_| Cache::new(&m.cfg)).collect() };
        let mut ca = mk(&m);
        let mut na = vec![0u32; n];
        for (i, c) in ca.iter_mut().enumerate() {
            let l = m.forward_cached(&[(100 + i) as u32], c).to_vec().await;
            na[i] = am(&l[l.len() - vn..]);
        }
        let mut sep = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                for i in 0..n {
                    let l = m.forward_cached(&[na[i]], &mut ca[i]).to_vec().await;
                    na[i] = am(&l[l.len() - vn..]);
                }
            }
            sep.push(t0.elapsed().as_secs_f64() * 1000.0 / 8.0);
        }
        sep.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut cb = mk(&m);
        let mut nb = vec![0u32; n];
        for (i, c) in cb.iter_mut().enumerate() {
            let l = m.forward_cached(&[(100 + i) as u32], c).to_vec().await;
            nb[i] = am(&l[l.len() - vn..]);
        }
        let mut bat = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                let mut refs: Vec<&mut Cache> = cb.iter_mut().collect();
                let lg = m.forward_batch(&nb, &mut refs).to_vec().await;
                for i in 0..n { nb[i] = am(&lg[i * vn..(i + 1) * vn]); }
            }
            bat.push(t0.elapsed().as_secs_f64() * 1000.0 / 8.0);
        }
        bat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sp = sep[2] / bat[2];
        best = best.max(sp);
        println!("  {n:>6} {:>10.2} ms {:>10.2} ms {:>10.2}x", sep[2], bat[2], sp);
    }

    println!("\n  ✅ Identical output, {best:.2}x throughput at the best batch size. The projections read");
    println!("     the weights once for N tokens instead of N times; attention stays a per-sequence loop");
    println!("     because each history differs in length, which is the part paged attention would fold in.");
    assert!(best > 1.5, "batching gained only {best:.2}x — the amortisation is not materialising");
}
