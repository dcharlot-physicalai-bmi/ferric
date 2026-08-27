//! **Gemma 4 batched decode: N sequences, one forward pass — and provably the same tokens.**
//!
//! `Gemma4::forward_batch` advances N independent sequences by one token each in a single pass, so the
//! ~5 GB of Q8_0 weights stream once for N tokens instead of N times. The projections batch; attention
//! stays a per-sequence loop, because each sequence attends its own KV history at its own position.
//!
//! ## Why this example exists, and why it asserts numbers rather than eyeballing text
//!
//! The failure mode of batched decode has **no symptom**. A path that crosses sequences — one shared
//! RoPE position, a K row appended to the wrong cache, a shared-KV read resolved against sequence 0's
//! buffer for every row — still emits fluent, confident text with finite logits and no error. Two of
//! those have already shipped in this repo: `rope_scaled` read its per-row positions out of bounds and
//! silently became a no-rope model, and the dense runtime's `forward_batch` diverged on rope-scaled
//! models because its batched rope was unscaled while the solo path's was not. Neither was visible in
//! the output.
//!
//! Gemma 4 adds three more ways to get this wrong that Qwen does not have:
//!
//!   1. **shared KV** — blocks 15..34 own no cache and read block 13's (sliding) or 14's (global).
//!      That indirection has to be resolved inside *each sequence's own* cache.
//!   2. **two head widths** — 512 on global blocks, 256 on sliding, in one model, so the rope width and
//!      the cache row width both change per block.
//!   3. **two rope variants** — proportional-scaled on global blocks, plain on sliding. The batched
//!      kernel must pick the same one, per block, that the solo path picks.
//!
//! ## Two gates, because one of them is not enough
//!
//!   * **token ids** — every sequence must generate exactly what it generates alone.
//!   * **max|Δ| over the full 262144-wide logit rows** — asserted, not just printed.
//!
//! The second gate exists because the first was **measured blind** to a real bug. Deleting the
//! proportional rope scaling from the batched global blocks — the exact defect that hit the dense
//! runtime — left every token id identical at short prompts, because that scaling only bends the
//! low-frequency angles and those angles are tiny below a few hundred positions. It moved the logits by
//! 8.2e-1 while the greedy pick never flipped. A token-only check would have signed off on it.
//!
//! Hence also the third scenario below, whose prompts run past the 512 sliding window so the scaled
//! branch is exercised where it actually bites.
//!
//! ```text
//!   cargo run -q -p ferric-llama --example gemma4_batched_decode --release -- <model.gguf>
//! ```
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::gemma4::{Cache, Gemma4};
use std::sync::Arc;

/// Logit agreement required between the batched and solo paths.
///
/// MEASURED, not guessed. Clean runs on gemma-4-E2B-it-Q8_0: **0.0 exactly** at n≤2 and **2.1e-5** at
/// n=4 — the residue of `matmul_q8_0` switching from its split-K kernel to the flat one above 2 rows,
/// which changes the reduction order of the 262144-wide head and nothing else. A structural
/// divergence is four orders of magnitude larger: dropping the global blocks' proportional rope moved
/// it to 8.2e-1. This threshold sits ~500x above the observed kernel noise and ~80x below the observed
/// break, so it separates them without being tuned to either.
const MAX_DELTA: f32 = 1e-2;

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

/// Prefill `p` and return the last position's logits, without ever materialising `[T, 262144]`.
///
/// A 700-token prompt's full logit tensor is 734 MB, and reading it back to compare one row would be
/// 734 MB of transfer to throw 699/700ths of away — on a machine where a large allocation burst has
/// already been seen to make Metal drop buffer contents silently. So the prompt runs through the
/// hidden-state path and only the final token goes through the head.
async fn prefill(m: &Gemma4, p: &[u32], cache: &mut Cache) -> Vec<f32> {
    assert!(p.len() >= 2, "prompts here are split as [..n-1] then [n-1]");
    let _ = m.forward_hidden_cached(&p[..p.len() - 1], cache);
    m.forward(&p[p.len() - 1..], cache).to_vec().await
}

/// Run `prompts` both ways and assert they agree. Returns `(min top-1 margin, max|Δ|)`.
async fn check(m: &Gemma4, prompts: &[Vec<u32>], steps: usize) -> (f32, f32) {
    let (n, vn) = (prompts.len(), m.cfg.n_vocab);
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();

    // ---- reference: every sequence decoded entirely on its own ----
    let mut solo_caches: Vec<Cache> = Vec::new();
    let mut solo_out: Vec<Vec<u32>> = Vec::new();
    for p in prompts {
        let mut cache = Cache::new(&m.cfg);
        let l = prefill(m, p, &mut cache).await;
        let mut tok = argmax(&l);
        let mut r#gen = vec![tok];
        for _ in 1..steps {
            let l = m.forward(&[tok], &mut cache).to_vec().await;
            tok = argmax(&l);
            r#gen.push(tok);
        }
        solo_out.push(r#gen);
        solo_caches.push(cache);
    }

    // ---- batched: same prompts, one forward per step for all N ----
    // Prefill stays per-sequence — the prompts differ in length, so there is nothing to batch there.
    // Only the decode steps go through forward_batch, which is the code under test.
    let mut bat_caches: Vec<Cache> = Vec::new();
    let mut next: Vec<u32> = Vec::new();
    for p in prompts {
        let mut cache = Cache::new(&m.cfg);
        let l = prefill(m, p, &mut cache).await;
        next.push(argmax(&l));
        bat_caches.push(cache);
    }
    let mut bat_out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
    // How close any greedy pick ever came to a tie. If this were tiny, token equality would be luck
    // rather than evidence, and the number belongs in the output rather than hidden behind a ✅.
    let mut min_margin = f32::MAX;
    for _ in 1..steps {
        let mut refs: Vec<&mut Cache> = bat_caches.iter_mut().collect();
        let logits = m.forward_batch(&next, &mut refs).to_vec().await;
        assert_eq!(logits.len(), n * vn, "forward_batch must return one logit row per sequence");
        for i in 0..n {
            let row = &logits[i * vn..(i + 1) * vn];
            let (mut top1, mut top2) = (f32::MIN, f32::MIN);
            for &x in row { if x > top1 { top2 = top1; top1 = x; } else if x > top2 { top2 = x; } }
            min_margin = min_margin.min(top1 - top2);
            next[i] = argmax(row);
            bat_out[i].push(next[i]);
        }
    }

    // ---- gate 1: token-identical, per sequence, and loud about WHICH one broke ----
    for i in 0..n {
        if solo_out[i] != bat_out[i] {
            let step = solo_out[i].iter().zip(&bat_out[i]).position(|(a, b)| a != b).unwrap_or(0);
            eprintln!("\n❌ SEQUENCE {i} of {n} DIVERGED at generated token {step} \
                       (prompt len {}, absolute position {})", lens[i], lens[i] + step);
            eprintln!("   solo : {:?}", &solo_out[i][..(step + 4).min(steps)]);
            eprintln!("   batch: {:?}", &bat_out[i][..(step + 4).min(steps)]);
            eprintln!("   min top-1 margin seen this run: {min_margin:.4}");
            panic!("sequence {i} of {n} is not token-identical to solo decode. Batching must change \
                    only HOW the work is scheduled. Suspects, in order: a shared RoPE position \
                    (positions[] must be each cache's own pos), a shared-KV read resolved against the \
                    wrong sequence's cache (c.kv[src], not caches[0].kv[src]), or the batched rope \
                    picking the unscaled variant on a global block.");
        }
    }

    // ---- gate 2: how much headroom that agreement has ----
    // Both cache sets now hold the SAME token history at the same positions, so one more step on each
    // is a like-for-like comparison of full logit rows.
    //
    // ⚠ This step must be ONE batched call over all N, not N calls of one sequence each. A
    // single-sequence forward_batch runs every matmul at rows=1 — the same kernel the solo path takes
    // — so it would report a flattering 0 while saying nothing about the rows=N kernels batched decode
    // actually uses. That is the measurement quietly answering a different question.
    let feed: Vec<u32> = solo_out.iter().map(|g| *g.last().unwrap()).collect();
    let mut solo_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        solo_rows.push(m.forward(&feed[i..i + 1], &mut solo_caches[i]).to_vec().await);
    }
    let bat_rows = {
        let mut refs: Vec<&mut Cache> = bat_caches.iter_mut().collect();
        m.forward_batch(&feed, &mut refs).to_vec().await
    };
    let mut worst = 0.0f32;
    for i in 0..n {
        let br = &bat_rows[i * vn..(i + 1) * vn];
        let dmax = solo_rows[i].iter().zip(br).fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert_eq!(argmax(&solo_rows[i]), argmax(br),
                   "sequence {i} of {n}: batched argmax differs from solo on the logit-diff step");
        assert!(dmax < MAX_DELTA,
                "sequence {i} of {n}: batched logits differ from solo by {dmax:.3e}, over the {MAX_DELTA:.0e} \
                 bar. That is far above kernel-selection noise (~2e-5) — it is a structural difference. \
                 Token ids can agree while this does not: dropping the global blocks' proportional rope \
                 scaling produced 8.2e-1 here with every token id still identical.");
        worst = worst.max(dmax);
    }
    (min_margin, worst)
}

fn main() { pollster::block_on(run()); }

async fn run() {
    // ⚠ argv, never a hardcoded path. The dense runtime's equivalence example hardcoded a Qwen
    // checkpoint and ignored argv, so it "passed" while testing a different model than the one whose
    // batched path was broken — a test that could not fail.
    let path = std::env::args().nth(1).expect(
        "usage: gemma4_batched_decode <model.gguf>\n\
         (no default: a hardcoded path is how an equivalence check ends up testing the wrong model)");
    println!("model: {path}");

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(&path).expect("open");
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    assert_eq!(arch, "gemma4", "this example is for gemma4, got {arch:?}");

    let m = Gemma4::load(&ctx, &g).expect("load gemma4");
    let c = &m.cfg;
    println!("gemma4 · {} blocks, d={}, {} heads x {}/{} (global/swa), {} kv, window {}",
             c.n_layer, c.d, c.n_head, c.head_dim, c.head_dim_swa, c.n_kv, c.window);
    // Look the source blocks up from the schedule rather than assuming which end of the range is
    // which: on E2B block 34 is GLOBAL, so `kv_src(n_layer-1)` is not the sliding answer.
    let shared_src = |swa: bool| (c.kv_from_start..c.n_layer).find(|&i| c.swa[i] == swa).map(|i| c.kv_src(i));
    println!("  KV-owning blocks 0..{} ({} shared); sliding blocks read {:?}, global read {:?}",
             c.kv_from_start, c.n_layer - c.kv_from_start, shared_src(true), shared_src(false));
    println!("  rope: global base {} (proportional factors: {}), sliding base {}\n",
             c.rope_base, m.has_rope_freqs(), c.rope_base_swa);

    // Deliberately UNEQUAL prompt lengths everywhere. If every sequence started at the same position, a
    // batched path that shared one RoPE position across rows would agree with the solo path anyway and
    // this whole example would prove nothing.
    //
    // Token 2 is BOS; the rest are ordinary ids, spread apart so no two sequences share a prefix.
    let mk = |n: usize, base: usize, step: usize| -> Vec<Vec<u32>> {
        (0..n).map(|i| {
            let mut p = vec![2u32];
            p.extend((0..(base + step * i)).map(|j| (500 + (j * 7 + 137 * i) % 60000) as u32));
            p
        }).collect()
    };

    for &(n, base, step, steps, label) in &[
        (2usize, 4usize, 3usize, 16usize, "short"),
        (4, 4, 3, 16, "short"),
        // Past the 512 sliding window, and far enough out that the global blocks' proportional rope
        // scaling actually bends the angles. At ~30 positions it does not, which is how an unscaled
        // batched rope passed a token-only check.
        (3, 560, 90, 8, "long (crosses the 512 window)"),
    ] {
        let prompts = mk(n, base, step);
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let t0 = std::time::Instant::now();
        let (margin, worst) = check(&m, &prompts, steps).await;
        println!("  n={n} {label}: all {n} sequences token-IDENTICAL to solo decode over {steps} steps");
        println!("        prompt lengths {lens:?} (all different, so all positions differ)");
        println!("        min top-1 margin {margin:.4} · max|Δ| over full logit rows {worst:.3e} \
                  (bar {MAX_DELTA:.0e}) · {:.1?}\n", t0.elapsed());
    }

    println!("✅ forward_batch is solo-equivalent on gemma4 at n=2, 3 and 4, at short and long positions.");
    println!("   Batched: q/k/v/o, gate/up/down, the per-layer gate+proj and the {}-wide head all run as", c.n_vocab);
    println!("   one matmul over [N, d]. Not batched: attention, because each sequence's KV history has");
    println!("   its own length — and on Gemma 4 also its own source block and its own head width.");
}
