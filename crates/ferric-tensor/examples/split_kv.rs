//! **Split-KV decode attention** — same answer, more of the GPU used.
//!
//! `fused_decode_attention` dispatches one workgroup per query head and walks the whole KV cache
//! *serially inside* that workgroup. At batch 1 with a long context — one user, on-device, the case
//! Ferric exists for — a 14-head model occupies 14 workgroups on a GPU with hundreds of cores.
//!
//! Every other parallelism Ferric has (paged KV, continuous batching, batched decode) works across
//! *sequences*, so none of it helps when there is only one. Splitting the KV axis does: `nh × splits`
//! workgroups, each owning a contiguous slice of the cache, merged by a second pass.
//!
//! ## What has to be true
//!
//! 1. **The answer must not change.** The merge is the same `exp(m_old - m_new)` rescale the kernel
//!    already applies between chunks, so it is mathematically identical — but the *reduction order*
//!    differs, so floating-point results may differ in the last bits. This measures by how much, rather
//!    than assuming, because Ferric's byte-identical stance means that number is a decision input.
//! 2. **It must actually pay**, and only past some context length — two dispatches replace one, so short
//!    caches should be left alone. `decode_splits` gates on that; this checks the gate is right.
//!
//!   cargo run -p ferric-tensor --example split_kv --release
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f32 / 8388608.0) - 1.0 }).collect()
}

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("Split-KV decode attention — correctness, then speed\n");

    // ---- correctness across head counts, GQA ratios, head dims and cache lengths ----
    println!("  {:>5} {:>5} {:>5} {:>8} {:>8} {:>13}", "nh", "nkv", "dh", "S", "splits", "max|Δ|");
    println!("  {:-<52}", "");
    let mut worst = 0f32;
    for &(nh, nkv, dh) in &[(14usize, 2usize, 64usize), (8, 8, 128), (32, 8, 64), (4, 1, 96)] {
        for &s in &[1usize, 700, 1024, 4096, 9000, 32768] {
            let q = Tensor::from_vec(&ctx, &fill(nh * dh, 7), &[1, nh * dh]);
            let k = Tensor::from_vec(&ctx, &fill(s * nkv * dh, 11), &[s, nkv * dh]);
            let v = Tensor::from_vec(&ctx, &fill(s * nkv * dh, 22), &[s, nkv * dh]);

            std::env::set_var("FERRIC_SPLITKV", "1");           // force the original single-pass path
            let base = q.fused_decode_attention(&k, &v, nh, nkv, dh).to_vec().await;
            std::env::remove_var("FERRIC_SPLITKV");             // let the heuristic choose
            let got = q.fused_decode_attention(&k, &v, nh, nkv, dh).to_vec().await;

            let d = base.iter().zip(&got).fold(0f32, |a, (&x, &y)| a.max((x - y).abs()));
            worst = worst.max(d);
            // Report the split count the heuristic actually picked, by probing the same rule.
            let splits = if s < 1024 { 1 } else { (s / 512).min(256 / nh.max(1)).max(1) };
            println!("  {nh:>5} {nkv:>5} {dh:>5} {s:>8} {splits:>8} {d:>13.3e}");
            assert!(
                d < 2e-5,
                "nh={nh} nkv={nkv} dh={dh} S={s}: split-KV differs by {d:.3e} — the merge is wrong, not \
                 merely reordered. Check the unnormalised partials and the empty-split case."
            );
        }
    }
    println!("\n  Worst deviation across all shapes: {worst:.3e}");
    if worst == 0.0 {
        println!("  Bit-identical — the merge happens to reproduce the sequential reduction order here.");
    } else {
        println!("  NOT bit-identical. The merge is mathematically the same rescale but reduces in a");
        println!("  different order, so the last bits move. That is a deliberate decision, not a bug:");
        println!("  the split path is only taken past the length gate, so short-context decode — where");
        println!("  reproducibility matters most for tests — is untouched and still exact.");
    }

    // ---- speed ----
    let Some(load) = load_avg() else { return };
    println!("\n  machine load average: {load:.2}");
    if load >= 8.0 {
        println!("  ⚠ too loaded to time anything — correctness above still holds. Re-run when quiet.");
        return;
    }
    let (nh, nkv, dh) = (14usize, 2usize, 64usize);
    println!("\n  {:>8} {:>12} {:>12} {:>10}", "S", "1 workgroup", "split-KV", "speedup");
    println!("  {:-<46}", "");
    for &s in &[1024usize, 4096, 16384, 32768] {
        let q = Tensor::from_vec(&ctx, &fill(nh * dh, 7), &[1, nh * dh]);
        let k = Tensor::from_vec(&ctx, &fill(s * nkv * dh, 11), &[s, nkv * dh]);
        let v = Tensor::from_vec(&ctx, &fill(s * nkv * dh, 22), &[s, nkv * dh]);
        let bench = |on: bool| {
            let (q, k, v) = (&q, &k, &v);
            async move {
                if on { std::env::remove_var("FERRIC_SPLITKV"); } else { std::env::set_var("FERRIC_SPLITKV", "1"); }
                let _ = q.fused_decode_attention(k, v, nh, nkv, dh).to_vec().await;
                let mut ms = Vec::new();
                for _ in 0..7 {
                    let t0 = std::time::Instant::now();
                    // Queue several, sync once: per-call readback latency would swamp the kernel.
                    let mut last = None;
                    for _ in 0..20 { last = Some(q.fused_decode_attention(k, v, nh, nkv, dh)); }
                    let _ = last.unwrap().to_vec().await;
                    ms.push(t0.elapsed().as_secs_f64() * 1000.0 / 20.0);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ms[3]
            }
        };
        let off = bench(false).await;
        let on = bench(true).await;
        println!("  {s:>8} {off:>9.3} ms {on:>9.3} ms {:>9.2}x", off / on);
    }
    std::env::remove_var("FERRIC_SPLITKV");

    // ---- the gate, in two dimensions, because one is not enough to set a constant ----
    //
    // The sweep above wins at EVERY length it tries, including its shortest. That is truncated rather
    // than clean: `decode_splits` refuses to split below 2*MIN_KEYS_PER_SPLIT (1024), so the sweep
    // started exactly at the first length the gate permits and never sampled the region the gate
    // governs.
    //
    // A first probe at splits in {1,2,4} showed 4 splits beating 1 by 2.2x at S=512 and 4.8x at S=768,
    // both below the gate. That is enough to say the threshold is wrong and NOT enough to say what it
    // should be: picking a constant off three columns would be fitting the rule to the sample. So this
    // sweeps both axes and reports the argmin, and the constant follows from the table.
    println!("\n  Gate calibration — context x split count. The heuristic currently refuses to split");
    println!("  below S=1024 and uses S/512 splits above it. Best column tells whether that is right.\n");
    const SPLITS: &[usize] = &[1, 2, 4, 8, 16];
    print!("  {:>7}", "S");
    for n in SPLITS { print!(" {:>9}", format!("{n} split")); }
    println!(" {:>7} {:>9}", "best", "vs 1");
    println!("  {:-<72}", "");
    // Three passes over the whole grid. The first version of this table reported an argmin from a
    // single pass and the winners bounced (16, 2, 16, 8, 8, 8, 16) while the cells they beat were
    // within 1-2% — that is an argmin of noise. What repeats show is which conclusions survive:
    // "some split beats no split" has a 2-10x effect and holds, "the optimum is exactly n" does not.
    const CAL_REPEATS: usize = 3;
    // (context, best split count or 0 for a tie, speedup of the best split over no-split)
    let mut recommend: Vec<(usize, usize, f64)> = Vec::new();
    for &s in &[128usize, 256, 512, 768, 1024, 2048, 4096] {
        let q = Tensor::from_vec(&ctx, &fill(nh * dh, 7), &[1, nh * dh]);
        let k = Tensor::from_vec(&ctx, &fill(s * nkv * dh, 11), &[s, nkv * dh]);
        let v = Tensor::from_vec(&ctx, &fill(s * nkv * dh, 22), &[s, nkv * dh]);
        // [split index][pass] so each cell can be reduced across passes.
        let mut grid: Vec<Vec<f64>> = SPLITS.iter().map(|_| Vec::new()).collect();
        for _ in 0..CAL_REPEATS {
            for (si, &n) in SPLITS.iter().enumerate() {
                std::env::set_var("FERRIC_SPLITKV", n.to_string());
                let _ = q.fused_decode_attention(&k, &v, nh, nkv, dh).to_vec().await;
                let mut ms = Vec::new();
                for _ in 0..7 {
                    let t0 = std::time::Instant::now();
                    let mut last = None;
                    for _ in 0..20 { last = Some(q.fused_decode_attention(&k, &v, nh, nkv, dh)); }
                    let _ = last.unwrap().to_vec().await;
                    ms.push(t0.elapsed().as_secs_f64() * 1000.0 / 20.0);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                grid[si].push(ms[3]);
            }
        }
        let times: Vec<f64> = grid.iter().map(|c| { let mut v = c.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }).collect();
        let bi = (0..times.len()).min_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap()).unwrap();
        // How much of a lead the winner actually has over the best of the rest. Under ~10% it is a
        // tie and naming a winner would be reporting noise.
        let runner = (0..times.len()).filter(|&i| i != bi)
            .map(|i| times[i]).fold(f64::INFINITY, f64::min);
        let lead = runner / times[bi] - 1.0;
        print!("  {s:>7}");
        for t in &times { print!(" {t:>9.3}"); }
        let tag = if lead < 0.10 { format!("{}~", SPLITS[bi]) } else { format!("{}", SPLITS[bi]) };
        println!(" {tag:>7} {:>8.2}x", times[0] / times[bi]);
        // TWO different questions, and collapsing them printed a wrong conclusion once already:
        //   (a) does ANY split beat no-split?  -> the gate question
        //   (b) WHICH split count is best?     -> the formula question
        // At S=512 every count from 2 to 16 beats 1-split by 1.79x while tying with each other. Marking
        // that row "tie" and then looking for the first non-tie row to answer (a) reported that
        // splitting first pays at S=4096, which the table plainly contradicts.
        let beats_one = times[0] / times[bi];
        recommend.push((s, if lead < 0.10 { 0 } else { SPLITS[bi] }, beats_one));
    }
    std::env::remove_var("FERRIC_SPLITKV");

    println!("\n  {CAL_REPEATS} passes over the grid, median per cell. A `~` marks a winner leading the");
    println!("  runner-up by under 10%, i.e. a tie the argmin should not be trusted to break.");
    println!("  Optimum per context: {}",
             recommend.iter().map(|(s, n, _)| if *n == 0 { format!("S={s}→tie") } else { format!("S={s}→{n}") })
                 .collect::<Vec<_>>().join("  "));
    // The gate question is answered by "does splitting beat not splitting", independent of which count
    // wins. 10% is the same tie threshold used above, applied to the right comparison this time.
    let first_split = recommend.iter().find(|(_, _, beats)| *beats > 1.10).map(|(s, _, _)| *s);
    match first_split {
        Some(s0) if s0 < 1024 => {
            println!("  ⚠ Splitting first pays at S={s0}, below the current gate of 1024, so the gate is");
            println!("    leaving a win on the floor for every context between {s0} and 1024.");
            // keys-per-split implied by the measured optimum at the smallest winning length
            let (_, n0, beats) = *recommend.iter().find(|(s, _, _)| *s == s0).unwrap();
            println!("    At S={s0} splitting beats not splitting by {beats:.2}x.");
            if n0 > 0 {
                println!("    Implied MIN_KEYS_PER_SPLIT there: {} (currently 512).", s0 / n0);
            } else {
                println!("    Every split count ties there, so the data fixes the GATE but not the");
                println!("    formula: lower the threshold, and leave the count rule until repeats");
                println!("    separate the counts.");
            }
        }
        Some(s0) => println!("  Splitting first pays at S={s0}; the gate at 1024 is consistent with that."),
        None => println!("  Splitting never pays in this range — the gate is right, or too permissive."),
    }
    println!("  Read the whole row before changing a constant: a single winning cell can be noise, and");
    println!("  the optimum drifting with S is the reason this is a formula and not a threshold.\n");

    println!("\n  Read the crossover, not the peak. Two dispatches replace one, so split-KV should LOSE");
    println!("  at short context and win as the cache grows — if it wins everywhere the gate is too");
    println!("  conservative, and if it never wins the heuristic is wrong for this device.");
    println!("\n  ⚠ Kernel-level numbers only. Whether this moves end-to-end tok/s depends on how much of");
    println!("  a decode step is attention at all — at short context it is a small share of a");
    println!("  weight-streaming problem. Confirm with a full decode benchmark before claiming a win.");
}
