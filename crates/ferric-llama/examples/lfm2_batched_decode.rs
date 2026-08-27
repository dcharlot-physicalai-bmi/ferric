//! **LFM2 batched decode: N sequences, one forward pass — and proof it changes nothing.**
//!
//! Decode is a weight-streaming problem. One token costs a full pass over every weight in the model, so
//! N sequences run as N separate forwards cost N× — the weights are re-read once per sequence with no
//! amortisation. `Lfm2::forward_batch` stacks the N tokens into `[N, d]` so every projection (the conv
//! block's in/out proj, q/k/v/o, the FFN, the head) reads its weights ONCE for all N rows.
//!
//! ## What this example exists to catch
//!
//! LFM2 carries **two** kinds of per-sequence state — KV for the attention blocks, and an `l_cache-1`
//! rolling window of the pre-conv signal for the conv blocks. Either one crossing between rows produces
//! **fluent, plausible text with finite logits and no error**. There is nothing to eyeball. So every
//! sequence is re-run ALONE through `Lfm2::forward` and compared.
//!
//! ### Two things this test needs in order to be able to fail at all
//!
//! **1. Real prompts.** The first version of this file used synthetic token ids (`100 + j + 29*i`), and
//! that made it a test that could not fail: garbage input drives this model straight into a fixed point
//! where it emits the SAME token forever (measured: seq 0 → `22883` repeated, seq 1 → `23123` repeated).
//! A greedy argmax over a saturated distribution is insensitive to almost any perturbation — deliberately
//! breaking `forward_batch` to share one RoPE position across all rows changed **nothing** in the token
//! ids. Real text keeps the model on a live decision boundary where a wrong position actually shows.
//!
//! **2. Logit comparison, not just argmax.** Even on real text, argmax discards everything but the
//! winner. So the full `[n_vocab]` logit row is compared per sequence per step and the max |Δ| reported.
//! That is the check with teeth; token equality is the property callers depend on.
//!
//! Prompt lengths differ, so every row sits at a different absolute position — a shared RoPE position
//! cannot cancel out. From n=3 up, the last two sequences are given the **same length but different
//! text**, so their positions agree and only genuinely per-sequence state (KV and the conv window) can
//! tell them apart. That is the case a position-only test would pass while the conv state was shared.
//!
//!   cargo run -q -p ferric-llama --example lfm2_batched_decode --release -- <lfm2.gguf>
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::lfm2::{Cache, Lfm2};
use std::sync::Arc;

const STEPS: usize = 24;

/// Not zero, and deliberately not: the projections are the SAME arithmetic on the same weights, but a
/// `[N, d]` matmul does not have to accumulate in the same ORDER as N separate `[1, d]` matmuls, and the
/// kernel picks its tiling from the row count. Measured on LFM2.5-1.2B-Q4_K_M: exactly 0.000e0 at n=2
/// (same kernel path as solo) and 4.768e-5 at n≥3 — float reassociation, ~800 ULP of 1.0 on logits of
/// magnitude ~30. The bound is set two orders above that so reassociation passes and anything structural
/// does not: a shared RoPE position moves these logits by ~1e0.
const LOGIT_TOL: f32 = 1e-3;

/// Eight distinct real texts, each long enough to survive truncation to the longest prompt length used
/// below. Distinct CONTENT is what separates the two equal-length sequences.
const TEXTS: [&str; 8] = [
    "The capital of France is Paris, and the capital city of Germany is Berlin, which sits on the river Spree in the north east.",
    "In 1969 a crew of three astronauts travelled to the Moon, and two of them walked on its surface while the third stayed in orbit above.",
    "Water boils at one hundred degrees Celsius at sea level, but on a high mountain the lower pressure means it boils at a cooler temperature.",
    "A short convolution keeps a rolling window of the previous timesteps, so the cost of one more token does not grow with how long the sequence already is.",
    "The library was quiet that afternoon, and the only sound came from the rain against the tall windows facing the garden and the old stone wall.",
    "To make bread you need flour, water, salt and yeast, and the most important ingredient after those four is simply enough time for the dough to rise.",
    "Every ship that leaves the harbour carries a log book, and the first entry of each voyage records the weather, the tide and the names of the crew.",
    "Electric motors turn current into torque, and the interesting engineering question is not peak power at all but how many joules the whole task costs.",
];

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }
/// The anti-saturation guard is itself checked, on a synthetic degenerate batch, before the model
/// loads. This exists because the guard shipped once counting distinct tokens over the FLATTENED
/// batch, and the one regime it is supposed to catch — every sequence stuck on its own repeated
/// token — scores `n` distinct that way and passes. A guard against vacuous tests that is itself
/// vacuous is worse than none: it reads like the hole is covered.
fn guard_self_check() {
    let degenerate: Vec<Vec<u32>> = vec![vec![22883; 24], vec![23123; 24]];
    let flattened: std::collections::BTreeSet<u32> = degenerate.iter().flatten().copied().collect();
    let per_seq: Vec<usize> = degenerate.iter()
        .map(|s| s.iter().copied().collect::<std::collections::BTreeSet<u32>>().len())
        .collect();
    assert!(flattened.len() > 1, "the OLD flattened guard admits this batch — that is the bug it had");
    assert!(!per_seq.iter().all(|&k| k > 1), "the per-sequence guard must REJECT a fully saturated batch");
    println!("  guard self-check: fully-saturated batch scores {} distinct flattened (old guard: PASS) \
              vs per-sequence {per_seq:?} (current guard: REJECT)", flattened.len());
}

async fn run() {
    // ⚠ The model path comes from argv and NOWHERE else. The qwen equivalence example once hardcoded a
    // checkpoint and ignored argv, so passing another model silently re-tested the hardcoded one — a
    // check that could not fail. No default here: a missing argument is an error, not a different model.
    let path = std::env::args().nth(1)
        .expect("usage: lfm2_batched_decode <model.gguf>  (the path is NOT defaulted on purpose)");
    println!("model: {path}");
    guard_self_check();

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(&path).expect("open");
    let m = Lfm2::load(&ctx, &g).expect("load");
    let n_attn = m.cfg.kv.iter().filter(|&&k| k > 0).count();
    println!("LFM2 · {} blocks ({} conv + {n_attn} attn) · d={} · l_cache={} · vocab={}",
             m.cfg.n_layer, m.cfg.n_layer - n_attn, m.cfg.d, m.cfg.conv_l, m.cfg.n_vocab);

    // ---- tokenizer, so the prompts are real text and the model stays off its degenerate fixed point ----
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer.ggml.tokens"),
    };
    let vocab: std::collections::HashMap<String, u32> =
        toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(p, q)| (p.into(), q.into())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);
    let bos = match (g.metadata.get("tokenizer.ggml.bos_token_id"), g.metadata.get("tokenizer.ggml.add_bos_token")) {
        (Some(Meta::U(b)), Some(Meta::Bool(true))) => Some(*b as u32),
        _ => None,
    };
    let full: Vec<Vec<u32>> = TEXTS.iter().map(|t| {
        let mut ids = bpe.encode(t);
        if let Some(b) = bos { ids.insert(0, b); }
        // The prompts below are PREFIXES of these; a text that tokenises shorter than the longest
        // prefix would panic on the slice, so say why here rather than at an opaque index.
        assert!(ids.len() >= 23, "text tokenises to {} tokens, need >= 23 for the longest prompt: {t:?}", ids.len());
        ids
    }).collect();

    let dec = |v: &[u32]| -> String {
        // GPT-2 byte-level unmapping, same as run_lfm2_cached.
        let mut mp = std::collections::HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
            let c = if p { b } else { let c = 256 + n; n += 1; c };
            mp.insert(char::from_u32(c).unwrap(), b as u8);
        }
        let s: String = v.iter().map(|&t| toks[t as usize].clone()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| mp.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };

    let vn = m.cfg.n_vocab;
    let am = |row: &[f32]| row.iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    println!("\nBatched decode — N sequences advanced by one token in ONE forward pass");
    println!("(each sequence then re-run ALONE; token ids AND full logit rows must match)\n");

    let mut worst = 0.0f32;
    for &n in &[2usize, 3, 4, 8] {
        // Unequal lengths ⇒ different absolute positions. From n=3 up, the LAST sequence deliberately
        // matches the second-to-last in LENGTH but not in TEXT, so those two are separated only by their
        // per-sequence state. n=2 keeps both lengths distinct — collapsing it to an equal-length pair
        // would have left no case where the positions differ.
        let plen = |i: usize| if n >= 3 && i + 1 == n { 5 + 3 * (i - 1) } else { 5 + 3 * i };
        let prompts: Vec<Vec<u32>> = (0..n).map(|i| full[i][..plen(i)].to_vec()).collect();

        // ---- reference: every sequence decoded entirely on its own ----
        let mut solo_tok: Vec<Vec<u32>> = Vec::new();
        let mut solo_lg: Vec<Vec<Vec<f32>>> = Vec::new();   // [seq][step][vocab]
        for p in &prompts {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward(p, &mut c).to_vec().await;
            let mut tok = am(&l[l.len() - vn..]);
            let (mut r#gen, mut lgs) = (vec![tok], Vec::new());
            for _ in 1..STEPS {
                let l = m.forward(&[tok], &mut c).to_vec().await;
                lgs.push(l[l.len() - vn..].to_vec());
                tok = am(&l[l.len() - vn..]);
                r#gen.push(tok);
            }
            solo_tok.push(r#gen);
            solo_lg.push(lgs);
        }

        // ---- batched: same sequences, one forward per step for all N ----
        // Prefill stays per-sequence (the prompts have different lengths); only decode is batched, which
        // is the part that repeats and the part that is weight-bound.
        let mut caches: Vec<Cache> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        for p in &prompts {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward(p, &mut c).to_vec().await;
            next.push(am(&l[l.len() - vn..]));
            caches.push(c);
        }
        let mut bat_tok: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
        let mut dmax = 0.0f32;
        let mut first_off: Option<(usize, usize, f32)> = None;   // (step, seq, delta)
        for s in 1..STEPS {
            let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let logits = m.forward_batch(&next, &mut refs).to_vec().await;
            for i in 0..n {
                let row = &logits[i * vn..(i + 1) * vn];
                // The check with teeth: the WHOLE distribution, not just its argmax. Argmax alone is a
                // weak detector — a deliberately shared RoPE position took until step 23 to flip a token
                // here, while it moves the logits on the very first batched step.
                let mut d = 0.0f32;
                for (a, b) in row.iter().zip(&solo_lg[i][s - 1]) { d = d.max((a - b).abs()); }
                dmax = dmax.max(d);
                if d > LOGIT_TOL && first_off.is_none() { first_off = Some((s, i, d)); }
                next[i] = am(row);
                bat_tok[i].push(next[i]);
            }
        }
        worst = worst.max(dmax);
        if let Some((s, i, d)) = first_off {
            println!("  ❌ n={n}: sequence {i} logits differ from solo decode by {d:.3e} at step {s}");
        }
        assert!(first_off.is_none(),
                "n={n}: batched logits are not solo-equivalent (max |Δ| {dmax:.3e} > {LOGIT_TOL:.0e}). \
                 The token ids may still agree — argmax hides small crossings — so this is the check \
                 that catches a shared RoPE position or a conv window read from the wrong row.");

        let mut bad = false;
        for i in 0..n {
            if let Some(s) = solo_tok[i].iter().zip(&bat_tok[i]).position(|(a, b)| a != b) {
                bad = true;
                println!("  ❌ n={n}: SEQUENCE {i} DIVERGED at step {s}: solo {} vs batched {}",
                         solo_tok[i][s], bat_tok[i][s]);
                println!("       solo    {:?}", solo_tok[i]);
                println!("       batched {:?}", bat_tok[i]);
            }
        }
        assert!(!bad, "n={n}: batched decode is not solo-equivalent. Batching must change only HOW the \
                       work is scheduled — a crossed KV cache, a shared RoPE position or a conv window \
                       read from the wrong row yields fluent, wrong text with no error.");
        // A saturated model would make the token check vacuous, so say out loud how many DISTINCT tokens
        // the sequences actually produced. This MUST be counted PER SEQUENCE, not over the flattened
        // batch: the exact degenerate regime this guard exists to catch is each sequence stuck on its
        // own repeated token (seq 0 → 22883 forever, seq 1 → 23123 forever), which scores 2 distinct
        // tokens flattened and sails through. Every sequence has to move on its own.
        let per_seq: Vec<usize> = bat_tok.iter()
            .map(|s| s.iter().copied().collect::<std::collections::BTreeSet<u32>>().len())
            .collect();
        let distinct: std::collections::BTreeSet<u32> = bat_tok.iter().flatten().copied().collect();
        println!("  ✅ n={n}: all {n} sequences token-identical to solo decode over {STEPS} steps");
        println!("       prompt lengths {:?} · max |Δlogit| {dmax:.3e} · {} distinct tokens generated",
                 prompts.iter().map(|p| p.len()).collect::<Vec<_>>(), distinct.len());
        assert!(per_seq.iter().all(|&k| k > 1),
                "a sequence emitted one repeated token for all {STEPS} steps (per-sequence distinct \
                 counts {per_seq:?}) — that sequence is saturated and its token comparison cannot \
                 detect anything, however many distinct tokens the OTHER sequences contributed. \
                 Change the prompts.");
        if n == 4 {
            for i in 0..n {
                println!("       seq {i} ({} tok): …{}⟩⟩{}", prompts[i].len(),
                         dec(&prompts[i][prompts[i].len().saturating_sub(4)..]).trim_end(),
                         dec(&bat_tok[i][..8]));
            }
        }
    }
    println!("\n  worst max |Δlogit| across every sequence and step: {worst:.3e}");

    // ---- throughput: does reading the weights once for N rows actually pay? ----
    let Some(load) = load_avg() else { return };
    let forced = std::env::var("FERRIC_FORCE_TIMING").is_ok();
    println!("\n  machine load average: {load:.2}{}", if forced { " (FERRIC_FORCE_TIMING set)" } else { "" });
    if load >= 8.0 && !forced {
        println!("  ⚠ too loaded to time anything — the equivalence above still holds. Re-run when quiet,");
        println!("    or set FERRIC_FORCE_TIMING=1 to time anyway and read the ratio as a lower bound.");
        return;
    }
    if load >= 8.0 {
        println!("  ⚠ timing under load {load:.2}: both sides are contended, so read the ratio as");
        println!("    indicative and the absolute ms as meaningless. Equivalence above is unaffected.");
    }

    println!("\n  {:>6} {:>13} {:>13} {:>11}", "seqs", "N separate", "1 batched", "speedup");
    println!("  {:-<48}", "");
    let mut best = 0.0f64;
    for &n in &[1usize, 2, 4, 8] {
        let mut ca: Vec<Cache> = (0..n).map(|_| Cache::new(&m.cfg)).collect();
        let mut na = vec![0u32; n];
        for (i, c) in ca.iter_mut().enumerate() {
            let l = m.forward(&full[i][..6], c).to_vec().await;
            na[i] = am(&l[l.len() - vn..]);
        }
        let mut sep = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                for i in 0..n {
                    let l = m.forward(&[na[i]], &mut ca[i]).to_vec().await;
                    na[i] = am(&l[l.len() - vn..]);
                }
            }
            sep.push(t0.elapsed().as_secs_f64() * 1000.0 / 8.0);
        }
        sep.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut cb: Vec<Cache> = (0..n).map(|_| Cache::new(&m.cfg)).collect();
        let mut nb = vec![0u32; n];
        for (i, c) in cb.iter_mut().enumerate() {
            let l = m.forward(&full[i][..6], c).to_vec().await;
            nb[i] = am(&l[l.len() - vn..]);
        }
        let mut bt = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                let mut refs: Vec<&mut Cache> = cb.iter_mut().collect();
                let lg = m.forward_batch(&nb, &mut refs).to_vec().await;
                for i in 0..n { nb[i] = am(&lg[i * vn..(i + 1) * vn]); }
            }
            bt.push(t0.elapsed().as_secs_f64() * 1000.0 / 8.0);
        }
        bt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sp = sep[2] / bt[2];
        best = best.max(sp);
        println!("  {n:>6} {:>10.2} ms {:>10.2} ms {:>10.2}x", sep[2], bt[2], sp);
    }

    println!("\n  ✅ identical tokens, {best:.2}x throughput at the best batch size. The projections read");
    println!("     the weights once for N rows instead of N times; attention and the short-conv stay");
    println!("     per-sequence loops because each row owns its own history — a KV of its own length and");
    println!("     its own rolling pre-conv window.");
}
