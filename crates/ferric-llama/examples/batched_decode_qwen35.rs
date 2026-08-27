//! **Batched decode on the Qwen3.5 hybrid** (qwen35 / qwen35moe / laguna) — identical output, one pass.
//!
//! `Qwen35::forward_batch` advances N independent sequences by one token each in a single forward, so the
//! projections read the weights ONCE for N tokens instead of N times. Everything that carries
//! per-sequence state stays a per-row loop — and on this architecture that is three separate things:
//! the KV cache on the full-attention layers, the gated-delta-net recurrent state `S [nv, dv, dk]`, and
//! the GDN short conv's carried tail.
//!
//! ## Why this example exists, and why it asserts on token ids
//!
//! A batched path that crosses sequences does not crash. Feed `gated_delta_rule_stateful` a stacked
//! `[N, …]` and it treats the N rows as N consecutive timesteps of ONE sequence, threading one state
//! through all of them; share a RoPE position and every row still rotates by a valid angle. Both produce
//! finite logits and fluent, plausible text. There is no symptom to eyeball. So each sequence is re-run
//! ALONE through `forward_cached` and the token ids are compared exactly.
//!
//! Two things this deliberately does that the dense runtime's equivalence example did not:
//!   * the model path comes from argv and is PRINTED — the dense example hardcoded a Qwen path and
//!     ignored argv, so it silently tested a model with no rope scaling and the equivalence claim never
//!     covered the scaled branch at all. A test that cannot fail is not a test.
//!   * prompt lengths DIFFER per sequence, so every row sits at a different absolute position. With equal
//!     lengths a shared position would cancel out and the check would pass for the wrong reason.
//!
//!   cargo run -q -p ferric-llama --example batched_decode_qwen35 --release -- <model.gguf> [steps]
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen35::{Cache, Mixer, Qwen35};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

/// 1-minute load average, for deciding whether the wall-clock numbers below mean anything.
fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

async fn run() {
    // ⚠ argv, never a hardcoded path: this runtime serves qwen35 (GDN + full attention, dense FFN),
    // qwen35moe and laguna (sliding-window + YaRN full attention, MoE FFN). They exercise DIFFERENT
    // mixers, so pinning one checkpoint would leave the others' batched path unverified.
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: batched_decode_qwen35 <model.gguf> [steps]");
        std::process::exit(2);
    });
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    println!("model: {path}");
    println!("steps: {steps}");

    let ctx = Arc::new(Context::new().await.unwrap());
    let t_load = std::time::Instant::now();
    let g = GgufFile::open(&path).unwrap();
    let m = Qwen35::load(&ctx, &g).unwrap();
    let n_layer = m.cfg.n_layer;
    let vn = m.cfg.n_vocab;
    println!("arch: {}  layers: {n_layer}  n_embd: {}  vocab: {vn}  moe: {}  loaded in {:.1}s",
             m.cfg.arch, m.cfg.n_embd, m.cfg.is_moe(), t_load.elapsed().as_secs_f64());
    // Count the mixers that were actually LOADED, not what the interval formula predicts: which batched
    // helper each layer takes is decided by tensor presence, and that is the thing under test here.
    let (mut n_gdn, mut n_attn, mut n_lag) = (0usize, 0usize, 0usize);
    for l in &m.layers {
        match l.mixer { Mixer::Gdn(_) => n_gdn += 1, Mixer::Attn(_) => n_attn += 1, Mixer::Lag(_) => n_lag += 1 }
    }
    println!("mixers loaded: {n_gdn} gated-delta-net, {n_attn} full-attention, {n_lag} laguna\n");
    assert!(m.batching_supported(), "this checkpoint refuses batching — nothing to verify");

    // Greedy argmax over one logits row.
    let am = |row: &[f32]| row.iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    // ⚠ Wall-clock on a contended machine is not a measurement. An earlier run of this example on a
    // box at load 24 printed "142.55x" at n=2 and "2.07x" at n=4 for the same work — the second number
    // is plausible and the first is pure scheduler noise, and nothing in the output said so. Timing is
    // therefore reported only when the machine is quiet; correctness is asserted either way, because
    // that is what this example is for.
    let load = load_avg().unwrap_or(0.0);
    let quiet = load < 4.0;
    println!("machine load average: {load:.2}{}", if quiet { "" } else { "  (too loaded to time — correctness only)" });

    println!("\nBatched decode — N sequences in one forward pass\n");

    for &n in &[2usize, 4] {
        // UNEQUAL prompt lengths on purpose: sequence i is 5 + 3i tokens, so after prefill the rows sit
        // at positions 5, 8, 11, 14. A crossed cache or a shared position cannot cancel out.
        let prompts: Vec<Vec<u32>> = (0..n)
            .map(|i| (0..(5 + 3 * i)).map(|j| ((100 + j + 29 * i) % vn) as u32).collect())
            .collect();

        // ---- reference: each sequence decoded entirely on its own ----
        // Prefill is timed separately from decode. Both paths prefill per sequence (the prompts have
        // different lengths — that IS prefill's shape), so folding it into the comparison would dilute
        // the number by work neither path batches. Only the decode steps are the claim.
        let mut ref_out: Vec<Vec<u32>> = Vec::new();
        // Solo logits per sequence per step, kept so the comparison can be made on the CONTINUOUS
        // quantity as well as the discrete one. See the max|Δ| check below for why that matters.
        let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut t_solo = 0.0f64;
        for p in &prompts {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward_cached(p, &mut c, n_layer).to_vec().await;
            let mut tok = am(&l[l.len() - vn..]);
            let mut r#gen = vec![tok];
            let mut rows: Vec<Vec<f32>> = Vec::new();
            let t0 = std::time::Instant::now();
            for _ in 1..steps {
                let l = m.forward_cached(&[tok], &mut c, n_layer).to_vec().await;
                rows.push(l[l.len() - vn..].to_vec());
                tok = am(&l[l.len() - vn..]);
                r#gen.push(tok);
            }
            t_solo += t0.elapsed().as_secs_f64();
            ref_out.push(r#gen);
            ref_logits.push(rows);
        }

        // ---- batched: same sequences, ONE forward per step ----
        let mut caches: Vec<Cache> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        for p in &prompts {
            let mut c = Cache::new(&m.cfg);
            let l = m.forward_cached(p, &mut c, n_layer).to_vec().await;
            next.push(am(&l[l.len() - vn..]));
            caches.push(c);
        }
        let mut bat_out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
        let mut max_dlogit = 0.0f32;      // largest |solo − batched| logit seen, over every step and row
        let mut worst: (usize, usize) = (0, 0); // (sequence, step) where it happened
        let t1 = std::time::Instant::now();
        for s in 1..steps {
            let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let logits = m.forward_batch(&next, &mut refs).to_vec().await;
            assert_eq!(logits.len(), n * vn, "forward_batch must return one logits row per sequence");
            for i in 0..n {
                let row = &logits[i * vn..(i + 1) * vn];
                // Valid comparison because both paths were fed the same token at this step — which the
                // token assert below independently confirms. If tokens ever disagree, that assert is the
                // one that fires and this number stops meaning anything.
                let d = row.iter().zip(&ref_logits[i][s - 1]).fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
                if d > max_dlogit { max_dlogit = d; worst = (i, s); }
                next[i] = am(row);
                bat_out[i].push(next[i]);
            }
        }
        let t_bat = t1.elapsed().as_secs_f64();

        // ---- the bar: token-identical, per sequence, named on failure ----
        let mut bad = 0usize;
        for i in 0..n {
            if ref_out[i] != bat_out[i] {
                bad += 1;
                let at = ref_out[i].iter().zip(&bat_out[i]).position(|(a, b)| a != b);
                eprintln!("\n❌ SEQUENCE {i} of {n} DIVERGED (prompt len {}, start pos {})",
                          prompts[i].len(), prompts[i].len());
                eprintln!("   first mismatch at generated token index {at:?}");
                eprintln!("   solo   : {:?}", ref_out[i]);
                eprintln!("   batched: {:?}", bat_out[i]);
            }
        }
        assert_eq!(bad, 0,
            "{bad} of {n} sequences diverged from solo decode. Batching must change only HOW the work is \
             scheduled — a crossed KV cache, a delta-rule state threaded through the wrong row, or a \
             shared RoPE position produces fluent, wrong text with no error at all.");

        // ---- the sharper bar: the LOGITS, not just the argmax over them ----
        //
        // Token equality is a coarse detector and this example has the receipt. A deliberately sabotaged
        // build that forced every row to share sequence 0's RoPE position still produced token-identical
        // output at n=2 — greedy argmax is a step function, and on a degenerate distribution a genuinely
        // wrong logit vector can land on the same top token for several steps running. The discrete check
        // passed while the model was, in fact, wrong.
        //
        // So the continuous quantity is checked too, and the threshold is placed from measurement rather
        // than taste. On Bonsai-27B: a correct batched path measured max|Δlogit| = 0.00001, and that same
        // shared-position sabotage measured 0.5934 — five orders of magnitude apart. 0.1 sits in the gap
        // with ~10⁴× headroom over f32 reassociation noise (matmul tiling differs between an [N, d] and a
        // [1, d] input) and ~6× margin under the smallest real leak observed.
        let tol = 0.1f32;
        assert!(max_dlogit < tol,
            "max |Δlogit| = {max_dlogit:.4} at sequence {} step {} — above the {tol} f32-reassociation \
             band. The tokens matched, but the LOGITS did not: this is a real cross-sequence leak that \
             greedy decoding happened to round away.", worst.0, worst.1);

        println!("  n={n}: all {n} sequences TOKEN-IDENTICAL to solo decode over {steps} steps");
        println!("        max |Δlogit| vs solo = {max_dlogit:.5} (f32 reassociation only; tol {tol})");
        println!("        prompt lengths {:?}  first seq {:?}",
                 prompts.iter().map(|p| p.len()).collect::<Vec<_>>(), &ref_out[0][..ref_out[0].len().min(8)]);
        if quiet {
            println!("        decode only ({} steps): {t_solo:.2}s as {n} separate streams vs {t_bat:.2}s \
                      batched  ({:.2}x)", steps - 1, t_solo / t_bat);
        }
        println!();
    }

    println!("✅ forward_batch is solo-equivalent on this checkpoint.");
    println!("   The projections (in_proj / qkv / q,k,v,o / gate_up / down / lm_head) read the weights once");
    println!("   for N tokens. The KV histories, the gated-delta-net recurrent state and the conv tail stay");
    println!("   per-sequence loops, because each belongs to one stream and to no other.");
}
