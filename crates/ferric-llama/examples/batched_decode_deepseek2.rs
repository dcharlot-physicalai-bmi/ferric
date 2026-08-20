//! **Batched decode on DeepSeek-V2 (MLA + DeepSeekMoE)** — N sequences, one forward pass, identical output.
//!
//! Decode is a weight-streaming problem: one token costs a full pass over the checkpoint's weights.
//! `forward_batch` stacks N sequences' next tokens into `[N, d]` so `attn_q`, `attn_kv_a_mqa`,
//! `attn_kv_b`, `attn_output`, the router, the shared expert and the LM head are read **once for N
//! tokens** instead of N times. Attention itself stays a per-sequence loop, because sequence `i`
//! attends its own latent KV history at its own position and those histories differ in length.
//!
//! ## Why this example exists, and why it compares token ids
//!
//! A batched path that leaked between sequences — a shared RoPE position, a KV row appended to the
//! wrong cache, the unscaled/NEOX rope instead of DeepSeek's YaRN-scaled NORM rope — still produces
//! **fluent text and finite logits**. There is no crash, no NaN, no shape mismatch. The only check
//! that can fail is running each sequence ALONE and comparing the token ids exactly, which is what
//! this does.
//!
//! Three things about this architecture make position leakage especially easy and especially quiet:
//!
//!   * the rope lanes are only 64 of a 192-wide query head, so a wrong rotation perturbs a third of Q
//!     and none of the 128 nope lanes — the output stays in-distribution;
//!   * the rope vector is SHARED across all 16 heads on the key side, so one bad position is one bad
//!     vector broadcast everywhere rather than an obviously broken head;
//!   * YaRN's per-dimension frequency scale lives in a separate table (`self.yarn`); dropping it is
//!     the difference between a 163k-context model and a 4k one, with no error either way.
//!
//! So the sequences are deliberately given DIFFERENT prompt lengths: at equal lengths a shared
//! position would coincide with the correct one and the check would pass for the wrong reason.
//!
//! ```text
//!   cargo run -q -p ferric-llama --example batched_decode_deepseek2 --release -- <model.gguf>
//! ```
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::deepseek2::{Cache, DeepSeek2};
use std::sync::Arc;

const STEPS: usize = 16;

/// 1-minute load average, or None if `uptime` cannot be parsed.
///
/// Used to REFUSE to report a speedup rather than print a wrong one. Measured here on a shared
/// machine, the same binary and the same weights gave 42.42x at n=2 and 1.61x at n=4 in one run —
/// the 42x is another process releasing the GPU, not amortisation. A throughput number nobody can
/// reproduce is worse than no throughput number; the equivalence claim is what this example is for.
fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }

async fn run() {
    // ⚠ From argv, never hardcoded. The dense runtime's equivalence example hardcoded a Qwen path and
    // ignored argv, so passing another checkpoint silently tested Qwen instead — a test that could not
    // fail, and the reason a real rope divergence shipped unnoticed.
    let path = std::env::args().nth(1).expect(
        "usage: batched_decode_deepseek2 <model.gguf>  (e.g. DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf)");
    println!("model: {path}");

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(&path).expect("open");
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    assert_eq!(arch, "deepseek2", "this example is for deepseek2, got {arch:?}");

    let t0 = std::time::Instant::now();
    let m = DeepSeek2::load(&ctx, &g).expect("load deepseek2");
    let c = &m.cfg;
    println!("deepseek2 · {} blocks, d={}, {} heads · loaded in {:.2?}", c.n_layer, c.d, c.n_head, t0.elapsed());
    println!("  MLA: qk {} = {} nope + {} rope (rope SHARED across heads), v {}, latent {}",
             c.qk_head(), c.qk_nope, c.qk_rope, c.v_head, c.kv_lora_rank);
    println!("  MoE: {} experts, {} used, {} shared, {} dense lead · renorm {}",
             c.n_expert, c.n_expert_used, c.n_expert_shared, c.dense_lead, c.expert_norm);
    println!("  YaRN: factor {} → mscale {:.5}, attn_factor {:.5}, q prescale {:.5}",
             c.yarn_factor, c.mscale(), c.attn_factor(), c.q_prescale());

    let vn = c.n_vocab;
    // Greedy: the batched and solo paths must agree on the ARGMAX, which is the strictest thing that
    // survives the reduction-order differences a [N, d] matmul has against N separate [1, d] ones.
    let am = |row: &[f32]| row.iter().enumerate()
        .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;

    println!("\nBatched decode — N sequences in one forward pass\n");

    for &n in &[2usize, 4] {
        // Deliberately UNEQUAL prompt lengths, so no two sequences share a position at any step —
        // at equal lengths a shared position would coincide with the right one and this whole example
        // would pass for the wrong reason.
        //
        // And deliberately LONG (40..109, not 5..14): measured here, dropping the YaRN frequency table
        // from the batched rope only — one of the two divergences this example exists to catch —
        // produced IDENTICAL tokens at prompt lengths 5 and 8 and first diverged at step 7 of n=4. At
        // positions under ~25 the per-dimension stretch barely moves the angle, so short prompts make
        // the check blind to exactly the bug it is for. Longer prompts also spread the four sequences
        // 40..~125 apart, so a crossed cache cannot land on a plausible history.
        let prompts: Vec<Vec<u32>> = (0..n)
            .map(|i| (0..(40 + 23 * i)).map(|j| (1200 + j + 137 * i) as u32).collect())
            .collect();
        for p in &prompts { assert!(p.iter().all(|&t| (t as usize) < vn), "token id out of vocab"); }

        // Warm-up on throwaway caches: the first batched forward at a given width compiles a pipeline
        // for every shape-specialised kernel it touches, which on the first run cost 49 s of the 50 s
        // "batched" time and reported a nonsensical 0.16x. Timing the second pass onward measures the
        // model, not the shader compiler.
        {
            let mut warm: Vec<Cache> = (0..n).map(|_| Cache::new(c)).collect();
            let mut tok = vec![0u32; n];
            for (i, wc) in warm.iter_mut().enumerate() {
                let l = m.forward(&prompts[i], wc).to_vec().await;
                tok[i] = am(&l[l.len() - vn..]);
            }
            let mut refs: Vec<&mut Cache> = warm.iter_mut().collect();
            let _ = m.forward_batch(&tok, &mut refs).to_vec().await;
        }

        // ---- reference: each sequence decoded entirely on its own ----
        // Only the DECODE forwards are timed; prefill is excluded from both sides so the comparison
        // below is the same work counted the same way.
        let mut solo: Vec<Vec<u32>> = Vec::new();
        let mut solo_ms = 0.0f64;
        for p in &prompts {
            let mut cache = Cache::new(c);
            let l = m.forward(p, &mut cache).to_vec().await;
            let mut tok = am(&l[l.len() - vn..]);
            let mut gen = vec![tok];
            for _ in 1..STEPS {
                let t0 = std::time::Instant::now();
                let l = m.forward(&[tok], &mut cache).to_vec().await;
                solo_ms += t0.elapsed().as_secs_f64() * 1000.0;
                tok = am(&l[l.len() - vn..]);
                gen.push(tok);
            }
            solo.push(gen);
        }

        // ---- batched: the same sequences, one forward per step ----
        // Prefill stays per-sequence (prompts have different lengths); only decode is batched, which
        // is the case that matters — prefill is already compute-bound and needs no amortising.
        let mut caches: Vec<Cache> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        for p in &prompts {
            let mut cache = Cache::new(c);
            let l = m.forward(p, &mut cache).to_vec().await;
            next.push(am(&l[l.len() - vn..]));
            caches.push(cache);
        }
        let mut bat: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
        let t0 = std::time::Instant::now();
        for _ in 1..STEPS {
            let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let logits = m.forward_batch(&next, &mut refs).to_vec().await;
            assert_eq!(logits.len(), n * vn, "forward_batch must return one logit row per sequence");
            assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits out of forward_batch");
            for i in 0..n {
                next[i] = am(&logits[i * vn..(i + 1) * vn]);
                bat[i].push(next[i]);
            }
        }
        let bat_ms = t0.elapsed().as_secs_f64() * 1000.0;

        for i in 0..n {
            if solo[i] != bat[i] {
                let at = solo[i].iter().zip(&bat[i]).position(|(a, b)| a != b).unwrap_or(0);
                panic!("SEQUENCE {i} of {n} DIVERGED at step {at} (prompt len {}): solo {:?} vs batched {:?}\n\
                        Batching must change only HOW the work is scheduled. A crossed cache, a shared \
                        RoPE position, or a rope kernel that dropped the YaRN scale / NORM pairing \
                        produces exactly this — fluent, wrong text with no error.",
                       prompts[i].len(), solo[i], bat[i]);
            }
        }
        println!("  n={n}: all {n} sequences IDENTICAL to solo decode over {STEPS} steps \
                  (prompt lengths {:?})", prompts.iter().map(|p| p.len()).collect::<Vec<_>>());
        println!("        first tokens {:?}", bat.iter().map(|g| g[0]).collect::<Vec<_>>());
        // Same n × (STEPS-1) decoded tokens on both sides, prefill excluded. Reported, never asserted:
        // a throughput floor that trips because another process took the GPU would say nothing about
        // the equivalence above, which is the only claim this example makes.
        let load = load_avg().unwrap_or(0.0);
        if load >= 4.0 {
            println!("        decode wall time: {solo_ms:.0} ms separate vs {bat_ms:.0} ms batched \
                      — NOT A MEASUREMENT, load average {load:.2}. Re-run on a quiet machine.");
        } else {
            println!("        decode wall time: {solo_ms:.0} ms as {n} separate streams vs {bat_ms:.0} \
                      ms batched, {} tokens each ({:.2}x, load {load:.2})",
                     STEPS - 1, solo_ms / bat_ms);
        }
    }

    println!("\n  ✅ token-identical at n=2 and n=4. The projections, the router, the shared expert and");
    println!("     the LM head read their weights once for N tokens; MLA attention stays a per-sequence");
    println!("     loop because each sequence's latent KV history is a different length.");
}
