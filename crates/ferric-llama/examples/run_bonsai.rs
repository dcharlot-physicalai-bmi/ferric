//! Run **PrismML Ternary Bonsai-27B** (Qwen3.5 hybrid, Q2_0 ternary) on Ferric.
//!
//! Takes token IDs directly rather than a prompt string, so a comparison against PrismML's
//! llama.cpp fork isolates the *model* — a tokenizer discrepancy can't be mistaken for a
//! numerical one. Prints the top-k next-token distribution.
//!
//!   cargo run --release -p ferric-llama --example run_bonsai -- <model.gguf> <id,id,...> [--layers N]
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen35::Qwen35;
use std::sync::Arc;

/// GPT-2 byte↔printable-unicode bijection, inverted — turns a vocab entry back into raw bytes.
fn unicode_to_byte() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u16> = Vec::new();
    bs.extend(b'!' as u16..=b'~' as u16);
    bs.extend(0xA1u16..=0xAC);
    bs.extend(0xAEu16..=0xFF);
    let mut m = std::collections::HashMap::new();
    let mut extra = 0u32;
    for b in 0u16..256 {
        let c = if bs.contains(&b) { char::from_u32(b as u32).unwrap() } else { let c = char::from_u32(256 + extra).unwrap(); extra += 1; c };
        m.insert(c, b as u8);
    }
    m
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: run_bonsai <model.gguf> <id,id,...> [--layers N]");
    let ids: Vec<u32> = args.get(2).map(|s| s.split(',').map(|x| x.trim().parse().expect("bad token id")).collect()).unwrap_or_default();
    let layers = args.iter().position(|a| a == "--layers").map(|i| args[i + 1].parse::<usize>().unwrap());

    let t0 = std::time::Instant::now();
    let g = GgufFile::open(path).unwrap();
    println!("GGUF: {} tensors, opened in {:?}", g.tensors.len(), t0.elapsed());

    // vocab (for decoding the result only)
    let vocab: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    };
    let u2b = unicode_to_byte();
    let detok = |s: &str| -> String { String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned() };

    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let m = Qwen35::load(&ctx, &g).unwrap();
    let c = &m.cfg;
    let n_recr = (0..c.n_layer).filter(|&i| c.is_recurrent(i)).count();
    println!(
        "loaded in {:?}\n  {} layers ({} gated-delta-net + {} full-attn) · d={} · ff={} · vocab={}\n  GDN: {} k-heads × {} v-heads, head_k={} head_v={}, conv={} · attn: {}q/{}kv × {}",
        t0.elapsed(), c.n_layer, n_recr, c.n_layer - n_recr, c.n_embd, c.n_ff, c.n_vocab,
        c.n_k_heads, c.n_v_heads, c.head_k_dim, c.head_v_dim(), c.conv_kernel, c.n_head, c.n_head_kv, c.head_dim
    );

    if args.iter().any(|a| a == "--ffnbench") {
        for t in [1usize, 2, 3, 5, 8] {
            println!("  ffn[1] t={t}: {:.2} ms (median of 20)", m.bench_ffn(1, t, 20));
        }
        return;
    }
    // `--fwdbench` — median cost of the speculative-verify shape (snapshot → t-token forward →
    // rollback) at t=1..3 against a warm cache. Isolates the verify's t-scaling.
    if args.iter().any(|a| a == "--fwdbench") {
        let mut cache = ferric_llama::qwen35::Cache::new(c);
        let (lg, _hid) = m.forward_spec(&ids, &mut cache, c.n_layer);
        lg.to_vec().await;
        for t in [1usize, 2, 3, 1, 2, 3] {
            let toks: Vec<u32> = ids.iter().copied().cycle().take(t).collect();
            let mut times = Vec::new();
            for i in 0..14 {
                let snap = cache.snapshot();
                let t0 = std::time::Instant::now();
                let (lg, _h) = m.forward_spec(&toks, &mut cache, c.n_layer);
                lg.to_vec().await;
                if i >= 2 { times.push(t0.elapsed().as_secs_f64() * 1e3); } // 2 warmups
                cache = snap;
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ferric_tensor::reset_op_counters();
            let snap = cache.snapshot();
            let (lg, _h) = m.forward_spec(&toks, &mut cache, c.n_layer);
            lg.to_vec().await;
            cache = snap;
            let (disp, subs) = ferric_tensor::op_counters();
            println!("  verify t={t}: min {:.1} · median {:.1} ms (12 runs) · {disp} dispatches, {subs} submits", times[0], times[6]);
        }
        return;
    }
    if ids.is_empty() { println!("(no token ids given — load-only)"); return; }
    println!("\ntokens: {ids:?}");
    for &i in &ids { print!("{}", detok(vocab.get(i as usize).map(|s| s.as_str()).unwrap_or("?"))); }
    println!();

    let t0 = std::time::Instant::now();
    let logits = m.forward_upto(&ids, layers.unwrap_or(c.n_layer));
    let v = logits.to_vec().await;
    let dt = t0.elapsed();

    let last = &v[(ids.len() - 1) * c.n_vocab..];
    let finite = last.iter().all(|x| x.is_finite());
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("\nforward {:?} ({} tok){}", dt, ids.len(), if layers.is_some() { format!(" [first {} layers]", layers.unwrap()) } else { String::new() });
    println!("  finite: {finite}");
    println!("  top-8 next-token:");
    for &i in idx.iter().take(8) {
        println!("    {:>10.5}  [{:>6}] {:?}", last[i], i, detok(vocab.get(i).map(|s| s.as_str()).unwrap_or("?")));
    }

    // `--ref id=logit,id=logit,...` — compare against reference logits (PrismML's llama.cpp fork,
    // via examples/dumplogits) on the SAME token ids, so this is a pure model-vs-model check.
    if let Some(p) = args.iter().position(|a| a == "--ref") {
        let mut worst = 0.0f32;
        println!("\n  vs reference (PrismML llama.cpp fork):");
        for pair in args[p + 1].split(',') {
            let (id, r) = pair.split_once('=').expect("--ref wants id=logit");
            let (id, r): (usize, f32) = (id.parse().unwrap(), r.parse().unwrap());
            let d = (last[id] - r).abs();
            worst = worst.max(d);
            println!("    [{:>6}] ferric {:>10.5}   ref {:>10.5}   Δ {:.2e}  {:?}", id, last[id], r, d, detok(vocab.get(id).map(|s| s.as_str()).unwrap_or("?")));
        }
        // logits run to ~15; f32 GPU reductions over 64 layers in a different op order than ggml
        // will not be bit-identical, so agreement at ~1e-2 on this scale is the real bar.
        let pass = worst < 5e-2;
        println!("\n{} max|Δ| = {worst:.2e} across {} layers / 27B ternary params",
            if pass { "✅ Ferric matches the reference implementation" } else { "❌ MISMATCH" }, c.n_layer);
        assert!(pass, "logits diverge from reference: {worst:.2e}");
    }
    assert!(finite, "non-finite logits");

    // `--gen N` — greedy continuation using the KV + recurrent-state cache: prompt once, then one
    // token per step. `--reprefill` forces the naive path (re-run the whole prefix each token),
    // which must produce identical text — that equivalence is the cache's correctness test.
    if let Some(p) = args.iter().position(|a| a == "--gen") {
        let n: usize = args[p + 1].parse().unwrap();
        let reprefill = args.iter().any(|a| a == "--reprefill");
        use std::io::Write;
        let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;

        print!("\ngenerating ({}): ", if reprefill { "re-prefill each token" } else { "cached decode" });
        for &i in &ids { print!("{}", detok(vocab.get(i as usize).map(|s| s.as_str()).unwrap_or("?"))); }
        std::io::stdout().flush().ok();

        let mut seq = ids.clone();
        let t0 = std::time::Instant::now();
        let mut cache = ferric_llama::qwen35::Cache::new(c);
        let mut prompt_ms = 0.0;
        for step in 0..n {
            let logits = if reprefill {
                m.forward(&seq)
            } else if step == 0 {
                m.forward_cached(&ids, &mut cache, c.n_layer)
            } else {
                m.forward_cached(&seq[seq.len() - 1..], &mut cache, c.n_layer)
            };
            let v = logits.to_vec().await;
            let rows = if reprefill || step == 0 { seq.len().min(ids.len().max(1)) } else { 1 };
            let _ = rows;
            let row = &v[v.len() - c.n_vocab..]; // last position's logits
            let next = argmax(row);
            if step == 0 { prompt_ms = t0.elapsed().as_secs_f64() * 1e3; ferric_tensor::prof_report(); }
            print!("{}", detok(vocab.get(next as usize).map(|s| s.as_str()).unwrap_or("?")));
            std::io::stdout().flush().ok();
            seq.push(next);
        }
        ferric_tensor::prof_report(); // accumulated over the decode steps
        let (disp, subs) = ferric_tensor::op_counters();
        println!("\n  [{disp} dispatches, {subs} submits total]");
        let total = t0.elapsed().as_secs_f64();
        let decode_ms = (total * 1e3 - prompt_ms) / (n - 1).max(1) as f64;
        println!("\n  {n} tokens in {:.2}s · prompt {:.0}ms · {:.0} ms/token after", total, prompt_ms, decode_ms);
        println!("  ids: {:?}", &seq[ids.len()..]);

        // `--verify-cache` — the cache is only correct if carrying state is indistinguishable from
        // re-running the prefix. Generate both ways and require identical tokens.
        // (see also `--spec` below: speculative output must equal `--gen` output token-for-token)
        if args.iter().any(|a| a == "--verify-cache") && !reprefill {
            let mut naive = ids.clone();
            let t1 = std::time::Instant::now();
            for _ in 0..n {
                let v = m.forward(&naive).to_vec().await;
                naive.push(argmax(&v[v.len() - c.n_vocab..]));
            }
            let slow = t1.elapsed().as_secs_f64();
            let same = naive[ids.len()..] == seq[ids.len()..];
            println!("\n  re-prefill path: {:.2}s → {:?}", slow, &naive[ids.len()..]);
            println!("{} cached decode == re-prefill (identical {n} tokens) · {:.1}× faster",
                if same { "✅" } else { "❌" }, slow / total);
            assert!(same, "cached decode diverged from re-prefill");
        }
    }

    // `--spec N` — greedy speculative decoding with the model's own MTP ("nextn") draft block.
    // The draft block proposes tokenᵢ₊₂ from (tokenᵢ₊₁, hiddenᵢ); the main model verifies every
    // proposal, so the output is LOSSLESS — it must match `--gen` token-for-token. The win: a
    // (k+1)-token verify forward costs barely more than a 1-token forward in the fixed-overhead-
    // bound decode regime, so each accepted draft is a nearly-free extra token.
    if let Some(p) = args.iter().position(|a| a == "--spec") {
        let n: usize = args[p + 1].parse().unwrap();
        use std::io::Write;
        let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
        assert!(m.mtp.is_some(), "model has no MTP draft block (nextn_predict_layers == 0)");
        let pr = |t: u32| { print!("{}", detok(vocab.get(t as usize).map(|s| s.as_str()).unwrap_or("?"))); std::io::stdout().flush().ok(); };

        print!("\ngenerating (speculative, MTP self-draft): ");
        for &i in &ids { print!("{}", detok(vocab.get(i as usize).map(|s| s.as_str()).unwrap_or("?"))); }
        std::io::stdout().flush().ok();

        let t0 = std::time::Instant::now();
        let mut cache = ferric_llama::qwen35::Cache::new(c);
        let mut mc = ferric_llama::qwen35::MtpCache::default();
        mc.pos = 1; // the first pair (prompt token 1, hidden 0) sits at position 1

        // Prompt: one prefill for the main model; its hidden rows also seed the draft block's cache
        // (pairs for every prompt position — without them the drafter is blind to the prompt).
        let (lg, hid) = m.forward_spec(&ids, &mut cache, c.n_layer);
        let v = lg.to_vec().await;
        let pend0 = argmax(&v[v.len() - c.n_vocab..]);
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        ferric_tensor::prof_report(); // prompt-phase categories
        pr(pend0);
        let mut out: Vec<u32> = vec![pend0];
        let mut unfed: Vec<u32> = vec![pend0];      // committed tokens the main cache hasn't seen yet
        let mut ptoks: Vec<u32> = ids[1..].iter().copied().chain([pend0]).collect();
        let mut phid = hid;                          // pair i = (ptoks[i], row i of phid)
        let (mut steps, mut acc) = (0usize, 0usize);
        let (mut draft_s, mut verify_s) = (0.0f64, 0.0f64);

        while out.len() < n {
            // 1. Draft: feed the pending pairs (keeps the draft cache aligned), propose one token.
            let td = std::time::Instant::now();
            let dlog = m.mtp_forward(&ptoks, &phid, &mut mc);
            let dv = dlog.to_vec().await;
            draft_s += td.elapsed().as_secs_f64();
            let d = argmax(&dv[dv.len() - c.n_vocab..]);
            // 2. Verify: one forward over [unfed…, draft]. Snapshot first — tensors are immutable
            //    Arc-shared buffers, so the snapshot is O(1) handle copies, and rollback is free.
            let tv = std::time::Instant::now();
            let snap = cache.snapshot();
            let k = unfed.len();
            let toks: Vec<u32> = unfed.iter().copied().chain([d]).collect();
            let (lg, hid) = m.forward_spec(&toks, &mut cache, c.n_layer);
            let v = lg.to_vec().await;
            verify_s += tv.elapsed().as_secs_f64();
            let truth = argmax(&v[0..c.n_vocab]); // row 0 = last unfed position (forward_spec heads only the last two)
            steps += 1;
            if truth == d {
                // Accept: everything in the cache is valid; the draft's own logits row is valid too,
                // so it immediately yields the next committed token — 2 tokens from this forward.
                acc += 1;
                let pend = argmax(&v[c.n_vocab..2 * c.n_vocab]);
                pr(d); pr(pend);
                out.push(d); out.push(pend);
                ptoks = vec![d, pend];
                phid = hid.narrow(0, k - 1, 2).contiguous(); // hiddens at d's and pend's predecessors
                unfed = vec![pend];
            } else {
                // Reject: the cache holds a wrong entry at the draft's position — roll back to the
                // snapshot. Nothing is wasted: the true token was still learned from this forward,
                // and next iteration's verify forward re-feeds the unfed tokens together.
                pr(truth);
                out.push(truth);
                cache = snap;
                unfed.push(truth);
                ptoks = vec![truth];
                phid = hid.narrow(0, k - 1, 1).contiguous(); // hidden at truth's predecessor
            }
        }
        out.truncate(n);
        ferric_tensor::prof_report(); // verify-forward categories accumulated over the loop
        let total = t0.elapsed().as_secs_f64();
        let decode_ms = (total * 1e3 - prompt_ms) / (out.len() - 1).max(1) as f64;
        println!("\n  {} tokens in {:.2}s · prompt {:.0}ms · {:.1} ms/token after", out.len(), total, prompt_ms, decode_ms);
        println!("  drafts: {acc}/{steps} accepted ({:.0}%) · draft {:.0}ms/iter · verify {:.0}ms/iter",
            100.0 * acc as f64 / steps.max(1) as f64, draft_s * 1e3 / steps.max(1) as f64, verify_s * 1e3 / steps.max(1) as f64);
        println!("  ids: {:?}", out);
    }

    // `--spec2 N` — REAL 2-token speculative decoding (measured d2 conditional acceptance ≈65% ⇒
    // +29% tokens/verify; verify is fixed-overhead-bound so ~14% wall-clock). The MTP block drafts
    // d1, then recursively drafts d2 from (d1, d1's own draft-hidden). ONE verify forward over
    // [unfed…, d1, d2] commits 1, 2, or 3 tokens. Two caches advance on separate ledgers: MAIN (fed
    // by verify forwards) and MTP `mc` (fed the committed pairs by draft forwards, each exactly once).
    // On partial accept (d1 ok, d2 wrong) we roll MAIN back to the snapshot and re-feed [d1, t2] next
    // iter — one wasted d1 recompute, but the rollback point stays trivially correct. Output == --gen.
    if let Some(p) = args.iter().position(|a| a == "--spec2") {
        let n: usize = args[p + 1].parse().unwrap();
        use std::io::Write;
        let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
        assert!(m.mtp.is_some(), "model has no MTP draft block");
        let pr = |t: u32| { print!("{}", detok(vocab.get(t as usize).map(|s| s.as_str()).unwrap_or("?"))); std::io::stdout().flush().ok(); };
        print!("\ngenerating (2-token speculative): ");
        for &i in &ids { pr(i); }

        let t0 = std::time::Instant::now();
        let mut cache = ferric_llama::qwen35::Cache::new(c);
        let mut mc = ferric_llama::qwen35::MtpCache::default();
        mc.pos = 1;
        let (lg, hid) = m.forward_spec(&ids, &mut cache, c.n_layer);
        let v = lg.to_vec().await;
        let pend0 = argmax(&v[v.len() - c.n_vocab..]);
        pr(pend0);
        let mut out: Vec<u32> = vec![pend0];
        let mut unfed: Vec<u32> = vec![pend0];        // committed, not yet in MAIN cache
        let mut ptoks: Vec<u32> = ids[1..].iter().copied().chain([pend0]).collect(); // committed, not yet in mc
        let mut phid = hid;
        let (mut steps, mut a1, mut a2) = (0usize, 0usize, 0usize);
        while out.len() < n {
            // Draft d1 (advances real mc past ptoks), then d2 recursively on a throwaway clone.
            let (l1, h1) = m.mtp_forward_h(&ptoks, &phid, &mut mc);
            let d1 = argmax(&l1.to_vec().await[..c.n_vocab]);
            let mut probe = mc.clone();
            let l2 = m.mtp_forward_h(&[d1], &h1, &mut probe).0;
            let d2 = argmax(&l2.to_vec().await[..c.n_vocab]);
            // Verify [unfed…, d1, d2] — head the last 3 rows (d1-check, d2-check, pend).
            let snap = cache.snapshot();
            let k = unfed.len();
            let toks: Vec<u32> = unfed.iter().copied().chain([d1, d2]).collect();
            let (lg, hid) = m.forward_spec_k(&toks, &mut cache, c.n_layer, 3);
            let v = lg.to_vec().await; // row 0 → after unfed (=d1?), 1 → after d1 (=d2?), 2 → after d2 (pend)
            steps += 1;
            let t1 = argmax(&v[0..c.n_vocab]);
            let dbg = std::env::var("FERRIC_SPEC_DBG").is_ok();
            if dbg {
                let r1 = &v[c.n_vocab..2 * c.n_vocab];
                let t2p = argmax(r1);
                // gap between the row-1 argmax and the runner-up — a near-tie means the multi-token
                // verify's fp-order can legally flip vs single-token decode.
                let mut sorted: Vec<f32> = r1.to_vec(); sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                eprintln!("iter {steps}: k={k} unfed={unfed:?} d1={d1} d2={d2} t1={t1} t2={t2p} · row1 top2 gap={:.4}", sorted[0] - sorted[1]);
            }
            if t1 != d1 {
                // Reject both. Roll MAIN back; re-feed [t1] next iter. mc next gets [t1].
                pr(t1); out.push(t1);
                cache = snap;
                unfed.push(t1);
                ptoks = vec![t1];
                phid = hid.narrow(0, k - 1, 1).contiguous(); // t1's predecessor (last unfed)
                continue;
            }
            a1 += 1;
            let t2 = argmax(&v[c.n_vocab..2 * c.n_vocab]);
            if t2 != d2 {
                // Accept d1 only. d2's cache entry is wrong → roll MAIN back to snap (loses d1's valid
                // entry too) and re-feed [d1, t2] next iter. hid rows k-1,k (d1/t2 predecessors) valid.
                pr(d1); pr(t2); out.push(d1); out.push(t2);
                cache = snap;
                unfed.push(d1); unfed.push(t2);
                ptoks = vec![d1, t2];
                phid = hid.narrow(0, k - 1, 2).contiguous();
            } else {
                // Accept both. MAIN holds [unfed…, d1, d2] valid; emit pend (not fed). mc gets d1,d2,pend.
                a2 += 1;
                let pend = argmax(&v[2 * c.n_vocab..3 * c.n_vocab]);
                pr(d1); pr(d2); pr(pend);
                out.push(d1); out.push(d2); out.push(pend);
                unfed = vec![pend];
                ptoks = vec![d1, d2, pend];
                phid = hid.narrow(0, k - 1, 3).contiguous();
            }
        }
        out.truncate(n);
        let total = t0.elapsed().as_secs_f64();
        println!("\n  {} tokens in {:.2}s · {:.1} ms/token", out.len(), total, total * 1e3 / out.len() as f64);
        println!("  d1 accept {a1}/{steps} ({:.0}%) · d2 accept {a2}/{} ({:.0}%) · {:.2} tok/verify",
            100.0 * a1 as f64 / steps.max(1) as f64, a1, 100.0 * a2 as f64 / a1.max(1) as f64,
            out.len() as f64 / steps.max(1) as f64);
        println!("  ids: {:?}", out);
    }
}
