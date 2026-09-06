//! **Run a hyv4 forward pass on a checkpoint this file writes.**
//!
//! Every component of `hyv4` is verified on its own — the hyper-connection closed form and both MLA
//! absorption folds exactly over GF(2⁶¹−1), the sink's softmax against its definition, the indexer's
//! sharing schedule by bounded model checking, the quant formats by Kani plus an interop check
//! against Tencent's published weights. None of that had ever been *composed*: no tensor had passed
//! through `Hyv4::forward`, because the smallest real checkpoint is 213.66 GiB.
//!
//! So write a small one. Two blocks, `d = 32`, four experts — the same architecture at a size that
//! fits, with the shapes and tensor names the real file uses.
//!
//! ⛔ **What this proves and what it cannot.** It proves the WIRING: that every tensor name
//! resolves, that the shapes agree end to end, that the block schedule composes, that a forward
//! terminates and produces finite logits of the right shape. It CANNOT prove fidelity to Tencent's
//! model, because the same conventions that write the file are used to read it back — a transposed
//! convention applied twice cancels. Fidelity needs the real weights, and that needs a bigger
//! machine. These are different claims and this file makes only the first.
//!
//! ```text
//! cargo run --release -p ferric-llama --example hyv4_synthetic
//! ```

use ferric_core::Context;
use ferric_gguf::write::GgufWriter;

use ferric_llama::hyv4::{Hyv4, Hyv4Cache};
use std::sync::Arc;

const HC: usize = 4;
const D: usize = 32;
const HEADS: usize = 2;
const QK_NOPE: usize = 8;
const ROPE: usize = 4;
const VH: usize = 8;
const KVL: usize = 12;
const QLORA: usize = 16;
const L: usize = 2;
const VOCAB: usize = 40;
const N_EXPERT: usize = 4;
const EXPERT_FF: usize = 16;
const FF: usize = 24;
const IDX_HEADS: usize = 2;
const IDX_DK: usize = 8;

/// Build a synthetic hyv4 checkpoint. `top_k` is a parameter so the DSA mask can be switched
/// between "selects everything" and "selects two positions" -- the only way to show, without the
/// real weights, that the indexer actually reaches the attention.
fn build(top_k: u32) -> Vec<u8> {
    let mut seed = 0xa5a5_1234u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Two-signed, and small: hyper-connections multiply four streams together and a wide
            // draw saturates the gates, which would test the plumbing on constants.
            (((seed >> 32) as f32 / (1u64 << 31) as f32) - 1.0) * 0.2
        }).collect()
    };

    let qk_head = QK_NOPE + ROPE;
    let mut w = GgufWriter::new("hyv4");
    w.kv_u32("hyv4.block_count", L as u32)
        .kv_u32("hyv4.embedding_length", D as u32)
        .kv_u32("hyv4.feed_forward_length", FF as u32)
        .kv_u32("hyv4.attention.head_count", HEADS as u32)
        .kv_u32("hyv4.attention.head_count_kv", 1)
        .kv_u32("hyv4.vocab_size", VOCAB as u32)
        .kv_u32("hyv4.context_length", 64)
        .kv_f32("hyv4.attention.layer_norm_rms_epsilon", 1e-5)
        .kv_u32("hyv4.attention.key_length_mla", qk_head as u32)
        .kv_u32("hyv4.attention.value_length_mla", VH as u32)
        .kv_u32("hyv4.rope.dimension_count", ROPE as u32)
        .kv_f32("hyv4.rope.freq_base", 10_000_000.0)
        .kv_u32("hyv4.attention.q_lora_rank", QLORA as u32)
        .kv_u32("hyv4.attention.kv_lora_rank", KVL as u32)
        .kv_u32("hyv4.leading_dense_block_count", 1)
        .kv_u32("hyv4.expert_count", N_EXPERT as u32)
        .kv_u32("hyv4.expert_used_count", 2)
        .kv_u32("hyv4.expert_shared_count", 1)
        .kv_u32("hyv4.expert_feed_forward_length", EXPERT_FF as u32)
        .kv_f32("hyv4.expert_weights_scale", 2.827)
        .kv_bool("hyv4.expert_weights_norm", true)
        .kv_u32("hyv4.expert_gating_func", 2)
        .kv_arr_f32("hyv4.swiglu_clamp_exp", &vec![10.0f32; L])
        .kv_u32("hyv4.hyper_connection.count", HC as u32)
        .kv_f32("hyv4.hyper_connection.epsilon", 1e-6)
        .kv_f32("hyv4.hyper_connection.magnitude", 2.0)
        .kv_u32("hyv4.attention.indexer.head_count", IDX_HEADS as u32)
        .kv_u32("hyv4.attention.indexer.key_length", IDX_DK as u32)
        .kv_u32("hyv4.attention.indexer.top_k", top_k)
        // Layer 0 full, layer 1 sharing — the smallest schedule that exercises BOTH branches of the
        // index reuse. A model where every layer were full would never test the sharing path.
        .kv_arr_i32("hyv4.attention.indexer.is_full", &[1, 0])   // I32, as Tencent's file stores it
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "hyv4");

    // ⭐ A MINIMAL BPE VOCAB, so this file is readable by ANOTHER IMPLEMENTATION and not just by the
    // code that wrote it. llama.cpp refuses a gpt2-model checkpoint with "cannot find tokenizer
    // merges in model file" before it will build a graph, and an oracle needs ONE file both sides
    // load — not two builders that agree with themselves. Ferric's own loader ignores these keys;
    // they exist purely so the reference can get as far as the forward pass.
    // SINGLE-CHARACTER tokens, so ordinary text actually tokenizes: a BPE whose vocabulary is
    // "t0".."t39" cannot segment anything, and the reference loads fine and then reports "there are
    // not input tokens to process" — a model that loads but cannot be given input is not an oracle.
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789+-*/";
    assert_eq!(ALPHABET.len(), VOCAB, "the alphabet must supply exactly one token per vocab slot");
    let tokens: Vec<String> = ALPHABET.iter().map(|c| (*c as char).to_string()).collect();
    let merges: Vec<String> = (0..VOCAB - 1)
        .map(|i| format!("{} {}", tokens[i], tokens[i + 1])).collect();
    let ttype: Vec<i32> = vec![1; VOCAB];   // 1 = LLAMA_TOKEN_TYPE_NORMAL
    w.kv_arr_str("tokenizer.ggml.tokens", &tokens)
        .kv_arr_str("tokenizer.ggml.merges", &merges)
        .kv_arr_i32("tokenizer.ggml.token_type", &ttype)
        .kv_u32("tokenizer.ggml.bos_token_id", 0)
        .kv_u32("tokenizer.ggml.eos_token_id", (VOCAB - 1) as u32);

    w.tensor_f32("token_embd.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
        .tensor_f32("output.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
        .tensor_f32("output_norm.weight", &[D as u64], &vec![1.0; D])
        .tensor_f32("output_hc_fn.weight", &[(HC * D) as u64, HC as u64], &rnd(HC * HC * D))
        .tensor_f32("output_hc_base.weight", &[HC as u64], &rnd(HC))
        .tensor_f32("output_hc_scale.weight", &[1], &[0.8]);

    for il in 0..L {
        let b = |s: &str| format!("blk.{il}.{s}");
        // ⚠ Dims are in GGUF ne order: ne0 is the FASTEST axis, so a [out, in] matrix is written
        // [in, out]. Writing them the other way round is the transposition class of bug — right
        // element count, wrong arrangement — and it would survive a round trip through this file
        // because the loader reverses whatever is written.
        w.tensor_f32(&b("attn_norm.weight"), &[D as u64], &vec![1.0; D])
            .tensor_f32(&b("ffn_norm.weight"), &[D as u64], &vec![1.0; D])
            .tensor_f32(&b("attn_q_a.weight"), &[D as u64, QLORA as u64], &rnd(QLORA * D))
            .tensor_f32(&b("attn_q_a_norm.weight"), &[QLORA as u64], &vec![1.0; QLORA])
            .tensor_f32(&b("attn_q_b.weight"), &[QLORA as u64, (HEADS * qk_head) as u64], &rnd(HEADS * qk_head * QLORA))
            .tensor_f32(&b("attn_kv_a_mqa.weight"), &[D as u64, (KVL + ROPE) as u64], &rnd((KVL + ROPE) * D))
            .tensor_f32(&b("attn_kv_a_norm.weight"), &[KVL as u64], &vec![1.0; KVL])
            .tensor_f32(&b("attn_k_b.weight"), &[QK_NOPE as u64, KVL as u64, HEADS as u64], &rnd(HEADS * KVL * QK_NOPE))
            .tensor_f32(&b("attn_v_b.weight"), &[KVL as u64, VH as u64, HEADS as u64], &rnd(HEADS * VH * KVL))
            .tensor_f32(&b("attn_gate.weight"), &[D as u64, (HEADS * VH) as u64], &rnd(HEADS * VH * D))
            .tensor_f32(&b("attn_output.weight"), &[(HEADS * VH) as u64, D as u64], &rnd(D * HEADS * VH))
            .tensor_f32(&b("attn_sinks.weight"), &[HEADS as u64], &rnd(HEADS))
            .tensor_f32(&b("hc_attn_fn.weight"), &[(HC * D) as u64, (2 * HC) as u64], &rnd(2 * HC * HC * D))
            .tensor_f32(&b("hc_attn_base.weight"), &[(2 * HC) as u64], &rnd(2 * HC))
            .tensor_f32(&b("hc_attn_scale.weight"), &[2], &[0.7, 1.3])
            .tensor_f32(&b("hc_ffn_fn.weight"), &[(HC * D) as u64, (2 * HC) as u64], &rnd(2 * HC * HC * D))
            .tensor_f32(&b("hc_ffn_base.weight"), &[(2 * HC) as u64], &rnd(2 * HC))
            .tensor_f32(&b("hc_ffn_scale.weight"), &[2], &[0.7, 1.3]);

        if il == 0 {
            w.tensor_f32(&b("indexer.attn_q_b.weight"), &[QLORA as u64, (IDX_HEADS * IDX_DK) as u64], &rnd(IDX_HEADS * IDX_DK * QLORA))
                .tensor_f32(&b("indexer.attn_k.weight"), &[D as u64, IDX_DK as u64], &rnd(IDX_DK * D))
                .tensor_f32(&b("indexer.k_norm.weight"), &[IDX_DK as u64], &vec![1.0; IDX_DK])
                .tensor_f32(&b("indexer.k_norm.bias"), &[IDX_DK as u64], &rnd(IDX_DK))
                .tensor_f32(&b("indexer.proj.weight"), &[D as u64, IDX_HEADS as u64], &rnd(IDX_HEADS * D));
            // Block 0 is the dense one (leading_dense_block_count = 1).
            w.tensor_f32(&b("ffn_gate.weight"), &[D as u64, FF as u64], &rnd(FF * D))
                .tensor_f32(&b("ffn_up.weight"), &[D as u64, FF as u64], &rnd(FF * D))
                .tensor_f32(&b("ffn_down.weight"), &[FF as u64, D as u64], &rnd(D * FF));
        } else {
            w.tensor_f32(&b("ffn_gate_inp.weight"), &[D as u64, N_EXPERT as u64], &rnd(N_EXPERT * D))
                .tensor_f32(&b("exp_probs_b.bias"), &[N_EXPERT as u64], &rnd(N_EXPERT))
                .tensor_f32(&b("ffn_gate_exps.weight"), &[D as u64, EXPERT_FF as u64, N_EXPERT as u64], &rnd(N_EXPERT * EXPERT_FF * D))
                .tensor_f32(&b("ffn_up_exps.weight"), &[D as u64, EXPERT_FF as u64, N_EXPERT as u64], &rnd(N_EXPERT * EXPERT_FF * D))
                .tensor_f32(&b("ffn_down_exps.weight"), &[EXPERT_FF as u64, D as u64, N_EXPERT as u64], &rnd(N_EXPERT * D * EXPERT_FF))
                .tensor_f32(&b("ffn_gate_shexp.weight"), &[D as u64, EXPERT_FF as u64], &rnd(EXPERT_FF * D))
                .tensor_f32(&b("ffn_up_shexp.weight"), &[D as u64, EXPERT_FF as u64], &rnd(EXPERT_FF * D))
                .tensor_f32(&b("ffn_down_shexp.weight"), &[EXPERT_FF as u64, D as u64], &rnd(D * EXPERT_FF));
        }
    }

    w.finish().expect("the synthetic checkpoint must be writable")
}

fn main() {
    // `hyv4_synthetic <path>` writes the checkpoint and exits, so the SAME bytes can be handed to
    // another implementation. That is the point: an oracle needs one file, not two builders.
    // `hyv4_synthetic --logits 3,11,7` prints the LAST row's logits and their sum, which is the
    // shape llama.cpp's eval-callback reports (`result_output` is one row: the last position).
    if std::env::args().nth(1).as_deref() == Some("--logits") {
        let ids: Vec<u32> = std::env::args().nth(2).expect("--logits <csv ids>")
            .split(',').map(|t| t.trim().parse().expect("token id")).collect();
        let ctx = Arc::new(pollster::block_on(Context::new()).expect("gpu"));
        let g = ferric_gguf::parse(build(64)).expect("parse");
        let m = Hyv4::load(&ctx, &g).expect("load");
        let l = pollster::block_on(m.forward(&ids).to_vec());
        let last = &l[(ids.len() - 1) * VOCAB..];
        println!("ferric last-row logits for {ids:?}:");
        for (i, v) in last.iter().enumerate() { print!("{v:>12.6}{}", if i % 8 == 7 { "\n" } else { "" }); }
        println!("sum = {:.6}", last.iter().sum::<f32>());
        return;
    }
    if let Some(out) = std::env::args().nth(1) {
        let bytes = build(64);
        std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
        println!("wrote {} bytes to {out}", bytes.len());
        return;
    }
    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context"); return };
    let ctx = Arc::new(ctx);

    let run = |top_k: u32| -> (Vec<f32>, Vec<usize>) {
        let bytes = build(top_k);
        let g = ferric_gguf::parse(bytes).expect("Ferric must read back what it wrote");
        let model = match Hyv4::load(&ctx, &g) {
            Ok(m) => m,
            Err(e) => { eprintln!("LOAD FAILED: {e}"); std::process::exit(1) }
        };
        let tokens: Vec<u32> = vec![3, 11, 7, 29, 1];
        let logits = model.forward(&tokens);
        (pollster::block_on(logits.to_vec()), logits.shape.clone())
    };

    println!("wrote a synthetic hyv4 checkpoint: {} bytes, {L} blocks, d={D}, {N_EXPERT} experts",
             build(64).len());

    // Dense selection: top_k >= seq, so the indexer's mask admits everything and the attention is
    // ordinary causal attention.
    let (dense, shape) = run(64);
    assert_eq!(shape, vec![5, VOCAB], "logits must be [seq, vocab]");
    assert!(dense.iter().all(|x| x.is_finite()), "a forward producing NaN or inf has not run");

    // ── the check with independent meaning ────────────────────────────────────────────────────
    //
    // Sparse selection: top_k = 2, so each query keeps at most two positions and the mask must
    // change the answer for every token that has more than two visible. If the indexer's mask never
    // reached `Mla::forward_masked`, these would be identical -- and every other assertion in this
    // file would still pass, because shape, finiteness and row-distinctness all survive a mask
    // that goes nowhere.
    let (sparse, _) = run(2);
    let moved = dense.iter().zip(&sparse).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(moved > 1e-4, "top_k=2 and top_k=64 give the same logits ({moved:.2e}); the DSA mask \
                           is not reaching the attention");
    // Token 0 sees only itself, so no selection can change it. If THAT moved, the mask is being
    // applied non-causally.
    let tok0 = dense[..VOCAB].iter().zip(&sparse[..VOCAB]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(tok0 < 1e-5, "the first token's logits moved under a tighter top_k ({tok0:.2e}); it has \
                          one visible position, so the mask is not respecting causality");

    // ── Cached decode must equal a full re-run ──────────────────────────────────────────────────
    //
    // The whole graph, not just MLA: hyper-connections, the DSA indexer with its per-full-layer key
    // cache and the schedule that shares it, the clamped-SwiGLU MoE, and absolute-position RoPE. Any
    // split into decode blocks must reproduce `forward` over the whole sequence.
    //
    // ⛔ Uneven splits on purpose, and 1+1+1+1+1 is NOT sufficient on its own: at t=1 the offset
    // causal mask is a no-op (see mla.rs), so single stepping cannot see a wrong offset. Only blocks
    // with t > 1 make intra-block causality observable.
    //
    // ⛔ AND IT MUST RUN SPARSE. At top_k=64 every visible position is selected, so the DSA mask is
    // all zeros and its COLUMN ORDER is unobservable — mutating the index cache to prepend instead
    // of append survived the dense-only version of this check. Under top_k=2 the mask actually
    // selects, and a cache whose column j is not position j changes the answer.
    for (top_k, reference) in [(64u32, &dense), (2, &sparse)] {
        let g = ferric_gguf::parse(build(top_k)).expect("parse");
        let model = Hyv4::load(&ctx, &g).expect("load");
        let tokens: Vec<u32> = vec![3, 11, 7, 29, 1];
        let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-3, "reference logits are ~zero; this comparison would pass on anything");

        for split in [vec![1usize, 1, 1, 1, 1], vec![1, 2, 2], vec![2, 1, 2], vec![3, 2], vec![5]] {
            let mut cache = Hyv4Cache::new(&model).expect("cache");
            assert_eq!(cache.index_slots(), model.index_schedule().live_cache_layers(),
                       "the cache reserved a different number of index slots than the schedule names");
            let (mut got, mut at) = (Vec::new(), 0usize);
            for n in &split {
                let out = model.decode(&tokens[at..at + n], &mut cache);
                got.extend(pollster::block_on(out.to_vec()));
                at += n;
            }
            assert_eq!(cache.len(), tokens.len(), "cache length disagrees with what was fed");
            let worst = reference.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("  top_k {top_k:>2} decode split {split:?}: max |Δ| vs full forward = {worst:.3e}");
            assert!(worst < 2e-5 * scale.max(1.0),
                    "top_k {top_k} decode split {split:?} diverges from the full forward by {worst}");
        }
        if top_k != 64 { continue }
        // The positional check the equality above would miss if RoPE restarted per block: feeding
        // the SAME token at two different positions must give different logits.
        let mut c2 = Hyv4Cache::new(&model).expect("cache");
        let a0 = pollster::block_on(model.decode(&[7], &mut c2).to_vec());
        let a1 = pollster::block_on(model.decode(&[7], &mut c2).to_vec());
        let moved_pos = a0.iter().zip(&a1).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(moved_pos > 1e-4, "the same token at position 0 and 1 gave identical logits \
                                   ({moved_pos:.2e}); RoPE is restarting at each block or the cache is not read");
        println!("  same token at position 0 vs 1 moves logits by {moved_pos:.4}");
    }

    // ── Batched decode must equal solo decode, sequence by sequence ─────────────────────────────
    //
    // ⛔ THE SEQUENCES HAVE DIFFERENT LENGTHS ON PURPOSE. Equal-length sequences hide the whole
    // class of bug this exists for: using sequence 0's `n_past` for everyone gives every row the
    // right answer when every row is at the same position. Priming to 1, 3 and 2 tokens makes the
    // per-sequence RoPE position, the per-sequence top-k offset and the per-sequence cache length
    // all distinct, so borrowing any of them across streams shows up immediately.
    //
    // Both arms prime identically and solo; only the steps after that differ, so any divergence is
    // attributable to batching and nothing else.
    {
        let g = ferric_gguf::parse(build(64)).expect("parse");
        let m = Hyv4::load(&ctx, &g).expect("load");
        // n = 2, 3 and 4, matching the bar the other batched runtimes in this repo were held to.
        let all_primes: [&[u32]; 4] = [&[3], &[11, 7, 29], &[1, 19], &[2, 5, 8, 13]];
        let all_steps: [[u32; 2]; 4] = [[5, 23], [2, 31], [17, 9], [21, 6]];
        for n in [2usize, 3, 4] {
            let primes = &all_primes[..n];
            let steps = &all_steps[..n];
            let prime = |cs: &mut Vec<Hyv4Cache>| {
                for (i, p) in primes.iter().enumerate() { let _ = m.decode(p, &mut cs[i]); }
            };
            let mut solo: Vec<Hyv4Cache> = (0..n).map(|_| Hyv4Cache::new(&m).expect("cache")).collect();
            let mut batched: Vec<Hyv4Cache> = (0..n).map(|_| Hyv4Cache::new(&m).expect("cache")).collect();
            prime(&mut solo);
            prime(&mut batched);
            for (i, c) in solo.iter().enumerate() {
                assert_eq!(c.len(), primes[i].len(), "priming did not advance sequence {i} as expected");
            }

            for step in 0..2 {
                let want: Vec<Vec<f32>> = (0..n)
                    .map(|i| pollster::block_on(m.decode(&[steps[i][step]], &mut solo[i]).to_vec()))
                    .collect();
                let toks: Vec<u32> = (0..n).map(|i| steps[i][step]).collect();
                let got = {
                    let mut refs: Vec<&mut Hyv4Cache> = batched.iter_mut().collect();
                    pollster::block_on(m.decode_batch(&toks, &mut refs).to_vec())
                };
                assert_eq!(got.len(), n * VOCAB, "decode_batch must return one row per sequence");
                let mut worst_any = 0.0f32;
                for i in 0..n {
                    let row = &got[i * VOCAB..(i + 1) * VOCAB];
                    let scale = want[i].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    assert!(scale > 1e-3, "solo logits for sequence {i} are ~zero; nothing would fail here");
                    let worst = want[i].iter().zip(row).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    worst_any = worst_any.max(worst);
                    assert!(worst < 2e-5 * scale.max(1.0),
                            "n={n}: batched sequence {i} diverges from solo decode by {worst} at step {step}");
                }
                println!("  n={n} batch step {step}: max |Δ| vs solo across {n} sequences = {worst_any:.3e}");
            }
            for i in 0..n {
                assert_eq!(solo[i].len(), batched[i].len(),
                           "n={n} sequence {i}: solo and batched caches disagree on length");
                assert_eq!(solo[i].len(), primes[i].len() + 2, "n={n} sequence {i} did not advance by two steps");
            }
        }
    }

    // ── Greedy generation must agree with a fresh full forward of what it produced ──────────────
    //
    // The last composition nothing exercised: decode -> argmax -> FEED THE TOKEN BACK. Every piece
    // under it is already pinned, but the loop that a server actually runs was not, and its failure
    // mode is silent — a cache that drifts one position produces perfectly plausible tokens.
    //
    // The check is self-referential in the useful direction: generate greedily through the cache,
    // then run `forward` over the sequence that produced, and require the argmax at each position to
    // be the token generation emitted there. Cache drift breaks it; a correct loop cannot.
    //
    // ⚠ HONEST LABEL: this is a COMPOSITION check, largely SUBSUMED by the decode oracle above. It
    // drives the same machinery, and the splits run first, so every library mutation I could
    // construct (rope from the wrong position, index cache prepended, n_past not advancing) is
    // caught there before this is reached. What it independently pins is the loop's SHAPE — which
    // row of a multi-token decode predicts the next token, and that feeding that token back lands it
    // at the right position — an interface property the splits do not state. Do not read it as
    // independent evidence about the arithmetic.
    {
        let g = ferric_gguf::parse(build(64)).expect("parse");
        let m = Hyv4::load(&ctx, &g).expect("load");
        let prompt: Vec<u32> = vec![3, 11, 7];
        let k = 4usize;
        let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;

        let mut cache = Hyv4Cache::new(&m).expect("cache");
        let first = pollster::block_on(m.decode(&prompt, &mut cache).to_vec());
        let mut emitted = vec![argmax(&first[(prompt.len() - 1) * VOCAB..])];
        for _ in 1..k {
            let last = *emitted.last().unwrap();
            let l = pollster::block_on(m.decode(&[last], &mut cache).to_vec());
            emitted.push(argmax(&l));
        }
        assert_eq!(cache.len(), prompt.len() + k - 1, "the cache advanced by the wrong number of steps");

        // ⛔ A degenerate model that emits one token forever would satisfy the agreement below
        // trivially. Require the run to actually vary before trusting it.
        let distinct = emitted.iter().collect::<std::collections::BTreeSet<_>>().len();
        assert!(distinct >= 2, "greedy generation emitted {distinct} distinct token(s) {emitted:?}; \
                                this comparison cannot discriminate on a constant sequence");

        let mut seq = prompt.clone();
        seq.extend(&emitted[..k - 1]);
        let full = pollster::block_on(m.forward(&seq).to_vec());
        for (j, t) in emitted.iter().enumerate() {
            let pos = prompt.len() - 1 + j;
            let got = argmax(&full[pos * VOCAB..(pos + 1) * VOCAB]);
            assert_eq!(got, *t, "position {pos}: a full forward predicts {got}, generation emitted {t}");
        }
        println!("  greedy generation {emitted:?} ({distinct} distinct) agrees with a full forward at every position");
    }

    // ── A streamed model must equal a resident one ──────────────────────────────────────────────
    //
    // The whole point of streaming is that it changes WHERE the bytes are, not what they compute.
    // So the check is equality against the resident load, on the same file, at a budget too small to
    // hold the model.
    //
    // ⚠ This is what makes a 213.66 GiB checkpoint runnable at all: loaded whole it peaked past
    // 250 GiB on a 256 GiB machine and was SIGKILLed. Blocks are visited 0..N once per token, so a
    // few slots is enough — but only if streaming is arithmetically identical, which is this.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = std::env::temp_dir().join("ferric_hyv4_stream_check.gguf");
        std::fs::write(&path, build(64)).expect("write checkpoint");
        let p = path.to_str().expect("utf-8 path");

        let resident = Hyv4::load(&ctx, &ferric_gguf::parse(build(64)).expect("parse")).expect("load");
        let toks: Vec<u32> = vec![3, 11, 7, 29, 1];
        let want = pollster::block_on(resident.forward(&toks).to_vec());

        // A budget of one block's run, so at least one block must be rebuilt per pass rather than
        // pinned — a budget large enough to pin everything would test nothing.
        for budget in [1u64 << 14, 1 << 16] {
            let streamed = match Hyv4::load_streaming(&ctx, p, budget) {
                Ok(m) => m,
                Err(e) => { println!("  streaming at {budget} B refused: {e}"); continue }
            };
            let got = pollster::block_on(streamed.forward(&toks).to_vec());
            let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let rebuilds = streamed.stream().map(|s| s.rebuilds()).unwrap_or(0);
            println!("  streamed at {budget} B: max |Δ| vs resident = {worst:.3e}, {rebuilds} block rebuild(s)");
            assert!(rebuilds > 0, "nothing was rebuilt — this budget pinned the model and tested nothing");
            assert_eq!(worst, 0.0, "streaming changed the arithmetic; it must only change where bytes live");
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── the regression lock, PER FABRIC ───────────────────────────────────────────────────────
    //
    // ⚠ This hash is SELF-REFERENTIAL: it was generated from this code, so it cannot say the
    // forward is correct against Tencent's model. What it does is pin what this code computes, so
    // a later edit cannot silently change it. Without it the assertions above pass under a stale
    // residual, a router bias carried into the weight, or the stream collapse replaced by "take
    // stream 0" -- all four were mutation-tested and all four survived until this line existed.
    //
    // ⛔ AND IT IS FABRIC-SPECIFIC, WHICH IS NOT A BUG. Kernel selection is capability-dependent, so
    // the same model on the same bytes reduces in a different order on a different backend and the
    // bit-hash differs. A single constant here made the FIRST CI run on lavapipe fail with a
    // perfectly healthy model. One hash per backend records that difference instead of hiding it;
    // an unknown backend reports rather than asserting, because a lock nobody has recorded is not
    // evidence about that fabric.
    //
    // ⭐ The checks ABOVE this point are all self-comparisons within one run — decode against
    // prefill, top_k 64 against top_k 2 — so they are fabric-independent and now run FIRST. They
    // used to sit after this lock, which meant a hash mismatch on a new backend hid every portable
    // check behind it. Order the portable evidence before the fabric-specific lock.
    let h = fnv(&dense);
    // Keyed by ADAPTER, matched by PREFIX. Backend is too coarse and CI proved it: the macOS runner
    // is an "Apple Paravirtual device" on the SAME Metal backend as this laptop's "Apple M5 Max" and
    // hashes to a third value. Prefix because llvmpipe carries an LLVM version that moves with the
    // runner image ("llvmpipe (LLVM 20.1.2, 256 bits)"), and a lock that breaks on an unrelated
    // image bump teaches people to ignore it.
    const GOLDEN: &[(&str, u64)] = &[
        ("Apple M5 Max",             0x29142d075f1beadc),
        ("Apple Paravirtual device", 0x13cfd14821cf04d5),  // GitHub macOS runner, run 33999082940
        // ⚠ ONE observation each. If an entry ever flakes the honest response is to DELETE IT — a
        // lock loosened until it stops failing is not a lock — and rely on the portable checks above.
        ("llvmpipe",                 0xf690016066e7574b),  // GitHub ubuntu runner (lavapipe)
    ];
    let adapter = &ctx.adapter_name;
    match GOLDEN.iter().find(|(a, _)| adapter.starts_with(a)) {
        Some((a, want)) => assert_eq!(h, *want,
            "the forward's output changed on {a:?}; if that was intended, update GOLDEN for this \
             adapter and say in the commit what moved and why"),
        None => println!("  logits hash {h:#018x} on {adapter:?} — no lock recorded for this adapter; \
                          add ({adapter:?}, {h:#018x}) to GOLDEN to lock it in"),
    }

    println!("forward ran: logits {shape:?}, all finite");
    println!("  top_k 64 -> 2 moved logits by {moved:.4}, first token unchanged by {tok0:.2e}");
    for t in 0..5 {
        let r = &dense[t * VOCAB..(t + 1) * VOCAB];
        let arg = r.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        println!("  pos {t} -> argmax {arg:>3}, range [{:+.4}, {:+.4}]",
                 r.iter().cloned().fold(f32::MAX, f32::min), r.iter().cloned().fold(f32::MIN, f32::max));
    }
    println!("\n  This proves the WIRING composes and that the sparse mask reaches the attention.\n  \
              It does not prove fidelity to Tencent's model: the conventions that wrote this file\n  \
              are the ones that read it back.");
}

/// FNV-1a over the logit bits. A hash, not a tolerance: the forward is deterministic on one
/// machine, and any change at all should be seen rather than absorbed.
fn fnv(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
